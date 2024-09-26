# coding=utf-8
import copy
import os
import sys
import time
import html
import json
from heapq import merge
from copy import deepcopy
from typing import Dict, List, Tuple
from pathlib import Path
import argparse
from argparse import Namespace
import logging
import math
import ast
import os
import random
from datetime import timedelta
import torch
from functools import partial
import subprocess
import numpy as np
from dill.pointers import parents
from torch.cuda import memory_allocated
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, gather_object, broadcast_object_list
import datasets
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
import wandb
import huggingface_hub
from accelerate.utils import DeepSpeedPlugin

import transformers
from transformers import (
    AutoConfig,
    PretrainedConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
import torch.nn.functional as F
import torch.multiprocessing as mp
import pickle
sys.path.append(Path(__file__).parents[1].absolute().as_posix())

from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from att import apply_att_template, encode_with_chat_template, DataCollatorForATT, has_responses

from eval.utils import maybe_create_reformatted_lora_checkpoint
from constants import BAD_MISTRAL_CHAT_TEMPLATE, ATT_SYSTEM_PROMPT, ATT_TEMPLATE
from load_utils import (add_common_training_args, pretty_print_chatml, preprocess_data_to_chatml, \
                        load_tokenizer_model, save_args, load_args, preprocess_hh_common, target_lora_modules)
from datetime import datetime
# from eval.truthfulqa.run_eval import main as run_eval
# from eval.truthfulqa.run_eval import parse_args as parse_args_eval

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256" - Cuda out of memory


logger = get_logger(__name__)
import pandas as pd
import gc

try:
    from hf_olmo import OLMoTokenizerFast
except ImportError:
    logger.warning("OLMo not installed. Ignore if using a different model.")

# wandb login stage
api_key_file = os.getenv('WANDB_API_KEY_FILE_AT')

if api_key_file:
    try:
        with open(api_key_file, 'r') as file:
            wandb_api_key = file.readline().strip()
            os.environ['WANDB_API_KEY'] = wandb_api_key  # Set the API key in the environment
    except Exception as e:
        raise ValueError(f"An error occurred while reading the WANDB_API_KEY from file: {e}")
else:
    raise ValueError("WANDB_API_KEY_FILE_AT environment variable not set")

def convert_batches_to_sentences(tokenizer, small_batches):
    all_sentences = []
    all_input_ids_length = []  

    for batch in small_batches:
        input_ids_batch = batch["input_ids"]
        attention_mask_batch = batch["attention_mask"]

        # Loop through each example in the batch
        for input_ids in input_ids_batch:
            # Decode the input_ids to a sentence (skip special tokens like padding)
            sentence = tokenizer.decode(input_ids, skip_special_tokens=True)
            all_sentences.append(sentence)
            all_input_ids_length.append(len(input_ids))  # Append the length of the input_ids
    
    return all_sentences, all_input_ids_length   

def generalized_jsd_loss(
        student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction="batchmean"
):
    # Apply temperature scaling
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature

    # Log probabilities
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

    # Interpolated log probabilities
    interpolated_log_probs = beta * student_log_probs + (1 - beta) * teacher_log_probs

    # KL divergence
    kl_teacher = F.kl_div(interpolated_log_probs, teacher_log_probs, reduction="none", log_target=True)
    kl_student = F.kl_div(interpolated_log_probs, student_log_probs, reduction="none", log_target=True)

    # Generalized JSD
    jsd = beta * kl_teacher + (1 - beta) * kl_student

    # Mask out padding tokens
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]

    # Apply reduction
    if reduction == "batchmean":
        return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / (jsd.size(0) * jsd.size(1))
    elif reduction == "sum":
        return jsd.sum()
    elif reduction == "mean":
        return jsd.mean()
    else:
        return jsd



def get_run_id(args: Namespace) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S") if not os.getenv('RUN_TIMESTAMP') else os.getenv('RUN_TIMESTAMP')
    # try to get the git commit hash
    # try:
    #     git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    # except subprocess.CalledProcessError:
    #     git_commit = "unknown"

    # hashes are not the same across processes, no idea why
    # arg_set = frozenset(vars(args).items())
    # args_hash = abs(hash(arg_set))
    # print(f"arg_set: {frozenset} args_hash: {args_hash}")
    return f"{timestamp}"
    # return f"{timestamp}_{args_hash}"
    # return f"{timestamp}_{git_commit}_{args_hash}"


def add_distill_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        "--remove_multiturn_data",
        action="store_true",
        help="If set, only \"prompt-response\" data is used and multi-turn dialogue data is filtered out.",
    )
    parser.add_argument(
        "--use_new_token_template",
        action="store_true",
        help="If set, the new token template is used for ATT traning."
    )
    parser.add_argument("--teacher_lora_model_name_or_path", type=str, help="Teacher model path")
    parser.add_argument(
        "--train_args",
        type=str,
        help="The tokenizer, model and lora config will be loaded with the same config as in training",
        required=True,
    )
    ## this script will directly take the path of generated rejected answers
    parser.add_argument(
        "--generation_storage", type=str, default="data_sft.json", help="A json file will store the generated data.",
        required=True
    )
    parser.add_argument(
        "--deepspeed_config_file", type=str, help="the path of deepspeed config file"
    )
    parser.add_argument(
        "--lambda_value", type=int, default=1, required=False
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, required=False
    )


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    logger.info(f"Saving model and tokenizer to {output_dir}", main_process_only=True)
    if accelerator.is_main_process:
        tokenizer_dir = Path(output_dir) / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tokenizer_dir)

    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone()
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    model_dir = Path(output_dir) / "model"
    if accelerator.is_main_process:
        model_dir.mkdir(exist_ok=True, parents=True)
    accelerator.wait_for_everyone()
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(model_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            model, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )

    state_dir = Path(output_dir) / "state"
    if accelerator.is_main_process:
        state_dir.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(state_dir)

    logger.info(f"Checkpointing done!", main_process_only=True)


def build_optimizer(args, model):
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    return optimizer


def build_lr_scheduler(accelerator, args, optimizer, train_dataloader):
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    return lr_scheduler, overrode_max_train_steps


def apply_formatting_and_encoding(example, tokenizer, max_seq_length, debug_print=False, logger=None):
    def empty_assistant_chosen(messages):
        for msg in messages:
            if msg["role"] == "assistant":
                msg["content"] = ""
        return messages

    example["chosen"] = empty_assistant_chosen(example["chosen"])
    teacher_encoded = apply_att_template(example, tokenizer, max_seq_length, debug_print, logger)
    input_ids = teacher_encoded['input_ids']
    labels = teacher_encoded['labels']
    prompt_ids = input_ids[labels == -100]
    teacher_encoded["input_ids"] = prompt_ids

    # prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    # print(f"Teacher Prompt: \n\n{prompt_text}\n\n")

    student_encoded = encode_with_chat_template(example["chosen"], tokenizer, max_seq_length, debug_print, logger)

    input_ids = student_encoded['input_ids']
    labels = student_encoded['labels']
    prompt_ids = input_ids[labels == -100]
    student_encoded["input_ids"] = prompt_ids
    if len(prompt_ids) > max_seq_length + 30:  # 30 is assumed to be the extra token amount.
        logger.warning(f"Skipping student data: input size  {len(prompt_ids)} exceeds {max_seq_length}")
        return None
    # prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    # print(f"Student Prompt: \n\n{prompt_text}\n\n")

    return {
        'teacher_encoded': teacher_encoded,
        'student_encoded': student_encoded,
    }


def main():
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="Finetune a transformers model using pairwise preference data.")
    add_common_training_args(parser)
    add_distill_args(parser)
    args = parser.parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        assert args.report_to in ["wandb", "all"], "Currently only wandb is supported for tracking."
        wandb_api_key = os.getenv('WANDB_API_KEY')
        # This should be done in accelerator.init_trackers
        # wandb.init(project="alignment_translation", entity="claire-labo")
        # Configure wandb logging within Accelerator
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # deepspeed_plugin=args.deepspeed_config_file,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    accelerator_teacher = Accelerator(
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
        # deepspeed_plugin=args.deepspeed_config_file
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    model, tokenizer, actual_eos_token, generation_config_nucleus, generation_config_greedy \
        = load_tokenizer_model(accelerator, args, substitute_eos_token=True, load_lora=False)

    model.generation_config = generation_config_greedy
    model.eos_token = actual_eos_token
    peft_model = PeftModel.from_pretrained(model, str(args.teacher_lora_model_name_or_path))

    teacher_model = peft_model.merge_and_unload()  # get the merged teacher_model

    # apply lora on top of student model
    if args.use_lora:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_lora_modules(model)
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    logger.info(str(model), main_process_only=True)
    """for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data Type: {param.dtype}")"""
    run_id = get_run_id(args)
    checkpointing_dir = Path(args.output_dir) / run_id
    # If the output directory already exists, the process was interrupted and restarted by the cluster.
    # We should resume from the last checkpoint.

    if accelerator.is_main_process:
        checkpointing_dir_exists = checkpointing_dir.exists()
        if not checkpointing_dir_exists:
            checkpointing_dir.mkdir(parents=True)
    else:
        checkpointing_dir_exists = None

    # Letting other processes know what the main process found. We don't want to create the directory multiple times.
    checkpointing_dir_actually_exists = broadcast_object_list([checkpointing_dir_exists], 0)
    if checkpointing_dir_actually_exists[0]:
        logger.info(f"Output directory {checkpointing_dir} already exists. Resuming training...")
        if args.resume_from_checkpoint is None:
            ckpt_dirs = [x for x in checkpointing_dir.iterdir() if
                         x.stem.startswith("step_") or x.stem.startswith("epoch_")]
            if len(ckpt_dirs) > 0:
                last_checkpoint = max(ckpt_dirs, key=os.path.getctime)
                args.resume_from_checkpoint = last_checkpoint
    logger.info(f"\n\n\nSaving checkpoints to {checkpointing_dir}\n\n\n")

    ######################################## Data Preprocessing ########################################
    dataset_train, dataset_test = preprocess_data_to_chatml(args)
    filtered_dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    # Replace the rejected answers in dataset with sft models' responses

    if args.generation_storage:
        generated_responses = []
        with open(args.generation_storage, "r") as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    generated_responses.extend(data.get("batch_generations", []))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line due to JSONDecodeError:{e}")
        print("Already generated answers are taken, ex:", generated_responses[0])

    for idx, example in enumerate(filtered_dataset["train"]):
        # Check if the example contains "rejected" and it's non-empty
        if "rejected" in example and len(example["rejected"]) > 0:
            # Access the last item in the "rejected" field
            last_message = example["rejected"][-1]

            # Check if the role is "assistant"
            if last_message["role"] == "assistant":
                # Replace the content with the corresponding generated response
                last_message["content"] = generated_responses[idx]

    encode_function = partial(
        apply_formatting_and_encoding,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        debug_print=False,
        logger=logger,
    )

    with accelerator.main_process_first():
        lm_datasets = filtered_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in filtered_dataset["train"].column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data", )

        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(has_responses)

    print("Size of training set:", len(filtered_dataset['train']))
    print("Size of test set:", len(filtered_dataset['test']))

    train_dataset = lm_datasets["train"]
    test_dataset = lm_datasets["test"]

    # COMMENT OUT! This is for debugging epoch checkpointing
    # train_dataset = train_dataset.select(range(192))
    # test_dataset = test_dataset.select(range(192))

    # TODO check that nothing breaks if we don't add special tokens
    # We only use instruct models, so tokens should already be there

    # Log a few random samples from the training set

    num_train_ex = accelerator.num_processes
    num_test_ex = accelerator.num_processes
    indices_train_ex = random.sample(range(len(train_dataset)), num_train_ex)
    indices_test_ex = random.sample(range(len(test_dataset)), num_test_ex)

    train_examples = [train_dataset[i] for i in indices_train_ex]
    test_examples = [test_dataset[i] for i in indices_test_ex]

    def put_big_to_front(dataset):
        logger.info("Putting the biggest examples to the front of the dataset to catch OOMs early.")
        lens = [x["teacher_encoded"]["input_ids"].shape[0] for x in dataset]
        order = list(sorted(range(len(lens)), key=lambda x: lens[x], reverse=True))
        if len(order) > 128:
            order = order[:128] + random.sample(order[128:], k=len(order[128:]))
        dataset = dataset.select(order)
        return dataset

    train_dataset = put_big_to_front(train_dataset)
    test_dataset = put_big_to_front(test_dataset)

    print("Size of training set:", len(train_dataset))
    print("Size of test set:", len(test_dataset))

    ######################################## Training Setup ########################################
    # DataLoaders creation:
    def get_dataloader(dataset, batch_size):
        return DataLoader(
            dataset,
            shuffle=True,
            collate_fn=DataCollatorForATT(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=batch_size
        )

    train_dataloader = get_dataloader(train_dataset, args.per_device_train_batch_size)
    test_dataloader = get_dataloader(test_dataset, args.per_device_train_batch_size)

    # Optimizer
    optimizer = build_optimizer(args, model)

    # Scheduler and math around the number of training steps.
    lr_scheduler, overrode_max_train_steps = build_lr_scheduler(accelerator, args, optimizer, train_dataloader)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(project_name="alignment_translation", config=experiment_config,
                                  init_kwargs={"wandb": {"entity": "claire-labo"}})

    # Define custom step metric for evaluation
    # run_id = wandb.run.id

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    # teacher_model= accelerator_teacher.prepare(teacher_model)
    test_dataloader = accelerator.prepare(test_dataloader)

    teacher_model = accelerator_teacher.prepare(teacher_model)

    for param in teacher_model.parameters():
        param.requires_grad = False

    # accelerator.state.select_deepspeed_plugin("student")
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    ######################################## Checkpointing ########################################
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        checkpoint_path = Path(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)

        state_path = Path(checkpoint_path) / "state"
        assert state_path.exists(), f"Checkpoint path {state_path} does not exist."
        logger.info(f"Resuming from checkpoint: {state_path}", main_process_only=True)
        accelerator.load_state(str(state_path))
        logger.info(f"Accelerator state loaded", main_process_only=True)

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = checkpoint_path.stem
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                    int(training_difference.replace("step_", ""))
                    * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    ######################################## Training Loop ########################################
    print(f"{accelerator.sync_gradients=}", flush=True)
    accelerator.wait_for_everyone()

    # TODO handle the eval after
    """    if accelerator.sync_gradients and not args.logging_examples_ignore_first:
        log_examples_to_wandb(completed_steps)"""

    torch.cuda.empty_cache()
    checkpoint_path = ""
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        epoch_data_count = 0  # To keep track of the number of data points in each epoch
        if (
                args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        """
        Algorithm: Generalized Knowledge Distillation (GKD)
        1: Given: Teacher model pT, Student Model pθS, Dataset (X, Y) containing (input, output) pairs
        2: Hyperparameters: Student data fraction λ ∈ [0, 1], Divergence D, Learning rate η
        3: for each step k = 1, . . . , K do
        4:   Generate a random value u ∼ Uniform(0, 1)
        5:   if u ≤ λ then
        6:       Sample inputs x from X and generate outputs y ∼ pθS(·|x) to obtain B = {(xb, yb)}_b=1
        7:   else
        8:       Sample batch of inputs and outputs from (X, Y) to obtain B = {(xb, yb)}_b=1.
        9:   end if
        10:  Update θ to minimize LGKD: θ ← θ - η * (1/B) * Σ_{(x,y)∈B} ∇θD(pT∥pθS)(y|x)
        11: end for
        """
        num_gpus = torch.cuda.device_count()  # Get the number of available GPUs (e.g., 8 GPUs)
        all_responses = []  # To store all responses across GPUs

        for step, batch in enumerate(active_dataloader):
            # Check if we want to use the student model's generated outputs (on-policy learning)
            if random.random() <= args.lambda_value:

                # Split the batch into 8 smaller batches, one for each GPU
                batch_size_per_gpu = len(batch["student_encoded"]["input_ids"]) // num_gpus
                small_batches = [
                    {
                        "input_ids": batch["student_encoded"]["input_ids"][i:i + batch_size_per_gpu],
                        "attention_mask": batch["student_encoded"]["attention_mask"][i:i + batch_size_per_gpu],
                    }
                    for i in range(0, len(batch["student_encoded"]["input_ids"]), batch_size_per_gpu)
                ]
                gpu_sentences_matrix =[[] for _ in range(num_gpus)]
                input_ids_len_matrix =[[] for _ in range(num_gpus)]
                # Prepare sampling params
                sampling_params = {
                    'max_new_tokens': 128,
                    'top_p': 0.9,
                    'temperature': 0.8,
                    'greedy': False,
                    'n_sample_per_prompt':1,
                    'use_vllm':True,
                    'is_lora': True,
                    'disable_sliding_window': True
                }
                sampling_params = Namespace(**sampling_params)

                # List of processes for parallel execution
                processes = []
                responses = [None] * num_gpus  # To store the responses from each GPU

                # Get the current timestamp to ensure file uniqueness
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                gpu_id= int(torch.cuda.current_device())
                print("executing ",gpu_id)

                gpu_sentences_matrix[gpu_id], input_ids_len_matrix[gpu_id] = convert_batches_to_sentences(tokenizer, [small_batches[gpu_id]])

                Path("pickles").mkdir(parents=True, exist_ok=True)
                # Create unique paths for pickle files for each GPU
                pickle_prompts_path = Path(f"pickles/prompts_{timestamp}_gpu{gpu_id}.pkl").absolute().as_posix()
                pickle_output_path = Path(f"pickles/output_{timestamp}_gpu{gpu_id}.pkl").absolute().as_posix()
                pickle_sampling_params_path = Path(f"pickles/sampling_params_{timestamp}_gpu{gpu_id}.pkl").absolute().as_posix()

                # Serialize the prompts and sampling parameters into pickle files
                with open(pickle_prompts_path, "wb") as f:
                    pickle.dump(gpu_sentences_matrix[gpu_id], f)

                with open(pickle_sampling_params_path, "wb") as f:
                    pickle.dump(sampling_params, f)

                # Run subprocess on the current GPU and get the results
                #todo get the results from subprocess - encode them and get the y+
                #todo crop if needed cases
                #calculate the losses
                """  
                p = subprocess.Popen(
                    ["python", "run_vllm_student.py", "--model_name_or_path", args.model_name_or_path,
                     "--lora_model_name_or_path", checkpoint_path,
                     "--tokenizer_name", args.tokenizer_name,
                     "--pickle_prompts", pickle_prompts_path,
                     "--pickle_sampling_params", pickle_sampling_params_path,
                     "--pickle_outputs",pickle_output_path,
                     #"--mem_util", str(getattr(sampling_params, "mem_util", 0.4)),
                     "--batch_size", batch_size_per_gpu],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                processes.append((p, gpu_id))
                

                # Wait for all processes to complete and gather their results
                for p, gpu_id in processes:
                    p.wait()  # Wait for the process to finish

                    # Read the responses from the output pickle file
                    with open(pickle_output_path, "rb") as f:
                        responses[gpu_id] = pickle.load(f)

                    # Optionally, clean up the pickle files after processing
                    Path(pickle_prompts_path).unlink()
                    Path(pickle_sampling_params_path).unlink()
                    Path(pickle_output_path).unlink()

                # Store the responses
                all_responses.extend(responses)

                generated_tokens = responses_log["input_ids"][:, original_input_length:]
                new_attention_mask = responses_log["attention_mask"][:, original_input_length:]

                # Use generated student outputs
                student_inputs = {
                    'input_ids': responses_log["input_ids"],
                    'attention_mask': responses_log["attention_mask"]  # New attention mask for generated tokens
                }
                
                # Keep teacher inputs from original data
                teacher_inputs = {
                    'input_ids': torch.cat([batch['teacher_input_ids'], generated_tokens], dim=1),
                    'attention_mask': torch.cat([batch['teacher_attention_mask'], new_attention_mask], dim=1)
                }
                """
            else:
                original_input_length = batch["attention_mask"].sum(dim=1)
                # Use original batch data for both student and teacher
                student_inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
                teacher_inputs = {
                    'input_ids': batch['teacher_input_ids'],
                    'attention_mask': batch['teacher_attention_mask']
                }

            # Forward pass for the student model
            with accelerator.accumulate(model):  # Accumulate gradients
                student_outputs = model(**student_inputs)

                # Forward pass for the teacher model (in evaluation mode)
                teacher_model.eval()
                # Get the teacher input and attention mask
                teacher_input_ids = batch['teacher_input_ids']
                teacher_attention_mask = batch['teacher_attention_mask']

                # Calculate the actual non-padding length of each sequence
                teacher_token_lengths = teacher_attention_mask.sum(
                    dim=1)  # Number of non-padding tokens for each sequence

                # Dynamically truncate the input sequences for the teacher model based on the non-padding length
                # We'll loop over the batch to truncate each sequence individually

                truncated_teacher_input_ids = []
                truncated_teacher_attention_masks = []

                for i in range(teacher_input_ids.size(0)):  # Iterate over the batch
                    seq_len = teacher_token_lengths[i].item()  # Get the non-padding length for this sequence
                    truncated_teacher_input_ids.append(teacher_input_ids[i, :seq_len])  # Truncate input_ids
                    truncated_teacher_attention_masks.append(
                        teacher_attention_mask[i, :seq_len])  # Truncate attention mask

                # Convert the lists back to tensors with dynamic batch sizes
                truncated_teacher_input_ids = torch.nn.utils.rnn.pad_sequence(truncated_teacher_input_ids,
                                                                              batch_first=True,
                                                                              padding_value=tokenizer.pad_token_id)
                truncated_teacher_attention_masks = torch.nn.utils.rnn.pad_sequence(truncated_teacher_attention_masks,
                                                                                    batch_first=True, padding_value=0)

                # Forward pass for the teacher model (in evaluation mode) using truncated inputs
                teacher_model.eval()
                with torch.no_grad():
                    teacher_outputs = teacher_model(**teacher_inputs)

                # Slice logits based on prompt lengths - to compare them over generated responses
                prompt_lengths_student = batch['input_ids'].shape[1]
                prompt_lengths_teacher = batch["teacher_input_ids"].shape[1]
                student_logits = student_outputs.logits[:, prompt_lengths_student - 1:-1,
                                 :]  # prompt_lengths_teacher - 1: -1, :]
                teacher_logits = teacher_outputs.logits[:, prompt_lengths_teacher - 1:-1,
                                 :]  # prompt_lengths_teacher - 1: -1, :]
                labels = batch['labels'][:, :prompt_lengths_teacher]  # [:, prompt_lengths_student:]

                # Compute the Generalized JSD loss
                loss = generalized_jsd_loss(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    beta=0.5,
                    temperature=1.0,
                )

                # Backpropagation and optimizer step
                accelerator.backward(loss)  # Use accelerator to handle distributed backward pass

                optimizer.step()
                optimizer.zero_grad()

                # Accumulate total loss and count for logging
                total_loss += loss.item()
                epoch_data_count += batch['input_ids'].size(0)

                # for cuda out of memory
                if step % 500 == 0:
                    torch.cuda.empty_cache()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(
                        total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    # logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    # Print number of examples processed so far in this epoch
                    # print(f"Step {completed_steps}: Processed {epoch_data_count} examples so far in Epoch {epoch + 1}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                        )

                    total_loss = 0

                """if args.logging_examples_steps and completed_steps % args.logging_examples_steps == 0:
                    log_examples_to_wandb(completed_steps)"""

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    output_dir = checkpointing_dir / f"step_{completed_steps}"
                    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

            print(f"Completed Epoch {epoch + 1}: Total processed examples = {epoch_data_count}")

        model.eval()
        eval_pbar = tqdm(test_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
        total_loss = 0.0
        total_num_labels = 0
        for step, batch in enumerate(eval_pbar):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            with torch.no_grad():
                outputs = model(**batch, use_cache=False)
                num_labels = batch["labels"].ne(-100).sum()
                loss = compute_loss(batch, outputs)
            batch_num_labels = accelerator.gather(num_labels.repeat(accelerator.num_processes)).sum().item()
            batch_loss = accelerator.gather(loss.repeat(accelerator.num_processes)).sum().item()
            total_loss += batch_loss
            total_num_labels += batch_num_labels
            eval_pbar.set_postfix({"loss": total_loss / total_num_labels})

        logs = {"eval_loss": total_loss / total_num_labels, "eval_ppl": math.exp(total_loss / total_num_labels)}
        if args.with_tracking:
            accelerator.log(logs)
        model.train()

        if args.checkpointing_steps == "epoch":
            output_dir = checkpointing_dir / f"epoch_{epoch}"
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)
            with open(output_dir / "logs.json", "w") as f:
                json.dump(logs, f)

    output_dir = checkpointing_dir / "final"
    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()


