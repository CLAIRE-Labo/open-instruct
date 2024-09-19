#!/usr/bin/env python
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

sys.path.append(Path(__file__).parents[1].absolute().as_posix())

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from utils import add_common_training_args, pretty_print_chatml, preprocess_hh_common, load_tokenizer_model
from att import apply_att_template

from constants import BAD_MISTRAL_CHAT_TEMPLATE, ATT_SYSTEM_PROMPT, ATT_TEMPLATE
from eval.utils import KeyWordsCriteria


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

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True,
                         disable_tqdm=False, generation_storage=None, **generation_kwargs):
    """Generate text completions with optimizations for GPU usage and batched JSON writing."""
    # Ensure the model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generations = []
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    all_json_entries = []

    # Using tqdm for progress tracking
    progress_bar = tqdm(range(0, len(prompts), batch_size), disable=disable_tqdm, desc="Generating batches")

    for i in progress_bar:
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize and move to GPU
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids.to(device)
        attention_mask = tokenized_prompts.attention_mask.to(device)

        batch_outputs = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
            **generation_kwargs
        )

        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx,
                           token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in
                           stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]
        generations += batch_generations

        # Store results for JSON writing
        json_entry = {
            "batch_index": i // batch_size,
            "batch_prompts": batch_prompts,
            "batch_generations": batch_generations
        }
        all_json_entries.append(json_entry)

        # Write to JSON every 100 batches
        if len(all_json_entries) >= 100:
            with open(generation_storage, 'a') as f:
                for entry in all_json_entries:
                    f.write(json.dumps(entry) + "\n")
            all_json_entries.clear()  # Clear the list after writing

    # Write any remaining entries in all_json_entries to the file
    if all_json_entries:
        with open(generation_storage, 'a') as f:
            for entry in all_json_entries:
                f.write(json.dumps(entry) + "\n")

    assert len(generations) == len(
        prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"

    return generations

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
        help="If set, the new token template is used for ATT taning."
    )
    parser.add_argument(
        "--teacher_tokenizer_name",
        type=str,
        help="tokenizer name for the teacher",
        required=False,
    )
    parser.add_argument(
        "--generated_data_path",
        type=str,
        help="collected/generated data from the base sft model for its use in teacher model",
        required=False,
    )

    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        help="Path to teacher finetuned model ",
        required=False,
    )

    parser.add_argument(
        "--generation_storage", type=str, default="data_sft.json", help="A json file will store the generated data."
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


def main():
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

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))


    # initialize student model before the accelerate - so that we can get inferences for the teacher model as rejected answers (just for inference)
    model, tokenizer,actual_eos_token, _, _ = load_tokenizer_model(accelerator=None, args=args)
    tokenizer.eos_token=actual_eos_token


    # teacher model
    # for teacher we need to manipulate the args
    teacher_args=copy.deepcopy(args)
    teacher_args.model_name_or_path= args.teacher_model_name_or_path
    teacher_args.tokenizer_name=args.teacher_tokenizer_name
    teacher_model, teacher_tokenizer, actual_eos_token, _, _ = load_tokenizer_model(accelerator=None, args=teacher_args)
    teacher_tokenizer.eos_token=actual_eos_token

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
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
    dataset_train, dataset_test = preprocess_hh_common(args)
    for i in range(10):
        logger.info(f"\n\nExample {i} chosen:\n{pretty_print_chatml(dataset_train[i]['chosen'])}\n\n"
                    f"Example {i} rejected:\n{pretty_print_chatml(dataset_train[i]['rejected'])}\n\n")
        apply_att_template(dataset_test[i], tokenizer, args.max_seq_length, debug_print=True, logger=logger)

    filtered_dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    student_msgs = []
    for example in filtered_dataset["train"]:
        content=example["rejected"][-2]["content"]
        msg= [{"role":"user", "content": content}]
        student_msgs.append(tokenizer.apply_chat_template(conversation=msg, add_generation_prompt=True, max_length= args.max_seq_length, tokenize=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.generated_data_path:
        generated_responses=[]
        with open(args.generated_data_path,"r") as file:
            for line in file:
                try:
                    data=json.loads(line.strip())
                    generated_responses.extend(data.get("batch_generations", []))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line due to JSONDecodeError:{e}")
        print("Already generated answers are taken, ex:", generated_responses[0])
    else:
        generated_responses=generate_completions(model, tokenizer, student_msgs, batch_size=1, max_new_tokens=300, disable_tqdm=False,
                                                 generation_storage= args.generation_storage, stop_id_sequences=None, do_sample=False)
    del model #which is not capable of using accelerate, just for generate_completions

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()



    for idx, example in enumerate(filtered_dataset["train"]):
        # Check if the example contains "rejected" and it's non-empty
        if "rejected" in example and len(example["rejected"]) > 0:
           # Access the last item in the "rejected" field
               last_message = example["rejected"][-1]

               # Check if the role is "assistant"
               if last_message["role"] == "assistant":
                 # Replace the content with the corresponding generated response
                 last_message["content"] = generated_responses[idx]



    # get the model with accelerator
    model, tokenizer, actual_eos_token, generation_config_nucleus, generation_config_greedy= load_tokenizer_model(accelerator, args)

    encode_function = partial(
        apply_att_template,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        logger=logger,
    )
    with accelerator.main_process_first():
        lm_datasets = filtered_dataset.map(
                 encode_function,
                 batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[name for name in filtered_dataset["train"].column_names if
                            name not in ["input_ids", "teacher_input_ids", "attention_mask", "teacher_attention_mask"]],
                desc="Tokenizing and reformatting instruction data",)

        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

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
    for index, ex in enumerate(train_examples):
        logger.info(f"Sample {index} of the training set: {ex}.")

    ######################################## Training Setup ########################################
    # DataLoaders creation:
    def get_dataloader(dataset, batch_size):
        return DataLoader(
            dataset,
            shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=batch_size
        )

    train_dataloader = get_dataloader(train_dataset, args.per_device_train_batch_size)
    test_dataloader = get_dataloader(test_dataset, args.per_device_train_batch_size)

    # Optimizer
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

    # Scheduler and math around the number of training steps.
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
    test_dataloader = accelerator.prepare(test_dataloader)

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

    # def get_cur_wandb_step():
    #     return accelerator.get_tracker("wandb").current_step

    def log_examples_to_wandb(step: int):
        def gen_examples(example, example_idx: int, gen_config: GenerationConfig) -> Tuple[Tuple[float, str, str, str]]:
            start_time = time.time()
            is_prompt = example["labels"] == -100
            is_response = ~is_prompt
            prompt_ids = example["input_ids"][is_prompt]
            # if prompt_ids.shape[0] >= args.logging_examples_max_length:
            #     logger.warning(
            #         f"Skipping logging of example {example_idx} because it is too long to log. "
            #         f"Length: {prompt_ids.shape[0]}. Max length: {args.logging_examples_max_length}"
            #     )
            #     return

            expected_response_ids = example["input_ids"][is_response]
            prompt_text = tokenizer.decode(prompt_ids)
            expected_response_text = tokenizer.decode(expected_response_ids)

            # unwrapped = accelerator.unwrap_model(model)
            # outputs = unwrapped.generate(input_ids=prompt_ids.to(model.device)[None, :],
            #                              generation_config=generation_config)

            outputs = model.generate(input_ids=prompt_ids.to(model.device)[None, :],
                                     generation_config=gen_config, tokenizer=tokenizer,
                                     synced_gpus=True)

            response_text = tokenizer.decode(outputs[0])
            if response_text[:len(prompt_text)] != prompt_text:
                logger.warning(
                    f"Response does not start with the prompt for example {example_idx}. "
                    f"Prompt: \n\"\"\"\n{prompt_text}\n\"\"\"\n. Response: \n\"\"\"\n{response_text}\n\"\"\"\n"
                )
            else:
                response_text = response_text[len(prompt_text):]

            # We updated the eos token to avoid the bugs in model.generate, so we have to truncate the output manually
            if actual_eos_token is not None and actual_eos_token in response_text:
                response_text = response_text.split(actual_eos_token)[0] + actual_eos_token

            # accelerator.get_tracker("wandb", unwrap=True).log(
            # wandb.log(
            #     {f"{prefix}_example_{example_idx + 1}/response": wandb.Html(f"<p>{html.escape(response_text)}</p>")})
            return ((time.time() - start_time, prompt_text, expected_response_text, response_text),)

        model.eval()
        sum_time = 0.0

        def gather_and_log(prefix, examples):
            nonlocal sum_time

            logger.info(f"Logging examples for {prefix}", main_process_only=True)
            logged_greedy = gen_examples(examples[accelerator.process_index], accelerator.process_index,
                                         generation_config_greedy)
            logged_nucleus = gen_examples(examples[accelerator.process_index], accelerator.process_index,
                                          generation_config_nucleus)
            logger.info(f"Logged examples for {prefix}", main_process_only=True)

            all_logged_greedy = gather_object(logged_greedy)
            all_logged_nucleus = gather_object(logged_nucleus)
            for i, (lg, ln) in enumerate(zip(all_logged_greedy, all_logged_nucleus)):
                tm_g, prompt, expected, actual_g = lg
                tm_n, _, _, actual_n = ln
                sum_time += tm_g + tm_n
                accelerator.log(
                    {f"{prefix}_example_{i + 1}/prompt": wandb.Html(f"<p>{html.escape(prompt)}</p>"),
                     f"{prefix}_example_{i + 1}/expected_response": wandb.Html(f"<p>{html.escape(expected)}</p>"),
                     f"{prefix}_example_{i + 1}/response_greedy": wandb.Html(f"<p>{html.escape(actual_g)}</p>"),
                     f"{prefix}_example_{i + 1}/response_nucleus": wandb.Html(f"<p>{html.escape(actual_n)}</p>"), }
                )

        gather_and_log("train", train_examples)
        gather_and_log("test", test_examples)

        model.train()
        accelerator.wait_for_everyone()
        print(f"Finished evals! pidx={accelerator.process_index}")

        accelerator.log(
            {"log_examples_time": sum_time,
             "avg_example_time": sum_time / 2 / (len(train_examples) + len(test_examples))})

    ######################################## Training Loop ########################################
    print(f"{accelerator.sync_gradients=}", flush=True)
    accelerator.wait_for_everyone()

    def compute_loss_distill(student_logits, teacher_logits, mode='student_teacher'):

        teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

        if mode == 'student_teacher':
            # Compute the softmax of student logits
            student_probs = torch.nn.functional.softmax(student_logits, dim=-1)

            # Compute KL Divergence between teacher and student distributions
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                teacher_probs,
                reduction='batchmean'
            )
        elif mode == 'teacher_only':
            # Use teacher probabilities directly and compute KL divergence
            loss = torch.nn.functional.kl_div(
                teacher_probs.log(),
                teacher_probs,
                reduction='batchmean'
            )

        return loss
    if accelerator.sync_gradients and not args.logging_examples_ignore_first:
        log_examples_to_wandb(completed_steps)
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
        # Main loop with compute_loss_distill integration
        for step, batch in enumerate(active_dataloader):
            # Student inputs
            student_inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }

            # Teacher inputs
            teacher_inputs = {
                'input_ids': batch['teacher_input_ids'],
                'attention_mask': batch['teacher_attention_mask']
            }

            # Random decision for lambda threshold
            u = random.uniform(0, 1)
            batch_size = len(batch['input_ids'])
            epoch_data_count += batch_size

            # Accumulate gradients
            with accelerator.accumulate(model):
                if u <= args.lambda_value:
                    # Forward pass for student model
                    student_outputs = model(**student_inputs, use_cache=False)
                    student_logits = student_outputs.logits

                    # Forward pass for teacher model without gradient
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**teacher_inputs, use_cache=False)
                        teacher_logits = teacher_outputs.logits

                    # Compute the student-teacher distillation loss
                    loss = compute_loss_distill(student_logits, teacher_logits, mode='student_teacher')

                else:
                    # Forward pass for teacher model only
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**teacher_inputs, use_cache=False)
                        teacher_logits = teacher_outputs.logits

                    # Compute loss based on teacher probabilities alone
                    loss = compute_loss_distill(None, teacher_logits, mode='teacher_only')

                # Track the total loss
                total_loss += loss.detach().float()

                # Backpropagation
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Scheduler step
                lr_scheduler.step()

                # for cuda out of memory
                if step % 2000 == 0:
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

                if args.logging_examples_steps and completed_steps % args.logging_examples_steps == 0:
                    log_examples_to_wandb(completed_steps)

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    output_dir = checkpointing_dir / f"step_{completed_steps}"
                    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        print(f"Completed Epoch {epoch + 1}: Total processed examples = {epoch_data_count}")

        """model.eval()
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
            accelerator.log(logs)"""
        model.train()

        if args.checkpointing_steps == "epoch":
            output_dir = checkpointing_dir / f"epoch_{epoch}"
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)
            """with open(output_dir / "logs.json", "w") as f:
                json.dump(logs, f)"""

    output_dir = checkpointing_dir / "final"
    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()