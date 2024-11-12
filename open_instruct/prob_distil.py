# coding=utf-8
import copy
import os
import sys
import time
import html
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
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

import deepspeed
import wandb
import huggingface_hub
from tqdm import tqdm
from datasets import Dataset, DatasetDict

import transformers
from transformers import (
    get_scheduler,
)
import torch.nn.functional as F
import torch.multiprocessing as mp
import pickle
sys.path.append(Path(__file__).parents[1].absolute().as_posix())

from peft import PeftConfig, PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from att import apply_att_template, encode_with_chat_template, DataCollatorForATT, preprocess_for_symmetric_att

from load_utils import (add_common_training_args, pretty_print_chatml, preprocess_data_to_chatml, \
                        load_tokenizer_model, save_args, load_args, preprocess_hh_common, target_lora_modules)
from datetime import datetime


logger = get_logger(__name__)

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

def get_run_id(args: Namespace) -> str:
    round = args.round
    # try to get the git commit hash
    # try:
    #     git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    # except subprocess.CalledProcessError:
    #     git_commit = "unknown"

    # hashes are not the same across processes, no idea why
    # arg_set = frozenset(vars(args).items())
    # args_hash = abs(hash(arg_set))
    # print(f"arg_set: {frozenset} args_hash: {args_hash}")
    return f"{round}"
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
    parser.add_argument("--wandb_id", type=str, default=None, required=False)

    parser.add_argument(
        "--deepspeed_config_file", type=str, help="the path of deepspeed config file"
    )
    parser.add_argument(
        "--lambda_value", type=int, default=1, required=False
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, required=False
    )
    parser.add_argument(
        "--pickle_batch", type=str, required=False
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


def preprocess_for_att(example, tokenizer, max_seq_length, slack_len=70, debug_print=False,
                                 logger=None):
    example = deepcopy(example)
    yplus_ref_orig = encode_with_chat_template(example['chosen'], tokenizer, max_seq_length, debug_print, logger)
    yminus_ref_orig = encode_with_chat_template(example['rejected'], tokenizer, max_seq_length, debug_print, logger)

    yplus_tokens = tokenizer.encode(example['chosen'][-1]['content'], add_special_tokens=False)
    yminus_tokens = tokenizer.encode(example['rejected'][-1]['content'], add_special_tokens=False)

    prompt_len = (yplus_ref_orig['labels'] == -100).sum().item()
    yplus_len = len(yplus_tokens)
    yminus_len = len(yminus_tokens)

    if prompt_len >= max_seq_length:
        logger.warning(f"Prompt is too long: {prompt_len}")
        dummy_input = {
            'input_ids': torch.tensor([tokenizer.bos_token_id], dtype=torch.long),
            'labels': torch.tensor([-100], dtype=torch.long),
            'attention_mask': torch.tensor([1], dtype=torch.long),
        }

        return dummy_input

    remaining_budget = max_seq_length - prompt_len - slack_len
    yminus_len_cropped = min(yminus_len, remaining_budget // 3)
    if prompt_len + yminus_len + yplus_len + slack_len <= max_seq_length:
        # No cropping needed
        pass
    elif prompt_len + yminus_len_cropped + yplus_len + slack_len <= max_seq_length:
        # Only need to crop yminus
        yminus_len_necessary = max_seq_length - prompt_len - yplus_len - slack_len
        yminus_cropped_tok = yminus_tokens[:yminus_len_necessary]
        example['rejected'][-1]['content'] = tokenizer.decode(yminus_cropped_tok, skip_special_tokens=False)
    else:
        # Crop both yminus and yplus
        yminus_cropped_tok = yminus_tokens[:yminus_len_cropped]
        example['rejected'][-1]['content'] = tokenizer.decode(yminus_cropped_tok, skip_special_tokens=False)
        yplus_len_necessary = max_seq_length - prompt_len - yminus_len_cropped - slack_len
        yplus_cropped_tok = yplus_tokens[:yplus_len_necessary]
        example['chosen'][-1]['content'] = tokenizer.decode(yplus_cropped_tok, skip_special_tokens=False)

    yplus_att = apply_att_template(example, tokenizer, max_seq_length, debug_print, logger)

    if yplus_att["input_ids"].shape[0]> max_seq_length:
        return None

    return yplus_att

def apply_formatting_and_encoding(example, tokenizer, max_seq_length, debug_print=False, logger=None):
    # Student encoding
    student_encoded = encode_with_chat_template(example["chosen"], tokenizer, max_seq_length, debug_print, logger)

    # Teacher encoding
    teacher_encoded = preprocess_for_att(example, tokenizer, max_seq_length, debug_print, logger)

    # If any of the encodings return None, return None to indicate the sample should be skipped
    if student_encoded is None or teacher_encoded is None:
        if logger:
            logger.warning("Skipping example due to encoding failure.")
        return {
            'teacher_encoded': None,
            'student_encoded': None
        }

    # Return the valid encoding
    return {
        'teacher_encoded': teacher_encoded,
        'student_encoded': student_encoded
    }

def has_responses(processed_example):
    # First, check if processed_example is None
    if processed_example is None:
        return False

    # Ensure that none of the values in processed_example are None
    for e in processed_example.values():
        if e is None or "labels" not in e:
            return False
        # Check if the labels contain valid entries
        if (e["labels"] == -100).all():
            return False

    # If all checks pass, return True
    return True



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
        # Configure wandb logging within Accelerator
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
    )

    accelerator_teacher = Accelerator(
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs],
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

    #if it is the very first run of this script use load lora true
    #model is the student model
    args.first_run=True #TODO: change this
    if args.first_run:
        model, tokenizer, actual_eos_token, generation_config_sampling, generation_config_greedy \
            = load_tokenizer_model(None, args, substitute_eos_token=True, load_lora=True)
    else:
        model, tokenizer, actual_eos_token, generation_config_sampling, generation_config_greedy \
             = load_tokenizer_model(None, args, substitute_eos_token=True, load_lora=False)
        model = PeftModel.from_pretrained(model, str(args.student_lora_model_name_or_path))
    tokenizer.eos_token = actual_eos_token
    #model.generation_config = generation_config_greedy

    teacher_model = PeftModel.from_pretrained(model, str(args.teacher_lora_model_name_or_path))

    logger.info(str(model), main_process_only=True)
    """for name, param in model.named_parameters():
        print(f"Parameter: {name}, Data Type: {param.dtype}")"""

    checkpointing_dir = Path(args.output_dir)
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
                args.resume_from_checkpoint = str(last_checkpoint)
    logger.info(f"\n\n\nSaving checkpoints to {checkpointing_dir}\n\n\n")

    ######################################## Data Preprocessing ########################################
     #read pickle file as dataset
    with open(args.pickle_batch, 'rb') as f:
        dataset = pickle.load(f)

    if isinstance(dataset, dict):
        if isinstance(dataset.get("chosen"), list) and isinstance(dataset.get("rejected"), list):
            chosen_dataset = Dataset.from_dict({"chosen": dataset["chosen"]})
            rejected_dataset = Dataset.from_dict({"rejected": dataset["rejected"]})

            chosen_split = chosen_dataset.train_test_split(test_size=0.2, shuffle=False)
            rejected_split = rejected_dataset.train_test_split(test_size=0.2, shuffle=False)

            combined_train_data = {
                "chosen": chosen_split["train"]["chosen"],
                "rejected": rejected_split["train"]["rejected"]
            }

            combined_test_data = {
                "chosen": chosen_split["test"]["chosen"],
                "rejected": rejected_split["test"]["rejected"]
            }

            train_dataset = Dataset.from_dict(combined_train_data)
            test_dataset = Dataset.from_dict(combined_test_data)

            dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
        else:
            raise ValueError("Both 'chosen' and 'rejected' must be present as lists.")
    else:
        raise TypeError(f"Expected a DatasetDict-like object, but got {type(dataset)}.")

    encode_function = partial(
        apply_formatting_and_encoding,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        debug_print=False,
        logger=logger,
    )

    with accelerator.main_process_first():
        lm_datasets = dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in dataset["train"].column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )

        # Set format to pytorch tensors
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(has_responses)

    print("Size of training set:", len(dataset['train']))

    train_dataset = lm_datasets["train"]
    test_dataset = lm_datasets["test"]

    num_train_ex = accelerator.num_processes

    print("Size of training set:", len(train_dataset))
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
                                  init_kwargs={"wandb": {"entity": "claire-labo", "id":args.wandb_id, "resume":"allow"}})

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
    teacher_model = accelerator_teacher.prepare(teacher_model)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = num_update_steps_per_epoch  #num of epochs 1
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    ######################################## Training Loop ########################################
    print(f"{accelerator.sync_gradients=}", flush=True)
    accelerator.wait_for_everyone()

    #will be run only 1 epoch all the time  - REVIEW
    total_steps = args.max_train_steps
    progress_bar = tqdm(total=total_steps, desc="Training Progress")

    completed_steps = 0  # Initialize the completed steps counter
    total_loss = 0  # Initialize total loss for logging

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
    model.train()
    teacher_model.eval()
    for step, batch in enumerate(active_dataloader):
        # Use original batch data for both student and teacher
        student_inputs = {
            'input_ids': batch['student_encoded']['input_ids'],
            'attention_mask': batch['student_encoded']['attention_mask'],
            'labels': batch['student_encoded']['labels'],
        }

        # Extract teacher inputs
        teacher_inputs = {
            'input_ids': batch['teacher_encoded']['input_ids'],
            'attention_mask': batch['teacher_encoded']['attention_mask'],
            'labels': batch['teacher_encoded']['labels'],
        }

        # Forward pass for the student model
        with accelerator.accumulate(model):  # Accumulate gradients
            student_outputs = model(**student_inputs)

            #print(student_outputs.logits.requires_grad)
            # Forward pass for the teacher model (in evaluation mode) using truncated inputs
            with torch.no_grad():
                teacher_outputs = teacher_model(**teacher_inputs)

            # Slice logits based on prompt lengths - to compare them over generated responses
            prompt_lengths_student = student_inputs['input_ids'].shape[1]
            prompt_lengths_teacher = teacher_inputs["input_ids"].shape[1]
            # Step 1: Calculate softmax for both teacher and student logits
            student_logits = student_outputs.logits[:, :prompt_lengths_student]
            teacher_logits = teacher_outputs.logits[:, :prompt_lengths_teacher]

            # Ensure that the logits are the same shape for both student and teacher, if not, align them. - TODO
            min_length = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits = student_logits[:, :min_length]
            teacher_logits = teacher_logits[:, :min_length]

            # Step 2: Apply softmax to get probabilities
            student_probs = F.softmax(student_logits, dim=-1)
            teacher_probs = F.softmax(teacher_logits, dim=-1)

            if not student_probs.requires_grad:
                student_probs.requires_grad_()

            #   Step 3: Compute KL Divergence
            loss = F.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')

            # Backpropagation and optimizer step
            accelerator.backward(loss)  # Use accelerator to handle distributed backward pass

            optimizer.step()
            optimizer.zero_grad()

            # Accumulate total loss and count for logging
            total_loss += loss.detach().float()

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

            if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                output_dir = checkpointing_dir / f"step_{completed_steps}"
                save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

            if completed_steps >= args.max_train_steps:
                break


        accelerator.wait_for_everyone()
        torch.cuda.empty_cache()
        #TODO : how should we evaluate the performance since in this version there is no yplus_att etc.?
        """model.eval()
        eval_pbar = tqdm(test_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
        total_loss = torch.zeros((), device=accelerator.device, dtype=torch.bfloat16)
        total_yplus_ce = 0.0
        num_labels_per_proc = torch.zeros((), device=accelerator.device, dtype=int)
        total_sum_logs = None
        step = 0
        for step, batch in enumerate(eval_pbar):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

            # Encountered a random deepspeed bug, goes away if I don't add no_grad
            # with (torch.no_grad()):
            num_labels = batch["yplus_att"]["labels"].ne(-100).sum()
            loss, logs = compute_loss_att(accelerator, model, batch, args, eval=True, debug=False)
            loss /= accelerator.num_processes
            all_logs = gather_object((logs,))
            sum_logs = {k: sum(log[k] for log in all_logs) for k in all_logs[0]}
            total_sum_logs = sum_logs if total_sum_logs is None \
                else {k: (total_sum_logs[k] + sum_logs[k]) for k in sum_logs}
            for proc_log in all_logs:
                accelerator.log(proc_log)

            batch_loss = accelerator.gather(loss).sum().item()
            total_loss += batch_loss
            num_labels_per_proc += num_labels
            eval_pbar.set_postfix({"avg_loss": (total_loss / (step + 1)).item()})
        # logs = {"eval/avg_loss": (total_loss / (step + 1)).item()}
        logs = {}
        all_num_labels = accelerator.gather(num_labels_per_proc)
        total_num_labels = all_num_labels.sum().item()

        logs["eval/avg_loss_per_label"] = (total_loss / total_num_labels).item()
        logs["eval/avg_loss_per_entry"] = (total_loss / len(test_dataset)).item()
        logs["eval/ppl"] = np.exp(total_sum_logs["mlog_pi_t_yplus_sum"] / total_num_labels)
        logs["eval/total_num_labels"] = total_num_labels
        logs["eval/total_num_entries"] = len(test_dataset)

        logs = {**logs, **{f"eval/per_entry_{k}": total_sum_logs[k] / len(test_dataset) for k in total_sum_logs}}
        logs = {**logs, **{f"eval/per_label_{k}": total_sum_logs[k] / total_num_labels for k in total_sum_logs}}

        if args.with_tracking:
            accelerator.log(logs)
        model.train()"""

        if args.checkpointing_steps == "epoch":

            if epoch_name is not None:
                epoch_name=epoch_name+1
                output_dir = checkpointing_dir / f"epoch_{epoch_name}"
            else:
                output_dir = checkpointing_dir / f"epoch_{epoch}"

            print("saved to ", output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)
            """if accelerator.is_main_process:
                print(logs)
                with open(output_dir / "logs.json", "w") as f:
                    json.dump(logs, f)"""
    print("process is finished")
    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()


