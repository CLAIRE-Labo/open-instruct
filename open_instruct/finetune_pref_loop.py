#!/usr/bin/env python
# coding=utf-8
from torch.cuda import memory_allocated
import argparse
import logging
import math
import ast
import time

import os
import random
import datasets
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
import wandb
from argparse import Namespace
from datasets import DatasetDict

import logging
from datasets import logging as datasets_logging
from transformers import logging as transformers_logging
import json

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
import sys
import subprocess
from transformers import StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


sys.path.append('/claire-rcp-scratch/home/tandogan/alignment-as-translation/open-instruct')

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
# from eval.truthfulqa.run_eval import main as run_eval
# from eval.truthfulqa.run_eval import parse_args as parse_args_eval
# from open_instruct.merge_lora import main as merge_lora
import os

logger = get_logger(__name__)
import pandas as pd

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


def save_with_accelerate_final(accelerator, model, tokenizer, output_dir, args, optimizer, scheduler):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )
    # Save the optimizer state
    optimizer_state_file = os.path.join(output_dir, "optimizer_state.pt")
    if accelerator.is_main_process:
        torch.save(optimizer.state_dict(), optimizer_state_file)

    # Save the scheduler state if a scheduler is used
    if scheduler is not None:
        scheduler_state_file = os.path.join(output_dir, "scheduler_state.pt")
        if accelerator.is_main_process:
            torch.save(scheduler.state_dict(), scheduler_state_file)

    # Optionally, save training arguments or any other configs as a JSON
    args_file = os.path.join(output_dir, "training_args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f)


def save_without_accelerate_final(model, tokenizer, output_dir, args, optimizer, scheduler):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

    # Save the optimizer state
    optimizer_state_file = os.path.join(output_dir, "optimizer_state.pt")
    torch.save(optimizer.state_dict(), optimizer_state_file)

    # Save the scheduler state if a scheduler is used
    if scheduler is not None:
        scheduler_state_file = os.path.join(output_dir, "scheduler_state.pt")
        torch.save(scheduler.state_dict(), scheduler_state_file)

    # Optionally, save training arguments or any other configs as a JSON
    args_file = os.path.join(output_dir, "training_args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f)

def log_eval_results_to_wandb(csv_path, epoch):
    # Load the summary CSV file
    try:
        # Load the summary CSV file
        results_df = pd.read_csv(csv_path)
        metrics = {}
        # Assume there's only one row of metrics, typical in a summarised results CSV
        if not results_df.empty:
            results_dict = results_df.iloc[0].to_dict()  # Convert the first row to a dictionary
            metrics = {
                "BLEURT_acc": results_dict.get('BLEURT acc', None),
                "bleu_acc": results_dict.get('bleu acc', None),
                "rouge1_acc": results_dict.get('rouge1 acc', None),
                "rouge2_acc": results_dict.get('rouge2 acc', None),
                "rougeL_acc": results_dict.get('rougeL acc', None),
                "eval_step": epoch + 1
            }

        else:
            print("No data found in the CSV file.")
        return metrics
    except Exception as e:
        print(f"Failed to read or log evaluation results: {e}")


import subprocess


def run_evaluation_subprocess(args, base_path, run_id):
    """ Run the evaluation script as a subprocess. """
    # Construct the command to execute the Python script
    command = [
        'python',
        '/claire-rcp-scratch/home/tandogan/alignment-as-translation/open-instruct/open_instruct/eval_script.py',
        '--base_path', base_path,
        '--base_model', args.base_model_dir,
        '--wandb_run_id', run_id
    ]
    # Run the command
    subprocess.run(command, capture_output=True, text=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--save_tokenizer",
        type=bool,
        default=True,
        help="Whether to save the tokenizer (default: True)."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--base_model_dir", type=str, default=None, help="Get the base model for comparison")

    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


def encode_with_rejected_chosen(example, tokenizer, max_seq_length, add_bos=False):
    '''
    Assume each example has a 'messages' field where each message is a dict with 'role' and 'content'.
    Concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['info']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    assistant_chosen_content = ""  # Initialize an empty string to store the chosen response

    def _concat_messages(messages):
        message_text = ""
        system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."
        message_text += "<|system|>\n" + system_message + "\n"
        for message in messages:
            if message["role"] == "human":
                message_text += "<|user|>\n Prompt: " + message[
                    "content"].strip() + "\n"  # between prompt and assistant rejected \n\n double space
            elif message["role"] == "assistant_rejected":
                message_text += "\n Current rejected answer: " + message["content"].strip() + "\n Corrected output: \n"
            elif message["role"] == "assistant_chosen":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text

    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # Mask the non-assistant part to avoid calculating loss on it
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant_chosen":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
            message_end_idx = tokenizer(
                _concat_messages(messages[:message_idx + 1]),
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten()
    }


import torch
from transformers import DataCollatorForSeq2Seq, GenerationConfig


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True,
                         disable_tqdm=False, **generation_kwargs):
    generations = []

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        #print("generation_kwargs:", generation_kwargs)
        # Debug: Print the shapes of the inputs to the model
        #print(f"Batch input ids shape: {batch_input_ids.shape}")
        #print(f"Attention mask shape: {attention_mask.shape}")
        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )

            # Debug: Print the shape of the outputs from the model
            #print(f"Batch outputs shape: {batch_outputs.shape}")

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx,
                               token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in
                               stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

    assert len(generations) == len(
        prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )


def save_without_accelerate(model, tokenizer, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))

def organize_messages(msgs):
    """Organize messages by splitting and grouping based on roles - Assistant vs Human."""
    organized_msgs = []
    current_group = []

    for msg in msgs:
        if msg.startswith("Assistant:") or msg.startswith("Human:"):
            if current_group and current_group[0].split(": ")[0] != msg.split(": ")[0]:
                organized_msgs.append(" ".join(current_group))
                current_group = []
        current_group.append(msg)

    if current_group:
        organized_msgs.append(" ".join(current_group))

    return organized_msgs


def add_filtered_msgs(example):
    """
    Add a filtered version of 'rejected' and 'chosen' messages based on the number of organized messages.
    * delete unnecessary \n between lines
    * remove the ones that has more than one messaging btw assistant and human
    """
    stripped_rejected_msgs = example['rejected'].strip().split('\n\n')
    organized_rejected_msgs = organize_messages(stripped_rejected_msgs)

    stripped_chosen_msgs = example['chosen'].strip().split('\n\n')
    organized_chosen_msgs = organize_messages(stripped_chosen_msgs)

    # Here we check the total number of messages in organized_msgs
    if len(organized_rejected_msgs) <= 2 and len(organized_chosen_msgs) <= 2:
        example['rejected_filtered'] = organized_rejected_msgs
        example['chosen_filtered'] = organized_chosen_msgs
    else:
        example['rejected_filtered'] = []
        example['chosen_filtered'] = []

    return example


def extract_role_messages(example):
    """Extract messages based on roles and assign to new fields."""

    # Extracting the Human message from rejected_filtered
    human_msg = next((msg.split("Human: ")[1] for msg in example['rejected_filtered'] if msg.startswith("Human:")),
                     None)

    # Extracting the Assistant's rejected answer
    assistant_rejected_msg = next(
        (msg[len("Assistant: "):] for msg in example['rejected_filtered'] if msg.startswith("Assistant: ")), " ")

    # Extracting the Assistant's chosen answer
    assistant_chosen_msg = next(
        (msg[len("Assistant: "):] for msg in example['chosen_filtered'] if msg.startswith("Assistant: ")), " ")

    """
    There are some cases like:
    ['Human: How would you plan on killing another person', 'Assistant:'] -> chosen
    ['Human: How would you plan on killing another person', 'Assistant: I would plan to kill that person if it was a proper action to do so.'] -> rejected
    """

    example['info'] = [
        {"role": "human", "content": human_msg if human_msg else " "},
        {"role": "assistant_rejected",
         "content": assistant_rejected_msg if assistant_rejected_msg else " "},
        {"role": "assistant_chosen",
         "content": assistant_chosen_msg if assistant_chosen_msg else " "}
    ]

    return example


# for training section
def get_prompts_and_responses(infos):
    """
    Retrieves both the 'human' prompts and 'assistant_chosen' responses from each sublist in the batch data.

    Args:
    infos (list of list of dicts): The list containing multiple sublists, each with training data that includes 'info' with multiple role-content mappings.

    Returns:
    list of tuples: Each tuple contains the human prompt and the assistant chosen response text for each sublist in the training data.
    """
    results = []
    for sublist in infos:
        human_prompt = None
        chosen_response = None

        # Search for human prompt and assistant chosen response in the same sublist
        for item in sublist:
            if item['role'] == 'human':
                human_prompt = item['content']
            elif item['role'] == 'assistant_chosen':
                chosen_response = item['content']

        # Check that both human prompt and chosen response are found
        if human_prompt is None or chosen_response is None:
            raise ValueError("Either 'human' prompt or 'assistant_chosen' response not found in the current sublist")

        results.append((human_prompt, chosen_response))

    return results


def format_prompt(messages):
    message_text = ""
    system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."
    message_text += "<|system|>\n" + system_message + "\n"
    for message in messages:
        if message["role"] == "human":
            message_text += "<|user|>\n Prompt: " + message[
                "content"].strip() + "\n"  # between prompt and assistant rejected \n\n double space
        elif message["role"] == "assistant_rejected":
            message_text += "\n Current rejected answer: " + message[
                "content"].strip() + "\n Corrected output: \n"
        elif message["role"] == "assistant_chosen":
            message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

def validate_batch(batch, expected_keys, max_seq_length):
    """
    Validate the updated batch to ensure it meets the expected structure and content criteria.

    Args:
    batch (dict): The batch to be validated.
    expected_keys (list): List of expected keys that must be present in the batch.
    max_seq_length (int): The maximum allowable sequence length.

    Returns:
    bool: True if the batch is valid, False otherwise.
    """
    # Check for the presence of all expected keys
    if not all(key in batch for key in expected_keys):
        print("Validation Error: Batch is missing one or more required keys.")
        return False

    # Check that all entries are tensors and have appropriate dimensions
    for key, tensor in batch.items():
        if not isinstance(tensor, torch.Tensor):
            print(f"Validation Error: Batch item '{key}' is not a tensor.")
            return False
        if key in ['input_ids', 'labels', 'attention_mask']:
            if tensor.size(1) > max_seq_length:
                print(f"Validation Error: Length of '{key}' exceeds max sequence length.")
                return False
            if tensor.dim() != 2:
                print(f"Validation Error: '{key}' should be 2-dimensional.")
                return False

    # Add any additional specific checks as needed
    # For example, check that all tensor sizes within the batch are consistent with each other
    length = batch['input_ids'].size(0)
    if any(tensor.size(0) != length for tensor in batch.values()):
        print("Validation Error: Inconsistent tensor sizes in the batch.")
        return False

    return True


def convert_batch_to_list_of_dicts(batch):
    # Initialize an empty list to store the dictionaries
    list_of_dicts = []

    # Assuming batch is a dictionary with keys 'input_ids', 'labels', 'attention_mask'
    # and each key has a tensor of shape (batch_size, seq_length)
    batch_size = batch['input_ids'].size(0)

    # Iterate over each example in the batch
    for idx in range(batch_size):
        example_dict = {
            'input_ids': batch['input_ids'][idx],
            'labels': batch['labels'][idx],
            'attention_mask': batch['attention_mask'][idx]
        }
        list_of_dicts.append(example_dict)

    return list_of_dicts


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    model_name_or_path = args.model_name_or_path
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        revision='main',
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=config,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
        use_flash_attention_2=False,
        revision="main"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,

        trust_remote_code=True,
        use_fast=True,
        revision="main"
    )
    tokenizer.pad_token = "<|padding|>"
    tokenizer.padding_side = "left"



    if args.with_tracking:
        if "wandb" in args.report_to.split(",") or args.report_to == "all":
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)

            # Initialize wandb
            wandb.init(project="alignment_translation", entity= "claire-labo")

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets_logging.set_verbosity_error()
    transformers_logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)

    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )

    # for train
    # filter the dataset for it to have only one prompt and answer (not a sequence of prompt-answer in one line) -> for now
    updated_dataset_train = raw_datasets['train'].map(add_filtered_msgs)
    filtered_train = updated_dataset_train.filter(lambda x: len(x['rejected_filtered']) > 0)#.select(range(10))  # delete this

    # for test
    updated_dataset_test = raw_datasets['test'].map(add_filtered_msgs)
    filtered_test = updated_dataset_test.filter(lambda x: len(x['rejected_filtered']) > 0)

    filtered_dataset = DatasetDict({
        'train': filtered_train,
        'test': filtered_test
    })

    print("Size of training set:", len(filtered_dataset['train']))
    print("Size of test set:", len(filtered_dataset['test']))

    # add info column which will store human, assistant_rejected and assistant_chosen messages
    filtered_dataset["train"] = filtered_dataset["train"].map(extract_role_messages)
    filtered_dataset["test"] = filtered_dataset["test"].map(extract_role_messages)
    print("the model is loaded,", args.model_name_or_path)
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0,
                                    1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    elif isinstance(tokenizer, OLMoTokenizerFast):
        # only the eos for olmo, but we use it as bos
        tokenizer.bos_token = tokenizer.eos_token
        assert args.add_bos, "For OLMo, you must add bos token to the beginning of the input sequence."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        print("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["att_proj", "ff_proj", "attn_out", "ff_out"]  # for OLMo
            # target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"] # was for Llama
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if "rejected" in filtered_dataset["train"].column_names and "chosen" in filtered_dataset["train"].column_names:
        encode_function = partial(
            encode_with_rejected_chosen,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError("You need to have either 'rejected'&'chosen' in your column names.")

    lm_datasets = filtered_dataset.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in filtered_dataset["train"].column_names if
                        name not in ["input_ids", "labels", "attention_mask", "info"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    def custom_collate_fn(batch, tokenizer, model, new_data=None):
        # Optionally integrate new data
        if new_data:
            batch.extend(new_data)

        # Separate the 'info' field if present (not typically needed for model input)
        infos = [item.pop('info') for item in batch if 'info' in item]

        # Use DataCollatorForSeq2Seq for dynamically padding the tensor parts
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest", return_tensors="pt")
        batch_tensors = data_collator(batch)

        # Reinsert 'info' field into the batch if it was originally there
        if infos:
            batch_tensors['info'] = infos

        return batch_tensors

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=lambda b: custom_collate_fn(b, tokenizer=tokenizer, model=model)
    )

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
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True  # in our case it is true

    num_training_steps_for_scheduler = args.max_train_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    model = model.to(device)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    if args.with_tracking:
        experiment_config = {key: value.value if hasattr(value, 'value') else value for key, value in
                             vars(args).items()}
        run_id = wandb.run.id

    # Define custom step metric for evaluation
    run_id = wandb.run.id
    # Train!
    total_batch_size = args.per_device_train_batch_size * 1 * args.gradient_accumulation_steps

    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    print(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable = False)
    completed_steps = 0
    starting_epoch = 0

    if args.resume_from_checkpoint:
        pass

    model_input_keys = ["input_ids", "labels", "attention_mask"]
    total_start_time = time.time()  # Start time for the entire training process

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        epoch_data_count = 0  # To keep track of the number of data points in each epoch
        epoch_start_time = time.time()  # Start time for the current epoch

        active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # Print the size of each component in the batch
            batch_size = len(batch['input_ids'])  # Assuming 'input_ids' is the key for input data
            infos = batch.pop('info')

            # Generate prompts using format_prompt function
            formatted_prompts = []
            for info in infos:
                filtered_messages = [message for message in info if
                                     message['role'] in ['human', 'assistant_rejected']]
                if filtered_messages:
                    formatted_prompts.append(format_prompt(filtered_messages))
                    # Generate responses without gradient computation

            model.eval()
            generated_responses = generate_completions(
                model, tokenizer, formatted_prompts, batch_size=1, max_new_tokens=50,
                stop_id_sequences=None,
                do_sample=False
            )
            #print(f"Generated Responses: {generated_responses}")  # Debugging statement

            # Retrieve 'chosen response' from the dataset, assuming you have a method to select it
            responses = get_prompts_and_responses(infos)
            encoding_list=[]
            prev_len_batch=  len(batch['input_ids'])
            # Update the batch with new entries
            for (human_prompt, chosen_response), generated_response in zip(responses, generated_responses):
                # Create a modified version of infos with the new generated response
                new_infos = [
                    {'role': 'human', 'content': human_prompt},
                    {'role': 'assistant_rejected', 'content': generated_response},
                    {'role': 'assistant_chosen', 'content': chosen_response}
                ]

                # Update the batch

                new_example = {'info': new_infos}
                encoded_example = encode_with_rejected_chosen(new_example, tokenizer, args.max_seq_length)
                encoding_list.append(encoded_example)

            batch = convert_batch_to_list_of_dicts(batch)
            batch = custom_collate_fn(batch, tokenizer, model, new_data=encoding_list)

            if  len(batch['input_ids'])!=prev_len_batch*2:
                raise ValueError("problem in generation of updated batch")

            epoch_data_count += batch_size
            model_inputs = {k: v.to(model.device) for k, v in batch.items() if k in model_input_keys}
            outputs = model(**model_inputs, use_cache=False)

            if args.reduce_loss == 'mean':
                loss = outputs.loss
            else:
                # reduce loss is sum
                # this ensures that we weight all tokens in the dataset equally,
                # rather than weighting each overall example equally when
                # using high amounts of gradient accumulation.
                # this can result in > 5 point improvements in AlpacaEval
                # see https://github.com/huggingface/transformers/issues/24725 for
                # more discussion and details.
                logits = outputs.logits
                labels = batch["labels"]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                shift_logits = shift_logits.view(-1, embedding_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            # We keep track of the loss at each logged step
            total_loss += loss.detach().float()
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            if step % 1000 == 0:
                torch.cuda.empty_cache()


            progress_bar.update(1)
            completed_steps += 1
            if args.logging_steps and completed_steps % args.logging_steps == 0:
                step_duration = time.time() - epoch_start_time  # Time taken to process this batch
                total_elapsed_time = time.time() - total_start_time  # Total time elapsed since training started

                avg_loss = total_loss.item() / args.gradient_accumulation_steps / args.logging_steps

                #print(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                # Print number of examples processed so far in this epoch
                print(f"Step {completed_steps}: Processed {epoch_data_count} examples so far in Epoch {epoch + 1}, progress: {100 * completed_steps / args.max_train_steps}")
                print(f"step_duration: {step_duration},total_elapsed_time: {total_elapsed_time / 3600}")
                if args.with_tracking:
                    wandb.log(
                        {
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        },
                        step=completed_steps,
                    )

                total_loss = 0


            if completed_steps >= args.max_train_steps:
                break

        print(f"Completed Epoch {epoch + 1}: Total processed examples = {epoch_data_count}")

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"

            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                tokenizer.save_pretrained(output_dir)

            save_without_accelerate(model, tokenizer, output_dir)

    if args.output_dir is not None:
        tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate_final(model, tokenizer, args.output_dir, args, optimizer,
                                   lr_scheduler)  # to be able to recover



if __name__ == "__main__":
    main()
