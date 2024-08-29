#!/usr/bin/env python
# coding=utf-8
import os
import sys
import time
import html
from heapq import merge
from typing import Dict, List
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
from accelerate.utils import set_seed, InitProcessGroupKwargs
import datasets
from datasets import load_dataset
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
from utils import add_common_training_args, parse_entry_anthropic_hh, merge_consecutive_messages_and_trim, \
    is_entry_ok_anthropic_hh, pretty_print_chatml

from constants import BAD_MISTRAL_CHAT_TEMPLATE, ATT_SYSTEM_PROMPT, ATT_TEMPLATE

# from eval.truthfulqa.run_eval import main as run_eval
# from eval.truthfulqa.run_eval import parse_args as parse_args_eval

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256" - Cuda out of memory


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


def run_evaluation_subprocess(args, base_path, run_id, tokenizer):
    """ Run the evaluation script as a subprocess. """
    command = [
        'python', str(Path(__file__).parent / 'eval_script.py'),
        '--base_path', base_path,
        '--base_model', args.base_model_dir,
        '--wandb_run_id', run_id,
    ]
    if isinstance(tokenizer, (LlamaTokenizer, LlamaTokenizerFast)):
        # If tokenizer is an instance of LlamaTokenizer or LlamaTokenizerFast, use phi3 chat format
        chat_format = "eval.templates.create_prompt_with_phi3_chat_format"
        command.extend(['--chat_formatting_function', chat_format])
    # Run the command
    subprocess.run(command, capture_output=True, text=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    add_common_training_args(parser)
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
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['info']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages):
        message_text = ""
        # system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."
        system_message = ATT_SYSTEM_PROMPT
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            message_text += "<|system|>\n" + system_message + '<|end|>' + "\n"
            for message in messages:
                if message["role"] == "human":
                    # between prompt and assistant rejected \n\n double space
                    message_text += "<|user|>\n Prompt: " + message["content"].strip() + '<|end|>' + "\n"
                elif message["role"] == "assistant_rejected":
                    message_text += "\n Current rejected answer: " + message["content"].strip() \
                                    + "\n Corrected output: \n" + '<|end|>' + "\n"
                elif message["role"] == "assistant_chosen":
                    message_text += "<|assistant|>\n" + message["content"].strip() + '<|end|>' + "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
        else:
            message_text += "<|system|>\n" + system_message + "\n"
            for message in messages:
                if message["role"] == "human":
                    # between prompt and assistant rejected \n\n double space
                    message_text += "<|user|>\n Prompt: " + message["content"].strip() + "\n"
                elif message["role"] == "assistant_rejected":
                    message_text += "\n Current rejected answer: " + message[
                        "content"].strip() + "\n Corrected output: \n"
                elif message["role"] == "assistant_chosen":
                    message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    # print(example_text) #to be commented out

    if add_bos:
        example_text = tokenizer.bos_token + example_text

    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant_chosen":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant_chosen":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx + 1])
            message_end_idx = tokenizer(
                messages_so_far,
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
        'attention_mask': attention_mask.flatten(),
    }


# OLMo chat template handles the BOS token, so we don't need to fiddle with add_bos.
# See https://huggingface.co/allenai/OLMo-7B-Instruct/commit/f09de2dc46d1e848f19dd7161bd998973e2b1272
def apply_att_template(example, tokenizer, max_seq_length, debug_print=False):
    chosen = example['chosen']
    rejected = example['rejected']

    assert len(chosen) == len(rejected), \
        f"Chosen and rejected should have the same length, got {len(chosen)} and {len(rejected)}.\n" \
        f"Chosen:\n{pretty_print_chatml(chosen)}\nRejected:\n{pretty_print_chatml(rejected)}"

    assert len(chosen) > 1, f"Chosen and rejected should have at least 2 messages, got {len(chosen)}"
    # All messages but the last one should be identical, like in the Anthropic HH data
    for i, (mc, mr) in enumerate(zip(chosen[:-1], rejected[:-1])):
        assert mc == mr, f"Chosen and rejected should be identical, got {mc} and {mr}"
    assert chosen[-1]['role'] == 'assistant' and rejected[-1]['role'] == 'assistant', \
        f"The last message in both chosen and rejected should be by the assistant, got {chosen[-1]['role']} and {rejected[-1]['role']}"

    messages = chosen[:-2]
    messages.append({
        'role': 'user',
        'content': ATT_TEMPLATE.format(last_user_message=chosen[-2]['content'], rejected=rejected[-1]['content'])
    })

    for i in range(0, len(messages) - 1):
        assert messages[i]['role'] != messages[i + 1]['role'], \
            f"Messages should alternate between user and assistant, got {messages[i]['role']} and {messages[i + 1]['role']}\n" \
            f"You can use mege_consecutive_messages to achieve that. \n" \
            f"Messages:\n{pretty_print_chatml(messages)}"

    # The chat template used by Mistral is not convenient because it only adds the system prompt to the last
    # user message, and only does it if the last message in the conversation is indeed by the user. This would make it
    # painful to set up the labels for SFT. Hence, we just prepend the system prompt to the first user message instead.
    if tokenizer.chat_template == BAD_MISTRAL_CHAT_TEMPLATE:
        messages[0]["content"] = ATT_SYSTEM_PROMPT + '\n\n' + messages[0]["content"]
    else:
        system_msg = {'role': 'system', 'content': ATT_SYSTEM_PROMPT}
        messages = [system_msg] + messages

    try:
        prompt_text = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True,
                                                    tokenize=False, max_length=max_seq_length)
    except Exception as e:
        logger.error(f"Error in apply_chat_template when generating the prompt: {e}")
        logger.error("Messages:")
        logger.error(pretty_print_chatml(messages))
        raise e

    if debug_print:
        logger.info("The prompt:\n\"\"\"\n" + prompt_text + "\n\"\"\"")
    tokens = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True,
                                           max_length=max_seq_length)
    end_idx = len(tokens)

    messages.append(chosen[-1])

    try:
        response_text = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=False,
                                                      tokenize=False, max_length=max_seq_length)
    except Exception as e:
        logger.error(f"Error in apply_chat_template when generating the response message: {e}")
        logger.error("Messages:")
        logger.error(pretty_print_chatml(messages))
        raise e

    assert prompt_text == response_text[:len(prompt_text)], \
        f"Currently it is assumed that the prompt and response should be the same up to the end of the prompt," \
        f" got \"{prompt_text}\" and \"{response_text}\""
    expected_response_text = response_text[len(prompt_text):]
    if debug_print:
        logger.info("Target response:\n\"\"\"\n" + expected_response_text + "\n\"\"\"")

    input_ids = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=False,
                                              max_length=max_seq_length)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :end_idx] = -100
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


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


def add_filtered_msgs(example, remove_multiturn_data=False):
    """
    Add a filtered version of 'rejected' and 'chosen' messages based on the number of organized messages.
    * delete unnecessary \n between lines
    * remove the ones that has more than one messaging btw assistant and human
    """
    stripped_rejected_msgs = example['rejected'].strip().split('\n\n')
    organized_rejected_msgs = organize_messages(stripped_rejected_msgs)

    stripped_chosen_msgs = example['chosen'].strip().split('\n\n')
    organized_chosen_msgs = organize_messages(stripped_chosen_msgs)

    example['rejected_filtered'] = organized_rejected_msgs
    example['chosen_filtered'] = organized_chosen_msgs
    if remove_multiturn_data and (len(organized_rejected_msgs) > 2 or len(organized_chosen_msgs) > 2):
        example['rejected_filtered'] = []
        example['chosen_filtered'] = []

    return example


# I think only works for single-turn dialogue
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


def target_lora_modules(model) -> List[str]:
    if model.__class__.__name__ == "Phi3ForCausalLM":
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif model.__class__.__name__ == "OLMoForCausalLM":
        return ["att_proj", "ff_proj", "attn_out", "ff_out"]
    elif model.__class__.__name__ == "MistralForCausalLM":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        raise ValueError(f"Model type {type(model)} not added yet. Model:\n {model}")


def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        assert args.report_to in ["wandb", "all"], "Currently only wandb is supported for tracking."
        wandb_api_key = os.getenv('WANDB_API_KEY')
        wandb.login(key=wandb_api_key)

        # Initialize wandb
        wandb.init(project="alignment_translation", entity="claire-labo")

        # Configure wandb logging within Accelerator
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

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

    ######################################## Config and Tokenizer Loading ########################################
    def try_load_config() -> PretrainedConfig:
        # Load pretrained model and tokenizer
        if args.config_name:
            return AutoConfig.from_pretrained(
                args.config_name,
                trust_remote_code=args.trust_remote_code,
                revision=args.model_revision,
            )
        elif args.model_name_or_path:
            return AutoConfig.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=args.trust_remote_code,
                revision=args.model_revision,
            )
        else:
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported by this script."
            )

    try:
        config = try_load_config()
    except OSError as e:
        print(f"Error loading config: {e}")
        print("Assuming it was a login issue, logging into Huggingface...")

        assert 'HUGGINGFACE_HUB_TOKEN' in os.environ, "Please set the HUGGINGFACE_HUB_TOKEN environment variable."
        huggingface_hub.login(token=os.getenv('HUGGINGFACE_HUB_TOKEN'))
        config = try_load_config()

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding=True,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    ######################################## Data Preprocessing ########################################
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
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

    dataset_train = raw_datasets['train'].map(parse_entry_anthropic_hh)
    dataset_test = raw_datasets['test'].map(parse_entry_anthropic_hh)

    dataset_train = dataset_train.filter(is_entry_ok_anthropic_hh)
    dataset_test = dataset_test.filter(is_entry_ok_anthropic_hh)
    print(f"After filtering, {len(dataset_train)} training examples and {len(dataset_test)} test examples remain.")

    if args.remove_multiturn_data:
        dataset_train = dataset_train.filter(lambda x: len(x['rejected']) == 2)
        dataset_test = dataset_test.filter(lambda x: len(x['rejected']) == 2)

    dataset_train = dataset_train.map(lambda x: {k: merge_consecutive_messages_and_trim(v) for k, v in x.items()})
    dataset_test = dataset_test.map(lambda x: {k: merge_consecutive_messages_and_trim(v) for k, v in x.items()})

    for i in range(10):
        logger.info(f"\n\nExample {i} chosen:\n{pretty_print_chatml(dataset_train[i]['chosen'])}\n\n"
                    f"Example {i} rejected:\n{pretty_print_chatml(dataset_train[i]['rejected'])}\n\n")
        apply_att_template(dataset_test[i], tokenizer, args.max_seq_length, debug_print=True)

    filtered_dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    # add info column which will store human, assistant_rejected and assistant_chosen messages
    # filtered_dataset["train"] = filtered_dataset["train"].map(extract_role_messages)
    # filtered_dataset["test"] = filtered_dataset["test"].map(extract_role_messages)

    # encode_function = partial(
    #     encode_with_rejected_chosen,
    #     tokenizer=tokenizer,
    #     max_seq_length=args.max_seq_length,
    #     add_bos=args.add_bos,
    # )
    encode_function = partial(
        apply_att_template,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length
    )

    with accelerator.main_process_first():
        lm_datasets = filtered_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in filtered_dataset["train"].column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    print("Size of training set:", len(filtered_dataset['train']))
    print("Size of test set:", len(filtered_dataset['test']))

    train_dataset = lm_datasets["train"]
    test_dataset = lm_datasets["test"]

    ######################################## Model Loading ########################################
    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index}  # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,  #
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # Add special tokens if they are not already added
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        # pad token is also equal to eos token in the case of phi3

        if model.__class__.__name__ == "Phi3ForCausalLM":
            assert num_added_tokens <= 2
            # Taken from https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-lora-python.ipynb
            tokenizer.pad_token = tokenizer.unk_token
            # The ID of the padding token is set to the ID of the unknown token.
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        else:
            assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token," \
                                               " or no tokens if pad token present."
        tokenizer.padding_side = 'left'
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})

    # Now OLMo supports chat templates, so we don't need to add bos token
    # elif isinstance(tokenizer, OLMoTokenizerFast):
    #     # only the eos for olmo, but we use it as bos
    #     tokenizer.bos_token = tokenizer.eos_token
    #     assert args.add_bos, "For OLMo, you must add bos token to the beginning of the input sequence."

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size

    generation_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=args.logging_examples_max_length,
        top_p=args.logging_examples_top_p,
        renormalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    embeddings = model.get_input_embeddings()

    # padding will enable tensorcores, hopefully will make it faster
    # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=128)
        embedding_size = embeddings.weight.shape[0]
        # padding to multiples creates a few "shadow" tokens that we don't want to be generated
        generation_config.suppress_tokens = list(range(len(tokenizer), embedding_size))

    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
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

    # Log a few random samples from the training set

    indices_train = random.sample(range(len(train_dataset)), 5)
    indices_test = random.sample(range(len(test_dataset)), 5)
    train_examples = [train_dataset[i] for i in indices_train]
    test_examples = [test_dataset[i] for i in indices_test]
    for index, ex in enumerate(train_examples):
        logger.info(f"Sample {index} of the training set: {ex}.")

    ######################################## Training Setup ########################################
    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
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
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct", experiment_config)

    # Define custom step metric for evaluation
    run_id = wandb.run.id

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    ######################################## Checkpointing ########################################
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        model.load_adapter(checkpoint_path, adapter_name=checkpoint_path)
        model.print_trainable_parameters()

        """
        There is a problem in the use of this loaded optimizer state. For now, it is commented out.
        optimizer_state_file = os.path.join(checkpoint_path, 'optimizer_state.pt')

        # Check if the optimizer state file exists before loading
        if os.path.exists(optimizer_state_file):
            optimizer_state_dict = torch.load(optimizer_state_file)
            optimizer.load_state_dict(optimizer_state_dict)
            print("Optimizer state loaded successfully.")
        else:
            print("Optimizer state file does not exist.")
        """
        # Path to the scheduler state file
        # scheduler_state_file = 'scheduler_state.pt'

        scheduler_state_file = os.path.join(checkpoint_path, 'scheduler_state.pt')

        # Check if the scheduler state file exists before loading
        if os.path.exists(scheduler_state_file):
            scheduler_state_dict = torch.load(scheduler_state_file)
            lr_scheduler.load_state_dict(scheduler_state_dict)
            print("Scheduler state loaded successfully.")
        else:
            print("Scheduler state file does not exist.")

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = path

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

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    example_tables = {
        "train": [wandb.Table(columns=["step", "response"]) for _ in range(len(train_examples))],
        "test": [wandb.Table(columns=["step", "response"]) for _ in range(len(test_examples))],
    }

    def log_examples_to_wandb(step: int):
        def log(example, example_idx: int, prefix: str) -> float:
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

            if step == 0:
                wandb.log(
                    {f"{prefix}_example_{example_idx + 1}/prompt": wandb.Html(f"<p>{html.escape(prompt_text)}</p>")})
                wandb.log({f"{prefix}_example_{example_idx + 1}/expected_response": wandb.Html(
                    f"<p>{html.escape(expected_response_text)}</p>")})

            outputs = model.generate(input_ids=prompt_ids.to(model.device)[None, :],
                                     generation_config=generation_config)
            response_text = tokenizer.decode(outputs[0])
            if response_text[:len(prompt_text)] != prompt_text:
                logger.warning(
                    f"Response does not start with the prompt for example {example_idx}. "
                    f"Prompt: \n\"\"\"\n{prompt_text}\n\"\"\"\n. Response: \n\"\"\"\n{response_text}\n\"\"\"\n"
                )
            else:
                response_text = response_text[len(prompt_text):]

            example_tables[prefix][example_idx].add_data(step, response_text)
            wandb.log({f"{prefix}_example_{example_idx + 1}/responses": example_tables[prefix][example_idx]})
            return time.time() - start_time

        model.eval()
        sum_time = 0.0
        for i, ex in enumerate(train_examples):
            sum_time += log(ex, i, "train")
        for i, ex in enumerate(test_examples):
            sum_time += log(ex, i, "test")

        wandb.log(
            {"log_examples_time": sum_time, "avg_example_time": sum_time / (len(train_examples) + len(test_examples))})

        model.train()

    ######################################## Training Loop ########################################
    if accelerator.sync_gradients:
        log_examples_to_wandb(0)
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
        for step, batch in enumerate(active_dataloader):
            # Print the size of each component in the batch
            batch_size = len(batch['input_ids'])  # Assuming 'input_ids' is the key for input data
            epoch_data_count += batch_size
            with accelerator.accumulate(model):
                # print(f"Memory allocated before forward pass: {memory_allocated() / 1e9} GB")
                outputs = model(**batch, use_cache=False)
                # print(f"Memory allocated after forward pass: {memory_allocated() / 1e9} GB")

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
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
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
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    # Print number of examples processed so far in this epoch
                    print(f"Step {completed_steps}: Processed {epoch_data_count} examples so far in Epoch {epoch + 1}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )

                    total_loss = 0

                if args.logging_examples_steps and completed_steps % args.logging_examples_steps == 0:
                    log_examples_to_wandb()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        print(f"Completed Epoch {epoch + 1}: Total processed examples = {epoch_data_count}")
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            torch.cuda.empty_cache()
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
                tokenizer.save_pretrained(output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)
            accelerator.wait_for_everyone()  # ensure that the files are created
            print(output_dir)

            print(f"Running evaluation at the end of epoch {epoch + 1}")
            run_evaluation_subprocess(args, output_dir, run_id, tokenizer)
            csv_path = output_dir + "/eval_results/summary.csv"
            print("log eval results to wandb")
            metrics_log = log_eval_results_to_wandb(csv_path, epoch)
            logger.info(f"Epoch {epoch} Evaluation Metrics: {metrics_log}")
            if args.with_tracking:
                # Log evaluation metrics
                wandb.log(metrics_log)

    if args.output_dir is not None:
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
