#!/usr/bin/env python
# coding=utf-8
from torch.cuda import memory_allocated
import argparse
import logging
import math
import ast
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
import subprocess
from merge_lora import main as merge_lora
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
from eval.utils import KeyWordsCriteria
from pathlib import Path
import gc

sys.path.append(Path(__file__).parents[1].absolute().as_posix())

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
#from eval.truthfulqa.run_eval import main as run_eval
#from eval.truthfulqa.run_eval import parse_args as parse_args_eval
import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256" - Cuda out of memory

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

@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True,
                         disable_tqdm=False, **generation_kwargs):
    """this function is taken from eval folder in order to see and track the changes it is copied to here."""
    generations = []

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in tqdm(range(0, len(prompts), batch_size), disable=disable_tqdm):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )


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
def log_eval_results_to_wandb(csv_path, epoch):
    # Load the summary CSV file
    try:
        # Load the summary CSV file
        results_df = pd.read_csv(csv_path)
        metrics={}
        # Assume there's only one row of metrics, typical in a summarised results CSV
        if not results_df.empty:
            results_dict = results_df.iloc[0].to_dict()  # Convert the first row to a dictionary
            metrics={
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

def run_evaluation_subprocess(args,base_path, run_id, tokenizer):
    """ Run the evaluation script as a subprocess. """
    command = [
        'python', '../open-instruct/open_instruct/eval_script.py',
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


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    '''
    This function is used for encodind teacher outputs for student model finetuning
    '''
    human_message = example['human_messages']
    completion = example['completions']

    messages = [
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": completion}
    ]

    if len(messages) == 0:
        raise ValueError('messages field is empty.')

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text

    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length,
                    truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx + 1]) + "\n"
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
        system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            message_text += "<|system|>\n" + system_message + +'<|end|>'+ "\n"
            for message in messages:
                if message["role"] == "human":
                    message_text += "<|user|>\n Prompt: " + message[
                        "content"].strip() + +'<|end|>'+ "\n" # between prompt and assistant rejected \n\n double space
                elif message["role"] == "assistant_rejected":
                    message_text += "\n Current rejected answer: " + message[
                        "content"].strip() + "\n Corrected output: \n" +'<|end|>'+ "\n"
                elif message["role"] == "assistant_chosen":
                    message_text += "<|assistant|>\n" + message["content"].strip() + +'<|end|>'+ "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
        else:
            message_text += "<|system|>\n" + system_message + "\n"
            for message in messages:
                if message["role"] == "human":
                    message_text += "<|user|>\n Prompt: " + message["content"].strip() + "\n"  # between prompt and assistant rejected \n\n double space
                elif message["role"] =="assistant_rejected":
                    message_text += "\n Current rejected answer: " + message["content"].strip() + "\n Corrected output: \n"
                elif message["role"] == "assistant_chosen":
                    message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    #print(example_text) #to be commented out

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
    if len(organized_rejected_msgs) <= 2 and len(organized_chosen_msgs)<=2:
        example['rejected_filtered'] = organized_rejected_msgs
        example['chosen_filtered'] = organized_chosen_msgs
    else:
        example['rejected_filtered'] = []
        example['chosen_filtered']=[]

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
        (msg[len("Assistant: "):] for msg in example['chosen_filtered'] if msg.startswith("Assistant: ")),  " ")

    """
    There are some cases like:
    ['Human: How would you plan on killing another person', 'Assistant:'] -> chosen
    ['Human: How would you plan on killing another person', 'Assistant: I would plan to kill that person if it was a proper action to do so.'] -> rejected
    """

    example['info'] = [
        {"role": "human", "content": human_msg if human_msg else " " },
        {"role": "assistant_rejected",
         "content": assistant_rejected_msg if assistant_rejected_msg else " "},
        {"role": "assistant_chosen",
         "content": assistant_chosen_msg if assistant_chosen_msg else " "}
    ]

    return example

def prepare_att_dataset(model, tokenizer, args, filtered_dataset):

    # Preprocessing the datasets.
    if "rejected" in filtered_dataset["train"].column_names and "chosen" in filtered_dataset["train"].column_names:
        encode_function = partial(
            encode_with_rejected_chosen,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError("You need to have either 'rejected'&'chosen' in your column names.")

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


    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )
    return train_dataloader

def create_optimizer(model, weight_decay, learning_rate, use_qlora, use_8bit_optimizer):
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            optim_bits=8 if use_8bit_optimizer else 32,
            is_paged=True
        )
    else:
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer

def create_scheduler(optimizer, num_training_steps, warmup_ratio, lr_scheduler_type):
    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        num_warmup_steps=int(num_training_steps * warmup_ratio),
    )
    return lr_scheduler

def prepare_config(model, args):
    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

        logger.info("Initializing LORA model...")
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                # target_modules=["att_proj", "ff_proj", "attn_out", "ff_out"]  # for OLMo
                target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # for phi3
            )
        else:
            # olmo target modules are used
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["att_proj", "ff_proj", "attn_out", "ff_out"]  # for OLMo
                # target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]  # for phi3
            )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model

def prepare_for_training(model, args, train_dataloader):

    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        use_qlora=args.use_qlora,
        use_8bit_optimizer=args.use_8bit_optimizer
    )

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
    # Create scheduler
    lr_scheduler = create_scheduler(
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type
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
    return optimizer,lr_scheduler, checkpointing_steps, args


def load_checkpoint(args, model, optimizer, lr_scheduler):
    completed_steps = 0
    starting_epoch = 0

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
        return starting_epoch, completed_steps, resume_step

def training(model, optimizer, train_dataloader, lr_scheduler, args, accelerator, tokenizer, run_id, starting_epoch=0, resume_step=None, save_directory=args.output_dir):
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.update(resume_step if resume_step else 0)
    completed_steps = resume_step if resume_step else 0
    checkpointing_steps = args.checkpointing_steps
    best_bleurt_score = -float('inf')
    best_epoch = -1

    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        epoch_data_count = 0  # To keep track of the number of data points in each epoch

        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader

        for step, batch in enumerate(active_dataloader):
            # Print the size of each component in the batch
            batch_size = len(batch['input_ids'])  # Assuming 'input_ids' is the key for input data
            epoch_data_count += batch_size
            with accelerator.accumulate(model):
                outputs = model(**batch, use_cache=False)

                if args.reduce_loss == 'mean':
                    loss = outputs.loss
                else:
                    logits = outputs.logits
                    labels = batch["labels"]
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    # Flatten the tokens
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
                    shift_logits = shift_logits.view(-1, logits.size(-1))
                    shift_labels = shift_labels.view(-1).to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)

                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                if step % 2000 == 0:
                    torch.cuda.empty_cache()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    print(f"Step {completed_steps}: Processed {epoch_data_count} examples so far in Epoch {epoch + 1}")
                    if args.with_tracking:
                        accelerator.log({
                            "learning_rate": lr_scheduler.get_last_lr()[0],
                            "train_loss": avg_loss,
                        }, step=completed_steps)

                    total_loss = 0

                if completed_steps >= args.max_train_steps:
                    break

        print(f"Completed Epoch {epoch + 1}: Total processed examples = {epoch_data_count}")
        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            torch.cuda.empty_cache()
            if save_directory is not None:
                output_dir = os.path.join(save_directory, output_dir)
                tokenizer.save_pretrained(output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)
            accelerator.wait_for_everyone()  # ensure that the files are created
            print(output_dir)

            print(f"Running evaluation at the end of epoch {epoch + 1}")
            run_evaluation_subprocess(args, output_dir, run_id, tokenizer)
            csv_path = os.path.join(output_dir, "eval_results", "summary.csv")
            print("log eval results to wandb")
            metrics_log = log_eval_results_to_wandb(csv_path, epoch)
            logger.info(f"Epoch {epoch} Evaluation Metrics: {metrics_log}")

            # Track the best BLEURT score
            bleurt_score = metrics_log.get("BLEURT_acc", None)
            if bleurt_score is not None and bleurt_score > best_bleurt_score:
                best_bleurt_score = bleurt_score
                best_epoch = epoch

            if args.with_tracking:
                wandb.log(metrics_log)

    print(f"Best BLEURT score was {best_bleurt_score} at epoch {best_epoch}")
    return best_epoch, model, output_dir


def create_prompt(original_questions, tag):
    # system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given output."
    formatted_questions = []  # Use a different variable name to store results

    for index, row in original_questions.iterrows():
        message_text = ""
        if pd.notna(row['Question']):
            message_text += row['Question'].strip() + "\n"
        if pd.notna(row[tag]):
            message_text += "\n Current rejected answer: " + row[tag].strip() + "\n Corrected output: \n "
            formatted_questions.append(message_text)  # append EOS here
        else:
            message_text += "\n Current rejected answer: " + " " + "\n Corrected output: \n "
            formatted_questions.append(message_text)
            print("Something wrong in the structure")

    return formatted_questions
def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        if "wandb" in args.report_to.split(",") or args.report_to == "all":
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

    # If passed along, set the training seed now. -> use this to split 90% the same everytime.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

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

    #for train
    # filter the dataset for it to have only one prompt and answer (not a sequence of prompt-answer in one line) -> for now
    updated_dataset_train = raw_datasets['train'].map(add_filtered_msgs)
    filtered_train = updated_dataset_train.filter(lambda x: len(x['rejected_filtered']) > 0)#.select(range(10)) # delete this

    #for test
    updated_dataset_test = raw_datasets['test'].map(add_filtered_msgs)
    filtered_test = updated_dataset_test.filter(lambda x: len(x['rejected_filtered']) > 0)

    filtered_dataset = DatasetDict({
        'train': filtered_train,
        'test': filtered_test
    })

    print("Size of training set:", len(filtered_dataset['train']))
    print("Size of test set:", len(filtered_dataset['test']))

    # add info column which will store human, assistant_rejected and assistant_chosen messages
    filtered_dataset["train_all"] = filtered_dataset["train"].map(extract_role_messages)
    train_size = 0.9
    filtered_dataset["train"], _ = train_test_split(filtered_dataset["train_all"], train_size=train_size, random_state=42)
    filtered_dataset["test"] = filtered_dataset["test"].map(extract_role_messages)

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

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
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

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
        assert num_added_tokens in [0,
                                    1,
                                    2], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        # specific to phi3 case
        # The padding token is set to the unknown token.
        tokenizer.pad_token = tokenizer.unk_token

        # The ID of the padding token is set to the ID of the unknown token.
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'left'

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

    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    lora_model = prepare_config(model)
    train_dataloader= prepare_att_dataset(lora_model, tokenizer, args, filtered_dataset)
    optimizer, lr_scheduler, checkpointing_steps, args= prepare_for_training(lora_model, args, train_dataloader)

    # Define custom step metric for evaluation
    run_id = wandb.run.id

    wandb.define_metric("eval_step")
    # Define evaluation metrics with their respective custom step
    wandb.define_metric("BLEURT_acc", step_metric="eval_step")
    wandb.define_metric("bleu_acc", step_metric="eval_step")
    wandb.define_metric("rouge1_acc", step_metric="eval_step")
    wandb.define_metric("rouge2_acc", step_metric="eval_step")
    wandb.define_metric("rougeL_acc", step_metric="eval_step")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training - ATT *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    starting_epoch, completed_steps, resume_step= load_checkpoint(args, lora_model, optimizer, lr_scheduler)
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    best_epoch, lora_model, epoch_output_dir =  training(lora_model, optimizer, train_dataloader, lr_scheduler, args, accelerator, tokenizer, run_id, starting_epoch, resume_step)

    if args.output_dir is not None:
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, lora_model, tokenizer, args.output_dir, args)

    #delete the model to save memory
    del lora_model
    gc.collect()
    # ATT finetuning finished
    lora_path = os.path.join(epoch_output_dir, str(best_epoch))

    #get the best epoch and merge it
    args_merged = argparse.Namespace(
            lora_model_name_or_path=lora_path,
            base_model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=lora_path,
            output_dir=args.att_save_dir,
            qlora=False,
            save_tokenizer=True,
            use_fast_tokenizer=True,
            tokenizer_revision=None
        )

    merge_lora(args_merged)
    # lora_model -> used for finetuning ATT
    # model-> is the version loaded at the beginning which is the base

    # take the outputs from the ATT model - two steps for whole anthropic
    # Extract human messages as prompts
    prompts = []
    for item in filtered_dataset["train"]:
        for message in item['info']:
            if message["role"] == "human":
                prompts.append(message["content"])
                break

    # first take the outputs from the base model - 1st step
    for idx, prompt in enumerate(prompts):
        messages = [{"role": "user", "content": prompt}]
        prompts[idx] = chat_formatting_function(messages, tokenizer)

    # What about sampling?
    batch_size=1
    completions = generate_completions(
        model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=512,
        stop_id_sequences=None, #not used stop id for now
        do_sample=False
    )
    assert len(completions) == len(prompts)
    # Create a DataFrame with prompts and completions
    questions = pd.DataFrame({
        "messages": prompts,
        "completions": completions
    })
    questions.to_csv(f"{args.data_completions_save_dir}/first_completions.csv", index=False)
    print("1st step answers are stored.")

    # Post-processing of completions
    for idx, completion in zip(questions.index, completions):
        questions.loc[idx, tag] = trim_answer(completion)

    questions["modified_input"] = create_prompt(questions, tag)
    #load the ATT model
    teacher_model = AutoModelForCausalLM.from_pretrained(
        args.att_save_dir,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        use_flash_attention_2=True if args.use_flash_attn else False )

    system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."
    prompts = questions["modified_input"].tolist()

    if chat_formatting_function is not None:
        for idx, prompt in enumerate(prompts):
            messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
            prompts[idx] = chat_formatting_function(messages, tokenizer, add_bos=False)

    completions = generate_completions(
        teacher_model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=512,
        stop_id_sequences=None,  # not used stop id for now
        do_sample=False
    )

    # store the results
    train_db = pd.DataFrame({
        "human_messages": human_messages,
        "messages": prompts,
        "completions": completions
    })
    train_db.to_csv(f"{args.data_completions_save_dir}/second_completions.csv", index=False)
    print("2nd step answers are stored.")

    # use teacher models' outputs and generate encodings
    embeddings = teacher_model.get_input_embeddings()

    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            teacher_model.resize_token_embeddings(len(tokenizer))

    encode_function = partial(
        encode_with_messages_format,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,  # Replace with args.max_seq_length if applicable
        add_bos=False  # Replace with args.add_bos if applicable
    )

    lora_student_model = prepare_config(teacher_model)
    with accelerator.main_process_first():
        #questions is the new dataset - created with the outputs of Teacher model
        lm_datasets = train_db.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in train_db.column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())


    train_dataset_student = lm_datasets["train"]
    # DataLoaders creation:
    train_dataloader_student = DataLoader(
        train_dataset_student,
        shuffle=True,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )

    optimizer_student, lr_scheduler_student, checkpointing_steps_student, args = prepare_for_training(lora_student_model, args, train_dataloader_student)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    # new training
    logger.info("***** Running training - Student model *****")
    logger.info(f"  Num examples = {len(train_dataset_student)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    starting_epoch, completed_steps, resume_step = load_checkpoint(args, lora_student_model, optimizer_student, lr_scheduler_student)
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    best_epoch, lora_student_model, epoch_output_dir = training(lora_student_model, optimizer_student, train_dataloader_student, lr_scheduler_student, args,
                                                        accelerator, tokenizer, run_id, starting_epoch, resume_step, save_directory= args.save_student_epochs)


    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
