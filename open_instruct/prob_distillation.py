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
from transformers import StoppingCriteria
from datasets import DatasetDict
import subprocess
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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
import json
import time

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

        # Log progress to W&B
        wandb.log({"progress": i + batch_size})

    # Write any remaining entries in all_json_entries to the file
    if all_json_entries:
        with open(generation_storage, 'a') as f:
            for entry in all_json_entries:
                f.write(json.dumps(entry) + "\n")

    assert len(generations) == len(
        prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"

    return generations


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
        "--generation_storage", type=str, default="data_new.json", help="A json file will store the generated data."
    )
    parser.add_argument(
        "--teacher_model_name_or_path",
        type=str,
        help="Path to teacher finetuned model ",
        required=False,
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
    parser.add_argument(
        "--generated_data_path",
        type=str,
        help="collected/generated data from the base sft model for its use in teacher model",
        required=False,
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


def save_with_accelerate(accelerator, model, output_dir, args):
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

    # Save the optimizer and scheduler states
    if accelerator.is_main_process:
        accelerator.save_state(output_dir)

    # save training arguments or any other configs as a JSON
    args_file = os.path.join(output_dir, "training_args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f)


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


def encode_with_rejected_chosen(example, tokenizer, max_seq_length, model_type, add_bos=False):
    '''
    Assume each example has a 'messages' field where each message is a dict with 'role' and 'content'.
    Concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['info']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')


    def _concat_messages(messages, model_type):
        message_text = ""
        system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."
        if model_type == "teacher":
            if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
                message_text += "<|system|>\n" + system_message + '<|end|>' + "\n"
                for message in messages:
                    if message["role"] == "human":
                        message_text += "<|user|>\n Prompt: " + message[
                            "content"].strip() + '<|end|>' + "\n"  # between prompt and assistant rejected \n\n double space
                    elif message["role"] == "assistant_rejected":
                        message_text += "\n Current rejected answer: " + message[
                            "content"].strip() + "\n Corrected output: \n" + '<|end|>' + "\n"
                    elif message["role"] == "assistant_chosen":
                        message_text += "<|assistant|>\n" + message["content"].strip() + "<|end|>" + "\n"
            else:
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
        elif model_type == "student":
            if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
                for message in messages:
                    if message["role"] == "human":
                        message_text += "<|user|>\n " + message[
                            "content"].strip() + '<|end|>' + "\n"
                    elif message["role"] == "assistant_chosen":
                        message_text += "<|assistant|>\n" + message["content"].strip() + "<|end|>" + "\n"

            else:
                for message in messages:
                    if message["role"] == "human":
                        message_text += "<|user|>\n " + message[
                            "content"].strip() + "\n"
                    elif message["role"] == "assistant_chosen":
                        message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
        else:
            raise ValueError("Invalid model type; must be student or teacher")

        return message_text


    example_text = _concat_messages(messages, model_type).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text

    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True, padding="max_length")
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # Mask the non-assistant part to avoid calculating loss on it
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant_chosen":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx], model_type), return_tensors='pt', max_length=max_seq_length,
                    truncation=True,
                    padding="max_length"
                ).input_ids.shape[1]
            message_end_idx = tokenizer(
                _concat_messages(messages[:message_idx + 1], model_type),
                return_tensors='pt',
                max_length=max_seq_length,
                truncation=True,
                padding="max_length"
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

def apply_formatting_and_encoding(example, tokenizer, max_seq_length, add_bos=False):
    teacher_encoded = encode_with_rejected_chosen(example, tokenizer, max_seq_length, "teacher", add_bos)
    student_encoded = encode_with_rejected_chosen(example, tokenizer, max_seq_length, "student", add_bos)

    return {
        'input_ids': student_encoded['input_ids'],
        'teacher_input_ids': teacher_encoded['input_ids'],
        'attention_mask': student_encoded['attention_mask'],
        'teacher_attention_mask': teacher_encoded['attention_mask'],
        'labels': teacher_encoded['labels'],
    }


def create_input_format_student(messages, tokenizer, add_bos=False):
    student_message_text = ""
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        for message in messages:
            if message["role"] == "human":
                student_message_text += "<|user|>\n " + message[
                    "content"].strip() + '<|end|>' + "\n"

    else:
        for message in messages:
            if message["role"] == "human":
                student_message_text += "<|user|>\n " + message[
                    "content"].strip() + "\n"
    student_message_text += "<|assistant|>\n"

    # Optionally add BOS token at the beginning
    if add_bos:
        tokenizer.bos_token = tokenizer.eos_token  # just for olmo
        student_message_text = tokenizer.bos_token + student_message_text
    return student_message_text


def main():
    args = parse_args()

    accelerator_log_kwargs = {}
    args.with_tracking = True
    args.lambda_value = 1

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

    ################
    # Model & Tokenizer
    ################

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
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_flash_attention_2=True if args.use_flash_attn else False,
            revision=args.model_revision
        )

        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_flash_attention_2=True if args.use_flash_attn else False,
            revision=args.model_revision)

    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # Print all named modules
    for name, module in model.named_modules():
        print(name)

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

    ################
    # Dataset & Lora Config
    ################
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name
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

    # for train
    # filter the dataset for it to have only one prompt and answer (not a sequence of prompt-answer in one line) -> for now
    updated_dataset_train = raw_datasets['train'].map(add_filtered_msgs)
    filtered_train = updated_dataset_train.filter(
        lambda x: len(x['rejected_filtered']) > 0)  # .select(range(1000)) # delete this

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

    # getting the responses from the base model as rejected
    # Apply the create_input_format_student function to each example in the training set
    # Directly process each 'info' entry in the dataset and collect the results in a list
    student_msgs_list = [
        create_input_format_student(info, tokenizer, add_bos=args.add_bos)
        for info in filtered_dataset['train']['info']
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    if args.generated_data_path:
        generated_responses = []
        # Open the JSON file and read each line
        with open(args.generated_data_path, 'r') as file:
            for line in file:
                try:
                    # Parse the JSON data from the line
                    data = json.loads(line.strip())
                    # Extract batch_generations and append to the list
                    generated_responses.extend(data.get("batch_generations", []))
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line due to JSONDecodeError: {e}")
        print("already generated responses taken", generated_responses[0])
    else:
        generated_responses = generate_completions(
            model, tokenizer, student_msgs_list, batch_size=1, max_new_tokens=300, disable_tqdm=False,
            generation_storage=args.generation_storage,
            stop_id_sequences=None,
            do_sample=False
        )
    # replace the rejected responses of anthropic with base models responses

    # Modifying the remaining entries
    for idx, info in enumerate(filtered_dataset["train"]["info"]):
        for message in info:
            if message['role'] == 'assistant_rejected':
                message['content'] = generated_responses[idx]

    # Preprocessing the datasets.

    if "rejected" in filtered_dataset["train"].column_names and "chosen" in filtered_dataset["train"].column_names:
        encode_function = partial(
            apply_formatting_and_encoding,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos= args.add_bos
        )
    else:
        raise ValueError("You need to have either 'rejected'&'chosen' in your column names.")

    lm_datasets = filtered_dataset.map(
        encode_function,
        batched=False,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        remove_columns=[name for name in filtered_dataset["train"].column_names if
                        name not in ["input_ids", "teacher_input_ids", "attention_mask", "teacher_attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")

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
    print("train dataloader:", len(train_dataloader))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print(num_update_steps_per_epoch)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

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

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    teacher_model.to(accelerator.device)
    total_start_time = time.time()  # Start time for the entire training process

    for epoch in range(starting_epoch, args.num_train_epochs):
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
        total_loss = 0
        epoch_data_count = 0  # To keep track of the number of data points in each epoch
        epoch_start_time = time.time()  # Start time for the current epoch

        for step, batch in enumerate(train_dataloader):
            # the main ones belong to the student model
            student_inputs = {
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask']
            }
            teacher_inputs = {
                'input_ids': batch['teacher_input_ids'],
                'attention_mask': batch['teacher_attention_mask']
            }
            u = random.uniform(0, 1)
            batch_size = len(batch['input_ids'])  # Assuming 'input_ids' is the key for input data
            epoch_data_count += batch_size
            # Accumulate gradients over the batch
            with accelerator.accumulate(model):
                if u <= args.lambda_value:
                    student_outputs = model(**student_inputs, use_cache=False)
                    student_logits = student_outputs.logits

                    with torch.no_grad():

                        teacher_outputs = teacher_model(**teacher_inputs, use_cache=False)
                        teacher_logits = teacher_outputs.logits
                    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

                    # Compute the softmax of student logits for loss calculation
                    student_probs = torch.nn.functional.softmax(student_logits, dim=-1)
                    loss = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(student_logits,dim=-1), teacher_probs, reduction='batchmean')

                else:
                    with torch.no_grad():
                        teacher_outputs = teacher_model(**teacher_inputs, use_cache=False)
                        teacher_logits = teacher_outputs.logits
                    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)

                    # Change it with labels - from the dataset - TODO
                    loss = torch.nn.functional.kl_div(teacher_probs.log(), teacher_probs, reduction='batchmean')

                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()

                # Backpropagation
                accelerator.backward(loss)

                # Clip gradient norm if needed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Scheduler step
                lr_scheduler.step()

            # Update progress bar and completed steps
            progress_bar.update(1)
            completed_steps += 1
            if args.logging_steps and completed_steps % args.logging_steps == 0:
                step_duration = time.time() - epoch_start_time  # Time taken to process this batch
                total_elapsed_time = time.time() - total_start_time  # Total time elapsed since training started

                avg_loss = total_loss / args.gradient_accumulation_steps / args.logging_steps

                # print(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                # Print number of examples processed so far in this epoch
                # print(
                #    f"Step {completed_steps}: Processed {epoch_data_count} examples so far in Epoch {epoch + 1}, progress: {100 * completed_steps / args.max_train_steps}")
                # print(f"step_duration: {step_duration},total_elapsed_time: {total_elapsed_time / 3600}")
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
            save_with_accelerate(accelerator, model, output_dir, args)

    if args.output_dir is not None:
        tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, args.output_dir, args)


if __name__ == "__main__":
    main()
