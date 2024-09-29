import os
import sys
from pathlib import Path
from typing import List, Dict
import re
import json
import argparse
import logging
from copy import deepcopy
from contextlib import nullcontext

import torch
import huggingface_hub
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import SchedulerType
from transformers import (
    AutoConfig,
    PretrainedConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig,
)
import deepspeed
from datasets import load_dataset
from accelerate.logging import get_logger

sys.path.append(str(Path(__file__).parents[1].absolute().as_posix()))
from open_instruct.constants import PHI2_CHAT_TEMPLATE, LLAMA_TULU_CHAT_TEMPLATE

logger = get_logger(__name__)


def get_hf_token() -> str:
    if 'HF_TOKEN' in os.environ:
        return os.environ['HF_TOKEN']
    elif 'HF_TOKEN_AT' in os.environ:
        with open(os.environ['HF_TOKEN_AT'], 'r') as file:
            return file.read().strip()
    else:
        raise ValueError("Please set the HF_TOKEN or HF_TOKEN_AT variable.")


def add_common_training_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_name",
        nargs="*",
        type=str,
        default=None,
        help="The name of the datasets to use (via the datasets library).",
    )
    parser.add_argument(
        '--dataset_subsample_size',
        type=int,
        default=None,
        help='Subsample the dataset to this size. Useful for debugging.'
    )
    # Don't need this for now
    # parser.add_argument(
    #     "--dataset_config_name",
    #     type=str,
    #     default=None,
    #     help="The configuration name of the dataset to use (via the datasets library).",
    # )
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
        "--half_dataset",
        action="store_true",
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
        default=1024,
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
        "--ignore_model_cache", action="store_true", help="Ignore the cached model and load from HF."
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
        "--logging_examples_steps",
        type=int,
        default=None,
        help="Log example LM generations every logging_examples_steps steps.",
    )
    parser.add_argument(
        "--logging_examples_max_length",
        type=int,
        default=64,
        help="The maximum length of the generated examples to log.",
    )
    parser.add_argument(
        "--logging_examples_top_p",
        type=int,
        default=0.9,
        help="The top-p value to use for the generated examples.",
    )
    parser.add_argument(
        "--logging_examples_temp",
        type=float,
        default=0.8,
        help="The temperature value to use for the generated examples."
    )
    parser.add_argument(
        "--logging_examples_ignore_first",
        action="store_true",
        help="For debugging: do not log sample outputs before the first training step.",
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
        "--wandb_entity",
        type=str,
        default="claire-labo",
        help="The entity to use for W&B logging.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        help="The name of the wandb run.",
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
        '--method',
        type=str,
        default='att',
        choices=['att', 'dpo', 'orpo'],
        help='The method to use for preference tuning.',
    )


def save_args(args: argparse.Namespace, filename: str):
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4)


def load_args(filename: str) -> argparse.Namespace:
    with open(filename, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)


def parse_messages_anthropic_hh(messages_raw: str) -> List[Dict[str, str]]:
    roles_mapping = {
        "Human:": "user",
        "Humans:": "user",
        "Assistant:": "assistant"
    }

    # Split the input string into individual messages
    messages = re.split('(' + '|'.join(roles_mapping.keys()) + ')', messages_raw)[1:]

    # Remove the prefixes and combine the messages
    chatml_messages = []
    for i in range(0, len(messages), 2):
        assert messages[i] in roles_mapping.keys(), f"Invalid role: {messages[i]}"

        if i + 1 >= len(messages) or messages[i + 1].strip() == "":
            message = " "
        else:
            message = messages[i + 1].strip()
        chatml_messages.append({
            "role": roles_mapping[messages[i]],
            "content": message
        })

    return chatml_messages


def parse_entry_anthropic_hh(entry: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    return {k: parse_messages_anthropic_hh(v) for k, v in entry.items()}


# I think it's better to just remove the conversations where this happens, keeping this function just in case
def merge_consecutive_messages_and_trim(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    merged_messages = []
    needs_merge = False
    for i in range(0, len(messages)):
        if i == 0 or messages[i]['role'] != messages[i - 1]['role']:
            merged_messages.append(deepcopy(messages[i]))
        else:
            merged_messages[-1]['content'] += '\n\n' + messages[i]['content']
            needs_merge = True
    if needs_merge:
        logger.warning(f"Consecutive messages from the same role have been merged:\n{pretty_print_chatml(messages)}")

    def trim(s: str) -> str:
        stripped = s.strip()
        return stripped if stripped != '' else ' '

    merged_messages = [{'role': m['role'], 'content': trim(m['content'])} for m in merged_messages]

    return merged_messages


# See eg https://huggingface.co/datasets/Anthropic/hh-rlhf/viewer/default/train?q=facebook+stalk&row=12696
def is_entry_ok_anthropic_hh(entry: Dict[str, List[Dict[str, str]]]) -> bool:
    if len(entry['chosen']) != len(entry['rejected']):
        return False

    chosen = entry['chosen']
    rejected = entry['rejected']
    for mc, mr in zip(chosen[:-1], rejected[:-1]):
        if mc != mr:
            return False

    def is_consecutive_alt(messages):
        for i in range(1, len(messages)):
            if messages[i]['role'] == messages[i - 1]['role']:
                return False
        return True

    if chosen[-1]['role'] != 'assistant' or rejected[-1]['role'] != 'assistant':
        return False

    return is_consecutive_alt(chosen) and is_consecutive_alt(rejected)


def pretty_print_chatml(messages: List[Dict[str, str]]):
    """
    Pretty print a list of chat messages in ChatML format.
    """
    chatml = ""
    for message in messages:
        if message["role"] == "system":
            chatml += f"SYS:\n{message['content']}\n"
        elif message["role"] == "user":
            chatml += f"USR:\n{message['content']}\n"
        elif message["role"] == "assistant":
            chatml += f"ASST:\n{message['content']}\n"
        else:
            raise ValueError(f"Invalid role: {message['role']}")
    return chatml


def preprocess_data_to_chatml(accelerator, args):
    chatml_datasets_train = []
    chatml_datasets_test = []
    for dataset_name in args.dataset_name:
        with accelerator.main_process_first() if accelerator is not None else nullcontext():
            raw_data = load_dataset(dataset_name)
        if dataset_name == "Anthropic/hh-rlhf":
            dataset_train, dataset_test = preprocess_hh_common(raw_data, remove_multiturn_data=True)
        elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
            dataset_train = raw_data['train_prefs']
            dataset_test = raw_data['test_prefs']
        elif dataset_name == "Magpie-Align/Magpie-Air-DPO-100K-v0.1":
            dataset_train = raw_data['train']
            dataset_test = raw_data['test']
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet.")
        chatml_datasets_train.append(dataset_train)
        chatml_datasets_test.append(dataset_test)

    # merge training datasets
    chatml_dataset_train = chatml_datasets_train[0]
    for dataset in chatml_datasets_train[1:]:
        chatml_dataset_train = chatml_dataset_train.concatenate(dataset)

    # merge test datasets
    chatml_dataset_test = chatml_datasets_test[0]
    for dataset in chatml_datasets_test[1:]:
        chatml_dataset_test = chatml_dataset_test.concatenate(dataset)

    # Subsampling is done later after processing and filtering
    # if args.dataset_subsample_size:
    #     chatml_dataset_train = chatml_dataset_train.shuffle(seed=args.seed).select(range(args.dataset_subsample_size))
    #     chatml_dataset_test = chatml_dataset_test.shuffle(seed=args.seed).select(range(args.dataset_subsample_size))

    return chatml_dataset_train, chatml_dataset_test


def preprocess_hh_common(raw_dataset, remove_multiturn_data=False):
    dataset_train = raw_dataset['train'].map(parse_entry_anthropic_hh)
    dataset_test = raw_dataset['test'].map(parse_entry_anthropic_hh)

    dataset_train = dataset_train.filter(is_entry_ok_anthropic_hh)
    dataset_test = dataset_test.filter(is_entry_ok_anthropic_hh)
    print(f"After filtering, {len(dataset_train)} training examples and {len(dataset_test)} test examples remain.")

    if remove_multiturn_data:
        dataset_train = dataset_train.filter(lambda x: len(x['rejected']) == 2)
        dataset_test = dataset_test.filter(lambda x: len(x['rejected']) == 2)

    dataset_train = dataset_train.map(lambda x: {k: merge_consecutive_messages_and_trim(v) for k, v in x.items()})
    dataset_test = dataset_test.map(lambda x: {k: merge_consecutive_messages_and_trim(v) for k, v in x.items()})

    return dataset_train, dataset_test


def target_lora_modules(model) -> List[str]:
    if model.__class__.__name__ == "Phi3ForCausalLM":
        return ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    elif model.__class__.__name__ == "PhiForCausalLM":
        return ["Wqkv", "out_proj", "fc1", "fc2"]
    elif model.__class__.__name__ == "OLMoForCausalLM":
        return ["att_proj", "ff_proj", "attn_out", "ff_out"]
    elif model.__class__.__name__ == "MistralForCausalLM":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif model.__class__.__name__ == "LlamaForCausalLM":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif model.__class__.__name__ == "OPTForCausalLM":
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    elif model.__class__.__name__ == "GPTNeoXForCausalLM":
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    else:
        raise ValueError(f"Model type {type(model)} not added yet. Model:\n {model}")


def load_tokenizer(args, substitute_eos_token=False):
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
        tokenizer_name = args.tokenizer_name
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            force_download=args.ignore_model_cache,
        )
    elif args.model_name_or_path:
        tokenizer_name = args.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            padding=True,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            force_download=args.ignore_model_cache,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    actual_eos_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
    # if substitute_eos_token:
    #     tokenizer.add_special_tokens({'eos_token': '<my_eos_token>'})

    # The SFTd Phi-2 model
    if tokenizer_name == 'lxuechen/phi-2-sft':
        tokenizer.chat_template = PHI2_CHAT_TEMPLATE
    # this also doesn't provide its own chat template, so we wrote it ourselves
    elif tokenizer_name in ['allenai/tulu-v1-llama2-7b', "allenai/open-instruct-opt-6.7b-tulu", "allenai/open-instruct-pythia-6.9b-tulu" ]:
        tokenizer.chat_template = LLAMA_TULU_CHAT_TEMPLATE
    if "facebook/opt" in tokenizer_name:
        tokenizer.chat_template = LLAMA_TULU_CHAT_TEMPLATE
    if "EleutherAI/pythia" in tokenizer_name:
        tokenizer.chat_template = LLAMA_TULU_CHAT_TEMPLATE
    else:
        assert hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None, \
            f"Tokenizer {tokenizer_name} does not have a chat template and we don't provide one."

    return tokenizer, actual_eos_token


# substitute_eos_token default value to not break existing code
def load_tokenizer_model(accelerator, args, substitute_eos_token=True, load_lora=True):
    ######################################## Config and Tokenizer Loading ########################################
    def try_load_config() -> PretrainedConfig:
        # Load pretrained model and tokenizer
        if args.config_name:
            return AutoConfig.from_pretrained(
                args.config_name,
                trust_remote_code=args.trust_remote_code,
                revision=args.model_revision,
                force_download=args.ignore_model_cache,
            )
        elif args.model_name_or_path:
            return AutoConfig.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=args.trust_remote_code,
                revision=args.model_revision,
                force_download=args.ignore_model_cache,
            )
        else:
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported by this script."
            )

    with accelerator.main_process_first() if accelerator is not None else nullcontext():
        try:
            config = try_load_config()
        except OSError as e:
            print(f"Error loading config: {e}")
            print("Assuming it was a login issue, logging into Huggingface...")

            huggingface_hub.login(token=get_hf_token())
            config = try_load_config()

        ######################################## Model Loading ########################################
        if args.model_name_or_path:
            def try_load_model():
                if args.use_qlora:
                    assert accelerator is not None
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                    device_index = accelerator.local_process_index
                    device_map = {"": device_index}  # force data-parallel training.
                    return AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=config,
                        load_in_4bit=True,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        trust_remote_code=args.trust_remote_code,
                        torch_dtype=torch.bfloat16,
                        use_flash_attention_2=True if args.use_flash_attn else False,
                        revision=args.model_revision,
                        force_download=args.ignore_model_cache,
                    )
                else:
                    return AutoModelForCausalLM.from_pretrained(
                        args.model_name_or_path,
                        from_tf=bool(".ckpt" in args.model_name_or_path),
                        config=config,
                        trust_remote_code=args.trust_remote_code,
                        low_cpu_mem_usage=args.low_cpu_mem_usage,  #
                        torch_dtype=torch.bfloat16,
                        use_flash_attention_2=True if args.use_flash_attn else False,
                        revision=args.model_revision,
                        force_download=args.ignore_model_cache,
                    )

            # Just a hack. Sometimes cache fails, and on the second time the model is loaded properly
            try:
                model = try_load_model()
            except Exception as e:
                model = try_load_model()
        else:
            print("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)

        # Have to use a "fake" eos token that the model never generates as a hack to work around a bug in model.generate
        tokenizer, actual_eos_token = load_tokenizer(args, substitute_eos_token=substitute_eos_token)

        generation_config = GenerationConfig(
            max_new_tokens=args.logging_examples_max_length,
            renormalize_logits=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # stop_strings=tokenizer.eos_token,
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

        if load_lora and args.use_lora:
            print("Initializing LORA model...")
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

        if (accelerator is None) or accelerator.is_main_process:
            print(str(model))

        generation_config_nucleus = deepcopy(generation_config)
        generation_config_nucleus.do_sample = True
        generation_config_nucleus.top_p = args.logging_examples_top_p
        generation_config_nucleus.temperature = args.logging_examples_temp

        generation_config_greedy = deepcopy(generation_config)
        generation_config_greedy.do_sample = False

    if accelerator is not None:
        accelerator.wait_for_everyone()

    return model, tokenizer, actual_eos_token, generation_config_nucleus, generation_config_greedy


if __name__ == "__main__":
    messages_raw = "\n\nHuman: Does time flow forward only, or does it move differently also? \n\nAssistant: It does. \n\nHuman: what else it does Assistant: It flows at different rates for different entities. \n\nHuman: whats the rates \n\nAssistant: Too difficult for me to calculate."
    print(parse_messages_anthropic_hh(messages_raw))

    messages_raw = "\n\nHuman: Does time flow forward only, or does it move differently also? \n\nAssistant: It does. \n\nHuman: what else it does Assistant: It flows at different rates for different entities. \n\nHuman: whats the rates \n\nAssistant:"
    print(parse_messages_anthropic_hh(messages_raw))

    messages_raw = "\n\nHuman: Does time flow forward only, or does it move differently also? \n\nAssistant: It does. \n\nHuman: what else it does Assistant: It flows at different rates for different entities. \n\nHuman: whats the rates \n\nAssistant: "
    print(parse_messages_anthropic_hh(messages_raw))
