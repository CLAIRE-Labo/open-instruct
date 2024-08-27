from typing import List, Dict
import re
import argparse
import logging
from copy import deepcopy

from transformers import SchedulerType


logger = logging.getLogger(__name__)


def add_common_training_args(parser: argparse.ArgumentParser):
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


if __name__ == "__main__":
    messages_raw = "\n\nHuman: Does time flow forward only, or does it move differently also? \n\nAssistant: It does. \n\nHuman: what else it does Assistant: It flows at different rates for different entities. \n\nHuman: whats the rates \n\nAssistant: Too difficult for me to calculate."
    print(parse_messages_anthropic_hh(messages_raw))

    messages_raw = "\n\nHuman: Does time flow forward only, or does it move differently also? \n\nAssistant: It does. \n\nHuman: what else it does Assistant: It flows at different rates for different entities. \n\nHuman: whats the rates \n\nAssistant:"
    print(parse_messages_anthropic_hh(messages_raw))

    messages_raw = "\n\nHuman: Does time flow forward only, or does it move differently also? \n\nAssistant: It does. \n\nHuman: what else it does Assistant: It flows at different rates for different entities. \n\nHuman: whats the rates \n\nAssistant: "
    print(parse_messages_anthropic_hh(messages_raw))
