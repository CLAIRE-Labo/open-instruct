'''
ORPO tuning script
'''
import json
import argparse
import logging
import math
import os
import random
from copy import deepcopy
import datasets
import torch
from functools import partial
from datetime import timedelta
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
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
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from dpo_utils import dpo_loss, concatenated_forward, DataCollatorForSeq2SeqDPO
from datasets import DatasetDict
import sys
import wandb
import os
import gc
from peft import PeftConfig, PeftModel
from trl import ORPOConfig, ORPOTrainer, setup_chat_format
# pip install trl -> update the docker container - TODO
# pip install --upgrade --no-cache-dir git+https://github.com/NVIDIA/apex.git -> update the docker container - TODO
logger = get_logger(__name__)

try:
    from hf_olmo import OLMoTokenizerFast
except ImportError:
    logger.warning("OLMo not installed. Ignore if using a different model.")

from transformers import Trainer
import torch
deepspeed_conf={
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "fp16": {
    "enabled": "auto"
  },
  "bf16": {
    "enabled": "auto"
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": True
    },
    "allgather_partitions": True,
    "allgather_bucket_size": 2e8,
    "overlap_comm": True,
    "reduce_scatter": True,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": True
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": 1e-8,
      "weight_decay": "auto"
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": "auto",
      "warmup_num_steps": "auto"
    }
  }
}

class CustomORPOTrainer(Trainer):
    """
    "Passing optimizers is not allowed if Deepspeed or PyTorch FSDP is enabled. "
                "You should subclass Trainer and override the create_optimizer_and_scheduler method."
    """
    def __init__(self, custom_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_args = custom_args

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a default implementation for convenience, but you can override this method in a subclass if you need.
        """
        if self.optimizer is None:
            no_decay = ["bias", "layer_norm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.custom_args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            if self.custom_args.use_qlora or self.custom_args.use_paged_optimizer:
                from bitsandbytes.optim import AdamW
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.custom_args.learning_rate,
                    optim_bits=8 if self.custom_args.use_8bit_optimizer else 32,
                    is_paged=True
                )
            else:
                self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.custom_args.learning_rate)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                name=self.custom_args.lr_scheduler_type,
                optimizer=self.optimizer,
                num_training_steps=num_training_steps,
                num_warmup_steps=int(num_training_steps * self.custom_args.warmup_ratio),
            )

        return self.optimizer, self.lr_scheduler

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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        '--beta',
        type=float,
        default=0.1,
        help='Beta parameter for DPO loss. Default is 0.1.',
    )
    parser.add_argument(
        '--use_paged_optimizer',
        action='store_true',
        help='Use paged optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
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
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args

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
def _concat_messages(messages, type, tokenizer, add_bos=False):
    message_text = ""
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        for message in messages:
            if message["role"] == "human":
                message_text += "\n" + message["content"].strip() + "\n"
            elif type == "chosen" and message["role"] == "assistant_chosen":
                message_text += "\n" + message["content"].strip() + "\n"
            elif type == "rejected" and message["role"] == "assistant_rejected":
                message_text += "\n" + message["content"].strip() + "\n"
            else:
                if message["role"] not in ["assistant_rejected", "assistant_chosen"]:
                    raise ValueError("Invalid role: {}".format(message["role"]))
    else:
        for message in messages:
            if message["role"] == "human":
                message_text += "\n" + message["content"].strip() + "\n"
            elif type == "chosen" and message["role"] == "assistant_chosen":
                message_text += "\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            elif type == "rejected" and message["role"] == "assistant_rejected":
                message_text += "\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                if message["role"] not in ["assistant_rejected", "assistant_chosen"]:
                    raise ValueError("Invalid role: {}".format(message["role"]))
    if add_bos:
        message_text = tokenizer.bos_token + message_text

    return message_text

def extract_human_message(messages):
    for message in messages:
        if message["role"] == "human":
            return message["content"].strip()
    return ""

def format_chat_template(row, tokenizer, add_bos=False):
    #column names with what TRL needs -> chosen, rejected, prompt
    row["chosen"] = _concat_messages(row["info"], "chosen", tokenizer, add_bos=add_bos)
    row["rejected"] = _concat_messages(row["info"], "rejected", tokenizer,add_bos=add_bos)
    row["prompt"] = extract_human_message(row["info"])
    return row
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
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
    )
    """
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    #with accelerate -> RuntimeError: The size of tensor a (0) must match the size of tensor b (2048) at non-singleton dimension 1
    """
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    """
    if args.seed is not None:
        set_seed(args.seed)
    """
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    """
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    #accelerator.wait_for_everyone()

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

    # for train
    # filter the dataset for it to have only one prompt and answer (not a sequence of prompt-answer in one line)
    updated_dataset_train = raw_datasets['train'].map(add_filtered_msgs)
    filtered_train = updated_dataset_train.filter(
        lambda x: len(x['rejected_filtered']) > 0).select(range(5000))

    # for test
    updated_dataset_test = raw_datasets['test'].map(add_filtered_msgs)
    filtered_test = updated_dataset_test.filter(lambda x: len(x['rejected_filtered']) > 0)

    filtered_dataset = DatasetDict({
        'train': filtered_train,
        'test': filtered_test
    })
    filtered_dataset["train"] = filtered_dataset["train"].map(extract_role_messages)
    filtered_dataset["test"] = filtered_dataset["test"].map(extract_role_messages)

    print("Size of training set:", len(filtered_dataset['train']))
    print("Size of test set:", len(filtered_dataset['test']))

    del raw_datasets, updated_dataset_test, updated_dataset_train
    gc.collect()  # delete unnecessary variables

    raw_datasets = filtered_dataset
    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:

        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer,
                                                  trust_remote_code=args.trust_remote_code)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    def load_model():
        if args.model_name_or_path:
            if args.use_qlora:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                #device_index = accelerator.local_process_index
                device_map = {"": ""}#device_index}  # force data-parallel training.

                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    load_in_4bit=True,
                    trust_remote_code=args.trust_remote_code,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=torch.bfloat16,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )
            else:

                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=args.trust_remote_code,
                    low_cpu_mem_usage=args.low_cpu_mem_usage,
                    use_flash_attention_2=True if args.use_flash_attn else False,
                )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForCausalLM.from_config(config)
        return model

    model = load_model()

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
    """
    # gather deepspeed to get "real" embedding size
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
   """
    #dont use deepspeed for now: RuntimeError: Passing `optimizers` is not allowed if Deepspeed or PyTorch FSDP is enabled. You should subclass `Trainer` and override the `create_optimizer_and_scheduler` method.

    embeddings = model.get_input_embeddings()
    if len(tokenizer) > embeddings.weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
    if args.use_lora:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model,
                                                    use_gradient_checkpointing=args.gradient_checkpointing)

        #logger.info("Initializing LORA model...")
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

        if "chosen" in raw_datasets["train"].column_names and "rejected" in raw_datasets[
            "train"].column_names:
            print("chosen- rejected")
            raw_datasets = raw_datasets.map(
                lambda row: format_chat_template(row, tokenizer, add_bos=args.add_bos),
            )
            raw_datasets["train"] = raw_datasets["train"].select_columns(["prompt", "rejected", "chosen"])
            raw_datasets["test"] = raw_datasets["test"].select_columns(["prompt", "rejected", "chosen"])

        else:
            raise ValueError("You need to have 'chosen' and 'rejected in your column names.")

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
        if args.use_qlora or args.use_paged_optimizer:
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
        num_update_steps_per_epoch = math.ceil(len(raw_datasets["train"]) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * 1  # accelerator.num_processes

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_training_steps=num_training_steps_for_scheduler,
            num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
        )

        orpo_args = ORPOConfig(
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            max_length=args.max_seq_length,
            max_prompt_length= args.max_seq_length,
            beta=args.beta,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim="adamw_torch",
            num_train_epochs=args.num_train_epochs,
            evaluation_strategy=args.checkpointing_steps,  # Evaluate after each epoch
            logging_steps=args.logging_steps,
            report_to=args.report_to,
            output_dir=args.output_dir,
            save_strategy=args.checkpointing_steps ,  # Save the model after each epoch
            deepspeed=deepspeed_conf,
            remove_unused_columns=False,
        )

        collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
        trainer = ORPOTrainer(
            model=model,
            args=orpo_args,
            train_dataset=raw_datasets["train"],
            peft_config=peft_config,
            tokenizer=tokenizer,
            data_collator=collate_fn,
            #optimizers=(optimizer, lr_scheduler)  # Pass the custom optimizer
        )
        trainer.train()
        trainer.save_model(args.output_dir)
if __name__ == "__main__":
    main()

