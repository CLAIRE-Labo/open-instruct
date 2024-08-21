import math

from datasets import load_dataset

from trl import  ORPOConfig, ORPOTrainer, get_peft_config
from datasets import DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    DataCollatorForSeq2Seq,
    SchedulerType
)
import argparse
from peft import LoraConfig, TaskType
import wandb
import os
# NOTE FOR TODO
# pip install trl -> update the docker container - TODO
# pip install --upgrade --no-cache-dir git+https://github.com/NVIDIA/apex.git -> update the docker container - TODO

try:
    from hf_olmo import OLMoTokenizerFast
except ImportError:
    print("OLMo not installed. Ignore if using a different model.")

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

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a language modeling task")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_peft",
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
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")

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
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        '--beta',
        type=float,
        default=0.1,
        help='Beta parameter for DPO loss. Default is 0.1.',
    )

    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )
    parser.add_argument(
            "--lr_scheduler_type",
            type=SchedulerType,
            default="linear",
            help="The scheduler type to use.",
            choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset is None:
        raise ValueError("dataset needed")

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
    # mention the None case -> to confirm alignment with other implementations

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        # phi3 is a llama tokenizer and requires different template compared to OLMo
        """
        for phi3:
          "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}
          {% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
        """
        for message in messages:
            if message["role"] == "human":
                message_text += "<|user|>\n" + message["content"].strip() +'<|end|>'+ "\n"
            elif type == "chosen" and message["role"] == "assistant_chosen":
                message_text += "<|assistant|>\n" + message["content"].strip() + '<|end|>' + "\n"
            elif type == "rejected" and message["role"] == "assistant_rejected":
                message_text += "<|assistant|>\n" + message["content"].strip() + '<|end|>' + "\n"
            else:
                if message["role"] not in ["assistant_rejected", "assistant_chosen"]:
                    raise ValueError("Invalid role: {}".format(message["role"]))
    else:
        for message in messages:
            if message["role"] == "human":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif type == "chosen" and message["role"] == "assistant_chosen":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            elif type == "rejected" and message["role"] == "assistant_rejected":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                if message["role"] not in ["assistant_rejected", "assistant_chosen"]:
                    raise ValueError("Invalid role: {}".format(message["role"]))
    # Optionally add BOS token at the beginning
    if add_bos:
        tokenizer.bos_token= tokenizer.eos_token #just for olmo
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

if __name__ == "__main__":
    args = parse_args()

    if args.with_tracking:
        if "wandb" in args.report_to.split(",") or args.report_to == "all":
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)
            # Initialize wandb
            wandb.init(project="alignment_translation", entity="claire-labo")

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset)
    # for train
    # filter the dataset for it to have only one prompt and answer (not a sequence of prompt-answer in one line)
    updated_dataset_train = ds['train'].map(add_filtered_msgs)
    filtered_train = updated_dataset_train.filter(
        lambda x: len(x['rejected_filtered']) > 0)#.select(range(5000)) #delete afterwards

    # for test
    updated_dataset_test = ds['test'].map(add_filtered_msgs)
    filtered_test = updated_dataset_test.filter(lambda x: len(x['rejected_filtered']) > 0)

    filtered_dataset = DatasetDict({
        'train': filtered_train,
        'test': filtered_test
    })
    filtered_dataset["train"] = filtered_dataset["train"].map(extract_role_messages)
    filtered_dataset["test"] = filtered_dataset["test"].map(extract_role_messages)

    print("Size of training set:", len(filtered_dataset['train']))
    print("Size of test set:", len(filtered_dataset['test']))

    ds = filtered_dataset.map(
        lambda row: format_chat_template(row, tokenizer, add_bos=True),
    )
    required_columns = ['chosen', 'rejected', 'prompt']
    ds['train'] = ds['train'].remove_columns(
        [col for col in ds['train'].column_names if col not in required_columns]
    )
    ds['test'] = ds['test'].remove_columns(
        [col for col in ds['test'].column_names if col not in required_columns]
    )

    train_dataset = ds["train"]
    eval_dataset = ds["test"]
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    orpo_args = ORPOConfig(
        learning_rate=args.learning_rate,
        optim="adamw_torch",  # Specify optim once, choose the correct one you want to use
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        lr_scheduler_type=args.lr_scheduler_type,
        max_length=args.max_seq_length,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        evaluation_strategy=args.checkpointing_steps,  # Evaluate after each epoch
        report_to=args.report_to,
        output_dir=args.output_dir,
        save_strategy=args.checkpointing_steps,  # Save the model after each epoch
        remove_unused_columns=False,
        max_steps=args.max_train_steps,
        gradient_checkpointing=args.gradient_checkpointing
    )

    ################
    # Training
    ################
    peft_config=None
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
    #collate_fn = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")
    # the one in use: DPODataCollatorWithPadding(pad_token_id=1, label_pad_token_id=-100, is_encoder_decoder=False)
    trainer = ORPOTrainer(
        model,
        args=orpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        #data_collator=collate_fn,
    )

    # train and save the model
    trainer.train()
    trainer.save_model(orpo_args.output_dir)