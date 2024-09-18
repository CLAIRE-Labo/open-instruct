import sys
import torch
import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import bitsandbytes as bnb
import os
from pathlib import Path
import copy
from bitsandbytes.functional import dequantize_4bit
from peft.utils import _get_submodules
import accelerate

sys.path.append(Path(__file__).parents[1].absolute().as_posix())
from open_instruct.load_utils import load_tokenizer_model, load_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--train_args", type=str, required=True,
                        help="If provided, the tokenizer, model, and lora configuration will be loaded with the same configuration as in training.")
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--save_tokenizer", action="store_true")
    return parser.parse_args()


def main(args):
    accelerator = accelerate.Accelerator()

    train_args = load_args(args.train_args)
    model, tokenizer, actual_eos_token, generation_config_sampling, generation_config_greedy \
        = load_tokenizer_model(accelerator, train_args, substitute_eos_token=True, load_lora=False)
    model.generation_config = generation_config_greedy
    tokenizer.eos_token = actual_eos_token
    peft_model = PeftModel.from_pretrained(model, str(args.lora_model_name_or_path))
    merged_model = peft_model.merge_and_unload()

    output_dir = args.output_dir if args.output_dir else args.lora_model_name_or_path
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)

    if args.save_tokenizer:
        print(f"Saving the tokenizer to {output_dir}...")
        tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)