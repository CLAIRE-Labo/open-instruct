import sys
import argparse
import pickle
from base64 import encode
from pathlib import Path
import torch
from ray.tune.examples.pbt_dcgan_mnist.common import batch_size
from tqdm import tqdm
import vllm
import sys
import json
import os
import multiprocessing

sys.path.append((Path(__file__).parents[1].absolute().as_posix()))
from eval.utils import maybe_create_reformatted_lora_checkpoint
from eval.chat import generate_responses_vllm

from load_utils import load_tokenizer
import torch.multiprocessing as mp


def run_model_for_student_vllm(args, prompts):
    model_name_or_path = args.model_name_or_path
    tokenizer_name_or_path = getattr(args, "tokenizer_name", None)


    hf_revision = getattr( args, "model_revision", "main")

    # Load tokenizer
    tokenizer, _ = load_tokenizer(args, substitute_eos_token=False)

    responses_log=[]
    if args.use_vllm:
        print("Loading vLLM model...")
        mem_util = args.mem_util if hasattr(args, "mem_util") else 0.9 if args.is_lora else 0.4

        base_model = vllm.LLM(
            model=model_name_or_path,
            revision=hf_revision,
            tokenizer=tokenizer_name_or_path,
            tokenizer_revision=hf_revision,
            tokenizer_mode="auto" if not args.use_slow_tokenizer else "slow",
            trust_remote_code=args.trust_remote_code,
            enable_lora=args.is_lora,
            max_lora_rank=128,
            disable_sliding_window=True if not hasattr(args, "disable_sliding_window") else args.disable_sliding_window,
            gpu_memory_utilization=mem_util
        )

        sampling_params = vllm.SamplingParams(
            top_p=1.0 if args.greedy else args.top_p,
            temperature=0.0 if args.greedy else args.temperature,
            seed=239,
            max_tokens=128,
            n=1 if not hasattr(args, "n_sample_per_prompt") else args.n_sample_per_prompt,
        )

        if args.is_lora:
            # Not ATT, using LoRA
            lora_request=None
            if args.lora_model_name_or_path:
                reformatted_checkpoint = maybe_create_reformatted_lora_checkpoint(args.lora_model_name_or_path, cache_dir=args.cache_dir)
                lora_request = vllm.lora.request.LoRARequest("adapter", 1, reformatted_checkpoint.absolute().as_posix())

            formatted_prompts, outputs= generate_responses_vllm(
                base_model,
                tokenizer,
                prompts,
                sampling_params,
                lora_request=lora_request,
                batch_size=int(args.batch_size)
            )
        else:
            # Not ATT, not using LoRA
            tuned_model = vllm.LLM(
                model=args.lora_model_name_or_path.absolute().as_posix(),
                revision=args.model_revision,
                tokenizer=tokenizer_name_or_path,
                tokenizer_revision=args.model_revision,
                tokenizer_mode="auto" if not args.use_slow_tokenizer else "slow",
                trust_remote_code=args.trust_remote_code,
                enable_lora=False,
                disable_sliding_window=True,
                gpu_memory_utilization=mem_util,
            )

            formatted_prompts, outputs= generate_responses_vllm(
                tuned_model, tokenizer, prompts, sampling_params
            )

        # Collect responses
        for prompt, output in zip(formatted_prompts, outputs):
            responses_log.append({
                "prompt": prompt,
                "output": output
            })

        del base_model  # Free up GPU memory
    return responses_log

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, help="Path to the model checkpoint")
parser.add_argument("--lora_model_name_or_path", type=str, help="Path to the lora model checkpoint")
parser.add_argument("--tokenizer_name", type=str, required=False, help="Name or path of the tokenizer")
parser.add_argument("--generated_res", type=Path, required=True, help="Path of resulting generations file")
parser.add_argument("--batch_file", type=Path, required=True, help="Path of prompts file")

parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
parser.add_argument("--top_p", type=float, default=0.9, help="Top p for sampling")
parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
parser.add_argument("--greedy", action="store_true", help="Whether to use greedy sampling")
parser.add_argument("--n_sample_per_prompt", type=int, default=1, help="Number of samples to generate per prompt")
parser.add_argument("--mem_util", type=float, default=0.4, help="GPU memory utilization")
parser.add_argument("--batch_size", type=str, default="30", help="Batch size for model generation")

parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading the tokenizer")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use the slow tokenizer")
parser.add_argument("--ignore_model_cache", action="store_true", help="Force download the model even if it's cached")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="The specific revision of the tokenizer to use")
parser.add_argument("--model_revision", type=str, default="main", help="The specific revision of the model to use")

parser.add_argument("--cache_dir", type=str, default= None, help= "cache directory path")

parser.add_argument('--is_lora', action='store_true', help='Enable LoRA support')
parser.add_argument('--use_vllm', action='store_true', help='Enable vLLM model usage')
parser.add_argument('--disable_sliding_window', action='store_true', help='Disable sliding window for inference')

args = parser.parse_args()

# Set multiprocessing to use 'spawn' method for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)

# read the prompts from json file
with open(args.batch_file, "r") as f:
    prompts = json.load(f)
# Call the function to generate responses
responses_log = run_model_for_student_vllm(args, prompts=prompts)

#write the things to the files
# Extract the directory and filename from the batch_file path
directory = os.path.dirname(args.batch_file)
filename = os.path.basename(args.batch_file)

# Create the new filename with the 'res_' prefix
new_filename = f"res_{filename}"

# Create the full path for the new file
result_file = os.path.join(directory, new_filename)

# Store the results in the new JSON file
with open(result_file, 'w') as f:
    json.dump(responses_log, f)

print(f"Results stored in: {result_file}")

