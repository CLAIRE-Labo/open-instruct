import sys
import argparse
import pickle
from base64 import encode
from pathlib import Path
import torch
from tqdm import tqdm
import vllm
import sys

sys.path.append((Path(__file__).parents[1].absolute().as_posix()))
from eval.utils import maybe_create_reformatted_lora_checkpoint

from eval.chat import generate_responses_vllm
from load_utils import load_tokenizer

def run_model_for_student_vllm(args,input_sampling_params, batch):
    model_name_or_path = args.model_name_or_path
    tokenizer_name_or_path = getattr(args, "tokenizer_name", None)


    hf_revision = getattr( args, "model_revision", "main")

    # Load tokenizer
    tokenizer = load_tokenizer(args, substitute_eos_token=False)
    max_new_tokens= input_sampling_params.max_new_tokens
    if input_sampling_params.use_vllm:
        print("Loading vLLM model...")
        mem_util = args.mem_util if hasattr(input_sampling_params, "mem_util") else 0.9 if input_sampling_params.is_lora else 0.4

        base_model = vllm.LLM(
            model=model_name_or_path,
            revision=hf_revision,
            tokenizer=tokenizer_name_or_path,
            tokenizer_revision=hf_revision,
            tokenizer_mode="auto" if not args.use_slow_tokenizer else "slow",
            trust_remote_code=args.trust_remote_code,
            enable_lora=input_sampling_params.is_lora,
            max_lora_rank=128,
            disable_sliding_window=True if not hasattr(input_sampling_params, "disable_sliding_window") else input_sampling_params.disable_sliding_window,
            gpu_memory_utilization=mem_util,
        )

        sampling_params = vllm.SamplingParams(
            top_p=1.0 if input_sampling_params.greedy else input_sampling_params.top_p,
            temperature=0.0 if input_sampling_params.greedy else input_sampling_params.temperature,
            seed=239,
            max_tokens=input_sampling_params.max_new_tokens,
            n=1 if not hasattr(input_sampling_params, "n_sample_per_prompt") else input_sampling_params.n_sample_per_prompt,
        )

        responses_log = []

        if input_sampling_params.is_lora:
            # Not ATT, using LoRA
            lora_request=None
            if args.lora_model_name_or_path:
                reformatted_checkpoint = maybe_create_reformatted_lora_checkpoint(args.lora_model_name_or_path, cache_dir=args.cache_dir)
                lora_request = vllm.lora.request.LoRARequest("adapter", 1, reformatted_checkpoint.absolute().as_posix())

            formatted_prompts, outputs, all_generated_tokens, all_new_attention_masks, all_new_labels = generate_responses_vllm(
                base_model,
                tokenizer,
                prompts,
                sampling_params,
                lora_request=lora_request,
                batch_size=args.batch_size
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

            formatted_prompts, outputs, all_generated_tokens, all_new_attention_masks, all_new_labels = generate_responses_vllm(
                tuned_model, tokenizer, prompts, sampling_params, max_new_tokens
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
parser.add_argument("--pickle_prompts", type=Path, required=True, help="Path to the pickle file for batch data")
parser.add_argument("--pickle_sampling_params", type=Path, required=True, help="Path to the pickle file for sampling parameters")
parser.add_argument("--pickle_output", type=Path, required=True, help="Path to the pickle file for output results")
parser.add_argument("--mem_util", type=float, default=0.4, help="GPU memory utilization")
parser.add_argument("--batch_size", type=int, default=30, help="Batch size for model generation")

parser.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading the tokenizer")
parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use the slow tokenizer")
parser.add_argument("--ignore_model_cache", action="store_true", help="Force download the model even if it's cached")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="The specific revision of the tokenizer to use")
parser.add_argument("--model_revision", type=str, default="main", help="The specific revision of the model to use")

parser.add_argument("--cache_dir", type=str, default= None, help= "cache directory path")
args = parser.parse_args()

# Load the batch
with open(args.pickle_prompts, "rb") as f:
    prompts = pickle.load(f)

# Load sampling params
with open(args.pickle_sampling_params, "rb") as f:
    input_sampling_params = pickle.load(f)

# Call the function to generate responses
responses_log = run_model_for_student_vllm(args, input_sampling_params, batch=prompts)

# Save the results in pickle format
with open(args.pickle_output, "wb") as f:
    pickle.dump(responses_log, f)


