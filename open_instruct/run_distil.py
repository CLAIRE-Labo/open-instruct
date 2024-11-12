import torch.cuda
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import wandb
sys.path.append(Path(__file__).parents[1].absolute().as_posix())
from load_utils import add_common_training_args,  preprocess_data_to_chatml_without_accelerator
import math

import argparse
import json
import os
import torch.cuda
from datasets import DatasetDict
from pathlib import Path
from tqdm import tqdm
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

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


# Add command-line arguments for autoregressive generation
def add_autoregressive_gen_args(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_model_name_or_path_student", type=str, help="Path to the LoRA model checkpoint")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p for sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--greedy", action="store_true", help="Whether to use greedy sampling")
    parser.add_argument("--n_sample_per_prompt", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--mem_util", type=float, default=0.4, help="GPU memory utilization")
    parser.add_argument("--batch_size", type=int, default=25, help="Batch size for model generation")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory path")
    parser.add_argument('--is_lora', action='store_true', help='Enable LoRA support')
    parser.add_argument('--use_vllm', action='store_true', help='Enable vLLM model usage')
    parser.add_argument('--disable_sliding_window', action='store_true', help='Disable sliding window for inference')
    parser.add_argument("--generation_storage_train", type=str, help="The path of yminus generations for train", required=True)
    parser.add_argument("--generation_storage_test", type=str, help="The path of yminus generations for test")


def read_generations(generation_storage):
    prompts_path = os.path.join(generation_storage, "prompts.json")
    generations_path = os.path.join(generation_storage, "generations.json")
    with open(prompts_path, "r") as fin:
        prompts = json.load(fin)
    with open(generations_path, "r") as fin:
        generations = json.load(fin)
    return prompts, generations

# Replace assistant content with generation content based on prompts
def replace_assistant_content(filtered_dataset, prompts_train, generations_train):
    replacements_count = 0
    for idx in tqdm(range(len(prompts_train)), desc="Processing replacements"):
        filtered_item = filtered_dataset[idx]
        if 'rejected' in filtered_item:
            user_content = None
            for item in filtered_item["rejected"]:
                if item['role'] == 'user':
                    user_content = item['content']
                    break

            if user_content == prompts_train[idx][0]['content']:
                for item in filtered_item['rejected']:
                    if item['role'] == 'assistant':
                        item['content'] = generations_train[idx]
                        replacements_count += 1
                        break
    return replacements_count


def chunk_data(dataset, num_gpus, min_chunk_size=1500):
    # Determine the number of chunks based on the minimum chunk size
    num_chunks = max(num_gpus, len(dataset) // min_chunk_size)

    # Calculate the actual chunk size
    chunk_size = len(dataset) // num_chunks

    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        if i == num_chunks - 1:
            # For the last chunk, take all remaining data
            chunks.append(dataset[start_idx:])
        else:
            chunks.append(dataset[start_idx:start_idx + chunk_size])

    return chunks


# Function to execute a subprocess with specific arguments for each GPU
def run_vllm_student(gpu_id, batch_folder, batch_file, args):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    vllm_script = (Path(__file__).parents[0] / "run_vllm_student.py").absolute().as_posix()

    command = [
                  'python', vllm_script,
                  '--batch_file', batch_file,
                  '--model_name_or_path', args.model_name_or_path,
                  '--generated_res', str(batch_folder) if batch_folder else batch_folder,
                  '--max_new_tokens', str(args.max_new_tokens) if args.max_new_tokens is not None else args.max_new_tokens,
                  '--top_p', str(args.top_p) if args.top_p is not None else args.top_p,
                  '--temperature', str(args.temperature) if args.temperature is not None else args.temperature,
                  '--n_sample_per_prompt', str(args.n_sample_per_prompt) if args.n_sample_per_prompt is not None else args.n_sample_per_prompt,
                  '--mem_util', str(args.mem_util) if args.mem_util is not None else args.mem_util,
                  '--batch_size', str(args.batch_size) if args.batch_size is not None else args.batch_size,
                  '--tokenizer_revision', args.tokenizer_revision if args.tokenizer_revision else 'main',
                  '--model_revision', args.model_revision if args.model_revision else 'main'
              ] + (['--lora_model_name_or_path', getattr(args, 'lora_model_name_or_path_student')] if getattr(args, 'lora_model_name_or_path_student', None) else []) + \
              (['--tokenizer_name', getattr(args, 'tokenizer_name')] if getattr(args, 'tokenizer_name', None) else []) + \
              (['--cache_dir', getattr(args, 'cache_dir')] if getattr(args, 'cache_dir', None) else [])

    flags = [
        ('--trust_remote_code', args.trust_remote_code),
        ('--use_slow_tokenizer', args.use_slow_tokenizer),
        ('--greedy', args.greedy),
        ('--is_lora', args.is_lora),
        ('--use_vllm', args.use_vllm),
        ('--disable_sliding_window', args.disable_sliding_window),
        ('--ignore_model_cache', args.ignore_model_cache)
    ]

    command += [flag for flag, condition in flags if condition]
    return subprocess.Popen(command, env=env)



def run_batch_on_gpu(gpu_id, batch_idx, batch_file, batch_folder, args):
    process = run_vllm_student(gpu_id, batch_folder, str(batch_file), args)
    process.wait()
    print(f"Completed batch {batch_idx} on GPU {gpu_id}.")

    response_file = batch_folder / f'res_batch_gpu_{gpu_id}_batch_{batch_idx}.json'
    try:
        if response_file.exists():
            with open(response_file, 'r') as res_f:
                responses = json.load(res_f)
            print(f"Responses for batch {batch_idx} on GPU {gpu_id} are taken.")
            return responses
        else:
            print(f"Response file {response_file} does not exist yet.")
            return None
    except Exception as e:
        print(f"Error reading response file {response_file}: {e}")
        return None

def create_batch_dataset(sub_batch_chosen, sub_batch_rejected):
    # Creating a new batch dataset with chosen and rejected batches
    batch_dataset = DatasetDict({
        'chosen': sub_batch_chosen,
        'rejected': sub_batch_rejected
    })

    return batch_dataset

def convert_responses_to_train_format(all_responses):
    prompts_train = []
    generations_train = []

    for entry in all_responses:
        prompt = entry['prompt']
        output = entry['output']

        prompts_train.append(prompt)
        generations_train.append(output)

    return prompts_train, generations_train

def main():
    parser = argparse.ArgumentParser(description="Main distillation file with subprocesses")
    add_common_training_args(parser)
    #add_distill_args(parser)
    add_autoregressive_gen_args(parser)
    args = parser.parse_args()
    run_id=None

    if args.with_tracking:
        if "wandb" in args.report_to.split(",") or args.report_to == "all":
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)

            # Initialize wandb
            wandb.init(project="alignment_translation", entity="claire-labo")
            run_id = wandb.run.id

    # Load prompts and generations from storage
    prompts_train, generations_train = read_generations(args.generation_storage_train)
    prompts_test, generations_test = read_generations(args.generation_storage_test)

    dataset_train, dataset_test = preprocess_data_to_chatml_without_accelerator(args)

    filtered_dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    # Replace rejected contents
    count_of_replacements_train = replace_assistant_content(filtered_dataset['train'], prompts_train, generations_train)
    count_of_replacements_test = replace_assistant_content(filtered_dataset['test'], prompts_test, generations_test)

    if count_of_replacements_test == len(prompts_test) and count_of_replacements_train == len(prompts_train):
        print("Replacement is done successfully")
    else:
        print("There is a mistake in replacements")

    num_gpus = torch.cuda.device_count()  # Get the number of available GPUs

    total_size = len(prompts_train)  # Total number of elements in prompts_train
    min_chunk_size = 1250  # Minimum size of each chunk  - each 1000 train 250 test

    dataset_chunks = chunk_data(prompts_train, num_gpus, min_chunk_size=min_chunk_size)
    batch_folder = Path(__file__).parents[0] / 'tmp_batches'
    batch_folder.mkdir(exist_ok=True)

    args.lora_model_name_or_path_student = None

    for epoch in range(args.num_train_epochs):
        # To collect all responses
        rounds_needed= math.ceil(len(dataset_chunks) / num_gpus)
        count=-1

        all_responses = []
        dataset_end=0
        # Iterate over all rounds
        for round_idx in range(rounds_needed):
            futures = []
            # Determine the start and end indices for the current round's chunks
            start_idx = round_idx * num_gpus  # Starting index for this round
            end_idx = min(start_idx + num_gpus, len(dataset_chunks))

            # Use ProcessPoolExecutor to handle multiple GPUs
            with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                dataset_start= dataset_end
                for gpu_id, chunk_idx in enumerate(range(start_idx, end_idx)):
                    sub_batch = dataset_chunks[chunk_idx]

                    # Define the batch file path for this chunk
                    batch_file = batch_folder / f'batch_gpu_{gpu_id}_round_{round_idx}.json'

                    # Save the sub-batch to a file
                    with open(batch_file, 'w') as f:
                        json.dump(sub_batch, f)
                    dataset_end+=len(dataset_chunks[chunk_idx])

                    futures.append(executor.submit(run_batch_on_gpu, gpu_id, round_idx, batch_file, batch_folder, args))

                # Wait for all GPUs to finish their sub-batches in this round
                for future in as_completed(futures):
                    result = future.result()

                print(f"Completed round {round_idx} for all GPUs.")


                for gpu_id in range(num_gpus):
                    res_file = batch_folder / f'res_batch_gpu_{gpu_id}_round_{round_idx}.json'

                    if res_file.exists():
                        with open(res_file, 'r') as f:
                            responses = json.load(f)

                        all_responses.extend(responses)
                        print(f"Read and collected results from {res_file}")
                    else:
                        print(f"Result file {res_file} not found.")

                sub_batch_chosen = filtered_dataset['train']["chosen"][dataset_start:dataset_end]

                sub_batch_rejected = []
                for item in all_responses:
                    # Create a list of dialogue pairs for each item
                    dialogue = []

                    # Add user prompts to the dialogue
                    for prompt in item['prompt']:
                        dialogue.append({
                            'content': prompt['content'],
                            'role': prompt['role']
                        })

                    # Add assistant output to the dialogue
                    dialogue.append({
                        'content': item['output'],
                        'role': 'assistant'
                    })

                    sub_batch_rejected.append(dialogue)

                #to make sure that the prompts are aligned
                for idx, (chosen_prompts, conversation) in enumerate(zip(sub_batch_chosen, sub_batch_rejected)):
                    # Extract only 'user' prompts from each conversation
                    chosen_user_prompts = [entry for entry in chosen_prompts if entry['role'] == 'user']
                    conversation_user_prompts = [entry for entry in conversation if entry['role'] == 'user']

                    # Check if the 'user' prompts match between sub_batch_chosen and sub_batch_rejected
                    if chosen_user_prompts != conversation_user_prompts:
                        raise ValueError(f"Mismatch found at index {idx}:\nExpected: {conversation_user_prompts}\nFound: {chosen_user_prompts}")

                print("Size of sub_batch_chosen:", len(sub_batch_chosen))
                print("Size of sub_batch_rejected:", len(sub_batch_rejected))

                batch_dataset = create_batch_dataset(sub_batch_chosen, sub_batch_rejected)

                # Save the manipulated batch into a pickle file
                pickle_file = batch_folder / f'batch_{round_idx}.pkl'
                with open(pickle_file, 'wb') as f:
                    pickle.dump(batch_dataset, f)

                # delete res_batch_idx_{idx}.json and batch_idx_{idx}.json
                for file_path in batch_folder.glob('res_batch_gpu_*.json'):
                    file_path.unlink()

                for file_path in batch_folder.glob('batch_gpu_*.json'):
                    file_path.unlink()


                #run_accelerate_subprocess(args, pickle_file, count, run_id)
                count=count+1 # starts in -1 and then 0 (first increment)
                checkpoint_path = Path(args.output_dir) / f"epoch_{count}"
                #get the output adapter as a lora model for run_vllm_student
                args.lora_model_name_or_path_student = checkpoint_path


if __name__ == "__main__":
    main()









