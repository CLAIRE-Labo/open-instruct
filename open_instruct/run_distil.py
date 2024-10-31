import argparse
import json
import os

import torch.cuda
from click.core import batch
from datasets import DatasetDict
import sys
from pathlib import Path
from tqdm import tqdm
import subprocess
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import wandb
sys.path.append(Path(__file__).parents[1].absolute().as_posix())
from load_utils import add_common_training_args, preprocess_data_to_chatml


import argparse
import json
import os
import torch.cuda
from datasets import DatasetDict
import sys
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


# Add command-line arguments for distillation
def add_distill_args(parser: argparse.ArgumentParser):
    parser.add_argument('--reduce_loss', default='mean', choices=['mean', 'sum'], help='How to reduce loss over tokens. Default is mean.')
    parser.add_argument("--remove_multiturn_data", action="store_true", help='If set, only "prompt-response" data is used and multi-turn dialogue data is filtered out.')
    parser.add_argument("--use_new_token_template", action="store_true", help="If set, the new token template is used for ATT training.")
    parser.add_argument("--teacher_lora_model_name_or_path", type=str, help="Teacher model path")
    parser.add_argument("--train_args", type=str, help="The tokenizer, model and lora config will be loaded with the same config as in training", required=True)
    parser.add_argument("--generation_storage_train", type=str, help="The path of yminus generations for train", required=True)
    parser.add_argument("--generation_storage_test", type=str, help="The path of yminus generations for test")
    parser.add_argument("--config_file", type=str, help="The path of config file")
    parser.add_argument("--lambda_value", type=int, default=1, required=False)


# Add command-line arguments for autoregressive generation
def add_autoregressive_gen_args(parser: argparse.ArgumentParser):
    parser.add_argument("--lora_model_name_or_path", type=str, help="Path to the LoRA model checkpoint")
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
              ] + (['--lora_model_name_or_path', getattr(args, 'lora_model_name_or_path')] if getattr(args, 'lora_model_name_or_path', None) else []) + \
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



import subprocess

def run_accelerate_subprocess(args, pickle_batch, count, run_id):
    command = [
        "accelerate", "launch",
        "--mixed_precision", "bf16",
        "--num_machines", "1",
        "--num_processes", "8",
        "--use_deepspeed",
        "--deepspeed_config_file", "open-instruct/ds_configs/stage3_no_offloading_accelerate.conf",
        "open-instruct/open_instruct/prob_distil.py",
        "--dataset_name", args.dataset_name[0],
        "--model_name_or_path", args.model_name_or_path,
        "--tokenizer_name", "allenai/tulu-v1-llama2-7b",
        "--add_bos",
        "--use_lora",
        "--lora_rank", str(args.lora_rank),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--max_seq_length", str(args.max_seq_length),
        "--preprocessing_num_workers", "16",
        "--checkpointing_steps", "epoch",
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.",
        "--weight_decay", "0.",
        "--num_train_epochs", "1",
        "--output_dir", args.output_dir,
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--logging_examples_ignore_first",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--trust_remote_code",
        "--logging_steps", "1",
        "--remove_multiturn_data",
        "--teacher_lora_model_name_or_path", args.teacher_lora_model_name_or_path,
        "--pickle_batch", str(pickle_batch),
        "--with_tracking",
        "--wandb_id", str(run_id)
    ]

    if count >= 0:
        checkpoint_path = Path(args.output_dir) / f"epoch_{count}"
        command += ["--resume_from_checkpoint", str(checkpoint_path)]

        # Check if the current epoch checkpoint exists
        if checkpoint_path.exists():
            # If we're at `epoch_n`, delete `epoch_(n-2)` if it exists
            prev_epoch_path = Path(args.output_dir) / f"epoch_{count - 2}"

            if prev_epoch_path.exists() and prev_epoch_path.is_dir():
                # Recursively delete the previous checkpoint folder
                print(f"Deleting old checkpoint: {prev_epoch_path}")
                for item in prev_epoch_path.glob('*'):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        os.rmdir(item)
                prev_epoch_path.rmdir()
            else:
                print(f"No previous checkpoint found for epoch {count - 2}")


    try:
        # Use Popen to run the command asynchronously
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Stream the output in real-time
        for stdout_line in iter(process.stdout.readline, ""):
            print(stdout_line, end="")

        # Wait for the process to finish and capture any remaining stderr output
        process.stdout.close()
        stderr = process.communicate()[1]

        # Check the return code
        if process.returncode == 0:
            print("Subprocess completed successfully.")
        else:
            print(f"Subprocess failed with error code {process.returncode}")
            print("Standard error output:", stderr)

    except Exception as ex:
        print(f"An unexpected error occurred: {str(ex)}")




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


def replace_chosen_assistant_content(filtered_item, prompts_train, generations_train):
    replacements_count = 0
    for idx in tqdm(range(len(prompts_train)), desc="Processing replacements"):
        if 'chosen' in filtered_item:  # Check if the 'chosen' key exists
            user_content = None

            # Find the corresponding user prompt content
            for item in filtered_item['chosen']:
                for sub_item in item:
                    if sub_item['role'] == 'user':
                        user_content = sub_item['content']
                        break

            # Compare user content with the current prompt in the train dataset
            if user_content == prompts_train[idx][0]['content']:
                for item in filtered_item['chosen']:
                    for sub_item in item:
                        if sub_item['role'] == 'assistant':
                            # Replace the assistant's content with the generated response
                            sub_item['content'] = generations_train[idx]
                            replacements_count += 1
                            break

    return replacements_count

# Split the dataset into parts based on the number of GPUs
def split_dataset(dataset, num_splits):
    # Calculate the split point for the dataset
    split_size = len(dataset) // num_splits

    # Create sequential splits
    splits = []
    for i in range(num_splits):
        start_idx = i * split_size
        if i == num_splits - 1:
            # For the last split, take all remaining data (in case of uneven split)
            splits.append(dataset[start_idx:])
        else:
            splits.append(dataset[start_idx:start_idx + split_size])

    return splits


# Function to process each batch on the assigned GPU
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


def convert_responses_to_train_format(all_responses):
    prompts_train = []
    generations_train = []

    for entry in all_responses:
        # Extracting the prompt and output
        prompt = entry['prompt']
        output = entry['output']

        # Adding the prompt to prompts_train
        prompts_train.append(prompt)

        # Adding the output to generations_train
        generations_train.append(output)

    return prompts_train, generations_train

def create_batch_dataset(sub_batch_chosen, sub_batch_rejected):
    # Creating a new batch dataset with chosen and rejected batches
    batch_dataset = DatasetDict({
        'chosen': sub_batch_chosen,
        'rejected': sub_batch_rejected
    })

    return batch_dataset

def main():
    parser = argparse.ArgumentParser(description="Main distillation file with subprocesses")
    add_common_training_args(parser)
    add_distill_args(parser)
    add_autoregressive_gen_args(parser)
    args = parser.parse_args()

    if args.with_tracking:
        if "wandb" in args.report_to.split(",") or args.report_to == "all":
            wandb_api_key = os.getenv('WANDB_API_KEY')
            wandb.login(key=wandb_api_key)

            # Initialize wandb
            wandb.init(project="alignment_translation", entity="claire-labo")
            run_id = wandb.run.id

    count=0
    # Load prompts and generations from storage
    prompts_train, generations_train = read_generations(args.generation_storage_train)
    prompts_test, generations_test = read_generations(args.generation_storage_test)

    dataset_train, dataset_test = preprocess_data_to_chatml(args)

    #for debugging
    prompts_train = prompts_train[:4000]  # Use only the first 4000 prompts
    generations_train = generations_train[:4000]  # Match the prompts with corresponding generations
    dataset_train = dataset_train.select(range(4000))  # Limit the preprocessed training dataset to 4000

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

    def create_chunks(prompts_train, min_chunk_size=1000):
        # Calculate the total size of the dataset
        total_size = len(prompts_train)

        # Initialize an empty list to hold the chunks
        chunks = []

        # Loop through the dataset, creating chunks of the specified minimum size
        for i in range(0, total_size, min_chunk_size):
            # Slice the prompts_train list to create a chunk of min_chunk_size
            chunk = prompts_train[i:i + min_chunk_size]
            chunks.append(chunk)

        return chunks

    total_size = len(prompts_train)  # Total number of elements in prompts_train
    min_chunk_size = 2000  # Minimum size of each chunk

    # Split the dataset into chunks of at least 1000 elements
    dataset_chunks = create_chunks(prompts_train, min_chunk_size)
    batch_folder = Path(__file__).parents[0] / 'tmp_batches'
    batch_folder.mkdir(exist_ok=True)
    for epoch in range(args.num_train_epochs):
        # To collect all responses
        all_responses = []
        dataset_end=0
        # Iterate over all rounds
        count=-1
        for idx, chunk in enumerate(dataset_chunks):
            dataset_start= dataset_end
            batch_file = batch_folder / f'batch_idx_{idx}.json'

            # Save the sub-batch to a file
            with open(batch_file, 'w') as f:
                json.dump(chunk, f)

            dataset_end+=len(chunk)
            gpu_id=0 #use only 1 gpu

            #TODO: tradeoff between the number of used gpus and the number of generations
            #we want to be as much as close to autoregressive generations - which means less number is better
            #but the other gpus are not working, if we include them the model will be downloaded to each - computational cost
            #so increase the number of generations??


            # Run the subprocess for each chunk
            process = run_vllm_student(gpu_id, batch_folder, batch_file, args)

            # Wait for the subprocess to finish before continuing
            process.wait()

            res_file = batch_folder / f'res_batch_idx_{idx}.json'

            # Check if the result file exists
            if res_file.exists():
                with open(res_file, 'r') as f:
                    responses = json.load(f)
            all_responses.extend(responses)

            if len(responses)>0:
                print("successfully getting the results")


            sub_batch_chosen = filtered_dataset['train']["chosen"][dataset_start:dataset_end]
            sub_batch_rejected= filtered_dataset["train"]["rejected"][dataset_start:dataset_end]
            print("Size of sub_batch_chosen:", len(sub_batch_chosen))
            print("Size of sub_batch_rejected:", len(sub_batch_rejected))


            batch_dataset = create_batch_dataset(sub_batch_chosen, sub_batch_rejected)

            # Convert all_responses to the desired format
            prompts_train_batch, generations_train_batch = convert_responses_to_train_format(all_responses)

            count_replace = replace_chosen_assistant_content(batch_dataset, prompts_train_batch, generations_train_batch)
            if count_replace != len(batch_dataset["chosen"]):
                raise ValueError("The count does not match the length of the 'chosen' field in batch_dataset.")

            # Save the manipulated batch into a pickle file
            pickle_file = batch_folder / f'batch_{idx}.pkl'
            with open(pickle_file, 'wb') as f:
                pickle.dump(batch_dataset, f)

            # delete res_batch_idx_{idx}.json and batch_idx_{idx}.json
            # Delete the batch and result files after processing
            if batch_file.exists():
                batch_file.unlink()  # Deletes batch_idx_{idx}.json
            if res_file.exists():
                res_file.unlink()  # Deletes res_batch_idx_{idx}.json


            run_accelerate_subprocess(args, pickle_file, count, run_id)
            count=count+1 # starts in -1 and then 0 (first increment)
            checkpoint_path = Path(args.output_dir) / f"epoch_{count}"
            #get the output adapter as a lora model for run_vllm_student
            args.lora_model_name_or_path = checkpoint_path


        print(f"Epoch {epoch + 1} finished processing.")
        #get the last epoch{count} that exists and rename the folder as final_{epoch}
        final_epoch_folder = batch_folder / f"epoch_{count}"
        final_folder = batch_folder / f"final_epoch_{epoch}"
        if final_epoch_folder.exists():
            final_epoch_folder.rename(final_folder)
            print(f"Renamed final epoch folder to {final_folder}")
            args.lora_model_name_or_path = final_folder

if __name__ == "__main__":
    main()
