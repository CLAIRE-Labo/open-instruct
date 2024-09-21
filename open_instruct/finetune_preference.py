#!/usr/bin/env python
# coding=utf-8
import os
import sys
import time
import html
import json
from heapq import merge
from copy import deepcopy
from typing import Dict, List, Tuple
from pathlib import Path
import argparse
from argparse import Namespace
import logging
import math
import ast
import os
import random
from datetime import timedelta, datetime
import torch
from functools import partial
import subprocess

import numpy as np
from torch.cuda import memory_allocated
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs, gather_object, broadcast_object_list
import datasets
from datasets import DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
import wandb
import huggingface_hub

import transformers
from transformers import (
    GenerationConfig,
    DataCollatorForSeq2Seq,
    get_scheduler,
)

sys.path.append(Path(__file__).parents[1].absolute().as_posix())
from load_utils import (add_common_training_args, pretty_print_chatml, preprocess_data_to_chatml, \
                        load_tokenizer_model, save_args, load_args)
from att import apply_att_template

from constants import BAD_MISTRAL_CHAT_TEMPLATE, ATT_SYSTEM_PROMPT, ATT_TEMPLATE

# from eval.truthfulqa.run_eval import main as run_eval
# from eval.truthfulqa.run_eval import parse_args as parse_args_eval

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256" - Cuda out of memory


logger = get_logger(__name__)
import pandas as pd

if __name__ == "__main__":
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


def get_run_id(args: Namespace) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") if not os.getenv('RUN_TIMESTAMP') else os.getenv('RUN_TIMESTAMP')
    # try to get the git commit hash
    # try:
    #     git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
    # except subprocess.CalledProcessError:
    #     git_commit = "unknown"

    # hashes are not the same across processes, no idea why
    # arg_set = frozenset(vars(args).items())
    # args_hash = abs(hash(arg_set))
    # print(f"arg_set: {frozenset} args_hash: {args_hash}")
    return f"{timestamp}"
    # return f"{timestamp}_{args_hash}"
    # return f"{timestamp}_{git_commit}_{args_hash}"


def add_att_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        "--remove_multiturn_data",
        action="store_true",
        help="If set, only \"prompt-response\" data is used and multi-turn dialogue data is filtered out.",
    )
    parser.add_argument(
        "--use_new_token_template",
        action="store_true",
        help="If set, the new token template is used for ATT taning."
    )


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    logger.info(f"Saving model and tokenizer to {output_dir}", main_process_only=True)
    if accelerator.is_main_process:
        tokenizer_dir = Path(output_dir) / "tokenizer"
        tokenizer_dir.mkdir(exist_ok=True, parents=True)
        tokenizer.save_pretrained(tokenizer_dir)

    unwrapped_model = accelerator.unwrap_model(model)

    accelerator.wait_for_everyone()
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    model_dir = Path(output_dir) / "model"
    if accelerator.is_main_process:
        model_dir.mkdir(exist_ok=True, parents=True)
    accelerator.wait_for_everyone()
    if args.use_lora:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(model_dir, state_dict=state_dict)
    else:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            model, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=False
        )

    state_dir = Path(output_dir) / "state"
    if accelerator.is_main_process:
        state_dir.mkdir(exist_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(state_dir)

    logger.info(f"Checkpointing done!", main_process_only=True)

def setup_checkpointing(accelerator, args):
    """accelerator can be a PartialState() object."""
    run_id = get_run_id(args)
    checkpointing_dir = Path(args.output_dir) / run_id
    # If the output directory already exists, the process was interrupted and restarted by the cluster.
    # We should resume from the last checkpoint.
    if accelerator.is_main_process:
        checkpointing_dir_exists = checkpointing_dir.exists()
        if not checkpointing_dir_exists:
            checkpointing_dir.mkdir(parents=True)
        args_file = checkpointing_dir / "args.json"
        if args_file.exists():
            args_file = checkpointing_dir / f"args_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_args(args, args_file)
    else:
        checkpointing_dir_exists = None
    # Letting other processes know what the main process found. We don't want to create the directory multiple times.
    checkpointing_dir_actually_exists = broadcast_object_list([checkpointing_dir_exists], 0)
    if checkpointing_dir_actually_exists[0]:
        logger.info(f"Output directory {checkpointing_dir} already exists. Resuming training...")
        if args.resume_from_checkpoint is None:
            ckpt_dirs = [x for x in checkpointing_dir.iterdir() if
                         x.stem.startswith("step_") or x.stem.startswith("epoch_")]
            if len(ckpt_dirs) > 0:
                last_checkpoint = max(ckpt_dirs, key=os.path.getctime)
                args.resume_from_checkpoint = last_checkpoint
    logger.info(f"\n\n\nSaving checkpoints to {checkpointing_dir}\n\n\n")
    return checkpointing_dir

def build_lr_scheduler(accelerator, args, optimizer, train_dataloader):
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    return lr_scheduler, overrode_max_train_steps


def build_optimizer(args, model):
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
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="Finetune a transformers model using pairwise preference data.")
    add_common_training_args(parser)
    add_att_args(parser)
    args = parser.parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        assert args.report_to in ["wandb", "all"], "Currently only wandb is supported for tracking."
        wandb_api_key = os.getenv('WANDB_API_KEY')
        # This should be done in accelerator.init_trackers
        # wandb.init(project="alignment_translation", entity="claire-labo")
        # Configure wandb logging within Accelerator
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

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

    accelerator.wait_for_everyone()

    # load_tokenizer_model is a function that loads the model and tokenizer
    model, tokenizer, actual_eos_token, generation_config_nucleus, generation_config_greedy \
        = load_tokenizer_model(accelerator, args)
    tokenizer.eos_token = actual_eos_token

    checkpointing_dir = setup_checkpointing(accelerator, args)

    ######################################## Data Preprocessing ########################################
    dataset_train, dataset_test = preprocess_data_to_chatml(args)
    for i in range(10):
        logger.info(f"\n\nExample {i} chosen:\n{pretty_print_chatml(dataset_train[i]['chosen'])}\n\n"
                    f"Example {i} rejected:\n{pretty_print_chatml(dataset_train[i]['rejected'])}\n\n")
        apply_att_template(dataset_test[i], tokenizer, args.max_seq_length, debug_print=True, logger=logger)

    filtered_dataset = DatasetDict({
        'train': dataset_train,
        'test': dataset_test
    })

    encode_function = partial(
        apply_att_template,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        logger=logger,
    )

    with accelerator.main_process_first():
        lm_datasets = filtered_dataset.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in filtered_dataset["train"].column_names if
                            name not in ["input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")
        lm_datasets = lm_datasets.filter(lambda example: (example['labels'] != -100).any())

    if accelerator.is_main_process:
        lens = [x["input_ids"].shape[0] for x in lm_datasets["train"]]
        len_response = [torch.sum(x["labels"] != -100) for x in lm_datasets["train"]]
        logger.info(f"Avg length of example: {np.mean(lens):.1f} +/- {np.std(lens):.1f}")
        logger.info(f"Avg length of response: {np.mean(len_response):.1f} +/- {np.std(len_response):.1f}")
    accelerator.wait_for_everyone()

    train_dataset = lm_datasets["train"]
    test_dataset = lm_datasets["test"]

    # COMMENT OUT! This is for debugging epoch checkpointing
    # train_dataset = train_dataset.select(range(192))
    # test_dataset = test_dataset.select(range(192))

    # TODO check that nothing breaks if we don't add special tokens
    # We only use instruct models, so tokens should already be there

    # Log a few random samples from the training set

    num_train_ex = accelerator.num_processes
    num_test_ex = accelerator.num_processes
    indices_train_ex = random.sample(range(len(train_dataset)), num_train_ex)
    indices_test_ex = random.sample(range(len(test_dataset)), num_test_ex)

    train_examples = [train_dataset[i] for i in indices_train_ex]
    test_examples = [test_dataset[i] for i in indices_test_ex]
    # for index, ex in enumerate(train_examples):
    #     logger.info(f"Sample {index} of the training set: {ex}.")

    ######################################## Training Setup ########################################
    def put_big_to_front(dataset):
        logger.info("Putting the biggest examples to the front of the dataset to catch OOMs early.")
        lens = [x["input_ids"].shape[0] for x in dataset]
        order = list(sorted(range(len(lens)), key=lambda x: lens[x], reverse=True))
        if len(order) > 128:
            order = order[:128] + random.sample(order[128:], k=len(order[128:]))
        dataset = dataset.select(order)
        return dataset

    train_dataset = put_big_to_front(train_dataset)
    test_dataset = put_big_to_front(test_dataset)

    if args.dataset_subsample_size is not None:
        train_dataset = train_dataset.select(range(min(args.dataset_subsample_size, len(train_dataset))))
        test_dataset = test_dataset.select(range(min(args.dataset_subsample_size, len(test_dataset))))

    print("Size of training set:", len(train_dataset))
    print("Size of test set:", len(test_dataset))

    # DataLoaders creation:
    def get_dataloader(dataset, batch_size):
        return DataLoader(
            dataset,
            shuffle=True,
            collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=batch_size
        )

    train_dataloader = get_dataloader(train_dataset, args.per_device_train_batch_size)
    test_dataloader = get_dataloader(test_dataset, args.per_device_train_batch_size)

    # Optimizer
    optimizer = build_optimizer(args, model)

    # Scheduler and math around the number of training steps.
    lr_scheduler, overrode_max_train_steps = build_lr_scheduler(accelerator, args, optimizer, train_dataloader)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(project_name="alignment_translation", config=experiment_config,
                                  init_kwargs={"wandb": {"entity": "claire-labo"}})

    # Define custom step metric for evaluation
    # run_id = wandb.run.id

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    test_dataloader = accelerator.prepare(test_dataloader)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    ######################################## Checkpointing ########################################
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint is not None:
        checkpoint_path = Path(args.resume_from_checkpoint)
        path = os.path.basename(args.resume_from_checkpoint)

        state_path = Path(checkpoint_path) / "state"
        assert state_path.exists(), f"Checkpoint path {state_path} does not exist."
        logger.info(f"Resuming from checkpoint: {state_path}", main_process_only=True)
        accelerator.load_state(str(state_path))
        logger.info(f"Accelerator state loaded", main_process_only=True)

        # Extract `epoch_{i}` or `step_{i}`
        training_difference = checkpoint_path.stem
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                    int(training_difference.replace("step_", ""))
                    * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    # def get_cur_wandb_step():
    #     return accelerator.get_tracker("wandb").current_step

    # TODO if we end up using phi2, look into the bug in the chat template we have for it
    def log_examples_to_wandb(step: int):
        def gen_examples(example, example_idx: int, gen_config: GenerationConfig) -> Tuple[Tuple[float, str, str, str]]:
            start_time = time.time()
            is_prompt = example["labels"] == -100
            is_response = ~is_prompt
            prompt_ids = example["input_ids"][is_prompt]
            # if prompt_ids.shape[0] >= args.logging_examples_max_length:
            #     logger.warning(
            #         f"Skipping logging of example {example_idx} because it is too long to log. "
            #         f"Length: {prompt_ids.shape[0]}. Max length: {args.logging_examples_max_length}"
            #     )
            #     return

            expected_response_ids = example["input_ids"][is_response]
            prompt_text = tokenizer.decode(prompt_ids)
            expected_response_text = tokenizer.decode(expected_response_ids)

            # unwrapped = accelerator.unwrap_model(model)
            # outputs = unwrapped.generate(input_ids=prompt_ids.to(model.device)[None, :],
            #                              generation_config=generation_config)

            outputs = model.generate(input_ids=prompt_ids.to(model.device)[None, :],
                                     generation_config=gen_config, tokenizer=tokenizer,
                                     synced_gpus=True)

            response_text = tokenizer.decode(outputs[0])
            if response_text[:len(prompt_text)] != prompt_text:
                logger.warning(
                    f"Response does not start with the prompt for example {example_idx}. "
                    f"Prompt: \n\"\"\"\n{prompt_text}\n\"\"\"\n. Response: \n\"\"\"\n{response_text}\n\"\"\"\n"
                )
            else:
                response_text = response_text[len(prompt_text):]

            # We updated the eos token to avoid the bugs in model.generate, so we have to truncate the output manually
            if actual_eos_token is not None and actual_eos_token in response_text:
                response_text = response_text.split(actual_eos_token)[0] + actual_eos_token

            # accelerator.get_tracker("wandb", unwrap=True).log(
            # wandb.log(
            #     {f"{prefix}_example_{example_idx + 1}/response": wandb.Html(f"<p>{html.escape(response_text)}</p>")})
            return ((time.time() - start_time, prompt_text, expected_response_text, response_text),)

        model.eval()
        sum_time = 0.0

        def gather_and_log(prefix, examples):
            nonlocal sum_time

            logger.info(f"Logging examples for {prefix}", main_process_only=True)
            logged_greedy = gen_examples(examples[accelerator.process_index], accelerator.process_index,
                                         generation_config_greedy)
            logged_nucleus = gen_examples(examples[accelerator.process_index], accelerator.process_index,
                                          generation_config_nucleus)
            logger.info(f"Logged examples for {prefix}", main_process_only=True)

            all_logged_greedy = gather_object(logged_greedy)
            all_logged_nucleus = gather_object(logged_nucleus)
            for i, (lg, ln) in enumerate(zip(all_logged_greedy, all_logged_nucleus)):
                tm_g, prompt, expected, actual_g = lg
                tm_n, _, _, actual_n = ln
                sum_time += tm_g + tm_n
                accelerator.log(
                    {f"{prefix}_example_{i + 1}/prompt": wandb.Html(f"<p>{html.escape(prompt)}</p>"),
                     f"{prefix}_example_{i + 1}/expected_response": wandb.Html(f"<p>{html.escape(expected)}</p>"),
                     f"{prefix}_example_{i + 1}/response_greedy": wandb.Html(f"<p>{html.escape(actual_g)}</p>"),
                     f"{prefix}_example_{i + 1}/response_nucleus": wandb.Html(f"<p>{html.escape(actual_n)}</p>"), }
                )

        gather_and_log("train", train_examples)
        gather_and_log("test", test_examples)

        model.train()
        accelerator.wait_for_everyone()
        print(f"Finished evals! pidx={accelerator.process_index}")

        accelerator.log(
            {"log_examples_time": sum_time,
             "avg_example_time": sum_time / 2 / (len(train_examples) + len(test_examples))})

    ######################################## Training Loop ########################################
    print(f"{accelerator.sync_gradients=}", flush=True)
    accelerator.wait_for_everyone()

    def compute_loss(batch, outputs):
        if args.reduce_loss == 'mean':
            loss = outputs.loss
        else:
            # reduce loss is sum
            # this ensures that we weight all tokens in the dataset equally,
            # rather than weighting each overall example equally when
            # using high amounts of gradient accumulation.
            # this can result in > 5 point improvements in AlpacaEval
            # see https://github.com/huggingface/transformers/issues/24725 for
            # more discussion and details.
            logits = outputs.logits
            labels = batch["labels"]
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
            shift_logits = shift_logits.view(-1, embedding_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        return loss

    if accelerator.sync_gradients and not args.logging_examples_ignore_first:
        log_examples_to_wandb(completed_steps)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        epoch_data_count = 0  # To keep track of the number of data points in each epoch
        if (
                args.resume_from_checkpoint
                and epoch == starting_epoch
                and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            # Print the size of each component in the batch
            batch_size = len(batch['input_ids'])  # Assuming 'input_ids' is the key for input data
            epoch_data_count += batch_size
            with accelerator.accumulate(model):
                # print(f"Memory allocated before forward pass: {memory_allocated() / 1e9} GB")
                outputs = model(**batch, use_cache=False)
                # print(f"Memory allocated after forward pass: {memory_allocated() / 1e9} GB")
                loss = compute_loss(batch, outputs)
                # We keep track of the loss at each logged step
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

                # for cuda out of memory
                if step % 2000 == 0:
                    torch.cuda.empty_cache()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(
                        total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    # logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    # Print number of examples processed so far in this epoch
                    # print(f"Step {completed_steps}: Processed {epoch_data_count} examples so far in Epoch {epoch + 1}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                        )

                    total_loss = 0

                if args.logging_examples_steps and completed_steps % args.logging_examples_steps == 0:
                    log_examples_to_wandb(completed_steps)

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    output_dir = checkpointing_dir / f"step_{completed_steps}"
                    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break

        print(f"Completed Epoch {epoch + 1}: Total processed examples = {epoch_data_count}")

        model.eval()
        eval_pbar = tqdm(test_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)
        total_loss = 0.0
        total_num_labels = 0
        for step, batch in enumerate(eval_pbar):
            if args.max_train_steps is not None and step >= args.max_train_steps:
                break
            with torch.no_grad():
                outputs = model(**batch, use_cache=False)
                num_labels = batch["labels"].ne(-100).sum()
                loss = compute_loss(batch, outputs)
            batch_num_labels = accelerator.gather(num_labels.repeat(accelerator.num_processes)).sum().item()
            batch_loss = accelerator.gather(loss.repeat(accelerator.num_processes)).sum().item()
            total_loss += batch_loss
            total_num_labels += batch_num_labels
            eval_pbar.set_postfix({"loss": total_loss / total_num_labels})

        logs = {"eval_loss": total_loss / total_num_labels, "eval_ppl": math.exp(total_loss / total_num_labels)}
        if args.with_tracking:
            accelerator.log(logs)
        model.train()

        if args.checkpointing_steps == "epoch":
            output_dir = checkpointing_dir / f"epoch_{epoch}"
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)
            with open(output_dir / "logs.json", "w") as f:
                json.dump(logs, f)

    output_dir = checkpointing_dir / "final"
    save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    accelerator.wait_for_everyone()
    if args.with_tracking:
        accelerator.end_training()


if __name__ == "__main__":
    main()
