import os
import sys
from typing import Dict, List
import argparse
from pathlib import Path
import json
from copy import deepcopy
from logging import getLogger

from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
import accelerate
from accelerate.utils import set_seed
# from accelerate.logging import get_logger
from alpaca_eval import evaluate as alpaca_farm_evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(str(Path(__file__).parents[2].absolute().as_posix()))
from eval.utils import query_openai_chat_model, query_openai_model, run_att_model_for_eval, add_eval_args, prepare_env
from open_instruct.load_utils import load_args, load_tokenizer_model

# logger = get_logger(__name__)
logger = getLogger(__name__)


# Taken from https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1
class ArmoRMPipeline:
    def __init__(self, model_id, device_map="auto", torch_dtype=torch.bfloat16, truncation=True,
                 trust_remote_code=False, max_length=2048):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
        )
        self.truncation = truncation
        self.device = self.model.device
        self.max_length = max_length

    def __call__(self, messages: List[Dict[str, str]]) -> Dict[str, float]:
        """
        messages: OpenAI chat messages to be scored
        Note: no batching since due to length differences, the model will have to pad to the max length which is not efficient
        Returns: a dictionary with the score between 0 and 1
        """
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            padding=True,
            truncation=self.truncation,
            max_length=self.max_length,
        ).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids)
            score = output.score.float().item()
        return {"score": score}


def evaluate(args):
    set_seed(239)

    train_args = load_args(args.train_run_args)
    if args.cache_dir is not None:
        rel_path = Path(*args.tuned_checkpoint.absolute().parts[1:])
        output_dir = args.cache_dir / rel_path / "eval" / args.output_dir
    else:
        output_dir = args.tuned_checkpoint / "eval" / args.output_dir
    if output_dir.exists():
        print(f"Output directory {output_dir} already exists. Will see if the outputs are cached.")
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_log_path = output_dir / "responses.json"
    responses_graded_path = output_dir / "responses_graded.json"
    if not responses_graded_path.exists():
        # Load the data
        data = load_dataset(args.prompt_dataset, split="test")
        prompt_chats = [x[:-1] for x in data["chosen"]]
        if args.num_instances is not None:
            prompt_chats = prompt_chats[:args.num_instances]

        if args.responses_base_log is not None:
            with open(args.responses_base_log, "r") as f:
                responses_base_log = json.load(f)
            responses_base = [r["output"] for r in responses_base_log]
        else:
            responses_base = None

        if responses_log_path.exists():
            with open(responses_log_path, "r") as f:
                responses_log = json.load(f)
            logger.info(f"Loaded responses from {responses_log_path}")
        else:
            responses_log = run_att_model_for_eval(train_args, args, prompt_chats, responses_base=responses_base)
            with open(responses_log_path, "w") as f:
                json.dump(responses_log, f)

        # Evaluate the responses
        chats = [deepcopy(prompt) + [{"role": "assistant", "content": log["output"]}] \
                 for prompt, log in zip(prompt_chats, responses_log)]

        assert args.reward_model == "RLHFlow/ArmoRM-Llama3-8B-v0.1", "Only the ArmoRM model is supported for now."
        reward_model = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

        scores = []
        score_key = f"{args.evaluated_name}_score"
        for i, chat in enumerate(tqdm(chats, "Scoring the responses")):
            chat_score = reward_model(chat)
            responses_log[i][score_key] = chat_score['score']
        score_vals = [log[score_key] for log in responses_log]

        if args.att_evaluate_base:
            base_chats = [deepcopy(prompt) + [{"role": "assistant", "content": log["response_base"]}] \
                          for prompt, log in zip(prompt_chats, responses_log)]
            base_scores = []
            base_score_key = f"{args.evaluated_name}_base_score"
            for i, chat in enumerate(tqdm(base_chats, "Scoring the base responses")):
                chat_score = reward_model(chat)
                responses_log[i][base_score_key] = chat_score['score']
            score_vals_base = [log[base_score_key] for log in responses_log]

        with open(responses_graded_path, "w") as f:
            json.dump(responses_log, f)
        print(f"Results saved to {responses_graded_path}")
    else:
        print(f"Results already exist at {responses_graded_path}.")

    responses_log = json.load(open(responses_graded_path, "r"))
    score_keys = [k for k in responses_log[0].keys() if k.endswith("_score") and not k.endswith("_base_score")]
    assert len(score_keys) == 1, "Only one score key is supported for now."
    score_key = score_keys[0]
    score_vals = np.array([log[score_key] for log in responses_log])

    score_base_keys = [k for k in responses_log[0].keys() if k.endswith("_base_score")]
    if score_base_keys:
        score_base_key = score_base_keys[0]
        score_vals_base = np.array([log[score_base_key] for log in responses_log])
        # Some comparison stats
        print(f"Mean score: {score_vals.mean():.4f}+-{score_vals.std():.4f}"
              f" vs {score_vals_base.mean():.4f}+-{score_vals_base.std():.4f}")
        print(f"Mean diff: {score_vals.mean() - score_vals_base.mean():.4f}")
        print(f"Median score: {np.median(score_vals):.4f} vs {np.median(score_vals_base):.4f}")
        print(f"Max score: {score_vals.max():.4f} vs {score_vals_base.max():.4f}")
        print(f"Min score: {score_vals.min():.4f} vs {score_vals_base.min():.4f}")
    else:
        print(f"Mean score: {score_vals.mean():.4f}+-{score_vals.std():.4f}")
        print(f"Median score: {np.median(score_vals):.4f}")
        print(f"Max score: {score_vals.max():.4f}")
        print(f"Min score: {score_vals.min():.4f}")


if __name__ == '__main__':
    prepare_env()

    parser = argparse.ArgumentParser()

    add_eval_args(parser)

    parser.add_argument(
        "--num_instances",
        type=int,
        help="The number of instances to subsample from the data."
    )
    parser.add_argument(
        "--prompt_dataset",
        type=str,
        default="Magpie-Align/Magpie-Air-DPO-100K-v0.1",
        help="The dataset to use for prompts. We will use the 'test' split of this dataset."
    )
    parser.add_argument('--att_evaluate_base', action='store_true',
                        help='If set, evaluation is also run on the base model. Like this we dont need to run the base eval separately.')
    parser.add_argument('--output_dir', type=Path,
                        default="reward",
                        help='Output directory to save results. Default is into {tuned_checkpoint}/eval/reward ')
    parser.add_argument(
        "--reward_model",
        type=str,
        default="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        help="The model to use for reward calculation."
    )
    args = parser.parse_args()

    evaluate(args)
