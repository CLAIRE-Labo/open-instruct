import sys
import argparse
from pathlib import Path
import json

import pandas as pd
import datasets
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from alpaca_eval import evaluate as alpaca_farm_evaluate

sys.path.append(str(Path(__file__).parents[2].absolute().as_posix()))
from eval.utils import query_openai_chat_model, query_openai_model, run_att_model_for_eval, add_eval_args
from open_instruct.load_utils import load_args, load_tokenizer_model

logger = get_logger(__name__)


def evaluate(args):
    set_seed(239)

    train_args = load_args(args.train_run_args)
    if args.cache_dir is not None:
        rel_path = Path(*args.tuned_checkpoint.absolute().parts[1:])
        output_dir = args.cache_dir / rel_path / "eval" / args.output_dir
    else:
        output_dir = args.tuned_checkpoint / "eval" / args.output_dir
    if output_dir.exists():
        logger.warning(f"Output directory {output_dir} already exists. Will see if the outputs are cached.")
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_log_path = output_dir / "responses.json"

    logger.info("loading data...")
    alpaca_eval_data = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", trust_remote_code=True)["eval"]
    prompts = []
    for example in alpaca_eval_data:
        prompt = example["instruction"]
        prompts.append(prompt)

    if args.num_instances is not None:
        prompts = prompts[:args.num_instances]

    chats = [[{"role": "user", "content": prompt}] for prompt in prompts]

    responses_log = []
    if responses_log_path.exists():
        with open(responses_log_path, "r") as f:
            responses_log = json.load(f)
        logger.info(f"Loaded responses from {responses_log_path}")
    else:
        responses_log = run_att_model_for_eval(train_args, args, chats)
        with open(responses_log_path, "w") as f:
            json.dump(responses_log, f)

    prompts = [resp["instruction"] for resp in responses_log]
    alpaca_eval_data = alpaca_eval_data.filter(lambda x: x["instruction"] in prompts)
    # responses = [resp[""] for resp in responses_log]

    df_leaderboard, annotations = alpaca_farm_evaluate(
        model_outputs=responses_log,
        reference_outputs=alpaca_eval_data,
        annotators_config="alpaca_eval_gpt4",
        output_path=output_dir,
        is_return_instead_of_print=True,
        caching_path=output_dir / "alpaca_eval_annotator_cache_att.json",
        precomputed_leaderboard=None,
        is_cache_leaderboard=False,
        metric_kwargs={"save_weights_dir": output_dir / "weights/finetuned"},
    )

    if (not args.not_att) and args.att_evaluate_base:
        # We also evaluate the base model
        base_data = []
        for resp in responses_log:
            base_data.append({
                "instruction": resp["instruction"],
                "output": resp["response_base"],
                "generator": train_args.model_name_or_path
            })
        base_leaderboard, base_annotations = alpaca_farm_evaluate(
            model_outputs=base_data,
            reference_outputs=alpaca_eval_data,
            annotators_config="alpaca_eval_gpt4",
            output_path=output_dir,
            is_return_instead_of_print=True,
            caching_path=output_dir / "alpaca_eval_annotator_cache_base.json",
            precomputed_leaderboard=None,
            is_cache_leaderboard=False,
            metric_kwargs={"save_weights_dir": output_dir / "weights/base"},
        )
        df_leaderboard = pd.concat([df_leaderboard, base_leaderboard])

    print(df_leaderboard.to_string(float_format="%.2f"))

    # save to json
    with open(output_dir / f"metrics.json", "w") as fout:
        json.dump(df_leaderboard.to_dict(), fout)
    return df_leaderboard.to_dict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_eval_args(parser)

    parser.add_argument(
        "--num_instances",
        type=int,
        help="The number of instances to subsample from the data."
    )
    parser.add_argument('--att_evaluate_base', action='store_true',
                        help='If set, evaluation is also run on the base model. Like this we dont need to run the base eval separately.')
    parser.add_argument('--output_dir', type=Path,
                        default="alpaca_farm",
                        help='Output directory to save results. Default is into {tuned_checkpoint}/eval/alpaca_farm ')
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
    )
    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    evaluate(args)
