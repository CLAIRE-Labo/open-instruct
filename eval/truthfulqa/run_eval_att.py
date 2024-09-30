import argparse
import os
import json
import torch
import pandas as pd
import os
import sys
from bleurt import score
import tensorflow as tf
import gpustat
import random
import wandb
from pathlib import Path
from logging import getLogger

from accelerate.utils import set_seed

sys.path.append(Path(__file__).parents[2].absolute().as_posix())
from open_instruct.load_utils import load_args
from eval.utils import (
    load_hf_lm,
    load_hf_tokenizer,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    query_openai_model,
    generate_completions,
    score_completions,
    dynamic_import_function,
    prepare_env,
)
from eval.truthfulqa.utilities import (
    format_prompt,
    split_multi_answer,
    format_best,
    set_columns,
    save_questions,
)
from eval.truthfulqa.metrics import run_gpt_classifier_eval, run_hf_classifier_eval, MC_calcs, run_bleu, run_rouge, \
    run_BLEURT, run_MAUVE
from eval.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL

logger = getLogger(__name__)


def get_gpu_memory():
    stats = gpustat.new_query()
    return [gpu.memory_total for gpu in stats.gpus]


# Get total memory of the first GPU in MB
total_memory = get_gpu_memory()[0]
memory_limit_percentage = 0.1  # limit it to use only 10% percent of memory for BLEURT
memory_limit = int(total_memory * memory_limit_percentage)

# Configure the GPU memory allocation
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)]
            )
    except RuntimeError as e:
        print("Error setting memory limit:", e)

sys.path.append(Path(__file__).parents[1].absolute().as_posix())

import warnings
from eval.utils import (
    add_eval_args,
    run_att_model_for_eval
)
from eval.truthfulqa.utilities import (
    format_prompt,
    split_multi_answer,
    format_best,
    set_columns,
    save_questions,
)
from eval.truthfulqa.metrics import run_gpt_classifier_eval, run_hf_classifier_eval, MC_calcs, run_bleu, run_rouge, \
    run_BLEURT, run_MAUVE
from eval.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL

bleurt_scorer = None


def trim_answer(answer):
    # remove spaces at the beginning and end
    answer = answer.strip()
    # remove the "A:" prefix if it exists
    if answer.startswith('A:'):
        answer = answer[2:].strip()
    # remove everything after "Q:" if it exists
    if 'Q:' in answer:
        answer = answer.split('Q:')[0].strip()
    # reformat line-breaks for long-form answers
    answer = answer.replace('\n\n', ' ')
    return answer


def format_frame(results):
    results = results[[x for x in results.columns if (x != 'Context') and (results[x].dtype != 'O')]]
    new_cols = []
    for col in results.columns:
        split = col.split(' ')
        new_cols.append((split[0], ' '.join(split[1:])))
    results.columns = pd.MultiIndex.from_tuples(new_cols)
    return results


def main(args):
    prepare_env()

    global bleurt_scorer

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

    model_key_tuned = train_args.model_name_or_path + "-tuned"
    model_keys = [model_key_tuned]

    questions = pd.read_csv(os.path.join(args.data_dir, "TruthfulQA.csv"))  # .head(10)

    # if args.wandb_run_id is not None: #to store results if needed
    #     wandb.init(project="alignment_translation", id=args.wandb_run_id, resume="allow")

    if args.num_instances is not None:
        questions = questions.sample(args.num_instances, random_state=42)

    # if args.model_name_or_path:
    #     print("Loading model and tokenizer...")

    def prepare_question(prompt):
        return prompt if args.not_ask_brevity else prompt + " Please answer in one sentence."

    chats = [[{"role": "user", "content": prepare_question(q)}] for q in questions["Question"]]

    responses_log = []
    if responses_log_path.exists():
        print(f"Loading responses from cache at {responses_log_path}")
        with open(responses_log_path, "r") as f:
            responses_log = json.load(f)
    else:
        responses_log = run_att_model_for_eval(train_args, args, chats)
        with open(responses_log_path, "w") as f:
            json.dump(responses_log, f)

    # set the answers to response["output"]
    questions[model_key_tuned] = [response["output"] for response in responses_log]
    if args.att_evaluate_base:
        assert not args.not_att, "Cannot evaluate base model when not using ATT"
        model_key_base = train_args.model_name_or_path
        model_keys.append(model_key_base)
        questions[model_key_base] = [response["response_base"] for response in responses_log]

    # run metrics
    print("Running metrics!")
    print(questions.columns)
    for model_key in model_keys:
        for metric in args.metrics:
            if metric == 'mc':
                continue
            elif metric in ['truth', 'info']:
                if model_key not in questions.columns:
                    raise ValueError("Answers missing for {0}!".format(model_key))
                if metric == 'truth':
                    print("entered truth")
                    # questions=questions[0:3]
                    if args.gpt_truth_model_name:
                        questions = run_gpt_classifier_eval(model_key, 'truth', args.gpt_truth_model_name, questions,
                                                            info=False)
                    elif args.hf_truth_model_name_or_path:
                        truth_classifier, truth_tokenizer = load_hf_lm_and_tokenizer(
                            model_name_or_path=args.hf_truth_model_name_or_path,
                            tokenizer_name_or_path=args.hf_truth_model_name_or_path,
                            # device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        )
                        questions = run_hf_classifier_eval(model_key, 'truth', truth_classifier, truth_tokenizer,
                                                           questions, info=False)
                    print(questions.columns)
                    save_questions(questions, output_dir, "predictions.csv")
                if metric == 'info':
                    print("entered info")
                    if args.gpt_info_model_name:
                        questions = run_gpt_classifier_eval(model_key, 'info', args.gpt_info_model_name, questions,
                                                            info=True)
                    elif args.hf_info_model_name_or_path:
                        info_classifier, info_tokenizer = load_hf_lm_and_tokenizer(
                            model_name_or_path=args.hf_info_model_name_or_path,
                            tokenizer_name_or_path=args.hf_info_model_name_or_path,
                            # device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        )
                        print(questions.columns)
                        questions = run_hf_classifier_eval(model_key, 'info', info_classifier, info_tokenizer,
                                                           questions, info=True)
                    save_questions(questions, output_dir, "predictions.csv")
            elif metric in ['bleu', 'rouge', 'bleurt', 'mauve']:
                try:
                    if metric == 'bleu':
                        print("bleu started")
                        questions = run_bleu(model_key, questions)
                    elif metric == 'rouge':
                        print("rouge started")
                        questions = run_rouge(model_key, questions)
                    elif metric == 'bleurt':
                        print("bleurt started")
                        checkpoint = "BLEURT-20"
                        if bleurt_scorer == None:
                            bleurt_scorer = score.BleurtScorer(checkpoint)

                        questions = run_BLEURT(model_key, questions, bleurt_scorer)
                    elif metric == 'mauve':
                        assert False, "Dropped support for now"
                        print("mauve started")
                        # questions = run_MAUVE(model_key, questions)
                    save_questions(questions, output_dir, "predictions.csv")
                except Exception as err:
                    print(f"Error running {metric} metric: {err}")
            else:
                warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)
        print("columns: ", questions.columns)
        if "truth" in args.metrics and "info" in args.metrics:
            questions["{} truth-info acc".format(model_key)] = questions["{} truth acc".format(model_key)] * questions[
                "{} info acc".format(model_key)]

    # save all
    save_questions(questions, output_dir, "predictions.csv")
    # questions.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'bleu acc',
                                              'rouge1 acc',
                                              'rouge2 acc',
                                              'rougeL acc',
                                              'BLEURT acc',
                                              'truth acc',
                                              'info acc',
                                              'truth-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    results.to_csv(os.path.join(output_dir, 'summary.csv'))

    print(results)

    results = results.loc[model_key].to_dict()
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()

    add_eval_args(parser)

    parser.add_argument(
        "--att_evaluate_base",
        action="store_true",
        help="Set to evaluate the base model in addition to ATT"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="truthfulqa",
        help="The directory to save the evaluation results. Overall the results are under {checkpoint_dir}/eval/{output_dir}"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="The directory containing the truthfulqa data. Download from https://github.com/sylinrl/TruthfulQA/tree/main/data."
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        help="The number of instances to subsample from the data."
    )
    parser.add_argument('--not_ask_brevity', action='store_true',
                        help='If set, we will not ask the model to reply in one sentence in the evaluation. This makes the results more "canonical", but ruins the ability of the judge to know whether the answer is correct, since the judge was fine-tuned on one-sentence responses.')
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['truth', 'info'],
        choices=['truth', 'info', 'mc', 'bleu', 'rouge', 'bleurt'],
        help='Metrics to run'
    )
    parser.add_argument(
        '--gpt_truth_model_name',
        type=str,
        help='A trained GPT judge model name to be used for computing the metrics for `truth` if it is specified.' \
             'Either `gpt_truth_model_name` or `hf_truth_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        '--gpt_info_model_name',
        type=str,
        help='A trained GPT judge model name to be used for computing the metrics for `info` if it is specified.' \
             'Either `gpt_info_model_name` or `hf_info_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        '--hf_truth_model_name_or_path',
        type=str,
        help='A trained HuggingFace judge model name to be used for computing the metrics for `truth` if it is specified.' \
             'Either `gpt_truth_model_name` or `hf_truth_model_name_or_path` should be specified for computing the metric.',
        default='allenai/truthfulqa-truth-judge-llama2-7B'
    )
    parser.add_argument(
        '--hf_info_model_name_or_path',
        type=str,
        help='A trained HuggingFace judge model name to be used for computing the metrics for `info` if it is specified.' \
             'Either `gpt_info_model_name` or `hf_info_model_name_or_path` should be specified for computing the metric.',
        default='allenai/truthfulqa-info-judge-llama2-7B'
    )
    parser.add_argument(
        "--recursive_test",
        action="store_true",
        help="Set to enable token, 3 recursive ATT test will be applied"
    )
    parser.add_argument('--wandb_run_id', type=str, help="wandb_run_id to store")

    return parser.parse_args()


if __name__ == '__main__':
    prepare_env()

    args = parse_args()
    main(args)
