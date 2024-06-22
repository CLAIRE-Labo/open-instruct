#this will be performed over already existing predictions in TruthfulQA.

import argparse
import pandas as pd
import os
import sys

# to find mauve_lib
sys.path.append('/claire-rcp-scratch/home/tandogan/alignment-as-translation')

from mauve_lib.src.mauve.compute_mauve import compute_mauve
import json
from transformers import AutoTokenizer

ANSWER_COL = 'Correct Answers'
def process_files(predictions_path, model_name):
    # Load CSV files
    predictions_df = pd.read_csv(predictions_path)
    answers=[]
    model_predictions=[]
    # Extract columns and store them in separate lists
    try:
        answers = predictions_df[ANSWER_COL].tolist()
        model_predictions = predictions_df[model_name].tolist()
        print(f"model predictions taken from,{model_name}")
    except KeyError as e:
        assert ValueError(f"Column not found in the predictions CSV: {e}")
    """data = {
        "answers": answers,
        "model_predictions": model_predictions
    }

    # Write to a JSON file
    try:
        output_json_path= "sample_input.json"
        with open(output_json_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data written to {output_json_path}")
    except IOError as e:
        raise IOError(f"Error writing to JSON file: {e}")"""
    return answers, model_predictions

def calculate_mauve(answers, model_predictions, args):
    # p_text -> human
    # q_text -> machine

    out = compute_mauve(p_text=answers, q_text=model_predictions, device_id=0,  verbose=False, featurize_model_name=args.model_name)
    print("mauve result: ", out.mauve)
    return out.mauve

def update_metrics(metrics_path, mauve_acc):
    # Load the metrics JSON file
    with open(metrics_path, 'r') as file:
        metrics = json.load(file)

    # Add the new MAUVE accuracy to the dictionary
    metrics["mauve acc"] = mauve_acc

    print("Updated Metrics:")
    print(metrics)

    # Write the updated dictionary back to the JSON file
    with open(metrics_path, 'w') as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prediction and metric CSV files for a specified model.")

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to the CSV file containing the predictions."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        required=True,
        help="Path to the json file containing the metrics results."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model used for generating predictions."
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )

    args = parser.parse_args()
    answers, model_predictions = process_files(args.predictions, args.model_name)

    mauve_acc = calculate_mauve(answers, model_predictions, args)
    update_metrics(args.metrics, mauve_acc)

