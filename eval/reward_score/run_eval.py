import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from tqdm import tqdm  # Make sure to import tqdm at the top of your script
import torch.nn.functional as F
import torch
def calculate_percentage_of_high_values(tensor):
    # Ensure the tensor is on the CPU and detach it from the computation graph
    tensor = tensor.detach()

    # Condition to check values greater than 0.5
    condition = tensor > 0.5

    # Count the number of values that are greater than 0.5
    count_high = torch.sum(condition)

    # Calculate the total number of values
    total_count = tensor.numel()

    print("higher than 0.5",count_high)
    print("all", total_count)
    # Calculate the percentage of values that are greater than 0.5
    percentage_high = (count_high / total_count).item() * 100  # Convert to percentage

    return percentage_high

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate the overall score for responses using a specified model.")
    parser.add_argument(
        "--reward_model",
        type=str,
        required=True,
        help="The HuggingFace model to be evaluated as base llm."
    )
    parser.add_argument(
        "--input_folder_att",
        type=str,
        required=True,
        help="Folder path that contains 'response_dict.jsonl' for ATT."
    )
    parser.add_argument(
        "--input_folder_dpo",
        type=str,
        required=True,
        help="Folder path that contains 'response_dict.jsonl' for DPO."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Folder path to save the 'score.json'."
    )
    return parser.parse_args()


def calculate_score(rank_model, tokenizer, question, answer):
    inputs = tokenizer(question, answer, return_tensors='pt',padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        score = rank_model(**inputs).logits
    print("logit is calculated")
    return score

def check_prompts_equality(prompts_dpo, prompts_att):
    # Check if both lists have the same length
    if len(prompts_dpo) != len(prompts_att):
        return False, "Lists are of different lengths."

    # Check each element in the lists
    for dpo, att in zip(prompts_dpo, prompts_att):
        if dpo != att:
            return False, "Elements differ."

    return True, "All elements are the same."

def process_file(args):
    rank_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model)
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model)

    input_path_att = os.path.join(args.input_folder_att, 'response_dict.jsonl')
    input_path_dpo = os.path.join(args.input_folder_dpo, 'response_dict.jsonl')

    output_path = os.path.join(args.output_folder, 'score.json')
    total_lines_att = sum(1 for _ in open(input_path_att, 'r', encoding='utf-8'))
    total_lines_dpo = sum(1 for _ in open(input_path_dpo, 'r', encoding='utf-8'))
    if total_lines_dpo!=total_lines_att:
        assert ValueError("they must be equal")


    prompts_att, prompts_dpo=[], []
    responses_att, responses_dpo=[],[]
    # Read and collect all prompts and responses
    with open(input_path_att, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompts_att.append(data['prompt'])
            responses_att.append(data['response'])

    with open(input_path_dpo, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompts_dpo.append(data['prompt'])
            responses_dpo.append(data['response'])
    result, message = check_prompts_equality(prompts_dpo, prompts_att)

    if result==False:
        assert ValueError("problem in the prompts")

    logits_att=calculate_score(rank_model, tokenizer, prompts_att, responses_att)
    logits_dpo=calculate_score(rank_model, tokenizer, prompts_dpo, responses_dpo)

    combined_logits = torch.cat((logits_att, logits_dpo), dim=1)

    # Apply softmax across the new dimension to normalize scores into probabilities
    probabilities = F.softmax(combined_logits, dim=1)
    wins = probabilities > 0.5

    # Count wins for each column
    win_counts = torch.sum(wins, dim=0)

    normalized_wins = win_counts / len(prompts_att)
    normalized_wins.tolist()
    win_counts.tolist()

    win_rates={
        "win_counts_att":win_counts[0].item(),
        "win_counts_dpo":win_counts[1].item(),
        "win_att": normalized_wins[0].item(),
        "win_dpo": normalized_wins[1].item()
    }
    print(win_rates)

    # Write the win rates to a JSON file
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(win_rates, outfile, indent=4)


def main():
    args = parse_args()
    process_file(args)


if __name__ == "__main__":
    main()

#results/ifeval/olmo1b_dpo
#model: OpenAssistant/reward-model-deberta-v3-large-v2