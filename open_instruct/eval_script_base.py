from argparse import Namespace
import sys
import os
sys.path.append('/claire-rcp-scratch/home/tandogan/alignment-as-translation/open-instruct')

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from eval.truthfulqa.run_eval import main as run_eval
from eval.truthfulqa.run_eval import parse_args as parse_args_eval
from open_instruct.merge_lora import main as merge_lora


def evaluate_models(base_path):
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    if not directories:
        evaluate_single_model(base_path)
    else:
        for epoch_dir in directories:
            full_path = os.path.join(base_path, epoch_dir)
            evaluate_single_model(full_path)


def evaluate_single_model(path):
    print(f"Evaluating model in {path}")
    filename = os.path.basename(path)

    eval_args = Namespace(
        model_name_or_path=path,
        tokenizer_name_or_path=path,
        base_llm_model= "outputs/olmo1b_lora_merged_",
        preference=False,
        data_dir="data/",
        save_dir=os.path.join(path, "eval_results"),  # Save results in a subdirectory
        metrics=['bleu', 'rouge', 'bleurt'],
        num_instances=None,
        preset='qa',
        eval_batch_size=1,
        use_chat_format=True,
        openai_engine=None,
        chat_formatting_function="eval.templates.create_prompt_with_finetuned_olmo1b_chat_format",
        use_slow_tokenizer=None,
        load_in_8bit=False,
        gptq=False,
        filename_answers=filename
    )
    print(f"{filename} started to be evaluated")
    # Run the evaluation
    run_eval(eval_args)
    print(f"{filename} is evaluated")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate models from directories under a base path.")
    parser.add_argument("--base_path", type=str, required=True,
                        help="The base path containing model directories to evaluate.")

    args = parser.parse_args()
    evaluate_models(args.base_path)
