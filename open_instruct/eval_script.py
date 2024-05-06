from argparse import Namespace
import sys
sys.path.append('/claire-rcp-scratch/home/tandogan/alignment-as-translation/open-instruct')

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from eval.truthfulqa.run_eval import main as run_eval
from eval.truthfulqa.run_eval import parse_args as parse_args_eval
from open_instruct.merge_lora import main as merge_lora


#evaluate the current epoch model
eval_args = Namespace(
    model_name_or_path="outputs/olmo1bsft_lora/epoch_0",
    tokenizer_name_or_path="outputs/olmo1bsft_lora/epoch_0",
    data_dir="data/",
    save_dir="outputs/olmo1bsft_lora/eval_results",  # Save evaluation results
    metrics=['bleu', 'rouge', 'bleurt'],  # Specify your metrics
    num_instances=None,
    preset='qa',
    eval_batch_size=1,
    use_chat_format=True,
    openai_engine=None,
    chat_formatting_function="eval.templates.create_prompt_with_finetuned_olmo1b_chat_format",
    use_slow_tokenizer=None,
    load_in_8bit=False,
    gptq=False,
    filename_answers=f"epoch_0"
)
run_eval(eval_args)