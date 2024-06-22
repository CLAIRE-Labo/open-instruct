import os
import subprocess

def run_evaluation_epoch(base_path):
    # List all directories within the base_path that start with "epoch"
    epochs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith('epoch')]
    #epochs=[base_path]
    epochs.remove('epoch_0')
    epochs.remove('epoch_1')
    epochs.remove('epoch_2')
    epochs.remove('epoch_3')
    epochs.remove('epoch_4')
    epochs.remove('epoch_5')
    # Loop through each found epoch directory
    for epoch in epochs:
        epoch_path = os.path.join(base_path, epoch)
        # Construct command to run run_eval.py with necessary arguments
        command = [
            "python", "open-instruct/eval/truthfulqa/run_eval.py",
            "--model_name_or_path", epoch_path,
            "--tokenizer_name_or_path", epoch_path,
            "--base_llm_model", "microsoft/Phi-3-mini-4k-instruct",
            #"--base_llm_model","allenai/OLMo-7B-Instruct",
            "--data_dir", "data/",
            "--save_dir", os.path.join(epoch_path, "eval_results"),
            "--eval_batch_size", "1",
            "--use_chat_format",
            "--preference",
            "--chat_formatting_function", "eval.templates.create_prompt_with_phi3_chat_format",
            #"--chat_formatting_function", "eval.templates.create_prompt_with_finetuned_olmo1b_chat_format",
            "--filename_answers", "answers_tmp_phi3",
            "--document", "/claire-rcp-scratch/home/tandogan/alignment-as-translation/results/phi3_Sft_res.csv",
        ]

        # Print the command fosr verification
        print("Executing command:", " ".join(command))

        # Execute the command using subprocess
        subprocess.run(command)
        print(epoch +" is completed")

def run_evaluation(base_path, save_path):
    command = [
        "python", "open-instruct/eval/truthfulqa/run_eval.py",
        "--model_name_or_path", base_path,
        "--tokenizer_name_or_path", base_path,
        "--base_llm_model", "microsoft/Phi-3-mini-4k-instruct",
        # "--base_llm_model","allenai/OLMo-7B-Instruct",
        "--data_dir", "data/",
        "--save_dir", os.path.join(save_path, "eval_results"),
        "--eval_batch_size", "1",
        "--use_chat_format",
        "--preference",
        "--chat_formatting_function", "eval.templates.create_prompt_with_phi3_chat_format",
        # "--chat_formatting_function", "eval.templates.create_prompt_with_finetuned_olmo1b_chat_format",
        "--filename_answers", "answers_tmp_phi3",
        "--document", "/claire-rcp-scratch/home/tandogan/alignment-as-translation/results/phi3_Sft_res.csv",
    ]

    # Print the command fosr verification
    print("Executing command:", " ".join(command))

    # Execute the command using subprocess
    subprocess.run(command)

base_path= "outputs/phi_att_3epochs_merged"

#run_evaluation_epoch(base_path)
# Define the base path for the model
#base_path="outputs/olmo7binstruct_lora_Dpo_merged"
#base_path= "outputs/olmo1b_pref_3epochs_merged"
save_path="outputs/finalRes_completion/phi3_att"
#
# Run the function
run_evaluation(base_path, save_path)