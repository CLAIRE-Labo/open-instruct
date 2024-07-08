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
import openai
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
)
from pathlib import Path


def get_gpu_memory():
    stats = gpustat.new_query()
    return [gpu.memory_total for gpu in stats.gpus]


# Get total memory of the first GPU in MB
total_memory = get_gpu_memory()[0]
memory_limit_percentage = 0.1 # limit it to use only 10% percent of memory for BLEURT
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
    load_hf_lm,
    load_hf_tokenizer,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    query_openai_model,
    generate_completions,
    score_completions,
    dynamic_import_function,
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


def run_chatgpt(questions, engine, tag, preset='qa', batch_size=1, cache_path=None, verbose=False):
    """Stores answers from ChatGPT / GPT4 models (requires an API key)"""

    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    instances = [
        {"prompt": format_prompt(questions.loc[idx], preset, format='general'), "id": idx} for idx in questions.index
    ]

    responses = query_openai_chat_model(
        engine=engine,
        output_path=cache_path,
        instances=instances,
        batch_size=batch_size,
        temperature=0.0
    )
    assert len(responses) == len(instances)
    for idx, response in zip(tqdm(questions.index, desc="Processing ChatGPT Responses"), responses):
        questions.loc[idx, tag] = trim_answer(response["output"])
    return questions


def run_gpt3(questions, engine, tag, preset='qa', batch_size=1, cache_path=None, verbose=False):
    """Stores answers from GPT-3 models (requires an API key)"""

    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    instances = [
        {"prompt": format_prompt(questions.loc[idx], preset, format='general'), "id": idx} for idx in questions.index
    ]

    responses = query_openai_model(
        engine=engine,
        instances=instances,
        output_path=cache_path,
        batch_size=batch_size,
        temperature=0.0,
        stop=None if preset == 'long' else '\n\n',
        max_tokens=50
    )
    assert len(responses) == len(instances)

    for idx, response in zip(tqdm(questions.index, desc="Processing GPT-3 Responses"), responses):
        questions.loc[idx, tag] = trim_answer(response["output"])

    return questions


def run_gpt3_mc(questions, engine, tag, preset='qa', batch_size=1, cache_path=None, verbose=False):
    """Runs multiple-choice metrics for GPT-3 models (requires an API key)"""

    set_columns(tag, questions)
    examples = []
    for idx in questions.index:
        # check that candidate answer exists
        if pd.isnull(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue
        if not len(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue

        # reference answers
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

        # prompt for all answers
        prompt = format_prompt(questions.loc[idx], preset, format='general') + "\nA:"

        # candidate completions
        examples.append({"prompt": prompt, "completions": ref_true + ref_false})

    instances, instance_id = [], 0
    for example in examples:
        for completion in example["completions"]:
            instances.append({"prompt": example["prompt"] + " " + completion, "id": instance_id})
            instance_id += 1
    responses = query_openai_model(
        engine=engine,
        instances=instances,
        output_path=cache_path,
        batch_size=batch_size,
        temperature=0.0,
        stop=["\n\n"],
        max_tokens=0,
        echo=True,
        logprobs=1
    )
    assert len(responses) == len(instances)
    responses = {response["id"]: response for response in responses}

    all_scores, instance_id = {}, 0
    for example in examples:
        all_scores[example["prompt"]] = {}
        for completion in example["completions"]:
            response = responses[instance_id]
            logprobs = response["response_metadata"]['choices'][0]['logprobs']
            # iterate through response to find the indexes of the start / end tokens for the ref answer
            idx_start = 0
            while idx_start < len(logprobs['text_offset']) - 1:
                if (logprobs['text_offset'][idx_start] >= len(example["prompt"])):
                    break
                idx_start += 1
            idx_end = idx_start
            while idx_end < len(logprobs['text_offset']) - 1:
                if (logprobs['text_offset'][idx_end] >= len(example["prompt"] + " " + completion)):
                    break
                idx_end += 1
            logprob_vals = logprobs['token_logprobs'][idx_start: idx_end]
            text_vals = logprobs['tokens'][idx_start + 3:idx_end]
            if True:
                print("LOGPROBS AND ANSWER TOKENS")
                print(logprob_vals)
                print(text_vals)
            all_scores[example["prompt"]][completion] = sum(logprob_vals)
            instance_id += 1

    for idx, example in zip(questions.index, examples):
        ref_best = format_best(questions.loc[idx, BEST_COL])
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])
        completion_scores = all_scores[example["prompt"]]
        scores_true = [completion_scores[ref] for ref in ref_true]
        scores_false = [completion_scores[ref] for ref in ref_false]

        MC_calcs(tag, questions, idx, scores_true, scores_false, ref_true, ref_best)
    return questions


def create_prompt(original_questions, tag, withToken=False, bos="<|endoftext|>", eos="<|endoftext|>"):
    # system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given output."
    formatted_questions = []  # Use a different variable name to store results

    if withToken:
        print("withToken")
        for index, row in original_questions.iterrows():
            message_text = ""
            if pd.notna(row['Question']):
                message_text += row['Question'].strip() + "\n"
            if pd.notna(row[tag]):
                message_text += "\n <|sample_answer|> " + row[tag].strip() + "\n <|sample_answer|> \n "
                formatted_questions.append(message_text)  # append EOS here
            else:
                print("Something wrong in the structure")
    else:
        for index, row in original_questions.iterrows():
            message_text = ""
            if pd.notna(row['Question']):
                message_text += row['Question'].strip() + "\n"
            if pd.notna(row[tag]):
                message_text += "\n Current rejected answer: " + row[tag].strip() + "\n Corrected output: \n "
                formatted_questions.append(message_text)  # append EOS here
            else:
                message_text += "\n Current rejected answer: " + " " + "\n Corrected output: \n "
                formatted_questions.append(message_text)
                print("Something wrong in the structure")

    return formatted_questions


def run_hf_model_create_input(withToken, questions, document, tag, preset="qa", batch_size=1, max_new_tokens=50,
                              chat_formatting_function=None, storage="tmp", idx_list=[]):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""
    try:
        # Check if the CSV already exists
        if document is not None and os.path.exists(document):
            questions = pd.read_csv(document)
            questions["modified_input"] = create_prompt(questions, tag, withToken=withToken, bos="", eos="")
            print(questions["modified_input"].to_list()[0])
            print("first stage input creation done")
        else:
            print("get initial responses from base llm")
            model = load_hf_lm(
                model_name_or_path=args.base_llm_model,
                load_in_8bit=args.load_in_8bit,
                device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                gptq_model=args.gptq
            )
            tokenizer = load_hf_tokenizer(
                model_name_or_path=args.base_llm_model,
                tokenizer_name_or_path=args.base_llm_model,
                use_fast_tokenizer=not args.use_slow_tokenizer,
            )
            if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast): #for phi3
                tokenizer.padding_side = 'left'
                num_added_tokens = tokenizer.add_special_tokens({
                    "bos_token": "<s>",
                    "eos_token": "</s>",
                    "unk_token": "<unk>",
                    "pad_token": "<pad>",
                })
                # pad token is also equal to eos token in the case of phi3
                assert num_added_tokens in [0,
                                            1,
                                            2], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
                # specific to phi3 case
                # The padding token is set to the unknown token.
                tokenizer.pad_token = tokenizer.unk_token

                # The ID of the padding token is set to the ID of the unknown token.
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
                tokenizer.padding_side = 'left'

            if tag not in questions.columns:
                questions[tag] = ''
            questions[tag].fillna('', inplace=True)
            questions[tag] = questions[tag].astype(str)
            prompts = [
                format_prompt(questions.loc[idx], preset, format='general') for idx in questions.index
            ]
            if chat_formatting_function is not None:
                for idx, prompt in enumerate(prompts):
                    messages = [{"role": "user", "content": prompt}]
                    prompts[idx] = chat_formatting_function(messages, tokenizer, add_bos=False)
                    prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"

            # get the last token because the tokenizer may add space tokens at the start.
            stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:]
            completions = generate_completions(
                model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
                stop_id_sequences=[stop_sequence] if chat_formatting_function is None else None,
                do_sample=False,
                # do_sample=True, top_k=50, top_p=0.95 for phi 3 case
            )
            assert len(completions) == len(prompts)

            # Post-processing of completions
            for idx, completion in zip(questions.index, completions):
                questions.loc[idx, tag] = trim_answer(completion)

            questions["modified_input"] = create_prompt(questions, tag, withToken=withToken, bos="", eos="")
            print("modified input sample:" ,questions['modified_input'].to_list()[0])


        row_data = []
        # Extract row data for specific indices
        if idx_list:
            for idx in idx_list:
                item = {
                    "epoch":"",
                    "prompt_before": questions['Question'][idx],
                    "prompt_after": "",
                    "before_translation": questions[tag][idx],
                    "after_translation": ""  # Placeholder for data to be filled later
                }
                row_data.append(item)

        return questions, row_data
    except Exception as e:
        print(f"An error occurred: {e}")


def run_hf_model_preference(save_dir, withToken, questions, model, tokenizer, tag, preset="qa", batch_size=1,
                            max_new_tokens=50, chat_formatting_function=None, idx_list=[],
                            text_table=None, row_data={}, wandb_run=None):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""
    print("entered run hf model")
    if tag not in questions.columns:
        questions[tag] = ''
    print(questions.columns)
    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)
    if withToken:
        prompts = [
            format_prompt(questions.loc[idx], preset="pref_with_token", format='general', use_manipulated=True) for idx
            in
            questions.index
        ]
    else:
        prompts = [
            format_prompt(questions.loc[idx], preset="pref", format='general', use_manipulated=True) for idx in
            questions.index
        ]
    system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."

    if chat_formatting_function is not None:
        for idx, prompt in enumerate(prompts):
            messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
            prompts[idx] = chat_formatting_function(messages, tokenizer, add_bos=False)
            prompt += "Prompt:" if prompt[-1] in ["\n", " "] else " Prompt:"
    stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:]
    completions = generate_completions(
        model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
        stop_id_sequences=[stop_sequence] if chat_formatting_function is None else None,
        do_sample=False,
    )
    assert len(completions) == len(prompts)
    if wandb_run:
        text_table = wandb.Table(
            columns=["epoch", "prompt_before", "prompt_after", "before_translation", "after_translation"])

    # print("Randomly selected prompt:", prompts[idx])
    # print("Completion:", completions[idx])
    for row, idx in zip(row_data, idx_list):
        # Update the 'after_translation' field
        row["epoch"]=tag.split('/')[-1]
        row["prompt_after"] = prompts[idx]
        row["after_translation"] = completions[idx]

    if wandb_run:
        for row in row_data:
            text_table.add_data(
                row["epoch"],
                row["prompt_before"],
                row["prompt_after"],
                row["before_translation"],
                row["after_translation"]
            )
        print("before after comparison is logged")
        wandb.log({f"samples_evaluation_{tag.split('/')[-1]}": text_table})
    # if it's not a chat format, we will do some post-processing for the answer to make sure it's valid
    # otherwise, we will just store the completions as is
    file_path = os.path.join(save_dir, 'before_after.json')
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(row_data, f, indent=2)

    for idx, completion in zip(questions.index, completions):
        questions.loc[idx, tag] = trim_answer(completion)
    print(questions.columns)
    return questions


def run_hf_model_preference_recursive(save_dir, withToken, questions, model, tokenizer, tag, preset="qa", batch_size=1,
                                      max_new_tokens=500,
                                      chat_formatting_function=None, document=None):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""

    if document is not None and os.path.exists(document):
        print("found doc")
        questions = pd.read_csv(document)

    print("entered run hf model")
    base_idx = tag.rfind('/')
    if base_idx != -1:  # Check if there is a backslash
        original_tag = tag[base_idx + 1:]

    for i in range(0, 4):

        tag = f"{original_tag}_{i}"
        storage = tag
        if tag not in questions.columns:
            questions[tag] = ''
        print(questions.columns)
        questions[tag].fillna('', inplace=True)
        questions[tag] = questions[tag].astype(str)
        if withToken:
            prompts = [
                format_prompt(questions.loc[idx], preset="pref_with_token", format='general', use_manipulated=True) for
                idx in
                questions.index
            ]
        else:
            prompts = [
                format_prompt(questions.loc[idx], preset="pref", format='general', use_manipulated=True) for idx in
                questions.index
            ]
        system_message = "For the following prompt and output, your task is to provide an improved response for the given prompt compared to the given rejected answer."

        if chat_formatting_function is not None:
            for idx, prompt in enumerate(prompts):
                messages = [{"role": "system", "content": system_message}, {"role": "user", "content": prompt}]
                prompts[idx] = chat_formatting_function(messages, tokenizer, add_bos=False)
                prompt += "Prompt:" if prompt[-1] in ["\n", " "] else " Prompt:"

        # get the last token because the tokenizer may add space tokens at the start.
        stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:]
        completions = generate_completions(
            model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
            stop_id_sequences=[stop_sequence] if chat_formatting_function is None else None,
            do_sample=False,
        )
        assert len(completions) == len(prompts)

        for idx, completion in zip(questions.index, completions):
            questions.loc[idx, tag] = trim_answer(completion)

        print(questions.columns)
        filename_to_store = tag.split('/')[-1]
        questions.to_csv(f"{save_dir}/{storage}.csv", index=False)

        questions["modified_input"] = create_prompt(questions, tag, withToken=withToken, bos="", eos="")

    return questions, tag


def run_hf_model(questions, model, tokenizer, tag, preset="qa", batch_size=1, max_new_tokens=50,
                 chat_formatting_function=None, document=None):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""

    if document is not None and os.path.exists(document): #document is used to directly read existing stored predictions
        questions = pd.read_csv(document)
        print(questions[tag].to_list()[0]) # a sample to see
    else:
        if tag not in questions.columns:
            questions[tag] = ''
        print(questions.columns)
        questions[tag].fillna('', inplace=True)
        questions[tag] = questions[tag].astype(str)
        prompts = [
            format_prompt(questions.loc[idx], preset, format='general') for idx in questions.index
        ]

        if chat_formatting_function is not None:
            for idx, prompt in enumerate(prompts):
                messages = [{"role": "user", "content": prompt}]
                prompts[idx] = chat_formatting_function(messages, tokenizer, add_bos=False)
                prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"

        # stop_sequence=tokenizer.encode(tokenizer.eos_token)
        stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:]
        print(stop_sequence)
        completions = generate_completions(
            model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
            stop_id_sequences=[stop_sequence] if chat_formatting_function is None else None,
            do_sample=False,
        )
        assert len(completions) == len(prompts)

        # if it's not a chat format, we will do some post-processing for the answer to make sure it's valid
        # otherwise, we will just store the completions as is
        for idx, completion in zip(questions.index, completions):
            questions.loc[idx, tag] = trim_answer(completion)
        if questions[tag].isnull().any():
            print("There are NaN values in the completions.")
        else:
            print("No NaN values in the completions.")
        filename_to_store= tag.split('/')[-1]
        questions.to_csv(f"results/{filename_to_store}.csv", index=False)
    return questions


def run_hf_model_mc(questions, model, tokenizer, tag, batch_size=1, preset='qa', chat_formatting_function=None):
    """Runs multiple-choice metrics for autoregressive HuggingFace models (GPT-2, GPT-Neo)"""

    set_columns(tag, questions)

    examples = []
    for idx in questions.index:
        # check that candidate answer exists
        if pd.isnull(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue
        if not len(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue

        # reference answers
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

        # prompt for all answers
        prompt = format_prompt(questions.loc[idx], preset, format='general')
        if chat_formatting_function is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
        else:
            prompt += "\nA:"

        # candidate completions
        examples.append({"prompt": prompt, "completions": ref_true + ref_false})

    all_scores = score_completions(model, tokenizer, examples, batch_size=batch_size, aggregation="sum")
    assert len(all_scores) == len(examples)

    for idx, example in zip(questions.index, examples):
        ref_best = format_best(questions.loc[idx, BEST_COL])
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

        completion_scores = all_scores[example["prompt"]]
        scores_true = [completion_scores[ref] for ref in ref_true]
        scores_false = [completion_scores[ref] for ref in ref_false]

        MC_calcs(tag, questions, idx, scores_true, scores_false, ref_true, ref_best)
    return questions


def format_frame(results):
    results = results[[x for x in results.columns if (x != 'Context') and (results[x].dtype != 'O')]]
    new_cols = []
    for col in results.columns:
        split = col.split(' ')
        new_cols.append((split[0], ' '.join(split[1:])))
    results.columns = pd.MultiIndex.from_tuples(new_cols)
    return results


def main(args):
    global bleurt_scorer
    os.makedirs(args.save_dir, exist_ok=True)
    questions = pd.read_csv(os.path.join(args.data_dir, "TruthfulQA.csv"))  # .head(10)
    random_indices = random.sample(range(len(questions)), 3)
    if args.wandb_run_id is not None: #to store results if needed
        wandb.init(project="alignment_translation", id=args.wandb_run_id, resume="allow")

    if args.num_instances is not None:
        questions = questions.sample(args.num_instances, random_state=42)

    if args.model_name_or_path:
        print("Loading model and tokenizer...")

        if args.preference and args.base_llm_model:
            questions, row_data = run_hf_model_create_input(
                args.withToken,
                questions,
                document=args.document,
                tag=args.base_llm_model,
                preset=args.preset,
                batch_size=args.eval_batch_size,
                chat_formatting_function=dynamic_import_function(
                    args.chat_formatting_function) if args.use_chat_format else None,
                idx_list=random_indices #random prompt ids for before-after comparison
            )
        else:
            print("no preference for now")


        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        model = load_hf_lm(
            model_name_or_path=args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast): #phi3
            tokenizer.padding_side = 'left'
            num_added_tokens = tokenizer.add_special_tokens({
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            })
            # pad token is also equal to eos token in the case of phi3
            assert num_added_tokens in [0,
                                        1,
                                        2], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
            # specific to phi3 case
            # The padding token is set to the unknown token.
            tokenizer.pad_token = tokenizer.unk_token

            # The ID of the padding token is set to the ID of the unknown token.
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)


        args.hf_truth_model_name_or_path = args.model_name_or_path
        args.hf_info_model_name_or_path = args.model_name_or_path
        model_key = args.model_name_or_path if args.model_name_or_path else args.openai_engine

        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        allowable_metrics = ['truth', 'info', 'mc', 'bleu', 'rouge', 'bleurt', 'mauve']
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
                model.config.max_position_embeddings))
        if any(metric in allowable_metrics for metric in args.metrics) and args.preference == False:
            print("Running generations - no pref!")
            print(args.document)
            questions = run_hf_model(
                questions,
                model,
                tokenizer,
                tag=args.model_name_or_path,
                preset=args.preset,
                batch_size=args.eval_batch_size,
                chat_formatting_function=dynamic_import_function(
                    args.chat_formatting_function) if args.use_chat_format else None,
                document=args.document
            )
        if any(metric in allowable_metrics for metric in args.metrics) and args.preference == True:
            print("Running generations preference!")
            if args.recursive_test:
                questions, model_key = run_hf_model_preference_recursive(
                    args.save_dir,
                    args.withToken,
                    questions,
                    model,
                    tokenizer,
                    tag=args.model_name_or_path,
                    preset=args.preset,
                    batch_size=args.eval_batch_size,
                    chat_formatting_function=dynamic_import_function(
                        args.chat_formatting_function) if args.use_chat_format else None,
                )
            else:
                run_hf_model_preference(
                    args.save_dir,
                    args.withToken,
                    questions,
                    model,
                    tokenizer,
                    tag=args.model_name_or_path,
                    preset=args.preset,
                    batch_size=args.eval_batch_size,
                    chat_formatting_function=dynamic_import_function(
                        args.chat_formatting_function) if args.use_chat_format else None,
                    idx_list=random_indices,
                    row_data=row_data
                )
        if "mc" in args.metrics:
            print("Running multiple-choice classification!")
            run_hf_model_mc(
                questions,
                model,
                tokenizer,
                tag=args.model_name_or_path,
                batch_size=args.eval_batch_size,
                preset=args.preset,
                chat_formatting_function=dynamic_import_function(
                    args.chat_formatting_function) if args.use_chat_format else None
            )

    elif args.openai_engine:
        # gpt-3 language models
        cache_path = os.path.join(args.save_dir, "openai_query_cache.jsonl")
        if args.openai_engine in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'text-davinci-002',
                                  'code-davinci-002']:
            if "truth" in args.metrics or "info" in args.metrics:
                print("Running generations")
                run_gpt3(questions, args.openai_engine, args.openai_engine, cache_path=cache_path,
                         batch_size=args.eval_batch_size, preset=args.preset)
            if 'mc' in args.metrics:
                print("Running multiple-choice classification!")
                run_gpt3_mc(questions, args.openai_engine, args.openai_engine, cache_path=cache_path,
                            batch_size=args.eval_batch_size, preset=args.preset)
        # other openai engines
        else:
            if "truth" in args.metrics or "info" in args.metrics:
                print("Running generations")
                run_chatgpt(questions, args.openai_engine, args.openai_engine, cache_path=cache_path,
                            batch_size=args.eval_batch_size, preset=args.preset)
            if "mc" in args.metrics:
                raise ValueError("OpenAI Chat engines does not support MC metrics.")

    # run metrics
    print("Running metrics!")
    print(questions.columns)
    for metric in args.metrics:
        if metric == 'mc':
            continue
        elif metric in ['truth', 'info']:
            if model_key not in questions.columns:
                raise ValueError("Answers missing for {0}!".format(model_key))
            try:
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
                    save_questions(questions, args.save_dir, "predictions.csv")
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
                    save_questions(questions, args.save_dir, "predictions.csv")
            except Exception as err:
                print(err)
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
                    print("mauve started")
                    questions = run_MAUVE(model_key, questions)
                save_questions(questions, args.save_dir, "predictions.csv")
            except Exception as err:
                print(f"Error running {metric} metric: {err}")
        else:
            warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)
    print("columns: ", questions.columns)
    if "truth" in args.metrics and "info" in args.metrics:
        questions["{} truth-info acc".format(model_key)] = questions["{} truth acc".format(model_key)] * questions[
            "{} info acc".format(model_key)]

    # save all
    save_questions(questions, args.save_dir, "predictions.csv")
    # questions.to_csv(os.path.join(args.save_dir, "predictions.csv"), index=False)

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
    results.to_csv(os.path.join(args.save_dir, 'summary.csv'))

    print(results)

    results = results.loc[model_key].to_dict()
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_llm_model",
        type=str,
        help="The HuggingFace model to be evaluated as base llm."
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="csv path of the base (rejected) results."
    )
    parser.add_argument(
        "--preference",
        action="store_true",
        default=False,
        help="Enable a specific preference. Default is False, pass this flag to enable."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The HuggingFace model to be evaluated."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will evaluate the OpenAI engine."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
        help="The directory containing the truthfulqa data. Download from https://github.com/sylinrl/TruthfulQA/tree/main/data."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/truthfulqa/",
        help="The directory to save the results."
    )
    parser.add_argument(
        "--filename_answers",
        type=str,
        default="answer_model_tmp",
        help="The directory to save the results/answers ."
    )
    parser.add_argument(
        "--num_instances",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq",
        action="store_true",
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_finetuned_olmo1b_chat_format",
        # eval.templates.create_prompt_with_phi3_chat_format for phi3
        # "eval.templates.create_prompt_with_tulu_chat_format",
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=['bleu', 'rouge', 'bleurt'],
        choices=['truth', 'info', 'mc', 'bleu', 'rouge', 'bleurt'],
        help='Metrics to run'
    )
    parser.add_argument(
        '--preset',
        type=str,
        default='qa',
        help='Preset to use for prompt generation. Please see presets.py for options.'
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
             'Either `gpt_truth_model_name` or `hf_truth_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        '--hf_info_model_name_or_path',
        type=str,
        help='A trained HuggingFace judge model name to be used for computing the metrics for `info` if it is specified.' \
             'Either `gpt_info_model_name` or `hf_info_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        "--withToken",
        action="store_true",
        help="Set to enable token, no value needed"
    )
    parser.add_argument(
        "--recursive_test",
        action="store_true",
        help="Set to enable token, 3 recursive ATT test will be applied"
    )
    parser.add_argument('--wandb_run_id', type=str, help="wandb_run_id to store")
    parser.add_argument('--with_token', action='store_true', help="Set to enable token, no value needed")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)