import sys
import torch
import tqdm
import json
import time
import asyncio
from pathlib import Path
import os
from importlib import import_module
import shutil
import warnings
from logging import getLogger
from datetime import datetime
import subprocess
from copy import deepcopy

from transformers import StoppingCriteria, AutoTokenizer
from safetensors import safe_open
from safetensors.torch import save_file
from eval.dispatch_openai_requests import dispatch_openai_chat_requesets, dispatch_openai_prompt_requesets

# from accelerate.logging import get_logger

# logger = get_logger(__name__)
# logger = getLogger(__name__)

sys.path.append(str(Path(__file__).parents[1].absolute().as_posix()))
from eval.chat import (
    generate_responses_vllm,
    generate_responses_vllm_att,
    generate_response_att_lora,
    generate_response,
)
from open_instruct.load_utils import load_tokenizer, save_args


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=True,
                         disable_tqdm=False, **generation_kwargs):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                eos_token_id=tokenizer.eos_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx,
                               token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in
                               stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts) // num_return_sequences)

    assert len(generations) == len(
        prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts, candidate_token_ids=None, batch_size=1,
                              return_token_predictions=False, add_special_tokens=True, disable_tqdm=False):
    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt",
                                      add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        batch_logits = model(input_ids=batch_input_ids, attention_mask=attention_mask).logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)
        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


@torch.no_grad()
def score_completions(model, tokenizer, scoring_examples, batch_size=1, aggregation="sum", disable_tqdm=False):
    '''
    Each scoring example is a dict, which contains the following keys:
    - prompt: the prompt to score
    - completions: a list of completions to score
    '''

    # unroll the scoring examples
    unrolled_examples = []
    for scoring_example in scoring_examples:
        prompt = scoring_example["prompt"]
        for completion in scoring_example["completions"]:
            unrolled_examples.append({
                "prompt": prompt,
                "completion": completion
            })

    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(unrolled_examples), desc="Scoring Completions")

    scores = []
    for i in range(0, len(unrolled_examples), batch_size):
        batch_prompts = [example["prompt"] for example in unrolled_examples[i:i + batch_size]]
        batch_examples = [
            (example["prompt"] if example["prompt"][-1] in ["\n", " "] else example["prompt"] + " ")
            + example["completion"] for example in unrolled_examples[i:i + batch_size]
        ]
        tokenized_batch = tokenizer(batch_examples, padding="longest", return_tensors="pt")
        if model.device.type == "cuda":
            tokenized_batch = {
                key: value.cuda() for key, value in tokenized_batch.items()
            }
        tokenized_batch.pop("token_type_ids", None)
        outputs = model(**tokenized_batch)

        for example_idx, (prompt, example) in enumerate(zip(batch_prompts, batch_examples)):
            tokenized_prompt = tokenizer(prompt, padding=False, return_tensors="pt").input_ids.squeeze(0)
            tokenized_example = tokenizer(example, padding=False, return_tensors="pt").input_ids.squeeze(0)
            completion_ids = tokenized_example[len(tokenized_prompt):]

            # get the logits for the entire example, removing the padding logits
            if tokenizer.padding_side == "right":
                example_logits = outputs.logits[example_idx, :len(tokenized_example), :]
            else:
                example_logits = outputs.logits[example_idx, -len(tokenized_example):, :]

            # get the logits for the completion portion - note we need to shift the index left by 1 because logits are computed for the next token
            completion_logits = example_logits[len(tokenized_prompt) - 1:len(tokenized_example) - 1, :]
            completion_log_probs = torch.log_softmax(completion_logits, dim=-1)[
                range(len(completion_ids)), completion_ids]

            if aggregation == "sum":
                score = completion_log_probs.sum().item()
            elif aggregation == "mean":
                score = completion_log_probs.mean().item()
            elif aggregation == "max":
                score = completion_log_probs.max().item()
            else:
                raise ValueError("Invalid aggregation method: {}".format(aggregation))
            scores.append(score)

        if not disable_tqdm:
            progress.update(len(batch_examples))

    # roll up the scores
    rolled_up_scores = {}
    for unrolled_example, score in zip(unrolled_examples, scores):
        prompt = unrolled_example["prompt"]
        completion = unrolled_example["completion"]
        if prompt not in rolled_up_scores:
            rolled_up_scores[prompt] = {}
        rolled_up_scores[prompt][completion] = score

    return rolled_up_scores


def load_hf_lm(
        model_name_or_path,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        gptq_model=False,
        token=os.getenv("HF_TOKEN", None),
        config=None,
        trust_other_remote_code=False
):
    # Loading OLMo models from HF requires `trust_remote_code=True`.
    # TODO: Implement this via command-line flag rather than hardcoded list.
    trusted_models = ["allenai/OLMo-7B", "allenai/OLMo-7B-Twin-2T", "allenai/OLMo-1B"]
    if model_name_or_path in trusted_models or trust_other_remote_code:
        trust_remote_code = True
    else:
        trust_remote_code = False

    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
    if gptq_model:
        from auto_gptq import AutoGPTQForCausalLM
        model_wrapper = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path, device="cuda:0", use_triton=True, trust_remote_code=trust_remote_code
        )
        model = model_wrapper.model
    elif load_in_8bit:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map=device_map,
            load_in_8bit=True,
            token=token,
            trust_remote_code=trust_remote_code,
            config=config
        )
    else:
        if device_map:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
                config=config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                token=token,
                trust_remote_code=trust_remote_code,
                config=config
            )
            if torch.cuda.is_available():
                model = model.cuda()
        if convert_to_half:
            model = model.half()
    model.eval()
    return model


def load_hf_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        use_fast_tokenizer=True,
        padding_side="left",
        token=os.getenv("HF_TOKEN", None),
):
    from transformers import AutoTokenizer

    # Need to explicitly import the olmo tokenizer.
    try:
        from hf_olmo import OLMoTokenizerFast
    except ImportError:
        warnings.warn("OLMo not installed. Ignore if using a different model.")

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=use_fast_tokenizer, token=token)
    except:
        # some tokenizers (e.g., GPTNeoXTokenizer) don't have the slow or fast version, so we just roll back to the default one
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=token)
    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_hf_lm_and_tokenizer(
        model_name_or_path,
        tokenizer_name_or_path=None,
        device_map="auto",
        torch_dtype="auto",
        load_in_8bit=False,
        convert_to_half=False,
        gptq_model=False,
        padding_side="left",
        use_fast_tokenizer=True,
        token=os.getenv("HF_TOKEN", None),
):
    tokenizer = load_hf_tokenizer(
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        use_fast_tokenizer=use_fast_tokenizer,
        padding_side=padding_side,
        token=token,
    )
    model = load_hf_lm(
        model_name_or_path=model_name_or_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        load_in_8bit=load_in_8bit,
        convert_to_half=convert_to_half,
        gptq_model=gptq_model,
        token=token,
    )
    from transformers import GPTNeoXForCausalLM, OPTForCausalLM
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
            model.config.max_position_embeddings))
    return model, tokenizer


def query_openai_chat_model(engine, instances, output_path=None, batch_size=10, retry_limit=5,
                            reuse_existing_outputs=True, **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = [{"role": "user", "content": instance["prompt"]}]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_chat_requesets(
                        messages_list=messages_list,
                        model=engine,
                        **completion_kwargs,
                    ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30 * retry_count} seconds.")
                time.sleep(30 * retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output.choices[0].message.content
            instance["response_metadata"] = output.json()
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


def query_openai_model(engine, instances, output_path=None, batch_size=10, retry_limit=5, reuse_existing_outputs=True,
                       **completion_kwargs):
    '''
    Query OpenAI chat model and save the results to output_path.
    `instances` is a list of dictionaries, each dictionary contains a key "prompt" and a key "id".
    '''
    existing_data = {}
    if reuse_existing_outputs and output_path is not None and os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                instance = json.loads(line)
                existing_data[instance["id"]] = instance

    # by default, we use temperature 0.0 to get the most likely completion.
    if "temperature" not in completion_kwargs:
        completion_kwargs["temperature"] = 0.0

    results = []
    if output_path is not None:
        fout = open(output_path, "w")

    retry_count = 0
    progress_bar = tqdm.tqdm(total=len(instances))
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i + batch_size]
        if all([x["id"] in existing_data for x in batch]):
            results.extend([existing_data[x["id"]] for x in batch])
            if output_path is not None:
                for instance in batch:
                    fout.write(json.dumps(existing_data[instance["id"]]) + "\n")
                    fout.flush()
            progress_bar.update(batch_size)
            continue
        messages_list = []
        for instance in batch:
            messages = instance["prompt"]
            messages_list.append(messages)
        while retry_count < retry_limit:
            try:
                outputs = asyncio.run(
                    dispatch_openai_prompt_requesets(
                        prompt_list=messages_list,
                        model=engine,
                        **completion_kwargs,
                    ))
                retry_count = 0
                break
            except Exception as e:
                retry_count += 1
                print(f"Error while requesting OpenAI API.")
                print(e)
                print(f"Sleep for {30 * retry_count} seconds.")
                time.sleep(30 * retry_count)
                print(f"Retry for the {retry_count} time.")
        if retry_count == retry_limit:
            raise RuntimeError(f"Failed to get response from OpenAI API after {retry_limit} retries.")
        assert len(outputs) == len(batch)
        for instance, output in zip(batch, outputs):
            instance[f"output"] = output.choices[0].text
            instance["response_metadata"] = output.json()
            results.append(instance)
            if output_path is not None:
                fout.write(json.dumps(instance) + "\n")
                fout.flush()
        progress_bar.update(batch_size)
    return results


def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function


def add_eval_args(parser):
    parser.add_argument(
        "train_run_args",
        type=Path,
        help="Path to the training run saved args.json. Allows us to faithfully load the model, tokenizer, and determine the type of ATT template to use.",
    )
    parser.add_argument(
        "tuned_checkpoint",
        type=Path,
        help="LoRA adapter or a full fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--base_model_root", type=Path, help="If provided, prepended to the base model path in the train_run_args."
    )
    parser.add_argument("--evaluated_name", required=True, type=str, help="The name of the model that will be saved.")

    parser.add_argument(
        '--cache_dir',
        type=Path,
        help="Directory to cache eval results and reformatted checkpoints.",
    )

    parser.add_argument(
        "--is_lora",
        action="store_true",
        help="If set, we assume that the checkpoint is a LoRA adapter rather than a full model.",
    )
    parser.add_argument(
        "--not_att",
        action="store_true",
        help="If set, we assume that the fine-tuned model is NOT an ATT model.",
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="If specified, we will load the model from a revision of the model in the hub. If not specified, will use model_revision from train_args.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here. If not specified, will use tokenizer_name from train_args.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer.",
    )

    # sampling
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If given, we will use vLLM to generate the predictions - much faster.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for generating predictions with vLLM (working around a bug in vLLM, setting smaller batch size can help). If not set, the entire dataset will be processed in one batch,  will be done by vLLM."
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="If specified, we will use greedy decoding instead of sampling."
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="The nucleus sampling parameter."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="The temperature for sampling."
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate."
    )


def copy_all_files(src_dir, dest_dir):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_dir = Path(src_dir)
    for file_path in src_dir.iterdir():
        if file_path.is_file():
            shutil.copy(file_path, dest_dir / file_path.name)


# vLLM unfortunately requires the checkpoints to be in a slightly different format compared to what HF provides.
def maybe_create_reformatted_lora_checkpoint(tuned_checkpoint, cache_dir=None):
    tuned_checkpoint = Path(tuned_checkpoint)
    if cache_dir is None:
        reformatted_checkpoint = tuned_checkpoint.parent / f"{tuned_checkpoint.stem}_reformatted"
    else:
        rel_path = Path(*tuned_checkpoint.parent.absolute().parts[1:])
        reformatted_checkpoint = Path(cache_dir) / rel_path / f"{tuned_checkpoint.stem}_reformatted"
    print(f"Reformatted checkpoint path: {reformatted_checkpoint}")
    if reformatted_checkpoint.exists():
        print("Reformatted checkpoint already exists. Skipping.")
        return reformatted_checkpoint

    print("Reformatting checkpoint...")
    reformatted_checkpoint.mkdir(parents=True)
    shutil.copyfile(tuned_checkpoint / "adapter_config.json", reformatted_checkpoint / "adapter_config.json")

    all_tensors = {}
    tensors_to_move = {}
    tensors_remaining = {}

    # Load the original tensors
    with safe_open(tuned_checkpoint / "adapter_model.safetensors", framework="pt") as f:
        for tensor_name in f.keys():
            tensor = f.get_tensor(tensor_name)
            all_tensors[tensor_name] = tensor

            # Check if the tensor should be removed
            if 'lm_head' in tensor_name or 'embed_tokens' in tensor_name:
                tensors_to_move[tensor_name] = tensor
            else:
                tensors_remaining[tensor_name] = tensor

    print(f"Saving the following weights into the embedding file: {list(tensors_to_move.keys())}")

    # Save the tensors without 'lm_head' and 'embeddings'
    save_file(tensors_remaining, reformatted_checkpoint / "adapter_model.safetensors")
    # Save the 'lm_head' and 'embeddings' tensors separately
    save_file(tensors_to_move, reformatted_checkpoint / "new_embeddings.safetensors")

    # Copy tokenizer if it exists
    tokenizer_path = tuned_checkpoint.parent / "tokenizer"
    if tokenizer_path.exists():
        shutil.copytree(tokenizer_path, reformatted_checkpoint, dirs_exist_ok=True)
    else:
        # Alternative layout of tokenizer files
        tokenizer_files = ["tokenizer.model", "tokenizer.json", "tokenizer_config.json", "added_tokens.json",
                           "special_tokens_map.json"]
        if all([(tuned_checkpoint.parent / file).exists() for file in tokenizer_files]):
            for file in tokenizer_files:
                shutil.copy(tuned_checkpoint.parent / file, reformatted_checkpoint)

    return reformatted_checkpoint


def run_att_model_for_eval(train_args, eval_args, chats):
    import vllm

    eval_args = deepcopy(eval_args)

    model_name_or_path = train_args.model_name_or_path
    if eval_args.base_model_root is not None and (eval_args.base_model_root / model_name_or_path).exists():
        model_name_or_path = str(eval_args.base_model_root / model_name_or_path)
    tokenizer_name_or_path = (
        train_args.tokenizer_name if train_args.tokenizer_name else model_name_or_path
    )
    hf_revision = train_args.model_revision

    tokenizer, _ = load_tokenizer(train_args, False)

    if eval_args.use_vllm:
        print("Loading vLLM model...")
        if hasattr(eval_args, "mem_util"):
            mem_util = eval_args.mem_util
        else:
            mem_util = 0.9 if eval_args.is_lora else 0.4

        merged_lora = None
        need_base = eval_args.is_lora or not eval_args.not_att
        if need_base:
            try:
                base_model = vllm.LLM(
                    model=model_name_or_path,
                    revision=hf_revision,
                    tokenizer=tokenizer_name_or_path,
                    tokenizer_revision=hf_revision,
                    tokenizer_mode="auto" if not eval_args.use_slow_tokenizer else "slow",
                    trust_remote_code=True,
                    enable_lora=eval_args.is_lora,
                    max_lora_rank=128,
                    disable_sliding_window=False,
                    # disable_sliding_window=True if not hasattr(eval_args, "disable_sliding_window") \
                    #     else eval_args.disable_sliding_window,
                    gpu_memory_utilization=mem_util,
                )
            except ValueError as e:
                if "does not support LoRA" in str(e):
                    print("Model does not support LoRA. We will merge the lora adapter into the base model.")
                    # Will have to store both base and tuned in memory
                    mem_util = 0.4
                    if eval_args.cache_dir is None:
                        merge_dir = eval_args.tuned_checkpoint.parent / f"{eval_args.tuned_checkpoint.name}_merged"
                    else:
                        rel_path = Path(*eval_args.tuned_checkpoint.parent.absolute().parts[1:])
                        merge_dir = Path(eval_args.cache_dir) / rel_path / f"{eval_args.tuned_checkpoint.name}_merged"
                    if merge_dir.exists():
                        print(f"The merged model directory {merge_dir} already exists. Skipping.")
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        tmp_args_fname = Path(f"tmp_args_{timestamp}.json")
                        save_args(train_args, tmp_args_fname)
                        merge_command = \
                            f'python {Path(__file__).parents[1] / "open_instruct/merge_lora_from_training.py"}' \
                            f' --lora_model_name_or_path {eval_args.tuned_checkpoint.absolute().as_posix()}' \
                            f' --train_args {tmp_args_fname.absolute().as_posix()}' \
                            f' --output_dir {merge_dir}' \
                            f' --save_tokenizer'
                        print(f"Running merge command:\n{merge_command}")
                        merge_subprocess = subprocess.run(merge_command, shell=True, capture_output=True)
                        tmp_args_fname.unlink()
                        if merge_subprocess.returncode != 0:
                            print("Error while merging the model.")
                            print(merge_subprocess.stdout.decode())
                            print(merge_subprocess.stderr.decode())
                            raise RuntimeError("Error while merging the model.")
                    eval_args.is_lora = False
                    eval_args.tuned_checkpoint = merge_dir
                    # Don't need the base model anymore, will load the merged model later
                else:
                    raise e

        sampling_params = vllm.SamplingParams(
            top_p=1.0 if eval_args.greedy else eval_args.top_p,
            temperature=0.0 if eval_args.greedy else eval_args.temperature,
            seed=239,
            max_tokens=eval_args.max_new_tokens,
            n=1 if not hasattr(eval_args, "n_sample_per_prompt") else eval_args.n_sample_per_prompt,
        )

        responses_log = []
        # Generate outputs based on ATT and LoRA settings
        if hasattr(eval_args, "is_base_model") and eval_args.is_base_model:
            formatted_prompts, outputs = generate_responses_vllm(
                base_model,
                tokenizer,
                chats,
                sampling_params,
            )
            return formatted_prompts, outputs
        elif eval_args.not_att:
            if eval_args.is_lora:
                # Not ATT, using LoRA
                reformatted_checkpoint = maybe_create_reformatted_lora_checkpoint(eval_args.tuned_checkpoint,
                                                                                  cache_dir=eval_args.cache_dir)
                lora_request = vllm.lora.request.LoRARequest(
                    "adapter", 1, reformatted_checkpoint.absolute().as_posix()
                )
                # generate_responses_vllm returns formatted prompts and responses
                formatted_prompts, outputs = generate_responses_vllm(
                    base_model,
                    tokenizer,
                    chats,
                    sampling_params,
                    lora_request=lora_request,
                    batch_size=eval_args.batch_size
                )
            else:
                # Not ATT, not using LoRA
                tuned_model = vllm.LLM(
                    model=eval_args.tuned_checkpoint.absolute().as_posix(),
                    revision=eval_args.hf_revision,
                    tokenizer=tokenizer_name_or_path,
                    tokenizer_revision=eval_args.hf_revision,
                    tokenizer_mode="auto" if not eval_args.use_slow_tokenizer else "slow",
                    trust_remote_code=True,
                    enable_lora=False,
                    disable_sliding_window=False,
                    # disable_sliding_window=True,
                    gpu_memory_utilization=mem_util,
                )
                formatted_prompts, outputs = generate_responses_vllm(
                    tuned_model, tokenizer, chats, sampling_params, batch_size=eval_args.batch_size
                )
            # Collect responses
            for chat, prompt, output in zip(chats, formatted_prompts, outputs):
                responses_log.append(
                    {
                        "instruction": chat[0]["content"],
                        "generator": eval_args.evaluated_name,
                        "prompt": prompt,
                        "output": output,
                    }
                )
        else:
            if eval_args.is_lora:
                # ATT, using LoRA
                reformatted_checkpoint = maybe_create_reformatted_lora_checkpoint(eval_args.tuned_checkpoint,
                                                                                  cache_dir=eval_args.cache_dir)
                lora_request = vllm.lora.request.LoRARequest(
                    "adapter", 1, reformatted_checkpoint.absolute().as_posix()
                )
                (
                    prompts_base,
                    responses_base,
                    prompts_att,
                    outputs,
                ) = generate_responses_vllm_att(base_model, tokenizer, chats, sampling_params,
                                                lora_request=lora_request,
                                                batch_size=eval_args.batch_size)
            else:
                # ATT, not using LoRA
                (
                    prompts_base,
                    responses_base,
                    prompts_att,
                    outputs,
                ) = generate_responses_vllm_att(base_model, tokenizer, chats, sampling_params,
                                                att_model_checkpoint=eval_args.tuned_checkpoint.absolute().as_posix(),
                                                tokenizer_name=tokenizer_name_or_path,
                                                batch_size=eval_args.batch_size)
            # Collect responses
            for chat, prompt_base, response_base, prompt_att, output in zip(
                    chats, prompts_base, responses_base, prompts_att, outputs
            ):
                responses_log.append(
                    {
                        "instruction": chat[0]["content"],
                        "generator": eval_args.evaluated_name,
                        "input_base": prompt_base,
                        "response_base": response_base,
                        "att_prompt": prompt_att,
                        "output": output,
                    }
                )

        del base_model  # Free up GPU memory
    else:
        assert False, "Currently only supporting vLLM generations."
        # If we need non-vLLM, need to debug and test this

        assert args.is_lora, f"Full model evaluation is not supported yet."

        base_model, tokenizer, actual_eos_token, _, _ = load_tokenizer_model(accelerator, train_args)
        actual_eos_token_id = tokenizer.encode(actual_eos_token)[0]
        base_model.load_adapter(str(args.tuned_checkpoint))
        base_model = base_model.cuda()
        base_model.eval()

        print(base_model)

        generation_config = GenerationConfig(
            max_length=args.max_seq_length,
            # Note that max_length in the config is overridden by max_new_tokens when model.generate is run. However, we use max_length to crop the prompts if necessary
            max_new_tokens=args.max_new_tokens,
            renormalize_logits=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=actual_eos_token_id,
            stop_strings=tokenizer.eos_token,
            do_sample=not args.greedy,
            top_p=args.top_p,
            temperature=args.temperature,
        )
        # filter instructions to only include those that are in prompts
        for prompt in tqdm(prompts, desc="Generating responses"):
            messages = [{"role": "user", "content": prompt}]
            log_entry = {"instruction": prompt, "generator": model_name}
            if not args.not_att:
                input_base, response_base, att_prompt, att_response_with_special_tokens, att_response \
                    = generate_response_att_lora(base_model, tokenizer, messages, generation_config)
                log_entry["input_base"] = input_base
                log_entry["response_base"] = response_base
                log_entry["att_prompt"] = att_prompt
                log_entry["response_with_special_tokens"] = att_response_with_special_tokens
                log_entry["output"] = att_response
            else:
                template_prompt, response_with_special_tokens, response \
                    = generate_response(base_model, tokenizer, messages, generation_config)
                log_entry["template_prompt"] = template_prompt
                log_entry["response_with_special_tokens"] = response_with_special_tokens
                log_entry["output"] = response
            responses_log.append(log_entry)
            print(f"Prompt: {prompt} \nResponse: {log_entry['output']}")

    return responses_log
