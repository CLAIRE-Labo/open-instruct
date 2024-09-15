import gc
import sys
import argparse
from pathlib import Path
import logging
import json

import pandas as pd
from tqdm import tqdm
import torch
import datasets
import accelerate
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import GenerationConfig, AutoTokenizer
from alpaca_eval import evaluate as alpaca_farm_evaluate

sys.path.append(str(Path(__file__).parents[2].absolute().as_posix()))
from eval.utils import query_openai_chat_model, query_openai_model
from eval.chat import generate_response, generate_response_att_lora, generate_responses_vllm, \
    generate_responses_vllm_att
from open_instruct.att import apply_att_template
from open_instruct.load_utils import load_args, load_tokenizer_model

logger = get_logger(__name__)


def evaluate(accelerator, args):
    set_seed(239)

    train_args = load_args(args.train_run_args)
    output_dir = args.tuned_checkpoint / "eval" / "alpaca_farm" if args.output_dir is None else args.output_dir
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
    # prompts = prompts[:10]

    responses_log = []
    if responses_log_path.exists():
        with open(responses_log_path, "r") as f:
            responses_log = json.load(f)
        logger.info(f"Loaded responses from {responses_log_path}")
    else:
        model_name = train_args.model_name_or_path
        if args.not_att:
            model_name += "+Alignment(?)"
        else:
            model_name += "+ATT"

        logger.info("loading the model...")
        if args.use_vllm:
            import vllm
            from vllm.distributed import destroy_model_parallel

            tokenizer_revision = (
                train_args.model_revision
                if train_args.tokenizer_revision is None
                else train_args.tokenizer_revision
            )
            tokenizer_name = train_args.tokenizer_name if train_args.tokenizer_name is not None else train_args.model_name_or_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision)
            # model_config = vllm.config.ModelConfig(
            #     model=train_args.model_name_or_path,
            #     revision=train_args.model_revision,
            #     tokenizer=tokenizer_name,
            #     tokenizer_revision=tokenizer_revision,
            #     tokenizer_mode="auto",
            #     trust_remote_code=True,
            #     dtype=torch.bfloat16,
            #     seed=239,
            #     disable_sliding_window=True,
            # )
            mem_util = 0.9 if args.is_lora else 0.4
            base_model = vllm.LLM(model=train_args.model_name_or_path,
                                  revision=train_args.model_revision,
                                  tokenizer=tokenizer_name,
                                  tokenizer_revision=tokenizer_revision,
                                  tokenizer_mode="auto",
                                  trust_remote_code=True,
                                  enable_lora=args.is_lora,
                                  disable_sliding_window=True,
                                  gpu_memory_utilization=mem_util,
                                  # model_config=model_config
                                  )

            def create_att_model():
                nonlocal base_model, tokenizer, tokenizer_name, tokenizer_revision, args, train_args, mem_util
                # Can't destroy vLLM state
                # destroy_model_parallel()
                # del base_model
                # gc.collect()
                # torch.cuda.empty_cache()
                # torch.distributed.destroy_process_group()
                return vllm.LLM(model=args.tuned_checkpoint.absolute().as_posix(),
                                revision=train_args.model_revision,
                                tokenizer=tokenizer_name,
                                tokenizer_revision=tokenizer_revision,
                                tokenizer_mode="auto",
                                trust_remote_code=True,
                                enable_lora=False,
                                disable_sliding_window=True,
                                gpu_memory_utilization=mem_util,
                                # model_config=model_config
                                )

            sampling_params = vllm.SamplingParams(
                top_p=args.top_p,
                temperature=args.temperature,
                seed=239,
                max_tokens=args.max_new_tokens,
            )

            chats = [[{"role": "user", "content": prompt}] for prompt in prompts]
            if args.not_att:
                if args.is_lora:
                    lora_request = vllm.lora.request.LoRARequest("adapter", 1,
                                                                 args.tuned_checkpoint.absolute().as_posix())
                    template_prompts, responses = generate_responses_vllm(base_model, tokenizer, chats, sampling_params,
                                                                          lora_request=lora_request)
                else:
                    template_prompts, responses = generate_responses_vllm(create_att_model(), tokenizer, chats,
                                                                          sampling_params)
                responses_log = [{"instruction": prompt, "generator": model_name, "template_prompt": template_prompt,
                                  "output": response} for
                                 prompt, template_prompt, response in zip(prompts, template_prompts, responses)]
            else:
                if args.is_lora:
                    lora_request = vllm.lora.request.LoRARequest("adapter", 1,
                                                                 args.tuned_checkpoint.absolute().as_posix())
                    prompts_base, responses_base, prompts_att, responses = \
                        generate_responses_vllm_att(base_model, tokenizer, chats, sampling_params,
                                                    lora_request=lora_request)
                else:
                    prompts_base, responses_base, prompts_att, responses = \
                        generate_responses_vllm_att(base_model, tokenizer, chats, sampling_params,
                                                    create_att_model=create_att_model)

                responses_log = [
                    {"instruction": prompt,
                     "generator": model_name,
                     "input_base": prompt_base,
                     "response_base": response_base,
                     "att_prompt": att_prompt,
                     # "response_with_special_tokens": response_with_special_tokens,
                     "output": response}
                    for prompt, prompt_base, response_base, att_prompt, response
                    in zip(prompts, prompts_base, responses_base, prompts_att, responses)
                ]

        else:
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
    parser.add_argument('train_run_args', type=Path,
                        help='Path to the training run saved args.json. Allows us to faithfully load the model, tokenizer, and determine the type of ATT template to use (NYI)')
    parser.add_argument('tuned_checkpoint', type=Path, help='LoRA adapter or a full fine-tuned model checkpoint.')
    parser.add_argument('--is_lora', action='store_true',
                        help='If set, we assume that the checkpoint is a LoRA adapter rather than a full model.')
    parser.add_argument('--not_att', action='store_true',
                        help='If set, we assume that the fine-tuned model is NOT an ATT model.')
    parser.add_argument('--att_evaluate_base', action='store_true',
                        help='If set, GPT-4 evaluator is also run on the base model. Like this we dont need to run the base eval separately.')
    parser.add_argument('--output_dir', type=Path,
                        help='Output directory to save results. Default is into {tuned_checkpoint}/eval/alpaca_farm ')
    parser.add_argument(
        "--openai_engine",
        type=str,
        default=None,
        help="If specified, we will use the OpenAI API to generate the predictions.",
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
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='Maximum length of the sequence (prompt + response). If the prompt is longer than this, we will truncate it.')
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=300,
        help="Maximum number of new tokens to generate."
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="If specified, we will use the vLLM to generate the predictions."
    )
    args = parser.parse_args()

    accelerator = accelerate.Accelerator()
    evaluate(accelerator, args)
