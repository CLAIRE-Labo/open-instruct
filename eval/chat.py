import sys
import argparse
from pathlib import Path
from copy import deepcopy
import logging

import torch
from ply.yacc import token
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import accelerate

sys.path.append(str(Path(__file__).parents[1] / "open_instruct"))
print(sys.path)
from att import apply_att_template
from load_utils import load_args, load_tokenizer_model

logger = logging.getLogger(__name__)


def generate_response(model, tokenizer, chat, generation_config):
    assert generation_config.max_new_tokens is not None
    max_prompt_len = generation_config.max_length - generation_config.max_new_tokens
    input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    if len(input_ids) > max_prompt_len:
        logger.warning(f"generate_response: tuncated input of size {len(input_ids)} to {max_prompt_len}")
        input_ids = input_ids[-max_prompt_len:]
    input_text = tokenizer.decode(input_ids)

    response_ids = model.generate(
        torch.tensor(input_ids, device='cuda').unsqueeze(0),
        generation_config=generation_config,
        tokenizer=tokenizer,
    )

    response_ids = response_ids[0].tolist()
    response_text_with_special_tokens = tokenizer.decode(response_ids)
    assert response_text_with_special_tokens.startswith(input_text), f"Response does not start with input"
    response_text_with_special_tokens = response_text_with_special_tokens[len(input_text):].strip()
    response_text = tokenizer.decode(tokenizer.encode(response_text_with_special_tokens),
                                     skip_special_tokens=True).strip()

    return input_text, response_text_with_special_tokens, response_text


def generate_response_att_lora(model, tokenizer, chat, generation_config):
    assert generation_config.max_new_tokens is not None
    max_prompt_len = generation_config.max_length - generation_config.max_new_tokens

    model.disable_adapters()
    inputs_base_text, response_base_text_with_special_tokens, response_base_text \
        = generate_response(model, tokenizer, chat, generation_config)
    # print(f"Input to base: \n\n{inputs_base_text}\n\n")
    print(f"Response from base (w/ sp tokens): \n\n{response_base_text_with_special_tokens}\n\n")
    model.enable_adapters()

    rejected = deepcopy(chat)
    rejected.append({"role": "assistant", "content": response_base_text})
    chosen = deepcopy(chat)
    chosen.append({"role": "assistant", "content": ""})

    data = apply_att_template({'chosen': chosen, 'rejected': rejected}, tokenizer=tokenizer, max_seq_length=1024,
                              debug_print=False, logger=logger)
    input_ids = data['input_ids']
    labels = data['labels']
    prompt_ids = input_ids[labels == -100]
    if len(prompt_ids) > max_prompt_len:
        logger.warning(f"generate_response_att_lora truncated input of size {len(prompt_ids)} to {max_prompt_len}")
        prompt_ids = input_ids[-max_prompt_len:]
    prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    print(f"Prompt: \n\n{prompt_text}\n\n")

    att_response_ids = model.generate(
        prompt_ids.cuda().unsqueeze(0),
        generation_config=generation_config,
        tokenizer=tokenizer,
    )

    att_response_ids = att_response_ids[0].tolist()
    att_response_text = tokenizer.decode(att_response_ids)
    assert att_response_text.startswith(prompt_text), f"Response does not start with prompt"
    att_response_text = att_response_text[len(prompt_text):].strip()
    print(f"Response from att (w/ sp tokens): \n\n{att_response_text}\n\n")
    att_response_text = tokenizer.decode(tokenizer.encode(att_response_text), skip_special_tokens=True).strip()
    return att_response_text


def main(args):
    accelerator = accelerate.Accelerator()

    train_args = load_args(args.train_run_args)
    model, tokenizer, actual_eos_token, _, _ = load_tokenizer_model(accelerator, train_args)
    actual_eos_token_id = tokenizer.encode(actual_eos_token)[0]
    logger.info(f"{actual_eos_token_id=}")
    print(f"{actual_eos_token_id=}")
    model.load_adapter(str(args.lora_checkpoint))
    model = model.cuda()
    model.eval()
    print(model)

    generation_config = GenerationConfig(
        max_length=args.max_seq_length,
        # Note that max_length in the config is overridden by max_new_tokens when model.generate is run. However, we use max_length to crop the prompts if necessary
        max_new_tokens=args.max_resp_length,
        renormalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=actual_eos_token_id,
        stop_strings=tokenizer.eos_token,
        do_sample=not args.greedy,
        top_p=args.top_p,
    )

    chat = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
    ]

    for i in range(args.max_interactions):
        print(f"Interaction {i + 1}")
        chat.append({"role": "user", "content": input("User: ")})
        response = generate_response_att_lora(model, tokenizer, chat, generation_config)
        chat.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat')
    parser.add_argument('train_run_args', type=Path,
                        help='Path to the training run saved args.json. Allows us to faithfully load the model, tokenizer, and determine the type of ATT template to use (NYI)')
    parser.add_argument('lora_checkpoint', type=Path, help='Adapter to load')
    parser.add_argument('--max_interactions', type=int, default=10, help='Max interactions')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='If the total length of the prompt at any point becomes larger than this, the beginning of the prompt is truncated.')
    parser.add_argument('--max_resp_length', type=int, default=300, help='Max length of the LLM response to generate')
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p for nucleus sampling')
    args = parser.parse_args()

    main(args)
