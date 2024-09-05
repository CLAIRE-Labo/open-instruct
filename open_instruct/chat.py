import argparse
from pathlib import Path
from copy import deepcopy
import logging

import torch
from transformers import (
    AutoConfig,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from att import apply_att_template


logger = logging.getLogger(__name__)


def generate_response(model, tokenizer, chat, generation_config):
    inputs_base_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
    inputs_base_text = tokenizer.decode(inputs_base_ids)
    # print(f"Input to base: \n\n{inputs_base_text}\n\n")

    model.disable_adapters()
    response_base_ids = model.generate(
        torch.tensor(inputs_base_ids, device='cuda').unsqueeze(0),
        generation_config=generation_config,
        tokenizer=tokenizer,
    )
    model.enable_adapters()

    response_base_ids = response_base_ids[0].tolist()
    response_base_text = tokenizer.decode(response_base_ids)
    assert response_base_text.startswith(inputs_base_text), f"Response does not start with input"
    response_base_text = response_base_text[len(inputs_base_text):].strip()
    print(f"Response from base (w/ sp tokens): \n\n{response_base_text}\n\n")
    response_base_text = tokenizer.decode(tokenizer.encode(response_base_text), skip_special_tokens=True).strip()

    rejected = deepcopy(chat)
    rejected.append({"role": "assistant", "content": response_base_text})
    chosen = deepcopy(chat)
    chosen.append({"role": "assistant", "content": ""})

    data = apply_att_template({'chosen': chosen, 'rejected': rejected}, tokenizer=tokenizer, max_seq_length=1024,
                              debug_print=False, logger=logger)
    input_ids = data['input_ids']
    labels = data['labels']
    prompt_ids = input_ids[labels == -100]
    prompt_text = tokenizer.decode(prompt_ids.tolist(), skip_special_tokens=False)
    # print(f"Prompt: \n\n{prompt_text}\n\n")

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
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    model.load_adapter(str(args.lora_checkpoint))
    model = model.cuda()
    print(model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    generation_config = GenerationConfig(
        max_new_tokens=args.max_resp_length,
        renormalize_logits=True,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
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
        response = generate_response(model, tokenizer, chat, generation_config)
        chat.append({"role": "assistant", "content": response})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat')
    parser.add_argument('base_model', type=str, help='Base HF model')
    parser.add_argument('lora_checkpoint', type=Path, help='Adapter to load')
    parser.add_argument('--max_interactions', type=int, default=10, help='Max interactions')
    parser.add_argument('--max_resp_length', type=int, default=256, help='Max interactions')
    parser.add_argument('--greedy', action='store_true', help='Use greedy decoding')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top p for nucleus sampling')
    args = parser.parse_args()

    main(args)
