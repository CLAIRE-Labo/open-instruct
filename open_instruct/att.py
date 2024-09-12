from constants import BAD_MISTRAL_CHAT_TEMPLATE, ATT_SYSTEM_PROMPT, ATT_TEMPLATE
from load_utils import pretty_print_chatml

import torch


# OLMo chat template handles the BOS token, so we don't need to fiddle with add_bos.
# See https://huggingface.co/allenai/OLMo-7B-Instruct/commit/f09de2dc46d1e848f19dd7161bd998973e2b1272
def apply_att_template(example, tokenizer, max_seq_length, debug_print=False, logger=None):
    chosen = example['chosen']
    rejected = example['rejected']

    assert len(chosen) == len(rejected), \
        f"Chosen and rejected should have the same length, got {len(chosen)} and {len(rejected)}.\n" \
        f"Chosen:\n{pretty_print_chatml(chosen)}\nRejected:\n{pretty_print_chatml(rejected)}"

    assert len(chosen) > 1, f"Chosen and rejected should have at least 2 messages, got {len(chosen)}"
    # All messages but the last one should be identical, like in the Anthropic HH data
    for i, (mc, mr) in enumerate(zip(chosen[:-1], rejected[:-1])):
        assert mc == mr, f"Chosen and rejected should be identical, got {mc} and {mr}"
    assert chosen[-1]['role'] == 'assistant' and rejected[-1]['role'] == 'assistant', \
        f"The last message in both chosen and rejected should be by the assistant, got {chosen[-1]['role']} and {rejected[-1]['role']}"

    if chosen[0]['role'] == 'system':
        existing_system_prompt = "\n\n" + chosen[0]['content']
        chosen = chosen[1:]
        rejected = rejected[1:]
    else:
        existing_system_prompt = ""

    messages = chosen[:-2]
    messages.append({
        'role': 'user',
        'content': ATT_TEMPLATE.format(last_user_message=chosen[-2]['content'], rejected=rejected[-1]['content'])
    })

    for i in range(0, len(messages) - 1):
        assert messages[i]['role'] != messages[i + 1]['role'], \
            f"Messages should alternate between user and assistant, got {messages[i]['role']} and {messages[i + 1]['role']}\n" \
            f"You can use mege_consecutive_messages to achieve that. \n" \
            f"Messages:\n{pretty_print_chatml(messages)}"

    # The chat template used by Mistral is not convenient because it only adds the system prompt to the last
    # user message, and only does it if the last message in the conversation is indeed by the user. This would make it
    # painful to set up the labels for SFT. Hence, we just prepend the system prompt to the first user message instead.
    if tokenizer.chat_template == BAD_MISTRAL_CHAT_TEMPLATE:
        messages[0]["content"] = ATT_SYSTEM_PROMPT + '\n\n' + messages[0]["content"]
    else:
        system_msg = {'role': 'system', 'content': ATT_SYSTEM_PROMPT + existing_system_prompt}
        messages = [system_msg] + messages

    try:
        # TODO Skander check max token length of HH data
        prompt_text = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True,
                                                    tokenize=False, max_length=max_seq_length)
    except Exception as e:
        if logger is not None:
            logger.error(f"Error in apply_chat_template when generating the prompt: {e}")
            logger.error("Messages:")
            logger.error(pretty_print_chatml(messages))
        raise e

    if debug_print and logger is not None:
        logger.info("The prompt:\n\"\"\"\n" + prompt_text + "\n\"\"\"")
    tokens = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True,
                                           max_length=max_seq_length)
    end_idx = len(tokens)

    messages.append(chosen[-1])

    try:
        response_text = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=False,
                                                      tokenize=False, max_length=max_seq_length)
    except Exception as e:
        if logger is not None:
            logger.error(f"Error in apply_chat_template when generating the response message: {e}")
            logger.error("Messages:")
            logger.error(pretty_print_chatml(messages))
        raise e

    assert prompt_text == response_text[:len(prompt_text)], \
        f"Currently it is assumed that the prompt and response should be the same up to the end of the prompt," \
        f" got \"{prompt_text}\" and \"{response_text}\""
    expected_response_text = response_text[len(prompt_text):]
    if debug_print and logger is not None:
        logger.info("Target response:\n\"\"\"\n" + expected_response_text + "\n\"\"\"")

    input_ids = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=False,
                                              max_length=max_seq_length)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :end_idx] = -100
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


