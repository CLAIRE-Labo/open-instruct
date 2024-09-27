import sys
from pathlib import Path
import argparse
from copy import deepcopy
import json

from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from datasets import Dataset

sys.path.append(str(Path(__file__).parents[1].absolute().as_posix()))
from open_instruct.constants import BAD_MISTRAL_CHAT_TEMPLATE, ATT_SYSTEM_PROMPT, ATT_TEMPLATE, ATT_RESPONSE_PREFIX
from open_instruct.load_utils import pretty_print_chatml


def add_att_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--reduce_loss',
        default='mean',
        choices=['mean', 'sum'],
        help='How to reduce loss over tokens. Default is mean, but using sum can improve chat model performance.',
    )
    parser.add_argument(
        "--remove_multiturn_data",
        action="store_true",
        help="If set, only \"prompt-response\" data is used and multi-turn dialogue data is filtered out.",
    )
    parser.add_argument(
        "--base_generations_dir",
        type=Path,
        default=None,
        help="If provided, these generations (produced by generate_vllm.py from the ATT repo) are used as y_ref. We extend the dataset, which originally contains triplets (x, y+, y-) to include also (x, y+, y_ref)."
    )
    parser.add_argument(
        "--use_new_token_template",
        action="store_true",
        help="If set, the new token template is used for ATT training."
    )
    parser.add_argument(
        "--loss",
        default="ce",
        choices=["ce", "symmetric", "symmetric_dpo", "symmetric_hinge", "ipo"],
    )

    parser.add_argument('--dpo_beta', type=float, default=0.1, help='Beta in the DPO loss.')
    parser.add_argument('--loss_lambda', type=float, default=1.0, help='Lambda in the symmetric and hinge losses.')
    parser.add_argument(
        '--neg_example_strength', type=float, default=1.0,
        help='The strength of the negative examples when using variants of the symmetric loss.'
    )
    parser.add_argument(
        '--hinge_delta', type=float, default=0.0,
        help='Delta used in the symmetric_hinge loss.'
    )


def load_base_generations(base_generations_dir: Path, existing_dataset: Dataset) -> Dataset:
    with open(base_generations_dir / "prompts.json") as f:
        prompts = json.load(f)
    with open(base_generations_dir / "generations.json") as f:
        responses = json.load(f)

    assert len(prompts) == len(responses), f"Prompt length mismatch: {len(prompts)} vs {len(responses)}"
    assert len(prompts) == len(existing_dataset), f"Dataset length mismatch: {len(prompts)} vs {len(existing_dataset)}"

    new_dataset = []
    for i, (prompt, response, example) in tqdm(enumerate(zip(prompts, responses, existing_dataset)),
                                               desc="Generating self-improvement data", total=len(prompts)):
        assert prompt == example['chosen'][:-1], f"Prompt mismatch at {i}" \
                                                 f"\"\"\"\n{pretty_print_chatml(prompt)}\n\"\"\"\nvs\n\"\"\"\n{pretty_print_chatml(example['chosen'][:-1])}\n\"\"\""
        new_example = deepcopy(example)
        new_example['rejected'][-1]['content'] = response
        new_dataset.append(new_example)

    # convert to huggingface dataset format
    return Dataset.from_list(new_dataset)


# Parses nested dicts. Each inner dict will become an input to a model.
class DataCollatorForATT(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        keys = features[0].keys()
        return {k: super(DataCollatorForATT, self).__call__([f[k] for f in features], return_tensors=return_tensors)
                for k in keys}


def _check_and_preprocess_chosen_rejected(example):
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
        existing_system_prompt = chosen[0]['content']
        chosen = chosen[1:]
        rejected = rejected[1:]
    else:
        existing_system_prompt = ""

    return chosen, rejected, existing_system_prompt


def _postprocess_input_ids(input_ids, end_idx, max_seq_length):
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :end_idx] = -100
    attention_mask = torch.ones_like(input_ids)

    # We do this at a later stage
    # input_ids = input_ids[:, :max_seq_length]
    # labels = labels[:, :max_seq_length]
    # attention_mask = attention_mask[:, :max_seq_length]

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


# OLMo chat template handles the BOS token, so we don't need to fiddle with add_bos.
# See https://huggingface.co/allenai/OLMo-7B-Instruct/commit/f09de2dc46d1e848f19dd7161bd998973e2b1272
def apply_att_template(example, tokenizer, max_seq_length, debug_print=False, logger=None):
    example = deepcopy(example)
    chosen, rejected, existing_system_prompt = _check_and_preprocess_chosen_rejected(example)

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
        system_prompt = ATT_SYSTEM_PROMPT + "\n\n" + existing_system_prompt if existing_system_prompt else ATT_SYSTEM_PROMPT
        system_msg = {'role': 'system', 'content': system_prompt}
        messages = [system_msg] + messages

    try:
        tokens = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=True)
    except Exception as e:
        if logger is not None:
            logger.error(f"Error in apply_chat_template when generating the prompt: {e}")
            logger.error("Messages:")
            logger.error(pretty_print_chatml(messages))
        raise e

    prompt_text = tokenizer.decode(tokens, skip_special_tokens=False)

    prompt_text += ATT_RESPONSE_PREFIX
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    end_idx = len(prompt_tokens)

    if debug_print and logger is not None:
        logger.info("The prompt:\n\"\"\"\n" + prompt_text + "\n\"\"\"")

    chosen_msg = chosen[-1]
    chosen_msg['content'] = ATT_RESPONSE_PREFIX + chosen_msg['content']
    messages.append(chosen[-1])

    try:
        input_ids = tokenizer.apply_chat_template(conversation=messages, add_generation_prompt=False)
    except Exception as e:
        if logger is not None:
            logger.error(f"Error in apply_chat_template when generating the response message: {e}")
            logger.error("Messages:")
            logger.error(pretty_print_chatml(messages))
        raise e

    assert input_ids[0] == tokenizer.bos_token_id, \
        f"The first token should be the BOS token, got {tokenizer.decode(input_ids[:10], skip_special_tokens=False)}"
    assert input_ids[1] != tokenizer.bos_token_id, \
        f"Got duplicated BOS token: {tokenizer.decode(input_ids[:10], skip_special_tokens=False)}"
    assert input_ids[-1] == tokenizer.eos_token_id, \
        f"The last token should be the EOS token, got {tokenizer.decode(input_ids[-10:], skip_special_tokens=False)}"
    assert input_ids[-2] != tokenizer.eos_token_id, \
        f"Got duplicated EOS token: {tokenizer.decode(input_ids[-10:], skip_special_tokens=False)}"

    response_text = tokenizer.decode(input_ids, skip_special_tokens=False)

    if prompt_text != response_text[:len(prompt_text)]:
        print(f"Prompt:\n\"\"\"\n{prompt_text}\n\"\"\"")
    assert prompt_text == response_text[:len(prompt_text)], \
        f"Currently it is assumed that the prompt and response should be the same up to the end of the prompt," \
        f" got \"{prompt_text}\" and \"{response_text}\""
    assert prompt_tokens == input_ids[:end_idx], \
        f"The prompt tokens should be the same as the first end_idx tokens of the response, " \
        f"got prompt ids\n{prompt_tokens}\nresponse ids\n{input_ids[:end_idx]}\n" \
        f" prompt:\n{prompt_text}\"\nresponse\n\"{response_text}\""

    expected_response_text = response_text[len(prompt_text):]
    if debug_print and logger is not None:
        logger.info("Target response:\n\"\"\"\n" + expected_response_text + "\n\"\"\"")

    return _postprocess_input_ids(input_ids, end_idx, max_seq_length)


def encode_with_chat_template(chat, tokenizer, max_seq_length, debug_print=False, logger=None):
    chat = deepcopy(chat)
    try:
        prompt_tokens = tokenizer.apply_chat_template(conversation=chat[:-1], add_generation_prompt=True)
    except Exception as e:
        if logger is not None:
            logger.error(f"Error in apply_chat_template when generating the prompt: {e}")
            logger.error("Messages:")
            logger.error(pretty_print_chatml(chat[:-1]))
        raise e

    prompt_text = tokenizer.decode(prompt_tokens, skip_special_tokens=False)

    input_ids = tokenizer.apply_chat_template(conversation=chat, add_generation_prompt=False)
    response_text = tokenizer.decode(input_ids, skip_special_tokens=False)
    assert prompt_text == response_text[:len(prompt_text)], \
        f"Currently it is assumed that the prompt and response should be the same up to the end of the prompt," \
        f" got \n\"\"\"\n{prompt_text}\n\"\"\"\nand\n\"\"\"\n{response_text}\n\"\"\"\n"

    assert prompt_tokens == input_ids[:len(prompt_tokens)], \
        f"The prompt tokens should be the same as the first end_idx tokens of the response, " \
        f"got prompt ids\n{prompt_tokens}\nresponse ids\n{input_ids[:len(prompt_tokens)]}\n" \
        f" prompt:\n\"\"\"\n{prompt_text}\n\"\"\"\nresponse\n\"\"\"\n{response_text}\n\"\"\"\n"

    if debug_print and logger is not None:
        expected_response_text = response_text[len(prompt_text):]
        logger.info("The prompt (ref model):\n\"\"\"\n" + prompt_text + "\n\"\"\"")
        logger.info("The response (ref model):\n\"\"\"\n" + expected_response_text + "\n\"\"\"")

    return _postprocess_input_ids(input_ids, len(prompt_tokens), max_seq_length)


# Cropping strategy for symmetric ATT:
# 1. Do not crop the prompt. If it is too long, discard the example.
# 2. The remaining context budget is max_seq_length - prompt_len - slack_len (slack_len is for the template tokens)
# 2. If the examples don't fit, first try cropping yminus to 1/3 of the remaining budget.
# 3. If the examples still don't fit, crop yplus until they do.
def preprocess_for_symmetric_att(example, tokenizer, max_seq_length, slack_len=70, debug_print=False,
                                 logger=None):
    example = deepcopy(example)
    yplus_ref_orig = encode_with_chat_template(example['chosen'], tokenizer, max_seq_length, debug_print, logger)
    yminus_ref_orig = encode_with_chat_template(example['rejected'], tokenizer, max_seq_length, debug_print, logger)

    yplus_tokens = tokenizer.encode(example['chosen'][-1]['content'], add_special_tokens=False)
    yminus_tokens = tokenizer.encode(example['rejected'][-1]['content'], add_special_tokens=False)

    prompt_len = (yplus_ref_orig['labels'] == -100).sum().item()
    yplus_len = len(yplus_tokens)
    yminus_len = len(yminus_tokens)

    if prompt_len >= max_seq_length:
        logger.warning(f"Prompt is too long: {prompt_len}")
        dummy_input = {
            'input_ids': torch.tensor([tokenizer.bos_token_id], dtype=torch.long),
            'labels': torch.tensor([-100], dtype=torch.long),
            'attention_mask': torch.tensor([1], dtype=torch.long),
        }

        return {
            "yplus_att": dummy_input,
            "yminus_att": dummy_input,
            "yplus_ref": dummy_input,
            "yminus_ref": dummy_input,
        }

    remaining_budget = max_seq_length - prompt_len - slack_len
    yminus_len_cropped = min(yminus_len, remaining_budget // 3)
    if prompt_len + yminus_len + yplus_len + slack_len <= max_seq_length:
        # No cropping needed
        pass
    elif prompt_len + yminus_len_cropped + yplus_len + slack_len <= max_seq_length:
        # Only need to crop yminus
        yminus_len_necessary = max_seq_length - prompt_len - yplus_len - slack_len
        yminus_cropped_tok = yminus_tokens[:yminus_len_necessary]
        example['rejected'][-1]['content'] = tokenizer.decode(yminus_cropped_tok, skip_special_tokens=False)
    else:
        # Crop both yminus and yplus
        yminus_cropped_tok = yminus_tokens[:yminus_len_cropped]
        example['rejected'][-1]['content'] = tokenizer.decode(yminus_cropped_tok, skip_special_tokens=False)
        yplus_len_necessary = max_seq_length - prompt_len - yminus_len_cropped - slack_len
        yplus_cropped_tok = yplus_tokens[:yplus_len_necessary]
        example['chosen'][-1]['content'] = tokenizer.decode(yplus_cropped_tok, skip_special_tokens=False)

    yplus_ref = encode_with_chat_template(example['chosen'], tokenizer, max_seq_length, debug_print, logger)
    yminus_ref = encode_with_chat_template(example['rejected'], tokenizer, max_seq_length, debug_print, logger)

    yplus_att = apply_att_template(example, tokenizer, max_seq_length, debug_print, logger)
    yminus_att = apply_att_template({'chosen': example['rejected'], 'rejected': example['chosen']}, tokenizer,
                                    max_seq_length, debug_print, logger)

    assert yplus_att['input_ids'].shape[
               0] <= max_seq_length, f"yplus_att is too long: {yplus_att['input_ids'].shape[0]}"
    assert yminus_att['input_ids'].shape[
               0] <= max_seq_length, f"yminus_att is too long: {yminus_att['input_ids'].shape[0]}"
    assert yplus_ref['input_ids'].shape[
               0] <= max_seq_length, f"yplus_ref is too long: {yplus_ref['input_ids'].shape[0]}"
    assert yminus_ref['input_ids'].shape[
               0] <= max_seq_length, f"yminus_ref is too long: {yminus_ref['input_ids'].shape[0]}"

    return {
        "yplus_att": yplus_att,
        "yminus_att": yminus_att,
        "yplus_ref": yplus_ref,
        "yminus_ref": yminus_ref,
    }


# Old cropping strategy for symmetric ATT
# def _crop(inputs, max_len):
#     inputs['input_ids'] = inputs['input_ids'][:max_len]
#     inputs['labels'] = inputs['labels'][:max_len]
#     inputs['attention_mask'] = inputs['attention_mask'][:max_len]
#     return inputs
#
#
# # We want to preserve the response length in yplus for the ATT model and for the reference model.
# # This function computes the response length (number of predicted tokens) after cropping
# def _cropped_resp_len(labels, max_len):
#     len_prompt = (labels == -100).sum().item()
#     len_resp = (labels != -100).sum().item()
#     len_resp_cropped = max(0, min(len_resp, max_len - len_prompt))
#     return len_prompt, len_resp_cropped
#
#
# def _crop_to_min(inputs1, inputs2, max_len):
#     len_prompt1, len_resp1 = _cropped_resp_len(inputs1['labels'], max_len)
#     len_prompt2, len_resp2 = _cropped_resp_len(inputs2['labels'], max_len)
#     min_len = min(len_resp1, len_resp2)
#     inputs1 = _crop(inputs1, min(len_prompt1 + min_len, max_len))
#     inputs2 = _crop(inputs2, min(len_prompt2 + min_len, max_len))
#     return inputs1, inputs2

# def preprocess_for_symmetric_att(example, tokenizer, max_seq_length, debug_print=False, logger=None):
#     yplus_att = apply_att_template(example, tokenizer, max_seq_length, debug_print, logger)
#
#     backward_example = {
#         'chosen': example['rejected'],
#         'rejected': example['chosen'],
#     }
#     yminus_att = apply_att_template(backward_example, tokenizer, max_seq_length, debug_print, logger)
#
#     yplus_ref = encode_with_chat_template(example['chosen'], tokenizer, max_seq_length, debug_print, logger)
#     yminus_ref = encode_with_chat_template(example['rejected'], tokenizer, max_seq_length, debug_print, logger)
#
#     # This makes sure that the following conditions are met:
#     # 1. Total length of the prompt + response is <= max_seq_length for all four inputs
#     # 2. The response length is the same for the ATT and reference models on yplus (yplus_att and yplus_ref)
#     # 3. The response length is the same for the ATT and reference models on yminus (yminus_att and yminus_ref)
#     yplus_att, yplus_ref = _crop_to_min(yplus_att, yplus_ref, max_seq_length)
#     yminus_att, yminus_ref = _crop_to_min(yminus_att, yminus_ref, max_seq_length)
#
#     return {
#         "yplus_att": yplus_att,
#         "yminus_att": yminus_att,
#         "yplus_ref": yplus_ref,
#         "yminus_ref": yminus_ref,
#     }


def has_responses(processed_example):
    return processed_example is not None and all([(e["labels"] != -100).any() for e in processed_example.values()])


def neg_crossentropy(outputs, labels, reduce_loss='sum'):
    if reduce_loss == 'mean':
        loss = outputs.loss
    elif reduce_loss == 'sum':
        # reduce loss is sum
        # this ensures that we weight all tokens in the dataset equally,
        # rather than weighting each overall example equally when
        # using high amounts of gradient accumulation.
        # this can result in > 5 point improvements in AlpacaEval
        # see https://github.com/huggingface/transformers/issues/24725 for
        # more discussion and details.
        num_labels = (labels != -100).sum()
        loss = outputs.loss * num_labels

        # logits = outputs.logits
        # # Shift so that tokens < n predict n
        # shift_logits = logits[..., :-1, :].contiguous()
        # shift_labels = labels[..., 1:].contiguous()
        # # Flatten the tokens
        # loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
        # shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        # shift_labels = shift_labels.view(-1)
        # # Enable model parallelism
        # shift_labels = shift_labels.to(shift_logits.device)
        # loss = loss_fct(shift_logits, shift_labels)
        # print(f"loss: {loss}, alt_loss: {alt_loss} rel_err = {(loss - alt_loss) / alt_loss}")

    else:
        raise ValueError(f"Unknown reduce_loss value: {reduce_loss}")
    return loss


def compute_loss_att(accelerator, model, batch, args, eval=False, debug=False):
    def log_neg_ce(value, labels, name):
        num_predicted = (labels != -100).sum().item()
        if args.reduce_loss == 'mean':
            return {name + "_avg": value.item(), name + "_sum": value.item() * num_predicted}
        elif args.reduce_loss == 'sum':
            return {name + "_sum": value.item(), name + "_avg": value.item() / num_predicted}

    logs = {}
    with torch.set_grad_enabled(not eval):
        yplus_outputs = model(**batch["yplus_att"])
        yplus_neg_ce = neg_crossentropy(yplus_outputs, batch["yplus_att"]["labels"], args.reduce_loss)
    logs = {**logs, **log_neg_ce(yplus_neg_ce, batch["yplus_att"]["labels"], "mlog_pi_t_yplus")}

    if args.loss == "ce":
        return yplus_neg_ce, logs

    with torch.set_grad_enabled(not eval):
        yminus_outputs = model(**batch["yminus_att"])
        yminus_neg_ce = neg_crossentropy(yminus_outputs, batch["yminus_att"]["labels"], args.reduce_loss)
    logs = {**logs, **log_neg_ce(yminus_neg_ce, batch["yminus_att"]["labels"], "mlog_pi_t_yminus")}

    if args.loss == "symmetric":
        # Loss1
        diff = yplus_neg_ce.detach() - args.neg_example_strength * yminus_neg_ce
        logs["log_pi_t_diff"] = diff.item()
        return yplus_neg_ce + args.loss_lambda * F.softplus(-diff), logs
    elif args.loss == "symmetric_hinge":
        # Loss2
        diff = yplus_neg_ce.detach() - args.neg_example_strength * yminus_neg_ce
        logs["log_pi_t_diff"] = diff.item()
        return yplus_neg_ce + args.loss_lambda * torch.relu(args.hinge_delta - diff), logs
    elif args.loss in ["symmetric_dpo", "ipo"]:
        # assert isinstance(model, )
        yplus_ref_outputs = None
        yminus_ref_outputs = None
        with torch.no_grad():
            if args.use_lora:
                if debug:
                    yplus_nodisable_outputs = model(**batch["yplus_ref"])
                    yminus_nodisable_outputs = model(**batch["yminus_ref"])
                    yplus_nodisable_ce = neg_crossentropy(yplus_nodisable_outputs, batch["yplus_ref"]["labels"],
                                                          args.reduce_loss)
                    yminus_nodisable_ce = neg_crossentropy(yminus_nodisable_outputs, batch["yminus_ref"]["labels"],
                                                           args.reduce_loss)
                    logs = {**logs,
                            **log_neg_ce(yplus_nodisable_ce, batch["yplus_ref"]["labels"], "mlog_pi_yplus_nodisable")}
                    logs = {**logs, **log_neg_ce(yminus_nodisable_ce, batch["yminus_ref"]["labels"],
                                                 "mlog_pi_yminus_nodisable")}
                with accelerator.unwrap_model(model).disable_adapter():
                    yplus_ref_outputs = model(**batch["yplus_ref"])
                    yminus_ref_outputs = model(**batch["yminus_ref"])
            else:
                assert False, "Not implemented"
                # reference_chosen_logps, reference_rejected_logps = concatenated_forward(reference_model, batch)

            yplus_ref_ce = neg_crossentropy(yplus_ref_outputs, batch["yplus_ref"]["labels"], args.reduce_loss)
            logs = {**logs, **log_neg_ce(yplus_ref_ce, batch["yplus_ref"]["labels"], "mlog_pi_ref_yplus")}
            yminus_ref_ce = neg_crossentropy(yminus_ref_outputs, batch["yminus_ref"]["labels"], args.reduce_loss)
            logs = {**logs, **log_neg_ce(yminus_ref_ce, batch["yminus_ref"]["labels"], "mlog_pi_ref_yminus")}

        diff = yplus_neg_ce - yplus_ref_ce \
               - args.neg_example_strength * (yminus_neg_ce - yminus_ref_ce)
        logs["log_pi_t_diff_dpo"] = diff.item()

        if args.loss == "symmetric_dpo":
            return F.softplus(-args.dpo_beta * diff), logs
        elif args.loss == "ipo":
            return (diff - 1 / (2 * args.dpo_beta)) ** 2, logs
    else:
        raise ValueError(f"Unknown loss type: {args.loss}")

# pos_inputs = {
# outputs_pos =
