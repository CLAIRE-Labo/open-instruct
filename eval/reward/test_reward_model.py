import sys
from copy import deepcopy
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from run_eval_att import ArmoRMPipeline


reward_model = ArmoRMPipeline("RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True)

prompt_chats = [
    {"role": "user", "content": "I am looking for a restaurant in San Francisco."},
    {"role": "user", "content": "Need a recommendation for a good book."},
    {"role": "user", "content": "What is the derivative of log(x)?"},
    {"role": "user", "content": "How to learn to drive?"},
]

good_responses = [
    {"role": "assistant", "content": "I recommend going to the House of Prime Rib."},
    {"role": "assistant", "content": "I recommend reading 'Brothers Karamazov'."},
    {"role": "assistant", "content": "The derivative of log(x) is 1/x."},
    {"role": "assistant", "content": "Start by going to a driving school."},
]

bad_responses = [
    {"role": "assistant", "content": "I don't know."},
    {"role": "assistant", "content": "Go watch a movie."},
    {"role": "assistant", "content": "It is 2 * x"},
    {"role": "assistant", "content": "To cook, you need a spoon."},
]

sample_good_chats = [deepcopy(prompt_chats) + [response] for response in good_responses]
sample_bad_chats = [deepcopy(prompt_chats) + [response] for response in bad_responses]

good_scores = [reward_model(chat)['score'] for chat in sample_good_chats]
bad_scores = [reward_model(chat)['score'] for chat in sample_bad_chats]

print("Good scores:", good_scores)
print("Bad scores:", bad_scores)