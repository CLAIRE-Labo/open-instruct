import sys
import argparse
from pathlib import Path
import pickle
from tqdm import tqdm

import vllm

sys.path.append(Path(__file__).parents[1].absolute().as_posix())
from open_instruct.load_utils import load_args, load_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("model_name_or_path", type=str)
parser.add_argument("--tokenizer_name", type=str, required=True)
parser.add_argument("--pickle_prompts", type=Path, required=True)
parser.add_argument("--pickle_sampling_params", type=Path, required=True)
parser.add_argument("--pickle_output", type=Path, required=True)
parser.add_argument("--mem_util", type=float, default=0.4)
parser.add_argument("--batch_size", type=int, default=30)
args = parser.parse_args()

# load prompts
with open(args.pickle_prompts, "rb") as f:
    prompts = pickle.load(f)

batches = [prompts[i : i + args.batch_size] for i in range(0, len(prompts), args.batch_size)]

# load sampling params
with open(args.pickle_sampling_params, "rb") as f:
    sampling_params = pickle.load(f)

model = vllm.LLM(
    model=args.model_name_or_path,
    tokenizer=args.tokenizer_name,
    tokenizer_mode="auto",
    trust_remote_code=True,
    enable_lora=False,
    disable_sliding_window=True,
    gpu_memory_utilization=args.mem_util,
)

responses = []
for batch in tqdm(batches, desc="Generating responses in batches"):
    cur_responses = model.generate(batch, sampling_params=sampling_params, use_tqdm=False)
    responses.extend(cur_responses)

responses_text = [t.outputs[0].text for t in responses]

with open(args.pickle_output, "wb") as f:
    pickle.dump(responses_text, f)