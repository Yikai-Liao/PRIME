import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import re
import time
import openai
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset
from utils.data import write_jsonl, read_problems
from utils.evaluation import evaluate_functional_correctness
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=10)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test", action="store_true", help="Test mode: evaluate only first 10 samples")
args = parser.parse_args()
data_path = os.path.join(args.data_dir, "HumanEval.jsonl.gz")
problems = read_problems(data_path)

# Apply test mode limitation
if args.test:
    problem_ids = sorted(problems.keys())[:10]
    problems = {pid: problems[pid] for pid in problem_ids}
    print(f"Test mode: Evaluating only first {len(problems)} problems")

STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)
def make_conv_hf(example, tokenizer):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    description = "\n".join(
                [
                    line.strip()
                    for line in re.search(
                        rf"(?:\"\"\"|''')(.*?)(?:\"\"\"|''')", example["prompt"], re.DOTALL
                    )
                    .group(1)
                    .split("\n")
                ]
            )
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python```\n"
                f"```python\n"
                f"{example['prompt']}\n"
                f"```\n"
            )
    system_prompt = open("system_prompt.md").read()

    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
    ]

    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def match_code(s):
    pattern = r'```python(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if 'def ' in code_block:
                return code_block
        return sol[-1]
    
    pattern = r'```(.*?)```'
    sol = re.findall(pattern, s, re.DOTALL)
    if len(sol) > 0:
        for code_block in reversed(sol):
            if 'def ' in code_block:
                return code_block
        return sol[-1]
    
    return s.split('```')[0]

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        # tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        seed=args.seed,
        
    )
    sampling_params = SamplingParams(max_tokens=32768,
                                    temperature=args.temperature,
                                    n=args.num_samples_per_task,
                                    stop=["<|eot_id|>","<|im_end|>"],
                                    seed=args.seed)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)

    completions = []
    for output in outputs:
        for i in range(args.num_samples_per_task):
            completions.append(match_code(output.outputs[i].text))
    return completions

def make_signature(example):
    signature = re.search(
                rf"def\s+({example['entry_point']}.*?):\s*\n", example["prompt"]
            ).group(1)
    return signature

samples = []
problems_df = pd.DataFrame(problems).T
problems_dataset = Dataset.from_pandas(problems_df)
problems_dataset = problems_dataset.map(lambda x: {"signature": make_signature(x)}, cache_file_name="cache/human_eval", load_from_cache_file=False)
problems_dataset = problems_dataset.map(lambda x: {"instruction": make_conv_hf(x, tokenizer)}, cache_file_name="cache/human_eval", load_from_cache_file=False)

completions = generate_sample_batch(problems_dataset["instruction"])

# Create samples for each completion of each problem
for i, problem in enumerate(problems_dataset):
    for j in range(args.num_samples_per_task):
        completion_idx = i * args.num_samples_per_task + j
        sample = {
            "task_id": problem["task_id"],
            "prompt": problem["prompt"],
            "canonical_solution": problem["canonical_solution"],
            "test": problem["test"],
            "entry_point": problem["entry_point"],
            "completion": problem["prompt"] + completions[completion_idx]
        }
        samples.append(sample)

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)
if not os.path.exists(os.path.join(args.save_dir)):
    os.mkdir(os.path.join(args.save_dir))
write_jsonl(output_filepath, samples)

k_values = list(range(1, args.num_samples_per_task + 1))  # k from 1 to n
# In test mode, pass the filtered problems to evaluation
if args.test:
    # Create a temporary problem file with only the test problems
    import tempfile
    import gzip
    import json
    
    temp_problems_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl.gz', delete=False)
    with gzip.open(temp_problems_file.name, 'wt') as f:
        for problem_id, problem in problems.items():
            f.write(json.dumps(problem) + '\n')
    
    score = evaluate_functional_correctness(sample_file=output_filepath, k=k_values, problem_file=temp_problems_file.name)
    
    # Clean up temp file
    import os
    os.unlink(temp_problems_file.name)
else:
    score = evaluate_functional_correctness(sample_file=output_filepath, k=k_values)
print(score)

# Save detailed results
score_path = os.path.join(args.save_dir, "result.txt")
with open(score_path, "w") as f:
    f.write(str(score))

# Save pass@k results for curve plotting
import json
pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
with open(pass_at_k_path, "w") as f:
    json.dump(score, f, indent=2)