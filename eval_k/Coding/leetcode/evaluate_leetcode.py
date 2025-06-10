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
import json
from transformers import AutoTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--input_data", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=10)
parser.add_argument("--temperature", type=float, default=0.8)
args = parser.parse_args()

problems = pd.read_json(args.input_data, lines=True)
STOP_WORDS =["\nassert", "assert"]

from vllm import LLM, SamplingParams
import torch

import re
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
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.80,
    )
    sampling_params = SamplingParams(max_tokens=32768,
                                    temperature=args.temperature,
                                    n=args.num_samples_per_task,
                                    stop=["<|eot_id|>"],)
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    
    completions = []
    raw_outputs = []
    for output in outputs:
        for i in range(args.num_samples_per_task):
            completion_text = "```python\n" + match_code(output.outputs[i].text) + "\n```"
            completions.append(completion_text)
            raw_outputs.append(output.outputs[i].text)
    
    return completions, raw_outputs


def make_conv_hf(example, tokenizer):
    system_prompt = open("system_prompt.md").read()
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example["prompt_sft"] + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat 

samples = []
del problems["start_time"]

tokenizer = AutoTokenizer.from_pretrained(args.model)
problems["instruction"] = problems.apply(lambda row: make_conv_hf(row, tokenizer), axis=1)


completions, raw_outputs = generate_sample_batch(problems["instruction"])

# Process completions to ensure class Solution format
for i in range(len(completions)):
    if "class Solution:" not in completions[i]:
        completions[i] = completions[i].replace("```python", "").replace("```", "")
        completions[i] = "\n".join(["    " + line for line in completions[i].split("\n")])
        completions[i] = "class Solution:\n" + completions[i]
        completions[i] = "```python\n" + completions[i] + "\n```"

# Create expanded dataset with multiple samples per task
expanded_data = []
for i, row in problems.iterrows():
    for j in range(args.num_samples_per_task):
        completion_idx = i * args.num_samples_per_task + j
        new_row = row.copy()
        new_row["task_id"] = f"{row.get('task_id', i)}_{j}"
        new_row["output"] = completions[completion_idx]
        new_row["raw_output"] = raw_outputs[completion_idx]
        expanded_data.append(new_row)

samples = expanded_data
output_filepath = os.path.join(args.save_dir, "samples.jsonl")
write_jsonl(output_filepath, samples)


# eval
import re
import json
from pathlib import Path
from collections import defaultdict
from utils.evaluation_leetcode import evaluate_functional_correctness

version = "20240121-Jul"

DATA_DIR = Path(__file__).parent / "data"

def extract_python_code(generation: str):
    generation = generation.replace("[PYTHON]", '```python').replace("[/PYTHON]", '```')
    if '```python' in generation:
        p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
        code_block = p_code.findall(generation)[0]
        return code_block
    else:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", generation)
        return codelist[0]
    
def evaluate_main(generation_path: str, result_path: str, temp_dir: str):
    problem_path = args.input_data
    print(problem_path)
    problems = [json.loads(line) for line in open(problem_path, 'r',encoding='utf-8')]

    id2problems = { x['task_id']: x for x in problems }

    results = [json.loads(line) for line in open(generation_path, 'r')]
    for result in results:
        if 'task_id' not in result:
            result['task_id'] = problems[result['index']]['task_id']

        if 'generation' not in result:
            try:
                if 'output' not in result:
                    result['output'] = result['response']
                if result['output'].startswith("\n        "):
                    func_code = extract_python_code(result['prompt_sft']).strip()
                    result['generation'] = func_code + '\n' + result['output']
                else:
                    result['generation'] = extract_python_code(result['output'])
            except:
                result['generation'] = result['output']
    
    with open(result_path, 'w') as fr:
        for result in results:
            fr.write(json.dumps(result) + "\n")

    # Calculate pass@k metrics
    from collections import defaultdict
    from math import comb
    import numpy as np
    
    def estimate_pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    # Group results by original task_id
    task_results = defaultdict(list)
    
    # First run evaluation on all samples
    score = evaluate_functional_correctness(
        input_file=result_path,
        tmp_dir=temp_dir,
        problem_file=problem_path,
        result_path=result_path
    )
    
    # Read evaluation results and group by original task
    eval_results = [json.loads(line) for line in open(result_path, 'r')]
    for result in eval_results:
        original_task_id = result['task_id'].split('_')[0] if '_' in str(result['task_id']) else result['task_id']
        task_results[original_task_id].append(result['passed'])
    
    # Calculate pass@k for different k values
    k_values = list(range(1, args.num_samples_per_task + 1))
    pass_at_k_results = {}
    
    for k in k_values:
        pass_at_k_scores = []
        for task_id, results_list in task_results.items():
            n = len(results_list)
            c = sum(results_list)
            if n >= k:
                pass_at_k_scores.append(estimate_pass_at_k(n, c, k))
        
        if pass_at_k_scores:
            pass_at_k_results[f"pass@{k}"] = np.mean(pass_at_k_scores)
    
    # Calculate hardness results using first sample of each task for legacy comparison
    hardness_results = defaultdict(int)
    for result in eval_results:
        if '_0' in str(result['task_id']) or '_' not in str(result['task_id']):  # Only count first sample
            original_task_id = result['task_id'].split('_')[0] if '_' in str(result['task_id']) else result['task_id']
            if original_task_id in id2problems:
                problem = id2problems[original_task_id]
                hardness = problem['meta']['difficulty']
                hardness_results[hardness] += 1
                hardness_results[hardness + "_correct"] += result['passed']

    print("="*100)
    print("Evaluate {} over.".format(generation_path))
    print("Legacy Pass@1: {:.3f}".format(score.get("pass@1", 0)))
    print("\npass@k results:")
    for k, v in pass_at_k_results.items():
        print(f"{k}: {v:.3f}")
    
    for key in ["Easy", "Medium", "Hard"]:
        if key.endswith("_correct") or hardness_results[key] == 0:
            continue
        acc = hardness_results[key+"_correct"] / hardness_results[key]
        print("{}: {:.3f}({}/{})".format(key, acc, hardness_results[key+"_correct"],  hardness_results[key]))
    
    score_path = os.path.join(args.save_dir, "result.txt")
    with open(score_path, "w") as f:
        f.write("Legacy Pass@1: {:.3f}\n".format(score.get("pass@1", 0)))
        f.write("\npass@k results:\n")
        for k, v in pass_at_k_results.items():
            f.write(f"{k}: {v:.3f}\n")
        for key in ["Easy", "Medium", "Hard"]:
            if key.endswith("_correct") or hardness_results[key] == 0:
                continue
            acc = hardness_results[key+"_correct"] / hardness_results[key]
            f.write("{}: {:.3f}({}/{})\n".format(key, acc, hardness_results[key+"_correct"],  hardness_results[key]))
    
    # Save pass@k results for curve plotting
    pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
    with open(pass_at_k_path, "w") as f:
        json.dump(pass_at_k_results, f, indent=2)
result_path = output_filepath.replace(".jsonl", "_result.jsonl")
if not os.path.exists("output"):
    os.makedirs("output")
if not os.path.exists("output/temp"):
    os.makedirs("output/temp")
evaluate_main(output_filepath, result_path, temp_dir="output/temp")
