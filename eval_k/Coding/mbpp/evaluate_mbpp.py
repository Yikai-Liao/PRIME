import json
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import os
import torch
import openai
import argparse
from vllm import LLM, SamplingParams
import time
import re

import sys
# sys.path.append("./scripts/eval")  # Commented out - this path doesn't exist

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

STOP_WORDS = []
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
        gpu_memory_utilization=0.9,
        max_model_len=8192,  # Limit context length
        max_num_seqs=32,
    )
    sampling_params = SamplingParams(max_tokens=4096,  # Reduced from 4096
                                    temperature=0.8,
                                    n=args.num_samples_per_task,
                                    stop=STOP_WORDS)
    print(f"Generating {len(question_list)} samples with {args.num_samples_per_task} completions each...")
    try:
        outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    except Exception as e:
        print(f"Error during generation: {e}")
        raise
    raw_completions = []
    completions = []
    for output in outputs:
        for i in range(args.num_samples_per_task):
            raw_completions.append(output.outputs[i].text)
            completions.append(match_code(output.outputs[i].text))
    return completions,raw_completions


def make_signature(code):
    signature = [line for line in code.split("\n") if line.strip().startswith("def ")][0]
    signature = signature.lstrip("def ").replace(" ", "").rstrip(":").strip().replace(",", ", ")
    assert ":" not in signature
    return signature


from transformers import AutoTokenizer
def make_conv_hf(signature, description, test_list, tokenizer):
    description = description.split(" https://www.")[0]
    testcase = test_list[0]
    prompt = (
                f"Write Python code to solve the task.\n"
                f"Write a Python function `{signature}` to solve the following problem: Present code in ```python and ```\n"
                f"{description}\n"
                f"The code should pass the following test cases:>>> {testcase}\n\n Let's coding step by step.\n"
            )
    system_prompt = open("system_prompt.md").read()
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + "\n\nWrite Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end."}
    ]

    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

import contextlib
import signal
class TimeoutException(Exception):
    pass
@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def exec_helper(code):
    with time_limit(3):
        exec(compile(code, filename="mbpp", mode='exec'), globals())

def evaluate(dataset):
    correct = 0
    format_error = 0
    exec_error = 0

    for example in dataset.to_dict(orient="records"):
        completion = example["completion"]
        # remove texts
        code = completion.split("\n")
        code_ = []
        for c in code:
            if len(c.lstrip()) == len(c) and not c.startswith("def"):
                continue
            code_.append(c)
        code = "\n".join(code_)

        function = code
        test_cases = "\n".join(example["test_list"]).replace("\/", "/")
        test_run = "\n".join([
            function,
            test_cases,
        ])

        # define function
        try:
            exec_helper(function)
        except Exception as e:
            print(function)
            print("Error",e)
            format_error += 1
            continue           

        try:
            # run test case
            exec_helper(test_cases)
            exec_helper(test_run)
        except:
            exec_error += 1
            continue
        else:
            correct += 1
    print("correct: ", correct)
    print("exec_error: ", exec_error)
    print("format_error: ", format_error)
    return 100 * (correct / len(dataset)), 100 * (exec_error / len(dataset)), 100 * (format_error / len(dataset))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--input_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num-samples-per-task", type=int, default=10)

    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = pd.read_json(args.input_data, lines=False)
    dataset["signature"] = dataset.apply(lambda row: make_signature(row["code"]), axis=1)
    for signature in dataset["signature"]:
        STOP_WORDS.append("\n\nprint(" + signature.split("(")[0].strip())
    dataset["prompt"] = dataset.apply(lambda row: make_conv_hf(row["signature"], row["prompt"], row["test_list"], tokenizer), axis=1)
    # Use full dataset instead of just first 16 samples
    # dataset = dataset[:16]  # Remove this limitation
    completions,raw_completions = generate_sample_batch(dataset["prompt"].tolist())
    
    # Create expanded dataset with multiple samples per task
    expanded_data = []
    for i, row in dataset.iterrows():
        for j in range(args.num_samples_per_task):
            completion_idx = i * args.num_samples_per_task + j
            new_row = row.copy()
            new_row["task_id"] = f"{row['task_id']}_{j}" if 'task_id' in row else f"{i}_{j}"
            new_row["raw_completion"] = raw_completions[completion_idx]
            new_row["completion"] = completions[completion_idx]
            new_row["completion"] = "def" + new_row["completion"] if "def" not in new_row["completion"] else new_row["completion"]
            expanded_data.append(new_row)
    
    expanded_dataset = pd.DataFrame(expanded_data)
    if "source_file" in expanded_dataset.columns:
        del expanded_dataset["source_file"]
    
    # Reset index to avoid duplicate index error when saving to JSON
    expanded_dataset = expanded_dataset.reset_index(drop=True)
    expanded_dataset.to_json(os.path.join(args.save_dir, "mbpp_completion.json"), orient='records', indent=2)
    
    # Calculate pass@k metrics
    from collections import defaultdict
    task_results = defaultdict(list)
    
    for i, example in enumerate(expanded_dataset.to_dict(orient="records")):
        original_task_id = int(example["task_id"].split("_")[0]) if isinstance(example["task_id"], str) else example.get("task_id", i // args.num_samples_per_task)
        completion = example["completion"]
        
        # Test the completion
        is_correct = False
        try:
            code = completion.split("\n")
            code_ = []
            for c in code:
                if len(c.lstrip()) == len(c) and not c.startswith("def"):
                    continue
                code_.append(c)
            code = "\n".join(code_)
            
            function = code
            test_cases = "\n".join(example["test_list"]).replace("\/", "/")
            test_run = "\n".join([function, test_cases])
            
            exec_helper(function)
            exec_helper(test_cases)
            exec_helper(test_run)
            is_correct = True
        except:
            is_correct = False
        
        task_results[original_task_id].append(is_correct)
    
    # Calculate pass@k for different k values
    import numpy as np
    from math import comb
    
    def estimate_pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    k_values = list(range(1, min(args.num_samples_per_task + 1, 11)))
    pass_at_k_results = {}
    
    for k in k_values:
        pass_at_k_scores = []
        for task_id, results in task_results.items():
            n = len(results)
            c = sum(results)
            if n >= k:
                pass_at_k_scores.append(estimate_pass_at_k(n, c, k))
        
        if pass_at_k_scores:
            pass_at_k_results[f"pass@{k}"] = np.mean(pass_at_k_scores)
    
    # Calculate legacy metrics using the same methodology as pass@1 
    # Extract first sample for each task to match pass@1 calculation
    first_samples = []
    for task_id in range(len(dataset)):
        first_sample_idx = task_id * args.num_samples_per_task
        if first_sample_idx < len(expanded_dataset):
            first_samples.append(expanded_dataset.iloc[first_sample_idx])
    
    first_samples_df = pd.DataFrame(first_samples).reset_index(drop=True)
    accuracy, exec_error, format_error = evaluate(first_samples_df)
    
    # Print results
    print("Legacy metrics (using first sample):")
    print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error})
    print("\npass@k results:")
    print(pass_at_k_results)
    
    # Save results
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        print("Legacy metrics:", file=f)
        print({"accuracy": accuracy, "exec_error": exec_error, "format_error": format_error}, file=f)
        print("\npass@k results:", file=f)
        print(pass_at_k_results, file=f)
    
    # Save pass@k results for curve plotting
    import json
    pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
    with open(pass_at_k_path, "w") as f:
        json.dump(pass_at_k_results, f, indent=2)
