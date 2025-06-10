import json
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import os
import openai
import argparse
from vllm import LLM, SamplingParams
import time
import re
import contextlib
import signal
import pickle
from transformers import AutoTokenizer
from collections import defaultdict
import numpy as np
from math import comb

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
        max_model_len=32768,
        max_num_seqs=32,
        seed=args.seed,
    )
    sampling_params = SamplingParams(max_tokens=32768,
                                    temperature=args.temperature,
                                    n=args.num_samples_per_task,
                                    stop=STOP_WORDS,
                                    seed=args.seed)
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

class TimeoutException(Exception):
    pass

@contextlib.contextmanager
def time_limit(seconds: float = 5):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def exec_helper(code, namespace=None):
    if namespace is None:
        namespace = {}
    with time_limit(5):
        exec(compile(code, filename="mbpp", mode='exec'), namespace)
    return namespace

def evaluate(dataset):
    correct = 0
    format_error = 0
    exec_error = 0
    for example in dataset.to_dict(orient="records"):
        completion = example["completion"]
        # 清理代码
        code_ = []
        for c in completion.split("\n"):
            if len(c.lstrip()) == len(c) and not c.startswith("def"):
                continue
            code_.append(c)
        function = "\n".join(code_)
        test_cases = "\n".join(example["test_list"]).replace("\/", "/")
        test_run = "\n".join([function, test_cases])

        namespace = {}
        try:
            exec_helper(function, namespace)
        except Exception as e:
            format_error += 1
            continue

        try:
            exec_helper(test_cases, namespace)
            exec_helper(test_run, namespace)
        except:
            exec_error += 1
            continue
        else:
            correct += 1
    total = len(dataset)
    return 100 * (correct / total), 100 * (exec_error / total), 100 * (format_error / total)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--input_data", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="")
    parser.add_argument("--num-samples-per-task", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true", help="Test mode: evaluate only first 10 samples")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = pd.read_json(args.input_data, lines=False)
    
    # Apply test mode limitation
    if args.test:
        dataset = dataset.head(10)
        print(f"Test mode: Evaluating only first {len(dataset)} problems")
    
    dataset["signature"] = dataset.apply(lambda row: make_signature(row["code"]), axis=1)
    for signature in dataset["signature"]:
        STOP_WORDS.append("\n\nprint(" + signature.split("(")[0].strip())
    dataset["prompt"] = dataset.apply(lambda row: make_conv_hf(row["signature"], row["prompt"], row["test_list"], tokenizer), axis=1)

    completions, raw_completions = generate_sample_batch(dataset["prompt"].tolist())
    with open(os.path.join(args.save_dir, "completions.pkl"), "wb") as f:
        pickle.dump(completions, f)
    with open(os.path.join(args.save_dir, "raw_completions.pkl"), "wb") as f:
        pickle.dump(raw_completions, f)

    # 加载 checkpoint
    with open(os.path.join(args.save_dir, "completions.pkl"), "rb") as f:
        completions = pickle.load(f)
    with open(os.path.join(args.save_dir, "raw_completions.pkl"), "rb") as f:
        raw_completions = pickle.load(f)

    expanded_data = []
    for i, row in dataset.iterrows():
        for j in range(args.num_samples_per_task):
            completion_idx = i * args.num_samples_per_task + j
            new_row = row.copy()
            new_row["task_id"] = f"{row['task_id']}_{j}" if 'task_id' in row else f"{i}_{j}"
            new_row["raw_completion"] = raw_completions[completion_idx]
            new_row["completion"] = completions[completion_idx]
            expanded_data.append(new_row)
    
    expanded_dataset = pd.DataFrame(expanded_data)
    if "source_file" in expanded_dataset.columns:
        del expanded_dataset["source_file"]
    
    expanded_dataset = expanded_dataset.reset_index(drop=True)
    expanded_dataset.to_json(os.path.join(args.save_dir, "mbpp_completion.json"), orient='records', indent=2)
    
    # 计算 pass@k 指标
    task_groups = defaultdict(list)
    for i, row in expanded_dataset.iterrows():
        original_task_id = int(row["task_id"].split("_")[0]) if isinstance(row["task_id"], str) else i // args.num_samples_per_task
        task_groups[original_task_id].append(row)
    
    task_results = {}
    for task_id, task_rows in task_groups.items():
        task_df = pd.DataFrame(task_rows).reset_index(drop=True)
        correct_count = 0
        for _, row in task_df.iterrows():
            temp_df = pd.DataFrame([row])
            accuracy, _, _ = evaluate(temp_df)
            if accuracy == 100:  # 该样本正确
                correct_count += 1
        task_results[task_id] = {
            'total': len(task_df),
            'correct': correct_count
        }
    
    # 计算 pass@k
    def estimate_pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    k_values = list(range(1, args.num_samples_per_task + 1))
    pass_at_k_results = {}
    
    for k in k_values:
        pass_at_k_scores = []
        for task_id, results in task_results.items():
            n = results['total']
            c = results['correct']
            if n >= k:
                pass_at_k_scores.append(estimate_pass_at_k(n, c, k))
        
        if pass_at_k_scores:
            pass_at_k_results[f"pass@{k}"] = np.mean(pass_at_k_scores)
    
    # 计算 legacy metrics（使用第一个样本）
    first_samples = []
    for task_id in range(len(dataset)):
        first_sample_idx = task_id * args.num_samples_per_task
        if first_sample_idx < len(expanded_dataset):
            first_samples.append(expanded_dataset.iloc[first_sample_idx])
    
    first_samples_df = pd.DataFrame(first_samples).reset_index(drop=True)
    legacy_accuracy, legacy_exec_error, legacy_format_error = evaluate(first_samples_df)
    
    # 打印结果
    print("Legacy metrics (using first sample):")
    print({"accuracy": legacy_accuracy, "exec_error": legacy_exec_error, "format_error": legacy_format_error})
    print("\npass@k results:")
    print(pass_at_k_results)
    
    # 保存结果
    with open(os.path.join(args.save_dir, "result.txt"), "w") as f:
        print("Legacy metrics:", file=f)
        print({"accuracy": legacy_accuracy, "exec_error": legacy_exec_error, "format_error": legacy_format_error}, file=f)
        print("\npass@k results:", file=f)
        print(pass_at_k_results, file=f)
        print("\nPer-task results:", file=f)
        for task_id, results in sorted(task_results.items()):
            print(f"Task {task_id}: {results}", file=f)
    
    # 保存 pass@k 结果以供曲线图
    pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
    with open(pass_at_k_path, "w") as f:
        json.dump(pass_at_k_results, f, indent=2)