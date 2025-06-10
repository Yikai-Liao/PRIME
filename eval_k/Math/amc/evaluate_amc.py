# Adapt from https://github.com/hendrycks/math/blob/main/modeling/evaluate_gpt3.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import time
import traceback
import openai
import argparse
import numpy as np
import operator
import json
import tqdm
import pandas as pd
from utils.util import clean_numbers, last_boxed_only, last_boxed_only_string
from utils.math_equivalence import is_equiv
from utils.grader import math_equal
from collections import defaultdict
from vllm import LLM, SamplingParams
import torch
import re
import math
from transformers import AutoTokenizer

os.environ["NCCL_IGNORE_DISABLED_P2P"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.90,
        dtype='bfloat16',
        enforce_eager=True,
        skip_tokenizer_init=False,
    )
    sampling_params = SamplingParams(max_tokens=32768,
                                     temperature=args.temperature,
                                     n=args.num_samples_per_task,
                                     stop=[], )
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)
    completions = []
    for output in outputs:
        for i in range(args.num_samples_per_task):
            completions.append(output.outputs[i].text)
    return completions


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None


def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1: right_brace_idx].strip()


def match_answer(response):
    is_matched = False
    ans_marker = 'The answer is: '
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]
            
    ans_marker = 'answer:\n'
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]

    ans_marker = 'answer: '
    ans_idx = response.lower().rfind(ans_marker)
    if ans_idx != -1:
        is_matched = True
        response = response[ans_idx + len(ans_marker):].strip()
        if response.endswith("\n"):
            response = response[:-2]

    # Find boxed
    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    # Grade
    return is_matched, response


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer_hf(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return eval(match_str)
    else:
        return INVALID_ANS





def make_conv_hf(question, tokenizer):
    system_prompt = open("system_prompt.md").read()
    content = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content}
    ]
    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat


def run(args, max=-1):
    all_problems = read_jsonl_file(os.path.join(args.data_dir, "aimo-validation-amc.jsonl"))
    print("reading problems done!")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    completions = generate_sample_batch(
        [make_conv_hf(problem_data["question"], tokenizer) for problem_data in all_problems])
    
    # Create expanded dataset with multiple samples per task
    tmp_data = []
    for i, problem_data in enumerate(all_problems):
        for j in range(args.num_samples_per_task):
            completion_idx = i * args.num_samples_per_task + j
            new_problem = problem_data.copy()
            new_problem["task_id"] = f"{i}_{j}"
            new_problem["completion"] = completions[completion_idx]
            tmp_data.append(new_problem)
    write_jsonl_file(os.path.join(args.save_dir, "completions.jsonl"), tmp_data)

    # Calculate pass@k metrics
    from collections import defaultdict
    from math import comb
    import numpy as np
    
    def estimate_pass_at_k(n, c, k):
        if n - c < k:
            return 1.0
        return 1.0 - comb(n - c, k) / comb(n, k)
    
    task_results = defaultdict(list)
    total = len(all_problems)
    correct = 0
    save_data = []
    
    for i, (problem_data, model_output) in enumerate(zip(tmp_data, [item["completion"] for item in tmp_data])):
        original_task_id = int(problem_data["task_id"].split("_")[0])
        original_problem = all_problems[original_task_id]

        answer = str(original_problem["answer"])
        problem_data["completion"] = model_output
        is_matched, model_output = match_answer(model_output)
        model_output = model_output.strip("The final answer is ").strip(". I hope it is correct.")
        try:
            if "\pi" in model_output or "\pi" in answer:
                equivs = []
                for pi in [math.pi, 3.14]:
                    equivs.append(math_equal(model_output, answer, timeout=True, pi=pi))
                equiv = any(equivs)
            else:
                equiv = math_equal(model_output, answer, timeout=True)
        except:
            equiv = False

        # Store result for pass@k calculation
        task_results[original_task_id].append(equiv)
        
        # Only save first sample of each task for compatibility
        if "_0" in problem_data["task_id"]:
            if equiv:
                correct += 1
            problem_data["success"] = equiv
            save_data.append(problem_data)

    print("##########AMC")
    print(f"total: {total}, success: {correct}, rate: {correct / total}")
    comp_name = []
    for line in save_data:
        comp_name.append(line["url"].split("/")[-2])
    comp_name = list(set(comp_name))
    dic = {}
    for line in comp_name:
        dic[line] = {}
        dic[line]["total"] = 0
        dic[line]["success"] = 0
    for line in save_data:
        dic[line["url"].split("/")[-2]]["total"] += 1
        if line["success"]:
            dic[line["url"].split("/")[-2]]["success"] += 1
    print(json.dumps(dic, indent=4))
    
    # Calculate pass@k metrics
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
    
    print("\npass@k results:")
    for k, v in pass_at_k_results.items():
        print(f"{k}: {v:.3f}")

    output_file = os.path.join(args.save_dir, "results_total.txt")
    with open(output_file, "w+") as f:
        f.write(f"total: {total}, success: {correct}, rate: {correct / total}")
        f.write("\n\npass@k results:\n")
        for k, v in pass_at_k_results.items():
            f.write(f"{k}: {v:.3f}\n")
    
    # Save pass@k results for curve plotting
    pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
    with open(pass_at_k_path, "w") as f:
        json.dump(pass_at_k_results, f, indent=2)
    output_file = os.path.join(args.save_dir, "results_split.txt")
    with open(output_file, "w+") as f:
        f.write(json.dumps(dic, indent=4))
    write_jsonl_file(os.path.join(args.save_dir, "results.json"), save_data)
    import pandas as pd
    df = pd.DataFrame(save_data)
    df.to_excel(os.path.join(args.save_dir, "results.xlsx"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="")
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--num-samples-per-task", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    run(args)
