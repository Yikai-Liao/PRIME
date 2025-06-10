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
from tqdm import tqdm
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
    )
    sampling_params = SamplingParams(max_tokens=32768,
                                    temperature=args.temperature,
                                    n=args.num_samples_per_task,
                                    stop=["\n###\nProblem: ", "<|eot_id|>"],)
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

    ans_marker = 'the answer is: '
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
    outputs = []
    answers = []
    types = []
    levels = []
    matches = []
    fnames_list = []

    cors = {}
    subject_cors = {}
    level_cors = {}
    correct = 0
    total = 0

    
    all_problems = pd.read_json(os.path.join(args.data_dir, "math_test_cleaned.json")).to_dict(orient="records")
    print("reading problems done!")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    completions = generate_sample_batch([make_conv_hf(problem_data["problem"], tokenizer) for problem_data in all_problems])

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
    
    for i, (problem_data, model_output) in enumerate(tqdm(zip(tmp_data, [item["completion"] for item in tmp_data]))):
        original_task_id = int(problem_data["task_id"].split("_")[0])

        prob_level = problem_data["level"]
        prob_type = problem_data["type"]
        try:
            prob_level = int(prob_level.split("Level ")[1])
        except:
            prob_level = None

        answer = problem_data["expected_answer"]

        levels.append(prob_level)
        types.append(prob_type)
        is_matched, model_output = match_answer(model_output)
        matches.append(is_matched)
        outputs.append(model_output)
        answers.append(answer)
        
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
        fnames_list.append(equiv)
        if (prob_level, prob_type) in cors:
            cors[(prob_level, prob_type)].append(equiv)
        else:
            cors[(prob_level, prob_type)] = [equiv]
        if prob_level in level_cors:
            level_cors[prob_level].append(equiv)
        else:
            if prob_level is not None:
                level_cors[prob_level] = [equiv]
        if prob_type in subject_cors:
            subject_cors[prob_type].append(equiv)
        else:
            if prob_type is not None:
                subject_cors[prob_type] = [equiv]
        if equiv:
            correct += 1
        
        # Store result for pass@k calculation
        task_results[original_task_id].append(equiv)
    
    output_file = os.path.join(args.save_dir, "results.txt")
    
    output_dict = {
        "outputs": [],
        "accuracy_by_subject_and_level": defaultdict(list),
        "accuracy_by_level": [],
        "accuracy_by_subject": [],
    }
    print("Match rate: ", np.mean(matches))
    with open(output_file, "w+") as f:
        for k, (output, answer, prob_type, prob_level, match, equiv) in enumerate(zip(outputs, answers, types, levels, matches, fnames_list)):
            try:
                f.write("{} TYPE: {} | LEVEL: {} | OUTPUT: {} | ANSWER: {} | MATCH: {} | CORRECT: {}\n".format(k, prob_type, prob_level, output, answer, match, equiv))
            except:
                f.write("Error line436")
                pass
            output_dict["outputs"].append({
                "type": prob_type,
                "level": prob_level,
                "output": output,
                "answer": answer,
                "match": match,
                "equiv": equiv
            })
        

        f.write("#####################\n")
        # also get accuracies for each
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            for level in range(1, 6):
                key = (level, subject)
                if key not in cors.keys():
                    print("Skipping", key)
                    continue
                cors_list = cors[key]
                print("{} Level {} Accuracy = {}/{} = {:.3f}".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                f.write("{} Level {} Accuracy = {}/{} = {:.3f}\n".format(subject, level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
                
                output_dict["accuracy_by_subject_and_level"][subject].append({
                    "level": level,
                    "num_correct": np.sum(cors_list),
                    "num_total": len(cors_list),
                    "accuracy": np.mean(cors_list)
                })

        print("#####################")
        f.write("#####################\n")
        for level in sorted(level_cors):
            if level not in level_cors.keys():
                print("Skipping", level)
                continue
            cors_list = level_cors[level]
            print("Level {} Accuracy = {}/{} = {:.3f}".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("Level {} Accuracy = {}/{} = {:.3f}\n".format(level, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            output_dict["accuracy_by_level"].append({
                "level": level,
                "num_correct": np.sum(cors_list),
                "num_total": len(cors_list),
                "accuracy": np.mean(cors_list)
            })

        print("#####################")
        f.write("#####################\n")
        for subject in ['Prealgebra', 'Algebra', 'Number Theory', 'Counting & Probability', 'Geometry', 'Intermediate Algebra', 'Precalculus']:
            if subject not in subject_cors.keys():
                print("Skipping", subject)
                continue
            cors_list = subject_cors[subject]
            print("{} Accuracy = {}/{} = {:.3f}".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            f.write("{} Accuracy = {}/{} = {:.3f}\n".format(subject, np.sum(cors_list), len(cors_list), np.mean(cors_list)))
            output_dict["accuracy_by_subject"].append({
                "subject": subject,
                "num_correct": np.sum(cors_list),
                "num_total": len(cors_list),
                "accuracy": np.mean(cors_list)
            })
        print("#####################")
        f.write("#####################\n")
        total = len(all_problems)
        # Adjust for calculation using first sample of each task
        first_sample_correct = sum([task_results[i][0] for i in range(len(all_problems)) if i in task_results])
        correct = first_sample_correct
        print("Overall Accuracy = {}/{} = {:.3f}".format(correct, total, correct/total * 100))
        f.write("Overall Accuracy = {}/{} = {:.3f}\n".format(correct, total, correct/total * 100))
        output_dict["overall_accuracy"] = {
            "num_correct": correct,
            "num_total": total,
            "accuracy": correct/total
        }
        
        # Calculate and save pass@k metrics
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
        f.write("\npass@k results:\n")
        for k, v in pass_at_k_results.items():
            print(f"{k}: {v:.3f}")
            f.write(f"{k}: {v:.3f}\n")
        
        output_dict["pass_at_k"] = pass_at_k_results
        
        # Save pass@k results for curve plotting
        pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
        with open(pass_at_k_path, "w") as jf:
            json.dump(pass_at_k_results, jf, indent=2)
        class JSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.int64):
                    return int(obj)
                return super(JSONEncoder, self).default(obj)
        with open(os.path.join(args.save_dir, "results.json"), "w") as jf:
            json.dump(output_dict, jf, cls=JSONEncoder)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-d", type=str, default="")
    parser.add_argument("--save_dir", "-s", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--num-samples-per-task", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    run(args)
