import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import json
import time
import argparse
import traceback
import pandas as pd
from tqdm import tqdm
from typing import List
from datasets import Dataset
from utils.data import write_jsonl
import reasoning_gym
os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="")
parser.add_argument("--save_dir", type=str)
parser.add_argument("--num-samples-per-task", type=int, default=10)
parser.add_argument("--temperature", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--test", action="store_true", help="Test mode: evaluate only first 10 samples")
parser.add_argument("--dataset_size", type=int, default=100, help="Size of boxnet dataset to generate")
args = parser.parse_args()

# Create boxnet dataset
print(f"Creating boxnet dataset with size={args.dataset_size}, seed={args.seed}")
data = reasoning_gym.create_dataset('boxnet', size=args.dataset_size, seed=args.seed)

# Convert to problems format
problems = {}
for i, x in enumerate(data):
    problems[f"boxnet_{i}"] = {
        "task_id": f"boxnet_{i}",
        "question": x['question'],
        "answer": x['answer'],
        "metadata": x['metadata'],
        "entry": x  # Store original entry for scoring
    }

# Apply test mode limitation
if args.test:
    problem_ids = sorted(problems.keys())[:10]
    problems = {pid: problems[pid] for pid in problem_ids}
    print(f"Test mode: Evaluating only first {len(problems)} problems")

from vllm import LLM, SamplingParams
import torch

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model)

def make_conv_hf(example, tokenizer):
    prompt = f"Solve the following problem and provide your answer in JSON format.\n\nProblem: {example['question']}\n\nProvide your answer as a valid JSON object."
    
    system_prompt = open("system_prompt.md").read()

    msg = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt + "\n\nSolve the problem and provide your answer in JSON format."}
    ]

    chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return chat

def extract_json_answer(text):
    """Extract JSON answer from model output"""
    try:
        import re
        
        # First try to find JSON arrays (for boxnet answers like [{"Agent": "action"}, ...])
        array_pattern = r'\[(?:[^\[\]]*(?:\[[^\]]*\])*)*\]'
        array_matches = re.findall(array_pattern, text, re.DOTALL)
        
        if array_matches:
            # Try to parse each array match, prioritizing the last one
            for match in reversed(array_matches):
                try:
                    parsed = json.loads(match)
                    # Validate it's a list of dictionaries as expected for boxnet
                    if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                        return parsed
                except:
                    continue
        
        # If no valid JSON array found, look for JSON objects using balanced brace matching
        brace_count = 0
        start_idx = -1
        valid_jsons = []
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found a complete JSON object
                    json_str = text[start_idx:i+1]
                    try:
                        parsed = json.loads(json_str)
                        valid_jsons.append(parsed)
                    except:
                        continue
        
        # Return the last valid JSON object found
        if valid_jsons:
            return valid_jsons[-1]
        
        # Last resort: try to parse the entire text
        return json.loads(text.strip())
    except:
        # Return None if no valid JSON can be extracted
        return None

def generate_sample_batch(question_list):
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        seed=args.seed,
        enforce_eager=True
    )
    sampling_params = SamplingParams(
        max_tokens=32768,
        temperature=args.temperature,
        n=args.num_samples_per_task,
        stop=["<|eot_id|>","<|im_end|>"],
        seed=args.seed
    )
    
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)

    completions = []
    for output in outputs:
        for i in range(args.num_samples_per_task):
            raw_output = output.outputs[i].text
            json_answer = extract_json_answer(raw_output)
            completions.append({
                "raw_output": raw_output,
                "json_answer": json_answer
            })
    return completions

def evaluate_boxnet_correctness(samples, data):
    """Evaluate correctness using reasoning_gym's score_answer method"""
    results = {}
    task_results = {}
    
    for sample in tqdm(samples, desc="Evaluating samples"):
        task_id = sample["task_id"]
        if task_id not in task_results:
            task_results[task_id] = []
        
        # Get the original entry for scoring
        entry = sample["entry"]
        json_answer = sample["json_answer"]
        
        try:
            if json_answer is not None:
                # Convert json_answer to string for reasoning_gym's score_answer method
                answer_str = json.dumps(json_answer) if isinstance(json_answer, (list, dict)) else str(json_answer)
                score = data.score_answer(answer=answer_str, entry=entry)
                # Use the continuous score (matches boxes matched / total boxes)
                # Score of 1.0 means all boxes matched, 0.33 means 1/3 boxes matched, etc.
                reward = score
            else:
                reward = 0.0
            
            task_results[task_id].append(reward)
        except Exception as e:
            print(f"Error evaluating task {task_id}: {e}")
            task_results[task_id].append(0.0)
    
    # Calculate pass@k for different k values
    k_values = list(range(1, args.num_samples_per_task + 1))
    for k in k_values:
        total = 0
        correct = 0
        for task_id, rewards in task_results.items():
            total += 1
            # Pass@k: at least one of the first k attempts has score >= 1.0 (perfect solution)
            if any(reward >= 1.0 for reward in rewards[:k]):
                correct += 1
        results[f"pass@{k}"] = correct / total if total > 0 else 0.0
    
    # Also calculate partial success rates (score >= 0.5, meaning at least half boxes matched)
    for k in k_values:
        total = 0
        correct = 0
        for task_id, rewards in task_results.items():
            total += 1
            # Partial success: at least one of the first k attempts has score >= 0.5
            if any(reward >= 0.5 for reward in rewards[:k]):
                correct += 1
        results[f"partial_pass@{k}"] = correct / total if total > 0 else 0.0
    
    return results

samples = []
problems_df = pd.DataFrame(problems).T
problems_dataset = Dataset.from_pandas(problems_df)
problems_dataset = problems_dataset.map(
    lambda x: {"instruction": make_conv_hf(x, tokenizer)}, 
    cache_file_name="cache/boxnet", 
    load_from_cache_file=False
)

# completions = generate_sample_batch(problems_dataset["instruction"])
import pickle
# Save completions for debugging
# with open(os.path.join(args.save_dir, "completions.pkl"), "wb") as f:
#     pickle.dump(completions, f)
# print(f"Saved completions to {os.path.join(args.save_dir, 'completions.pkl')}")
# Create samples for each completion of each problem
with open(os.path.join(args.save_dir, "completions.pkl"), "rb") as f:
    completions = pickle.load(f)

for i, problem in enumerate(problems_dataset):
    for j in range(args.num_samples_per_task):
        completion_idx = i * args.num_samples_per_task + j
        if completion_idx >= len(completions):
            print(f"Warning: completion_idx {completion_idx} exceeds completions length {len(completions)}")
            break
        completion = completions[completion_idx]
        # Use the original entry from the data source, not the processed one
        original_entry = data[i]  # Use original entry from reasoning_gym
        sample = {
            "task_id": problem["task_id"],
            "question": problem["question"],
            "answer": problem["answer"],
            "metadata": problem["metadata"],
            "entry": original_entry,
            "raw_output": completion["raw_output"],
            "json_answer": completion["json_answer"]
        }
        samples.append(sample)

output_filepath = os.path.join(args.save_dir, "samples.jsonl")
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

write_jsonl(output_filepath, samples)

# Evaluate using reasoning_gym
score = evaluate_boxnet_correctness(samples, data)
print(score)

# Save detailed results
score_path = os.path.join(args.save_dir, "result.txt")
with open(score_path, "w") as f:
    f.write(str(score))

# Save pass@k results for curve plotting
pass_at_k_path = os.path.join(args.save_dir, "pass_at_k.json")
with open(pass_at_k_path, "w") as f:
    json.dump(score, f, indent=2)