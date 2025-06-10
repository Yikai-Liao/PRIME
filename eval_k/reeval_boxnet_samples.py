#!/usr/bin/env python3
"""
Re-evaluate existing boxnet samples with the fixed evaluation
"""
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from tqdm import tqdm
import reasoning_gym

def reeval_samples():
    # Load existing samples
    samples_file = "results/Nemotron-Research-Reasoning-Qwen-1.5B/boxnet_chat/samples.jsonl"
    
    samples = []
    with open(samples_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"Loaded {len(samples)} samples")
    
    # Create dataset for evaluation
    data = reasoning_gym.create_dataset('boxnet', size=100, seed=42)
    
    # Re-evaluate samples
    task_results = {}
    scores = []
    
    for sample in tqdm(samples, desc="Re-evaluating samples"):
        task_id = sample["task_id"]
        if task_id not in task_results:
            task_results[task_id] = []
        
        entry = sample["entry"]
        json_answer = sample["json_answer"]
        
        try:
            if json_answer is not None:
                # Convert json_answer to string for reasoning_gym's score_answer method
                answer_str = json.dumps(json_answer) if isinstance(json_answer, (list, dict)) else str(json_answer)
                score = data.score_answer(answer=answer_str, entry=entry)
                reward = score
            else:
                reward = 0.0
            
            task_results[task_id].append(reward)
            scores.append(reward)
        except Exception as e:
            print(f"Error evaluating task {task_id}: {e}")
            task_results[task_id].append(0.0)
            scores.append(0.0)
    
    # Calculate statistics
    num_samples_per_task = 10
    k_values = list(range(1, num_samples_per_task + 1))
    results = {}
    
    # Perfect solutions (score = 1.0)
    for k in k_values:
        total = 0
        correct = 0
        for task_id, rewards in task_results.items():
            total += 1
            if any(reward >= 1.0 for reward in rewards[:k]):
                correct += 1
        results[f"pass@{k}"] = correct / total if total > 0 else 0.0
    
    # Partial solutions (score >= 0.5)
    for k in k_values:
        total = 0
        correct = 0
        for task_id, rewards in task_results.items():
            total += 1
            if any(reward >= 0.5 for reward in rewards[:k]):
                correct += 1
        results[f"partial_pass@{k}"] = correct / total if total > 0 else 0.0
    
    # Any progress (score > 0)
    for k in k_values:
        total = 0
        correct = 0
        for task_id, rewards in task_results.items():
            total += 1
            if any(reward > 0 for reward in rewards[:k]):
                correct += 1
        results[f"any_progress@{k}"] = correct / total if total > 0 else 0.0
    
    print("\n=== Results ===")
    print(f"Total samples: {len(samples)}")
    print(f"Average score: {sum(scores) / len(scores):.3f}")
    print(f"Samples with perfect score (1.0): {sum(1 for s in scores if s >= 1.0)}")
    print(f"Samples with partial score (>= 0.5): {sum(1 for s in scores if s >= 0.5)}")
    print(f"Samples with any progress (> 0): {sum(1 for s in scores if s > 0)}")
    
    for k in [1, 5, 10]:
        print(f"pass@{k}: {results[f'pass@{k}']:.3f}")
        print(f"partial_pass@{k}: {results[f'partial_pass@{k}']:.3f}")
        print(f"any_progress@{k}: {results[f'any_progress@{k}']:.3f}")
        print()
    
    # Save updated results
    pass_at_k_path = "results/Nemotron-Research-Reasoning-Qwen-1.5B/boxnet_chat/pass_at_k.json"
    with open(pass_at_k_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Updated results saved to {pass_at_k_path}")

if __name__ == "__main__":
    reeval_samples()