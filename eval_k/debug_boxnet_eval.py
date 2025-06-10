#!/usr/bin/env python3
"""
Debug script to test boxnet evaluation directly
"""
import json
import reasoning_gym
from pathlib import Path

def test_sample_evaluation():
    # Load a sample
    samples_file = Path("results/Nemotron-Research-Reasoning-Qwen-1.5B/boxnet_chat/samples.jsonl")
    
    # Create the dataset for evaluation
    data = reasoning_gym.create_dataset('boxnet', size=100, seed=42)
    
    with open(samples_file, 'r') as f:
        sample = json.loads(f.readline())
    
    print("Sample analysis:")
    print(f"Task ID: {sample['task_id']}")
    print(f"JSON Answer: {sample['json_answer']}")
    print(f"Expected Answer: {sample['answer']}")
    print()
    
    # Test evaluation
    entry = sample["entry"]
    json_answer = sample["json_answer"]
    
    print("Testing evaluation:")
    print(f"Entry type: {type(entry)}")
    print(f"JSON answer type: {type(json_answer)}")
    
    try:
        score = data.score_answer(answer=json_answer, entry=entry)
        print(f"Score: {score}")
        print(f"Reward (binarized): {1.0 if score == 1.0 else 0.0}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    # Test with the expected answer
    print("\nTesting with expected answer:")
    try:
        expected_answer = sample["answer"]
        if expected_answer:
            score = data.score_answer(answer=expected_answer, entry=entry)
            print(f"Expected answer score: {score}")
        else:
            print("No expected answer found")
    except Exception as e:
        print(f"Error with expected answer: {e}")
    
    # Let's examine the structure of the entry
    print(f"\nEntry structure:")
    print(f"Entry keys: {list(entry.keys()) if isinstance(entry, dict) else 'Not a dict'}")
    if isinstance(entry, dict):
        for key, value in entry.items():
            print(f"  {key}: {type(value)} - {str(value)[:100]}...")

if __name__ == "__main__":
    test_sample_evaluation()