#!/usr/bin/env python3
"""
Debug the sample structure to understand the issue
"""
import json

def debug_sample_structure():
    samples_file = "results/Nemotron-Research-Reasoning-Qwen-1.5B/boxnet_chat/samples.jsonl"
    
    with open(samples_file, 'r') as f:
        sample = json.loads(f.readline())
    
    print("Sample keys:", list(sample.keys()))
    print()
    
    for key, value in sample.items():
        print(f"{key}: {type(value)}")
        if key == "entry":
            if isinstance(value, dict):
                print(f"  Entry keys: {list(value.keys())}")
                for entry_key, entry_value in value.items():
                    print(f"    {entry_key}: {type(entry_value)} - {str(entry_value)[:100] if entry_value else 'None'}...")
            else:
                print(f"  Entry value: {value}")
        elif key in ["json_answer", "answer"]:
            print(f"  Value: {value}")
        print()

if __name__ == "__main__":
    debug_sample_structure()