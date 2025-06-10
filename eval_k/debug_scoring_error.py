#!/usr/bin/env python3
"""
Debug the exact scoring error
"""
import json
import reasoning_gym
import traceback

def debug_scoring_error():
    # Load a sample
    samples_file = "results/Nemotron-Research-Reasoning-Qwen-1.5B/boxnet_chat/samples.jsonl"
    
    with open(samples_file, 'r') as f:
        sample = json.loads(f.readline())
    
    # Create dataset
    data = reasoning_gym.create_dataset('boxnet', size=100, seed=42)
    
    print("Sample entry structure:")
    entry = sample["entry"]
    print(f"Entry keys: {list(entry.keys())}")
    print(f"Metadata keys: {list(entry['metadata'].keys())}")
    
    json_answer = sample["json_answer"]
    print(f"JSON answer: {json_answer}")
    
    try:
        # Convert to string
        answer_str = json.dumps(json_answer)
        print(f"Answer string: {answer_str}")
        
        # Try scoring
        score = data.score_answer(answer=answer_str, entry=entry)
        print(f"Score: {score}")
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        
        # Let's examine the action_from_response function more carefully
        print("\nDebugging action_from_response function...")
        from reasoning_gym.games.boxnet import action_from_response
        
        try:
            initial_state = entry["metadata"]["initial_state"]
            print(f"Initial state: {initial_state}")
            
            result = action_from_response(initial_state, json_answer)
            print(f"Action result: {result}")
            
        except Exception as e2:
            print(f"Error in action_from_response: {e2}")
            traceback.print_exc()

if __name__ == "__main__":
    debug_scoring_error()