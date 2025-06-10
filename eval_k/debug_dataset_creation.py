#!/usr/bin/env python3
"""
Debug dataset creation to understand the structure
"""
import reasoning_gym

def debug_dataset():
    print("Creating boxnet dataset...")
    data = reasoning_gym.create_dataset('boxnet', size=5, seed=42)
    
    print(f"Dataset type: {type(data)}")
    print(f"Dataset length: {len(data)}")
    
    # Check first few samples
    for i in range(min(3, len(data))):
        sample = data[i]
        print(f"\n=== Sample {i} ===")
        print(f"Keys: {list(sample.keys()) if hasattr(sample, 'keys') else 'No keys method'}")
        
        if isinstance(sample, dict):
            for key, value in sample.items():
                print(f"  {key}: {type(value)}")
                if key == 'answer' and value is not None:
                    print(f"    Answer: {value}")
                elif key == 'metadata':
                    print(f"    Metadata keys: {list(value.keys()) if isinstance(value, dict) else 'Not a dict'}")
    
    # Test score_answer method
    print("\n=== Testing score_answer method ===")
    sample = data[0]
    
    # Try to understand what constitutes a valid answer
    test_answers = [
        None,
        [],
        [{"Agent[0.5, 0.5]": "move(box_red, target_red)"}],
        [{"Agent[0.5, 0.5]": "move(box_blue, target_blue)"}],
        [{"Agent[0.5, 0.5]": "move(box_green, target_green)"}]
    ]
    
    for test_answer in test_answers:
        try:
            score = data.score_answer(answer=test_answer, entry=sample)
            print(f"Answer {test_answer} -> Score: {score}")
        except Exception as e:
            print(f"Answer {test_answer} -> Error: {e}")

if __name__ == "__main__":
    debug_dataset()