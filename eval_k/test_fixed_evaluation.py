#!/usr/bin/env python3
"""
Test the fixed evaluation logic
"""
import json
import reasoning_gym

def test_fixed_evaluation():
    # Create the dataset
    data = reasoning_gym.create_dataset('boxnet', size=5, seed=42)
    sample = data[0]
    
    # Test different answer formats
    test_answers = [
        [{"Agent[0.5, 0.5]": "move(box_red, target_red)"}],
        [{"Agent[0.5, 0.5]": "move(box_blue, target_blue)"}], 
        [{"Agent[0.5, 0.5]": "move(box_green, target_green)"}],
        [
            {"Agent[0.5, 0.5]": "move(box_red, target_red)"},
            {"Agent[0.5, 0.5]": "move(box_blue, target_blue)"},
            {"Agent[0.5, 0.5]": "move(box_green, target_green)"}
        ]
    ]
    
    print("Testing fixed evaluation (converting to JSON string):")
    for i, answer in enumerate(test_answers):
        try:
            # Convert to JSON string as the scoring function expects
            answer_str = json.dumps(answer)
            score = data.score_answer(answer=answer_str, entry=sample)
            print(f"Answer {i+1}: {answer}")
            print(f"Score: {score}")
            print(f"Reward (binarized): {1.0 if score == 1.0 else 0.0}")
            print()
        except Exception as e:
            print(f"Answer {i+1}: Error - {e}")
            print()

if __name__ == "__main__":
    test_fixed_evaluation()