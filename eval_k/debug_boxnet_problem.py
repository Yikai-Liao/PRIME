#!/usr/bin/env python3
"""
Debug specific boxnet problem to understand the solution format
"""
import reasoning_gym
import json

def debug_specific_problem():
    # Create dataset
    data = reasoning_gym.create_dataset('boxnet', size=5, seed=42)
    sample = data[0]
    
    print("=== Problem Analysis ===")
    print("Question:")
    print(sample['question'])
    print("\nMetadata:")
    print(json.dumps(sample['metadata'], indent=2))
    
    # Parse the problem to understand the setup
    question = sample['question']
    
    # Extract the current state
    import re
    agent_pattern = r"Agent\[([\d.]+),\s*([\d.]+)\]:\s*I am in square\[([\d.]+),\s*([\d.]+)\],\s*I can observe \[(.*?)\],\s*I can do \[(.*?)\]"
    
    agents = re.findall(agent_pattern, question)
    
    print(f"\n=== Agents Found ===")
    for i, agent in enumerate(agents):
        x, y, square_x, square_y, observe, actions = agent
        print(f"Agent {i+1}: [{x}, {y}]")
        print(f"  Square: [{square_x}, {square_y}]")
        print(f"  Observes: {observe}")
        print(f"  Actions: {actions}")
        print()
    
    # The sample shows this is a simple case where Agent[0.5, 0.5] has all boxes and targets in same square
    # Let's try different solution formats
    
    test_solutions = [
        # Single step - move all to targets
        [{"Agent[0.5, 0.5]": "move(box_red, target_red)"}],
        [{"Agent[0.5, 0.5]": "move(box_blue, target_blue)"}], 
        [{"Agent[0.5, 0.5]": "move(box_green, target_green)"}],
        
        # Multi-step solutions
        [
            {"Agent[0.5, 0.5]": "move(box_red, target_red)"},
            {"Agent[0.5, 0.5]": "move(box_blue, target_blue)"},
            {"Agent[0.5, 0.5]": "move(box_green, target_green)"}
        ],
        
        # Single step with all moves
        [{"Agent[0.5, 0.5]": "move(box_red, target_red)", "Agent[0.5, 0.5]": "move(box_blue, target_blue)", "Agent[0.5, 0.5]": "move(box_green, target_green)"}],
    ]
    
    print("=== Testing Solutions ===")
    for i, solution in enumerate(test_solutions):
        try:
            score = data.score_answer(answer=solution, entry=sample)
            print(f"Solution {i+1}: {solution}")
            print(f"Score: {score}")
            print()
        except Exception as e:
            print(f"Solution {i+1}: Error - {e}")
            print()

if __name__ == "__main__":
    debug_specific_problem()