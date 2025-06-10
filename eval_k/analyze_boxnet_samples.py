#!/usr/bin/env python3
"""
Script to analyze boxnet samples.jsonl and understand evaluation issues
"""
import json
import sys
from pathlib import Path

def analyze_sample(sample):
    """Analyze a single sample from the JSONL file"""
    print(f"=== Task ID: {sample.get('task_id', 'Unknown')} ===")
    
    # Check if there's a raw output
    raw_output = sample.get('raw_output', '')
    if raw_output:
        print(f"Raw output length: {len(raw_output)} characters")
        print("Raw output preview (first 500 chars):")
        print(raw_output[:500])
        print("..." if len(raw_output) > 500 else "")
        print()
    else:
        print("No raw_output found!")
    
    # Check if there's a json_answer
    json_answer = sample.get('json_answer')
    if json_answer:
        print(f"JSON answer: {json_answer}")
        print(f"JSON answer type: {type(json_answer)}")
        print()
    else:
        print("No json_answer found!")
    
    # Check the expected answer format
    question = sample.get('question', '')
    if 'json' in question.lower():
        print("Question expects JSON format")
    
    # Look for any parsing issues
    if raw_output and not json_answer:
        print("WARNING: Raw output exists but no json_answer - possible parsing failure")
        
        # Try to find JSON in raw output
        if '[{' in raw_output and '}]' in raw_output:
            start = raw_output.find('[{')
            end = raw_output.rfind('}]') + 2
            potential_json = raw_output[start:end]
            print(f"Potential JSON found: {potential_json[:200]}...")
            
            try:
                parsed = json.loads(potential_json)
                print(f"Successfully parsed JSON with {len(parsed)} items")
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
    
    print("-" * 80)

def main():
    samples_file = Path("results/Nemotron-Research-Reasoning-Qwen-1.5B/boxnet_chat/samples.jsonl")
    
    if not samples_file.exists():
        print(f"File not found: {samples_file}")
        return
    
    print(f"Analyzing samples from: {samples_file}")
    print("=" * 80)
    
    sample_count = 0
    samples_with_output = 0
    samples_with_json = 0
    
    with open(samples_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    sample = json.loads(line)
                    sample_count += 1
                    
                    if sample.get('raw_output'):
                        samples_with_output += 1
                    if sample.get('json_answer'):
                        samples_with_json += 1
                    
                    # Analyze first few samples in detail
                    if sample_count <= 3:
                        analyze_sample(sample)
                        
                except json.JSONDecodeError as e:
                    print(f"Failed to parse line {line_num}: {e}")
    
    print("=" * 80)
    print(f"SUMMARY:")
    print(f"Total samples: {sample_count}")
    print(f"Samples with raw_output: {samples_with_output}")
    print(f"Samples with json_answer: {samples_with_json}")
    print(f"Success rate for JSON extraction: {samples_with_json/sample_count*100:.1f}%" if sample_count > 0 else "N/A")
    
    if samples_with_json == 0:
        print("\nWARNING: No samples have json_answer - this explains why pass@k is 0!")
        print("The evaluation likely fails because it can't find properly formatted answers.")

if __name__ == "__main__":
    main()