#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from pathlib import Path

def load_pass_at_k_data(results_file):
    """Load pass@k data from results file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Extract k values and pass@k scores
    k_values = []
    pass_at_k_scores = []
    
    for key, value in results.items():
        if key.startswith('pass@') and key != 'pass@detail':
            k = int(key.split('@')[1])
            k_values.append(k)
            pass_at_k_scores.append(value)
    
    # Sort by k values
    sorted_data = sorted(zip(k_values, pass_at_k_scores))
    k_values, pass_at_k_scores = zip(*sorted_data)
    
    return k_values, pass_at_k_scores

def load_aime_split_data(results_split_file):
    """Load AIME split results to identify sub-experiments"""
    with open(results_split_file, 'r') as f:
        content = f.read()
    
    # Parse the content to extract sub-experiments
    sub_experiments = {}
    lines = content.strip().split('\n')
    
    for line in lines:
        if line.strip() and ':' in line:
            parts = line.split(':')
            if len(parts) >= 2:
                key = parts[0].strip().strip('{}')
                # Extract numbers after key
                rest = ':'.join(parts[1:]).strip()
                if 'total' in rest and 'success' in rest:
                    # Parse like: "total": 15, "success": 9
                    total_idx = rest.find('"total"')
                    success_idx = rest.find('"success"')
                    if total_idx != -1 and success_idx != -1:
                        try:
                            total_str = rest[total_idx:].split(',')[0]
                            total = int(total_str.split(':')[1].strip())
                            success_str = rest[success_idx:].split(',')[0]  
                            success = int(success_str.split(':')[1].strip())
                            sub_experiments[key] = {'total': total, 'success': success}
                        except:
                            continue
    
    return sub_experiments

def plot_comparison(model1_name, model1_data, model2_name, model2_data, title, output_path=None):
    """Plot comparison between two models"""
    k_values1, scores1 = model1_data
    k_values2, scores2 = model2_data
    
    # Ensure both have same k values for comparison
    all_k_values = sorted(list(set(k_values1) & set(k_values2)))
    
    # Filter data to common k values
    scores1_filtered = [scores1[k_values1.index(k)] for k in all_k_values]
    scores2_filtered = [scores2[k_values2.index(k)] for k in all_k_values]
    
    plt.figure(figsize=(12, 8))
    
    # Plot both models
    plt.plot(all_k_values, scores1_filtered, 'b-o', linewidth=2, markersize=6, label=model1_name)
    plt.plot(all_k_values, scores2_filtered, 'r-^', linewidth=2, markersize=6, label=model2_name)
    
    plt.xlabel('k (number of samples)', fontsize=12)
    plt.ylabel('pass@k', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(all_k_values)
    
    # Add value labels on points
    for k, score1, score2 in zip(all_k_values, scores1_filtered, scores2_filtered):
        plt.annotate(f'{score1:.3f}', (k, score1), textcoords="offset points", 
                    xytext=(-15, 10), ha='center', fontsize=9, color='blue')
        plt.annotate(f'{score2:.3f}', (k, score2), textcoords="offset points", 
                    xytext=(15, 10), ha='center', fontsize=9, color='red')
    
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(output_path.replace('.png', '.png'), dpi=300, bbox_inches='tight')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
        print(f"Saved comparison plot to {output_path}")
    
    plt.show()
    plt.close()

def compare_models_on_experiment(results_dir, model1_name, model2_name, experiment_name, output_dir=None):
    """Compare two models on a specific experiment"""
    model1_path = os.path.join(results_dir, model1_name, f"{experiment_name}_chat", "pass_at_k.json")
    model2_path = os.path.join(results_dir, model2_name, f"{experiment_name}_chat", "pass_at_k.json")
    
    if not os.path.exists(model1_path) or not os.path.exists(model2_path):
        print(f"Missing results for {experiment_name}: {model1_path} or {model2_path}")
        return
    
    model1_data = load_pass_at_k_data(model1_path)
    model2_data = load_pass_at_k_data(model2_path)
    
    title = f"pass@k Comparison: {experiment_name.upper()}"
    output_path = None
    if output_dir:
        output_path = os.path.join(output_dir, f"{experiment_name}_comparison.png")
    
    plot_comparison(model1_name, model1_data, model2_name, model2_data, title, output_path)

def compare_aime_sub_experiments(results_dir, model1_name, model2_name, output_dir=None):
    """Compare AIME sub-experiments separately"""
    model1_split_path = os.path.join(results_dir, model1_name, "aime_chat", "results_split.txt")
    model2_split_path = os.path.join(results_dir, model2_name, "aime_chat", "results_split.txt")
    
    if not os.path.exists(model1_split_path) or not os.path.exists(model2_split_path):
        print("Missing AIME split results")
        return
    
    # Load AIME split data
    model1_splits = load_aime_split_data(model1_split_path)
    model2_splits = load_aime_split_data(model2_split_path)
    
    # Find common sub-experiments
    common_experiments = set(model1_splits.keys()) & set(model2_splits.keys())
    
    print(f"Found {len(common_experiments)} common AIME sub-experiments")
    
    # Create comparison plots for each sub-experiment
    for exp_name in sorted(common_experiments):
        if exp_name == "unknown":
            continue  # Skip unknown category
            
        model1_success_rate = model1_splits[exp_name]['success'] / model1_splits[exp_name]['total']
        model2_success_rate = model2_splits[exp_name]['success'] / model2_splits[exp_name]['total']
        
        # Create a simple bar chart for this sub-experiment
        plt.figure(figsize=(8, 6))
        models = [model1_name, model2_name]
        success_rates = [model1_success_rate, model2_success_rate]
        colors = ['blue', 'red']
        
        bars = plt.bar(models, success_rates, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.3f}', ha='center', va='bottom', fontsize=11)
        
        plt.ylabel('Success Rate', fontsize=12)
        plt.title(f'AIME Success Rate Comparison: {exp_name}', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add problem counts
        plt.text(0.5, 0.9, f"Problems: {model1_splits[exp_name]['total']}", 
                transform=plt.gca().transAxes, ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
        
        plt.tight_layout()
        
        if output_dir:
            safe_exp_name = exp_name.replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_dir, f"aime_{safe_exp_name}_comparison.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
            print(f"Saved AIME sub-experiment plot to {output_path}")
        
        plt.show()
        plt.close()
    
    # Create overall AIME comparison using pass@k data
    compare_models_on_experiment(results_dir, model1_name, model2_name, "aime", output_dir)

def main():
    parser = argparse.ArgumentParser(description='Compare pass@k curves between two models')
    parser.add_argument('--results_dir', type=str, default='results', 
                       help='Directory containing results for both models')
    parser.add_argument('--model1', type=str, required=True, 
                       help='First model name (directory name in results)')
    parser.add_argument('--model2', type=str, required=True, 
                       help='Second model name (directory name in results)')
    parser.add_argument('--experiment', type=str, 
                       help='Specific experiment to compare (e.g., human_eval, mbpp, math)')
    parser.add_argument('--output_dir', type=str, 
                       help='Directory to save comparison plots')
    parser.add_argument('--all', action='store_true', 
                       help='Compare all available experiments')
    
    args = parser.parse_args()
    
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Available experiments
    experiments = ['human_eval', 'mbpp', 'math', 'amc', 'boxnet', 'leetcode']
    
    if args.experiment:
        if args.experiment == 'aime':
            compare_aime_sub_experiments(args.results_dir, args.model1, args.model2, args.output_dir)
        else:
            compare_models_on_experiment(args.results_dir, args.model1, args.model2, args.experiment, args.output_dir)
    elif args.all:
        # Compare all available experiments
        for exp in experiments:
            print(f"\nComparing {exp}...")
            compare_models_on_experiment(args.results_dir, args.model1, args.model2, exp, args.output_dir)
        
        # Special handling for AIME
        print(f"\nComparing AIME (with sub-experiments)...")
        compare_aime_sub_experiments(args.results_dir, args.model1, args.model2, args.output_dir)
    else:
        print("Please specify --experiment or use --all to compare all experiments")
        print("Available experiments:", experiments + ['aime'])

if __name__ == "__main__":
    main()