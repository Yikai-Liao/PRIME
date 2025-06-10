import json
import matplotlib.pyplot as plt
import argparse
import os

def plot_pass_at_k_curve(results_file, output_dir=None):
    """
    Plot pass@k curve from evaluation results
    
    Args:
        results_file: Path to the pass_at_k.json file
        output_dir: Directory to save the plot (optional)
    """
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
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, pass_at_k_scores, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('k (number of samples)', fontsize=12)
    plt.ylabel('pass@k', fontsize=12)
    plt.title('pass@k Performance Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    
    # Add value labels on points
    for k, score in zip(k_values, pass_at_k_scores):
        plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'pass_at_k_curve.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'pass_at_k_curve.pdf'), bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print(f"\npass@k Results:")
    print(f"{'k':>3} | {'pass@k':>8}")
    print("-" * 15)
    for k, score in zip(k_values, pass_at_k_scores):
        print(f"{k:>3} | {score:>8.4f}")

def main():
    parser = argparse.ArgumentParser(description='Plot pass@k curves from evaluation results')
    parser.add_argument('--results_file', type=str, required=True, 
                       help='Path to the pass_at_k.json file')
    parser.add_argument('--output_dir', type=str, 
                       help='Directory to save the plot')
    
    args = parser.parse_args()
    plot_pass_at_k_curve(args.results_file, args.output_dir)

if __name__ == "__main__":
    main()