import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_training_logs(base_directory='.'):
    """
    Load training logs from directory structure and calculate averages.
    
    Parameters:
    base_directory (str): Base directory containing class folders
    
    Returns:
    dict: Dictionary with averaged metrics for each class and fold
    """
    
    results = {}
    
    # Expected class directories
    class_dirs = ['Crescent', 'Hypercelularidade', 'Membranous', 
                  'Normal', 'Podocitopatia', 'Sclerosis']
    
    for class_name in class_dirs:
        class_path = Path(base_directory) / class_name
        log_file = class_path / 'training_log.txt'
        
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping {class_name}")
            continue
        
        try:
            # Load JSON data
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results[class_name] = {}
            
            # Process each fold configuration
            for fold_key, metrics in data.items():
                # Extract fold number from key (e.g., "2_folds" -> 2)
                fold_num = int(fold_key.split('_')[0])
                
                # Calculate averages for each metric
                averaged_metrics = {}
                for metric_name, values in metrics.items():
                    if isinstance(values, list) and len(values) > 0:
                        averaged_metrics[metric_name] = np.mean(values)
                    else:
                        averaged_metrics[metric_name] = values
                
                results[class_name][fold_num] = averaged_metrics
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing {log_file}: {e}")
            continue
    
    return results

def create_performance_graph_from_logs(base_directory='.', save_path=None):
    """
    Creates a performance graph showing F1-Score vs Number of Folds for different classes.
    Reads data from training_log.txt files in class directories.
    
    Parameters:
    base_directory (str): Base directory containing class folders
    save_path (str): Optional path to save the graph
    """
    
    # Load data from training logs
    data = load_training_logs(base_directory)
    
    if not data:
        print("No data found. Please check your directory structure and file paths.")
        return
    
    # Set up the plot with Portuguese title
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    # Define colors and markers for each class
    colors = {
        'Sclerosis': '#1f77b4',      # Blue
        'Podocitopatia': '#ff7f0e',  # Orange
        'Normal': '#2ca02c',         # Green
        'Hypercelularidade': '#d62728', # Red
        'Crescent': '#9467bd',       # Purple
        'Membranous': '#8c564b'      # Brown
    }
    
    markers = {
        'Sclerosis': 'o',
        'Podocitopatia': 's',
        'Normal': 'D',
        'Hypercelularidade': '^',
        'Crescent': 'v',
        'Membranous': '<'
    }
    
    # Plot each class
    for class_name, class_data in data.items():
        folds = []
        f1_scores = []
        
        # Sort by fold number and extract F1 scores
        for fold_num in sorted(class_data.keys()):
            if 'f1_score' in class_data[fold_num]:
                folds.append(fold_num)
                f1_scores.append(class_data[fold_num]['f1_score'])
        
        if folds and f1_scores:
            plt.plot(folds, f1_scores, 
                    color=colors.get(class_name, '#000000'),
                    marker=markers.get(class_name, 'o'),
                    linewidth=2,
                    markersize=8,
                    label=class_name)
            
            print(f"{class_name}: Folds {folds}, F1-Scores {[f'{score:.4f}' for score in f1_scores]}")
    
    # Customize the plot
    plt.title('Desempenho da EfficientNet-B0 por Número de Folds', fontsize=16, pad=20)
    plt.xlabel('Número de Folds', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    
    # Set axis limits and ticks
    all_folds = set()
    all_f1_scores = []
    
    for class_data in data.values():
        for fold_num, metrics in class_data.items():
            if 'f1_score' in metrics:
                all_folds.add(fold_num)
                all_f1_scores.append(metrics['f1_score'])
    
    if all_folds and all_f1_scores:
        min_fold = min(all_folds)
        max_fold = max(all_folds)
        min_f1 = min(all_f1_scores)
        max_f1 = max(all_f1_scores)
        
        plt.xlim(min_fold - 0.2, max_fold + 0.2)
        plt.ylim(min_f1 - 0.01, max_f1 + 0.01)
        plt.xticks(sorted(all_folds))
    
    # Add grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    plt.legend(loc='center right', bbox_to_anchor=(1.0, 0.5), frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
    # Show plot
    plt.show()

def print_summary_statistics(base_directory='.'):
    """
    Print summary statistics for all classes and folds.
    """
    data = load_training_logs(base_directory)
    
    if not data:
        print("No data found.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS (Averaged across Cross-Validation Steps)")
    print("="*80)
    
    for class_name, class_data in data.items():
        print(f"\n{class_name}:")
        print("-" * 50)
        
        for fold_num in sorted(class_data.keys()):
            metrics = class_data[fold_num]
            print(f"  {fold_num} Folds:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name.capitalize()}: {value:.4f}")

def main(base_directory):
    """
    Main function to run the script.
    """
    print("Loading training logs and creating performance graph...")
    
    # Create the graph
    create_performance_graph_from_logs(
        base_directory=base_directory,
        save_path='effnetb0_performance_graph.png'  # Remove this line if you don't want to save
    )
    
    # Print detailed statistics
    print_summary_statistics(base_directory)
    
    print("\nGraph generated successfully!")
    print(f"Make sure your directory structure matches:")
    print("├── Crescent/")
    print("│   └── training_log.txt")
    print("├── Hypercelularidade/")
    print("│   └── training_log.txt")
    print("├── ... (other class directories)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", help="Directory where logs are saved.")
    args = parser.parse_args()

    main(args.logs_dir)
