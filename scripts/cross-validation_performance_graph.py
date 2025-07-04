import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import Config

def load_training_logs(base_directory='.'):
    """
    Load training logs from directory structure and calculate averages.
    
    Parameters:
    base_directory (str): Base directory containing class folders
    
    Returns:
    dict: Dictionary with averaged metrics for each class and fold
    """
    
    results = {}
    
    for class_name in Config.CLASSES:
        class_path = Path(base_directory) / class_name
        log_file = class_path / 'training_log.txt'
        
        if not log_file.exists():
            print(f"Warning: {log_file} not found, skipping {class_name}")
            continue
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            results[class_name] = {}
            # Process each fold configuration
            for fold_key, metrics in data.items():
                fold_num = int(fold_key.split('_')[0])
                
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
    
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    # Define colors and markers for each class
    name_translation = {
        'Sclerosis':  'Sclerosis',
        'Podocitopatia': 'ECDC',
        'Normal':  'Normal',
        'Hypercelularidade': 'Hypercellularity',
        'Crescent': 'Crescent',
        'Membranous': 'Membranous',
    }

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
                    label=name_translation[class_name])
            
            print(f"{class_name}: Folds {folds}, F1-Scores {[f'{score:.4f}' for score in f1_scores]}")
    
    #plt.title('Desempenho da EfficientNet-B0 por Número de Folds', fontsize=16, pad=20)
    plt.xlabel('#Folds', fontsize=12)
    plt.ylabel('F1-score', fontsize=12)
    
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
    
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
    plt.ylim(bottom=0.8, top=1.0)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    
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
    
    create_performance_graph_from_logs(
        base_directory=base_directory,
        save_path='./results/effnetb0_performance_graph.pdf'
    )
    
    print_summary_statistics(base_directory)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", help="Directory where logs are saved.")
    args = parser.parse_args()

    main(args.logs_dir)
