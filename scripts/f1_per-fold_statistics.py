import os
import json
import argparse
from pathlib import Path

import numpy as np

from config import Config

def load_training_logs(base_directory='.'):
    """
    Load training logs from directory structure.
    
    Parameters:
    base_directory (str): Base directory containing class folders
    
    Returns:
    dict: Dictionary with raw metrics for each class and fold
    """
    
    results = {}
    
    for class_name in Config.CLASSES:
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
                results[class_name][fold_num] = metrics
                
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing {log_file}: {e}")
            continue
    
    return results

def calculate_f1_statistics(data):
    """
    Calculate mean and standard deviation for F1 scores.
    
    Parameters:
    data (dict): Raw training log data
    
    Returns:
    dict: Statistics organized by fold number and class
    """
    
    statistics = {}
    all_folds = set()
    for class_data in data.values():
        all_folds.update(class_data.keys())
    
    # Calculate statistics for each fold
    for fold_num in sorted(all_folds):
        statistics[fold_num] = {}
        
        for class_name, class_data in data.items():
            if fold_num in class_data and 'f1_score' in class_data[fold_num]:
                f1_scores = class_data[fold_num]['f1_score']
                
                if isinstance(f1_scores, list) and len(f1_scores) > 0:
                    mean_f1 = np.mean(f1_scores)
                    std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
                    
                    statistics[fold_num][class_name] = {
                        'mean': mean_f1,
                        'std': std_f1,
                        'raw_scores': f1_scores
                    }
    
    return statistics

def print_f1_statistics(base_directory='.', show_raw_scores=False):
    """
    Print F1-score statistics in the requested format.
    
    Parameters:
    base_directory (str): Base directory containing class folders
    show_raw_scores (bool): Whether to show individual CV scores
    """

    data = load_training_logs(base_directory)
    if not data:
        print("No data found. Please check your directory structure and file paths.")
        return
    
    statistics = calculate_f1_statistics(data)
    if not statistics:
        print("No F1-score data found in the training logs.")
        return
    
    # Print statistics for each fold
    for fold_num in sorted(statistics.keys()):
        print(f"===== {fold_num} Folds =====")
        
        for class_name in Config.CLASSES:
            if class_name in statistics[fold_num]:
                stats = statistics[fold_num][class_name]
                
                mean_percent = stats['mean'] * 100
                std_percent = stats['std'] * 100
                
                print(f"{class_name}: F1 = {mean_percent:.1f}% +- {std_percent:.1f}%")
                
                if show_raw_scores:
                    raw_scores_percent = [score * 100 for score in stats['raw_scores']]
                    print(f"  Raw scores: {[f'{score:.1f}%' for score in raw_scores_percent]}")
        
        print()  # Empty line between folds

def generate_detailed_report(base_directory='.', output_file=None):
    """
    Generate a detailed report with all metrics statistics.
    
    Parameters:
    base_directory (str): Base directory containing class folders
    output_file (str): Optional file to save the report
    """
    
    data = load_training_logs(base_directory)
    
    if not data:
        print("No data found.")
        return
    
    report_lines = []
    report_lines.append("DETAILED F1-SCORE STATISTICS REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Get all unique fold numbers
    all_folds = set()
    for class_data in data.values():
        all_folds.update(class_data.keys())
    
    for fold_num in sorted(all_folds):
        report_lines.append(f"===== {fold_num} FOLDS =====")
        report_lines.append("")
        
        for class_name in Config.CLASSES:
            if class_name in data and fold_num in data[class_name]:
                if 'f1_score' in data[class_name][fold_num]:
                    f1_scores = data[class_name][fold_num]['f1_score']
                    
                    if isinstance(f1_scores, list) and len(f1_scores) > 0:
                        mean_f1 = np.mean(f1_scores)
                        std_f1 = np.std(f1_scores, ddof=1) if len(f1_scores) > 1 else 0.0
                        min_f1 = np.min(f1_scores)
                        max_f1 = np.max(f1_scores)
                        
                        report_lines.append(f"{class_name}:")
                        report_lines.append(f"  Mean: {mean_f1*100:.2f}%")
                        report_lines.append(f"  Std:  {std_f1*100:.2f}%")
                        report_lines.append(f"  Min:  {min_f1*100:.2f}%")
                        report_lines.append(f"  Max:  {max_f1*100:.2f}%")
                        report_lines.append(f"  Raw:  {[f'{score*100:.2f}%' for score in f1_scores]}")
                        report_lines.append("")
        
        report_lines.append("")
    
    for line in report_lines:
        print(line)
    
    # Save to file if specified
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"\nDetailed report saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", help="Directory where logs are saved.")
    args = parser.parse_args()

    print("F1-Score Statistics by Fold")
    print("=" * 40)
    print_f1_statistics(args.logs_dir)
    print("Analysis complete!")
