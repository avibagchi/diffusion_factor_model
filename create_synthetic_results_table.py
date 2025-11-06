#!/usr/bin/env python3
"""
Script to recreate Table 1 from the PDF: Relative error of the estimated top-k eigenvalues
and k-dimensional principal components for varying sample sizes.

This script evaluates the performance of the diffusion model on synthetic data
by comparing generated samples with training data across different sample sizes.

Usage:
    python create_synthetic_results_table.py --generated_data <path_to_generated_data> [options]

Examples:
    # Using a directory of sample batches:
    python create_synthetic_results_table.py --generated_data samples/dfm_training_data_example_ts1762385835_seed42
    
    # With custom parameters:
    python create_synthetic_results_table.py \
        --training_data simulation_experiment_data/training_data_example.npy \
        --generated_data samples/dfm_training_data_example_ts1762385835_seed42 \
        --num_trials 10 \
        --k 5
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# Import directly to avoid __init__.py issues
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval'))
from mean_cov import calculate_mean_cov
from simulation_eval import svd

def load_data(training_data_path, generated_data_path):
    """
    Load and prepare training and generated data.
    
    Args:
        training_data_path: Path to training data .npy file
        generated_data_path: Path to generated data .npy file (can be directory with batch files)
    
    Returns:
        tuple: (training_data, generated_data) both as 2D arrays (samples, features)
    """
    # Load training data
    training_data = np.load(training_data_path)
    
    # Reshape 3D data to 2D if needed (samples, height, width) -> (samples, height*width)
    if len(training_data.shape) == 3:
        samples, height, width = training_data.shape
        training_data = training_data.reshape(samples, height * width)
    
    # Load generated data
    if os.path.isdir(generated_data_path):
        # Load all batch files and concatenate
        batch_files = sorted([f for f in os.listdir(generated_data_path) if f.startswith('sample_batch') and f.endswith('.npy')])
        if not batch_files:
            raise ValueError(f"No sample batch files found in {generated_data_path}")
        
        generated_batches = []
        for batch_file in batch_files:
            batch_data = np.load(os.path.join(generated_data_path, batch_file))
            if len(batch_data.shape) == 3:
                samples, height, width = batch_data.shape
                batch_data = batch_data.reshape(samples, height * width)
            generated_batches.append(batch_data)
        
        generated_data = np.concatenate(generated_batches, axis=0)
    else:
        # Load single file
        generated_data = np.load(generated_data_path)
        if len(generated_data.shape) == 3:
            samples, height, width = generated_data.shape
            generated_data = generated_data.reshape(samples, height * width)
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Generated data shape: {generated_data.shape}")
    
    return training_data, generated_data

def calculate_ground_truth_statistics(training_data):
    """
    Calculate ground truth mean and covariance from training data.
    
    Args:
        training_data: Training data array
    
    Returns:
        tuple: (ground_truth_mean, ground_truth_cov)
    """
    # Use BS=True, OLSE=True, LW=True, clip=0.01 as in test.py
    ground_truth_mean, ground_truth_cov = calculate_mean_cov(
        data=training_data, BS=True, OLSE=True, LW=True, clip=0.01
    )
    return ground_truth_mean, ground_truth_cov

def calculate_relative_error_eigenvalues(sample_cov, ground_truth_cov, k):
    """
    Calculate relative error for top-k eigenvalues.
    
    Args:
        sample_cov: Sample covariance matrix
        ground_truth_cov: Ground truth covariance matrix
        k: Number of top eigenvalues to consider
    
    Returns:
        float: Relative error for eigenvalues
    """
    # Get top-k eigenvalues from ground truth
    U_gt, S_gt, _ = np.linalg.svd(ground_truth_cov)
    ground_truth_eigenvalues = S_gt[:k]
    
    # Get top-k eigenvalues from sample
    U_sample, S_sample, _ = np.linalg.svd(sample_cov)
    sample_eigenvalues = S_sample[:k]
    
    # Calculate relative error: mean(|lambda_sample / lambda_gt - 1|)
    relative_error = np.abs(sample_eigenvalues / ground_truth_eigenvalues - 1).mean()
    
    return relative_error

def calculate_relative_error_principal_components(sample_cov, ground_truth_cov, k):
    """
    Calculate relative error for k-dimensional principal components (Frobenius norm).
    
    Args:
        sample_cov: Sample covariance matrix
        ground_truth_cov: Ground truth covariance matrix
        k: Number of principal components to consider
    
    Returns:
        float: Relative error for principal components (Frobenius norm)
    """
    # Get top-k principal components from ground truth
    U_gt, S_gt, VT_gt = np.linalg.svd(ground_truth_cov)
    U_gt_k = U_gt[:, :k]
    S_gt_k = np.diag(S_gt[:k])
    ground_truth_pc = U_gt_k @ S_gt_k @ VT_gt[:k, :]
    
    # Get top-k principal components from sample
    U_sample, S_sample, VT_sample = np.linalg.svd(sample_cov)
    U_sample_k = U_sample[:, :k]
    S_sample_k = np.diag(S_sample[:k])
    sample_pc = U_sample_k @ S_sample_k @ VT_sample[:k, :]
    
    # Calculate relative error: ||sample_pc - ground_truth_pc||_F / ||ground_truth_pc||_F
    numerator = np.linalg.norm(sample_pc - ground_truth_pc, ord='fro')
    denominator = np.linalg.norm(ground_truth_pc, ord='fro')
    relative_error = numerator / denominator if denominator > 0 else np.nan
    
    return relative_error

def evaluate_sample_size(sample_data, ground_truth_cov, n_samples, k, num_trials=10):
    """
    Evaluate performance for a given sample size by subsampling.
    
    Args:
        sample_data: Full sample data array
        ground_truth_cov: Ground truth covariance matrix
        n_samples: Number of samples to use (will subsample)
        k: Number of top eigenvalues/components
        num_trials: Number of random subsamples to average over
    
    Returns:
        tuple: (mean_eig_error, std_eig_error, mean_pc_error, std_pc_error)
    """
    eig_errors = []
    pc_errors = []
    
    available_samples = sample_data.shape[0]
    
    # If we don't have enough samples, use what we have (but note this limits variance)
    if available_samples < n_samples:
        # Use all available samples, but only if we have at least k+1 samples
        if available_samples < k + 1:
            return np.nan, 0.0, np.nan, 0.0
        # Use all samples for each "trial" (will be identical, but allows code to run)
        actual_trials = min(num_trials, 1)  # Use 1 trial if not enough samples
        subsample = sample_data
    else:
        actual_trials = num_trials
        subsample = None  # Will be created per trial
    
    for trial in range(actual_trials):
        if subsample is None:
            # Random subsample
            indices = np.random.choice(available_samples, size=n_samples, replace=False)
            trial_sample = sample_data[indices]
        else:
            # Use all available samples
            trial_sample = subsample
        
        # Calculate sample covariance
        sample_cov = np.cov(trial_sample.T)
        
        # Calculate errors
        eig_error = calculate_relative_error_eigenvalues(sample_cov, ground_truth_cov, k)
        pc_error = calculate_relative_error_principal_components(sample_cov, ground_truth_cov, k)
        
        eig_errors.append(eig_error)
        pc_errors.append(pc_error)
    
    if len(eig_errors) == 0:
        return np.nan, 0.0, np.nan, 0.0
    
    eig_mean = np.mean(eig_errors)
    eig_std = np.std(eig_errors, ddof=1) if len(eig_errors) > 1 else 0.0
    pc_mean = np.mean(pc_errors)
    pc_std = np.std(pc_errors, ddof=1) if len(pc_errors) > 1 else 0.0
    
    return eig_mean, eig_std, pc_mean, pc_std

def create_table_1(training_data, generated_data, ground_truth_cov, k=5, num_trials=10):
    """
    Create Table 1: Relative error of eigenvalues and principal components for varying sample sizes.
    
    Args:
        training_data: Training data array
        generated_data: Generated data array
        ground_truth_cov: Ground truth covariance matrix
        k: Number of top eigenvalues/components
        num_trials: Number of random subsamples per sample size
    
    Returns:
        pd.DataFrame: Results table matching Table 1 format
    """
    # Sample sizes: 2^9, 2^10, 2^11, 2^12, 2^13
    sample_sizes = [2**9, 2**10, 2**11, 2**12, 2**13]
    
    results = []
    
    print(f"\nCalculating errors for {len(sample_sizes)} sample sizes with {num_trials} trials each...")
    
    for n in sample_sizes:
        print(f"  Processing N = {n} ({2**int(np.log2(n))})...")
        
        # Evaluate diffusion-generated samples
        diff_eig_mean, diff_eig_std, diff_pc_mean, diff_pc_std = evaluate_sample_size(
            generated_data, ground_truth_cov, n, k, num_trials
        )
        
        # Evaluate empirical (training) samples
        emp_eig_mean, emp_eig_std, emp_pc_mean, emp_pc_std = evaluate_sample_size(
            training_data, ground_truth_cov, n, k, num_trials
        )
        
        # Calculate ratios
        eig_ratio = diff_eig_mean / emp_eig_mean if emp_eig_mean > 0 else np.nan
        # For ratio std, we use propagation of errors approximation
        eig_ratio_std = eig_ratio * np.sqrt((diff_eig_std/diff_eig_mean)**2 + (emp_eig_std/emp_eig_mean)**2) if emp_eig_mean > 0 and diff_eig_mean > 0 else 0.0
        
        pc_ratio = diff_pc_mean / emp_pc_mean if emp_pc_mean > 0 else np.nan
        pc_ratio_std = pc_ratio * np.sqrt((diff_pc_std/diff_pc_mean)**2 + (emp_pc_std/emp_pc_mean)**2) if emp_pc_mean > 0 and diff_pc_mean > 0 else 0.0
        
        # Format N as 2^power
        n_formatted = f"2^{int(np.log2(n))}" if n == 2**int(np.log2(n)) else str(n)
        
        # Panel A: Eigenvalues
        results.append({
            'Panel': 'A',
            'N': n_formatted,
            'Diff_RE1': diff_eig_mean,
            'Diff_RE1_std': diff_eig_std,
            'Emp_RE1': emp_eig_mean,
            'Ratio1': eig_ratio,
            'Ratio1_std': eig_ratio_std,
            'Metric': 'Eigenvalues'
        })
        
        # Panel B: Principal Components
        results.append({
            'Panel': 'B',
            'N': n_formatted,
            'Diff_RE2': diff_pc_mean,
            'Diff_RE2_std': diff_pc_std,
            'Emp_RE2': emp_pc_mean,
            'Ratio2': pc_ratio,
            'Ratio2_std': pc_ratio_std,
            'Metric': 'Principal Components'
        })
    
    df = pd.DataFrame(results)
    return df

def print_table_1(df):
    """Print Table 1 in the format matching the PDF."""
    print("\n" + "="*100)
    print("Table 1: Relative error of the estimated top-k eigenvalues and k-dimensional")
    print("         principal components for varying sample sizes (standard deviations in parentheses)")
    print("="*100)
    
    # Panel A: Eigenvalues
    print("\nPanel A: Eigenvalues")
    print("-" * 100)
    print(f"{'N':<10} {'Diff RE₁':<20} {'Emp RE₁':<15} {'Diff RE₁/Emp RE₁':<25}")
    print("-" * 100)
    
    panel_a = df[df['Panel'] == 'A']
    for _, row in panel_a.iterrows():
        diff_re = f"{row['Diff_RE1']:.3f} (± {row['Diff_RE1_std']:.3f})"
        emp_re = f"{row['Emp_RE1']:.3f}"
        ratio = f"{row['Ratio1']:.3f} (± {row['Ratio1_std']:.3f})"
        print(f"{row['N']:<10} {diff_re:<25} {emp_re:<15} {ratio:<30}")
    
    # Panel B: Principal Components
    print("\nPanel B: Principal Components")
    print("-" * 100)
    print(f"{'N':<10} {'Diff RE₂':<25} {'Emp RE₂':<15} {'Diff RE₂/Emp RE₂':<30}")
    print("-" * 100)
    
    panel_b = df[df['Panel'] == 'B']
    for _, row in panel_b.iterrows():
        diff_re = f"{row['Diff_RE2']:.3f} (± {row['Diff_RE2_std']:.3f})"
        emp_re = f"{row['Emp_RE2']:.3f}"
        ratio = f"{row['Ratio2']:.3f} (± {row['Ratio2_std']:.3f})"
        print(f"{row['N']:<10} {diff_re:<25} {emp_re:<15} {ratio:<30}")
    
    print("="*100)
    
    # Also create LaTeX format
    print("\nLaTeX Format:")
    print("="*100)
    
    # Create LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\caption{Relative error of the estimated top-k eigenvalues and k-dimensional principal components for varying sample sizes (standard deviations in parentheses).}")
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\multicolumn{4}{c}{Panel A: Eigenvalues} \\\\")
    latex_lines.append("\\midrule")
    latex_lines.append("N & Diff RE$_1$ & Emp RE$_1$ & Diff RE$_1$/Emp RE$_1$ \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in panel_a.iterrows():
        diff_re = f"{row['Diff_RE1']:.3f} ($\\pm$ {row['Diff_RE1_std']:.3f})"
        emp_re = f"{row['Emp_RE1']:.3f}"
        ratio = f"{row['Ratio1']:.3f} ($\\pm$ {row['Ratio1_std']:.3f})"
        n_str = row['N']
        if '^' in n_str:
            power = int(n_str.split('^')[1])
            n_val = f"$2^{{{power}}}$"
        else:
            n_val = n_str
        latex_lines.append(f"{n_val} & {diff_re} & {emp_re} & {ratio} \\\\")
    
    latex_lines.append("\\midrule")
    latex_lines.append("\\multicolumn{4}{c}{Panel B: Principal Components} \\\\")
    latex_lines.append("\\midrule")
    latex_lines.append("N & Diff RE$_2$ & Emp RE$_2$ & Diff RE$_2$/Emp RE$_2$ \\\\")
    latex_lines.append("\\midrule")
    
    for _, row in panel_b.iterrows():
        diff_re = f"{row['Diff_RE2']:.3f} ($\\pm$ {row['Diff_RE2_std']:.3f})"
        emp_re = f"{row['Emp_RE2']:.3f}"
        ratio = f"{row['Ratio2']:.3f} ($\\pm$ {row['Ratio2_std']:.3f})"
        n_str = row['N']
        if '^' in n_str:
            power = int(n_str.split('^')[1])
            n_val = f"$2^{{{power}}}$"
        else:
            n_val = n_str
        latex_lines.append(f"{n_val} & {diff_re} & {emp_re} & {ratio} \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    print("\n".join(latex_lines))
    print("="*100)

def main():
    """Main function to run the table generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Table 1: Relative error of eigenvalues and principal components')
    parser.add_argument('--training_data', type=str, 
                       default='simulation_experiment_data/training_data_example.npy',
                       help='Path to training data .npy file')
    parser.add_argument('--generated_data', type=str, required=True,
                       help='Path to generated data .npy file or directory containing sample batches')
    parser.add_argument('--output', type=str, default='table1_results.csv',
                       help='Output CSV file path')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of top eigenvalues/components (default: 5)')
    parser.add_argument('--num_trials', type=int, default=10,
                       help='Number of random subsamples per sample size (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load data
    print("Loading data...")
    training_data, generated_data = load_data(args.training_data, args.generated_data)
    
    # Calculate ground truth statistics
    print("Calculating ground truth statistics...")
    ground_truth_mean, ground_truth_cov = calculate_ground_truth_statistics(training_data)
    
    # Create Table 1
    print(f"\nCreating Table 1 with k={args.k}, num_trials={args.num_trials}...")
    results_df = create_table_1(training_data, generated_data, ground_truth_cov, k=args.k, num_trials=args.num_trials)
    
    # Print table
    print_table_1(results_df)
    
    # Save to CSV
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
