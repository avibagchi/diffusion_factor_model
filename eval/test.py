import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from mean_cov import calculate_mean_cov

if __name__ == "__main__":
    # Load data
    training_data = np.load('/home/ubuntu/diffustionproject/diffusion_factor_model/simulation_experiment_data/training_data_example.npy')
    generated_data = np.load('/home/ubuntu/diffustionproject/diffusion_factor_model/samples/dfm_training_data_example_ts1761971629_seed42/sample_batch1.npy')
    
    # Reshape 3D data to 2D if needed (samples, height, width) -> (samples, height*width)
    if len(training_data.shape) == 3:
        samples, height, width = training_data.shape
        training_data = training_data.reshape(samples, height * width)
    
    if len(generated_data.shape) == 3:
        samples, height, width = generated_data.shape
        generated_data = generated_data.reshape(samples, height * width)
    
    # Calculate mean and covariance
    mean_cov_result = calculate_mean_cov(data=training_data, BS=True, OLSE=True, LW=True, clip=0.01)
    ground_truth_mean = mean_cov_result[0]
    ground_truth_cov = mean_cov_result[1]
    
    # Find asset indices
    variances = np.diag(ground_truth_cov)  # Diagonal of covariance matrix gives variances
    largest_variance_idx = np.argmax(variances)
    smallest_variance_idx = np.argmin(variances)
    largest_mean_idx = np.argmax(ground_truth_mean)
    smallest_mean_idx = np.argmin(ground_truth_mean)
    
    print(f"Largest variance asset index: {largest_variance_idx}")
    print(f"Smallest variance asset index: {smallest_variance_idx}")
    print(f"Largest mean asset index: {largest_mean_idx}")
    print(f"Smallest mean asset index: {smallest_mean_idx}")
    
    # Create figure with 4 rows (asset types) x 2 columns (Generated, Training)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16), dpi=400)
    
    # Define the 4 assets to plot
    assets = [
        (largest_variance_idx, "Largest Variance"),
        (smallest_variance_idx, "Smallest Variance"),
        (largest_mean_idx, "Largest Mean"),
        (smallest_mean_idx, "Smallest Mean")
    ]
    
    bins_num = 100
    
    # Plot each asset with side-by-side Generated and Training
    for row_idx, (asset_idx, title) in enumerate(assets):
        # Get data for this asset
        gen_data = generated_data[:, asset_idx]
        train_data = training_data[:, asset_idx]
        
        # Calculate x-axis bounds: center on median, edges at min and max
        all_data = np.concatenate([gen_data, train_data])
        x_min = np.min(all_data)  # Most extreme minimum
        x_max = np.max(all_data)  # Most extreme maximum
        x_median = np.median(all_data)  # Center value
        
        # Determine symmetric bounds around median
        # Find the distance from median to each extreme
        dist_to_min = abs(x_median - x_min)
        dist_to_max = abs(x_max - x_median)
        max_dist = max(dist_to_min, dist_to_max)
        
        # Set bounds symmetrically around median
        x_bound_left = x_median - max_dist
        x_bound_right = x_median + max_dist
        
        # Create bin edges covering the full x-axis range
        bin_edges = np.linspace(x_bound_left, x_bound_right, bins_num + 1)
        
        # Create x-axis ticks (round to nice values)
        # Find nice tick spacing
        x_range = x_bound_right - x_bound_left
        tick_spacing = 10 ** np.floor(np.log10(x_range))  # Rough spacing
        if x_range / tick_spacing > 10:
            tick_spacing *= 2
        elif x_range / tick_spacing < 4:
            tick_spacing /= 2
        
        # Generate ticks around median
        num_ticks_left = int(np.ceil((x_median - x_bound_left) / tick_spacing))
        num_ticks_right = int(np.ceil((x_bound_right - x_median) / tick_spacing))
        x_ticks = np.concatenate([
            np.arange(x_median - num_ticks_left * tick_spacing, x_median, tick_spacing),
            np.arange(x_median, x_median + (num_ticks_right + 1) * tick_spacing, tick_spacing)
        ])
        # Ensure we include the median
        if x_median not in x_ticks:
            x_ticks = np.append(x_ticks, x_median)
        x_ticks = np.sort(x_ticks)
        # Filter ticks to be within bounds
        x_ticks = x_ticks[(x_ticks >= x_bound_left) & (x_ticks <= x_bound_right)]
        
        # Plot ground truth normal distribution (same for both)
        return_cdf = norm.cdf(bin_edges, ground_truth_mean[asset_idx], np.sqrt(ground_truth_cov[asset_idx, asset_idx]))
        cdf_diff = np.diff(return_cdf)
        
        # Plot Generated data (left column)
        ax_gen = axes[row_idx, 0]
        gen_hist, _ = np.histogram(gen_data, bins=bin_edges, density=False)
        gen_prop = gen_hist / len(gen_data) if len(gen_data) > 0 else gen_hist
        
        sns.histplot(ax=ax_gen, data=gen_data, bins=bin_edges, alpha=1,
                    stat="proportion", color="C0", label="Generated")
        ax_gen.plot(bin_edges[1:], cdf_diff, label='Ground truth', color="C3", linestyle="--", linewidth=3)
        
        # Calculate y-axis bound for Generated - match tallest actual value
        max_gen_prop = gen_prop.max() if len(gen_prop) > 0 else 0
        max_cdf_diff = cdf_diff.max()
        y_bound_gen = max(max_gen_prop, max_cdf_diff)  # No padding, use actual max
        
        # Generate y-ticks - nice spacing based on the bound
        if y_bound_gen <= 0.05:
            y_ticks_gen = np.arange(0, 0.06, 0.01)
        else:
            # Create 6 ticks
            y_ticks_gen = np.linspace(0, y_bound_gen, 6)
        
        ax_gen.set_xlim(x_bound_left, x_bound_right)
        ax_gen.set_xticks(x_ticks)
        ax_gen.tick_params(axis='x', labelsize=11)
        ax_gen.set_ylim(0, y_bound_gen)
        ax_gen.set_yticks(y_ticks_gen)
        ax_gen.tick_params(axis='y', labelsize=11)
        if row_idx == 3:  # Only label bottom row
            ax_gen.set_xlabel("Return", fontsize=12)
        ax_gen.set_ylabel("Frequency", fontsize=12)
        ax_gen.set_title(f"({chr(97+row_idx)}) Generated - Asset with {title}", fontsize=13, fontweight='bold')
        ax_gen.legend(fontsize=9, loc='upper right')
        
        # Set border style
        for spine in ax_gen.spines.values():
            spine.set_color("black")
            spine.set_linestyle("-")
            spine.set_linewidth(1)
        
        # Plot Training data (right column)
        ax_train = axes[row_idx, 1]
        train_hist, _ = np.histogram(train_data, bins=bin_edges, density=False)
        train_prop = train_hist / len(train_data) if len(train_data) > 0 else train_hist
        
        sns.histplot(ax=ax_train, data=train_data, bins=bin_edges, alpha=1,
                    stat="proportion", color="C2", label="Training")
        ax_train.plot(bin_edges[1:], cdf_diff, label='Ground truth', color="C3", linestyle="--", linewidth=3)
        
        # Calculate y-axis bound for Training - match tallest actual value
        max_train_prop = train_prop.max() if len(train_prop) > 0 else 0
        y_bound_train = max(max_train_prop, max_cdf_diff)  # No padding, use actual max
        
        # Generate y-ticks - nice spacing based on the bound
        if y_bound_train <= 0.05:
            y_ticks_train = np.arange(0, 0.06, 0.01)
        else:
            num_ticks = 6
            y_ticks_train = np.linspace(0, y_bound_train, num_ticks)
        
        ax_train.set_xlim(x_bound_left, x_bound_right)
        ax_train.set_xticks(x_ticks)
        ax_train.tick_params(axis='x', labelsize=11)
        ax_train.set_ylim(0, y_bound_train)
        ax_train.set_yticks(y_ticks_train)
        ax_train.tick_params(axis='y', labelsize=11)
        if row_idx == 3:  # Only label bottom row
            ax_train.set_xlabel("Return", fontsize=12)
        ax_train.set_ylabel("Frequency", fontsize=12)
        ax_train.set_title(f"({chr(97+row_idx)}) Training - Asset with {title}", fontsize=13, fontweight='bold')
        ax_train.legend(fontsize=9, loc='upper right')
        
        # Set border style
        for spine in ax_train.spines.values():
            spine.set_color("black")
            spine.set_linestyle("-")
            spine.set_linewidth(1)
        
        print(f"{title}: x_range=[{x_bound_left:.3f}, {x_bound_right:.3f}], median={x_median:.3f}, y_max_gen={y_bound_gen:.4f}, y_max_train={y_bound_train:.4f}")
    
    plt.tight_layout()
    plt.savefig('comparison_histplot_simulation.png', dpi=400, bbox_inches='tight')
    plt.show()
    print("Figure saved as 'comparison_histplot_simulation.png'")