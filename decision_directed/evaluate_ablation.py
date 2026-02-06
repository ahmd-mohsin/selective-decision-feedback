import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from decision_directed.config import DecisionDirectedConfig
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.dataset_generator import load_dataset_hdf5
from transmission_system.receiver_frontend import compute_channel_nmse


def evaluate_ablation(
    dataset_path: str,
    diffusion_checkpoint: Optional[str],
    num_samples: int = 100,
    device: str = 'cuda'
):
    
    data, config = load_dataset_hdf5(dataset_path)
    
    dd_config = DecisionDirectedConfig(
        llr_threshold=4.0,
        normalizer_step_size=0.01,
        noise_step_size=0.05
    )
    
    estimator = IntegratedEstimator(
        diffusion_checkpoint=diffusion_checkpoint,
        dd_config=dd_config,
        modulation_order=config.modulation_order,
        device=device
    )
    
    results = {
        'pilot_only': [],
        'diffusion_only': [],
        'dd_only': [],
        'full_pipeline': []
    }
    
    num_samples = min(num_samples, data['H_true'].shape[0])
    
    for idx in tqdm(range(num_samples), desc="Evaluating"):
        H_true = data['H_true'][idx]
        Y_grid = data['Y_grid'][idx]
        pilot_mask = data['pilot_mask'][idx]
        H_pilot = data['H_pilot_full'][idx]
        noise_var = data['noise_var'][idx]
        
        X_grid = np.zeros_like(Y_grid)
        pilot_positions = np.where(pilot_mask)
        X_grid[pilot_positions] = np.exp(1j * np.random.uniform(0, 2*np.pi, len(pilot_positions[0])))
        
        pilot_result = estimator.estimate_pilot_only(Y_grid, X_grid, pilot_mask, noise_var)
        nmse_pilot = compute_channel_nmse(pilot_result['H_estimate'], H_true)
        results['pilot_only'].append(nmse_pilot)
        
        if diffusion_checkpoint is not None:
            diffusion_result = estimator.estimate_diffusion_only(Y_grid, H_pilot, pilot_mask)
            nmse_diffusion = compute_channel_nmse(diffusion_result['H_estimate'], H_true)
            results['diffusion_only'].append(nmse_diffusion)
        
        dd_result = estimator.estimate_dd_only(H_pilot, Y_grid, X_grid, pilot_mask, noise_var)
        nmse_dd = compute_channel_nmse(dd_result['H_estimate'], H_true)
        results['dd_only'].append(nmse_dd)
        
        full_result = estimator.estimate_full_pipeline(Y_grid, X_grid, pilot_mask, H_pilot, noise_var)
        nmse_full = compute_channel_nmse(full_result['H_final'], H_true)
        results['full_pipeline'].append(nmse_full)
    
    return results


def plot_ablation_results(results: dict, output_path: str):
    
    methods = []
    nmse_means = []
    nmse_stds = []
    
    for method, nmse_list in results.items():
        if len(nmse_list) > 0:
            methods.append(method.replace('_', ' ').title())
            nmse_db = 10 * np.log10(nmse_list)
            nmse_means.append(np.mean(nmse_db))
            nmse_stds.append(np.std(nmse_db))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(methods))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    bars = ax.bar(x_pos, nmse_means, yerr=nmse_stds, capsize=10, 
                   color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('NMSE (dB)', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Channel Estimation Performance', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (mean_val, std_val) in enumerate(zip(nmse_means, nmse_stds)):
        ax.text(i, mean_val + std_val + 0.5, f'{mean_val:.2f} dB', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()


def save_results_table(results: dict, output_path: str):
    
    with open(output_path, 'w') as f:
        f.write("Ablation Study Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'Method':<25} {'NMSE (dB)':<15} {'Std Dev (dB)':<15} {'Samples':<10}\n")
        f.write("-" * 80 + "\n")
        
        for method, nmse_list in results.items():
            if len(nmse_list) > 0:
                nmse_db = 10 * np.log10(nmse_list)
                mean_nmse = np.mean(nmse_db)
                std_nmse = np.std(nmse_db)
                
                f.write(f"{method.replace('_', ' ').title():<25} {mean_nmse:<15.2f} {std_nmse:<15.2f} {len(nmse_list):<10}\n")
        
        f.write("\n")
    
    print(f"Results table saved to: {output_path}")


def main():
    
    parser = argparse.ArgumentParser(description='Ablation study for integrated channel estimation')
    
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--diffusion_checkpoint', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./results/ablation')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("ABLATION STUDY - INTEGRATED CHANNEL ESTIMATION")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Diffusion Checkpoint: {args.diffusion_checkpoint}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Device: {args.device}")
    
    print("\n" + "=" * 70)
    print("RUNNING EVALUATION")
    print("=" * 70 + "\n")
    
    results = evaluate_ablation(
        args.dataset,
        args.diffusion_checkpoint,
        args.num_samples,
        args.device
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70 + "\n")
    
    for method, nmse_list in results.items():
        if len(nmse_list) > 0:
            nmse_db = 10 * np.log10(nmse_list)
            mean_nmse = np.mean(nmse_db)
            std_nmse = np.std(nmse_db)
            print(f"{method.replace('_', ' ').title():<25} {mean_nmse:>10.2f} dB  (±{std_nmse:.2f} dB)")
    
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70 + "\n")
    
    plot_path = os.path.join(args.output_dir, 'ablation_study.png')
    plot_ablation_results(results, plot_path)
    
    table_path = os.path.join(args.output_dir, 'ablation_results.txt')
    save_results_table(results, table_path)
    
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()