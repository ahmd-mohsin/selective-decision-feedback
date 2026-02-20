#!/usr/bin/env python3
"""
SNR Sweep Evaluation: Compare all channel estimation methods across different SNR values
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from decision_directed.config import DecisionDirectedConfig
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample
from transmission_system.receiver_frontend import compute_channel_nmse


def evaluate_at_snr(
    snr_db: float,
    diffusion_checkpoint: str,
    num_samples: int = 50,
    device: str = 'cuda'
):
    """Evaluate all methods at a specific SNR"""
    
    # Create config for this SNR
    config = TransmissionConfig(
        Nfft=64,
        Ncp=16,
        Nsym=100,
        pilot_spacing=4,
        modulation_order=16,
        snr_db=snr_db,
        doppler_hz=100.0,
        time_varying=True,
        seed=42
    )
    
    # Setup estimator
    dd_config = DecisionDirectedConfig(
        llr_threshold=12.0,  # Higher threshold to be more selective
        adaptive_threshold=False
    )
    
    estimator = IntegratedEstimator(
        diffusion_checkpoint=diffusion_checkpoint,
        dd_config=dd_config,
        modulation_order=16,
        device=device
    )
    
    results = {
        'pilot_only': [],
        'diffusion_only': [],
        'dd_only': [],
        'full_pipeline': [],
        'acceptance_rates': []
    }
    
    rng = np.random.default_rng(config.seed)
    
    for i in tqdm(range(num_samples), desc=f"SNR={snr_db:2.0f}dB", leave=False):
        # Generate single sample at this SNR
        sample = generate_single_sample(config, rng, i)
        
        H_true = sample['H_true']
        Y_grid = sample['Y_grid']
        X_grid = sample['X_grid']
        pilot_mask = sample['pilot_mask']
        H_pilot_full = sample['H_pilot_full']
        
        # 1. Pilot Only
        nmse_pilot = compute_channel_nmse(H_pilot_full, H_true)
        results['pilot_only'].append(nmse_pilot)
        
        # 2. Diffusion Only
        try:
            diff_result = estimator.estimate_diffusion_only(Y_grid, H_pilot_full, pilot_mask)
            nmse_diff = compute_channel_nmse(diff_result['H_estimate'], H_true)
            results['diffusion_only'].append(nmse_diff)
        except Exception as e:
            print(f"\nWarning: Diffusion failed at SNR={snr_db}dB, sample {i}: {e}")
            results['diffusion_only'].append(nmse_pilot)  # Fallback
        
        # 3. DD Only
        pilot_positions = np.where(pilot_mask)
        pilot_errors = Y_grid[pilot_positions] - H_pilot_full[pilot_positions] * X_grid[pilot_positions]
        noise_var = np.mean(np.abs(pilot_errors)**2)
        
        dd_result = estimator.estimate_dd_only(H_pilot_full, Y_grid, X_grid, pilot_mask, noise_var)
        nmse_dd = compute_channel_nmse(dd_result['H_estimate'], H_true)
        results['dd_only'].append(nmse_dd)
        
        # 4. Full Pipeline (DD before Diffusion)
        try:
            full_result = estimator.estimate_full_pipeline(
                Y_grid, X_grid, pilot_mask, H_pilot_full, noise_var,
                num_iterations=1, use_dd_before_diffusion=True
            )
            nmse_full = compute_channel_nmse(full_result['H_final'], H_true)
            results['full_pipeline'].append(nmse_full)
            
            if 'acceptance_rates' in full_result:
                results['acceptance_rates'].append(np.mean(full_result['acceptance_rates']))
        except Exception as e:
            print(f"\nWarning: Full pipeline failed at SNR={snr_db}dB, sample {i}: {e}")
            results['full_pipeline'].append(nmse_diff if len(results['diffusion_only']) > 0 else nmse_pilot)
    
    # Convert to dB
    return {
        'pilot_only': 10 * np.log10(np.mean(results['pilot_only'])),
        'diffusion_only': 10 * np.log10(np.mean(results['diffusion_only'])),
        'dd_only': 10 * np.log10(np.mean(results['dd_only'])),
        'full_pipeline': 10 * np.log10(np.mean(results['full_pipeline'])),
        'acceptance_rate': np.mean(results['acceptance_rates']) if results['acceptance_rates'] else 0.0
    }


def plot_snr_sweep(snr_values, results_dict, output_path):
    """Plot NMSE vs SNR for all methods"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: NMSE vs SNR
    methods = ['pilot_only', 'diffusion_only', 'dd_only', 'full_pipeline']
    labels = ['Pilot Only', 'Diffusion Only', 'DD Only', 'Full Pipeline (DD+Diffusion)']
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71']
    markers = ['o', 's', '^', 'D']
    
    for method, label, color, marker in zip(methods, labels, colors, markers):
        nmse_values = [results_dict[snr][method] for snr in snr_values]
        ax1.plot(snr_values, nmse_values, marker=marker, linewidth=2.5, 
                markersize=8, label=label, color=color, alpha=0.9)
    
    ax1.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('NMSE (dB)', fontsize=14, fontweight='bold')
    ax1.set_title('Channel Estimation Performance vs SNR', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.set_xlim([min(snr_values)-2, max(snr_values)+2])
    
    # Plot 2: Improvement over Pilot Only
    for method, label, color, marker in zip(methods[1:], labels[1:], colors[1:], markers[1:]):
        pilot_nmse = [results_dict[snr]['pilot_only'] for snr in snr_values]
        method_nmse = [results_dict[snr][method] for snr in snr_values]
        improvement = [pilot - method for pilot, method in zip(pilot_nmse, method_nmse)]
        ax2.plot(snr_values, improvement, marker=marker, linewidth=2.5,
                markersize=8, label=label, color=color, alpha=0.9)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Improvement over Pilot Only (dB)', fontsize=14, fontweight='bold')
    ax2.set_title('Gain Compared to Baseline', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='upper left')
    ax2.set_xlim([min(snr_values)-2, max(snr_values)+2])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def save_results_table(snr_values, results_dict, output_path):
    """Save detailed results to text file"""
    
    with open(output_path, 'w') as f:
        f.write("SNR Sweep Results - Channel Estimation Performance\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'SNR (dB)':<10} {'Pilot Only':<15} {'Diffusion':<15} {'DD Only':<15} "
                f"{'Full Pipeline':<15} {'Accept %':<12}\n")
        f.write("-" * 100 + "\n")
        
        for snr in snr_values:
            res = results_dict[snr]
            f.write(f"{snr:<10.0f} {res['pilot_only']:<15.2f} {res['diffusion_only']:<15.2f} "
                   f"{res['dd_only']:<15.2f} {res['full_pipeline']:<15.2f} "
                   f"{res['acceptance_rate']*100:<12.1f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("Improvements over Pilot Only (positive = better)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'SNR (dB)':<10} {'Diffusion Gain':<18} {'DD Gain':<18} {'Full Pipeline Gain':<20}\n")
        f.write("-" * 100 + "\n")
        
        for snr in snr_values:
            res = results_dict[snr]
            diff_gain = res['pilot_only'] - res['diffusion_only']
            dd_gain = res['pilot_only'] - res['dd_only']
            full_gain = res['pilot_only'] - res['full_pipeline']
            f.write(f"{snr:<10.0f} {diff_gain:<18.2f} {dd_gain:<18.2f} {full_gain:<20.2f}\n")
    
    print(f"Results table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SNR Sweep Evaluation for Channel Estimation')
    
    parser.add_argument('--diffusion_checkpoint', type=str, required=True,
                       help='Path to diffusion model checkpoint')
    parser.add_argument('--snr_min', type=int, default=5,
                       help='Minimum SNR in dB (default: 5)')
    parser.add_argument('--snr_max', type=int, default=30,
                       help='Maximum SNR in dB (default: 30)')
    parser.add_argument('--snr_step', type=int, default=5,
                       help='SNR step size in dB (default: 5)')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples per SNR (default: 50)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')
    parser.add_argument('--output_dir', type=str, default='./results/snr_sweep',
                       help='Output directory (default: ./results/snr_sweep)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate SNR values
    snr_values = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    
    print("=" * 80)
    print("SNR SWEEP EVALUATION - CHANNEL ESTIMATION")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Diffusion Checkpoint: {args.diffusion_checkpoint}")
    print(f"  SNR Range: {args.snr_min} to {args.snr_max} dB (step {args.snr_step} dB)")
    print(f"  Samples per SNR: {args.num_samples}")
    print(f"  Device: {args.device}")
    print(f"  Output Directory: {args.output_dir}")
    print(f"\n{'='*80}\n")
    
    results_dict = {}
    
    # Evaluate at each SNR
    for snr in tqdm(snr_values, desc="SNR Sweep"):
        results_dict[snr] = evaluate_at_snr(
            snr,
            args.diffusion_checkpoint,
            args.num_samples,
            args.device
        )
    
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print("=" * 80 + "\n")
    
    print(f"{'SNR (dB)':<10} {'Pilot Only':<15} {'Diffusion':<15} {'DD Only':<15} {'Full Pipeline':<15}")
    print("-" * 80)
    for snr in snr_values:
        res = results_dict[snr]
        print(f"{snr:<10.0f} {res['pilot_only']:<15.2f} {res['diffusion_only']:<15.2f} "
              f"{res['dd_only']:<15.2f} {res['full_pipeline']:<15.2f}")
    
    # Save results
    plot_path = os.path.join(args.output_dir, 'snr_sweep.png')
    plot_snr_sweep(snr_values, results_dict, plot_path)
    
    table_path = os.path.join(args.output_dir, 'snr_sweep_results.txt')
    save_results_table(snr_values, results_dict, table_path)
    
    print(f"\n{'='*80}")
    print("SNR SWEEP COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
