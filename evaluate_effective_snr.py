"""
Calculate Effective SNR (Post-Equalization SNR) for Channel Estimation

This metric measures: Signal Power / Residual Error Power
- More practical than NMSE
- Shows how good the channel estimate is for actual data decoding
"""

import numpy as np
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample
from decision_directed.pipeline import IntegratedEstimator
from decision_directed.config import DecisionDirectedConfig as DDConfig


def calculate_effective_snr(Y_grid, X_grid, H_estimated, pilot_mask):
    """
    Calculate effective SNR after equalization with estimated channel.
    
    Effective SNR = Signal_Power / Residual_Error_Power
    
    Where:
    - Signal = H_true * X (what we want)
    - Residual Error = Y - H_estimated * X (estimation error + noise)
    
    Returns effective SNR in dB, calculated only on data positions.
    """
    # Calculate at data positions only (where we don't know X in advance)
    data_mask = ~pilot_mask
    
    # Signal power (assuming unit power symbols)
    # E[|H * X|^2] = E[|H|^2] * E[|X|^2] ≈ E[|H|^2] for normalized symbols
    signal_power = np.mean(np.abs(H_estimated[data_mask])**2)
    
    # Residual error after equalization with estimated channel
    # Y = H_true * X + N
    # Ŷ = H_estimated * X
    # Residual = Y - Ŷ = (H_true - H_estimated) * X + N
    Y_estimated = H_estimated * X_grid
    residual = Y_grid - Y_estimated
    
    # Error power (on data positions)
    error_power = np.mean(np.abs(residual[data_mask])**2)
    
    # Effective SNR
    if error_power > 0:
        eff_snr = signal_power / error_power
        eff_snr_db = 10 * np.log10(eff_snr)
    else:
        eff_snr_db = float('inf')
    
    return eff_snr_db


def calculate_symbol_error_rate(Y_grid, X_grid, H_estimated, pilot_mask, modulation_order=16):
    """
    Calculate Symbol Error Rate (SER) after equalization.
    Practical metric: How many symbols are decoded incorrectly?
    """
    from transmission_system.constellation import get_constellation
    
    # Get constellation points
    constellation = get_constellation(modulation_order, 'qam')
    
    # Equalize at data positions
    data_mask = ~pilot_mask
    Y_data = Y_grid[data_mask]
    X_true = X_grid[data_mask]
    H_est = H_estimated[data_mask]
    
    # Equalize
    X_equalized = Y_data / (H_est + 1e-10)
    
    # Hard decisions
    distances = np.abs(X_equalized[:, None] - constellation[None, :])
    decisions_idx = np.argmin(distances, axis=1)
    X_decisions = constellation[decisions_idx]
    
    # Calculate SER
    true_idx = np.argmin(np.abs(X_true[:, None] - constellation[None, :]), axis=1)
    errors = np.sum(decisions_idx != true_idx)
    ser = errors / len(X_true)
    
    return ser


def evaluate_effective_snr_at_snr(
    snr_db,
    num_samples,
    config,
    dd_config,
    pipeline
):
    """Evaluate effective SNR and SER at a given input SNR."""
    
    config.snr_db = snr_db
    rng = np.random.default_rng(42 + int(snr_db * 100))
    
    results = {
        'pilot_only': {'eff_snr': [], 'ser': []},
        'diffusion_only': {'eff_snr': [], 'ser': []},
        'full_pipeline': {'eff_snr': [], 'ser': []}
    }
    
    for i in tqdm(range(num_samples), desc=f"SNR={snr_db:2.0f}dB", leave=False):
        sample = generate_single_sample(config, rng, i)
        
        Y_grid = sample['Y_grid']
        X_grid = sample['X_grid']
        H_true = sample['H_true']
        pilot_mask = sample['pilot_mask']
        noise_var = sample['noise_var']
        
        # 1. Pilot Only
        H_pilot = sample['H_pilot_full']
        eff_snr_pilot = calculate_effective_snr(Y_grid, X_grid, H_pilot, pilot_mask)
        ser_pilot = calculate_symbol_error_rate(Y_grid, X_grid, H_pilot, pilot_mask)
        results['pilot_only']['eff_snr'].append(eff_snr_pilot)
        results['pilot_only']['ser'].append(ser_pilot)
        
        # 2. Diffusion Only
        try:
            diffusion_result = pipeline.estimate_diffusion_only(
                Y_grid, H_pilot, pilot_mask
            )
            H_diffusion = diffusion_result['H_estimate']
            eff_snr_diff = calculate_effective_snr(Y_grid, X_grid, H_diffusion, pilot_mask)
            ser_diff = calculate_symbol_error_rate(Y_grid, X_grid, H_diffusion, pilot_mask)
            results['diffusion_only']['eff_snr'].append(eff_snr_diff)
            results['diffusion_only']['ser'].append(ser_diff)
        except Exception as e:
            results['diffusion_only']['eff_snr'].append(eff_snr_pilot)
            results['diffusion_only']['ser'].append(ser_pilot)
        
        # 3. Full Pipeline
        try:
            full_result = pipeline.estimate_full_pipeline(
                Y_grid, X_grid, pilot_mask, noise_var,
                num_iterations=1, use_dd_before_diffusion=True
            )
            H_full = full_result['H_final']
            eff_snr_full = calculate_effective_snr(Y_grid, X_grid, H_full, pilot_mask)
            ser_full = calculate_symbol_error_rate(Y_grid, X_grid, H_full, pilot_mask)
            results['full_pipeline']['eff_snr'].append(eff_snr_full)
            results['full_pipeline']['ser'].append(ser_full)
        except Exception as e:
            results['full_pipeline']['eff_snr'].append(eff_snr_diff if 'eff_snr_diff' in locals() else eff_snr_pilot)
            results['full_pipeline']['ser'].append(ser_diff if 'ser_diff' in locals() else ser_pilot)
    
    # Average results
    summary = {}
    for method in results:
        summary[method] = {
            'eff_snr_mean': np.mean(results[method]['eff_snr']),
            'eff_snr_std': np.std(results[method]['eff_snr']),
            'ser_mean': np.mean(results[method]['ser']),
            'ser_std': np.std(results[method]['ser'])
        }
    
    return summary


def plot_effective_snr_results(snr_values, results_dict, output_dir):
    """Plot effective SNR and SER vs input SNR."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    methods = ['pilot_only', 'diffusion_only', 'full_pipeline']
    labels = ['Pilot Only', 'Diffusion Only', 'Full Pipeline (DD+Diffusion)']
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    markers = ['o', 's', 'D']
    
    # Plot 1: Effective SNR
    for method, label, color, marker in zip(methods, labels, colors, markers):
        eff_snr_values = [results_dict[snr][method]['eff_snr_mean'] for snr in snr_values]
        ax1.plot(snr_values, eff_snr_values, marker=marker, linewidth=2.5, 
                markersize=8, label=label, color=color)
    
    ax1.set_xlabel('Input SNR (dB)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Effective SNR (dB)', fontsize=14, fontweight='bold')
    ax1.set_title('Effective SNR: Signal Power / Residual Error Power', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Add diagonal reference line (ideal case: output = input)
    ax1.plot(snr_values, snr_values, 'k--', linewidth=1.5, alpha=0.5, label='Ideal (no estimation error)')
    ax1.legend(fontsize=11, loc='lower right')
    
    # Plot 2: Symbol Error Rate
    for method, label, color, marker in zip(methods, labels, colors, markers):
        ser_values = [results_dict[snr][method]['ser_mean'] * 100 for snr in snr_values]
        ax2.semilogy(snr_values, ser_values, marker=marker, linewidth=2.5,
                    markersize=8, label=label, color=color)
    
    ax2.set_xlabel('Input SNR (dB)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Symbol Error Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_title('Symbol Error Rate After Equalization', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/effective_snr_results.png', dpi=150, bbox_inches='tight')
    plt.close()


def save_effective_snr_table(snr_values, results_dict, output_dir):
    """Save results table."""
    
    with open(f'{output_dir}/effective_snr_results.txt', 'w') as f:
        f.write("Effective SNR Results (Practical Metric for Real Systems)\n")
        f.write("=" * 110 + "\n\n")
        
        f.write("EFFECTIVE SNR (Signal Power / Residual Error Power) in dB\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Input SNR':>10}  {'Pilot Only':>12}  {'Diffusion':>12}  {'Full Pipeline':>15}  {'Gain vs Pilot':>15}\n")
        f.write("-" * 110 + "\n")
        
        for snr in snr_values:
            pilot_eff = results_dict[snr]['pilot_only']['eff_snr_mean']
            diff_eff = results_dict[snr]['diffusion_only']['eff_snr_mean']
            full_eff = results_dict[snr]['full_pipeline']['eff_snr_mean']
            gain = full_eff - pilot_eff
            
            f.write(f"{snr:>10.0f}  {pilot_eff:>12.2f}  {diff_eff:>12.2f}  {full_eff:>15.2f}  {gain:>+15.2f}\n")
        
        f.write("\n\n")
        f.write("SYMBOL ERROR RATE (%) After Equalization\n")
        f.write("-" * 110 + "\n")
        f.write(f"{'Input SNR':>10}  {'Pilot Only':>12}  {'Diffusion':>12}  {'Full Pipeline':>15}  {'Reduction':>15}\n")
        f.write("-" * 110 + "\n")
        
        for snr in snr_values:
            pilot_ser = results_dict[snr]['pilot_only']['ser_mean'] * 100
            diff_ser = results_dict[snr]['diffusion_only']['ser_mean'] * 100
            full_ser = results_dict[snr]['full_pipeline']['ser_mean'] * 100
            reduction = pilot_ser - full_ser
            
            f.write(f"{snr:>10.0f}  {pilot_ser:>12.2f}  {diff_ser:>12.2f}  {full_ser:>15.2f}  {reduction:>+15.2f}\n")
        
        f.write("\n" + "=" * 110 + "\n")
        f.write("\nInterpretation:\n")
        f.write("- Effective SNR: Higher is better (less residual error)\n")
        f.write("- Symbol Error Rate: Lower is better (more accurate decoding)\n")
        f.write("- This metric is what matters in real systems for data decoding!\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate Effective SNR (Practical Metric)')
    parser.add_argument('--diffusion_checkpoint', type=str, required=True)
    parser.add_argument('--snr_min', type=int, default=5)
    parser.add_argument('--snr_max', type=int, default=30)
    parser.add_argument('--snr_step', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./results/effective_snr')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup
    config = TransmissionConfig()
    dd_config = DDConfig(llr_threshold=8.0, adaptive_threshold=False)
    
    pipeline = IntegratedEstimator(
        diffusion_checkpoint=args.diffusion_checkpoint,
        dd_config=dd_config,
        device=args.device
    )
    
    # Evaluate
    snr_values = list(range(args.snr_min, args.snr_max + 1, args.snr_step))
    results_dict = {}
    
    print("\n" + "="*80)
    print("EFFECTIVE SNR EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Diffusion Checkpoint: {args.diffusion_checkpoint}")
    print(f"  SNR Range: {args.snr_min} to {args.snr_max} dB (step {args.snr_step} dB)")
    print(f"  Samples per SNR: {args.num_samples}")
    print(f"  Output Directory: {args.output_dir}")
    print("\n" + "="*80 + "\n")
    
    for snr in tqdm(snr_values, desc="SNR Sweep"):
        results_dict[snr] = evaluate_effective_snr_at_snr(
            snr, args.num_samples, config, dd_config, pipeline
        )
    
    # Save results
    plot_effective_snr_results(snr_values, results_dict, args.output_dir)
    save_effective_snr_table(snr_values, results_dict, args.output_dir)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n{'Input SNR':>10}  {'Pilot Only':>12}  {'Full Pipeline':>15}  {'Gain (dB)':>12}")
    print("-"*55)
    
    for snr in snr_values:
        pilot_eff = results_dict[snr]['pilot_only']['eff_snr_mean']
        full_eff = results_dict[snr]['full_pipeline']['eff_snr_mean']
        gain = full_eff - pilot_eff
        print(f"{snr:>10.0f}  {pilot_eff:>12.2f}  {full_eff:>15.2f}  {gain:>+12.2f}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - effective_snr_results.png")
    print(f"  - effective_snr_results.txt")
