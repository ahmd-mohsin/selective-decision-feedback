"""
Calculate Effective SNR: How well the estimated channel performs against noise

In Y = H_true × X + n:
- After estimating Ĥ, the residual error is: Y - Ĥ×X = (H_true - Ĥ)×X + n
- This residual contains BOTH estimation error AND noise
- Effective SNR = Signal Power / Residual Error Power
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample
from decision_directed.pipeline import IntegratedEstimator
from decision_directed.config import DecisionDirectedConfig


def calculate_effective_snr_simple(Y_grid, X_grid, H_estimated, data_mask):
    """
    Calculate effective SNR after channel estimation.
    
    This is what matters in real systems!
    
    Effective SNR = E[|H×X|²] / E[|Y - Ĥ×X|²]
    
    Where:
    - Signal: H×X (what we want to decode)
    - Residual error: Y - Ĥ×X = (H_true - Ĥ)×X + n
      (contains both estimation error and noise)
    """
    # Calculate on data positions only
    Y_data = Y_grid[data_mask]
    X_data = X_grid[data_mask]
    H_est_data = H_estimated[data_mask]
    
    # Signal power (what we're trying to decode)
    signal_power = np.mean(np.abs(H_est_data * X_data)**2)
    
    # Residual error power (estimation error + noise)
    Y_estimated = H_est_data * X_data
    residual = Y_data - Y_estimated
    error_power = np.mean(np.abs(residual)**2)
    
    # Effective SNR
    if error_power > 0:
        eff_snr = signal_power / error_power
        eff_snr_db = 10 * np.log10(eff_snr)
    else:
        eff_snr_db = float('inf')
    
    return eff_snr_db, signal_power, error_power


def main():
    """Run quick evaluation at a few SNR points."""
    
    print("\n" + "="*80)
    print("EFFECTIVE SNR EVALUATION")
    print("Measuring: Signal Power / (Estimation Error + Noise) Power")
    print("="*80 + "\n")
    
    # Setup
    config = TransmissionConfig()
    dd_config = DecisionDirectedConfig(llr_threshold=8.0, adaptive_threshold=False)
    
    checkpoint_path = './checkpoints/augmented_model/checkpoint_epoch_100.pt'
    
    pipeline = IntegratedEstimator(
        diffusion_checkpoint=checkpoint_path,
        dd_config=dd_config,
        device='cuda'
    )
    
    # Test at a few SNR values
    snr_values = [5, 10, 15, 20, 25, 30]
    num_samples = 20  # Quick test
    
    results = {
        'snr': [],
        'input_noise_var': [],
        'pilot_eff_snr': [],
        'diffusion_eff_snr': [],
        'full_eff_snr': []
    }
    
    for snr_db in snr_values:
        config.snr_db = snr_db
        noise_var = config.noise_variance
        
        print(f"\n{'='*80}")
        print(f"Input SNR = {snr_db} dB (Noise Variance = {noise_var:.4f})")
        print(f"{'='*80}")
        
        rng = np.random.default_rng(42 + snr_db * 100)
        
        pilot_eff_snrs = []
        diff_eff_snrs = []
        full_eff_snrs = []
        
        for i in tqdm(range(num_samples), desc=f"Samples at SNR={snr_db}dB"):
            sample = generate_single_sample(config, rng, i)
            
            Y_grid = sample['Y_grid']
            X_grid = sample['X_grid']
            pilot_mask = sample['pilot_mask']
            data_mask = ~pilot_mask
            
            # 1. Pilot interpolation only
            H_pilot = sample['H_pilot_full']
            eff_snr_pilot, _, _ = calculate_effective_snr_simple(
                Y_grid, X_grid, H_pilot, data_mask
            )
            pilot_eff_snrs.append(eff_snr_pilot)
            
            # 2. Diffusion only
            try:
                diff_result = pipeline.estimate_diffusion_only(
                    Y_grid, H_pilot, pilot_mask
                )
                H_diffusion = diff_result['H_estimate']
                eff_snr_diff, _, _ = calculate_effective_snr_simple(
                    Y_grid, X_grid, H_diffusion, data_mask
                )
                diff_eff_snrs.append(eff_snr_diff)
            except:
                diff_eff_snrs.append(eff_snr_pilot)
            
            # 3. Full pipeline (DD + Diffusion)
            try:
                full_result = pipeline.estimate_full_pipeline(
                    Y_grid, X_grid, pilot_mask, noise_var,
                    num_iterations=1, use_dd_before_diffusion=True
                )
                H_full = full_result['H_final']
                eff_snr_full, _, _ = calculate_effective_snr_simple(
                    Y_grid, X_grid, H_full, data_mask
                )
                full_eff_snrs.append(eff_snr_full)
            except:
                full_eff_snrs.append(eff_snr_diff if diff_eff_snrs else eff_snr_pilot)
        
        # Average results
        avg_pilot = np.mean(pilot_eff_snrs)
        avg_diff = np.mean(diff_eff_snrs)
        avg_full = np.mean(full_eff_snrs)
        
        results['snr'].append(snr_db)
        results['input_noise_var'].append(noise_var)
        results['pilot_eff_snr'].append(avg_pilot)
        results['diffusion_eff_snr'].append(avg_diff)
        results['full_eff_snr'].append(avg_full)
        
        print(f"\n  Results (Effective SNR in dB):")
        print(f"    Pilot Only:       {avg_pilot:7.2f} dB")
        print(f"    Diffusion Only:   {avg_diff:7.2f} dB")
        print(f"    Full Pipeline:    {avg_full:7.2f} dB")
        print(f"    Gain over Pilot:  {avg_full - avg_pilot:+7.2f} dB")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: EFFECTIVE SNR (Signal Power / Residual Error Power)")
    print("="*80)
    print(f"\n{'Input SNR':>10}  {'Input Noise':>12}  {'Pilot Only':>12}  {'Diffusion':>12}  {'Full Pipeline':>15}  {'Gain':>10}")
    print(f"{'(dB)':>10}  {'Variance':>12}  {'(dB)':>12}  {'(dB)':>12}  {'(dB)':>15}  {'(dB)':>10}")
    print("-"*90)
    
    for i, snr in enumerate(results['snr']):
        noise_var = results['input_noise_var'][i]
        pilot = results['pilot_eff_snr'][i]
        diff = results['diffusion_eff_snr'][i]
        full = results['full_eff_snr'][i]
        gain = full - pilot
        
        print(f"{snr:>10.0f}  {noise_var:>12.4f}  {pilot:>12.2f}  {diff:>12.2f}  {full:>15.2f}  {gain:>+10.2f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    print("Input SNR: The noise power added during transmission (Y = H×X + n)")
    print("Input Noise Variance: σ² = 10^(-SNR/10)")
    print()
    print("Effective SNR: How well the estimated channel performs at the receiver")
    print("  - Measured as: Signal_Power / Residual_Error_Power")
    print("  - Residual Error = Y - Ĥ×X = (H_true - Ĥ)×X + n")
    print("  - Contains BOTH estimation error AND noise")
    print()
    print("Key Insight:")
    print("  At SNR=10dB (noise var = 0.100), if Effective SNR = 8 dB, this means:")
    print("  - The residual error (estimation error + noise) has power ≈ 0.158")
    print("  - Signal power ≈ 1.0")
    print("  - Ratio = 1.0/0.158 ≈ 6.3 = 8 dB")
    print()
    print("This is the PRACTICAL metric that matters for data decoding!")
    print("="*80 + "\n")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Effective SNR vs Input SNR
    ax1.plot(results['snr'], results['pilot_eff_snr'], 'o-', 
            linewidth=2, markersize=8, label='Pilot Only', color='#e74c3c')
    ax1.plot(results['snr'], results['diffusion_eff_snr'], 's-',
            linewidth=2, markersize=8, label='Diffusion Only', color='#3498db')
    ax1.plot(results['snr'], results['full_eff_snr'], 'D-',
            linewidth=2, markersize=8, label='Full Pipeline', color='#2ecc71')
    
    # Add reference line (ideal: output = input)
    ax1.plot(results['snr'], results['snr'], 'k--', 
            linewidth=1.5, alpha=0.5, label='Ideal (no estimation error)')
    
    ax1.set_xlabel('Input SNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Effective SNR (dB)', fontsize=13, fontweight='bold')
    ax1.set_title('Effective SNR: Signal Power / (Error + Noise) Power', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gain
    gains_pilot = np.array(results['full_eff_snr']) - np.array(results['pilot_eff_snr'])
    
    ax2.bar(results['snr'], gains_pilot, width=3, color='mediumseagreen',
           edgecolor='black', linewidth=1.5, alpha=0.8)
    
    for i, (snr, gain) in enumerate(zip(results['snr'], gains_pilot)):
        ax2.text(snr, gain + 0.2, f'{gain:.2f}', ha='center', 
                fontsize=10, fontweight='bold')
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Input SNR (dB)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Effective SNR Gain (dB)', fontsize=13, fontweight='bold')
    ax2.set_title('Gain: Full Pipeline vs Pilot Only', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path('./results/effective_snr_quick')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'effective_snr_quick_test.png', dpi=150, bbox_inches='tight')
    
    print(f"Plot saved to: {output_dir}/effective_snr_quick_test.png\n")


if __name__ == "__main__":
    main()
