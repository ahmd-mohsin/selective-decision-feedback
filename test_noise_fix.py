#!/usr/bin/env python3
"""
Test the noise variance fix
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.dataset_generator import load_dataset_hdf5
from transmission_system.receiver_frontend import compute_channel_nmse

def test_noise_var_fix():
    print("=" * 80)
    print("Testing Noise Variance Fix")
    print("=" * 80)
    
    dataset_path = "transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
    checkpoint_path = "checkpoints/module2/checkpoint_epoch_100.pt"
    
    data, config = load_dataset_hdf5(dataset_path)
    
    estimator = IntegratedEstimator(
        diffusion_checkpoint=checkpoint_path,
        dd_config=None,
        modulation_order=16,
        device='cuda'
    )
    
    idx = 0
    H_true = data['H_true'][idx]
    H_pilot_interp = data['H_pilot_full'][idx]
    Y_grid = data['Y_grid'][idx]
    X_grid = data['X_grid'][idx]
    pilot_mask = data['pilot_mask'][idx]
    
    # OLD: noise variance from pilot interpolation (wrong!)
    pilot_positions = np.where(pilot_mask)
    pilot_errors_interp = Y_grid[pilot_positions] - H_pilot_interp[pilot_positions] * X_grid[pilot_positions]
    noise_var_wrong = np.mean(np.abs(pilot_errors_interp)**2)
    
    print(f"\nNoise variance (OLD method from pilot interp): {noise_var_wrong:.6f}")
    
    # Run full pipeline (it should now use correct noise variance internally)
    result = estimator.estimate_full_pipeline(
        Y_grid, X_grid, pilot_mask, H_pilot_interp, noise_var_wrong, num_iterations=2
    )
    
    H_final = result['H_final']
    H_diffusion = result['H_diffusion']
    
    nmse_pilot = compute_channel_nmse(H_pilot_interp, H_true)
    nmse_diffusion = compute_channel_nmse(H_diffusion, H_true)
    nmse_final = compute_channel_nmse(H_final, H_true)
    
    print(f"\nResults:")
    print(f"  Pilot interpolation: {10*np.log10(nmse_pilot):.2f} dB")
    print(f"  Diffusion: {10*np.log10(nmse_diffusion):.2f} dB")
    print(f"  Full Pipeline (DD+Diffusion): {10*np.log10(nmse_final):.2f} dB")
    
    print(f"\nAcceptance rates: {result['acceptance_rates']}")
    print(f"Average acceptance: {np.mean(result['acceptance_rates'])*100:.2f}%")
    
    print(f"\nPilot mask stats:")
    print(f"  Original pilots: {np.sum(pilot_mask)}")
    print(f"  Augmented pilots: {np.sum(result['augmented_pilot_mask'])}")
    print(f"  New virtual pilots: {np.sum(result['augmented_pilot_mask']) - np.sum(pilot_mask)}")
    
    if nmse_final < nmse_diffusion:
        print(f"\n✓ SUCCESS: Full pipeline improved over diffusion by {10*np.log10(nmse_diffusion/nmse_final):.2f} dB!")
    else:
        print(f"\n✗ PROBLEM: Full pipeline worse than diffusion by {10*np.log10(nmse_final/nmse_diffusion):.2f} dB")
    
    print("=" * 80)

if __name__ == '__main__':
    test_noise_var_fix()
