#!/usr/bin/env python3
"""
Quick test: Verify pilot density capping works
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from decision_directed.config import DecisionDirectedConfig
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample
from transmission_system.receiver_frontend import compute_channel_nmse

def test_high_snr():
    print("=" * 80)
    print("Testing Pilot Density Capping at High SNR")
    print("=" * 80)
    
    checkpoint = "checkpoints/module2/checkpoint_epoch_100.pt"
    
    # Test at SNR=25 dB where we had problems
    config = TransmissionConfig(
        Nfft=64,
        Nsym=100,
        pilot_spacing=4,
        modulation_order=16,
        snr_db=25.0,
        doppler_hz=100.0,
        seed=42
    )
    
    dd_config = DecisionDirectedConfig(
        llr_threshold=8.0,  # Base threshold
        adaptive_threshold=False
    )
    
    estimator = IntegratedEstimator(
        diffusion_checkpoint=checkpoint,
        dd_config=dd_config,
        modulation_order=16,
        device='cuda'
    )
    
    rng = np.random.default_rng(42)
    sample = generate_single_sample(config, rng, 0)
    
    H_true = sample['H_true']
    Y_grid = sample['Y_grid']
    X_grid = sample['X_grid']
    pilot_mask = sample['pilot_mask']
    H_pilot_full = sample['H_pilot_full']
    
    print(f"\nOriginal pilot density: {np.sum(pilot_mask)/pilot_mask.size*100:.1f}%")
    
    # Test diffusion only
    diff_result = estimator.estimate_diffusion_only(Y_grid, H_pilot_full, pilot_mask)
    nmse_diff = compute_channel_nmse(diff_result['H_estimate'], H_true)
    
    # Test full pipeline with capping
    full_result = estimator.estimate_full_pipeline(
        Y_grid, X_grid, pilot_mask, H_pilot_full, None,
        num_iterations=1, use_dd_before_diffusion=True
    )
    nmse_full = compute_channel_nmse(full_result['H_final'], H_true)
    
    augmented_density = np.sum(full_result['augmented_pilot_mask']) / full_result['augmented_pilot_mask'].size
    
    print(f"\nResults at SNR=25 dB:")
    print(f"  Diffusion Only:  {10*np.log10(nmse_diff):.2f} dB")
    print(f"  Full Pipeline:   {10*np.log10(nmse_full):.2f} dB")
    print(f"\n  Acceptance rate: {np.mean(full_result['acceptance_rates'])*100:.1f}%")
    print(f"  Final pilot density: {augmented_density*100:.1f}%")
    
    if nmse_full < nmse_diff:
        print(f"\n✓ SUCCESS! Full pipeline improved by {10*np.log10(nmse_diff/nmse_full):.2f} dB")
    else:
        print(f"\n✗ PROBLEM: Full pipeline worse by {10*np.log10(nmse_full/nmse_diff):.2f} dB")
    
    if augmented_density <= 0.60:
        print(f"✓ Pilot density capped correctly (<= 60%)")
    else:
        print(f"✗ Pilot density too high: {augmented_density*100:.1f}%")
    
    print("=" * 80)

if __name__ == '__main__':
    test_high_snr()
