#!/usr/bin/env python3
"""
Test new approach: DD BEFORE diffusion
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

def test_dd_before_diffusion():
    print("=" * 80)
    print("Testing: DD BEFORE Diffusion (Pseudo-Pilots Approach)")
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
    H_pilot = data['H_pilot_full'][idx]
    Y_grid = data['Y_grid'][idx]
    X_grid = data['X_grid'][idx]
    pilot_mask = data['pilot_mask'][idx]
    
    # 1. Pilot interpolation baseline
    nmse_pilot = compute_channel_nmse(H_pilot, H_true)
    
    # 2. Diffusion only (original approach)
    print(f"\nRunning Diffusion Only (baseline)...")
    diff_result = estimator.estimate_diffusion_only(Y_grid, H_pilot, pilot_mask)
    H_diffusion = diff_result['H_estimate']
    nmse_diffusion = compute_channel_nmse(H_diffusion, H_true)
    
    # 3. DD + Diffusion (new approach)
    print(f"Running DD + Diffusion (new approach)...")
    result = estimator.estimate_full_pipeline(
        Y_grid, X_grid, pilot_mask, H_pilot, None, 
        num_iterations=1, use_dd_before_diffusion=True
    )
    
    H_final = result['H_final']
    nmse_final = compute_channel_nmse(H_final, H_true)
    
    print(f"\n{'='*80}")
    print(f"RESULTS COMPARISON")
    print(f"{'='*80}")
    print(f"  1. Pilot interpolation:     {10*np.log10(nmse_pilot):.2f} dB")
    print(f"  2. Diffusion Only:          {10*np.log10(nmse_diffusion):.2f} dB")
    print(f"  3. DD + Diffusion (NEW):    {10*np.log10(nmse_final):.2f} dB")
    print(f"\n{'='*80}")
    print(f"IMPROVEMENTS")
    print(f"{'='*80}")
    print(f"  Diffusion vs Pilot:         {10*np.log10(nmse_pilot/nmse_diffusion):.2f} dB")
    print(f"  DD+Diffusion vs Pilot:      {10*np.log10(nmse_pilot/nmse_final):.2f} dB")
    print(f"  DD+Diffusion vs Diffusion:  {10*np.log10(nmse_diffusion/nmse_final):.2f} dB")
    
    if nmse_final < nmse_diffusion:
        print(f"\n✓ SUCCESS! DD+Diffusion is {10*np.log10(nmse_diffusion/nmse_final):.2f} dB better than Diffusion Only!")
    else:
        print(f"\n✗ PROBLEM: DD+Diffusion is {10*np.log10(nmse_final/nmse_diffusion):.2f} dB worse than Diffusion Only")
    
    print(f"\nAcceptance rate: {np.mean(result['acceptance_rates'])*100:.1f}%")
    print(f"Augmented pilots: {np.sum(result['augmented_pilot_mask'])} (from {np.sum(pilot_mask)} original)")
    
    print("=" * 80)

if __name__ == '__main__':
    test_dd_before_diffusion()
