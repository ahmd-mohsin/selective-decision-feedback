#!/usr/bin/env python3
"""
Check if diffusion is actually using augmented pilots
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

def check_augmented_pilot_effect():
    print("=" * 80)
    print("Testing if Augmented Pilots Actually Help Diffusion")
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
    
    print(f"\n1. Baseline: Diffusion with ORIGINAL pilots")
    result1 = estimator.estimate_diffusion_only(Y_grid, H_pilot, pilot_mask)
    H_diff1 = result1['H_estimate']
    nmse1 = compute_channel_nmse(H_diff1, H_true)
    print(f"   Pilot density: {np.sum(pilot_mask)/pilot_mask.size*100:.1f}%")
    print(f"   NMSE: {10*np.log10(nmse1):.2f} dB")
    
    # Create fake "perfect" augmented pilots using TRUE channel at all positions
    print(f"\n2. Oracle: Diffusion with PERFECT augmented pilots (all positions)")
    augmented_mask_perfect = np.ones_like(pilot_mask, dtype=bool)
    H_perfect = H_true.copy()  # Perfect channel knowledge
    
    result2 = estimator.estimate_diffusion_only(Y_grid, H_perfect, augmented_mask_perfect)
    H_diff2 = result2['H_estimate']
    nmse2 = compute_channel_nmse(H_diff2, H_true)
    print(f"   Pilot density: {np.sum(augmented_mask_perfect)/augmented_mask_perfect.size*100:.1f}%")
    print(f"   NMSE: {10*np.log10(nmse2):.2f} dB")
    print(f"   Improvement over baseline: {10*np.log10(nmse1/nmse2):.2f} dB")
    
    # Create augmented pilots from DIFFUSION output (high quality but not perfect)
    print(f"\n3. Realistic: Diffusion with augmented pilots from its own output")
    # Use diffusion's own output as "virtual pilots" everywhere
    augmented_mask_realistic = np.ones_like(pilot_mask, dtype=bool)
    H_augmented = H_diff1.copy()
    
    result3 = estimator.estimate_diffusion_only(Y_grid, H_augmented, augmented_mask_realistic)
    H_diff3 = result3['H_estimate']
    nmse3 = compute_channel_nmse(H_diff3, H_true)
    print(f"   Pilot density: {np.sum(augmented_mask_realistic)/augmented_mask_realistic.size*100:.1f}%")
    print(f"   NMSE: {10*np.log10(nmse3):.2f} dB")
    print(f"   Change from baseline: {10*np.log10(nmse1/nmse3):.2f} dB")
    
    print(f"\n4. Summary:")
    print(f"   Original pilots only: {10*np.log10(nmse1):.2f} dB")
    print(f"   Perfect oracle: {10*np.log10(nmse2):.2f} dB")
    print(f"   Self-augmented: {10*np.log10(nmse3):.2f} dB")
    
    if nmse3 < nmse1:
        print(f"\n   ✓ Augmentation helps! Improvement: {10*np.log10(nmse1/nmse3):.2f} dB")
    else:
        print(f"\n   ✗ Augmentation hurts! Degradation: {10*np.log10(nmse3/nmse1):.2f} dB")
    
    print("=" * 80)

if __name__ == '__main__':
    check_augmented_pilot_effect()
