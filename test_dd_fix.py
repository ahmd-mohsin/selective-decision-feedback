#!/usr/bin/env python3
"""
Quick test script to verify decision-directed feedback augments pilot mask correctly
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from decision_directed.config import DecisionDirectedConfig
from decision_directed.estimator import DecisionDirectedEstimator
from transmission_system.dataset_generator import load_dataset_hdf5

def test_augmented_pilot_mask():
    print("=" * 70)
    print("Testing Augmented Pilot Mask Fix")
    print("=" * 70)
    
    # Load a small sample
    dataset_path = "transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
    data, config = load_dataset_hdf5(dataset_path)
    
    idx = 0
    H_initial = data['H_pilot_full'][idx]
    Y_grid = data['Y_grid'][idx]
    X_grid = data['X_grid'][idx]
    pilot_mask = data['pilot_mask'][idx]
    
    print(f"\nOriginal pilot_mask shape: {pilot_mask.shape}")
    print(f"Number of original pilots: {np.sum(pilot_mask)}")
    print(f"Pilot density: {np.sum(pilot_mask) / pilot_mask.size * 100:.2f}%")
    
    # Run DD estimator
    dd_config = DecisionDirectedConfig(llr_threshold=4.0)
    estimator = DecisionDirectedEstimator(dd_config, modulation_order=16)
    
    result = estimator.estimate(H_initial, Y_grid, X_grid, pilot_mask)
    
    print(f"\nDecision-directed results:")
    print(f"  DD mask created: {np.sum(result['dd_mask'])} reliable decisions")
    
    if 'augmented_pilot_mask' in result:
        augmented_mask = result['augmented_pilot_mask']
        print(f"\n✓ Augmented pilot mask created!")
        print(f"  Original pilots: {np.sum(pilot_mask)}")
        print(f"  Augmented pilots: {np.sum(augmented_mask)}")
        print(f"  New virtual pilots: {np.sum(augmented_mask) - np.sum(pilot_mask)}")
        print(f"  New pilot density: {np.sum(augmented_mask) / augmented_mask.size * 100:.2f}%")
        
        # Verify augmented mask includes original pilots
        assert np.all(pilot_mask <= augmented_mask), "ERROR: Augmented mask should include all original pilots!"
        print(f"\n✓ Verification passed: All original pilots preserved in augmented mask")
        
        print(f"\nAcceptance rate: {np.mean(result['acceptance_rates'])*100:.2f}%")
        
        return True
    else:
        print("\n✗ ERROR: 'augmented_pilot_mask' not found in result!")
        return False

if __name__ == '__main__':
    success = test_augmented_pilot_mask()
    
    if success:
        print("\n" + "=" * 70)
        print("TEST PASSED: Decision-directed feedback correctly augments pilot mask!")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("TEST FAILED")
        print("=" * 70)
        sys.exit(1)
