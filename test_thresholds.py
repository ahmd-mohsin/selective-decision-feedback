#!/usr/bin/env python3
"""
Test different LLR thresholds to find optimal operating point
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from decision_directed.config import DecisionDirectedConfig
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.dataset_generator import load_dataset_hdf5
from transmission_system.receiver_frontend import compute_channel_nmse

def test_thresholds():
    print("=" * 80)
    print("Testing Different LLR Thresholds")
    print("=" * 80)
    
    dataset_path = "transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
    checkpoint_path = "checkpoints/module2/checkpoint_epoch_100.pt"
    
    data, config = load_dataset_hdf5(dataset_path)
    
    print(f"\nTesting on 10 samples...")
    print(f"{'Threshold':<12} {'Accept %':<12} {'Pilot (dB)':<15} {'Diffusion (dB)':<18} {'Full (dB)':<15} {'Gain (dB)':<10}")
    print("=" * 100)
    
    thresholds = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    
    for threshold in thresholds:
        dd_config = DecisionDirectedConfig(
            llr_threshold=threshold,
            adaptive_threshold=False
        )
        
        estimator = IntegratedEstimator(
            diffusion_checkpoint=checkpoint_path,
            dd_config=dd_config,
            modulation_order=16,
            device='cuda'
        )
        
        nmse_pilot_list = []
        nmse_diff_list = []
        nmse_full_list = []
        accept_rates = []
        
        for idx in range(10):
            H_true = data['H_true'][idx]
            H_pilot = data['H_pilot_full'][idx]
            Y_grid = data['Y_grid'][idx]
            X_grid = data['X_grid'][idx]
            pilot_mask = data['pilot_mask'][idx]
            
            nmse_pilot = compute_channel_nmse(H_pilot, H_true)
            nmse_pilot_list.append(nmse_pilot)
            
            diff_result = estimator.estimate_diffusion_only(Y_grid, H_pilot, pilot_mask)
            nmse_diff = compute_channel_nmse(diff_result['H_estimate'], H_true)
            nmse_diff_list.append(nmse_diff)
            
            # Temporarily disable debug output
            import sys
            import os
            sys.stdout = open(os.devnull, 'w')
            
            full_result = estimator.estimate_full_pipeline(
                Y_grid, X_grid, pilot_mask, H_pilot, None, num_iterations=1
            )
            
            sys.stdout = sys.__stdout__
            
            nmse_full = compute_channel_nmse(full_result['H_final'], H_true)
            nmse_full_list.append(nmse_full)
            accept_rates.append(np.mean(full_result['acceptance_rates']))
        
        avg_pilot_db = 10*np.log10(np.mean(nmse_pilot_list))
        avg_diff_db = 10*np.log10(np.mean(nmse_diff_list))
        avg_full_db = 10*np.log10(np.mean(nmse_full_list))
        avg_accept = np.mean(accept_rates)*100
        gain_db = avg_diff_db - avg_full_db  # Positive means full is better
        
        print(f"{threshold:<12.1f} {avg_accept:<11.1f}% {avg_pilot_db:<15.2f} {avg_diff_db:<18.2f} {avg_full_db:<15.2f} {gain_db:<10.2f}")
    
    print("=" * 100)
    print("\nNote: Positive gain means Full Pipeline is better than Diffusion Only")

if __name__ == '__main__':
    test_thresholds()
