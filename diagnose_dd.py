#!/usr/bin/env python3
"""
Diagnostic script to understand why DD is performing poorly
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from decision_directed.config import DecisionDirectedConfig
from decision_directed.estimator import DecisionDirectedEstimator
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.dataset_generator import load_dataset_hdf5
from transmission_system.receiver_frontend import compute_channel_nmse

def diagnose_dd_performance():
    print("=" * 80)
    print("DIAGNOSTIC: Why is Decision-Directed Performing Poorly?")
    print("=" * 80)
    
    dataset_path = "transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
    data, config = load_dataset_hdf5(dataset_path)
    
    idx = 0
    H_true = data['H_true'][idx]
    H_pilot_interp = data['H_pilot_full'][idx]
    Y_grid = data['Y_grid'][idx]
    X_grid = data['X_grid'][idx]
    pilot_mask = data['pilot_mask'][idx]
    
    # Estimate noise variance
    pilot_positions = np.where(pilot_mask)
    pilot_errors = Y_grid[pilot_positions] - H_pilot_interp[pilot_positions] * X_grid[pilot_positions]
    noise_var = np.mean(np.abs(pilot_errors)**2)
    
    print(f"\n1. GROUND TRUTH METRICS")
    print(f"   Noise variance (from pilots): {noise_var:.6f}")
    print(f"   SNR (theoretical): 20.0 dB")
    print(f"   SNR (estimated): {10*np.log10(1/noise_var):.2f} dB")
    
    # Check pilot interpolation quality
    nmse_pilot = compute_channel_nmse(H_pilot_interp, H_true)
    print(f"\n2. PILOT INTERPOLATION")
    print(f"   NMSE: {10*np.log10(nmse_pilot):.2f} dB")
    
    # Test with different thresholds
    print(f"\n3. DECISION-DIRECTED WITH DIFFERENT THRESHOLDS")
    print(f"   {'Threshold':<12} {'Accept %':<12} {'NMSE (dB)':<12} {'DD NMSE (dB)':<15}")
    print(f"   {'-'*60}")
    
    for threshold in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        dd_config = DecisionDirectedConfig(
            llr_threshold=threshold,
            adaptive_threshold=False
        )
        estimator = DecisionDirectedEstimator(dd_config, modulation_order=16)
        
        result = estimator.estimate(H_pilot_interp, Y_grid, X_grid, pilot_mask, noise_var)
        
        H_dd = result['H_dd']
        nmse_dd = compute_channel_nmse(H_dd, H_true)
        
        acceptance_rate = np.mean(result['acceptance_rates'])
        
        # Check quality of DD channel estimates at accepted positions
        dd_positions = result['dd_mask']
        if np.any(dd_positions):
            dd_channel_estimates = H_dd[dd_positions]
            true_channel = H_true[dd_positions]
            dd_nmse_at_accepted = np.mean(np.abs(dd_channel_estimates - true_channel)**2) / np.mean(np.abs(true_channel)**2)
            dd_nmse_db = 10*np.log10(dd_nmse_at_accepted)
        else:
            dd_nmse_db = float('inf')
        
        print(f"   {threshold:<12.1f} {acceptance_rate*100:<11.1f}% {10*np.log10(nmse_dd):<12.2f} {dd_nmse_db:<15.2f}")
    
    # Detailed analysis at default threshold
    print(f"\n4. DETAILED ANALYSIS (Threshold = 4.0)")
    dd_config = DecisionDirectedConfig(llr_threshold=4.0, adaptive_threshold=False)
    estimator = DecisionDirectedEstimator(dd_config, modulation_order=16)
    result = estimator.estimate(H_pilot_interp, Y_grid, X_grid, pilot_mask, noise_var)
    
    # Check equalization quality
    sym_idx = 50  # Middle symbol
    data_mask = ~pilot_mask[sym_idx]
    H_current = H_pilot_interp[sym_idx]
    Y_equalized = Y_grid[sym_idx, data_mask] / (H_current[data_mask] + 1e-10)
    
    # True transmitted symbols
    X_true = X_grid[sym_idx, data_mask]
    
    # Hard decisions
    from decision_directed.llr_computer import LLRComputer
    llr_comp = LLRComputer(16)
    hard_decisions = llr_comp.hard_decision(Y_equalized)
    
    # Symbol error rate
    symbol_errors = np.sum(hard_decisions != X_true)
    ser = symbol_errors / len(X_true)
    
    print(f"   Symbol index: {sym_idx}")
    print(f"   Data symbols: {len(X_true)}")
    print(f"   Symbol Error Rate: {ser*100:.2f}%")
    print(f"   Accepted as reliable: {np.sum(result['dd_mask'][sym_idx])}")
    
    # Check if accepted symbols are actually correct
    dd_mask_sym = result['dd_mask'][sym_idx]
    if np.any(dd_mask_sym):
        data_indices = np.where(data_mask)[0]
        accepted_indices = data_indices[dd_mask_sym[data_mask]]
        
        decisions_at_accepted = result['X_dd'][sym_idx, accepted_indices]
        true_at_accepted = X_grid[sym_idx, accepted_indices]
        
        correct_decisions = np.sum(decisions_at_accepted == true_at_accepted)
        accuracy = correct_decisions / len(accepted_indices) if len(accepted_indices) > 0 else 0
        
        print(f"   Accuracy of accepted decisions: {accuracy*100:.2f}%")
        print(f"   Incorrectly accepted: {len(accepted_indices) - correct_decisions}")
    
    # Check if the problem is in the channel estimation at DD positions
    print(f"\n5. CHANNEL ESTIMATION QUALITY AT DD POSITIONS")
    dd_positions = result['dd_mask']
    
    if np.any(dd_positions):
        # True channel
        H_true_at_dd = H_true[dd_positions]
        
        # DD estimated channel
        H_dd_at_dd = result['H_dd'][dd_positions]
        
        # Theoretical channel (if we had perfect decisions)
        Y_at_dd = Y_grid[dd_positions]
        X_at_dd = X_grid[dd_positions]  # True symbols
        H_perfect = Y_at_dd / (X_at_dd + 1e-10)
        
        nmse_dd_estimate = np.mean(np.abs(H_dd_at_dd - H_true_at_dd)**2) / np.mean(np.abs(H_true_at_dd)**2)
        nmse_perfect = np.mean(np.abs(H_perfect - H_true_at_dd)**2) / np.mean(np.abs(H_true_at_dd)**2)
        
        print(f"   NMSE of DD channel estimates: {10*np.log10(nmse_dd_estimate):.2f} dB")
        print(f"   NMSE with perfect decisions: {10*np.log10(nmse_perfect):.2f} dB")
        print(f"   Gap: {10*np.log10(nmse_dd_estimate/nmse_perfect):.2f} dB")
    
    print(f"\n" + "=" * 80)

if __name__ == '__main__':
    diagnose_dd_performance()
