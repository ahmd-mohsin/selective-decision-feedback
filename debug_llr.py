#!/usr/bin/env python3
"""
Debug LLR computation
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from decision_directed.config import DecisionDirectedConfig
from decision_directed.estimator import DecisionDirectedEstimator
from decision_directed.llr_computer import LLRComputer
from transmission_system.dataset_generator import load_dataset_hdf5

def debug_llr():
    print("=" * 80)
    print("Debug LLR Computation")
    print("=" * 80)
    
    dataset_path = "transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
    data, config = load_dataset_hdf5(dataset_path)
    
    idx = 0
    H_pilot = data['H_pilot_full'][idx]
    Y_grid = data['Y_grid'][idx]
    X_grid = data['X_grid'][idx]
    pilot_mask = data['pilot_mask'][idx]
    
    # Compute noise variance at pilots
    pilot_pos = np.where(pilot_mask)
    pilot_errors = Y_grid[pilot_pos] - H_pilot[pilot_pos] * X_grid[pilot_pos]
    noise_var = np.mean(np.abs(pilot_errors)**2)
    
    print(f"\nNoise variance: {noise_var:.6f}")
    print(f"SNR: {10*np.log10(1/noise_var) if noise_var > 0 else float('inf'):.2f} dB")
    
    # Equalize one symbol
    sym_idx = 50
    data_mask = ~pilot_mask[sym_idx]
    
    H_est = H_pilot[sym_idx, data_mask]
    Y = Y_grid[sym_idx, data_mask]
    X_true = X_grid[sym_idx, data_mask]
    
    Y_eq = Y / (H_est + 1e-10)
    
    # Compute LLRs
    llr_comp = LLRComputer(16)
    llrs = llr_comp.compute_llrs(Y_eq, noise_var)
    min_llrs = np.min(np.abs(llrs), axis=-1)
    
    print(f"\nSymbol {sym_idx}:")
    print(f"  Data symbols: {len(Y_eq)}")
    print(f"  Min LLR statistics:")
    print(f"    Min: {np.min(min_llrs):.4f}")
    print(f"    Max: {np.max(min_llrs):.4f}")
    print(f"    Mean: {np.mean(min_llrs):.4f}")
    print(f"    Median: {np.median(min_llrs):.4f}")
    
    # Check with different thresholds
    for threshold in [1.0, 2.0, 4.0, 8.0, 16.0]:
        reliable = np.sum(min_llrs > threshold)
        print(f"  Threshold {threshold:.1f}: {reliable}/{len(min_llrs)} ({reliable/len(min_llrs)*100:.1f}%) reliable")
    
    # Check if noise_var == 0 causes issues
    print(f"\n" + "=" * 80)
    print("Testing with noise_var = 0.01 (forcing reasonable value)")
    print("=" * 80)
    
    noise_var_forced = 0.01
    llrs_forced = llr_comp.compute_llrs(Y_eq, noise_var_forced)
    min_llrs_forced = np.min(np.abs(llrs_forced), axis=-1)
    
    print(f"\nMin LLR statistics with forced noise:")
    print(f"  Min: {np.min(min_llrs_forced):.4f}")
    print(f"  Max: {np.max(min_llrs_forced):.4f}")
    print(f"  Mean: {np.mean(min_llrs_forced):.4f}")
    print(f"  Median: {np.median(min_llrs_forced):.4f}")
    
    for threshold in [1.0, 2.0, 4.0, 8.0, 16.0]:
        reliable = np.sum(min_llrs_forced > threshold)
        print(f"  Threshold {threshold:.1f}: {reliable}/{len(min_llrs_forced)} ({reliable/len(min_llrs_forced)*100:.1f}%) reliable")
    
    # Check hard decisions
    hard_dec = llr_comp.hard_decision(Y_eq)
    errors = np.sum(hard_dec != X_true)
    ser = errors / len(X_true)
    print(f"\nSymbol Error Rate: {ser*100:.2f}% ({errors}/{len(X_true)} errors)")
    
    print("=" * 80)

if __name__ == '__main__':
    debug_llr()
