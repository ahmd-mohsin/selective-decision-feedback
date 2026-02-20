#!/usr/bin/env python3
"""
Check what's happening with the diffusion model
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from decision_directed.pipeline import IntegratedEstimator
from transmission_system.dataset_generator import load_dataset_hdf5
from transmission_system.receiver_frontend import compute_channel_nmse

def check_diffusion_output():
    print("=" * 80)
    print("Checking Diffusion Model Behavior")
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
    
    print(f"\n1. INPUT STATISTICS")
    print(f"   H_true: mean={np.abs(H_true).mean():.4f}, std={np.abs(H_true).std():.4f}")
    print(f"   H_pilot_interp: mean={np.abs(H_pilot_interp).mean():.4f}, std={np.abs(H_pilot_interp).std():.4f}")
    print(f"   Y_grid: mean={np.abs(Y_grid).mean():.4f}, std={np.abs(Y_grid).std():.4f}")
    print(f"   X_grid: mean={np.abs(X_grid).mean():.4f}, std={np.abs(X_grid).std():.4f}")
    
    # Run diffusion
    print(f"\n2. RUNNING DIFFUSION MODEL")
    diffusion_result = estimator.estimate_diffusion_only(Y_grid, H_pilot_interp, pilot_mask)
    H_diffusion = diffusion_result['H_estimate']
    
    print(f"   H_diffusion: mean={np.abs(H_diffusion).mean():.4f}, std={np.abs(H_diffusion).std():.4f}")
    
    nmse_pilot = compute_channel_nmse(H_pilot_interp, H_true)
    nmse_diffusion = compute_channel_nmse(H_diffusion, H_true)
    
    print(f"\n3. NMSE COMPARISON")
    print(f"   Pilot interpolation: {10*np.log10(nmse_pilot):.2f} dB")
    print(f"   Diffusion output: {10*np.log10(nmse_diffusion):.2f} dB")
    print(f"   Improvement: {10*np.log10(nmse_pilot/nmse_diffusion):.2f} dB")
    
    # Check noise estimation from diffusion output
    pilot_positions = np.where(pilot_mask)
    
    # Noise from pilot interpolation
    pilot_errors_interp = Y_grid[pilot_positions] - H_pilot_interp[pilot_positions] * X_grid[pilot_positions]
    noise_var_interp = np.mean(np.abs(pilot_errors_interp)**2)
    
    # Noise from diffusion
    pilot_errors_diff = Y_grid[pilot_positions] - H_diffusion[pilot_positions] * X_grid[pilot_positions]
    noise_var_diff = np.mean(np.abs(pilot_errors_diff)**2)
    
    print(f"\n4. NOISE VARIANCE ESTIMATION")
    print(f"   From pilot interpolation: {noise_var_interp:.6f} (SNR: {10*np.log10(1/noise_var_interp) if noise_var_interp > 0 else float('inf'):.2f} dB)")
    print(f"   From diffusion output: {noise_var_diff:.6f} (SNR: {10*np.log10(1/noise_var_diff) if noise_var_diff > 0 else float('inf'):.2f} dB)")
    
    # Check at data positions
    data_mask = ~pilot_mask
    data_errors_interp = Y_grid[data_mask] - H_pilot_interp[data_mask] * X_grid[data_mask]
    data_errors_diff = Y_grid[data_mask] - H_diffusion[data_mask] * X_grid[data_mask]
    
    noise_var_data_interp = np.mean(np.abs(data_errors_interp)**2)
    noise_var_data_diff = np.mean(np.abs(data_errors_diff)**2)
    
    print(f"\n5. NOISE AT DATA POSITIONS (using true X)")
    print(f"   From pilot interpolation: {noise_var_data_interp:.6f}")
    print(f"   From diffusion output: {noise_var_data_diff:.6f}")
    
    # The real issue: check equalization
    print(f"\n6. EQUALIZATION CHECK (Symbol 50)")
    sym_idx = 50
    data_mask_sym = ~pilot_mask[sym_idx]
    
    H_interp_sym = H_pilot_interp[sym_idx, data_mask_sym]
    H_diff_sym = H_diffusion[sym_idx, data_mask_sym]
    H_true_sym = H_true[sym_idx, data_mask_sym]
    Y_sym = Y_grid[sym_idx, data_mask_sym]
    X_true_sym = X_grid[sym_idx, data_mask_sym]
    
    # Equalized symbols
    Y_eq_interp = Y_sym / (H_interp_sym + 1e-10)
    Y_eq_diff = Y_sym / (H_diff_sym + 1e-10)
    Y_eq_true = Y_sym / (H_true_sym + 1e-10)
    
    # Distance to true symbols
    dist_interp = np.mean(np.abs(Y_eq_interp - X_true_sym)**2)
    dist_diff = np.mean(np.abs(Y_eq_diff - X_true_sym)**2)
    dist_true = np.mean(np.abs(Y_eq_true - X_true_sym)**2)
    
    print(f"   MSE to true symbols:")
    print(f"      Pilot interpolation: {dist_interp:.6f}")
    print(f"      Diffusion: {dist_diff:.6f}")
    print(f"      Perfect H: {dist_true:.6f} (noise floor)")
    
    # Hard decision SER
    from decision_directed.llr_computer import LLRComputer
    llr_comp = LLRComputer(16)
    
    dec_interp = llr_comp.hard_decision(Y_eq_interp)
    dec_diff = llr_comp.hard_decision(Y_eq_diff)
    
    ser_interp = np.mean(dec_interp != X_true_sym)
    ser_diff = np.mean(dec_diff != X_true_sym)
    
    print(f"\n   Symbol Error Rate:")
    print(f"      Pilot interpolation: {ser_interp*100:.2f}%")
    print(f"      Diffusion: {ser_diff*100:.2f}%")
    
    print(f"\n" + "=" * 80)

if __name__ == '__main__':
    check_diffusion_output()
