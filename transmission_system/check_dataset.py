import h5py
import numpy as np

filepath = '../transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5'

with h5py.File(filepath, 'r') as f:
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)
    print(f"\nDataset keys: {list(f.keys())}")
    
    if 'noise_var' in f.keys():
        noise_var = f['noise_var'][:]
        print(f"\nnoise_var shape: {noise_var.shape}")
        print(f"noise_var first 10 values: {noise_var[:10]}")
        print(f"noise_var statistics:")
        print(f"  Min: {np.min(noise_var)}")
        print(f"  Max: {np.max(noise_var)}")
        print(f"  Mean: {np.mean(noise_var)}")
        print(f"  Std: {np.std(noise_var)}")
        print(f"  Number of zeros: {np.sum(noise_var == 0)}")
    else:
        print("\nERROR: 'noise_var' not found in dataset!")
    
    if 'snr_db' in f.keys():
        snr_db = f['snr_db'][:]
        print(f"\nsnr_db values: {snr_db[:10]}")
    
    H_true = f['H_true'][0]
    Y_grid = f['Y_grid'][0]
    
    signal_power = np.mean(np.abs(H_true)**2)
    received_power = np.mean(np.abs(Y_grid)**2)
    
    print(f"\nSample 0 statistics:")
    print(f"  Signal power (|H|²): {signal_power:.6f}")
    print(f"  Received power (|Y|²): {received_power:.6f}")
    
    snr_linear = 10**(20.0/10.0)
    expected_noise_var = signal_power / snr_linear
    print(f"  Expected noise_var at SNR=20dB: {expected_noise_var:.6f}")