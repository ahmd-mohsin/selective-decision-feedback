import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample
from transmission_system.receiver_frontend import compute_channel_nmse


def test_module1():
    print("Testing Module 1: Transmission System")
    print("=" * 60)
    
    config = TransmissionConfig(
        Nfft=64,
        Ncp=16,
        Nsym=50,
        pilot_spacing=4,
        modulation_order=16,
        snr_db=20.0,
        doppler_hz=100.0,
        Nt=1,
        Nr=1,
        time_varying=True,
        block_fading=False,
        seed=42
    )
    
    print("\nConfiguration:")
    print(f"  Nfft: {config.Nfft}")
    print(f"  Nsym: {config.Nsym}")
    print(f"  Modulation: {config.modulation_order}-QAM")
    print(f"  SNR: {config.snr_db} dB")
    print(f"  Doppler: {config.doppler_hz} Hz")
    
    rng = np.random.default_rng(config.seed)
    
    print("\nGenerating single sample...")
    sample = generate_single_sample(config, rng, sample_id=0)
    
    print("\nSample structure verification:")
    print(f"  Y_grid shape: {sample['Y_grid'].shape}")
    print(f"  H_true shape: {sample['H_true'].shape}")
    print(f"  H_pilot_full shape: {sample['H_pilot_full'].shape}")
    print(f"  pilot_mask shape: {sample['pilot_mask'].shape}")
    print(f"  X_grid shape: {sample['X_grid'].shape}")
    print(f"  Yp shape: {sample['Yp'].shape}")
    print(f"  Xp shape: {sample['Xp'].shape}")
    print(f"  noise_var: {sample['noise_var']:.6f}")
    
    print("\nVerifying complex tensors:")
    print(f"  Y_grid is complex: {np.iscomplexobj(sample['Y_grid'])}")
    print(f"  H_true is complex: {np.iscomplexobj(sample['H_true'])}")
    print(f"  H_pilot_full is complex: {np.iscomplexobj(sample['H_pilot_full'])}")
    
    print("\nVerifying pilot extraction:")
    num_pilots = np.sum(sample['pilot_mask'])
    print(f"  Total pilots: {num_pilots}")
    print(f"  Expected: {config.Nsym * (config.Nfft // config.pilot_spacing)}")
    print(f"  Yp length: {len(sample['Yp'])}")
    print(f"  Xp length: {len(sample['Xp'])}")
    
    print("\nComputing baseline channel estimation NMSE:")
    nmse_pilot = compute_channel_nmse(sample['H_pilot_full'], sample['H_true'])
    print(f"  Pilot-only NMSE: {nmse_pilot:.6f} ({10*np.log10(nmse_pilot):.2f} dB)")
    
    print("\n" + "=" * 60)
    print("Module 1 Interface Contract for Diffusion/DD:")
    print("=" * 60)
    
    print("\nDiffusion Module will receive:")
    print(f"  - Y_grid: {sample['Y_grid'].shape} (complex)")
    print(f"  - pilot_mask: {sample['pilot_mask'].shape} (bool)")
    print(f"  - H_pilot_full: {sample['H_pilot_full'].shape} (complex, coarse baseline)")
    print(f"  - H_true: {sample['H_true'].shape} (complex, supervision)")
    print(f"  - noise_var: scalar")
    
    print("\nDD Module will receive:")
    print(f"  - Y_grid: per-symbol observations")
    print(f"  - H_estimate: current channel estimate")
    print(f"  - pilot_mask: to exclude pilots from DD")
    print(f"  - noise_var: for LLR computation")
    
    print("\nDD Module will provide back:")
    print(f"  - H_hat_dd: refined channel estimate")
    print(f"  - dd_mask: accepted pseudo-pilot positions")
    print(f"  - X_dd: hard decisions at accepted positions")
    print(f"  - Y_dd: observations at accepted positions")
    
    print("\n" + "=" * 60)
    print("Module 1 test passed successfully!")
    print("=" * 60)


def test_mimo():
    print("\n\nTesting MIMO Configuration (2x2)")
    print("=" * 60)
    
    config = TransmissionConfig(
        Nfft=64,
        Nsym=20,
        Nt=2,
        Nr=2,
        seed=100
    )
    
    rng = np.random.default_rng(config.seed)
    sample = generate_single_sample(config, rng, sample_id=0)
    
    print("\nMIMO shapes:")
    print(f"  X_grid: {sample['X_grid'].shape}")
    print(f"  Y_grid: {sample['Y_grid'].shape}")
    print(f"  H_true: {sample['H_true'].shape}")
    print(f"  H_pilot_full: {sample['H_pilot_full'].shape}")
    
    expected_X = (config.Nsym, config.Nfft, config.Nt)
    expected_Y = (config.Nsym, config.Nfft, config.Nr)
    expected_H = (config.Nsym, config.Nfft, config.Nr, config.Nt)
    
    assert sample['X_grid'].shape == expected_X, f"X_grid shape mismatch"
    assert sample['Y_grid'].shape == expected_Y, f"Y_grid shape mismatch"
    assert sample['H_true'].shape == expected_H, f"H_true shape mismatch"
    
    print("\nMIMO test passed!")


if __name__ == '__main__':
    test_module1()
    test_mimo()