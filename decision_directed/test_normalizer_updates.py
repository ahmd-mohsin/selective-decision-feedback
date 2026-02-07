import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from transmission_system.dataset_generator import load_dataset_hdf5
from decision_directed import DecisionDirectedConfig, DecisionDirectedEstimator
from transmission_system.receiver_frontend import compute_channel_nmse

data, config = load_dataset_hdf5(
    '../transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5'
)

idx = 0
H_true = data['H_true'][idx]
Y_grid = data['Y_grid'][idx]
pilot_mask = data['pilot_mask'][idx]
H_pilot = data['H_pilot_full'][idx]
X_grid = data['X_grid'][idx]

pilot_positions = np.where(pilot_mask)
pilot_errors = Y_grid[pilot_positions] - H_pilot[pilot_positions] * X_grid[pilot_positions]
noise_var = np.mean(np.abs(pilot_errors)**2)

print("=" * 70)
print("NORMALIZER UPDATE TEST")
print("=" * 70)

configs_to_test = [
    ("Conservative (0.01)", DecisionDirectedConfig(normalizer_step_size=0.01, llr_threshold=4.0)),
    ("Moderate (0.05)", DecisionDirectedConfig(normalizer_step_size=0.05, llr_threshold=4.0)),
    ("Aggressive (0.1)", DecisionDirectedConfig(normalizer_step_size=0.1, llr_threshold=4.0)),
    ("Very Aggressive (0.2)", DecisionDirectedConfig(normalizer_step_size=0.2, llr_threshold=4.0)),
]

for name, dd_config in configs_to_test:
    estimator = DecisionDirectedEstimator(dd_config, modulation_order=16)
    result = estimator.estimate(H_pilot, Y_grid, X_grid, pilot_mask, noise_var)
    
    H_dd = result['H_dd']
    nmse_dd = compute_channel_nmse(H_dd, H_true)
    nmse_pilot = compute_channel_nmse(H_pilot, H_true)
    improvement = 10*np.log10(nmse_pilot/nmse_dd)
    
    print(f"\n{name}:")
    print(f"  Step size: {dd_config.normalizer_step_size}")
    print(f"  Mean |W|: {np.mean(np.abs(result['normalizers'])):.6f}")
    print(f"  Std |W|: {np.std(np.abs(result['normalizers'])):.6f}")
    print(f"  Acceptance: {np.mean(result['acceptance_rates'])*100:.1f}%")
    print(f"  Pilot NMSE: {10*np.log10(nmse_pilot):.2f} dB")
    print(f"  DD NMSE: {10*np.log10(nmse_dd):.2f} dB")
    print(f"  Improvement: {improvement:.2f} dB")

print("\n" + "=" * 70)
print("THRESHOLD TEST")
print("=" * 70)

thresholds_to_test = [2.0, 3.0, 4.0, 5.0, 6.0]

for threshold in thresholds_to_test:
    dd_config = DecisionDirectedConfig(normalizer_step_size=0.1, llr_threshold=threshold)
    estimator = DecisionDirectedEstimator(dd_config, modulation_order=16)
    result = estimator.estimate(H_pilot, Y_grid, X_grid, pilot_mask, noise_var)
    
    H_dd = result['H_dd']
    nmse_dd = compute_channel_nmse(H_dd, H_true)
    nmse_pilot = compute_channel_nmse(H_pilot, H_true)
    improvement = 10*np.log10(nmse_pilot/nmse_dd)
    
    print(f"\nThreshold {threshold}:")
    print(f"  Acceptance: {np.mean(result['acceptance_rates'])*100:.1f}%")
    print(f"  Mean |W|: {np.mean(np.abs(result['normalizers'])):.6f}")
    print(f"  DD NMSE: {10*np.log10(nmse_dd):.2f} dB")
    print(f"  Improvement: {improvement:.2f} dB")