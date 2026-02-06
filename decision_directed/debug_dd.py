import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from transmission_system.dataset_generator import load_dataset_hdf5
from decision_directed import DecisionDirectedConfig, DecisionDirectedEstimator

data, config = load_dataset_hdf5(
    '../transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5'
)

idx = 0
H_true = data['H_true'][idx]
Y_grid = data['Y_grid'][idx]
pilot_mask = data['pilot_mask'][idx]
H_pilot = data['H_pilot_full'][idx]
noise_var = data['noise_var'][idx]

X_grid = np.ones_like(Y_grid, dtype=complex)
pilot_positions = np.where(pilot_mask)
X_grid[pilot_positions] = Y_grid[pilot_positions] / (H_true[pilot_positions] + 1e-10)

rng = np.random.default_rng(0)
config_dd = DecisionDirectedConfig(llr_threshold=4.0)
estimator = DecisionDirectedEstimator(config_dd, modulation_order=16)

data_positions = np.where(~pilot_mask)
X_grid[data_positions] = rng.choice(estimator.llr_computer.constellation, size=len(data_positions[0]))

print("=" * 70)
print("DEBUG: Decision-Directed Behavior")
print("=" * 70)
print(f"\nNoise variance: {noise_var:.6f}")
print(f"LLR threshold: {config_dd.llr_threshold}")
print(f"Number of pilots: {np.sum(pilot_mask)}")
print(f"Number of data symbols: {np.sum(~pilot_mask)}")

result = estimator.estimate(H_pilot, Y_grid, X_grid, pilot_mask, noise_var)

print(f"\nAcceptance rates per symbol (first 10):")
for i, rate in enumerate(result['acceptance_rates'][:10]):
    print(f"  Symbol {i}: {rate*100:.1f}%")

print(f"\nOverall statistics:")
print(f"  Mean acceptance rate: {np.mean(result['acceptance_rates'])*100:.1f}%")
print(f"  Total accepted: {np.sum(result['dd_mask'])} / {np.sum(~pilot_mask)}")
print(f"  Mean normalizer mag: {np.mean(np.abs(result['normalizers'])):.4f}")
print(f"  Tracked noise var: {np.mean(result['noise_var']):.6f}")

sym_idx = 0
data_mask = ~pilot_mask[sym_idx]
Y_eq = Y_grid[sym_idx, data_mask] / (H_pilot[sym_idx, data_mask] + 1e-10)

llrs = estimator.llr_computer.compute_llrs(Y_eq, noise_var)
min_llrs = np.min(np.abs(llrs), axis=-1)

print(f"\nSample LLR values (symbol 0, first 10 data tones):")
for i, llr_val in enumerate(min_llrs[:10]):
    status = "ACCEPT" if llr_val > config_dd.llr_threshold else "REJECT"
    print(f"  Tone {i}: min|LLR| = {llr_val:.2f} [{status}]")

print(f"\nLLR statistics:")
print(f"  Min: {np.min(min_llrs):.2f}")
print(f"  Max: {np.max(min_llrs):.2f}")
print(f"  Mean: {np.mean(min_llrs):.2f}")
print(f"  Fraction > threshold: {np.mean(min_llrs > config_dd.llr_threshold)*100:.1f}%")