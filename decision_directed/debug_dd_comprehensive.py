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

signal_power = np.mean(np.abs(Y_grid)**2)
snr_linear = 10**(20.0 / 10.0)
noise_var = signal_power / snr_linear

print("=" * 70)
print("COMPREHENSIVE DD DEBUG")
print("=" * 70)
print(f"\nDataset info:")
print(f"  Stored noise_var: {data['noise_var'][idx]:.2e}")
print(f"  Computed noise_var: {noise_var:.6f}")
print(f"  Signal power: {signal_power:.6f}")
print(f"  SNR: 20.0 dB (linear: {snr_linear:.2f})")

print(f"\nPilot info:")
print(f"  Total pilots: {np.sum(pilot_mask)}")
print(f"  Total data: {np.sum(~pilot_mask)}")
print(f"  Pilot NMSE: {10*np.log10(compute_channel_nmse(H_pilot, H_true)):.2f} dB")

config_dd = DecisionDirectedConfig(
    llr_threshold=4.0,
    normalizer_step_size=0.01,
    adaptive_threshold=True
)
estimator = DecisionDirectedEstimator(config_dd, modulation_order=16)

print(f"\nDD Config:")
print(f"  LLR threshold: {config_dd.llr_threshold}")
print(f"  Normalizer step: {config_dd.normalizer_step_size}")
print(f"  Adaptive threshold: {config_dd.adaptive_threshold}")

print("\n" + "=" * 70)
print("RUNNING DD ESTIMATION")
print("=" * 70)

result = estimator.estimate(H_pilot, Y_grid, X_grid, pilot_mask, noise_var)

print(f"\nAcceptance statistics:")
print(f"  Mean acceptance rate: {np.mean(result['acceptance_rates'])*100:.1f}%")
print(f"  Min acceptance rate: {np.min(result['acceptance_rates'])*100:.1f}%")
print(f"  Max acceptance rate: {np.max(result['acceptance_rates'])*100:.1f}%")
print(f"  Total accepted: {np.sum(result['dd_mask'])} / {np.sum(~pilot_mask)}")

print(f"\nNormalizer statistics:")
print(f"  Mean magnitude: {np.mean(np.abs(result['normalizers'])):.6f}")
print(f"  Std magnitude: {np.std(np.abs(result['normalizers'])):.6f}")
print(f"  Min magnitude: {np.min(np.abs(result['normalizers'])):.6f}")
print(f"  Max magnitude: {np.max(np.abs(result['normalizers'])):.6f}")

print(f"\nNoise tracking:")
print(f"  Initial noise_var: {result['noise_var'][0]:.6f}")
print(f"  Final noise_var: {result['noise_var'][-1]:.6f}")
print(f"  Mean noise_var: {np.mean(result['noise_var']):.6f}")

H_dd = result['H_dd']
nmse_dd = compute_channel_nmse(H_dd, H_true)
nmse_pilot = compute_channel_nmse(H_pilot, H_true)

print(f"\nPerformance:")
print(f"  Pilot NMSE: {10*np.log10(nmse_pilot):.2f} dB")
print(f"  DD NMSE: {10*np.log10(nmse_dd):.2f} dB")
print(f"  Improvement: {10*np.log10(nmse_pilot/nmse_dd):.2f} dB")

sym_idx = 0
data_mask = ~pilot_mask[sym_idx]
data_indices = np.where(data_mask)[0]

print(f"\n" + "=" * 70)
print(f"DETAILED ANALYSIS: Symbol {sym_idx}")
print("=" * 70)

H_current = H_pilot[sym_idx, data_mask]
Y_eq = Y_grid[sym_idx, data_mask] / (H_current + 1e-10)

print(f"\nEqualized signal statistics:")
print(f"  Mean |Y_eq|: {np.mean(np.abs(Y_eq)):.4f}")
print(f"  Std |Y_eq|: {np.std(np.abs(Y_eq)):.4f}")

llrs = estimator.llr_computer.compute_llrs(Y_eq, noise_var)
min_llrs = np.min(np.abs(llrs), axis=-1)

print(f"\nLLR statistics:")
print(f"  Mean min|LLR|: {np.mean(min_llrs):.2f}")
print(f"  Std min|LLR|: {np.std(min_llrs):.2f}")
print(f"  Min min|LLR|: {np.min(min_llrs):.2f}")
print(f"  Max min|LLR|: {np.max(min_llrs):.2f}")
print(f"  Fraction > 4.0: {np.mean(min_llrs > 4.0)*100:.1f}%")
print(f"  Fraction > 3.0: {np.mean(min_llrs > 3.0)*100:.1f}%")
print(f"  Fraction > 2.0: {np.mean(min_llrs > 2.0)*100:.1f}%")

hard_decisions = estimator.llr_computer.hard_decision(Y_eq)
X_true = X_grid[sym_idx, data_mask]

decision_errors = np.sum(hard_decisions != X_true)
print(f"\nDecision accuracy:")
print(f"  Correct decisions: {len(X_true) - decision_errors} / {len(X_true)}")
print(f"  Error rate: {decision_errors / len(X_true) * 100:.1f}%")

print(f"\nSample LLRs and decisions (first 10 data tones):")
for i in range(min(10, len(min_llrs))):
    llr_val = min_llrs[i]
    decision = hard_decisions[i]
    true_sym = X_true[i]
    correct = "✓" if decision == true_sym else "✗"
    status = "ACCEPT" if llr_val > config_dd.llr_threshold else "REJECT"
    print(f"  Tone {data_indices[i]:2d}: LLR={llr_val:6.2f} [{status}] Decision={correct} |X̂|={abs(decision):.3f} |X|={abs(true_sym):.3f}")

if np.mean(min_llrs) < 2.0:
    print("\n⚠️  WARNING: Very low LLRs! Possible issues:")
    print("  - Noise variance too high")
    print("  - Channel estimate very poor")
    print("  - Equalization failing")
elif np.mean(result['acceptance_rates']) < 0.1:
    print("\n⚠️  WARNING: Very low acceptance rate! Possible issues:")
    print("  - LLR threshold too high")
    print("  - Noise variance estimation incorrect")
elif abs(np.mean(np.abs(result['normalizers'])) - 1.0) > 0.1:
    print("\n⚠️  WARNING: Normalizers diverging from 1.0!")
    print("  - Check step size")
    print("  - Check error computation")