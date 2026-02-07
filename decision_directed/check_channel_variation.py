import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from transmission_system.dataset_generator import load_dataset_hdf5

data, config = load_dataset_hdf5(
    '../transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5'
)

idx = 0
H_true = data['H_true'][idx]

print("=" * 70)
print("CHANNEL TIME VARIATION ANALYSIS")
print("=" * 70)

print(f"\nChannel shape: {H_true.shape} (symbols × subcarriers)")
print(f"Doppler: {data['doppler_hz'][idx]} Hz")

H_mag = np.abs(H_true)
H_phase = np.angle(H_true)

print(f"\nMagnitude variation across symbols:")
for k in [0, 16, 32, 48, 63]:
    mag_over_time = H_mag[:, k]
    print(f"  Subcarrier {k:2d}: mean={np.mean(mag_over_time):.4f}, std={np.std(mag_over_time):.6f}, range=[{np.min(mag_over_time):.4f}, {np.max(mag_over_time):.4f}]")

print(f"\nPhase variation across symbols (first subcarrier):")
phase_diff = np.diff(H_phase[:, 0])
phase_diff_wrapped = np.angle(np.exp(1j * phase_diff))
print(f"  Mean phase change: {np.mean(np.abs(phase_diff_wrapped)):.6f} rad/symbol")
print(f"  Max phase change: {np.max(np.abs(phase_diff_wrapped)):.6f} rad/symbol")

correlation_adjacent = []
for k in range(64):
    corr = np.corrcoef(H_true[:-1, k], H_true[1:, k])[0, 1]
    correlation_adjacent.append(corr)

print(f"\nTemporal correlation (symbol n vs n+1):")
print(f"  Mean correlation: {np.mean(correlation_adjacent):.6f}")
print(f"  Min correlation: {np.min(correlation_adjacent):.6f}")

if np.mean(correlation_adjacent) > 0.999:
    print("\n⚠️  PROBLEM: Channel is nearly STATIC (corr > 0.999)!")
    print("   DD tracking has nothing to track!")
    print("   Doppler might be too low or channel model issue.")
elif np.mean(correlation_adjacent) > 0.99:
    print("\n✓ Channel varies slowly (good for DD tracking)")
elif np.mean(correlation_adjacent) > 0.95:
    print("\n✓ Channel varies moderately (ideal for DD tracking)")
else:
    print("\n⚠️  Channel varies very fast (DD may struggle)")

H_first_10 = H_true[:10, 32]
H_last_10 = H_true[-10:, 32]
print(f"\nChannel drift (subcarrier 32):")
print(f"  First 10 symbols mean: {np.mean(H_first_10):.4f}")
print(f"  Last 10 symbols mean: {np.mean(H_last_10):.4f}")
print(f"  Drift: {np.abs(np.mean(H_last_10) - np.mean(H_first_10)):.6f}")