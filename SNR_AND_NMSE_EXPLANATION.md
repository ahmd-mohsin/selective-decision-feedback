# SNR Definition and NMSE Calculation Explained

## Question from Professor
"What is the SNR against noise? What is the dB value of the noise we are adding during transmission compared to the signal?"

## Answer

### SNR Definition in This System

The **SNR (Signal-to-Noise Ratio)** in this work is defined as:

```
SNR_dB = 10 × log₁₀(Signal_Power / Noise_Power)
```

**Key Implementation Details:**

1. **Signal Power = 1.0** (normalized)
   - All transmitted symbols are normalized to unit power
   - This is standard in wireless communication systems

2. **Noise Variance Calculation:**
   ```python
   # From transmission_system/config.py line 71-72
   noise_variance = 10^(-SNR_dB / 10)
   ```

3. **Noise Addition:**
   ```python
   # From transmission_system/channel.py line 178-180
   noise = sqrt(noise_var/2) × (randn + j×randn)
   Y_received = Y_clean + noise
   ```

### Example: What Noise Power is Added at Different SNRs?

| SNR (dB) | Noise Variance | Noise Std Dev | Signal:Noise Ratio |
|----------|----------------|---------------|-------------------|
| 5 dB     | 0.3162        | 0.5623        | 3.16:1           |
| 10 dB    | 0.1000        | 0.3162        | 10:1             |
| 15 dB    | 0.0316        | 0.1778        | 31.6:1           |
| 20 dB    | 0.0100        | 0.1000        | 100:1            |
| 25 dB    | 0.0032        | 0.0562        | 316:1            |
| 30 dB    | 0.0010        | 0.0316        | 1000:1           |

### Mathematical Derivation

Given:
- Signal power: `P_signal = 1.0` (normalized)
- SNR in dB: `SNR_dB`

The noise variance is:

```
SNR_dB = 10 × log₁₀(P_signal / σ²_noise)

Rearranging:
σ²_noise = P_signal / 10^(SNR_dB/10)

Since P_signal = 1.0:
σ²_noise = 10^(-SNR_dB/10)
```

For complex Gaussian noise (I and Q components):
```
noise_real ~ N(0, σ²_noise/2)
noise_imag ~ N(0, σ²_noise/2)

Total noise power = E[|noise|²] = σ²_noise
```

---

## NMSE Definition

The **Normalized Mean Square Error (NMSE)** is calculated as:

```
NMSE = E[||H_estimated - H_true||²] / E[||H_true||²]

NMSE_dB = 10 × log₁₀(NMSE)
```

**Important:** NMSE is calculated **against the true channel H_true**, NOT against the baseline!

### Example from Results

At SNR = 10 dB:
- **Pilot Only NMSE:** -10.22 dB
  - This means: `||H_pilot - H_true||² / ||H_true||² = 10^(-10.22/10) = 0.0950`
  - Error power is 9.5% of signal power

- **Full Pipeline NMSE:** -15.56 dB  
  - This means: `||H_full - H_true||² / ||H_true||² = 10^(-15.56/10) = 0.0278`
  - Error power is 2.78% of signal power

- **Improvement:** 5.34 dB
  - This is the difference: -15.56 - (-10.22) = 5.34 dB
  - Proportionally: 0.0950 / 0.0278 = 3.42× better (in linear scale)

---

## Relationship Between Input SNR and Output NMSE

### Key Insight
The **input SNR** (noise added during transmission) determines how difficult the channel estimation problem is:

| Input SNR | Difficulty | Typical Pilot Interp NMSE |
|-----------|-----------|---------------------------|
| 5 dB      | Very Hard | ~-6 dB (error = 25% of signal) |
| 10 dB     | Hard      | ~-10 dB (error = 10% of signal) |
| 20 dB     | Medium    | ~-16 dB (error = 2.5% of signal) |
| 30 dB     | Easy      | ~-17.5 dB (error = 1.8% of signal) |

### Why Doesn't NMSE Keep Improving at High SNR?

At very high SNR (25-30 dB), the NMSE plateaus around -17 to -18 dB because:

1. **Interpolation Error Dominates:** With only 25% pilots, there's inherent uncertainty about the channel between pilots, even with zero noise
2. **Time-Varying Channel:** The channel changes over time (Doppler = 100 Hz), so interpolation has fundamental limits
3. **Quantization and Numerical Effects:** At some point, numerical precision limits improvements

---

## Summary for Your Professor

### What SNR Means
- **SNR = X dB** means the received signal power is `10^(X/10)` times larger than the noise power
- We add complex Gaussian noise with variance `σ² = 10^(-SNR_dB/10)` to the normalized signal (power = 1.0)
- At SNR = 20 dB: noise variance = 0.01, noise std dev = 0.1

### What NMSE Means  
- **NMSE measures estimation error relative to the true channel**, not relative to baseline
- **NMSE = -X dB** means the estimation error power is `10^(-X/10)` times the true channel power
- At NMSE = -15 dB: error is 3.16% of channel power

### Key Result
At SNR = 10 dB (noise variance = 0.1):
- Baseline pilot interpolation: NMSE = -10.22 dB (9.5% error)
- Our Full Pipeline: NMSE = -15.56 dB (2.78% error)  
- **Improvement: 5.34 dB** (3.42× better in linear scale)

This means our method reduces the channel estimation error by more than 3× compared to standard pilot-based interpolation, when the transmission noise variance is 0.1 (SNR = 10 dB).
