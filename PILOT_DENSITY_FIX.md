# CRITICAL FIX: Pilot Density Capping

## Problem Discovered from SNR Sweep

At **high SNR (25-30 dB)**, Full Pipeline was **WORSE** than Diffusion Only:

```
SNR    Diffusion   Full Pipeline   Accept Rate   Issue
20dB   -23.82      -24.31 (better)   44.6%       OK
25dB   -24.54      -23.99 (WORSE!)   71.2%       Too many pilots!
30dB   -24.74      -24.02 (WORSE!)   77.7%       Too many pilots!
```

### Root Cause

1. At high SNR, DD accepts **71-77%** of symbols
2. Original pilots (25%) + pseudo-pilots (71%) = **~90% pilot density**
3. **Diffusion was trained on 25% pilot density**
4. **High pilot density confuses the model** (out-of-distribution)

We proved this earlier:
- 25% pilots: -23.81 dB (good)
- 100% pilots: -20.68 dB (WORSE by 3 dB!)

## The Fix

### 1. Cap Maximum Pilot Density at 60%

Added logic in `decision_directed/pipeline.py`:

```python
max_pilot_density = 0.60  # Cap at 60%

if current_density > max_pilot_density:
    # Randomly subsample pseudo-pilots to stay under cap
    # Keep: original pilots + reduced pseudo-pilots
```

### 2. Increase LLR Threshold to 12.0

Changed in `evaluate_snr_sweep.py`:
```python
llr_threshold=12.0  # Was 8.0 - more selective at high SNR
```

This reduces acceptance rate at high SNR while maintaining benefit at medium SNR.

## Expected Behavior After Fix

### Low SNR (5-15 dB)
- Few reliable decisions (low acceptance)
- Full Pipeline ≈ Diffusion Only
- Pilot density: ~30-40%

### Medium SNR (20 dB)
- Good acceptance rate (~40-50%)
- Full Pipeline > Diffusion Only
- Pilot density: ~50-60% (optimal!)

### High SNR (25-30 dB)
- High acceptance but capped
- Full Pipeline should still beat or match Diffusion
- Pilot density: capped at 60%

## How to Test

### Quick test at SNR=25 dB:
```bash
pkill -9 python
sleep 3
conda activate rc_flow_env
python3 test_pilot_density_cap.py
```

Expected output:
- Pilot density: ~60% (capped)
- Full Pipeline should be better or equal to Diffusion

### Re-run full SNR sweep:
```bash
pkill -9 python
sleep 3
./run_snr_sweep.sh
```

Expected: Full Pipeline should now beat or match Diffusion at ALL SNR values!

## Key Insight

**There's a sweet spot for pilot density:**
- Too few (25%): Under-determined, needs strong prior
- Just right (40-60%): Best balance of conditioning + generative prior
- Too many (>70%): Over-constrains, degrades generative model

Decision-directed feedback should add pilots to reach the **optimal density range**, not blindly accept everything at high SNR.
