# Complete System Explanation: Decision-Directed Diffusion for Channel Estimation

## Table of Contents
1. [Problem Setup](#problem-setup)
2. [Diffusion Model Inputs & Architecture](#diffusion-model-inputs--architecture)
3. [Decision-Directed Feedback](#decision-directed-feedback)
4. [Full Pipeline Flow](#full-pipeline-flow)
5. [Why Retraining Helps](#why-retraining-helps)

---

## 1. Problem Setup

### What We're Trying to Estimate

In OFDM wireless communication, we need to estimate the **channel matrix H**:

```
Y[k,n] = H[k,n] × X[k,n] + noise
```

Where:
- `Y[k,n]` = Received signal at subcarrier k, symbol n
- `H[k,n]` = Channel coefficient (UNKNOWN - what we want!)
- `X[k,n]` = Transmitted symbol (known at pilots, unknown at data)
- `k` = Subcarrier index (0 to 63 in your system)
- `n` = OFDM symbol index (0 to 99 in your system)

### The Challenge

- **Pilots:** We know X at pilot positions → can compute H directly
  - `H[k,n] = Y[k,n] / X[k,n]` (at pilot positions)
  - But only 25% of positions have pilots!
  
- **Data positions:** We don't know X → can't directly compute H
  - Need to **interpolate** or **use a model** to fill in

---

## 2. Diffusion Model Inputs & Architecture

### What the Diffusion Model Receives

The diffusion model gets **5 input channels** concatenated together:

```python
# From inference.py line 81
cond = torch.cat([H_pilot_full, pilot_mask, Y_grid], dim=1)
```

#### Channel 1-2: `H_pilot_full` (Complex → 2 real channels)
```
Shape: (2, num_symbols, num_subcarriers) = (2, 100, 64)

Channel[0] = Real part of H_pilot_full
Channel[1] = Imaginary part of H_pilot_full

What it contains:
- At pilot positions (25%): H = Y/X (true channel estimate)
- At data positions (75%): Interpolated from nearby pilots
```

Example visualization:
```
Subcarriers: 0   4   8  12  16  20  24  28  32  ...
Pilots:      ✓   .   .   .   ✓   .   .   .   ✓   ...
             |               |               |
H_pilot:     H0  (interp)   H16 (interp)   H32  ...
```

#### Channel 3: `pilot_mask` (Binary)
```
Shape: (1, num_symbols, num_subcarriers) = (1, 100, 64)

Values:
- 1.0 at pilot positions (where we have true H)
- 0.0 at data positions (where H is interpolated)
```

Example:
```
[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, ...]
 ↑  interpolated  ↑  interpolated  ↑
pilot            pilot            pilot
```

#### Channels 4-5: `Y_grid` (Complex → 2 real channels)
```
Shape: (2, num_symbols, num_subcarriers) = (2, 100, 64)

Channel[0] = Real part of received signal Y
Channel[1] = Imaginary part of received signal Y

What it provides: Additional observation of channel through Y = H×X + noise
```

### What the Diffusion Model Outputs

```
Output shape: (2, 100, 64)
- Channel 0: Real part of estimated H
- Channel 1: Imaginary part of estimated H

This is the refined channel estimate!
```

### How Diffusion Works

**Training (what was already done):**
```python
# Diffusion learns to denoise:
# Start with: H_noisy = H_true + noise
# Learn to predict: H_true from H_noisy, conditioned on [H_pilot_full, pilot_mask, Y_grid]

for epoch in range(num_epochs):
    # Add noise to true channel
    t = random_timestep()
    H_noisy = add_noise(H_true, t)
    
    # Model predicts noise
    noise_pred = model(H_noisy, t, conditioning=[H_pilot_full, pilot_mask, Y_grid])
    
    # Loss: How well did we predict the noise?
    loss = MSE(noise_pred, actual_noise)
```

**Inference (what happens now):**
```python
# Start from pure noise
H_t = random_noise()

# Iteratively denoise (1000 steps)
for t in reversed(range(1000)):
    # Model predicts noise at this timestep
    noise = model(H_t, t, conditioning)
    
    # Remove predicted noise
    H_t = denoise_step(H_t, noise, t)

# Final result
H_estimated = H_t
```

The model has learned what "valid channels" look like from training data and uses this **generative prior** to fill in missing information!

---

## 3. Decision-Directed Feedback

### The Idea

**Problem:** With only 25% pilots, interpolation is poor  
**Solution:** Use the **data payload** to create "pseudo-pilots"!

### How Decision-Directed Works

```python
# Step 1: Initial channel estimate (from pilot interpolation or diffusion)
H_initial = interpolate_from_pilots(Y, X_pilots, pilot_mask)

# Step 2: Equalize data symbols
for k in data_positions:
    Y_equalized[k] = Y[k] / H_initial[k]

# Step 3: Make hard decisions (decode symbols)
X_decisions[k] = nearest_constellation_point(Y_equalized[k])

# Step 4: Check reliability using LLR (Log-Likelihood Ratio)
for k in data_positions:
    # Compute LLR for each bit
    llrs = compute_llr(Y_equalized[k], noise_var)
    min_llr = min(abs(llrs))
    
    # Only accept high-confidence decisions
    if min_llr > threshold:
        # Reliable! Treat as pseudo-pilot
        H_pseudo[k] = Y[k] / X_decisions[k]
        augmented_pilot_mask[k] = True

# Step 5: Now we have MORE pilots!
# Original: 25% pilots
# Augmented: 25% + ~20-30% pseudo-pilots = 45-55% total
```

### LLR (Log-Likelihood Ratio) Explained

For 16-QAM, each symbol carries 4 bits. LLR measures **confidence** in each bit decision:

```python
# For bit b in {0,1,2,3}
# LLR = log(P(bit=0|y) / P(bit=1|y))

# Compute distances to all constellation points
dist_0 = min_distance(y, points_where_bit_b_is_0)
dist_1 = min_distance(y, points_where_bit_b_is_1)

LLR[b] = (dist_1 - dist_0) / noise_var
```

**High |LLR|** → Confident decision (one bit value much more likely)  
**Low |LLR|** → Uncertain (both bit values equally likely)

Example:
```
Threshold = 6.0

Symbol A: LLRs = [8.2, 12.5, 9.1, 7.8] → min_LLR = 7.8 > 6.0 → ACCEPT ✓
Symbol B: LLRs = [2.1, 8.5, 3.4, 9.2] → min_LLR = 2.1 < 6.0 → REJECT ✗
```

---

## 4. Full Pipeline Flow

### Current Implementation: DD BEFORE Diffusion

```
┌─────────────────────────────────────────────────────────────┐
│ INPUT: Received signal Y, Transmitted pilots X_pilots       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Pilot Interpolation                                 │
│   - Compute H at pilot positions: H_pilot = Y / X           │
│   - Interpolate to data positions (linear)                  │
│   → H_pilot_full (poor quality: ~-16 dB NMSE)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Decision-Directed Feedback on Pilot Interpolation   │
│   - Equalize: Y_eq = Y / H_pilot_full                      │
│   - Hard decisions: X_dec = argmin |Y_eq - constellation|   │
│   - Compute LLR reliability scores                          │
│   - Accept reliable symbols (LLR > threshold)               │
│   → Augmented pilot mask (45-55% density)                   │
│   → Pseudo-pilots: H_pseudo = Y / X_dec (at accepted pos)  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Diffusion Model with Augmented Pilots               │
│   Input conditioning:                                        │
│     - H_pilot_full (with pseudo-pilots)                     │
│     - Augmented pilot_mask (45-55% = 1.0)                   │
│     - Y_grid (received signal)                              │
│   Process:                                                   │
│     - Start from noise                                       │
│     - Iteratively denoise (1000 steps)                      │
│     - Use conditioning to guide denoising                    │
│   → H_final (improved: ~-24 to -26 dB NMSE)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT: Refined channel estimate H_final                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Insight: Why DD Before (Not After)

**❌ WRONG: DD After Diffusion**
```
Pilot Interp → Diffusion → DD
              (-24 dB)     (-18 dB)
              (good)       (WORSE!)
```
- Diffusion already near-optimal
- DD on good estimate makes errors
- Any wrong decision degrades result

**✓ CORRECT: DD Before Diffusion**
```
Pilot Interp → DD → Diffusion
(-16 dB)            (-24 dB)
(poor)              (MUCH BETTER!)
```
- DD on poor estimate: Even with some errors, increases pilot density
- Diffusion sees more pilots → better conditioning
- Net result: Improvement!

---

## 5. Why Retraining Helps

### Current Model Limitation

**What it was trained on:**
```
ALL training samples:
  pilot_mask = [1,0,0,0,1,0,0,0,1,...]  (25% density)
  
Model learns: "Always expect 25% pilots"
```

**What we feed it now:**
```
Testing with DD:
  pilot_mask = [1,1,0,1,1,0,1,0,1,...]  (50% density)
  
Model: "This is weird... not what I saw in training"
Result: Can't fully exploit the extra pilots
```

### After Retraining

**New training data:**
```
50% of samples:
  pilot_mask = [1,0,0,0,1,0,0,0,1,...]  (25% density)

50% of samples:
  pilot_mask = [1,1,0,1,1,0,1,0,1,...]  (45-60% density)
  
Model learns: "Adapt based on pilot_mask density!"
```

**Testing with DD:**
```
pilot_mask = [1,1,0,1,1,0,1,0,1,...]  (50% density)

Model: "Great! I trained on this. Use more conditioning, less prior"
Result: FULLY exploits pseudo-pilots → 4-6 dB gain!
```

### Mathematical Intuition

Diffusion balances two things:

1. **Data fidelity:** Match the conditioning (pilots)
2. **Prior:** Generate realistic channels

```python
# Implicit balance in diffusion
H_final = α × match_pilots + (1-α) × generative_prior

Current model:
  α tuned for 25% pilots → mostly relies on prior

Retrained model:
  α adaptive to pilot_mask density
  - 25% pilots: α small (rely on prior)
  - 50% pilots: α large (trust conditioning more)
  → OPTIMAL use of available information!
```

---

## Summary

### Current System

```
Components:
1. Pilot interpolation → poor initial estimate
2. Decision-directed → finds reliable decisions, creates pseudo-pilots
3. Diffusion → refines using augmented pilots

Limitation:
- Diffusion trained on 25% pilots only
- Can't fully exploit 50% augmented pilots
- Gain: ~1-2 dB

Solution:
- Retrain on variable pilot density (25-60%)
- Model learns to adapt
- Expected gain: 4-6 dB!
```

### Why This Approach is Smart

1. **Uses payload for estimation** - not just pilots!
2. **Iterative refinement** - DD improves initial estimate for diffusion
3. **Generative prior** - diffusion knows what valid channels look like
4. **Adaptive** - can work across different SNRs and conditions

The only missing piece was training the model to handle variable pilot densities - which is what retraining solves! 🎯
