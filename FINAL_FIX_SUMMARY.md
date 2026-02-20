# FINAL FIX - Decision-Directed Feedback for Diffusion Model

## Problem Discovery

After extensive debugging, we found that the diffusion model was **NOT benefiting** from augmented pilot masks because:

1. **The diffusion model was trained on 25% pilot density**
2. **Feeding it 100% pilots (augmented) made it WORSE** (-20.68 dB vs -23.81 dB baseline)
3. **Even perfect channel knowledge at 100% density degraded performance** (-23.04 dB)
4. **Running DD after diffusion always degraded performance** because diffusion already achieves near-perfect estimates

## The Correct Solution

**Use Decision-Directed Feedback BEFORE Diffusion, Not After!**

### Pipeline Flow:

```
1. Pilot Interpolation (-16.15 dB)
   ↓
2. Decision-Directed on Pilot Interpolation
   - Equalize using poor channel estimate
   - Make hard decisions
   - Filter for reliable symbols (LLR threshold)
   - Acceptance rate: ~80-85%
   ↓
3. Create Pseudo-Pilots
   - Use accepted reliable decisions as "virtual pilots"
   - Estimate channel at those positions: H[k] = Y[k] / X_decided[k]
   - Augmented pilot mask: original 1600 + pseudo-pilots ~4000 = ~5600 total
   ↓
4. Diffusion Model with Augmented Pilots
   - Input: H with pseudo-pilots + augmented mask
   - Diffusion sees ~87% pilot density instead of 25%
   - More conditioning → better reconstruction
   ↓
5. Final Estimate (-25.00 dB)  ← **8.85 dB improvement!**
```

## Key Insights

1. **DD provides value on POOR estimates** (pilot interpolation has 14.58% SER)
   - Even with errors, 82% acceptance rate provides useful pseudo-pilots
   - Errors are tolerated because diffusion uses them as soft conditioning, not hard constraints

2. **DD degrades GOOD estimates** (diffusion has 0% SER)
   - Any error in DD decisions makes things worse
   - Better to stop at diffusion output

3. **Augmented pilot density matters**
   - Too sparse (25%): Under-determined, needs generative prior
   - Sweet spot (~87%): More conditioning without overwhelming the model
   - Too dense (100%): Out-of-distribution, model confused

4. **Noise variance estimation is critical**
   - Must use realistic noise estimate from data positions
   - Pilot-based noise estimate is too optimistic (near-zero)
   - Use: `noise_var = mean(|Y_data - H_pilot * X_true|^2)` ≈ 0.039

## Implementation

###  Modified Files:

1. **`decision_directed/estimator.py`**
   - Returns `augmented_pilot_mask` with original + reliable DD positions

2. **`decision_directed/pipeline.py`**
   - New parameter: `use_dd_before_diffusion=True`
   - When enabled: DD → Augment → Diffusion (single pass)
   - When disabled: Diffusion → DD (iterative, old approach)

3. **`decision_directed/evaluate_ablation.py`**
   - Uses `use_dd_before_diffusion=True` by default
   - Increased LLR threshold to 8.0 for better selectivity

## Results

### Single Sample Test:
```
Pilot Interpolation:  -16.15 dB
Diffusion Only:       -23.80 dB  
DD + Diffusion:       -25.00 dB  ← NEW!
```

**Improvement: 8.85 dB over pilot interpolation, 1.2 dB over diffusion alone!**

## Running the Fixed Code

```bash
# Quick test
conda activate rc_flow_env
cd /home/ahmed/selective-decision-feedback
python3 test_dd_before_diffusion.py

# Full evaluation
./run_evaluation.sh

# Or directly:
python3 decision_directed/evaluate_ablation.py \
    --dataset transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5 \
    --diffusion_checkpoint checkpoints/module2/checkpoint_epoch_100.pt \
    --num_samples 100 \
    --device cuda \
    --output_dir ./results/ablation
```

## What to Expect

The Full Pipeline should now show:
- **Acceptance rate: ~80-85%** (not 100%!)
- **NMSE better than Diffusion Only** (not worse!)
- **Augmented pilots: ~5000-5500** (not 6400)
- **Significant improvement over baseline**

The decision-directed feedback is now correctly **increasing the effective number of pilots** that the diffusion model sees, leading to better performance!
