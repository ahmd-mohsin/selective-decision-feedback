# Decision-Directed Feedback Fix - Summary of Changes

## Problem Identified

The decision-directed (DD) feedback was **NOT increasing the effective number of pilots** for the diffusion model because:

1. **Old behavior**: DD estimator found reliable decisions but only used nearest-neighbor interpolation internally
2. **Critical issue**: The `pilot_mask` passed to the diffusion model remained **unchanged** 
3. **Result**: The diffusion model didn't know about the additional reliable "virtual pilots" from DD decisions

## What Was Fixed

### 1. Modified `decision_directed/estimator.py`

**Key changes:**
- Added `augmented_pilot_mask` to track original pilots + reliable DD decisions
- Changed channel refinement from nearest-neighbor interpolation to direct estimation at reliable positions
- Returns the augmented mask so the diffusion model can use it

**Before (lines 76-87):**
```python
# Used nearest-neighbor interpolation - didn't actually add pilots
all_pilot_indices = np.concatenate([original_pilots, accepted_indices])
all_pilot_values = np.concatenate([...])
for k in range(num_subcarriers):
    distances = np.abs(all_pilot_indices - k)
    H_refined[sym_idx, k] = all_pilot_values[nearest_idx]
```

**After (lines 59-64):**
```python
# Direct channel estimation at reliable positions
if len(accepted_indices) > 0:
    for idx, decision in zip(accepted_indices, hard_decisions[reliable_mask]):
        H_refined[sym_idx, idx] = Y_grid[sym_idx, idx] / (decision + 1e-10)
    
    augmented_pilot_mask[sym_idx, accepted_indices] = True
```

**New return value:**
- Added `'augmented_pilot_mask'` to the returned dictionary

### 2. Modified `decision_directed/pipeline.py`

**Key changes:**
- Added iterative refinement with `num_iterations` parameter (default=2)
- Uses augmented pilot mask in subsequent diffusion calls
- Tracks all DD iterations

**New function signature:**
```python
def estimate_full_pipeline(
    self,
    Y_grid: np.ndarray,
    X_grid: np.ndarray,
    pilot_mask: np.ndarray,
    H_pilot_full: Optional[np.ndarray] = None,
    noise_var: Optional[float] = None,
    num_iterations: int = 2  # NEW PARAMETER
) -> Dict[str, np.ndarray]:
```

**Iterative refinement loop (lines 46-69):**
```python
for iteration in range(num_iterations):
    # Run DD estimation
    dd_result = self.dd_estimator.estimate(...)
    
    # Get augmented mask with virtual pilots
    augmented_mask = dd_result['augmented_pilot_mask']
    
    # Feed back to diffusion model with MORE pilots
    if self.diffusion is not None and iteration < num_iterations - 1:
        H_augmented = dd_result['H_dd']
        diffusion_result = self.estimate_diffusion_only(
            Y_grid, H_augmented, augmented_mask  # USES AUGMENTED MASK!
        )
        H_current = diffusion_result['H_estimate']
```

### 3. Modified `decision_directed/evaluate_ablation.py`

**Key changes:**
- Added `num_iterations` parameter to evaluation
- Tracks acceptance rates across iterations
- Command-line argument for configurable iterations

## How It Works Now

### Pipeline Flow:

```
1. Initial Pilot Interpolation
   └─> H_pilot_full with original pilot_mask (e.g., 25% density)

2. Diffusion Model (First Pass)
   └─> Input: H_pilot_full + pilot_mask (25% pilots)
   └─> Output: H_diffusion (improved estimate)

3. Decision-Directed Feedback (Iteration 1)
   └─> Input: H_diffusion
   └─> Equalization → Hard decisions → Reliability check (LLR threshold)
   └─> Output: 
       - augmented_pilot_mask (25% + new virtual pilots, e.g., 100%)
       - H_refined with channel estimates at reliable positions

4. Diffusion Model (Second Pass) **← KEY FIX**
   └─> Input: H_refined + augmented_pilot_mask (100% pilots!)
   └─> Output: H_final (even better estimate with MORE conditioning)

5. Decision-Directed Feedback (Iteration 2)
   └─> Further refinement if needed
```

## Expected Improvements

### Before Fix:
- Diffusion saw only 16 pilots per OFDM symbol (25% density)
- DD decisions were wasted (interpolation only)
- No iterative improvement

### After Fix:
- Diffusion sees 16-64 pilots per symbol (25%-100% density)
- DD decisions become "virtual pilots" 
- Iterative refinement: DD → Diffusion → DD → Diffusion
- **Should significantly improve accuracy!**

## Test Results

```bash
$ python3 test_dd_fix.py

Original pilot_mask shape: (100, 64)
Number of original pilots: 1600
Pilot density: 25.00%

Decision-directed results:
  DD mask created: 4800 reliable decisions

✓ Augmented pilot mask created!
  Original pilots: 1600
  Augmented pilots: 6400
  New virtual pilots: 4800
  New pilot density: 100.00%

✓ Verification passed: All original pilots preserved in augmented mask

Acceptance rate: 100.00%

TEST PASSED: Decision-directed feedback correctly augments pilot mask!
```

## Files Modified

1. `decision_directed/estimator.py` - Core DD estimator with augmented mask
2. `decision_directed/pipeline.py` - Iterative pipeline with feedback loop
3. `decision_directed/evaluate_ablation.py` - Evaluation script with iteration support

## Files Created

1. `test_dd_fix.py` - Unit test for augmented pilot mask
2. `run_evaluation.sh` - Convenience script to run full evaluation
3. `CHANGES_SUMMARY.md` - This file

## How to Run

### Quick Test:
```bash
conda activate rc_flow_env
python3 test_dd_fix.py
```

### Full Evaluation:
```bash
# Method 1: Using the shell script
./run_evaluation.sh

# Method 2: Direct command
conda activate rc_flow_env
python3 decision_directed/evaluate_ablation.py \
    --dataset transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5 \
    --diffusion_checkpoint checkpoints/module2/checkpoint_epoch_100.pt \
    --num_samples 100 \
    --num_iterations 2 \
    --device cuda \
    --output_dir ./results/ablation
```

### Custom iterations:
```bash
# Try 1 iteration (no feedback)
python3 decision_directed/evaluate_ablation.py ... --num_iterations 1

# Try 3 iterations (more refinement)
python3 decision_directed/evaluate_ablation.py ... --num_iterations 3
```

## Expected Output

The evaluation will produce:
- `results/ablation/ablation_study.png` - Bar plot comparing methods
- `results/ablation/ablation_results.txt` - Detailed NMSE results

Methods compared:
1. **Pilot Only** - Linear interpolation baseline
2. **Diffusion Only** - Diffusion model with original pilots
3. **DD Only** - Decision-directed on pilot interpolation
4. **Full Pipeline** - Iterative DD + Diffusion (FIXED!)

The Full Pipeline should now show **significant improvement** because the diffusion model actually benefits from the additional virtual pilots!
