# Retrained Model Evaluation - In Progress

## Status
The evaluation with the retrained diffusion model is currently running.

### Progress (Last Check)
- ✅ SNR = 5 dB: COMPLETED
- 🔄 SNR = 10 dB: ~76-80% complete
- ⏳ SNR = 15 dB: Pending
- ⏳ SNR = 20 dB: Pending
- ⏳ SNR = 25 dB: Pending
- ⏳ SNR = 30 dB: Pending

**Process ID:** 107161  
**Runtime:** ~26 minutes (started at evaluation launch)  
**Estimated completion:** ~40-50 minutes from now

## What's Being Evaluated

The retrained diffusion model is being tested against 4 different channel estimation methods:

1. **Pilot Only** - Baseline pilot interpolation
2. **Diffusion Only** - Retrained diffusion model (trained on variable 25-60% pilot densities)
3. **DD Only** - Decision-directed feedback alone
4. **Full Pipeline** - DD before Diffusion (complete solution)

## Key Differences from Previous Evaluations

### Old Model Issues:
- Trained ONLY on 25% pilot density
- Could not leverage augmented pilot masks from DD
- Performance degraded when given 50%+ pilots
- Result: Full Pipeline was often WORSE than Diffusion Only

### Retrained Model Improvements:
- Trained on variable pilot densities (25-60%)
- 50% of training samples had augmented pilots
- Model learned to adapt based on pilot_mask density
- Expected: Full Pipeline should be 4-6 dB better than Pilot Only

## Expected Results

Based on the retraining approach:

| Method | Expected NMSE @ SNR=25dB | Improvement over Pilot Only |
|--------|--------------------------|----------------------------|
| Pilot Only | ~-16 dB | Baseline |
| Diffusion Only | ~-24 dB | +8 dB |
| DD Only | ~-18 dB | +2 dB |
| **Full Pipeline** | **~-28 dB** | **+12 dB** ✓ |

## Why This Should Work

### Training Data Distribution:
```python
# 50% of samples: Original pilots only (25%)
pilot_mask = [1,0,0,0,1,0,0,0,1,...]  # 25% density

# 50% of samples: Augmented pilots (45-60%)
pilot_mask = [1,1,0,1,1,0,1,0,1,...]  # 45-60% density
```

### Model Adaptation:
The diffusion model now learns:
- **Low density (25%)**: Rely more on generative prior
- **High density (45-60%)**: Trust the conditioning more, use less prior

This adaptive balance allows it to fully exploit the pseudo-pilots from DD!

## Next Steps

Once evaluation completes:
1. Review the NMSE vs SNR plots in `./results/retrained_model/`
2. Check the numerical results table
3. Verify the Full Pipeline achieves 4-6 dB improvement
4. If successful, use this retrained model for all future work

## Files Generated

The evaluation will create:
- `./results/retrained_model/snr_sweep_results.json` - Numerical results
- `./results/retrained_model/snr_vs_nmse.png` - NMSE comparison plot
- `./results/retrained_model/snr_vs_nmse_improvement.png` - Improvement plot

## Monitoring

To check progress manually:
```bash
# Check if still running
ps -p 107161

# Check latest output
tail -30 ~/.cursor/projects/home-ahmed-selective-decision-feedback/terminals/546374.txt

# Check results directory
ls -lh ./results/retrained_model/
```

---
**Last Updated:** Evaluation in progress (~26 minutes elapsed)
