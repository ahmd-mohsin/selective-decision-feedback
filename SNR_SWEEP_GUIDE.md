# SNR Sweep Evaluation Guide

## What This Does

Evaluates all channel estimation methods across different SNR values (5-30 dB) to show:
- How NMSE decreases with increasing SNR
- Which method performs best at each SNR
- Performance gaps between methods

## Methods Compared

1. **Pilot Only**: Linear interpolation baseline
2. **Diffusion Only**: Diffusion model with original pilots
3. **DD Only**: Decision-directed on pilot interpolation
4. **Full Pipeline**: DD + Diffusion (pseudo-pilots approach)

## Quick Start

### Option 1: Using the shell script (easiest)
```bash
cd /home/ahmed/selective-decision-feedback
chmod +x run_snr_sweep.sh
./run_snr_sweep.sh
```

### Option 2: Direct command
```bash
# Kill any GPU processes first
pkill -9 python
sleep 3

# Activate environment
conda activate rc_flow_env
cd /home/ahmed/selective-decision-feedback

# Run evaluation
python3 evaluate_snr_sweep.py \
    --diffusion_checkpoint checkpoints/module2/checkpoint_epoch_100.pt \
    --snr_min 5 \
    --snr_max 30 \
    --snr_step 5 \
    --num_samples 50 \
    --device cuda \
    --output_dir ./results/snr_sweep
```

## Command Options

```
--diffusion_checkpoint PATH  Path to trained diffusion model checkpoint (required)
--snr_min INT               Minimum SNR in dB (default: 5)
--snr_max INT               Maximum SNR in dB (default: 30)
--snr_step INT              SNR step size (default: 5)
--num_samples INT           Samples to average per SNR point (default: 50)
--device STR                Device: 'cuda' or 'cpu' (default: cuda)
--output_dir PATH           Where to save results (default: ./results/snr_sweep)
```

## Output Files

After running, you'll get:

1. **`snr_sweep.png`** - Visualization with two plots:
   - Left: NMSE (dB) vs SNR (dB) for all methods
   - Right: Improvement over Pilot Only baseline

2. **`snr_sweep_results.txt`** - Detailed table with:
   - NMSE values for each method at each SNR
   - Acceptance rates for Full Pipeline
   - Gain calculations

## Expected Results

At **low SNR** (5-10 dB):
- High noise → DD has low acceptance rate
- Full Pipeline similar to Diffusion Only
- Both much better than Pilot Only

At **medium SNR** (15-20 dB):
- DD acceptance rate increases (~70-85%)
- Full Pipeline starts to outperform Diffusion Only
- Clear benefit from pseudo-pilots

At **high SNR** (25-30 dB):
- DD acceptance rate very high (~90%+)
- Full Pipeline significantly better than Diffusion
- Maximum benefit from pseudo-pilots

## Example Results

```
SNR (dB)   Pilot Only   Diffusion   DD Only    Full Pipeline
---------------------------------------------------------------
5          -10.5        -15.2       -9.8       -15.3
10         -13.2        -18.5       -12.5      -18.9
15         -15.8        -21.7       -14.9      -22.5
20         -18.1        -24.3       -17.2      -25.8
25         -20.3        -26.8       -19.4      -28.9
30         -22.4        -29.1       -21.5      -31.7
```

## Runtime

- ~50 samples per SNR point
- 6 SNR points (5, 10, 15, 20, 25, 30 dB)
- Estimated time: **15-25 minutes** depending on GPU

For faster results, reduce `--num_samples` to 20 or 10.

## Troubleshooting

**GPU Out of Memory:**
```bash
pkill -9 python
sleep 5
# Then try again with fewer samples
python3 evaluate_snr_sweep.py ... --num_samples 20
```

**Slow execution:**
- Reduce SNR range: `--snr_min 10 --snr_max 25`
- Fewer samples: `--num_samples 20`
- Larger steps: `--snr_step 10`
