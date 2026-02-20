#!/bin/bash
# Run SNR Sweep Evaluation

echo "=========================================="
echo "SNR Sweep Evaluation"
echo "=========================================="
echo ""

# Kill any existing Python processes
echo "Cleaning GPU memory..."
pkill -9 python 2>/dev/null
sleep 3

# Configuration
CHECKPOINT="/home/ahmed/selective-decision-feedback/checkpoints/module2/checkpoint_epoch_100.pt"
OUTPUT_DIR="./results/snr_sweep"
SNR_MIN=5
SNR_MAX=30
SNR_STEP=5
NUM_SAMPLES=50
DEVICE="cuda"

# Activate environment
export PATH="/home/ahmed/miniforge3/condabin:$PATH"
source /home/ahmed/miniforge3/etc/profile.d/conda.sh
conda activate rc_flow_env

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  SNR Range: ${SNR_MIN} to ${SNR_MAX} dB (step ${SNR_STEP})"
echo "  Samples per SNR: $NUM_SAMPLES"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "=========================================="
echo ""

# Run evaluation
python3 evaluate_snr_sweep.py \
    --diffusion_checkpoint "$CHECKPOINT" \
    --snr_min $SNR_MIN \
    --snr_max $SNR_MAX \
    --snr_step $SNR_STEP \
    --num_samples $NUM_SAMPLES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Plot: $OUTPUT_DIR/snr_sweep.png"
echo "  - Table: $OUTPUT_DIR/snr_sweep_results.txt"
