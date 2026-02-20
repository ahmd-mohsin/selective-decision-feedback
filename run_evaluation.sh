#!/bin/bash
# Run Decision-Directed Feedback Ablation Study with Fixed Augmented Pilot Mask
# This script evaluates the iterative DD + Diffusion pipeline

set -e  # Exit on error

echo "=========================================="
echo "Decision-Directed Feedback Evaluation"
echo "=========================================="
echo ""

# Configuration
DATASET="/home/ahmed/selective-decision-feedback/transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
CHECKPOINT="/home/ahmed/selective-decision-feedback/checkpoints/module2/checkpoint_epoch_100.pt"
OUTPUT_DIR="./results/ablation"
NUM_SAMPLES=100
NUM_ITERATIONS=2
DEVICE="cuda"

# Activate conda environment
export PATH="/home/ahmed/miniforge3/condabin:$PATH"
source /home/ahmed/miniforge3/etc/profile.d/conda.sh
conda activate rc_flow_env

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Num Iterations: $NUM_ITERATIONS"
echo "  Device: $DEVICE"
echo ""
echo "=========================================="
echo ""

# Run the evaluation
python3 decision_directed/evaluate_ablation.py \
    --dataset "$DATASET" \
    --diffusion_checkpoint "$CHECKPOINT" \
    --num_samples $NUM_SAMPLES \
    --num_iterations $NUM_ITERATIONS \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - ablation_study.png (plot)"
echo "  - ablation_results.txt (detailed results)"
