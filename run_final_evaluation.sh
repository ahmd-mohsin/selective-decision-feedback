#!/bin/bash
# Complete Evaluation with GPU Memory Management

echo "========================================"
echo "Decision-Directed + Diffusion Evaluation"
echo "========================================"
echo ""

# Kill any existing Python processes using GPU
echo "Cleaning GPU memory..."
pkill -9 python 2>/dev/null
sleep 3

# Configuration
DATASET="/home/ahmed/selective-decision-feedback/transmission_system/datasets/diffusion_training/ofdm_Nfft64_Nsym100_M16_SNR20.0dB_Doppler100.0Hz_test.h5"
CHECKPOINT="/home/ahmed/selective-decision-feedback/checkpoints/module2/checkpoint_epoch_100.pt"
OUTPUT_DIR="./results/ablation_fixed"
NUM_SAMPLES=100
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
echo "  Device: $DEVICE"
echo ""
echo "========================================"
echo ""

# Set memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the evaluation
python3 decision_directed/evaluate_ablation.py \
    --dataset "$DATASET" \
    --diffusion_checkpoint "$CHECKPOINT" \
    --num_samples $NUM_SAMPLES \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "Evaluation Complete!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "  - ablation_study.png (plot)"
echo "  - ablation_results.txt (detailed results)"
echo ""
echo "Expected Results:"
echo "  Pilot Only:      ~-16 dB"
echo "  Diffusion Only:  ~-24 dB"
echo "  DD Only:         ~-15 dB"
echo "  Full Pipeline:   ~-25 dB (BEST!)"
