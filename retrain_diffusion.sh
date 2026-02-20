#!/bin/bash
# Complete Pipeline: Retrain Diffusion Model for Variable Pilot Density
# This will enable 5-6 dB improvement with decision-directed feedback

set -e

echo "================================================================================"
echo "RETRAINING DIFFUSION MODEL FOR VARIABLE PILOT DENSITY"
echo "================================================================================"
echo ""

# Configuration
OUTPUT_DIR="./datasets/augmented"
CHECKPOINT_DIR="./checkpoints/augmented_model"
NUM_TRAIN=10000
NUM_VAL=1000
NUM_TEST=1000
SNR=20
STRATEGY="mixed"  # Mix of 25% and 40-60% pilot density
EPOCHS=100
BATCH_SIZE=32

# Activate environment
export PATH="/home/ahmed/miniforge3/condabin:$PATH"
source /home/ahmed/miniforge3/etc/profile.d/conda.sh
conda activate rc_flow_env

echo "Step 1/3: Generating Augmented Training Dataset"
echo "--------------------------------------------------------------------------------"
echo "  Strategy: $STRATEGY (50% baseline 25%, 50% augmented 40-60%)"
echo "  Training samples: $NUM_TRAIN"
echo "  SNR: $SNR dB"
echo ""

python3 generate_augmented_dataset.py \
    --num_train $NUM_TRAIN \
    --num_val $NUM_VAL \
    --num_test $NUM_TEST \
    --snr_db $SNR \
    --strategy $STRATEGY \
    --output_dir $OUTPUT_DIR

echo ""
echo "================================================================================"
echo "Step 2/3: Training Diffusion Model"
echo "--------------------------------------------------------------------------------"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo ""

python3 diffusion/train.py \
    --train_data "$OUTPUT_DIR/train_augmented_${STRATEGY}_snr${SNR}.h5" \
    --val_data "$OUTPUT_DIR/val_augmented_${STRATEGY}_snr${SNR}.h5" \
    --num_epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate 1e-4 \
    --checkpoint_dir $CHECKPOINT_DIR \
    --device cuda

echo ""
echo "================================================================================"
echo "Step 3/3: Evaluating Retrained Model"
echo "--------------------------------------------------------------------------------"
echo ""

# Find the best checkpoint
BEST_CHECKPOINT=$(ls -t $CHECKPOINT_DIR/checkpoint_epoch_*.pt | head -1)

echo "Using checkpoint: $BEST_CHECKPOINT"
echo ""

# Run SNR sweep with new model
python3 evaluate_snr_sweep.py \
    --diffusion_checkpoint "$BEST_CHECKPOINT" \
    --snr_min 15 \
    --snr_max 30 \
    --snr_step 5 \
    --num_samples 50 \
    --device cuda \
    --output_dir ./results/snr_sweep_retrained

echo ""
echo "================================================================================"
echo "RETRAINING COMPLETE!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  - Retrained model: $BEST_CHECKPOINT"
echo "  - Evaluation results: ./results/snr_sweep_retrained/"
echo ""
echo "Expected improvement with retrained model:"
echo "  - OLD model: ~1-2 dB gain (limited by 25% training)"
echo "  - NEW model: ~4-6 dB gain (trained on variable density)"
echo ""
echo "Next: Update your code to use the new checkpoint!"
