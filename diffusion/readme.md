# Module 2 Quick Start Guide

## Step 1: Generate Dataset (if not done)

```bash
cd ~/selective-decision-feedback

python transmission_system/scripts/make_dataset.py \
    --Nfft 64 \
    --Nsym 100 \
    --pilot_spacing 4 \
    --modulation_order 16 \
    --snr_db 20.0 \
    --doppler_hz 100.0 \
    --num_train 50000 \
    --num_val 5000 \
    --num_test 5000 \
    --time_varying \
    --dataset_path ./datasets/module2
```

## Step 2: Train Diffusion Model

```bash
python diffusion/train.py \
    --train_data datasets/module2/ofdm_*_train.h5 \
    --val_data datasets/module2/ofdm_*_val.h5 \
    --diffusion_steps 1000 \
    --beta_schedule cosine \
    --model_channels 128 \
    --num_res_blocks 2 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --device cuda \
    --checkpoint_dir ./checkpoints/module2 \
    --log_dir ./logs/module2
```

## Step 3: Monitor Training

```bash
tensorboard --logdir ./logs/module2
```

Open browser to: http://localhost:6006

## Step 4: Evaluate Model

```bash
python diffusion/evaluate.py \
    --checkpoint checkpoints/module2/checkpoint_epoch_100.pt \
    --test_data datasets/module2/ofdm_*_test.h5 \
    --save_plots \
    --output_dir ./results/module2
```

## Step 5: Use for Inference

```python
from diffusion.inference import DiffusionInference
from diffusion.dataset import DiffusionDataset
import torch

inference = DiffusionInference(
    'checkpoints/module2/checkpoint_epoch_100.pt',
    device='cuda'
)

dataset = DiffusionDataset('datasets/module2/ofdm_*_test.h5')

sample = dataset[0]
for key in sample:
    sample[key] = sample[key].unsqueeze(0).cuda()

H_reconstructed = inference.reconstruct_batch(sample)

H_true = sample['H_true']
nmse = inference.compute_nmse(H_reconstructed, H_true)
nmse_db = 10 * torch.log10(torch.tensor(nmse))

print(f"NMSE: {nmse_db:.2f} dB")

H_complex = inference.channels_to_complex(H_reconstructed)
```

## Expected Training Time

- GPU (RTX 3090): ~30 minutes per epoch with 50k samples
- Total training (100 epochs): ~50 hours

## Expected Performance

| Method | NMSE (dB) | Improvement |
|--------|-----------|-------------|
| Pilot-Only | -16 dB | Baseline |
| Diffusion | -22 dB | +6 dB |
| Diffusion + DD | -26 dB | +10 dB (with Module 3) |

## Tips

1. **Start small**: Test with 1000 samples first
2. **Use cosine schedule**: Better than linear
3. **Monitor validation**: Stop if val_loss stops decreasing
4. **Save best model**: Keep checkpoint with lowest val_nmse
5. **Use EMA model**: For inference, always use the EMA weights