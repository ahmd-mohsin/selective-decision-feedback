import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import yaml
import torch

from diffusion.config import DiffusionConfig
from diffusion.dataset import DiffusionDataset
from diffusion.trainer import DiffusionTrainer


def main():
    parser = argparse.ArgumentParser(description='Train diffusion model for channel estimation')
    
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training HDF5 dataset')
    parser.add_argument('--val_data', type=str, default=None,
                       help='Path to validation HDF5 dataset')
    
    parser.add_argument('--diffusion_steps', type=int, default=1000)
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                       choices=['linear', 'cosine', 'quadratic'])
    
    parser.add_argument('--model_channels', type=int, default=128)
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = DiffusionConfig(**config_dict)
    else:
        config = DiffusionConfig(
            diffusion_steps=args.diffusion_steps,
            beta_schedule=args.beta_schedule,
            model_channels=args.model_channels,
            num_res_blocks=args.num_res_blocks,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            grad_clip=args.grad_clip,
            num_workers=args.num_workers,
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=args.log_dir
        )
    
    print("=" * 70)
    print("DIFFUSION MODEL TRAINING - MODULE 2")
    print("=" * 70)
    
    print("\nConfiguration:")
    print(f"  Diffusion Steps:      {config.diffusion_steps}")
    print(f"  Beta Schedule:        {config.beta_schedule}")
    print(f"  Model Channels:       {config.model_channels}")
    print(f"  Num Res Blocks:       {config.num_res_blocks}")
    print(f"  Learning Rate:        {config.learning_rate}")
    print(f"  Batch Size:           {config.batch_size}")
    print(f"  Num Epochs:           {config.num_epochs}")
    print(f"  Device:               {config.device}")
    
    print("\nLoading datasets...")
    train_dataset = DiffusionDataset(args.train_data, normalize=True)
    print(f"  Training samples: {len(train_dataset)}")
    
    val_dataset = None
    if args.val_data is not None:
        val_dataset = DiffusionDataset(args.val_data, normalize=True)
        print(f"  Validation samples: {len(val_dataset)}")
    
    print("\nInitializing trainer...")
    trainer = DiffusionTrainer(config, train_dataset, val_dataset)
    
    if args.resume is not None:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70 + "\n")
    
    trainer.train()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()