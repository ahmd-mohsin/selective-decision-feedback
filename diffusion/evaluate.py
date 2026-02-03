import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from diffusion.inference import DiffusionInference
from diffusion.dataset import DiffusionDataset


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained diffusion model')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test HDF5 dataset')
    
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to evaluate (default: 100)')
    
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--save_plots', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DIFFUSION MODEL EVALUATION")
    print("=" * 70)
    
    print("\nLoading model...")
    inference = DiffusionInference(args.checkpoint, device=args.device)
    
    print("\nLoading test dataset...")
    test_dataset = DiffusionDataset(args.test_data, normalize=True)
    
    num_samples = args.num_samples if args.num_samples is not None else len(test_dataset)
    print(f"  Evaluating on {num_samples} samples (use --num_samples to change)")
    
    print("\n" + "=" * 70)
    print("COMPUTING METRICS")
    print("=" * 70 + "\n")
    
    print("Running diffusion sampling (this may take a few minutes)...")
    metrics = inference.evaluate_dataset(test_dataset, num_samples)
    
    print("\nResults:")
    print(f"  Pilot-Only NMSE:      {metrics['pilot_nmse_db']:.2f} dB")
    print(f"  Diffusion NMSE:       {metrics['diffusion_nmse_db']:.2f} dB")
    print(f"  Improvement:          {metrics['improvement_db']:.2f} dB")
    
    if args.save_plots:
        print("\n" + "=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70 + "\n")
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        sample_idx = 0
        batch = test_dataset[sample_idx]
        
        for key in batch:
            batch[key] = batch[key].unsqueeze(0)
        
        H_true = batch['H_true'].to(args.device)
        H_pilot = batch['H_pilot_full'].to(args.device)
        H_pred = inference.reconstruct_batch(batch)
        
        H_true_complex = inference.channels_to_complex(H_true).cpu().numpy()[0]
        H_pilot_complex = inference.channels_to_complex(H_pilot).cpu().numpy()[0]
        H_pred_complex = inference.channels_to_complex(H_pred).cpu().numpy()[0]
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        
        im0 = axes[0, 0].imshow(np.abs(H_true_complex), aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Ground Truth |H|')
        axes[0, 0].set_xlabel('Subcarrier')
        axes[0, 0].set_ylabel('OFDM Symbol')
        plt.colorbar(im0, ax=axes[0, 0])
        
        im1 = axes[0, 1].imshow(np.angle(H_true_complex), aspect='auto', cmap='twilight')
        axes[0, 1].set_title('Ground Truth ∠H')
        axes[0, 1].set_xlabel('Subcarrier')
        axes[0, 1].set_ylabel('OFDM Symbol')
        plt.colorbar(im1, ax=axes[0, 1])
        
        im2 = axes[1, 0].imshow(np.abs(H_pilot_complex), aspect='auto', cmap='viridis')
        axes[1, 0].set_title('Pilot-Only Estimate |H|')
        axes[1, 0].set_xlabel('Subcarrier')
        axes[1, 0].set_ylabel('OFDM Symbol')
        plt.colorbar(im2, ax=axes[1, 0])
        
        im3 = axes[1, 1].imshow(np.abs(H_pred_complex), aspect='auto', cmap='viridis')
        axes[1, 1].set_title('Diffusion Estimate |H|')
        axes[1, 1].set_xlabel('Subcarrier')
        axes[1, 1].set_ylabel('OFDM Symbol')
        plt.colorbar(im3, ax=axes[1, 1])
        
        error_pilot = np.abs(H_pilot_complex - H_true_complex)
        error_diff = np.abs(H_pred_complex - H_true_complex)
        
        im4 = axes[2, 0].imshow(error_pilot, aspect='auto', cmap='hot')
        axes[2, 0].set_title('Pilot-Only Error')
        axes[2, 0].set_xlabel('Subcarrier')
        axes[2, 0].set_ylabel('OFDM Symbol')
        plt.colorbar(im4, ax=axes[2, 0])
        
        im5 = axes[2, 1].imshow(error_diff, aspect='auto', cmap='hot')
        axes[2, 1].set_title('Diffusion Error')
        axes[2, 1].set_xlabel('Subcarrier')
        axes[2, 1].set_ylabel('OFDM Symbol')
        plt.colorbar(im5, ax=axes[2, 1])
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/channel_comparison.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {args.output_dir}/channel_comparison.png")
        
        fig2, axes2 = plt.subplots(2, 1, figsize=(10, 8))
        
        symbol_idx = H_true_complex.shape[0] // 2
        
        axes2[0].plot(np.abs(H_true_complex[symbol_idx, :]), 'k-', label='Ground Truth', linewidth=2)
        axes2[0].plot(np.abs(H_pilot_complex[symbol_idx, :]), 'b--', label='Pilot-Only', linewidth=1.5)
        axes2[0].plot(np.abs(H_pred_complex[symbol_idx, :]), 'r-', label='Diffusion', linewidth=1.5)
        axes2[0].set_xlabel('Subcarrier Index')
        axes2[0].set_ylabel('|H|')
        axes2[0].set_title(f'Channel Magnitude (Symbol {symbol_idx})')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)
        
        axes2[1].plot(np.angle(H_true_complex[symbol_idx, :]), 'k-', label='Ground Truth', linewidth=2)
        axes2[1].plot(np.angle(H_pilot_complex[symbol_idx, :]), 'b--', label='Pilot-Only', linewidth=1.5)
        axes2[1].plot(np.angle(H_pred_complex[symbol_idx, :]), 'r-', label='Diffusion', linewidth=1.5)
        axes2[1].set_xlabel('Subcarrier Index')
        axes2[1].set_ylabel('∠H (radians)')
        axes2[1].set_title(f'Channel Phase (Symbol {symbol_idx})')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{args.output_dir}/channel_profile.png", dpi=150, bbox_inches='tight')
        print(f"  Saved: {args.output_dir}/channel_profile.png")
        
        plt.close('all')
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()