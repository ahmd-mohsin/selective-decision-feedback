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
import os

from diffusion.inference import DiffusionInference
from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample
from transmission_system.receiver_frontend import compute_channel_nmse


def generate_snr_sweep_data(
    config: TransmissionConfig,
    snr_range: np.ndarray,
    num_samples_per_snr: int = 100
):
    results = {
        'snr_values': snr_range,
        'pilot_nmse': [],
        'diffusion_nmse': [],
        'pilot_nmse_std': [],
        'diffusion_nmse_std': []
    }
    
    for snr_db in tqdm(snr_range, desc="SNR Sweep"):
        config.snr_db = snr_db
        
        pilot_nmse_samples = []
        diffusion_nmse_samples = []
        
        rng = np.random.default_rng(config.seed)
        
        for i in range(num_samples_per_snr):
            sample = generate_single_sample(config, rng, sample_id=i)
            
            H_true = sample['H_true']
            H_pilot = sample['H_pilot_full']
            
            nmse_pilot = compute_channel_nmse(H_pilot, H_true)
            pilot_nmse_samples.append(nmse_pilot)
        
        results['pilot_nmse'].append(np.mean(pilot_nmse_samples))
        results['pilot_nmse_std'].append(np.std(pilot_nmse_samples))
        results['diffusion_nmse'].append(0)
        results['diffusion_nmse_std'].append(0)
    
    return results


def evaluate_snr_sweep_with_diffusion(
    checkpoint_path: str,
    config: TransmissionConfig,
    snr_range: np.ndarray,
    num_samples_per_snr: int = 100,
    device: str = 'cuda'
):
    inference = DiffusionInference(checkpoint_path, device=device)
    
    results = {
        'snr_values': snr_range,
        'pilot_nmse': [],
        'diffusion_nmse': [],
        'pilot_nmse_std': [],
        'diffusion_nmse_std': []
    }
    
    for snr_db in tqdm(snr_range, desc="SNR Sweep with Diffusion"):
        config.snr_db = snr_db
        
        pilot_nmse_samples = []
        diffusion_nmse_samples = []
        
        rng = np.random.default_rng(config.seed)
        
        for i in range(num_samples_per_snr):
            sample = generate_single_sample(config, rng, sample_id=i)
            
            H_true_np = sample['H_true']
            H_pilot_np = sample['H_pilot_full']
            pilot_mask_np = sample['pilot_mask']
            Y_grid_np = sample['Y_grid']
            
            nmse_pilot = compute_channel_nmse(H_pilot_np, H_true_np)
            pilot_nmse_samples.append(nmse_pilot)
            
            H_true = torch.from_numpy(np.stack([np.real(H_true_np), np.imag(H_true_np)], axis=0)).float().unsqueeze(0)
            H_pilot = torch.from_numpy(np.stack([np.real(H_pilot_np), np.imag(H_pilot_np)], axis=0)).float().unsqueeze(0)
            pilot_mask = torch.from_numpy(pilot_mask_np).float().unsqueeze(0).unsqueeze(0)
            Y_grid = torch.from_numpy(np.stack([np.real(Y_grid_np), np.imag(Y_grid_np)], axis=0)).float().unsqueeze(0)
            
            batch = {
                'H_true': H_true.to(device),
                'H_pilot_full': H_pilot.to(device),
                'pilot_mask': pilot_mask.to(device),
                'Y_grid': Y_grid.to(device)
            }
            
            H_pred = inference.reconstruct_batch(batch)
            
            nmse_diff = inference.compute_nmse(H_pred, H_true.to(device))
            diffusion_nmse_samples.append(nmse_diff)
        
        results['pilot_nmse'].append(np.mean(pilot_nmse_samples))
        results['pilot_nmse_std'].append(np.std(pilot_nmse_samples))
        results['diffusion_nmse'].append(np.mean(diffusion_nmse_samples))
        results['diffusion_nmse_std'].append(np.std(diffusion_nmse_samples))
    
    return results


def plot_snr_vs_nmse(results: dict, output_path: str):
    snr_values = results['snr_values']
    pilot_nmse_db = 10 * np.log10(np.array(results['pilot_nmse']))
    diffusion_nmse_db = 10 * np.log10(np.array(results['diffusion_nmse']))
    
    pilot_std_db = 10 * np.log10(np.array(results['pilot_nmse']) + np.array(results['pilot_nmse_std'])) - pilot_nmse_db
    diffusion_std_db = 10 * np.log10(np.array(results['diffusion_nmse']) + np.array(results['diffusion_nmse_std'])) - diffusion_nmse_db
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(snr_values, pilot_nmse_db, yerr=pilot_std_db, 
                 marker='o', linestyle='--', linewidth=2, capsize=5,
                 label='Pilot-Only LS + Interpolation', color='blue')
    
    if np.any(results['diffusion_nmse']):
        plt.errorbar(snr_values, diffusion_nmse_db, yerr=diffusion_std_db,
                     marker='s', linestyle='-', linewidth=2, capsize=5,
                     label='Diffusion Model', color='red')
    
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title('Channel Estimation NMSE vs SNR', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.figure(figsize=(10, 6))
    improvement_db = pilot_nmse_db - diffusion_nmse_db
    
    if np.any(results['diffusion_nmse']):
        plt.plot(snr_values, improvement_db, marker='D', linestyle='-', 
                linewidth=2, color='green', markersize=8)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('NMSE Improvement (dB)', fontsize=12)
        plt.title('Diffusion Model Improvement over Pilot-Only', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        improvement_path = output_path.replace('.png', '_improvement.png')
        plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
        print(f"Improvement plot saved to: {improvement_path}")
    
    plt.close('all')


def save_results_table(results: dict, output_path: str):
    with open(output_path, 'w') as f:
        f.write("SNR vs NMSE Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'SNR (dB)':<12} {'Pilot NMSE (dB)':<18} {'Diffusion NMSE (dB)':<20} {'Improvement (dB)':<15}\n")
        f.write("-" * 80 + "\n")
        
        for i, snr in enumerate(results['snr_values']):
            pilot_nmse_db = 10 * np.log10(results['pilot_nmse'][i])
            
            if results['diffusion_nmse'][i] > 0:
                diff_nmse_db = 10 * np.log10(results['diffusion_nmse'][i])
                improvement = pilot_nmse_db - diff_nmse_db
                f.write(f"{snr:<12.1f} {pilot_nmse_db:<18.2f} {diff_nmse_db:<20.2f} {improvement:<15.2f}\n")
            else:
                f.write(f"{snr:<12.1f} {pilot_nmse_db:<18.2f} {'N/A':<20} {'N/A':<15}\n")
        
        f.write("\n")
    
    print(f"Results table saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate NMSE vs SNR')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to diffusion model checkpoint (optional, compares with pilot-only if not provided)')
    
    parser.add_argument('--snr_min', type=float, default=0.0,
                       help='Minimum SNR in dB')
    parser.add_argument('--snr_max', type=float, default=30.0,
                       help='Maximum SNR in dB')
    parser.add_argument('--snr_step', type=float, default=5.0,
                       help='SNR step size in dB')
    
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples per SNR point')
    
    parser.add_argument('--Nfft', type=int, default=64)
    parser.add_argument('--Nsym', type=int, default=100)
    parser.add_argument('--pilot_spacing', type=int, default=4)
    parser.add_argument('--doppler_hz', type=float, default=100.0)
    
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='./results/snr_sweep')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 70)
    print("SNR vs NMSE EVALUATION")
    print("=" * 70)
    
    config = TransmissionConfig(
        Nfft=args.Nfft,
        Nsym=args.Nsym,
        pilot_spacing=args.pilot_spacing,
        doppler_hz=args.doppler_hz,
        modulation_order=16,
        time_varying=True,
        seed=args.seed
    )
    
    snr_range = np.arange(args.snr_min, args.snr_max + args.snr_step, args.snr_step)
    
    print(f"\nConfiguration:")
    print(f"  SNR Range: {args.snr_min} to {args.snr_max} dB (step: {args.snr_step} dB)")
    print(f"  Number of SNR points: {len(snr_range)}")
    print(f"  Samples per SNR: {args.num_samples}")
    print(f"  FFT Size: {args.Nfft}")
    print(f"  OFDM Symbols: {args.Nsym}")
    print(f"  Pilot Spacing: {args.pilot_spacing}")
    print(f"  Doppler: {args.doppler_hz} Hz")
    
    if args.checkpoint is not None:
        print(f"  Diffusion Model: {args.checkpoint}")
        print("\n" + "=" * 70)
        print("EVALUATING WITH DIFFUSION MODEL")
        print("=" * 70 + "\n")
        
        results = evaluate_snr_sweep_with_diffusion(
            args.checkpoint,
            config,
            snr_range,
            args.num_samples,
            args.device
        )
    else:
        print("\n" + "=" * 70)
        print("EVALUATING PILOT-ONLY (NO DIFFUSION)")
        print("=" * 70 + "\n")
        
        results = generate_snr_sweep_data(
            config,
            snr_range,
            args.num_samples
        )
    
    print("\n" + "=" * 70)
    print("GENERATING PLOTS AND TABLES")
    print("=" * 70 + "\n")
    
    plot_path = os.path.join(args.output_dir, 'snr_vs_nmse.png')
    plot_snr_vs_nmse(results, plot_path)
    
    table_path = os.path.join(args.output_dir, 'snr_vs_nmse_results.txt')
    save_results_table(results, table_path)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70 + "\n")
    
    print(f"{'SNR (dB)':<12} {'Pilot NMSE (dB)':<18} {'Diffusion NMSE (dB)':<20} {'Improvement (dB)':<15}")
    print("-" * 70)
    
    for i, snr in enumerate(snr_range):
        pilot_nmse_db = 10 * np.log10(results['pilot_nmse'][i])
        
        if results['diffusion_nmse'][i] > 0:
            diff_nmse_db = 10 * np.log10(results['diffusion_nmse'][i])
            improvement = pilot_nmse_db - diff_nmse_db
            print(f"{snr:<12.1f} {pilot_nmse_db:<18.2f} {diff_nmse_db:<20.2f} {improvement:<15.2f}")
        else:
            print(f"{snr:<12.1f} {pilot_nmse_db:<18.2f} {'N/A':<20} {'N/A':<15}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()