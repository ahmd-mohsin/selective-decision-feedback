#!/usr/bin/env python3
"""
Generate augmented training dataset with variable pilot densities
This enables the diffusion model to learn to exploit pseudo-pilots
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import h5py
from tqdm import tqdm
import argparse

from transmission_system.config import TransmissionConfig
from transmission_system.dataset_generator import generate_single_sample


def augment_pilot_mask(pilot_mask: np.ndarray, augmentation_ratio: float, seed: int = None) -> np.ndarray:
    """
    Augment pilot mask by adding pseudo-pilots
    
    Args:
        pilot_mask: Original pilot mask (num_symbols, num_subcarriers)
        augmentation_ratio: Ratio of data positions to add as pseudo-pilots (0.0 to 0.5)
        seed: Random seed for reproducibility
    
    Returns:
        Augmented pilot mask with additional pseudo-pilots
    """
    if seed is not None:
        np.random.seed(seed)
    
    augmented_mask = pilot_mask.copy()
    
    # For each symbol, randomly select data positions to add as pseudo-pilots
    for sym_idx in range(pilot_mask.shape[0]):
        data_positions = np.where(~pilot_mask[sym_idx])[0]
        
        if len(data_positions) > 0:
            num_to_add = int(len(data_positions) * augmentation_ratio)
            
            if num_to_add > 0:
                # Randomly select positions to augment
                selected = np.random.choice(data_positions, size=num_to_add, replace=False)
                augmented_mask[sym_idx, selected] = True
    
    return augmented_mask


def generate_augmented_dataset(
    config: TransmissionConfig,
    num_samples: int,
    augmentation_strategy: str = 'mixed',
    seed_offset: int = 0
):
    """
    Generate dataset with variable pilot densities
    
    augmentation_strategy:
        'none': No augmentation (baseline)
        'fixed': Fixed augmentation ratio (e.g., 20%)
        'mixed': Mix of 0%, 20%, 30%, 40% augmentation
        'progressive': Gradually increasing augmentation
    """
    
    rng = np.random.default_rng(config.seed + seed_offset)
    
    dataset = {
        'H_true': [],
        'H_pilot_full': [],
        'pilot_mask': [],
        'Y_grid': [],
        'X_grid': [],
        'noise_var': [],
        'snr_db': [],
        'doppler_hz': [],
        'augmentation_ratio': []
    }
    
    for i in tqdm(range(num_samples), desc="Generating augmented dataset"):
        # Generate base sample
        sample = generate_single_sample(config, rng, i)
        
        # Determine augmentation ratio based on strategy
        if augmentation_strategy == 'none':
            aug_ratio = 0.0
        elif augmentation_strategy == 'fixed':
            aug_ratio = 0.25  # Add 25% more pilots
        elif augmentation_strategy == 'mixed':
            # Randomly choose: 50% no augmentation, 50% augmented
            if rng.random() < 0.5:
                aug_ratio = 0.0
            else:
                # Choose augmentation between 15% and 40%
                aug_ratio = rng.uniform(0.15, 0.40)
        elif augmentation_strategy == 'progressive':
            # Gradually increase augmentation through dataset
            aug_ratio = (i / num_samples) * 0.35
        else:
            raise ValueError(f"Unknown augmentation strategy: {augmentation_strategy}")
        
        # Augment pilot mask
        pilot_mask_orig = sample['pilot_mask']
        if aug_ratio > 0.0:
            pilot_mask_aug = augment_pilot_mask(pilot_mask_orig, aug_ratio, seed=i)
            
            # Recompute H_pilot_full with augmented mask
            H_pilot_full_aug = np.zeros_like(sample['H_pilot_full'])
            for sym_idx in range(pilot_mask_aug.shape[0]):
                pilot_indices = np.where(pilot_mask_aug[sym_idx])[0]
                if len(pilot_indices) > 0:
                    # Use true channel at augmented pilot positions
                    H_pilot_full_aug[sym_idx, pilot_indices] = sample['Y_grid'][sym_idx, pilot_indices] / (sample['X_grid'][sym_idx, pilot_indices] + 1e-10)
                
                # Interpolate for non-pilot positions
                data_indices = np.where(~pilot_mask_aug[sym_idx])[0]
                for data_idx in data_indices:
                    left_pilots = pilot_indices[pilot_indices < data_idx]
                    right_pilots = pilot_indices[pilot_indices > data_idx]
                    
                    if len(left_pilots) > 0 and len(right_pilots) > 0:
                        left_idx = left_pilots[-1]
                        right_idx = right_pilots[0]
                        alpha = (data_idx - left_idx) / (right_idx - left_idx)
                        H_pilot_full_aug[sym_idx, data_idx] = (1 - alpha) * H_pilot_full_aug[sym_idx, left_idx] + alpha * H_pilot_full_aug[sym_idx, right_idx]
                    elif len(left_pilots) > 0:
                        H_pilot_full_aug[sym_idx, data_idx] = H_pilot_full_aug[sym_idx, left_pilots[-1]]
                    elif len(right_pilots) > 0:
                        H_pilot_full_aug[sym_idx, data_idx] = H_pilot_full_aug[sym_idx, right_pilots[0]]
        else:
            pilot_mask_aug = pilot_mask_orig
            H_pilot_full_aug = sample['H_pilot_full']
        
        # Store augmented sample
        dataset['H_true'].append(sample['H_true'])
        dataset['H_pilot_full'].append(H_pilot_full_aug)
        dataset['pilot_mask'].append(pilot_mask_aug)
        dataset['Y_grid'].append(sample['Y_grid'])
        dataset['X_grid'].append(sample['X_grid'])
        dataset['noise_var'].append(sample['noise_var'])
        dataset['snr_db'].append(sample['snr_db'])
        dataset['doppler_hz'].append(sample['doppler_hz'])
        dataset['augmentation_ratio'].append(aug_ratio)
    
    # Stack arrays
    for key in dataset:
        if key in ['noise_var', 'snr_db', 'doppler_hz', 'augmentation_ratio']:
            dataset[key] = np.array(dataset[key])
        else:
            dataset[key] = np.stack(dataset[key])
    
    return dataset


def save_augmented_dataset(dataset: dict, filepath: str, config: TransmissionConfig):
    """Save augmented dataset to HDF5"""
    
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        for key, value in dataset.items():
            if np.iscomplexobj(value):
                f.create_dataset(f'{key}/real', data=np.real(value), compression='gzip')
                f.create_dataset(f'{key}/imag', data=np.imag(value), compression='gzip')
            else:
                f.create_dataset(key, data=value, compression='gzip')
        
        # Save config
        config_group = f.create_group('config')
        config_group.attrs['Nfft'] = config.Nfft
        config_group.attrs['Ncp'] = config.Ncp
        config_group.attrs['Nsym'] = config.Nsym
        config_group.attrs['pilot_spacing'] = config.pilot_spacing
        config_group.attrs['modulation_order'] = config.modulation_order
        config_group.attrs['snr_db'] = config.snr_db
        config_group.attrs['doppler_hz'] = config.doppler_hz


def main():
    parser = argparse.ArgumentParser(description='Generate augmented training dataset')
    
    parser.add_argument('--num_train', type=int, default=10000, help='Training samples')
    parser.add_argument('--num_val', type=int, default=1000, help='Validation samples')
    parser.add_argument('--num_test', type=int, default=1000, help='Test samples')
    parser.add_argument('--snr_db', type=float, default=20.0, help='SNR in dB')
    parser.add_argument('--doppler', type=float, default=100.0, help='Doppler in Hz')
    parser.add_argument('--strategy', type=str, default='mixed', 
                       choices=['none', 'fixed', 'mixed', 'progressive'],
                       help='Augmentation strategy')
    parser.add_argument('--output_dir', type=str, default='./datasets/augmented',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = TransmissionConfig(
        Nfft=64,
        Ncp=16,
        Nsym=100,
        pilot_spacing=4,
        modulation_order=16,
        snr_db=args.snr_db,
        doppler_hz=args.doppler,
        time_varying=True,
        seed=42
    )
    
    print("=" * 80)
    print("GENERATING AUGMENTED TRAINING DATASET")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Strategy: {args.strategy}")
    print(f"  SNR: {args.snr_db} dB")
    print(f"  Doppler: {args.doppler} Hz")
    print(f"  Training samples: {args.num_train}")
    print(f"  Validation samples: {args.num_val}")
    print(f"  Test samples: {args.num_test}")
    print(f"  Output: {args.output_dir}")
    print()
    
    # Generate datasets
    print("\nGenerating training set...")
    train_data = generate_augmented_dataset(config, args.num_train, args.strategy, seed_offset=0)
    train_path = f"{args.output_dir}/train_augmented_{args.strategy}_snr{int(args.snr_db)}.h5"
    save_augmented_dataset(train_data, train_path, config)
    print(f"Saved: {train_path}")
    
    print("\nGenerating validation set...")
    val_data = generate_augmented_dataset(config, args.num_val, args.strategy, seed_offset=100000)
    val_path = f"{args.output_dir}/val_augmented_{args.strategy}_snr{int(args.snr_db)}.h5"
    save_augmented_dataset(val_data, val_path, config)
    print(f"Saved: {val_path}")
    
    print("\nGenerating test set (no augmentation)...")
    test_data = generate_augmented_dataset(config, args.num_test, 'none', seed_offset=200000)
    test_path = f"{args.output_dir}/test_snr{int(args.snr_db)}.h5"
    save_augmented_dataset(test_data, test_path, config)
    print(f"Saved: {test_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    for name, data in [('Train', train_data), ('Val', val_data)]:
        aug_ratios = data['augmentation_ratio']
        pilot_densities = [np.sum(data['pilot_mask'][i]) / data['pilot_mask'][i].size 
                          for i in range(len(data['pilot_mask']))]
        
        print(f"\n{name} Set:")
        print(f"  Augmentation ratio: {aug_ratios.min():.2f} - {aug_ratios.max():.2f} (mean: {aug_ratios.mean():.2f})")
        print(f"  Pilot density: {min(pilot_densities)*100:.1f}% - {max(pilot_densities)*100:.1f}% (mean: {np.mean(pilot_densities)*100:.1f}%)")
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nNext step: Retrain diffusion model with:")
    print(f"  python3 diffusion/train.py --train_data {train_path} --val_data {val_path}")


if __name__ == '__main__':
    main()
