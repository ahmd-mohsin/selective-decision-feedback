import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transmission_system.dataset_generator import load_dataset_hdf5
from typing import Dict, Optional, Tuple


class OFDMDiffusionDataset(Dataset):
    
    def __init__(
        self, 
        hdf5_path: str,
        mode: str = 'pilots_only',
        normalize: bool = True,
        return_complex: bool = False
    ):
        self.hdf5_path = hdf5_path
        self.mode = mode
        self.normalize = normalize
        self.return_complex = return_complex
        
        self.data, self.config = load_dataset_hdf5(hdf5_path)
        
        self.num_samples = self.data['H_true'].shape[0]
        
        if normalize:
            self.channel_mean = np.mean(np.abs(self.data['H_true']))
            self.channel_std = np.std(np.abs(self.data['H_true']))
        else:
            self.channel_mean = 0.0
            self.channel_std = 1.0
    
    def __len__(self) -> int:
        return self.num_samples
    
    def complex_to_channels(self, x: np.ndarray) -> np.ndarray:
        real = np.real(x)
        imag = np.imag(x)
        return np.stack([real, imag], axis=0)
    
    def normalize_channel(self, H: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return H
        H_abs = np.abs(H)
        H_normalized = H * (1.0 / (H_abs + 1e-8))
        H_normalized = H_normalized * (H_abs - self.channel_mean) / (self.channel_std + 1e-8)
        return H_normalized
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        H_true = self.data['H_true'][idx]
        H_pilot_full = self.data['H_pilot_full'][idx]
        pilot_mask = self.data['pilot_mask'][idx]
        Y_grid = self.data['Y_grid'][idx]
        noise_var = self.data['noise_var'][idx]
        
        if self.return_complex:
            sample = {
                'H_true': torch.from_numpy(H_true),
                'H_pilot_full': torch.from_numpy(H_pilot_full),
                'pilot_mask': torch.from_numpy(pilot_mask),
                'Y_grid': torch.from_numpy(Y_grid),
                'noise_var': torch.tensor(noise_var, dtype=torch.float32)
            }
        else:
            H_true_normalized = self.normalize_channel(H_true)
            H_pilot_normalized = self.normalize_channel(H_pilot_full)
            
            H_true_channels = self.complex_to_channels(H_true_normalized)
            H_pilot_channels = self.complex_to_channels(H_pilot_normalized)
            Y_channels = self.complex_to_channels(Y_grid)
            
            pilot_mask_float = pilot_mask.astype(np.float32)
            
            sample = {
                'H_true': torch.from_numpy(H_true_channels).float(),
                'H_pilot_full': torch.from_numpy(H_pilot_channels).float(),
                'pilot_mask': torch.from_numpy(pilot_mask_float).unsqueeze(0),
                'Y_grid': torch.from_numpy(Y_channels).float(),
                'noise_var': torch.tensor(noise_var, dtype=torch.float32),
                'idx': torch.tensor(idx, dtype=torch.long)
            }
            
            if 'dd_mask' in self.data:
                dd_mask = self.data['dd_mask'][idx]
                X_dd = self.data['X_dd'][idx] if 'X_dd' in self.data else None
                Y_dd = self.data['Y_dd'][idx] if 'Y_dd' in self.data else None
                
                sample['dd_mask'] = torch.from_numpy(dd_mask.astype(np.float32)).unsqueeze(0)
                if X_dd is not None:
                    sample['X_dd'] = torch.from_numpy(self.complex_to_channels(X_dd)).float()
                if Y_dd is not None:
                    sample['Y_dd'] = torch.from_numpy(self.complex_to_channels(Y_dd)).float()
        
        return sample
    
    def get_channel_stats(self) -> Dict[str, float]:
        return {
            'mean': float(self.channel_mean),
            'std': float(self.channel_std),
            'min': float(np.min(np.abs(self.data['H_true']))),
            'max': float(np.max(np.abs(self.data['H_true'])))
        }


def create_dataloaders(
    train_path: str,
    val_path: str,
    test_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    normalize: bool = True,
    mode: str = 'pilots_only'
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    
    train_dataset = OFDMDiffusionDataset(
        train_path, 
        mode=mode, 
        normalize=normalize
    )
    
    val_dataset = OFDMDiffusionDataset(
        val_path, 
        mode=mode, 
        normalize=normalize
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = None
    if test_path is not None:
        test_dataset = OFDMDiffusionDataset(
            test_path, 
            mode=mode, 
            normalize=normalize
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader


def test_dataloader():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    
    print("Testing OFDMDiffusionDataset...")
    print("=" * 60)
    
    dataset = OFDMDiffusionDataset(args.dataset, normalize=True, return_complex=False)
    
    print(f"\nDataset Info:")
    print(f"  Number of samples: {len(dataset)}")
    print(f"  Channel statistics: {dataset.get_channel_stats()}")
    
    print(f"\nLoading first sample...")
    sample = dataset[0]
    
    print(f"\nSample tensor shapes:")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {tuple(value.shape):30s} dtype={value.dtype}")
    
    print(f"\nTensor value ranges:")
    print(f"  H_true:        [{sample['H_true'].min():.4f}, {sample['H_true'].max():.4f}]")
    print(f"  H_pilot_full:  [{sample['H_pilot_full'].min():.4f}, {sample['H_pilot_full'].max():.4f}]")
    print(f"  pilot_mask:    [{sample['pilot_mask'].min():.4f}, {sample['pilot_mask'].max():.4f}]")
    print(f"  noise_var:     {sample['noise_var'].item():.6f}")
    
    print(f"\nCreating DataLoader...")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"\nLoading first batch...")
    batch = next(iter(loader))
    
    print(f"\nBatch tensor shapes:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key:20s}: {tuple(value.shape)}")
    
    print(f"\nBatch statistics:")
    print(f"  Batch size:       {batch['H_true'].shape[0]}")
    print(f"  H_true channels:  {batch['H_true'].shape[1]} (should be 2: Re, Im)")
    print(f"  Spatial dims:     {batch['H_true'].shape[2:]} (Nsym, Nfft)")
    
    print("\n" + "=" * 60)
    print("DataLoader test passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_dataloader()