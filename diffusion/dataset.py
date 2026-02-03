import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from torch.utils.data import Dataset
from transmission_system.dataset_generator import load_dataset_hdf5
from typing import Dict, Tuple


class DiffusionDataset(Dataset):
    
    def __init__(
        self,
        hdf5_path: str,
        normalize: bool = False,
        return_metadata: bool = False
    ):
        self.hdf5_path = hdf5_path
        self.normalize = normalize
        self.return_metadata = return_metadata
        
        self.data, self.config = load_dataset_hdf5(hdf5_path)
        
        self.num_samples = self.data['H_true'].shape[0]
        
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        H_abs = np.abs(self.data['H_true'])
        self.H_mean = np.mean(H_abs)
        self.H_std = np.std(H_abs)
    
    def _normalize_channel(self, H: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return H
        H_abs = np.abs(H)
        H_normalized = H / (self.H_std + 1e-8)
        return H_normalized
    
    def _denormalize_channel(self, H: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return H
        return H * self.H_std
    
    def _complex_to_channels(self, x: np.ndarray) -> np.ndarray:
        real = np.real(x)
        imag = np.imag(x)
        return np.stack([real, imag], axis=0)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H_true = self.data['H_true'][idx]
        H_pilot_full = self.data['H_pilot_full'][idx]
        pilot_mask = self.data['pilot_mask'][idx]
        Y_grid = self.data['Y_grid'][idx]
        noise_var = self.data['noise_var'][idx]
        
        if self.normalize:
            H_true = self._normalize_channel(H_true)
            H_pilot_full = self._normalize_channel(H_pilot_full)
            Y_grid = self._normalize_channel(Y_grid)
        
        H_true_channels = self._complex_to_channels(H_true)
        H_pilot_channels = self._complex_to_channels(H_pilot_full)
        Y_channels = self._complex_to_channels(Y_grid)
        pilot_mask_float = pilot_mask.astype(np.float32)
        
        sample = {
            'H_true': torch.from_numpy(H_true_channels).float(),
            'H_pilot_full': torch.from_numpy(H_pilot_channels).float(),
            'pilot_mask': torch.from_numpy(pilot_mask_float).unsqueeze(0).float(),
            'Y_grid': torch.from_numpy(Y_channels).float(),
            'noise_var': torch.tensor(noise_var, dtype=torch.float32)
        }
        
        if self.return_metadata:
            sample['idx'] = idx
            sample['snr_db'] = self.data['snr_db'][idx]
            sample['doppler_hz'] = self.data['doppler_hz'][idx]
        
        return sample
    
    def get_conditioning(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond_list = [batch['H_pilot_full'], batch['pilot_mask']]
        
        if 'Y_grid' in batch:
            cond_list.append(batch['Y_grid'])
        
        return torch.cat(cond_list, dim=1)
    
    def get_normalization_stats(self) -> Tuple[float, float]:
        if self.normalize:
            return self.H_mean, self.H_std
        return 0.0, 1.0