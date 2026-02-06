import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from typing import Dict, Optional

from decision_directed.config import DecisionDirectedConfig
from decision_directed.estimator import DecisionDirectedEstimator


class IntegratedEstimator:
    
    def __init__(
        self,
        diffusion_checkpoint: Optional[str] = None,
        dd_config: Optional[DecisionDirectedConfig] = None,
        modulation_order: int = 16,
        device: str = 'cuda'
    ):
        self.device = device
        self.modulation_order = modulation_order
        
        if diffusion_checkpoint is not None:
            from diffusion.inference import DiffusionInference
            self.diffusion = DiffusionInference(diffusion_checkpoint, device=device)
        else:
            self.diffusion = None
        
        if dd_config is None:
            dd_config = DecisionDirectedConfig()
        
        self.dd_estimator = DecisionDirectedEstimator(dd_config, modulation_order)
    
    def estimate_pilot_only(
        self,
        Y_grid: np.ndarray,
        X_grid: np.ndarray,
        pilot_mask: np.ndarray,
        noise_var: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        
        num_symbols, num_subcarriers = Y_grid.shape
        H_pilot = np.zeros_like(Y_grid)
        
        for sym_idx in range(num_symbols):
            pilot_indices = np.where(pilot_mask[sym_idx])[0]
            
            if len(pilot_indices) == 0:
                continue
            
            H_pilot[sym_idx, pilot_indices] = Y_grid[sym_idx, pilot_indices] / (X_grid[sym_idx, pilot_indices] + 1e-10)
            
            data_indices = np.where(~pilot_mask[sym_idx])[0]
            
            for data_idx in data_indices:
                left_pilots = pilot_indices[pilot_indices < data_idx]
                right_pilots = pilot_indices[pilot_indices > data_idx]
                
                if len(left_pilots) > 0 and len(right_pilots) > 0:
                    left_idx = left_pilots[-1]
                    right_idx = right_pilots[0]
                    
                    alpha = (data_idx - left_idx) / (right_idx - left_idx)
                    H_pilot[sym_idx, data_idx] = (1 - alpha) * H_pilot[sym_idx, left_idx] + alpha * H_pilot[sym_idx, right_idx]
                elif len(left_pilots) > 0:
                    H_pilot[sym_idx, data_idx] = H_pilot[sym_idx, left_pilots[-1]]
                elif len(right_pilots) > 0:
                    H_pilot[sym_idx, data_idx] = H_pilot[sym_idx, right_pilots[0]]
        
        return {
            'H_estimate': H_pilot,
            'method': 'pilot_only'
        }
    
    def estimate_diffusion_only(
        self,
        Y_grid: np.ndarray,
        H_pilot_full: np.ndarray,
        pilot_mask: np.ndarray
    ) -> Dict[str, np.ndarray]:
        
        if self.diffusion is None:
            raise ValueError("Diffusion model not loaded")
        
        H_pilot_torch = torch.from_numpy(
            np.stack([np.real(H_pilot_full), np.imag(H_pilot_full)], axis=0)
        ).float().unsqueeze(0).to(self.device)
        
        pilot_mask_torch = torch.from_numpy(pilot_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
        
        Y_torch = torch.from_numpy(
            np.stack([np.real(Y_grid), np.imag(Y_grid)], axis=0)
        ).float().unsqueeze(0).to(self.device)
        
        batch = {
            'H_pilot_full': H_pilot_torch,
            'pilot_mask': pilot_mask_torch,
            'Y_grid': Y_torch
        }
        
        H_recon = self.diffusion.reconstruct_batch(batch)
        
        H_complex = self.diffusion.channels_to_complex(H_recon).cpu().numpy()[0]
        
        return {
            'H_estimate': H_complex,
            'method': 'diffusion_only'
        }
    
    def estimate_dd_only(
        self,
        H_pilot_full: np.ndarray,
        Y_grid: np.ndarray,
        X_grid: np.ndarray,
        pilot_mask: np.ndarray,
        noise_var: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        
        dd_result = self.dd_estimator.estimate(
            H_pilot_full,
            Y_grid,
            X_grid,
            pilot_mask,
            noise_var
        )
        
        return {
            'H_estimate': dd_result['H_dd'],
            'dd_mask': dd_result['dd_mask'],
            'X_dd': dd_result['X_dd'],
            'Y_dd': dd_result['Y_dd'],
            'acceptance_rates': dd_result['acceptance_rates'],
            'statistics': dd_result['statistics'],
            'method': 'dd_only'
        }
    
    def estimate_full_pipeline(
        self,
        Y_grid: np.ndarray,
        X_grid: np.ndarray,
        pilot_mask: np.ndarray,
        H_pilot_full: Optional[np.ndarray] = None,
        noise_var: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        
        if H_pilot_full is None:
            num_symbols, num_subcarriers = Y_grid.shape
            H_pilot_full = np.zeros_like(Y_grid)
            
            for sym_idx in range(num_symbols):
                pilot_indices = np.where(pilot_mask[sym_idx])[0]
                
                if len(pilot_indices) > 0:
                    H_pilot_full[sym_idx, pilot_indices] = Y_grid[sym_idx, pilot_indices] / (X_grid[sym_idx, pilot_indices] + 1e-10)
                    
                    data_indices = np.where(~pilot_mask[sym_idx])[0]
                    
                    for data_idx in data_indices:
                        left_pilots = pilot_indices[pilot_indices < data_idx]
                        right_pilots = pilot_indices[pilot_indices > data_idx]
                        
                        if len(left_pilots) > 0 and len(right_pilots) > 0:
                            left_idx = left_pilots[-1]
                            right_idx = right_pilots[0]
                            
                            alpha = (data_idx - left_idx) / (right_idx - left_idx)
                            H_pilot_full[sym_idx, data_idx] = (1 - alpha) * H_pilot_full[sym_idx, left_idx] + alpha * H_pilot_full[sym_idx, right_idx]
                        elif len(left_pilots) > 0:
                            H_pilot_full[sym_idx, data_idx] = H_pilot_full[sym_idx, left_pilots[-1]]
                        elif len(right_pilots) > 0:
                            H_pilot_full[sym_idx, data_idx] = H_pilot_full[sym_idx, right_pilots[0]]
        
        if self.diffusion is not None:
            diffusion_result = self.estimate_diffusion_only(Y_grid, H_pilot_full, pilot_mask)
            H_diffusion = diffusion_result['H_estimate']
        else:
            H_diffusion = H_pilot_full
        
        dd_result = self.dd_estimator.estimate(
            H_diffusion,
            Y_grid,
            X_grid,
            pilot_mask,
            noise_var
        )
        
        return {
            'H_pilot': H_pilot_full,
            'H_diffusion': H_diffusion,
            'H_final': dd_result['H_dd'],
            'dd_mask': dd_result['dd_mask'],
            'X_dd': dd_result['X_dd'],
            'Y_dd': dd_result['Y_dd'],
            'normalizers': dd_result['normalizers'],
            'acceptance_rates': dd_result['acceptance_rates'],
            'noise_var': dd_result['noise_var'],
            'statistics': dd_result['statistics'],
            'method': 'full_pipeline'
        }