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
        noise_var: Optional[float] = None,
        num_iterations: int = 2,
        use_dd_before_diffusion: bool = True
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
        
        # NEW APPROACH: Use DD on pilot interpolation to create pseudo-pilots BEFORE diffusion
        if use_dd_before_diffusion and self.diffusion is not None:
            # Estimate noise from data positions (more realistic)
            data_positions = ~pilot_mask
            data_errors = Y_grid[data_positions] - H_pilot_full[data_positions] * X_grid[data_positions]
            noise_var_dd = np.mean(np.abs(data_errors)**2)
            noise_var_dd = max(noise_var_dd, 1e-3)  # Use reasonable floor
            
            dd_result_pre = self.dd_estimator.estimate(
                H_pilot_full,
                Y_grid,
                X_grid,
                pilot_mask,
                noise_var_dd
            )
            
            # Use augmented mask (with pseudo-pilots) as input to diffusion
            H_with_pseudopilots = dd_result_pre['H_dd']
            augmented_mask = dd_result_pre['augmented_pilot_mask']
            
            # CRITICAL: Cap pilot density to avoid overwhelming the diffusion model
            # The model was trained on 25% pilots - too many pilots degrades performance
            current_density = np.sum(augmented_mask) / augmented_mask.size
            max_pilot_density = 0.60  # Cap at 60% pilot density
            
            avg_accept = np.mean(dd_result_pre['acceptance_rates'])
            
            if current_density > max_pilot_density:
                # Too many pilots - randomly sample to reduce density
                num_original_pilots = np.sum(pilot_mask)
                target_total_pilots = int(max_pilot_density * augmented_mask.size)
                max_pseudo_pilots = target_total_pilots - num_original_pilots
                
                # Get DD positions
                dd_positions = dd_result_pre['dd_mask']
                num_dd_pilots = np.sum(dd_positions)
                
                if num_dd_pilots > max_pseudo_pilots:
                    # Need to subsample - keep the most reliable ones
                    # Flatten and get indices
                    dd_flat = dd_positions.flatten()
                    dd_indices = np.where(dd_flat)[0]
                    
                    # Randomly select subset (could also use LLR scores for smarter selection)
                    np.random.seed(42)
                    selected = np.random.choice(dd_indices, size=max_pseudo_pilots, replace=False)
                    
                    # Create new mask
                    augmented_mask_reduced = pilot_mask.copy()
                    dd_mask_reduced = np.zeros_like(dd_positions)
                    dd_mask_flat = dd_mask_reduced.flatten()
                    dd_mask_flat[selected] = True
                    dd_mask_reduced = dd_mask_flat.reshape(dd_positions.shape)
                    augmented_mask_reduced[dd_mask_reduced] = True
                    
                    augmented_mask = augmented_mask_reduced
                    current_density = np.sum(augmented_mask) / augmented_mask.size
            
            # Only use augmented pilots if acceptance is reasonable (not too few, not everything)
            if 0.2 < avg_accept < 0.95 and current_density <= max_pilot_density:
                diffusion_result = self.estimate_diffusion_only(Y_grid, H_with_pseudopilots, augmented_mask)
                H_final = diffusion_result['H_estimate']
            else:
                # Fall back to original pilot mask
                diffusion_result = self.estimate_diffusion_only(Y_grid, H_pilot_full, pilot_mask)
                H_final = diffusion_result['H_estimate']
            
            return {
                'H_pilot': H_pilot_full,
                'H_diffusion': H_final,
                'H_final': H_final,
                'dd_mask': dd_result_pre['dd_mask'],
                'augmented_pilot_mask': augmented_mask,
                'X_dd': dd_result_pre['X_dd'],
                'Y_dd': dd_result_pre['Y_dd'],
                'normalizers': dd_result_pre['normalizers'],
                'acceptance_rates': dd_result_pre['acceptance_rates'],
                'all_acceptance_rates': [dd_result_pre['acceptance_rates']],
                'noise_var': dd_result_pre['noise_var'],
                'statistics': dd_result_pre['statistics'],
                'num_iterations': 1,
                'method': 'full_pipeline'
            }
        
        # OLD APPROACH (kept for compatibility)
        if self.diffusion is not None:
            diffusion_result = self.estimate_diffusion_only(Y_grid, H_pilot_full, pilot_mask)
            H_diffusion = diffusion_result['H_estimate']
        else:
            H_diffusion = H_pilot_full
        
        H_current = H_diffusion
        current_pilot_mask = pilot_mask.copy()
        all_dd_masks = []
        all_acceptance_rates = []
        
        for iteration in range(num_iterations):
            # ALWAYS recompute noise variance from current channel estimate at pilot positions
            pilot_positions = np.where(pilot_mask)
            current_noise_errors = Y_grid[pilot_positions] - H_current[pilot_positions] * X_grid[pilot_positions]
            noise_var_for_dd = np.mean(np.abs(current_noise_errors)**2)
            noise_var_for_dd = max(noise_var_for_dd, 1e-6)  # Floor to avoid division by zero
            
            dd_result = self.dd_estimator.estimate(
                H_current,
                Y_grid,
                X_grid,
                pilot_mask,
                noise_var_for_dd
            )
            
            all_dd_masks.append(dd_result['dd_mask'])
            all_acceptance_rates.append(dd_result['acceptance_rates'])
            
            # For the final iteration, use the DD result directly
            # DO NOT feed augmented mask back to diffusion (it makes it worse!)
            if iteration < num_iterations - 1 and self.diffusion is not None:
                # Use DD refined channel but KEEP ORIGINAL pilot mask for diffusion
                H_augmented = dd_result['H_dd']
                diffusion_result = self.estimate_diffusion_only(Y_grid, H_augmented, pilot_mask)
                H_current = diffusion_result['H_estimate']
            else:
                H_current = dd_result['H_dd']
        
        return {
            'H_pilot': H_pilot_full,
            'H_diffusion': H_diffusion,
            'H_final': H_current,
            'dd_mask': dd_result['dd_mask'],
            'augmented_pilot_mask': dd_result['augmented_pilot_mask'],
            'X_dd': dd_result['X_dd'],
            'Y_dd': dd_result['Y_dd'],
            'normalizers': dd_result['normalizers'],
            'acceptance_rates': dd_result['acceptance_rates'],
            'all_acceptance_rates': all_acceptance_rates,
            'noise_var': dd_result['noise_var'],
            'statistics': dd_result['statistics'],
            'num_iterations': num_iterations,
            'method': 'full_pipeline'
        }