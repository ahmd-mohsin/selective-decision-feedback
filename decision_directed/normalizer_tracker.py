import numpy as np
from typing import Dict, Tuple, Optional


class NormalizerTracker:
    
    def __init__(
        self,
        num_symbols: int,
        num_subcarriers: int,
        step_size: float = 0.01,
        noise_step_size: float = 0.05,
        enable_noise_tracking: bool = True,
        clip_value: Optional[float] = 3.0
    ):
        self.num_symbols = num_symbols
        self.num_subcarriers = num_subcarriers
        self.step_size = step_size
        self.noise_step_size = noise_step_size
        self.enable_noise_tracking = enable_noise_tracking
        self.clip_value = clip_value
        
        self.normalizers = np.ones((num_symbols, num_subcarriers), dtype=complex)
        self.noise_var = np.ones(num_symbols) * 0.01
        
        self.acceptance_history = []
        self.error_history = []
    
    def initialize_from_pilots(
        self,
        H_ref: np.ndarray,
        pilot_mask: np.ndarray,
        Y_grid: np.ndarray,
        X_grid: np.ndarray
    ):
        pilot_positions = np.where(pilot_mask)
        
        if len(pilot_positions[0]) > 0:
            pilot_errors = Y_grid[pilot_positions] - H_ref[pilot_positions] * X_grid[pilot_positions]
            initial_noise_var = np.mean(np.abs(pilot_errors)**2)
            self.noise_var[:] = initial_noise_var
    
    def update_normalizer(
        self,
        symbol_idx: int,
        subcarrier_idx: int,
        error: complex,
        decision: complex,
        H_ref: complex
    ):
        if np.abs(H_ref) < 1e-10:
            return
        
        gradient = np.conj(error) * decision / (np.abs(H_ref)**2 + 1e-10)
        self.normalizers[symbol_idx, subcarrier_idx] += self.step_size * gradient
        
        if self.clip_value is not None:
            norm_mag = np.abs(self.normalizers[symbol_idx, subcarrier_idx])
            if norm_mag > self.clip_value:
                self.normalizers[symbol_idx, subcarrier_idx] *= self.clip_value / norm_mag
    
    def update_noise_variance(
        self,
        symbol_idx: int,
        errors: np.ndarray
    ):
        if not self.enable_noise_tracking or len(errors) == 0:
            return
        
        measured_var = np.mean(np.abs(errors)**2)
        
        if symbol_idx > 0:
            self.noise_var[symbol_idx] = (
                (1 - self.noise_step_size) * self.noise_var[symbol_idx - 1] +
                self.noise_step_size * measured_var
            )
        else:
            self.noise_var[symbol_idx] = measured_var
    
    def get_refined_channel(
        self,
        H_ref: np.ndarray
    ) -> np.ndarray:
        return self.normalizers * H_ref
    
    def get_statistics(self) -> Dict[str, float]:
        return {
            'mean_normalizer_mag': float(np.mean(np.abs(self.normalizers))),
            'std_normalizer_mag': float(np.std(np.abs(self.normalizers))),
            'mean_noise_var': float(np.mean(self.noise_var)),
            'mean_acceptance_rate': float(np.mean(self.acceptance_history)) if self.acceptance_history else 0.0
        }