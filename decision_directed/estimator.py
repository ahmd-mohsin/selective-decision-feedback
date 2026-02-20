import numpy as np
from typing import Dict, Tuple, Optional

from decision_directed.config import DecisionDirectedConfig
from decision_directed.llr_computer import LLRComputer
from decision_directed.normalizer_tracker import NormalizerTracker


class DecisionDirectedEstimator:
    
    def __init__(
        self,
        config: DecisionDirectedConfig,
        modulation_order: int = 16
    ):
        self.config = config
        self.llr_computer = LLRComputer(modulation_order)
        self.current_threshold = config.llr_threshold
    
    def estimate(
        self,
        H_initial: np.ndarray,
        Y_grid: np.ndarray,
        X_grid: np.ndarray,
        pilot_mask: np.ndarray,
        noise_var: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        
        num_symbols, num_subcarriers = H_initial.shape
        
        tracker = NormalizerTracker(
            num_symbols=num_symbols,
            num_subcarriers=num_subcarriers,
            step_size=self.config.normalizer_step_size,
            noise_step_size=self.config.noise_step_size,
            enable_noise_tracking=self.config.enable_noise_tracking,
            clip_value=self.config.normalizer_clip_value if self.config.clip_normalizer else None
        )
        
        if self.config.initialize_noise_from_pilots and noise_var is None:
            tracker.initialize_from_pilots(H_initial, pilot_mask, Y_grid, X_grid)
        elif noise_var is not None:
            tracker.noise_var[:] = noise_var
        
        dd_mask = np.zeros((num_symbols, num_subcarriers), dtype=bool)
        X_decisions = np.zeros((num_symbols, num_subcarriers), dtype=complex)
        H_refined = H_initial.copy()
        augmented_pilot_mask = pilot_mask.copy()
        
        for sym_idx in range(num_symbols):
            data_mask = ~pilot_mask[sym_idx]
            
            if not np.any(data_mask):
                continue
            
            H_current = H_refined[sym_idx]
            
            Y_equalized = np.zeros(num_subcarriers, dtype=complex)
            Y_equalized[data_mask] = Y_grid[sym_idx, data_mask] / (H_current[data_mask] + 1e-10)
            
            current_noise_var = tracker.noise_var[sym_idx]
            
            reliable_mask, min_llrs = self.llr_computer.compute_reliability(
                Y_equalized[data_mask],
                current_noise_var,
                self.current_threshold
            )
            
            hard_decisions = self.llr_computer.hard_decision(Y_equalized[data_mask])
            
            data_indices = np.where(data_mask)[0]
            accepted_indices = data_indices[reliable_mask]
            
            dd_mask[sym_idx, accepted_indices] = True
            X_decisions[sym_idx, accepted_indices] = hard_decisions[reliable_mask]
            
            if len(accepted_indices) > 0:
                for idx, decision in zip(accepted_indices, hard_decisions[reliable_mask]):
                    H_refined[sym_idx, idx] = Y_grid[sym_idx, idx] / (decision + 1e-10)
                
                augmented_pilot_mask[sym_idx, accepted_indices] = True
            
            acceptance_rate = np.sum(reliable_mask) / np.sum(data_mask) if np.sum(data_mask) > 0 else 0.0
            tracker.acceptance_history.append(acceptance_rate)
            
            if self.config.adaptive_threshold:
                self._adapt_threshold(acceptance_rate)
        
        return {
            'H_dd': H_refined,
            'dd_mask': dd_mask,
            'augmented_pilot_mask': augmented_pilot_mask,
            'X_dd': X_decisions,
            'Y_dd': Y_grid * dd_mask,
            'noise_var': tracker.noise_var,
            'normalizers': tracker.normalizers,
            'acceptance_rates': np.array(tracker.acceptance_history),
            'statistics': tracker.get_statistics()
        }
    
    def _adapt_threshold(self, acceptance_rate: float):
        if acceptance_rate < self.config.min_acceptance_rate:
            self.current_threshold -= self.config.threshold_adapt_rate
            self.current_threshold = max(self.current_threshold, 1.0)
        elif acceptance_rate > self.config.max_acceptance_rate:
            self.current_threshold += self.config.threshold_adapt_rate
            self.current_threshold = min(self.current_threshold, 10.0)