import numpy as np
from typing import Tuple, Optional


class LLRComputer:
    
    def __init__(self, modulation_order: int = 16):
        self.modulation_order = modulation_order
        self.bits_per_symbol = int(np.log2(modulation_order))
        self.constellation = self._generate_qam_constellation()
        self.bit_mapping = self._generate_gray_mapping()
    
    def _generate_qam_constellation(self) -> np.ndarray:
        if self.modulation_order == 4:
            points = np.array([-1-1j, -1+1j, 1-1j, 1+1j])
        elif self.modulation_order == 16:
            real_vals = np.array([-3, -1, 1, 3])
            imag_vals = np.array([-3, -1, 1, 3])
            points = np.array([r + 1j*i for i in imag_vals for r in real_vals])
        elif self.modulation_order == 64:
            real_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            imag_vals = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
            points = np.array([r + 1j*i for i in imag_vals for r in real_vals])
        else:
            raise ValueError(f"Unsupported modulation order: {self.modulation_order}")
        
        energy = np.mean(np.abs(points)**2)
        return points / np.sqrt(energy)
    
    def _generate_gray_mapping(self) -> np.ndarray:
        bit_mapping = np.zeros((self.modulation_order, self.bits_per_symbol), dtype=int)
        
        if self.modulation_order == 16:
            gray_map = [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10]
            for idx, gray_idx in enumerate(gray_map):
                for bit_pos in range(self.bits_per_symbol):
                    bit_mapping[idx, bit_pos] = (gray_idx >> (self.bits_per_symbol - 1 - bit_pos)) & 1
        else:
            for idx in range(self.modulation_order):
                for bit_pos in range(self.bits_per_symbol):
                    bit_mapping[idx, bit_pos] = (idx >> (self.bits_per_symbol - 1 - bit_pos)) & 1
        
        return bit_mapping
    
    def compute_llrs(
        self,
        y: np.ndarray,
        noise_var: float
    ) -> np.ndarray:
        y_flat = y.flatten()
        num_symbols = len(y_flat)
        llrs = np.zeros((num_symbols, self.bits_per_symbol))
        
        for sym_idx in range(num_symbols):
            y_val = y_flat[sym_idx]
            
            for bit_pos in range(self.bits_per_symbol):
                dist_0 = np.min(np.abs(y_val - self.constellation[self.bit_mapping[:, bit_pos] == 0])**2)
                dist_1 = np.min(np.abs(y_val - self.constellation[self.bit_mapping[:, bit_pos] == 1])**2)
                
                llrs[sym_idx, bit_pos] = (dist_1 - dist_0) / noise_var
        
        return llrs.reshape(y.shape + (self.bits_per_symbol,))
    
    def compute_min_llr(
        self,
        y: np.ndarray,
        noise_var: float
    ) -> np.ndarray:
        llrs = self.compute_llrs(y, noise_var)
        return np.min(np.abs(llrs), axis=-1)
    
    def hard_decision(
        self,
        y: np.ndarray
    ) -> np.ndarray:
        y_flat = y.flatten()
        decisions = np.zeros_like(y_flat)
        
        for idx, y_val in enumerate(y_flat):
            distances = np.abs(y_val - self.constellation)
            decisions[idx] = self.constellation[np.argmin(distances)]
        
        return decisions.reshape(y.shape)
    
    def compute_reliability(
        self,
        y: np.ndarray,
        noise_var: float,
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        min_llrs = self.compute_min_llr(y, noise_var)
        reliable_mask = min_llrs > threshold
        return reliable_mask, min_llrs