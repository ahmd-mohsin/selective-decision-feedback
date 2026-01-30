# transmission/constellation.py

import numpy as np
from typing import Tuple

class QAMConstellation:
    def __init__(self, order: int):
        self.order = order
        self.M = int(np.sqrt(order))
        self.constellation = self._generate_constellation()
        self.normalization = self._compute_normalization()
        
    def _generate_constellation(self) -> np.ndarray:
        m = np.arange(self.M)
        real_parts = 2 * m - (self.M - 1)
        imag_parts = 2 * m - (self.M - 1)
        real_grid, imag_grid = np.meshgrid(real_parts, imag_parts)
        constellation = real_grid + 1j * imag_grid
        return constellation.flatten()
    
    def _compute_normalization(self) -> float:
        avg_power = np.mean(np.abs(self.constellation) ** 2)
        return np.sqrt(avg_power)
    
    def get_normalized_constellation(self) -> np.ndarray:
        return self.constellation / self.normalization
    
    def get_symbol_from_bits(self, bits: np.ndarray) -> complex:
        bits_per_symbol = int(np.log2(self.order))
        if len(bits) != bits_per_symbol:
            raise ValueError(f"Expected {bits_per_symbol} bits")
        index = int(''.join(bits.astype(str)), 2)
        return self.constellation[index] / self.normalization