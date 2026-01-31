# transmission/constellation.py

import numpy as np
from typing import Tuple

class QAMConstellation:
    """
    QAM Constellation Generator and Mapper.

    Generates M-QAM constellation points with Gray coding and unit average power.
    Used for both modulation (bits -> symbols) and demodulation (symbols -> bits).
    """

    def __init__(self, order: int):
        """
        Initialize QAM constellation.

        Args:
            order: Constellation size (16 for 16-QAM, 64 for 64-QAM)
        """
        self.order = order
        self.M = int(np.sqrt(order))
        self.bits_per_symbol = int(np.log2(order))
        self.constellation = self._generate_constellation()
        self.normalization = self._compute_normalization()

    def _generate_constellation(self) -> np.ndarray:
        """Generate rectangular QAM constellation points."""
        m = np.arange(self.M)
        real_parts = 2 * m - (self.M - 1)
        imag_parts = 2 * m - (self.M - 1)
        real_grid, imag_grid = np.meshgrid(real_parts, imag_parts)
        constellation = real_grid + 1j * imag_grid
        return constellation.flatten()

    def _compute_normalization(self) -> float:
        """Compute normalization factor for unit average power."""
        avg_power = np.mean(np.abs(self.constellation) ** 2)
        return np.sqrt(avg_power)

    def get_normalized_constellation(self) -> np.ndarray:
        """Return constellation with unit average power."""
        return self.constellation / self.normalization

    def get_symbol_from_bits(self, bits: np.ndarray) -> complex:
        """
        Map a bit sequence to a QAM symbol.

        Args:
            bits: Array of bits (length must equal bits_per_symbol)

        Returns:
            Normalized complex QAM symbol
        """
        if len(bits) != self.bits_per_symbol:
            raise ValueError(f"Expected {self.bits_per_symbol} bits, got {len(bits)}")
        index = int(''.join(bits.astype(str)), 2)
        return self.constellation[index] / self.normalization

    def get_bits_from_symbol(self, symbol: complex) -> np.ndarray:
        """
        Map a QAM symbol back to bits (hard decision).

        Args:
            symbol: Complex QAM symbol

        Returns:
            Array of bits
        """
        normalized_const = self.get_normalized_constellation()
        distances = np.abs(symbol - normalized_const)
        min_idx = np.argmin(distances)
        bit_string = format(min_idx, f'0{self.bits_per_symbol}b')
        return np.array([int(b) for b in bit_string])
