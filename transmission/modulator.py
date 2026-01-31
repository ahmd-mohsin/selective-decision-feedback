# transmission/modulator.py

import numpy as np
from typing import Tuple
from .constellation import QAMConstellation
from .config import SystemConfig


class Modulator:
    """
    OFDM Transmitter Module.

    Handles bit generation, QAM modulation, pilot insertion, and OFDM grid creation.
    This is the Tx side of the physical layer for massive MIMO-OFDM systems.
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize modulator with system configuration.

        Args:
            config: SystemConfig with Nr, Nt, Np, modulation_order, etc.
        """
        self.config = config
        self.constellation = QAMConstellation(config.modulation_order)
        self.bits_per_symbol = config.bits_per_symbol

    def generate_random_bits(self, num_symbols: int) -> np.ndarray:
        """
        Generate random binary data for transmission.

        Args:
            num_symbols: Number of QAM symbols to generate bits for

        Returns:
            Array of random bits (length = num_symbols * bits_per_symbol)
        """
        total_bits = num_symbols * self.bits_per_symbol
        return np.random.randint(0, 2, size=total_bits)

    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Map bit sequence to QAM symbols.

        Args:
            bits: Binary array (must be divisible by bits_per_symbol)

        Returns:
            Array of complex QAM symbols
        """
        num_symbols = len(bits) // self.bits_per_symbol
        symbols = np.zeros(num_symbols, dtype=complex)

        for i in range(num_symbols):
            start_idx = i * self.bits_per_symbol
            end_idx = start_idx + self.bits_per_symbol
            bit_group = bits[start_idx:end_idx]
            symbols[i] = self.constellation.get_symbol_from_bits(bit_group)

        return symbols

    def generate_pilots(self, num_pilots: int) -> np.ndarray:
        """
        Generate QPSK pilot symbols.

        Pilots use QPSK (4 phases) for robustness. They are known at both
        Tx and Rx, enabling channel estimation.

        Args:
            num_pilots: Number of pilot symbols

        Returns:
            Array of unit-magnitude QPSK pilot symbols
        """
        phases = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], size=num_pilots)
        pilots = np.exp(1j * phases)
        return pilots

    def create_ofdm_grid(
        self, data_symbols: np.ndarray, pilots: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create OFDM resource grid with interleaved pilots and data.

        The grid has dimensions (Nt, num_data_symbols) where:
        - Np subcarriers per OFDM symbol carry pilots
        - (Nt - Np) subcarriers carry data

        Args:
            data_symbols: Modulated data symbols
            pilots: QPSK pilot symbols

        Returns:
            Tuple of:
                - ofdm_grid: Complex grid (Nt x num_data_symbols)
                - pilot_mask: Boolean mask indicating pilot positions
        """
        grid_size = (self.config.Nt, self.config.num_data_symbols)
        ofdm_grid = np.zeros(grid_size, dtype=complex)
        pilot_mask = np.zeros(grid_size, dtype=bool)

        # Fixed pilot positions across all OFDM symbols (comb-type pilots)
        pilot_indices = np.random.choice(
            self.config.Nt, size=len(pilots), replace=False
        )

        data_idx = 0
        for time_idx in range(self.config.num_data_symbols):
            # Insert pilots at fixed subcarrier positions
            ofdm_grid[pilot_indices, time_idx] = pilots
            pilot_mask[pilot_indices, time_idx] = True

            # Insert data at remaining positions
            data_indices = np.setdiff1d(np.arange(self.config.Nt), pilot_indices)
            num_data_per_symbol = len(data_indices)

            if data_idx + num_data_per_symbol <= len(data_symbols):
                ofdm_grid[data_indices, time_idx] = data_symbols[
                    data_idx : data_idx + num_data_per_symbol
                ]
                data_idx += num_data_per_symbol

        return ofdm_grid, pilot_mask

    def get_pilot_indices(self) -> np.ndarray:
        """
        Get the subcarrier indices used for pilots.

        Returns:
            Array of pilot subcarrier indices
        """
        return np.random.choice(self.config.Nt, size=self.config.Np, replace=False)
