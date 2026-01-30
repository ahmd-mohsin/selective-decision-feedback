# transmission/config.py

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class SystemConfig:
    Nr: int = 16
    Nt: int = 64
    Np: int = 38
    fft_size: int = 64
    carrier_freq: float = 28e9
    subcarrier_spacing: float = 120e3
    num_data_symbols: int = 100
    modulation_order: int = 16
    snr_db: float = 10.0
    channel_model: str = 'CDL-C'
    num_samples: int = 10000
    
    @property
    def pilot_density(self) -> float:
        return self.Np / self.Nt
    
    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.modulation_order))