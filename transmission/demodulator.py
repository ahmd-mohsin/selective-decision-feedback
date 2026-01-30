# transmission/demodulator.py

import numpy as np
from .constellation import QAMConstellation
from .config import SystemConfig

class Demodulator:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.constellation = QAMConstellation(config.modulation_order)
        self.normalized_const = self.constellation.get_normalized_constellation()
        self.bits_per_symbol = config.bits_per_symbol
        
    def hard_decision(self, soft_symbols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hard_symbols = np.zeros_like(soft_symbols, dtype=complex)
        distances = np.zeros(len(soft_symbols), dtype=float)
        
        for i, symbol in enumerate(soft_symbols):
            dists = np.abs(symbol - self.normalized_const)
            min_idx = np.argmin(dists)
            hard_symbols[i] = self.normalized_const[min_idx]
            distances[i] = dists[min_idx]
            
        return hard_symbols, distances
    
    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        bits = []
        
        for symbol in symbols:
            dists = np.abs(symbol - self.normalized_const)
            min_idx = np.argmin(dists)
            
            bit_string = format(min_idx, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in bit_string])
            
        return np.array(bits)
    
    def calculate_evm(self, soft_symbols: np.ndarray, hard_symbols: np.ndarray) -> np.ndarray:
        error_vector = soft_symbols - hard_symbols
        evm = np.abs(error_vector)
        return evm
    
    def calculate_confidence(self, evm: np.ndarray, threshold: float = None) -> np.ndarray:
        if threshold is None:
            threshold = np.percentile(evm, 25)
        
        confidence_mask = evm < threshold
        return confidence_mask