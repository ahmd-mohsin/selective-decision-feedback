# transmission/receiver.py

import numpy as np
from typing import Tuple
from .demodulator import Demodulator
from .config import SystemConfig

class Receiver:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.demodulator = Demodulator(config)
        
    def zero_forcing_equalization(
        self, 
        received: np.ndarray, 
        H_est: np.ndarray
    ) -> np.ndarray:
        try:
            H_pinv = np.linalg.pinv(H_est)
            equalized = H_pinv @ received
        except np.linalg.LinAlgError:
            equalized = np.zeros_like(received)
            
        return equalized
    
    def mmse_equalization(
        self, 
        received: np.ndarray, 
        H_est: np.ndarray,
        noise_power: float = 0.01
    ) -> np.ndarray:
        Nr, Nt = H_est.shape
        H_hermitian = H_est.conj().T
        
        try:
            G = H_hermitian @ np.linalg.inv(
                H_est @ H_hermitian + noise_power * np.eye(Nr)
            )
            equalized = G @ received
        except np.linalg.LinAlgError:
            equalized = self.zero_forcing_equalization(received, H_est)
            
        return equalized
    
    def process_received_signal(
        self,
        received: np.ndarray,
        H_est: np.ndarray,
        equalization_type: str = 'zf',
        confidence_threshold: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if equalization_type == 'mmse':
            equalized = self.mmse_equalization(received, H_est)
        else:
            equalized = self.zero_forcing_equalization(received, H_est)
        
        soft_symbols = equalized.flatten()
        hard_symbols, distances = self.demodulator.hard_decision(soft_symbols)
        
        evm = self.demodulator.calculate_evm(soft_symbols, hard_symbols)
        confidence_mask = self.demodulator.calculate_confidence(evm, confidence_threshold)
        
        decoded_bits = self.demodulator.symbols_to_bits(hard_symbols)
        
        return hard_symbols, decoded_bits, evm, confidence_mask