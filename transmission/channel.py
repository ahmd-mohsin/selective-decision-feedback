# transmission/channel.py

import numpy as np
from scipy.linalg import sqrtm
from .config import SystemConfig

class CDLChannel:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.channel_models = {
            'CDL-A': self._generate_cdl_a,
            'CDL-B': self._generate_cdl_b,
            'CDL-C': self._generate_cdl_c,
            'CDL-D': self._generate_cdl_d
        }
        
    def generate_response(self) -> np.ndarray:
        generator = self.channel_models.get(self.config.channel_model, self._generate_cdl_c)
        H = generator()
        H = H / np.sqrt(np.mean(np.abs(H)**2))
        return H
    
    def _generate_cdl_a(self) -> np.ndarray:
        num_paths = 23
        return self._generate_multipath(num_paths, rician_k=0)
    
    def _generate_cdl_b(self) -> np.ndarray:
        num_paths = 25
        return self._generate_multipath(num_paths, rician_k=0)
    
    def _generate_cdl_c(self) -> np.ndarray:
        num_paths = 24
        return self._generate_multipath(num_paths, rician_k=0)
    
    def _generate_cdl_d(self) -> np.ndarray:
        num_paths = 16
        return self._generate_multipath(num_paths, rician_k=13.3)
    
    def _generate_multipath(self, num_paths: int, rician_k: float = 0) -> np.ndarray:
        H = np.zeros((self.config.Nr, self.config.Nt), dtype=complex)
        
        if rician_k > 0:
            K_linear = 10 ** (rician_k / 10)
            los_component = np.sqrt(K_linear / (K_linear + 1))
            nlos_component = np.sqrt(1 / (K_linear + 1))
            
            los_aod = np.random.uniform(0, 2*np.pi)
            los_aoa = np.random.uniform(0, 2*np.pi)
            
            a_tx = np.exp(1j * np.pi * np.arange(self.config.Nt) * np.sin(los_aod))
            a_rx = np.exp(1j * np.pi * np.arange(self.config.Nr) * np.sin(los_aoa))
            
            H_los = np.outer(a_rx, a_tx.conj())
            H += los_component * H_los
            
            for _ in range(num_paths - 1):
                aod = np.random.uniform(0, 2*np.pi)
                aoa = np.random.uniform(0, 2*np.pi)
                
                a_tx = np.exp(1j * np.pi * np.arange(self.config.Nt) * np.sin(aod))
                a_rx = np.exp(1j * np.pi * np.arange(self.config.Nr) * np.sin(aoa))
                
                gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                H += nlos_component * gain * np.outer(a_rx, a_tx.conj()) / np.sqrt(num_paths - 1)
        else:
            for _ in range(num_paths):
                aod = np.random.uniform(0, 2*np.pi)
                aoa = np.random.uniform(0, 2*np.pi)
                
                a_tx = np.exp(1j * np.pi * np.arange(self.config.Nt) * np.sin(aod))
                a_rx = np.exp(1j * np.pi * np.arange(self.config.Nr) * np.sin(aoa))
                
                gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                H += gain * np.outer(a_rx, a_tx.conj()) / np.sqrt(num_paths)
        
        return H
    
    def apply(self, signal: np.ndarray, H: np.ndarray, add_noise: bool = True) -> np.ndarray:
        received = H @ signal
        
        if add_noise:
            noise_power = self._compute_noise_power(signal, H)
            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*received.shape) + 1j * np.random.randn(*received.shape)
            )
            received = received + noise
            
        return received
    
    def _compute_noise_power(self, signal: np.ndarray, H: np.ndarray) -> float:
        signal_power = np.mean(np.abs(H @ signal) ** 2)
        snr_linear = 10 ** (self.config.snr_db / 10)
        noise_power = signal_power / snr_linear
        return noise_power