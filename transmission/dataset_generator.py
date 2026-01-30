# transmission/dataset_generator.py

import numpy as np
from typing import Dict, List
from pathlib import Path
from .config import SystemConfig
from .modulator import Modulator
from .channel import CDLChannel
from .receiver import Receiver

class DatasetGenerator:
    def __init__(self, config: SystemConfig):
        self.config = config
        self.modulator = Modulator(config)
        self.channel = CDLChannel(config)
        self.receiver = Receiver(config)
        
    def generate_single_sample(self) -> Dict:
        total_data_symbols = self.config.num_data_symbols * (self.config.Nt - self.config.Np)
        data_bits = self.modulator.generate_random_bits(total_data_symbols)
        data_symbols = self.modulator.bits_to_symbols(data_bits)
        
        pilots = self.modulator.generate_pilots(self.config.Np)
        
        ofdm_grid, pilot_mask = self.modulator.create_ofdm_grid(data_symbols, pilots)
        
        H_true = self.channel.generate_response()
        
        received_grid = np.zeros_like(ofdm_grid, dtype=complex)
        for t in range(self.config.num_data_symbols):
            transmitted = ofdm_grid[:, t]
            received_grid[:, t] = self.channel.apply(transmitted, H_true)
        
        return {
            'H_true': H_true,
            'transmitted_grid': ofdm_grid,
            'received_grid': received_grid,
            'pilot_mask': pilot_mask,
            'pilots': pilots,
            'data_bits': data_bits,
            'data_symbols': data_symbols,
            'snr_db': self.config.snr_db
        }
    
    def generate_dataset(self, num_samples: int = None) -> List[Dict]:
        if num_samples is None:
            num_samples = self.config.num_samples
            
        dataset = []
        for i in range(num_samples):
            sample = self.generate_single_sample()
            dataset.append(sample)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: str):
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(save_path), dataset, allow_pickle=True)
        
    def load_dataset(self, filepath: str) -> List[Dict]:
        return np.load(filepath, allow_pickle=True)