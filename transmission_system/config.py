import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class TransmissionConfig:
    seed: int = 42
    
    Nfft: int = 64
    Ncp: int = 16
    Nsym: int = 100
    
    pilot_spacing: int = 4
    pilot_value: complex = 1.0 + 0.0j
    
    Nt: int = 1
    Nr: int = 1
    
    modulation_order: int = 16
    constellation_type: Literal['qam', 'psk'] = 'qam'
    
    snr_db: float = 20.0
    
    channel_model: Literal['rayleigh', 'rician', 'tapped_delay'] = 'rayleigh'
    num_taps: int = 6
    delay_spread_ns: float = 50.0
    
    doppler_hz: float = 100.0
    carrier_freq_ghz: float = 2.4
    symbol_rate_khz: float = 1000.0
    
    time_varying: bool = True
    block_fading: bool = False
    
    dd_tau_threshold: float = 4.0
    dd_gate_mode: Literal['bit', 'symbol'] = 'symbol'
    dd_mu_normalizer: float = 0.01
    dd_mu_noise: float = 0.05
    
    code_rate: Optional[float] = None
    
    num_samples_train: int = 10000
    num_samples_val: int = 1000
    num_samples_test: int = 1000
    
    dataset_path: str = './datasets'
    
    @property
    def Nsc(self) -> int:
        return self.Nfft
    
    @property
    def num_data_tones(self) -> int:
        num_pilots = self.Nfft // self.pilot_spacing
        return self.Nfft - num_pilots
    
    @property
    def bits_per_symbol(self) -> int:
        return int(np.log2(self.modulation_order))
    
    @property
    def symbol_duration_us(self) -> float:
        return (self.Nfft + self.Ncp) / self.symbol_rate_khz
    
    @property
    def max_doppler_norm(self) -> float:
        return self.doppler_hz * self.symbol_duration_us * 1e-6
    
    @property
    def noise_variance(self) -> float:
        return 10.0 ** (-self.snr_db / 10.0)
    
    def get_pilot_indices(self) -> np.ndarray:
        return np.arange(0, self.Nfft, self.pilot_spacing)
    
    def get_data_indices(self) -> np.ndarray:
        pilot_idx = self.get_pilot_indices()
        all_idx = np.arange(self.Nfft)
        return np.setdiff1d(all_idx, pilot_idx)


def load_config(config_path: Optional[str] = None) -> TransmissionConfig:
    if config_path is None:
        return TransmissionConfig()
    
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return TransmissionConfig(**config_dict)