import numpy as np
from typing import Tuple, Optional
from transmission_system.config import TransmissionConfig


def build_pilot_pattern(config: TransmissionConfig) -> Tuple[np.ndarray, np.ndarray]:
    pilot_indices = config.get_pilot_indices()
    data_indices = config.get_data_indices()
    
    return pilot_indices, data_indices


def build_resource_grid(data_symbols: np.ndarray, 
                       pilot_value: complex,
                       config: TransmissionConfig) -> Tuple[np.ndarray, np.ndarray]:
    
    pilot_indices, data_indices = build_pilot_pattern(config)
    
    if config.Nt == 1:
        X_grid = np.zeros((config.Nsym, config.Nfft), dtype=complex)
        pilot_mask = np.zeros((config.Nsym, config.Nfft), dtype=bool)
        
        for n in range(config.Nsym):
            X_grid[n, pilot_indices] = pilot_value
            pilot_mask[n, pilot_indices] = True
            
            num_data_this_symbol = len(data_indices)
            start_idx = n * num_data_this_symbol
            end_idx = start_idx + num_data_this_symbol
            
            if end_idx <= len(data_symbols):
                X_grid[n, data_indices] = data_symbols[start_idx:end_idx]
            else:
                available = len(data_symbols) - start_idx
                if available > 0:
                    X_grid[n, data_indices[:available]] = data_symbols[start_idx:]
    
    else:
        X_grid = np.zeros((config.Nsym, config.Nfft, config.Nt), dtype=complex)
        pilot_mask = np.zeros((config.Nsym, config.Nfft), dtype=bool)
        
        for n in range(config.Nsym):
            for tx in range(config.Nt):
                X_grid[n, pilot_indices, tx] = pilot_value
            
            pilot_mask[n, pilot_indices] = True
            
            num_data_this_symbol = len(data_indices)
            symbols_needed = num_data_this_symbol * config.Nt
            start_idx = n * symbols_needed
            end_idx = start_idx + symbols_needed
            
            if end_idx <= len(data_symbols):
                data_block = data_symbols[start_idx:end_idx].reshape(num_data_this_symbol, config.Nt)
                X_grid[n, data_indices, :] = data_block
    
    return X_grid, pilot_mask


def ofdm_modulate(X_grid: np.ndarray, config: TransmissionConfig) -> np.ndarray:
    
    if X_grid.ndim == 2:
        Nsym, Nfft = X_grid.shape
        Nt = 1
        X_grid_3d = X_grid[:, :, None]
    else:
        Nsym, Nfft, Nt = X_grid.shape
        X_grid_3d = X_grid
    
    x_time = np.zeros((Nsym, Nfft + config.Ncp, Nt), dtype=complex)
    
    for tx in range(Nt):
        x_freq = X_grid_3d[:, :, tx]
        
        x_no_cp = np.fft.ifft(x_freq, axis=1) * np.sqrt(Nfft)
        
        x_time[:, :config.Ncp, tx] = x_no_cp[:, -config.Ncp:]
        x_time[:, config.Ncp:, tx] = x_no_cp
    
    if Nt == 1:
        x_time = x_time[:, :, 0]
    
    return x_time


def ofdm_demodulate(y_time: np.ndarray, config: TransmissionConfig) -> np.ndarray:
    
    if y_time.ndim == 2:
        Nsym, total_len = y_time.shape
        Nr = 1
        y_time_3d = y_time[:, :, None]
    else:
        Nsym, total_len, Nr = y_time.shape
        y_time_3d = y_time
    
    assert total_len == config.Nfft + config.Ncp
    
    Y_grid = np.zeros((Nsym, config.Nfft, Nr), dtype=complex)
    
    for rx in range(Nr):
        y_no_cp = y_time_3d[:, config.Ncp:, rx]
        
        Y_grid[:, :, rx] = np.fft.fft(y_no_cp, axis=1) / np.sqrt(config.Nfft)
    
    if Nr == 1:
        Y_grid = Y_grid[:, :, 0]
    
    return Y_grid


def add_cyclic_prefix(x: np.ndarray, Ncp: int) -> np.ndarray:
    
    if x.ndim == 2:
        Nsym, Nfft = x.shape
        x_cp = np.zeros((Nsym, Nfft + Ncp), dtype=complex)
        x_cp[:, :Ncp] = x[:, -Ncp:]
        x_cp[:, Ncp:] = x
    else:
        Nsym, Nfft, Nt = x.shape
        x_cp = np.zeros((Nsym, Nfft + Ncp, Nt), dtype=complex)
        x_cp[:, :Ncp, :] = x[:, -Ncp:, :]
        x_cp[:, Ncp:, :] = x
    
    return x_cp


def remove_cyclic_prefix(y: np.ndarray, Ncp: int) -> np.ndarray:
    
    if y.ndim == 2:
        return y[:, Ncp:]
    else:
        return y[:, Ncp:, :]


def serialize_ofdm_symbols(x_time: np.ndarray) -> np.ndarray:
    
    if x_time.ndim == 2:
        return x_time.flatten()
    else:
        Nsym, total_len, Nt = x_time.shape
        return x_time.reshape(-1, Nt)


def deserialize_ofdm_symbols(x_serial: np.ndarray, Nsym: int, total_len: int) -> np.ndarray:
    
    if x_serial.ndim == 1:
        return x_serial.reshape(Nsym, total_len)
    else:
        Nt = x_serial.shape[1]
        return x_serial.reshape(Nsym, total_len, Nt)


def normalize_ofdm_power(x: np.ndarray) -> np.ndarray:
    
    avg_power = np.mean(np.abs(x) ** 2)
    if avg_power > 0:
        return x / np.sqrt(avg_power)
    return x


def get_subcarrier_spacing(config: TransmissionConfig) -> float:
    
    return config.symbol_rate_khz * 1000.0 / config.Nfft