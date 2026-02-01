import numpy as np
from typing import Tuple, Optional
from transmission_system.config import TransmissionConfig


def generate_channel_taps(config: TransmissionConfig, rng: np.random.Generator) -> np.ndarray:
    
    if config.channel_model == 'rayleigh':
        h_taps = (rng.standard_normal(config.num_taps) + 
                 1j * rng.standard_normal(config.num_taps)) / np.sqrt(2)
        
        delays = np.arange(config.num_taps)
        decay = np.exp(-delays / (config.num_taps / 3))
        h_taps = h_taps * np.sqrt(decay)
        h_taps = h_taps / np.linalg.norm(h_taps)
        
    elif config.channel_model == 'rician':
        K_factor_db = 10.0
        K_linear = 10.0 ** (K_factor_db / 10.0)
        
        los_component = np.sqrt(K_linear / (K_linear + 1))
        scatter_component = np.sqrt(1 / (K_linear + 1))
        
        h_taps = np.zeros(config.num_taps, dtype=complex)
        h_taps[0] = los_component
        
        scatter = (rng.standard_normal(config.num_taps) + 
                  1j * rng.standard_normal(config.num_taps)) / np.sqrt(2)
        
        delays = np.arange(config.num_taps)
        decay = np.exp(-delays / (config.num_taps / 3))
        scatter = scatter * np.sqrt(decay)
        
        h_taps = h_taps + scatter_component * scatter
        h_taps = h_taps / np.linalg.norm(h_taps)
        
    elif config.channel_model == 'tapped_delay':
        h_taps = (rng.standard_normal(config.num_taps) + 
                 1j * rng.standard_normal(config.num_taps)) / np.sqrt(2)
        
        h_taps = h_taps / np.linalg.norm(h_taps)
    
    else:
        raise ValueError(f"Unknown channel model: {config.channel_model}")
    
    return h_taps


def taps_to_frequency_response(h_taps: np.ndarray, Nfft: int) -> np.ndarray:
    
    H_freq = np.fft.fft(h_taps, n=Nfft)
    return H_freq


def generate_time_varying_channel(config: TransmissionConfig, 
                                  rng: np.random.Generator) -> np.ndarray:
    
    if config.Nt == 1 and config.Nr == 1:
        if config.time_varying and not config.block_fading:
            H_freq = np.zeros((config.Nsym, config.Nfft), dtype=complex)
            
            h_taps_initial = generate_channel_taps(config, rng)
            
            fd_norm = config.max_doppler_norm
            
            for n in range(config.Nsym):
                phase_drift = np.exp(2j * np.pi * fd_norm * n * rng.standard_normal(config.num_taps))
                
                amplitude_drift = 1.0 + 0.1 * rng.standard_normal(config.num_taps)
                
                h_taps = h_taps_initial * phase_drift * amplitude_drift
                h_taps = h_taps / np.linalg.norm(h_taps)
                
                H_freq[n, :] = taps_to_frequency_response(h_taps, config.Nfft)
        
        elif config.block_fading:
            h_taps = generate_channel_taps(config, rng)
            H_single = taps_to_frequency_response(h_taps, config.Nfft)
            H_freq = np.tile(H_single, (config.Nsym, 1))
        
        else:
            h_taps = generate_channel_taps(config, rng)
            H_single = taps_to_frequency_response(h_taps, config.Nfft)
            H_freq = np.tile(H_single, (config.Nsym, 1))
    
    else:
        if config.time_varying and not config.block_fading:
            H_freq = np.zeros((config.Nsym, config.Nfft, config.Nr, config.Nt), dtype=complex)
            
            h_taps_initial = np.zeros((config.Nr, config.Nt, config.num_taps), dtype=complex)
            for rx in range(config.Nr):
                for tx in range(config.Nt):
                    h_taps_initial[rx, tx, :] = generate_channel_taps(config, rng)
            
            fd_norm = config.max_doppler_norm
            
            for n in range(config.Nsym):
                for rx in range(config.Nr):
                    for tx in range(config.Nt):
                        phase_drift = np.exp(2j * np.pi * fd_norm * n * rng.standard_normal(config.num_taps))
                        amplitude_drift = 1.0 + 0.1 * rng.standard_normal(config.num_taps)
                        
                        h_taps = h_taps_initial[rx, tx, :] * phase_drift * amplitude_drift
                        h_taps = h_taps / np.linalg.norm(h_taps)
                        
                        H_freq[n, :, rx, tx] = taps_to_frequency_response(h_taps, config.Nfft)
        
        else:
            H_freq = np.zeros((config.Nsym, config.Nfft, config.Nr, config.Nt), dtype=complex)
            
            for rx in range(config.Nr):
                for tx in range(config.Nt):
                    h_taps = generate_channel_taps(config, rng)
                    H_single = taps_to_frequency_response(h_taps, config.Nfft)
                    H_freq[:, :, rx, tx] = np.tile(H_single, (config.Nsym, 1))
    
    return H_freq


def apply_frequency_domain_channel(Y_freq: np.ndarray, H_freq: np.ndarray) -> np.ndarray:
    
    if Y_freq.ndim == 2 and H_freq.ndim == 2:
        return Y_freq * H_freq
    
    elif Y_freq.ndim == 3 and H_freq.ndim == 4:
        Nsym, Nfft, Nt = Y_freq.shape
        Nr = H_freq.shape[2]
        
        Y_received = np.zeros((Nsym, Nfft, Nr), dtype=complex)
        
        for n in range(Nsym):
            for k in range(Nfft):
                H_nk = H_freq[n, k, :, :]
                X_nk = Y_freq[n, k, :]
                Y_received[n, k, :] = H_nk @ X_nk
        
        return Y_received
    
    else:
        raise ValueError(f"Incompatible shapes: Y_freq {Y_freq.shape}, H_freq {H_freq.shape}")


def apply_channel(x_time: np.ndarray, config: TransmissionConfig, 
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    
    H_freq = generate_time_varying_channel(config, rng)
    
    if x_time.ndim == 2:
        Nsym, total_len = x_time.shape
        Nt = 1
        x_time_3d = x_time[:, :, None]
    else:
        Nsym, total_len, Nt = x_time.shape
        x_time_3d = x_time
    
    Nfft = config.Nfft
    Ncp = config.Ncp
    
    X_freq = np.zeros((Nsym, Nfft, Nt), dtype=complex)
    for tx in range(Nt):
        x_no_cp = x_time_3d[:, Ncp:, tx]
        X_freq[:, :, tx] = np.fft.fft(x_no_cp, axis=1) / np.sqrt(Nfft)
    
    if H_freq.ndim == 2:
        Y_freq = X_freq[:, :, 0] * H_freq
    else:
        Y_freq = apply_frequency_domain_channel(X_freq, H_freq)
    
    noise_var = config.noise_variance
    
    if Y_freq.ndim == 2:
        Nr = 1
        Y_freq_3d = Y_freq[:, :, None]
    else:
        Nr = Y_freq.shape[2]
        Y_freq_3d = Y_freq
    
    noise = np.sqrt(noise_var / 2) * (rng.standard_normal(Y_freq_3d.shape) + 
                                       1j * rng.standard_normal(Y_freq_3d.shape))
    Y_freq_noisy = Y_freq_3d + noise
    
    y_time = np.zeros((Nsym, total_len, Nr), dtype=complex)
    for rx in range(Nr):
        y_no_cp = np.fft.ifft(Y_freq_noisy[:, :, rx], axis=1) * np.sqrt(Nfft)
        y_time[:, Ncp:, rx] = y_no_cp
        y_time[:, :Ncp, rx] = y_no_cp[:, -Ncp:]
    
    if Nr == 1:
        y_time = y_time[:, :, 0]
    
    return y_time, H_freq


def estimate_channel_from_pilots_simple(Y_pilots: np.ndarray, 
                                       X_pilots: np.ndarray) -> np.ndarray:
    
    H_pilots = Y_pilots / (X_pilots + 1e-12)
    return H_pilots


def get_noise_variance_estimate(Y_grid: np.ndarray, 
                                H_est: np.ndarray, 
                                X_est: np.ndarray) -> float:
    
    if Y_grid.ndim == 2:
        Y_pred = H_est * X_est
        error = Y_grid - Y_pred
    else:
        error = Y_grid - H_est * X_est[:, :, None]
    
    noise_var = np.mean(np.abs(error) ** 2)
    return np.real(noise_var)