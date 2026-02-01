import numpy as np
from typing import Tuple, Dict, Optional
from scipy.interpolate import interp1d
from transmission_system.config import TransmissionConfig


def extract_pilots(Y_grid: np.ndarray, 
                  pilot_mask: np.ndarray, 
                  pilot_value: complex) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    if Y_grid.ndim == 2:
        Nsym, Nfft = Y_grid.shape
        Nr = 1
        
        Yp = Y_grid[pilot_mask]
        Xp = np.full(Yp.shape, pilot_value, dtype=complex)
        
    else:
        Nsym, Nfft, Nr = Y_grid.shape
        
        num_pilots = np.sum(pilot_mask)
        Yp = np.zeros((num_pilots, Nr), dtype=complex)
        
        for rx in range(Nr):
            Yp[:, rx] = Y_grid[:, :, rx][pilot_mask]
        
        Xp = np.full(num_pilots, pilot_value, dtype=complex)
    
    return Yp, Xp, pilot_mask


def pilot_ls_estimate(Yp: np.ndarray, Xp: np.ndarray) -> np.ndarray:
    
    H_pilots = Yp / (Xp[..., None] if Yp.ndim > 1 and Xp.ndim == 1 else Xp + 1e-12)
    
    return H_pilots


def interpolate_channel_linear(H_pilot_sparse: np.ndarray, 
                               pilot_indices: np.ndarray,
                               Nfft: int, 
                               Nsym: int) -> np.ndarray:
    
    if H_pilot_sparse.ndim == 1:
        
        H_pilot_reshaped = H_pilot_sparse.reshape(Nsym, len(pilot_indices))
        
        H_full = np.zeros((Nsym, Nfft), dtype=complex)
        
        for n in range(Nsym):
            real_interp = interp1d(pilot_indices, np.real(H_pilot_reshaped[n, :]), 
                                  kind='linear', fill_value='extrapolate')
            imag_interp = interp1d(pilot_indices, np.imag(H_pilot_reshaped[n, :]), 
                                  kind='linear', fill_value='extrapolate')
            
            all_indices = np.arange(Nfft)
            H_full[n, :] = real_interp(all_indices) + 1j * imag_interp(all_indices)
        
        return H_full
    
    else:
        
        num_pilots_total, Nr = H_pilot_sparse.shape
        num_pilots_per_sym = len(pilot_indices)
        
        H_pilot_reshaped = H_pilot_sparse.reshape(Nsym, num_pilots_per_sym, Nr)
        
        H_full = np.zeros((Nsym, Nfft, Nr), dtype=complex)
        
        for n in range(Nsym):
            for rx in range(Nr):
                real_interp = interp1d(pilot_indices, np.real(H_pilot_reshaped[n, :, rx]), 
                                      kind='linear', fill_value='extrapolate')
                imag_interp = interp1d(pilot_indices, np.imag(H_pilot_reshaped[n, :, rx]), 
                                      kind='linear', fill_value='extrapolate')
                
                all_indices = np.arange(Nfft)
                H_full[n, :, rx] = real_interp(all_indices) + 1j * imag_interp(all_indices)
        
        return H_full


def interpolate_channel_cubic(H_pilot_sparse: np.ndarray, 
                              pilot_indices: np.ndarray,
                              Nfft: int, 
                              Nsym: int) -> np.ndarray:
    
    if H_pilot_sparse.ndim == 1:
        H_pilot_reshaped = H_pilot_sparse.reshape(Nsym, len(pilot_indices))
        H_full = np.zeros((Nsym, Nfft), dtype=complex)
        
        for n in range(Nsym):
            if len(pilot_indices) >= 4:
                real_interp = interp1d(pilot_indices, np.real(H_pilot_reshaped[n, :]), 
                                      kind='cubic', fill_value='extrapolate')
                imag_interp = interp1d(pilot_indices, np.imag(H_pilot_reshaped[n, :]), 
                                      kind='cubic', fill_value='extrapolate')
            else:
                real_interp = interp1d(pilot_indices, np.real(H_pilot_reshaped[n, :]), 
                                      kind='linear', fill_value='extrapolate')
                imag_interp = interp1d(pilot_indices, np.imag(H_pilot_reshaped[n, :]), 
                                      kind='linear', fill_value='extrapolate')
            
            all_indices = np.arange(Nfft)
            H_full[n, :] = real_interp(all_indices) + 1j * imag_interp(all_indices)
        
        return H_full
    
    else:
        num_pilots_total, Nr = H_pilot_sparse.shape
        num_pilots_per_sym = len(pilot_indices)
        H_pilot_reshaped = H_pilot_sparse.reshape(Nsym, num_pilots_per_sym, Nr)
        
        H_full = np.zeros((Nsym, Nfft, Nr), dtype=complex)
        
        for n in range(Nsym):
            for rx in range(Nr):
                if len(pilot_indices) >= 4:
                    real_interp = interp1d(pilot_indices, np.real(H_pilot_reshaped[n, :, rx]), 
                                          kind='cubic', fill_value='extrapolate')
                    imag_interp = interp1d(pilot_indices, np.imag(H_pilot_reshaped[n, :, rx]), 
                                          kind='cubic', fill_value='extrapolate')
                else:
                    real_interp = interp1d(pilot_indices, np.real(H_pilot_reshaped[n, :, rx]), 
                                          kind='linear', fill_value='extrapolate')
                    imag_interp = interp1d(pilot_indices, np.imag(H_pilot_reshaped[n, :, rx]), 
                                          kind='linear', fill_value='extrapolate')
                
                all_indices = np.arange(Nfft)
                H_full[n, :, rx] = real_interp(all_indices) + 1j * imag_interp(all_indices)
        
        return H_full


def smooth_channel_time(H_full: np.ndarray, window_size: int = 5) -> np.ndarray:
    
    if window_size % 2 == 0:
        window_size += 1
    
    pad_width = window_size // 2
    
    if H_full.ndim == 2:
        Nsym, Nfft = H_full.shape
        
        H_padded = np.pad(H_full, ((pad_width, pad_width), (0, 0)), mode='edge')
        
        H_smoothed = np.zeros_like(H_full)
        for n in range(Nsym):
            H_smoothed[n, :] = np.mean(H_padded[n:n+window_size, :], axis=0)
        
        return H_smoothed
    
    else:
        Nsym, Nfft, Nr = H_full.shape
        
        H_padded = np.pad(H_full, ((pad_width, pad_width), (0, 0), (0, 0)), mode='edge')
        
        H_smoothed = np.zeros_like(H_full)
        for n in range(Nsym):
            H_smoothed[n, :, :] = np.mean(H_padded[n:n+window_size, :, :], axis=0)
        
        return H_smoothed


def smooth_channel_frequency(H_full: np.ndarray, window_size: int = 3) -> np.ndarray:
    
    if window_size % 2 == 0:
        window_size += 1
    
    pad_width = window_size // 2
    
    if H_full.ndim == 2:
        Nsym, Nfft = H_full.shape
        
        H_padded = np.pad(H_full, ((0, 0), (pad_width, pad_width)), mode='edge')
        
        H_smoothed = np.zeros_like(H_full)
        for k in range(Nfft):
            H_smoothed[:, k] = np.mean(H_padded[:, k:k+window_size], axis=1)
        
        return H_smoothed
    
    else:
        Nsym, Nfft, Nr = H_full.shape
        
        H_padded = np.pad(H_full, ((0, 0), (pad_width, pad_width), (0, 0)), mode='edge')
        
        H_smoothed = np.zeros_like(H_full)
        for k in range(Nfft):
            H_smoothed[:, k, :] = np.mean(H_padded[:, k:k+window_size, :], axis=1)
        
        return H_smoothed


def receiver_frontend_process(Y_grid: np.ndarray, 
                              pilot_mask: np.ndarray, 
                              config: TransmissionConfig) -> Dict[str, np.ndarray]:
    
    pilot_indices = config.get_pilot_indices()
    
    Yp, Xp, mask_p = extract_pilots(Y_grid, pilot_mask, config.pilot_value)
    
    H_pilot_sparse = pilot_ls_estimate(Yp, Xp)
    
    H_pilot_full = interpolate_channel_linear(H_pilot_sparse, pilot_indices, 
                                              config.Nfft, config.Nsym)
    
    H_pilot_full_smoothed = smooth_channel_time(H_pilot_full, window_size=5)
    
    output = {
        'Y_grid': Y_grid,
        'pilot_mask': pilot_mask,
        'Yp': Yp,
        'Xp': Xp,
        'H_pilot_sparse': H_pilot_sparse,
        'H_pilot_full': H_pilot_full,
        'H_pilot_full_smoothed': H_pilot_full_smoothed,
        'pilot_indices': pilot_indices
    }
    
    return output


def equalize_zf(Y_grid: np.ndarray, H_est: np.ndarray) -> np.ndarray:
    
    Y_eq = Y_grid / (H_est + 1e-12)
    return Y_eq


def equalize_mmse(Y_grid: np.ndarray, H_est: np.ndarray, noise_var: float) -> np.ndarray:
    
    if Y_grid.ndim == 2:
        H_conj = np.conj(H_est)
        denominator = np.abs(H_est) ** 2 + noise_var
        Y_eq = (H_conj * Y_grid) / (denominator + 1e-12)
    
    else:
        Nsym, Nfft, Nr = Y_grid.shape
        Nt = H_est.shape[3] if H_est.ndim == 4 else 1
        
        Y_eq = np.zeros((Nsym, Nfft, Nt), dtype=complex)
        
        for n in range(Nsym):
            for k in range(Nfft):
                H_nk = H_est[n, k, :, :]
                y_nk = Y_grid[n, k, :]
                
                H_H = np.conj(H_nk.T)
                W_mmse = H_H @ np.linalg.inv(H_nk @ H_H + noise_var * np.eye(Nr))
                
                Y_eq[n, k, :] = W_mmse @ y_nk
    
    return Y_eq


def compute_channel_nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
    
    error = H_est - H_true
    nmse = np.mean(np.abs(error) ** 2) / np.mean(np.abs(H_true) ** 2)
    return nmse


def compute_evm(X_hat: np.ndarray, X_true: np.ndarray) -> float:
    
    error = X_hat - X_true
    evm = np.sqrt(np.mean(np.abs(error) ** 2)) / np.sqrt(np.mean(np.abs(X_true) ** 2))
    return evm


def estimate_noise_variance_from_pilots(Yp: np.ndarray, 
                                       Xp: np.ndarray, 
                                       H_pilot_sparse: np.ndarray) -> float:
    
    if Yp.ndim == 1:
        Y_pred = H_pilot_sparse * Xp
    else:
        Y_pred = H_pilot_sparse * Xp[:, None]
    
    error = Yp - Y_pred
    noise_var = np.mean(np.abs(error) ** 2)
    
    return np.real(noise_var)