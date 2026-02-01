import numpy as np
from typing import Tuple, Optional


def generate_qam_constellation(M: int) -> np.ndarray:
    assert M in [4, 16, 64, 256], "M must be 4, 16, 64, or 256"
    
    k = int(np.sqrt(M))
    assert k * k == M, "M must be a perfect square for QAM"
    
    real_levels = np.arange(-(k-1), k, 2)
    imag_levels = np.arange(-(k-1), k, 2)
    
    constellation = np.array([r + 1j*i for i in imag_levels for r in real_levels])
    
    avg_power = np.mean(np.abs(constellation)**2)
    constellation = constellation / np.sqrt(avg_power)
    
    return constellation


def generate_psk_constellation(M: int) -> np.ndarray:
    angles = 2 * np.pi * np.arange(M) / M
    constellation = np.exp(1j * angles)
    return constellation


def get_constellation(M: int, constellation_type: str = 'qam') -> np.ndarray:
    if constellation_type == 'qam':
        return generate_qam_constellation(M)
    elif constellation_type == 'psk':
        return generate_psk_constellation(M)
    else:
        raise ValueError(f"Unknown constellation type: {constellation_type}")


def map_bits_to_symbols(bits: np.ndarray, M: int, constellation_type: str = 'qam') -> np.ndarray:
    k = int(np.log2(M))
    
    num_bits = len(bits)
    num_symbols = num_bits // k
    
    bits = bits[:num_symbols * k]
    
    bits_reshaped = bits.reshape(num_symbols, k)
    
    indices = np.packbits(bits_reshaped, axis=1, bitorder='big')[:, 0] >> (8 - k)
    
    constellation = get_constellation(M, constellation_type)
    
    return constellation[indices]


def symbols_to_bit_indices(symbols: np.ndarray, M: int) -> np.ndarray:
    constellation = get_constellation(M, 'qam')
    
    distances = np.abs(symbols[:, None] - constellation[None, :])
    indices = np.argmin(distances, axis=1)
    
    return indices


def hard_decision(Y_eq: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    Y_flat = Y_eq.reshape(-1)
    
    distances = np.abs(Y_flat[:, None] - constellation[None, :])
    indices = np.argmin(distances, axis=1)
    
    X_hat = constellation[indices]
    
    return X_hat.reshape(Y_eq.shape)


def compute_bit_llrs(Y_eq: np.ndarray, noise_var: float, M: int, 
                     constellation_type: str = 'qam') -> np.ndarray:
    constellation = get_constellation(M, constellation_type)
    k = int(np.log2(M))
    
    Y_flat = Y_eq.reshape(-1)
    num_symbols = len(Y_flat)
    
    llrs = np.zeros((num_symbols, k))
    
    for sym_idx in range(num_symbols):
        y = Y_flat[sym_idx]
        
        for bit_idx in range(k):
            
            bit_mask = 1 << (k - 1 - bit_idx)
            
            indices_0 = np.array([i for i in range(M) if (i & bit_mask) == 0])
            indices_1 = np.array([i for i in range(M) if (i & bit_mask) != 0])
            
            symbols_0 = constellation[indices_0]
            symbols_1 = constellation[indices_1]
            
            dist_0 = np.abs(y - symbols_0) ** 2
            dist_1 = np.abs(y - symbols_1) ** 2
            
            min_dist_0 = np.min(dist_0)
            min_dist_1 = np.min(dist_1)
            
            llr = (min_dist_1 - min_dist_0) / noise_var
            
            llrs[sym_idx, bit_idx] = llr
    
    original_shape = Y_eq.shape
    llrs_reshaped = llrs.reshape(original_shape + (k,))
    
    return llrs_reshaped


def compute_symbol_reliability(llrs: np.ndarray) -> np.ndarray:
    if llrs.ndim == 3:
        min_abs_llr = np.min(np.abs(llrs), axis=-1)
    else:
        min_abs_llr = np.abs(llrs)
    
    return min_abs_llr


def bits_to_symbols_gray(bits: np.ndarray, M: int) -> np.ndarray:
    k = int(np.log2(M))
    num_bits = len(bits)
    num_symbols = num_bits // k
    
    bits = bits[:num_symbols * k]
    bits_reshaped = bits.reshape(num_symbols, k)
    
    gray_to_bin = lambda g: g ^ (g >> 1)
    
    indices = np.zeros(num_symbols, dtype=int)
    for i in range(num_symbols):
        gray_val = 0
        for j in range(k):
            gray_val = (gray_val << 1) | bits_reshaped[i, j]
        indices[i] = gray_to_bin(gray_val)
    
    constellation = get_constellation(M, 'qam')
    return constellation[indices]


def get_euclidean_distance_to_constellation(Y: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    Y_flat = Y.reshape(-1)
    
    distances = np.abs(Y_flat[:, None] - constellation[None, :])
    min_distances = np.min(distances, axis=1)
    
    return min_distances.reshape(Y.shape)