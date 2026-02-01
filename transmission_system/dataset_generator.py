import numpy as np
import os
import h5py
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from transmission_system.config import TransmissionConfig
from transmission_system.constellation import map_bits_to_symbols, get_constellation
from transmission_system.modulator import build_resource_grid, ofdm_modulate, ofdm_demodulate
from transmission_system.channel import apply_channel
from transmission_system.receiver_frontend import receiver_frontend_process, estimate_noise_variance_from_pilots


def generate_single_sample(config: TransmissionConfig, 
                          rng: np.random.Generator,
                          sample_id: int) -> Dict[str, np.ndarray]:
    
    num_data_symbols = config.num_data_tones * config.Nsym
    if config.Nt > 1:
        num_data_symbols *= config.Nt
    
    num_bits = num_data_symbols * config.bits_per_symbol
    bits = rng.integers(0, 2, size=num_bits, dtype=np.uint8)
    
    data_symbols = map_bits_to_symbols(bits, config.modulation_order, config.constellation_type)
    
    X_grid, pilot_mask = build_resource_grid(data_symbols, config.pilot_value, config)
    
    x_time = ofdm_modulate(X_grid, config)
    
    y_time, H_true = apply_channel(x_time, config, rng)
    
    Y_grid = ofdm_demodulate(y_time, config)
    
    rx_output = receiver_frontend_process(Y_grid, pilot_mask, config)
    
    noise_var = estimate_noise_variance_from_pilots(
        rx_output['Yp'], 
        rx_output['Xp'], 
        rx_output['H_pilot_sparse']
    )
    
    sample = {
        'sample_id': sample_id,
        'bits': bits,
        'X_grid': X_grid,
        'Y_grid': Y_grid,
        'pilot_mask': pilot_mask,
        'Yp': rx_output['Yp'],
        'Xp': rx_output['Xp'],
        'H_true': H_true,
        'H_pilot_sparse': rx_output['H_pilot_sparse'],
        'H_pilot_full': rx_output['H_pilot_full'],
        'H_pilot_full_smoothed': rx_output['H_pilot_full_smoothed'],
        'pilot_indices': rx_output['pilot_indices'],
        'noise_var': noise_var,
        'snr_db': config.snr_db,
        'doppler_hz': config.doppler_hz,
    }
    
    return sample


def generate_dataset(config: TransmissionConfig, 
                    num_samples: int,
                    split: str = 'train',
                    seed_offset: int = 0) -> Dict[str, np.ndarray]:
    
    rng = np.random.default_rng(config.seed + seed_offset)
    
    samples = []
    for i in tqdm(range(num_samples), desc=f"Generating {split} samples"):
        sample = generate_single_sample(config, rng, i)
        samples.append(sample)
    
    dataset = {}
    for key in samples[0].keys():
        if key == 'sample_id':
            dataset[key] = np.array([s[key] for s in samples])
        elif key in ['noise_var', 'snr_db', 'doppler_hz']:
            dataset[key] = np.array([s[key] for s in samples])
        else:
            dataset[key] = np.stack([s[key] for s in samples], axis=0)
    
    return dataset


def save_dataset_hdf5(dataset: Dict[str, np.ndarray], 
                     filepath: str,
                     config: TransmissionConfig):
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with h5py.File(filepath, 'w') as f:
        
        for key, value in dataset.items():
            if np.iscomplexobj(value):
                f.create_dataset(f'{key}/real', data=np.real(value), compression='gzip')
                f.create_dataset(f'{key}/imag', data=np.imag(value), compression='gzip')
            else:
                f.create_dataset(key, data=value, compression='gzip')
        
        config_group = f.create_group('config')
        config_group.attrs['Nfft'] = config.Nfft
        config_group.attrs['Ncp'] = config.Ncp
        config_group.attrs['Nsym'] = config.Nsym
        config_group.attrs['pilot_spacing'] = config.pilot_spacing
        config_group.attrs['Nt'] = config.Nt
        config_group.attrs['Nr'] = config.Nr
        config_group.attrs['modulation_order'] = config.modulation_order
        config_group.attrs['constellation_type'] = config.constellation_type
        config_group.attrs['snr_db'] = config.snr_db
        config_group.attrs['channel_model'] = config.channel_model
        config_group.attrs['doppler_hz'] = config.doppler_hz
        config_group.attrs['time_varying'] = config.time_varying
        config_group.attrs['block_fading'] = config.block_fading


def load_dataset_hdf5(filepath: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    
    dataset = {}
    config_dict = {}
    
    with h5py.File(filepath, 'r') as f:
        
        for key in f.keys():
            if key == 'config':
                continue
            
            if isinstance(f[key], h5py.Group):
                real_part = f[f'{key}/real'][:]
                imag_part = f[f'{key}/imag'][:]
                dataset[key] = real_part + 1j * imag_part
            else:
                dataset[key] = f[key][:]
        
        if 'config' in f:
            config_group = f['config']
            for attr_name in config_group.attrs:
                config_dict[attr_name] = config_group.attrs[attr_name]
    
    return dataset, config_dict


def generate_and_save_all_splits(config: TransmissionConfig):
    
    base_path = Path(config.dataset_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    dataset_name = f"ofdm_Nfft{config.Nfft}_Nsym{config.Nsym}_M{config.modulation_order}_SNR{config.snr_db}dB_Doppler{config.doppler_hz}Hz"
    
    print(f"Generating training dataset ({config.num_samples_train} samples)...")
    train_dataset = generate_dataset(config, config.num_samples_train, split='train', seed_offset=0)
    train_path = base_path / f"{dataset_name}_train.h5"
    save_dataset_hdf5(train_dataset, str(train_path), config)
    print(f"Saved training dataset to {train_path}")
    
    print(f"\nGenerating validation dataset ({config.num_samples_val} samples)...")
    val_dataset = generate_dataset(config, config.num_samples_val, split='val', seed_offset=100000)
    val_path = base_path / f"{dataset_name}_val.h5"
    save_dataset_hdf5(val_dataset, str(val_path), config)
    print(f"Saved validation dataset to {val_path}")
    
    print(f"\nGenerating test dataset ({config.num_samples_test} samples)...")
    test_dataset = generate_dataset(config, config.num_samples_test, split='test', seed_offset=200000)
    test_path = base_path / f"{dataset_name}_test.h5"
    save_dataset_hdf5(test_dataset, str(test_path), config)
    print(f"Saved test dataset to {test_path}")
    
    return {
        'train': str(train_path),
        'val': str(val_path),
        'test': str(test_path)
    }


def verify_dataset_shapes(dataset: Dict[str, np.ndarray], config: TransmissionConfig):
    
    print("\nDataset Shape Verification:")
    print("=" * 60)
    
    num_samples = dataset['Y_grid'].shape[0]
    print(f"Number of samples: {num_samples}")
    
    print(f"\nY_grid shape: {dataset['Y_grid'].shape}")
    expected_Y = (num_samples, config.Nsym, config.Nfft) if config.Nr == 1 else (num_samples, config.Nsym, config.Nfft, config.Nr)
    print(f"Expected: {expected_Y}")
    
    print(f"\nH_true shape: {dataset['H_true'].shape}")
    if config.Nt == 1 and config.Nr == 1:
        expected_H = (num_samples, config.Nsym, config.Nfft)
    else:
        expected_H = (num_samples, config.Nsym, config.Nfft, config.Nr, config.Nt)
    print(f"Expected: {expected_H}")
    
    print(f"\nH_pilot_full shape: {dataset['H_pilot_full'].shape}")
    print(f"Expected: same as H_true")
    
    print(f"\npilot_mask shape: {dataset['pilot_mask'].shape}")
    print(f"Expected: ({num_samples}, {config.Nsym}, {config.Nfft})")
    
    print(f"\nX_grid shape: {dataset['X_grid'].shape}")
    
    print(f"\nnoise_var shape: {dataset['noise_var'].shape}")
    print(f"Values range: [{dataset['noise_var'].min():.6f}, {dataset['noise_var'].max():.6f}]")
    
    print("\n" + "=" * 60)


class OFDMDataset:
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset, self.config_dict = load_dataset_hdf5(filepath)
        
    def __len__(self) -> int:
        return self.dataset['Y_grid'].shape[0]
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = {}
        for key, value in self.dataset.items():
            sample[key] = value[idx]
        return sample
    
    def get_batch(self, indices: np.ndarray) -> Dict[str, np.ndarray]:
        batch = {}
        for key, value in self.dataset.items():
            batch[key] = value[indices]
        return batch
    
    def get_diffusion_conditioning(self, idx: int) -> Dict[str, np.ndarray]:
        
        return {
            'H_pilot_full': self.dataset['H_pilot_full'][idx],
            'pilot_mask': self.dataset['pilot_mask'][idx],
            'Yp': self.dataset['Yp'][idx],
            'Xp': self.dataset['Xp'][idx],
            'Y_grid': self.dataset['Y_grid'][idx],
            'noise_var': self.dataset['noise_var'][idx]
        }
    
    def get_ground_truth(self, idx: int) -> np.ndarray:
        return self.dataset['H_true'][idx]
    
    def get_pilot_baseline(self, idx: int) -> np.ndarray:
        return self.dataset['H_pilot_full'][idx]