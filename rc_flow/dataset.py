import torch
import numpy as np
from typing import Tuple, Dict
from .config import RCFlowConfig


class ChannelDataset:
    def __init__(self, config: RCFlowConfig):
        self.config = config
        self.Nr = config.Nr
        self.Nt = config.Nt
        self.Np = config.Np

    def generate_channel(self, batch_size: int, num_paths: int = 5) -> torch.Tensor:
        H = torch.zeros(batch_size, self.Nr, self.Nt, dtype=torch.complex64)

        for _ in range(num_paths):
            aod = torch.rand(batch_size) * 2 * np.pi
            aoa = torch.rand(batch_size) * 2 * np.pi

            a_tx = torch.exp(1j * np.pi * torch.arange(self.Nt).float().unsqueeze(0) *
                           torch.sin(aod).unsqueeze(1))
            a_rx = torch.exp(1j * np.pi * torch.arange(self.Nr).float().unsqueeze(0) *
                           torch.sin(aoa).unsqueeze(1))

            gain = (torch.randn(batch_size) + 1j * torch.randn(batch_size)) / np.sqrt(2)

            H_path = gain.unsqueeze(1).unsqueeze(2) * a_rx.unsqueeze(2) * a_tx.unsqueeze(1).conj()
            H = H + H_path / np.sqrt(num_paths)

        H = H / torch.sqrt(torch.mean(torch.abs(H) ** 2, dim=(1, 2), keepdim=True))
        return H

    def add_noise(self, H: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        if snr_db is None:
            snr_db = self.config.snr_db

        snr_linear = 10 ** (snr_db / 10)
        signal_power = torch.mean(torch.abs(H) ** 2)
        noise_power = signal_power / snr_linear

        noise = torch.sqrt(noise_power / 2) * (
            torch.randn_like(H.real) + 1j * torch.randn_like(H.imag)
        )
        return H + noise

    def generate_pilot_mask(self, batch_size: int) -> torch.Tensor:
        mask = torch.zeros(batch_size, self.Nt, dtype=torch.bool)
        pilot_indices = torch.randperm(self.Nt)[:self.Np]
        mask[:, pilot_indices] = True
        return mask

    def extract_pilot_observations(
        self,
        H: torch.Tensor,
        pilot_mask: torch.Tensor
    ) -> torch.Tensor:
        H_observed = torch.zeros_like(H)
        mask_expanded = pilot_mask.unsqueeze(1).expand_as(H)
        H_observed[mask_expanded] = H[mask_expanded]
        return H_observed

    def generate_dataset(self, num_samples: int) -> Dict[str, torch.Tensor]:
        H_true = self.generate_channel(num_samples)
        pilot_mask = self.generate_pilot_mask(num_samples)
        H_noisy = self.add_noise(H_true)
        H_pilot_obs = self.extract_pilot_observations(H_noisy, pilot_mask)

        return {
            'H_true': H_true,
            'H_noisy': H_noisy,
            'H_pilot_obs': H_pilot_obs,
            'pilot_mask': pilot_mask
        }

    def generate_train_val_test(
        self,
        num_train: int = 8000,
        num_val: int = 1000,
        num_test: int = 1000
    ) -> Tuple[Dict, Dict, Dict]:
        train_data = self.generate_dataset(num_train)
        val_data = self.generate_dataset(num_val)
        test_data = self.generate_dataset(num_test)
        return train_data, val_data, test_data
