import torch
import numpy as np
from typing import Tuple, Dict
from .config import RCFlowConfig


class ChannelDataset:
    def __init__(self, config: RCFlowConfig, num_angles: int = 8):
        self.config = config
        self.Nr = config.Nr
        self.Nt = config.Nt
        self.Np = config.Np
        self.num_angles = num_angles

        self.angle_grid_tx = torch.linspace(-np.pi/2, np.pi/2, num_angles)
        self.angle_grid_rx = torch.linspace(-np.pi/2, np.pi/2, num_angles)

        self.A_tx = self._create_steering_matrix(self.Nt, self.angle_grid_tx)
        self.A_rx = self._create_steering_matrix(self.Nr, self.angle_grid_rx)

    def _create_steering_matrix(self, num_antennas: int, angles: torch.Tensor) -> torch.Tensor:
        n = torch.arange(num_antennas).float()
        A = torch.exp(1j * np.pi * n.unsqueeze(1) * torch.sin(angles).unsqueeze(0))
        return A

    def generate_channel(self, batch_size: int, sparsity: int = 3) -> torch.Tensor:
        H = torch.zeros(batch_size, self.Nr, self.Nt, dtype=torch.complex64)

        for b in range(batch_size):
            tx_indices = torch.randperm(self.num_angles)[:sparsity]
            rx_indices = torch.randperm(self.num_angles)[:sparsity]

            for i in range(sparsity):
                gain = (torch.randn(1) + 1j * torch.randn(1)) / np.sqrt(2)
                a_tx = self.A_tx[:, tx_indices[i]]
                a_rx = self.A_rx[:, rx_indices[i]]
                H[b] += gain * torch.outer(a_rx, a_tx.conj())

            H[b] = H[b] / torch.sqrt(torch.mean(torch.abs(H[b]) ** 2))

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
        total_elements = self.Nr * self.Nt
        num_pilots = self.Np

        mask = torch.zeros(batch_size, self.Nr, self.Nt, dtype=torch.bool)

        for b in range(batch_size):
            pilot_positions = torch.randperm(total_elements)[:num_pilots]
            rows = pilot_positions // self.Nt
            cols = pilot_positions % self.Nt
            mask[b, rows, cols] = True

        return mask

    def create_observation(
        self,
        H_noisy: torch.Tensor,
        pilot_mask: torch.Tensor
    ) -> torch.Tensor:
        H_observed = torch.zeros_like(H_noisy)
        H_observed[pilot_mask] = H_noisy[pilot_mask]
        return H_observed

    def generate_dataset(self, num_samples: int) -> Dict[str, torch.Tensor]:
        H_true = self.generate_channel(num_samples)
        pilot_mask = self.generate_pilot_mask(num_samples)
        H_noisy = self.add_noise(H_true)
        H_observed = self.create_observation(H_noisy, pilot_mask)

        return {
            'H_true': H_true,
            'H_noisy': H_noisy,
            'H_observed': H_observed,
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
