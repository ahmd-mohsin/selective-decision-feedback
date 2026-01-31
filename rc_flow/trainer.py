import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional
from .flow_matching import FlowMatching
from .projector import PhysicsProjector
from .config import RCFlowConfig


class RCFlowTrainer:
    def __init__(self, config: RCFlowConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.flow = FlowMatching(config).to(self.device)
        self.projector = PhysicsProjector(config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.flow.parameters(),
            lr=config.learning_rate
        )

    def prepare_data(self, H_true: torch.Tensor) -> torch.Tensor:
        H_real = torch.cat([H_true.real, H_true.imag], dim=-1)
        return H_real.view(H_true.shape[0], -1)

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.flow.train()
        total_loss = 0.0

        for batch in dataloader:
            H_batch = batch[0].to(self.device)
            H_flat = self.prepare_data(H_batch)

            self.optimizer.zero_grad()
            loss = self.flow.training_step(H_flat)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        train_dataset = TensorDataset(train_data)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'val_nmse': []}

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            if val_data is not None and (epoch + 1) % 10 == 0:
                val_nmse = self.evaluate(val_data)
                history['val_nmse'].append(val_nmse)
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f} | Val NMSE: {val_nmse:.4f} dB")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f}")

        return history

    @torch.no_grad()
    def reconstruct(
        self,
        H_noisy: torch.Tensor,
        pilot_mask: torch.Tensor,
        num_iterations: Optional[int] = None
    ) -> torch.Tensor:
        if num_iterations is None:
            num_iterations = self.config.num_outer_iterations

        self.flow.eval()
        device = self.device

        H_noisy = H_noisy.to(device)
        pilot_mask = pilot_mask.to(device)

        H_noisy_flat = self.prepare_data(H_noisy)

        H_anchor = H_noisy_flat.clone()

        for outer_iter in range(num_iterations):
            H_refined = self.flow.denoise(H_anchor, start_t=0.3)

            H_complex = self.projector.real_to_complex(H_refined)
            H_noisy_complex = self.projector.real_to_complex(H_noisy_flat)

            H_projected = self.projector.project_simple(
                H_complex, H_noisy_complex, pilot_mask
            )

            H_anchor = self.prepare_data(H_projected)

            beta = self.config.beta_anchor
            H_anchor = (1 - beta) * H_anchor + beta * H_refined

        return self.projector.real_to_complex(H_anchor)

    @torch.no_grad()
    def evaluate(self, H_true: torch.Tensor, pilot_mask: Optional[torch.Tensor] = None) -> float:
        self.flow.eval()

        if pilot_mask is None:
            pilot_mask = torch.zeros(H_true.shape[0], self.config.Nt, dtype=torch.bool)
            pilot_indices = torch.randperm(self.config.Nt)[:self.config.Np]
            pilot_mask[:, pilot_indices] = True

        H_noisy = self.add_noise(H_true)
        H_reconstructed = self.reconstruct(H_noisy, pilot_mask)

        H_true_device = H_true.to(self.device)
        nmse = self.compute_nmse(H_reconstructed, H_true_device)
        return nmse

    def add_noise(self, H: torch.Tensor) -> torch.Tensor:
        snr_linear = 10 ** (self.config.snr_db / 10)
        signal_power = torch.mean(torch.abs(H) ** 2)
        noise_power = signal_power / snr_linear
        noise = torch.sqrt(noise_power / 2) * (
            torch.randn_like(H.real) + 1j * torch.randn_like(H.imag)
        )
        return H + noise

    def compute_nmse(self, H_est: torch.Tensor, H_true: torch.Tensor) -> float:
        error = torch.abs(H_est - H_true) ** 2
        power = torch.abs(H_true) ** 2
        nmse = torch.mean(error) / torch.mean(power)
        return 10 * torch.log10(nmse).item()

    def save(self, path: str):
        torch.save({
            'flow_state_dict': self.flow.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.flow.load_state_dict(checkpoint['flow_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
