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
            lr=config.learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs, eta_min=1e-6
        )

        self.best_val_nmse = float('inf')
        self.best_state = None
        self.patience_counter = 0

    def complex_to_real(self, H: torch.Tensor) -> torch.Tensor:
        return torch.cat([H.real, H.imag], dim=-1).reshape(H.shape[0], -1)

    def real_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.config.Nr, self.config.Nt, 2)
        return torch.complex(x[..., 0], x[..., 1])

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.flow.train()
        total_loss = 0.0

        for batch in dataloader:
            H_batch = batch[0].to(self.device)
            H_flat = self.complex_to_real(H_batch)

            self.optimizer.zero_grad()
            loss = self.flow.training_step(H_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_mask: Optional[torch.Tensor] = None,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 40
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
        self.best_val_nmse = float('inf')
        self.patience_counter = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            self.scheduler.step()

            if val_data is not None and (epoch + 1) % 5 == 0:
                val_nmse = self.evaluate(val_data, val_mask)
                history['val_nmse'].append(val_nmse)

                if val_nmse < self.best_val_nmse:
                    self.best_val_nmse = val_nmse
                    self.best_state = {k: v.cpu().clone() for k, v in self.flow.state_dict().items()}
                    self.patience_counter = 0
                    marker = " *"
                else:
                    self.patience_counter += 1
                    marker = ""

                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f} | Val NMSE: {val_nmse:.2f} dB | LR: {lr:.2e}{marker}")

                if self.patience_counter >= early_stopping_patience // 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f}")

        if self.best_state is not None:
            print(f"\nRestoring best model with Val NMSE: {self.best_val_nmse:.2f} dB")
            self.flow.load_state_dict(self.best_state)

        return history

    @torch.no_grad()
    def reconstruct(
        self,
        H_observed: torch.Tensor,
        pilot_mask: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.config.num_flow_steps

        self.flow.eval()
        device = self.device

        H_observed = H_observed.to(device)
        pilot_mask = pilot_mask.to(device)

        batch_size = H_observed.shape[0]

        x = torch.randn(batch_size, self.config.channel_dim, device=device)

        H_obs_flat = self.complex_to_real(H_observed)
        mask_flat = torch.cat([pilot_mask, pilot_mask], dim=-1).reshape(batch_size, -1)

        dt = 1.0 / num_steps
        lambda_g = self.config.lambda_proj

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            v = self.flow.network(x, t)
            x = x + v * dt

            strength = lambda_g * min(1.0, (i + 1) / (num_steps * 0.5))
            x = torch.where(mask_flat, (1 - strength) * x + strength * H_obs_flat, x)

        H_est = self.real_to_complex(x)
        H_est = torch.where(pilot_mask, H_observed, H_est)

        return H_est

    @torch.no_grad()
    def evaluate(
        self,
        H_true: torch.Tensor,
        pilot_mask: Optional[torch.Tensor] = None
    ) -> float:
        self.flow.eval()

        if pilot_mask is None:
            batch_size = H_true.shape[0]
            pilot_mask = torch.zeros(batch_size, self.config.Nr, self.config.Nt, dtype=torch.bool)
            for b in range(batch_size):
                pos = torch.randperm(self.config.total_elements)[:self.config.Np]
                rows, cols = pos // self.config.Nt, pos % self.config.Nt
                pilot_mask[b, rows, cols] = True

        H_noisy = self.add_noise(H_true)
        H_observed = torch.zeros_like(H_noisy)
        H_observed[pilot_mask] = H_noisy[pilot_mask]

        H_est = self.reconstruct(H_observed, pilot_mask)

        return self.compute_nmse(H_est, H_true.to(self.device))

    def add_noise(self, H: torch.Tensor) -> torch.Tensor:
        snr_linear = 10 ** (self.config.snr_db / 10)
        signal_power = torch.mean(torch.abs(H) ** 2)
        noise_power = signal_power / snr_linear
        noise = torch.sqrt(noise_power / 2) * (
            torch.randn_like(H.real) + 1j * torch.randn_like(H.imag)
        )
        return H + noise

    def compute_nmse(self, H_est: torch.Tensor, H_true: torch.Tensor) -> float:
        mse = torch.mean(torch.abs(H_est - H_true) ** 2)
        power = torch.mean(torch.abs(H_true) ** 2)
        nmse = mse / power
        return 10 * torch.log10(nmse).item()

    def save(self, path: str):
        torch.save({
            'flow_state_dict': self.flow.state_dict(),
            'config': self.config,
            'best_val_nmse': self.best_val_nmse
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.flow.load_state_dict(checkpoint['flow_state_dict'])
        self.best_val_nmse = checkpoint.get('best_val_nmse', float('inf'))
