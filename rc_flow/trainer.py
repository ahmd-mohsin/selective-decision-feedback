import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
from .flow_matching import FlowMatching
from .config import RCFlowConfig


class RCFlowTrainer:
    def __init__(self, config: RCFlowConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.flow = FlowMatching(config).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.flow.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=config.num_epochs * 100,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        self.best_val_nmse = float('inf')
        self.best_state = None

    def complex_to_real(self, H: torch.Tensor) -> torch.Tensor:
        return torch.cat([H.real, H.imag], dim=-1).reshape(H.shape[0], -1)

    def real_to_complex(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.reshape(batch_size, self.config.Nr, self.config.Nt, 2)
        return torch.complex(x[..., 0], x[..., 1])

    def train_epoch(
        self,
        H_true: torch.Tensor,
        pilot_mask: torch.Tensor
    ) -> float:
        self.flow.train()

        batch_size = self.config.batch_size
        num_samples = H_true.shape[0]
        indices = torch.randperm(num_samples)

        total_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]

            H_batch = H_true[batch_idx].to(self.device)
            mask_batch = pilot_mask[batch_idx].to(self.device)

            H_flat = self.complex_to_real(H_batch)

            H_noisy = self.add_noise(H_batch)
            H_obs = torch.zeros_like(H_noisy)
            H_obs[mask_batch] = H_noisy[mask_batch]
            H_obs_flat = self.complex_to_real(H_obs)

            mask_flat = torch.cat([mask_batch, mask_batch], dim=-1).reshape(H_batch.shape[0], -1).float()

            self.optimizer.zero_grad()
            loss = self.flow.training_step(H_flat, H_obs_flat, mask_flat)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def train(
        self,
        train_H: torch.Tensor,
        train_mask: torch.Tensor,
        val_H: torch.Tensor = None,
        val_mask: torch.Tensor = None,
        num_epochs: Optional[int] = None
    ) -> Dict[str, List[float]]:
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        total_steps = num_epochs * (len(train_H) // self.config.batch_size + 1)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )

        history = {'train_loss': [], 'val_nmse': []}
        self.best_val_nmse = float('inf')

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_H, train_mask)
            history['train_loss'].append(train_loss)

            if val_H is not None and (epoch + 1) % 5 == 0:
                val_nmse = self.evaluate(val_H, val_mask)
                history['val_nmse'].append(val_nmse)

                if val_nmse < self.best_val_nmse:
                    self.best_val_nmse = val_nmse
                    self.best_state = {k: v.cpu().clone() for k, v in self.flow.state_dict().items()}
                    marker = " *"
                else:
                    marker = ""

                lr = self.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f} | Val NMSE: {val_nmse:.2f} dB | LR: {lr:.2e}{marker}")
            elif (epoch + 1) % 10 == 0:
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

        H_observed = H_observed.to(self.device)
        pilot_mask = pilot_mask.to(self.device)

        batch_size = H_observed.shape[0]

        H_obs_flat = self.complex_to_real(H_observed)
        mask_flat = torch.cat([pilot_mask, pilot_mask], dim=-1).reshape(batch_size, -1).float()

        x = self.flow.sample(
            batch_size,
            self.device,
            obs=H_obs_flat,
            mask=mask_flat,
            num_steps=num_steps
        )

        H_est = self.real_to_complex(x)
        H_est = torch.where(pilot_mask, H_observed, H_est)

        return H_est

    @torch.no_grad()
    def evaluate(self, H_true: torch.Tensor, pilot_mask: torch.Tensor) -> float:
        self.flow.eval()

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
        return 10 * torch.log10(mse / power).item()

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
