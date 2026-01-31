import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple
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

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )

        self.best_val_nmse = float('inf')
        self.best_state = None
        self.patience_counter = 0

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

            torch.nn.utils.clip_grad_norm_(self.flow.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def train(
        self,
        train_data: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        num_epochs: Optional[int] = None,
        early_stopping_patience: int = 20
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

            if val_data is not None and (epoch + 1) % 5 == 0:
                val_nmse = self.evaluate(val_data)
                history['val_nmse'].append(val_nmse)

                self.scheduler.step(val_nmse)

                if val_nmse < self.best_val_nmse:
                    self.best_val_nmse = val_nmse
                    self.best_state = {
                        'flow': self.flow.state_dict(),
                        'optimizer': self.optimizer.state_dict()
                    }
                    self.patience_counter = 0
                    marker = " *"
                else:
                    self.patience_counter += 1
                    marker = ""

                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f} | Val NMSE: {val_nmse:.4f} dB{marker}")

                if self.patience_counter >= early_stopping_patience // 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.6f}")

        if self.best_state is not None:
            print(f"\nRestoring best model with Val NMSE: {self.best_val_nmse:.4f} dB")
            self.flow.load_state_dict(self.best_state['flow'])

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

        batch_size = H_noisy.shape[0]
        H_est = self.flow.sample(batch_size, device, num_steps=self.config.num_flow_steps)
        H_est_complex = self.projector.real_to_complex(H_est)

        H_noisy_complex = H_noisy

        for outer_iter in range(num_iterations):
            lambda_val = self.config.lambda_proj * (outer_iter + 1) / num_iterations

            H_projected = self.projector.project_simple(
                H_est_complex, H_noisy_complex, pilot_mask, lambda_reg=lambda_val
            )

            H_projected_flat = self.prepare_data(H_projected)
            H_refined = self.flow.denoise(H_projected_flat, start_t=0.2)
            H_est_complex = self.projector.real_to_complex(H_refined)

            beta = self.config.beta_anchor
            H_est_complex = (1 - beta) * H_est_complex + beta * H_noisy_complex
            mask_expanded = pilot_mask.unsqueeze(1).expand_as(H_est_complex)
            H_est_complex = torch.where(mask_expanded, H_noisy_complex, H_est_complex)

        return H_est_complex

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
            'config': self.config,
            'best_val_nmse': self.best_val_nmse
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.flow.load_state_dict(checkpoint['flow_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'best_val_nmse' in checkpoint:
            self.best_val_nmse = checkpoint['best_val_nmse']
