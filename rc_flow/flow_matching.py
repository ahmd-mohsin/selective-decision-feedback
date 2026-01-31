import torch
import torch.nn as nn
from typing import Tuple, Optional
from .network import FlowNetwork
from .config import RCFlowConfig


class FlowMatching(nn.Module):
    def __init__(self, config: RCFlowConfig):
        super().__init__()
        self.config = config
        self.network = FlowNetwork(
            input_dim=config.channel_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers
        )

    def get_velocity(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return x_1 - x_0

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        return (1 - t) * x_0 + t * x_1

    def training_step(self, H_true: torch.Tensor) -> torch.Tensor:
        batch_size = H_true.shape[0]
        device = H_true.device

        x_1 = H_true
        x_0 = torch.randn_like(x_1)

        t = torch.rand(batch_size, device=device)

        x_t = self.interpolate(x_0, x_1, t)
        target_velocity = self.get_velocity(x_0, x_1, t)

        predicted_velocity = self.network(x_t, t)

        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.config.num_flow_steps

        x = torch.randn(batch_size, self.config.channel_dim, device=device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            velocity = self.network(x, t)
            x = x + velocity * dt

        return x

    @torch.no_grad()
    def denoise(
        self,
        x_noisy: torch.Tensor,
        start_t: float = 0.5,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.config.num_flow_steps // 2

        device = x_noisy.device
        batch_size = x_noisy.shape[0]

        x = x_noisy
        dt = (1.0 - start_t) / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), start_t + i * dt, device=device)
            velocity = self.network(x, t)
            x = x + velocity * dt

        return x
