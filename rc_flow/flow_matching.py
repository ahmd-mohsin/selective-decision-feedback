import torch
import torch.nn as nn
from typing import Optional
from .network import ConditionalFlowNetwork
from .config import RCFlowConfig


class FlowMatching(nn.Module):
    def __init__(self, config: RCFlowConfig):
        super().__init__()
        self.config = config
        self.network = ConditionalFlowNetwork(
            input_dim=config.channel_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=8,
            mult=4,
            dropout=0.1
        )

    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        return (1 - t) * x_0 + t * x_1

    def training_step(
        self,
        H_true: torch.Tensor,
        H_obs: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = H_true.shape[0]
        device = H_true.device

        x_1 = H_true
        x_0 = torch.randn_like(x_1)

        t = torch.rand(batch_size, device=device)
        x_t = self.interpolate(x_0, x_1, t)

        target_velocity = x_1 - x_0

        predicted_velocity = self.network(x_t, t, H_obs, mask)

        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        device: torch.device,
        obs: torch.Tensor = None,
        mask: torch.Tensor = None,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.config.num_flow_steps

        x = torch.randn(batch_size, self.config.channel_dim, device=device)

        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            velocity = self.network(x, t, obs, mask)
            x = x + velocity * dt

            if obs is not None and mask is not None:
                x = torch.where(mask.bool(), obs, x)

        return x
