import torch
import torch.nn as nn
from typing import Tuple
from .config import RCFlowConfig


class PhysicsProjector(nn.Module):
    def __init__(self, config: RCFlowConfig):
        super().__init__()
        self.config = config
        self.Nr = config.Nr
        self.Nt = config.Nt

    def complex_to_real(self, H_complex: torch.Tensor) -> torch.Tensor:
        H_real = torch.cat([H_complex.real, H_complex.imag], dim=-1)
        return H_real.view(H_complex.shape[0], -1)

    def real_to_complex(self, H_real: torch.Tensor) -> torch.Tensor:
        batch_size = H_real.shape[0]
        H_real = H_real.view(batch_size, self.Nr, self.Nt, 2)
        return torch.complex(H_real[..., 0], H_real[..., 1])

    def project(
        self,
        H_est: torch.Tensor,
        Y_pilot: torch.Tensor,
        P: torch.Tensor,
        pilot_mask: torch.Tensor,
        lambda_reg: float = None
    ) -> torch.Tensor:
        if lambda_reg is None:
            lambda_reg = self.config.lambda_proj

        batch_size = H_est.shape[0]

        H_projected = H_est.clone()

        for b in range(batch_size):
            pilot_indices = torch.where(pilot_mask[b])[0]
            if len(pilot_indices) == 0:
                continue

            P_sub = P[b, pilot_indices]
            Y_sub = Y_pilot[b, :, :len(pilot_indices)]

            P_dag = torch.linalg.pinv(P_sub.unsqueeze(0)).squeeze(0)
            H_ls = Y_sub @ P_dag

            H_projected[b] = (1 - lambda_reg) * H_est[b] + lambda_reg * H_ls

        return H_projected

    def project_simple(
        self,
        H_est: torch.Tensor,
        H_pilot_obs: torch.Tensor,
        pilot_mask: torch.Tensor,
        lambda_reg: float = None
    ) -> torch.Tensor:
        if lambda_reg is None:
            lambda_reg = self.config.lambda_proj

        H_projected = H_est.clone()

        mask_expanded = pilot_mask.unsqueeze(1).expand_as(H_est)

        H_projected = torch.where(
            mask_expanded,
            (1 - lambda_reg) * H_est + lambda_reg * H_pilot_obs,
            H_est
        )

        return H_projected
