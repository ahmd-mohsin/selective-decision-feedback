from dataclasses import dataclass
import torch

@dataclass
class RCFlowConfig:
    Nr: int = 4
    Nt: int = 3
    Np: int = 2
    snr_db: float = 10.0

    hidden_dim: int = 128
    num_layers: int = 4

    num_flow_steps: int = 50
    num_outer_iterations: int = 3

    lambda_proj: float = 0.5
    beta_anchor: float = 0.3

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def pilot_density(self) -> float:
        return self.Np / self.Nt

    @property
    def channel_dim(self) -> int:
        return self.Nr * self.Nt * 2
