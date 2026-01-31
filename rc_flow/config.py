from dataclasses import dataclass
import torch

@dataclass
class RCFlowConfig:
    Nr: int = 4
    Nt: int = 3
    Np: int = 6
    snr_db: float = 10.0

    hidden_dim: int = 512
    num_layers: int = 12

    num_flow_steps: int = 100

    learning_rate: float = 1e-3
    batch_size: int = 64
    num_epochs: int = 300

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def total_elements(self) -> int:
        return self.Nr * self.Nt

    @property
    def pilot_density(self) -> float:
        return self.Np / self.total_elements

    @property
    def channel_dim(self) -> int:
        return self.Nr * self.Nt * 2
