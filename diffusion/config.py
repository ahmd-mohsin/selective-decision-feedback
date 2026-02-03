import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class DiffusionConfig:
    diffusion_steps: int = 1000
    beta_schedule: str = 'linear'
    beta_start: float = 1e-4
    beta_end: float = 0.02
    
    model_channels: int = 64
    num_res_blocks: int = 2
    channel_mult: tuple = (1, 2, 4)
    attention_resolutions: tuple = (8,)
    dropout: float = 0.1
    
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    grad_clip: float = 1.0
    use_gradient_checkpointing: bool = False
    
    ema_decay: float = 0.9999
    log_interval: int = 100
    save_interval: int = 1000
    
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    checkpoint_dir: str = './checkpoints'
    log_dir: str = './logs'
    
    conditioning_mode: str = 'concat'
    use_pilot_mask: bool = True
    use_observations: bool = True
    
    noise_schedule_type: str = 'cosine'
    
    validation_samples: int = 16
    
    @property
    def in_channels(self) -> int:
        return 2
    
    @property
    def out_channels(self) -> int:
        return 2
    
    @property
    def cond_channels(self) -> int:
        base = 2
        if self.use_pilot_mask:
            base += 1
        if self.use_observations:
            base += 2
        return base