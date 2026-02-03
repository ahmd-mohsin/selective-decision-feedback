import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import numpy as np
from typing import Dict, Optional, Tuple

from diffusion.config import DiffusionConfig
from diffusion.model import UNet
from diffusion.scheduler import NoiseScheduler


class DiffusionInference:
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.config = checkpoint['config']
        
        self.model = UNet(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            cond_channels=self.config.cond_channels,
            model_channels=self.config.model_channels,
            channel_mult=self.config.channel_mult,
            num_res_blocks=self.config.num_res_blocks,
            attention_resolutions=self.config.attention_resolutions,
            dropout=0.0
        ).to(self.device)
        
        if 'ema_model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['ema_model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        
        self.scheduler = NoiseScheduler(
            num_steps=self.config.diffusion_steps,
            beta_start=self.config.beta_start,
            beta_end=self.config.beta_end,
            schedule_type=self.config.noise_schedule_type,
            device=self.device
        )
    
    @torch.no_grad()
    def reconstruct_channel(
        self,
        H_pilot_full: torch.Tensor,
        pilot_mask: torch.Tensor,
        Y_grid: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        if num_inference_steps is None:
            num_inference_steps = self.config.diffusion_steps
        
        if H_pilot_full.device != self.device:
            H_pilot_full = H_pilot_full.to(self.device)
            pilot_mask = pilot_mask.to(self.device)
            Y_grid = Y_grid.to(self.device)
        
        if H_pilot_full.dim() == 3:
            H_pilot_full = H_pilot_full.unsqueeze(0)
            pilot_mask = pilot_mask.unsqueeze(0)
            Y_grid = Y_grid.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        cond = torch.cat([H_pilot_full, pilot_mask, Y_grid], dim=1)
        
        shape = (H_pilot_full.shape[0], 2, H_pilot_full.shape[2], H_pilot_full.shape[3])
        
        if return_intermediates:
            H_recon, intermediates = self.scheduler.p_sample_loop(
                self.model,
                shape,
                cond,
                clip_denoised=True,
                return_intermediates=True
            )
        else:
            H_recon = self.scheduler.p_sample_loop(
                self.model,
                shape,
                cond,
                clip_denoised=True
            )
        
        if squeeze_output:
            H_recon = H_recon.squeeze(0)
        
        if return_intermediates:
            return H_recon, intermediates
        return H_recon
    
    @torch.no_grad()
    def reconstruct_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        H_pilot_full = batch['H_pilot_full'].to(self.device)
        pilot_mask = batch['pilot_mask'].to(self.device)
        Y_grid = batch['Y_grid'].to(self.device)
        
        return self.reconstruct_channel(H_pilot_full, pilot_mask, Y_grid)
    
    @torch.no_grad()
    def compute_nmse(
        self,
        H_pred: torch.Tensor,
        H_true: torch.Tensor
    ) -> float:
        error = torch.sum((H_pred - H_true) ** 2)
        signal = torch.sum(H_true ** 2)
        return (error / (signal + 1e-8)).item()
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataset,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        from torch.utils.data import DataLoader
        
        if num_samples is None:
            num_samples = len(dataset)
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        total_nmse = 0.0
        pilot_nmse = 0.0
        count = 0
        
        from tqdm import tqdm
        
        for i, batch in enumerate(tqdm(loader, total=num_samples, desc="Evaluating")):
            if i >= num_samples:
                break
            
            H_true = batch['H_true'].to(self.device)
            H_pilot = batch['H_pilot_full'].to(self.device)
            
            H_pred = self.reconstruct_batch(batch)
            
            total_nmse += self.compute_nmse(H_pred, H_true)
            pilot_nmse += self.compute_nmse(H_pilot, H_true)
            count += 1
        
        metrics = {
            'diffusion_nmse': total_nmse / count,
            'diffusion_nmse_db': 10 * np.log10(total_nmse / count),
            'pilot_nmse': pilot_nmse / count,
            'pilot_nmse_db': 10 * np.log10(pilot_nmse / count),
            'improvement_db': 10 * np.log10(pilot_nmse / total_nmse)
        }
        
        return metrics
    
    def channels_to_complex(self, H_channels: torch.Tensor) -> torch.Tensor:
        real = H_channels[:, 0, :, :] if H_channels.dim() == 4 else H_channels[0, :, :]
        imag = H_channels[:, 1, :, :] if H_channels.dim() == 4 else H_channels[1, :, :]
        return torch.complex(real, imag)
    
    def complex_to_channels(self, H_complex: torch.Tensor) -> torch.Tensor:
        real = H_complex.real
        imag = H_complex.imag
        return torch.stack([real, imag], dim=1 if H_complex.dim() == 3 else 0)