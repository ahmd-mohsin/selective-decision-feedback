import torch
import numpy as np
from typing import Optional, Tuple


class NoiseScheduler:
    
    def __init__(
        self,
        num_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule_type: str = 'linear',
        device: str = 'cuda'
    ):
        self.num_steps = num_steps
        self.device = device
        
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_steps, device=device)
        elif schedule_type == 'quadratic':
            self.betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_steps, device=device) ** 2
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]])
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008, device: str = 'cuda') -> torch.Tensor:
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_noise = model(x_t, t, cond)
        
        x_recon = self._predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1.0, 1.0)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(
            x_recon, x_t, t
        )
        
        return model_mean, posterior_variance, posterior_log_variance
    
    def _predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            self._extract(self.sqrt_recip_alphas, t, x_t.shape) * x_t -
            self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    @torch.no_grad()
    def p_sample(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        model_mean, _, model_log_variance = self.p_mean_variance(
            model, x_t, t, cond, clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))
        
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        shape: tuple,
        cond: torch.Tensor,
        clip_denoised: bool = True,
        return_intermediates: bool = False
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        x_t = torch.randn(shape, device=device)
        
        intermediates = [] if return_intermediates else None
        
        for i in reversed(range(self.num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, cond, clip_denoised)
            
            if return_intermediates:
                intermediates.append(x_t)
        
        if return_intermediates:
            return x_t, intermediates
        return x_t
    
    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))