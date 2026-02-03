import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
from typing import Dict, Optional

from diffusion.config import DiffusionConfig
from diffusion.model import UNet
from diffusion.scheduler import NoiseScheduler
from diffusion.dataset import DiffusionDataset


class DiffusionTrainer:
    
    def __init__(
        self,
        config: DiffusionConfig,
        train_dataset: DiffusionDataset,
        val_dataset: Optional[DiffusionDataset] = None
    ):
        self.config = config
        self.device = torch.device(config.device)
        
        self.model = UNet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            cond_channels=config.cond_channels,
            model_channels=config.model_channels,
            channel_mult=config.channel_mult,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            dropout=config.dropout
        ).to(self.device)
        
        self.scheduler = NoiseScheduler(
            num_steps=config.diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            schedule_type=config.noise_schedule_type,
            device=self.device
        )
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        self.ema_model = self._create_ema_model()
        
        self.writer = SummaryWriter(config.log_dir)
        
        self.global_step = 0
        self.epoch = 0
        
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    def _create_ema_model(self):
        ema_model = UNet(
            in_channels=self.config.in_channels,
            out_channels=self.config.out_channels,
            cond_channels=self.config.cond_channels,
            model_channels=self.config.model_channels,
            channel_mult=self.config.channel_mult,
            num_res_blocks=self.config.num_res_blocks,
            attention_resolutions=self.config.attention_resolutions,
            dropout=self.config.dropout
        ).to(self.device)
        
        ema_model.load_state_dict(self.model.state_dict())
        ema_model.eval()
        
        for param in ema_model.parameters():
            param.requires_grad = False
        
        return ema_model
    
    def _update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.config.ema_decay).add_(
                    param.data, alpha=1 - self.config.ema_decay
                )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.model.train()
        
        H_true = batch['H_true'].to(self.device)
        cond = torch.cat([
            batch['H_pilot_full'].to(self.device),
            batch['pilot_mask'].to(self.device),
            batch['Y_grid'].to(self.device)
        ], dim=1)
        
        batch_size = H_true.shape[0]
        t = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=self.device)
        
        noise = torch.randn_like(H_true)
        H_noisy = self.scheduler.q_sample(H_true, t, noise)
        
        pred_noise = self.model(H_noisy, t, cond)
        
        loss = nn.functional.mse_loss(pred_noise, noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        self._update_ema()
        
        return loss.item()
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            H_true = batch['H_true'].to(self.device)
            cond = torch.cat([
                batch['H_pilot_full'].to(self.device),
                batch['pilot_mask'].to(self.device),
                batch['Y_grid'].to(self.device)
            ], dim=1)
            
            batch_size = H_true.shape[0]
            t = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=self.device)
            
            noise = torch.randn_like(H_true)
            H_noisy = self.scheduler.q_sample(H_true, t, noise)
            
            pred_noise = self.model(H_noisy, t, cond)
            loss = nn.functional.mse_loss(pred_noise, noise)
            
            total_loss += loss.item()
            num_batches += 1
        
        metrics = {
            'val_loss': total_loss / num_batches
        }
        
        return metrics
    
    def _compute_nmse(self, H_pred: torch.Tensor, H_true: torch.Tensor) -> float:
        error = torch.sum((H_pred - H_true) ** 2)
        signal = torch.sum(H_true ** 2)
        return (error / (signal + 1e-8)).item()
    
    def train(self):
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in pbar:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                self.global_step += 1
                
                if self.global_step % self.config.log_interval == 0:
                    self.writer.add_scalar('train/loss', loss, self.global_step)
                
                pbar.set_postfix({'loss': f'{loss:.6f}'})
            
            avg_train_loss = epoch_loss / num_batches
            
            val_metrics = self.validate()
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.6f}")
                
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)
            
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
        
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'ema_model_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch+1}, step {self.global_step}")