import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimeEmbedding(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResidualBlock(nn.Module):
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        if H * W > 1024:
            return x
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W).transpose(2, 3)
        
        scale = (C // self.num_heads) ** -0.5
        
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False
        ) if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else self._manual_attention(q, k, v, scale)
        
        out = attn.transpose(2, 3).contiguous().view(B, C, H, W)
        
        out = self.proj(out)
        
        return x + out
    
    def _manual_attention(self, q, k, v, scale):
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class Downsample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        cond_channels: int = 5,
        model_channels: int = 128,
        channel_mult: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (16, 8),
        dropout: float = 0.1,
        time_embed_dim: Optional[int] = None
    ):
        super().__init__()
        
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        
        if time_embed_dim is None:
            time_embed_dim = model_channels * 4
        
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        self.input_conv = nn.Conv2d(in_channels + cond_channels, model_channels, 3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        channels_list = [model_channels]
        now_channels = model_channels
        
        for level, mult in enumerate(channel_mult):
            out_channels_level = model_channels * mult
            
            for block_idx in range(num_res_blocks):
                layers = [ResidualBlock(now_channels, out_channels_level, time_embed_dim, dropout)]
                now_channels = out_channels_level
                
                if out_channels_level in [model_channels * m for m in attention_resolutions]:
                    layers.append(AttentionBlock(now_channels))
                
                self.down_blocks.append(nn.ModuleList(layers))
                channels_list.append(now_channels)
            
            if level != len(channel_mult) - 1:
                self.down_samples.append(Downsample(now_channels))
                channels_list.append(now_channels)
        
        self.middle_block1 = ResidualBlock(now_channels, now_channels, time_embed_dim, dropout)
        self.middle_attn = AttentionBlock(now_channels)
        self.middle_block2 = ResidualBlock(now_channels, now_channels, time_embed_dim, dropout)
        
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_channels_level = model_channels * mult
            
            for block_idx in range(num_res_blocks + 1):
                layers = [ResidualBlock(
                    now_channels + channels_list.pop(),
                    out_channels_level,
                    time_embed_dim,
                    dropout
                )]
                now_channels = out_channels_level
                
                if out_channels_level in [model_channels * m for m in attention_resolutions]:
                    layers.append(AttentionBlock(now_channels))
                
                self.up_blocks.append(nn.ModuleList(layers))
            
            if level != 0:
                self.up_samples.append(Upsample(now_channels))
        
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        x_cond = torch.cat([x, cond], dim=1)
        h = self.input_conv(x_cond)
        
        hs = [h]
        
        for level, mult in enumerate(self.channel_mult):
            for block_modules in self.down_blocks[level * self.num_res_blocks:(level + 1) * self.num_res_blocks]:
                for layer in block_modules:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                hs.append(h)
            
            if level != len(self.channel_mult) - 1:
                h = self.down_samples[level](h)
                hs.append(h)
        
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        up_block_idx = 0
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            for i in range(self.num_res_blocks + 1):
                h = torch.cat([h, hs.pop()], dim=1)
                
                for layer in self.up_blocks[up_block_idx]:
                    if isinstance(layer, ResidualBlock):
                        h = layer(h, t_emb)
                    else:
                        h = layer(h)
                
                up_block_idx += 1
            
            if level != 0:
                h = self.up_samples[len(self.channel_mult) - 1 - level](h)
        
        return self.output_conv(h)