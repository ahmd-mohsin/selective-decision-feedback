import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm).reshape(B, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).reshape(B, D)
        return x + self.proj(out)


class DenseResidualBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * mult

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim * 2)
        )

        self.mlp1 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.mlp1(h)
        x = x + h

        h = self.norm2(x)
        h = self.mlp2(h)
        return x + h


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, time_dim: int, num_heads: int = 8, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = SelfAttention(dim, num_heads, dropout)
        self.res_block = DenseResidualBlock(dim, time_dim, mult, dropout)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.res_block(x, t_emb)
        return x


class FlowNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        time_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 3 == 2:
                self.layers.append(TransformerBlock(hidden_dim, time_dim, num_heads, mult, dropout))
            else:
                self.layers.append(DenseResidualBlock(hidden_dim, time_dim, mult, dropout))

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        h = self.input_proj(x)

        for layer in self.layers:
            h = layer(h, t_emb)

        h = self.output_norm(h)
        return self.output_proj(h)


class ConditionalFlowNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        time_dim = hidden_dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.input_proj = nn.Linear(input_dim * 3, hidden_dim)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 4 == 3:
                self.layers.append(TransformerBlock(hidden_dim, time_dim, num_heads, mult, dropout))
            else:
                self.layers.append(DenseResidualBlock(hidden_dim, time_dim, mult, dropout))

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)

        if obs is not None and mask is not None:
            mask_float = mask.float()
            h = torch.cat([x, obs * mask_float, mask_float], dim=-1)
        else:
            h = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)

        h = self.input_proj(h)

        for layer in self.layers:
            h = layer(h, t_emb)

        h = self.output_norm(h)
        return self.output_proj(h)
