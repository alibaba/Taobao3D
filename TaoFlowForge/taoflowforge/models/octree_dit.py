"""Inference-only raw-space octree DiT used by Stage 1."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .attention import flash_attn_qkv


def cell_centers(coordinates: torch.Tensor, level: int) -> torch.Tensor:
    """Convert integer octree cells to centers in ``[-1, 1]``."""
    resolution = float(2**level)
    return (coordinates.float() + 0.5) / resolution * 2.0 - 1.0


def build_octree_rope(
    positions: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the checkpoint's normalized-coordinate 3D RoPE tables."""
    dimensions_per_axis = (head_dim // 3) // 2 * 2
    half_dim = dimensions_per_axis // 2
    inverse_frequency = 1.0 / (
        10_000.0
        ** (
            torch.arange(
                half_dim,
                device=positions.device,
                dtype=torch.float32,
            )
            / half_dim
        )
    )
    theta = positions.float().unsqueeze(-1) * inverse_frequency
    theta = theta * math.pi
    cosine = theta.cos()
    sine = theta.sin()
    cosine = torch.cat([cosine, cosine], dim=-1).flatten(-2).unsqueeze(1)
    sine = torch.cat([-sine, sine], dim=-1).flatten(-2).unsqueeze(1)
    return cosine, sine


def apply_octree_rope(
    tensor: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply cached 3D RoPE with the legacy float32 arithmetic order."""
    rope_dim = cosine.shape[-1]
    rotated_input = tensor[..., :rope_dim].float()
    first, second = rotated_input.unflatten(-1, (3, 2, -1)).unbind(-2)
    swapped = torch.cat([second, first], dim=-1).flatten(-2)
    result = (rotated_input * cosine + swapped * sine).to(tensor.dtype)
    if tensor.shape[-1] > rope_dim:
        result = torch.cat([result, tensor[..., rope_dim:]], dim=-1)
    return result


class MultiHeadRMSNorm(nn.Module):
    """Per-head RMS normalization used by the release checkpoint."""

    def __init__(self, head_dim: int, num_heads: int):
        super().__init__()
        self.scale = head_dim**0.5
        self.gamma = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(tensor.float(), dim=-1)
        return (normalized * self.gamma * self.scale).to(tensor.dtype)


class FeedForwardNet(nn.Module):
    def __init__(
        self,
        channels: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_channels, channels),
        )
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.out_drop(self.mlp(tensor))


class OctreeSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_qkv = nn.Linear(channels, channels * 3, bias=True)
        self.to_out = nn.Linear(channels, channels)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(
        self,
        tensor: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        qkv = self.to_qkv(tensor).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = self.q_rms_norm(query).transpose(1, 2)
        key = self.k_rms_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query = apply_octree_rope(query, rope[0], rope[1])
        key = apply_octree_rope(key, rope[0], rope[1])
        hidden = flash_attn_qkv(query, key, value)
        hidden = hidden.transpose(1, 2).reshape(batch, tokens, channels)
        return self.out_drop(self.to_out(hidden))


class OctreeCrossAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.to_q = nn.Linear(channels, channels, bias=True)
        self.to_kv = nn.Linear(context_channels, channels * 2, bias=True)
        self.to_out = nn.Linear(channels, channels)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)

    def forward(
        self,
        tensor: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch, tokens, channels = tensor.shape
        context_tokens = context.shape[1]
        query = self.to_q(tensor).reshape(
            batch,
            tokens,
            self.num_heads,
            self.head_dim,
        )
        key_value = self.to_kv(context).reshape(
            batch,
            context_tokens,
            2,
            self.num_heads,
            self.head_dim,
        )
        key, value = key_value.unbind(dim=2)
        query = self.q_rms_norm(query).transpose(1, 2)
        key = self.k_rms_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        hidden = flash_attn_qkv(query, key, value)
        hidden = hidden.transpose(1, 2).reshape(batch, tokens, channels)
        return self.out_drop(self.to_out(hidden))


class OctreeDiTBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        context_channels: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(6 * channels) / channels**0.5)
        self.norm1 = nn.LayerNorm(channels, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(channels, elementwise_affine=False)
        self.self_attn = OctreeSelfAttention(channels, num_heads, dropout)
        self.cross_attn = OctreeCrossAttention(
            channels,
            context_channels,
            num_heads,
            dropout,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio, dropout)

    def forward(
        self,
        tensor: torch.Tensor,
        shared_modulation: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
    ) -> torch.Tensor:
        modulation = (self.modulation + shared_modulation).to(
            shared_modulation.dtype
        )
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation.chunk(6, dim=-1)

        hidden = self.norm1(tensor)
        hidden = hidden * (1 + scale_attention.unsqueeze(1))
        hidden = hidden + shift_attention.unsqueeze(1)
        hidden = self.self_attn(hidden, rope)
        tensor = tensor + hidden * gate_attention.unsqueeze(1)

        tensor = tensor + self.cross_attn(self.norm2(tensor), context)

        hidden = self.norm3(tensor)
        hidden = hidden * (1 + scale_mlp.unsqueeze(1))
        hidden = hidden + shift_mlp.unsqueeze(1)
        hidden = self.mlp(hidden)
        return tensor + hidden * gate_mlp.unsqueeze(1)


class RawOctreeDenoiser(nn.Module):
    """Fixed level expert for raw 8-child occupancy prediction."""

    def __init__(
        self,
        hidden_dim: int = 1_536,
        num_heads: int = 12,
        num_blocks: int = 28,
        mlp_ratio: float = 5.3334,
        context_dim: int = 1_024,
        num_levels: int = 9,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.input_layer = nn.Linear(8, hidden_dim)
        self.t_embedder = nn.Module()
        self.t_embedder.frequency_embedding_size = 256
        self.t_embedder.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6, bias=True),
        )
        self.depth_emb = nn.Embedding(num_levels, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                OctreeDiTBlock(
                    hidden_dim,
                    context_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(hidden_dim, 8)

    @staticmethod
    def _timestep_embedding(
        timestep: torch.Tensor,
        dimensions: int = 256,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        half = dimensions // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(
                half,
                dtype=torch.float32,
                device=timestep.device,
            )
            / half
        )
        arguments = timestep[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)

    def forward(
        self,
        noisy_occupancy: torch.Tensor,
        positions: torch.Tensor,
        level: int | torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        batch = noisy_occupancy.shape[0]
        hidden = self.input_layer(noisy_occupancy)
        if isinstance(level, int):
            level_ids = torch.full(
                (batch,),
                level,
                device=noisy_occupancy.device,
                dtype=torch.long,
            )
        else:
            level_ids = level
        depth_embedding = self.depth_emb(level_ids)
        hidden = hidden + depth_embedding.unsqueeze(1)

        frequency = self._timestep_embedding(timestep * 1_000.0)
        time_embedding = self.t_embedder.mlp(frequency)
        shared_modulation = self.adaLN_modulation(
            torch.cat([time_embedding, depth_embedding], dim=-1)
        )
        rope = build_octree_rope(positions, self.hidden_dim // self.num_heads)
        for block in self.blocks:
            hidden = block(hidden, shared_modulation, rope, context)
        hidden = F.layer_norm(hidden, (self.hidden_dim,))
        return self.out_layer(hidden)
