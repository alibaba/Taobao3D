"""Inference-only Stage 2 DiT with voxel RoPE and adaLN-Zero."""

from __future__ import annotations

import math

import torch
from torch import nn

from .attention import (
    flash_attn_qkv,
    flash_attn_varlen_cross_qkv,
    flash_attn_varlen_qkv,
)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, max_period: int = 10_000):
        super().__init__()
        self.num_channels = num_channels
        self.max_period = max_period

    def forward(self, timesteps):
        half = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(
            half, dtype=torch.float32, device=timesteps.device
        ) / half
        embedding = timesteps[:, None].float() * torch.exp(exponent)[None, :]
        embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
        if self.num_channels % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=True),
        )
        self.time_embed = Timesteps(hidden_size)

    def forward(self, timesteps):
        frequency = self.time_embed(timesteps).type(self.mlp[0].weight.dtype)
        return self.mlp(frequency).unsqueeze(1)


class MLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qk_norm: bool,
        dtype=None,
    ):
        super().__init__()
        self.qdim = query_dim
        self.kdim = context_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.to_k = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        self.to_v = nn.Linear(context_dim, query_dim, bias=qkv_bias)
        norm_eps = 1.0 / 65_530 if dtype == torch.float16 else 1e-6
        norm = nn.RMSNorm
        self.q_norm = (
            norm(self.head_dim, elementwise_affine=True, eps=norm_eps)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            norm(self.head_dim, elementwise_affine=True, eps=norm_eps)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(query_dim, query_dim, bias=True)

    def forward(self, x, context):
        batch, query_tokens, _ = x.shape
        context_tokens = context.shape[1]
        q = self.to_q(x).view(
            batch, query_tokens, self.num_heads, self.head_dim
        )
        k = self.to_k(context)
        v = self.to_v(context)
        # Preserve the historical K/V packing used to train the release
        # checkpoint. It concatenates projected channels before splitting each
        # head into K and V halves; standard independent reshapes are not
        # numerically compatible with those learned weights.
        packed_kv = torch.cat((k, v), dim=-1).view(
            1,
            -1,
            self.num_heads,
            self.head_dim * 2,
        )
        k, v = torch.split(packed_kv, self.head_dim, dim=-1)
        k = k.view(batch, context_tokens, self.num_heads, self.head_dim)
        v = v.view(batch, context_tokens, self.num_heads, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        valid_mask = (context != -1).any(dim=-1)
        if bool(valid_mask.all()):
            output = flash_attn_qkv(q, k, v)
        else:
            output = flash_attn_varlen_cross_qkv(q, k, v, valid_mask)
        output = output.transpose(1, 2).reshape(batch, query_tokens, -1)
        return self.out_proj(output)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qk_norm: bool,
        dtype=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        norm_eps = 1.0 / 65_530 if dtype == torch.float16 else 1e-6
        self.q_norm = (
            nn.RMSNorm(self.head_dim, elementwise_affine=True, eps=norm_eps)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            nn.RMSNorm(self.head_dim, elementwise_affine=True, eps=norm_eps)
            if qk_norm
            else nn.Identity()
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, rotary_cos, rotary_sin, sequence_mask=None):
        batch, tokens, _ = x.shape
        q = self.to_q(x).reshape(
            batch, tokens, self.num_heads, self.head_dim
        )
        k = self.to_k(x).reshape(
            batch, tokens, self.num_heads, self.head_dim
        )
        v = self.to_v(x).reshape(
            batch, tokens, self.num_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = apply_rotary_embedding(q, rotary_cos, rotary_sin)
        k = apply_rotary_embedding(k, rotary_cos, rotary_sin)
        if sequence_mask is None:
            output = flash_attn_qkv(q, k, v)
        else:
            output = flash_attn_varlen_qkv(q, k, v, sequence_mask)
        output = output.transpose(1, 2).reshape(batch, tokens, -1)
        return self.out_proj(output)


class DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        context_dim: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qk_norm: bool,
        dtype=None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn1 = Attention(
            hidden_size,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            dtype=dtype,
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.attn2 = CrossAttention(
            hidden_size,
            context_dim,
            num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            dtype=dtype,
        )
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.mlp = MLP(width=hidden_size)

    def forward(
        self,
        x,
        condition,
        context,
        rotary_cos,
        rotary_sin,
        sequence_mask=None,
    ):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        hidden = _modulate(self.norm1(x), shift_msa, scale_msa)
        hidden = self.attn1(
            hidden,
            rotary_cos,
            rotary_sin,
            sequence_mask,
        )
        x = x + hidden * gate_msa.unsqueeze(1)
        x = x + self.attn2(self.norm2(x), context)
        hidden = _modulate(self.norm3(x), shift_mlp, scale_mlp)
        hidden = self.mlp(hidden)
        return x + hidden * gate_mlp.unsqueeze(1)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_channels: int):
        super().__init__()
        self.final_hidden_size = hidden_size
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6
        )
        self.linear = nn.Linear(hidden_size, output_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm_final(x))


class RefineDiT(nn.Module):
    """Fixed Stage 2 latent velocity model."""

    def __init__(
        self,
        *,
        in_channels: int = 32,
        hidden_size: int = 1_536,
        context_dim: int = 1_024,
        depth: int = 28,
        num_heads: int = 24,
        qk_norm: bool = True,
        qkv_bias: bool = False,
        dtype=None,
    ):
        super().__init__()
        self.t_encoding = "adaLN_zero"
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.pos_embed_type = "voxel_rope"
        self.x_embedder = nn.Linear(in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    context_dim,
                    num_heads,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )
        self.depth = depth
        self.final_layer = FinalLayer(hidden_size, in_channels)

    def forward(self, x, timesteps, contexts, *, voxel_cond, **_unused):
        context = contexts["main"]
        condition = self.t_embedder(timesteps).squeeze(1)
        x = self.x_embedder(x)
        head_dim = self.blocks[0].attn1.head_dim
        rotary_cos, rotary_sin = precompute_voxel_rope(head_dim, voxel_cond)
        for block in self.blocks:
            x = block(
                x,
                condition,
                context,
                rotary_cos,
                rotary_sin,
            )
        return self.final_layer(x)


def apply_rotary_embedding(x, cosine, sine):
    """Apply float32 rotary arithmetic, then restore the input dtype."""
    first, second = x.float().chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    cosine = cosine.unsqueeze(1).float()
    sine = sine.unsqueeze(1).float()
    return (x.float() * cosine + rotated * sine).to(x.dtype)


def precompute_voxel_rope(
    dim: int,
    voxel_coordinates: torch.Tensor,
    theta: float = 10_000.0,
):
    """Build the exact three-axis RoPE used by the Stage 2 checkpoint."""
    if dim % 2:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    half = dim // 2
    pairs_x = half // 3
    pairs_y = half // 3
    pairs_z = half - pairs_x - pairs_y
    device = voxel_coordinates.device

    def frequencies(count):
        return 1.0 / (
            theta
            ** (
                torch.arange(count, device=device).float()
                / max(count, 1)
            )
        )

    coordinates = voxel_coordinates.float()
    arguments = torch.cat(
        [
            coordinates[..., 0:1] * frequencies(pairs_x),
            coordinates[..., 1:2] * frequencies(pairs_y),
            coordinates[..., 2:3] * frequencies(pairs_z),
        ],
        dim=-1,
    )
    arguments = torch.cat([arguments, arguments], dim=-1)
    return torch.cos(arguments), torch.sin(arguments)
