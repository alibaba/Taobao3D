"""Inference-only TRELLIS.2 sparse-structure flow transformer."""

from __future__ import annotations

import math
import os
from functools import partial
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


_MIXED_PRECISION_MODULES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def _convert_module(module: nn.Module, dtype: torch.dtype) -> None:
    if isinstance(module, _MIXED_PRECISION_MODULES):
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype)


def _manual_cast(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if not torch.is_autocast_enabled():
        return tensor.to(dtype=dtype)
    return tensor


class LayerNorm32(nn.LayerNorm):
    """Layer normalization computed in float32."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = super().forward(inputs.float())
        return _manual_cast(normalized, inputs.dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(inputs.float(), dim=-1)
        return (normalized * self.gamma * self.scale).to(inputs.dtype)


class RotaryPositionEmbedder(nn.Module):
    def __init__(
        self,
        head_dim: int,
        dim: int = 3,
        rope_freq: tuple[float, float] = (1.0, 10_000.0),
    ):
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be divisible by two")
        self.head_dim = head_dim
        self.dim = dim
        self.freq_dim = head_dim // 2 // dim
        frequencies = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = rope_freq[0] / rope_freq[1] ** frequencies

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        return torch.polar(torch.ones_like(phases), phases)

    @staticmethod
    def apply_rotary_embedding(
        inputs: torch.Tensor,
        phases: torch.Tensor,
    ) -> torch.Tensor:
        complex_inputs = torch.view_as_complex(
            inputs.float().reshape(*inputs.shape[:-1], -1, 2)
        )
        rotated = complex_inputs * phases.unsqueeze(-2)
        return torch.view_as_real(rotated).reshape(
            *rotated.shape[:-1], -1
        ).to(inputs.dtype)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} coordinate dimensions")
        phases = self._get_phases(indices.reshape(-1)).reshape(
            *indices.shape[:-1], -1
        )
        if phases.shape[-1] < self.head_dim // 2:
            padding = self.head_dim // 2 - phases.shape[-1]
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(
                            *phases.shape[:-1],
                            padding,
                            device=phases.device,
                        ),
                        torch.zeros(
                            *phases.shape[:-1],
                            padding,
                            device=phases.device,
                        ),
                    ),
                ],
                dim=-1,
            )
        return phases


def _naive_attention(q, k, v, attention_mask=None):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    weights = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if attention_mask is not None:
        weights = weights + attention_mask
    output = torch.softmax(weights, dim=-1) @ v
    return output.permute(0, 2, 1, 3)


def _attention(q, k, v, attention_mask=None):
    backend = os.environ.get("ATTN_BACKEND", "sdpa")
    if backend == "sdpa":
        output = F.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3),
            k.permute(0, 2, 1, 3),
            v.permute(0, 2, 1, 3),
            attn_mask=attention_mask,
        )
        return output.permute(0, 2, 1, 3)
    if backend == "naive":
        return _naive_attention(q, k, v, attention_mask)
    if backend == "xformers":
        import xformers.ops as xops

        return xops.memory_efficient_attention(q, k, v, attn_bias=attention_mask)
    if backend == "flash_attn":
        import flash_attn

        return flash_attn.flash_attn_func(q, k, v)
    if backend == "flash_attn_3":
        import flash_attn_interface

        return flash_attn_interface.flash_attn_func(q, k, v)
    raise ValueError(f"Unsupported attention backend: {backend!r}")


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: str = "self",
        attn_mode: str = "full",
        window_size=None,
        shift_window=None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10_000.0),
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        if channels % num_heads:
            raise ValueError("Attention channels must be divisible by num_heads")
        if type not in {"self", "cross"}:
            raise ValueError(f"Unsupported attention type: {type!r}")
        if attn_mode != "full":
            raise NotImplementedError("Only full attention is supported")
        self.channels = channels
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_window = shift_window
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.to_out = nn.Linear(channels, channels)

    def forward(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        phases: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, length, _ = inputs.shape
        if self._type == "self":
            qkv = self.to_qkv(inputs).reshape(
                batch_size, length, 3, self.num_heads, -1
            )
            q, k, v = qkv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
            if self.use_rope:
                if phases is None:
                    raise ValueError("RoPE phases are required")
                q = RotaryPositionEmbedder.apply_rotary_embedding(q, phases)
                k = RotaryPositionEmbedder.apply_rotary_embedding(k, phases)
        else:
            if context is None:
                raise ValueError("Cross-attention context is required")
            context_length = context.shape[1]
            q = self.to_q(inputs).reshape(
                batch_size, length, self.num_heads, -1
            )
            kv = self.to_kv(context).reshape(
                batch_size, context_length, 2, self.num_heads, -1
            )
            k, v = kv.unbind(dim=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
        hidden = _attention(q, k, v, cross_attn_mask)
        return self.to_out(hidden.reshape(batch_size, length, -1))


class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0):
        super().__init__()
        expanded_channels = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, expanded_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(expanded_channels, channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class ModulatedTransformerCrossBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        rope_freq: tuple[float, float] = (1.0, 10_000.0),
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads,
            type="self",
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            rope_freq=rope_freq,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if share_mod:
            self.modulation = nn.Parameter(
                torch.randn(6 * channels) / channels**0.5
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(
        self,
        inputs: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.share_mod:
            parameters = (self.modulation + modulation).to(modulation.dtype)
        else:
            parameters = self.adaLN_modulation(modulation)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            parameters.chunk(6, dim=1)
        )
        hidden = self.norm1(inputs)
        hidden = hidden * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        hidden = self.self_attn(hidden, phases=phases)
        inputs = inputs + hidden * gate_msa.unsqueeze(1)
        inputs = inputs + self.cross_attn(self.norm2(inputs), context)
        hidden = self.norm3(inputs)
        hidden = hidden * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        hidden = self.mlp(hidden)
        return inputs + hidden * gate_mlp.unsqueeze(1)

    def forward(
        self,
        inputs: torch.Tensor,
        modulation: torch.Tensor,
        context: torch.Tensor,
        phases: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                inputs,
                modulation,
                context,
                phases,
                use_reentrant=False,
            )
        return self._forward(inputs, modulation, context, phases)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(
        timestep: torch.Tensor,
        dim: int,
        max_period: int = 10_000,
    ) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -np.log(max_period)
            * torch.arange(0, half, dtype=torch.float32)
            / half
        ).to(timestep.device)
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat(
            [torch.cos(arguments), torch.sin(arguments)], dim=-1
        )
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        frequencies = self.timestep_embedding(
            timestep, self.frequency_embedding_size
        )
        return self.mlp(frequencies)


class SparseStructureFlowModel(nn.Module):
    """Dense 16-cube latent flow model used by Stage 0."""

    def __init__(
        self,
        resolution: int = 16,
        in_channels: int = 8,
        model_channels: int = 1_536,
        cond_channels: int = 1_024,
        out_channels: int = 8,
        num_blocks: int = 30,
        num_heads: int = 12,
        mlp_ratio: float = 5.3334,
        pe_mode: str = "rope",
        rope_freq: tuple[float, float] = (1.0, 10_000.0),
        dtype: str = "float32",
        use_checkpoint: bool = False,
        share_mod: bool = True,
        initialization: str = "scaled",
        qk_rms_norm: bool = True,
        qk_rms_norm_cross: bool = True,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.initialization = initialization
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[dtype]

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        if pe_mode != "rope":
            raise NotImplementedError("Stage 0 only supports 3D RoPE")
        position_embedder = RotaryPositionEmbedder(
            model_channels // num_heads, 3
        )
        coordinates = torch.meshgrid(
            *[
                torch.arange(resolution, device=self.device)
                for _ in range(3)
            ],
            indexing="ij",
        )
        coordinates = torch.stack(coordinates, dim=-1).reshape(-1, 3)
        self.register_buffer("rope_phases", position_embedder(coordinates))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList(
            [
                ModulatedTransformerCrossBlock(
                    model_channels,
                    cond_channels,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_checkpoint=use_checkpoint,
                    use_rope=True,
                    rope_freq=rope_freq,
                    share_mod=share_mod,
                    qk_rms_norm=qk_rms_norm,
                    qk_rms_norm_cross=qk_rms_norm_cross,
                )
                for _ in range(num_blocks)
            ]
        )
        self.out_layer = nn.Linear(model_channels, out_channels)
        self.initialize_weights()
        self.blocks.apply(partial(_convert_module, dtype=self.dtype))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        if self.initialization != "scaled":
            raise NotImplementedError("Stage 0 requires scaled initialization")

        def initialize_basic(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    std=np.sqrt(2.0 / (5.0 * self.model_channels)),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(initialize_basic)

        def initialize_scaled(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    std=1.0
                    / np.sqrt(5 * self.num_blocks * self.model_channels),
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        for block in self.blocks:
            block.self_attn.to_out.apply(initialize_scaled)
            block.cross_attn.to_out.apply(initialize_scaled)
            block.mlp.mlp[2].apply(initialize_scaled)
        nn.init.normal_(
            self.input_layer.weight, std=1.0 / np.sqrt(self.in_channels)
        )
        nn.init.zeros_(self.input_layer.bias)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.share_mod:
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            inputs.shape[0],
            self.in_channels,
            self.resolution,
            self.resolution,
            self.resolution,
        )
        if tuple(inputs.shape) != expected:
            raise ValueError(f"Expected latent shape {expected}, got {inputs.shape}")
        hidden = inputs.view(*inputs.shape[:2], -1).permute(0, 2, 1).contiguous()
        hidden = self.input_layer(hidden)
        time_embedding = self.t_embedder(timestep)
        if self.share_mod:
            time_embedding = self.adaLN_modulation(time_embedding)
        time_embedding = _manual_cast(time_embedding, self.dtype)
        hidden = _manual_cast(hidden, self.dtype)
        condition = _manual_cast(condition, self.dtype)
        for block in self.blocks:
            hidden = block(hidden, time_embedding, condition, self.rope_phases)
        hidden = _manual_cast(hidden, inputs.dtype)
        hidden = F.layer_norm(hidden, hidden.shape[-1:])
        hidden = self.out_layer(hidden)
        return hidden.permute(0, 2, 1).view(
            hidden.shape[0],
            hidden.shape[2],
            self.resolution,
            self.resolution,
            self.resolution,
        ).contiguous()
