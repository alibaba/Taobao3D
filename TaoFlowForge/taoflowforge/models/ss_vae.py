"""Inference-only sparse-structure VAE decoder for Stage 0."""

from __future__ import annotations

from typing import Literal, Optional

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
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        output = super().forward(_manual_cast(inputs, torch.float32))
        return _manual_cast(output, dtype)


class GroupNorm32(nn.GroupNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dtype = inputs.dtype
        output = super().forward(_manual_cast(inputs, torch.float32))
        return _manual_cast(output, dtype)


class ChannelLayerNorm32(LayerNorm32):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        dimensions = inputs.dim()
        inputs = inputs.permute(0, *range(2, dimensions), 1).contiguous()
        inputs = super().forward(inputs)
        return inputs.permute(
            0, dimensions - 1, *range(1, dimensions - 1)
        ).contiguous()


def _normalization(
    normalization_type: str,
    channels: int,
) -> nn.Module:
    if normalization_type == "group":
        return GroupNorm32(32, channels)
    if normalization_type == "layer":
        return ChannelLayerNorm32(channels)
    raise ValueError(f"Unsupported normalization type: {normalization_type!r}")


def _zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def pixel_shuffle_3d(inputs: torch.Tensor, scale_factor: int) -> torch.Tensor:
    batch_size, channels, height, width, depth = inputs.shape
    output_channels = channels // scale_factor**3
    inputs = inputs.reshape(
        batch_size,
        output_channels,
        scale_factor,
        scale_factor,
        scale_factor,
        height,
        width,
        depth,
    )
    inputs = inputs.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return inputs.reshape(
        batch_size,
        output_channels,
        height * scale_factor,
        width * scale_factor,
        depth * scale_factor,
    )


class ResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        norm_type: Literal["group", "layer"] = "layer",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.norm1 = _normalization(norm_type, channels)
        self.norm2 = _normalization(norm_type, self.out_channels)
        self.conv1 = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        self.conv2 = _zero_module(
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1)
        )
        self.skip_connection = (
            nn.Conv3d(channels, self.out_channels, 1)
            if channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(inputs)
        hidden = F.silu(hidden)
        hidden = self.conv1(hidden)
        hidden = self.norm2(hidden)
        hidden = F.silu(hidden)
        hidden = self.conv2(hidden)
        return hidden + self.skip_connection(inputs)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_3d(self.conv(inputs), 2)


class SparseStructureDecoder(nn.Module):
    """Decode ``(B,8,16,16,16)`` latents to ``(B,1,64,64,64)`` logits."""

    def __init__(
        self,
        out_channels: int = 1,
        latent_channels: int = 8,
        num_res_blocks: int = 2,
        channels: tuple[int, ...] = (512, 128, 32),
        num_res_blocks_middle: int = 2,
        norm_type: Literal["group", "layer"] = "layer",
        use_fp16: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.num_res_blocks = num_res_blocks
        self.channels = channels
        self.num_res_blocks_middle = num_res_blocks_middle
        self.norm_type = norm_type
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(
            *[
                ResBlock3d(channels[0], channels[0], norm_type)
                for _ in range(num_res_blocks_middle)
            ]
        )
        self.blocks = nn.ModuleList()
        for index, channels_at_level in enumerate(channels):
            self.blocks.extend(
                [
                    ResBlock3d(channels_at_level, channels_at_level, norm_type)
                    for _ in range(num_res_blocks)
                ]
            )
            if index < len(channels) - 1:
                self.blocks.append(
                    UpsampleBlock3d(channels_at_level, channels[index + 1])
                )
        self.out_layer = nn.Sequential(
            _normalization(norm_type, channels[-1]),
            nn.SiLU(),
            nn.Conv3d(channels[-1], out_channels, 3, padding=1),
        )
        if use_fp16:
            self.blocks.apply(
                lambda module: _convert_module(module, torch.float16)
            )
            self.middle_block.apply(
                lambda module: _convert_module(module, torch.float16)
            )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(latent)
        hidden = hidden.to(self.dtype)
        hidden = self.middle_block(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = hidden.to(latent.dtype)
        return self.out_layer(hidden)
