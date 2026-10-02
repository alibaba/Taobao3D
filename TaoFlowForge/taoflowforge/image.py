"""Shared image loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

ImageInput = Union[str, Path, Image.Image]


def composite_rgba_on_white(image: Image.Image) -> Image.Image:
    """Convert an image to RGB, compositing transparency over white."""
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def pad_to_square(image: Image.Image) -> Image.Image:
    """Center an RGB image on a square white canvas."""
    width, height = image.size
    if width == height:
        return image
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas


def prepare_image(image: ImageInput, size: int = 1_024) -> torch.Tensor:
    """Load an image as a float32 ``(3, size, size)`` tensor in ``[0, 1]``."""
    source = Image.open(image) if isinstance(image, (str, Path)) else image
    rgb = composite_rgba_on_white(source)
    square = pad_to_square(rgb)
    resized = square.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
