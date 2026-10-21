"""Shared image loading and preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image

ImageInput = Union[str, Path, Image.Image]

REMBG_DISABLED_WARNING = (
    "WARNING: No RMBG model supplied (--rembg-model / --rembg_model). "
    "Background removal is DISABLED; existing image preprocessing is unchanged. "
    "Real image backgrounds are retained."
)


class BackgroundRemover:
    """Local RMBG-2.0 / BiRefNet model, matching master's soft-mask preprocessing."""

    def __init__(self, model_path: str | Path, device: str | torch.device) -> None:
        model_path = Path(model_path)
        if not model_path.is_dir():
            raise FileNotFoundError(f"RMBG model directory does not exist: {model_path}")
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        print(f"Loading background-removal model (RMBG-2.0/BiRefNet): {model_path}", flush=True)
        self.model = AutoModelForImageSegmentation.from_pretrained(
            str(model_path), trust_remote_code=True, local_files_only=True,
        )
        torch.set_float32_matmul_precision("high")
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(device=device, dtype=torch.float32)
        self.transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"Background-removal model loaded (device={device}, dtype=float32).", flush=True)
        if torch.device(device).type == "cpu":
            print("WARNING: RMBG is running on CPU; background removal may be slow.", flush=True)

    @torch.no_grad()
    def __call__(self, image: Image.Image) -> Image.Image:
        from torchvision import transforms

        image = composite_rgba_on_white(image)
        device = next(self.model.parameters()).device
        print("Removing image background with RMBG-2.0/BiRefNet...", flush=True)
        inp = self.transform(image).unsqueeze(0).to(device)
        pred = self.model(inp)[-1].sigmoid().cpu()[0].squeeze()
        if not torch.isfinite(pred).all():
            raise ValueError("RMBG predicted a non-finite foreground mask")
        mask = transforms.ToPILImage()(pred).resize(image.size)
        white = Image.new("RGB", image.size, (255, 255, 255))
        result = Image.composite(image, white, mask)
        print("Image background removal complete.", flush=True)
        return result


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


def prepare_image(
    image: ImageInput,
    size: int = 1_024,
    *,
    background_remover: BackgroundRemover | None = None,
) -> torch.Tensor:
    """Optionally remove background, then load a float32 ``(3, size, size)`` image."""
    source = Image.open(image) if isinstance(image, (str, Path)) else image
    rgb = composite_rgba_on_white(source)
    if background_remover is not None:
        rgb = background_remover(rgb)
    square = pad_to_square(rgb)
    resized = square.resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
