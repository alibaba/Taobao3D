"""Checkpoint-backed DINOv3 image conditioner."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import transforms
from transformers import DINOv3ViTConfig, DINOv3ViTModel


class DinoConditioner(nn.Module):
    """DINOv3-L/16 wrapper matching the training-time image transform."""

    hidden_size = 1_024
    num_register_tokens = 4

    def __init__(self, image_size: int = 1_024):
        super().__init__()
        config = DINOv3ViTConfig(
            patch_size=16,
            hidden_size=self.hidden_size,
            intermediate_size=4_096,
            num_hidden_layers=24,
            num_attention_heads=16,
            hidden_act="gelu",
            attention_dropout=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            rope_theta=100.0,
            image_size=224,
            num_channels=3,
            query_bias=True,
            key_bias=False,
            value_bias=True,
            proj_bias=True,
            mlp_bias=True,
            layerscale_value=1.0,
            drop_path_rate=0.0,
            use_gated_mlp=False,
            num_register_tokens=self.num_register_tokens,
            pos_embed_shift=None,
            pos_embed_jitter=None,
            pos_embed_rescale=2.0,
            apply_layernorm=True,
        )
        self.model = DINOv3ViTModel(config)
        self.image_size = image_size
        self.num_patches = (image_size // config.patch_size) ** 2
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    image_size,
                    transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.model.eval()
        self.model.requires_grad_(False)

    def materialize_nonpersistent_buffers(self) -> None:
        """Restore deterministic buffers omitted from a checkpoint state dict."""
        rope = self.model.rope_embeddings
        if rope.inv_freq.is_meta:
            target_dtype = rope.inv_freq.dtype
            rope.inv_freq = (
                1
                / rope.base
                ** torch.arange(
                    0,
                    1,
                    4 / rope.head_dim,
                    dtype=torch.float32,
                )
            ).to(dtype=target_dtype)

    def forward(self, image):
        image = image.to(
            device=next(self.model.parameters()).device,
            dtype=next(self.model.parameters()).dtype,
        )
        output = self.model(self.transform(image)).last_hidden_state
        return output[:, 1 + self.num_register_tokens :, :]


def load_pretrained_dino_state_dict(
    pretrained_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Load DINOv3 weights from a HuggingFace pretrained directory.

    Returns a state_dict whose keys match :class:`DinoConditioner.state_dict`
    (i.e. prefixed with ``model.``).
    """
    pretrained = DINOv3ViTModel.from_pretrained(str(pretrained_path))
    return {"model." + k: v for k, v in pretrained.state_dict().items()}
