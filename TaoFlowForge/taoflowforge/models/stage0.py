"""Stage 0 image-to-occupancy model and strict checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from ..random import deterministic_fps
from .dino import DinoConditioner, load_pretrained_dino_state_dict
from .flow import EulerFlowSampler
from .ss_flow import SparseStructureFlowModel
from .ss_vae import SparseStructureDecoder


_DENOISER_TIME_SCALE = 1_000.0
_DINO_LAYER_SOURCE = "dino_encoder.model.layer."
_DINO_LAYER_TARGET = "model.model.layer."
_DINO_SOURCE = "dino_encoder."
_DENOISER_SOURCE = "denoiser."
_VAE_DECODER_SOURCE = "vae.decoder."
_VAE_ENCODER_SOURCE = "vae.encoder."


class Stage0Model(nn.Module):
    """Generate level-6 occupied voxels through a 16-cube VAE latent."""

    latent_channels = 8
    latent_resolution = 16
    output_resolution = 64

    def __init__(self):
        super().__init__()
        self._dino_encoder = DinoConditioner(image_size=512)
        self._dino_encoder.to(dtype=torch.bfloat16)
        self.denoiser = SparseStructureFlowModel()
        self.vae_decoder = SparseStructureDecoder()
        self.register_buffer(
            "_latent_mean",
            torch.zeros(1, self.latent_channels, 1, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_latent_std",
            torch.ones(1, self.latent_channels, 1, 1, 1),
            persistent=False,
        )
        self.flow_sampler = EulerFlowSampler(prediction_type="velocity")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one ``(3,H,W)`` RGB image in ``[0,1]`` to patch tokens."""
        device = next(self.parameters()).device
        image_batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            condition = self._dino_encoder(image_batch)
        return condition.float()

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        *,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(
            latent,
            timestep * _DENOISER_TIME_SCALE,
            condition,
        )

    @torch.no_grad()
    def sample_latent(
        self,
        condition: torch.Tensor,
        *,
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        time_shift: float = 2.718,
        start_time: float = 0.013,
        initial_state: torch.Tensor | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        unconditional = None
        if cfg_scale != 1.0:
            unconditional = {"condition": torch.zeros_like(condition)}
        device = condition.device
        shape = (
            condition.shape[0],
            self.latent_channels,
            self.latent_resolution,
            self.latent_resolution,
            self.latent_resolution,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return self.flow_sampler.sample(
                self,
                shape,
                num_steps=num_steps,
                device=device,
                dtype=condition.dtype,
                model_kwargs={"condition": condition},
                cfg_scale=cfg_scale,
                unconditional_kwargs=unconditional,
                time_shift=time_shift,
                start_time=start_time,
                initial_state=initial_state,
                show_progress=show_progress,
                description="[Stage 0/3] occupancy",
            )

    @torch.no_grad()
    def decode_logits(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        latent = normalized_latent.float() * self._latent_std + self._latent_mean
        device = latent.device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits = self.vae_decoder(latent)
        return logits.float()

    @torch.no_grad()
    def decode_occupancy(self, normalized_latent: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decode_logits(normalized_latent))

    @staticmethod
    def occupancy_to_coordinates(
        occupancy: torch.Tensor,
        threshold: float = 0.5,
        *,
        mid_keep: int = 2,
    ) -> torch.Tensor:
        """Apply the release ``mid`` filter and return sorted voxel coordinates."""
        if occupancy.ndim != 5 or occupancy.shape[:2] != (1, 1):
            raise ValueError(
                "Expected occupancy with shape (1,1,resolution,resolution,resolution)"
            )
        field = occupancy[0, 0]
        if field.shape[0] != field.shape[1] or field.shape[1] != field.shape[2]:
            raise ValueError(f"Expected a cubic occupancy field, got {tuple(field.shape)}")

        occupied = field >= float(threshold)
        occupancy_int = occupied.long()
        keep_depth = max(1, int(mid_keep))

        def shift_previous(values: torch.Tensor, axis: int) -> torch.Tensor:
            shifted = torch.zeros_like(values)
            destination = [slice(None)] * 3
            source = [slice(None)] * 3
            destination[axis] = slice(1, None)
            source[axis] = slice(0, -1)
            shifted[tuple(destination)] = values[tuple(source)]
            return shifted

        def forward_distance(values: torch.Tensor, axis: int) -> torch.Tensor:
            distances = torch.zeros_like(values)
            for _ in range(values.shape[axis]):
                updated = values * (shift_previous(distances, axis) + 1)
                if torch.equal(updated, distances):
                    return updated
                distances = updated
            return distances

        keep = torch.zeros_like(occupied)
        for axis in range(3):
            forward = forward_distance(occupancy_int, axis)
            reverse = forward_distance(
                occupancy_int.flip(axis),
                axis,
            ).flip(axis)
            run_length = forward + reverse - 1
            boundary_distance = torch.minimum(forward, reverse)
            protected = boundary_distance <= keep_depth
            median = (2 * forward == run_length) | (2 * forward == run_length + 1)
            keep |= occupied & (protected | median)
        return keep.nonzero(as_tuple=False)

    @staticmethod
    def coordinates_to_centers(
        coordinates: torch.Tensor,
        level: int = 6,
    ) -> torch.Tensor:
        return (coordinates.float() + 0.5) / float(2**level) * 2.0 - 1.0

    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        *,
        num_steps: int = 50,
        cfg_scale: float = 7.5,
        time_shift: float = 2.718,
        start_time: float = 0.013,
        occupancy_threshold: float = 0.5,
        local_max_keep: int = 2,
        max_cells: int | None = 15_000,
        show_progress: bool = True,
    ) -> torch.Tensor:
        condition = self.encode_image(image)
        latent = self.sample_latent(
            condition,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            time_shift=time_shift,
            start_time=start_time,
            show_progress=show_progress,
        )
        occupancy = self.decode_occupancy(latent)
        coordinates = self.occupancy_to_coordinates(
            occupancy,
            occupancy_threshold,
            mid_keep=local_max_keep,
        )
        coordinates, _ = deterministic_fps(
            coordinates,
            max_cells,
            device=coordinates.device,
        )
        return coordinates


def _strip_compile_prefix(key: str) -> str:
    return key.replace("module.", "").replace("_orig_mod.", "")


def _map_dino_key(source_key: str) -> str:
    if source_key.startswith(_DINO_LAYER_SOURCE):
        return _DINO_LAYER_TARGET + source_key[len(_DINO_LAYER_SOURCE) :]
    return source_key[len(_DINO_SOURCE) :]


def _load_component(
    module: nn.Module,
    state: dict[str, torch.Tensor],
    component_name: str,
    *,
    cast_to_expected: bool = False,
) -> None:
    expected = module.state_dict()
    unexpected = sorted(set(state) - set(expected))
    missing = sorted(set(expected) - set(state))
    mismatches = [
        f"{key}: {tuple(state[key].shape)} != {tuple(expected[key].shape)}"
        for key in set(state) & set(expected)
        if tuple(state[key].shape) != tuple(expected[key].shape)
    ]
    if unexpected:
        raise RuntimeError(
            f"{component_name} checkpoint has unexpected tensors: "
            + ", ".join(unexpected[:20])
        )
    if missing:
        raise RuntimeError(
            f"{component_name} checkpoint is missing tensors: "
            + ", ".join(missing[:20])
        )
    if mismatches:
        raise RuntimeError(
            f"{component_name} checkpoint has incompatible tensor shapes: "
            + "; ".join(mismatches[:20])
        )
    if cast_to_expected:
        state = {
            key: value.to(dtype=expected[key].dtype)
            for key, value in state.items()
        }
    module.load_state_dict(state, strict=True, assign=True)


def load_stage0_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    use_ema: bool = True,
    dino_checkpoint_path: str | Path | None = None,
) -> Stage0Model:
    """Load every Stage 0 inference tensor from a single merged checkpoint.

    Expected checkpoint layout::

        {
            "model_state_dict": {
                "dino_encoder.*": ...,   # optional when *dino_checkpoint_path* given
                "denoiser.*": ...,
                "vae.decoder.*": ...,
                "vae.encoder.*": ...,    # silently ignored
            },
            "ema_state_dict": {...},     # optional; denoiser EMA weights
            "channel_mean": Tensor,      # latent-space per-channel mean (8,)
            "channel_std": Tensor,       # latent-space per-channel std  (8,)
        }
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    source = {
        _strip_compile_prefix(key): value
        for key, value in checkpoint.get("model_state_dict", checkpoint).items()
    }
    if use_ema and "ema_state_dict" in checkpoint:
        ema = {
            _strip_compile_prefix(key): value
            for key, value in checkpoint["ema_state_dict"].items()
        }
        stray_ema = sorted(set(ema) - set(source))
        invalid_ema = sorted(
            key for key in ema if not key.startswith(_DENOISER_SOURCE)
        )
        if stray_ema or invalid_ema:
            details = stray_ema[:10] + invalid_ema[:10]
            raise RuntimeError(
                "Stage 0 EMA is not a denoiser-only subset: "
                + ", ".join(details)
            )
        source.update(ema)

    dino_state: dict[str, torch.Tensor] = {}
    denoiser_state: dict[str, torch.Tensor] = {}
    decoder_state: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    for key, value in source.items():
        if key.startswith(_DINO_SOURCE):
            target = _map_dino_key(key)
            if target in dino_state:
                raise RuntimeError(f"Duplicate Stage 0 DINO tensor: {target}")
            dino_state[target] = value
        elif key.startswith(_DENOISER_SOURCE):
            target = key[len(_DENOISER_SOURCE) :]
            if target in denoiser_state:
                raise RuntimeError(f"Duplicate Stage 0 denoiser tensor: {target}")
            denoiser_state[target] = value
        elif key.startswith(_VAE_DECODER_SOURCE):
            target = key[len(_VAE_DECODER_SOURCE) :]
            decoder_state[target] = value
        elif key.startswith(_VAE_ENCODER_SOURCE):
            pass  # encoder weights are not needed for inference
        else:
            unexpected.append(key)
    if unexpected:
        raise RuntimeError(
            "Stage 0 checkpoint has unexpected tensors: "
            + ", ".join(sorted(unexpected)[:20])
        )

    if not dino_state:
        if dino_checkpoint_path is None:
            raise RuntimeError(
                "Stage 0 checkpoint contains no DINO weights; "
                "pass --dino-checkpoint to supply a HuggingFace DINOv3 directory"
            )
        dino_state = load_pretrained_dino_state_dict(dino_checkpoint_path)

    if not decoder_state:
        raise RuntimeError("Stage 0 checkpoint is missing VAE decoder weights (vae.decoder.*)")

    if "channel_mean" not in checkpoint or "channel_std" not in checkpoint:
        raise RuntimeError("Stage 0 checkpoint is missing latent normalization statistics (channel_mean / channel_std)")
    mean = checkpoint["channel_mean"].float().reshape(-1)
    std = checkpoint["channel_std"].float().reshape(-1)
    if mean.numel() != Stage0Model.latent_channels:
        raise RuntimeError(f"Expected 8 latent means, got {mean.numel()}")
    if std.numel() != Stage0Model.latent_channels or torch.any(std <= 0):
        raise RuntimeError("Stage 0 latent standard deviations are invalid")

    with torch.device("meta"):
        model = Stage0Model()
    _load_component(model._dino_encoder, dino_state, "Stage 0 DINO")
    _load_component(model.denoiser, denoiser_state, "Stage 0 denoiser")
    _load_component(
        model.vae_decoder,
        decoder_state,
        "Stage 0 VAE decoder",
        cast_to_expected=True,
    )
    model._dino_encoder.materialize_nonpersistent_buffers()
    model._latent_mean = mean.reshape(1, -1, 1, 1, 1)
    model._latent_std = std.reshape(1, -1, 1, 1, 1)
    model.requires_grad_(False)
    model.eval()
    return model.to(device)
