"""Stage 1 raw-space octree refinement and strict checkpoint loader."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ..random import deterministic_fps
from .dino import DinoConditioner, load_pretrained_dino_state_dict
from .octree_dit import RawOctreeDenoiser, cell_centers


_DINO_LAYER_SOURCE = "dino_encoder.model.layer."
_DINO_LAYER_TARGET = "dino_encoder.model.model.layer."
_EXPERT_LEVELS = (6, 7, 8)


class Stage1Model(nn.Module):
    """Refine level-6 cells to level-9 cells with three raw-space experts."""

    input_level = 6
    output_level = 9

    def __init__(self):
        super().__init__()
        self.dino_encoder = DinoConditioner(image_size=768)
        self.dino_encoder.to(dtype=torch.bfloat16)
        self.denoisers = nn.ModuleList(
            [RawOctreeDenoiser() for _ in _EXPERT_LEVELS]
        )
        self._target_device = torch.device("cpu")
        self._active_expert: int | None = None

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one ``(3,H,W)`` RGB image in ``[0,1]``."""
        device = next(self.dino_encoder.parameters()).device
        image_batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            condition = self.dino_encoder(image_batch)
        return condition.float()

    def expert_for_level(self, level: int) -> RawOctreeDenoiser:
        try:
            expert_index = _EXPERT_LEVELS.index(int(level))
        except ValueError as error:
            raise ValueError(
                f"Stage 1 only owns levels {_EXPERT_LEVELS}, got {level}"
            ) from error
        return self.denoisers[expert_index]

    def prepare_device(self, device: str | torch.device) -> None:
        """Place DINO on the target device while keeping experts paged on CPU."""
        self._target_device = torch.device(device)
        self.dino_encoder.to(self._target_device)
        for expert in self.denoisers:
            expert.to("cpu")
        self._active_expert = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _activate_expert(self, level: int) -> RawOctreeDenoiser:
        expert_index = _EXPERT_LEVELS.index(level)
        if self._active_expert == expert_index:
            return self.denoisers[expert_index]
        for index, expert in enumerate(self.denoisers):
            target = (
                self._target_device
                if index == expert_index
                else torch.device("cpu")
            )
            if next(expert.parameters()).device != target:
                expert.to(target)
        self._active_expert = expert_index
        if self._target_device.type == "cuda":
            torch.cuda.empty_cache()
        return self.denoisers[expert_index]

    def offload(self) -> None:
        """Move the complete stage to CPU."""
        self.dino_encoder.to("cpu")
        for expert in self.denoisers:
            expert.to("cpu")
        self._active_expert = None
        self._target_device = torch.device("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _validate_seed(coordinates: torch.Tensor) -> torch.Tensor:
        coordinates = torch.as_tensor(coordinates, dtype=torch.long)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3:
            raise ValueError(
                "Stage 1 seed must have shape (N,3), got "
                f"{tuple(coordinates.shape)}"
            )
        if coordinates.shape[0] == 0:
            raise ValueError("Stage 1 seed is empty")
        minimum = int(coordinates.min())
        maximum = int(coordinates.max())
        if minimum < 0 or maximum >= 2**Stage1Model.input_level:
            raise ValueError(
                "Stage 1 seed coordinates must lie in [0,63], got "
                f"min={minimum}, max={maximum}"
            )
        return torch.unique(coordinates, dim=0)

    @staticmethod
    def _expand_children(
        parent_coordinates: torch.Tensor,
        occupancy: torch.Tensor,
    ) -> torch.Tensor:
        children = []
        for octant in range(8):
            selected = occupancy[:, octant].nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            offset = torch.tensor(
                [
                    (octant >> 2) & 1,
                    (octant >> 1) & 1,
                    octant & 1,
                ],
                device=parent_coordinates.device,
                dtype=torch.long,
            )
            children.append(parent_coordinates[selected] * 2 + offset)
        if not children:
            return parent_coordinates.new_zeros((0, 3))
        return torch.cat(children, dim=0)

    @torch.no_grad()
    def refine_coordinates(
        self,
        condition: torch.Tensor,
        seed_coordinates: torch.Tensor,
        *,
        num_steps: int = 30,
        cfg_scale: float = 7.0,
        time_shift: float = 1.0,
        threshold: float = 0.5,
        initial_states: Mapping[int, torch.Tensor] | None = None,
        show_progress: bool = True,
        offload_after: bool = True,
    ) -> torch.Tensor:
        """Return integer occupied cells on the final ``512^3`` grid."""
        if condition.ndim != 3 or condition.shape[0] != 1:
            raise ValueError(
                "Stage 1 condition must have shape (1,T,1024), got "
                f"{tuple(condition.shape)}"
            )
        if condition.shape[-1] != 1_024:
            raise ValueError("Stage 1 condition channel count must be 1024")
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0,1)")

        device = condition.device
        if device != self._target_device:
            self.prepare_device(device)
        coordinates = self._validate_seed(seed_coordinates).to(device)
        unconditional = torch.zeros_like(condition) if cfg_scale != 1.0 else None
        linear_times = np.linspace(0.0, 1.0, num_steps + 1)
        if time_shift > 0:
            times = linear_times / (
                linear_times + time_shift * (1.0 - linear_times)
            )
        else:
            times = linear_times

        progress = tqdm(
            total=len(_EXPERT_LEVELS) * num_steps,
            disable=not show_progress,
            desc="[Stage 1/3] refinement",
            leave=False,
        )
        try:
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                for level in _EXPERT_LEVELS:
                    if coordinates.shape[0] == 0:
                        break
                    progress.set_description(
                        f"[Stage 1/3] {2**level}->{2 ** (level + 1)}"
                    )
                    expert = self._activate_expert(level)
                    positions = cell_centers(coordinates, level).unsqueeze(0)
                    expected_shape = (1, coordinates.shape[0], 8)
                    if initial_states is not None and level in initial_states:
                        state = initial_states[level].to(
                            device=device,
                            dtype=torch.float32,
                        )
                        if tuple(state.shape) != expected_shape:
                            raise ValueError(
                                f"Initial state for level {level} has shape "
                                f"{tuple(state.shape)}, expected {expected_shape}"
                            )
                    else:
                        state = torch.randn(*expected_shape, device=device)

                    for step in range(num_steps):
                        current_time = float(times[step])
                        next_time = float(times[step + 1])
                        timestep = torch.full((1,), current_time, device=device)
                        conditional_prediction = expert(
                            state,
                            positions,
                            level,
                            timestep,
                            condition,
                        )
                        if unconditional is None:
                            prediction = conditional_prediction
                        else:
                            unconditional_prediction = expert(
                                state,
                                positions,
                                level,
                                timestep,
                                unconditional,
                            )
                            prediction = unconditional_prediction + cfg_scale * (
                                conditional_prediction - unconditional_prediction
                            )
                        x1_prediction = torch.sigmoid(prediction)
                        coefficient = (next_time - current_time) / max(
                            1.0 - current_time,
                            1e-3,
                        )
                        state = state + coefficient * (x1_prediction - state)
                        progress.update(1)

                    occupancy = state[0].clamp(0.0, 1.0) > threshold
                    coordinates = self._expand_children(coordinates, occupancy)
        finally:
            progress.close()
            if offload_after and self._target_device.type == "cuda":
                for expert in self.denoisers:
                    expert.to("cpu")
                self._active_expert = None
                torch.cuda.empty_cache()
        return coordinates

    @staticmethod
    def normalize_vertices(vertices: torch.Tensor) -> torch.Tensor:
        """Uniformly center and scale a point cloud to fill ``[-1,1]``."""
        if vertices.shape[0] == 0:
            return vertices
        center = (vertices.amax(dim=0) + vertices.amin(dim=0)) / 2.0
        centered = vertices - center
        max_extent = centered.abs().amax()
        if float(max_extent) < 1e-8:
            return centered
        return centered / max_extent

    @staticmethod
    def snap_vertices(vertices: torch.Tensor, resolution: int = 512) -> torch.Tensor:
        """Snap vertices to the ``q/(R-1)*2-1`` lattice used by Stage 2."""
        if vertices.shape[0] == 0:
            return vertices
        resolution = int(resolution)
        if resolution < 2:
            raise ValueError("voxel resolution must be at least 2")
        vertices64 = vertices.to(dtype=torch.float64)
        quantized = torch.round(
            (vertices64 + 1.0) / 2.0 * (resolution - 1)
        ).clamp_(0, resolution - 1)
        return (quantized / (resolution - 1) * 2.0 - 1.0).float()

    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        seed_coordinates: torch.Tensor,
        *,
        max_vertices: int | None = 20_000,
        voxel_resolution: int = 512,
        **sampling_options,
    ) -> torch.Tensor:
        """Refine, cap, normalize, and snap vertices for Stage 2."""
        condition = self.encode_image(image)
        coordinates = self.refine_coordinates(
            condition,
            seed_coordinates,
            **sampling_options,
        )
        vertices = cell_centers(coordinates, self.output_level)
        vertices, _ = deterministic_fps(
            vertices,
            max_vertices,
            device=vertices.device,
        )
        vertices = self.normalize_vertices(vertices)
        return self.snap_vertices(vertices, voxel_resolution)


def _strip_compile_prefix(key: str) -> str:
    return key.replace("module.", "").replace("_orig_mod.", "")


def _map_checkpoint_key(source_key: str) -> str:
    if source_key.startswith(_DINO_LAYER_SOURCE):
        return _DINO_LAYER_TARGET + source_key[len(_DINO_LAYER_SOURCE) :]
    return source_key


def _validate_release_config(checkpoint: dict) -> None:
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("Stage 1 checkpoint is missing its architecture config")
    model = config.get("model") or {}
    flow = config.get("flow_matching") or {}
    expected_model = {
        "hidden_dim": 1_536,
        "num_heads": 12,
        "num_blocks": 28,
        "ffn_ratio": 5.3334,
        "use_parent_cross_attn": False,
        "use_vert_count_cond": False,
        "qk_rms_norm": True,
        "qk_rms_norm_cross": True,
        "share_mod": True,
        "cond_fusion_mode": "concat",
        "rope_pos_mode": "normalized",
    }
    mismatches = [
        f"model.{key}={model.get(key)!r} (expected {value!r})"
        for key, value in expected_model.items()
        if model.get(key) != value
    ]
    groups = (model.get("level_moe") or {}).get("expert_level_groups")
    if groups != [[6], [7], [8]]:
        mismatches.append(
            f"model.level_moe.expert_level_groups={groups!r} "
            "(expected [[6], [7], [8]])"
        )
    expected_flow = {
        "prediction_target": "occupancy",
        "occ_threshold": 0.5,
        "noise_type": "randn",
        "supervision_mode": "x1_logit_bce",
    }
    mismatches.extend(
        f"flow_matching.{key}={flow.get(key)!r} (expected {value!r})"
        for key, value in expected_flow.items()
        if flow.get(key) != value
    )
    x1_normalize = flow.get("x1_normalize") or {}
    if bool(x1_normalize.get("enabled", False)):
        mismatches.append("flow_matching.x1_normalize.enabled must be false")
    if mismatches:
        raise RuntimeError(
            "Stage 1 checkpoint does not match the release architecture: "
            + "; ".join(mismatches)
        )


def load_stage1_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
    *,
    dino_checkpoint_path: str | Path | None = None,
) -> Stage1Model:
    """Load every release tensor and page experts one at a time at inference."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    _validate_release_config(checkpoint)
    source = checkpoint.get("model_state_dict", checkpoint)
    with torch.device("meta"):
        model = Stage1Model()
    expected = model.state_dict()
    mapped = {}
    unexpected = []
    shape_mismatches = []
    for raw_key, value in source.items():
        source_key = _strip_compile_prefix(raw_key)
        target_key = _map_checkpoint_key(source_key)
        if target_key not in expected:
            unexpected.append(source_key)
            continue
        if target_key in mapped:
            raise RuntimeError(
                f"Stage 1 checkpoint maps multiple tensors to {target_key!r}"
            )
        actual_shape = tuple(value.shape)
        expected_shape = tuple(expected[target_key].shape)
        if actual_shape != expected_shape:
            shape_mismatches.append(
                f"{source_key}: {actual_shape} != {expected_shape}"
            )
        else:
            mapped[target_key] = value

    if unexpected:
        raise RuntimeError(
            "Stage 1 checkpoint has unexpected tensors: "
            + ", ".join(sorted(unexpected)[:20])
        )
    if shape_mismatches:
        raise RuntimeError(
            "Stage 1 checkpoint has incompatible tensor shapes: "
            + "; ".join(shape_mismatches[:20])
        )

    _dino_prefix = "dino_encoder."
    if not any(k.startswith(_dino_prefix) for k in mapped):
        if dino_checkpoint_path is None:
            raise RuntimeError(
                "Stage 1 checkpoint contains no DINO weights; "
                "pass --dino-checkpoint to supply a HuggingFace DINOv3 directory"
            )
        for k, v in load_pretrained_dino_state_dict(dino_checkpoint_path).items():
            target_key = _dino_prefix + k
            if target_key in expected:
                exp_shape = tuple(expected[target_key].shape)
                act_shape = tuple(v.shape)
                if act_shape != exp_shape:
                    shape_mismatches.append(
                        f"DINO pretrained {k}: {act_shape} != {exp_shape}"
                    )
                else:
                    mapped[target_key] = v
        if shape_mismatches:
            raise RuntimeError(
                "DINO pretrained weights have incompatible shapes: "
                + "; ".join(shape_mismatches[:20])
            )

    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise RuntimeError(
            "Stage 1 checkpoint is missing inference tensors: "
            + ", ".join(missing[:20])
        )

    model.load_state_dict(mapped, strict=True, assign=True)
    model.dino_encoder.materialize_nonpersistent_buffers()
    model.requires_grad_(False)
    model.eval()
    model.prepare_device(device)
    return model
