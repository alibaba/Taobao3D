"""Deterministic random-state utilities for resumable stage inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    from pytorch3d.ops import sample_farthest_points as _p3d_fps
except ImportError:  # pragma: no cover - optional acceleration
    _p3d_fps = None


_FPS_TIE_BREAK_SEED = 0
_P3D_FPS_UNUSABLE = False


@dataclass
class RandomState:
    """NumPy and torch RNG states captured at a stage boundary."""

    numpy_state: tuple
    torch_cpu_state: torch.Tensor
    torch_cuda_states: list[torch.Tensor]


def seed_all(seed: int) -> None:
    """Seed NumPy and torch without changing backend numerical settings."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_random_state() -> RandomState:
    """Capture all RNG streams consumed by the inference pipeline."""
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return RandomState(
        numpy_state=np.random.get_state(),
        torch_cpu_state=torch.get_rng_state().clone(),
        torch_cuda_states=[state.clone() for state in cuda_states],
    )


def restore_random_state(state: RandomState) -> None:
    """Restore a state previously returned by :func:`capture_random_state`."""
    np.random.set_state(state.numpy_state)
    torch.set_rng_state(state.torch_cpu_state)
    if state.torch_cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Saved CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(state.torch_cuda_states)


def deterministic_fps(
    points: torch.Tensor,
    max_points: int | None,
    *,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Cap a point cloud with deterministic farthest-point sampling.

    A private fixed-seed permutation provides spatially unbiased tie-breaking on
    lattice point clouds without consuming any inference RNG stream. Returned
    indices are CPU tensors sorted in the input order.
    """
    points = torch.as_tensor(points)
    if points.ndim != 2:
        raise ValueError(f"Expected a 2D point cloud, got shape {tuple(points.shape)}")
    point_count = int(points.shape[0])
    keep_count = int(max_points) if max_points else 0
    if keep_count <= 0 or point_count <= keep_count:
        return points, None

    sample_device = torch.device(device) if device is not None else points.device
    samples = points.detach().to(device=sample_device, dtype=torch.float32).contiguous()
    permutation = torch.randperm(
        point_count,
        generator=torch.Generator().manual_seed(_FPS_TIE_BREAK_SEED),
    ).to(sample_device)
    samples = samples[permutation]

    global _P3D_FPS_UNUSABLE
    selected = None
    if _p3d_fps is not None and samples.is_cuda and not _P3D_FPS_UNUSABLE:
        try:
            _, selected = _p3d_fps(
                samples.unsqueeze(0),
                K=keep_count,
                random_start_point=False,
            )
            selected = selected[0]
        except RuntimeError as error:
            selected = None
            if not isinstance(error, torch.cuda.OutOfMemoryError):
                _P3D_FPS_UNUSABLE = True

    if selected is None:
        minimum_distances = torch.full(
            (point_count,),
            float("inf"),
            device=sample_device,
        )
        selected = torch.zeros(keep_count, dtype=torch.long, device=sample_device)
        current = samples[0]
        for index in range(1, keep_count):
            squared_distances = (samples - current).pow(2).sum(dim=-1)
            minimum_distances = torch.minimum(
                minimum_distances,
                squared_distances,
            )
            next_index = minimum_distances.argmax()
            selected[index] = next_index
            current = samples[next_index]

    original_indices = permutation[selected].sort().values.cpu()
    kept = points.index_select(0, original_indices.to(points.device))
    return kept, original_indices
