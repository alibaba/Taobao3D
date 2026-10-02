"""Portable NPZ artifacts for independent stage execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .random import RandomState

_RESERVED_PREFIX = "__taoflowforge_"
_LEGACY_RESERVED_PREFIX = "__swifttopo_"
_RESERVED_PREFIXES = (_RESERVED_PREFIX, _LEGACY_RESERVED_PREFIX)


def save_stage_artifact(
    path: str | Path,
    *,
    stage: int,
    seed: int,
    outputs: Mapping[str, Any],
    random_state: RandomState,
) -> None:
    """Save stage outputs and the exact RNG continuation state."""
    collisions = [
        key
        for key in outputs
        if any(key.startswith(prefix) for prefix in _RESERVED_PREFIXES)
    ]
    if collisions:
        raise ValueError(f"Reserved artifact keys: {collisions}")

    np_name, np_keys, np_pos, np_has_gauss, np_cached = random_state.numpy_state
    payload: dict[str, Any] = {
        **outputs,
        f"{_RESERVED_PREFIX}format": np.array(1, dtype=np.int64),
        f"{_RESERVED_PREFIX}stage": np.array(stage, dtype=np.int64),
        f"{_RESERVED_PREFIX}seed": np.array(seed, dtype=np.int64),
        f"{_RESERVED_PREFIX}numpy_name": np.array(np_name),
        f"{_RESERVED_PREFIX}numpy_keys": np.asarray(np_keys, dtype=np.uint32),
        f"{_RESERVED_PREFIX}numpy_pos": np.array(np_pos, dtype=np.int64),
        f"{_RESERVED_PREFIX}numpy_has_gauss": np.array(np_has_gauss, dtype=np.int64),
        f"{_RESERVED_PREFIX}numpy_cached_gaussian": np.array(np_cached, dtype=np.float64),
        f"{_RESERVED_PREFIX}torch_cpu": random_state.torch_cpu_state.cpu().numpy(),
        f"{_RESERVED_PREFIX}torch_cuda_count": np.array(
            len(random_state.torch_cuda_states), dtype=np.int64
        ),
    }
    for index, cuda_state in enumerate(random_state.torch_cuda_states):
        payload[f"{_RESERVED_PREFIX}torch_cuda_{index}"] = cuda_state.cpu().numpy()

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)


def load_stage_artifact(
    path: str | Path,
) -> tuple[int, int, dict[str, np.ndarray], RandomState]:
    """Load outputs and RNG state without pickle-backed arrays."""
    with np.load(path, allow_pickle=False) as archive:
        if f"{_RESERVED_PREFIX}format" in archive.files:
            prefix = _RESERVED_PREFIX
        elif f"{_LEGACY_RESERVED_PREFIX}format" in archive.files:
            prefix = _LEGACY_RESERVED_PREFIX
        else:
            raise ValueError("Not a TaoFlowForge stage artifact")

        version = int(archive[f"{prefix}format"])
        if version != 1:
            raise ValueError(f"Unsupported stage artifact version: {version}")
        stage = int(archive[f"{prefix}stage"])
        seed = int(archive[f"{prefix}seed"])
        numpy_state = (
            str(archive[f"{prefix}numpy_name"]),
            archive[f"{prefix}numpy_keys"].copy(),
            int(archive[f"{prefix}numpy_pos"]),
            int(archive[f"{prefix}numpy_has_gauss"]),
            float(archive[f"{prefix}numpy_cached_gaussian"]),
        )
        cpu_state = torch.from_numpy(archive[f"{prefix}torch_cpu"].copy())
        cuda_count = int(archive[f"{prefix}torch_cuda_count"])
        cuda_states = [
            torch.from_numpy(archive[f"{prefix}torch_cuda_{index}"].copy())
            for index in range(cuda_count)
        ]
        outputs = {
            key: archive[key].copy()
            for key in archive.files
            if not key.startswith(_RESERVED_PREFIXES)
        }
    return stage, seed, outputs, RandomState(numpy_state, cpu_state, cuda_states)
