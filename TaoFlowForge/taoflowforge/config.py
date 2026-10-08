"""Typed inference configuration for the fixed three-stage pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


OffloadMode = Literal["auto", "on", "off"]


@dataclass(frozen=True)
class Stage0Config:
    """Stage 0 occupancy generation settings."""

    checkpoint: Path
    num_steps: int = 50
    cfg_scale: float = 7.5
    t_shift: float = 2.718
    occupancy_threshold: float = 0.5
    local_max_neighbors: Literal["mid"] = "mid"
    local_max_keep: int = 2
    max_cells: int = 15_000
    compile_model: bool = False


@dataclass(frozen=True)
class Stage1Config:
    """Stage 1 raw-space vertex upsampling settings."""

    checkpoint: Path
    num_steps: int = 30
    cfg_scale: float = 7.0
    vc_cfg_scale: float = 1.0
    t_shift: float = 1.0
    occupancy_threshold: float = 0.5
    max_vertices: int = 20_000


@dataclass(frozen=True)
class Stage2Config:
    """Stage 2 feature-space topology generation settings."""

    checkpoint: Path
    num_steps: int = 50
    cfg_scale: float = 3.0
    t_shift: float = 1.0
    edge_threshold: float = 0.5


@dataclass(frozen=True)
class InferenceConfig:
    """Complete image-to-mesh pipeline settings."""

    stage0: Stage0Config
    stage1: Stage1Config
    stage2: Stage2Config
    dino_checkpoint: Path | None = None
    seed: int = 42
    target_num_vertices: int = 3_000
    offload: OffloadMode = "auto"
    fill_holes: bool = True
    image_size: int = 1_024
    output_dir: Path = field(default_factory=lambda: Path("output"))

    def __post_init__(self) -> None:
        if self.offload not in {"auto", "on", "off"}:
            raise ValueError(f"Unsupported offload mode: {self.offload!r}")
        if self.target_num_vertices <= 0:
            raise ValueError("target_num_vertices must be positive")
