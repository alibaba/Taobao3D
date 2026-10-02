"""Unified deterministic runner for the fixed TaoFlowForge inference pipeline."""

from __future__ import annotations

import gc
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn.functional as functional
import trimesh
from PIL import Image

from .artifacts import load_stage_artifact, save_stage_artifact
from .config import InferenceConfig, OffloadMode, Stage0Config, Stage1Config, Stage2Config
from .image import ImageInput, prepare_image
from .models.stage0 import Stage0Model, load_stage0_model
from .models.stage1 import Stage1Model, load_stage1_model
from .models.stage2 import Stage2Model, load_stage2_model
from .postprocess import export_normalized_glb, export_obj, fill_mesh_holes
from .random import capture_random_state, restore_random_state, seed_all


@dataclass(frozen=True)
class PipelineResult:
    """Files and in-memory mesh produced by a complete pipeline run."""

    mesh: trimesh.Trimesh
    output_dir: Path
    raw_mesh_path: Path
    final_mesh_path: Path
    glb_path: Path | None
    artifact_paths: dict[int, Path]
    metadata_path: Path


@dataclass(frozen=True)
class PipelineUpdate:
    """A stage-boundary update emitted by :meth:`TaoFlowForgePipeline.iter_run`."""

    stage: int | None
    message: str
    result: PipelineResult | None = None


class TaoFlowForgePipeline:
    """Load and execute any subset of the release's three inference stages."""

    def __init__(
        self,
        *,
        stage0: Stage0Config | None = None,
        stage1: Stage1Config | None = None,
        stage2: Stage2Config | None = None,
        seed: int = 42,
        offload: OffloadMode = "auto",
        fill_holes: bool = True,
        image_size: int = 1_024,
        output_dir: str | Path = "output",
        device: str | torch.device | None = None,
    ) -> None:
        if offload not in {"auto", "on", "off"}:
            raise ValueError(f"Unsupported offload mode: {offload!r}")
        self.stage0_config = stage0
        self.stage1_config = stage1
        self.stage2_config = stage2
        self.seed = self._validate_seed(seed)
        self.offload = offload
        self.fill_holes = bool(fill_holes)
        self.image_size = int(image_size)
        self.output_dir = Path(output_dir)
        self.device = self._resolve_device(device)
        self._offload_enabled = offload == "on" or (
            offload == "auto" and self.device.type == "cuda"
        )
        self._stage0: Stage0Model | None = None
        self._stage1: Stage1Model | None = None
        self._stage2: Stage2Model | None = None

    @classmethod
    def from_config(
        cls,
        config: InferenceConfig,
        *,
        device: str | torch.device | None = None,
    ) -> "TaoFlowForgePipeline":
        return cls(
            stage0=config.stage0,
            stage1=config.stage1,
            stage2=config.stage2,
            seed=config.seed,
            offload=config.offload,
            fill_holes=config.fill_holes,
            image_size=config.image_size,
            output_dir=config.output_dir,
            device=device,
        )

    @staticmethod
    def _validate_seed(seed: int) -> int:
        seed = int(seed)
        if not 0 <= seed < 2**32:
            raise ValueError("seed must lie in [0, 2**32)")
        return seed

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        resolved = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return resolved

    def _require_config(self, stage: int) -> Stage0Config | Stage1Config | Stage2Config:
        config = getattr(self, f"stage{stage}_config")
        if config is None:
            raise RuntimeError(f"Stage {stage} configuration is required")
        return config

    def load_models(self, stages: Iterable[int] = (0, 1, 2)) -> None:
        """Load requested models before seeding so loading cannot shift sampling RNG."""
        requested = set(int(stage) for stage in stages)
        invalid = requested - {0, 1, 2}
        if invalid:
            raise ValueError(f"Unsupported stage numbers: {sorted(invalid)}")

        if 0 in requested and self._stage0 is None:
            config = self._require_config(0)
            assert isinstance(config, Stage0Config)
            self._stage0 = load_stage0_model(
                config.checkpoint,
                config.vae_checkpoint,
                config.latent_norm,
                "cpu",
            )
            if config.compile_model:
                try:
                    self._stage0.denoiser = torch.compile(
                        self._stage0.denoiser,
                        dynamic=False,
                    )
                except Exception as error:
                    warnings.warn(
                        f"Stage 0 torch.compile failed; using eager mode: {error}",
                        stacklevel=2,
                    )
        if 1 in requested and self._stage1 is None:
            config = self._require_config(1)
            assert isinstance(config, Stage1Config)
            self._stage1 = load_stage1_model(config.checkpoint, "cpu")
        if 2 in requested and self._stage2 is None:
            config = self._require_config(2)
            assert isinstance(config, Stage2Config)
            self._stage2 = load_stage2_model(config.checkpoint, "cpu")

        if not self._offload_enabled:
            if 0 in requested and self._stage0 is not None:
                self._stage0.to(self.device)
            if 1 in requested and self._stage1 is not None:
                self._stage1.prepare_device(self.device)
            if 2 in requested and self._stage2 is not None:
                self._stage2.to(self.device)

    def _prepare_image(self, image: ImageInput | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(image):
            return prepare_image(image, self.image_size)
        tensor = image.detach().to(device="cpu", dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[0] != 3:
            raise ValueError(f"Expected image shape (3,H,W), got {tuple(tensor.shape)}")
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )[0]
        if not torch.isfinite(tensor).all():
            raise ValueError("Image tensor contains non-finite values")
        return tensor.clamp(0.0, 1.0).contiguous()

    def _activate_stage(self, stage: int) -> None:
        if not self._offload_enabled:
            return
        if stage != 0 and self._stage0 is not None:
            self._stage0.to("cpu")
        if stage != 1 and self._stage1 is not None:
            self._stage1.offload()
        if stage != 2 and self._stage2 is not None:
            self._stage2.to("cpu")
        if stage == 0 and self._stage0 is not None:
            self._stage0.to(self.device)
        elif stage == 1 and self._stage1 is not None:
            self._stage1.prepare_device(self.device)
        elif stage == 2 and self._stage2 is not None:
            self._stage2.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _infer_stage0(self, image: torch.Tensor, show_progress: bool) -> np.ndarray:
        config = self._require_config(0)
        assert isinstance(config, Stage0Config) and self._stage0 is not None
        self._activate_stage(0)
        coordinates = self._stage0.infer(
            image,
            num_steps=config.num_steps,
            cfg_scale=config.cfg_scale,
            time_shift=config.t_shift,
            occupancy_threshold=config.occupancy_threshold,
            local_max_keep=config.local_max_keep,
            max_cells=config.max_cells,
            show_progress=show_progress,
        )
        return coordinates.detach().cpu().numpy().astype(np.int64, copy=False)

    @torch.no_grad()
    def _infer_stage1(
        self,
        image: torch.Tensor,
        coordinates: np.ndarray,
        show_progress: bool,
    ) -> np.ndarray:
        config = self._require_config(1)
        assert isinstance(config, Stage1Config) and self._stage1 is not None
        self._activate_stage(1)
        vertices = self._stage1.infer(
            image,
            torch.from_numpy(np.asarray(coordinates, dtype=np.int64)),
            num_steps=config.num_steps,
            cfg_scale=config.cfg_scale,
            time_shift=config.t_shift,
            threshold=config.occupancy_threshold,
            max_vertices=config.max_vertices,
            show_progress=show_progress,
        )
        return vertices.detach().cpu().numpy().astype(np.float32, copy=False)

    @torch.no_grad()
    def _infer_stage2(
        self,
        image: torch.Tensor,
        vertices: np.ndarray,
        show_progress: bool,
    ) -> trimesh.Trimesh:
        config = self._require_config(2)
        assert isinstance(config, Stage2Config) and self._stage2 is not None
        self._activate_stage(2)
        return self._stage2.infer(
            image,
            vertices,
            num_steps=config.num_steps,
            cfg_scale=config.cfg_scale,
            time_shift=config.t_shift,
            edge_threshold=config.edge_threshold,
            show_progress=show_progress,
        )

    def run_stage0(
        self,
        image: ImageInput | torch.Tensor,
        artifact_path: str | Path,
        *,
        seed: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        self.load_models((0,))
        image_tensor = self._prepare_image(image)
        use_seed = self.seed if seed is None else self._validate_seed(seed)
        seed_all(use_seed)
        coordinates = self._infer_stage0(image_tensor, show_progress)
        save_stage_artifact(
            artifact_path,
            stage=0,
            seed=use_seed,
            outputs={"coordinates": coordinates},
            random_state=capture_random_state(),
        )
        return coordinates

    def run_stage1(
        self,
        image: ImageInput | torch.Tensor,
        stage0_artifact: str | Path,
        artifact_path: str | Path,
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        stage, seed, outputs, random_state = load_stage_artifact(stage0_artifact)
        if stage != 0 or "coordinates" not in outputs:
            raise ValueError("Stage 1 requires a Stage 0 artifact with coordinates")
        self.load_models((1,))
        image_tensor = self._prepare_image(image)
        restore_random_state(random_state)
        vertices = self._infer_stage1(image_tensor, outputs["coordinates"], show_progress)
        save_stage_artifact(
            artifact_path,
            stage=1,
            seed=seed,
            outputs={"vertices": vertices},
            random_state=capture_random_state(),
        )
        return vertices

    def run_stage2(
        self,
        image: ImageInput | torch.Tensor,
        stage1_artifact: str | Path,
        artifact_path: str | Path,
        *,
        show_progress: bool = True,
    ) -> trimesh.Trimesh:
        stage, seed, outputs, random_state = load_stage_artifact(stage1_artifact)
        if stage != 1 or "vertices" not in outputs:
            raise ValueError("Stage 2 requires a Stage 1 artifact with vertices")
        self.load_models((2,))
        image_tensor = self._prepare_image(image)
        restore_random_state(random_state)
        mesh = self._infer_stage2(image_tensor, outputs["vertices"], show_progress)
        save_stage_artifact(
            artifact_path,
            stage=2,
            seed=seed,
            outputs={"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces)},
            random_state=capture_random_state(),
        )
        return mesh

    def iter_run(
        self,
        image: ImageInput | torch.Tensor,
        *,
        output_dir: str | Path | None = None,
        seed: int | None = None,
        resume_from: str | Path | None = None,
        fill_holes: bool | None = None,
        show_progress: bool = True,
    ) -> Iterator[PipelineUpdate]:
        """Run the pipeline and yield only at RNG-safe stage boundaries."""
        destination = Path(output_dir) if output_dir is not None else self.output_dir
        destination.mkdir(parents=True, exist_ok=True)
        image_tensor = self._prepare_image(image)
        image_array = (
            image_tensor.permute(1, 2, 0).mul(255).round().byte().numpy()
        )
        Image.fromarray(image_array).save(destination / "input.png")

        resume_stage = -1
        outputs: dict[str, np.ndarray] = {}
        boundary_state = None
        use_seed = self.seed if seed is None else self._validate_seed(seed)
        artifact_paths: dict[int, Path] = {}
        if resume_from is not None:
            resume_stage, use_seed, outputs, boundary_state = load_stage_artifact(resume_from)
            if resume_stage not in {0, 1, 2}:
                raise ValueError(f"Unsupported resume artifact stage: {resume_stage}")
            artifact_paths[resume_stage] = Path(resume_from)

        needed = tuple(stage for stage in (0, 1, 2) if stage > resume_stage)
        self.load_models(needed)
        if boundary_state is None:
            seed_all(use_seed)
        else:
            restore_random_state(boundary_state)

        metadata: dict[str, Any] = {
            "seed": int(use_seed),
            "device": str(self.device),
            "offload": self.offload,
            "image_size": self.image_size,
            "resumed_from": str(resume_from) if resume_from is not None else None,
            "stages": {},
        }
        coordinates = outputs.get("coordinates") if resume_stage == 0 else None
        vertices = outputs.get("vertices") if resume_stage == 1 else None
        mesh = None
        if resume_stage == 2:
            if "vertices" not in outputs or "faces" not in outputs:
                raise ValueError("Stage 2 artifact must contain vertices and faces")
            mesh = trimesh.Trimesh(
                vertices=outputs["vertices"],
                faces=outputs["faces"],
                process=False,
            )

        if resume_stage < 0:
            started = time.perf_counter()
            coordinates = self._infer_stage0(image_tensor, show_progress)
            boundary_state = capture_random_state()
            artifact_paths[0] = destination / "stage0.npz"
            save_stage_artifact(
                artifact_paths[0],
                stage=0,
                seed=use_seed,
                outputs={"coordinates": coordinates},
                random_state=boundary_state,
            )
            metadata["stages"]["stage0"] = {
                "seconds": time.perf_counter() - started,
                "cells": int(len(coordinates)),
            }
            yield PipelineUpdate(0, f"Stage 0 complete: {len(coordinates)} occupied cells")
            restore_random_state(boundary_state)

        if resume_stage < 1:
            if coordinates is None:
                raise ValueError("No Stage 0 coordinates are available")
            started = time.perf_counter()
            vertices = self._infer_stage1(image_tensor, coordinates, show_progress)
            boundary_state = capture_random_state()
            artifact_paths[1] = destination / "stage1.npz"
            save_stage_artifact(
                artifact_paths[1],
                stage=1,
                seed=use_seed,
                outputs={"vertices": vertices},
                random_state=boundary_state,
            )
            metadata["stages"]["stage1"] = {
                "seconds": time.perf_counter() - started,
                "vertices": int(len(vertices)),
            }
            yield PipelineUpdate(1, f"Stage 1 complete: {len(vertices)} vertices")
            restore_random_state(boundary_state)

        if resume_stage < 2:
            if vertices is None:
                raise ValueError("No Stage 1 vertices are available")
            started = time.perf_counter()
            mesh = self._infer_stage2(image_tensor, vertices, show_progress)
            boundary_state = capture_random_state()
            artifact_paths[2] = destination / "stage2.npz"
            save_stage_artifact(
                artifact_paths[2],
                stage=2,
                seed=use_seed,
                outputs={
                    "vertices": np.asarray(mesh.vertices),
                    "faces": np.asarray(mesh.faces),
                },
                random_state=boundary_state,
            )
            metadata["stages"]["stage2"] = {
                "seconds": time.perf_counter() - started,
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            }
            yield PipelineUpdate(2, f"Stage 2 complete: {len(mesh.faces)} faces")

        assert mesh is not None
        raw_mesh_path = export_obj(mesh, destination / "mesh_raw.obj")
        do_fill = self.fill_holes if fill_holes is None else bool(fill_holes)
        post_started = time.perf_counter()
        if do_fill:
            final_mesh, postprocess = fill_mesh_holes(mesh)
        else:
            final_mesh, postprocess = mesh, {"status": "disabled"}
        final_mesh_path = export_obj(final_mesh, destination / "mesh.obj")
        glb_path, normalization = export_normalized_glb(
            final_mesh,
            destination / "mesh.glb",
        )
        metadata["postprocess"] = postprocess
        metadata["postprocess_seconds"] = time.perf_counter() - post_started
        metadata["normalization"] = normalization
        metadata["artifacts"] = {
            str(stage): str(path) for stage, path in artifact_paths.items()
        }
        metadata["config"] = {
            "stage0": asdict(self.stage0_config) if self.stage0_config else None,
            "stage1": asdict(self.stage1_config) if self.stage1_config else None,
            "stage2": asdict(self.stage2_config) if self.stage2_config else None,
        }
        metadata_path = destination / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, default=str),
            encoding="utf-8",
        )
        result = PipelineResult(
            mesh=final_mesh,
            output_dir=destination,
            raw_mesh_path=raw_mesh_path,
            final_mesh_path=final_mesh_path,
            glb_path=glb_path,
            artifact_paths=artifact_paths,
            metadata_path=metadata_path,
        )
        yield PipelineUpdate(None, "Pipeline complete", result)

    def run(self, image: ImageInput | torch.Tensor, **kwargs: Any) -> PipelineResult:
        """Run the complete pipeline and return its final result."""
        result = None
        for update in self.iter_run(image, **kwargs):
            if update.result is not None:
                result = update.result
        if result is None:
            raise RuntimeError("Pipeline terminated without a result")
        return result

    def close(self) -> None:
        """Move loaded models to CPU and release cached CUDA allocations."""
        if self._stage0 is not None:
            self._stage0.to("cpu")
        if self._stage1 is not None:
            self._stage1.offload()
        if self._stage2 is not None:
            self._stage2.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "TaoFlowForgePipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
