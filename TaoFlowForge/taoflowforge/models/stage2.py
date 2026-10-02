"""Stage 2 feature-DiT model and strict legacy checkpoint loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import trimesh
from torch import nn

from .dino import DinoConditioner
from .dit import RefineDiT
from .flow import EulerFlowSampler
from .topology_decoder import TopologyDecoder


class Stage2Model(nn.Module):
    """Inference-only image-conditioned topology generator."""

    latent_dim = 32
    latent_norm_clamp = 0.1

    def __init__(self):
        super().__init__()
        self._dino_encoder = DinoConditioner(image_size=1_024)
        self.dit = RefineDiT()
        self.vertex_decoder = TopologyDecoder()
        self.register_buffer("_latent_norm_mean", torch.zeros(self.latent_dim))
        self.register_buffer("_latent_norm_std", torch.ones(self.latent_dim))
        self.flow_sampler = EulerFlowSampler(prediction_type="velocity")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a ``(3,H,W)`` image tensor in ``[0,1]``."""
        device = next(self.parameters()).device
        image_batch = image.unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            features = self._dino_encoder(image_batch)
        return features.float()

    def denormalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        scale = self._latent_norm_std.clamp(min=self.latent_norm_clamp)
        return latent * scale + self._latent_norm_mean

    @torch.no_grad()
    def sample_latent(
        self,
        condition: torch.Tensor,
        voxel_coordinates: torch.Tensor,
        *,
        num_steps: int = 50,
        cfg_scale: float = 3.0,
        time_shift: float = 1.0,
        initial_state: torch.Tensor | None = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        contexts = {"main": condition}
        unconditional_contexts = None
        if cfg_scale != 1.0:
            unconditional_contexts = {"main": torch.zeros_like(condition)}
        model_kwargs = {
            "contexts": contexts,
            "voxel_cond": voxel_coordinates,
        }
        unconditional_kwargs = None
        if unconditional_contexts is not None:
            unconditional_kwargs = {
                "contexts": unconditional_contexts,
                "voxel_cond": voxel_coordinates,
            }
        device = condition.device
        shape = (condition.shape[0], voxel_coordinates.shape[1], self.latent_dim)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return self.flow_sampler.sample(
                self.dit,
                shape,
                num_steps=num_steps,
                device=device,
                dtype=condition.dtype,
                model_kwargs=model_kwargs,
                cfg_scale=cfg_scale,
                unconditional_kwargs=unconditional_kwargs,
                time_shift=time_shift,
                initial_state=initial_state,
                show_progress=show_progress,
                description="[Stage 2/3] topology",
            )

    @staticmethod
    def prepare_vertices(
        vertices: torch.Tensor | np.ndarray,
        resolution: int = 512,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize and sort vertices by the training-time ``(z,x,y)`` order."""
        resolution = int(resolution)
        if resolution < 2:
            raise ValueError("voxel resolution must be at least 2")
        if torch.is_tensor(vertices):
            vertices_numpy = vertices.detach().cpu().numpy()
        else:
            vertices_numpy = vertices
        vertices_numpy = np.array(vertices_numpy, dtype=np.float32, copy=True)
        if vertices_numpy.ndim != 2 or vertices_numpy.shape[1] != 3:
            raise ValueError(
                f"Stage 2 vertices must have shape (N,3), got {vertices_numpy.shape}"
            )
        if vertices_numpy.shape[0] == 0:
            raise ValueError("Stage 2 vertices are empty")
        if not np.isfinite(vertices_numpy).all():
            raise ValueError("Stage 2 vertices contain non-finite values")

        voxel_coordinates = (
            (vertices_numpy + 1.0) / 2.0 * float(resolution - 1)
        )
        voxel_coordinates = np.clip(
            np.round(voxel_coordinates),
            0,
            resolution - 1,
        ).astype(np.int64)
        sort_indices = np.lexsort(
            (
                voxel_coordinates[:, 1],
                voxel_coordinates[:, 0],
                voxel_coordinates[:, 2],
            )
        )
        sorted_vertices = torch.from_numpy(vertices_numpy[sort_indices].copy())
        sorted_voxels = torch.from_numpy(voxel_coordinates[sort_indices].copy())
        return sorted_vertices, sorted_voxels.unsqueeze(0)

    @torch.no_grad()
    def decode_mesh(
        self,
        denormalized_latent: torch.Tensor,
        sorted_vertices: torch.Tensor,
        *,
        edge_threshold: float = 0.5,
    ) -> trimesh.Trimesh:
        """Decode denormalized vertex latents into a mesh without reordering."""
        if denormalized_latent.ndim != 3 or denormalized_latent.shape[0] != 1:
            raise ValueError(
                "Stage 2 latent must have shape (1,N,32), got "
                f"{tuple(denormalized_latent.shape)}"
            )
        if denormalized_latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Stage 2 latent channel count must be {self.latent_dim}")
        if tuple(sorted_vertices.shape) != (denormalized_latent.shape[1], 3):
            raise ValueError(
                "Sorted vertex shape does not match the latent token count: "
                f"{tuple(sorted_vertices.shape)} versus {denormalized_latent.shape[1]}"
            )

        device = denormalized_latent.device
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            decoder_output = self.vertex_decoder(denormalized_latent)
        decoder_output = {
            key: (
                value.float()
                if torch.is_tensor(value) and value.is_floating_point()
                else value
            )
            for key, value in decoder_output.items()
        }
        coordinates = sorted_vertices.to(device=device, dtype=torch.float32)
        normals = decoder_output.get("normal_pred")
        faces = self.vertex_decoder.extract_mesh(
            decoder_output["edge_logits"][0],
            threshold=edge_threshold,
            vertex_normals=normals[0] if normals is not None else None,
            vertex_coordinates=coordinates,
        )
        faces_numpy = np.asarray(faces, dtype=np.int64)
        if faces_numpy.size == 0:
            faces_numpy = np.zeros((0, 3), dtype=np.int64)
        else:
            faces_numpy = faces_numpy.reshape(-1, 3)
            valid = faces_numpy.max(axis=1) < sorted_vertices.shape[0]
            faces_numpy = faces_numpy[valid]

        mesh = trimesh.Trimesh(
            vertices=sorted_vertices.cpu().numpy(),
            faces=faces_numpy,
            process=False,
        )
        if len(mesh.faces) > 0:
            nondegenerate = mesh.area_faces > 1e-10
            if not nondegenerate.all():
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices,
                    faces=mesh.faces[nondegenerate],
                    process=False,
                )
        return mesh

    @torch.no_grad()
    def infer(
        self,
        image: torch.Tensor,
        vertices: torch.Tensor | np.ndarray,
        *,
        voxel_resolution: int = 512,
        num_steps: int = 50,
        cfg_scale: float = 3.0,
        time_shift: float = 1.0,
        edge_threshold: float = 0.5,
        initial_state: torch.Tensor | None = None,
        show_progress: bool = True,
    ) -> trimesh.Trimesh:
        """Generate topology for a Stage 1 vertex cloud."""
        condition = self.encode_image(image)
        sorted_vertices, voxel_coordinates = self.prepare_vertices(
            vertices,
            voxel_resolution,
        )
        voxel_coordinates = voxel_coordinates.to(condition.device)
        normalized_latent = self.sample_latent(
            condition,
            voxel_coordinates,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            time_shift=time_shift,
            initial_state=initial_state,
            show_progress=show_progress,
        )
        denormalized_latent = self.denormalize_latent(normalized_latent)
        del normalized_latent
        if condition.device.type == "cuda":
            torch.cuda.empty_cache()
        return self.decode_mesh(
            denormalized_latent,
            sorted_vertices,
            edge_threshold=edge_threshold,
        )


_DINO_LAYER_SOURCE = "_dino_encoder.model.layer."
_DINO_LAYER_TARGET = "_dino_encoder.model.model.layer."
_ALLOWED_UNUSED_PREFIXES = (
    "vertex_encoder.",
    "loss_fn.",
    "vertex_decoder.coord_head.",
    "vertex_decoder.perm_head.",
)
_ALLOWED_UNUSED_KEYS = {
    "vertex_decoder.gumbel_temperature",
    "vertex_decoder.adjacency_threshold",
}


def _map_checkpoint_key(key: str) -> str:
    if key.startswith(_DINO_LAYER_SOURCE):
        return _DINO_LAYER_TARGET + key[len(_DINO_LAYER_SOURCE) :]
    return key


def load_stage2_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Stage2Model:
    """Load all inference-required tensors and reject silent mismatches."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    source = checkpoint.get("model_state_dict", checkpoint)
    with torch.device("meta"):
        model = Stage2Model()
    expected = model.state_dict()
    mapped = {}
    unexpected = []
    shape_mismatches = []
    for source_key, value in source.items():
        target_key = _map_checkpoint_key(source_key)
        if target_key in expected:
            if target_key in mapped:
                raise RuntimeError(
                    f"Stage 2 checkpoint maps multiple tensors to {target_key!r}"
                )
            expected_shape = tuple(expected[target_key].shape)
            actual_shape = tuple(value.shape)
            if actual_shape != expected_shape:
                shape_mismatches.append(
                    f"{source_key}: {actual_shape} != {expected_shape}"
                )
            else:
                mapped[target_key] = value
        elif not (
            source_key in _ALLOWED_UNUSED_KEYS
            or source_key.startswith(_ALLOWED_UNUSED_PREFIXES)
        ):
            unexpected.append(source_key)

    if unexpected:
        raise RuntimeError(
            "Stage 2 checkpoint has unexpected tensors: "
            + ", ".join(sorted(unexpected)[:20])
        )
    if shape_mismatches:
        raise RuntimeError(
            "Stage 2 checkpoint has incompatible tensor shapes: "
            + "; ".join(shape_mismatches[:20])
        )

    missing = sorted(set(expected) - set(mapped))
    if missing:
        raise RuntimeError(
            "Stage 2 checkpoint is missing inference weights: "
            + ", ".join(missing[:20])
        )

    model.load_state_dict(mapped, strict=True, assign=True)
    model._dino_encoder.materialize_nonpersistent_buffers()
    model.requires_grad_(False)
    model.eval()
    return model.to(device)
