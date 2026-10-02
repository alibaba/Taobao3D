"""Inference-only no-query topology decoder."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention block with checkpoint-compatible names."""

    def __init__(
        self,
        hidden_dim: int = 1_024,
        num_heads: int = 8,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        ffn_hidden = hidden_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        normalized = self.norm_attn(x)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.ffn(self.norm_ffn(x))


class TopologyDecoder(nn.Module):
    """Decode per-vertex latents into edges and oriented triangle faces."""

    def __init__(self):
        super().__init__()
        latent_dim = 32
        hidden_dim = 1_024
        edge_input_dim = 256
        edge_hidden_dim = 512
        self.edge_start = 512
        self.edge_end = self.edge_start + edge_input_dim
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.latent_self_layers = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, 8, dropout=0.1) for _ in range(6)]
        )
        self.latent_self_norm = nn.LayerNorm(hidden_dim)
        self.edge_feat_head = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(edge_hidden_dim * 2, edge_hidden_dim),
        )
        self.edge_src_proj = nn.Linear(
            edge_hidden_dim, edge_hidden_dim, bias=False
        )
        self.edge_dst_proj = nn.Linear(
            edge_hidden_dim, edge_hidden_dim, bias=False
        )
        self.edge_bias = nn.Parameter(torch.zeros(1))
        self.normal_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 3),
        )

    def forward(self, latent_tokens):
        hidden = self.latent_proj(latent_tokens)
        key_padding_mask = torch.zeros(
            latent_tokens.shape[:2],
            dtype=torch.bool,
            device=latent_tokens.device,
        )
        for layer in self.latent_self_layers:
            hidden = layer(hidden, key_padding_mask=key_padding_mask)
        hidden = self.latent_self_norm(hidden)

        edge_input = hidden[..., self.edge_start : self.edge_end]
        edge_features = self.edge_feat_head(edge_input)
        edge_source = self.edge_src_proj(edge_features)
        edge_target = self.edge_dst_proj(edge_features)
        edge_logits = torch.bmm(
            edge_source, edge_target.transpose(1, 2)
        ) + self.edge_bias
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2.0
        diagonal = torch.eye(
            edge_logits.shape[1],
            device=edge_logits.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        edge_logits = edge_logits.masked_fill(diagonal, -1e9)
        normals = F.normalize(self.normal_head(hidden), dim=-1)
        vertex_mask = torch.ones(
            hidden.shape[:2], dtype=torch.bool, device=hidden.device
        )
        return {
            "edge_logits": edge_logits,
            "normal_pred": normals,
            "vert_mask": vertex_mask,
        }

    @staticmethod
    def extract_mesh(
        edge_logits,
        *,
        threshold: float = 0.5,
        vertex_normals=None,
        vertex_coordinates=None,
    ):
        edge_probability = torch.sigmoid(edge_logits)
        edge_matrix = (edge_probability > threshold).float()
        edge_matrix.fill_diagonal_(0.0)
        edge_matrix = ((edge_matrix + edge_matrix.T) > 0).float()
        faces = TopologyDecoder._extract_triangles_from_edges(edge_matrix)
        if (
            vertex_normals is not None
            and vertex_coordinates is not None
            and faces
        ):
            faces = TopologyDecoder._orient_faces_by_normals(
                faces, vertex_coordinates, vertex_normals
            )
        return faces

    @staticmethod
    def _extract_triangles_from_edges(edge_matrix, max_faces: int = 100_000):
        adjacency = edge_matrix > 0.5
        vertex_count = adjacency.shape[0]
        adjacency_float = adjacency.float()
        edge_count = int(adjacency_float.sum().item()) // 2
        if edge_count > max_faces * 3:
            degree = adjacency_float.sum(dim=1)
            max_degree = max(
                6,
                int((2 * max_faces / max(vertex_count, 1)) ** 0.5) + 1,
            )
            if degree.max().item() > max_degree:
                _, top_indices = adjacency_float.topk(
                    min(max_degree, vertex_count), dim=1
                )
                sparse = torch.zeros_like(adjacency_float)
                sparse.scatter_(1, top_indices, 1.0)
                adjacency = (sparse + sparse.T) > 0

        adjacency_numpy = adjacency.cpu().numpy()
        faces = []
        for first in range(vertex_count):
            if len(faces) >= max_faces:
                break
            neighbors = [
                second
                for second in range(first + 1, vertex_count)
                if adjacency_numpy[first, second]
            ]
            for second in neighbors:
                if len(faces) >= max_faces:
                    break
                for third in range(second + 1, vertex_count):
                    if (
                        adjacency_numpy[first, third]
                        and adjacency_numpy[second, third]
                    ):
                        faces.append([first, second, third])
                        if len(faces) >= max_faces:
                            break
        return faces

    @staticmethod
    def _orient_faces_by_normals(faces, coordinates, normals):
        coordinates = coordinates.detach().cpu().numpy()
        normals = normals.detach().cpu().numpy()
        for index, face in enumerate(faces):
            first, second, third = face
            v0, v1, v2 = coordinates[first], coordinates[second], coordinates[third]
            face_normal = np.cross(v1 - v0, v2 - v0)
            average_normal = normals[first] + normals[second] + normals[third]
            if np.dot(face_normal, average_normal) < 0:
                faces[index] = [first, third, second]
        return faces
