#!/usr/bin/env python3
"""Orient mesh faces outward using visibility voting and graph cuts."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- I/O


def load_obj(path):
    """Load and triangulate OBJ geometry."""
    verts, faces = [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                idx = [i - 1 if i > 0 else len(verts) + i for i in idx]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
    return np.asarray(verts, np.float64), np.asarray(faces, np.int64).reshape(-1, 3)


def save_obj_flat(path, V, F, N):
    """Write an OBJ with one flat normal per face."""
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    parts = ["# normal_correct.py : outward flat normals\n"]
    parts.append("\n".join("v %.9g %.9g %.9g" % tuple(p) for p in V) + "\n")
    parts.append("\n".join("vn %.6f %.6f %.6f" % tuple(n) for n in N) + "\n")
    parts.append(
        "\n".join(
            "f %d//%d %d//%d %d//%d" % (a, k, b, k, c, k)
            for (a, b, c), k in zip(F + 1, np.arange(1, len(F) + 1))
        )
        + "\n"
    )
    with open(path, "w") as fh:
        fh.write("".join(parts))


# ---------------------------------------------------------- Geometry


def unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-300)


def face_normals(V, F):
    """Return unnormalized face normals and areas."""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    return cr, 0.5 * np.linalg.norm(cr, axis=1)


def bounding_sphere(V, pad=0.02):
    """Approximate the bounding sphere with Ritter's algorithm."""
    p = V[np.argmin(V[:, 0])]
    q = V[np.argmax(np.sum((V - p) ** 2, axis=1))]
    r = V[np.argmax(np.sum((V - q) ** 2, axis=1))]
    c = 0.5 * (q + r)
    rad = 0.5 * float(np.linalg.norm(q - r))
    for _ in range(16):
        d = np.linalg.norm(V - c, axis=1)
        i = int(np.argmax(d))
        if d[i] <= rad * (1 + 1e-12):
            break
        newrad = 0.5 * (rad + d[i])
        c = c + (newrad - rad) / max(d[i], 1e-300) * (V[i] - c)
        rad = newrad
    rad = max(float(np.max(np.linalg.norm(V - c, axis=1))), 1e-12)
    return c, rad * (1.0 + pad)


def tangent_frame(n):
    """Build orthogonal tangent frames from unit normals."""
    sign = np.where(n[:, 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t = np.stack([1.0 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], axis=1)
    s = np.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], axis=1)
    return t, s


def cosine_hemisphere(k, rng):
    """Sample cosine-weighted hemisphere directions."""
    i = np.arange(k, dtype=np.float64)
    u1 = (i + 0.5) / k
    bits = np.arange(k, dtype=np.uint32)
    for shift, mask_a, mask_b in (
        (16, 0xFFFFFFFF, 0),
        (1, 0x55555555, 0xAAAAAAAA),
        (2, 0x33333333, 0xCCCCCCCC),
        (4, 0x0F0F0F0F, 0xF0F0F0F0),
        (8, 0x00FF00FF, 0xFF00FF00),
    ):
        if mask_b == 0:
            bits = (bits << np.uint32(shift)) | (bits >> np.uint32(shift))
        else:
            bits = ((bits & np.uint32(mask_a)) << np.uint32(shift)) | (
                (bits & np.uint32(mask_b)) >> np.uint32(shift)
            )
    u2 = (bits.astype(np.float64) * 2.3283064365386963e-10 + rng.random()) % 1.0
    r, phi = np.sqrt(u1), 2.0 * np.pi * u2
    return np.stack(
        [r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1.0 - u1))], 1
    )


def fibonacci_sphere(d, rng=None):
    """Sample near-uniform directions on a sphere."""
    i = np.arange(d, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / d
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * (np.pi * (3.0 - np.sqrt(5.0)))
    if rng is not None:
        phi = phi + 2.0 * np.pi * rng.random()
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    if rng is None:
        return dirs
    q = rng.normal(size=4)                             # Random unit quaternion
    q /= max(np.linalg.norm(q), 1e-300)
    w, x, y, zq = q
    R = np.array([
        [1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)],
        [2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)],
        [2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return dirs @ R.T


# ---------------------------------------------------------- Stage 0


@dataclass
class Prepared:
    wmap: np.ndarray          # Original-to-welded vertex map
    WV: np.ndarray            # Welded vertices
    WF: np.ndarray            # Welded faces
    valid: np.ndarray         # Non-degenerate face mask
    repr_of: np.ndarray       # Duplicate representative per face
    same_wind: np.ndarray     # Winding relative to the representative
    bvh_id: np.ndarray        # Face-to-BVH primitive map
    prim2face: np.ndarray     # BVH primitive-to-face map
    center: np.ndarray
    radius: float
    diag: float
    n_unit: np.ndarray        # Input face normals
    area: np.ndarray
    fc: np.ndarray            # Face centroids
    stats: dict = field(default_factory=dict)


def prepare(V, F, weld_tol_rel=1e-6):
    """Prepare welded geometry and face metadata."""
    center, radius = bounding_sphere(V)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))

    # Weld coincident vertices by quantized position.
    tol = max(diag * weld_tol_rel, 1e-12)
    key = np.round((V - V.min(0)) / tol).astype(np.int64)
    _, wfirst, wmap = np.unique(key, axis=0, return_index=True, return_inverse=True)
    wmap = np.asarray(wmap).ravel()
    WV, WF = V[wfirst], wmap[F]

    cr, area = face_normals(V, F)
    valid = area > 1e-12 * max(diag, 1e-12) ** 2
    valid &= (WF[:, 0] != WF[:, 1]) & (WF[:, 1] != WF[:, 2]) & (WF[:, 2] != WF[:, 0])

    # Merge faces sharing the same welded vertex triplet.
    _, first, inv = np.unique(np.sort(WF, axis=1), axis=0,
                              return_index=True, return_inverse=True)
    repr_of = first[np.asarray(inv).ravel()]

    def cyc(x):                                   # Canonical cyclic order
        r = np.argmin(x, axis=1)
        return np.stack([x[np.arange(len(x)), (r + k) % 3] for k in range(3)], 1)

    same_wind = (cyc(WF) == cyc(WF[repr_of])).all(axis=1)

    in_bvh = valid & (repr_of == np.arange(len(F)))
    bvh_id = np.full(len(F), -1, np.int64)
    bvh_id[in_bvh] = np.arange(int(in_bvh.sum()))
    prim2face = np.nonzero(in_bvh)[0]              # BVH primitive order
    bvh_id = bvh_id[repr_of]                      # Reuse representative primitives
    bvh_id[~valid] = -1

    stats = dict(
        n_vert=len(V), n_vert_welded=len(WV), n_face=len(F),
        n_degenerate=int((~valid).sum()),
        n_duplicate=int((repr_of != np.arange(len(F))).sum()),
        n_bvh_tri=int(in_bvh.sum()),
    )
    return Prepared(wmap, WV, WF, valid, repr_of, same_wind, bvh_id, prim2face,
                    center, radius, diag, unit(cr), area, V[F].mean(axis=1),
                    stats), in_bvh


# ---------------------------------------------------------- Stage 1


class RayEngine:
    """Use Open3D ray casting with a trimesh fallback."""

    def __init__(self, V, F):
        try:
            import open3d as o3d

            self._o3d = o3d
            self.scene = o3d.t.geometry.RaycastingScene()
            self.scene.add_triangles(
                o3d.core.Tensor(np.ascontiguousarray(V, np.float32)),
                o3d.core.Tensor(np.ascontiguousarray(F, np.uint32)),
            )
            self.backend = "open3d-embree"
        except Exception:
            import trimesh

            self.inter = trimesh.Trimesh(vertices=V, faces=F, process=False).ray
            self.backend = "trimesh"

    def first_hit(self, origins, dirs):
        """Return hit distances and primitive IDs."""
        if self.backend.startswith("open3d"):
            rays = np.concatenate(
                [np.asarray(origins, np.float32), np.asarray(dirs, np.float32)], 1
            )
            res = self.scene.cast_rays(self._o3d.core.Tensor(rays))
            t = res["t_hit"].numpy().astype(np.float64)
            pid = res["primitive_ids"].numpy().astype(np.int64)
            pid[~np.isfinite(t)] = -1
            return t, pid
        loc, ray_idx, tri_idx = self.inter.intersects_location(
            origins, dirs, multiple_hits=False
        )
        t = np.full(len(origins), np.inf)
        pid = np.full(len(origins), -1, np.int64)
        if len(ray_idx):
            t[ray_idx] = np.linalg.norm(loc - origins[ray_idx], axis=1)
            pid[ray_idx] = tri_idx
        return t, pid


def visibility_vote(prep, engine, n_rays=32, chunk=400_000, seed=0):
    """Estimate bidirectional per-face visibility."""
    rng = np.random.default_rng(seed)
    nf = len(prep.n_unit)
    vis = np.zeros((nf, 2))
    eps = max(prep.diag * 1e-5, 1e-12)
    local = cosine_hemisphere(n_rays, rng)
    idx_all = np.nonzero(prep.valid)[0]
    per_face = n_rays * 2
    step = max(1, chunk // per_face)

    for beg in range(0, len(idx_all), step):
        fid = idx_all[beg: beg + step]
        n = prep.n_unit[fid]
        t_ax, s_ax = tangent_frame(n)
        d_pos = (local[None, :, 0, None] * t_ax[:, None, :]
                 + local[None, :, 1, None] * s_ax[:, None, :]
                 + local[None, :, 2, None] * n[:, None, :])
        dirs = np.concatenate([d_pos, -d_pos], axis=1).reshape(-1, 3)
        org = np.repeat(prep.fc[fid], per_face, axis=0) + eps * dirs
        t, pid = engine.first_hit(org, dirs)
        self_id = np.repeat(prep.bvh_id[fid], per_face)
        escaped = ((~np.isfinite(t)) | (pid == self_id)).reshape(len(fid), 2, n_rays)
        # Average cosine samples to estimate visibility.
        vis[fid, 0] = escaped[:, 0].mean(axis=1)
        vis[fid, 1] = escaped[:, 1].mean(axis=1)
    return vis[:, 0], vis[:, 1]


def shell_view_vote(prep, engine, n_views=64, res=128, seed=0, chunk=800_000):
    """Estimate signed visible area from shell viewpoints."""
    rng = np.random.default_rng(seed + 12345)
    nf = len(prep.n_unit)
    pos = np.zeros(nf)
    neg = np.zeros(nf)
    R = prep.radius
    px = (2.0 * R / res) ** 2                       # World area per ray
    om_all = fibonacci_sphere(n_views, rng)
    t_ax, s_ax = tangent_frame(om_all)

    g = (np.arange(res) + 0.5) / res * 2.0 - 1.0    # Jittered projection grid
    gu, gv = np.meshgrid(g, g, indexing="ij")
    gu, gv = gu.reshape(-1), gv.reshape(-1)
    step = max(1, chunk // (res * res))

    for beg in range(0, n_views, step):
        om = om_all[beg: beg + step]
        nb = len(om)
        ju = (rng.random((nb, res * res)) - 0.5) * (2.0 / res)
        jv = (rng.random((nb, res * res)) - 0.5) * (2.0 / res)
        u = (gu[None, :] + ju) * R
        v = (gv[None, :] + jv) * R
        org = (prep.center[None, None, :]
               - 1.5 * R * om[:, None, :]
               + u[:, :, None] * t_ax[beg: beg + step][:, None, :]
               + v[:, :, None] * s_ax[beg: beg + step][:, None, :]).reshape(-1, 3)
        dr = np.repeat(om, res * res, axis=0)
        _, pid = engine.first_hit(org, dr)
        hit = pid >= 0
        if not hit.any():
            continue
        f = prep.prim2face[pid[hit]]
        front = np.einsum("ij,ij->i", prep.n_unit[f], -dr[hit]) > 0
        np.add.at(pos, f[front], px)
        np.add.at(neg, f[~front], px)

    return (pos - neg) / n_views, (pos + neg) / n_views


# ---------------------------------------------------------- Adjacency graph


@dataclass
class FaceGraph:
    dst: np.ndarray       # CSR neighbors
    off: np.ndarray       # CSR offsets
    incompat: np.ndarray  # Current winding mismatch
    strong: np.ndarray    # Manifold-edge constraint
    elen: np.ndarray      # Shared-edge length
    bd_per_face: np.ndarray
    nm_per_face: np.ndarray
    stats: dict


def build_face_graph(prep, F):
    """Build manifold and non-manifold face adjacency."""
    WF = prep.WF
    keep = np.nonzero(prep.valid & (prep.repr_of == np.arange(len(F))))[0]
    he_f = np.repeat(keep, 3)
    he_a = WF[keep][:, [0, 1, 2]].reshape(-1)
    he_b = WF[keep][:, [1, 2, 0]].reshape(-1)
    he_o = WF[keep][:, [2, 0, 1]].reshape(-1)          # Opposite vertices
    lo, hi = np.minimum(he_a, he_b), np.maximum(he_a, he_b)
    fwd = he_a < he_b
    order = np.argsort(lo * (len(prep.WV) + 1) + hi, kind="stable")
    ekey_s = (lo * (len(prep.WV) + 1) + hi)[order]
    he_f_s, fwd_s = he_f[order], fwd[order]
    lo_s, hi_s, he_o_s = lo[order], hi[order], he_o[order]
    elen_s = np.linalg.norm(prep.WV[hi_s] - prep.WV[lo_s], axis=1)

    starts = np.concatenate([[0], np.nonzero(np.diff(ekey_s))[0] + 1])
    counts = np.diff(np.concatenate([starts, [len(ekey_s)]]))
    nm_groups = np.nonzero(counts >= 3)[0]

    # Vectorized manifold edges
    mf = np.nonzero(counts == 2)[0]
    i0, i1 = starts[mf], starts[mf] + 1
    P = [np.stack([he_f_s[i0], he_f_s[i1]], 1)]
    IC = [fwd_s[i0] == fwd_s[i1]]                      # Same direction is incompatible
    ST = [np.ones(len(mf), bool)]
    EL = [elen_s[i0]]

    # Connect adjacent wedges around non-manifold edges.
    for gi in nm_groups:
        b, cnt = int(starts[gi]), int(counts[gi])
        sl = slice(b, b + cnt)
        u, v = int(lo_s[b]), int(hi_s[b])
        ax = prep.WV[v] - prep.WV[u]
        nrm = float(np.linalg.norm(ax))
        if nrm < 1e-300:
            continue
        ax /= nrm
        e1 = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = e1 - np.dot(e1, ax) * ax
        e1 /= max(np.linalg.norm(e1), 1e-300)
        e2 = np.cross(ax, e1)
        w = prep.WV[he_o_s[sl]] - prep.WV[u]
        w = w - np.outer(w @ ax, ax)
        srt = np.argsort(np.arctan2(w @ e2, w @ e1))
        fs, fw = he_f_s[sl][srt], fwd_s[sl][srt]
        nxt = np.roll(np.arange(cnt), -1)
        ok = fs != fs[nxt]
        if not ok.any():
            continue
        P.append(np.stack([fs[ok], fs[nxt][ok]], 1))
        IC.append(fw[ok] == fw[nxt][ok])
        ST.append(np.zeros(int(ok.sum()), bool))
        EL.append(np.full(int(ok.sum()), nrm))

    nf = len(F)
    bd_per_face = np.zeros(nf, np.int32)
    nm_per_face = np.zeros(nf, np.int32)
    bd = np.nonzero(counts == 1)[0]
    if len(bd):
        np.add.at(bd_per_face, he_f_s[starts[bd]], 1)
    for gi in nm_groups:
        b, cnt = int(starts[gi]), int(counts[gi])
        np.add.at(nm_per_face, he_f_s[b: b + cnt], 1)

    stats = dict(n_boundary_edge=int((counts == 1).sum()),
                 n_manifold_edge=int((counts == 2).sum()),
                 n_nonmanifold_edge=len(nm_groups))
    P = np.concatenate(P, 0)
    if len(P) == 0:
        z = np.zeros(0, np.int64)
        return FaceGraph(z, np.zeros(nf + 1, np.int64), z.astype(bool), z.astype(bool),
                         z.astype(float), bd_per_face, nm_per_face, stats)

    IC, ST = np.concatenate(IC).astype(bool), np.concatenate(ST).astype(bool)
    EL = np.concatenate(EL).astype(float)
    src = np.concatenate([P[:, 0], P[:, 1]])
    dst = np.concatenate([P[:, 1], P[:, 0]])
    ic, st, el = np.concatenate([IC, IC]), np.concatenate([ST, ST]), np.concatenate([EL, EL])
    o = np.argsort(src, kind="stable")
    src, dst, ic, st, el = src[o], dst[o], ic[o], st[o], el[o]
    off = np.zeros(nf + 1, np.int64)
    np.add.at(off, src + 1, 1)
    return FaceGraph(dst, np.cumsum(off), ic, st, el, bd_per_face, nm_per_face, stats)


# ---------------------------------------------------------- Stage 2


class ParityDSU:
    """Track relative signs with parity union-find."""

    def __init__(self, n):
        self.p = list(range(n))
        self.par = [False] * n

    def find(self, x):
        path, cum, acc = [], [], False
        while self.p[x] != x:
            path.append(x)
            cum.append(acc)
            acc ^= self.par[x]
            x = self.p[x]
        for nd, c in zip(path, cum):
            self.p[nd] = x
            self.par[nd] = acc ^ c
        return x, acc

    def union(self, a, b, rel):
        """Merge a relative-sign constraint."""
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == rel
        self.p[rb] = ra
        self.par[rb] = pa ^ pb ^ rel
        return True


def gauge_fix(prep, g):
    """Convert winding constraints into graph-cut-compatible labels."""
    nf = len(prep.valid)
    flip0 = np.zeros(nf, bool)
    pid = np.full(nf, -1, np.int64)
    dst, off, ic, st = g.dst, g.off, g.incompat, g.strong

    # Propagate across manifold edges.
    npatch = 0
    for s0 in np.nonzero(prep.valid & (prep.repr_of == np.arange(nf)))[0]:
        if pid[s0] != -1:
            continue
        pid[s0] = npatch
        dq = deque([s0])
        while dq:
            f = dq.popleft()
            for k in range(off[f], off[f + 1]):
                if not st[k]:
                    continue                       # Skip weak edges
                h = int(dst[k])
                if pid[h] != -1 or not prep.valid[h]:
                    continue
                pid[h] = npatch
                flip0[h] = flip0[f] ^ bool(ic[k])
                dq.append(h)
        npatch += 1

    # Reconcile patches across weak non-manifold edges.
    src_e = np.repeat(np.arange(nf), np.diff(off))
    wk = np.nonzero((~st) & (src_e < dst))[0]
    if len(wk) and npatch > 1:
        f, h = src_e[wk], dst[wk]
        P, Q = pid[f], pid[h]
        good = (P >= 0) & (Q >= 0) & (P != Q)
        if good.any():
            f, h, P, Q = f[good], h[good], P[good], Q[good]
            rel = (ic[wk][good] ^ flip0[f] ^ flip0[h]).astype(np.int64)
            key = np.minimum(P, Q) * npatch + np.maximum(P, Q)
            uk, inv = np.unique(key, return_inverse=True)
            inv = np.asarray(inv).ravel()
            w = prep.area[f] + prep.area[h]
            acc = np.zeros((len(uk), 2))
            np.add.at(acc, (inv, rel), w)
            dsu = ParityDSU(npatch)
            for t in np.argsort(-(acc[:, 0] + acc[:, 1]), kind="stable"):
                dsu.union(int(uk[t] // npatch), int(uk[t] % npatch),
                          bool(acc[t, 1] > acc[t, 0]))
            offs = np.array([dsu.find(q)[1] for q in range(npatch)], bool)
            flip0 ^= np.where(pid >= 0, offs[np.where(pid >= 0, pid, 0)], False)

    compat = (flip0[src_e] ^ flip0[dst]) == ic
    n_frust = int((~compat).sum() // 2)
    return flip0, pid, npatch, n_frust, src_e, compat


# ---------------------------------------------------------- Stage 3


def mincut_flip(prep, g, flip0, src_e, compat, ev, beta=0.2, weak_ratio=0.3):
    """Solve face flips by minimizing the graph-cut energy."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    nf = len(prep.valid)
    A = prep.area
    u = ev * np.where(flip0, -1.0, 1.0)             # Outward score in the fixed gauge
    unary = np.abs(u) * prep.valid
    cap_s = np.where(u > 0, unary, 0.0)             # Prefer keeping
    cap_t = np.where(u < 0, unary, 0.0)             # Prefer flipping

    keep = (src_e < g.dst) & compat & prep.valid[src_e] & prep.valid[g.dst]
    i, j = src_e[keep], g.dst[keep]
    ltot = float(g.elen[src_e < g.dst].sum())
    ell = float(A[prep.valid].sum()) / max(ltot, 1e-300)
    w = np.where(g.strong[keep], beta, beta * weak_ratio) * g.elen[keep] * ell

    tot = max(cap_s.sum(), cap_t.sum(), 1e-300)
    sc = 1.0e9 / tot                                # Keep flow within int32
    cs = np.rint(cap_s * sc) + 1.0                  # Keep zero-evidence faces
    ct = np.rint(cap_t * sc)
    wq = np.clip(np.rint(w * sc), 0.0, 2.0 ** 30)

    S, T = nf, nf + 1
    fidx = np.arange(nf)
    z = np.zeros(nf)
    I = np.concatenate([np.full(nf, S), fidx, i, j, fidx, np.full(nf, T)])
    J = np.concatenate([fidx, np.full(nf, T), j, i, np.full(nf, S), fidx])
    C = np.concatenate([cs, ct, wq, wq, z, z])
    G = coo_matrix((C, (I, J)), shape=(nf + 2, nf + 2)).tocsr()
    G.data = G.data.astype(np.int32)

    res = maximum_flow(G, S, T)
    R = (G - res.flow).tocsr()
    R.eliminate_zeros()
    reach = np.zeros(nf + 2, bool)
    reach[breadth_first_order(R, S, directed=True, return_predecessors=False)] = True

    y = ~reach[:nf]                                 # Sink-side faces flip
    y &= prep.valid
    return flip0 ^ y, int(y.sum())


# ---------------------------------------------------------- Stage 4


def fix_blind_components(prep, V, F, g, src_e, flip, ev, conf_tau=0.03):
    """Orient components with insufficient visibility evidence."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nf, A, m = len(F), prep.area, prep.valid
    em = m[src_e] & m[g.dst]
    adj = coo_matrix((np.ones(int(em.sum()), np.int8), (src_e[em], g.dst[em])),
                     shape=(nf, nf))
    ncomp, lab = connected_components(adj, directed=False)

    den = np.maximum(np.bincount(lab[m], weights=A[m], minlength=ncomp), 1e-300)
    conf = np.bincount(lab[m], weights=np.abs(ev)[m], minlength=ncomp) / den
    closed = np.bincount(lab[m], weights=(g.bd_per_face + g.nm_per_face)[m],
                         minlength=ncomp) == 0
    blind = np.nonzero(conf < conf_tau)[0]

    Fo = F.copy()
    Fo[flip] = Fo[flip][:, [0, 2, 1]]
    a, b, c = V[Fo[:, 0]], V[Fo[:, 1]], V[Fo[:, 2]]
    vol = np.bincount(lab[m], weights=np.einsum("ij,ij->i", a, np.cross(b, c))[m],
                      minlength=ncomp)
    cc = np.stack([np.bincount(lab[m], weights=(prep.fc[:, k] * A)[m], minlength=ncomp)
                   for k in range(3)], 1) / den[:, None]
    nrm = np.where(flip[:, None], -prep.n_unit, prep.n_unit)
    rad = np.einsum("ij,ij->i", nrm, unit(prep.fc - cc[lab]))
    scr = np.bincount(lab[m], weights=(A * rad)[m], minlength=ncomp)

    need = np.zeros(ncomp, bool)
    bv, br = blind[closed[blind]], blind[~closed[blind]]
    need[bv] = vol[bv] < 0
    need[br] = scr[br] < 0
    stat = dict(n_comp=int(ncomp), n_blind_comp=len(blind), n_blind_vol=len(bv),
                n_blind_radial=len(br), n_blind_flipped=int(need.sum()))
    return flip ^ (need[lab] & m), stat


def add_backfaces(V, F, N, back_area, diag, eps_rel=1e-4):
    """Add offset reverse faces for visible back sides."""
    idx = np.nonzero(back_area > 0)[0]
    if len(idx) == 0:
        return V, F, N, 0
    off = V[F[idx]] - (eps_rel * diag) * N[idx][:, None, :]
    add = len(V) + np.arange(3 * len(idx)).reshape(-1, 3)[:, [0, 2, 1]]
    return (np.concatenate([V, off.reshape(-1, 3)]),
            np.concatenate([F, add]),
            np.concatenate([N, -N[idx]]),
            len(idx))


# ---------------------------------------------------------- Main pipeline


def correct_normals(V, F, n_rays=32, n_views=64, vres=128, conf_tau=0.03, beta=0.2,
                    seed=0, verbose=True):
    """Return oriented faces, flat normals, metrics, and debug data."""
    t = {}
    t0 = time.time()
    prep, in_bvh = prepare(V, F)
    engine = RayEngine(V, F[in_bvh])
    t["prep"] = time.time() - t0

    t0 = time.time()
    sview, cover = shell_view_vote(prep, engine, n_views=n_views, res=vres, seed=seed)
    vis_pos, vis_neg = visibility_vote(prep, engine, n_rays=n_rays, seed=seed)
    s = vis_pos - vis_neg
    # Fall back to hemisphere evidence for shell-occluded faces.
    ev = np.where(cover > 0, sview, prep.area * s)
    t["vote"] = time.time() - t0

    t0 = time.time()
    g = build_face_graph(prep, F)
    t["graph"] = time.time() - t0

    t0 = time.time()
    flip0, pid, npatch, n_frust, src_e, compat = gauge_fix(prep, g)
    t["gauge"] = time.time() - t0

    t0 = time.time()
    flip, n_cut = mincut_flip(prep, g, flip0, src_e, compat, ev, beta=beta)
    flip, bstat = fix_blind_components(prep, V, F, g, src_e, flip, ev, conf_tau)
    t["cut"] = time.time() - t0

    # Match duplicate faces to their representatives.
    flip = np.where(prep.same_wind, flip[prep.repr_of], ~flip[prep.repr_of])
    flip[~prep.valid] = False                     # Keep degenerate faces unchanged

    F_out = F.copy()
    F_out[flip] = F_out[flip][:, [0, 2, 1]]
    N = np.where(flip[:, None], -prep.n_unit, prep.n_unit)
    degen = ~prep.valid
    N[degen] = unit(prep.fc - prep.center)[degen]  # Avoid NaNs on degenerate faces

    # Compute visibility metrics on supported faces.
    sgn = np.where(flip, -1.0, 1.0)
    hc = prep.valid & (np.abs(s) >= conf_tau)
    hw = prep.area * hc
    hw = hw / max(hw.sum(), 1e-300)
    aw = prep.area / max(prep.area.sum(), 1e-300)
    st = g.strong
    if st.any():
        coh_b = float((~g.incompat[st]).mean())
        coh_a = float(((flip[src_e[st]] ^ flip[g.dst[st]]) == g.incompat[st]).mean())
    else:
        coh_b = coh_a = float("nan")

    # Measure visible front- and back-facing area.
    cvt = max(float(cover.sum()), 1e-300)
    pos_v = 0.5 * (cover + sview)
    neg_v = 0.5 * (cover - sview)
    back_area = np.where(flip, pos_v, neg_v)
    fv_b = float(pos_v.sum()) / cvt
    fv_a = 1.0 - float(back_area.sum()) / cvt
    fv_ceil = 1.0 - float(np.minimum(pos_v, neg_v).sum()) / cvt

    info = dict(
        **prep.stats, **g.stats, **bstat,
        n_rays=n_rays, n_views=n_views, vres=vres,
        backend=engine.backend, n_patch=int(npatch),
        n_flip=int(flip.sum()), n_frustrated_edge=n_frust, n_cut_flip=n_cut,
        outward_before=float(hw[s > 0].sum()),
        outward_after=float(hw[s * sgn > 0].sum()),
        frontvis_before=fv_b, frontvis_after=fv_a, frontvis_ceiling=fv_ceil,
        coherence_before=coh_b, coherence_after=coh_a,
        occluded_ratio=float(aw[prep.valid & ~hc].sum()),
        hidden_ratio=float(aw[prep.valid & (cover <= 0)].sum()),
        **{"t_" + k: v for k, v in t.items()},
    )
    if verbose:
        print(f"  [prep ] V={info['n_vert']}(welded {info['n_vert_welded']}) "
              f"F={info['n_face']} degenerate={info['n_degenerate']} duplicate={info['n_duplicate']}")
        print(f"  [graph] boundary={info['n_boundary_edge']} manifold={info['n_manifold_edge']} "
              f"nonmanifold={info['n_nonmanifold_edge']} patches={npatch} components={bstat['n_comp']} "
              f"frustrated={n_frust}")
        print(f"  [vote ] {engine.backend} views={n_views}x{vres}^2 K={n_rays}/side "
              f"hidden_area={info['hidden_ratio']:.3f}")
        print(f"  [cut  ] cut_flips={n_cut} | blind={bstat['n_blind_comp']}"
              f"(volume={bstat['n_blind_vol']}/radial={bstat['n_blind_radial']})"
              f" component_flips={bstat['n_blind_flipped']} | total_flips={info['n_flip']}")
        print(f"  [metric] front visibility {coh_or(fv_b)} -> {coh_or(fv_a)} "
              f"(ceiling {coh_or(fv_ceil)}) | "
              f"supported outwardness {coh_or(info['outward_before'])} -> "
              f"{coh_or(info['outward_after'])} | "
              f"manifold coherence {coh_or(coh_b)} -> {coh_or(coh_a)}")
        print("  [time ] " + "  ".join(f"{k} {v:.2f}s" for k, v in t.items()))
    extra = dict(vis_pos=vis_pos, vis_neg=vis_neg, sview=sview, cover=cover,
                 back_area=back_area, diag=prep.diag, flip=flip, patch=pid,
                 area=prep.area, normals=N)
    return F_out, N, info, extra


def coh_or(x):
    return "n/a" if x != x else f"{x:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Orient mesh normals using visibility voting and graph cuts.")
    ap.add_argument("input", help="Input OBJ file or directory")
    ap.add_argument("-o", "--output", default="out", help="Output file or directory")
    ap.add_argument("-d", "--views", type=int, default=64, help="Number of shell views")
    ap.add_argument("--vres", type=int, default=128, help="Ray-grid resolution per view")
    ap.add_argument("-k", "--rays", type=int, default=32, help="Hemisphere rays per side")
    ap.add_argument("--beta", type=float, default=0.2, help="Structural consistency weight")
    ap.add_argument("--conf-tau", type=float, default=0.03, help="Blind-component threshold")
    ap.add_argument("--backfaces", action="store_true",
                    help="Add reverse geometry for visible back sides")
    ap.add_argument("--bf-eps", type=float, default=1e-4,
                    help="Back-face offset relative to the bounding-box diagonal")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    ap.add_argument("--dump", action="store_true", help="Save per-face debug data as NPZ")
    a = ap.parse_args(argv)

    if os.path.isdir(a.input):
        files = sorted(os.path.join(a.input, f)
                       for f in os.listdir(a.input) if f.endswith(".obj"))
        os.makedirs(a.output, exist_ok=True)
        outs = [os.path.join(a.output, os.path.basename(f)) for f in files]
    else:
        files, outs = [a.input], [a.output]

    for src, dstp in zip(files, outs):
        print(f"== {src}")
        V, F = load_obj(src)
        F2, N, info, extra = correct_normals(V, F, n_rays=a.rays, n_views=a.views,
                                             vres=a.vres, beta=a.beta,
                                             conf_tau=a.conf_tau, seed=a.seed)
        Vo = V
        if a.backfaces:
            Vo, F2, N, nadd = add_backfaces(V, F2, N, extra["back_area"],
                                            extra["diag"], a.bf_eps)
            print(f"  [bf   ] added {nadd} back faces (+{nadd / max(len(F), 1) * 100:.1f}%)")
        save_obj_flat(dstp, Vo, F2, N)
        if a.dump:
            np.savez_compressed(os.path.splitext(dstp)[0] + "_dbg.npz", **extra)
        print(f"   -> {dstp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
