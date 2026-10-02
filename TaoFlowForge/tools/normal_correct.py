#!/usr/bin/env python3
"""法向矫正：包围球球壳视角可见性投票 + 结构传播定向。

目标：任意三角网格（面片汤 / 非流形 / 开放边界 / 多连通分量 / 重复面）的所有
面法向一律朝外，即朝向包围球球壳的那一侧。

外向性的严格判据（与输入朝向无关）——在包围球球壳上均匀取观察方向 ω，沿 ω 打一
束正交平行光线，只取第一命中面。第一命中只由遮挡决定，不依赖任何面的朝向，故该
判据对输入的错误 winding 完全鲁棒。被看到的面若满足 n_f·(-ω) > 0，本次露的是正面：

    sview(f) = (1/D) * Σ_ω sign(-ω·n_f) * Area_vis(f, ω)

sview > 0 即"从球壳望过去这个面更多地露正面" -> 朝外。关键性质：|sview| 就是单面
光照下该面的可见发黑面积，所以最小化它 = 最小化"在三维软件里看到的黑面"，判据与
验收指标严格同一，不存在代理目标失配。

从球壳完全看不到的面（被外层包住的内层结构）sview = 0，退回半球逃逸测试兜底：

    Vis±(f) = (1/pi) * ∫_{Ω±} max(±n_f·ω, 0) * 1[ray(c_f, ω) 无遮挡抵达球壳] dω

设计要点：逐面证据是有噪声的采样估计，而"共享边两侧 winding 必须反向"是精确的组合
约束。二者都不能单独用：只信投票会打碎原本正确的一致性，只信结构则无法修正整片
翻转。故把两者写成一个全局二值能量并精确最小化：

    E(y) = Σ_f  |ev_f| · [y_f ≠ vote_f]           (可见性一元项, 单位: 面积)
         + Σ_e  w_e · [y_f ≠ y_g]                 (共享边二元项, 结构一致性)

流程：

    Stage 0  预处理：顶点合并 / 退化面剔除 / 重复面归并 / 包围球
    Stage 1  证据 ev：球壳视角首命中投票（主）+ 半球逃逸投票（隐藏面兜底）
    Stage 2  gauge fixing：沿流形边 BFS + 片级奇偶并查集，把"应反向"的约束统一
             改写成"同标签更优"，使 E 次模
    Stage 3  s-t 最小割精确最小化 E（scipy maximum_flow），得到逐面翻转标签
    Stage 4  证据总量近零的连通块整体定符号：闭合块用带符号体积，否则用径向兜底
    Stage 5  按矫正后 winding 输出每面独立 flat 法向

二元项让"确定的朝向沿共享边传递"成为最优解的自然结果：无证据区域的一元项为 0，
其标签完全由邻域证据经边传播决定。

顶点坐标、面数量与面顺序均不改变，只可能翻转面内顶点次序。

不可消除的残余：对每个面，"翻或不翻"至多只能消掉 max(pos, neg) 那一侧，剩下
min(pos, neg) 必然发黑。因此正面可见率有一个纯几何的硬上限

    ceil = Σ_f max(pos_f, neg_f) / Σ_f (pos_f + neg_f)

在这批真实家具模型上 ceil ≈ 0.963，本算法达到 0.952。剩下约 3.7% 来自"两侧都能被
看到"的单层面片（薄板、开口壳体的内壁、零厚度重合面）——这类几何的"朝外"没有唯一
解，翻法向解决不了，只能靠双面材质渲染或加背面几何。
"""

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
    """读取 obj 几何。多边形面按扇形三角化。返回 (V[nv,3], F[nf,3])。"""
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
    """写出 obj，每面一个独立法向（flat shading），面格式 f v//vn。"""
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


# ---------------------------------------------------------- 几何基础


def unit(x, axis=-1):
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-300)


def face_normals(V, F):
    """未归一化面法向（模长 = 2*面积）与面积。"""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    return cr, 0.5 * np.linalg.norm(cr, axis=1)


def bounding_sphere(V, pad=0.02):
    """Ritter 近似最小包围球 + 逐点扩张收敛，返回 (center, radius)。"""
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
    """由单位法向构造正交切空间基（Duff et al. 分支自由法）。"""
    sign = np.where(n[:, 2] >= 0.0, 1.0, -1.0)
    a = -1.0 / (sign + n[:, 2])
    b = n[:, 0] * n[:, 1] * a
    t = np.stack([1.0 + sign * n[:, 0] ** 2 * a, sign * b, -sign * n[:, 0]], axis=1)
    s = np.stack([b, sign + n[:, 1] ** 2 * a, -n[:, 1]], axis=1)
    return t, s


def cosine_hemisphere(k, rng):
    """cosine 加权半球方向的 Hammersley 低差异采样，返回 (k,3) 局部坐标。"""
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
    """球面近均匀方向（黄金角螺旋），返回 (d,3) 单位向量。整体随机旋转以去偏。"""
    i = np.arange(d, dtype=np.float64)
    z = 1.0 - (2.0 * i + 1.0) / d
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * (np.pi * (3.0 - np.sqrt(5.0)))
    if rng is not None:
        phi = phi + 2.0 * np.pi * rng.random()
    dirs = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    if rng is None:
        return dirs
    q = rng.normal(size=4)                             # 均匀随机旋转（单位四元数）
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
    wmap: np.ndarray          # 原始顶点 -> welded 顶点
    WV: np.ndarray            # welded 顶点坐标
    WF: np.ndarray            # welded 面索引
    valid: np.ndarray         # 非退化面掩码
    repr_of: np.ndarray       # 原始面 -> 代表面（重复面归并）
    same_wind: np.ndarray     # 该面与其代表面 winding 是否同向
    bvh_id: np.ndarray        # 原始面 -> BVH 图元下标（-1 表示不在 BVH 中）
    prim2face: np.ndarray     # BVH 图元下标 -> 原始面（代表面）
    center: np.ndarray
    radius: float
    diag: float
    n_unit: np.ndarray        # 输入 winding 下的单位面法向
    area: np.ndarray
    fc: np.ndarray            # 面重心
    stats: dict = field(default_factory=dict)


def prepare(V, F, weld_tol_rel=1e-6):
    """Stage 0。仅为内部计算服务，不改动输出拓扑。"""
    center, radius = bounding_sphere(V)
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))

    # 按容差量化合并重合顶点（输入常来自量化网格，重合点未必同索引）
    tol = max(diag * weld_tol_rel, 1e-12)
    key = np.round((V - V.min(0)) / tol).astype(np.int64)
    _, wfirst, wmap = np.unique(key, axis=0, return_index=True, return_inverse=True)
    wmap = np.asarray(wmap).ravel()
    WV, WF = V[wfirst], wmap[F]

    cr, area = face_normals(V, F)
    valid = area > 1e-12 * max(diag, 1e-12) ** 2
    valid &= (WF[:, 0] != WF[:, 1]) & (WF[:, 1] != WF[:, 2]) & (WF[:, 2] != WF[:, 0])

    # 重复面归并：同一 welded 顶点三元组视为同一片几何（零厚度双层会互相遮挡）
    _, first, inv = np.unique(np.sort(WF, axis=1), axis=0,
                              return_index=True, return_inverse=True)
    repr_of = first[np.asarray(inv).ravel()]

    def cyc(x):                                   # 旋转到最小元素开头，比较循环序
        r = np.argmin(x, axis=1)
        return np.stack([x[np.arange(len(x)), (r + k) % 3] for k in range(3)], 1)

    same_wind = (cyc(WF) == cyc(WF[repr_of])).all(axis=1)

    in_bvh = valid & (repr_of == np.arange(len(F)))
    bvh_id = np.full(len(F), -1, np.int64)
    bvh_id[in_bvh] = np.arange(int(in_bvh.sum()))
    prim2face = np.nonzero(in_bvh)[0]              # BVH 图元顺序 = in_bvh 的面序
    bvh_id = bvh_id[repr_of]                      # 重复面共享代表面的图元下标
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
    """embree(open3d) 优先，缺失时回退 trimesh 纯 numpy 求交。"""

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
        """返回 (t_hit, prim_id)；未命中为 (inf, -1)。"""
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
    """Stage 1：逐面双向球壳可见性投票，返回 (Vis+, Vis-)，取值 [0,1]。

    所有几何都在包围球内，故"无命中"等价于"无遮挡抵达球壳"。
    起点沿射线方向偏移 eps，使自身三角面落在 t<0 一侧，天然免自交。
    """
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
        # cosine 采样下方向的算术平均即 cosine 加权可见度的无偏估计
        vis[fid, 0] = escaped[:, 0].mean(axis=1)
        vis[fid, 1] = escaped[:, 1].mean(axis=1)
    return vis[:, 0], vis[:, 1]


def shell_view_vote(prep, engine, n_views=64, res=128, seed=0, chunk=800_000):
    """Stage 1 主证据：球壳视角首命中投票，返回 (sview, cover)，单位均为面积。

    对每个方向 ω（球面近均匀），从包围球外沿 ω 打一束正交平行光线（覆盖球的截面
    圆），只取第一命中面。每条光线代表固定的世界面积 px，故累加 px 得到的就是该面
    在这一视角下的可见投影面积。第一命中完全由遮挡决定，与任何面的朝向无关。

        sview(f) = (1/D) Σ_ω sign(-ω·n_f) · Area_vis(f, ω)     符号化可见投影面积
        cover(f) = (1/D) Σ_ω              Area_vis(f, ω)       总可见投影面积

    sview > 0 -> 当前 winding 下这个面从球壳看更多地露正面，已朝外。
    |sview| 就是"翻错时能看到的黑面面积"，最小割最小化的目标因此与验收指标同一。
    cover = 0 的面从球壳完全看不到（内层结构），交给半球逃逸投票兜底。
    """
    rng = np.random.default_rng(seed + 12345)
    nf = len(prep.n_unit)
    pos = np.zeros(nf)
    neg = np.zeros(nf)
    R = prep.radius
    px = (2.0 * R / res) ** 2                       # 每条光线代表的世界面积
    om_all = fibonacci_sphere(n_views, rng)
    t_ax, s_ax = tangent_frame(om_all)

    g = (np.arange(res) + 0.5) / res * 2.0 - 1.0    # 截面网格（含随机抖动去锯齿）
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


# ---------------------------------------------------------- 邻接图


@dataclass
class FaceGraph:
    dst: np.ndarray       # CSR 邻居
    off: np.ndarray       # CSR 偏移
    incompat: np.ndarray  # True = 两面当前 winding 不相容（接受传播需翻转）
    strong: np.ndarray    # True = 流形边（精确约束）；False = 非流形弱边
    elen: np.ndarray      # 共享边长度（图割二元项权重用）
    bd_per_face: np.ndarray
    nm_per_face: np.ndarray
    stats: dict


def build_face_graph(prep, F):
    """构造面邻接图。

    相容性判据（纯组合，不依赖几何）：两面共享一条边时，若各自的有向半边方向
    相反则 winding 相容。

      · 恰好 2 面的边（流形边）  -> 强约束
      · >=3 面的边（非流形边）   -> 按绕边方位角排序，只连接环状相邻的楔形对，弱约束
      · 1 面的边（边界边）       -> 无约束
    """
    WF = prep.WF
    keep = np.nonzero(prep.valid & (prep.repr_of == np.arange(len(F))))[0]
    he_f = np.repeat(keep, 3)
    he_a = WF[keep][:, [0, 1, 2]].reshape(-1)
    he_b = WF[keep][:, [1, 2, 0]].reshape(-1)
    he_o = WF[keep][:, [2, 0, 1]].reshape(-1)          # 对顶点，供绕边排序
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

    # 流形边：全向量化
    mf = np.nonzero(counts == 2)[0]
    i0, i1 = starts[mf], starts[mf] + 1
    P = [np.stack([he_f_s[i0], he_f_s[i1]], 1)]
    IC = [fwd_s[i0] == fwd_s[i1]]                      # 同向 -> 不相容
    ST = [np.ones(len(mf), bool)]
    EL = [elen_s[i0]]

    # 非流形边：绕边方位角排序后连接环状相邻对
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
    """带奇偶标记的并查集：维护每个元素相对其根的相对符号。"""

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
        """要求 sign(a) xor sign(b) == rel。返回是否与已有约束一致。"""
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == rel
        self.p[rb] = ra
        self.par[rb] = pa ^ pb ^ rel
        return True


def gauge_fix(prep, g):
    """Stage 2：gauge fixing——把"共享边两侧应反向"改写成"两端同标签更优"。

    2a  沿流形边（精确约束）做 BFS 森林，逐面赋 flip0，使每个"片"内部 winding 一致。
    2b  片之间只由非流形弱边相连；按面积加权多数定出每对片的相对符号，再用最大
        生成森林（奇偶并查集，重边优先）统一各片的 gauge。

    完成后除极少数"受挫边"（输入 winding 自相矛盾处，必须丢弃）外，所有边的相容
    条件都变成 y_f == y_g，Stage 3 的能量因此次模，可由最小割精确最小化。

    返回 (flip0, pid, npatch, n_frustrated)。
    """
    nf = len(prep.valid)
    flip0 = np.zeros(nf, bool)
    pid = np.full(nf, -1, np.int64)
    dst, off, ic, st = g.dst, g.off, g.incompat, g.strong

    # 2a 面级：仅沿流形边
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
                    continue                       # 弱边不参与面级传播
                h = int(dst[k])
                if pid[h] != -1 or not prep.valid[h]:
                    continue
                pid[h] = npatch
                flip0[h] = flip0[f] ^ bool(ic[k])
                dq.append(h)
        npatch += 1

    # 2b 片级：非流形弱边给出片对相对符号（面积加权多数），最大生成森林统一 gauge
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
    """Stage 3：最小割精确最小化

        E(y) = Σ_f |ev_f|·[y_f ≠ vote_f] + Σ_e w_e·[y_f ≠ y_g]

    gauge 之后所有保留边都是"同标签更优"（Ising 铁磁项），能量次模，s-t 最小割
    给出全局最优解。相比"整片一票"，它让强证据区域（如外壳大面板）能带着邻域一起
    翻转，同时无证据区域（一元项为 0）的标签完全由邻域经边传播决定。

    一元项 ev_f 是带符号的可见投影面积（Stage 1），量纲为面积，其绝对值就是判错时
    在单面光照下暴露的黑面面积——能量的一元部分因此**就是**验收指标本身。

    边权 w_e = beta·L_e·ℓ，ℓ = ΣA/ΣL（网格自身的"面积/周长"标度，使权重与整体尺度
    和网格密度无关，beta 因此是无量纲松紧旋钮）。非流形弱边按 weak_ratio 折减。

    beta 定档（9 个真实家具模型实测，均值）：

        beta   正面可见率   流形边相容率
        0.0     0.9626      0.8758      纯投票，等于逐面独立最优
        0.1     0.9558      0.9870
        0.2     0.9516      0.9914      <- 默认：相容率首次超过输入基线
        0.5     0.9348      0.9949
        1.0     0.9180      0.9959
        输入    0.8441      0.9903

    "逐面独立最优上限" = 0.9629：每个面单独取 max(pos,neg) 时的正面可见率。beta=0.2
    离这个硬上限只差 1.1pp，说明结构项几乎没有付出代价。剩下的 3.7% 是几何决定的
    不可消除量（见模块 docstring 末尾说明），不是求解质量问题。

    返回 (flip, n_cut_flip)。
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    nf = len(prep.valid)
    A = prep.area
    u = ev * np.where(flip0, -1.0, 1.0)             # gauge 下的外向性得分
    unary = np.abs(u) * prep.valid
    cap_s = np.where(u > 0, unary, 0.0)             # 偏好 y=0（保持）
    cap_t = np.where(u < 0, unary, 0.0)             # 偏好 y=1（翻转）

    keep = (src_e < g.dst) & compat & prep.valid[src_e] & prep.valid[g.dst]
    i, j = src_e[keep], g.dst[keep]
    ltot = float(g.elen[src_e < g.dst].sum())
    ell = float(A[prep.valid].sum()) / max(ltot, 1e-300)
    w = np.where(g.strong[keep], beta, beta * weak_ratio) * g.elen[keep] * ell

    tot = max(cap_s.sum(), cap_t.sum(), 1e-300)
    sc = 1.0e9 / tot                                # 最大流 <= 1e9，稳在 int32 内
    cs = np.rint(cap_s * sc) + 1.0                  # +1：零证据面默认留在源侧
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

    y = ~reach[:nf]                                 # 未被源侧到达 -> 割到汇侧 -> 翻转
    y &= prep.valid
    return flip0 ^ y, int(y.sum())


# ---------------------------------------------------------- Stage 4


def fix_blind_components(prep, V, F, g, src_e, flip, ev, conf_tau=0.03):
    """Stage 4：证据总量近零的连通块整体定符号。

    被外层完全包住的内层结构，从球壳看不到、半球射线也逃不出去，一元项为 0，最小割
    无从判断（只保证块内一致）。对这类块整体决定一次符号：

      · 块闭合（无边界边、无非流形边）-> 带符号体积；外向定向的闭合面必围出正体积
      · 否则 -> 相对块自身重心的径向判据兜底
    """
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
    """给"矫正后仍会露出背面"的面补一层反向几何，追加在表尾。

    这是消除残余黑面的唯一手段：单层面片的两侧都能被看到时，翻法向只能挑一侧，
    另一侧必然发黑（见模块 docstring 的硬上限）。补一张反向的面就把那一侧也覆盖了。

    副本沿 -n 平移 eps·diag，于是从背面观察时副本总位于原面之前，渲染器无论开不开
    背面剔除都不会 z-fighting，也不会出现"一半亮一半黑"的闪烁。

    返回 (V2, F2, N2, n_added)。原有顶点坐标、面顺序完全不变。
    """
    idx = np.nonzero(back_area > 0)[0]
    if len(idx) == 0:
        return V, F, N, 0
    off = V[F[idx]] - (eps_rel * diag) * N[idx][:, None, :]
    add = len(V) + np.arange(3 * len(idx)).reshape(-1, 3)[:, [0, 2, 1]]
    return (np.concatenate([V, off.reshape(-1, 3)]),
            np.concatenate([F, add]),
            np.concatenate([N, -N[idx]]),
            len(idx))


# ---------------------------------------------------------- 主流程


def correct_normals(V, F, n_rays=32, n_views=64, vres=128, conf_tau=0.03, beta=0.2,
                    seed=0, verbose=True):
    """返回 (F_out, N_flat, info, extra)。V、面数、面序不变，只可能翻转 winding。"""
    t = {}
    t0 = time.time()
    prep, in_bvh = prepare(V, F)
    engine = RayEngine(V, F[in_bvh])
    t["prep"] = time.time() - t0

    t0 = time.time()
    sview, cover = shell_view_vote(prep, engine, n_views=n_views, res=vres, seed=seed)
    vis_pos, vis_neg = visibility_vote(prep, engine, n_rays=n_rays, seed=seed)
    s = vis_pos - vis_neg
    # 主证据 = 球壳视角带符号可见面积；球壳完全看不到的面退回半球逃逸证据
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

    # 重复面：与代表面取得同一最终朝向（否则共享三边会被判为"应互为反向"）
    flip = np.where(prep.same_wind, flip[prep.repr_of], ~flip[prep.repr_of])
    flip[~prep.valid] = False                     # 退化面法向无定义，保持原 winding

    F_out = F.copy()
    F_out[flip] = F_out[flip][:, [0, 2, 1]]
    N = np.where(flip[:, None], -prep.n_unit, prep.n_unit)
    degen = ~prep.valid
    N[degen] = unit(prep.fc - prep.center)[degen]  # 退化面用径向占位，避免 NaN

    # ---- 指标：外向率只在「可见性有实证」的面上统计（全遮挡面 s≈0，统计无意义）
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

    # 正面可见率 = 1 - 单面光照下的发黑比例，直接来自 64 视角的可见面积统计
    cvt = max(float(cover.sum()), 1e-300)
    pos_v = 0.5 * (cover + sview)                  # 当前 winding 下露正面的可见面积
    neg_v = 0.5 * (cover - sview)
    back_area = np.where(flip, pos_v, neg_v)       # 矫正后仍会露出的背面可见面积
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
              f"F={info['n_face']} 退化={info['n_degenerate']} 重复={info['n_duplicate']}")
        print(f"  [graph] 边界边={info['n_boundary_edge']} 流形边={info['n_manifold_edge']} "
              f"非流形边={info['n_nonmanifold_edge']} 片={npatch} 连通块={bstat['n_comp']} "
              f"受挫边={n_frust}")
        print(f"  [vote ] {engine.backend} 视角={n_views}x{vres}^2 K={n_rays}/侧 "
              f"球壳不可见面积占比={info['hidden_ratio']:.3f}")
        print(f"  [cut  ] 图割翻转={n_cut} | 盲块={bstat['n_blind_comp']}"
              f"(体积{bstat['n_blind_vol']}/径向{bstat['n_blind_radial']})"
              f" 整块翻转={bstat['n_blind_flipped']} | 总翻转面={info['n_flip']}")
        print(f"  [metric] 正面可见率 {coh_or(fv_b)} -> {coh_or(fv_a)} "
              f"(几何硬上限 {coh_or(fv_ceil)}) | "
              f"外向率(有实证面) {coh_or(info['outward_before'])} -> "
              f"{coh_or(info['outward_after'])} | "
              f"流形边相容率 {coh_or(coh_b)} -> {coh_or(coh_a)}")
        print("  [time ] " + "  ".join(f"{k} {v:.2f}s" for k, v in t.items()))
    extra = dict(vis_pos=vis_pos, vis_neg=vis_neg, sview=sview, cover=cover,
                 back_area=back_area, diag=prep.diag, flip=flip, patch=pid,
                 area=prep.area, normals=N)
    return F_out, N, info, extra


def coh_or(x):
    return "n/a" if x != x else f"{x:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="包围球球壳视角可见性投票 + 图割定向的法向矫正")
    ap.add_argument("input", help="obj 文件或目录")
    ap.add_argument("-o", "--output", default="out", help="输出文件或目录")
    ap.add_argument("-d", "--views", type=int, default=64, help="球壳观察方向数")
    ap.add_argument("--vres", type=int, default=128, help="每个方向的平行光线网格边长")
    ap.add_argument("-k", "--rays", type=int, default=32, help="每侧半球射线数（隐藏面兜底）")
    ap.add_argument("--beta", type=float, default=0.2, help="结构一致性权重（越大越保守）")
    ap.add_argument("--conf-tau", type=float, default=0.03, help="盲块判定阈值")
    ap.add_argument("--backfaces", action="store_true",
                    help="给仍会露背面的面补一层反向几何，任何渲染器都不再发黑")
    ap.add_argument("--bf-eps", type=float, default=1e-4,
                    help="背面副本的内偏移量（相对包围盒对角线）")
    ap.add_argument("--seed", type=int, default=0, help="采样随机种子")
    ap.add_argument("--dump", action="store_true", help="另存逐面中间量 npz")
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
            print(f"  [bf   ] 补背面 {nadd} 面 (+{nadd / max(len(F), 1) * 100:.1f}%)")
        save_obj_flat(dstp, Vo, F2, N)
        if a.dump:
            np.savez_compressed(os.path.splitext(dstp)[0] + "_dbg.npz", **extra)
        print(f"   -> {dstp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
