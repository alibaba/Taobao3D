"""
组合补洞 + 全局法向统一矫正 —— 单文件自包含实现。

在 fill_combine_all 的组合补洞(meshflow + fill_liepa)之后，追加一步「全局法向统一矫正」，
解决输入网格自身法向就混乱、补洞时因「锚定原始面绝不翻转」而把错误朝向继承传播的问题。

流程：
  1) meshflow 图算法对原始网格 append 补一遍（仅追加面，默认只补空四环 ring_max_n=4）；
  2) fill_liepa 完整 pipeline 再补（去重->迭代补洞+补空多边形+双链/逐边缝合）；
  3) 局部朝向对齐：把新增面与存活原始面对齐（沿用 combine 逻辑）；
  4) 【新增】全局法向统一矫正 global_orient_fix：
     A. 按焊接流形边建面邻接，逐连通分量 BFS 传播绕向，使共享边方向必相反
        （此步会翻转部分原始面 —— 这正是修正输入法向混乱所必需的）；
     B. 逐分量判定整体朝内/朝外并统一翻转为朝外：闭合度高的分量用有向体积(散度定理)，
        开放分量用「多方向支撑面投票」(沿随机方向的最外侧面法向必须朝该方向)；
     C. 全程只改面绕向(顶点顺序)，顶点数量/坐标 100% 不变、面数不变。

输出：./orient_output/{输入文件名}_orient.obj

用法：
  python3 fill_orient.py input_obj/*.obj --out orient_output
  python3 fill_orient.py input_obj/*.obj --no-global-orient   # 关闭全局矫正(等价 combine)
  python3 fill_orient.py in/*.obj --orient-only               # 只做法向矫正、不补洞
"""
import os
import sys
import io
import time
import argparse
import contextlib
from collections import defaultdict, deque

import numpy as np
import trimesh


# --------------------------------------------------------------------------- #
# 基础几何工具
# --------------------------------------------------------------------------- #
def tri_normal(a, b, c):
    """三角形单位法线，退化返回零向量。"""
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-20 else np.zeros(3)


def tri_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


# --------------------------------------------------------------------------- #
# 边界环（洞）检测
# --------------------------------------------------------------------------- #
def canonical_index_map(mesh, decimals=8):
    """坐标焊接：位置重合(坐标相同、索引不同)的顶点映射到同一「代表索引」。
    仅用于检测/组环/定向，不改变、不删除任何原始顶点。"""
    rep = {}
    canon = np.empty(len(mesh.vertices), dtype=np.int64)
    for k, v in enumerate(mesh.vertices):
        key = tuple(np.round(v, decimals))
        if key not in rep:
            rep[key] = k
        canon[k] = rep[key]
    return canon


def build_edge_face_map(mesh, canon=None):
    """边(min,max) -> [face_idx,...]，传入 canon 时按焊接后代表索引建边。"""
    emap = defaultdict(list)
    faces = mesh.faces
    if canon is not None:
        faces = canon[np.asarray(faces)]
    for fi, tri in enumerate(faces):
        x, y, z = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((x, y), (y, z), (z, x)):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            emap[key].append(fi)
    return emap


def find_boundary_loops(mesh, canon=None, weld=True):
    """边界环（洞）检测，对「绕向不一致」的非流形网格鲁棒。

    两级策略，每条边界边恰好消费一次：
      1) 无向简单环：边界边构成的连通分量中「所有顶点度数=2」者，直接按无向图走成
         闭环（不依赖面绕向，能找回被绕向打断的干净四/五/多边形洞）。
      2) 剩余带 junction（度>2，非流形分叉）的边界边：用「有向半边」贪心组环
         （洞边界方向 = 相邻面内该边方向的反向），分叉处选转角最平直的后继。
    返回的环索引为「代表原始索引」。"""
    if weld and canon is None:
        canon = canonical_index_map(mesh)
    cf = canon[np.asarray(mesh.faces)] if canon is not None else np.asarray(mesh.faces)

    cnt = defaultdict(int)
    for tri in cf:
        x, y, z = int(tri[0]), int(tri[1]), int(tri[2])
        for a, b in ((x, y), (y, z), (z, x)):
            if a == b:
                continue
            cnt[(a, b) if a < b else (b, a)] += 1
    boundary = set(e for e, c in cnt.items() if c == 1)
    if not boundary:
        return []

    verts = mesh.vertices

    def pick_next(prev, cur, cands):
        """分叉处选择转角最连续(最平直)的后继顶点。"""
        if len(cands) == 1:
            return cands[0]
        v_in = verts[cur] - verts[prev]
        n_in = np.linalg.norm(v_in)
        if n_in < 1e-20:
            return cands[0]
        v_in = v_in / n_in
        best, best_score = cands[0], -2.0
        for c in cands:
            v_out = verts[c] - verts[cur]
            n_out = np.linalg.norm(v_out)
            if n_out < 1e-20:
                continue
            score = float(np.dot(v_in, v_out / n_out))
            if score > best_score:
                best, best_score = c, score
        return best

    loops = []
    consumed = set()

    # ---- 1) 无向简单环（全度2 连通分量），不依赖面绕向 ----
    g = defaultdict(set)
    for u, w in boundary:
        g[u].add(w)
        g[w].add(u)
    seen = set()
    for s in list(g):
        if s in seen:
            continue
        comp = set()
        stack = [s]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            for y in g[x]:
                if y not in seen:
                    stack.append(y)
        if not all(len(g[x]) == 2 for x in comp):
            continue                         # 含 junction，留给第 2 级处理
        start = next(iter(comp))
        loop = [start]
        prev, cur = None, start
        while True:
            nxts = [y for y in g[cur] if y != prev]
            if not nxts:
                break
            nxt = nxts[0]
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
            if len(loop) > len(comp):
                break
        if len(loop) >= 3:
            loops.append(loop)
            for k in range(len(loop)):
                a, b = loop[k], loop[(k + 1) % len(loop)]
                consumed.add((a, b) if a < b else (b, a))

    # ---- 2) 剩余（带 junction）边界边：有向半边贪心组环 ----
    remain = boundary - consumed
    if remain:
        out = defaultdict(list)
        for tri in cf:
            x, y, z = int(tri[0]), int(tri[1]), int(tri[2])
            for a, b in ((x, y), (y, z), (z, x)):
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in remain:
                    out[b].append(a)         # 洞边界走向取反向
        max_steps = sum(len(v) for v in out.values()) + 5
        for s in list(out.keys()):
            while out[s]:
                loop = [s]
                prev, cur = s, out[s].pop()
                steps = 0
                while cur != s and steps < max_steps:
                    loop.append(cur)
                    nexts = out[cur]
                    if not nexts:
                        break
                    nn = pick_next(prev, cur, nexts)
                    nexts.remove(nn)
                    prev, cur = cur, nn
                    steps += 1
                if cur == s and len(loop) >= 3:
                    loops.append(loop)
    return loops


def boundary_edge_normal(mesh, emap, a, b):
    """边界边 (a,b) 相邻的唯一原始面的法线。"""
    key = (a, b) if a < b else (b, a)
    fs = emap.get(key, [])
    if not fs:
        return np.zeros(3)
    return mesh.face_normals[fs[0]]


# --------------------------------------------------------------------------- #
# 最小权重三角化（Liepa 2003）
# --------------------------------------------------------------------------- #
def min_weight_triangulation(pts, edge_normals, ref_normal):
    """对有序闭合多边形 pts[0..n-1] 做最小权重三角化。
    权重 = (相邻面最大二面角度量, 总面积) 字典序最小。
    返回三角形局部索引列表 [(i,m,j), ...]。"""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    def norm_oriented(i, m, j):
        nrm = tri_normal(pts[i], pts[m], pts[j])
        if np.dot(nrm, ref_normal) < 0:
            nrm = -nrm
        return nrm

    def dihedral(n1, n2):
        d = np.dot(n1, n2)
        return 1.0 - max(-1.0, min(1.0, d))

    INF = float("inf")
    W = {}
    O = {}
    for i in range(n - 1):
        W[(i, i + 1)] = (0.0, 0.0)

    def edge_neighbor_normal(i, m):
        if m == i + 1:
            en = edge_normals[i]
            if np.dot(en, ref_normal) < 0:
                en = -en
            return en
        return norm_oriented(i, O[(i, m)], m)

    for gap in range(2, n):
        for i in range(0, n - gap):
            k = i + gap
            best = (INF, INF)
            best_m = -1
            for m in range(i + 1, k):
                wim, aim = W[(i, m)]
                wmk, amk = W[(m, k)]
                nrm = norm_oriented(i, m, k)
                ang = max(wim, wmk)
                ang = max(ang, dihedral(nrm, edge_neighbor_normal(i, m)))
                ang = max(ang, dihedral(nrm, edge_neighbor_normal(m, k)))
                if i == 0 and k == n - 1:
                    en = edge_normals[n - 1]
                    if np.dot(en, ref_normal) < 0:
                        en = -en
                    ang = max(ang, dihedral(nrm, en))
                area = aim + amk + tri_area(pts[i], pts[m], pts[k])
                cand = (ang, area)
                if cand < best:
                    best = cand
                    best_m = m
            W[(i, k)] = best
            O[(i, k)] = best_m

    tris = []
    stack = [(0, n - 1)]
    while stack:
        i, k = stack.pop()
        if k - i < 2:
            continue
        m = O[(i, k)]
        tris.append((i, m, k))
        stack.append((i, m))
        stack.append((m, k))
    return tris


# --------------------------------------------------------------------------- #
# 四边形洞：补回缺失的对角线（显式快速通道）
# --------------------------------------------------------------------------- #
def best_quad_diagonal(pts, edge_normals, ref_normal):
    """四边形洞(边界环恰 4 顶点)显式补回对角线。

    这类洞多是「网格四边形忘了加对角线」形成的。两条候选对角线：
      - 选项A: 对角线 0-2 -> 三角形 (0,1,2) + (0,2,3)
      - 选项B: 对角线 1-3 -> 三角形 (0,1,3) + (1,2,3)
    **先做内部性校验再按二面角择优**：对凸四边形两条对角线都在内部；对凹(非凸)
    四边形只有一条对角线在内部，选到外部那条会生成相互重叠、戳出到邻面的三角形
    （表现为交叉面）；对退化/自交(bowtie)四边形两条都无效，返回空跳过。
    内部性判据：以四边形自身绕向法线(Newell)为基准，对角线有效 <=> 其切出的两个
    三角形有向面积同号且非退化（即该对角线未跨越反角顶点、落在多边形内部）。
    edge_normals[k] 为边 pts[k]->pts[k+1] 相邻原始面法线，用于评估二面角过渡。
    返回两个三角形的局部索引列表；无有效对角线时返回 []。
    """
    pts = [np.asarray(p, dtype=float) for p in pts]

    # —— 四边形自身绕向法线（Newell），用于内部性(有向面积)判据 ——
    poly_n = _newell_normal(np.array(pts))
    pn = np.linalg.norm(poly_n)
    if pn < 1e-20:
        return []                              # 退化(共线/零面积)四边形，跳过
    poly_n = poly_n / pn

    def signed_area(i, j, k):
        return 0.5 * float(np.dot(np.cross(pts[j] - pts[i], pts[k] - pts[i]), poly_n))

    quad_area = tri_area(pts[0], pts[1], pts[2]) + tri_area(pts[0], pts[2], pts[3])
    eps_a = 1e-9 * max(quad_area, 1e-30)       # 相对面积余量，判定退化/翻转

    # 选项A(0-2)：三角形 (0,1,2)、(0,2,3) 均需正向且非退化
    validA = signed_area(0, 1, 2) > eps_a and signed_area(0, 2, 3) > eps_a
    # 选项B(1-3)：三角形 (0,1,3)、(1,2,3)
    validB = signed_area(0, 1, 3) > eps_a and signed_area(1, 2, 3) > eps_a

    if not validA and not validB:
        return []                              # 无内部对角线(bowtie/退化)，跳过
    if validA and not validB:
        return [(0, 1, 2), (0, 2, 3)]
    if validB and not validA:
        return [(0, 1, 3), (1, 2, 3)]

    # —— 两条对角线都有效(凸四边形)：按二面角过渡最平滑者择优 ——
    def orient(nrm):
        return -nrm if np.dot(nrm, ref_normal) < 0 else nrm

    def en(k):
        return orient(edge_normals[k])

    def tnorm(i, j, k):
        return orient(tri_normal(pts[i], pts[j], pts[k]))

    def dih(n1, n2):
        return 1.0 - max(-1.0, min(1.0, float(np.dot(n1, n2))))

    # 选项A：对角线 0-2；(0,1,2)邻边0,1  (0,2,3)邻边2,3
    nA1, nA2 = tnorm(0, 1, 2), tnorm(0, 2, 3)
    costA = max(dih(nA1, en(0)), dih(nA1, en(1)),
                dih(nA2, en(2)), dih(nA2, en(3)), dih(nA1, nA2))
    areaA = tri_area(pts[0], pts[1], pts[2]) + tri_area(pts[0], pts[2], pts[3])

    # 选项B：对角线 1-3；(0,1,3)邻边0,3  (1,2,3)邻边1,2
    nB1, nB2 = tnorm(0, 1, 3), tnorm(1, 2, 3)
    costB = max(dih(nB1, en(0)), dih(nB1, en(3)),
                dih(nB2, en(1)), dih(nB2, en(2)), dih(nB1, nB2))
    areaB = tri_area(pts[0], pts[1], pts[3]) + tri_area(pts[1], pts[2], pts[3])

    if (costA, areaA) <= (costB, areaB):
        return [(0, 1, 2), (0, 2, 3)]        # 对角线 0-2
    return [(0, 1, 3), (1, 2, 3)]            # 对角线 1-3


# --------------------------------------------------------------------------- #
# 补丁细化（边中点分裂）+ Laplacian 公平化
# --------------------------------------------------------------------------- #
def refine_and_fair(local_pts, local_faces, n_boundary, target_len,
                    fair_iters=10, max_refine=200):
    """前 n_boundary 个为固定边界顶点；仅分裂内部共享边，避免拼接处 T 顶点。"""
    pts = [p.copy() for p in local_pts]
    faces = [list(f) for f in local_faces]

    def rebuild_edge_map():
        em = defaultdict(list)
        for fi, (a, b, c) in enumerate(faces):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                em[key].append(fi)
        return em

    for _ in range(max_refine):
        em = rebuild_edge_map()
        longest = None
        longest_len = target_len
        for (u, v), fs in em.items():
            if len(fs) != 2:
                continue
            L = np.linalg.norm(pts[u] - pts[v])
            if L > longest_len:
                longest_len = L
                longest = (u, v, fs[0], fs[1])
        if longest is None:
            break
        u, v, f1, f2 = longest
        newp = 0.5 * (pts[u] + pts[v])
        pi = len(pts)
        pts.append(newp)

        def split(face):
            a, b, c = face
            seq = [a, b, c]
            for idx in range(3):
                x, y = seq[idx], seq[(idx + 1) % 3]
                if {x, y} == {u, v}:
                    w = seq[(idx + 2) % 3]
                    return [[x, pi, w], [pi, y, w]]
            return [face]

        new_f1 = split(faces[f1])
        new_f2 = split(faces[f2])
        for fi in sorted((f1, f2), reverse=True):
            faces.pop(fi)
        faces.extend(new_f1)
        faces.extend(new_f2)

    n_fixed = len(local_pts)
    nbr = defaultdict(set)
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            nbr[u].add(v)
            nbr[v].add(u)
    movable = [i for i in range(len(pts)) if i >= n_fixed]
    for _ in range(fair_iters):
        for i in movable:
            ns = list(nbr[i])
            if ns:
                pts[i] = np.mean([pts[j] for j in ns], axis=0)

    return np.array(pts), np.array(faces, dtype=np.int64)


# --------------------------------------------------------------------------- #
# 法向矫正：补丁绕向一致性传播
# --------------------------------------------------------------------------- #
def orient_patch_faces(patch, boundary_dir, verts, ref_normal):
    """让一块补丁三角形的绕向与相邻原始面一致。

    原理（朝向一致的网格）：一条边在其两个相邻面中方向相反。
      - 补丁边界边应遵循洞边界有向边 boundary_dir（= 相邻原始面方向的反向），
        从而补丁面与原始面在共享边上方向相反 -> 朝向一致。
      - 从含边界边的三角形作为种子确定绝对朝向，再沿补丁内部共享边 BFS 传播
        （相邻两三角形在共享边上方向必须相反）。
      - 不接触任何边界边的孤立块用 ref_normal 兜底定向。

    patch: [[a,b,c], ...] 全局索引三角形；boundary_dir: set{(u,v)} 有向边界边。
    返回定向后的三角形列表。
    """
    m = len(patch)
    if m == 0:
        return []

    # 补丁内部共享边邻接
    edge2tris = defaultdict(list)
    for ti, tri in enumerate(patch):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:                       # 退化边（补丁含退化三角形）跳过
                continue
            edge2tris[frozenset((u, v))].append(ti)
    adj = defaultdict(list)
    for key, tl in edge2tris.items():
        if len(key) != 2:                    # 冗余守卫：非二元边跳过
            continue
        if len(tl) == 2:
            adj[tl[0]].append((tl[1], tuple(key)))
            adj[tl[1]].append((tl[0], tuple(key)))

    def dirs_of(tri):
        a, b, c = tri
        return ((a, b), (b, c), (c, a))

    def flip(tri):
        return (tri[0], tri[2], tri[1])

    oriented = [None] * m
    visited = [False] * m
    dq = deque()

    # 1) 用含边界有向边的三角形作种子
    for ti, tri in enumerate(patch):
        t = (int(tri[0]), int(tri[1]), int(tri[2]))
        s = 0
        for (u, v) in dirs_of(t):
            if (u, v) in boundary_dir:
                s = 1
                break
            if (v, u) in boundary_dir:
                s = -1
                break
        if s != 0:
            oriented[ti] = flip(t) if s == -1 else t
            visited[ti] = True
            dq.append(ti)

    # 2) BFS 传播：邻居在共享边上取与当前相反方向
    while dq:
        ti = dq.popleft()
        cur = set(dirs_of(oriented[ti]))
        for (tj, key) in adj[ti]:
            if visited[tj]:
                continue
            u, v = key
            if (u, v) in cur:
                need = (v, u)
            elif (v, u) in cur:
                need = (u, v)
            else:
                need = None
            tj_tri = (int(patch[tj][0]), int(patch[tj][1]), int(patch[tj][2]))
            if need is not None and need not in set(dirs_of(tj_tri)):
                tj_tri = flip(tj_tri)
            oriented[tj] = tj_tri
            visited[tj] = True
            dq.append(tj)

    # 3) 兜底：未被边界约束到的孤立块，用 ref_normal 定向后再传播
    for ti in range(m):
        if visited[ti]:
            continue
        tri = (int(patch[ti][0]), int(patch[ti][1]), int(patch[ti][2]))
        fn = tri_normal(verts[tri[0]], verts[tri[1]], verts[tri[2]])
        if np.dot(fn, ref_normal) < 0:
            tri = flip(tri)
        oriented[ti] = tri
        visited[ti] = True
        dq.append(ti)
        while dq:
            t0 = dq.popleft()
            cur = set(dirs_of(oriented[t0]))
            for (tj, key) in adj[t0]:
                if visited[tj]:
                    continue
                u, v = key
                if (u, v) in cur:
                    need = (v, u)
                elif (v, u) in cur:
                    need = (u, v)
                else:
                    need = None
                tj_tri = (int(patch[tj][0]), int(patch[tj][1]), int(patch[tj][2]))
                if need is not None and need not in set(dirs_of(tj_tri)):
                    tj_tri = flip(tj_tri)
                oriented[tj] = tj_tri
                visited[tj] = True
                dq.append(tj)

    return [list(t) for t in oriented]


# --------------------------------------------------------------------------- #
# 法向朝向修正：锁定原始老面，仅修正新增补洞面的朝向
# --------------------------------------------------------------------------- #
def orient_new_faces_by_neighbors(mesh, n_orig, verbose=True):
    """只修正「新增补洞面」的朝向，原始老面 [0, n_orig) 绝对不动。

    动机：老面朝向本来是对的，全局翻转会破坏它们。这里以老面为「锚」，
    沿焊接流形边做 BFS/多源传播，让每个新增面与相邻已定向面(优先老面)绕向一致
    （共享边方向必须相反）。仅翻转 idx >= n_orig 的新增面，不改顶点、不改老面。

    锚不到任何老面的孤立补丁(被非流形边包围)：退化为补丁内部互相一致。
    """
    canon = canonical_index_map(mesh)
    faces = np.asarray(mesh.faces)
    cf = canon[faces]
    F = len(faces)

    # 焊接边 -> [(面, 该面内有向边)]
    edge2 = defaultdict(list)
    for fi in range(F):
        a, b, c = int(cf[fi, 0]), int(cf[fi, 1]), int(cf[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                edge2[(u, v) if u < v else (v, u)].append((fi, (u, v)))
    # 仅用流形边(恰 2 面)构建面邻接
    adj = defaultdict(list)
    for key, lst in edge2.items():
        if len(lst) == 2:
            (fi, di), (fj, dj) = lst
            adj[fi].append((fj, di, dj))
            adj[fj].append((fi, dj, di))

    flip = np.zeros(F, dtype=bool)
    vis = np.zeros(F, dtype=bool)

    # 1) 锚定所有老面（不翻转、作为 BFS 源），向新增面传播
    dq = deque()
    for fi in range(n_orig):
        vis[fi] = True
        dq.append(fi)
    while dq:
        fi = dq.popleft()
        for (fj, di, dj) in adj[fi]:
            if vis[fj] or fj < n_orig:      # 老面已锚定，绝不改动
                continue
            flip[fj] = (not flip[fi]) if di == dj else flip[fi]
            vis[fj] = True
            dq.append(fj)

    # 2) 锚不到老面的孤立新增补丁：内部互相一致
    for s in range(n_orig, F):
        if vis[s]:
            continue
        vis[s] = True
        dq = deque([s])
        while dq:
            fi = dq.popleft()
            for (fj, di, dj) in adj[fi]:
                if vis[fj] or fj < n_orig:
                    continue
                flip[fj] = (not flip[fi]) if di == dj else flip[fi]
                vis[fj] = True
                dq.append(fj)

    # 只翻转新增面，老面强制保持
    flip[:n_orig] = False
    nf = faces.copy()
    nf[flip] = nf[flip][:, ::-1]
    mesh.faces = nf

    if verbose:
        n_new = F - n_orig
        # 焊接口径统计新增面与相邻老面/补洞面之间的残留朝向矛盾边
        ncf = canon[nf]
        e2 = defaultdict(list)
        for fi in range(F):
            a, b, c = int(ncf[fi, 0]), int(ncf[fi, 1]), int(ncf[fi, 2])
            for u, v in ((a, b), (b, c), (c, a)):
                if u != v:
                    e2[(u, v) if u < v else (v, u)].append((fi, (u, v)))
        conflict_new = 0
        for key, lst in e2.items():
            if len(lst) == 2:
                (fi, di), (fj, dj) = lst
                if di == dj and (fi >= n_orig or fj >= n_orig):
                    conflict_new += 1
        print(f"  [orient] 仅修正新增补洞面朝向: 翻转 {int(flip[n_orig:].sum())}/{n_new} 个新增面"
              f"（老面 {n_orig} 个保持不变）| 涉及新增面的残留矛盾边 {conflict_new}")
    return mesh


# --------------------------------------------------------------------------- #
# 全局法向统一矫正：允许翻转任何面（含原始面），使整体绕向一致且朝外
# --------------------------------------------------------------------------- #
def _build_face_adjacency(mesh):
    """按焊接流形边(恰 2 面共享)建面邻接：fi -> [(fj, 本面有向边, 邻面有向边)]。
    同时返回焊接后的面数组 cf，供矛盾边统计复用。"""
    canon = canonical_index_map(mesh)
    faces = np.asarray(mesh.faces)
    cf = canon[faces]
    edge2 = defaultdict(list)
    for fi in range(len(faces)):
        a, b, c = int(cf[fi, 0]), int(cf[fi, 1]), int(cf[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                edge2[(u, v) if u < v else (v, u)].append((fi, (u, v)))
    adj = defaultdict(list)
    for lst in edge2.values():
        if len(lst) == 2:
            (fi, di), (fj, dj) = lst
            adj[fi].append((fj, di, dj))
            adj[fj].append((fi, dj, di))
    return adj, cf, edge2


def count_orientation_conflicts(mesh):
    """焊接口径下「朝向矛盾边」数量与流形边总数。

    两个面共享一条流形边时，若朝向一致，该边在两面内的有向走向必然相反(u→v / v→u)；
    若走向相同(di == dj)则说明两面朝向矛盾。返回 (矛盾边数, 流形边数)。"""
    _, _, edge2 = _build_face_adjacency(mesh)
    mani = confl = 0
    for lst in edge2.values():
        if len(lst) == 2:
            mani += 1
            if lst[0][1] == lst[1][1]:
                confl += 1
    return confl, mani


def _visibility_vote(V, F, n_dirs=32, res=128, samples=3, seed=0):
    """多视角可见性投票：估计每个面的朝向是否「朝外」。返回每面净票数(正=朝外)。

    动机：本项目输入常为严重非流形面片汤，纯拓扑传播会碎成大量孤立单面分量，
    此时有向体积/支撑面判据近似随机。可见性投票直接对应「渲染时看到的是正面还是背面」，
    对非流形同样鲁棒。

    做法(点采样 z-buffer)：在每个面上按**面积比例**布置重心采样点(每面至少 1 个)，使点云
    近似曲面均匀密度 —— 这样 z-buffer 胜出计数就近似「像素面积」，票数天然按面积加权，
    与光栅化观感一致。对 n_dirs 个视角方向 d(相机位于 +∞·d 沿 -d 观察)做正交投影，在
    res×res 网格内保留深度(V·d)最大即离相机最近的采样点 —— 其所属面即该视角可见面。
    可见面法向与 d 同向(朝向相机)记 +1 票，背向记 -1 票。"""
    nf = len(F)
    if nf == 0:
        return np.zeros(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-20
    fnu = np.zeros_like(fn)
    fnu[ok] = fn[ok] / ln[ok, None]

    # 按面积比例分配采样点数（每面至少 1 个），总预算 ≈ samples × nf
    area = 0.5 * ln
    budget = max(1, int(samples)) * nf
    tot_area = float(area.sum())
    if tot_area <= 0:
        return np.zeros(nf)
    cnt = np.maximum(1, np.round(area / tot_area * budget).astype(np.int64))
    cnt = np.minimum(cnt, 64)                       # 防极端大面爆内存
    pf = np.repeat(np.arange(nf), cnt)              # 采样点 -> 面
    # 三角形内均匀重心采样（第 1 个点固定取质心，保证小面稳定）
    rng = np.random.default_rng(seed)
    m = len(pf)
    r1 = np.sqrt(rng.random(m))
    r2 = rng.random(m)
    w0 = 1.0 - r1
    w1 = r1 * (1.0 - r2)
    w2 = r1 * r2
    first = np.zeros(m, dtype=bool)
    first[np.cumsum(np.concatenate([[0], cnt[:-1]]))] = True
    w0[first] = w1[first] = w2[first] = 1.0 / 3.0
    P = w0[:, None] * a[pf] + w1[:, None] * b[pf] + w2[:, None] * c[pf]

    # 视角方向：球面均匀(黄金螺旋)，避免随机方向聚簇
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)

    lo = V.min(0)
    ext = float(np.abs(np.ptp(V, axis=0)).max())
    if ext <= 0:
        return np.zeros(nf)
    votes = np.zeros(nf)
    # 注：macOS Accelerate BLAS 对 matmul 会发出伪 FP 告警(结果实测全为有限值)，此处屏蔽
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for d in dirs:
            # 视平面正交基
            tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(d, tmp)
            u /= max(np.linalg.norm(u), 1e-20)
            w = np.cross(d, u)
            # 像素坐标与深度(沿 d 越大越靠近相机)
            pu = (P - lo) @ u
            pw = (P - lo) @ w
            depth = (P - lo) @ d
            gi = np.clip(((pu - pu.min()) / ext * (res - 1)).astype(np.int64), 0, res - 1)
            gj = np.clip(((pw - pw.min()) / ext * (res - 1)).astype(np.int64), 0, res - 1)
            cell = gi * res + gj
            # 每像素取深度最大的采样点：按 (cell, depth) 排序后取每 cell 最后一个
            order = np.lexsort((depth, cell))
            cs = cell[order]
            last = np.ones(len(cs), dtype=bool)
            last[:-1] = cs[:-1] != cs[1:]          # 该 cell 的最大深度样本
            win = order[last]
            wf = pf[win]
            s = fnu[wf] @ d                        # >0 法向朝相机(正面)
            np.add.at(votes, wf, np.sign(s))
    return votes


def visible_back_ratio(V, F, n_dirs=32, res=128, seed=0):
    """可见采样点中「背面」占比 [0,1]，0=看到的全是正面(法向朝外)。

    与 _visibility_vote 同一套点采样 z-buffer 口径，作为「渲染观感」的量化代理，
    用于矫正前后的效果比对与防回退护栏。"""
    v = _visibility_vote(V, F, n_dirs=n_dirs, res=res, seed=seed)
    pos = float(np.clip(v, 0, None).sum())
    neg = float(np.clip(-v, 0, None).sum())
    tot = pos + neg
    return (neg / tot) if tot > 0 else 0.0


def global_orient_fix(mesh, verbose=True, seed=0, n_dirs=32, res=128,
                      vis_conf=0.25, guard_tol=0.02):
    """全局法向统一矫正：使整个网格绕向一致且朝外。仅改面绕向，顶点与面数完全不变。

    与 orient_new_faces_by_neighbors 的区别：本函数**不锚定原始面**，允许翻转任意面。
    这是修正「输入网格自身法向就混乱、补洞把错误朝向继承传播」的必要手段。

    步骤：
      A. 按焊接流形边建面邻接，逐连通分量 BFS 传播绕向(共享边走向必相反)，使分量内一致；
      B. 逐分量定「朝外」符号，采用「可见性证据 + 原朝向先验」混合判据：
         · 可见性票据充分(|净票| >= vis_conf × 总票)时按票决 —— 对整体翻反的网格能果断纠正；
         · 票据不足(分量小/被遮挡, 此时投票近似噪声)时退回原朝向先验，即选择「与输入原
           绕向偏离最小」的一侧；
         · 仍无法判定且分量近闭合时用有向体积兜底。
      C. 防回退护栏：输入本身散布少量矛盾边时，强制全局一致会连带反转大片区域，可能反而
         更差。故比较矫正前后的「可见背面比」，若矫正后差出 guard_tol 以上则放弃本次矫正、
         保留原绕向(guard_tol=None 关闭护栏, 始终矫正)。
      D. 返回 (mesh, stats)，含翻转面数、分量数、翻转分量数、矫正前后矛盾边与可见背面比。
    """
    faces = np.asarray(mesh.faces)
    F = len(faces)
    if F == 0:
        return mesh, {"n_flip": 0, "n_comp": 0, "n_comp_flip": 0,
                      "confl_before": 0, "confl_after": 0, "mani": 0,
                      "by_vis": 0, "by_prior": 0, "back_before": 0.0,
                      "back_after": 0.0, "reverted": False}
    V = np.asarray(mesh.vertices, dtype=np.float64)
    adj, cf, edge2 = _build_face_adjacency(mesh)
    confl_before = sum(1 for l in edge2.values() if len(l) == 2 and l[0][1] == l[1][1])
    mani = sum(1 for l in edge2.values() if len(l) == 2)

    # —— A. 逐分量 BFS 传播绕向一致性 ——
    flip = np.zeros(F, dtype=bool)
    comp = np.full(F, -1, dtype=np.int64)
    n_comp = 0
    for s in range(F):
        if comp[s] >= 0:
            continue
        comp[s] = n_comp
        dq = deque([s])
        while dq:
            fi = dq.popleft()
            for (fj, di, dj) in adj[fi]:
                if comp[fj] >= 0:
                    continue
                # 朝向一致要求共享边走向相反；若相同则 fj 需与 fi 反相
                flip[fj] = (not flip[fi]) if di == dj else flip[fi]
                comp[fj] = n_comp
                dq.append(fj)
        n_comp += 1

    faces_c = faces.copy()
    faces_c[flip] = faces_c[flip][:, ::-1]

    # —— B. 逐分量判定朝内/朝外（可见性证据 + 原朝向先验 混合）——
    vote = _visibility_vote(V, faces_c, n_dirs=n_dirs, res=res, seed=seed)
    comp_vote = np.zeros(n_comp)                  # 净票(正=朝外)
    comp_conf = np.zeros(n_comp)                  # 总票量(置信度)
    np.add.at(comp_vote, comp, vote)
    np.add.at(comp_conf, comp, np.abs(vote))
    # 原朝向先验：步骤 A 使 flip[i]=True 的面已偏离输入原绕向。
    # 保持现状的偏离数 = Σflip；整体翻转后的偏离数 = size - Σflip。
    comp_size = np.bincount(comp, minlength=n_comp).astype(np.float64)
    comp_dev = np.zeros(n_comp)
    np.add.at(comp_dev, comp, flip.astype(np.float64))
    # 有向体积兜底所需
    a0 = V[faces_c[:, 0]]
    tetvol = np.einsum("ij,ij->i", a0, np.cross(V[faces_c[:, 1]], V[faces_c[:, 2]])) / 6.0
    comp_vol = np.zeros(n_comp)
    np.add.at(comp_vol, comp, tetvol)
    comp_bnd = np.zeros(n_comp)
    comp_edge = np.zeros(n_comp)
    for lst in edge2.values():
        ci = comp[lst[0][0]]
        comp_edge[ci] += 1
        if len(lst) == 1:
            comp_bnd[ci] += 1

    need_flip = np.zeros(n_comp, dtype=bool)
    n_by_vis = n_by_prior = 0
    for ci in range(n_comp):
        conf = comp_conf[ci]
        vis = comp_vote[ci]
        if conf > 0 and abs(vis) >= vis_conf * conf:
            need_flip[ci] = vis < 0            # 可见性证据充分 -> 按票决
            n_by_vis += 1
            continue
        # 证据不足 -> 原朝向先验：选偏离输入原绕向更小的一侧
        if comp_dev[ci] * 2 != comp_size[ci]:
            need_flip[ci] = comp_dev[ci] * 2 > comp_size[ci]
            n_by_prior += 1
            continue
        # 先验也无法区分(恰好一半)：近闭合分量用有向体积
        ratio = comp_bnd[ci] / max(comp_edge[ci], 1)
        if ratio < 0.05 and comp_vol[ci] < 0:
            need_flip[ci] = True
    if np.any(need_flip):
        sel = need_flip[comp]
        faces_c[sel] = faces_c[sel][:, ::-1]
        flip[sel] = ~flip[sel]

    # —— C. 防回退护栏：矫正后若可见背面比明显变差则放弃矫正 ——
    back_before = visible_back_ratio(V, faces, n_dirs=n_dirs, res=res, seed=seed)
    back_after = visible_back_ratio(V, faces_c, n_dirs=n_dirs, res=res, seed=seed)
    reverted = False
    if guard_tol is not None and back_after > back_before + guard_tol:
        reverted = True
        faces_c = faces                          # 保留输入原绕向
        flip[:] = False

    mesh.faces = faces_c
    confl_after, _ = count_orientation_conflicts(mesh)
    stats = {"n_flip": int(flip.sum()), "n_comp": n_comp,
             "n_comp_flip": 0 if reverted else int(need_flip.sum()),
             "confl_before": confl_before, "confl_after": confl_after,
             "mani": mani, "by_vis": n_by_vis, "by_prior": n_by_prior,
             "back_before": back_before, "back_after": back_after,
             "reverted": reverted}
    if verbose:
        if reverted:
            print(f"  [全局法向矫正] 已放弃(护栏): 可见背面比 {back_before * 100:.1f}% → "
                  f"{back_after * 100:.1f}% 反而变差, 保留输入原绕向")
        else:
            print(f"  [全局法向矫正] 分量 {n_comp} 个(整体翻转 {stats['n_comp_flip']} 个; "
                  f"可见性判定 {n_by_vis}, 原朝向先验 {n_by_prior}) | "
                  f"翻转面 {stats['n_flip']}/{F} | 朝向矛盾边 {confl_before}→{confl_after} "
                  f"(流形边 {mani}) | 可见背面比 {back_before * 100:.1f}%→"
                  f"{back_after * 100:.1f}% | 顶点不变")
    return mesh, stats


# --------------------------------------------------------------------------- #
# 冗余重叠面剔除：消除渲染时的 z-fighting（重合穿模）
# --------------------------------------------------------------------------- #
def zfight_ratio(V, F, n_dirs=16, res=256, samples=6, eps_ratio=2e-3, seed=0):
    """z-fighting 像素比 [0,100]：多视角点采样 z-buffer 下，同一像素「最近两层表面」
    深度差 < eps 且分属不同面的比例 —— 直接量化渲染时的重合穿模程度。"""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    if len(F) == 0:
        return 0.0
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    ln = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    area = 0.5 * ln
    tot = float(area.sum())
    if tot <= 0:
        return 0.0
    cnt = np.maximum(1, np.round(area / tot * (samples * len(F))).astype(np.int64))
    cnt = np.minimum(cnt, 64)
    pf = np.repeat(np.arange(len(F)), cnt)
    m = len(pf)
    rng = np.random.default_rng(seed)
    r1 = np.sqrt(rng.random(m))
    r2 = rng.random(m)
    P = ((1 - r1)[:, None] * a[pf] + (r1 * (1 - r2))[:, None] * b[pf]
         + (r1 * r2)[:, None] * c[pf])
    i = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * i / n_dirs)
    th = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                     np.cos(phi)], axis=1)
    lo = V.min(0)
    ext = float(np.abs(np.ptp(V, axis=0)).max())
    if ext <= 0:
        return 0.0
    eps = eps_ratio * ext
    zf = nc = 0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for d in dirs:
            tmp = np.array([0., 0., 1.]) if abs(d[2]) < 0.9 else np.array([1., 0., 0.])
            u = np.cross(d, tmp)
            u /= max(np.linalg.norm(u), 1e-20)
            w = np.cross(d, u)
            pu = (P - lo) @ u
            pw = (P - lo) @ w
            dep = (P - lo) @ d
            gi = np.clip(((pu - pu.min()) / ext * (res - 1)).astype(np.int64), 0, res - 1)
            gj = np.clip(((pw - pw.min()) / ext * (res - 1)).astype(np.int64), 0, res - 1)
            cell = gi * res + gj
            order = np.lexsort((-dep, cell))          # 每 cell 按深度降序(最近在前)
            cs = cell[order]
            ds = dep[order]
            fs = pf[order]
            firstm = np.ones(len(cs), dtype=bool)
            firstm[1:] = cs[1:] != cs[:-1]
            idx1 = np.nonzero(firstm)[0]              # 每 cell 最近样本
            nc += len(idx1)
            idx2 = idx1 + 1
            ok = (idx2 < len(cs)) & (cs[np.minimum(idx2, len(cs) - 1)] == cs[idx1])
            if not np.any(ok):
                continue
            i1 = idx1[ok]
            i2 = idx2[ok]
            close = (np.abs(ds[i1] - ds[i2]) < eps) & (fs[i1] != fs[i2])
            zf += int(close.sum())
    return zf / max(nc, 1) * 100.0


# --------------------------------------------------------------------------- #
# Liepa 补洞主流程
# --------------------------------------------------------------------------- #
def fill_liepa(mesh, do_refine=False, fix_orientation=True, verbose=True):
    """默认 do_refine=False：原始顶点数量与位置完全不变，仅向洞内追加三角面。
    fix_orientation=True 时对最终网格做「朝内/朝外」统一修正（绕向一致化+朝外）。"""
    mesh = mesh.copy()
    canon = canonical_index_map(mesh)
    emap = build_edge_face_map(mesh, canon=canon)
    loops = find_boundary_loops(mesh, canon=canon)
    if verbose:
        print(f"  [liepa] 检测到 {len(loops)} 个洞，开始三角化...")

    verts = mesh.vertices.copy()
    faces = mesh.faces.tolist()
    n_orig_faces = len(faces)          # 原始老面数，之后仅追加补洞面
    new_verts = list(verts)
    filled = 0

    for loop in loops:
        n = len(loop)
        pts = np.array([verts[idx] for idx in loop])
        edge_normals = np.array([
            boundary_edge_normal(mesh, emap, loop[k], loop[(k + 1) % n])
            for k in range(n)
        ])
        ref = edge_normals.sum(axis=0)
        rn = np.linalg.norm(ref)
        ref_normal = ref / rn if rn > 1e-12 else tri_normal(pts[0], pts[1], pts[2])

        try:
            if n == 4:
                # 四边形洞：显式补回缺失的对角线（最优二面角）
                tris = best_quad_diagonal(pts, edge_normals, ref_normal)
            else:
                tris = min_weight_triangulation(pts, edge_normals, ref_normal)
        except Exception as e:
            if verbose:
                print(f"    洞(n={n})三角化失败，跳过: {e}")
            continue
        if not tris:
            continue

        if do_refine and n >= 4:
            blens = [np.linalg.norm(pts[k] - pts[(k + 1) % n]) for k in range(n)]
            target = float(np.mean(blens)) if blens else 0.0
            local_faces = [list(t) for t in tris]
            try:
                rp, rf = refine_and_fair(pts, local_faces, n, target)
            except Exception:
                rp, rf = pts, np.array(tris, dtype=np.int64)
        else:
            rp, rf = pts, np.array(tris, dtype=np.int64)

        # 局部索引 -> 全局索引：前 n 个是边界(用 loop 全局索引)，其余为新增顶点
        base = len(new_verts)
        idx_map = {}
        for k in range(n):
            idx_map[k] = loop[k]
        for k in range(n, len(rp)):
            new_verts.append(rp[k])
            idx_map[k] = base + (k - n)

        # 组装补丁（全局索引），并做绕向一致性法向矫正
        patch = [[idx_map[int(a)], idx_map[int(b)], idx_map[int(c)]] for (a, b, c) in rf]
        boundary_dir = {(loop[k], loop[(k + 1) % n]) for k in range(n)}
        oriented = orient_patch_faces(patch, boundary_dir, new_verts, ref_normal)
        faces.extend(oriented)
        filled += 1

    if verbose:
        print(f"  [liepa] 已填充 {filled}/{len(loops)} 个洞，"
              f"新增顶点 {len(new_verts) - len(verts)}")
    out = trimesh.Trimesh(vertices=np.array(new_verts),
                          faces=np.array(faces), process=False)
    if fix_orientation:
        orient_new_faces_by_neighbors(out, n_orig_faces, verbose=verbose)
    return out


# --------------------------------------------------------------------------- #
# 统计
# --------------------------------------------------------------------------- #
def count_holes(mesh):
    """可闭合洞（能组成闭合边界环）的数量。"""
    return len(find_boundary_loops(mesh))


def count_boundary_edges(mesh):
    """真实残留边界边数量（焊接后仅被 1 个面引用的边，含悬挂/非流形碎边）。"""
    canon = canonical_index_map(mesh)
    cf = canon[np.asarray(mesh.faces)]
    cnt = defaultdict(int)
    for t in cf:
        x, y, z = int(t[0]), int(t[1]), int(t[2])
        for a, b in ((x, y), (y, z), (z, x)):
            if a != b:
                cnt[(a, b) if a < b else (b, a)] += 1
    return sum(1 for c in cnt.values() if c == 1)


# --------------------------------------------------------------------------- #
# 缝合接缝裂缝（zippering，只追加桥接面、不动顶点）
# --------------------------------------------------------------------------- #
def zipper_boundary_cracks(mesh, tol=None, tol_ratio=0.005, max_iter=12,
                           verbose=True):
    """缝合「接缝裂缝」：相邻面片沿同一棱线相接但两侧三角化不一致、留有细小间隙，
    形成互不共享的两条边界链（长条形、带分叉/断头，组不成闭合环，Liepa 无法补）。

    做法：对每条边界边(count==1)，在其中点附近 tol 半径内寻找「对侧」边界顶点 c，
    追加桥接三角形 (a,b,c) 把缝拉合。仅追加面、复用已有顶点（焊接代表索引），
    不移动/不新增任何顶点。迭代进行，直到无缝可缝或达到 max_iter。

    tol       : 绝对间距阈值(坐标单位)。None 时取 包围盒对角线 * tol_ratio。
    tol_ratio : tol 为 None 时的相对阈值(相对包围盒对角线)。
    """
    verts = np.asarray(mesh.vertices)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    if tol is None:
        tol = diag * tol_ratio
    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    total_added = 0
    rounds = 0
    for _ in range(max_iter):
        faces = np.asarray(mesh.faces)
        canon = canonical_index_map(mesh)
        cf = canon[faces]
        # 边引用计数 + 已存在面集合(焊接口径, 去重用)
        cnt = defaultdict(int)
        face_set = set()
        for t in cf:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            key = frozenset((a, b, c))
            if len(key) == 3:
                face_set.add(key)
            for u, w in ((a, b), (b, c), (c, a)):
                if u != w:
                    cnt[(u, w) if u < w else (w, u)] += 1
        boundary = [e for e, c in cnt.items() if c == 1]
        if not boundary:
            break
        # 边界顶点(焊接代表索引) + KDTree
        bverts = sorted({x for e in boundary for x in e})
        P = verts[bverts]
        pos_of = {vid: verts[vid] for vid in bverts}
        if cKDTree is not None:
            tree = cKDTree(P)
        # 为每条边界边挑选最优对侧顶点 c
        candidates = []            # (质量分, a, b, c)
        for (a, b) in boundary:
            Pa, Pb = pos_of[a], pos_of[b]
            mid = 0.5 * (Pa + Pb)
            ab = Pb - Pa
            ab2 = float(ab @ ab)
            if cKDTree is not None:
                idxs = tree.query_ball_point(mid, tol)
                cand_ids = [bverts[i] for i in idxs]
            else:
                cand_ids = [vid for vid in bverts
                            if np.linalg.norm(pos_of[vid] - mid) <= tol]
            best = None
            best_d = tol
            for c in cand_ids:
                if c == a or c == b:
                    continue
                Pc = pos_of[c]
                # c 到线段 ab 的垂距(缝宽)，越小越好
                if ab2 > 1e-20:
                    tproj = max(0.0, min(1.0, float((Pc - Pa) @ ab) / ab2))
                else:
                    tproj = 0.0
                foot = Pa + tproj * ab
                d = float(np.linalg.norm(Pc - foot))
                if d >= best_d:
                    continue
                if tri_area(Pa, Pb, Pc) < 1e-14:     # 退化(共线)
                    continue
                best = c
                best_d = d
            if best is not None:
                candidates.append((best_d, a, b, best))
        if not candidates:
            break
        # 缝宽小的优先，贪心追加；实时更新边计数，避免重复/过度缝合
        candidates.sort(key=lambda x: x[0])
        new_faces = []
        for _d, a, b, c in candidates:
            e_ab = (a, b) if a < b else (b, a)
            if cnt.get(e_ab, 0) != 1:               # 该缝已被前面缝合
                continue
            key_tri = frozenset((a, b, c))
            if len(key_tri) < 3 or key_tri in face_set:
                continue
            new_faces.append((a, b, c))
            face_set.add(key_tri)
            for u, w in ((a, b), (b, c), (c, a)):
                cnt[(u, w) if u < w else (w, u)] = cnt.get((u, w) if u < w else (w, u), 0) + 1
        if not new_faces:
            break
        mesh.faces = np.vstack([faces, np.array(new_faces, dtype=faces.dtype)])
        total_added += len(new_faces)
        rounds += 1
    if verbose:
        print(f"  [zipper] 缝合裂缝: 追加桥接面 {total_added} 个 (tol={tol:.4f}, {rounds} 轮)")
    return total_added


# --------------------------------------------------------------------------- #
# 双链推进缝合（advancing-front zipper，专治长条接缝裂缝）
# --------------------------------------------------------------------------- #
def zipper_seams(mesh, tol=None, tol_ratio=0.008, max_iter=8, verbose=True):
    """双链推进缝合：把残留边界边按 junction(度!=2) 切成「链」，配对近似反向且相互
    靠近的两条链，用双指针推进(每步选更短对角线)生成规整三角形带，把长条裂缝连续
    拉合。仅追加面、复用已有顶点(焊接代表索引)，不移动/不新增顶点。

    与 zipper_boundary_cracks(逐边就近桥接) 互补：本函数负责成条的长裂缝，
    对分叉/两侧顶点数不等(T 型接头)的缝更鲁棒，净闭合率更高。
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return 0
    verts = np.asarray(mesh.vertices)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    if tol is None:
        tol = diag * tol_ratio
    rung_max = 2.5 * tol                      # 单根「横档」超过此长度则停止该对缝合

    total_added = 0
    rounds = 0
    for _ in range(max_iter):
        faces = np.asarray(mesh.faces)
        canon = canonical_index_map(mesh)
        cf = canon[faces]
        cnt = defaultdict(int)
        face_set = set()
        for t in cf:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            if len({a, b, c}) == 3:
                face_set.add(frozenset((a, b, c)))
            for u, w in ((a, b), (b, c), (c, a)):
                if u != w:
                    cnt[(u, w) if u < w else (w, u)] += 1
        boundary = [e for e, c in cnt.items() if c == 1]
        if not boundary:
            break

        g = defaultdict(set)
        for u, w in boundary:
            g[u].add(w)
            g[w].add(u)

        # ---- 按 junction/端点(度!=2) 切成链 ----
        visited_edges = set()

        def walk_chain(u, w):
            chain = [u, w]
            e0 = (u, w) if u < w else (w, u)
            visited_edges.add(e0)
            prev, cur = u, w
            while len(g[cur]) == 2:
                nxts = [y for y in g[cur] if y != prev]
                if not nxts:
                    break
                nn = nxts[0]
                e = (cur, nn) if cur < nn else (nn, cur)
                if e in visited_edges:
                    break
                visited_edges.add(e)
                chain.append(nn)
                prev, cur = cur, nn
            return chain

        chains = []
        for s in [x for x in g if len(g[x]) != 2]:
            for w in list(g[s]):
                e = (s, w) if s < w else (w, s)
                if e not in visited_edges:
                    chains.append(walk_chain(s, w))
        for (u, w) in boundary:              # 剩余纯环
            e = (u, w) if u < w else (w, u)
            if e not in visited_edges:
                chains.append(walk_chain(u, w))
        chains = [c for c in chains if len(c) >= 2]
        if len(chains) < 2:
            break

        # ---- 顶点 -> 链id, KDTree 投票配对 ----
        chain_of = {}
        for ci, ch in enumerate(chains):
            for v in ch:
                chain_of.setdefault(v, ci)
        allv = sorted(chain_of.keys())
        tree = cKDTree(verts[allv])

        def dist(x, y):
            return float(np.linalg.norm(verts[x] - verts[y]))

        used = set()
        new_faces = []

        def emit(tri):
            if len(set(tri)) < 3:
                return
            key = frozenset(tri)
            if key in face_set:
                return
            if tri_area(verts[tri[0]], verts[tri[1]], verts[tri[2]]) < 1e-16:
                return
            new_faces.append(tri)
            face_set.add(key)

        order = sorted(range(len(chains)), key=lambda i: -len(chains[i]))
        for ai in order:
            if ai in used:
                continue
            A = chains[ai]
            votes = defaultdict(int)
            for v in A:
                for k in tree.query_ball_point(verts[v], tol):
                    cid = chain_of[allv[k]]
                    if cid != ai and cid not in used:
                        votes[cid] += 1
            if not votes:
                continue
            bi = max(votes, key=votes.get)
            if bi in used or votes[bi] < max(2, 0.25 * len(A)):
                continue
            B = chains[bi]
            # 对齐：使 A[0] 与 Bc[0] 为同一物理端
            if dist(A[0], B[0]) > dist(A[0], B[-1]):
                B = B[::-1]
            # ---- 双指针推进缝合 ----
            i = j = 0
            La, Lb = len(A), len(B)
            stop = False
            while (i < La - 1 or j < Lb - 1) and not stop:
                advA = i < La - 1
                advB = j < Lb - 1
                dA = dist(A[i + 1], B[j]) if advA else float("inf")
                dB = dist(A[i], B[j + 1]) if advB else float("inf")
                if min(dA, dB) > rung_max:
                    stop = True
                    break
                if advA and dA <= dB:
                    emit((A[i], A[i + 1], B[j]))
                    i += 1
                elif advB:
                    emit((A[i], B[j + 1], B[j]))
                    j += 1
                else:
                    break
            used.add(ai)
            used.add(bi)

        if not new_faces:
            break
        mesh.faces = np.vstack([faces, np.array(new_faces, dtype=faces.dtype)])
        total_added += len(new_faces)
        rounds += 1
    if verbose:
        print(f"  [seam] 双链推进缝合: 追加面 {total_added} 个 (tol={tol:.4f}, {rounds} 轮)")
    return total_added


# --------------------------------------------------------------------------- #
# 补「空多边形洞」（未匹配半边判据，count 不敏感，四边形/五边形/...通用）
# --------------------------------------------------------------------------- #
def _newell_normal(pts):
    """Newell 法求多边形法线（对非严格平面也稳健）。"""
    n = np.zeros(3)
    m = len(pts)
    for i in range(m):
        a = pts[i]
        b = pts[(i + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-20 else np.array([0.0, 0.0, 1.0])


def fill_empty_polygons(mesh, max_n=6, max_iter=8, verbose=True):
    """补回「空多边形洞」：一圈 n 条边都存在、内部无面的多边形（n=3..max_n）。

    这类洞的边常被外侧相邻面共享(count>=2)，count==1 的边界检测识别不到。改用
    「未匹配有向半边」判据：有向边 (u,v) 被面使用、而反向 (v,u) 未被任何面使用，
    即为未匹配半边（对 count>=2 同样成立）。把未匹配半边串成小闭环(长度<=max_n)，
    即被外侧面统一绕向包住、内部为空的多边形洞；按外侧反向做三角化补上：
      - n==4: best_quad_diagonal（补一条对角线）
      - n>=5: min_weight_triangulation（Liepa 最小权重）
      - n==3: 单个三角形
    仅追加面、复用已有顶点(焊接代表索引)，不移动/不新增顶点。分叉处按转角最平直续走。
    """
    verts = np.asarray(mesh.vertices)
    total = 0
    n_holes = 0
    rounds = 0

    def pick_next(prev, cur, cands):
        """分叉处选转角最平直(最连续)的后继。"""
        if len(cands) == 1:
            return cands[0]
        v_in = verts[cur] - verts[prev]
        nin = np.linalg.norm(v_in)
        if nin < 1e-20:
            return cands[0]
        v_in = v_in / nin
        best, best_s = cands[0], -2.0
        for c in cands:
            v_out = verts[c] - verts[cur]
            no = np.linalg.norm(v_out)
            if no < 1e-20:
                continue
            s = float(v_in @ (v_out / no))
            if s > best_s:
                best, best_s = c, s
        return best

    for _ in range(max_iter):
        faces = np.asarray(mesh.faces)
        canon = canonical_index_map(mesh)
        cf = canon[faces]
        C = defaultdict(int)              # 有向半边计数
        faceset = set()
        for t in cf:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            if len({a, b, c}) == 3:
                faceset.add(frozenset((a, b, c)))
            for u, w in ((a, b), (b, c), (c, a)):
                if u != w:
                    C[(u, w)] += 1
        # 未匹配半边(可用余量) + 出边邻接
        avail = defaultdict(int)
        out = defaultdict(list)
        for (u, w), c in C.items():
            ex = c - C.get((w, u), 0)
            if ex > 0:
                avail[(u, w)] = ex
                out[u].append(w)
        if not avail:
            break

        new_faces = []
        round_added = 0
        for start in list(out.keys()):
            while True:
                first = [w for w in out[start] if avail[(start, w)] > 0]
                if not first:
                    break
                v0 = first[0]
                path = [start]
                used = []
                avail[(start, v0)] -= 1
                used.append((start, v0))
                prev, cur = start, v0
                closed = False
                while len(path) <= max_n:
                    if cur == start:
                        closed = True
                        break
                    path.append(cur)
                    cands = [w for w in out[cur] if avail[(cur, w)] > 0]
                    if not cands:
                        break
                    nn = pick_next(prev, cur, cands)
                    avail[(cur, nn)] -= 1
                    used.append((cur, nn))
                    prev, cur = cur, nn
                loop = path        # 若闭合, path=[start, v1, ..., v_{n-1}]
                ok = closed and 3 <= len(loop) <= max_n and len(set(loop)) == len(loop)
                if ok:
                    # 外侧遍历方向为 loop(未匹配方向)，填充取反向以对齐周围面朝向
                    fill_loop = loop[::-1]
                    pts = [verts[x] for x in fill_loop]
                    n = len(fill_loop)
                    if n == 3:
                        tris_local = [(0, 1, 2)]
                    else:
                        ref = _newell_normal(np.array(pts))
                        edge_normals = [ref] * n
                        if n == 4:
                            tris_local = best_quad_diagonal(pts, edge_normals, ref)
                        else:
                            tris_local = min_weight_triangulation(pts, edge_normals, ref)
                    # 排除会与现有面重复的三角形
                    good = []
                    for (i, j, k) in tris_local:
                        tri = (fill_loop[i], fill_loop[j], fill_loop[k])
                        if len(set(tri)) < 3:
                            continue
                        if frozenset(tri) in faceset:
                            continue
                        good.append(tri)
                        faceset.add(frozenset(tri))
                    if good:
                        new_faces.extend(good)
                        round_added += 1
                else:
                    # walk 失败：回退本次消费，避免破坏数据/死循环
                    for e in used:
                        avail[e] += 1
                    break
        if not new_faces:
            break
        mesh.faces = np.vstack([faces, np.array(new_faces, dtype=faces.dtype)])
        total += len(new_faces)
        n_holes += round_added
        rounds += 1
    if verbose:
        print(f"  [polyfill] 补空多边形洞: 追加面 {total} 个 "
              f"(约 {n_holes} 个洞, n<={max_n}, {rounds} 轮)")
    return total


# --------------------------------------------------------------------------- #
# 删除自相交面（三角形面片穿插相交）
# --------------------------------------------------------------------------- #
def _seg_tri_cross(P0, P1, V0, V1, V2, eps=1e-9):
    """向量化线段-三角形「穿刺」判定（Möller–Trumbore）。

    P0,P1: (N,3) 线段端点；V0,V1,V2: (N,3) 三角形顶点。
    返回 (N,) 布尔：线段是否真正穿过三角形内部（严格内部 + 交点严格落在
    线段中间），用于识别面片穿插相交，排除仅共点/共边的贴合。
    """
    d = P1 - P0
    e1 = V1 - V0
    e2 = V2 - V0
    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)          # 行列式
    parallel = np.abs(a) < eps                # 平行/共面 -> 不算穿刺
    a_safe = np.where(parallel, 1.0, a)
    f = 1.0 / a_safe
    s = P0 - V0
    u = f * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = f * np.einsum('ij,ij->i', d, q)
    t = f * np.einsum('ij,ij->i', e2, q)
    m = eps                                   # 严格内部余量，排除边/顶点贴合
    ok = (~parallel) & (u > m) & (v > m) & (u + v < 1.0 - m) & (t > m) & (t < 1.0 - m)
    return ok


def remove_self_intersections(mesh, verbose=True, seed=0):
    """检测三角形面片「穿插相交」，随机删除相交对中的一个面（仅删面，不动顶点）。

    - broad-phase：用 trimesh 的 AABB 空间索引(triangles_tree) 找包围盒重叠的候选面对。
    - narrow-phase：向量化线段-三角形穿刺测试（任一三角形的边穿过另一三角形内部
      即判为穿插），排除共享顶点(焊接口径)的相邻面。
    - 消解：对每个相交面对，若两面都还在，则随机删除其一（贪心，保证每对被覆盖）。
      删面可能暴露新的边界，交由后续补洞/缝合处理。
    """
    faces = np.asarray(mesh.faces)
    F = len(faces)
    if F == 0:
        return 0
    try:
        tree = mesh.triangles_tree
    except Exception:
        if verbose:
            print("  [selfx] 缺少 rtree，跳过自相交检测")
        return 0
    tris = np.asarray(mesh.triangles)         # (F,3,3)
    canon = canonical_index_map(mesh)
    cf = canon[faces]
    tri_bounds = np.hstack([tris.min(axis=1), tris.max(axis=1)])  # (F,6)

    # ---- broad-phase：AABB 重叠候选面对 ----
    cand = []
    cf_sets = [set(map(int, cf[i])) for i in range(F)]
    for i in range(F):
        for j in tree.intersection(tri_bounds[i]):
            j = int(j)
            if j <= i:
                continue
            if cf_sets[i] & cf_sets[j]:        # 焊接口径下共享顶点 -> 相邻面，跳过
                continue
            cand.append((i, j))
    if not cand:
        if verbose:
            print("  [selfx] 未检测到穿插相交面对")
        return 0
    cand = np.asarray(cand, dtype=np.int64)
    ai, bi = cand[:, 0], cand[:, 1]

    # ---- narrow-phase：A 的三条边穿 B 内部，或 B 的三条边穿 A 内部 ----
    A = tris[ai]                              # (M,3,3)
    B = tris[bi]
    hit = np.zeros(len(cand), dtype=bool)
    for (p, q) in ((0, 1), (1, 2), (2, 0)):   # A 的边 vs 三角形 B
        hit |= _seg_tri_cross(A[:, p], A[:, q], B[:, 0], B[:, 1], B[:, 2])
    for (p, q) in ((0, 1), (1, 2), (2, 0)):   # B 的边 vs 三角形 A
        hit |= _seg_tri_cross(B[:, p], B[:, q], A[:, 0], A[:, 1], A[:, 2])
    inter = cand[hit]
    if len(inter) == 0:
        if verbose:
            print("  [selfx] 未检测到穿插相交面对")
        return 0

    # ---- 消解：每对随机删一个，贪心覆盖 ----
    rng = np.random.RandomState(seed)
    removed = set()
    for i, j in inter:
        i, j = int(i), int(j)
        if i in removed or j in removed:
            continue
        removed.add(i if rng.rand() < 0.5 else j)
    keep = [k for k in range(F) if k not in removed]
    mesh.faces = faces[keep]
    if verbose:
        print(f"  [selfx] 穿插相交面对 {len(inter)} 组 | 删除相交面 {len(removed)} 个 | 剩余面 {len(keep)}")
    return len(removed)


# --------------------------------------------------------------------------- #
# 删除重复面 / 退化面（焊接口径）
# --------------------------------------------------------------------------- #
def remove_duplicate_faces(mesh, verbose=True):
    """删除「重复面」与「退化面」（焊接口径，仅删面不动顶点）。

    - 重复面：焊接后顶点集合(无序,忽略绕向)相同的面，只保留首次出现的一个。
      删除完全重合的副本不会产生新洞（原位置仍被保留的那一份覆盖），
      但能让被副本"多重覆盖"的边引用数下降，暴露出真实边界，利于后续补洞。
    - 退化面：焊接后不足 3 个不同顶点的面（无面积），直接删除。
    保留原始出现顺序中的首个，尽量维持原拓扑。
    """
    canon = canonical_index_map(mesh)
    faces = np.asarray(mesh.faces)
    cf = canon[faces]
    seen = set()
    keep = []
    n_dup = 0
    n_degen = 0
    for i, t in enumerate(cf):
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        key = frozenset((a, b, c))
        if len(key) < 3:                     # 退化面
            n_degen += 1
            continue
        if key in seen:                      # 重复面
            n_dup += 1
            continue
        seen.add(key)
        keep.append(i)
    mesh.faces = faces[keep]
    if verbose:
        print(f"  [dedup] 删除重复面 {n_dup} 个 + 退化面 {n_degen} 个 | 剩余面 {len(keep)}")
    return n_dup + n_degen


# --------------------------------------------------------------------------- #
# 删除悬空面
# --------------------------------------------------------------------------- #
def remove_dangling_faces(mesh, verbose=True, max_iter=1000):
    """删除「悬空面」：焊接口径下，某面 3 条边中恰有 2 条是边界边(仅属该面, count==1)，
    且剩下 1 条是非流形边(被 >2 面共享, count>2)。这类面仅靠一条非流形边挂在网格上，
    两条自由边悬空，属明显冗余碎片。

    迭代删除直到收敛（删除后暴露的新悬空面继续删）。仅删除面，不删除/移动任何顶点。
    """
    faces = np.asarray(mesh.faces)
    canon = canonical_index_map(mesh)          # 顶点不变 -> canon 全程不变
    total_removed = 0
    rounds = 0
    while rounds < max_iter:
        cf = canon[faces]
        cnt = defaultdict(int)
        for tri in cf:
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v in ((a, b), (b, c), (c, a)):
                if u != v:
                    cnt[(u, v) if u < v else (v, u)] += 1

        remove_mask = np.zeros(len(faces), dtype=bool)
        for fi, tri in enumerate(cf):
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            counts = []
            for u, v in ((a, b), (b, c), (c, a)):
                if u != v:
                    counts.append(cnt[(u, v) if u < v else (v, u)])
            if len(counts) != 3:               # 退化面(有重合顶点)跳过
                continue
            n_free = sum(1 for e in counts if e == 1)      # 边界边(悬空)
            n_nonmanifold = sum(1 for e in counts if e > 2)  # 非流形边
            if n_free == 2 and n_nonmanifold == 1:
                remove_mask[fi] = True

        if not remove_mask.any():
            break
        faces = faces[~remove_mask]
        total_removed += int(remove_mask.sum())
        rounds += 1

    mesh.faces = faces
    if verbose:
        print(f"  [dangling] 删除悬空面 {total_removed} 个（{rounds} 轮迭代）| 剩余面 {len(faces)}")
    return total_removed


# --------------------------------------------------------------------------- #
# 导出（保留全部顶点，含删除悬空面后产生的孤立顶点）
# --------------------------------------------------------------------------- #
def export_obj_keep_all_vertices(mesh, path, write_normals=True):
    """手动写 OBJ：写出 mesh.vertices 中的每一个顶点（包括未被任何面引用的孤立点），
    避免 trimesh.export 自动剔除未引用顶点。用 17 位有效数字保证 float64 精确 round-trip。"""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    vn = np.asarray(mesh.vertex_normals) if write_normals else None
    with open(path, "w") as fp:
        for p in v:
            fp.write(f"v {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
        if write_normals:
            for nrm in vn:
                fp.write(f"vn {nrm[0]:.17g} {nrm[1]:.17g} {nrm[2]:.17g}\n")
        for tri in f:
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1  # OBJ 为 1-based
            if write_normals:
                fp.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                fp.write(f"f {a} {b} {c}\n")


# --------------------------------------------------------------------------- #
# 核心 pipeline（在内存网格上执行，可被其它脚本复用）
# --------------------------------------------------------------------------- #
def run_fill_pipeline(mesh, refine=False, remove_dangling=False, dedup=True,
                      zipper=True, zipper_tol=None, quadfill=True, poly_max_n=6,
                      remove_intersect=False, orient_new=True, verbose=True,
                      orient_anchor=None):
    """在内存网格上执行完整补洞 pipeline，返回 (result_mesh, stats)。

    流程：去相交(可选) -> 去重复面(可选) -> 迭代「补洞->补空多边形->双链缝合->
    逐边缝合->删悬空面(可选)」直到收敛 -> 仅新增面朝向对齐(可选)。
    仅追加面、ΔV=0（顶点不变）。评估(count_holes/边界边)在朝向修正之前完成，
    避免绕向翻转干扰指标。stats 含各阶段耗时 tm 与各类计数。

    orient_anchor：朝向对齐时作为「不可翻转锚点」的前缀面数；None 时默认取去重后
    的全部输入面(orig_faces)。组合补洞(combine)场景传入「真实原始面」数量，使去重
    后前缀恰为存活原始面，从而让 meshflow 追加面也被对齐(而非被当成锚点)。"""
    tm = {"selfx": 0.0, "dedup": 0.0, "fill": 0.0, "quad": 0.0,
          "seam": 0.0, "zip": 0.0, "dangling": 0.0, "orient": 0.0}
    n_selfx = 0
    if remove_intersect:
        mesh = mesh.copy()
        _ts = time.perf_counter()
        n_selfx = remove_self_intersections(mesh, verbose=verbose)
        tm["selfx"] += time.perf_counter() - _ts
    n_dedup = 0
    if dedup:
        mesh = mesh.copy()
        _ts = time.perf_counter()
        n_dedup = remove_duplicate_faces(mesh, verbose=verbose)
        tm["dedup"] += time.perf_counter() - _ts
    holes_before = count_holes(mesh)         # 预处理后可闭合洞

    result = mesh
    orig_faces = len(mesh.faces)
    total_removed = total_zip = total_seam = total_quad = 0
    iters = 0
    max_iter = 8
    for it in range(max_iter):
        before_faces = len(result.faces)
        _ts = time.perf_counter()
        result = fill_liepa(result, do_refine=refine, fix_orientation=False,
                            verbose=(verbose and it == 0))
        tm["fill"] += time.perf_counter() - _ts
        added = len(result.faces) - before_faces
        _ts = time.perf_counter()
        q = fill_empty_polygons(result, max_n=poly_max_n,
                                verbose=(verbose and it == 0)) if quadfill else 0
        tm["quad"] += time.perf_counter() - _ts
        total_quad += q
        _ts = time.perf_counter()
        s = zipper_seams(result, tol=zipper_tol,
                         verbose=(verbose and it == 0)) if zipper else 0
        tm["seam"] += time.perf_counter() - _ts
        total_seam += s
        _ts = time.perf_counter()
        z = zipper_boundary_cracks(result, tol=zipper_tol,
                                   verbose=(verbose and it == 0)) if zipper else 0
        tm["zip"] += time.perf_counter() - _ts
        total_zip += z
        _ts = time.perf_counter()
        r = remove_dangling_faces(result, verbose=(verbose and it == 0)) if remove_dangling else 0
        tm["dangling"] += time.perf_counter() - _ts
        total_removed += r
        iters = it + 1
        if added == 0 and q == 0 and s == 0 and z == 0 and r == 0:  # 全部收敛
            break

    # —— 评估（count_holes 依赖绕向，必须在朝向修正之前） ——
    holes_after = count_holes(result)
    be_after = count_boundary_edges(result)
    add_faces = len(result.faces) - orig_faces + total_removed

    # 收尾：仅修正「新增面」朝向，使其法向朝内/朝外与相邻锚点面对齐（锚点面绝对不动）
    n_flip = 0
    anchor = orig_faces if orient_anchor is None else max(0, min(orient_anchor, len(result.faces)))
    if orient_new and not remove_dangling and len(result.faces) > anchor:
        _ts = time.perf_counter()
        before = np.asarray(result.faces).copy()
        orient_new_faces_by_neighbors(result, anchor, verbose=verbose)
        n_flip = int(np.any(before != np.asarray(result.faces), axis=1).sum())
        tm["orient"] += time.perf_counter() - _ts

    stats = {"tm": tm, "n_selfx": n_selfx, "n_dedup": n_dedup,
             "total_quad": total_quad, "total_seam": total_seam,
             "total_zip": total_zip, "total_removed": total_removed,
             "iters": iters, "orig_faces": orig_faces, "n_flip": n_flip,
             "add_faces": add_faces, "holes_before": holes_before,
             "holes_after": holes_after, "be_after": be_after}
    return result, stats


# --------------------------------------------------------------------------- #
# meshflow 图算法（原 fill_meshflow.py，几何盲、纯拓扑）
# --------------------------------------------------------------------------- #
def triangulate_quad_rings(adj: np.ndarray) -> np.ndarray:
    # thinks meshflow@CVPR2026!
    num = int(adj.shape[0])
    if num < 4:
        return np.empty((0, 3), dtype=np.int32)
    nbr_sets = [set(np.nonzero(adj[v])[0].tolist()) for v in range(num)]
    new_faces: list[list[int]] = []
    seen: set[tuple[int, int, int]] = set()
    for a in range(num):
        cand = sorted(v for v in nbr_sets[a] if v > a)
        n_cand = len(cand)
        if n_cand < 2:
            continue
        for i in range(n_cand):
            b = cand[i]
            set_b = nbr_sets[b]
            for j in range(i + 1, n_cand):
                d = cand[j]
                if d in set_b:
                    continue
                for c in set_b & nbr_sets[d]:
                    if c <= a or c in nbr_sets[a]:
                        continue
                    for tri in ([a, b, c], [a, c, d]):
                        key = tuple(sorted(tri))
                        if key in seen:
                            continue
                        seen.add(key)
                        new_faces.append(tri)
    if not new_faces:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(new_faces, dtype=np.int32)


def triangulate_poly_rings(adj: np.ndarray, max_n: int = 6) -> np.ndarray:
    """triangulate_quad_rings 的泛化版：补全长度 4..max_n 的「空多边形环」。

    「空多边形环」= 图中长度为 k 的无弦环(chordless cycle)：环上相邻顶点有边、
    任意不相邻顶点之间无边(无弦)。找到后从环上最小顶点 a 做扇形三角化。
    max_n=4 时与 triangulate_quad_rings 完全等价。"""
    num = int(adj.shape[0])
    if num < 4 or max_n < 4:
        return np.empty((0, 3), dtype=np.int32)
    nbr = [set(np.nonzero(adj[v])[0].tolist()) for v in range(num)]
    new_faces: list[list[int]] = []
    seen_tri: set[tuple[int, int, int]] = set()
    seen_cycle: set[tuple] = set()

    def emit_cycle(cyc: list[int]) -> None:
        interior = tuple(cyc[1:])
        rev = tuple(reversed(interior))
        key = (cyc[0],) + (interior if interior <= rev else rev)
        if key in seen_cycle:
            return
        seen_cycle.add(key)
        a = cyc[0]
        for i in range(1, len(cyc) - 1):           # 从 a 扇形三角化
            tri = [a, cyc[i], cyc[i + 1]]
            tkey = tuple(sorted(tri))
            if len(set(tkey)) < 3 or tkey in seen_tri:
                continue
            seen_tri.add(tkey)
            new_faces.append(tri)

    def dfs(path: list[int], inpath: set) -> None:
        a = path[0]
        last = path[-1]
        interior = path[1:-1]
        if len(path) >= max_n:
            return
        for w in nbr[last]:
            if w <= a or w in inpath:
                continue
            if any(w in nbr[p] for p in interior):  # 与内部相邻=弦，跳过
                continue
            if w in nbr[a]:                          # 与 a 相邻 -> 只能在此闭环
                if len(path) + 1 >= 4:
                    emit_cycle(path + [w])
                continue
            path.append(w)
            inpath.add(w)
            dfs(path, inpath)
            path.pop()
            inpath.discard(w)

    for a in range(num):
        for b in nbr[a]:
            if b <= a:
                continue
            dfs([a, b], {a, b})

    if not new_faces:
        return np.empty((0, 3), dtype=np.int32)
    return np.asarray(new_faces, dtype=np.int32)


def edges_to_faces(
    edges: np.ndarray, num_valid: int, fill_quad_rings: bool, ring_max_n: int = 4
) -> np.ndarray:
    adj = np.zeros((num_valid, num_valid), dtype=bool)
    if edges.shape[0] > 0:
        adj[edges[:, 0], edges[:, 1]] = True
        adj[edges[:, 1], edges[:, 0]] = True
    faces_list = []
    for ei in range(edges.shape[0]):
        u = int(edges[ei, 0])
        v = int(edges[ei, 1])
        common = np.where(adj[u] & adj[v])[0]
        common = common[common > v]
        for w in common:
            faces_list.append([u, v, int(w)])
    faces = (
        np.asarray(faces_list, dtype=np.int32)
        if faces_list
        else np.empty((0, 3), dtype=np.int32)
    )
    if fill_quad_rings:
        quad = triangulate_poly_rings(adj, max_n=ring_max_n)
        if quad.shape[0] > 0:
            faces = np.concatenate([faces, quad], axis=0) if faces.shape[0] else quad
    return faces


def mesh_edges(mesh):
    """从网格现有面提取无向唯一边 (E,2)，剔除退化边(u==v)。"""
    f = np.asarray(mesh.faces)
    if len(f) == 0:
        return np.empty((0, 2), dtype=np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0)
    return e.astype(np.int64)


# --------------------------------------------------------------------------- #
# 组合补洞（原 fill_combine.py）
# --------------------------------------------------------------------------- #
def _surviving_orig_count(mesh):
    """去重复面后仍会存活的「原始面」数量 = 原始面按焊接口径去重后的不同面数。

    meshflow_append 把原始面放在最前，run_fill_pipeline 内部去重保留每个焊接键的
    首次出现，故存活原始面恰好是这些不同键、且位于面数组最前缀。该值用作朝向对齐
    的锚点前缀，使 meshflow 追加面也被对齐。"""
    canon = canonical_index_map(mesh)
    keyset = set()
    for t in np.asarray(mesh.faces):
        ct = canon[t]
        fk = frozenset((int(ct[0]), int(ct[1]), int(ct[2])))
        if len(fk) == 3:                      # 排除退化面(去重也会删)
            keyset.add(fk)
    return len(keyset)


def meshflow_append(mesh, fill_quad_rings=True, ring_max_n=6, bnd_verts=2,
                    verbose=False):
    """meshflow 图算法补洞(append 模式)：保留原始面，仅追加新生成、原本不存在的面。
    仅追加面、顶点不变。返回 (new_mesh, n_added)。

    bnd_verts：候选面必须至少含该数量个「原始边界顶点」才被采纳（0=不筛，全量追加）。
    动机：edges_to_faces 会物化邻接图里**所有** 3-团，其中大量三角位于网格内部、
    根本不接触任何边界 —— 它们封不了任何洞，纯属冗余，却会在渲染时造成大量 z-fighting
    （实测这类冗余面使 z-fighting 由输入的 3.8% 升到 6.6%）。只保留贴着原始边界的候选面
    即可在几乎不牺牲封边能力的前提下显著减少重合穿模。取值越大越激进(1/2/3)。"""
    orig_faces = np.asarray(mesh.faces)
    N = int(len(mesh.vertices))
    edges = mesh_edges(mesh)
    gen = edges_to_faces(edges, N, fill_quad_rings, ring_max_n=ring_max_n)
    seen = set(tuple(sorted(int(x) for x in t)) for t in orig_faces)

    canon = None
    bnd_v = None
    if bnd_verts and bnd_verts > 0:
        # 焊接口径下的原始边界顶点集合（边界边=仅被 1 个面引用的边）
        canon = canonical_index_map(mesh)
        cf = canon[orig_faces]
        ec = defaultdict(int)
        for t in cf:
            x, y, z = int(t[0]), int(t[1]), int(t[2])
            for p, q in ((x, y), (y, z), (z, x)):
                if p != q:
                    ec[(p, q) if p < q else (q, p)] += 1
        bnd_v = set()
        for (p, q), v in ec.items():
            if v == 1:
                bnd_v.add(p)
                bnd_v.add(q)

    new = []
    n_drop = 0
    for t in gen:
        k = tuple(sorted(int(x) for x in t))
        if len(set(k)) < 3 or k in seen:
            continue
        seen.add(k)
        if bnd_v is not None:
            nb = sum(1 for i in t if int(canon[int(i)]) in bnd_v)
            if nb < bnd_verts:
                n_drop += 1
                continue                          # 不挨边界 -> 封不了洞, 丢弃
        new.append([int(t[0]), int(t[1]), int(t[2])])
    if new:
        faces = np.vstack([orig_faces, np.array(new, dtype=orig_faces.dtype)])
    else:
        faces = orig_faces
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                          faces=faces, process=False)
    if verbose and n_drop:
        print(f"  [meshflow] 丢弃不接触边界的冗余候选面 {n_drop} 个"
              f"(bnd_verts={bnd_verts})")
    return out, len(faces) - len(orig_faces)


def process(path, out_dir, fill_quad_rings=True, write_normals=True,
            poly_max_n=6, zipper_tol=None, ring_max_n=4,
            global_orient=True, orient_only=False, suffix="orient",
            mf_bnd_verts=2):
    print(f"\n>>> 处理 {path}")
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    n_vert = len(mesh.vertices)
    orig_faces_in = len(mesh.faces)
    n_anchor = _surviving_orig_count(mesh)      # 朝向对齐锚点=存活原始面数
    holes0 = count_holes(mesh)
    be0 = count_boundary_edges(mesh)
    cf0, mani0 = count_orientation_conflicts(mesh)
    zf0 = zfight_ratio(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    print(f"    原始: 顶点 {n_vert} 面 {orig_faces_in} "
          f"可闭合洞 {holes0} 个 | 真实边界边 {be0} | "
          f"朝向矛盾边 {cf0}/{mani0} ({cf0 / max(mani0, 1) * 100:.1f}%) | "
          f"z-fighting {zf0:.1f}%")
    stem = os.path.splitext(os.path.basename(path))[0]

    t0 = time.perf_counter()
    if orient_only:
        # 只做法向矫正、不补洞
        result = mesh
        st = {"iters": 0, "n_dedup": 0, "total_quad": 0, "total_seam": 0,
              "total_zip": 0, "n_flip": 0, "add_faces": 0,
              "holes_before": holes0, "holes_after": holes0, "be_after": be0}
        mf_added = 0
    else:
        # —— 阶段1：meshflow 图算法先补 ——
        _ts = time.perf_counter()
        mf_mesh, mf_added = meshflow_append(mesh, fill_quad_rings=fill_quad_rings,
                                            ring_max_n=ring_max_n,
                                            bnd_verts=mf_bnd_verts, verbose=True)
        t_mf = time.perf_counter() - _ts
        holes1 = count_holes(mf_mesh)
        be1 = count_boundary_edges(mf_mesh)
        print(f"  [阶段1 meshflow] max_n={ring_max_n} 追加面 {mf_added} | 可闭合洞 {holes0}→{holes1} | "
              f"真实边界边 {be0}→{be1} | 耗时 {t_mf:.3f}s")

        # —— 阶段2：fill_liepa 完整 pipeline 再补 ——
        _ts = time.perf_counter()
        result, st = run_fill_pipeline(
            mf_mesh, dedup=True, zipper=True, zipper_tol=zipper_tol,
            quadfill=True, poly_max_n=poly_max_n, remove_intersect=False,
            orient_new=True, verbose=True, orient_anchor=n_anchor)
        t_lp = time.perf_counter() - _ts
        print(f"  [阶段2 fill_liepa] {st['iters']} 轮 | 去重复面 {st['n_dedup']} | "
              f"补空多边形面 {st['total_quad']} | 双链缝合面 {st['total_seam']} | "
              f"缝合裂缝面 {st['total_zip']} | 对齐翻转面 {st['n_flip']} | "
              f"可闭合洞 {st['holes_before']}→{st['holes_after']} | "
              f"真实边界边 {be1}→{st['be_after']} | 耗时 {t_lp:.3f}s")

    # —— 阶段3：全局法向统一矫正（允许翻转原始面；仅改绕向, 顶点/面数不变）——
    gst = None
    if global_orient:
        _ts = time.perf_counter()
        V_chk = np.asarray(result.vertices).copy()
        nf_chk = len(result.faces)
        result, gst = global_orient_fix(result, verbose=True)
        t_go = time.perf_counter() - _ts
        # 不变量自检：顶点坐标与面数必须完全不变
        assert np.array_equal(V_chk, np.asarray(result.vertices)), "全局矫正改动了顶点！"
        assert nf_chk == len(result.faces), "全局矫正改动了面数！"
        print(f"  [阶段3 全局法向矫正] 耗时 {t_go:.3f}s")

    if write_normals:
        _ = result.vertex_normals

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stem}_{suffix}.obj")
    export_obj_keep_all_vertices(result, out_path, write_normals=write_normals)
    elapsed = time.perf_counter() - t0

    dv = len(result.vertices) - n_vert
    total_add = len(result.faces) - orig_faces_in
    filled = holes0 - st["holes_after"]
    fill_rate = (filled / holes0 * 100.0) if holes0 > 0 else 0.0
    print(f"     已保存 {out_path}"
          + ("（含顶点法线）" if write_normals else ""))
    # 面数守恒: 最终 = 原始 + meshflow追加 - 去重 + fill_liepa净补
    print(f"       总耗时 {elapsed:.3f}s | 合计追加面 {total_add} "
          f"(meshflow +{mf_added}, 去重 -{st['n_dedup']}, fill_liepa +{st['add_faces']}) | "
          f"可闭合洞 {holes0}→{st['holes_after']} (填 {filled}, 补洞率 {fill_rate:.1f}%) | "
          f"真实边界边 {be0}→{st['be_after']}")
    cf1, mani1 = count_orientation_conflicts(result)
    zf1 = zfight_ratio(np.asarray(result.vertices), np.asarray(result.faces))
    print(f"       最终面 {len(result.faces)} | 顶点 {len(result.vertices)} "
          f"(Δ{dv:+d}, 原顶点保持) | 朝向矛盾边 {cf0}/{mani0} → {cf1}/{mani1} "
          f"({cf0 / max(mani0, 1) * 100:.1f}% → {cf1 / max(mani1, 1) * 100:.1f}%) | "
          f"z-fighting {zf0:.1f}% → {zf1:.1f}%")


def _batch_worker(job):
    """子进程执行体：调用 process_fn 并整块捕获其 stdout 返回，避免多进程输出交错。"""
    fn, path, out_dir, kwargs = job
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(path, out_dir, **kwargs)
        except Exception:
            import traceback
            traceback.print_exc()
    return buf.getvalue()


def run_batch(process_fn, inputs, out_dir, kwargs, jobs=0):
    """按文件多进程并行执行 process_fn(path, out_dir, **kwargs)。

    各文件互不依赖(仅读取输入、独立追加/导出)，故按文件并行不改变任何补洞结果。
    每个文件的 stdout 整块捕获后按输入顺序打印，输出与串行完全一致、不交错。
    jobs: <=0 自动取 min(cpu, 文件数); 1 串行; >1 指定进程数。"""
    n = len(inputs)
    if jobs is None or jobs <= 0:
        jobs = min(os.cpu_count() or 1, n)
    jobs = max(1, min(jobs, n))
    job_list = [(process_fn, p, out_dir, dict(kwargs)) for p in inputs]
    if jobs == 1 or n == 1:
        for job in job_list:
            print(_batch_worker(job), end="")
        return
    print(f"[并行] 使用 {jobs} 个进程处理 {n} 个文件")
    from multiprocessing import Pool
    with Pool(processes=jobs) as pool:
        for out in pool.imap(_batch_worker, job_list):
            print(out, end="")


def main():
    ap = argparse.ArgumentParser(
        description="组合补洞 + 全局法向统一矫正（单文件自包含）")
    ap.add_argument("inputs", nargs="+", help="输入 .obj 文件")
    ap.add_argument("--out", default="orient_output",
                    help="输出目录 (默认 ./orient_output)")
    ap.add_argument("--suffix", default="orient",
                    help="输出文件名后缀 (默认 orient -> {stem}_orient.obj)")
    ap.add_argument("--no-normals", dest="normals", action="store_false",
                    help="不写入顶点法线(默认写入)")
    ap.add_argument("--no-quad-rings", dest="quad_rings", action="store_false",
                    help="meshflow 阶段不补空多边形环(默认补)")
    ap.add_argument("--ring-max-n", dest="ring_max_n", type=int, default=4,
                    help="meshflow 阶段补空多边形环的最大边数(默认4=仅四环; 6=含五/六边, 面数约翻倍)")
    ap.add_argument("--poly-max-n", dest="poly_max_n", type=int, default=6,
                    help="fill_liepa 空多边形洞最大边数(默认6)")
    ap.add_argument("--zipper-tol", dest="zipper_tol", type=float, default=None,
                    help="缝合间距阈值(坐标单位); 默认取包围盒对角线*0.005")
    ap.add_argument("--mf-bnd-verts", dest="mf_bnd_verts", type=int, default=2,
                    help="meshflow 候选面需含的最少「原始边界顶点」数(默认2; 0=不筛=全量追加)。"
                         "越大丢弃的内部冗余面越多、渲染 z-fighting 越低, 但封边略弱")
    ap.add_argument("--no-global-orient", dest="global_orient", action="store_false",
                    help="关闭补洞后的全局法向统一矫正(默认开启; 关闭则等价纯组合补洞)")
    ap.add_argument("--orient-only", dest="orient_only", action="store_true",
                    help="只做全局法向矫正、不补洞(用于单独修复法向混乱的网格)")
    ap.add_argument("-j", "--jobs", dest="jobs", type=int, default=0,
                    help="并行进程数(默认0=自动取min(CPU核数,文件数); 1=串行)。按文件并行")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"输出目录: {os.path.abspath(args.out)}")

    t_all = time.perf_counter()
    kwargs = dict(fill_quad_rings=args.quad_rings, write_normals=args.normals,
                  poly_max_n=args.poly_max_n, zipper_tol=args.zipper_tol,
                  ring_max_n=args.ring_max_n, global_orient=args.global_orient,
                  orient_only=args.orient_only, suffix=args.suffix,
                  mf_bnd_verts=args.mf_bnd_verts)
    run_batch(process, args.inputs, args.out, kwargs, jobs=args.jobs)
    print(f"\n总计耗时 {time.perf_counter() - t_all:.3f}s")


if __name__ == "__main__":
    main()
