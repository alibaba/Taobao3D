"""Fill mesh holes and normalize face orientation without moving vertices."""
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
# Geometry
# --------------------------------------------------------------------------- #
def tri_normal(a, b, c):
    """Return a unit triangle normal."""
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-20 else np.zeros(3)


def tri_area(a, b, c):
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


# --------------------------------------------------------------------------- #
# Boundary loops
# --------------------------------------------------------------------------- #
def canonical_index_map(mesh, decimals=8):
    """Map coincident vertices to canonical indices."""
    rep = {}
    canon = np.empty(len(mesh.vertices), dtype=np.int64)
    for k, v in enumerate(mesh.vertices):
        key = tuple(np.round(v, decimals))
        if key not in rep:
            rep[key] = k
        canon[k] = rep[key]
    return canon


def build_edge_face_map(mesh, canon=None):
    """Map undirected edges to incident faces."""
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
    """Find closed boundary loops in inconsistent meshes."""
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
        """Choose the straightest continuation at a junction."""
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

    # Extract simple undirected loops.
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
            continue                         # Defer junctions
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

    # Greedily trace remaining directed half-edges.
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
                    out[b].append(a)         # Reverse the adjacent face edge
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
    """Return the face normal adjacent to a boundary edge."""
    key = (a, b) if a < b else (b, a)
    fs = emap.get(key, [])
    if not fs:
        return np.zeros(3)
    return mesh.face_normals[fs[0]]


# --------------------------------------------------------------------------- #
# Minimum-weight triangulation
# --------------------------------------------------------------------------- #
def min_weight_triangulation(pts, edge_normals, ref_normal):
    """Triangulate an ordered polygon by dihedral and area cost."""
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
# Quadrilateral holes
# --------------------------------------------------------------------------- #
def best_quad_diagonal(pts, edge_normals, ref_normal):
    """Choose the valid, smoothest diagonal for a quadrilateral hole."""
    pts = [np.asarray(p, dtype=float) for p in pts]

    # Use the Newell normal for signed-area tests.
    poly_n = _newell_normal(np.array(pts))
    pn = np.linalg.norm(poly_n)
    if pn < 1e-20:
        return []                              # Degenerate quadrilateral
    poly_n = poly_n / pn

    def signed_area(i, j, k):
        return 0.5 * float(np.dot(np.cross(pts[j] - pts[i], pts[k] - pts[i]), poly_n))

    quad_area = tri_area(pts[0], pts[1], pts[2]) + tri_area(pts[0], pts[2], pts[3])
    eps_a = 1e-9 * max(quad_area, 1e-30)       # Relative area tolerance

    # Candidate diagonal 0-2.
    validA = signed_area(0, 1, 2) > eps_a and signed_area(0, 2, 3) > eps_a
    # Candidate diagonal 1-3.
    validB = signed_area(0, 1, 3) > eps_a and signed_area(1, 2, 3) > eps_a

    if not validA and not validB:
        return []                              # No valid internal diagonal
    if validA and not validB:
        return [(0, 1, 2), (0, 2, 3)]
    if validB and not validA:
        return [(0, 1, 3), (1, 2, 3)]

    # Choose the smoother valid diagonal.
    def orient(nrm):
        return -nrm if np.dot(nrm, ref_normal) < 0 else nrm

    def en(k):
        return orient(edge_normals[k])

    def tnorm(i, j, k):
        return orient(tri_normal(pts[i], pts[j], pts[k]))

    def dih(n1, n2):
        return 1.0 - max(-1.0, min(1.0, float(np.dot(n1, n2))))

    # Score diagonal 0-2.
    nA1, nA2 = tnorm(0, 1, 2), tnorm(0, 2, 3)
    costA = max(dih(nA1, en(0)), dih(nA1, en(1)),
                dih(nA2, en(2)), dih(nA2, en(3)), dih(nA1, nA2))
    areaA = tri_area(pts[0], pts[1], pts[2]) + tri_area(pts[0], pts[2], pts[3])

    # Score diagonal 1-3.
    nB1, nB2 = tnorm(0, 1, 3), tnorm(1, 2, 3)
    costB = max(dih(nB1, en(0)), dih(nB1, en(3)),
                dih(nB2, en(1)), dih(nB2, en(2)), dih(nB1, nB2))
    areaB = tri_area(pts[0], pts[1], pts[3]) + tri_area(pts[1], pts[2], pts[3])

    if (costA, areaA) <= (costB, areaB):
        return [(0, 1, 2), (0, 2, 3)]
    return [(0, 1, 3), (1, 2, 3)]


# --------------------------------------------------------------------------- #
# Patch refinement
# --------------------------------------------------------------------------- #
def refine_and_fair(local_pts, local_faces, n_boundary, target_len,
                    fair_iters=10, max_refine=200):
    """Refine and smooth a patch while fixing boundary vertices."""
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
# Patch orientation
# --------------------------------------------------------------------------- #
def orient_patch_faces(patch, boundary_dir, verts, ref_normal):
    """Align patch winding with adjacent faces."""
    m = len(patch)
    if m == 0:
        return []

    # Build patch adjacency.
    edge2tris = defaultdict(list)
    for ti, tri in enumerate(patch):
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u == v:                       # Skip degenerate edges
                continue
            edge2tris[frozenset((u, v))].append(ti)
    adj = defaultdict(list)
    for key, tl in edge2tris.items():
        if len(key) != 2:
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

    # Seed faces touching directed boundary edges.
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

    # Propagate opposite directions across shared edges.
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

    # Orient unconstrained components by the reference normal.
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
# New-face orientation
# --------------------------------------------------------------------------- #
def orient_new_faces_by_neighbors(mesh, n_orig, verbose=True):
    """Orient added faces while preserving original face winding."""
    canon = canonical_index_map(mesh)
    faces = np.asarray(mesh.faces)
    cf = canon[faces]
    F = len(faces)

    # Map welded edges to directed face edges.
    edge2 = defaultdict(list)
    for fi in range(F):
        a, b, c = int(cf[fi, 0]), int(cf[fi, 1]), int(cf[fi, 2])
        for u, v in ((a, b), (b, c), (c, a)):
            if u != v:
                edge2[(u, v) if u < v else (v, u)].append((fi, (u, v)))
    # Build adjacency from manifold edges.
    adj = defaultdict(list)
    for key, lst in edge2.items():
        if len(lst) == 2:
            (fi, di), (fj, dj) = lst
            adj[fi].append((fj, di, dj))
            adj[fj].append((fi, dj, di))

    flip = np.zeros(F, dtype=bool)
    vis = np.zeros(F, dtype=bool)

    # Propagate from fixed original faces.
    dq = deque()
    for fi in range(n_orig):
        vis[fi] = True
        dq.append(fi)
    while dq:
        fi = dq.popleft()
        for (fj, di, dj) in adj[fi]:
            if vis[fj] or fj < n_orig:
                continue
            flip[fj] = (not flip[fi]) if di == dj else flip[fi]
            vis[fj] = True
            dq.append(fj)

    # Orient isolated added patches consistently.
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

    # Flip only added faces.
    flip[:n_orig] = False
    nf = faces.copy()
    nf[flip] = nf[flip][:, ::-1]
    mesh.faces = nf

    if verbose:
        n_new = F - n_orig
        # Count residual conflicts touching added faces.
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
        print(f"  [orient] flipped {int(flip[n_orig:].sum())}/{n_new} added faces "
              f"({n_orig} original faces fixed) | residual conflicts {conflict_new}")
    return mesh


# --------------------------------------------------------------------------- #
# Global orientation
# --------------------------------------------------------------------------- #
def _build_face_adjacency(mesh):
    """Build face adjacency from welded manifold edges."""
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
    """Count winding conflicts across welded manifold edges."""
    _, _, edge2 = _build_face_adjacency(mesh)
    mani = confl = 0
    for lst in edge2.values():
        if len(lst) == 2:
            mani += 1
            if lst[0][1] == lst[1][1]:
                confl += 1
    return confl, mani


def _visibility_vote(V, F, n_dirs=32, res=128, samples=3, seed=0):
    """Estimate per-face outwardness by multi-view visibility voting."""
    nf = len(F)
    if nf == 0:
        return np.zeros(0)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    ln = np.linalg.norm(fn, axis=1)
    ok = ln > 1e-20
    fnu = np.zeros_like(fn)
    fnu[ok] = fn[ok] / ln[ok, None]

    # Allocate samples by face area.
    area = 0.5 * ln
    budget = max(1, int(samples)) * nf
    tot_area = float(area.sum())
    if tot_area <= 0:
        return np.zeros(nf)
    cnt = np.maximum(1, np.round(area / tot_area * budget).astype(np.int64))
    cnt = np.minimum(cnt, 64)                       # Bound memory use
    pf = np.repeat(np.arange(nf), cnt)              # Sample-to-face map
    # Sample triangle interiors uniformly.
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

    # Sample view directions with a Fibonacci sphere.
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
    # Suppress spurious BLAS floating-point warnings.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for d in dirs:
            # View-plane basis
            tmp = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
            u = np.cross(d, tmp)
            u /= max(np.linalg.norm(u), 1e-20)
            w = np.cross(d, u)
            # Pixel coordinates and depth
            pu = (P - lo) @ u
            pw = (P - lo) @ w
            depth = (P - lo) @ d
            gi = np.clip(((pu - pu.min()) / ext * (res - 1)).astype(np.int64), 0, res - 1)
            gj = np.clip(((pw - pw.min()) / ext * (res - 1)).astype(np.int64), 0, res - 1)
            cell = gi * res + gj
            # Keep the nearest sample per pixel.
            order = np.lexsort((depth, cell))
            cs = cell[order]
            last = np.ones(len(cs), dtype=bool)
            last[:-1] = cs[:-1] != cs[1:]          # Maximum-depth sample
            win = order[last]
            wf = pf[win]
            s = fnu[wf] @ d                        # Positive faces the camera
            np.add.at(votes, wf, np.sign(s))
    return votes


def visible_back_ratio(V, F, n_dirs=32, res=128, seed=0):
    """Estimate the visible back-facing sample ratio."""
    v = _visibility_vote(V, F, n_dirs=n_dirs, res=res, seed=seed)
    pos = float(np.clip(v, 0, None).sum())
    neg = float(np.clip(-v, 0, None).sum())
    tot = pos + neg
    return (neg / tot) if tot > 0 else 0.0


def global_orient_fix(mesh, verbose=True, seed=0, n_dirs=32, res=128,
                      vis_conf=0.25, guard_tol=0.02):
    """Orient all components outward without changing topology."""
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

    # Propagate consistent winding within each component.
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
                # Shared edge directions must be opposite.
                flip[fj] = (not flip[fi]) if di == dj else flip[fi]
                comp[fj] = n_comp
                dq.append(fj)
        n_comp += 1

    faces_c = faces.copy()
    faces_c[flip] = faces_c[flip][:, ::-1]

    # Choose each component's outward sign.
    vote = _visibility_vote(V, faces_c, n_dirs=n_dirs, res=res, seed=seed)
    comp_vote = np.zeros(n_comp)                  # Signed visibility vote
    comp_conf = np.zeros(n_comp)                  # Vote confidence
    np.add.at(comp_vote, comp, vote)
    np.add.at(comp_conf, comp, np.abs(vote))
    # Prefer the orientation closest to the input.
    comp_size = np.bincount(comp, minlength=n_comp).astype(np.float64)
    comp_dev = np.zeros(n_comp)
    np.add.at(comp_dev, comp, flip.astype(np.float64))
    # Signed-volume fallback
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
            need_flip[ci] = vis < 0            # Use confident visibility
            n_by_vis += 1
            continue
        # Fall back to the input orientation.
        if comp_dev[ci] * 2 != comp_size[ci]:
            need_flip[ci] = comp_dev[ci] * 2 > comp_size[ci]
            n_by_prior += 1
            continue
        # Use signed volume for tied near-closed components.
        ratio = comp_bnd[ci] / max(comp_edge[ci], 1)
        if ratio < 0.05 and comp_vol[ci] < 0:
            need_flip[ci] = True
    if np.any(need_flip):
        sel = need_flip[comp]
        faces_c[sel] = faces_c[sel][:, ::-1]
        flip[sel] = ~flip[sel]

    # Revert if visible back faces increase significantly.
    back_before = visible_back_ratio(V, faces, n_dirs=n_dirs, res=res, seed=seed)
    back_after = visible_back_ratio(V, faces_c, n_dirs=n_dirs, res=res, seed=seed)
    reverted = False
    if guard_tol is not None and back_after > back_before + guard_tol:
        reverted = True
        faces_c = faces                          # Restore input winding
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
            print(f"  [global orient] reverted: back ratio {back_before * 100:.1f}% -> "
                  f"{back_after * 100:.1f}%")
        else:
            print(f"  [global orient] components={n_comp} flipped={stats['n_comp_flip']} "
                  f"(visibility={n_by_vis}, prior={n_by_prior}) | "
                  f"faces={stats['n_flip']}/{F} | conflicts={confl_before}->{confl_after} "
                  f"(manifold={mani}) | back ratio={back_before * 100:.1f}%->"
                  f"{back_after * 100:.1f}%")
    return mesh, stats


# --------------------------------------------------------------------------- #
# Z-fighting metric
# --------------------------------------------------------------------------- #
def zfight_ratio(V, F, n_dirs=16, res=256, samples=6, eps_ratio=2e-3, seed=0):
    """Estimate the percentage of pixels with near-overlapping surfaces."""
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
            order = np.lexsort((-dep, cell))          # Sort nearest first per cell
            cs = cell[order]
            ds = dep[order]
            fs = pf[order]
            firstm = np.ones(len(cs), dtype=bool)
            firstm[1:] = cs[1:] != cs[:-1]
            idx1 = np.nonzero(firstm)[0]              # Nearest sample per cell
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
# Liepa hole filling
# --------------------------------------------------------------------------- #
def fill_liepa(mesh, do_refine=False, fix_orientation=True, verbose=True):
    """Fill detected holes without moving original vertices."""
    mesh = mesh.copy()
    canon = canonical_index_map(mesh)
    emap = build_edge_face_map(mesh, canon=canon)
    loops = find_boundary_loops(mesh, canon=canon)
    if verbose:
        print(f"  [liepa] triangulating {len(loops)} holes...")

    verts = mesh.vertices.copy()
    faces = mesh.faces.tolist()
    n_orig_faces = len(faces)          # Original face count
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
                # Restore the best quadrilateral diagonal.
                tris = best_quad_diagonal(pts, edge_normals, ref_normal)
            else:
                tris = min_weight_triangulation(pts, edge_normals, ref_normal)
        except Exception as e:
            if verbose:
                print(f"    skipped hole (n={n}): {e}")
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

        # Map local patch indices to global vertices.
        base = len(new_verts)
        idx_map = {}
        for k in range(n):
            idx_map[k] = loop[k]
        for k in range(n, len(rp)):
            new_verts.append(rp[k])
            idx_map[k] = base + (k - n)

        # Assemble and orient the patch.
        patch = [[idx_map[int(a)], idx_map[int(b)], idx_map[int(c)]] for (a, b, c) in rf]
        boundary_dir = {(loop[k], loop[(k + 1) % n]) for k in range(n)}
        oriented = orient_patch_faces(patch, boundary_dir, new_verts, ref_normal)
        faces.extend(oriented)
        filled += 1

    if verbose:
        print(f"  [liepa] filled {filled}/{len(loops)} holes, "
              f"added {len(new_verts) - len(verts)} vertices")
    out = trimesh.Trimesh(vertices=np.array(new_verts),
                          faces=np.array(faces), process=False)
    if fix_orientation:
        orient_new_faces_by_neighbors(out, n_orig_faces, verbose=verbose)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def count_holes(mesh):
    """Count closable boundary loops."""
    return len(find_boundary_loops(mesh))


def count_boundary_edges(mesh):
    """Count welded edges referenced by one face."""
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
# Boundary crack stitching
# --------------------------------------------------------------------------- #
def zipper_boundary_cracks(mesh, tol=None, tol_ratio=0.005, max_iter=12,
                           verbose=True):
    """Bridge narrow boundary cracks using existing vertices."""
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
        # Count welded edges and existing faces.
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
        # Index welded boundary vertices.
        bverts = sorted({x for e in boundary for x in e})
        P = verts[bverts]
        pos_of = {vid: verts[vid] for vid in bverts}
        if cKDTree is not None:
            tree = cKDTree(P)
        # Select the best opposite vertex per edge.
        candidates = []            # (score, a, b, c)
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
                # Prefer the shortest distance to edge ab.
                if ab2 > 1e-20:
                    tproj = max(0.0, min(1.0, float((Pc - Pa) @ ab) / ab2))
                else:
                    tproj = 0.0
                foot = Pa + tproj * ab
                d = float(np.linalg.norm(Pc - foot))
                if d >= best_d:
                    continue
                if tri_area(Pa, Pb, Pc) < 1e-14:     # Degenerate triangle
                    continue
                best = c
                best_d = d
            if best is not None:
                candidates.append((best_d, a, b, best))
        if not candidates:
            break
        # Add narrow bridges first.
        candidates.sort(key=lambda x: x[0])
        new_faces = []
        for _d, a, b, c in candidates:
            e_ab = (a, b) if a < b else (b, a)
            if cnt.get(e_ab, 0) != 1:               # Already stitched
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
        print(f"  [zipper] added {total_added} bridge faces (tol={tol:.4f}, {rounds} rounds)")
    return total_added


# --------------------------------------------------------------------------- #
# Advancing-front seam stitching
# --------------------------------------------------------------------------- #
def zipper_seams(mesh, tol=None, tol_ratio=0.008, max_iter=8, verbose=True):
    """Stitch paired boundary chains with triangle strips."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return 0
    verts = np.asarray(mesh.vertices)
    diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
    if tol is None:
        tol = diag * tol_ratio
    rung_max = 2.5 * tol                      # Maximum bridge length

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

        # Split boundaries into chains at junctions.
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
        for (u, w) in boundary:              # Remaining closed loops
            e = (u, w) if u < w else (w, u)
            if e not in visited_edges:
                chains.append(walk_chain(u, w))
        chains = [c for c in chains if len(c) >= 2]
        if len(chains) < 2:
            break

        # Match nearby chains with a KD-tree.
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
            # Align corresponding chain endpoints.
            if dist(A[0], B[0]) > dist(A[0], B[-1]):
                B = B[::-1]
            # Advance along both chains.
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
        print(f"  [seam] added {total_added} faces (tol={tol:.4f}, {rounds} rounds)")
    return total_added


# --------------------------------------------------------------------------- #
# Empty polygon filling
# --------------------------------------------------------------------------- #
def _newell_normal(pts):
    """Estimate a polygon normal with Newell's method."""
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
    """Fill small chordless polygonal holes."""
    verts = np.asarray(mesh.vertices)
    total = 0
    n_holes = 0
    rounds = 0

    def pick_next(prev, cur, cands):
        """Choose the straightest continuation at a junction."""
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
        C = defaultdict(int)              # Directed half-edge counts
        faceset = set()
        for t in cf:
            a, b, c = int(t[0]), int(t[1]), int(t[2])
            if len({a, b, c}) == 3:
                faceset.add(frozenset((a, b, c)))
            for u, w in ((a, b), (b, c), (c, a)):
                if u != w:
                    C[(u, w)] += 1
        # Build unmatched half-edge adjacency.
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
                loop = path        # Closed path excludes repeated start
                ok = closed and 3 <= len(loop) <= max_n and len(set(loop)) == len(loop)
                if ok:
                    # Reverse the loop to match surrounding winding.
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
                    # Exclude duplicate triangles.
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
                    # Restore consumed edges after a failed walk.
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
        print(f"  [polyfill] added {total} faces "
              f"(~{n_holes} holes, n<={max_n}, {rounds} rounds)")
    return total


# --------------------------------------------------------------------------- #
# Self-intersection removal
# --------------------------------------------------------------------------- #
def _seg_tri_cross(P0, P1, V0, V1, V2, eps=1e-9):
    """Test segment-triangle crossings with Möller-Trumbore."""
    d = P1 - P0
    e1 = V1 - V0
    e2 = V2 - V0
    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)          # Determinant
    parallel = np.abs(a) < eps                # Ignore parallel or coplanar pairs
    a_safe = np.where(parallel, 1.0, a)
    f = 1.0 / a_safe
    s = P0 - V0
    u = f * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = f * np.einsum('ij,ij->i', d, q)
    t = f * np.einsum('ij,ij->i', e2, q)
    m = eps                                   # Exclude edge and vertex contact
    ok = (~parallel) & (u > m) & (v > m) & (u + v < 1.0 - m) & (t > m) & (t < 1.0 - m)
    return ok


def remove_self_intersections(mesh, verbose=True, seed=0):
    """Remove one face from each detected intersecting pair."""
    faces = np.asarray(mesh.faces)
    F = len(faces)
    if F == 0:
        return 0
    try:
        tree = mesh.triangles_tree
    except Exception:
        if verbose:
            print("  [selfx] skipped: rtree unavailable")
        return 0
    tris = np.asarray(mesh.triangles)         # (F,3,3)
    canon = canonical_index_map(mesh)
    cf = canon[faces]
    tri_bounds = np.hstack([tris.min(axis=1), tris.max(axis=1)])  # (F,6)

    # Find AABB-overlapping candidates.
    cand = []
    cf_sets = [set(map(int, cf[i])) for i in range(F)]
    for i in range(F):
        for j in tree.intersection(tri_bounds[i]):
            j = int(j)
            if j <= i:
                continue
            if cf_sets[i] & cf_sets[j]:        # Skip adjacent welded faces
                continue
            cand.append((i, j))
    if not cand:
        if verbose:
            print("  [selfx] no intersections found")
        return 0
    cand = np.asarray(cand, dtype=np.int64)
    ai, bi = cand[:, 0], cand[:, 1]

    # Test edges of each triangle against the other.
    A = tris[ai]                              # (M,3,3)
    B = tris[bi]
    hit = np.zeros(len(cand), dtype=bool)
    for (p, q) in ((0, 1), (1, 2), (2, 0)):   # A edges against B
        hit |= _seg_tri_cross(A[:, p], A[:, q], B[:, 0], B[:, 1], B[:, 2])
    for (p, q) in ((0, 1), (1, 2), (2, 0)):   # B edges against A
        hit |= _seg_tri_cross(B[:, p], B[:, q], A[:, 0], A[:, 1], A[:, 2])
    inter = cand[hit]
    if len(inter) == 0:
        if verbose:
            print("  [selfx] no intersections found")
        return 0

    # Remove one face from each pair.
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
        print(f"  [selfx] pairs={len(inter)} removed={len(removed)} remaining={len(keep)}")
    return len(removed)


# --------------------------------------------------------------------------- #
# Duplicate-face removal
# --------------------------------------------------------------------------- #
def remove_duplicate_faces(mesh, verbose=True):
    """Remove duplicate and degenerate faces."""
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
        if len(key) < 3:                     # Degenerate face
            n_degen += 1
            continue
        if key in seen:                      # Duplicate face
            n_dup += 1
            continue
        seen.add(key)
        keep.append(i)
    mesh.faces = faces[keep]
    if verbose:
        print(f"  [dedup] duplicates={n_dup} degenerate={n_degen} remaining={len(keep)}")
    return n_dup + n_degen


# --------------------------------------------------------------------------- #
# Dangling-face removal
# --------------------------------------------------------------------------- #
def remove_dangling_faces(mesh, verbose=True, max_iter=1000):
    """Remove faces attached only through one non-manifold edge."""
    faces = np.asarray(mesh.faces)
    canon = canonical_index_map(mesh)          # Vertices stay fixed
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
            if len(counts) != 3:               # Skip degenerate faces
                continue
            n_free = sum(1 for e in counts if e == 1)      # Boundary edges
            n_nonmanifold = sum(1 for e in counts if e > 2)  # Non-manifold edges
            if n_free == 2 and n_nonmanifold == 1:
                remove_mask[fi] = True

        if not remove_mask.any():
            break
        faces = faces[~remove_mask]
        total_removed += int(remove_mask.sum())
        rounds += 1

    mesh.faces = faces
    if verbose:
        print(f"  [dangling] removed={total_removed} rounds={rounds} remaining={len(faces)}")
    return total_removed


# --------------------------------------------------------------------------- #
# OBJ export
# --------------------------------------------------------------------------- #
def export_obj_keep_all_vertices(mesh, path, write_normals=True):
    """Write OBJ data without dropping unreferenced vertices."""
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
            a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1  # OBJ is 1-based
            if write_normals:
                fp.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
            else:
                fp.write(f"f {a} {b} {c}\n")


# --------------------------------------------------------------------------- #
# In-memory pipeline
# --------------------------------------------------------------------------- #
def run_fill_pipeline(mesh, refine=False, remove_dangling=False, dedup=True,
                      zipper=True, zipper_tol=None, quadfill=True, poly_max_n=6,
                      remove_intersect=False, orient_new=True, verbose=True,
                      orient_anchor=None):
    """Run the complete in-memory hole-filling pipeline."""
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
    holes_before = count_holes(mesh)         # Closable holes after preprocessing

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
        if added == 0 and q == 0 and s == 0 and z == 0 and r == 0:  # Converged
            break

    # Evaluate before winding changes.
    holes_after = count_holes(result)
    be_after = count_boundary_edges(result)
    add_faces = len(result.faces) - orig_faces + total_removed

    # Align added faces to fixed anchor faces.
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
# Meshflow graph filling
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
    """Triangulate chordless polygonal cycles."""
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
        for i in range(1, len(cyc) - 1):           # Fan triangulation from a
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
            if any(w in nbr[p] for p in interior):  # Skip chords
                continue
            if w in nbr[a]:                          # Close the cycle
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
    """Extract unique non-degenerate undirected mesh edges."""
    f = np.asarray(mesh.faces)
    if len(f) == 0:
        return np.empty((0, 2), dtype=np.int64)
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    e = np.unique(e, axis=0)
    return e.astype(np.int64)


# --------------------------------------------------------------------------- #
# Combined hole filling
# --------------------------------------------------------------------------- #
def _surviving_orig_count(mesh):
    """Count original faces that survive welded deduplication."""
    canon = canonical_index_map(mesh)
    keyset = set()
    for t in np.asarray(mesh.faces):
        ct = canon[t]
        fk = frozenset((int(ct[0]), int(ct[1]), int(ct[2])))
        if len(fk) == 3:                      # Exclude degenerate faces
            keyset.add(fk)
    return len(keyset)


def meshflow_append(mesh, fill_quad_rings=True, ring_max_n=6, bnd_verts=2,
                    verbose=False):
    """Append meshflow faces that touch enough boundary vertices."""
    orig_faces = np.asarray(mesh.faces)
    N = int(len(mesh.vertices))
    edges = mesh_edges(mesh)
    gen = edges_to_faces(edges, N, fill_quad_rings, ring_max_n=ring_max_n)
    seen = set(tuple(sorted(int(x) for x in t)) for t in orig_faces)

    canon = None
    bnd_v = None
    if bnd_verts and bnd_verts > 0:
        # Collect original welded boundary vertices.
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
                continue                          # Discard interior candidates
        new.append([int(t[0]), int(t[1]), int(t[2])])
    if new:
        faces = np.vstack([orig_faces, np.array(new, dtype=orig_faces.dtype)])
    else:
        faces = orig_faces
    out = trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                          faces=faces, process=False)
    if verbose and n_drop:
        print(f"  [meshflow] discarded {n_drop} interior candidates "
              f"(bnd_verts={bnd_verts})")
    return out, len(faces) - len(orig_faces)


def process(path, out_dir, fill_quad_rings=True, write_normals=True,
            poly_max_n=6, zipper_tol=None, ring_max_n=4,
            global_orient=True, orient_only=False, suffix="orient",
            mf_bnd_verts=2):
    print(f"\n>>> Processing {path}")
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    n_vert = len(mesh.vertices)
    orig_faces_in = len(mesh.faces)
    n_anchor = _surviving_orig_count(mesh)      # Surviving original anchors
    holes0 = count_holes(mesh)
    be0 = count_boundary_edges(mesh)
    cf0, mani0 = count_orientation_conflicts(mesh)
    zf0 = zfight_ratio(np.asarray(mesh.vertices), np.asarray(mesh.faces))
    print(f"    Input: vertices={n_vert} faces={orig_faces_in} "
          f"holes={holes0} | boundary edges={be0} | "
          f"orientation conflicts={cf0}/{mani0} ({cf0 / max(mani0, 1) * 100:.1f}%) | "
          f"z-fighting={zf0:.1f}%")
    stem = os.path.splitext(os.path.basename(path))[0]

    t0 = time.perf_counter()
    if orient_only:
        # Orient normals without filling holes.
        result = mesh
        st = {"iters": 0, "n_dedup": 0, "total_quad": 0, "total_seam": 0,
              "total_zip": 0, "n_flip": 0, "add_faces": 0,
              "holes_before": holes0, "holes_after": holes0, "be_after": be0}
        mf_added = 0
    else:
        # Stage 1: meshflow filling
        _ts = time.perf_counter()
        mf_mesh, mf_added = meshflow_append(mesh, fill_quad_rings=fill_quad_rings,
                                            ring_max_n=ring_max_n,
                                            bnd_verts=mf_bnd_verts, verbose=True)
        t_mf = time.perf_counter() - _ts
        holes1 = count_holes(mf_mesh)
        be1 = count_boundary_edges(mf_mesh)
        print(f"  [stage 1 meshflow] max_n={ring_max_n} added={mf_added} | holes={holes0}->{holes1} | "
              f"boundary edges={be0}->{be1} | time={t_mf:.3f}s")

        # Stage 2: full Liepa pipeline
        _ts = time.perf_counter()
        result, st = run_fill_pipeline(
            mf_mesh, dedup=True, zipper=True, zipper_tol=zipper_tol,
            quadfill=True, poly_max_n=poly_max_n, remove_intersect=False,
            orient_new=True, verbose=True, orient_anchor=n_anchor)
        t_lp = time.perf_counter() - _ts
        print(f"  [stage 2 fill_liepa] rounds={st['iters']} dedup={st['n_dedup']} | "
              f"polygon faces={st['total_quad']} seam faces={st['total_seam']} | "
              f"bridge faces={st['total_zip']} aligned faces={st['n_flip']} | "
              f"holes={st['holes_before']}->{st['holes_after']} | "
              f"boundary edges={be1}->{st['be_after']} | time={t_lp:.3f}s")

    # Stage 3: global face orientation
    gst = None
    if global_orient:
        _ts = time.perf_counter()
        V_chk = np.asarray(result.vertices).copy()
        nf_chk = len(result.faces)
        result, gst = global_orient_fix(result, verbose=True)
        t_go = time.perf_counter() - _ts
        # Verify topology invariants.
        assert np.array_equal(V_chk, np.asarray(result.vertices)), "Global orientation changed vertices"
        assert nf_chk == len(result.faces), "Global orientation changed face count"
        print(f"  [stage 3 global orient] time={t_go:.3f}s")

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
    print(f"     Saved {out_path}"
          + (" with vertex normals" if write_normals else ""))
    # Final faces = input + meshflow - deduplication + Liepa.
    print(f"       time={elapsed:.3f}s | added faces={total_add} "
          f"(meshflow +{mf_added}, dedup -{st['n_dedup']}, fill_liepa +{st['add_faces']}) | "
          f"holes={holes0}->{st['holes_after']} (filled={filled}, rate={fill_rate:.1f}%) | "
          f"boundary edges={be0}->{st['be_after']}")
    cf1, mani1 = count_orientation_conflicts(result)
    zf1 = zfight_ratio(np.asarray(result.vertices), np.asarray(result.faces))
    print(f"       faces={len(result.faces)} | vertices={len(result.vertices)} "
          f"(delta={dv:+d}) | orientation conflicts={cf0}/{mani0} -> {cf1}/{mani1} "
          f"({cf0 / max(mani0, 1) * 100:.1f}% -> {cf1 / max(mani1, 1) * 100:.1f}%) | "
          f"z-fighting={zf0:.1f}% -> {zf1:.1f}%")


def _batch_worker(job):
    """Run one job while capturing its output."""
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
    """Process independent files in parallel with ordered output."""
    n = len(inputs)
    if jobs is None or jobs <= 0:
        jobs = min(os.cpu_count() or 1, n)
    jobs = max(1, min(jobs, n))
    job_list = [(process_fn, p, out_dir, dict(kwargs)) for p in inputs]
    if jobs == 1 or n == 1:
        for job in job_list:
            print(_batch_worker(job), end="")
        return
    print(f"[parallel] processing {n} files with {jobs} workers")
    from multiprocessing import Pool
    with Pool(processes=jobs) as pool:
        for out in pool.imap(_batch_worker, job_list):
            print(out, end="")


def main():
    ap = argparse.ArgumentParser(
        description="Fill mesh holes and normalize face orientation.")
    ap.add_argument("inputs", nargs="+", help="Input OBJ files")
    ap.add_argument("--out", default="orient_output",
                    help="Output directory")
    ap.add_argument("--suffix", default="orient",
                    help="Output filename suffix")
    ap.add_argument("--no-normals", dest="normals", action="store_false",
                    help="Do not write vertex normals")
    ap.add_argument("--no-quad-rings", dest="quad_rings", action="store_false",
                    help="Disable meshflow polygon-ring filling")
    ap.add_argument("--ring-max-n", dest="ring_max_n", type=int, default=4,
                    help="Maximum meshflow polygon size")
    ap.add_argument("--poly-max-n", dest="poly_max_n", type=int, default=6,
                    help="Maximum Liepa polygon size")
    ap.add_argument("--zipper-tol", dest="zipper_tol", type=float, default=None,
                    help="Absolute stitching distance")
    ap.add_argument("--mf-bnd-verts", dest="mf_bnd_verts", type=int, default=2,
                    help="Minimum original boundary vertices per meshflow face")
    ap.add_argument("--no-global-orient", dest="global_orient", action="store_false",
                    help="Disable global orientation")
    ap.add_argument("--orient-only", dest="orient_only", action="store_true",
                    help="Orient faces without filling holes")
    ap.add_argument("-j", "--jobs", dest="jobs", type=int, default=0,
                    help="Worker count; use 0 for automatic selection")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"Output directory: {os.path.abspath(args.out)}")

    t_all = time.perf_counter()
    kwargs = dict(fill_quad_rings=args.quad_rings, write_normals=args.normals,
                  poly_max_n=args.poly_max_n, zipper_tol=args.zipper_tol,
                  ring_max_n=args.ring_max_n, global_orient=args.global_orient,
                  orient_only=args.orient_only, suffix=args.suffix,
                  mf_bnd_verts=args.mf_bnd_verts)
    run_batch(process, args.inputs, args.out, kwargs, jobs=args.jobs)
    print(f"\nTotal time: {time.perf_counter() - t_all:.3f}s")


if __name__ == "__main__":
    main()
