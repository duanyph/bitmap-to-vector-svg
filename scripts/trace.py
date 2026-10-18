# -*- coding: utf-8 -*-
"""
轮廓追踪层：彩色位图 -> 真矢量 SVG（多笔画 / 闭合轮廓 / 颜色分层）
=================================================================

管线：
1. 颜色量化：PIL FASTOCTREE 降到 K 色
2. 分层：每色的"归属场" = 到其他最近色的距离 - 到本色的距离，
   零水平集即该色区域的亚像素边界（抗锯齿边缘自动落在正确位置）
3. marching squares 提取有向闭合轮廓（方向一致，孔洞自动反向）
4. RDP 折线简化（减小 DP 规模）
5. 复用 fit_curve 的最优分段（DP 最少基元 + 基元分级 + G1）
6. 同色所有轮廓合并为一个 path，fill-rule="evenodd" 自动处理孔洞
"""
import numpy as np
from PIL import Image

from fit_curve import (detect_corners, segment_minimal, enforce_g1,
                       pieces_to_path, sample_pieces)

# ---------------------------------------------------------------------------
# Marching squares（带向查表，前景恒在左侧）
# ---------------------------------------------------------------------------
# 角点编号: 0=(y,x) 1=(y,x+1) 2=(y+1,x+1) 3=(y+1,x)
# 边: T=上 R=右 B=下 L=左
_MS = {
    1:  [("L", "T")], 2:  [("T", "R")], 3:  [("L", "R")], 4:  [("R", "B")],
    5:  [("L", "T"), ("R", "B")], 6: [("T", "B")], 7: [("L", "B")],
    8:  [("B", "L")], 9:  [("B", "T")], 10: [("T", "R"), ("B", "L")],
    11: [("B", "R")], 12: [("R", "L")], 13: [("R", "T")], 14: [("T", "L")],
}


def marching_squares(field, level=0.0):
    """field: (h,w) 标量场。返回有向线段列表 [(p0, p1), ...]，前景在左。"""
    v = field > level
    h, w = v.shape
    idx = (v[:-1, :-1].astype(np.int8)
           | (v[:-1, 1:].astype(np.int8) << 1)
           | (v[1:, 1:].astype(np.int8) << 2)
           | (v[1:, :-1].astype(np.int8) << 3))
    ys, xs = np.nonzero((idx > 0) & (idx < 15))
    segs = []
    f = field
    for y, x in zip(ys, xs):
        v00, v10 = f[y, x], f[y, x + 1]
        v11, v01 = f[y + 1, x + 1], f[y + 1, x]
        def ip(pa, va, pb, vb):
            t = (level - va) / (vb - va) if abs(vb - va) > 1e-12 else 0.5
            return (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))
        T = ip((x, y), v00, (x + 1, y), v10)
        R = ip((x + 1, y), v10, (x + 1, y + 1), v11)
        B = ip((x, y + 1), v01, (x + 1, y + 1), v11)
        L = ip((x, y), v00, (x, y + 1), v01)
        pts = {"T": T, "R": R, "B": B, "L": L}
        for a, b in _MS[int(idx[y, x])]:
            segs.append((pts[a], pts[b]))
    return segs


def _key(p):
    return (round(p[0], 6), round(p[1], 6))


def assemble_contours(segs, min_points=4):
    """把有向线段按首尾相接组装成闭合轮廓列表。"""
    outgoing = {}
    for si, (a, b) in enumerate(segs):
        outgoing.setdefault(_key(a), []).append(si)
    used = [False] * len(segs)
    contours = []
    for si0 in range(len(segs)):
        if used[si0]:
            continue
        used[si0] = True
        ring = [segs[si0][0], segs[si0][1]]
        cur = segs[si0][1]
        while True:
            nxt = None
            for sj in outgoing.get(_key(cur), ()):
                if not used[sj]:
                    nxt = sj
                    break
            if nxt is None:
                break
            used[nxt] = True
            cur = segs[nxt][1]
            ring.append(cur)
        if len(ring) >= min_points:
            contours.append(np.array(ring[:-1], dtype=float))  # 去重复终点
    return contours


# ---------------------------------------------------------------------------
# RDP 折线简化（支持强制保留拐点，返回保留的索引）
# ---------------------------------------------------------------------------

def _rdp_open_indices(pts, eps, offset=0):
    """开放折线 RDP，返回全局索引集合（offset 为起始全局索引）"""
    n = len(pts)
    if n < 3:
        return {offset + i for i in range(n)}
    keep = {0, n - 1}
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        d = b - a
        L = np.hypot(*d)
        seg = pts[i + 1:j]
        if L < 1e-12:
            dist = np.linalg.norm(seg - a, axis=1)
        else:
            w = seg - a
            dist = np.abs(w[:, 0] * d[1] - w[:, 1] * d[0]) / L
        k = int(np.argmax(dist))
        if dist[k] > eps:
            m = i + 1 + k
            keep.add(m)
            stack.append((i, m))
            stack.append((m, j))
    return {offset + i for i in keep}


def rdp_closed_indices(pts, eps, force=None):
    """闭合折线 RDP：直径两端切开分别简化。force 中的索引强制保留。
    返回排序后的保留索引列表。"""
    n = len(pts)
    force = set(force or [])
    if n < 4:
        return sorted(set(range(n)) | force)
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)
    i0, j0 = np.unravel_index(np.argmax(d2), d2.shape)
    if j0 < i0:
        i0, j0 = j0, i0
    keep = _rdp_open_indices(pts[i0:j0 + 1], eps, offset=i0)
    keep |= {(j0 + i) % n for i in _rdp_open_indices(
        np.concatenate([pts[j0:], pts[:i0 + 1]]), eps, offset=0)}
    keep |= force
    return sorted(keep)


def rdp_closed(pts, eps):
    """兼容接口：直接返回简化后的点列"""
    return pts[rdp_closed_indices(pts, eps)]


def rdp_open(pts, eps):
    """开放折线 RDP，直接返回简化后的点列"""
    keep = sorted(_rdp_open_indices(pts, eps))
    return pts[keep]


# ---------------------------------------------------------------------------
# 颜色分层
# ---------------------------------------------------------------------------

def quantize_layers(rgb_img, n_colors=8, min_ratio=0.003):
    """量化到 K 色，再把面积过小的量化色（抗锯齿产生的混合色）
    合并到 RGB 最近的大色，返回 (idx(h,w), palette)。
    不合并的话，混合色像素恰好落在归属场零水平集上，会把轮廓撕碎。"""
    q = rgb_img.convert("RGB").quantize(colors=n_colors,
                                        method=Image.FASTOCTREE)
    pal = q.getpalette()
    idx0 = np.asarray(q, dtype=np.int32)
    palette0 = [tuple(pal[i * 3:i * 3 + 3]) for i in range(n_colors)]
    counts = np.bincount(idx0.ravel(), minlength=n_colors)
    total = idx0.size
    big = [c for c in range(n_colors) if counts[c] >= max(20, total * min_ratio)]
    if not big:
        big = [int(np.argmax(counts))]
    P0 = np.asarray(palette0, dtype=float)
    remap = {}
    for c in range(n_colors):
        if c in big:
            remap[c] = big.index(c)
        else:
            d = np.linalg.norm(P0[big] - P0[c], axis=1)
            remap[c] = int(np.argmin(d))
    lut = np.asarray([remap[c] for c in range(n_colors)], dtype=np.int32)
    idx = lut[idx0]
    palette = [palette0[c] for c in big]
    return idx, palette


def layer_field(rgb_arr, palette, c):
    """色 c 的归属场：>0 表示该像素归色 c。零水平集 = 区域边界。"""
    P = np.asarray(rgb_arr, dtype=float)
    c_arr = np.asarray(palette[c], dtype=float)
    d_c = np.sqrt(((P - c_arr) ** 2).sum(-1))
    d_other = np.full(d_c.shape, np.inf)
    for c2 in range(len(palette)):
        if c2 == c:
            continue
        o = np.asarray(palette[c2], dtype=float)
        d_other = np.minimum(d_other, np.sqrt(((P - o) ** 2).sum(-1)))
    return d_other - d_c  # 场值：内部为正，边界为 0


# ---------------------------------------------------------------------------
# 闭合轮廓拟合（复用 fit_curve 的 DP 最少基元）
# ---------------------------------------------------------------------------

def fit_closed_contour(pts, tol=0.6, simplify_eps=0.5, min_turn_deg=22.0):
    """闭合轮廓 -> 最少基元分段。
    拐角在稠密原始轮廓上检测（光滑曲率每点转角小、尖角转角集中），
    RDP 简化时强制保留拐点，闭合缝优先落在拐角上。"""
    n = len(pts)
    corners = detect_corners(pts, min_turn_deg=min_turn_deg, win=3,
                             closed=True)
    keep = rdp_closed_indices(pts, simplify_eps, force=corners)
    cset = set(corners)
    cpos = [k for k, gi in enumerate(keep) if gi in cset]
    start = cpos[0] if cpos else 0
    rot_idx = keep[start:] + keep[:start] + [keep[start]]
    rot = pts[rot_idx]
    m = len(keep)
    corners2 = [(p - start) % m for p in cpos]
    pieces = segment_minimal(rot, tol, corners2)
    enforce_g1(pieces)
    return pieces


def trace_layer(rgb_arr, palette, c, tol=0.6, simplify_eps=0.5, min_area=30.0):
    """追踪色 c 的全部轮廓并拟合，返回 [(contour_pts, pieces), ...]"""
    field = layer_field(rgb_arr, palette, c)
    segs = marching_squares(field, level=0.0)
    contours = assemble_contours(segs)
    out = []
    for ring in contours:
        # 过滤微小轮廓（噪点/量化杂色）
        x, y = ring[:, 0], ring[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < min_area:
            continue
        pieces = fit_closed_contour(ring, tol=tol, simplify_eps=simplify_eps)
        out.append((ring, pieces))
    return out


def layer_path_d(contours_pieces):
    """同层所有轮廓拼成一个 path（evenodd 处理孔洞）"""
    return " ".join(pieces_to_path(pieces, closed=True)
                    for _ring, pieces in contours_pieces)


def render_pieces(contours_pieces, per_piece=40):
    """渲染用：每轮廓一条离散化闭合折线"""
    out = []
    for _ring, pieces in contours_pieces:
        pts = np.concatenate(sample_pieces(pieces, per_piece))
        out.append(np.concatenate([pts, pts[:1]]))
    return out
