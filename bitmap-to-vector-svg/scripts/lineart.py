# -*- coding: utf-8 -*-
"""
线稿抽取与矢量化（只出 stroke，不管填色）
========================================
管线：
1. 去背景：从图像四边区域生长（局部色差判据，沿平滑渐变推进，
   被深色线条/明暗交界阻挡）
2. 抽线条：黑帽变换（grey_closing - gray）高亮比周围暗的细结构，
   + 阈值 + 小连通域清除。渐变背景和亮斑天然不会被检出
3. 骨架化：skimage skeletonize 把笔画区域缩成单像素中心线
4. 中心线追踪：骨架建图（8 邻域），从端点/关节/环路中追踪出
   一笔一笔的折线
5. 拟合：开放笔画 = 平滑 + RDP + DP 最少基元；
   闭合笔画（眼廓、圆球）= fit_closed_contour
6. 输出：全部笔画合并为一个 stroke path 的线稿 SVG
"""
import numpy as np
from collections import deque

from fit_curve import (detect_corners, segment_minimal, enforce_g1,
                       pieces_to_path, sample_pieces)
from trace import fit_closed_contour, rdp_open


# ---------------------------------------------------------------------------
# 1. 去背景
# ---------------------------------------------------------------------------

def remove_background(rgb, tol=35):
    """从图像四边泛洪去除背景。判据用相邻像素局部色差（而非全局阈值），
    这样能沿平滑渐变一路推进，遇到深色线条/明暗交界才停。
    返回前景 bool 掩膜。"""
    rgb = np.asarray(rgb, dtype=np.int32)
    h, w, _ = rgb.shape
    fg = np.ones((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not visited[y, x]:
                visited[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        fg[y, x] = False
        c = rgb[y, x]
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                if np.abs(rgb[ny, nx] - c).sum() <= tol:
                    dq.append((ny, nx))
    return fg


# ---------------------------------------------------------------------------
# 2. 抽线条
# ---------------------------------------------------------------------------

def extract_line_mask(gray, fg, ksize=5, thresh=28, min_area=6,
                      prefilter=0):
    """黑帽变换提取比局部背景暗的细结构（线条），
    清除面积过小的噪点。ksize 决定能检出多粗的线。
    prefilter>0 时先用该大小的中值滤波去 JPEG 噪点（保边去噪）。"""
    from scipy import ndimage
    g = np.asarray(gray, dtype=np.int32)
    if prefilter >= 3:
        g = ndimage.median_filter(g, size=prefilter)
    closed = ndimage.grey_closing(g, size=ksize)
    blackhat = closed - g
    mask = (blackhat > thresh) & fg
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:][np.asarray(sizes) >= min_area] = True
    return keep[lbl]


# ---------------------------------------------------------------------------
# 2b. 照度平坦化线稿提取（区分线条与软阴影/暗区）
# ---------------------------------------------------------------------------

def flat_line_mask(gray, fg, sigma=8.0, thresh=22.0, prefilter=3,
                   min_area=6, spur_len=6):
    """照度平坦化提取线稿并骨架化，返回修剪过的骨架。

    原理：插画里的阴影/暗区是低频信号，线条是高频信号。
    用大 sigma 高斯估计阴影底图减掉后，只剩下细的暗结构（线条）。
    比黑帽/ XDoG 都更适合本类图：
    - 黑帽按尺度选择，但会把软阴影的边界也抓成线（阈值难以分离）；
    - XDoG 只按对比度筛选，会把大块暗区（发丝阴影、暗饰品）整个抓进来；
    - 平坦化按频率分离，线条（高频）保留，阴影（低频）消除。

    流程：中值去噪(JPEG) -> 减阴影底图 -> 阈值 -> 去小斑
          -> 骨架化 -> 剪毛刺 -> 去小连通域
    """
    from scipy import ndimage
    g = np.asarray(gray, dtype=float)
    if prefilter >= 3:
        g = ndimage.median_filter(g, size=prefilter)
    bg = ndimage.gaussian_filter(g, sigma)
    flat = g - bg
    mask = (flat < -thresh) & fg
    lbl, n = ndimage.label(mask)
    if n:
        sizes = np.asarray(ndimage.sum(mask, lbl, range(1, n + 1)))
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:][sizes >= min_area] = True
        mask = keep[lbl]
    skel = skeletonize_mask(mask)
    if spur_len > 0:
        skel = prune_spurs(skel, min_len=spur_len)
    skel = drop_small_components(skel, min_size=8)
    return skel


# ---------------------------------------------------------------------------
# 3. 骨架化
# ---------------------------------------------------------------------------

def close_line_gaps(mask, size=3):
    """二值闭合，桥接细线在 JPEG 噪点下产生的 1-2px 断点。
    闭合后会稍微变粗，但骨架化会重新抽细，不影响中心线位置。"""
    from scipy import ndimage
    st = np.ones((size, size), dtype=bool)
    return np.asarray(ndimage.binary_closing(mask, structure=st), dtype=bool)


def prune_spurs(skel, min_len=6):
    """修剪骨架毛刺：迭代删除短于 min_len 的端点分支。
    JPEG 噪点会让骨架长出大量 1-3px 毛刺，不剪的话追踪出的笔画
    中位长度只有 1px，全是没有意义的碎片。"""
    skel = skel.copy()
    h, w = skel.shape

    def nbs(p):
        y, x = p
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w \
                        and skel[y + dy, x + dx]:
                    out.append((y + dy, x + dx))
        return out

    changed = True
    while changed:
        changed = False
        ys, xs = np.nonzero(skel)
        for y, x in zip(ys.tolist(), xs.tolist()):
            if not skel[y, x]:
                continue
            n1 = nbs((y, x))
            if len(n1) != 1:
                continue
            # 从端点沿分支走，量长度
            prev, cur = (y, x), n1[0]
            length = 1
            while True:
                nn = [q for q in nbs(cur) if q != prev]
                if len(nn) != 1:
                    break
                prev, cur = cur, nn[0]
                length += 1
            if length < min_len:
                prev, cur = (y, x), n1[0]
                skel[y, x] = False
                while True:
                    nn = [q for q in nbs(cur) if q != prev]
                    skel[cur[0], cur[1]] = False
                    if len(nn) != 1:
                        break
                    prev, cur = cur, nn[0]
                changed = True
    return skel


def drop_small_components(skel, min_size=8):
    """删除孤立小连通域（噪点斑）"""
    from scipy import ndimage
    lbl, n = ndimage.label(skel, structure=np.ones((3, 3)))
    if n == 0:
        return skel
    sizes = np.asarray(ndimage.sum(skel, lbl, range(1, n + 1)))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:][sizes >= min_size] = True
    return keep[lbl]


def skeletonize_mask(mask):
    from skimage.morphology import skeletonize
    return np.asarray(skeletonize(mask), dtype=bool)


# ---------------------------------------------------------------------------
# 4. 中心线追踪
# ---------------------------------------------------------------------------

def trace_skeleton(skel, min_extent=3.0):
    """把骨架追踪成一笔一笔的折线。返回 [(pts(n,2), closed), ...]"""
    ys, xs = np.nonzero(skel)
    S = set(zip(ys.tolist(), xs.tolist()))
    if not S:
        return []
    adj = {}
    for (y, x) in S:
        lst = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    q = (y + dy, x + dx)
                    if q in S:
                        lst.append(q)
        adj[(y, x)] = lst
    deg = {p: len(adj[p]) for p in S}
    used = set()

    def ekey(a, b):
        return (a, b) if a <= b else (b, a)

    def walk(a, b):
        path = [a, b]
        used.add(ekey(a, b))
        prev, cur = a, b
        while deg[cur] == 2:
            nxt = None
            for q in adj[cur]:
                if q != prev and ekey(cur, q) not in used:
                    nxt = q
                    break
            if nxt is None:
                break
            used.add(ekey(cur, nxt))
            path.append(nxt)
            prev, cur = cur, nxt
        return path

    raw = []
    # 端点出发
    for p in sorted(S):
        if deg[p] == 1:
            for q in adj[p]:
                if ekey(p, q) not in used:
                    raw.append(walk(p, q))
    # 关节出发（每条未用分支）
    for p in sorted(S):
        if deg[p] >= 3:
            for q in adj[p]:
                if ekey(p, q) not in used:
                    raw.append(walk(p, q))
    # 剩余的是环
    for p in sorted(S):
        for q in adj[p]:
            if ekey(p, q) not in used:
                raw.append(walk(p, q))

    out = []
    for st in raw:
        if len(st) < 2:
            continue
        # nonzero 给出 (row, col)，这里转成下游通用的 (x, y)
        pts = np.asarray(st, dtype=float)[:, ::-1]
        closed = len(pts) > 4 and np.linalg.norm(pts[0] - pts[-1]) < 1e-9
        if closed:
            pts = pts[:-1]
        # 过滤碎刺和极小圈
        if np.linalg.norm(pts.max(0) - pts.min(0)) < min_extent:
            continue
        out.append((pts, closed))
    return out


# ---------------------------------------------------------------------------
# 5. 笔画拟合
# ---------------------------------------------------------------------------

def _smooth_open(pts, passes=2):
    p = pts.copy()
    for _ in range(passes):
        q = p.copy()
        q[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
        p = q
    return p


def _smooth_closed(pts, passes=2):
    p = pts.copy()
    for _ in range(passes):
        p = 0.25 * np.roll(p, 1, axis=0) + 0.5 * p \
            + 0.25 * np.roll(p, -1, axis=0)
    return np.concatenate([p, p[:1]])


def fit_stroke(pts, closed, tol=0.5, rdp_eps=0.35, smooth_passes=2):
    """单笔画 -> 最少基元分段。先平滑去锯齿，再 RDP 抽稀（DP 是 O(n^2)），
    最后 DP 求全局最少基元。
    smooth_passes：平滑次数。干净二值线稿用 1 次即可（平滑会挪动中心线，
    次数越多保真度越低）；有噪点的位图用 2 次。"""
    if closed:
        pts_s = _smooth_closed(pts, passes=smooth_passes)
        return fit_closed_contour(pts_s, tol=tol, simplify_eps=rdp_eps)
    pts_s = rdp_open(_smooth_open(pts, passes=smooth_passes), rdp_eps)
    if len(pts_s) < 2:
        return []
    corners = detect_corners(pts_s)
    pieces = segment_minimal(pts_s, tol, corners)
    enforce_g1(pieces)
    return pieces


# ---------------------------------------------------------------------------
# 6. 输出 SVG
# ---------------------------------------------------------------------------

def strokes_to_svg(stroke_pieces, W, H, color="#1c2b3a", width=1.1):
    """全部笔画合并为一个 stroke path（线稿，无填充）"""
    d = " ".join(pieces_to_path(p) for p in stroke_pieces if p)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
            f'width="{W}" height="{H}">'
            f'<rect width="{W}" height="{H}" fill="#ffffff"/>'
            f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" '
            f'stroke-linejoin="round"/></svg>')


def stroke_widths(strokes, mask):
    """用距离变换估算每笔的线宽：骨架像素处 mask 距离变换值的 2 倍
    就是局部线宽，取沿笔中位数（对端点/毛刺稳健）。"""
    from scipy import ndimage
    dt = ndimage.distance_transform_edt(mask)
    h, w = mask.shape
    out = []
    for p, _c in strokes:
        idx = np.round(p).astype(int)
        idx[:, 0] = np.clip(idx[:, 0], 0, w - 1)
        idx[:, 1] = np.clip(idx[:, 1], 0, h - 1)
        out.append(float(2.0 * np.median(dt[idx[:, 1], idx[:, 0]])))
    return out


def strokes_to_svg_grouped(stroke_pieces, widths, W, H, color="#111111",
                           n_buckets=4):
    """按笔画宽度分组输出线稿 SVG：每组一个 path，stroke-width 取组内
    中位数。还原线稿的粗细变化（轮廓粗、发丝细），分组只为控制 path 数。
    粗线先画、细线后画（细节覆盖在轮廓上）。

    注意：不能用 percentile 分桶 + digitize——细线占绝大多数时 25/50/75
    分位会全部相等，digitize(right=False) 会把所有值扔进最后一个桶，
    分组静默塌缩成一个。改用"最近级别"赋值，分布集中也能正确分组。"""
    ws = np.asarray(widths, dtype=float)
    levels = np.unique(np.round(np.linspace(ws.min(), ws.max(), n_buckets),
                                2))
    bids = np.argmin(np.abs(ws[:, None] - levels[None, :]), axis=1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             f'width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    order = sorted(range(len(levels)),
                   key=lambda li: -float(np.median(ws[bids == li])))
    for li in order:
        idx = np.where(bids == li)[0]
        w = float(np.median(ws[idx]))
        d = " ".join(pieces_to_path(stroke_pieces[i])
                     for i in idx if stroke_pieces[i])
        if not d:
            continue
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" '
                     f'stroke-width="{w:.2f}" stroke-linecap="round" '
                     f'stroke-linejoin="round"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_strokes(stroke_pieces, per_piece=24):
    """渲染用：每笔画一条离散化折线"""
    out = []
    for pieces in stroke_pieces:
        if not pieces:
            continue
        pts = np.concatenate(sample_pieces(pieces, per_piece))
        out.append(pts)
    return out
