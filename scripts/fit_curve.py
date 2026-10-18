# -*- coding: utf-8 -*-
"""
位图曲线 -> 最少元素 SVG 路径还原库
====================================

解决的问题：位图提取出的中心线是一串密密麻麻的点（几十~上百个），
如何用最少的 SVG 基元（直线 L / 圆弧 A / 三次贝塞尔 C）还原它。

方法：
1. 拐角检测：切线方向突变处强制分段（不连续点无法用一个基元表达）
2. 基元分级拟合：每段先试直线 -> 再试圆弧 -> 最后三次贝塞尔，
   谁在容差内就用谁（圆弧用 SVG A 命令，一段圆弧 = 1 个元素）
3. 最优分段：动态规划(DP)求"最大误差 <= tol 约束下的最少基元数"，
   而不是固定长度切块（固定切块会为直线上的一小段也分配一个贝塞尔）
4. G1 连续修正：相邻贝塞尔接头处控制点共线对齐，消除折痕

三次贝塞尔拟合算法：Schneider(1990) 最小二乘 + 正交距离重参数化，
端点固定为数据首尾点，保证 C0 连续。
"""
import numpy as np

# ---------------------------------------------------------------------------
# 三次贝塞尔基础
# ---------------------------------------------------------------------------

def _bezier_eval(P, t):
    """P: (4,2) 控制点; t: (n,) -> (n,2)"""
    t = np.asarray(t, dtype=float)
    mt = 1.0 - t
    b0 = mt ** 3
    b1 = 3 * mt * mt * t
    b2 = 3 * mt * t * t
    b3 = t ** 3
    return (np.outer(b0, P[0]) + np.outer(b1, P[1]) +
            np.outer(b2, P[2]) + np.outer(b3, P[3]))


def _bezier_d1(P, t):
    t = np.asarray(t, dtype=float)
    mt = 1.0 - t
    return (np.outer(3 * mt * mt, P[1] - P[0]) +
            np.outer(6 * mt * t, P[2] - P[1]) +
            np.outer(3 * t * t, P[3] - P[2]))


def _bezier_d2(P, t):
    t = np.asarray(t, dtype=float)
    mt = 1.0 - t
    return (np.outer(6 * mt, P[2] - 2 * P[1] + P[0]) +
            np.outer(6 * t, P[3] - 2 * P[2] + P[1]))


def _project_all(pts, P, t):
    """批量牛顿投影：把所有点投影到曲线上。t: (n,)，返回新 t: (n,)"""
    t = np.clip(t, 0.0, 1.0)
    for _ in range(6):
        B = _bezier_eval(P, t)
        d1 = _bezier_d1(P, t)
        d2 = _bezier_d2(P, t)
        r = pts - B
        f = np.einsum("ij,ij->i", d1, r)
        fp = np.einsum("ij,ij->i", d2, r) - np.einsum("ij,ij->i", d1, d1)
        fp[np.abs(fp) < 1e-12] = 1e-12
        t = np.clip(t - f / fp, 0.0, 1.0)
    return t


# ---------------------------------------------------------------------------
# 三种基元拟合
# ---------------------------------------------------------------------------

def fit_line(pts):
    """直线拟合：返回最大垂(正交)距离误差"""
    p0, p1 = pts[0], pts[-1]
    v = p1 - p0
    L = np.hypot(*v)
    if L < 1e-12:
        return float(np.max(np.linalg.norm(pts - p0, axis=1)))
    v = v / L
    w = pts - p0
    perp = np.abs(w[:, 0] * v[1] - w[:, 1] * v[0])  # 2D 叉积 = 垂距
    return float(np.max(perp))


def fit_arc(pts):
    """代数圆拟合(Kasa)：x^2+y^2+Dx+Ey+F=0 最小二乘。
    返回 (center, radius, max_err, a0, span)；点太少或退化时返回 None。
    a0/span 为沿点列累积的真实角跨度（首尾弦会低估近闭合圆弧的跨度）。"""
    if len(pts) < 5:
        return None
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x * x + y * y)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    D, E, F = sol
    cx, cy = -D / 2.0, -E / 2.0
    r2 = cx * cx + cy * cy - F
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    if r < 3.0:  # 半径太小没有矢量意义，按直线/贝塞尔处理
        return None
    err = float(np.max(np.abs(np.hypot(x - cx, y - cy) - r)))
    # 沿点列累积角跨度（y-down 坐标系，正角度 = 顺时针）
    angs = np.arctan2(y - cy, x - cx)
    d = np.diff(angs)
    d = (d + np.pi) % (2 * np.pi) - np.pi  # 缠绕到 [-pi, pi]
    span = float(d.sum())
    if abs(span) > 5.2:  # > ~300 度，单条 SVG A 命令无法表达
        return None
    return np.array([cx, cy]), float(r), err, float(angs[0]), span


def fit_cubic(pts, iters=6):
    """Schneider 最小二乘三次贝塞尔拟合。
    返回 (P(4,2), max_err)；失败返回 None。"""
    n = len(pts)
    if n < 2:
        return None
    P0, P3 = pts[0], pts[-1]
    # 弦长参数化
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if cum[-1] < 1e-12:
        return None
    t = cum / cum[-1]
    P1 = P0 + (P3 - P0) / 3.0
    P2 = P0 + (P3 - P0) * 2.0 / 3.0
    for _ in range(iters):
        A = 3 * (1 - t) ** 2 * t
        C = 3 * (1 - t) * t * t
        ux = pts[:, 0] - (1 - t) ** 3 * P0[0] - t ** 3 * P3[0]
        uy = pts[:, 1] - (1 - t) ** 3 * P0[1] - t ** 3 * P3[1]
        M = np.array([[float((A * A).sum()), float((A * C).sum())],
                      [float((A * C).sum()), float((C * C).sum())]])
        if abs(np.linalg.det(M)) < 1e-14:
            return None
        sx = np.linalg.solve(M, [float((A * ux).sum()), float((C * ux).sum())])
        sy = np.linalg.solve(M, [float((A * uy).sum()), float((C * uy).sum())])
        P1 = np.array([sx[0], sy[0]])
        P2 = np.array([sx[1], sy[1]])
        P = np.array([P0, P1, P2, P3])
        t_new = _project_all(pts, P, t)
        if np.max(np.abs(t_new - t)) < 1e-10:
            t = t_new
            break
        t = t_new
    P = np.array([P0, P1, P2, P3])
    err = float(np.max(np.linalg.norm(_bezier_eval(P, t) - pts, axis=1)))
    return P, err


def fit_piece(pts, tol):
    """对一段点列尝试三种基元，返回误差最小的可行基元；不可行返回 None。
    优先级：直线/圆弧（元素语义更干净）优先于贝塞尔。"""
    if len(pts) < 2:
        return None
    e_line = fit_line(pts)
    if e_line <= tol:
        return {"kind": "L", "P": np.array([pts[0], pts[-1]]), "err": e_line}
    arc = fit_arc(pts)
    if arc is not None and arc[2] <= tol:
        c, r, e, a0, span = arc
        return {"kind": "A", "P": np.array([pts[0], pts[-1]]),
                "center": c, "r": r, "a0": a0, "span": span, "err": e}
    cub = fit_cubic(pts)
    if cub is not None and cub[1] <= tol:
        return {"kind": "C", "P": cub[0], "err": cub[1]}
    return None


# ---------------------------------------------------------------------------
# 拐角检测
# ---------------------------------------------------------------------------

def detect_corners(pts, min_turn_deg=22.0, win=None, closed=False, rel_k=2.5):
    """检测切线方向突变点（拐角），返回拐角索引列表。
    稠密点列（如 marching squares 原始轮廓）上判据：
    转角 > 绝对阈值 且 明显大于邻域平均转角（局部离群）。
    尖角 = 直边上转角突然爆发；光滑曲线的转角是渐变的，不会误检。
    win=None 时按点列密度自适应；closed=True 时按环形处理。"""
    n = len(pts)
    if n < 3:
        return []
    if win is None:
        win = 1 if n <= 20 else max(2, n // 40)
    if closed:
        if n < 2 * win + 1:
            win = 1
    else:
        if n < 2 * win + 3:
            win = max(1, (n - 3) // 2)
        if win < 1:
            return []
    turns = np.zeros(n)
    for i in range(n):
        if closed:
            im, ip = (i - win) % n, (i + win) % n
        else:
            im, ip = i - win, i + win
            if im < 0 or ip > n - 1:
                continue
        v0 = pts[i] - pts[im]
        v1 = pts[ip] - pts[i]
        a0 = np.arctan2(v0[1], v0[0])
        a1 = np.arctan2(v1[1], v1[0])
        turns[i] = np.degrees(abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi))
    # 邻域基线：局部窗口中转角的中位数
    w2 = max(win, n // 8)
    corners = []
    for i in range(n):
        if turns[i] <= min_turn_deg:
            continue
        if closed:
            widx = (np.arange(-w2, w2 + 1) + i) % n
        else:
            lo, hi = max(0, i - w2), min(n, i + w2 + 1)
            widx = np.arange(lo, hi)
        base = float(np.median(turns[widx]))
        if turns[i] < max(min_turn_deg, rel_k * base):
            continue
        if closed:
            prev, nxt = (i - 1) % n, (i + 1) % n
        else:
            prev, nxt = i - 1, i + 1
            if prev < 0 or nxt > n - 1:
                continue
        if turns[i] >= turns[prev] and turns[i] >= turns[nxt]:
            corners.append(i)
    return corners


# ---------------------------------------------------------------------------
# 最优分段：动态规划（全局最少基元数）
# ---------------------------------------------------------------------------

def segment_minimal(pts, tol, corners=None):
    """DP: 把 pts 分成尽量少的段，每段都能用单个基元在 tol 内拟合。
    返回 [(i, j, piece)]，i/j 为闭区间索引。"""
    n = len(pts)
    if n < 2:
        return []
    forced = set(corners or [])
    forced.update({0, n - 1})
    INF = float("inf")
    cost = [INF] * n
    prev = [-1] * n
    cache = {}
    cost[0] = 0
    for j in range(1, n):
        # 段起点 i 必须从上一个强制边界之后开始，且段内不含强制拐角
        lo = 0
        for c in sorted(forced):
            if c < j:
                lo = c
        for i in range(lo, j):
            if cost[i] == INF:
                continue
            if any(i < c < j for c in forced):
                continue
            key = (i, j)
            if key not in cache:
                cache[key] = fit_piece(pts[i:j + 1], tol)
            piece = cache[key]
            if piece is None:
                continue
            if cost[i] + 1 < cost[j]:
                cost[j] = cost[i] + 1
                prev[j] = i
    if cost[n - 1] == INF:  # 容差太苛刻，放宽重试一次
        return segment_minimal(pts, tol * 2.5, corners)
    # 回溯
    out = []
    j = n - 1
    while j > 0:
        i = prev[j]
        out.append((i, j, cache[(i, j)]))
        j = i
    out.reverse()
    return out


# ---------------------------------------------------------------------------
# G1 连续修正
# ---------------------------------------------------------------------------

def enforce_g1(pieces, max_angle_deg=22.0):
    """相邻两段都是贝塞尔时，若接头两侧控制点方向接近共线，
    把入控制点对齐到出控制点方向，消除视觉折痕（不改变元素个数）。"""
    for k in range(1, len(pieces)):
        a, b = pieces[k - 1][2], pieces[k][2]
        if a["kind"] != "C" or b["kind"] != "C":
            continue
        joint = b["P"][0]
        vin = a["P"][2] - joint   # 指向接头内部（反向）
        vout = b["P"][1] - joint  # 指出接头（正向）
        nv = np.linalg.norm(vout)
        if nv < 1e-9:
            continue
        ang = np.degrees(np.arccos(np.clip(
            np.dot(vin, -vout) / (np.linalg.norm(vin) * nv + 1e-12), -1, 1)))
        if ang < max_angle_deg:
            a["P"][2] = joint + (-vout / nv) * np.linalg.norm(vin)


# ---------------------------------------------------------------------------
# 输出 SVG path
# ---------------------------------------------------------------------------

def _fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("-0", "") else "0"


def pieces_to_path(pieces, closed=False):
    """把分段结果转成 SVG path d 字符串。closed=True 时追加 Z（闭合轮廓）。"""
    d = []
    for idx, (i, j, p) in enumerate(pieces):
        P = p["P"]
        if idx == 0:
            d.append(f"M{_fmt(P[0][0])} {_fmt(P[0][1])}")
        if p["kind"] == "L":
            d.append(f"L{_fmt(P[1][0])} {_fmt(P[1][1])}")
        elif p["kind"] == "C":
            d.append(f"C{_fmt(P[1][0])} {_fmt(P[1][1])} "
                     f"{_fmt(P[2][0])} {_fmt(P[2][1])} "
                     f"{_fmt(P[3][0])} {_fmt(P[3][1])}")
        elif p["kind"] == "A":
            r, span = p["r"], p["span"]
            large = 1 if abs(span) > np.pi else 0
            sweep = 1 if span > 0 else 0  # y-down 系：正角 = 顺时针 = sweep 1
            d.append(f"A{_fmt(r)} {_fmt(r)} 0 {large} {sweep} "
                     f"{_fmt(P[1][0])} {_fmt(P[1][1])}")
    if closed:
        d.append("Z")
    return " ".join(d)


def sample_pieces(pieces, per_piece=40):
    """把拟合结果离散化，用于渲染对比图。"""
    out = []
    for _idx, (_i, _j, p) in enumerate(pieces):
        P = p["P"]
        if p["kind"] == "L":
            ts = np.linspace(0, 1, per_piece)
            out.append(np.outer(1 - ts, P[0]) + np.outer(ts, P[1]))
        elif p["kind"] == "C":
            out.append(_bezier_eval(P, np.linspace(0, 1, per_piece)))
        elif p["kind"] == "A":
            c, r = p["center"], p["r"]
            ts = np.linspace(p["a0"], p["a0"] + p["span"], per_piece)
            out.append(c + r * np.column_stack([np.cos(ts), np.sin(ts)]))
    return out


# ---------------------------------------------------------------------------
# 位图 -> 中心线点列
# ---------------------------------------------------------------------------

def stroke_centerline(gray_img, thresh=128):
    """从笔画位图提取中心线：逐列取暗像素的加权平均位置。
    仅适用于 x 单调的笔画（波浪线等）。"""
    arr = np.asarray(gray_img, dtype=float)
    h, w = arr.shape
    dark = 255.0 - arr  # 越暗权重越大
    dark[dark < (255 - thresh)] = 0.0
    pts = []
    for x in range(w):
        col = dark[:, x]
        s = col.sum()
        if s > 1e-6:
            ys = np.arange(h)
            pts.append([float(x), float((col * ys).sum() / s)])
    if len(pts) < 2:
        return np.zeros((0, 2))
    return np.array(pts)


def subsample(pts, max_n=120):
    """等间隔抽稀到 max_n 个点（DP 是 O(n^2)，控制规模）。"""
    n = len(pts)
    if n <= max_n:
        return pts
    idx = np.unique(np.linspace(0, n - 1, max_n).astype(int))
    return pts[idx]
