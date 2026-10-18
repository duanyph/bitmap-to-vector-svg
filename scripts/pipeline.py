# -*- coding: utf-8 -*-
"""
pipeline.py —— 位图转矢量 SVG【正式流程】主程序
=================================================
给定「生图阶段产出的干净线稿位图」+「原彩图（取色来源）」，一条命令产出带色、
分层、可编辑的矢量 SVG（真 path/fill，非栅格嵌入）。

    python pipeline.py                      # 全流程（固定流程，无参数可调）
    python pipeline.py --stage lineart      # 只出线稿矢量
    python pipeline.py --stage fill         # 复用已有线稿产物，只重跑上色
    python pipeline.py --name myart         # 指定产物名前缀

产物（输出目录默认为 当前工作目录/output）
    <name>_lineart.svg    线稿矢量（单 path，统一 2px）
    <name>_lineart.png    原线稿 / 归一化位图 / 矢量化 三方对照
    <name>_normmask.png   归一化线稿掩膜（上色前置）
    <name>_colored.svg    正式成品（白底 → fills-base → fills-main → 线稿）
    <name>_colored.png    成品栅格预览
    <name>_check.png      原彩图 vs 成品（含局部放大）

输入（缺省时在输出目录自动探测）
    lineart_src.png  生图阶段产出的线稿位图（必须已“线条归一化”：统一线宽/纯黑白）
    original.jpg     原彩图，用于配准取色

本脚本为**正式版**：算法参数（阈值/桥接/合并/降曲率/挡位/上色路线等）全部固定为定稿配置，
命令行只暴露作业控制参数（--stage / --name / --outdir / --src-*）。需要调参请用开发版。

处理流程
------------------------------------------------------------------
【线稿链】
  [1]   二值化（阈值 200，保住抗锯齿淡线）+ 外框裁剪
  [2.5] 位图断缝修复：方向判据桥接
        —— 仅"短碎片(extent<6px)"发起；桥连线必须与参照侧笔画局部走向
           (PCA)|cos| >= bridge_cos 才连；参照侧 = 长侧 / 任一侧（可调）
  [3]   骨架化 + 剪毛刺 + 去小连通域
  [4]   2x 超采样统一宽度重渲染（全体线条统一 2.0px，骨架拓扑零损失）
  [5]   再骨架化 -> trace_skeleton 追踪点列（结点处拆笔）
  [9a]  近距连接 = 真正合并点列（<=2.0px，短笔画发起 + 单趟贪心，禁传递闭包）
  [9b]  降低线曲率（点列级）：顶点到弦垂距 <0.75px 拉向弦中点，3 轮
  [9c]  删除孤立噪点：extent<6px 且 5px 内无任何其它笔画的点
  [9d]  拟合 fit_stroke：RDP 抽稀 -> 拐角检测 -> DP 最优分段 -> 基元分级 L/A/C
  [10]  输出单一 path（stroke-width=2.0）+ 归一化掩膜 png
【上色链】
  [P1] 归一化线稿掩膜（画布 W×H）+ 线稿矢量 path
  [P2] 原彩图配准：墨迹 vs 梯度 FFT 互相关，s/dx/dy 缓存在 output/_regist.json
       取色 = 区域内部(EDT>1.5px)像素中位数；渐变 = 大区按亮度分 N 挡分层
  [P3] 分区：墙 = dilate(线稿,2px)，泛洪分区 + 小块并入最大邻接域
  [R]  上色路线（固定 = B+C）：
       B  骨架平面图最小环遍历（闭合区色块，边缘天生贴线）
       C  泛洪分区 + mask 级边缘吸附（补齐 B 未覆盖区域）
       BC B 主基底 + C 补齐（C 只画 B 覆盖率<0.6 的区域，底层）  ← 正式流程固定路线
  [P4] 层序：白底 -> fills-base(补齐) -> fills-main(主基底) -> 线稿 -> 指标
"""
import os
import sys
import time
import json
import argparse
import warnings

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree
from collections import defaultdict

warnings.filterwarnings("ignore")

from skimage.measure import find_contours
from skimage.draw import polygon as sk_polygon
from skimage.draw import line as sk_line

# 保证优先导入本脚本同目录的自研模块（trace.py 等与标准库同名，需屏蔽）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lineart import (skeletonize_mask, prune_spurs, drop_small_components,
                     trace_skeleton, fit_stroke)
from fit_curve import sample_pieces, pieces_to_path

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "output")

# ============================ 默认参数 ============================
# —— 线稿链 ——
DEF_THRESH = 200        # 二值化阈值（AI 线稿抗锯齿淡线灰度中位 167，128 会切掉）
DEF_TARGET_W = 2.0      # 统一线宽(px)
DEF_BRIDGE_GAP = 3      # 位图桥接最大间隙：3（定稿）/ 5（更激进）
DEF_BRIDGE_SIDE = "long"    # 方向判据参照侧：long=较长一侧 / either=任一侧
DEF_BRIDGE_COS = 0.8    # 桥连线与参照侧走向的 |cos| 下限
DEF_SHORT_LEN = 6.0     # 短碎片判定(extent)
DEF_MERGE_DIST = 2.0    # 矢量级合并点列距离（5.0 会误并）
DEF_CURV_DEV = 0.75     # 降曲率：顶点到弦垂距阈值(px)
DEF_CURV_ITERS = 3      # 降曲率迭代轮数
DEF_ISO_DIST = 5.0      # 孤立噪点判定半径
DEF_FIT_TOL = 0.25      # 线稿拟合容差
DEF_RDP_EPS = 0.15
DEF_SMOOTH = 1

# —— 上色链 ——
DEF_ROUTE = "bc"        # b / c / bc(默认) / cb
DEF_MIN_REGION_AREA = 40    # 小于该面积的泛洪区并入邻域
DEF_MIN_FACE_AREA = 50      # B: 小于该面积的环面丢弃
DEF_MAX_FACES = 1600        # B: 最多取多少环面
DEF_BAND_AREA = 20000       # 挡位分层面积门槛（过低会把脸部误判成渐变区）
DEF_BAND_STD = 14           # 挡位分层的亮度标准差门槛
DEF_BANDS = 3               # 渐变挡位数（3 定稿；6/10 可再降色差，体积上升）
DEF_SNAP_R = 1.8            # C: 轮廓点吸附到骨架的半径
DEF_SKIP_BC = 0.6           # B+C: 区域被 B 覆盖达该比例则不再补底
DEF_SKIP_CB = 1.01          # C+B: 底层不剔除任何块（全量打底）
DEF_FILL_FIT_TOL = 0.3
DEF_SMALL_FIT_AREA = 300    # 小区域直接 RDP 折线，不做贝塞尔拟合

SKIP_PREFIX = ("dev_", "fill_", "_")     # 自动探测输入时跳过的产物前缀

T0 = time.time()


def log(msg):
    print(f"[{time.time()-T0:5.0f}s] {msg}", flush=True)


def reg_cache_path():
    return os.path.join(OUT, "_regist.json")


def auto_src(outdir, std_names, exts, hints=()):
    """在 outdir 里自动探测输入素材：优先标准名 -> 名字含关键词 -> 唯一候选。
    找不到返回 None（调用方需提示用户显式传参）。"""
    if not os.path.isdir(outdir):
        return None
    files = sorted(f for f in os.listdir(outdir)
                   if f.lower().endswith(exts) and not f.startswith(SKIP_PREFIX))
    for n in std_names:
        if n in files:
            return os.path.join(outdir, n)
    for f in files:
        lf = f.lower()
        if any(h in lf for h in hints):
            return os.path.join(outdir, f)
    return os.path.join(outdir, files[0]) if len(files) == 1 else None


# ==================================================================
#                        线稿链：工具函数
# ==================================================================
def reduce_curvature(pts, dev_thresh=0.75, iters=3):
    """剔除栅格化锯齿：顶点到前后邻居弦的垂距 < dev_thresh 则拉向弦中点，
    偏离大的真实拐角保留（向量化）。"""
    p = np.asarray(pts, dtype=float).copy()
    if len(p) < 3:
        return p
    for _ in range(iters):
        a, b, c = p[:-2], p[1:-1], p[2:]
        ac = c - a
        L = np.hypot(ac[:, 0], ac[:, 1])
        L[L < 1e-9] = 1e-9
        cross = ac[:, 0] * (b[:, 1] - a[:, 1]) - ac[:, 1] * (b[:, 0] - a[:, 0])
        mask = (np.abs(cross) / L) < dev_thresh
        new = p.copy()
        new[1:-1][mask] = (0.5 * (a + c))[mask]
        p = new
    return p


def stroke_dir(tree, pts, q, r_in=1.2, r_out=6.0, min_pts=4):
    """查询点 q 处本分量的局部走向（PCA 主方向，单位向量）；像素不足返回 None。"""
    idx = tree.query_ball_point(q, r_out)
    if len(idx) < min_pts:
        return None
    P = np.asarray(pts)[idx]
    d = np.linalg.norm(P - q, axis=1)
    P = P[d > r_in]
    if len(P) < min_pts:
        return None
    Pc = P - P.mean(0)
    _w, v = np.linalg.eigh(Pc.T @ Pc / len(Pc))
    return v[:, -1]


def bridge_components(mask, max_gap=3, short_extent=6, dir_cos=0.8,
                      side="long"):
    """位图级断缝修复：方向判据桥接。

    只有"短碎片(extent<short_extent)"发起；桥连线必须与参照侧笔画在连接点处的
    局部走向一致(|cos| >= dir_cos)才连：
        side='long'   —— 参照"较长一侧"的走向（定稿，误连最少）
        side='either' —— 任一侧走向一致即连（更宽松，桥更多，误连风险更高）
    返回 (桥接后掩膜, 桥接处数)。
    """
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask.copy(), 0
    comps, extents, trees = [], [], []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        pts = np.column_stack([xs, ys]).astype(float)
        comps.append(pts)
        extents.append(max(xs.max() - xs.min(), ys.max() - ys.min()))
        trees.append(cKDTree(pts))
    cell = int(max_gap) + 1
    grid = {}
    for i, pts in enumerate(comps):
        for gx in range(int(pts[:, 0].min()) // cell,
                        int(pts[:, 0].max()) // cell + 1):
            for gy in range(int(pts[:, 1].min()) // cell,
                            int(pts[:, 1].max()) // cell + 1):
                grid.setdefault((gx, gy), []).append(i)
    bridge = np.zeros_like(mask, bool)
    done = set()
    n_bridge = 0
    for i, pts_i in enumerate(comps):
        cand = set()
        for gx in range((int(pts_i[:, 0].min()) - int(max_gap)) // cell,
                        (int(pts_i[:, 0].max()) + int(max_gap)) // cell + 1):
            for gy in range((int(pts_i[:, 1].min()) - int(max_gap)) // cell,
                            (int(pts_i[:, 1].max()) + int(max_gap)) // cell + 1):
                for j in grid.get((gx, gy), []):
                    if j != i:
                        cand.add(j)
        for j in cand:
            if (i, j) in done or (j, i) in done:
                continue
            if min(extents[i], extents[j]) > short_extent:
                continue
            done.add((i, j))
            d, idx = trees[j].query(pts_i, k=1)
            k = int(d.argmin())
            if d[k] > max_gap:
                continue
            a, b = pts_i[k], comps[j][idx[k]]
            u = (b - a) / max(d[k], 1e-9)
            if side == "long":
                li = i if extents[i] >= extents[j] else j
                dl = stroke_dir(trees[li], comps[li], a if li == i else b)
                ok = dl is not None and abs(float(u @ dl)) >= dir_cos
            else:   # either：任一侧走向一致即通过
                di = stroke_dir(trees[i], comps[i], a)
                dj = stroke_dir(trees[j], comps[j], b)
                ok = any(dv is not None and abs(float(u @ dv)) >= dir_cos
                         for dv in (di, dj))
            if not ok:
                continue
            rr, cc = sk_line(int(round(a[1])), int(round(a[0])),
                             int(round(b[1])), int(round(b[0])))
            bridge[rr, cc] = True
            n_bridge += 1
    return mask | bridge, n_bridge


def bridge_points(a, b, d):
    """a->b 之间按 ~1px 间距插值桥接点（含两端）。"""
    nseg = max(1, int(np.ceil(d)))
    t = np.linspace(0.0, 1.0, nseg + 1).reshape(-1, 1)
    return a.astype(float) + (b - a) * t


def merge_near_strokes(strokes, dist, short_len):
    """矢量级近距连接 = 真正合并点列（单趟贪心，禁传递性并查集）：
    仅短笔画(extent<short_len)发起，找端点距离<=dist 的最近一笔，两端点首尾
    相接并用插值桥点补实间隙。返回 (合并后点列, 合并对数)。"""
    work = [{"pts": np.asarray(p, float), "closed": c, "alive": True}
            for p, c in strokes]
    exts = np.array([float(np.linalg.norm(w["pts"].max(0) - w["pts"].min(0)))
                     for w in work])
    ends, own = [], []
    for idx, w in enumerate(work):
        ends.append(w["pts"][0])
        ends.append(w["pts"][-1])
        own += [idx, idx]
    tree = cKDTree(np.asarray(ends, float))
    n_merged = 0
    for i in range(len(work)):
        w = work[i]
        if not w["alive"] or exts[i] >= short_len:
            continue
        best = None
        for pe, P in ((0, w["pts"][0]), (1, w["pts"][-1])):
            d, jj = tree.query(P, k=24)
            for dd, j in zip(np.atleast_1d(d), np.atleast_1d(jj)):
                if dd > dist:
                    break
                o = int(own[int(j)])
                if o == i or not work[o]["alive"]:
                    continue
                je = int(j) % 2
                Q = work[o]["pts"][0] if je == 0 else work[o]["pts"][-1]
                real = float(np.linalg.norm(P - Q))
                if real <= dist and (best is None or real < best[0]):
                    best = (real, o, pe, je)
        if best is None:
            continue
        real, o, pe_i, je_o = best
        ith = w["pts"] if pe_i == 1 else w["pts"][::-1]
        oth = work[o]["pts"] if je_o == 1 else work[o]["pts"][::-1]
        if real < 0.5:
            merged = np.vstack([oth, ith[:-1]])
        else:
            merged = np.vstack([oth, bridge_points(oth[-1], ith[-1],
                                                   real)[1:], ith[:-1]])
        work[o]["pts"] = merged
        work[o]["closed"] = False
        w["alive"] = False
        n_merged += 1
    return [(w["pts"], w["closed"]) for w in work if w["alive"]], n_merged


def delete_isolated_noise(strokes, iso_dist, short_len):
    """删除孤立噪点：extent<short_len 且两端点在 iso_dist 内找不到任何其它
    笔画的点（对全部点判定）的短笔画。"""
    pts_all, own = [], []
    for i, (p, _) in enumerate(strokes):
        pts_all.append(p)
        own.append(np.full(len(p), i))
    pts_all = np.vstack(pts_all)
    own = np.concatenate(own)
    tree = cKDTree(pts_all)
    keep, n_del = [], 0
    for i, (p, c) in enumerate(strokes):
        if float(np.linalg.norm(p.max(0) - p.min(0))) >= short_len:
            keep.append((p, c))
            continue
        iso = True
        for P in (p[0], p[-1]):
            for j in tree.query_ball_point(P, iso_dist):
                if own[int(j)] != i:
                    iso = False
                    break
            if not iso:
                break
        if iso:
            n_del += 1
        else:
            keep.append((p, c))
    return keep, n_del


# ==================================================================
#                        线稿链：阶段
# ==================================================================
def load_binary(cfg):
    gen = Image.open(cfg["src_lineart"]).convert("L")
    arr = np.asarray(gen)
    mask0 = arr < cfg["thresh"]
    ys, xs = np.nonzero(mask0)
    y0, y1 = max(0, ys.min() - 10), min(arr.shape[0], ys.max() + 10)
    x0, x1 = max(0, xs.min() - 10), min(arr.shape[1], xs.max() + 10)
    return mask0[y0:y1, x0:x1]


def stage_lineart(cfg):
    """线稿链全流程，产出 <name>_lineart.svg / <name>_normmask.png。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]

    tag = cfg["tag"]
    mask0 = load_binary(cfg)
    H, W = mask0.shape
    log(f"[1] 二值化(阈值{cfg['thresh']}) 掩膜 {mask0.sum()} px, 画布 {W}x{H}")

    clean, n_bridge = bridge_components(
        mask0, max_gap=cfg["bridge_gap"], short_extent=cfg["short_len"],
        dir_cos=cfg["bridge_cos"], side=cfg["bridge_side"])
    log(f"[2.5] 位图桥接(gap={cfg['bridge_gap']}, side={cfg['bridge_side']}, "
        f"cos>={cfg['bridge_cos']}): {n_bridge} 处 -> {clean.sum()} px")

    skel = skeletonize_mask(clean)
    skel = prune_spurs(skel, min_len=3)
    skel = drop_small_components(skel, min_size=4)
    log(f"[3] 骨架 {skel.sum()} px")

    big = np.kron(skel, np.ones((2, 2), bool))
    mb = ndimage.distance_transform_edt(~big) <= 1.0
    Hb, Wb = mb.shape
    mb = mb[: Hb // 2 * 2, : Wb // 2 * 2]
    norm = mb.reshape(Hb // 2, 2, Wb // 2, 2).mean(axis=(1, 3)) > 0.5
    log(f"[4] 统一重渲染 {cfg['target_w']}px: {norm.sum()} px")
    p_norm = os.path.join(OUT, f"{cfg['name']}_normmask.png")
    Image.fromarray((norm * 255).astype(np.uint8)).save(p_norm)

    skel2 = skeletonize_mask(norm)
    skel2 = prune_spurs(skel2, min_len=3)
    skel2 = drop_small_components(skel2, min_size=4)
    strokes_raw = trace_skeleton(skel2, min_extent=2.0)
    log(f"[5] 追踪 {len(strokes_raw)} 笔")

    m_strokes, n_merged = merge_near_strokes(strokes_raw, cfg["merge_dist"],
                                             cfg["short_len"])
    log(f"[9a] 合并点列(<={cfg['merge_dist']}px): {n_merged} 对 / "
        f"{len(strokes_raw)} -> {len(m_strokes)} 笔")

    m_strokes = [(reduce_curvature(p, cfg["curv_dev"], cfg["curv_iters"]), c)
                 for p, c in m_strokes]
    log(f"[9b] 降曲率: 垂距<{cfg['curv_dev']}px × {cfg['curv_iters']} 轮")

    m_strokes, n_del = delete_isolated_noise(m_strokes, cfg["iso_dist"],
                                             cfg["short_len"])
    log(f"[9c] 删孤立噪点(半径{cfg['iso_dist']}px): -{n_del} -> "
        f"{len(m_strokes)} 笔")

    t1 = time.time()
    pieces = [fit_stroke(p, c, tol=cfg["fit_tol"], rdp_eps=cfg["rdp_eps"],
                         smooth_passes=cfg["smooth"]) for p, c in m_strokes]
    kinds = {"L": 0, "A": 0, "C": 0}
    for pc in pieces:
        for _i, _j, pp in pc:
            kinds[pp["kind"]] += 1
    n_el = sum(len(pc) for pc in pieces)
    log(f"[9d] 拟合 {n_el} 基元 (L{kinds['L']}+A{kinds['A']}+C{kinds['C']}) "
        f"({time.time()-t1:.0f}s)")

    ds = [pieces_to_path(pc, closed=c) for (pts, c), pc in zip(m_strokes, pieces)]
    d_all = " ".join(ds)
    svg = "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<path d="{d_all}" fill="none" stroke="#111111" '
        f'stroke-width="{cfg["target_w"]:.2f}" stroke-linecap="round" '
        f'stroke-linejoin="round"/>', "</svg>"])
    p_svg = os.path.join(OUT, f"{cfg['name']}_lineart.svg")
    open(p_svg, "w", encoding="utf-8").write(svg)
    log(f"[10] {os.path.basename(p_svg)} {len(svg.encode())/1024:.0f}KB / "
        f"{len(m_strokes)} 笔 / {n_el} 基元")

    fit_pts = np.concatenate([seg for pc in pieces for seg in sample_pieces(pc, 12)])
    sk_pts = np.column_stack(np.nonzero(skel2)).astype(float)[:, ::-1]
    dm, _ = cKDTree(fit_pts).query(sk_pts)
    log(f"[10b] 覆盖率 @1px {(dm<=1).mean()*100:.1f}% / "
        f"@2px {(dm<=2).mean()*100:.1f}%")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    axes[0].imshow(Image.open(cfg["src_lineart"]), cmap="gray")
    axes[0].set_title("原线稿", fontsize=12)
    axes[1].imshow(norm, cmap="gray")
    axes[1].set_title(f"归一化位图（统一 {cfg['target_w']}px）", fontsize=12)
    ax = axes[2]
    for (pts, c), pc in zip(m_strokes, pieces):
        for seg in sample_pieces(pc, 20):
            ax.plot(seg[:, 0], seg[:, 1], "-", lw=cfg["target_w"], color="#111111")
        if c:
            ax.plot([pts[-1, 0], pts[0, 0]], [pts[-1, 1], pts[0, 1]],
                    "-", lw=cfg["target_w"], color="#111111")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    ax.set_title(f"矢量化 {len(m_strokes)} 笔 / {n_el} 基元", fontsize=12)
    for a_ in axes:
        a_.axis("off")
    fig.suptitle(f"{cfg['name']}  gap={cfg['bridge_gap']} side={cfg['bridge_side']} "
                 f"cos≥{cfg['bridge_cos']}  桥{n_bridge} 合{n_merged} 删{n_del}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p_png = os.path.join(OUT, f"{cfg['name']}_lineart.png")
    fig.savefig(p_png, dpi=95)
    log(f"saved: {os.path.basename(p_svg)} / {os.path.basename(p_norm)} / "
        f"{os.path.basename(p_png)}")
    return dict(norm=norm, svg_path=p_svg, norm_path=p_norm, W=W, H=H)


# ==================================================================
#                        上色链：工具函数
# ==================================================================
def hexc(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb)


def rdp(pts, eps):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        L = float(np.hypot(*ab))
        seg = pts[i + 1:j]
        d = (np.hypot(*(seg - a).T) if L < 1e-9
             else np.abs(np.cross(ab, seg - a)) / L)
        k = int(np.argmax(d))
        if d[k] > eps:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    return pts[keep]


def run_registration(norm, col):
    gcol = col.mean(axis=2)
    gy, gx = np.gradient(gcol)
    grad = np.hypot(gx, gy)
    grad /= (grad.max() + 1e-9)
    ink = norm.astype(np.float32)
    DS = 4
    ink4 = ndimage.gaussian_filter(ink, 2.0)[::DS, ::DS]
    ink4 -= ink4.mean()
    H, W = norm.shape

    def ncc_map(a, b):
        from scipy.signal import fftconvolve
        b0 = b - b.mean()
        return (fftconvolve(a, b0[::-1, ::-1], mode="full") /
                np.sqrt((a * a).sum() * (b0 * b0).sum() + 1e-9))

    best = None
    for s in np.arange(1.0, 2.21, 0.05):
        hh, ww = int(round(gcol.shape[0] * s)), int(round(gcol.shape[1] * s))
        if hh < ink4.shape[0] // 2 or ww < ink4.shape[1] // 2:
            continue
        g4 = ndimage.zoom(grad, (hh / gcol.shape[0], ww / gcol.shape[1]),
                          order=1)[::DS, ::DS]
        g4 = ndimage.gaussian_filter(g4, 1.0)
        m = ncc_map(ink4, g4)
        ij = np.unravel_index(np.argmax(m), m.shape)
        dy4 = ij[0] - (g4.shape[0] - 1)
        dx4 = ij[1] - (g4.shape[1] - 1)
        if best is None or m[ij] > best[0]:
            best = (float(m[ij]), float(s), dy4 * DS, dx4 * DS)
    _, S0, DY0, DX0 = best
    log(f"配准粗扫: s={S0:.2f} dy={DY0:.0f} dx={DX0:.0f}")

    fb = None
    for s in np.arange(S0 - 0.06, S0 + 0.061, 0.01):
        hh, ww = int(round(gcol.shape[0] * s)), int(round(gcol.shape[1] * s))
        gr = ndimage.zoom(grad, (hh / gcol.shape[0], ww / gcol.shape[1]), order=1)
        for dy in range(DY0 - 8, DY0 + 9, 2):
            for dx in range(DX0 - 8, DX0 + 9, 2):
                y0, x0 = max(0, -dy), max(0, -dx)
                y1, x1 = min(hh, H - dy), min(ww, W - dx)
                if y1 - y0 < H * 0.6 or x1 - x0 < W * 0.6:
                    continue
                sub = gr[y0:y1, x0:x1]
                tgt = ink[max(0, dy):max(0, dy) + sub.shape[0],
                          max(0, dx):max(0, dx) + sub.shape[1]]
                if sub.shape != tgt.shape:
                    continue
                a, b = sub - sub.mean(), tgt - tgt.mean()
                c = float((a * b).sum() /
                          np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))
                if fb is None or c > fb[0]:
                    fb = (c, float(s), dy, dx)
    return float(fb[1]), int(fb[2]), int(fb[3])


class Filler(object):
    """上色链：一次配准 + 一次分区，按需构建 A / B / C 三层并组合。"""

    def __init__(self, cfg, norm, line_d):
        self.cfg = cfg
        self.norm = norm
        self.line_d = line_d
        self.H, self.W = norm.shape
        self._load_color()
        self._partition()
        self._skeleton()
        self.fillsA = self.rasterA = None
        self.fillsC = self.rasterC = None
        self.fillsB = self.rasterB = self.coveredB = None

    # ---- P2 配准 ----
    def _load_color(self):
        cfg = self.cfg
        col = np.asarray(Image.open(cfg["src_color"]).convert("RGB"),
                         dtype=np.float32)
        rc = reg_cache_path()
        if os.path.exists(rc):
            r = json.load(open(rc))
            S, DY, DX = float(r["s"]), int(r["dy"]), int(r["dx"])
            log(f"[P2] 配准缓存命中: s={S:.3f} dy={DY} dx={DX}")
        else:
            S, DY, DX = run_registration(self.norm, col)
            json.dump({"s": S, "dy": DY, "dx": DX}, open(rc, "w"))
            log(f"[P2] 配准完成并缓存: s={S:.3f} dy={DY} dx={DX}")
        H, W = self.H, self.W
        yy, xx = np.mgrid[0:H, 0:W]
        self.C = np.stack([ndimage.map_coordinates(
            col[:, :, ch], [(yy - DY) / S, (xx - DX) / S],
            order=1, mode="nearest") for ch in range(3)], -1)
        self.C = np.clip(self.C, 0, 255).astype(np.uint8)

    # ---- P3 分区（A/C 共用）----
    def _partition(self):
        cfg = self.cfg
        wall0 = ndimage.binary_dilation(self.norm, iterations=2)
        lab, n0 = ndimage.label(~wall0)
        log(f"[P3] 泛洪分区初始 {n0} 个")
        for _ in range(4):
            sizes = np.bincount(lab.ravel(), minlength=n0 + 1)
            sizes[0] = 0
            small = np.where((sizes > 0) & (sizes < cfg["min_region_area"]))[0]
            if len(small) == 0:
                break
            h, w = lab.shape
            pairs = []
            for dy, dx in ((0, 1), (1, 0)):
                a = lab[:h - dy, :w - dx].ravel()
                b = lab[dy:, dx:].ravel()
                m = (a != b) & (a > 0) & (b > 0)
                pairs.append(np.stack([a[m], b[m]], 1))
            P = np.vstack(pairs)
            ss = set(small.tolist())
            best_nb = {}
            if len(P):
                uq, cnt = np.unique(P, axis=0, return_counts=True)
                for (a_, b_), c_ in zip(uq.tolist(), cnt.tolist()):
                    if a_ in ss and (a_ not in best_nb or c_ > best_nb[a_][0]):
                        best_nb[a_] = (c_, b_)
            lookup = np.arange(n0 + 1)
            for a_, (_, b_) in best_nb.items():
                lookup[a_] = b_
            lab = lookup[lab]
        lab, inv = np.unique(lab, return_inverse=True)
        self.lab = inv.reshape(self.H, self.W)
        self.nr = int(self.lab.max())
        self.sizes = np.bincount(self.lab.ravel())[1:]
        log(f"[P3] 合并后区域 {self.nr} 个 (最大 {self.sizes.max()} px)")

        self.edt = ndimage.distance_transform_edt(~wall0)
        _, (ri, ci) = ndimage.distance_transform_edt(wall0, return_indices=True)
        self.lab_full = self.lab.copy()
        self.lab_full[wall0] = self.lab[ri[wall0], ci[wall0]]
        log("[P3] 边缘吸附：墙像素按最近区域重分配（边界落到线稿中线）")

    def _skeleton(self):
        skel = skeletonize_mask(self.norm)
        skel = prune_spurs(skel, min_len=3)
        skel = drop_small_components(skel, min_size=4)
        self.skel = skel
        sk_xy = np.column_stack(np.nonzero(skel))[:, ::-1].astype(float)
        self.tree_sk = cKDTree(sk_xy)
        self.sk_xy = sk_xy
        log(f"[P3] 骨架 {len(sk_xy)} px")

    # ---- 取色 / 轮廓 / 出块 ----
    def snap_pts(self, pts):
        d, j = self.tree_sk.query(pts)
        out = np.asarray(pts, float).copy()
        m = d <= self.cfg["snap_r"]
        out[m] = self.sk_xy[j[m]]
        return out

    def region_color(self, mask, edt):
        sel = mask & (edt > 1.5)
        if sel.sum() < 8:
            sel = mask
        if sel.sum() < 4:
            return np.median(self.C.reshape(-1, 3), axis=0)
        return np.median(self.C[sel].reshape(-1, 3), axis=0)

    def emit_region_path(self, cs_xy, color, use_fit):
        ds = []
        for pts in cs_xy:
            if len(pts) < 8:
                continue
            if use_fit:
                pc = fit_stroke(pts, True, tol=self.cfg["fill_fit_tol"],
                                rdp_eps=0.6, smooth_passes=1)
                ds.append(pieces_to_path(pc, closed=True))
            else:
                simp = rdp(pts, 0.7)
                ds.append("M" + " ".join(f"{p[0]:.1f} {p[1]:.1f}"
                                         for p in simp) + "Z")
        if not ds:
            return None
        return ('<path d="%s" fill="%s" fill-rule="evenodd"/>' %
                (" ".join(ds), hexc(color)))

    def contours_of(self, mask, snap, min_len=8):
        cs = find_contours(np.pad(mask.astype(np.float32), 1), 0.5)
        cs = [c[:, ::-1] - 1.0 for c in cs if len(c) >= min_len]
        return [self.snap_pts(p) for p in cs] if snap else cs

    def _lum(self, mask):
        sel = mask & (self.edt > 1.5)
        if sel.sum() < 8:
            sel = mask
        return self.C[sel].astype(np.float32) @ [0.299, 0.587, 0.114]

    def build_region_fills(self, snap, include_wall, tag):
        """A = (snap=False, wall=False)；C = (snap=True, wall=True)。"""
        cfg = self.cfg
        H, W = self.H, self.W
        fills, raster = [], np.full((H, W, 3), 255, np.uint8)
        n_band = 0
        for r in np.argsort(-self.sizes) + 1:
            m_free = self.lab == r
            if m_free.sum() < 2:
                continue
            area = int(m_free.sum())
            m = (self.lab_full == r) if include_wall else m_free
            colr = self.region_color(m_free, self.edt)
            lum = self._lum(m_free)
            use_fit = area >= cfg["small_fit_area"]
            if area > cfg["band_area"] and float(lum.std()) > cfg["band_std"]:
                # 大渐变区：先整区打底（挡位腐蚀后会留 1~2px 缝，打底消除留白）
                p0 = self.emit_region_path(self.contours_of(m, snap), colr,
                                           use_fit)
                if p0:
                    fills.append((area, p0, int(r)))
                    raster[m] = colr
                ths = np.percentile(lum, np.linspace(0, 100, cfg["bands"] + 1)
                                    [1:-1])
                px = np.argwhere(m_free)
                lum_all = self.C[m_free].astype(np.float32) @ \
                    [0.299, 0.587, 0.114]
                edges = np.concatenate([[-1e9], ths, [1e9]])
                for k in range(cfg["bands"]):
                    bm = np.zeros((H, W), bool)
                    sel = px[(lum_all > edges[k]) & (lum_all <= edges[k + 1])]
                    bm[sel[:, 0], sel[:, 1]] = True
                    bm = ndimage.binary_erosion(bm, iterations=1)
                    if bm.sum() < 30:
                        continue
                    bcol = np.median(self.C[bm].reshape(-1, 3), axis=0)
                    pth = self.emit_region_path(
                        self.contours_of(bm, snap, 12), bcol, use_fit)
                    if pth:
                        fills.append((area, pth, int(r)))
                        raster[bm] = bcol
                        n_band += 1
            else:
                pth = self.emit_region_path(self.contours_of(m, snap), colr,
                                            use_fit)
                if pth:
                    fills.append((area, pth, int(r)))
                    raster[m] = colr
        fills.sort(key=lambda t: -t[0])
        log(f"[R] {tag} 层: 色块 {len(fills)} 条（挡位分层 {n_band} 段）")
        return fills, raster

    def ensure_A(self):
        if self.fillsA is None:
            self.fillsA, self.rasterA = self.build_region_fills(
                False, False, "A")
        return self.fillsA, self.rasterA

    def ensure_C(self):
        if self.fillsC is None:
            self.fillsC, self.rasterC = self.build_region_fills(True, True, "C")
        return self.fillsC, self.rasterC

    def ensure_B(self):
        if self.fillsB is not None:
            return self.fillsB, self.rasterB, self.coveredB
        cfg = self.cfg
        H, W = self.H, self.W
        pts_rc = np.column_stack(np.nonzero(self.skel))
        N = len(pts_rc)
        idx_map = -np.ones(self.skel.shape, np.int64)
        idx_map[pts_rc[:, 0], pts_rc[:, 1]] = np.arange(N)
        NB = defaultdict(list)
        rr, cc = pts_rc[:, 0], pts_rc[:, 1]
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                r2, c2 = rr + dy, cc + dx
                ok = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
                j = np.full(N, -1, np.int64)
                j[ok] = idx_map[r2[ok], c2[ok]]
                hit = j >= 0
                for i, jj in zip(np.nonzero(hit)[0], j[hit]):
                    NB[int(i)].append(int(jj))
        for i, lst in NB.items():
            r0, c0 = pts_rc[i]
            lst.sort(key=lambda j: np.arctan2(pts_rc[j][0] - r0,
                                              pts_rc[j][1] - c0))
        visited, faces = set(), []
        for i0 in range(N):
            for jj in NB.get(i0, []):
                if (i0, jj) in visited:
                    continue
                cycle = [i0, jj]
                visited.add((i0, jj))
                prev, cur = i0, jj
                okf = True
                while True:
                    nb = NB.get(cur, [])
                    if not nb:
                        okf = False
                        break
                    k = nb.index(prev) if prev in nb else 0
                    nxt = nb[(k - 1) % len(nb)]
                    if (cur, nxt) in visited:
                        if (cur, nxt) != (i0, jj):
                            okf = False
                        break
                    visited.add((cur, nxt))
                    cycle.append(nxt)
                    prev, cur = cur, nxt
                    if len(cycle) > 20000:
                        okf = False
                        break
                if okf and len(cycle) >= 4:
                    xy = pts_rc[cycle][:, ::-1].astype(float)
                    x, y = xy[:, 0], xy[:, 1]
                    a_ = abs(0.5 * float(np.dot(x, np.roll(y, -1)) -
                                         np.dot(y, np.roll(x, -1))))
                    if a_ >= cfg["min_face_area"]:
                        faces.append((a_, xy))
        faces.sort(key=lambda t: -t[0])
        faces = faces[:cfg["max_faces"]]
        log(f"[R] B 层: 环面 {len(faces)} 个")
        covered = np.zeros((H, W), bool)
        fills, raster = [], np.full((H, W, 3), 255, np.uint8)
        for area, xy in faces:
            rr_, cc_ = sk_polygon(xy[:, 1].astype(int), xy[:, 0].astype(int),
                                  (H, W))
            inside = np.zeros((H, W), bool)
            inside[rr_, cc_] = True
            if inside.sum() < 8:
                continue
            colr = np.median(self.C[inside].reshape(-1, 3), axis=0)
            pc = fit_stroke(xy, True, tol=cfg["fill_fit_tol"], rdp_eps=0.6,
                            smooth_passes=1)
            fills.append('<path d="%s" fill="%s"/>' %
                         (pieces_to_path(pc, closed=True), hexc(colr)))
            covered |= inside
            raster[inside] = colr
        log(f"[R] B 层: 色块 {len(fills)} 条，覆盖 {covered.mean()*100:.1f}%")
        self.fillsB, self.rasterB, self.coveredB = fills, raster, covered
        return fills, raster, covered

    # ---- 组合与指标 ----
    def combine(self, route):
        """返回 (svg填充层字符串列表, 栅格预览, 摘要dict)。"""
        cfg = self.cfg
        if route == "a":
            fills, ras = self.ensure_A()
            paths = [p for _, p, _ in fills]
            out = ras.copy()
            out[self.norm] = (40, 40, 40)
            return paths, out, dict(base=0, main=len(paths))
        if route == "c":
            fills, ras = self.ensure_C()
            paths = [p for _, p, _ in fills]
            out = ras.copy()
            out[self.norm] = (40, 40, 40)
            return paths, out, dict(base=0, main=len(paths))
        if route == "b":
            fills, ras, cov = self.ensure_B()
            out = ras.copy()
            out[self.norm] = (40, 40, 40)
            return list(fills), out, dict(base=0, main=len(fills))
        # bc / cb：C 底层 + B 上层
        fillsC, rasC = self.ensure_C()
        fillsB, rasB, cov = self.ensure_B()
        ratio = cfg["skip_bc"] if route == "bc" else cfg["skip_cb"]
        ids = set()
        for r in range(1, self.nr + 1):
            m = self.lab == r
            if m.sum() < 4:
                continue
            if float(cov[m].mean()) < ratio:
                ids.add(r)
        base = [p for _, p, r in fillsC if r in ids]
        out = np.full((self.H, self.W, 3), 255, np.uint8)
        mb = np.isin(self.lab_full, list(ids))
        out[mb] = rasC[mb]
        out[cov] = rasB[cov]
        out[self.norm] = (40, 40, 40)
        return base + list(fillsB), out, dict(base=len(base), main=len(fillsB))

    def metrics(self, img):
        non = ~self.norm
        white = float(((img.sum(2) > 720) & non).mean()) * 100
        mad = float(np.abs(img.astype(np.int16)[non] - self.C[non]).mean())
        a = img.astype(np.int16)
        eh = (a[:, 1:, :] != a[:, :-1, :]).any(2)
        ev = (a[1:, :, :] != a[:-1, :, :]).any(2)
        e = np.zeros((self.H, self.W), bool)
        e[:, 1:] |= eh
        e[:, :-1] |= eh
        e[1:, :] |= ev
        e[:-1, :] |= ev
        e &= non
        hit = float((ndimage.distance_transform_edt(~self.norm)[e] <= 1.5).mean()) \
            * 100 if e.sum() > 1 else 0.0
        return white, mad, hit

# ==================================================================
#                        上色链：阶段
# ==================================================================
def stage_fill(cfg, norm, line_d, routes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]

    F = Filler(cfg, norm, line_d)
    tag = cfg["tag"]
    H, W = F.H, F.W
    made = {}
    tbl = []

    for route in routes:
        t1 = time.time()
        paths, raster, info = F.combine(route)
        white, mad, hit = F.metrics(raster)
        # ---- SVG（分层结构）----
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
                 f'viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
                 f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
        if route in ("bc", "cb"):
            n_main = info["main"]
            base_p, main_p = paths[:len(paths) - n_main], paths[len(paths) - n_main:]
            parts.append('<!-- 底层补齐 C -->')
            parts.append('<g id="fills-base">')
            parts += base_p
            parts.append('</g>')
            parts.append('<!-- 主基底 B：骨架环面遍历 -->')
            parts.append('<g id="fills-main">')
            parts += main_p
            parts.append('</g>')
        else:
            parts.append(f'<g id="fills-{route}">')
            parts += paths
            parts.append('</g>')
        parts.append(f'<path id="lineart" d="{line_d}" fill="none" '
                     f'stroke="#111111" stroke-width="2.00" '
                     f'stroke-linecap="round" stroke-linejoin="round"/>')
        parts.append('</svg>')
        svg = "\n".join(parts)
        fname = (f"{cfg['name']}_colored.svg" if route == DEF_ROUTE
                 else f"{cfg['name']}_colored_{route}.svg")
        open(os.path.join(OUT, fname), "w", encoding="utf-8").write(svg)
        kb = len(svg.encode()) / 1024
        made[route] = dict(svg=fname, raster=raster, kb=kb, n=len(paths))
        log(f"[P4] {route.upper():3s}: 色块 {len(paths)} (底{info['base']}+"
            f"主{info['main']}) {kb:.0f}KB | 留白 {white:.2f}% | 色差 "
            f"{mad:.1f} | 边界贴合 {hit:.1f}% ({time.time()-t1:.0f}s)")
        tbl.append([route.upper(), len(paths), f"{kb:.0f}KB",
                    f"{white:.2f}%", f"{mad:.1f}", f"{hit:.1f}%"])
        Image.fromarray(raster).save(
            os.path.join(OUT, fname[:-4] + ".png"))

    # ---- 每个产物出一张对照图（原彩图 vs 成品 + 局部放大）----
    head = (max(0, int(H * 0.02)), int(H * 0.34),
            max(0, int(W * 0.28)), int(W * 0.62))
    for k, route in enumerate(routes):
        raster = made[route]["raster"]
        base = (cfg["name"] if route == DEF_ROUTE
                else f"{cfg['name']}_{route}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 11))
        for i, (t, img) in enumerate([("原彩图(配准对齐)", F.C),
                                      (f"成品 {route.upper()} "
                                       f"{made[route]['kb']:.0f}KB", raster)]):
            axes[0, i].imshow(img)
            axes[0, i].set_title(t, fontsize=12)
            y0, y1, x0, x1 = head
            axes[1, i].imshow(img[y0:y1, x0:x1], extent=[x0, x1, y1, y0])
            axes[1, i].set_title("局部放大", fontsize=11)
            for a_ in axes[:, i]:
                a_.axis("off")
        fig.suptitle(f"{cfg['name']} | route={route.upper()}   "
                     f"留白 {tbl[k][3]} / 色差 {tbl[k][4]} / "
                     f"边界贴合 {tbl[k][5]}", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        p = os.path.join(OUT, f"{base}_check.png")
        fig.savefig(p, dpi=110)
        log(f"saved: {made[route]['svg']} / {os.path.basename(p)}")


# ==================================================================
#                              CLI
# ==================================================================
def build_parser():
    """正式版 CLI：只暴露作业控制参数（输入/输出/阶段），算法参数全部固定为定稿配置。

    需要调参（桥接强度、合并阈值、渐变挡位、上色路线等）请使用开发版
    dev_pipeline.py，本脚本不提供任何算法旋钮。
    """
    p = argparse.ArgumentParser(
        description="位图转矢量 SVG 正式流程（固定流程：线稿矢量化 + B+C 分层上色）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--stage", default="all", choices=["lineart", "fill", "all"],
                   help="运行阶段：lineart=只出线稿 / fill=只上色（复用已有线稿产物）/ all=全流程")
    p.add_argument("--name", default="out",
                   help="产物名前缀，用于区分同一输出目录下的不同任务")
    p.add_argument("--src-lineart", default=None,
                   help="线稿位图（生图阶段产物）；缺省在输出目录自动探测")
    p.add_argument("--src-color", default=None,
                   help="原彩图（取色来源）；缺省在输出目录自动探测")
    p.add_argument("--outdir", default=None,
                   help="输出目录，默认 = 当前工作目录/output")
    return p


def main():
    global OUT
    a = build_parser().parse_args()
    OUT = os.path.abspath(a.outdir) if a.outdir else \
        os.path.join(os.getcwd(), "output")
    os.makedirs(OUT, exist_ok=True)

    if a.src_lineart is None:
        a.src_lineart = auto_src(OUT, ("lineart_src.png",), (".png",),
                                 ("line", "mono", "art", "sketch"))
        if a.src_lineart is None:
            sys.exit("[x] 未找到线稿位图：请用 --src-lineart 指定，"
                     "或把生图产物存为 <outdir>/lineart_src.png\n"
                     "    （要求：线条归一化——统一线宽、纯黑白二值、闭合轮廓、白底无噪点）")
    if a.src_color is None:
        a.src_color = auto_src(OUT, ("original.jpg", "original.png"),
                               (".jpg", ".jpeg", ".png"),
                               ("original", "input", "color", "src"))
        if a.src_color is None and a.stage in ("fill", "all"):
            sys.exit("[x] 未找到原彩图：请用 --src-color 指定，"
                     "或存为 <outdir>/original.jpg")
        if a.src_color is None:
            a.src_color = ""      # lineart 阶段不需要取色源

    # 正式版：算法参数全部固定为定稿配置，不接受命令行覆盖
    routes = [DEF_ROUTE]
    cfg = dict(name=a.name, tag=a.name, stage=a.stage, route=DEF_ROUTE,
               bridge_gap=DEF_BRIDGE_GAP, bridge_side=DEF_BRIDGE_SIDE,
               bridge_cos=DEF_BRIDGE_COS, thresh=DEF_THRESH,
               target_w=DEF_TARGET_W, short_len=DEF_SHORT_LEN,
               merge_dist=DEF_MERGE_DIST, curv_dev=DEF_CURV_DEV,
               curv_iters=DEF_CURV_ITERS, iso_dist=DEF_ISO_DIST,
               fit_tol=DEF_FIT_TOL, rdp_eps=DEF_RDP_EPS, smooth=DEF_SMOOTH,
               bands=DEF_BANDS, band_area=DEF_BAND_AREA,
               band_std=DEF_BAND_STD, snap_r=DEF_SNAP_R,
               skip_bc=DEF_SKIP_BC, skip_cb=DEF_SKIP_CB,
               min_face_area=DEF_MIN_FACE_AREA, max_faces=DEF_MAX_FACES,
               min_region_area=DEF_MIN_REGION_AREA,
               small_fit_area=DEF_SMALL_FIT_AREA,
               fill_fit_tol=DEF_FILL_FIT_TOL,
               src_lineart=a.src_lineart, src_color=a.src_color)

    log(f"outdir = {OUT}")
    log(f"线稿源 = {a.src_lineart}")
    log(f"彩图源 = {a.src_color}")
    log(f"== pipeline(正式版) | stage={a.stage} route={DEF_ROUTE} "
        f"name={a.name} ==")

    if a.stage in ("lineart", "all"):
        res = stage_lineart(cfg)
    else:
        p_norm = os.path.join(OUT, f"{a.name}_normmask.png")
        p_svg = os.path.join(OUT, f"{a.name}_lineart.svg")
        if not (os.path.exists(p_norm) and os.path.exists(p_svg)):
            sys.exit(f"[x] 缺少线稿产物 {os.path.basename(p_norm)} / "
                     f"{os.path.basename(p_svg)}，请先跑 --stage lineart")
        res = dict(norm=np.asarray(Image.open(p_norm).convert("L")) > 128,
                   svg_path=p_svg)

    if a.stage == "lineart":
        log(f"总耗时 {time.time()-T0:.0f}s")
        return

    d = open(res["svg_path"], encoding="utf-8").read().split('d="', 1)[1]
    line_d = d.split('"', 1)[0]
    stage_fill(cfg, res["norm"], line_d, routes)
    log(f"总耗时 {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
