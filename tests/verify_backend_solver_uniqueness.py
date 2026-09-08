# -*- coding: utf-8 -*-
"""第一步验证：最优解是否已经唯一。"""
import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import backend_solver as BS
from fieldii_loader import find_shards, load_shard

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "step1b.txt")
lines = []
w = lines.append

cap = load_shard(find_shards(phantom_type="uniform", depth_mm=42.0,
                             frequency_mhz=5.0, focus_mm=15.0)[0])
db = cap.db_image
PIVOT = cap.tissue_median_db
NROWS = db.shape[0]


def render(levels, gain_db, dr_ui):
    return S.render(db_image=db, tgc_levels=levels, gain_db=gain_db,
                    dynamic_range_db=S.dr_ui_to_window_db(dr_ui),
                    reference_db=PIVOT, depth_response_db=None)


def objective(levels, gain_db, dr_ui):
    return OBJ.backend_objective(render(levels, gain_db, dr_ui), db)


# ---------------------------------------------------------------- A
w(u"=========== A. 简并的代数来源：插值权重每行加起来正好是 1 ===========")
basis = BS.tgc_basis(NROWS)
sums = basis.sum(axis=1)
w(u"  逐行权重和：最小 %.12f  最大 %.12f" % (sums.min(), sums.max()))
w(u"  → 8 个滑块同时抬 d 档，每一行的增益都恰好抬 d×0.06559 dB，与深度无关。")
w(u"    这和「总增益抬同样多」是同一件事，所以两者可以互相抵消。")

# ---------------------------------------------------------------- B
w(u"")
w(u"=========== B. 规范化：把整条脊压成一个点，且不改变图像 ===========")
weights = BS.tissue_weight_from_mask(OBJ.signal_mask(db))
ridge = []
for shift in [-60, -30, -10, 0, 10, 30, 60]:
    lv = np.full(8, 127.0) + shift
    g = -shift * S.DEFAULT_DB_PER_LEVEL
    ridge.append((g, lv))
ref_img = render(ridge[0][1], ridge[0][0], 67)
w(u"%10s %12s %14s %14s %14s" % (u"滑块偏移", u"原增益dB", u"规范后增益dB", u"规范后滑块", u"与首张图差"))
canon = []
for g, lv in ridge:
    cg, clv = BS.canonical_action(g, lv, weights, basis)
    img = render(lv, g, 67)
    canon.append((cg, clv))
    w(u"%10d %12.4f %14.6f %14.4f %14d" % (
        int(lv[0] - 127), g, cg, clv[0],
        int(np.abs(img.astype(int) - ref_img.astype(int)).max())))
spread_g = max(c[0] for c in canon) - min(c[0] for c in canon)
spread_t = max(c[1][0] for c in canon) - min(c[1][0] for c in canon)
w(u"  规范化后增益极差 %.3e dB，滑块极差 %.3e 档 → 7 个不同的起点收敛到同一点。"
  % (spread_g, spread_t))

# ---------------------------------------------------------------- C
w(u"")
w(u"=========== C. 修之前：同一份数据，遍历顺序不同就给出不同答案 ===========")
rng = np.random.RandomState(0)
grid = []
for off in range(-60, 61, 10):          # 滑块整体偏移档
    for gdb in np.arange(-8.0, 24.01, 0.5):
        for dr in [30, 45, 67, 100, 150]:
            grid.append((off, gdb, dr))
scored = [(objective(np.full(8, 127.0) + o, g, d), o, g, d) for o, g, d in grid]
best_j = min(s[0] for s in scored)
w(u"  搜索空间 %d 个组合，最小代价 %.6f" % (len(scored), best_j))
w(u"%14s %12s %12s %12s" % (u"遍历顺序", u"滑块偏移档", u"增益dB", u"动态范围"))
seen = set()
for trial in range(5):
    order = rng.permutation(len(scored))
    pick = None
    for idx in order:
        j, o, g, d = scored[idx]
        if pick is None or j < pick[0] - 1e-12:
            pick = (j, o, g, d)
    seen.add((pick[1], pick[2], pick[3]))
    w(u"%14d %12d %12.1f %12d" % (trial + 1, pick[1], pick[2], pick[3]))
w(u"  5 次遍历得到 %d 个不同答案，代价全部等于 %.6f —— 它们描述的是同一张图。"
  % (len(seen), best_j))
tie = [s for s in scored if s[0] <= best_j + 1e-12]
w(u"  代价严格并列最小的组合共 %d 个，横跨滑块偏移 %d…%d 档。"
  % (len(tie), min(s[1] for s in tie), max(s[1] for s in tie)))

# ---------------------------------------------------------------- D
w(u"")
w(u"=========== D. 修之后：解是唯一的，且与起点无关 ===========")
w(u"%14s %14s %14s %12s %10s %10s" % (
    u"起点(增益dB)", u"起点(滑块)", u"起点(动态范围)", u"解出增益dB", u"解出滑块0", u"动态范围"))
answers = set()
for g0, t0, d0 in [(0.0, 127, 67), (12.0, 60, 30), (-6.0, 200, 400),
                   (5.0, 90, 100), (20.0, 160, 150)]:
    res = BS.solve_backend(db, reference_db=PIVOT,
                           current=(g0, np.full(8, float(t0)), d0))
    answers.add((round(res["gain_db"], 6), tuple(res["tgc_levels"].tolist()), res["dr_ui"]))
    w(u"%14.1f %14d %14d %12.3f %10d %10d" % (
        g0, t0, d0, res["gain_db"], res["tgc_levels"][0], res["dr_ui"]))
w(u"  不同起点得到 %d 个不同答案（期望 1）。" % len(answers))

# ---------------------------------------------------------------- E
w(u"")
w(u"=========== E. 解出来的滑块形状确实在压平深度趋势 ===========")
lv, info = BS.solve_tgc_shape(db)
res_auto = BS.solve_backend(db, reference_db=PIVOT)
w(u"  解出滑块：%s" % np.array2string(lv))
w(u"  拟合残差 %.3f dB，可用行数 %d / %d，规范化残差 %.2e dB"
  % (info["fit_rms_db"], info["usable_rows"], NROWS, info["residual_gain_db"]))
res_solved = BS.solve_backend(db, reference_db=PIVOT, shape_scales=(1.0,))
flat_best = min(objective(np.full(8, 127.0), g, d)
                for g in np.arange(-8.0, 24.01, 0.5) for d in BS.DEFAULT_DR_UI_CANDIDATES)
w(u"")
w(u"%22s %12s" % (u"", u"最小代价"))
w(u"%22s %12.4f" % (u"滑块全 127（平的）", flat_best))
w(u"%22s %12.4f" % (u"解出的滑块形状(全量)", res_solved["objective"]))
w(u"%22s %12.4f  (缩放系数 %.2f)" % (u"让代价自己挑缩放", res_auto["objective"], res_auto["shape_scale"]))
g_solved = render(res_auto["tgc_levels"], res_auto["gain_db"], res_auto["dr_ui"])
g_flat = render(np.full(8, 127), res_auto["gain_db"], res_auto["dr_ui"])
w(u"  深度不均匀项（同增益同动态范围下）：平滑块 %.4f → 选中形状 %.4f"
  % (OBJ.depth_uniformity_cost(g_flat, db), OBJ.depth_uniformity_cost(g_solved, db)))

# ---------------------------------------------------------------- F
w(u"")
w(u"=========== F. 三种体模 / 四种设置上跑一遍，看解是否落在搜索范围内部 ===========")
w(u"%-9s %6s %7s %9s %8s %8s %7s %7s %9s %8s" % (
    u"体模", u"深度", u"频率", u"增益dB", u"滑块首档", u"动态范围", u"缩放", u"代价", u"等效集大小", u"顶边界"))
gmin, gmax = BS.DEFAULT_GAIN_DB_GRID[0], BS.DEFAULT_GAIN_DB_GRID[-1]
drs = BS.DEFAULT_DR_UI_CANDIDATES
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        r = BS.solve_backend(c.db_image, reference_db=c.tissue_median_db)
        edge = r["at_gain_edge"] or r["at_dr_edge"]
        w(u"%-9s %6.0f %7.1f %9.2f %8d %8.0f %7.2f %7.3f %9d %8s" % (
            ptype, depth, freq, r["gain_db"], r["tgc_levels"][0], r["dr_ui"],
            r["shape_scale"], r["objective"], r["equivalent_count"],
            (u"增益" if r["at_gain_edge"] else u"") + (u"动态范围" if r["at_dr_edge"] else u"")
            or u"否"))

io.open(OUT, "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
