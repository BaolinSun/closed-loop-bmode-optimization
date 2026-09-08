# -*- coding: utf-8 -*-
"""第二步验证：换成真值/实测掩膜之后，代价函数的行为变了吗。"""
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import backend_solver as BS
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import band_edges

lines = []; w = lines.append

w(u"=========== X. 掩膜换了以后，多少像素被算成组织 ===========")
w(u"%-9s %7s %7s %14s %14s %12s" % (
    u"体模", u"深度", u"频率", u"旧：估计噪声底", u"新：结构真值", u"差"))
caps = {}
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        caps[(ptype, depth, freq)] = c
        old = OBJ.signal_mask(c.db_image).mean()
        new = T.fieldii_tissue_mask(c).mean()
        w(u"%-9s %7.0f %7.1f %13.1f%% %13.1f%% %11.1f%%" % (
            ptype, depth, freq, 100 * old, 100 * new, 100 * (new - old)))

w(u"")
w(u"=========== Y. 裁黑这一项活过来了吗 ===========")
c = caps[("uniform", 42.0, 5.0)]
db, pivot = c.db_image, c.tissue_median_db
mask = T.fieldii_tissue_mask(c)
w(u"  体模 uniform / 42 mm / 5 MHz，逐档取该档最优增益")
w(u"")
w(u"%9s %10s %10s %10s %10s %10s %11s %10s" % (
    u"动态范围", u"窗宽dB", u"最优增益", u"裁黑", u"饱和", u"不均匀", u"利用率代价", u"总代价"))
sweep = BS.GainWindowSweep(db, np.full(8, 127), mask)
for dr in BS.DEFAULT_DR_UI_CANDIDATES:
    window = S.dr_ui_to_window_db(dr)
    best = None
    for gain in BS.DEFAULT_GAIN_DB_GRID:
        total, terms = sweep.evaluate(gain, window, pivot, return_terms=True)
        if best is None or total < best[0]:
            best = (total, gain, terms)
    total, gain, terms = best
    w(u"%9d %10.1f %10.1f %10.4f %10.4f %10.4f %11.4f %10.4f" % (
        dr, window, gain, terms["crushed"], terms["saturated"],
        terms["uniformity"], terms["utilisation"], total))

w(u"")
w(u"=========== Z. 最优解怎么变的 ===========")
w(u"%-9s %7s %7s %20s %22s" % (u"体模", u"深度", u"频率", u"旧掩膜（增益/范围/缩放）",
                               u"新掩膜（增益/范围/缩放）"))
changed = 0
for key, c in caps.items():
    old = BS.solve_backend(c.db_image, reference_db=c.tissue_median_db)
    new = BS.solve_backend(c.db_image, reference_db=c.tissue_median_db,
                           valid_mask=T.fieldii_tissue_mask(c))
    if (old["dr_ui"], old["shape_scale"]) != (new["dr_ui"], new["shape_scale"]):
        changed += 1
    w(u"%-9s %7.0f %7.1f %20s %22s" % (
        key[0], key[1], key[2],
        u"%+.1f / %.0f / %.2f" % (old["gain_db"], old["dr_ui"], old["shape_scale"]),
        u"%+.1f / %.0f / %.2f" % (new["gain_db"], new["dr_ui"], new["shape_scale"])))
w(u"  12 帧中动态范围或滑块缩放发生变化的：%d 帧" % changed)

w(u"")
w(u"  新掩膜下，动态范围还顶到最小档 30 吗：")
edge = sum(1 for c in caps.values()
           if BS.solve_backend(c.db_image, reference_db=c.tissue_median_db,
                               valid_mask=T.fieldii_tissue_mask(c))["dr_ui"] == 30)
w(u"    12 帧中 %d 帧仍在 30" % edge)

w(u"")
w(u"=========== AA. 最优点上深部的黑像素，现在被看见了吗 ===========")
r = BS.solve_backend(db, reference_db=pivot, valid_mask=mask)
gray = S.render(db_image=db, tgc_levels=r["tgc_levels"], gain_db=r["gain_db"],
                dynamic_range_db=S.dr_ui_to_window_db(r["dr_ui"]),
                reference_db=pivot, depth_response_db=None)
edges = band_edges(db.shape[0], 8)
mm = c.geometry.mm_per_point if False else caps[("uniform", 42.0, 5.0)].geometry.mm_per_point
w(u"  最优：增益 %+.1f dB，动态范围 %.0f，滑块缩放 %.2f"
  % (r["gain_db"], r["dr_ui"], r["shape_scale"]))
w(u"")
w(u"%6s %14s %12s %14s %16s" % (u"深度带", u"深度mm", u"中位灰阶", u"该带黑像素", u"被裁黑项计入"))
for k in range(8):
    lo, hi = edges[k], edges[k + 1]
    blk = gray[lo:hi] <= 2
    seen = blk & mask[lo:hi]
    w(u"%6d %14s %12d %13.1f%% %15.1f%%" % (
        k + 1, u"%.1f–%.1f" % (lo * mm, hi * mm), int(np.median(gray[lo:hi])),
        100 * blk.mean(), 100 * seen.mean()))
w(u"")
w(u"  全图黑像素 %.1f%%，被裁黑项计入 %.1f%%"
  % (100 * (gray <= 2).mean(), 100 * ((gray <= 2) & mask).mean()))

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_verify.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
