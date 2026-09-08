# -*- coding: utf-8 -*-
"""动态范围到底该由什么决定：窗宽与信号跨度的关系，以及 gCNR 的走向。"""
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import backend_solver as BS
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

lines = []; w = lines.append

w(u"=========== AI. 旧的「利用率」其实就是「窗宽 vs 信号跨度」 ===========")
w(u"  灰阶跨度 = 信号dB跨度 / 窗宽 × 255，所以")
w(u"  1 − 灰阶跨度/255  ==  1 − 信号dB跨度/窗宽。两个公式是同一个。")
cap = load_shard(find_shards(phantom_type="uniform", depth_mm=42.0,
                             frequency_mhz=5.0, focus_mm=15.0)[0])
db, pivot = cap.db_image, cap.tissue_median_db
mask = T.fieldii_tissue_mask(cap)
span = OBJ.signal_span_db(db, mask)
w(u"")
w(u"%9s %9s %14s %16s %10s" % (u"动态范围", u"窗宽dB", u"旧利用率代价", u"1−跨度/窗宽", u"差"))
sweep = BS.GainWindowSweep(db, np.full(8, 127), mask)
for dr in BS.DEFAULT_DR_UI_CANDIDATES:
    window = S.dr_ui_to_window_db(dr)
    best = min((sweep.evaluate(g, window, pivot, return_terms=True)
                for g in BS.DEFAULT_GAIN_DB_GRID), key=lambda x: x[0])
    old = best[1]["utilisation"]
    new = OBJ.window_fit_cost(span, window)
    w(u"%9d %9.1f %14.4f %16.4f %10.4f" % (dr, window, old, new, abs(old - new)))
w(u"")
w(u"  → 我原以为「改成按组织电平算」能修好，实测不能：它只是同一个量。")

w(u"")
w(u"=========== AJ. 各帧的信号 dB 跨度 vs 最窄可设窗宽 ===========")
w(u"  控制台最窄档 动态范围=30，窗宽 %.1f dB" % S.dr_ui_to_window_db(30))
w(u"")
w(u"%-9s %7s %7s %14s %16s %12s" % (
    u"体模", u"深度", u"频率", u"信号跨度dB", u"最优窗宽应≈跨度", u"实选动态范围"))
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        m = T.fieldii_tissue_mask(c)
        sp = OBJ.signal_span_db(c.db_image, m)
        r = BS.solve_backend(c.db_image, reference_db=c.tissue_median_db, valid_mask=m)
        w(u"%-9s %7.0f %7.1f %14.1f %16s %12.0f" % (
            ptype, depth, freq, sp,
            u"低于最窄档" if sp < S.dr_ui_to_window_db(30) else u"%.0f dB" % sp,
            r["dr_ui"]))

w(u"")
w(u"=========== AK. 实机帧的信号跨度（空间复合过，斑点起伏更小）===========")
groups = {}
for sess in ["20260903", "20260903_GEN", "20260904"]:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)
floors = T.measure_session_noise_floors(groups, lambda cap: S.bc0_to_db(cap.bc0))
w(u"%-9s %9s %8s %14s %20s" % (u"模式", u"显示深度", u"帧数", u"信号跨度dB", u"该跨度对应的动态范围档"))
rows = {}
for key, caps in groups.items():
    f = floors.get(key)
    if f is None:
        continue
    for c in caps:
        d = S.bc0_to_db(c.bc0)
        m = T.console_tissue_mask(d, f["floor_db"])
        sp = OBJ.signal_span_db(d, m)
        if np.isfinite(sp):
            rows.setdefault((u"通用GEN" if key[1] == 0 else u"谐波THI",
                             round(c.geometry.depth_mm, 1)), []).append(sp)
for k in sorted(rows):
    v = np.array(rows[k])
    ui = (v.mean() - S.DR_WINDOW_INTERCEPT) / S.DR_WINDOW_SLOPE
    w(u"%-9s %9.1f %8d %14.1f %20s" % (
        k[0], k[1], len(v), v.mean(),
        u"低于 30" if ui < 30 else u"%.0f" % ui))

w(u"")
w(u"=========== AL. gCNR 随动态范围怎么走（囊肿体模）===========")
cc = load_shard(find_shards(phantom_type="cyst", depth_mm=42.0,
                            frequency_mhz=5.0, focus_mm=15.0)[0])
cdb, cpivot = cc.db_image, cc.tissue_median_db
cmask = T.fieldii_tissue_mask(cc)
lesion = np.asarray(cc.truth_mask) > 0
rows_l = np.where(lesion.any(axis=1))[0]
bg = cmask.copy(); bg[:rows_l.min()] = False; bg[rows_l.max() + 1:] = False
csweep = BS.GainWindowSweep(cdb, np.full(8, 127), cmask)
w(u"%9s %9s %10s %10s %14s %12s" % (
    u"动态范围", u"窗宽dB", u"最优增益", u"gCNR", u"可辨性代价", u"囊肿全黑比例"))
for dr in BS.DEFAULT_DR_UI_CANDIDATES:
    window = S.dr_ui_to_window_db(dr)
    gain = min(BS.DEFAULT_GAIN_DB_GRID,
               key=lambda g: csweep.evaluate(g, window, cpivot))
    gray = S.render(db_image=cdb, tgc_levels=np.full(8, 127), gain_db=gain,
                    dynamic_range_db=window, reference_db=cpivot, depth_response_db=None)
    g = OBJ.gcnr(gray[lesion], gray[bg])
    w(u"%9d %9.1f %10.1f %10.4f %14.4f %11.1f%%" % (
        dr, window, gain, g, 1 - g, 100 * (gray[lesion] <= 2).mean()))

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_span.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
