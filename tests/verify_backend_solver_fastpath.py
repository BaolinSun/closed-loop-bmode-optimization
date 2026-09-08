# -*- coding: utf-8 -*-
"""快速评估器与逐帧渲染的一致性检验，以及提速倍数。"""
import io
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import backend_solver as BS
from fieldii_loader import find_shards, load_shard
import tissue as T

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_fast.txt")
lines = []
w = lines.append

w(u"=========== G. 快速评估器与真实渲染逐点比对 ===========")
w(u"  比对方式：对每个 (增益, 动态范围) 组合，分别用「渲染整幅图再算代价」和")
w(u"  「预排序后查表」两条路算，看两个代价数字差多少。")
w(u"")
w(u"%-9s %6s %7s %9s %14s %14s %10s" % (
    u"体模", u"深度", u"频率", u"组合数", u"最大代价差", u"最大项差", u"结论"))

worst_all = 0.0
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 8.0), (60.0, 4.0)]:
        cap = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                     frequency_mhz=freq, focus_mm=15.0)[0])
        db = cap.db_image
        pivot = cap.tissue_median_db
        mask = T.fieldii_tissue_mask(cap)
        levels, _ = BS.solve_tgc_shape(db)
        sweep = BS.GainWindowSweep(db, levels, mask)

        worst, worst_term, count = 0.0, 0.0, 0
        for dr in [30, 67, 150, 400]:
            window = S.dr_ui_to_window_db(dr)
            for gain in np.arange(-8.0, 24.01, 2.0):
                gray = S.render(db_image=db, tgc_levels=levels, gain_db=gain,
                                dynamic_range_db=window, reference_db=pivot,
                                depth_response_db=None)
                slow, slow_terms = OBJ.backend_objective(gray, db, return_terms=True,
                                                         valid_mask=mask)
                fast, fast_terms = sweep.evaluate(gain, window, pivot, return_terms=True)
                worst = max(worst, abs(slow - fast))
                for key in ("crushed", "saturated", "uniformity", "utilisation"):
                    worst_term = max(worst_term, abs(slow_terms[key] - fast_terms[key]))
                count += 1
        worst_all = max(worst_all, worst)
        w(u"%-9s %6.0f %7.1f %9d %14.2e %14.2e %10s" % (
            ptype, depth, freq, count, worst, worst_term,
            u"一致" if worst < 1e-9 else u"不一致"))

w(u"")
w(u"  全部组合的最大代价差 %.3e" % worst_all)

# ---------------------------------------------------------------- 速度
w(u"")
w(u"=========== H. 提速 ===========")
cap = load_shard(find_shards(phantom_type="cyst", depth_mm=42.0,
                             frequency_mhz=5.0, focus_mm=15.0)[0])
db, pivot = cap.db_image, cap.tissue_median_db

t0 = time.time()
mask_c = T.fieldii_tissue_mask(cap)
res_slow = BS.solve_backend(db, reference_db=pivot, fast=False, valid_mask=mask_c)
t_slow = time.time() - t0

t0 = time.time()
res_fast = BS.solve_backend(db, reference_db=pivot, fast=True, valid_mask=mask_c)
t_fast = time.time() - t0

n_eval = 2 * len(BS.DEFAULT_DR_UI_CANDIDATES) * len(BS.DEFAULT_GAIN_DB_GRID)
w(u"  每帧评估 %d 个组合" % n_eval)
w(u"%22s %12s %14s" % (u"", u"耗时秒", u"4686帧预计"))
w(u"%22s %12.2f %14s" % (u"逐帧渲染", t_slow, u"%.1f 小时" % (t_slow * 4686 / 3600.0)))
w(u"%22s %12.2f %14s" % (u"预排序查表", t_fast, u"%.1f 分钟" % (t_fast * 4686 / 60.0)))
w(u"  提速 %.0f 倍" % (t_slow / max(t_fast, 1e-9)))
w(u"")
w(u"  两条路给出的解：")
w(u"%22s %12s %12s %10s" % (u"", u"增益dB", u"动态范围", u"代价"))
w(u"%22s %12.4f %12.0f %10.5f" % (u"逐帧渲染", res_slow["gain_db"], res_slow["dr_ui"],
                                  res_slow["objective"]))
w(u"%22s %12.4f %12.0f %10.5f" % (u"预排序查表", res_fast["gain_db"], res_fast["dr_ui"],
                                  res_fast["objective"]))
same = (abs(res_slow["gain_db"] - res_fast["gain_db"]) < 1e-9
        and res_slow["dr_ui"] == res_fast["dr_ui"]
        and np.array_equal(res_slow["tgc_levels"], res_fast["tgc_levels"]))
w(u"  滑块是否一致：%s；整体是否同解：%s"
  % (u"是" if np.array_equal(res_slow["tgc_levels"], res_fast["tgc_levels"]) else u"否",
     u"是" if same else u"否"))

io.open(OUT, "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
