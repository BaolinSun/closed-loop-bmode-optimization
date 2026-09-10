# -*- coding: utf-8 -*-
"""第三步验证：新目标函数的快慢路径一致性、解的唯一性与内部性。"""
import io, os, sys, time
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import backend_solver as BS
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

lines = []; w = lines.append

w(u"=========== AR. 快速路径与逐帧渲染是否仍然一致（新目标函数）===========")
w(u"%-9s %7s %7s %9s %14s %14s %8s" % (
    u"体模", u"深度", u"频率", u"组合数", u"最大代价差", u"最大项差", u"结论"))
worst_all = 0.0
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 8.0), (60.0, 4.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        db, pivot = c.db_image, c.tissue_median_db
        vm = T.fieldii_tissue_mask(c)
        vd = ~vm
        levels, _ = BS.solve_tgc_shape(db, valid_mask=vm)
        sweep = BS.GainSweep(db, levels, vm, vd)
        shaped = S.apply_tgc(db, levels)
        worst = wt = n = 0
        for dr in [30, 67, 150, 400]:
            window = S.dr_ui_to_window_db(dr)
            for gain in np.arange(-8.0, 24.01, 2.0):
                gray = S.render(db_image=db, tgc_levels=levels, gain_db=gain,
                                dynamic_range_db=window, reference_db=pivot,
                                depth_response_db=None)
                slow, st = OBJ.backend_objective(gray, shaped, vm, vd, return_terms=True)
                fast, ft = sweep.evaluate(gain, window, pivot, return_terms=True)
                worst = max(worst, abs(slow - fast))
                for k in ("crushed", "saturated", "uniformity", "noise_brightening"):
                    wt = max(wt, abs(st[k] - ft[k]))
                n += 1
        worst_all = max(worst_all, worst)
        w(u"%-9s %7.0f %7.1f %9d %14.2e %14.2e %8s" % (
            ptype, depth, freq, n, worst, wt, u"一致" if worst < 1e-9 else u"不一致"))
w(u"  最大代价差 %.3e" % worst_all)

w(u"")
w(u"=========== AS. 解是否唯一、是否在网格内部（Field II 12 帧）===========")
w(u"%-9s %7s %7s %10s %9s %7s %10s %9s %9s" % (
    u"体模", u"深度", u"频率", u"增益dB", u"滑块首档", u"缩放", u"代价", u"等效集", u"顶边界"))
edge = 0
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        vm = T.fieldii_tissue_mask(c)
        answers = set()
        for g0, t0 in [(0.0, 127), (12.0, 60), (-6.0, 200), (20.0, 160)]:
            r = BS.solve_backend(c.db_image, vm, dr_ui=67, reference_db=c.tissue_median_db,
                                 current=(g0, np.full(8, float(t0)), 67))
            answers.add((round(r["gain_db"], 6), tuple(r["tgc_levels"].tolist())))
        edge += r["at_gain_edge"]
        w(u"%-9s %7.0f %7.1f %10.2f %9d %7.2f %10.4f %9d %9s" % (
            ptype, depth, freq, r["gain_db"], r["tgc_levels"][0], r["shape_scale"],
            r["objective"], r["equivalent_count"],
            u"是" if r["at_gain_edge"] else u"否"))
        assert len(answers) == 1, "%s 起点不同解不同：%d" % (ptype, len(answers))
w(u"  4 个不同起点全部收敛到同一解（断言通过）；增益顶边界 %d / 12" % edge)

w(u"")
w(u"=========== AT. 滑块缩放不再恒为 0 了吗（对比第二步）===========")
w(u"  第二步（旧目标函数）：12 帧中 7 帧缩放为 0，动态范围 10 帧顶在 30")
scales = []
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        r = BS.solve_backend(c.db_image, T.fieldii_tissue_mask(c), dr_ui=67,
                             reference_db=c.tissue_median_db)
        scales.append(r["shape_scale"])
w(u"  第三步（新目标函数）：缩放取值 %s，为 0 的有 %d 帧"
  % (sorted(set(scales)), sum(1 for s in scales if s == 0.0)))

w(u"")
w(u"=========== AU. 实机上跑一遍 ===========")
groups = {}
for sess in ["20260903", "20260903_GEN", "20260904"]:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)
floors = T.measure_session_noise_floors(groups, lambda cap: S.bc0_to_db(cap.bc0))
w(u"%-22s %7s %8s %10s %9s %7s %12s %12s" % (
    u"帧", u"深度", u"模式", u"增益dB", u"滑块首档", u"缩放", u"Δ增益(档)", u"噪声提亮项"))
count = 0
for key in sorted(groups):
    f = floors.get(key)
    if f is None:
        continue
    for c in groups[key][:3]:
        d = S.bc0_to_db(c.bc0)
        vm = T.console_tissue_mask(d, f["floor_db"])
        r = BS.solve_backend(
            d, vm, dr_ui=c.dynamic_range_level, reference_db=float(np.median(d[vm])),
            current=(S.capture_gain_db(c),
                     np.asarray(c.tgc_levels, dtype=np.float64),
                     float(c.dynamic_range_level)))
        gray = S.render(db_image=d, tgc_levels=r["tgc_levels"], gain_db=r["gain_db"],
                        dynamic_range_db=S.dr_ui_to_window_db(r["dr_ui"]),
                        reference_db=float(np.median(d[vm])), depth_response_db=None)
        nb = OBJ.noise_brightening_cost(gray, ~vm)
        w(u"%-22s %7.1f %8s %10.2f %9d %7.2f %12.1f %12.4f" % (
            c.name[:22], c.geometry.depth_mm,
            u"通用" if key[1] == 0 else u"谐波",
            r["gain_db"], r["tgc_levels"][0], r["shape_scale"],
            r["delta_gain_levels"], nb))
        count += 1
w(u"  共 %d 帧" % count)

w(u"")
w(u"=========== AV. 速度 ===========")
c = load_shard(find_shards(phantom_type="cyst", depth_mm=42.0,
                           frequency_mhz=5.0, focus_mm=15.0)[0])
vm = T.fieldii_tissue_mask(c)
t0 = time.time(); BS.solve_backend(c.db_image, vm, 67, reference_db=c.tissue_median_db,
                                   fast=False); t_slow = time.time() - t0
t0 = time.time(); BS.solve_backend(c.db_image, vm, 67, reference_db=c.tissue_median_db,
                                   fast=True); t_fast = time.time() - t0
n = len(BS.DEFAULT_SHAPE_SCALES) * len(BS.DEFAULT_GAIN_DB_GRID)
w(u"  每帧 %d 个组合：逐帧渲染 %.2f 秒，预排序 %.2f 秒，提速 %.0f 倍"
  % (n, t_slow, t_fast, t_slow / max(t_fast, 1e-9)))
w(u"  4686 帧：%.1f 小时 → %.1f 分钟" % (t_slow * 4686 / 3600, t_fast * 4686 / 60))

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_verify.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
