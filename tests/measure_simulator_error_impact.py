# -*- coding: utf-8 -*-
"""重建的视觉差异，会不会传到目标函数和最优解上。"""
import io, json, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
import objective as OBJ
import backend_solver as BS
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append
cal_data = json.load(io.open("bmode_opt/console_calibration.json", encoding="utf-8"))
cals = {(g["session"], g["image_mode"]): CAL.GroupCalibration(
    g["counts_per_db"], g["pivot_db"], g["screenshot_gray_error"], 0,
    np.array(g["depth_axis_mm"]), np.array(g["depth_response_db"]))
    for g in cal_data["groups"]}
meta = {(g["session"], g["image_mode"]): (g["noise_floor_db"], g["target_gray"])
        for g in cal_data["groups"]}

SESS = ["20260903", "20260903_GEN", "20260903_replication_check", "20260904", "20260904_DR"]
groups = {}
for sess in SESS:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)

w(u"=========== CF. 有没有配准偏移（整体平移能不能减小误差）===========")
w(u"%-28s %7s %12s %12s %14s" % (u"场次", u"模式", u"最佳行偏移", u"最佳列偏移", u"误差降低"))
for key in sorted(cals):
    cal = cals[key]
    cap = sorted([c for c in groups[key] if S.is_flat_tgc(c)],
                 key=lambda c: c.geometry.depth_mm)[len(groups[key]) // 2]
    shot = crop_capture_image(cap)[0]
    db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
    reb = S.render(db_image=db, tgc_levels=cap.tgc_levels,
                   gain_db=S.capture_gain_db(cap),
                   dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                   reference_db=cal.pivot_db, depth_response_db=None,
                   out_shape=shot.shape).astype(np.float64)
    base = np.median(np.abs(shot - reb))
    best = (base, 0, 0)
    for dy in range(-6, 7):
        for dx in range(-6, 7):
            a = shot[max(0, dy):shot.shape[0] + min(0, dy),
                     max(0, dx):shot.shape[1] + min(0, dx)]
            b = reb[max(0, -dy):reb.shape[0] + min(0, -dy),
                    max(0, -dx):reb.shape[1] + min(0, -dx)]
            v = np.median(np.abs(a - b))
            if v < best[0]:
                best = (v, dy, dx)
    w(u"%-28s %7s %12d %12d %14s" % (
        key[0][:26], u"谐波" if key[1] else u"通用", best[1], best[2],
        u"%.1f → %.1f" % (base, best[0])))

w(u"")
w(u"=========== CG. 目标函数在截图上算 vs 在重建上算 ===========")
w(u"  用同一份掩膜、同一份 dB 图，只换被打分的灰阶图像。")
w(u"  若两者接近，说明视觉差异没有传到标签上。")
w(u"")
w(u"%-28s %7s %9s %9s %9s %11s %11s %9s" % (
    u"场次", u"模式", u"J(截图)", u"J(重建)", u"差", u"裁黑差", u"亮度差", u"均匀性差"))
rows = []
for key in sorted(cals):
    cal = cals[key]
    floor, target = meta[key]
    caps = [c for c in groups[key] if S.is_flat_tgc(c)]
    caps = caps[::max(1, len(caps) // 6)][:6]
    for cap in caps:
        shot = crop_capture_image(cap)[0]
        db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
        db_s = S.scan_convert_linear(db, shot.shape[0], shot.shape[1])
        vm = T.console_tissue_mask(db_s, floor)
        reb = S.render(db_image=db, tgc_levels=cap.tgc_levels,
                       gain_db=S.capture_gain_db(cap),
                       dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                       reference_db=cal.pivot_db, depth_response_db=None,
                       out_shape=shot.shape)
        shaped = db_s + float(S.tgc_level_to_db(cap.tgc_levels[0]))
        a, ta = OBJ.backend_objective(shot, shaped, vm, ~vm, target_gray=target,
                                      return_terms=True)
        b, tb = OBJ.backend_objective(reb, shaped, vm, ~vm, target_gray=target,
                                      return_terms=True)
        rows.append((key, a, b, ta, tb))
    a = np.array([r[1] for r in rows[-len(caps):]])
    b = np.array([r[2] for r in rows[-len(caps):]])
    ta = [r[3] for r in rows[-len(caps):]]; tb = [r[4] for r in rows[-len(caps):]]
    w(u"%-28s %7s %9.4f %9.4f %9.4f %11.4f %11.4f %9.4f" % (
        key[0][:26], u"谐波" if key[1] else u"通用", a.mean(), b.mean(),
        abs(a - b).mean(),
        np.mean([abs(x["crushed"] - y["crushed"]) for x, y in zip(ta, tb)]),
        np.mean([abs(x["brightness"] - y["brightness"]) for x, y in zip(ta, tb)]),
        np.mean([abs(x["uniformity"] - y["uniformity"]) for x, y in zip(ta, tb)])))
allA = np.array([r[1] for r in rows]); allB = np.array([r[2] for r in rows])
w(u"")
w(u"  全部 %d 帧：J 平均绝对差 %.4f（J 本身量级 %.3f），相对 %.1f%%"
  % (len(rows), np.abs(allA - allB).mean(), allA.mean(),
     100 * np.abs(allA - allB).mean() / allA.mean()))
w(u"  作为对照：每帧扰动增益 ±0.22 dB（S0 复现极差）引起的 J 变化见下")

cal = cals[("20260903", 1)]
floor, target = meta[("20260903", 1)]
cap = [c for c in groups[("20260903", 1)] if S.is_flat_tgc(c)][0]
db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
vm = T.console_tissue_mask(db, floor)
sweep = BS.GainSweep(db, cap.tgc_levels, vm, ~vm, target_gray=target)
win = S.dr_ui_to_window_db(cap.dynamic_range_level)
g0 = S.capture_gain_db(cap)
base = sweep.evaluate(g0, win, cal.pivot_db)
w(u"    增益 ±0.22 dB → J 变化 %.4f / %.4f"
  % (abs(sweep.evaluate(g0 + 0.22, win, cal.pivot_db) - base),
     abs(sweep.evaluate(g0 - 0.22, win, cal.pivot_db) - base)))
io.open(os.path.join(HERE, "s5_impact.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
