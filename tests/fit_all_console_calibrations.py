# -*- coding: utf-8 -*-
"""标定全部场次，并测出每组的「标签不确定度」。"""
import io, json, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
import objective as OBJ
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append
MODE = {0: u"通用GEN", 1: u"谐波THI"}

SESSIONS = ["20260814", "20260819", "20260831", "20260901", "20260901_E2",
            "20260901_E3", "20260903", "20260903_GEN",
            "20260903_replication_check", "20260904", "20260904_DR",
            "20260828/GEN", "20260828/THI"]
groups = {}
for sess in SESSIONS:
    try:
        paths = find_captures(DEFAULT_DATA_DIR / sess)
    except Exception:
        continue
    for path in paths:
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)

w(u"=========== CJ. 全部场次逐组标定 ===========")
w(u"%-30s %8s %6s %11s %10s %12s %11s" % (
    u"场次", u"模式", u"帧数", u"counts/dB", u"pivot_dB", u"拟合灰阶误差", u"后端有扫描"))
cals, swept = {}, {}
for key in sorted(groups):
    caps = groups[key]
    cal = CAL.fit_group(caps, limit=min(16, len(caps)))
    if cal is None:
        continue
    cals[key] = cal
    n_settings = len({(c.gain_level, c.tgc_levels[0], c.dynamic_range_level) for c in caps})
    swept[key] = n_settings >= 4
    w(u"%-30s %8s %6d %11.1f %10.2f %12.2f %11s" % (
        key[0][:28], MODE[key[1]], len(caps), cal.counts_per_db, cal.pivot_db,
        cal.gray_error, u"是(%d档)" % n_settings if swept[key] else u"否"))

w(u"")
w(u"=========== CK. 标签不确定度：J(重建)−J(截图) 去掉常数偏置后的残差 ===========")
w(u"  只有后端有扫描的组才测得出来（需要同一位置多个设置的截图）")
w(u"")
w(u"%-30s %8s %8s %12s %12s %14s %12s" % (
    u"场次", u"模式", u"设置数", u"偏置(均值)", u"残差标准差", u"J跨设置跨度", u"占跨度"))
uncertainty = {}
for key in sorted(cals):
    if not swept.get(key):
        continue
    cal = cals[key]
    caps = groups[key]
    fl = T.measure_noise_floor(caps, lambda c: S.bc0_to_db(c.bc0, cal.counts_per_db))
    floor = fl["floor_db"] if fl else float(np.percentile(
        S.bc0_to_db(caps[0].bc0, cal.counts_per_db), 5))
    a, b = [], []
    for cap in caps:
        shot = crop_capture_image(cap)[0]
        db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
        db_s = S.scan_convert_linear(db, shot.shape[0], shot.shape[1])
        vm = T.console_tissue_mask(db_s, floor)
        if vm.sum() < 1000:
            continue
        reb = S.render(db_image=db, tgc_levels=cap.tgc_levels,
                       gain_db=S.gain_level_to_db(cap.gain_level),
                       dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                       reference_db=cal.pivot_db, depth_response_db=None,
                       out_shape=shot.shape).astype(np.float64)
        shaped = db_s + S.expand_tgc_to_depth(cap.tgc_levels, db_s.shape[0])[:, None]
        a.append(OBJ.backend_objective(shot, shaped, vm, ~vm, target_gray=40.0))
        b.append(OBJ.backend_objective(reb, shaped, vm, ~vm, target_gray=40.0))
    if len(a) < 4:
        continue
    a = np.array(a); b = np.array(b); d = b - a
    span = a.max() - a.min()
    uncertainty[key] = float(d.std())
    w(u"%-30s %8s %8d %12.4f %12.4f %14.4f %11.1f%%" % (
        key[0][:28], MODE[key[1]], len(a), d.mean(), d.std(), span,
        100 * d.std() / max(span, 1e-9)))

w(u"")
w(u"  ⚠ 只有谐波模式有后端扫描。通用模式没有任何「同一位置多个后端设置 + 截图」")
w(u"    的数据，所以它的不确定度只能借用谐波值，尚未验证。")

# 借用：同模式取中位；无同模式则取全局中位
allu = list(uncertainty.values())
for key in cals:
    if key in uncertainty:
        continue
    same = [v for k, v in uncertainty.items() if k[1] == key[1]]
    uncertainty[key] = float(np.median(same if same else allu))

out = {"note": ("Per (session, imaging mode) calibration. counts_per_db and pivot_db are "
                "only meaningful as a pair and only within their own group. "
                "label_uncertainty is the standard deviation of J(rebuild) - J(screenshot) "
                "after removing the constant bias, which is the part that can move an argmin."),
       "groups": []}
for key in sorted(cals):
    cal = cals[key]
    caps = groups[key]
    fl = T.measure_noise_floor(caps, lambda c: S.bc0_to_db(c.bc0, cal.counts_per_db))
    floor = fl["floor_db"] if fl else None
    out["groups"].append({
        "session": key[0], "image_mode": int(key[1]),
        "image_mode_name": T.IMAGE_MODE_NAMES[int(key[1])],
        "num_frames": len(caps),
        "counts_per_db": round(cal.counts_per_db, 2),
        "pivot_db": round(cal.pivot_db, 3),
        "screenshot_gray_error": round(cal.gray_error, 3),
        "noise_floor_db": None if floor is None else round(floor, 3),
        "noise_floor_measured": floor is not None,
        "label_uncertainty": round(uncertainty[key], 4),
        "label_uncertainty_measured": bool(swept.get(key)),
        "depth_axis_mm": [round(float(v), 3) for v in cal.depth_axis_mm],
        "depth_response_db": [round(float(v), 4) for v in cal.depth_response_db],
    })
io.open("bmode_opt/console_calibration.json", "w", encoding="utf-8").write(
    json.dumps(out, indent=2, ensure_ascii=False))
w(u"")
w(u"  写出 bmode_opt/console_calibration.json：%d 组，其中噪声底实测 %d 组、"
  u"不确定度实测 %d 组"
  % (len(out["groups"]), sum(1 for g in out["groups"] if g["noise_floor_measured"]),
     sum(1 for g in out["groups"] if g["label_uncertainty_measured"])))
io.open(os.path.join(HERE, "s6_calall.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
