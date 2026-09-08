# -*- coding: utf-8 -*-
"""第四步：用逐组标定重测噪声底与亮度目标，并重跑求解器。"""
import io, os, sys, pickle
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
import objective as OBJ
import backend_solver as BS
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append
SESS = ["20260903", "20260903_GEN", "20260903_replication_check", "20260904", "20260904_DR"]
MODE = {0: u"通用GEN", 1: u"谐波THI"}

groups = {}
for sess in SESS:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)
with open(os.path.join(HERE, "cals.pkl"), "rb") as fh:
    raw = pickle.load(fh)
cals = {k: CAL.GroupCalibration(v[0], v[1], v[4], 0, v[2], v[3]) for k, v in raw.items()}


def db_of(cap, key):
    return S.bc0_to_db(cap.bc0, cals[key].counts_per_db)


def render_own(cap, key):
    cal = cals[key]
    return S.render(cap.bc0, tgc_levels=cap.tgc_levels,
                    gain_db=S.gain_level_to_db(cap.gain_level),
                    dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                    depth_response_db=CAL.depth_response_for(cap, cal),
                    reference_db=cal.pivot_db, counts_per_db=cal.counts_per_db)


w(u"=========== BK. 用逐组标定重测噪声底 ===========")
w(u"  第二步用的是全局 counts_per_db=877.3，dB 刻度对通用模式是错的。")
w(u"")
w(u"%-30s %9s %8s %14s %14s %10s" % (
    u"场次", u"模式", u"够深帧", u"旧（全局877.3）", u"新（组自身）", u"标准差"))
floors_old = T.measure_session_noise_floors(groups, lambda cap: S.bc0_to_db(cap.bc0))
floors_new = {}
for key in sorted(groups):
    if key not in cals:
        continue
    r = T.measure_noise_floor(groups[key], lambda cap, k=key: db_of(cap, k))
    if r is None:
        continue
    floors_new[key] = r
    w(u"%-30s %9s %8d %14.2f %14.2f %10.2f" % (
        key[0], MODE[key[1]], r["num_frames"],
        floors_old[key]["floor_db"], r["floor_db"], r["std_db"]))
# borrow for groups without deep frames
for key in sorted(groups):
    if key in floors_new or key not in cals:
        continue
    donors = [(k, v) for k, v in floors_new.items() if k[1] == key[1]]
    if donors:
        floors_new[key] = dict(donors[0][1], borrowed_from=donors[0][0])

w(u"")
w(u"=========== BL. 亮度目标：第三步 vs 第四步 ===========")
w(u"  第三步全局常数下：通用 92/90/89，谐波 34/34/36/36，相差 2.6 倍")
w(u"")
w(u"%-30s %9s %7s %12s %10s %14s" % (
    u"场次", u"模式", u"帧数", u"目标灰阶", u"标准差", u"四分位"))
targets = {}
for key in sorted(groups):
    if key not in floors_new:
        continue
    r = T.measure_accepted_brightness(
        groups[key], lambda cap, k=key: render_own(cap, k),
        lambda cap, k=key: T.console_tissue_mask(db_of(cap, k), floors_new[k]["floor_db"]))
    if r is None:
        continue
    targets[key] = r["target_gray"]
    w(u"%-30s %9s %7d %12.0f %10.1f %14s" % (
        key[0], MODE[key[1]], r["num_frames"], r["target_gray"], r["std_gray"],
        u"%.0f–%.0f" % r["quartiles_gray"]))
g = [v for k, v in targets.items() if k[1] == 0]
h = [v for k, v in targets.items() if k[1] == 1]
w(u"")
w(u"  通用均值 %.0f 灰阶，谐波均值 %.0f 灰阶，相差 %.2f 倍（第三步是 2.6 倍）"
  % (np.mean(g), np.mean(h), max(np.mean(g), np.mean(h)) / min(np.mean(g), np.mean(h))))

w(u"")
w(u"=========== BM. 用完整标定重跑求解器 ===========")
w(u"%-22s %7s %6s %10s %8s %7s %11s %9s" % (
    u"帧", u"深度", u"模式", u"增益dB", u"滑块首档", u"缩放", u"Δ增益(档)", u"顶边界"))
deltas = {0: [], 1: []}
n_edge = n = 0
for key in sorted(groups):
    if key not in targets:
        continue
    cal = cals[key]
    for c in groups[key][:3]:
        d = db_of(c, key)
        resp = CAL.depth_response_for(c, cal)
        d = d + resp[:, None]
        vm = T.console_tissue_mask(d, floors_new[key]["floor_db"] )
        r = BS.solve_backend(
            d, vm, dr_ui=c.dynamic_range_level, reference_db=cal.pivot_db,
            target_gray=targets[key],
            current=(S.gain_level_to_db(c.gain_level),
                     np.asarray(c.tgc_levels, dtype=np.float64),
                     float(c.dynamic_range_level)))
        n += 1; n_edge += r["at_gain_edge"]
        deltas[key[1]].append(r["delta_gain_levels"])
        w(u"%-22s %7.1f %6s %10.2f %8d %7.2f %11.1f %9s" % (
            c.name[:22], c.geometry.depth_mm, MODE[key[1]][:2], r["gain_db"],
            r["tgc_levels"][0], r["shape_scale"], r["delta_gain_levels"],
            u"是" if r["at_gain_edge"] else u"否"))
w(u"  %d 帧中增益顶边界 %d 帧" % (n, n_edge))
for m in (0, 1):
    if deltas[m]:
        a = np.array(deltas[m])
        w(u"  %s Δ增益：中位 %+.1f 档，绝对值中位 %.1f 档，范围 %+.1f…%+.1f"
          % (MODE[m], np.median(a), np.median(np.abs(a)), a.min(), a.max()))

with open(os.path.join(HERE, "targets.pkl"), "wb") as fh:
    pickle.dump({"targets": targets,
                 "floors": {k: v["floor_db"] for k, v in floors_new.items()}}, fh)
io.open(os.path.join(HERE, "s4_recheck.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
