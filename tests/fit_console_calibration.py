# -*- coding: utf-8 -*-
"""第四步：逐组联合标定（含深度响应），并用真实截图检验。"""
import io, os, sys, time, pickle
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
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

w(u"=========== BG. 逐组联合标定（两个标量 + 深度响应交替求解）===========")
w(u"%-30s %9s %7s %12s %11s %14s %14s" % (
    u"场次", u"模式", u"拟合帧", u"counts/dB", u"pivot_dB", u"深度响应跨度dB", u"拟合灰阶误差"))
cals = {}
t0 = time.time()
for key in sorted(groups):
    cal = CAL.fit_group(groups[key])
    if cal is None:
        continue
    cals[key] = cal
    rng = cal.depth_response_db.max() - cal.depth_response_db.min()
    w(u"%-30s %9s %7d %12.1f %11.2f %14.1f %14.2f" % (
        key[0], MODE[key[1]], cal.num_fit_frames, cal.counts_per_db,
        cal.pivot_db, rng, cal.gray_error))
w(u"  耗时 %.0f 秒" % (time.time() - t0))

w(u"")
w(u"=========== BH. 同模式跨场次一致性 ===========")
for m in (0, 1):
    same = [(k[0], v) for k, v in sorted(cals.items()) if k[1] == m]
    if len(same) < 2:
        continue
    cs = [v.counts_per_db for _, v in same]
    ps = [v.pivot_db for _, v in same]
    w(u"  %s" % MODE[m])
    w(u"    counts/dB  %s → 极差 %.1f（%.1f%%）"
      % ("  ".join("%.0f" % v.counts_per_db for _, v in same),
         max(cs) - min(cs), 100 * (max(cs) - min(cs)) / np.mean(cs)))
    w(u"    pivot_dB   %s → 极差 %.2f"
      % ("  ".join("%.1f" % v.pivot_db for _, v in same), max(ps) - min(ps)))
    w(u"    场次        %s" % "  ".join(n[:12] for n, _ in same))
gen = [v.counts_per_db for k, v in cals.items() if k[1] == 0]
har = [v.counts_per_db for k, v in cals.items() if k[1] == 1]
w(u"")
w(u"  通用均值 %.0f vs 谐波均值 %.0f，相差 %.0f%%"
  % (np.mean(gen), np.mean(har), 100 * (np.mean(har) / np.mean(gen) - 1)))

w(u"")
w(u"=========== BI. 检验：预测真实截图的逐带灰阶误差（全部帧，含非拟合帧）===========")
w(u"%-30s %9s %7s %16s %18s %10s" % (
    u"场次", u"模式", u"帧数", u"全局默认常数", u"组自身+深度响应", u"改善"))
for key in sorted(cals):
    cal = cals[key]
    frames = [c for c in groups[key] if S.is_flat_tgc(c)]
    old = np.mean([CAL.screenshot_gray_error(c, S.DEFAULT_COUNTS_PER_DB, S.DEFAULT_PIVOT_DB)
                   for c in frames])
    new = np.mean([CAL.screenshot_gray_error(c, cal.counts_per_db, cal.pivot_db,
                                             CAL.depth_response_for(c, cal))
                   for c in frames])
    w(u"%-30s %9s %7d %16.2f %18.2f %9.0f%%" % (
        key[0], MODE[key[1]], len(frames), old, new, 100 * (1 - new / old)))

w(u"")
w(u"=========== BJ. 深度响应曲线长什么样 ===========")
w(u"%-30s %9s %s" % (u"场次", u"模式", u"每 5 mm 的 C(z) dB"))
for key in sorted(cals):
    cal = cals[key]
    pts = np.arange(0, 68, 5.0)
    vals = np.interp(pts, cal.depth_axis_mm, cal.depth_response_db)
    w(u"%-30s %9s %s" % (key[0], MODE[key[1]],
                         " ".join("%6.1f" % v for v in vals)))
w(u"  深度 mm                              %s"
  % " ".join("%6.0f" % p for p in np.arange(0, 68, 5.0)))

with open(os.path.join(HERE, "cals.pkl"), "wb") as fh:
    pickle.dump({k: (v.counts_per_db, v.pivot_db, v.depth_axis_mm, v.depth_response_db,
                     v.gray_error) for k, v in cals.items()}, fh)
io.open(os.path.join(HERE, "s4_fit3.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
