# -*- coding: utf-8 -*-
"""重建为什么肉眼看着比截图差：分解成色调曲线、锐度、纹理三部分。"""
import io, json, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append
cal_data = json.load(io.open("bmode_opt/console_calibration.json", encoding="utf-8"))
cals = {(g["session"], g["image_mode"]): CAL.GroupCalibration(
    g["counts_per_db"], g["pivot_db"], g["screenshot_gray_error"], 0,
    np.array(g["depth_axis_mm"]), np.array(g["depth_response_db"])) for g in cal_data["groups"]}

PICKS = [("20260903", 1), ("20260903_GEN", 0), ("20260904", 0),
         ("20260903_replication_check", 1)]
pairs = []
for sess, mode in PICKS:
    caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / sess)]
    caps = [c for c in caps if T.capture_image_mode(c) == mode]
    caps = sorted(caps, key=lambda c: (c.geometry.depth_mm, c.name))
    for cap in [caps[len(caps) // 2]]:
        cal = cals[(sess, mode)]
        shot = crop_capture_image(cap)[0]
        db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
        rebuilt = S.render(db_image=db, tgc_levels=cap.tgc_levels,
                           gain_db=S.gain_level_to_db(cap.gain_level),
                           dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                           reference_db=cal.pivot_db, depth_response_db=None,
                           out_shape=shot.shape).astype(np.float64)
        pairs.append(((sess, mode), cap, shot, rebuilt))

w(u"=========== CA. 逐像素误差有多大（对比逐带中位的 1.65–6.87）===========")
w(u"%-30s %7s %12s %12s %12s %12s" % (
    u"场次", u"模式", u"逐带中位误差", u"逐像素平均", u"逐像素中位", u"90分位"))
for key, cap, shot, reb in pairs:
    d = np.abs(shot - reb)
    band = CAL.screenshot_gray_error(cap, cals[key].counts_per_db, cals[key].pivot_db,
                                     CAL.depth_response_for(cap, cals[key]))
    w(u"%-30s %7s %12.2f %12.1f %12.1f %12.1f" % (
        key[0], u"谐波" if key[1] else u"通用", band,
        d.mean(), np.median(d), np.percentile(d, 90)))
w(u"")
w(u"  → 逐带中位掩盖了真实误差：它先在带内取中位，把纹理差异全平均掉了。")

w(u"")
w(u"=========== CB. 色调曲线：截图灰阶 vs 重建灰阶 ===========")
w(u"  若两者只差一条单调曲线，说明缺的是显示响应，可以用查找表补上。")
w(u"")
key, cap, shot, reb = pairs[0]
w(u"%-30s %s" % (u"", u"重建灰阶落在下列区间时，截图的中位灰阶"))
w(u"%-30s %s" % (u"场次/模式", " ".join(u"%5d" % b for b in range(10, 250, 20))))
for key, cap, shot, reb in pairs:
    row = []
    for b in range(10, 250, 20):
        sel = (reb >= b - 10) & (reb < b + 10)
        row.append(np.median(shot[sel]) if sel.sum() > 200 else np.nan)
    w(u"%-30s %s" % (
        u"%s/%s" % (key[0][:18], u"谐波" if key[1] else u"通用"),
        " ".join(u"%5.0f" % v if np.isfinite(v) else u"    -" for v in row)))
w(u"%-30s %s" % (u"（若完全一致应等于）", " ".join(u"%5d" % b for b in range(10, 250, 20))))

w(u"")
w(u"=========== CC. 锐度与纹理 ===========")
w(u"  梯度能量 = 相邻像素差的均方根；斑点起伏 = 均匀区的标准差/均值")
w(u"")
w(u"%-30s %7s %14s %14s %10s %14s %14s" % (
    u"场次", u"模式", u"截图梯度", u"重建梯度", u"比值", u"截图斑点起伏", u"重建斑点起伏"))
for key, cap, shot, reb in pairs:
    def grad(x):
        dy = np.diff(x, axis=0)[:, :-1]
        dx = np.diff(x, axis=1)[:-1, :]
        return float(np.sqrt(np.mean(dy ** 2 + dx ** 2)))
    h, wd = shot.shape
    box = (slice(h // 3, 2 * h // 3), slice(wd // 3, 2 * wd // 3))
    def speck(x):
        r = x[box]
        return float(np.std(r) / max(np.mean(r), 1e-6))
    gs, gr = grad(shot), grad(reb)
    w(u"%-30s %7s %14.2f %14.2f %10.2f %14.3f %14.3f" % (
        key[0][:28], u"谐波" if key[1] else u"通用", gs, gr, gr / gs,
        speck(shot), speck(reb)))

io.open(os.path.join(HERE, "s5_diag.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
