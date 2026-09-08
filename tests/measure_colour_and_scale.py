# -*- coding: utf-8 -*-
"""两个剩余嫌疑：截图不是纯灰、以及重采样比例有偏。"""
import io, json, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
from PIL import Image
import hisense_backend_sim as S
import calibration as CAL
import tissue as T
from hisense_loader import (DEFAULT_DATA_DIR, find_captures, load_capture,
                            crop_capture_image, load_screenshot, crop_image_area)

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append
cal_data = json.load(io.open("bmode_opt/console_calibration.json", encoding="utf-8"))
cals = {(g["session"], g["image_mode"]): CAL.GroupCalibration(
    g["counts_per_db"], g["pivot_db"], g["screenshot_gray_error"], 0,
    np.array(g["depth_axis_mm"]), np.array(g["depth_response_db"]))
    for g in cal_data["groups"]}

PICKS = [("20260903", 1), ("20260903_GEN", 0), ("20260904", 0),
         ("20260903_replication_check", 1)]

w(u"=========== DA. B 图区域是不是纯灰（load_screenshot 走的是亮度加权）===========")
w(u"%-28s %7s %14s %16s %16s" % (
    u"场次", u"模式", u"R=G=B 的比例", u"最大通道差", u"亮度转换偏差"))
picked = []
for sess, mode in PICKS:
    caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / sess)]
    caps = [c for c in caps if T.capture_image_mode(c) == mode]
    cap = sorted(caps, key=lambda c: (c.geometry.depth_mm, c.name))[len(caps) // 2]
    picked.append(((sess, mode), cap))
    rgb = np.asarray(Image.open(cap.path / "Screenthum.bmp").convert("RGB"), dtype=np.float64)
    lum = np.asarray(Image.open(cap.path / "Screenthum.bmp").convert("L"), dtype=np.float64)
    _, (r0, r1, c0, c1) = crop_capture_image(cap)
    sub = rgb[r0:r1, c0:c1]
    same = ((sub[..., 0] == sub[..., 1]) & (sub[..., 1] == sub[..., 2])).mean()
    chan = np.abs(sub.max(axis=2) - sub.min(axis=2)).max()
    dev = np.abs(lum[r0:r1, c0:c1] - sub.mean(axis=2)).max()
    w(u"%-28s %7s %13.1f%% %16.0f %16.1f" % (
        sess[:26], u"谐波" if mode else u"通用", 100 * same, chan, dev))

w(u"")
w(u"=========== DB. 重采样比例：小幅缩放能否降低逐像素误差 ===========")
w(u"  几何声明：BImageWidth/Height 与 BC0 的宽深比只差 0.15%%，但 766 行上")
w(u"  0.5%% 的比例误差在底部就是约 4 个像素的漂移，足以让深部斑点失配。")
w(u"")
w(u"%-28s %7s %10s %10s %10s %12s %12s" % (
    u"场次", u"模式", u"原误差", u"最佳缩放y", u"最佳缩放x", u"最佳平移", u"缩放后误差"))
for key, cap in picked:
    cal = cals[key]
    shot = crop_capture_image(cap)[0]
    db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
    full = S.render_db(db_image=db, tgc_levels=cap.tgc_levels,
                       gain_db=S.gain_level_to_db(cap.gain_level))
    H, W = shot.shape
    base = None; best = None
    for sy in [1.0, 1.005, 1.01, 1.02, 1.03, 0.995, 0.99, 0.98]:
        for sx in [1.0, 1.005, 1.01, 0.995, 0.99]:
            big = S.scan_convert_linear(full, int(round(H * sy)), int(round(W * sx)))
            gray = S.db_to_gray(big, S.dr_ui_to_window_db(cap.dynamic_range_level),
                                cal.pivot_db).astype(np.float64)
            for dy in (-4, -2, 0, 2, 4):
                for dx in (-2, 0, 2):
                    a = shot[max(0, dy):H + min(0, dy), max(0, dx):W + min(0, dx)]
                    b = gray[max(0, -dy):max(0, -dy) + a.shape[0],
                             max(0, -dx):max(0, -dx) + a.shape[1]]
                    if b.shape != a.shape:
                        continue
                    v = float(np.median(np.abs(a - b)))
                    if sy == 1.0 and sx == 1.0 and dy == 0 and dx == 0:
                        base = v
                    if best is None or v < best[0]:
                        best = (v, sy, sx, (dy, dx))
    w(u"%-28s %7s %10.1f %10.3f %10.3f %12s %12.1f" % (
        key[0][:26], u"谐波" if key[1] else u"通用", base, best[1], best[2],
        str(best[3]), best[0]))

io.open(os.path.join(HERE, "s7_scale.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
