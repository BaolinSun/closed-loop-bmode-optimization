# -*- coding: utf-8 -*-
"""仿真器误差是常数偏置，还是随设置变化？后者才会移动最优点。"""
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
import objective as OBJ
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append


def analyse(session, label):
    caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
    cal = CAL.fit_group(caps, limit=len(caps))
    floor_r = T.measure_noise_floor(caps, lambda c: S.bc0_to_db(c.bc0, cal.counts_per_db))
    # 这些场次都不够深，噪声底借用同模式的谐波值（按本组 counts_per_db 换算量级）
    floor = floor_r["floor_db"] if floor_r else np.percentile(
        S.bc0_to_db(caps[0].bc0, cal.counts_per_db), 5)
    w(u"")
    w(u"=========== %s ===========" % label)
    w(u"  counts/dB %.1f  pivot %.2f  拟合灰阶误差 %.2f  噪声底 %.2f"
      % (cal.counts_per_db, cal.pivot_db, cal.gray_error, floor))
    w(u"")
    w(u"%-22s %7s %8s %8s %10s %10s %10s %12s" % (
        u"帧", u"增益档", u"滑块首档", u"动态范围", u"J(截图)", u"J(重建)", u"差", u"逐像素误差"))
    rows = []
    for cap in sorted(caps, key=lambda c: (c.gain_level, c.tgc_levels[0],
                                           c.dynamic_range_level, c.name)):
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
        target = 40.0   # 只比较排序，绝对目标不影响两者之差
        a = OBJ.backend_objective(shot, shaped, vm, ~vm, target_gray=target)
        b = OBJ.backend_objective(reb, shaped, vm, ~vm, target_gray=target)
        rows.append((cap, a, b))
        w(u"%-22s %7d %8d %8d %10.4f %10.4f %10.4f %12.1f" % (
            cap.name[:22], cap.gain_level, cap.tgc_levels[0], cap.dynamic_range_level,
            a, b, b - a, np.median(np.abs(shot - reb))))
    if len(rows) < 3:
        return
    d = np.array([r[2] - r[1] for r in rows])
    w(u"")
    w(u"  误差 J(重建)-J(截图)：均值 %+.4f，标准差 %.4f" % (d.mean(), d.std()))
    w(u"  → 均值是常数偏置（不影响 argmin），标准差才是会移动最优点的部分")
    a = np.array([r[1] for r in rows]); b = np.array([r[2] for r in rows])
    w(u"  J(截图) 跨设置的跨度 %.4f，J(重建) 跨度 %.4f" % (a.max() - a.min(), b.max() - b.min()))
    w(u"  偏置去掉后的残差标准差 / J 跨度 = %.1f%%"
      % (100 * d.std() / max(a.max() - a.min(), 1e-9)))
    order_a = np.argsort(a); order_b = np.argsort(b)
    w(u"")
    w(u"  按 J 从小到大排序（0 = 最优）：")
    w(u"    截图  %s" % " ".join("%d" % i for i in np.argsort(order_a)))
    w(u"    重建  %s" % " ".join("%d" % i for i in np.argsort(order_b)))
    w(u"  最优设置是否一致：%s（截图选第 %d 帧，重建选第 %d 帧）"
      % (u"是" if order_a[0] == order_b[0] else u"否", order_a[0], order_b[0]))
    rho = np.corrcoef(np.argsort(order_a), np.argsort(order_b))[0, 1]
    w(u"  排序相关系数 %.3f" % rho)


analyse("20260901_E3", u"20260901_E3：增益扫描（59/75/91/105 档）+ 滑块 6/127/242")
analyse("20260904_DR", u"20260904_DR：增益 75/115 + 滑块 3/69/127/176/254 + 动态范围 30–82")
analyse("20260819", u"20260819：增益 75/125 + 滑块全档扫描")

io.open(os.path.join(HERE, "s6_bias.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
