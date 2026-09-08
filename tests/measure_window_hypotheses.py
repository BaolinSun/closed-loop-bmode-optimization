# -*- coding: utf-8 -*-
"""窗口该覆盖什么：从噪声底到最亮结构，而不是组织的 p1–p99。"""
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

lines = []; w = lines.append

w(u"=========== AM. 实机：从噪声底到最亮结构，跨度是多少 ===========")
w(u"  假设：显示窗口应当从噪声底一直盖到最亮的结构。")
w(u"  临床颈动脉预设用的是 动态范围=67，窗宽 %.1f dB —— 看能不能对上。"
  % S.dr_ui_to_window_db(67))
w(u"")
groups = {}
for sess in ["20260903", "20260903_GEN", "20260904", "20260904_DR",
             "20260903_replication_check"]:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)
floors = T.measure_session_noise_floors(groups, lambda cap: S.bc0_to_db(cap.bc0))

w(u"%-9s %9s %7s %11s %12s %12s %14s %14s" % (
    u"模式", u"显示深度", u"帧数", u"噪声底dB", u"组织p99", u"全图p99.9", u"底→p99.9跨度",
    u"折合动态范围"))
rows = {}
for key, caps in groups.items():
    f = floors.get(key)
    if f is None:
        continue
    for c in caps:
        d = S.bc0_to_db(c.bc0)
        m = T.console_tissue_mask(d, f["floor_db"])
        if m.sum() < 1000:
            continue
        rows.setdefault((u"通用GEN" if key[1] == 0 else u"谐波THI",
                         round(c.geometry.depth_mm, 1)), []).append(
            (f["floor_db"], float(np.percentile(d[m], 99)),
             float(np.percentile(d, 99.9))))
for k in sorted(rows):
    v = np.array(rows[k])
    span = (v[:, 2] - v[:, 0]).mean()
    ui = (span - S.DR_WINDOW_INTERCEPT) / S.DR_WINDOW_SLOPE
    w(u"%-9s %9.1f %7d %11.2f %12.1f %12.1f %14.1f %14s" % (
        k[0], k[1], len(v), v[:, 0].mean(), v[:, 1].mean(), v[:, 2].mean(), span,
        u"低于30" if ui < 30 else u"%.0f" % ui))

w(u"")
w(u"=========== AN. Field II 同样口径 ===========")
w(u"%-9s %7s %7s %12s %12s %14s %14s" % (
    u"体模", u"深度", u"频率", u"组织p1", u"全图p99.9", u"p1→p99.9跨度", u"折合动态范围"))
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        m = T.fieldii_tissue_mask(c)
        lo = float(np.percentile(c.db_image[m], 1))
        hi = float(np.percentile(c.db_image, 99.9))
        ui = (hi - lo - S.DR_WINDOW_INTERCEPT) / S.DR_WINDOW_SLOPE
        w(u"%-9s %7.0f %7.1f %12.1f %12.1f %14.1f %14s" % (
            ptype, depth, freq, lo, hi, hi - lo,
            u"低于30" if ui < 30 else u"%.0f" % ui))

w(u"")
w(u"=========== AO. 数据里有没有低对比病灶（决定动态范围的唯一任务）===========")
cc = load_shard(find_shards(phantom_type="cyst", depth_mm=42.0,
                            frequency_mhz=5.0, focus_mm=15.0)[0])
tm = np.asarray(cc.truth_mask)
bgm = T.fieldii_tissue_mask(cc)
w(u"  生成代码 make_phantom() 对囊肿的处理是 amplitudes(inside)=0，即完全无回声。")
w(u"")
w(u"%8s %14s %16s %14s" % (u"囊肿编号", u"内部中位dB", u"同深度背景中位dB", u"对比dB"))
for idx in np.unique(tm):
    if idx == 0:
        continue
    inside = tm == idx
    rows_i = np.where(inside.any(axis=1))[0]
    bg = bgm.copy(); bg[:rows_i.min()] = False; bg[rows_i.max() + 1:] = False
    w(u"%8d %14.1f %16.1f %14.1f" % (
        idx, np.median(cc.db_image[inside]), np.median(cc.db_image[bg]),
        np.median(cc.db_image[bg]) - np.median(cc.db_image[inside])))
w(u"")
w(u"  → 对比都在 20 dB 以上。动态范围真正影响的是差几 dB 的低对比病灶，")
w(u"    这套数据里一个都没有。")

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_window.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
