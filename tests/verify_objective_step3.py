# -*- coding: utf-8 -*-
"""第三步最终验证：加入亮度锚点后的行为。"""
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
SESS = ["20260903", "20260903_GEN", "20260903_replication_check", "20260904", "20260904_DR"]
MODE = {0: u"通用GEN", 1: u"谐波THI"}

groups = {}
for sess in SESS:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)
floors = T.measure_session_noise_floors(groups, lambda cap: S.bc0_to_db(cap.bc0))


def mask_of(cap, key):
    return T.console_tissue_mask(S.bc0_to_db(cap.bc0), floors[key]["floor_db"])


def render_own(cap):
    return S.render(db_image=S.bc0_to_db(cap.bc0), tgc_levels=cap.tgc_levels,
                    gain_db=S.gain_level_to_db(cap.gain_level),
                    dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                    reference_db=S.DEFAULT_PIVOT_DB, depth_response_db=None)


w(u"=========== AW. 各组「操作员接受的组织灰阶」 ===========")
w(u"%-30s %9s %7s %12s %9s %14s" % (
    u"场次", u"模式", u"帧数", u"目标灰阶", u"标准差", u"四分位"))
targets = {}
for key in sorted(groups):
    if floors.get(key) is None:
        continue
    r = T.measure_accepted_brightness(groups[key], render_own,
                                      lambda cap, k=key: mask_of(cap, k))
    if r is None:
        continue
    targets[key] = r["target_gray"]
    w(u"%-30s %9s %7d %12.0f %9.1f %14s" % (
        key[0], MODE[key[1]], r["num_frames"], r["target_gray"], r["std_gray"],
        u"%.0f–%.0f" % r["quartiles_gray"]))
w(u"")
w(u"  ⚠ 谐波与通用相差很大。这不能全归给操作员：DEFAULT_PIVOT_DB=25.09 和")
w(u"    counts_per_db=877.3 是在一个场次上标定的，跨模式套用会平移渲染灰阶。")
w(u"    该差值要等第四步逐场次标定后才能定性。")

w(u"")
w(u"=========== AX. 加锚点后，实机解还合理吗 ===========")
w(u"%-22s %7s %7s %10s %8s %7s %11s %10s" % (
    u"帧", u"深度", u"模式", u"增益dB", u"滑块首档", u"缩放", u"Δ增益(档)", u"顶边界"))
n_edge = n = 0
for key in sorted(groups):
    if key not in targets:
        continue
    for c in groups[key][:3]:
        d = S.bc0_to_db(c.bc0)
        vm = mask_of(c, key)
        r = BS.solve_backend(
            d, vm, dr_ui=c.dynamic_range_level, reference_db=S.DEFAULT_PIVOT_DB,
            target_gray=targets[key],
            current=(S.gain_level_to_db(c.gain_level),
                     np.asarray(c.tgc_levels, dtype=np.float64),
                     float(c.dynamic_range_level)))
        n += 1; n_edge += r["at_gain_edge"]
        w(u"%-22s %7.1f %7s %10.2f %8d %7.2f %11.1f %10s" % (
            c.name[:22], c.geometry.depth_mm, MODE[key[1]][:2], r["gain_db"],
            r["tgc_levels"][0], r["shape_scale"], r["delta_gain_levels"],
            u"是" if r["at_gain_edge"] else u"否"))
w(u"  %d 帧中增益顶边界 %d 帧" % (n, n_edge))

w(u"")
w(u"=========== AY. 滑块缩放：加锚点前后 ===========")
w(u"  加锚点前（AT）：12 帧 Field II 全部缩放 1.00（另一端顶死）")
gen_target = targets.get(("20260903_GEN", 0))
scales_a, scales_b, edges = [], [], 0
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        vm = T.fieldii_tissue_mask(c)
        a = BS.solve_backend(c.db_image, vm, 67, reference_db=c.tissue_median_db)
        b = BS.solve_backend(c.db_image, vm, 67, reference_db=c.tissue_median_db,
                             target_gray=gen_target)
        scales_a.append(a["shape_scale"]); scales_b.append(b["shape_scale"])
        edges += b["at_gain_edge"]
w(u"  无锚点：缩放取值 %s；增益顶边界 %d/12"
  % (sorted(set(scales_a)), sum(1 for x in [1] if False) or
     sum(BS.solve_backend(load_shard(find_shards(phantom_type=p, depth_mm=d,
         frequency_mhz=f, focus_mm=15.0)[0]).db_image,
         T.fieldii_tissue_mask(load_shard(find_shards(phantom_type=p, depth_mm=d,
         frequency_mhz=f, focus_mm=15.0)[0])), 67,
         reference_db=load_shard(find_shards(phantom_type=p, depth_mm=d,
         frequency_mhz=f, focus_mm=15.0)[0]).tissue_median_db)["at_gain_edge"]
         for p in ["uniform", "cyst", "point"]
         for d, f in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)])))
w(u"  有锚点（借用通用模式目标 %.0f 灰阶）：缩放取值 %s；增益顶边界 %d/12"
  % (gen_target, sorted(set(scales_b)), edges))

w(u"")
w(u"=========== AZ. 快慢路径一致性（含亮度项）与速度 ===========")
c = load_shard(find_shards(phantom_type="cyst", depth_mm=42.0,
                           frequency_mhz=5.0, focus_mm=15.0)[0])
db, pivot, vm = c.db_image, c.tissue_median_db, T.fieldii_tissue_mask(c)
levels, _ = BS.solve_tgc_shape(db, valid_mask=vm)
sweep = BS.GainSweep(db, levels, vm, ~vm, target_gray=gen_target)
shaped = S.apply_tgc(db, levels)
worst = 0.0
for dr in [30, 67, 400]:
    for gain in np.arange(-8.0, 24.01, 2.0):
        gray = S.render(db_image=db, tgc_levels=levels, gain_db=gain,
                        dynamic_range_db=S.dr_ui_to_window_db(dr),
                        reference_db=pivot, depth_response_db=None)
        slow = OBJ.backend_objective(gray, shaped, vm, ~vm, target_gray=gen_target)
        fast = sweep.evaluate(gain, S.dr_ui_to_window_db(dr), pivot)
        worst = max(worst, abs(slow - fast))
w(u"  最大代价差 %.3e" % worst)
t0 = time.time(); BS.solve_backend(db, vm, 67, reference_db=pivot, target_gray=gen_target,
                                   fast=False); ts = time.time() - t0
t0 = time.time(); BS.solve_backend(db, vm, 67, reference_db=pivot, target_gray=gen_target,
                                   fast=True); tf = time.time() - t0
w(u"  逐帧渲染 %.2f 秒 / 预排序 %.2f 秒，提速 %.0f 倍；4686 帧 %.1f 小时 → %.1f 分钟"
  % (ts, tf, ts / max(tf, 1e-9), ts * 4686 / 3600, tf * 4686 / 60))

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_final.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
