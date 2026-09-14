# -*- coding: utf-8 -*-
"""频率标签规则的物理前提：实机上发射频率越高，点靶是否真的越锐利。

    为什么要测

前端频率标签的规则是「在穿透仍能覆盖显示深度的频率里，选最高的那个」。它隐含一个
前提：频率越高分辨率越好。这个前提只在 Field II 上验证过（侧向半高全宽 4->8 MHz
从 1.076 降到 0.599 mm）。

2026-09-14 在实机 E8 上用散斑宽度测，结果与前提相反：基波 5->11.4 MHz 轴向散斑宽度
0.165->0.257 mm（+55.5%，理想 1/f 应为 -56.1%），侧向 -9.1%；谐波侧向 +6.9%。
但那次测量有两个弱点，本脚本逐一排除：

  一、【采样太粗】BC0 永远是 870 个轴向采样点，与显示深度无关，所以轴向间隔 =
     显示深度 / 870。E8 在 67 mm 下是 0.0770 mm，恰好是最粗的一档。散斑宽度只有
     2-3 个采样点，高频的变化可能被采样吃掉甚至反转。
     对照：同样的频率扫描在 25.1 mm（0.0289 mm）与 41.9 mm（0.0481 mm）下也采过。
     若轴向变差只出现在粗采样下，就是假象；三档采样都变差，才是真的。

  二、【散斑不是点扩散函数】散斑宽度受斑点统计、深度带内平均、以及点靶截断的影响。
     点靶的半高全宽才是分辨率的直接定义。

    点靶怎么挑

体模里有成排成列的尼龙线靶。在 BC0 dB 图上找比周围 3 mm 中位数高出
PIN_EXCESS_DB 的局部极大，再剔除三类假靶：

  近场        深度 < NEAR_FIELD_MM：探头表面振铃，一排排超出背景 15-18 dB
  次级回波    同一横向位置、在更强的靶正下方 SECONDARY_MM 以内：同一根线的混响，
              实测每根靶下方约 1 mm 跟着一个低 14 dB 左右的回波
  过弱        在比较集的任一频率下超出背景不足 PIN_EXCESS_DB：高频穿透不够时深处的
              靶会沉入噪声，那时量到的是噪声而不是点扩散函数

同一根靶必须在比较集的每个频率下都量到，才参与比较——否则不同频率比的是不同的靶。

    半高全宽怎么量

BC0 是 dB，-6 dB 即半幅。沿过峰值的行（侧向）与列（轴向）取剖面，峰值用抛物线插值
修正（离散网格上峰值可能偏半个采样点，峰值低估会把 -6 dB 线放低、宽度高估），两侧
交点线性插值。

    饱和

BC0 是 float32、未见统一上限，但仍逐靶检查峰顶是否平台化（连续多个采样点贴着峰值）。

用法：python tests/measure_point_target_resolution.py
"""

import collections
import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np
from scipy import ndimage

import calibration as CAL
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

# 比较集：同一场景族（探头不动）、同一显示深度、同一聚焦，只有频率变化。
# 每种模式三档显示深度，即三档轴向采样密度。
COMPARISONS = [
    ("fundamental", "20260903_GEN", 25.1, 15.0),
    ("fundamental", "20260903_GEN", 41.9, 15.0),
    ("fundamental", "20260911_E8_GEN", 67.0, 15.0),
    ("harmonic", "20260903", 25.1, 15.0),
    ("harmonic", "20260903", 41.9, 15.0),
    ("harmonic", "20260911_E8_THI", 67.0, 15.0),
]

COUNTS_PER_DB = 560.0         # 只影响 dB 刻度的缩放；超出背景与 -6 dB 都是比值，与它无关
PIN_EXCESS_DB = 25.0
NEAR_FIELD_MM = 3.0
SECONDARY_MM = 2.0
SECONDARY_LATERAL_MM = 0.6
MATCH_RADIUS_MM = 0.5
FWHM_DROP_DB = 6.0
PLATEAU_DB = 0.05             # 贴着峰值这么近的采样点算平台
PLATEAU_SAMPLES = 3


def depth_axis(geometry):
    return geometry.min_depth_mm + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point


def lateral_axis(geometry):
    return (np.arange(geometry.num_lines) - (geometry.num_lines - 1) / 2.0) * geometry.mm_per_line


def excess_map(db, geometry):
    """每个像素比周围 3 mm 中位数高出多少 dB。"""
    window = (max(3, int(3.0 / geometry.mm_per_point)), max(3, int(3.0 / geometry.mm_per_line)))
    return db - ndimage.median_filter(db, size=window)


def detect_pins(db, geometry):
    """点靶的 (行, 列)，已剔除近场与次级回波。"""
    excess = excess_map(db, geometry)
    neighbourhood = (max(3, int(1.0 / geometry.mm_per_point)),
                     max(3, int(1.0 / geometry.mm_per_line)))
    peaks = np.argwhere((db == ndimage.maximum_filter(db, size=neighbourhood))
                        & (excess > PIN_EXCESS_DB))
    z, x = depth_axis(geometry), lateral_axis(geometry)
    candidates = [(int(r), int(c), float(excess[r, c])) for r, c in peaks if z[r] >= NEAR_FIELD_MM]
    candidates.sort(key=lambda t: -t[2])
    kept = []
    for r, c, e in candidates:
        secondary = any(abs(x[c] - x[kc]) <= SECONDARY_LATERAL_MM
                        and 0.0 < z[r] - z[kr] <= SECONDARY_MM for kr, kc, _ in kept)
        if not secondary:
            kept.append((r, c, e))
    return kept


def refine_peak(profile, index):
    """抛物线插值的峰值。返回 (峰值 dB, 亚采样位置)。"""
    if index <= 0 or index >= profile.size - 1:
        return float(profile[index]), float(index)
    a, b, c = profile[index - 1], profile[index], profile[index + 1]
    denominator = a - 2.0 * b + c
    if denominator >= 0:
        return float(b), float(index)
    offset = 0.5 * (a - c) / denominator
    return float(b - 0.25 * (a - c) * offset), float(index + offset)


def fwhm(profile, index, spacing):
    """-6 dB 全宽，单位 mm。任一侧找不到交点返回 nan。"""
    peak, _ = refine_peak(profile, index)
    level = peak - FWHM_DROP_DB
    left = right = None
    for k in range(index, 0, -1):
        if profile[k - 1] < level <= profile[k]:
            left = (k - 1) + (level - profile[k - 1]) / (profile[k] - profile[k - 1])
            break
    for k in range(index, profile.size - 1):
        if profile[k + 1] < level <= profile[k]:
            right = k + (profile[k] - level) / (profile[k] - profile[k + 1])
            break
    if left is None or right is None:
        return float("nan")
    return float((right - left) * spacing)


def plateau(profile, index):
    """峰顶贴着峰值的连续采样点数。"""
    top = profile[index]
    count, k = 1, index - 1
    while k >= 0 and top - profile[k] <= PLATEAU_DB:
        count, k = count + 1, k - 1
    k = index + 1
    while k < profile.size and top - profile[k] <= PLATEAU_DB:
        count, k = count + 1, k + 1
    return count


def measure_pin(db, excess, geometry, row, col):
    """在 (row, col) 附近重新找峰，量侧向与轴向半高全宽。"""
    half_rows = max(1, int(MATCH_RADIUS_MM / geometry.mm_per_point))
    half_cols = max(1, int(MATCH_RADIUS_MM / geometry.mm_per_line))
    r0, r1 = max(0, row - half_rows), min(geometry.num_points, row + half_rows + 1)
    c0, c1 = max(0, col - half_cols), min(geometry.num_lines, col + half_cols + 1)
    local = np.unravel_index(np.argmax(db[r0:r1, c0:c1]), (r1 - r0, c1 - c0))
    r, c = r0 + int(local[0]), c0 + int(local[1])
    if excess[r, c] < PIN_EXCESS_DB:
        return None
    return {
        "lateral": fwhm(db[r, :], c, geometry.mm_per_line),
        "axial": fwhm(db[:, c], r, geometry.mm_per_point),
        "plateau": max(plateau(db[r, :], c), plateau(db[:, c], r)),
        "excess": float(excess[r, c]),
    }


MODE_CODES = {"fundamental": 0, "harmonic": 1}


def load_set(mode, session, depth, focus):
    """{频率: 帧}。同一频率多帧时取第一帧（同设置重复帧 BC0 几乎一致）。

    必须按成像模式筛。第一版没筛，20260903 场次在 41.9 mm 恰好有一帧 8.0 MHz 的基波，
    被当成谐波组的「最高频率」，把谐波 4.4 MHz 与基波 8.0 MHz 放在一起比，得出侧向
    +48.5%、轴向 +22.4% 的假结果。去掉那一帧后谐波轴向是 -19.4%。
    """
    frames = {}
    for path in sorted(find_captures(DEFAULT_DATA_DIR / session)):
        capture = load_capture(path)
        if (S.capture_image_mode(capture) == MODE_CODES[mode]
                and abs(capture.geometry.depth_mm - depth) < 0.5 and capture.focus_mm == focus):
            frames.setdefault(round(CAL.capture_frequency(capture), 2), capture)
    return frames


def analyse(mode, session, depth, focus, emit):
    frames = load_set(mode, session, depth, focus)
    freqs = sorted(frames)
    reference = frames[freqs[len(freqs) // 2]]
    geometry = reference.geometry
    db_ref = S.bc0_to_db(reference.bc0, COUNTS_PER_DB)
    pins = detect_pins(db_ref, geometry)
    z, x = depth_axis(geometry), lateral_axis(geometry)

    prepared = {}
    for f in freqs:
        db = S.bc0_to_db(frames[f].bc0, COUNTS_PER_DB)
        prepared[f] = (db, excess_map(db, frames[f].geometry))

    table = []
    clipped = 0
    for row, col, _ in pins:
        per_freq = {}
        for f in freqs:
            db, excess = prepared[f]
            m = measure_pin(db, excess, frames[f].geometry, row, col)
            if m is None or not (np.isfinite(m["lateral"]) and np.isfinite(m["axial"])):
                per_freq = None
                break
            if m["plateau"] >= PLATEAU_SAMPLES:
                clipped += 1
            per_freq[f] = m
        if per_freq:
            table.append((z[row], x[col], per_freq))
    table.sort(key=lambda t: t[0])

    emit(u"")
    emit(u"--- %s  %s  display %.1f mm  focus %.0f mm  axial %.4f mm/sample  lateral %.4f mm/line ---"
         % (mode, session, depth, focus, geometry.mm_per_point, geometry.mm_per_line))
    emit(u"  pins detected %d, measurable at every frequency %d, flat-topped peaks %d"
         % (len(pins), len(table), clipped))
    if not table:
        return None
    for label, key in [(u"lateral", "lateral"), (u"axial", "axial")]:
        emit(u"  %s FWHM (mm)" % label)
        emit(u"  %-17s %s" % (u"pin depth,lateral", u" ".join(u"%8s" % (u"%g MHz" % f) for f in freqs)))
        for depth_mm, lateral_mm, per_freq in table:
            emit(u"  %6.1f %+7.1f mm   %s" % (depth_mm, lateral_mm,
                                            u" ".join(u"%8.3f" % per_freq[f][key] for f in freqs)))
        means = [np.mean([p[f][key] for _, _, p in table]) for f in freqs]
        # 逐靶的相对变化再取中位，避免少数宽靶主导均值。
        change = np.median([p[freqs[-1]][key] / p[freqs[0]][key] - 1.0 for _, _, p in table])
        emit(u"  %-17s %s   per-pin median change %+.1f%%"
             % (u"mean", u" ".join(u"%8.3f" % v for v in means), 100.0 * change))
    # 按发射聚焦把靶分成聚焦前、聚焦后两组分别统计。2026-09-14 发现基波 10/11.4 MHz
    # 的轴向变粗只出现在深处：41.9 mm 显示时 <=10 mm 的靶随频率平滑变锐，>=20 mm 的
    # 5 根靶在 8->10 MHz 全部跳粗 17-43%。把所有靶混在一起取中位数会掩盖这件事。
    for label, chosen in [(u"before focus", [t_ for t_ in table if t_[0] < focus]),
                          (u"beyond focus", [t_ for t_ in table if t_[0] >= focus])]:
        if not chosen:
            continue
        ax = [np.mean([p[f]["axial"] for _, _, p in chosen]) for f in freqs]
        la = [np.mean([p[f]["lateral"] for _, _, p in chosen]) for f in freqs]
        emit(u"  %-13s %2d pins  axial %s   lateral %s"
             % (label, len(chosen), u" ".join(u"%.3f" % v for v in ax),
                u" ".join(u"%.3f" % v for v in la)))
        emit(u"  %-13s          sharpest axial at %g MHz, sharpest lateral at %g MHz"
             % (u"", freqs[int(np.argmin(ax))], freqs[int(np.argmin(la))]))
    overall = [np.mean([p[f]["axial"] for _, _, p in table]) for f in freqs]
    emit(u"  whole image: sharpest axial at %g MHz (mean over every measured pin)"
         % freqs[int(np.argmin(overall))])
    return {
        "mode": mode, "depth": depth, "spacing": geometry.mm_per_point, "freqs": freqs,
        "best_axial": freqs[int(np.argmin(overall))],
        "pins": len(table),
        "lateral": float(np.median([p[freqs[-1]]["lateral"] / p[freqs[0]]["lateral"] - 1.0
                                   for _, _, p in table])),
        "axial": float(np.median([p[freqs[-1]]["axial"] / p[freqs[0]]["axial"] - 1.0
                                 for _, _, p in table])),
        "axial_samples": float(np.mean([p[freqs[-1]]["axial"] for _, _, p in table])
                               / geometry.mm_per_point),
    }


def main():
    lines = []
    emit = lines.append
    emit(u"=========== point-target FWHM versus transmit frequency ===========")
    results = [r for r in (analyse(*c, emit=emit) for c in COMPARISONS) if r]

    emit(u"")
    emit(u"=========== summary: lowest -> highest frequency, per-pin median change ===========")
    emit(u"  negative = sharper at high frequency (what the frequency rule assumes)")
    emit(u"")
    emit(u"%-12s %8s %12s %6s %10s %10s %18s %12s" % (u"mode", u"display", u"axial mm/smp", u"pins",
                                                u"lateral", u"axial", u"axial FWHM samples",
                                                u"best axial"))
    for r in results:
        emit(u"%-12s %8.1f %12.4f %6d %+9.1f%% %+9.1f%% %18.1f %9g MHz"
             % (r["mode"], r["depth"], r["spacing"], r["pins"], 100 * r["lateral"],
                100 * r["axial"], r["axial_samples"], r["best_axial"]))
    emit(u"")
    emit(u"  ideal 1/f scaling: fundamental 5.0->11.4 MHz %+.1f%%, harmonic 4.4->5.7 MHz %+.1f%%"
         % (100 * (5.0 / 11.4 - 1), 100 * (4.4 / 5.7 - 1)))
    emit(u"  If the axial trend changes sign as sampling gets coarser, the E8 speckle result")
    emit(u"  was a sampling artefact. If it holds at 0.0289 mm, it is real.")

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_point_target_resolution.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
