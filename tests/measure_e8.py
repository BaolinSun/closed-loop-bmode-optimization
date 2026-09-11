# -*- coding: utf-8 -*-
"""E8 分析：组织电平与噪声底之差随发射频率怎么变。

    这批数据要回答什么

Field II 的频率标签定不下来，因为那边没有噪声底（noise_enabled 全是 0）。重新仿真
要注入噪声，而注入多少、随频率怎么变，必须以实机为准。协议 §5e 据此定了 E8。

采之前只有 12 帧够深能看到噪声底，多数频率点只有 1 帧，而且基波算出的斜率是
-0.25 dB/MHz——频率翻一倍多而间隔纹丝不动，物理上讲不通（按 0.5 dB/(cm*MHz) 估，
4 cm 深往返应多耗约 25 dB）。两点各一帧，判断不了真假。

E8 每个 (模式, 频率, 显示深度) 采 3 帧，两个显示深度，两种成像模式共 72 帧。

    三个问题

一、【最深处到底是不是噪声】噪声底与深度无关，所以深部电平随深度的斜率应当趋零。
   若仍在下降，那里量到的是信号，间隔就是假的——之前基波那个「平的间隔」就是
   这么来的（当时用 50 mm 门槛，而实测要到 67 mm 才平）。

二、【间隔随频率怎么变】谐波之前量出 -8.24 dB/MHz，基波 -0.25。基波那个数要么是
   真的（说明基波的噪声另有来源），要么是 1 帧的测量噪声。

三、【噪声底是否随显示深度变】67.0 与 75.4 两档给出交叉验证。显示深度会改变脉冲
   重复频率与线密度，噪声底跟着变是可能的，之前无从判断。

    counts_per_db 必须各场次自己定

它把 BC0 计数换算成 dB，而间隔是 dB 之差，所以整体按 1/counts_per_db 缩放。已有
场次里基波的 counts_per_db 只差 6.2%，谐波却差 44%，不能借用。这里从每帧自己的
截图拟合：调色板反解后的灰度对重采样 BC0 是仿射的，斜率给出 counts_per_db。

用 display_palette 而不是 PIL 亮度——主机经调色板显示，亮度会偏 3 到 16 个灰阶。

用法：python tests/measure_e8.py
"""

import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import re

import calibration as CAL
import display_palette as DP
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

SESSIONS = [("20260911_E8_GEN", 0), ("20260911_E8_THI", 1)]
MODE_NAMES = {0: "fundamental", 1: "harmonic"}

# 组织电平取这一段：浅到任何频率下都一定还有信号。
TISSUE_DEPTH_MM = (10.0, 20.0)
# 噪声底取最深这一段。
NOISE_TAIL_FRACTION = 0.08
# 检查平坦度用最深这一段。
FLATNESS_TAIL_FRACTION = 0.20


def fit_counts_per_db(capture, palette):
    """从一帧的截图拟合 counts_per_db。

    gray = GRAY_PIVOT + (bc0/counts_per_db + gain_db - pivot_db)/W*255
    对 bc0 是仿射的，斜率 a = 255/(W*counts_per_db)，所以 counts_per_db = 255/(W*a)。
    增益只进截距，不进斜率，所以谐波那几帧增益不同也不影响。
    """
    gray, _ = DP.capture_display_gray(capture, palette=palette)
    gray = np.asarray(gray, dtype=np.float64)
    counts = S.scan_convert_linear(capture.bc0, gray.shape[0], gray.shape[1])
    start, stop = 3 * gray.shape[0] // 8, 5 * gray.shape[0] // 8
    flat_gray = gray[start:stop].ravel()
    flat_counts = counts[start:stop].ravel()
    usable = (flat_gray > 5) & (flat_gray < 250)
    if usable.sum() < 100:
        return None
    slope = np.polyfit(flat_counts[usable], flat_gray[usable], 1)[0]
    if not np.isfinite(slope) or slope <= 0:
        return None
    return float(S.GRAY_MAX / (S.capture_window_db(capture) * slope))


def mechanical_index(capture):
    """主机报告的机械指数。发射声压除以频率的平方根，所以它跟着发射功率走。"""
    text = io.open(capture.path / "Algo_FeParam.pdt",
                   encoding="utf-8", errors="ignore").read()
    found = re.search(r"^\s*MI:(.*)$", text, re.M)
    return float(found.group(1)) if found else float("nan")


def penetration_mm(db_image, geometry, floor_db, margin_db=3.0):
    """组织仍高出噪声底 margin_db 的最深处。进入标签规则的正是这个量。"""
    rows = np.median(db_image, axis=1)
    good = np.where(rows >= floor_db + margin_db)[0]
    if good.size == 0:
        return float(geometry.min_depth_mm)
    return float(geometry.min_depth_mm + (good[-1] + 0.5) * geometry.mm_per_point)


def depth_axis(capture):
    geometry = capture.geometry
    return (geometry.min_depth_mm
            + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)


def main():
    lines = []
    emit = lines.append

    loaded = {}
    for session, mode in SESSIONS:
        captures = sorted((load_capture(p) for p in
                           find_captures(DEFAULT_DATA_DIR / session)),
                          key=lambda c: c.name)
        palette = DP.session_palette(captures)[0]
        loaded[session] = (mode, captures, palette)

    emit(u"=========== 0. counts per dB, fitted per session ===========")
    emit(u"  From each frame's own screenshot, read through the console palette. The gap")
    emit(u"  is a difference of dB values, so it scales with 1 over this number; the")
    emit(u"  existing harmonic sessions spread 44%%, which is too much to borrow.")
    emit(u"")
    emit(u"%-22s %-12s %7s %10s %10s %10s"
         % (u"session", u"mode", u"frames", u"median", u"min", u"max"))
    counts_per_db = {}
    for session, (mode, captures, palette) in loaded.items():
        values = [fit_counts_per_db(c, palette) for c in captures]
        values = np.array([v for v in values if v is not None])
        counts_per_db[session] = float(np.median(values))
        emit(u"%-22s %-12s %7d %10.1f %10.1f %10.1f"
             % (session, MODE_NAMES[mode], values.size, np.median(values),
                values.min(), values.max()))

    emit(u"")
    emit(u"=========== 1. is the deepest part really noise ===========")
    emit(u"  Slope of the level over the deepest %.0f%% of rows. A noise floor does not"
         % (100 * FLATNESS_TAIL_FRACTION))
    emit(u"  depend on depth, so this must approach zero. Still falling means signal,")
    emit(u"  and any gap measured there is fiction.")
    emit(u"")
    emit(u"%-12s %7s %7s %7s %12s %10s"
         % (u"mode", u"depth", u"freq", u"n", u"slope dB/mm", u"verdict"))
    for session, (mode, captures, palette) in loaded.items():
        rows = {}
        for capture in captures:
            db = S.bc0_to_db(capture.bc0, counts_per_db[session])
            depth = depth_axis(capture)
            tail = depth >= (capture.geometry.depth_mm
                             - FLATNESS_TAIL_FRACTION
                             * (capture.geometry.depth_mm - capture.geometry.min_depth_mm))
            profile = np.median(db[tail, :], axis=1)
            slope = float(np.polyfit(depth[tail], profile, 1)[0])
            key = (round(capture.geometry.depth_mm, 1),
                   round(CAL.capture_frequency(capture), 2))
            rows.setdefault(key, []).append(slope)
        for key in sorted(rows):
            values = np.array(rows[key])
            verdict = (u"noise" if abs(values.mean()) < 0.10
                       else (u"mixed" if abs(values.mean()) < 0.25 else u"SIGNAL"))
            emit(u"%-12s %7.1f %7.2f %7d %12.3f %10s"
                 % (MODE_NAMES[mode], key[0], key[1], values.size,
                    values.mean(), verdict))

    emit(u"")
    emit(u"=========== 2. the gap, by mode, depth and frequency ===========")
    emit(u"  tissue = median over %g-%g mm, noise = median over the deepest %.0f%%."
         % (TISSUE_DEPTH_MM[0], TISSUE_DEPTH_MM[1], 100 * NOISE_TAIL_FRACTION))
    emit(u"")
    emit(u"%-12s %7s %7s %5s %9s %9s %9s %8s"
         % (u"mode", u"depth", u"freq", u"n", u"tissue", u"noise", u"gap", u"sd"))
    table = {}
    for session, (mode, captures, palette) in loaded.items():
        rows = {}
        for capture in captures:
            db = S.bc0_to_db(capture.bc0, counts_per_db[session])
            depth = depth_axis(capture)
            band = ((depth >= TISSUE_DEPTH_MM[0]) & (depth <= TISSUE_DEPTH_MM[1]))
            tail = int(round(capture.geometry.num_points * (1 - NOISE_TAIL_FRACTION)))
            tissue = float(np.median(db[band, :]))
            noise = float(np.median(db[tail:, :]))
            key = (round(capture.geometry.depth_mm, 1),
                   round(CAL.capture_frequency(capture), 2))
            rows.setdefault(key, []).append((tissue, noise))
        for key in sorted(rows):
            values = np.array(rows[key])
            gaps = values[:, 0] - values[:, 1]
            table[(mode,) + key] = (float(values[:, 0].mean()),
                                    float(values[:, 1].mean()), float(gaps.mean()))
            emit(u"%-12s %7.1f %7.2f %5d %9.2f %9.2f %9.2f %8.2f"
                 % (MODE_NAMES[mode], key[0], key[1], values.shape[0],
                    values[:, 0].mean(), values[:, 1].mean(),
                    gaps.mean(), gaps.std()))

    emit(u"")
    emit(u"=========== 3. how fast the gap falls with frequency ===========")
    emit(u"  Before E8 the harmonic estimate was -8.24 dB per MHz from 3 frames, and the")
    emit(u"  fundamental -0.25 from 2 frames, which is physically implausible.")
    emit(u"")
    emit(u"%-12s %8s %10s %10s %14s"
         % (u"mode", u"depth", u"low freq", u"high freq", u"slope dB/MHz"))
    for mode in (0, 1):
        for display_depth in sorted({k[1] for k in table if k[0] == mode}):
            keys = sorted(k for k in table if k[0] == mode and k[1] == display_depth)
            if len(keys) < 2:
                continue
            frequencies = [k[2] for k in keys]
            gaps = [table[k][2] for k in keys]
            slope = float(np.polyfit(frequencies, gaps, 1)[0])
            emit(u"%-12s %8.1f %10.2f %10.2f %14.2f"
                 % (MODE_NAMES[mode], display_depth, gaps[0], gaps[-1], slope))

    emit(u"")
    emit(u"=========== 4. does the noise floor depend on the display depth ===========")
    emit(u"  Deeper settings change the pulse repetition frequency and the line density,")
    emit(u"  so the floor moving with them is possible and was never testable before.")
    emit(u"")
    emit(u"%-12s %8s %12s %12s %12s"
         % (u"mode", u"freq", u"floor 67mm", u"floor 75.4mm", u"difference"))
    for mode in (0, 1):
        differences = []
        for frequency in sorted({k[2] for k in table if k[0] == mode}):
            shallow = table.get((mode, 67.0, frequency))
            deep = table.get((mode, 75.4, frequency))
            if shallow is None or deep is None:
                continue
            differences.append(deep[1] - shallow[1])
            emit(u"%-12s %8.2f %12.2f %12.2f %12.2f"
                 % (MODE_NAMES[mode], frequency, shallow[1], deep[1],
                    deep[1] - shallow[1]))
        if differences:
            emit(u"  %-12s mean difference %.2f dB, spread %.2f"
                 % (MODE_NAMES[mode], float(np.mean(differences)),
                    float(np.max(differences) - np.min(differences))))

    emit(u"")
    emit(u"=========== 5. transmit power is not constant along the frequency axis ===========")
    emit(u"  The console reports a mechanical index per frame. It is a fixed five-rung")
    emit(u"  ladder, identical in both imaging modes and indexed by the frequency STEP")
    emit(u"  rather than the frequency itself: every rung up in frequency is also a rung")
    emit(u"  down in transmit power. Field II transmits the same excitation at every")
    emit(u"  frequency, so this is a domain gap, not a modelling choice we get to make.")
    emit(u"")
    emit(u"  Penetration is the quantity the constrained label rule uses. It is read at")
    emit(u"  the deeper display depth: at 67 mm the lowest fundamental frequency never")
    emit(u"  reaches its own noise floor, so the image is not deep enough to measure it.")
    emit(u"")
    emit(u"%-12s %8s %8s %10s %14s %10s"
         % (u"mode", u"freq", u"MI", u"floor", u"penetration mm", u"covered"))
    for session, (mode, captures, palette) in loaded.items():
        rows = {}
        for capture in captures:
            db = S.bc0_to_db(capture.bc0, counts_per_db[session])
            tail = int(round(capture.geometry.num_points * (1 - NOISE_TAIL_FRACTION)))
            floor = float(np.median(db[tail:, :]))
            reach = penetration_mm(db, capture.geometry, floor)
            key = (round(capture.geometry.depth_mm, 1),
                   round(CAL.capture_frequency(capture), 2))
            rows.setdefault(key, []).append(
                (mechanical_index(capture), floor, reach,
                 reach / capture.geometry.depth_mm))
        deepest = max(k[0] for k in rows)
        for key in sorted(k for k in rows if k[0] == deepest):
            values = np.array(rows[key])
            emit(u"%-12s %8.2f %8.3f %10.2f %14.1f %10.3f"
                 % (MODE_NAMES[mode], key[1], values[0, 0], values[:, 1].mean(),
                    values[:, 2].mean(), values[:, 3].mean()))

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_e8.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
