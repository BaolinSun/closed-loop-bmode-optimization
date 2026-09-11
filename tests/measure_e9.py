# -*- coding: utf-8 -*-
"""E9 分析：最优发射聚焦随显示深度怎么走。

    这批数据要回答什么

聚焦是三根前端轴里图像特征最强、标签覆盖最弱的一根：改动它 BC0 相关掉到 0.15
（频率 0.23、深度 0.30，而探头移位只掉到 0.69），可标注帧却只有 99。

Field II 上聚焦已经证明有真正的内部极值——侧向半高全宽在 10/15/25/35 mm 上是
0.853/0.806/0.795/0.814 mm，最优落在 25 mm，跨 20 个散斑种子一致率 0.925。判据
成立，缺的是实机数据。E9 就是补这个。

    用什么量分辨率

散斑的侧向自相关宽度。它正比于波束宽度，不需要点靶，全图每个深度都能算，所以
能直接回答「这个聚焦设置在哪个深度上最锐利」。

做法：取一个深度带，逐行减去该行均值（去掉亮度，只留斑点结构），算侧向自相关，
取降到 0.5 的滞后，换算成 mm。

    两个必须避开的陷阱

一、【噪声区】噪声在横向不相关，自相关宽度会异常地窄，看着像「分辨率极好」。
   所以先按 E8 的办法定出噪声底，组织电平不足噪声底 +10 dB 的深度带一律不算。

二、【点靶】体模里有成列的点靶，它们比斑点亮得多，会把自相关拉宽。按行做稳健
   化：超过该行中位数若干倍的像素截断。

第一版用「组织电平不足帧内噪声底 +10 dB」来排除噪声带，那是错的：显示深度
41.9 mm 时图像最深处（38-42 mm）还远没到穿透极限（谐波 45 mm、基波 46 mm），
拿最深 8% 当噪声底等于把信号当噪声，几乎所有带都被误排除。改成直接识别噪声
本身的特征——噪声在横向完全不相关，自相关宽度会塌到线间距量级。

    这个脚本报告什么

一、【逐深度带的侧向宽度】每个显示深度下，5 个聚焦档各自在各深度带上的宽度。
二、【最优聚焦随深度怎么走】每个深度带上宽度最小的聚焦档。
三、【重复性】同一设置 2-3 帧之间的离散，以及它相对于聚焦档之间差异的大小。
   若档间差异不比帧间离散大，这个判据在实机上就不成立。

用法：python tests/measure_e9.py
"""

import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import calibration as CAL
import display_palette as DP
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

SESSIONS = [("20260911_E9_GEN", 0), ("20260911_E9_THI", 1)]
MODE_NAMES = {0: "fundamental", 1: "harmonic"}

# 这两个由 tests/measure_e8.py 在同一天、同一体模上拟合得到。
COUNTS_PER_DB = {0: 571.3, 1: 619.9}

# 深度带的宽度，单位 mm。太窄则行数不足，太宽则把聚焦的局部效应抹平。
BAND_MM = 5.0
# 宽度窄到线间距的这个倍数以下，就认为该带是噪声而不是斑点。
NOISE_WIDTH_LINES = 1.5

# 噪声底，由 tests/measure_e8.py 在同一天同一体模上独立测得，用的也是同一组
# counts_per_db。噪声底是接收机的性质，不随探头位置或耦合变化，所以可以跨场次用。
NOISE_FLOOR_DB = {0: 6.2, 1: 17.9}

# 组织电平要高出噪声底这么多，该带才算可用。
#
# 12 dB 比穿透判据的 3 dB 严得多，是必要的：自相关宽度比电平更早被噪声污染。
# 噪声横向不相关，掺进来会把宽度压窄，看着像「分辨率极好」。谐波 58.6 mm 下
# 37.5 与 42.5 mm 两带（高出噪声底 10.4 和 6.3 dB）就是这样翻向聚焦 10 的，
# 而 32.5 mm 带（13.9 dB）仍然正常。
BAND_MARGIN_DB = 12.0
# 自相关降到这个值时的滞后，算作宽度。
CORRELATION_LEVEL = 0.5
# 逐行截断：超过该行中位数这么多倍的像素压下去，挡住点靶。
CLIP_FACTOR = 4.0


def lateral_correlation_width_mm(envelope, mm_per_line):
    """一个深度带的散斑侧向自相关宽度，单位 mm。

    逐行去均值再做自相关，行间平均。返回相关降到 CORRELATION_LEVEL 的滞后。
    """
    block = np.asarray(envelope, dtype=np.float64)
    if block.shape[0] < 4 or block.shape[1] < 16:
        return float("nan")
    # 点靶比斑点亮得多，会把自相关拉宽，逐行截断挡住它们。
    median = np.median(block, axis=1, keepdims=True)
    block = np.minimum(block, CLIP_FACTOR * np.maximum(median, 1e-12))
    block = block - block.mean(axis=1, keepdims=True)

    lines = block.shape[1]
    spectrum = np.fft.rfft(block, n=2 * lines, axis=1)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), axis=1)[:, :lines]
    zero = correlation[:, :1]
    good = (zero[:, 0] > 0)
    if good.sum() < 4:
        return float("nan")
    correlation = (correlation[good] / zero[good]).mean(axis=0)

    below = np.where(correlation < CORRELATION_LEVEL)[0]
    if below.size == 0:
        return float("nan")
    index = below[0]
    if index == 0:
        return 0.0
    # 线性插值找过零点。
    high, low = correlation[index - 1], correlation[index]
    lag = (index - 1) + (high - CORRELATION_LEVEL) / (high - low)
    return float(lag * mm_per_line)


def frame_widths(capture, counts_per_db, floor_db):
    """一帧的逐深度带侧向宽度。噪声带返回 nan。"""
    db = S.bc0_to_db(capture.bc0, counts_per_db)
    geometry = capture.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
    envelope = 10.0 ** (db / 20.0)
    out = {}
    edges = np.arange(geometry.min_depth_mm, geometry.depth_mm, BAND_MM)
    for low in edges:
        mask = (depth >= low) & (depth < low + BAND_MM)
        if mask.sum() < 8:
            continue
        centre = low + BAND_MM / 2.0
        level = float(np.median(db[mask, :]))
        if level < floor_db + BAND_MARGIN_DB or low + BAND_MM > geometry.depth_mm:
            continue        # 已被噪声污染，或该带被图像下边缘截断
        width = lateral_correlation_width_mm(envelope[mask, :],
                                             geometry.mm_per_line)
        # 噪声在横向不相关，宽度会塌到线间距量级；那不是「分辨率极好」。
        if np.isfinite(width) and width < NOISE_WIDTH_LINES * geometry.mm_per_line:
            width = float("nan")
        out[centre] = (width, level)
    return out


def main():
    lines = []
    emit = lines.append
    summaries = []

    for session, mode in SESSIONS:
        root = DEFAULT_DATA_DIR / session
        if not root.exists():
            emit(u"MISSING %s" % session)
            continue
        captures = sorted((load_capture(p) for p in find_captures(root)),
                          key=lambda c: c.name)
        emit(u"")
        emit(u"=========== %s  (%s, %d frames) ==========="
             % (session, MODE_NAMES[mode], len(captures)))

        grouped = {}
        for capture in captures:
            key = (round(capture.geometry.depth_mm, 1), capture.focus_mm)
            grouped.setdefault(key, []).append(
                frame_widths(capture, COUNTS_PER_DB[mode], NOISE_FLOOR_DB[mode]))

        table = []
        for display_depth in sorted({k[0] for k in grouped}):
            focuses = sorted(k[1] for k in grouped if k[0] == display_depth)
            centres = sorted({c for f in focuses
                              for c in grouped[(display_depth, f)][0]})
            levels = {}
            for centre in centres:
                pool = [w[centre][1] for f in focuses
                        for w in grouped[(display_depth, f)] if centre in w]
                levels[centre] = float(np.mean(pool)) if pool else float("nan")
            emit(u"")
            emit(u"  --- display depth %.1f mm ---" % display_depth)
            emit(u"  lateral speckle correlation width in mm, smaller is sharper")
            emit(u"  %-10s %s   %s"
                 % (u"band mm",
                    u" ".join(u"%9s" % (u"focus %g" % f) for f in focuses),
                    u"best     level"))
            for centre in centres:
                row, spread = [], []
                for focus in focuses:
                    values = [w.get(centre, (float("nan"), 0.0))[0]
                              for w in grouped[(display_depth, focus)]]
                    values = np.array(values, dtype=np.float64)
                    finite = values[np.isfinite(values)]
                    row.append(float(finite.mean()) if finite.size else float("nan"))
                    spread.append(float(finite.std()) if finite.size > 1 else 0.0)
                if all(not np.isfinite(v) for v in row):
                    continue
                best = focuses[int(np.nanargmin(row))]
                table.append((display_depth, centre, best,
                              float(np.nanmax(row) - np.nanmin(row)),
                              float(np.mean(spread))))
                emit(u"  %-10.1f %s   %6g   %7.1f dB  (sd %.3f, spread %.3f)"
                     % (centre,
                        u" ".join(u"%9.3f" % v for v in row),
                        best, levels[centre], float(np.mean(spread)),
                        float(np.nanmax(row) - np.nanmin(row))))

        summaries.append((session, mode, table))

    emit(u"")
    emit(u"=========== summary: does the best focus follow the band ===========")
    emit(u"  If the focus label means anything, the sharpest setting for a band should")
    emit(u"  be the one focused near that band. Perfect tracking would put every entry")
    emit(u"  on the diagonal.")
    emit(u"")
    emit(u"%-14s %8s %10s %12s %10s %10s"
         % (u"session", u"display", u"band mm", u"best focus", u"error mm", u"spread/sd"))
    for session, mode, table in summaries:
        for display_depth, centre, best, spread, sd in table:
            ratio = (spread / sd) if sd > 1e-6 else float("inf")
            emit(u"%-14s %8.1f %10.1f %12g %10.1f %10.0f"
                 % (session[-6:], display_depth, centre, best, best - centre, ratio))
    emit(u"")
    emit(u"  error is best focus minus band centre. spread/sd is how many times bigger")
    emit(u"  the difference between focus settings is than the difference between two")
    emit(u"  frames of the same setting; under about 3 the criterion is not usable.")

    emit(u"")
    emit(u"=========== what to look for ===========")
    emit(u"  The best column should move down as the band gets deeper: that is the")
    emit(u"  focus following the region of interest, which is the whole premise of the")
    emit(u"  focus label. If it does not move, the label has nothing to learn.")
    emit(u"")
    emit(u"  The focus spread must also be larger than the repeat sd. If turning the")
    emit(u"  knob changes the width by less than two frames of the same setting differ,")
    emit(u"  the criterion does not survive on the console however well it worked in")
    emit(u"  simulation.")

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measure_e9.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
