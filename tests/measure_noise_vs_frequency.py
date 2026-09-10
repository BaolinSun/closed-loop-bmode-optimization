# -*- coding: utf-8 -*-
"""组织与噪声底之差随发射频率怎么变——现有实机数据够不够回答。

    为什么要这个量

measure_frequency_tradeoff 证明：Field II 的 noise_enabled 全是 0，没有噪声底，
高频只剩好处，频率标签必然railing到最高频。注入噪声能恢复内部极值，但结果对噪声
电平极其敏感——间隔 30 dB 时约束几乎不咬合（各深度都选 8 MHz），20 dB 才咬合并
给出浅高频、深低频的正确行为。

所以「组织高出噪声底多少 dB」是决定频率标签的那个数，而且它必须随频率变化：高频
衰减快，深部更早掉进噪声。现有的 console_calibration.json 每个 (场次, 模式) 只有
一个噪声底，从显示深度 >=58 mm 的帧上量的，没有分频率。

    怎么量

一帧的最深处若已超出穿透，那里剩下的就是噪声。所以：
    噪声底  = 最深 8% 行的中位电平
    组织电平 = 固定浅深度（10-20 mm）上的中位电平，那里一定还有信号
    间隔    = 两者之差
只用显示深度足够大的帧，否则「最深处」仍在组织里，量到的不是噪声。

    这个脚本要回答的

一、间隔随频率掉得有多快。若掉得明显，频率的穿透代价可以直接从实机标定，不必猜。
二、现有帧覆盖够不够——每个 (模式, 频率) 上有几帧深度够大。不够的就是要补采的。

用法：python tests/measure_noise_vs_frequency.py
"""

import collections
import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import calibration as CAL
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

# 显示深度至少这么大，最深处才可能已经超出穿透。
#
# 67 mm 不是拍的。逐帧量深部 20% 行的电平随深度的斜率：50.2 mm 时是 -0.18 到
# -0.84 dB/mm，还在下降，那里仍是信号；58.6 mm 是 -0.03 到 -0.38 的过渡带；
# 到 67 mm 才落到 -0.014 到 -0.06，平了，才是噪声。用 50 mm 当门槛会把信号
# 当噪声量，基波的间隔因此看起来完全不随频率变化——那是假象。
# 66 而不是 67：67.0 那一档的实际值略小于 67，卡在边界上会被整档滤掉。
MIN_DEPTH_MM = 66.0
# 最深这一段算噪声。
NOISE_TAIL_FRACTION = 0.08
# 组织电平在这一段上取，浅到一定还有信号。
TISSUE_DEPTH_MM = (10.0, 20.0)

MODE_NAMES = {0: "fundamental", 1: "harmonic"}


def sessions():
    found = []
    for path in sorted(DEFAULT_DATA_DIR.rglob("Algo_BC0.bin")):
        label = str(path.parent.parent.relative_to(DEFAULT_DATA_DIR)).replace("\\", "/")
        if label not in found:
            found.append(label)
    return found


def gap_for(capture, counts_per_db):
    """一帧的组织电平、噪声底与两者之差，单位 dB。"""
    db = S.bc0_to_db(capture.bc0, counts_per_db)
    geometry = capture.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)

    tail = int(round(geometry.num_points * (1.0 - NOISE_TAIL_FRACTION)))
    noise_db = float(np.median(db[tail:, :]))

    band = (depth >= TISSUE_DEPTH_MM[0]) & (depth <= TISSUE_DEPTH_MM[1])
    if band.sum() < 10:
        return None
    tissue_db = float(np.median(db[band, :]))
    return tissue_db, noise_db, tissue_db - noise_db


def main():
    lines = []
    emit = lines.append

    calibration = {}
    data = None
    path = "bmode_opt/console_calibration.json"
    if os.path.exists(path):
        import json
        data = json.load(io.open(path, encoding="utf-8"))
        for group in data["groups"]:
            calibration[(group["session"], group["image_mode"])] = group["counts_per_db"]

    rows = []
    for label in sessions():
        for capture_path in find_captures(DEFAULT_DATA_DIR / label):
            capture = load_capture(capture_path)
            if capture.geometry.depth_mm < MIN_DEPTH_MM:
                continue
            mode = S.capture_image_mode(capture)
            counts = calibration.get((label, mode), S.DEFAULT_COUNTS_PER_DB)
            found = gap_for(capture, counts)
            if found is None:
                continue
            rows.append((mode, round(CAL.capture_frequency(capture), 2),
                         round(capture.geometry.depth_mm, 1), label,
                         capture.name) + found)

    emit(u"=========== coverage: frames deep enough to show a noise floor ===========")
    emit(u"  A frame only reveals the noise floor if its display depth runs past the")
    emit(u"  penetration limit. Frames shallower than %g mm are skipped." % MIN_DEPTH_MM)
    emit(u"")
    emit(u"%-13s %8s %s"
         % (u"mode", u"freq", u" ".join(u"%8s" % (u"%g mm" % d)
                                        for d in [50.2, 58.6, 67.0, 75.4])))
    counts = collections.Counter((r[0], r[1], r[2]) for r in rows)
    for mode in sorted({r[0] for r in rows}):
        for frequency in sorted({r[1] for r in rows if r[0] == mode}):
            cells = [counts.get((mode, frequency, d), 0)
                     for d in [50.2, 58.6, 67.0, 75.4]]
            emit(u"%-13s %8.2f %s"
                 % (MODE_NAMES[mode], frequency,
                    u" ".join(u"%8d" % c for c in cells)))

    emit(u"")
    emit(u"=========== the gap, by mode and frequency ===========")
    emit(u"  tissue = median over %g-%g mm, noise = median over the deepest %.0f%% of rows."
         % (TISSUE_DEPTH_MM[0], TISSUE_DEPTH_MM[1], 100 * NOISE_TAIL_FRACTION))
    emit(u"  A gap that falls with frequency is the penetration cost, measured rather")
    emit(u"  than assumed.")
    emit(u"")
    emit(u"%-13s %8s %6s %10s %10s %10s %10s"
         % (u"mode", u"freq", u"n", u"tissue", u"noise", u"gap", u"gap sd"))
    for mode in sorted({r[0] for r in rows}):
        for frequency in sorted({r[1] for r in rows if r[0] == mode}):
            pool = [r for r in rows if r[0] == mode and r[1] == frequency]
            gaps = np.array([r[7] for r in pool])
            emit(u"%-13s %8.2f %6d %10.2f %10.2f %10.2f %10.2f"
                 % (MODE_NAMES[mode], frequency, len(pool),
                    float(np.mean([r[5] for r in pool])),
                    float(np.mean([r[6] for r in pool])),
                    float(gaps.mean()), float(gaps.std())))

    emit(u"")
    emit(u"=========== how fast does the gap fall with frequency ===========")
    for mode in sorted({r[0] for r in rows}):
        pool = [r for r in rows if r[0] == mode]
        frequencies = sorted({r[1] for r in pool})
        if len(frequencies) < 2:
            continue
        means = [float(np.mean([r[7] for r in pool if r[1] == f])) for f in frequencies]
        slope = np.polyfit(frequencies, means, 1)[0]
        emit(u"  %-13s %g to %g MHz : gap %.1f to %.1f dB, slope %.2f dB per MHz"
             % (MODE_NAMES[mode], frequencies[0], frequencies[-1],
                means[0], means[-1], slope))

    emit(u"")
    emit(u"  %d frames deep enough, out of %d captures"
         % (len(rows), sum(1 for _ in DEFAULT_DATA_DIR.rglob("Algo_BC0.bin"))))

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_noise_vs_frequency.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
