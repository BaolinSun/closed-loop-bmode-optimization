# -*- coding: utf-8 -*-
"""检查噪声改用固定参考之后是否生效，并量出配置该填多少。

    2026-09-11 改写：第一版的测法把结论搞反了

第一版对**全图**求 sqrt(mean(包络_有噪声^2 - 包络_无噪声^2))。问题是绝大多数像素
上噪声远低于信号，那里的残差没有意义，混进平均里只会得到一个跟着信号走的数。
于是它报告「底噪仍随频率变化 14.36 dB，改动没生效」——而实际上改动是生效的。

正确的测法是**逐深度带**，并且只取噪声真的抬高了图像的那些带：

    抬高量 = 有噪声电平 - 无噪声电平
    底噪   = 无噪声电平 + 10*log10(10^(抬高量/10) - 1)

包络存成 float32，平方域的相对精度约 -66 dB，所以抬高量低到 0.005 dB 仍然是真实
信号。低于这个值的带就是量不出来，必须报「--」而不是硬给一个数。

用这个办法重看 pilot2（N = -78）：

    6.5 MHz @ 50-60 mm   -115.21
    8.0 MHz @ 40-50 mm   -113.84
    8.0 MHz @ 50-60 mm   -114.71     离散 1.36 dB

两个频率给出同一个底噪，改动确实生效。4 和 5 MHz 量不到，是因为那里噪声比信号
低 60 多 dB——电平设得太低，不是改动失败。

    这个脚本报告什么

一、【逐带的底噪】以及哪些带量得出、哪些量不出。
二、【底噪是否与频率无关】能量出的频率之间必须一致。这是改动是否生效的判据。
三、【偏移与正式配置】偏移 = 底噪 - N配置，是一个纯 dB 平移量。

用法：
    python tests/verify_noise_fix.py --pilot-dir <hdf5 目录> --noise-db <配置里填的 N>
"""

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, "bmode_opt")
import numpy as np

from fieldii_loader import DEFAULT_FIELDII_DIR, HDF5_SUBDIR, load_shard

# 抬高量低于此值就认为量不出来。float32 在平方域的相对精度约 -66 dB，对应
# 约 1e-6 dB 的抬高，所以 0.005 dB 留了很大余量。
MIN_LIFT_DB = 0.005

# 能量出底噪的频率之间，允许的离散。
MAX_FLOOR_SPREAD_DB = 4.0

# 至少要有这么多个频率量得出底噪，频率无关性才算被检验过。
MIN_FREQUENCIES = 2

# 正式配置要命中的图像噪声底区间，来自 tests/measure_noise_reference_choice.py。
TARGET_FLOOR_DB = (-95.0, -60.0)

BANDS_MM = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]


def band_levels(shard, low_mm, high_mm):
    geometry = shard.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
    mask = (depth >= low_mm) & (depth < high_mm)
    if mask.sum() < 10:
        return None
    envelope = 10.0 ** (shard.db_image[mask, :] / 20.0)
    return 20.0 * np.log10(float(np.sqrt(np.mean(envelope ** 2))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--noiseless-dir",
                        default=str(DEFAULT_FIELDII_DIR / HDF5_SUBDIR))
    parser.add_argument("--noise-db", type=float, required=True,
                        help="试跑配置里填的 electronic_noise_db")
    args = parser.parse_args()

    lines = []
    emit = lines.append
    failures = []
    noiseless_root = Path(args.noiseless_dir)

    emit(u"=========== 1. noise floor, band by band ===========")
    emit(u"  floor = clean + 10*log10(10^(lift/10) - 1). A band whose lift is under")
    emit(u"  %.3f dB carries no measurable noise and is reported as --, not guessed."
         % MIN_LIFT_DB)
    emit(u"")
    emit(u"%-8s %10s %12s %10s %12s"
         % (u"freq", u"band mm", u"clean", u"lift dB", u"floor dB"))

    per_frequency = {}
    for path in sorted(Path(args.pilot_dir).glob("*.h5")):
        twin = noiseless_root / path.name
        if not twin.exists():
            failures.append("no noiseless twin for %s" % path.name)
            continue
        clean, noisy = load_shard(twin), load_shard(path)
        frequency = round(noisy.frequency_mhz, 2)
        for low, high in BANDS_MM:
            clean_db = band_levels(clean, low, high)
            noisy_db = band_levels(noisy, low, high)
            if clean_db is None or noisy_db is None:
                continue
            lift = noisy_db - clean_db
            if lift < MIN_LIFT_DB:
                emit(u"%-8g %10s %12.2f %10.4f %12s"
                     % (frequency, u"%d-%d" % (low, high), clean_db, lift, u"--"))
                continue
            floor = clean_db + 10.0 * np.log10(10.0 ** (lift / 10.0) - 1.0)
            per_frequency.setdefault(frequency, []).append(floor)
            emit(u"%-8g %10s %12.2f %10.4f %12.2f"
                 % (frequency, u"%d-%d" % (low, high), clean_db, lift, floor))

    emit(u"")
    emit(u"=========== 2. is the floor independent of frequency ===========")
    emit(u"  Before the change the floor tracked the signal: -47.6 at 4 MHz down to")
    emit(u"  -78.4 at 8 MHz, a 30.8 dB slide. With a fixed per-phantom reference the")
    emit(u"  frequencies that can be measured must agree.")
    emit(u"")
    if not per_frequency:
        failures.append("no band anywhere carried measurable noise; raise "
                        "electronic_noise_db and run the pilot again")
        emit(u"  nothing measurable")
    else:
        emit(u"%-8s %8s %12s %12s" % (u"freq", u"bands", u"median floor", u"spread"))
        medians = {}
        for frequency in sorted(per_frequency):
            values = np.array(per_frequency[frequency])
            medians[frequency] = float(np.median(values))
            emit(u"%-8g %8d %12.2f %12.2f"
                 % (frequency, values.size, medians[frequency],
                    values.max() - values.min()))
        emit(u"")
        if len(medians) < MIN_FREQUENCIES:
            failures.append(
                "only %d frequency carried measurable noise, so frequency "
                "independence was not tested. Raise electronic_noise_db so the "
                "floor comes up near the deep tissue level at 4 MHz." % len(medians))
        else:
            spread = max(medians.values()) - min(medians.values())
            emit(u"  spread across %d frequencies: %.2f dB  (limit %.1f)"
                 % (len(medians), spread, MAX_FLOOR_SPREAD_DB))
            if spread > MAX_FLOOR_SPREAD_DB:
                failures.append("floor still varies %.2f dB with frequency; the "
                                "fixed reference did not take effect" % spread)

    emit(u"")
    emit(u"=========== 3. offset, and what to put in the config ===========")
    if per_frequency:
        everything = np.array([v for values in per_frequency.values() for v in values])
        floor = float(np.median(everything))
        offset = floor - args.noise_db
        emit(u"  electronic_noise_db used   %8.2f dB" % args.noise_db)
        emit(u"  measured floor (median)    %8.2f dB" % floor)
        emit(u"  offset                     %8.2f dB" % offset)
        emit(u"")
        emit(u"  target floor range         %g to %g dB" % TARGET_FLOOR_DB)
        emit(u"")
        if failures:
            emit(u"  Checks failed, so this number is not safe to use yet.")
        else:
            emit(u"  PUT THIS IN configs/full.json:")
            emit(u"")
            emit(u'      "noise": {')
            emit(u'        "enabled": true,')
            emit(u'        "electronic_noise_db": [%.1f, %.1f]'
                 % (TARGET_FLOOR_DB[0] - offset, TARGET_FLOOR_DB[1] - offset))
            emit(u'      }')

    emit(u"")
    emit(u"=========== result ===========")
    if failures:
        for line in failures:
            emit(u"  FAIL  " + line)
        emit(u"")
        emit(u"  Do NOT start the full run.")
    else:
        emit(u"  All checks pass. The generator is ready for the full run.")

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "verify_noise_fix.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
