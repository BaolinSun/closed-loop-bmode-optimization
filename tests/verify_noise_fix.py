# -*- coding: utf-8 -*-
"""检查噪声参考基准的改动是否生效——正式跑一周之前的最后一道关。

    改了什么

fieldii_simulate_line_chunk.m 里噪声幅度的参考，从「每条线自己的射频均方根」
改成「每个体模算一次的固定值」（fieldii_noise_reference_rms）。真实接收机的底噪
是机器的固有性质，与发射频率无关；原来的写法让噪声跟着信号走，组织与噪声之比
几乎不随频率变化，频率标签因此定不下来。

    要验证的三件事

一、【噪声底与频率无关】四个频率上recover出的噪声电平必须基本相同。改之前实测是
   -47.61 / -56.89 / -68.62 / -78.37，跟着信号掉了 30.8 dB；改之后应当持平。

二、【电平落在预期位置】试跑配置填 -78.0，参考采集上量到的偏移是 +0.39 dB，
   所以四个频率都应当落在 -77.6 dB 附近。

三、【信噪比随频率下降】组织电平减噪声底的斜率应当接近无噪声数据预测的
   -8.6 dB/MHz，与实机实测的 -8.24 同量级。这是这次重仿真要买的东西。

噪声与信号非相干叠加，所以逐像素在线性域上与无噪声孪生帧相减，精确解出噪声，
不依赖任何模型。

用法：
    python tests/verify_noise_fix.py --pilot-dir <noise_pilot2 的 hdf5 目录>
"""

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, "bmode_opt")
import numpy as np

from fieldii_loader import DEFAULT_FIELDII_DIR, HDF5_SUBDIR, load_shard

# 试跑配置里填的值，configs/noise_pilot2.json 两端都是它。
EXPECTED_CONFIG_DB = -78.0
# 参考采集上实测的偏移，来自第一次试跑。
CONFIG_TO_FLOOR_OFFSET_DB = 0.39

# 四个频率之间噪声底的允许离散。不同频率的接收带宽不同，几 dB 的差异是正常的；
# 改之前那 30.8 dB 的落差则完全不是。
MAX_FLOOR_SPREAD_DB = 4.0
# 噪声底与预期位置的允许偏差。
MAX_FLOOR_ERROR_DB = 3.0
# 信噪比随频率下降的斜率，至少要有这么陡才算买到了频率依赖。
MIN_GAP_SLOPE_DB_PER_MHZ = 4.0

TISSUE_DEPTH_MM = (10.0, 20.0)


def recover_noise_db(clean, noisy):
    a = 10.0 ** (clean.db_image / 20.0)
    b = 10.0 ** (noisy.db_image / 20.0)
    residual = b ** 2 - a ** 2
    residual = residual[residual > 0]
    if residual.size < 1000:
        return None
    return 20.0 * np.log10(float(np.sqrt(np.mean(residual))))


def tissue_db(shard):
    geometry = shard.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
    band = (depth >= TISSUE_DEPTH_MM[0]) & (depth <= TISSUE_DEPTH_MM[1])
    return float(np.median(shard.db_image[band, :]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True)
    parser.add_argument("--noiseless-dir",
                        default=str(DEFAULT_FIELDII_DIR / HDF5_SUBDIR))
    parser.add_argument("--config-db", type=float, default=EXPECTED_CONFIG_DB)
    args = parser.parse_args()

    lines = []
    emit = lines.append
    failures = []

    noiseless_root = Path(args.noiseless_dir)
    rows = []
    for path in sorted(Path(args.pilot_dir).glob("*.h5")):
        twin = noiseless_root / path.name
        if not twin.exists():
            failures.append("no noiseless twin for %s" % path.name)
            continue
        clean, noisy = load_shard(twin), load_shard(path)
        noise = recover_noise_db(clean, noisy)
        if noise is None:
            failures.append("noise not recoverable at %g MHz" % noisy.frequency_mhz)
            continue
        rows.append({"freq": round(noisy.frequency_mhz, 2), "floor": noise,
                     "tissue": tissue_db(clean)})
    rows.sort(key=lambda r: r["freq"])
    if not rows:
        raise SystemExit("nothing to check")

    expected = args.config_db + CONFIG_TO_FLOOR_OFFSET_DB

    emit(u"=========== 1. is the noise floor now independent of frequency ===========")
    emit(u"  Before the change the floor tracked the signal and fell 30.8 dB across the")
    emit(u"  ladder. It should now sit still.")
    emit(u"")
    emit(u"%-8s %14s %14s %14s"
         % (u"freq", u"noise floor", u"expected", u"error"))
    for row in rows:
        emit(u"%-8g %14.2f %14.2f %14.2f"
             % (row["freq"], row["floor"], expected, row["floor"] - expected))
    spread = max(r["floor"] for r in rows) - min(r["floor"] for r in rows)
    worst = max(abs(r["floor"] - expected) for r in rows)
    emit(u"")
    emit(u"  spread across frequency %6.2f dB  (limit %.1f)" % (spread, MAX_FLOOR_SPREAD_DB))
    emit(u"  worst error vs expected %6.2f dB  (limit %.1f)" % (worst, MAX_FLOOR_ERROR_DB))
    if spread > MAX_FLOOR_SPREAD_DB:
        failures.append("floor still varies %.2f dB with frequency; the fixed "
                        "reference did not take effect" % spread)
    if worst > MAX_FLOOR_ERROR_DB:
        failures.append("floor is %.2f dB from where the config says it should be"
                        % worst)

    emit(u"")
    emit(u"=========== 2. does the tissue to noise gap now fall with frequency ===========")
    emit(u"  This is what the whole re-simulation is for. The noiseless data predicts")
    emit(u"  about -8.6 dB per MHz; the console measures -8.24.")
    emit(u"")
    emit(u"%-8s %14s %14s %10s" % (u"freq", u"tissue 10-20mm", u"noise floor", u"gap"))
    for row in rows:
        emit(u"%-8g %14.2f %14.2f %10.2f"
             % (row["freq"], row["tissue"], row["floor"], row["tissue"] - row["floor"]))
    frequencies = [r["freq"] for r in rows]
    gaps = [r["tissue"] - r["floor"] for r in rows]
    slope = float(np.polyfit(frequencies, gaps, 1)[0]) if len(rows) > 1 else 0.0
    emit(u"")
    emit(u"  gap slope %6.2f dB per MHz  (needs to be steeper than -%.1f)"
         % (slope, MIN_GAP_SLOPE_DB_PER_MHZ))
    if slope > -MIN_GAP_SLOPE_DB_PER_MHZ:
        failures.append("gap slope is only %.2f dB per MHz; the frequency dependence "
                        "the re-simulation is meant to buy is not there" % slope)

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
