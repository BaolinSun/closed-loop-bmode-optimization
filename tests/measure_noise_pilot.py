# -*- coding: utf-8 -*-
"""读取 Field II 噪声试跑的结果，算出正式配置该填的 electronic_noise_db。

    2026-09-11 改写：第一版的读数办法是错的

第一版从图像最深处读噪声底，再按孔径模型反推。试跑证明那样读不到噪声：在 -48 dB
这个电平下最深处仍然是组织信号，读出来的「噪声底」其实是信号，于是算出的斜率是
+4.03 dB/MHz，把「相对基准抹平频率依赖」这个**正确**的判断误判成不成立。

现在改用**与无噪声孪生帧在线性域相减**。噪声与信号非相干叠加：

    包络_有噪声^2 = 包络_无噪声^2 + 包络_噪声^2

两边相减就精确解出注入的噪声，不需要孔径模型，也不怕深处还是信号。试跑的种子在
原数据集里有同设置的孪生帧，两者衰减系数完全相同（0.66841981），可逐像素比。

    第一版那条换算链为什么算不对

原本以为 图像噪声底 = R + N配置 + 3.18 - 10*log10(接收阵元数)，按孔径模型那几项
合计 -17.5 dB。实测是 +13.4 dB，差 31 dB。原因是 signal_rms 取的是**通道射频**，
而每个阵元收到的是整个被照射区域的回波；波束合成只挑出焦线上那一份，抑制掉的
离轴杂波正好是这 31 dB。这条链路只能实测，不能算。

好在实测结果非常好用：在参考采集自己那一帧上，

    包络噪声底 = electronic_noise_db + 0.4 dB

也就是**代码改成固定参考之后，配置里填的数几乎就是图像上的噪声底**。

    这个脚本报告什么

一、【注入的噪声到底多大】线性域相减，逐频率解出。
二、【相对基准是否抹平了频率依赖】噪声底随频率的斜率与信号随频率的斜率若相等，
   就说明噪声完全跟着信号走，组织与噪声之比不随频率变化——那是要求改
   fieldii_simulate_line_chunk.m 的全部理由。
三、【正式配置填多少】把目标包络噪声底区间平移过去。

用法：
    python tests/measure_noise_pilot.py --pilot-dir <试跑输出目录>
"""

import argparse
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, "bmode_opt")
import numpy as np

from fieldii_loader import DEFAULT_FIELDII_DIR, HDF5_SUBDIR, load_shard

# 试跑配置里填的值。configs/noise_pilot.json 把区间两端都设成这个数。
PILOT_NOISE_DB = -48.0

# 浅层组织电平取这一段，浅到任何频率下都一定还有信号。
TISSUE_DEPTH_MM = (10.0, 20.0)

# 正式配置要命中的图像噪声底区间，来自 tests/measure_noise_reference_choice.py：
# 落在这个区间内，整条频率阶梯才都会在某些场景下成为正确答案。
TARGET_FLOOR_DB = (-95.0, -60.0)

# 线性域相减后，正残差少于这么多像素就认为噪声没露头，该帧不参与。
MIN_POSITIVE_PIXELS = 1000


def recover_noise_db(clean_shard, noisy_shard):
    """线性域相减，解出注入噪声的包络均方根，单位 dB。"""
    clean = 10.0 ** (clean_shard.db_image / 20.0)
    noisy = 10.0 ** (noisy_shard.db_image / 20.0)
    residual = noisy ** 2 - clean ** 2
    residual = residual[residual > 0]
    if residual.size < MIN_POSITIVE_PIXELS:
        return None
    return 20.0 * np.log10(float(np.sqrt(np.mean(residual))))


def envelope_rms_db(shard):
    envelope = 10.0 ** (shard.db_image / 20.0)
    return 20.0 * np.log10(float(np.sqrt(np.mean(envelope ** 2))))


def tissue_db(shard):
    geometry = shard.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
    band = (depth >= TISSUE_DEPTH_MM[0]) & (depth <= TISSUE_DEPTH_MM[1])
    return float(np.median(shard.db_image[band, :]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True,
                        help="试跑输出的 hdf5 目录")
    parser.add_argument("--noiseless-dir", default=str(DEFAULT_FIELDII_DIR / HDF5_SUBDIR),
                        help="原无噪声数据集的 hdf5 目录，用来找孪生帧")
    parser.add_argument("--noise-db", type=float, default=PILOT_NOISE_DB,
                        help="试跑配置里填的 electronic_noise_db")
    args = parser.parse_args()

    lines = []
    emit = lines.append

    paths = sorted(Path(args.pilot_dir).glob("*.h5"))
    if not paths:
        raise SystemExit("no shards under %s" % args.pilot_dir)
    noiseless_root = Path(args.noiseless_dir)

    emit(u"=========== pilot shards ===========")
    rows = []
    for path in paths:
        twin = noiseless_root / path.name
        noisy = load_shard(path)
        if not twin.exists():
            emit(u"  %-46s NO NOISELESS TWIN, skipped" % path.stem[:46])
            continue
        clean = load_shard(twin)
        noise_db = recover_noise_db(clean, noisy)
        rows.append({"freq": round(noisy.frequency_mhz, 2),
                     "noise_db": noise_db,
                     "rms_db": envelope_rms_db(clean),
                     "tissue_db": tissue_db(clean)})
        emit(u"  %-46s %g MHz  focus %g mm  depth %g mm"
             % (path.stem[:46], noisy.frequency_mhz, noisy.focus_mm,
                noisy.geometry.depth_mm))
    emit(u"  electronic_noise_db used in the pilot: %g" % args.noise_db)
    rows = [r for r in rows if r["noise_db"] is not None]
    if not rows:
        raise SystemExit("no shard had a recoverable noise residual")
    rows.sort(key=lambda r: r["freq"])

    emit(u"")
    emit(u"=========== 1. how big the injected noise actually is ===========")
    emit(u"  Recovered by subtracting the noiseless twin in the linear domain, so this is")
    emit(u"  exact rather than read off a region that may still hold signal.")
    emit(u"")
    emit(u"%-8s %14s %14s %14s"
         % (u"freq", u"noise env dB", u"image rms dB", u"noise - rms"))
    for row in rows:
        emit(u"%-8g %14.2f %14.2f %14.2f"
             % (row["freq"], row["noise_db"], row["rms_db"],
                row["noise_db"] - row["rms_db"]))

    emit(u"")
    emit(u"=========== 2. does the relative reference flatten frequency ===========")
    frequencies = [r["freq"] for r in rows]
    noise_slope = float(np.polyfit(frequencies, [r["noise_db"] for r in rows], 1)[0])
    signal_slope = float(np.polyfit(frequencies, [r["rms_db"] for r in rows], 1)[0])
    drift = (max(r["noise_db"] - r["rms_db"] for r in rows)
             - min(r["noise_db"] - r["rms_db"] for r in rows))
    emit(u"  noise floor slope  %7.2f dB per MHz" % noise_slope)
    emit(u"  signal level slope %7.2f dB per MHz" % signal_slope)
    emit(u"  the two differ by  %7.2f dB per MHz" % abs(noise_slope - signal_slope))
    emit(u"  noise-minus-signal drifts %.2f dB over the whole ladder" % drift)
    emit(u"")
    if abs(noise_slope - signal_slope) < 1.0:
        emit(u"  CONFIRMED. The floor tracks the signal almost exactly, so the tissue to")
        emit(u"  noise ratio barely moves with frequency. fieldii_simulate_line_chunk.m")
        emit(u"  line 89 must take a fixed per-phantom reference instead of each line's")
        emit(u"  own root mean square, before the full run.")
    else:
        emit(u"  NOT CONFIRMED. The floor does not simply track the signal; re-examine")
        emit(u"  before changing the generator.")

    emit(u"")
    emit(u"=========== 3. what to put in the config ===========")
    reference = rows[0]
    offset = reference["noise_db"] - args.noise_db
    emit(u"  The reference acquisition is the lowest frequency, %g MHz, because that is"
         % reference["freq"])
    emit(u"  the first one the generator's loop reaches and therefore the one a fixed")
    emit(u"  per-phantom reference would be taken from.")
    emit(u"")
    emit(u"  electronic_noise_db in the pilot   %8.2f dB" % args.noise_db)
    emit(u"  envelope noise floor it produced   %8.2f dB" % reference["noise_db"])
    emit(u"  offset between the two             %8.2f dB" % offset)
    emit(u"")
    emit(u"  Once the generator uses that one reference for every frequency, the offset")
    emit(u"  above is all that separates the config number from the image noise floor.")
    emit(u"")
    emit(u"  target envelope noise floor        %g to %g dB" % TARGET_FLOOR_DB)
    emit(u"")
    emit(u"  PUT THIS IN configs/full.json:")
    emit(u"")
    emit(u'      "noise": {')
    emit(u'        "enabled": true,')
    emit(u'        "reference": "per_phantom",')
    emit(u'        "electronic_noise_db": [%.1f, %.1f]'
         % (TARGET_FLOOR_DB[0] - offset, TARGET_FLOOR_DB[1] - offset))
    emit(u'      }')

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_noise_pilot.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
