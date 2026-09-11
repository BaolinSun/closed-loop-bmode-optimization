# -*- coding: utf-8 -*-
"""读取 Field II 噪声试跑的结果，算出正式配置该填的 electronic_noise_db。

    这个脚本解决什么

从 configs/full.json 里的 electronic_noise_db 到图像上真正的噪声底，中间隔着
一整条换算链。链上除了一个量，其余都已经能精确算出来：

    图像噪声底(dB) = R + N配置 + 3.18 - 10*log10(接收阵元数)

  N配置   就是 electronic_noise_db 填的数
  3.18 dB 射频转包络的固定换算。带通高斯噪声的解析包络服从瑞利分布，
          中位数是射频标准差的 sqrt(2*ln2)=1.177 倍，20*log10(1.177)=1.41；
          再加上 fieldii_beamform_line 用归一化汉宁权重带来的 sqrt(3/2)，
          合计 3.18 dB
  阵元数  波束合成对噪声的抑制。信号相干相加、噪声非相干相加，而权重和为 1
          （fieldii_beamform_line.m 第 47 行 weights/sum(weights)），所以
          信号幅度不变、噪声降 10*log10(N)。接收孔径按 f-number 1.5 随深度
          张开，阵元数 = 深度/(1.5*阵元间距)，下限 8 上限 128
  R       参考采集的通道射频均方根，相对数据集显示参考值，单位 dB。
          **唯一测不出只能跑出来的量**，因为它取决于 Field II 内部怎么缩放
          激励与散射幅度，而 HDF5 里只存了包络，没存射频

试跑就是为了量 R。R 只是一个纯 dB 平移量，量准一次，正式配置就精确，不用试第二次。

    为什么试跑要用未修改的代码

原代码里 noise_rms = 该线自己的射频均方根 * 10^(N/20)。对**参考采集**
（4.0 MHz、聚焦 10 mm）而言，"该线自己的均方根"正好就是要找的参考均方根，
所以在 4.0 MHz 那一帧上读出的就是 R 本身。

顺带，另外三个频率给出一次直接验证：若四个频率的「浅层组织减噪声底」几乎不变，
就证实了「相对基准抹掉频率依赖」这个判断——那是要求改代码的全部理由。

用法：python tests/measure_noise_pilot.py --pilot-dir <试跑输出目录>
"""

import argparse
import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

from fieldii_loader import load_shard

# 试跑配置里填的值。configs/noise_pilot.json 把区间两端都设成这个数，
# 所以采样结果必然是它，不受随机数影响。
PILOT_NOISE_DB = -48.0

# 探头与波束合成参数，取自 configs/full.json 的 probe 与 beamforming 两段。
PITCH_MM = 0.3
RX_F_NUMBER = 1.5
MIN_RX_ELEMENTS = 8
MAX_RX_ELEMENTS = 128

# 射频转包络加上归一化汉宁权重的固定换算，见模块开头。
ENVELOPE_OFFSET_DB = 3.18

# 浅层组织电平取这一段，浅到任何频率下都一定还有信号。
TISSUE_DEPTH_MM = (10.0, 20.0)

# 正式配置要命中的图像噪声底区间，来自 tests/measure_noise_reference_choice.py：
# 落在这个区间内，整条频率阶梯才都会在某些场景下成为正确答案。
TARGET_FLOOR_DB = (-95.0, -60.0)

# 折算 N配置 时用的参考深度。取 30 mm，那是显示深度阶梯的中段。
REFERENCE_DEPTH_MM = 30.0


def rx_elements(depth_mm):
    """某深度上的接收阵元数。孔径按 f-number 张开，两端截断。"""
    count = np.asarray(depth_mm, dtype=np.float64) / (RX_F_NUMBER * PITCH_MM)
    return np.clip(count, MIN_RX_ELEMENTS, MAX_RX_ELEMENTS)


def beamforming_suppression_db(depth_mm):
    """波束合成把噪声压低多少 dB。"""
    return 10.0 * np.log10(rx_elements(depth_mm))


def depth_axis(geometry):
    return (geometry.min_depth_mm
            + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dir", required=True,
                        help="试跑输出的 hdf5 目录，例如 "
                             "D:/MyProjects/matlab/fieldii-dataset-generation/"
                             "data/noise_pilot/hdf5")
    parser.add_argument("--noise-db", type=float, default=PILOT_NOISE_DB,
                        help="试跑配置里填的 electronic_noise_db")
    args = parser.parse_args()

    lines = []
    emit = lines.append

    from pathlib import Path
    paths = sorted(Path(args.pilot_dir).glob("*.h5"))
    if not paths:
        raise SystemExit("no shards under %s" % args.pilot_dir)
    shards = [load_shard(p) for p in paths]
    shards.sort(key=lambda s: s.frequency_mhz)

    emit(u"=========== pilot shards ===========")
    for shard in shards:
        emit(u"  %-50s %g MHz  focus %g mm  depth %g mm"
             % (shard.name[:50], shard.frequency_mhz, shard.focus_mm,
                shard.geometry.depth_mm))
    emit(u"  electronic_noise_db used in the pilot: %g" % args.noise_db)

    emit(u"")
    emit(u"=========== 1. is the noise floor where the model says ===========")
    emit(u"  The floor is read off the deepest eighth of each image, then corrected by")
    emit(u"  the beamforming suppression that depth carries. If the corrected profile is")
    emit(u"  flat with depth, the aperture model is right and R can be trusted.")
    emit(u"")
    emit(u"%-8s %12s %12s %12s %12s"
         % (u"freq", u"raw floor", u"rx elements", u"corrected", u"flatness"))
    corrected_by_frequency = {}
    for shard in shards:
        depth = depth_axis(shard.geometry)
        tail = depth >= shard.geometry.depth_mm - 0.25 * (
            shard.geometry.depth_mm - shard.geometry.min_depth_mm)
        rows = np.median(shard.db_image[tail, :], axis=1)
        corrected = rows + beamforming_suppression_db(depth[tail])
        corrected_by_frequency[round(shard.frequency_mhz, 2)] = float(np.median(corrected))
        emit(u"%-8g %12.2f %12.1f %12.2f %12.2f"
             % (shard.frequency_mhz, float(np.median(rows)),
                float(np.median(rx_elements(depth[tail]))),
                float(np.median(corrected)), float(np.std(corrected))))
    emit(u"  flatness is the spread of the corrected profile; a couple of dB is fine,")
    emit(u"  ten or more means the deep part is still signal rather than noise.")

    emit(u"")
    emit(u"=========== 2. does the relative reference flatten frequency ===========")
    emit(u"  Shallow tissue minus noise floor, per frequency. The prediction from the")
    emit(u"  noiseless data was about -1.1 dB per MHz with the code as it stands, against")
    emit(u"  -8.6 with an absolute reference and -8.24 measured on the console.")
    emit(u"")
    emit(u"%-8s %14s %12s %10s" % (u"freq", u"tissue 10-20mm", u"floor", u"gap"))
    frequencies, gaps = [], []
    for shard in shards:
        depth = depth_axis(shard.geometry)
        band = (depth >= TISSUE_DEPTH_MM[0]) & (depth <= TISSUE_DEPTH_MM[1])
        tissue = float(np.median(shard.db_image[band, :]))
        tail = depth >= shard.geometry.depth_mm - 0.25 * (
            shard.geometry.depth_mm - shard.geometry.min_depth_mm)
        floor = float(np.median(shard.db_image[tail, :]))
        frequencies.append(shard.frequency_mhz)
        gaps.append(tissue - floor)
        emit(u"%-8g %14.2f %12.2f %10.2f" % (shard.frequency_mhz, tissue, floor,
                                             tissue - floor))
    if len(frequencies) > 1:
        slope = float(np.polyfit(frequencies, gaps, 1)[0])
        emit(u"")
        emit(u"  measured slope %.2f dB per MHz" % slope)
        if abs(slope) < 3.0:
            emit(u"  CONFIRMED: the relative reference does flatten the frequency")
            emit(u"  dependence. fieldii_simulate_line_chunk.m line 89 must be changed to")
            emit(u"  a fixed per-phantom reference before the full run.")
        else:
            emit(u"  NOT CONFIRMED: the slope is steeper than predicted. Do not change the")
            emit(u"  code on the strength of this analysis; re-examine first.")

    emit(u"")
    emit(u"=========== 3. R, and what to put in the config ===========")
    reference = min(corrected_by_frequency)
    r_value = (corrected_by_frequency[reference] - args.noise_db - ENVELOPE_OFFSET_DB)
    emit(u"  The reference acquisition is the lowest frequency, %g MHz, because that is" % reference)
    emit(u"  the first one the generator's loop reaches and therefore the one a fixed")
    emit(u"  per-phantom reference would be taken from.")
    emit(u"")
    emit(u"  corrected floor at %g MHz        %8.2f dB" % (reference, corrected_by_frequency[reference]))
    emit(u"  minus electronic_noise_db        %8.2f" % args.noise_db)
    emit(u"  minus envelope offset            %8.2f" % ENVELOPE_OFFSET_DB)
    emit(u"  R                                %8.2f dB" % r_value)
    emit(u"  (the estimate from the noiseless data was about -15.9 dB)")

    suppression = beamforming_suppression_db(REFERENCE_DEPTH_MM)
    low = TARGET_FLOOR_DB[0] - r_value - ENVELOPE_OFFSET_DB + suppression
    high = TARGET_FLOOR_DB[1] - r_value - ENVELOPE_OFFSET_DB + suppression
    emit(u"")
    emit(u"  target image noise floor         %g to %g dB" % TARGET_FLOOR_DB)
    emit(u"  at the %g mm reference depth the beamformer suppresses %.2f dB"
         % (REFERENCE_DEPTH_MM, suppression))
    emit(u"")
    emit(u"  PUT THIS IN configs/full.json:")
    emit(u"")
    emit(u'      "noise": {')
    emit(u'        "enabled": true,')
    emit(u'        "reference": "per_phantom",')
    emit(u'        "electronic_noise_db": [%.1f, %.1f]' % (low, high))
    emit(u'      }')

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_noise_pilot.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
