# -*- coding: utf-8 -*-
"""Field II 带噪数据的逐行噪声底：由生成器的参数解析算出，不从图像估计。

    为什么能解析算

生成器（fieldii_simulate_line_chunk.m）在每个接收通道的射频上加独立高斯白噪声，均方根为
    noise_rms = noise_reference_rms * 10^(electronic_noise_db / 20)
两个量都写在分片属性里。之后是逐线延时叠加（fieldii_beamform_line.m）：
    权重 w = Hann 窗（非零端点）/ sum(Hann)，孔径阵元数 n(z) = round(z / (F数 * 阵元间距))，
    限制在 [rx_min_elements, rx_max_elements]，再取偶数。
独立噪声经加权求和后功率乘以 sum(w^2)。Hann 窗归一化到和为 1，sum(w^2) 约为 3/(2(n+1))，
孔径越大噪声越低。所以噪声底随深度下降：3.6 mm（n=8）到 57.6 mm（n=128）之间下降
10*log10(129/9) = 11.6 dB，约 -0.21 dB/mm，与 2026-09-11 pilot 实测的 -0.19 dB/mm 一致。

剩下与 n 无关的因子（延时插值对白噪声方差的缩减、包络检波把功率翻倍、中位数与均方根之比、
深度裁剪时的重采样）合成一个常数 NOISE_OFFSET_DB，由 tests/measure_fieldii_noise_floor.py
在噪声主导的行上实测。

    输出的刻度

与 fieldii_loader.load_shard 的 db_image 相同：20*log10(包络 / display_reference)，并且是
「纯噪声像素的 dB 中位数」，与实机 tissue.measure_noise_floor 取深部行 dB 中位数的口径一致。
"""

import numpy as np

# 与孔径无关的常数，见模块说明。tests/measure_fieldii_noise_floor.py 于 2026-09-15 在 38 个
# 最深 15 mm 全为噪声的 60 mm 分片上测得 -0.313 dB（p10 -0.376，p90 -0.247），体模、频率
# 之间一致；同一次仿真 42 / 50 / 60 mm 三种裁剪相差 0.04 dB，35 到 60 mm（孔径 78 到 128
# 阵元）逐段持平，说明模型的孔径项是对的。
NOISE_OFFSET_DB = -0.313


def _attr(attrs, key):
    return float(np.ravel(attrs[key])[0])


def aperture_count(shard):
    """逐行接收孔径阵元数，直接读分片里的 /rx_aperture_count。

    不按 fieldii_beamform_line.m 的公式在裁剪后的深度轴上重算：分片是从 65 mm 仿真
    按最近邻裁剪重采样出来的，重算会在孔径跳变的行上差一档。
    """
    import h5py
    with h5py.File(shard.path, "r") as handle:
        counts = np.asarray(handle["/rx_aperture_count"][()], dtype=np.float64).ravel()
    if counts.size != shard.geometry.num_points:
        raise ValueError("%s: /rx_aperture_count has %d rows, image has %d"
                         % (shard.path.name, counts.size, shard.geometry.num_points))
    return counts


def hann_sum_of_squares(count):
    """归一化非零端点 Hann 窗的 sum(w^2)。"""
    count = int(count)
    index = np.arange(1, count + 1, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * index / (count + 1))
    window /= window.sum()
    return float(np.sum(window ** 2))


def model_without_offset_db(shard):
    """逐行噪声底（不含常数 NOISE_OFFSET_DB），刻度同 shard.db_image。"""
    attrs = shard.attrs
    counts = aperture_count(shard)
    table = {c: hann_sum_of_squares(c) for c in np.unique(counts)}
    sum_sq = np.array([table[c] for c in counts])
    injected = _attr(attrs, "noise_reference_rms") * 10.0 ** (_attr(attrs, "electronic_noise_db") / 20.0)
    return (20.0 * np.log10(injected / _attr(attrs, "display_reference"))
            + 10.0 * np.log10(sum_sq))


def noise_floor_db(shard, offset_db=None):
    """逐行噪声底（纯噪声像素的 dB 中位数），形状 (num_points,)。"""
    offset = NOISE_OFFSET_DB if offset_db is None else offset_db
    if offset is None:
        raise ValueError("NOISE_OFFSET_DB is not measured yet; run tests/measure_fieldii_noise_floor.py")
    if int(_attr(shard.attrs, "noise_enabled")) != 1:
        raise ValueError("%s was generated without noise; there is no floor to model" % shard.path.name)
    return model_without_offset_db(shard) + float(offset)
