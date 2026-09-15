# -*- coding: utf-8 -*-
"""标定 Field II 带噪数据的噪声底常数，并定出散斑宽度的噪声过滤阈值。

    一、噪声底常数 NOISE_OFFSET_DB

bmode_opt/fieldii_noise.py 按生成器参数解析算出逐行噪声底，只差一个与孔径无关的常数。
这里在噪声主导的行上量它：

  残差 r(z) = 行中位数 dB - 模型（不含常数）
  纯噪声行上 r(z) 就是常数；有组织信号的行 r(z) 更大。

噪声主导的认定：60 mm 裁剪分片里，最深 15 mm 按 5 mm 分三段，三段残差的极差不超过
FLAT_DB 才用。组织信号随深度单调衰减，还有信号的分片浅段一定更高；三段持平说明这 15 mm
（孔径约 100 到 128 阵元）全是噪声。常数取这些分片最深 5 mm 残差的中位数。

第一版用「残差进入初值 + 1 dB 以内的尾段」，尾段起点仍带着组织信号，孔径小的一端偏高
2 到 3 dB，得出 +0.69 dB。改用持平判据后在最噪的四个体模上逐段核对：35 到 60 mm 残差
都在 -0.2 到 -0.45 dB，同一次仿真 42 / 50 / 60 mm 三种裁剪在 37 到 42 mm 带上分别为
-0.38 / -0.39 / -0.42 dB。

报告常数在体模、频率、裁剪深度、孔径大小之间是否一致——一致才说明模型的深度项是对的。

    二、散斑宽度的噪声过滤阈值

实机按 1.5 个线距过滤（0.1488 mm/线）。Field II 线距 0.225 mm，而聚焦良好的波束散斑
本身就可能窄于 1.5 线。这里分别量：
  纯噪声带的宽度（横向不相关，理论上约 0.5 线）
  高出噪声底 12 dB 以上的带（聚焦判据实际使用的带）里最窄的宽度
阈值取在两者之间。

用法：python tests/measure_fieldii_noise_floor.py
"""

import collections
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
os.chdir(ROOT)

import numpy as np

import fieldii_loader as FL
import fieldii_noise as FN
import front_end as FE

DATA = FL.Path("data/field_ii/full_noise")
FREQUENCIES = (6.5, 8.0)
CROPS = (42.0, 60.0)
FLAT_DB = 0.3


def complete_phantoms(paths):
    count = collections.Counter(re.sub(r"_d\d+p\d_.*", "", p.name) for p in paths)
    return {k for k, v in count.items() if v == 168}


def smooth(values, rows):
    kernel = np.ones(rows) / rows
    return np.convolve(values, kernel, mode="valid")


def main():
    lines = []
    emit = lines.append
    paths = FL.find_shards(DATA)
    keep = complete_phantoms(paths)
    shards = [p for p in paths if re.sub(r"_d\d+p\d_.*", "", p.name) in keep]
    chosen = []
    for p in shards:
        meta = FL.parse_shard_name(p)
        if meta["frequency_mhz"] in FREQUENCIES and float(re.search(r"_d(\d+)p", p.name).group(1)) in CROPS:
            chosen.append(p)
    emit(u"complete phantoms %d, shards used %d (frequencies %s, crops %s mm)"
         % (len(keep), len(chosen), FREQUENCIES, CROPS))

    residuals = []
    for p in chosen:
        s = FL.load_shard(p)
        r = np.median(s.db_image, axis=1) - FN.model_without_offset_db(s)
        z = s.geometry.min_depth_mm + (np.arange(s.geometry.num_points) + 0.5) * s.geometry.mm_per_point
        residuals.append((p, s, r, z))

    offsets = []
    by = collections.defaultdict(list)
    bands = collections.defaultdict(list)
    for p, s, r, z in residuals:
        if float(re.search(r"_d(\d+)p", p.name).group(1)) != 60.0:
            continue
        top = z[-1]
        segments = [float(np.median(r[(z >= top - 15.0 + 5.0 * k) & (z < top - 10.0 + 5.0 * k)]))
                    for k in range(3)]
        if max(segments) - min(segments) > FLAT_DB:
            continue
        value = segments[-1]
        offsets.append(value)
        meta = FL.parse_shard_name(p)
        by[("phantom", "%s %d" % (meta["phantom_type"], meta["seed"]))].append(value)
        by[("frequency", meta["frequency_mhz"])].append(value)
        for low in range(20, 60, 5):
            m = (z >= low) & (z < low + 5)
            bands[low].append(float(np.median(r[m])) - value)
    offset = float(np.median(offsets))
    emit(u"")
    emit(u"=========== 1. noise floor offset ===========")
    emit(u"  60 mm shards whose last 15 mm is flat within %.1f dB: %d" % (FLAT_DB, len(offsets)))
    emit(u"  NOISE_OFFSET_DB = %.3f dB   (p10 %.3f, p90 %.3f)"
         % (offset, float(np.percentile(offsets, 10)), float(np.percentile(offsets, 90))))
    for key in sorted(by):
        v = by[key]
        emit(u"  %-10s %-18s shards %3d  median %+.3f  range %+.3f .. %+.3f"
             % (key[0], key[1], len(v), float(np.median(v)), min(v), max(v)))
    emit(u"  residual above the deepest 5 mm, by band (0 all the way up = noise and model agree;")
    emit(u"  positive at shallow bands = tissue signal still present there):")
    for low in sorted(bands):
        v = bands[low]
        emit(u"    %2d-%2d mm  median %+.2f  p10 %+.2f" % (low, low + 5, float(np.median(v)),
                                                        float(np.percentile(v, 10))))

    emit(u"")
    emit(u"=========== 2. lateral width of noise bands vs usable tissue bands (in lines) ===========")
    noise_w, tissue_w = [], collections.defaultdict(list)
    for p in shards:
        meta = FL.parse_shard_name(p)
        if float(re.search(r"_d(\d+)p", p.name).group(1)) != 60.0:
            continue
        s = FL.load_shard(p)
        floor = FN.noise_floor_db(s, offset)
        db = s.db_image
        z = s.geometry.min_depth_mm + (np.arange(s.geometry.num_points) + 0.5) * s.geometry.mm_per_point
        env = 10.0 ** (db / 20.0)
        for low in np.arange(5.0, 55.0, FE.SPECKLE_BAND_MM):
            mask = (z >= low) & (z < low + FE.SPECKLE_BAND_MM)
            excess = float(np.median(db[mask, :]) - np.mean(floor[mask]))
            width = FE.lateral_speckle_width_mm(env[mask, :], s.geometry.mm_per_line)
            if not np.isfinite(width):
                continue
            lines_w = width / s.geometry.mm_per_line
            if excess < 1.0:
                noise_w.append(lines_w)
            elif excess >= FE.SPECKLE_MARGIN_DB:
                tissue_w[meta["frequency_mhz"]].append(lines_w)
    if noise_w:
        emit(u"  noise bands (median within 1 dB of floor): %d bands, width lines p50 %.2f  p99 %.2f  max %.2f"
             % (len(noise_w), np.median(noise_w), np.percentile(noise_w, 99), np.max(noise_w)))
    for f in sorted(tissue_w):
        v = tissue_w[f]
        emit(u"  tissue bands >= %d dB over floor, %.1f MHz: %4d bands, width lines min %.2f  p1 %.2f  p50 %.2f"
             % (FE.SPECKLE_MARGIN_DB, f, len(v), np.min(v), np.percentile(v, 1), np.median(v)))
    emit(u"  console filter 1.5 lines would drop %.1f%% of these tissue bands"
         % (100.0 * np.mean(np.concatenate([np.asarray(v) for v in tissue_w.values()]) < 1.5)))

    text = u"\n".join(lines)
    io.open(os.path.join(ROOT, "tests", "measure_fieldii_noise_floor.txt"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
