# -*- coding: utf-8 -*-
"""Field II 重新仿真时噪声该怎么配：参考基准选错，一周机时买不到频率依赖。

    背景

measure_frequency_tradeoff 证明 Field II 现有数据定不了频率标签，因为
noise_enabled 全是 0：没有噪声底，TGC 把深部抬多高都不会同时抬起噪声，高频只剩
好处。重新仿真加噪声是可行的（约一周），所以要把 configs/full.json 的 noise 段
配对。

    生成器现在怎么加噪声

fieldii_simulate_line_chunk.m 第 88-95 行：

    signal_rms = sqrt(mean(rf_multi(:).^2));          % 该线自己的射频均方根
    noise_rms  = signal_rms*10^(electronic_noise_db/20);
    rf_multi   = rf_multi + noise_rms*randn(...);

噪声电平**跟着每一条线自己的信号走**。真实接收机的底噪是机器的性质，与发射频率
无关；这里高频信号弱、噪声也按比例弱下去，组织与噪声之比因此几乎不随频率变化。

    这个脚本回答两件事

一、【参考基准】相对基准把频率依赖削弱了多少。拿现有无噪声数据算两种方案下
   「浅层组织电平减噪声底」随频率的斜率，与实机实测的 -8.24 dB/MHz 对照。

二、【电平】绝对基准下噪声底取多少。判据是约束咬合得好不好：逐显示深度看
   「穿透仍能覆盖整幅图的最高发射频率」，这一列要随深度变化，且整条频率阶梯
   都要被用到，否则标签仍然退化。

用法：python tests/measure_noise_reference_choice.py [--scenes 3]
"""

import argparse
import collections
import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import front_end as FE
from fieldii_loader import find_shards, load_shard

# 实机实测：谐波 4.4-5.7 MHz 上「浅层组织减噪声底」的斜率。
# 见 tests/measure_noise_vs_frequency.py。
CONSOLE_SLOPE_DB_PER_MHZ = -8.24

# 浅层组织电平在这一段上取，浅到任何频率下都一定还有信号。
TISSUE_DEPTH_MM = (10.0, 20.0)

# 待评估的绝对噪声底，单位 dB，用的是整个数据集共享的那个刻度（db_image 的刻度）。
FLOORS_DB = [-60.0, -70.0, -80.0, -90.0, -100.0, -110.0, -120.0]

# 覆盖率高于此值才算「穿透够得着整幅图」。
COVERED = 0.995

RAYLEIGH_MEDIAN = np.sqrt(2.0 * np.log(2.0))


def inject(db_image, floor_db, rng):
    """把绝对噪声底非相干地加进包络。"""
    envelope = 10.0 ** (db_image / 20.0)
    noise = rng.rayleigh(10.0 ** (floor_db / 20.0) / RAYLEIGH_MEDIAN,
                         size=db_image.shape)
    return 20.0 * np.log10(np.sqrt(envelope ** 2 + noise ** 2))


def uniform_scenes(count):
    grouped = collections.defaultdict(list)
    for path in find_shards():
        parts = path.stem.split("_")
        grouped[(parts[1], parts[2])].append(path)
    picked = [members for (phantom, _), members in sorted(grouped.items())
              if phantom == "uniform"]
    return picked[:count]


def question_one(shards, emit):
    """相对基准把频率依赖削弱了多少。"""
    rows = []
    for shard in shards:
        geometry = shard.geometry
        depth = (geometry.min_depth_mm
                 + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
        band = (depth >= TISSUE_DEPTH_MM[0]) & (depth <= TISSUE_DEPTH_MM[1])
        envelope = 10.0 ** (shard.db_image / 20.0)
        rows.append((round(shard.frequency_mhz, 2),
                     float(np.median(shard.db_image[band])),
                     20.0 * np.log10(float(np.sqrt(np.mean(envelope ** 2))))))

    frequencies = sorted({r[0] for r in rows})
    emit(u"=========== 1. which reference the noise level should use ===========")
    emit(u"  RELATIVE is what the generator does now: the floor tracks each line's own")
    emit(u"  radio-frequency root mean square, so a weak high-frequency acquisition also")
    emit(u"  gets weak noise. ABSOLUTE is what a real receiver does: the floor stays put.")
    emit(u"")
    emit(u"%-8s %14s %12s %14s %14s"
         % (u"freq", u"tissue 10-20mm", u"line rms", u"gap RELATIVE", u"gap ABSOLUTE"))
    relative, absolute = [], []
    for frequency in frequencies:
        pool = [r for r in rows if r[0] == frequency]
        tissue = float(np.mean([r[1] for r in pool]))
        line_rms = float(np.mean([r[2] for r in pool]))
        relative.append(tissue - line_rms)
        absolute.append(tissue)
        emit(u"%-8g %14.2f %12.2f %14.2f %14.2f"
             % (frequency, tissue, line_rms, tissue - line_rms, tissue))

    emit(u"")
    for name, values in [(u"relative", relative), (u"absolute", absolute)]:
        slope = float(np.polyfit(frequencies, values, 1)[0])
        emit(u"  %-9s %g to %g MHz : %7.2f -> %7.2f dB   slope %6.2f dB per MHz"
             % (name, frequencies[0], frequencies[-1], values[0], values[-1], slope))
    emit(u"  %-9s harmonic 4.4 to 5.7 MHz, measured               slope %6.2f dB per MHz"
         % (u"console", CONSOLE_SLOPE_DB_PER_MHZ))
    emit(u"")
    emit(u"  The absolute reference lands in the same range as the console. The relative")
    emit(u"  one is far too weak, so running a week with the code as it stands would buy")
    emit(u"  almost no frequency dependence - the whole point of adding noise.")
    emit(u"")
    emit(u"  Caveat worth keeping: the two do not decompose the same way. On the console")
    emit(u"  the tissue level falls 4.9 dB per MHz while the noise floor RISES 1.6, and")
    emit(u"  the two add to 8.24. In the simulation the whole 8.61 comes from the tissue")
    emit(u"  side, because an absolute floor is flat by construction. The agreement in")
    emit(u"  the total is partly coincidence; what is solid is the factor of seven")
    emit(u"  between the two reference choices.")


def question_two(shards, emit):
    """绝对噪声底取多少，约束才咬合得好。"""
    frequencies = sorted({round(s.frequency_mhz, 2) for s in shards})
    depths = sorted({round(s.geometry.depth_mm, 1) for s in shards})

    emit(u"")
    emit(u"=========== 2. what the absolute floor should be ===========")
    emit(u"  Highest transmit frequency whose penetration still covers the whole image.")
    emit(u"  That is the constrained label rule and it needs no weights. A row reading")
    emit(u"  the same value everywhere means the constraint never bites and the label")
    emit(u"  stays degenerate.")
    emit(u"")
    emit(u"%-10s %s" % (u"floor dB", u" ".join(u"%9s" % (u"%g mm" % d) for d in depths)))
    usable = {}
    for floor in FLOORS_DB:
        rng = np.random.RandomState(20260911)
        coverage = collections.defaultdict(list)
        for shard in shards:
            noisy = inject(shard.db_image, floor, rng)
            reach = FE.penetration_depth_mm(noisy, shard.geometry, floor)
            span = shard.geometry.depth_mm - shard.geometry.min_depth_mm
            coverage[(round(shard.frequency_mhz, 2),
                      round(shard.geometry.depth_mm, 1))].append(
                (reach - shard.geometry.min_depth_mm) / span)
        row, winners = [], set()
        for depth in depths:
            best = None
            for frequency in frequencies:
                pool = coverage.get((frequency, depth))
                if pool and float(np.mean(pool)) >= COVERED:
                    best = frequency
            row.append(u"%9s" % (u"%g" % best if best else u"none"))
            if best:
                winners.add(best)
        usable[floor] = winners
        emit(u"%-10g %s" % (floor, u" ".join(row)))

    emit(u"")
    emit(u"  frequencies the ladder actually uses, by floor")
    for floor in FLOORS_DB:
        emit(u"    %-8g %s" % (floor, sorted(usable[floor]) or u"none"))
    emit(u"")
    emit(u"  No single floor exercises all four frequencies, which is the argument for")
    emit(u"  the range the config already has: sample the floor per phantom and every")
    emit(u"  frequency becomes the right answer for some scene. The sampling happens in")
    emit(u"  randomized_acoustics, called once per (phantom type, seed) OUTSIDE the")
    emit(u"  frequency, focus and depth loops, so the floor is constant within a scene")
    emit(u"  family and front-end comparisons stay clean.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=3)
    args = parser.parse_args()

    lines = []
    emit = lines.append
    shards = [load_shard(p) for members in uniform_scenes(args.scenes)
              for p in members]
    emit(u"  %d uniform shards over %d scenes" % (len(shards), args.scenes))
    emit(u"")

    question_one(shards, emit)
    question_two(shards, emit)

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_noise_reference_choice.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
