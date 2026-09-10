# -*- coding: utf-8 -*-
"""频率标签在 Field II 上定不下来，为什么，以及注入噪声能不能救回来。

    问题

measure_front_end_criteria 量出来两组判据都完美稳定，却指向相反的两端：
    分辨率（侧向半高全宽 1.076 -> 0.599 mm）  一致率 1.000  永远选 8 MHz
    衰减斜率（-0.719 -> -1.116 dB/mm）        一致率 1.000  永远选 4 MHz
两边都顶在量程端点，没有内部极值。频率的最优完全由两者的权重决定，而这个网格
定不出权重——这跟动态范围定不下来是同一类困境。

    为什么无噪声时必然如此

高频的代价在物理上是【穿透】：信号衰减更快，深部掉进噪声里就再也捞不出来。
Field II 数据集的 noise_enabled 全是 0，没有噪声底。没有噪声，TGC 就可以把深部
无代价地抬起来——抬多少都不会同时抬起噪声。于是高频只剩好处，最优必然railing到
最高频。衰减斜率之所以看着像个判据，是因为我在原始 db_image 上算的，没有加上
求解出的 TGC；而拉平深度衰减正是 TGC 的本职。

所以这个脚本查两件事：

一、【TGC 之后衰减代价还剩多少】把后端解出的 TGC 加上去再算深度不均匀度。若归零，
   说明无噪声下频率确实没有代价，前面那个「一致率 1.000」是假象。

二、【注入噪声能否恢复内部极值】按实机实测的组织-噪声间隔（谐波 6-20 dB、基波
   41-43 dB）往包络里加瑞利噪声，量【有效穿透深度】——组织仍高出噪声底若干 dB
   的最深处。有了穿透，频率的判据就不再需要权重，而是约束形式：
       在穿透能覆盖目标深度的频率里，挑分辨率最好的那个。
   这个形式天然有内部极值。

噪声按非相干叠加加进包络：env' = sqrt(env^2 + n^2)，n 服从瑞利分布，尺度取到
其中位数正好落在目标噪声底上。

用法：python tests/measure_frequency_tradeoff.py [--seeds 2]
"""

import argparse
import collections
import io
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import backend_solver as BS
import hisense_backend_sim as S
import objective as OBJ
import tissue as T
from fieldii_loader import find_shards, load_shard

# 实机实测的组织电平与噪声底之差，取自 console_calibration.json。谐波 6.2-20.4，
# 基波 41.3-43.1。扫这一串是为了看穿透在哪个量级上开始咬住频率。
NOISE_GAPS_DB = [None, 60.0, 50.0, 40.0, 30.0, 20.0]

# 组织高出噪声底这么多才算「还看得见」。3 dB 是 console_tissue_mask 用的余量。
PENETRATION_MARGIN_DB = 3.0

RAYLEIGH_MEDIAN = np.sqrt(2.0 * np.log(2.0))     # 1.1774


def inject_noise(db_image, floor_db, rng):
    """把噪声底加进包络。返回新的 dB 图。"""
    envelope = 10.0 ** (db_image / 20.0)
    scale = 10.0 ** (floor_db / 20.0) / RAYLEIGH_MEDIAN
    noise = rng.rayleigh(scale, size=db_image.shape)
    return 20.0 * np.log10(np.sqrt(envelope ** 2 + noise ** 2))


def penetration_depth_mm(db_image, geometry, floor_db, margin_db=PENETRATION_MARGIN_DB):
    """组织仍高出噪声底 margin_db 的最深处。

    逐行取该深度上的中位电平——中位数对散斑起伏稳健，对点靶也不敏感。从最深处
    往上找第一行越过门槛的位置。
    """
    rows = np.median(db_image, axis=1)
    good = np.where(rows >= floor_db + margin_db)[0]
    if good.size == 0:
        return float(geometry.min_depth_mm)
    return float(geometry.min_depth_mm + (good[-1] + 0.5) * geometry.mm_per_point)


def shaped_uniformity(shard, db_image, valid, solution):
    """加上后端解出的 TGC 之后，深度不均匀度还剩多少。"""
    shaped = S.apply_tgc(db_image, solution["tgc_levels"],
                         db_per_level=S.tgc_db_per_level(0))
    return OBJ.depth_uniformity_db_cost(shaped, valid)


def measure(shard, gap_db, rng, target_gray=64.0):
    db_image = shard.db_image
    floor_db = None
    if gap_db is not None:
        floor_db = shard.tissue_median_db - gap_db
        db_image = inject_noise(db_image, floor_db, rng)

    valid = T.fieldii_tissue_mask(shard)
    if valid.sum() < 1000:
        return None
    solution = BS.solve_backend(
        db_image, valid, dr_ui=shard.dynamic_range_level,
        reference_db=shard.tissue_median_db, target_gray=target_gray,
        j_uncertainty=0.0, image_mode=0)

    out = {
        "uniformity_raw_db": OBJ.depth_uniformity_db_cost(db_image, valid),
        "uniformity_after_tgc_db": shaped_uniformity(shard, db_image, valid, solution),
    }
    if floor_db is not None:
        reach = penetration_depth_mm(db_image, shard.geometry, floor_db)
        out["penetration_mm"] = reach
        # 视野有多少被穿透覆盖。1.0 表示到底部都还有信号，高频会掉下来。
        out["covered_fraction"] = ((reach - shard.geometry.min_depth_mm)
                                   / (shard.geometry.depth_mm - shard.geometry.min_depth_mm))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    lines = []
    emit = lines.append

    by_scene = collections.defaultdict(list)
    for path in find_shards():
        parts = path.stem.split("_")
        by_scene[(parts[1], parts[2])].append(path)
    chosen = collections.defaultdict(list)
    for (phantom, seed), members in sorted(by_scene.items()):
        if phantom == "uniform" and len(chosen[phantom]) < args.seeds:
            chosen[phantom].append(members)

    shards = [load_shard(p) for members in chosen["uniform"] for p in members]
    emit(u"=========== loaded ===========")
    emit(u"  %d uniform shards over %d seeds" % (len(shards), len(chosen["uniform"])))
    emit(u"  Uniform phantoms only: penetration is about the tissue level against the")
    emit(u"  noise floor, and a cyst or a point target would only get in the way.")

    frequencies = sorted({round(s.frequency_mhz, 2) for s in shards})
    depths = sorted({round(s.geometry.depth_mm, 1) for s in shards})

    results = {}
    for gap in NOISE_GAPS_DB:
        rng = np.random.RandomState(20260910)
        rows = []
        for shard in shards:
            values = measure(shard, gap, rng)
            if values:
                rows.append((round(shard.frequency_mhz, 2),
                             round(shard.geometry.depth_mm, 1), values))
        results[gap] = rows

    emit(u"")
    emit(u"=========== 1. does the TGC absorb the attenuation cost ===========")
    emit(u"  Depth non-uniformity in dB, before and after the solved TGC curve is applied.")
    emit(u"  If the after column is flat across frequency, the simulation charges high")
    emit(u"  frequency nothing and the frequency label cannot be determined from it.")
    emit(u"")
    emit(u"%-10s %14s %14s" % (u"frequency", u"raw", u"after TGC"))
    for frequency in frequencies:
        raw = [v["uniformity_raw_db"] for f, _, v in results[None] if f == frequency]
        after = [v["uniformity_after_tgc_db"] for f, _, v in results[None] if f == frequency]
        emit(u"%-10.2f %14.4f %14.4f"
             % (frequency, float(np.nanmean(raw)), float(np.nanmean(after))))
    raw_spread = max(float(np.nanmean([v["uniformity_raw_db"]
                                       for f, _, v in results[None] if f == x]))
                     for x in frequencies) - \
        min(float(np.nanmean([v["uniformity_raw_db"]
                              for f, _, v in results[None] if f == x]))
            for x in frequencies)
    after_values = [float(np.nanmean([v["uniformity_after_tgc_db"]
                                      for f, _, v in results[None] if f == x]))
                    for x in frequencies]
    emit(u"  spread across frequency:  raw %.4f    after TGC %.4f"
         % (raw_spread, max(after_values) - min(after_values)))

    emit(u"")
    emit(u"=========== 2. penetration once a noise floor exists ===========")
    emit(u"  Deepest place the tissue still stands %.0f dB above the noise floor, in mm."
         % PENETRATION_MARGIN_DB)
    emit(u"  The gap is the tissue-to-noise separation the floor was set to; the console")
    emit(u"  measures 6-20 dB in harmonic and 41-43 dB in fundamental.")
    emit(u"")
    emit(u"%-12s %s" % (u"gap dB", u" ".join(u"%12s" % (u"%g MHz" % f) for f in frequencies)))
    for gap in NOISE_GAPS_DB:
        if gap is None:
            continue
        row = []
        for frequency in frequencies:
            pool = [v["penetration_mm"] for f, _, v in results[gap] if f == frequency]
            row.append(float(np.mean(pool)) if pool else float("nan"))
        emit(u"%-12g %s" % (gap, u" ".join(u"%12.1f" % x for x in row)))

    emit(u"")
    emit(u"  the same as a fraction of the displayed depth")
    emit(u"%-12s %s" % (u"gap dB", u" ".join(u"%12s" % (u"%g MHz" % f) for f in frequencies)))
    for gap in NOISE_GAPS_DB:
        if gap is None:
            continue
        row = []
        for frequency in frequencies:
            pool = [v["covered_fraction"] for f, _, v in results[gap] if f == frequency]
            row.append(float(np.mean(pool)) if pool else float("nan"))
        emit(u"%-12g %s" % (gap, u" ".join(u"%12.3f" % x for x in row)))

    emit(u"")
    emit(u"=========== 3. does the constraint bite ===========")
    emit(u"  For each display depth, the highest frequency whose penetration still covers")
    emit(u"  the whole image. That is the constrained label rule, and it needs no weights.")
    emit(u"  A column reading %g MHz everywhere means the constraint never bites and the"
         % frequencies[-1])
    emit(u"  optimum still rails at the top of the ladder.")
    emit(u"")
    emit(u"%-12s %s" % (u"gap dB", u" ".join(u"%10s" % (u"%g mm" % d) for d in depths)))
    for gap in NOISE_GAPS_DB:
        if gap is None:
            continue
        row = []
        for depth in depths:
            best = None
            for frequency in frequencies:
                pool = [v["covered_fraction"] for f, d, v in results[gap]
                        if f == frequency and d == depth]
                if pool and float(np.mean(pool)) >= 0.995:
                    best = frequency
            row.append(u"%10s" % (u"%g" % best if best else u"none"))
        emit(u"%-12g %s" % (gap, u" ".join(row)))

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_frequency_tradeoff.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
