# -*- coding: utf-8 -*-
"""前端目标函数的候选项有没有分辨力：在 Field II 上逐项量。

    为什么先量再建

后端目标函数吃过两次亏。「压黑」项在每个动态范围下都是 0.0000，因为信号掩膜和压黑
阈值只差 2.6 dB，凡是压黑的早就被当噪声排除了——18.7% 的像素是黑的，计入 0.0%。
gCNR 定不了动态范围，因为它对单调变换不变，而动态范围恰恰是单调变换，整个阶梯上
0.6996 到 0.6979 纹丝不动。两次都是先建后发现失效。

所以这次反过来：先问每个候选项在三根前端轴上动不动，再决定要不要它。

    分辨力怎么定义

一个判据有用，要同时满足两条：
  一、沿轴变化要大——不同频率下取值明显不同；
  二、跨种子要稳——同一个最优设置要在不同散斑实现下重复出现。
只满足第一条的是噪声放大器。所以报的是【argmin 一致率】：同一 (体模, 深度, 其余轴)
下换 20 个种子，argmin 落在同一档的比例。随机猜是 1/候选数。

    比较必须在各自的后端最优上

4.0 MHz 和 8.0 MHz 两帧若用同一增益比较，8.0 MHz 那帧一定更暗（衰减更大），于是
「看起来更差」——这跟频率选得对不对无关，纯粹是增益没跟上。所以 gCNR 这类要过显示
的量，每个成员先各自解一次后端最优再算。分辨率与 dB 域的量不过显示，直接在 db_image
上算，不需要解后端。

用法：python tests/measure_front_end_criteria.py [--seeds 4]
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

AXES = ("depth_mm", "frequency_mhz", "focus_mm")

# 点靶半高全宽在 -6 dB 处量（dB 图是 20log10(包络)，-6 dB 即半幅）。
FWHM_DROP_DB = 6.0
# 靶点周围取这么大的窗做侧向/轴向剖面。
TARGET_WINDOW_MM = (3.0, 3.0)


def shard_key(shard):
    return (shard.phantom_type, shard.seed)


def setting_of(shard):
    return (round(shard.geometry.depth_mm, 1), round(shard.frequency_mhz, 2),
            round(shard.focus_mm, 1))


def target_pixel(shard, depth_mm, lateral_mm):
    """靶点的 (行, 列)。超出视野返回 None。"""
    geometry = shard.geometry
    row = geometry.row_of_depth(depth_mm)
    half = geometry.width_mm / 2.0
    column = (lateral_mm + half) / geometry.mm_per_line
    if not (0 <= row < geometry.num_points and 0 <= column < geometry.num_lines):
        return None
    return int(round(row)), int(round(column))


def _width_at_drop(profile, spacing, peak_index):
    """剖面上 -6 dB 处的全宽，线性插值两侧交点。"""
    peak = profile[peak_index]
    level = peak - FWHM_DROP_DB
    left = right = None
    for k in range(peak_index, 0, -1):
        if profile[k - 1] <= level <= profile[k]:
            span = profile[k] - profile[k - 1]
            left = (k - 1) + (level - profile[k - 1]) / span if span else k
            break
    for k in range(peak_index, profile.size - 1):
        if profile[k + 1] <= level <= profile[k]:
            span = profile[k] - profile[k + 1]
            right = k + (profile[k] - level) / span if span else k
            break
    if left is None or right is None:
        return float("nan")
    return float((right - left) * spacing)


def point_resolution(shard):
    """视野内全部点靶的侧向与轴向半高全宽的中位数，单位 mm。"""
    db = shard.db_image
    geometry = shard.geometry
    lateral, axial = [], []
    for depth_mm, offset_mm in np.asarray(shard.point_targets_mm).reshape(-1, 2):
        found = target_pixel(shard, depth_mm, offset_mm)
        if found is None:
            continue
        row, column = found
        half_rows = max(2, int(TARGET_WINDOW_MM[0] / geometry.mm_per_point))
        half_cols = max(2, int(TARGET_WINDOW_MM[1] / geometry.mm_per_line))
        r0, r1 = max(0, row - half_rows), min(geometry.num_points, row + half_rows + 1)
        c0, c1 = max(0, column - half_cols), min(geometry.num_lines, column + half_cols + 1)
        patch = db[r0:r1, c0:c1]
        if patch.size < 9:
            continue
        local = np.unravel_index(np.argmax(patch), patch.shape)
        lateral.append(_width_at_drop(patch[local[0], :], geometry.mm_per_line, local[1]))
        axial.append(_width_at_drop(patch[:, local[1]], geometry.mm_per_point, local[0]))
    clean = lambda v: float(np.nanmedian(v)) if v and not np.all(np.isnan(v)) else float("nan")
    return clean(lateral), clean(axial)


def attenuation_slope_db_per_mm(shard, valid_mask):
    """组织电平随深度的斜率。频率越高越陡，这是穿透代价在 Field II 里的形态。

    Field II 的 noise_enabled 全是 0，没有噪声底，信号不会消失在噪声里。高频的代价
    因此不表现为「看不见了」，而表现为浅深之间的 dB 落差变大——增益抬深部就烧浅部。
    """
    levels = OBJ.band_levels_db(shard.db_image, valid_mask)
    if levels.size < 2 or np.all(np.isnan(levels)):
        return float("nan")
    geometry = shard.geometry
    depth = np.linspace(geometry.min_depth_mm, geometry.depth_mm, levels.size)
    good = ~np.isnan(levels)
    if good.sum() < 2:
        return float("nan")
    return float(np.polyfit(depth[good], levels[good], 1)[0])


def measure_shard(shard, target_gray, solve_backend):
    """一帧的全部候选判据。"""
    valid = T.fieldii_tissue_mask(shard)
    out = {}
    if valid.sum() >= 1000:
        out["uniformity_db"] = OBJ.depth_uniformity_db_cost(shard.db_image, valid)
        out["span_db"] = OBJ.signal_span_db(shard.db_image, valid)
        out["attenuation_db_per_mm"] = attenuation_slope_db_per_mm(shard, valid)

    if shard.phantom_type == "point" and len(shard.point_targets_mm):
        lateral, axial = point_resolution(shard)
        out["lateral_fwhm_mm"] = lateral
        out["axial_fwhm_mm"] = axial

    if shard.phantom_type == "cyst" and solve_backend and valid.sum() >= 1000:
        solution = BS.solve_backend(
            shard.db_image, valid, dr_ui=shard.dynamic_range_level,
            reference_db=shard.tissue_median_db, target_gray=target_gray,
            j_uncertainty=0.0, image_mode=0)
        gray = S.render(
            db_image=shard.db_image, tgc_levels=solution["tgc_levels"],
            gain_db=solution["gain_db"],
            dynamic_range_db=S.dr_ui_to_window_db(solution["dr_ui"]),
            reference_db=shard.tissue_median_db, depth_response_db=None,
            out_shape=shard.db_image.shape)
        lesion = shard.truth_mask == 1
        background = shard.truth_mask == 0
        out["gcnr_cyst"] = OBJ.gcnr(gray[lesion], gray[background])
        out["lesion_contrast_db"] = float(
            np.median(shard.db_image[background]) - np.median(shard.db_image[lesion]))
    return out


def agreement(records, axis, metric, prefer_low):
    """同一 (体模, 深度/其余轴) 下换种子，argmin 落在同一档的比例。

    比较集 = 除 axis 之外的前端设置都相同的一组帧。种子是散斑的随机实现，最优设置
    不该跟着种子跑。
    """
    others = [a for a in AXES if a != axis]
    grouped = collections.defaultdict(dict)
    for key, setting, values in records:
        if metric not in values or not np.isfinite(values[metric]):
            continue
        fixed = tuple(setting[AXES.index(a)] for a in others)
        grouped[(key[0], key[1], fixed)][setting[AXES.index(axis)]] = values[metric]

    winners = collections.defaultdict(list)
    sizes = []
    for (phantom, seed, fixed), members in grouped.items():
        if len(members) < 2:
            continue
        pick = (min if prefer_low else max)(members, key=lambda v: members[v])
        winners[(phantom, fixed)].append(pick)
        sizes.append(len(members))

    total, agreed, candidates = 0, 0, []
    for picks in winners.values():
        if len(picks) < 2:
            continue
        modal = collections.Counter(picks).most_common(1)[0]
        total += len(picks)
        agreed += modal[1]
        candidates.append(modal[0])
    if not total:
        return None
    return {"agreement": agreed / float(total), "n": total,
            "candidates": float(np.mean(sizes)) if sizes else 0.0,
            "modal_pick": collections.Counter(candidates).most_common(3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4,
                        help="how many seeds per phantom type to load")
    args = parser.parse_args()

    lines = []
    emit = lines.append

    paths = find_shards()
    by_scene = collections.defaultdict(list)
    for path in paths:
        parts = path.stem.split("_")
        by_scene[(parts[1], parts[2])].append(path)
    chosen = collections.defaultdict(list)
    for (phantom, seed), members in sorted(by_scene.items()):
        if len(chosen[phantom]) < args.seeds:
            chosen[phantom].append((seed, members))

    emit(u"=========== what was loaded ===========")
    records = []
    for phantom in sorted(chosen):
        for seed, members in chosen[phantom]:
            for path in members:
                shard = load_shard(path)
                values = measure_shard(shard, target_gray=64.0, solve_backend=True)
                records.append((shard_key(shard), setting_of(shard), values))
            emit(u"  %-8s %-12s %3d shards" % (phantom, seed, len(members)))
    emit(u"  %d shards total" % len(records))

    metrics = [(u"gcnr_cyst", False), (u"lesion_contrast_db", False),
               (u"lateral_fwhm_mm", True), (u"axial_fwhm_mm", True),
               (u"uniformity_db", True), (u"span_db", True),
               (u"attenuation_db_per_mm", False)]

    emit(u"")
    emit(u"=========== 1. does the metric move along the axis at all ===========")
    emit(u"  Mean over every scene, by axis setting. A flat row cannot choose anything.")
    for axis in AXES:
        emit(u"")
        emit(u"  --- %s ---" % axis)
        values = sorted({s[AXES.index(axis)] for _, s, _ in records})
        emit(u"  %-24s %s" % (u"metric", u" ".join(u"%9g" % v for v in values)))
        for metric, _ in metrics:
            row = []
            for value in values:
                pool = [v[metric] for _, s, v in records
                        if s[AXES.index(axis)] == value and metric in v
                        and np.isfinite(v[metric])]
                row.append(np.mean(pool) if pool else float("nan"))
            if all(np.isnan(x) for x in row):
                continue
            finite = [x for x in row if np.isfinite(x)]
            spread = (max(finite) - min(finite)) if finite else 0.0
            emit(u"  %-24s %s   spread %.4g"
                 % (metric, u" ".join(u"%9.4g" % x for x in row), spread))

    emit(u"")
    emit(u"=========== 2. is the winner stable across speckle seeds ===========")
    emit(u"  Same phantom and same other-axis settings, 20 different speckle seeds.")
    emit(u"  A metric that picks a different winner every seed is fitting noise.")
    emit(u"  Chance level is 1 / candidates.")
    emit(u"")
    emit(u"%-16s %-24s %8s %8s %10s %10s %s"
         % (u"axis", u"metric", u"sets", u"cands", u"chance", u"agreement", u"modal pick"))
    for axis in AXES:
        for metric, prefer_low in metrics:
            result = agreement(records, axis, metric, prefer_low)
            if result is None:
                continue
            chance = 1.0 / result["candidates"] if result["candidates"] else float("nan")
            emit(u"%-16s %-24s %8d %8.1f %10.3f %10.3f   %s"
                 % (axis, metric, result["n"], result["candidates"], chance,
                    result["agreement"],
                    u", ".join(u"%g x%d" % (v, n) for v, n in result["modal_pick"])))

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_front_end_criteria.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
