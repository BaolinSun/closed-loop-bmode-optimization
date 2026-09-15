# -*- coding: utf-8 -*-
"""Field II 带噪数据的 6 参数标签：后端（增益、TGC、动态范围）+ 前端（深度、频率、聚焦）。

写出 data/labels_fieldii.jsonl，每个分片一行，字段与 data/labels_console.jsonl 相同，另加
reference_db 等 Field II 专有字段。数据目录 data/field_ii/full_noise（2026-09-12 起的带噪
正式仿真）。

    与实机标签用同一套判据

前端三轴直接复用 tools_generate_console_labels 的求解函数（聚焦 -> 频率 -> 深度、6 dB
底部余量、分辨率表、可接受集合），只换掉三样与设备有关的输入：

  噪声底    实机是每个场次一个常数；Field II 由 fieldii_noise 按生成器参数逐行解析算出
            （接收孔径随深度变大，底噪随深度下降约 0.2 dB/mm）
  阶梯      Field II 的深度 25/30/35/42/50/60 mm、频率 4/5/6.5/8 MHz、聚焦 5 到 40 mm
  分辨率表  Field II 自己的，由无噪声旧数据集的点靶体模测得（point_targets.fieldii_resolution_table）

    2026-09-13 推迟、仿真完成后要做的两处修改，在这里落实

  组织掩膜排除噪声   tissue.fieldii_tissue_mask 传入逐行底噪，低于底噪 + 3 dB 的像素不算
                     组织。不排除时 8 MHz 分片的增益标签最多差 8.6 dB。
  散斑宽度过滤器     按 Field II 实测定为 0.75 线（front_end.FIELDII_SPECKLE_NOISE_WIDTH_LINES），
                     实机的 1.5 线会丢掉 24.6% 的真实组织带。

    后端

与旧的 tools_generate_labels.label_fieldii 相同：每帧解一次最优，按帧名为种子抽一个起点。
改动两处：组织掩膜排除噪声；曝光参考 reference_db 取组织像素的 dB 中位数（旧版取整幅中间
一半行的中位数，带噪数据里深部噪声会把它拉低），并写进标签，渲染时必须用同一个值。

    只标完整的体模

仿真仍在进行，正在生成的体模分片不全，比较集缺档会给出错误的条件最优。每个体模要有
全部档位组合（与已完成体模的组合数一致）才标。仿真结束后重跑即可覆盖全部 36 个体模。

用法：python tools_generate_fieldii_labels.py [--workers 8]
"""

import argparse
import collections
import io
import json
import os
import re
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "bmode_opt")
import numpy as np

import fieldii_noise as FN
import front_end as FE
import labels as LB
import point_targets as PT
import tissue as T
from fieldii_loader import find_shards, load_shard, parse_shard_name
import tools_generate_console_labels as G

DATA_DIR = "data/field_ii/full_noise"
POINT_TABLE_DIR = "data/field_ii/full"
RESOLUTION_CACHE = "bmode_opt/fieldii_resolution_table.json"
OUT_PATH = "data/labels_fieldii.jsonl"
REPORT_PATH = "tools_generate_fieldii_labels.txt"
FIELDII_MODE = 0                     # 线性仿真，对应基波成像
START_SEED = 20260909                # 与旧版 label_fieldii 相同，起点可复现
MIN_TISSUE_PIXELS = 1000


def shard_depth(path):
    return float("%s.%s" % re.search(r"_d(\d+)p(\d)", path.name).groups())


def phantom_key(path):
    meta = parse_shard_name(path)
    return meta["split"], meta["phantom_type"], meta["seed"]


def load_resolution_table():
    """Field II 分辨率表，缓存在 bmode_opt/fieldii_resolution_table.json。"""
    if os.path.exists(RESOLUTION_CACHE):
        data = json.load(io.open(RESOLUTION_CACHE, encoding="utf-8"))
    else:
        tables, pins, sets = PT.fieldii_resolution_table(POINT_TABLE_DIR, split="train")
        data = {"source": POINT_TABLE_DIR + " (train point phantoms, noiseless)",
                "created": time.strftime("%Y-%m-%d %H:%M"),
                "tables": [{"display_depth_mm": d, "scores": {"%g" % f: s for f, s in sc.items()},
                            "pins": pins[(m, d)], "comparison_sets": sets[(m, d)]}
                           for (m, d), sc in sorted(tables.items())]}
        io.open(RESOLUTION_CACHE, "w", encoding="utf-8").write(json.dumps(data, indent=2) + "\n")
    return data


def process_shard(args):
    """一个分片：后端标签 + 前端判据要用的逐帧量。在子进程里跑。"""
    path, target_gray = args
    shard = load_shard(path)
    db = shard.db_image
    floor = FN.noise_floor_db(shard)
    tissue = T.fieldii_tissue_mask(shard, floor_db=floor)
    setting = (FIELDII_MODE, round(shard_depth(path), 1), round(shard.frequency_mhz, 2),
               float(shard.focus_mm))
    measured = {
        "row_excess_db": G.row_excess_db(db, floor),
        "mm_per_point": float(shard.geometry.mm_per_point),
        "min_depth_mm": float(shard.geometry.min_depth_mm),
        "widths": FE.band_speckle_widths(db, shard.geometry, floor,
                                         noise_width_lines=FE.FIELDII_SPECKLE_NOISE_WIDTH_LINES),
        "display_depth_mm": float(setting[1]),
    }
    extra = {
        "noise_floor_top_db": float(floor[0]),
        "noise_floor_bottom_db": float(floor[-1]),
        "tissue_pixel_fraction": float(tissue.mean()),
        "electronic_noise_db": float(np.ravel(shard.attrs["electronic_noise_db"])[0]),
        "attenuation_db_cm_mhz": float(np.ravel(shard.attrs["attenuation_db_cm_mhz"])[0]),
        "sound_speed_mps": float(np.ravel(shard.attrs["sound_speed_mps"])[0]),
        "phantom_type": shard.phantom_type,
        "data_dir": DATA_DIR,
    }
    if tissue.sum() < MIN_TISSUE_PIXELS:
        return path.name, setting, None, measured, extra
    reference_db = float(np.median(db[tissue]))
    rng = np.random.RandomState((START_SEED + zlib.crc32(shard.name.encode("utf-8"))) % (2 ** 32))
    row = LB.label_frame(
        db, tissue, dr_ui=shard.dynamic_range_level, reference_db=reference_db,
        current=None, rng=rng, target_gray=target_gray, source="fieldii", frame_id=shard.name,
        group_id="%s/%s/%d" % phantom_key(path), imaging_mode="fundamental",
        depth_mm=setting[1], frequency_mhz=setting[2], focus_mm=setting[3],
        split=shard.split, label_uncertainty=0.0, image_mode=FIELDII_MODE,
        notes=["render is ground truth; no screenshot to match",
               "starting point drawn, not an operator's",
               "tissue mask excludes pixels within 3 dB of the modelled noise floor"]).as_dict()
    extra["reference_db"] = reference_db
    return path.name, setting, row, measured, extra


def console_target_gray():
    """与旧版相同：实机基波各组可接受组织灰阶的中位数。"""
    import tools_generate_labels as TG
    cal = TG.load_calibration()
    targets = TG.console_targets(cal)
    fundamental = [v for k, v in targets.items() if k[1] == 0]
    return float(np.median(fundamental)) if fundamental else 64.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-phantoms", type=int, default=None)
    args = parser.parse_args()
    started = time.time()
    lines = []
    emit = lines.append

    paths = find_shards(DATA_DIR)
    by_phantom = collections.defaultdict(list)
    for p in paths:
        by_phantom[phantom_key(p)].append(p)
    expected = max(len(v) for v in by_phantom.values())
    complete = {k: v for k, v in by_phantom.items() if len(v) == expected}
    skipped = {k: len(v) for k, v in by_phantom.items() if len(v) != expected}
    if args.limit_phantoms:
        complete = dict(sorted(complete.items())[:args.limit_phantoms])
    emit(u"=========== data ===========")
    emit(u"  %s: %d shards, %d phantoms; complete (%d shards each) %d; skipped as still generating: %s"
         % (DATA_DIR, len(paths), len(by_phantom), expected, len(complete),
            u", ".join(u"%s/%s/%d (%d)" % (k + (n,)) for k, n in sorted(skipped.items())) or u"none"))

    depths = sorted({round(shard_depth(p), 1) for v in complete.values() for p in v})
    frequencies = sorted({round(parse_shard_name(p)["frequency_mhz"], 2) for v in complete.values() for p in v})
    G.CONSOLE_LADDERS.clear()
    G.CONSOLE_LADDERS.update({(FIELDII_MODE, "depth_mm"): depths,
                              (FIELDII_MODE, "frequency_mhz"): frequencies})
    resolution = load_resolution_table()
    G.RESOLUTION.clear()
    for entry in resolution["tables"]:
        G.RESOLUTION[(FIELDII_MODE, entry["display_depth_mm"])] = {
            float(f): s for f, s in entry["scores"].items()}
    emit(u"  ladders: depth %s mm, frequency %s MHz" % (depths, frequencies))
    emit(u"")
    emit(u"=========== Field II resolution table (%s; lower = sharper) ===========" % resolution["source"])
    for entry in resolution["tables"]:
        s = {float(f): v for f, v in entry["scores"].items()}
        emit(u"  display %4.0f mm  sets %3d  pins %4d   %s   sharpest %g MHz"
             % (entry["display_depth_mm"], entry["comparison_sets"], entry["pins"],
                u"  ".join(u"%g:%.3f" % (f, s[f]) for f in sorted(s)), min(s, key=s.get)))

    target_gray = console_target_gray()
    emit(u"")
    emit(u"  back-end target gray (median of console fundamental groups): %.1f" % target_gray)

    jobs = [(p, target_gray) for v in complete.values() for p in v]
    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_shard, jobs, chunksize=4)):
                results.append(r)
                if (i + 1) % 200 == 0:
                    print("  %d / %d shards  (%.0f s)" % (i + 1, len(jobs), time.time() - started))
    else:
        results = [process_shard(j) for j in jobs]
    emit(u"  processed %d shards in %.0f s" % (len(results), time.time() - started))

    by_key = collections.defaultdict(list)
    for name, setting, row, measured, extra in results:
        by_key[phantom_key(Path(name))].append((name, setting, row, measured, extra))

    merged = []
    for key in sorted(by_key):
        items = sorted(by_key[key], key=lambda t: t[0])
        measured = {name: m for name, _, _, m, _ in items}
        family = SimpleNamespace(family_id="%s/%s/%d" % key,
                                 frame_names=[name for name, _, _, _, _ in items],
                                 settings=[setting for _, setting, _, _, _ in items],
                                 unbracketed=[], anchor_starved=False)
        front = G.frontend_labels(family, measured)
        for name, setting, row, _, extra in items:
            if row is None:
                row = {"source": "fieldii", "frame_id": name, "group_id": family.family_id,
                       "split": key[0], "depth_mm": setting[1], "frequency_mhz": setting[2],
                       "focus_mm": setting[3], "imaging_mode": "fundamental",
                       "backend_determined": False}
            else:
                row["backend_determined"] = True
            row["depth_mm"] = setting[1]
            row.update(extra)
            row.update(front[name])
            merged.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    emit(u"")
    emit(u"=========== coverage ===========")
    emit(u"  %d rows written to %s" % (len(merged), args.out))
    emit(u"  back-end determined %d" % sum(r.get("backend_determined", False) for r in merged))
    backend = [r for r in merged if r.get("backend_determined")]
    emit(u"  gain       %s" % dict(collections.Counter(r["gain_direction"] for r in backend)))
    for group in ("near", "mid", "far"):
        emit(u"  TGC %-6s %s" % (group, dict(collections.Counter(r["slider_directions"][group] for r in backend))))
    emit(u"  gain at search edge %d" % sum(r["at_gain_edge"] for r in backend))
    for stem, names in [("depth", G.DEPTH_DIRECTIONS), ("frequency", G.FREQUENCY_DIRECTIONS),
                        ("focus", G.FOCUS_DIRECTIONS)]:
        determined = [r for r in merged if r.get("%s_determined" % stem)]
        counts = collections.Counter(r["%s_direction" % stem] for r in determined)
        optimum = collections.Counter(r["optimal_%s_mm" % stem if stem != "frequency" else "optimal_frequency_mhz"]
                                      for r in determined)
        emit(u"  %-10s determined %4d / %d   %s   at edge %d   optimum %s"
             % (stem, len(determined), len(merged),
                u"  ".join(u"%s %d" % (n, counts.get(n, 0)) for n in names),
                sum(r["%s_at_edge" % stem] for r in determined), dict(sorted(optimum.items()))))
    freq = [r for r in merged if r.get("frequency_determined")]
    emit(u"  frequency  kind %s   confidence %s"
         % (dict(collections.Counter(r["frequency_label_kind"] for r in freq)),
            dict(collections.Counter(r["frequency_confidence"] for r in freq))))
    emit(u"  frequency  conditioned on %s"
         % dict(collections.Counter(r["frequency_conditioned_on"] for r in freq)))
    emit(u"  depth      conditioned on %s"
         % dict(collections.Counter(r["depth_conditioned_on"] for r in merged if r.get("depth_determined"))))
    emit(u"  dynamic range determined %d (no criterion; masked in training)"
         % sum(r.get("dr_determined", False) for r in merged))
    emit(u"")
    emit(u"done in %.0f s" % (time.time() - started))
    text = u"\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
