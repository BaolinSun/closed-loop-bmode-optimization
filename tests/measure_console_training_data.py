# -*- coding: utf-8 -*-
"""实机侧数据能为六参数训练提供多少监督。

    为什么行数不是答案

data/labels_console.jsonl 有 422 行，但行与行高度相关：

  后端  BC0 不随增益、TGC、动态范围变化，所以同一族、同一前端设置、同一动态范围的帧，
        对后端求解器是同一个场景，最优解相同。重复帧不增加信息。
  前端  同一比较集（族、模式、其余两根轴）里所有行共用一个条件最优。监督的单位是集，
        不是行。
  切分  同一族的帧共享探头位置，必须整族进同一个集合，否则测试集泄漏。

这个脚本报告五件事：

  A. 独立性：场次、族、每族帧数的集中程度。整个实机侧只有一个体模。
  B. 后端：不同场景数、标签在重复帧间的可重复性、每个场景的起点多样性。
  C. 类别均衡：每根轴的方向分布与最优值集中度。
  D. 前端：比较集数，以及按保守条件过滤后剩多少。
  E. 按族切分之后，测试集里还剩几个比较集。
  F. 频率标签规则的物理前提——「频率越高分辨率越好」——在实机 BC0 上是否成立。

用法：python tests/measure_console_training_data.py
"""

import collections
import io
import json
import os
import random
import sys

sys.path.insert(0, "bmode_opt")
# 工程根目录：F 节要导入根目录下的 tools_generate_*。脚本在 tests/ 里运行时它不在路径上。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

LABELS = "data/labels_console.jsonl"
SPLIT_TRIALS = 500
SPLIT_SEED = 20260914

AXES = [("depth", "depth_mm", ["frequency_mhz", "focus_mm"]),
        ("frequency", "frequency_mhz", ["depth_mm", "focus_mm"]),
        ("focus", "focus_mm", ["depth_mm", "frequency_mhz"])]


def family(row):
    """family_id 会跨模式重名（20260903/0 与 20260904/0 两种模式都有），必须带上模式。"""
    return "%s|%s" % (row["family_id"], row["imaging_mode"])


def independence(rows, emit):
    emit(u"=========== A. how independent is the console data ===========")
    families = collections.Counter(family(r) for r in rows)
    sessions = {r["group_id"].rsplit("/", 1)[0] for r in rows}
    sizes = sorted(families.values(), reverse=True)
    emit(u"  frames %d   sessions %d   scene families (with mode) %d   phantoms 1"
         % (len(rows), len(sessions), len(families)))
    emit(u"  frames per family: %s" % sizes)
    emit(u"  largest family %.0f%% of all frames, top three %.0f%%"
         % (100.0 * sizes[0] / len(rows), 100.0 * sum(sizes[:3]) / len(rows)))
    for mode in ("harmonic", "fundamental"):
        emit(u"  %-11s families %2d, frames %3d"
             % (mode, len({family(r) for r in rows if r["imaging_mode"] == mode}),
                sum(r["imaging_mode"] == mode for r in rows)))


def backend(rows, emit):
    emit(u"")
    emit(u"=========== B. back-end: distinct scenes and label repeatability ===========")
    key = lambda r: (family(r), r["depth_mm"], r["frequency_mhz"], r["focus_mm"], r["dr_ui"])
    scenes = collections.defaultdict(list)
    for r in rows:
        scenes[key(r)].append(r)
    repeated = [v for v in scenes.values() if len(v) > 1]
    emit(u"  distinct back-end scenes %d from %d frames" % (len(scenes), len(rows)))
    for mode in ("harmonic", "fundamental"):
        emit(u"    %-11s %d" % (mode, sum(1 for k in scenes if k[0].endswith(mode))))
    gain_sd = [np.std([x["optimal_gain_db"] for x in v]) for v in repeated]
    tgc = [np.array([x["optimal_tgc_levels"] for x in v]) for v in repeated]
    tgc_dev = [np.mean(np.abs(t - np.median(t, axis=0))) for t in tgc]
    starts = [len({(x["gain_db"], tuple(x["tgc_levels"])) for x in v}) for v in repeated]
    typical = np.median(np.abs([r["delta_gain_db"] for r in rows]))
    emit(u"  scenes seen more than once: %d" % len(repeated))
    emit(u"    optimal gain sd across repeats  median %.2f dB  p90 %.2f dB"
         % (np.median(gain_sd), np.percentile(gain_sd, 90)))
    emit(u"    optimal TGC abs dev             median %.1f   p90 %.1f levels"
         % (np.median(tgc_dev), np.percentile(tgc_dev, 90)))
    emit(u"    typical |delta gain| %.2f dB, so repeat noise is %.0f%% of it"
         % (typical, 100.0 * np.median(gain_sd) / typical))
    emit(u"  distinct starting back-ends per repeated scene: median %d, max %d"
         % (np.median(starts), max(starts)))
    rich = collections.Counter()
    for k, v in scenes.items():
        if len({(x["gain_db"], tuple(x["tgc_levels"])) for x in v}) >= 4:
            rich[k[0].split("|")[0].rsplit("/", 1)[0]] += 1
    emit(u"  scenes captured at >= 4 different back-ends: %d  %s"
         % (sum(rich.values()), dict(rich)))


def balance(rows, emit):
    emit(u"")
    emit(u"=========== C. class balance ===========")
    for group in ("near", "mid", "far"):
        c = collections.Counter(r["slider_directions"][group] for r in rows)
        emit(u"  TGC %-4s low %3d  correct %3d  high %3d   majority class %.1f%%"
             % (group, c["low"], c["correct"], c["high"], 100.0 * max(c.values()) / len(rows)))
    c = collections.Counter(r["gain_direction"] for r in rows)
    emit(u"  gain     dark %3d  correct %3d  bright %3d" % (c["dark"], c["correct"], c["bright"]))
    for stem, field in [("depth", "optimal_depth_mm"), ("frequency", "optimal_frequency_mhz"),
                        ("focus", "optimal_focus_mm")]:
        d = [r for r in rows if r["%s_determined" % stem]]
        value, count = collections.Counter(r[field] for r in d).most_common(1)[0]
        emit(u"  %-9s most common optimum %g on %d/%d rows (%.1f%%)"
             % (stem, value, count, len(d), 100.0 * count / len(d)))


def frontend(rows, emit):
    emit(u"")
    emit(u"=========== D. front-end comparison sets ===========")
    emit(u"  filters: not at edge, bracketed by anchors, family not anchor starved;")
    emit(u"  depth additionally requires that the recommended depth was actually captured")
    emit(u"  in that set (otherwise it is inferred framing, not a measured comparison).")
    emit(u"")
    emit(u"%-10s %-11s %6s %6s %12s %12s" % (u"axis", u"mode", u"rows", u"sets",
                                             u"kept sets", u"kept rows"))
    kept = {}
    for stem, field, others in AXES:
        for mode in ("harmonic", "fundamental"):
            rs = [r for r in rows if r["%s_determined" % stem] and r["imaging_mode"] == mode]
            sets = collections.defaultdict(list)
            for r in rs:
                sets[(family(r),) + tuple(r[k] for k in others)].append(r)
            good = {}
            for k, v in sets.items():
                ok = [r for r in v if not r["%s_at_edge" % stem]
                      and not r["family_unbracketed"] and not r["family_anchor_starved"]]
                if stem == "depth":
                    captured = {x[field] for x in v}
                    ok = [r for r in ok if r["optimal_depth_mm"] in captured]
                if ok:
                    good[k] = ok
            kept[(stem, mode)] = good
            emit(u"%-10s %-11s %6d %6d %12d %12d"
                 % (stem, mode, len(rs), len(sets), len(good),
                    sum(len(v) for v in good.values())))
    return kept


def splits(rows, kept, emit):
    emit(u"")
    emit(u"=========== E. comparison sets left after a family-level split ===========")
    emit(u"  %d random 60/20/20 splits by family, per mode. Reported as p10 / median / p90."
         % SPLIT_TRIALS)
    families = sorted({family(r) for r in rows})
    tally = collections.defaultdict(lambda: collections.defaultdict(list))
    rng = random.Random(SPLIT_SEED)
    for _ in range(SPLIT_TRIALS):
        for mode in ("harmonic", "fundamental"):
            pool = [f for f in families if f.endswith(mode)]
            rng.shuffle(pool)
            size = max(1, round(0.2 * len(pool)))
            test, val = set(pool[:size]), set(pool[size:2 * size])
            for stem, _, _ in AXES:
                sets = kept[(stem, mode)]
                tally[(stem, mode)]["test"].append(sum(1 for k in sets if k[0] in test))
                tally[(stem, mode)]["val"].append(sum(1 for k in sets if k[0] in val))
    emit(u"")
    emit(u"%-10s %-11s %20s %20s %18s" % (u"axis", u"mode", u"test sets", u"val sets",
                                          u"P(test empty)"))
    for stem, _, _ in AXES:
        for mode in ("harmonic", "fundamental"):
            t = np.array(tally[(stem, mode)]["test"])
            v = np.array(tally[(stem, mode)]["val"])
            emit(u"%-10s %-11s %8d / %3d / %3d %8d / %3d / %3d %17.0f%%"
                 % (stem, mode, np.percentile(t, 10), np.median(t), np.percentile(t, 90),
                    np.percentile(v, 10), np.median(v), np.percentile(v, 90),
                    100.0 * np.mean(t == 0)))
    counts = collections.Counter(f.split("|")[1] for f in families)
    emit(u"  families available: harmonic %d, fundamental %d"
         % (counts["harmonic"], counts["fundamental"]))


def frequency_premise(emit):
    """频率规则假设高频分辨率更好。在 E8 的 67 mm 帧上，于各频率都有信号的深度带里量。"""
    import calibration as CAL
    import front_end as FE
    import hisense_backend_sim as S
    import tools_generate_console_labels as G
    import tools_generate_labels as TG
    from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

    emit(u"")
    emit(u"=========== F. does resolution improve with frequency on the console ===========")
    emit(u"  The frequency rule picks the highest frequency that still penetrates, which")
    emit(u"  assumes higher frequency resolves better. That was shown in Field II (lateral")
    emit(u"  FWHM 1.076 -> 0.599 mm from 4 to 8 MHz), never on the console. E8, 67 mm,")
    emit(u"  focus 15 mm; speckle width in bands 10-30 mm where every frequency has signal.")
    calibrations = TG.load_calibration()
    G.floors_in_counts(calibrations)
    for session, mode in [("20260911_E8_GEN", 0), ("20260911_E8_THI", 1)]:
        entry = calibrations[(session, mode)]
        captures = [c for c in (load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session))
                    if abs(c.geometry.depth_mm - 67.0) < 1.0]
        lateral = collections.defaultdict(list)
        axial = collections.defaultdict(list)
        for capture in captures:
            db = S.bc0_to_db(capture.bc0, entry["cal"].counts_per_db)
            geometry = capture.geometry
            depth = (geometry.min_depth_mm
                     + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
            band = (depth >= 10.0) & (depth <= 30.0)
            envelope = 10.0 ** (db[band, :] / 20.0)
            frequency = round(CAL.capture_frequency(capture), 2)
            lateral[frequency].append(FE.lateral_speckle_width_mm(envelope, geometry.mm_per_line))
            axial[frequency].append(FE.lateral_speckle_width_mm(envelope.T, geometry.mm_per_point))
        freqs = sorted(lateral)
        lat = [np.nanmean(lateral[f]) for f in freqs]
        ax = [np.nanmean(axial[f]) for f in freqs]
        emit(u"")
        emit(u"  %s" % session)
        emit(u"    %-9s %s" % (u"freq MHz", u" ".join(u"%7g" % f for f in freqs)))
        emit(u"    %-9s %s   %+.1f%%" % (u"lateral", u" ".join(u"%7.3f" % v for v in lat),
                                        100.0 * (lat[-1] / lat[0] - 1)))
        emit(u"    %-9s %s   %+.1f%%   (ideal 1/f: %+.1f%%)"
             % (u"axial", u" ".join(u"%7.3f" % v for v in ax),
                100.0 * (ax[-1] / ax[0] - 1), 100.0 * (freqs[0] / freqs[-1] - 1)))


def main():
    lines = []
    emit = lines.append
    rows = [json.loads(l) for l in io.open(LABELS, encoding="utf-8")]
    independence(rows, emit)
    backend(rows, emit)
    balance(rows, emit)
    kept = frontend(rows, emit)
    splits(rows, kept, emit)
    frequency_premise(emit)
    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_console_training_data.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
