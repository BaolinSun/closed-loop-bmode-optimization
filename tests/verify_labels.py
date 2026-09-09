# -*- coding: utf-8 -*-
"""检查 data/labels_backend.jsonl：结构、覆盖、标签本身，以及两个来源可不可比。

最后一项才是要害。计划是 Field II 预训练、实机微调，那两边的 Δ 必须在同一个尺度上
——若仿真给的是 ±20 档而实机只有 ±3 档，预训练学到的响应幅度到微调时就是错的，
模型得先把学到的东西忘掉一半。

其余各项是常规体检：有没有字段缺失、有没有帧顶在搜索边界、某个轴是不是几乎全落在
同一类（那样的标签不携带信息）。

用法：python tests/verify_labels.py [--path data/labels_backend.jsonl]
"""

import argparse
import collections
import io
import json
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

from hisense_backend_sim import GAIN_DB_PER_LEVEL

DEFAULT_PATH = "data/labels_backend.jsonl"

# 某个方向轴上超过这个比例落在同一类，就等于没有标签。
DEGENERATE_FRACTION = 0.95


def load(path):
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentiles(values, label, unit=""):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "%-28s (none)" % label
    return ("%-28s n=%5d  p5 %8.2f  median %8.2f  p95 %8.2f  max|.| %8.2f %s"
            % (label, values.size, np.percentile(values, 5), np.median(values),
               np.percentile(values, 95), np.abs(values).max(), unit))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=DEFAULT_PATH)
    args = parser.parse_args()

    rows = load(args.path)
    emit = []
    problems = []

    def say(line=""):
        emit.append(line)

    say("=========== structure ===========")
    say("  rows: %d" % len(rows))
    by_source = collections.Counter(r["source"] for r in rows)
    for source, count in sorted(by_source.items()):
        say("  %-10s %d" % (source, count))

    ids = [r["frame_id"] for r in rows]
    duplicates = [k for k, v in collections.Counter(ids).items() if v > 1]
    if duplicates:
        problems.append("%d duplicate frame_id(s), e.g. %s"
                        % (len(duplicates), duplicates[:3]))

    required = ["source", "frame_id", "group_id", "depth_mm", "imaging_mode",
                "gain_db", "tgc_levels", "dr_ui", "optimal_gain_db", "optimal_tgc_levels",
                "gain_direction", "slider_directions", "dr_direction",
                "delta_gain_levels", "delta_tgc_levels", "delta_dr_ui",
                "objective", "tolerance", "label_uncertainty", "deadband_gain_levels",
                "equivalent_count", "dr_determined", "at_gain_edge"]
    missing = {key for r in rows for key in required if key not in r}
    if missing:
        problems.append("missing field(s): %s" % sorted(missing))

    numeric = ["delta_gain_levels", "objective", "tolerance", "deadband_gain_levels"]
    bad = [r["frame_id"] for r in rows
           if any(not np.isfinite(float(r[k])) for k in numeric)]
    if bad:
        problems.append("%d row(s) with a non-finite number, e.g. %s" % (len(bad), bad[:3]))

    if any(r["dr_determined"] for r in rows):
        problems.append("dr_determined is true somewhere; it should be false everywhere")
    if any(abs(float(r["delta_dr_ui"])) > 1e-9 for r in rows):
        problems.append("delta_dr_ui is non-zero somewhere; dynamic range is held")

    say()
    say("=========== coverage ===========")
    field = [r for r in rows if r["source"] == "fieldii"]
    console = [r for r in rows if r["source"] == "console"]

    say("  Field II")
    say("    splits      %s" % dict(collections.Counter(r["split"] for r in field)))
    phantom = collections.Counter(r["group_id"].split("/")[-1] for r in field)
    say("    phantoms    %s" % dict(phantom))
    say("    scenes      %d" % len({r["group_id"] for r in field}))
    for axis in ["depth_mm", "frequency_mhz", "focus_mm"]:
        say("    %-11s %s" % (axis, sorted({r[axis] for r in field})))

    say("  console")
    groups = collections.Counter(r["group_id"] for r in console)
    for key, count in sorted(groups.items()):
        say("    %-32s %3d" % (key, count))
    say("    depths      %s" % sorted({round(r["depth_mm"], 1) for r in console}))
    say("    modes       %s" % dict(collections.Counter(r["imaging_mode"] for r in console)))

    say()
    say("=========== direction labels ===========")
    for source, subset in [("fieldii", field), ("console", console)]:
        if not subset:
            continue
        say("  %s (%d rows)" % (source, len(subset)))
        counts = collections.Counter(r["gain_direction"] for r in subset)
        say("    %-14s %s" % ("gain", dict(counts)))
        top = counts.most_common(1)[0]
        if top[1] / len(subset) > DEGENERATE_FRACTION:
            problems.append("%s gain_direction is %.0f%% '%s' - no information"
                            % (source, 100 * top[1] / len(subset), top[0]))
        for band in ["near", "mid", "far"]:
            counts = collections.Counter(r["slider_directions"][band] for r in subset)
            say("    %-14s %s" % ("sliders " + band, dict(counts)))
            top = counts.most_common(1)[0]
            if top[1] / len(subset) > DEGENERATE_FRACTION:
                problems.append("%s slider_directions[%s] is %.0f%% '%s' - no information"
                                % (source, band, 100 * top[1] / len(subset), top[0]))
        say("    %-14s %s" % ("dynamic range",
                              dict(collections.Counter(r["dr_direction"] for r in subset))))

    say()
    say("=========== are the two sources on the same scale ===========")
    say("  This is what pretraining then fine-tuning depends on.")
    say()
    for source, subset in [("fieldii", field), ("console", console)]:
        if not subset:
            continue
        say("  " + percentiles([r["delta_gain_levels"] for r in subset],
                               "%s d(gain), clicks" % source))
        slider = [v for r in subset for v in r["delta_tgc_levels"]]
        say("  " + percentiles(slider, "%s d(slider), levels" % source))
        say("  " + percentiles([r["deadband_gain_levels"] for r in subset],
                               "%s deadband, clicks" % source))
        say("  " + percentiles([r["label_uncertainty"] for r in subset],
                               "%s label uncertainty" % source))
        say("  " + percentiles([r["equivalent_count"] for r in subset],
                               "%s equivalent set size" % source))
        say("  " + percentiles([r["objective"] for r in subset],
                               "%s objective" % source))
        say()

    say("=========== boundaries and flags ===========")
    for source, subset in [("fieldii", field), ("console", console)]:
        if not subset:
            continue
        edge = sum(1 for r in subset if r["at_gain_edge"])
        say("  %-10s at gain grid edge: %d / %d (%.1f%%)"
            % (source, edge, len(subset), 100.0 * edge / len(subset)))
        if edge > 0.05 * len(subset):
            problems.append("%s has %.0f%% of frames pinned at the gain grid edge"
                            % (source, 100.0 * edge / len(subset)))
    borrowed = collections.Counter(
        note for r in console for note in r.get("notes", []))
    say("  console notes:")
    for note, count in sorted(borrowed.items(), key=lambda kv: -kv[1]):
        say("    %4d  %s" % (count, note))

    say()
    say("=========== arithmetic and derived fields ===========")
    worst_gain = max(abs(r["delta_gain_levels"]
                         - (r["optimal_gain_db"] - r["gain_db"]) / GAIN_DB_PER_LEVEL)
                     for r in rows)
    worst_slider = max(
        float(np.abs(np.asarray(r["delta_tgc_levels"])
                     - (np.asarray(r["optimal_tgc_levels"], dtype=np.float64)
                        - np.asarray(r["tgc_levels"], dtype=np.float64))).max())
        for r in rows)
    say("  max |delta_gain   - (optimal - current)| = %.6f clicks" % worst_gain)
    say("  max |delta_slider - (optimal - current)| = %.6f levels" % worst_slider)
    if worst_gain > 1e-6 or worst_slider > 1e-6:
        problems.append("delta fields do not equal optimal minus current")

    mismatched = 0
    for r in rows:
        delta, band = r["delta_gain_levels"], r["deadband_gain_levels"]
        want = "correct" if abs(delta) <= band else ("dark" if delta > 0 else "bright")
        mismatched += want != r["gain_direction"]
    say("  gain_direction disagreeing with its own delta and deadband: %d" % mismatched)
    if mismatched:
        problems.append("%d gain_direction label(s) disagree with the delta" % mismatched)

    say()
    say("=========== split integrity (Field II) ===========")
    scenes = collections.defaultdict(set)
    for r in field:
        scenes[r["split"]].add(r["group_id"].split("/")[0])
    for first in ["train", "val", "test"]:
        for second in ["train", "val", "test"]:
            if first >= second:
                continue
            shared = scenes[first] & scenes[second]
            say("  %-6s (%2d scenes) vs %-5s (%2d): %d shared"
                % (first, len(scenes[first]), second, len(scenes[second]), len(shared)))
            if shared:
                problems.append("%s and %s share %d scene(s)"
                                % (first, second, len(shared)))

    say()
    say("=========== how much the optimum itself moves with the scene ===========")
    say("  reference_db is anchored per frame, so the optimum is a relative target and is")
    say("  expected to be nearly constant. The task the labels pose is the delta, which the")
    say("  model reads off the rendered image; this table is here so that stays visible.")
    for axis in ["depth_mm", "frequency_mhz"]:
        buckets = collections.defaultdict(list)
        for r in field:
            buckets[r[axis]].append(r["optimal_gain_db"])
        say("  optimal_gain_db by %s" % axis)
        for key in sorted(k for k in buckets if k is not None):
            values = np.asarray(buckets[key])
            say("    %-6s n=%4d  median %6.2f dB  p5 %6.2f  p95 %6.2f"
                % (key, values.size, np.median(values),
                   np.percentile(values, 5), np.percentile(values, 95)))

    say()
    say("=========== optimal sliders sitting at an end ===========")
    for source, subset in [("fieldii", field), ("console", console)]:
        if not subset:
            continue
        railed = sum(1 for r in subset
                     if min(r["optimal_tgc_levels"]) <= 1
                     or max(r["optimal_tgc_levels"]) >= 254)
        say("  %-10s %d / %d (%.0f%%)" % (source, railed, len(subset),
                                          100.0 * railed / len(subset)))
    say("  Field II runs high because its phantom falls about 25 dB over the display depth")
    say("  while the sliders span 254 x %.5f = %.1f dB. The console's BC0 arrives already"
        % (0.07734, 254 * 0.07734))
    say("  partly compensated, so its optimum lands inside the range more often.")

    say()
    say("=========== verdict ===========")
    if problems:
        for problem in problems:
            say("  PROBLEM  %s" % problem)
    else:
        say("  no structural problem found")

    text = "\n".join(emit)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_labels.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
