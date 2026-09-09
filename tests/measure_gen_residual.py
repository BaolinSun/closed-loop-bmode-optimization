# -*- coding: utf-8 -*-
"""Why 20260903_GEN rebuilds worse than every other group.

Its per-pixel error is 11.0 while its band error is only 5.21 and 89% of its pixels are
unclipped, so it is not an exposure problem and it is not the fit failing to find a level.

That group is also the one with by far the widest sweep: 44 frames over six display depths and
five transmit frequencies, where 20260903 fundamental holds six frames at one depth and one
frequency. A calibration is fitted once per group, so if any of its constants actually moves
with depth or with frequency, this is the group where that would show.

This breaks the error down along both axes and, separately, checks whether the screenshot is
simply smoother than the rebuild - that would point at console post-processing rather than at
the calibration.

Usage: python tests/measure_gen_residual.py
"""

import collections
import io
import json
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import calibration as CAL
import display_palette as DP
import hisense_backend_sim as S
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, get_leaf, load_capture

CAL_PATH = "bmode_opt/console_calibration.json"
FUNDAMENTAL_SESSIONS = ["20260903_GEN", "20260904", "20260903"]


def gradient_energy(image):
    dy = np.diff(image, axis=0)[:, :-1]
    dx = np.diff(image, axis=1)[:-1, :]
    return float(np.sqrt(np.mean(dy ** 2 + dx ** 2)))


def main():
    lines = []
    emit = lines.append

    data = json.load(io.open(CAL_PATH, encoding="utf-8"))
    entries = {(g["session"], g["image_mode"]): g for g in data["groups"]}

    records = []
    for session in FUNDAMENTAL_SESSIONS:
        entry = entries.get((session, 0))
        if entry is None:
            continue
        captures = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        captures = [c for c in captures if T.capture_image_mode(c) == 0]
        if not captures:
            continue
        palette = DP.session_palette(captures)[0]
        calibration = CAL.GroupCalibration(
            entry["counts_per_db"], entry["pivot_db"], 0.0, 0,
            np.array(entry["depth_axis_mm"]), np.array(entry["depth_response_db"]),
            {float(k): np.array(v) for k, v in
             entry.get("depth_response_by_frequency", {}).items()})
        for capture in captures:
            actual = DP.capture_display_gray(capture, palette=palette)[0]
            predicted = S.render(
                capture.bc0, tgc_levels=capture.tgc_levels,
                gain_db=S.gain_level_to_db(capture.gain_level),
                dynamic_range_db=S.capture_window_db(capture),
                depth_response_db=CAL.depth_response_for(capture, calibration),
                reference_db=calibration.pivot_db,
                counts_per_db=calibration.counts_per_db,
                out_shape=actual.shape).astype(np.float64)
            try:
                frequency = round(float(get_leaf(capture.fe_params, "BFreqValue")), 1)
            except Exception:
                frequency = None
            records.append({
                "session": session,
                "depth": round(capture.geometry.depth_mm, 1),
                "frequency": frequency,
                "error": float(np.median(np.abs(actual - predicted))),
                "bias": float(np.median(actual - predicted)),
                "gradient_ratio": gradient_energy(predicted) / max(gradient_energy(actual), 1e-9),
            })

    emit("=========== per-pixel error by session ===========")
    emit("%-16s %7s %10s %10s %14s" % ("session", "frames", "error", "bias", "gradient ratio"))
    for session in FUNDAMENTAL_SESSIONS:
        subset = [r for r in records if r["session"] == session]
        if not subset:
            continue
        emit("%-16s %7d %10.1f %10.1f %14.2f" % (
            session, len(subset),
            np.median([r["error"] for r in subset]),
            np.median([r["bias"] for r in subset]),
            np.median([r["gradient_ratio"] for r in subset])))
    emit("  bias is the median signed difference: a rebuild that is uniformly too dark or too")
    emit("  bright shows up here, whereas the error column cannot tell that from texture.")
    emit("  gradient ratio below 1 means the rebuild is smoother than the screenshot.")

    subset = [r for r in records if r["session"] == "20260903_GEN"]
    for axis in ["depth", "frequency"]:
        emit("")
        emit("=========== 20260903_GEN error by %s ===========" % axis)
        emit("%-10s %7s %10s %10s %14s" % (axis, "frames", "error", "bias", "gradient ratio"))
        buckets = collections.defaultdict(list)
        for record in subset:
            buckets[record[axis]].append(record)
        for key in sorted(k for k in buckets if k is not None):
            group = buckets[key]
            emit("%-10s %7d %10.1f %10.1f %14.2f" % (
                key, len(group),
                np.median([r["error"] for r in group]),
                np.median([r["bias"] for r in group]),
                np.median([r["gradient_ratio"] for r in group])))

    emit("")
    emit("=========== the same two axes on the other fundamental sessions ===========")
    emit("  If the error only climbs with the spread of conditions, one calibration per group")
    emit("  is the thing that cannot hold, not that session's data.")
    emit("%-16s %10s %14s %14s" % ("session", "frames", "depths spanned", "freqs spanned"))
    for session in FUNDAMENTAL_SESSIONS:
        group = [r for r in records if r["session"] == session]
        if not group:
            continue
        emit("%-16s %10d %14d %14d" % (
            session, len(group), len({r["depth"] for r in group}),
            len({r["frequency"] for r in group})))

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_gen_residual.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
