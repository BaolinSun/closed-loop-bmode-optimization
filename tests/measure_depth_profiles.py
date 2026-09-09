# -*- coding: utf-8 -*-
"""Tissue level against depth, Field II beside the console, on one plot.

Both sources attenuate. The question is how much of that attenuation is still present in the
data the back end sees, because that is what the TGC sliders are asked to undo.

Field II gives the beamformed envelope with nothing removed. The console's BC0 tap sits after
whatever depth gain the machine applies upstream of it, which is not operator controlled and
not visible in any parameter file. So the two arrive at the sliders with different amounts of
slope left in them, and a slider correction fitted on one does not mean the same thing on the
other.

Usage: python tests/measure_depth_profiles.py [--out docs/depth_profiles.png]
"""

import argparse
import io
import json
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import calibration as CAL
import display_palette as DP
import hisense_backend_sim as S
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

CAL_PATH = "bmode_opt/console_calibration.json"


def row_levels(db_image, mask, min_pixels=8):
    """Median dB of the tissue in each row, so the curve is the depth trend alone."""
    masked = np.ma.masked_array(db_image, mask=~mask)
    levels = np.ma.median(masked, axis=1).filled(np.nan)
    levels[mask.sum(axis=1) < min_pixels] = np.nan
    return levels


def anchor(depth_mm, levels, at_mm=15.0, window_mm=1.0):
    """The level at a common depth, so two curves can be laid over each other fairly."""
    near = np.isfinite(levels) & (np.abs(depth_mm - at_mm) <= window_mm)
    return float(np.median(levels[near])) if near.any() else float(np.nanmedian(levels))


def slope_over(depth_mm, levels, low, high):
    keep = np.isfinite(levels) & (depth_mm >= low) & (depth_mm <= high)
    if keep.sum() < 10:
        return float("nan")
    return float(np.polyfit(depth_mm[keep], levels[keep], 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/depth_profiles.png")
    args = parser.parse_args()

    lines, curves = [], []
    emit = lines.append
    emit("Tissue level against depth, every curve shifted to pass through 0 dB at 15 mm so the")
    emit("shapes can be laid over each other. Only the slope matters here.")
    emit("")
    emit("%-34s %10s %12s %12s %12s" % (
        "source", "depth mm", "slope 5-15mm", "slope 15-30", "total fall"))

    for phantom in ["uniform", "cyst", "point"]:
        for depth, freq in [(42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
            shard = load_shard(find_shards(phantom_type=phantom, depth_mm=depth,
                                           frequency_mhz=freq, focus_mm=15.0)[0])
            mask = T.fieldii_tissue_mask(shard)
            levels = row_levels(shard.db_image, mask)
            axis = (np.arange(levels.size) * shard.geometry.mm_per_point
                    + shard.geometry.min_depth_mm)
            base = anchor(axis, levels)
            label = "Field II %s %.0fmm %.1fMHz" % (phantom, depth, freq)
            curves.append((label, axis, levels - base, "tab:orange"))
            emit("%-34s %10.0f %12.3f %12.3f %12.1f" % (
                label, depth, slope_over(axis, levels, 5, 15),
                slope_over(axis, levels, 15, 30),
                np.nanmin(levels) - base))

    data = json.load(io.open(CAL_PATH, encoding="utf-8"))
    entries = {(g["session"], g["image_mode"]): g for g in data["groups"]
               if g.get("calibratable", True)}
    for session, mode in [("20260903", 1), ("20260903_GEN", 0), ("20260904", 0)]:
        entry = entries.get((session, mode))
        if entry is None:
            continue
        captures = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        captures = [c for c in captures if T.capture_image_mode(c) == mode]
        palette = DP.session_palette(captures)[0]
        calibration = CAL.GroupCalibration(
            entry["counts_per_db"], entry["pivot_db"], 0.0, 0,
            np.array(entry["depth_axis_mm"]), np.array(entry["depth_response_db"]),
            {float(k): np.array(v) for k, v in
             entry.get("depth_response_by_frequency", {}).items()})
        for capture in captures[:2]:
            db = (S.bc0_to_db(capture.bc0, calibration.counts_per_db)
                  + CAL.depth_response_for(capture, calibration)[:, None])
            mask = T.console_tissue_mask(db, entry["noise_floor_db"])
            levels = row_levels(db, mask)
            axis = np.linspace(0.0, capture.geometry.depth_mm, levels.size)
            if np.isfinite(levels).sum() < 20:
                continue
            base = anchor(axis, levels)
            label = "console %s %s" % (session[:14], entry["image_mode_name"])
            curves.append((label, axis, levels - base, "tab:blue"))
            emit("%-34s %10.0f %12.3f %12.3f %12.1f" % (
                label, capture.geometry.depth_mm, slope_over(axis, levels, 5, 15),
                slope_over(axis, levels, 15, 30), np.nanmin(levels) - base))

    emit("")
    emit("Both attenuate. Field II falls faster because nothing has compensated it; the")
    emit("console's BC0 already sits downstream of the machine's own depth gain. The slider")
    emit("range is 254 levels at %.5f dB, i.e. %.1f dB in total."
         % (S.DEFAULT_DB_PER_LEVEL, 254 * S.DEFAULT_DB_PER_LEVEL))

    figure, axis_plot = plt.subplots(figsize=(9, 5.5))
    seen = set()
    for label, depth_mm, level, colour in curves:
        show = label.split()[0]
        axis_plot.plot(depth_mm, level, color=colour, alpha=0.75, linewidth=1.3,
                       label=None if show in seen else show)
        seen.add(show)
    axis_plot.set_xlabel("depth (mm)")
    axis_plot.set_ylabel("tissue level, aligned at 15 mm (dB)")
    axis_plot.axvline(15.0, color="0.6", linestyle="--", linewidth=1)
    axis_plot.set_title("Depth trend of the tissue as the back end receives it")
    axis_plot.grid(alpha=0.3)
    axis_plot.legend()
    figure.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    figure.savefig(args.out, dpi=120)

    text = "\n".join(lines)
    out_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "measure_depth_profiles.txt")
    io.open(out_txt, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s and %s" % (args.out, out_txt))


if __name__ == "__main__":
    main()
