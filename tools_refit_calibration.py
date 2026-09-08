# -*- coding: utf-8 -*-
"""Refit every group's calibration against the console's own gray index.

Everything fitted so far was fitted against PIL's luma of a tinted screenshot. This refits it
against the index recovered through the display palette, then reports how much the rebuild
improved, and rewrites bmode_opt/console_calibration.json.

Run:  python tools_refit_calibration.py
"""
import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np

import calibration as CAL
import display_palette as DP
import hisense_backend_sim as S
import objective as OBJ
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

SESSIONS = ["20260814", "20260819", "20260831", "20260901", "20260901_E2",
            "20260901_E3", "20260903", "20260903_GEN",
            "20260903_replication_check", "20260904", "20260904_DR",
            "20260828/GEN", "20260828/THI"]
MODE_NAME = {0: "general", 1: "harmonic"}


def load_groups():
    groups = {}
    for session in SESSIONS:
        try:
            paths = find_captures(DEFAULT_DATA_DIR / session)
        except Exception:
            continue
        for path in paths:
            capture = load_capture(path)
            groups.setdefault((session, T.capture_image_mode(capture)), []).append(capture)
    return groups


def per_pixel_error(capture, counts_per_db, pivot_db, depth_response_db):
    """Median absolute error between the rebuild and the console's own gray index."""
    actual = DP.capture_display_gray(capture)[0]
    predicted = S.render(
        capture.bc0, tgc_levels=capture.tgc_levels,
        gain_db=S.gain_level_to_db(capture.gain_level),
        dynamic_range_db=S.capture_window_db(capture),
        depth_response_db=depth_response_db, reference_db=pivot_db,
        counts_per_db=counts_per_db, out_shape=actual.shape).astype(np.float64)
    return float(np.median(np.abs(actual - predicted)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bmode_opt/console_calibration.json")
    parser.add_argument("--fit-frames", type=int, default=16)
    args = parser.parse_args()

    started = time.time()
    groups = load_groups()
    print("groups: %d, captures: %d\n" % (len(groups), sum(len(v) for v in groups.values())))

    print("%-30s %-9s %6s %11s %10s %11s %13s" % (
        "session", "mode", "frames", "counts/dB", "pivot_dB", "band error", "per-pixel err"))
    calibrations, swept = {}, {}
    for key in sorted(groups):
        captures = groups[key]
        calibration = CAL.fit_group(captures, limit=min(args.fit_frames, len(captures)))
        if calibration is None:
            continue
        calibrations[key] = calibration
        swept[key] = len({(c.gain_level, c.tgc_levels[0], c.dynamic_range_level)
                          for c in captures}) >= 4
        check = [c for c in captures if S.is_flat_tgc(c)]
        check = check[::max(1, len(check) // 6)][:6]
        errors = [per_pixel_error(c, calibration.counts_per_db, calibration.pivot_db,
                                  CAL.depth_response_for(c, calibration)) for c in check]
        print("%-30s %-9s %6d %11.1f %10.2f %11.2f %13.1f" % (
            key[0][:28], MODE_NAME[key[1]], len(captures), calibration.counts_per_db,
            calibration.pivot_db, calibration.gray_error, float(np.mean(errors))))

    print("\nnoise floors and brightness targets")
    print("%-30s %-9s %12s %12s %14s" % (
        "session", "mode", "floor dB", "target gray", "uncertainty"))
    floors, targets, uncertainty = {}, {}, {}
    for key, calibration in sorted(calibrations.items()):
        captures = groups[key]
        measured = T.measure_noise_floor(
            captures, lambda c: S.bc0_to_db(c.bc0, calibration.counts_per_db))
        floors[key] = None if measured is None else measured["floor_db"]
    for key in calibrations:
        if floors.get(key) is None:
            donors = [v for k, v in floors.items() if k[1] == key[1] and v is not None]
            floors[key] = float(np.median(donors)) if donors else None

    for key, calibration in sorted(calibrations.items()):
        if floors.get(key) is None:
            continue
        captures = groups[key]
        result = T.measure_accepted_brightness(
            captures,
            lambda c: S.render(c.bc0, tgc_levels=c.tgc_levels,
                               gain_db=S.gain_level_to_db(c.gain_level),
                               dynamic_range_db=S.capture_window_db(c),
                               depth_response_db=CAL.depth_response_for(c, calibration),
                               reference_db=calibration.pivot_db,
                               counts_per_db=calibration.counts_per_db),
            lambda c: T.console_tissue_mask(
                S.bc0_to_db(c.bc0, calibration.counts_per_db), floors[key]))
        if result:
            targets[key] = result["target_gray"]

        if swept.get(key):
            screenshot, rebuilt = [], []
            for capture in captures:
                actual = DP.capture_display_gray(capture)[0]
                db = (S.bc0_to_db(capture.bc0, calibration.counts_per_db)
                      + CAL.depth_response_for(capture, calibration)[:, None])
                shaped_screen = S.scan_convert_linear(db, actual.shape[0], actual.shape[1])
                mask = T.console_tissue_mask(shaped_screen, floors[key])
                if mask.sum() < 1000:
                    continue
                rebuild = S.render(
                    db_image=db, tgc_levels=capture.tgc_levels,
                    gain_db=S.gain_level_to_db(capture.gain_level),
                    dynamic_range_db=S.capture_window_db(capture),
                    reference_db=calibration.pivot_db, depth_response_db=None,
                    out_shape=actual.shape)
                shaped = shaped_screen + S.expand_tgc_to_depth(
                    capture.tgc_levels, shaped_screen.shape[0])[:, None]
                screenshot.append(OBJ.backend_objective(actual, shaped, mask, ~mask,
                                                        target_gray=40.0))
                rebuilt.append(OBJ.backend_objective(rebuild, shaped, mask, ~mask,
                                                     target_gray=40.0))
            if len(screenshot) >= 4:
                uncertainty[key] = float(np.std(np.array(rebuilt) - np.array(screenshot)))
        print("%-30s %-9s %12s %12s %14s" % (
            key[0][:28], MODE_NAME[key[1]],
            "%.2f" % floors[key] if floors.get(key) is not None else "-",
            "%.0f" % targets[key] if key in targets else "-",
            "%.4f" % uncertainty[key] if key in uncertainty else "borrowed"))

    measured_values = list(uncertainty.values())
    for key in calibrations:
        if key in uncertainty:
            continue
        same = [v for k, v in uncertainty.items() if k[1] == key[1]]
        uncertainty[key] = float(np.median(same if same else measured_values or [0.1]))

    payload = {
        "note": ("Per (session, imaging mode) calibration, fitted against the console's own "
                 "gray index recovered through display_palette rather than PIL luma. "
                 "counts_per_db and pivot_db are meaningful only as a pair and only within "
                 "their own group. label_uncertainty is the standard deviation of "
                 "J(rebuild) - J(screenshot) after removing the constant bias."),
        "groups": [],
    }
    for key, calibration in sorted(calibrations.items()):
        payload["groups"].append({
            "session": key[0], "image_mode": int(key[1]),
            "image_mode_name": MODE_NAME[key[1]], "num_frames": len(groups[key]),
            "counts_per_db": round(calibration.counts_per_db, 2),
            "pivot_db": round(calibration.pivot_db, 3),
            "screenshot_gray_error": round(calibration.gray_error, 3),
            "noise_floor_db": (None if floors.get(key) is None
                               else round(floors[key], 3)),
            "noise_floor_measured": floors.get(key) is not None,
            "target_gray": (None if key not in targets else round(targets[key], 1)),
            "label_uncertainty": round(uncertainty[key], 4),
            "label_uncertainty_measured": bool(swept.get(key)),
            "depth_axis_mm": [round(float(v), 3) for v in calibration.depth_axis_mm],
            "depth_response_db": [round(float(v), 4) for v in calibration.depth_response_db],
        })
    io.open(args.out, "w", encoding="utf-8").write(
        json.dumps(payload, indent=2, ensure_ascii=False))
    print("\nwrote %s (%d groups) in %.0f s" % (args.out, len(payload["groups"]),
                                                time.time() - started))


if __name__ == "__main__":
    main()
