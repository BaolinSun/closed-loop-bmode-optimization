# -*- coding: utf-8 -*-
"""Refit every group's calibration against the console's own gray index.

Everything fitted so far was fitted against PIL's luma of a tinted screenshot. This refits it
against the index recovered through the display palette, then reports how much the rebuild
improved, and rewrites the calibration file of the chosen dataset.

Sessions are discovered (bmode_opt/console_sessions.py), not listed here:
  --dataset phantom   every phantom session found -> bmode_opt/console_calibration.json
  --dataset invivo    every in-vivo carotid subject in the manifest
                      -> bmode_opt/console_calibration_invivo_carotid.json

Groups are independent, so they are fitted in parallel worker processes (--workers).
A group without a frame deep enough to see the noise floor borrows one from a group of the
same imaging mode and the same dataset; phantom and in-vivo floors are never pooled (the
2026-10 firmware sits about 8-9 dB lower in BC0 counts).

Run:  python tools_refit_calibration.py [--dataset invivo] [--sessions S1,S2]
"""
import argparse
import concurrent.futures
import io
import json
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np

import calibration as CAL
import console_sessions as CS
import display_palette as DP
import hisense_backend_sim as S
import objective as OBJ
import tissue as T
from hisense_loader import image_fits_window, load_capture

MODE_NAME = {0: "fundamental", 1: "harmonic"}


def load_session_groups(session):
    """{(session, mode): [captures]} for one session."""
    groups = {}
    for path in CS.session_captures(session):
        capture = load_capture(path)
        groups.setdefault((session, T.capture_image_mode(capture)), []).append(capture)
    return groups


def per_pixel_error(capture, counts_per_db, pivot_db, depth_response_db,
                    palette=None):
    """Median absolute error between the rebuild and the console's own gray index."""
    actual = DP.capture_display_gray(capture, palette=palette)[0]
    predicted = S.render(
        capture.bc0, tgc_levels=capture.tgc_levels,
        gain_db=S.capture_gain_db(capture),
        dynamic_range_db=S.capture_window_db(capture),
        depth_response_db=depth_response_db, reference_db=pivot_db,
        counts_per_db=counts_per_db, out_shape=actual.shape).astype(np.float64)
    return float(np.median(np.abs(actual - predicted)))


def fit_one(captures, fit_frames):
    """Palette, fit, rebuild error and measured noise floor of one group."""
    # One palette per group, taken as the median over the frames whose colour bar reads as
    # a ramp. Reading it per frame silently misfires: six frames across the archive return
    # a bar that is flat for most of its length, and on 20260819 that made two frames read
    # 33 dB too bright and poisoned the whole group's fit.
    palette, _, rejected = DP.session_palette(captures)
    calibration = CAL.fit_group(captures, limit=min(fit_frames, len(captures)),
                                palette=palette)
    if calibration is None:
        return {"rejected": rejected, "calibration": None}
    # 显示放大（2026-10-09/10 人体采集用过 1.4-1.8）让图像超出 B 窗口时截图只有一部分，
    # 与整幅 BC0 比不了；拟合本身已经跳过这些帧（calibration._select_fit_frames），这里同样跳过。
    check = [c for c in captures if S.is_flat_tgc(c) and image_fits_window(c.geometry)]
    check = check[::max(1, len(check) // 6)][:6]
    errors = [per_pixel_error(c, calibration.counts_per_db, calibration.pivot_db,
                              CAL.depth_response_for(c, calibration), palette)
              for c in check]
    measured = T.measure_noise_floor(
        captures, lambda c: S.bc0_to_db(c.bc0, calibration.counts_per_db))
    return {
        "rejected": rejected, "calibration": calibration, "palette": palette,
        "errors": errors, "num_frames": len(captures),
        "swept": len({(c.gain_level, c.tgc_levels[0], c.dynamic_range_level)
                      for c in captures}) >= 4,
        "floor": None if measured is None else measured["floor_db"],
    }


def target_and_uncertainty(captures, calibration, floor, palette, swept):
    """Accepted brightness and, for back-end-swept groups, the label uncertainty."""
    out = {"target": None, "uncertainty": None}
    result = T.measure_accepted_brightness(
        captures,
        lambda c: S.render(c.bc0, tgc_levels=c.tgc_levels,
                           gain_db=S.capture_gain_db(c),
                           dynamic_range_db=S.capture_window_db(c),
                           depth_response_db=CAL.depth_response_for(c, calibration),
                           reference_db=calibration.pivot_db,
                           counts_per_db=calibration.counts_per_db),
        lambda c: T.console_tissue_mask(
            S.bc0_to_db(c.bc0, calibration.counts_per_db), floor))
    if result:
        out["target"] = result["target_gray"]
    if swept:
        screenshot, rebuilt = [], []
        for capture in captures:
            if not image_fits_window(capture.geometry):
                continue
            actual = DP.capture_display_gray(capture, palette=palette)[0]
            db = (S.bc0_to_db(capture.bc0, calibration.counts_per_db)
                  + CAL.depth_response_for(capture, calibration)[:, None])
            shaped_screen = S.scan_convert_linear(db, actual.shape[0], actual.shape[1])
            mask = T.console_tissue_mask(shaped_screen, floor)
            if mask.sum() < 1000:
                continue
            rebuild = S.render(
                db_image=db, tgc_levels=capture.tgc_levels,
                gain_db=S.capture_gain_db(capture),
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
            out["uncertainty"] = float(np.std(np.array(rebuilt) - np.array(screenshot)))
    return out


def session_job(session, fit_frames, donor_floors=None):
    """Worker: everything for one session's groups.

    donor_floors {mode: floor dB} is given only on the second pass, for groups that could not
    measure their own floor. On the first pass those groups stop after the fit and come back.
    """
    results = {}
    try:
        groups = load_session_groups(session)
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}
    for key, captures in groups.items():
        captures.sort(key=lambda c: c.name)
        try:
            entry = fit_one(captures, fit_frames)
            calibration = entry["calibration"]
            if calibration is None:
                results[key] = entry
                continue
            floor = entry["floor"]
            if floor is None and donor_floors is not None:
                floor = donor_floors.get(key[1])
            if floor is not None:
                entry.update(target_and_uncertainty(captures, calibration, floor,
                                                    entry["palette"], entry["swept"]))
                entry["floor_used"] = floor
            results[key] = entry
        except Exception as exc:
            results[key] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    return {"groups": results}


def run_jobs(sessions, fit_frames, workers, donor_floors=None):
    out, errors = {}, {}
    with concurrent.futures.ProcessPoolExecutor(workers) as pool:
        futures = {pool.submit(session_job, s, fit_frames, donor_floors): s for s in sessions}
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            session = futures[future]
            result = future.result()
            if "error" in result:
                errors[session] = result["error"]
            else:
                for key, entry in result["groups"].items():
                    if "error" in entry:
                        errors["%s/%d" % key] = entry["error"]
                    else:
                        out[key] = entry
            if done % 20 == 0 or done == len(futures):
                print("  %d / %d sessions" % (done, len(futures)), flush=True)
    return out, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CS.DATASETS), default="phantom")
    parser.add_argument("--sessions", default=None,
                        help="comma separated subset of the dataset's sessions")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fit-frames", type=int, default=16)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    out_path = args.out or CS.output_path(args.dataset, "calibration")
    report = CS.report_path(__file__, args.dataset,
                            args.out or (out_path if args.sessions else None))

    lines = []

    def emit(text=""):
        lines.append(text)
        print(text.encode("ascii", "replace").decode("ascii"), flush=True)

    started = time.time()
    sessions = CS.sessions(args.dataset)
    if args.sessions:
        wanted = args.sessions.split(",")
        sessions = [s for s in sessions if s in wanted]
    emit("dataset %s: %d session(s); fitting with %d worker(s)"
         % (args.dataset, len(sessions), args.workers))
    results, errors = run_jobs(sessions, args.fit_frames, args.workers)

    # 底噪借用只在同一套数据、同一成像模式内。
    floors = {key: entry["floor"] for key, entry in results.items()
              if entry["calibration"] is not None}
    donors = {}
    for mode in {k[1] for k in floors}:
        values = [v for k, v in floors.items() if k[1] == mode and v is not None]
        donors[mode] = float(np.median(values)) if values else None
    redo = sorted({key[0] for key, v in floors.items() if v is None and donors.get(key[1])})
    if redo:
        emit("borrowing a noise floor for %d session(s) with no deep frame" % len(redo))
        again, more_errors = run_jobs(redo, args.fit_frames, args.workers, donors)
        errors.update(more_errors)
        for key, entry in again.items():
            if results.get(key, {}).get("floor") is None:
                results[key] = entry

    rejected_total = sum(entry["rejected"] for entry in results.values())
    calibrations = {k: e for k, e in results.items() if e["calibration"] is not None}
    emit("palette: %d group(s), %d frame(s) rejected as unreadable" % (len(results), rejected_total))
    emit("")
    emit("%-30s %-9s %6s %11s %10s %11s %13s %7s %6s" % (
        "session", "mode", "frames", "counts/dB", "pivot_dB", "band error", "per-pixel err",
        "usable", "ok"))
    for key in sorted(calibrations):
        entry = calibrations[key]
        calibration = entry["calibration"]
        emit("%-30s %-9s %6d %11.1f %10.2f %11.2f %13.1f %6.0f%% %6s" % (
            key[0][:28], MODE_NAME[key[1]], entry["num_frames"], calibration.counts_per_db,
            calibration.pivot_db, calibration.gray_error, float(np.mean(entry["errors"])),
            100 * calibration.usable_fraction,
            "yes" if calibration.usable_fraction >= CAL.MIN_USABLE_FRACTION else "NO"))
    for key in sorted(set(results) - set(calibrations)):
        emit("%-30s %-9s no fit frames (no flat-TGC frame that fits the window)"
             % (key[0][:28], MODE_NAME[key[1]]))

    emit("")
    emit("noise floors and brightness targets")
    emit("%-30s %-9s %12s %12s %12s %14s" % (
        "session", "mode", "floor dB", "floor counts", "target gray", "uncertainty"))
    uncertainty = {k: e.get("uncertainty") for k, e in calibrations.items()
                   if e.get("uncertainty") is not None}
    for key in sorted(calibrations):
        entry = calibrations[key]
        floor = entry.get("floor_used")
        emit("%-30s %-9s %12s %12s %12s %14s" % (
            key[0][:28], MODE_NAME[key[1]],
            "%.2f" % floor if floor is not None else "-",
            "%.0f" % (floor * entry["calibration"].counts_per_db) if floor is not None else "-",
            "%.0f" % entry["target"] if entry.get("target") is not None else "-",
            "%.4f" % uncertainty[key] if key in uncertainty else "borrowed"))

    measured_values = list(uncertainty.values())
    borrowed_uncertainty = {}
    for key in calibrations:
        if key in uncertainty:
            continue
        same = [v for k, v in uncertainty.items() if k[1] == key[1]]
        borrowed_uncertainty[key] = float(np.median(same if same else measured_values or [0.1]))
    uncertainty.update(borrowed_uncertainty)

    payload = {
        "note": ("Per (session, imaging mode) calibration, fitted against the console's own "
                 "gray index recovered through display_palette rather than PIL luma. "
                 "counts_per_db and pivot_db are meaningful only as a pair and only within "
                 "their own group. label_uncertainty is the standard deviation of "
                 "J(rebuild) - J(screenshot) after removing the constant bias."),
        "dataset": args.dataset,
        "groups": [],
    }
    for key, entry in sorted(calibrations.items()):
        calibration = entry["calibration"]
        floor = entry.get("floor_used")
        payload["groups"].append({
            "session": key[0], "image_mode": int(key[1]),
            "image_mode_name": MODE_NAME[key[1]], "num_frames": entry["num_frames"],
            "counts_per_db": round(calibration.counts_per_db, 2),
            "pivot_db": round(calibration.pivot_db, 3),
            "screenshot_gray_error": round(calibration.gray_error, 3),
            "usable_fraction": round(float(calibration.usable_fraction), 3),
            "calibratable": bool(calibration.usable_fraction >= CAL.MIN_USABLE_FRACTION),
            "noise_floor_db": None if floor is None else round(floor, 3),
            # 只有本组自己量到的才算 measured；借来的不算（旧版把借来的也写成 True）。
            "noise_floor_measured": entry["floor"] is not None,
            "target_gray": (None if entry.get("target") is None else round(entry["target"], 1)),
            "label_uncertainty": round(uncertainty[key], 4),
            "label_uncertainty_measured": key not in borrowed_uncertainty,
            "depth_axis_mm": [round(float(v), 3) for v in calibration.depth_axis_mm],
            "depth_response_db": [round(float(v), 4) for v in calibration.depth_response_db],
            "depth_response_by_frequency": {
                str(frequency): [round(float(v), 4) for v in curve]
                for frequency, curve in sorted(
                    calibration.depth_response_by_frequency.items())},
        })
    io.open(out_path, "w", encoding="utf-8").write(
        json.dumps(payload, indent=2, ensure_ascii=False))

    usable = [g["usable_fraction"] for g in payload["groups"]]
    counts = [g["counts_per_db"] for g in payload["groups"]]
    emit("")
    emit("summary: %d group(s) written, %d calibratable; counts/dB %.0f-%.0f (median %.0f); "
         "usable %.0f-%.0f%%" % (
             len(payload["groups"]), sum(g["calibratable"] for g in payload["groups"]),
             min(counts or [0]), max(counts or [0]), float(np.median(counts or [0])),
             100 * min(usable or [0]), 100 * max(usable or [0])))
    if errors:
        emit("failures: %d" % len(errors))
        for name, error in sorted(errors.items()):
            emit("  %s  %s" % (name, error))
    emit("wrote %s (%d groups) in %.0f s" % (out_path, len(payload["groups"]),
                                             time.time() - started))
    io.open(report, "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
