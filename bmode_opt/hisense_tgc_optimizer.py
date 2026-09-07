"""Closed-loop TGC optimiser for Hisense B-mode captures, run offline against the simulator.

The loop observes Algo_BC0.bin, which sits upstream of the back end and is therefore
unaffected by the very TGC setting being solved for. That removes the usual hazard of
feeding a controller an observation already contaminated by its own actuator, and it means
the whole loop converges offline without the console in the loop.

The TGC actuator was identified from the 20260814 sweep as separable, static and monotonic:
a slider level applies an additive dB gain to its depth band, 0.0656 dB per level about a
neutral point at 127. So the control law inverts it analytically instead of searching, and
only iterates to settle the mild coupling introduced by interpolating eight band gains onto
the depth axis.

The result is a suggested slider setting for the console operator, plus the metric change it
is predicted to produce.

Known limitation, from the 20260819 sweep: the calibration this loop stands on flows through
hisense_backend_sim's linear gray-to-dB mapping, and that mapping is wrong - the display
compresses the dark end about 3x. Both the depth response and the dB-per-level constant are
distorted by it, which moves the suggested sliders by a mean of 9 levels and up to 18 against
a correctly-calibrated solve. The loop is self-consistent and highly repeatable (0.4 levels
across repeated captures, and 0.4 levels across a 50-level gain change, because a global
offset cancels), so the shape of its answer is sound; the absolute levels are not yet.
See hisense_display_response for the recovered curve and the evidence.
"""

import argparse
from pathlib import Path

import numpy as np

from hisense_backend_sim import (
    CALIBRATED_LEVEL_RANGE,
    DEFAULT_DB_PER_LEVEL,
    TGC_CENTER_LEVEL,
    TGC_MAX_LEVEL,
    TGC_MIN_LEVEL,
    calibrate_counts_per_db,
    calibrate_depth_response,
    calibrate_tgc_db_per_level,
    flat_tgc_capture,
    render,
    render_db,
    tgc_level_to_db,
)
from hisense_loader import (
    DEFAULT_DATA_DIR,
    NUM_TGC_BANDS,
    find_captures,
    load_capture,
)
from hisense_metrics import depth_band_levels, summarise, uniformity_cost


# Frame-to-frame band spread over the sweep was at most 0.17 dB with a clamped probe, so this
# deadband sits just above the observed noise. Freehand scanning will need sequence S0 of the
# acquisition protocol to re-establish it.
DEFAULT_DEADBAND_DB = 0.2
DEFAULT_LOOP_GAIN = 0.7
DEFAULT_MAX_STEP = 30
DEFAULT_MAX_ITERATIONS = 25


def simulate_band_levels(bc0, tgc_levels, depth_response_db, counts_per_db, db_per_level, num_bands=NUM_TGC_BANDS):
    """Median dB level of each depth band for a candidate TGC setting."""
    db_image = render_db(
        bc0,
        tgc_levels=tgc_levels,
        depth_response_db=depth_response_db,
        counts_per_db=counts_per_db,
        db_per_level=db_per_level,
    )
    return depth_band_levels(db_image, num_bands)


def solve_tgc_levels(
    bc0,
    depth_response_db,
    counts_per_db,
    db_per_level=DEFAULT_DB_PER_LEVEL,
    initial_levels=None,
    target_db=None,
    loop_gain=DEFAULT_LOOP_GAIN,
    deadband_db=DEFAULT_DEADBAND_DB,
    max_step=DEFAULT_MAX_STEP,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    num_bands=NUM_TGC_BANDS,
    level_bounds=(TGC_MIN_LEVEL, TGC_MAX_LEVEL),
):
    """Drive the depth band levels onto a common target by adjusting the TGC sliders.

    Returns (levels, history). history holds one dict per iteration with the band levels,
    the residual error and the uniformity, so convergence can be inspected or plotted.

    target_db defaults to the median of the *TGC-free* band levels offset by the mean gain of
    the starting sliders. Anchoring to the tissue rather than to the current band levels keeps
    the suggestion independent of whatever TGC the operator happened to have set, while the
    offset preserves overall image brightness.
    """
    levels = (
        np.full(num_bands, TGC_CENTER_LEVEL, dtype=np.float64)
        if initial_levels is None
        else np.asarray(initial_levels, dtype=np.float64).copy()
    )
    low_bound, high_bound = level_bounds

    if target_db is None:
        base_bands = simulate_band_levels(
            bc0, None, depth_response_db, counts_per_db, db_per_level, num_bands
        )
        mean_gain_db = float(np.mean(tgc_level_to_db(levels, db_per_level)))
        target_db = float(np.median(base_bands)) + mean_gain_db

    bands = simulate_band_levels(bc0, levels, depth_response_db, counts_per_db, db_per_level, num_bands)
    history = []
    for iteration in range(int(max_iterations)):
        error = target_db - bands
        # A band pinned at a bound whose residual error pushes further outward is settled: no
        # slider movement can reduce it. Treating that as non-convergence would report failure
        # for the constrained optimum.
        blocked = ((error > 0) & (levels >= high_bound - 1e-9)) | ((error < 0) & (levels <= low_bound + 1e-9))
        settled = (np.abs(error) < deadband_db) | blocked
        history.append(
            {
                "iteration": iteration,
                "levels": levels.copy(),
                "band_levels_db": bands.copy(),
                "error_db": error.copy(),
                "blocked": blocked.copy(),
                "uniformity_db": uniformity_cost(bands, target_db),
            }
        )
        if np.all(settled):
            break

        step = np.clip(loop_gain * error / db_per_level, -max_step, max_step)
        # Bands already inside the deadband are held still to stop dither.
        step[np.abs(error) < deadband_db] = 0.0
        levels = np.clip(levels + step, low_bound, high_bound)
        bands = simulate_band_levels(bc0, levels, depth_response_db, counts_per_db, db_per_level, num_bands)

    final = np.clip(np.round(levels), low_bound, high_bound).astype(np.int32)
    return final, history


def optimise_capture(
    capture,
    reference_capture=None,
    target_db=None,
    loop_gain=DEFAULT_LOOP_GAIN,
    deadband_db=DEFAULT_DEADBAND_DB,
    max_iterations=DEFAULT_MAX_ITERATIONS,
    db_per_level=None,
    calibration_captures=None,
    level_bounds=CALIBRATED_LEVEL_RANGE,
):
    """Suggest a TGC setting for one capture and report the predicted metric change."""
    reference_capture = reference_capture or capture
    counts_per_db, reference_db, _ = calibrate_counts_per_db(reference_capture)
    depth_response_db, _ = calibrate_depth_response(reference_capture, reference_db, counts_per_db)

    if db_per_level is None:
        if calibration_captures:
            db_per_level, *_ = calibrate_tgc_db_per_level(calibration_captures)
        else:
            db_per_level = DEFAULT_DB_PER_LEVEL

    levels, history = solve_tgc_levels(
        capture.bc0,
        depth_response_db,
        counts_per_db,
        db_per_level=db_per_level,
        initial_levels=capture.tgc_levels,
        target_db=target_db,
        loop_gain=loop_gain,
        deadband_db=deadband_db,
        max_iterations=max_iterations,
        level_bounds=level_bounds,
    )

    dynamic_range_db = float(capture.dynamic_range_level)
    reports = {}
    for name, candidate in (("before", capture.tgc_levels), ("after", levels)):
        db_image = render_db(
            capture.bc0,
            tgc_levels=candidate,
            depth_response_db=depth_response_db,
            counts_per_db=counts_per_db,
            db_per_level=db_per_level,
        )
        gray = render(
            capture.bc0,
            tgc_levels=candidate,
            dynamic_range_db=dynamic_range_db,
            depth_response_db=depth_response_db,
            reference_db=reference_db,
            counts_per_db=counts_per_db,
            db_per_level=db_per_level,
        )
        reports[name] = summarise(db_image, capture.geometry, gray)

    return {
        "capture": capture,
        "suggested_levels": levels,
        "history": history,
        "before": reports["before"],
        "after": reports["after"],
        "db_per_level": db_per_level,
        "converged": bool(
            np.all((np.abs(history[-1]["error_db"]) < deadband_db) | history[-1]["blocked"])
        ),
        "target_db": float(history[0]["band_levels_db"][0] + history[0]["error_db"][0]),
        "level_bounds": level_bounds,
        "clamped_bands": [
            index
            for index, level in enumerate(levels)
            if level in (level_bounds[0], level_bounds[1])
        ],
    }


def format_report(result):
    """Render an optimisation result as a readable console report."""
    capture = result["capture"]
    before, after = result["before"], result["after"]
    levels = result["suggested_levels"]
    current = np.asarray(capture.tgc_levels)

    low, high = result["level_bounds"]
    lines = [
        f"{capture.name}  (gain {capture.gain_level}, DR {capture.dynamic_range_level})",
        f"  iterations      : {len(result['history'])}"
        f"{'' if result['converged'] else '  [did not reach deadband]'}",
        f"  slider bounds   : {low}..{high}"
        f"{' (calibrated range)' if (low, high) == CALIBRATED_LEVEL_RANGE else ' (extrapolating)'}",
        f"  target level    : {result['target_db']:.2f} dB"
        f"   (every band is driven onto this)",
        "",
        f"  {'band':>4} {'depth mm':>9} {'now':>5} {'suggest':>8} {'delta':>6} {'band dB':>9} {'to go':>7}",
    ]
    geometry = capture.geometry
    band_mm = geometry.depth_mm / len(levels)
    for index, (old, new) in enumerate(zip(current, levels)):
        centre = (index + 0.5) * band_mm
        flag = "  at bound" if index in result["clamped_bands"] else ""
        lines.append(
            f"  {index:>4} {centre:>9.1f} {old:>5d} {new:>8d} {new - old:>+6d}"
            f" {before['band_levels_db'][index]:>9.1f}"
            f" {result['target_db'] - before['band_levels_db'][index]:>+7.1f}{flag}"
        )

    lines += [
        "",
        f"  {'metric':<22}{'before':>10}{'after':>10}",
        f"  {'uniformity (dB)':<22}{before['uniformity_db']:>10.2f}{after['uniformity_db']:>10.2f}",
        f"  {'speckle SNR':<22}{before['speckle_snr']:>10.2f}{after['speckle_snr']:>10.2f}",
        f"  {'mean cyst CNR':<22}{before.get('mean_cyst_cnr', float('nan')):>10.2f}"
        f"{after.get('mean_cyst_cnr', float('nan')):>10.2f}",
        f"  {'crushed %':<22}{before['crushed_fraction'] * 100:>10.1f}{after['crushed_fraction'] * 100:>10.1f}",
        f"  {'saturated %':<22}{before['saturated_fraction'] * 100:>10.2f}"
        f"{after['saturated_fraction'] * 100:>10.2f}",
        "",
        f"  set the eight console TGC sliders to: {levels.tolist()}",
    ]
    if result["clamped_bands"]:
        lines.append(
            f"  bands {result['clamped_bands']} hit the slider bound. The bound is no longer about"
            " missing sweep data - the 20260819 sweep covers 0..255 - but about this module still"
            " assuming a linear display response, which that sweep disproved. Widening it safely"
            " needs hisense_display_response wired into the back-end model; --allow-extrapolation"
            " overrides it meanwhile."
        )
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Suggest a flattened TGC setting for a Hisense capture using the offline simulator."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory holding capture folders.")
    parser.add_argument("--capture", type=Path, default=None, help="Capture to optimise; default is all of them.")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Flat-TGC capture used to calibrate the depth response; default auto-detects one.",
    )
    parser.add_argument("--target-db", type=float, default=None, help="Target band level; default is the current median.")
    parser.add_argument("--loop-gain", type=float, default=DEFAULT_LOOP_GAIN, help="Proportional loop gain.")
    parser.add_argument("--deadband-db", type=float, default=DEFAULT_DEADBAND_DB, help="Per-band deadband in dB.")
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS, help="Iteration limit.")
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow sliders outside the calibrated range, where the dB-per-level fit is unverified.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    capture_dirs = find_captures(args.data_dir)
    if not capture_dirs:
        raise SystemExit(f"No captures found under {args.data_dir}")

    all_captures = [load_capture(path) for path in capture_dirs]
    if args.reference:
        reference = load_capture(args.reference)
    else:
        try:
            reference = flat_tgc_capture(all_captures)
        except ValueError:
            raise SystemExit(
                "No flat-TGC capture found for depth-response calibration; pass --reference explicitly."
            )

    targets = [load_capture(args.capture)] if args.capture else all_captures
    print(f"depth-response reference: {reference.name}\n")
    for capture in targets:
        result = optimise_capture(
            capture,
            reference_capture=reference,
            target_db=args.target_db,
            loop_gain=args.loop_gain,
            deadband_db=args.deadband_db,
            max_iterations=args.max_iterations,
            calibration_captures=all_captures if len(all_captures) > 1 else None,
            level_bounds=(TGC_MIN_LEVEL, TGC_MAX_LEVEL) if args.allow_extrapolation else CALIBRATED_LEVEL_RANGE,
        )
        print(format_report(result))
        print()


if __name__ == "__main__":
    main()
