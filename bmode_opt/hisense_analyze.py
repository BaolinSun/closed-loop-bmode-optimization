"""Reproduce the Hisense export analysis as figures and a Markdown summary.

Regenerates every result the TGC work rests on: the back-end model calibration, the held-out
validation of that model against the console's own screenshots, the phantom metrics, and the
before/after of the TGC optimiser. Outputs land in output/hisense, which .gitignore excludes.
"""

import argparse
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hisense_backend_sim import (
    CALIBRATED_LEVEL_RANGE,
    GRAY_MAX,
    TGC_CENTER_LEVEL,
    calibrate_counts_per_db,
    calibrate_depth_response,
    calibrate_tgc_db_per_level,
    flat_tgc_capture,
    is_flat_tgc,
    render,
    render_db,
    tgc_level_to_db,
    validate_capture,
)
from hisense_loader import (
    DEFAULT_DATA_DIR,
    crop_image_area,
    find_captures,
    load_capture,
    load_screenshot,
    crop_capture_image,
)
from hisense_metrics import depth_band_levels, summarise
from hisense_tgc_optimizer import optimise_capture


DEFAULT_OUT_DIR = Path(r"D:\MyProjects\py_prj\ultrasound_dev_kit\output\hisense")


def plot_capture_overview(captures, calibration, out_path):
    """Simulated render beside the console screenshot for every capture."""
    fig, axes = plt.subplots(2, len(captures), figsize=(4.2 * len(captures), 8.4))
    axes = np.atleast_2d(axes)
    for column, capture in enumerate(captures):
        actual = crop_capture_image(capture)[0]
        predicted = render(
            capture.bc0,
            tgc_levels=capture.tgc_levels,
            dynamic_range_db=float(capture.dynamic_range_level),
            depth_response_db=calibration["depth_response"],
            reference_db=calibration["reference_db"],
            counts_per_db=calibration["counts_per_db"],
            db_per_level=calibration["db_per_level"],
            out_shape=actual.shape,
        )
        axes[0, column].imshow(actual, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[0, column].set_title(f"{capture.name}\nconsole screenshot", fontsize=9)
        axes[1, column].imshow(predicted, cmap="gray", vmin=0, vmax=255, aspect="auto")
        axes[1, column].set_title(f"simulated from BC0\nTGC {capture.tgc_levels.tolist()}", fontsize=8)
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_tgc_identification(captures, calibration, out_path):
    """Measured slider-to-dB samples against the fitted line.

    Captures are grouped by (gain, dynamic range) and each group is referenced to its own
    flat-TGC capture, matching what calibrate_tgc_db_per_level does. Referencing everything to
    a single capture would fold a group's gain offset into its apparent TGC gain, which for
    the 20260819 sweep put non-zero gain on the centre slider.
    """
    groups = {}
    for capture in captures:
        groups.setdefault((capture.gain_level, capture.dynamic_range_level), []).append(capture)

    fig, axis = plt.subplots(figsize=(7.5, 5))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for index, ((gain, dynamic_range), group) in enumerate(sorted(groups.items())):
        flats = [c for c in group if is_flat_tgc(c) and c.tgc_levels[0] == TGC_CENTER_LEVEL]
        swept = [c for c in group if not (is_flat_tgc(c) and c.tgc_levels[0] == TGC_CENTER_LEVEL)]
        if not flats or not swept:
            continue

        gray_per_db = GRAY_MAX / float(dynamic_range)
        reference_bands = np.mean(
            [depth_band_levels(crop_capture_image(f)[0].astype(float)) for f in flats],
            axis=0,
        )
        colour = colours[index % len(colours)]
        levels_in, gains_in, levels_out, gains_out = [], [], [], []
        for capture in swept:
            bands = depth_band_levels(crop_capture_image(capture)[0].astype(float))
            gains_db = (bands - reference_bands) / gray_per_db
            for level, gain_db in zip(capture.tgc_levels, gains_db):
                inside = CALIBRATED_LEVEL_RANGE[0] <= level <= CALIBRATED_LEVEL_RANGE[1]
                (levels_in if inside else levels_out).append(level)
                (gains_in if inside else gains_out).append(gain_db)

        label = f"gain {gain}, DR {dynamic_range}  (ref: {', '.join(f.name for f in flats)})"
        axis.scatter(levels_in, gains_in, s=55, color=colour, label=label)
        axis.scatter(levels_out, gains_out, s=55, marker="x", color=colour,
                     label=f"gain {gain}: outside fitted range")

    levels = np.linspace(0, 255, 200)
    axis.plot(
        levels,
        tgc_level_to_db(levels, calibration["db_per_level"]),
        "k--",
        label=f"fit: {calibration['db_per_level']:.5f} dB/level",
    )
    axis.axvline(TGC_CENTER_LEVEL, color="grey", lw=0.8)
    axis.axvspan(*CALIBRATED_LEVEL_RANGE, color="tab:green", alpha=0.08, label="calibrated range")
    axis.set_xlabel("TGC slider level")
    axis.set_ylabel("applied gain (dB)")
    axis.set_title("TGC actuator identification")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_depth_profiles(captures, calibration, out_path):
    """Actual versus simulated depth profiles, and the calibrated depth response."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for capture in captures:
        actual = crop_capture_image(capture)[0]
        predicted = render(
            capture.bc0,
            tgc_levels=capture.tgc_levels,
            dynamic_range_db=float(capture.dynamic_range_level),
            depth_response_db=calibration["depth_response"],
            reference_db=calibration["reference_db"],
            counts_per_db=calibration["counts_per_db"],
            db_per_level=calibration["db_per_level"],
            out_shape=actual.shape,
        )
        depth = np.linspace(0, capture.geometry.depth_mm, actual.shape[0])
        line = axes[0].plot(depth, np.median(actual, axis=1), lw=1.6, label=f"{capture.name} actual")[0]
        axes[0].plot(depth, np.median(predicted, axis=1), "--", lw=1.2, color=line.get_color(), label="simulated")

    axes[0].set_xlabel("depth (mm)")
    axes[0].set_ylabel("median display gray")
    axes[0].set_title("console vs simulator depth profile")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    geometry = calibration["reference"].geometry
    depth_axis = np.arange(calibration["depth_response"].size) * geometry.mm_per_point
    axes[1].plot(depth_axis, calibration["depth_response"], lw=1.6)
    measured = calibration["depth_response_mask"]
    axes[1].fill_between(
        depth_axis,
        calibration["depth_response"].min(),
        calibration["depth_response"].max(),
        where=~measured,
        color="tab:red",
        alpha=0.15,
        label="interpolated (display clipped)",
    )
    axes[1].set_xlabel("depth (mm)")
    axes[1].set_ylabel("correction (dB)")
    axes[1].set_title("calibrated BC0-to-display depth response")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_optimisation(result, calibration, out_path):
    """Before/after image, band levels and convergence for one optimisation."""
    capture = result["capture"]
    shared = dict(
        dynamic_range_db=float(capture.dynamic_range_level),
        depth_response_db=calibration["depth_response"],
        reference_db=calibration["reference_db"],
        counts_per_db=calibration["counts_per_db"],
        db_per_level=result["db_per_level"],
        out_shape=(int(capture.geometry.image_height_px), int(capture.geometry.image_width_px)),
    )
    before = render(capture.bc0, tgc_levels=capture.tgc_levels, **shared)
    after = render(capture.bc0, tgc_levels=result["suggested_levels"], **shared)

    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    axes[0].imshow(before, cmap="gray", vmin=0, vmax=255, aspect="auto")
    axes[0].set_title(f"before  TGC {capture.tgc_levels.tolist()}", fontsize=8)
    axes[1].imshow(after, cmap="gray", vmin=0, vmax=255, aspect="auto")
    axes[1].set_title(f"after  TGC {result['suggested_levels'].tolist()}", fontsize=8)
    for axis in axes[:2]:
        axis.set_xticks([])
        axis.set_yticks([])

    bands = np.arange(len(result["suggested_levels"]))
    axes[2].plot(result["before"]["band_levels_db"], bands, "o-", label="before")
    axes[2].plot(result["after"]["band_levels_db"], bands, "s-", label="after")
    axes[2].invert_yaxis()
    axes[2].axvline(result["target_db"], color="grey", ls=":", lw=1.2, label="target")
    axes[2].set_xlabel("band level (dB)")
    axes[2].set_ylabel("depth band")
    axes[2].set_title("depth uniformity")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    axes[3].plot([step["uniformity_db"] for step in result["history"]], "o-")
    axes[3].axhline(0.0, color="grey", lw=0.8)
    axes[3].set_xlabel("iteration")
    # Deliberately not the same quantity as the uniformity in the metric table: that one is
    # measured against each frame's own median, this one against the fixed target. A rising
    # curve here means the target was unreachable inside the slider bounds, not divergence.
    axes[3].set_ylabel("RMS distance to target (dB)")
    axes[3].set_title(f"loop convergence   (target {result['target_db']:.1f} dB)")
    axes[3].grid(alpha=0.3)

    fig.suptitle(capture.name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def build_calibration(captures):
    """Fit every back-end model constant from the capture set."""
    reference = flat_tgc_capture(captures)
    counts_per_db, reference_db, _ = calibrate_counts_per_db(reference)
    depth_response, depth_response_mask = calibrate_depth_response(reference, reference_db, counts_per_db)
    db_per_level, intercept, rms, samples = calibrate_tgc_db_per_level(captures)
    return {
        "reference": reference,
        "counts_per_db": counts_per_db,
        "reference_db": reference_db,
        "depth_response": depth_response,
        "depth_response_mask": depth_response_mask,
        "db_per_level": db_per_level,
        "tgc_intercept_db": intercept,
        "tgc_rms_db": rms,
        "tgc_samples": samples,
    }


def write_summary(captures, calibration, results, out_path):
    """Write the Markdown analysis summary."""
    reference = calibration["reference"]
    gray_per_db = GRAY_MAX / float(reference.dynamic_range_level)
    lines = [
        "# Hisense export analysis",
        "",
        f"Captures analysed: {len(captures)} from `{captures[0].path.parent}`",
        f"Depth-response reference (flat TGC): `{reference.name}`",
        "",
        "## Back-end model calibration",
        "",
        "| constant | value |",
        "| --- | --- |",
        f"| BC0 counts per dB | {calibration['counts_per_db']:.1f} |",
        f"| display reference level | {calibration['reference_db']:.2f} dB |",
        f"| TGC dB per slider level | {calibration['db_per_level']:.5f} |",
        f"| TGC span over 0..255 | {calibration['db_per_level'] * 255:.2f} dB |",
        f"| TGC gain at slider {TGC_CENTER_LEVEL} | {calibration['tgc_intercept_db']:+.3f} dB |",
        f"| TGC fit RMS residual | {calibration['tgc_rms_db']:.3f} dB over {calibration['tgc_samples']} samples |",
        f"| depth response range | {calibration['depth_response'].min():+.1f} .. "
        f"{calibration['depth_response'].max():+.1f} dB |",
        "",
        "The fitted gain at the centre slider coming out near zero means the swept measurements",
        "extrapolate linearly back to zero gain at the anchor. It is a linearity and consistency",
        "check, not independent proof that 127 is the neutral point.",
        "",
        "## Forward model validation",
        "",
        "Per-TGC-band median gray, simulated from BC0 against the console's own screenshot.",
        "",
        "| capture | role | mean abs error (gray) | mean abs error (dB) |",
        "| --- | --- | --- | --- |",
    ]
    for capture in captures:
        actual, predicted = validate_capture(
            capture,
            calibration["depth_response"],
            calibration["reference_db"],
            calibration["counts_per_db"],
            calibration["db_per_level"],
        )
        error = float(np.abs(predicted - actual).mean())
        role = "reference" if capture.path == reference.path else "held out"
        lines.append(f"| `{capture.name}` | {role} | {error:.1f} | {error / gray_per_db:.2f} |")

    lines += [
        "",
        "## Phantom metrics and TGC optimisation",
        "",
        "| capture | TGC before | uniformity before | uniformity after | speckle SNR | cyst CNR | TGC suggested |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        capture = result["capture"]
        lines.append(
            f"| `{capture.name}` | {capture.tgc_levels.tolist()} "
            f"| {result['before']['uniformity_db']:.2f} dB "
            f"| {result['after']['uniformity_db']:.2f} dB "
            f"| {result['before']['speckle_snr']:.2f} "
            f"| {result['before'].get('mean_cyst_cnr', float('nan')):.2f} "
            f"| {result['suggested_levels'].tolist()} |"
        )

    lines += [
        "",
        "Speckle SNR and cyst CNR are expected to stay essentially constant across the sweep:",
        "TGC redistributes brightness with depth, it does not change speckle statistics or",
        "target contrast. Their stability is a check on the metrics, not a result.",
        "",
        f"Slider bounds in use: {results[0]['level_bounds']}. Bands that reach a bound need",
        "acquisition sequence S1 before the recommendation beyond it can be trusted.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Analyse Hisense exports and regenerate figures.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory holding capture folders.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Where figures and summary are written.")
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Let the optimiser leave the calibrated slider range.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    capture_dirs = find_captures(args.data_dir)
    if not capture_dirs:
        raise SystemExit(f"No captures found under {args.data_dir}")
    captures = [load_capture(path) for path in capture_dirs]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    calibration = build_calibration(captures)

    level_bounds = (0, 255) if args.allow_extrapolation else CALIBRATED_LEVEL_RANGE
    results = [
        optimise_capture(
            capture,
            reference_capture=calibration["reference"],
            db_per_level=calibration["db_per_level"],
            level_bounds=level_bounds,
        )
        for capture in captures
    ]

    plot_capture_overview(captures, calibration, args.out_dir / "capture_overview.png")
    plot_tgc_identification(captures, calibration, args.out_dir / "tgc_identification.png")
    plot_depth_profiles(captures, calibration, args.out_dir / "depth_profiles.png")
    for result in results:
        plot_optimisation(result, calibration, args.out_dir / f"optimisation_{result['capture'].name}.png")

    summary_path = args.out_dir / "summary.md"
    write_summary(captures, calibration, results, summary_path)

    print(f"wrote {summary_path}")
    for path in sorted(args.out_dir.glob("*.png")):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
