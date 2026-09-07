"""Recover the console's display response curve from a uniform-TGC sweep.

hisense_backend_sim assumes displayed gray is linear in dB, i.e. gray = (dB - floor)/DR*255.
The 20260819 sweep shows that assumption is wrong: measuring the TGC gain band by band in
raw gray makes the slider look almost twice as effective at mid depth as in the near field.

That apparent depth dependence is an artefact, not a property of the TGC. The sweep contains
a decisive control: S1-H-3 (gain 75, TGC 254 flat) and S1-L-0 (gain 125, TGC 127 flat) render
the same scene to within 3 gray everywhere. Gain is a global scalar and physically cannot
carry a depth weighting, so a depth-weighted TGC would have made those two images differ by
an arch of roughly 18 gray. They do not. The TGC is depth uniform, and it is the gray scale
that is nonlinear - the per-band slope tracks band brightness with r = 0.95.

This module recovers that response nonparametrically. For a spatially uniform shift of the
underlying signal, a correct response phi must satisfy

    phi(gray_shifted) - phi(gray_reference) = constant

for every gray level. Fitting phi to enforce that over several shift magnitudes recovers the
curve up to an affine transform, which is the classic camera-response-from-multiple-exposures
problem; here the sweep supplies the exposures and a clamped probe supplies the static scene.

phi is returned in slider-equivalent units: one unit is one TGC slider level. Converting to dB
needs an absolute anchor that this data does not provide - see the module notes in main().
"""

import argparse
from pathlib import Path

import numpy as np

from hisense_backend_sim import (
    TGC_CENTER_LEVEL,
    is_flat_tgc,
    scan_convert_linear,
)
from hisense_loader import (
    DEFAULT_DATA_DIR,
    NUM_TGC_BANDS,
    band_edges,
    crop_image_area,
    find_captures,
    load_capture,
    load_screenshot,
    crop_capture_image,
)


DEFAULT_GRID_STEP = 4.0
DEFAULT_GRID_MAX = 136.0
DEFAULT_BIN_STEP = 4
DEFAULT_MIN_BIN_PIXELS = 300
DEFAULT_SMOOTHNESS = 3.0
# Gray levels outside this window are quantisation- or clip-dominated and are not fitted.
USABLE_GRAY = (6, 120)


def common_crop(captures):
    """Crop every capture's screenshot to a shared shape so pixels correspond."""
    images = {c.name: crop_capture_image(c)[0] for c in captures}
    height = min(v.shape[0] for v in images.values())
    width = min(v.shape[1] for v in images.values())
    return {name: value[:height, :width] for name, value in images.items()}, (height, width)


def comparagram(reference, image, bin_step=DEFAULT_BIN_STEP, min_pixels=DEFAULT_MIN_BIN_PIXELS,
                gray_max=DEFAULT_GRID_MAX):
    """Median gray of image as a function of reference gray, using pixel correspondence."""
    points = []
    for low in np.arange(0, gray_max, bin_step):
        selected = (reference >= low) & (reference < low + bin_step)
        if selected.sum() < min_pixels:
            continue
        points.append((float(reference[selected].mean()), float(np.median(image[selected]))))
    return points


def uniform_shift_sets(captures):
    """Group captures into (swept capture, flat reference image key, shift) by gain level.

    Only flat-TGC captures qualify, because the recovery assumes a spatially uniform shift.
    """
    groups = {}
    for capture in captures:
        groups.setdefault(capture.gain_level, []).append(capture)

    sets = []
    for group in groups.values():
        flats = [c for c in group if is_flat_tgc(c) and c.tgc_levels[0] == TGC_CENTER_LEVEL]
        if not flats:
            continue
        for capture in group:
            shift = float(np.mean(capture.tgc_levels) - TGC_CENTER_LEVEL)
            if abs(shift) < 1.0:
                continue
            sets.append((capture, [f.name for f in flats], shift))
    return sets


def recover_response(captures, images, grid_step=DEFAULT_GRID_STEP, grid_max=DEFAULT_GRID_MAX,
                     smoothness=DEFAULT_SMOOTHNESS):
    """Fit the monotone response phi(gray) that makes uniform shifts constant in phi.

    Returns (grid, phi, num_points). phi is in slider-equivalent units and anchored at phi(0)=0.
    """
    sets = uniform_shift_sets(captures)
    if not sets:
        raise ValueError("No flat-TGC captures with a centre reference; cannot recover response")

    grid = np.arange(0.0, grid_max + grid_step, grid_step)
    size = grid.size

    def weights(value):
        row = np.zeros(size)
        value = float(np.clip(value, grid[0], grid[-1]))
        index = max(0, min(int(np.searchsorted(grid, value)) - 1, size - 2))
        frac = (value - grid[index]) / (grid[index + 1] - grid[index])
        row[index], row[index + 1] = 1.0 - frac, frac
        return row

    rows, targets = [], []
    for capture, reference_names, shift in sets:
        reference = np.mean([images[name] for name in reference_names], axis=0)
        for gray, shifted in comparagram(reference, images[capture.name], gray_max=grid_max):
            rows.append(weights(shifted) - weights(gray))
            targets.append(shift)

    design = np.array(rows)
    target = np.array(targets)

    # Second-difference smoothing keeps the curve stable where few pixels constrain it, and a
    # hard anchor at phi(0)=0 removes the additive freedom.
    curvature = np.zeros((size - 2, size))
    for index in range(size - 2):
        curvature[index, index:index + 3] = (1.0, -2.0, 1.0)
    anchor = np.eye(size)[0] * 100.0

    design = np.vstack([design, smoothness * curvature, anchor])
    target = np.concatenate([target, np.zeros(size - 2), [0.0]])
    phi, *_ = np.linalg.lstsq(design, target, rcond=None)
    return grid, phi, len(rows)


def response_units_per_count(capture, grid, phi, usable_gray=USABLE_GRAY, row_step=25):
    """Fit d(phi)/d(BC0 count) from lateral variation at fixed depth.

    At one depth the depth response is constant, so phi versus BC0 traces the same straight
    line on every row. Fitting rows independently and taking the median is robust to the cyst
    and wire targets that sit on only some rows.
    """
    image = crop_capture_image(capture)[0]
    bc0 = scan_convert_linear(capture.bc0, image.shape[0], image.shape[1])
    slopes = []
    for row in range(row_step * 2, image.shape[0] - row_step, row_step):
        gray, counts = image[row], bc0[row]
        usable = (gray > usable_gray[0]) & (gray < usable_gray[1])
        if usable.sum() < 200:
            continue
        slopes.append(np.polyfit(counts[usable], np.interp(gray[usable], grid, phi), 1)[0])
    if not slopes:
        raise ValueError("No usable rows for the BC0 scale fit")
    return float(np.median(slopes)), np.array(slopes)


def band_slopes(captures, images, grid=None, phi=None, gray_per_db=None, num_bands=NUM_TGC_BANDS):
    """Per-band gain per slider level, measured either in raw gray/dB or in phi units.

    Pass grid and phi for the corrected measurement; pass gray_per_db for the raw one. The
    corrected slopes collapsing onto a single value is the evidence that the TGC is uniform.
    """
    height = next(iter(images.values())).shape[0]
    edges = band_edges(height, num_bands)

    def bands(image):
        return np.array([np.median(image[edges[k]:edges[k + 1]]) for k in range(num_bands)])

    rows = []
    for capture, reference_names, shift in uniform_shift_sets(captures):
        reference = bands(np.mean([images[name] for name in reference_names], axis=0))
        measured = bands(images[capture.name])
        if phi is not None:
            gain = np.interp(measured, grid, phi) - np.interp(reference, grid, phi)
        else:
            gain = (measured - reference) / gray_per_db
        rows.append((capture.name, gain / (np.asarray(capture.tgc_levels) - TGC_CENTER_LEVEL)))
    return rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Recover the Hisense display response curve from a uniform-TGC sweep."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory holding the sweep's capture folders.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional .npy path to save the (grid, phi) curve.")
    return parser


def main():
    args = build_parser().parse_args()
    captures = [load_capture(path) for path in find_captures(args.data_dir)]
    if not captures:
        raise SystemExit(f"No captures found under {args.data_dir}")

    images, (height, _) = common_crop(captures)
    grid, phi, num_points = recover_response(captures, images)
    print(f"recovered response from {num_points} comparagram points over "
          f"{len(uniform_shift_sets(captures))} uniform-shift captures")
    print(f"  monotone: {bool(np.all(np.diff(phi) > 0))}")

    slope = np.gradient(phi, grid)
    dark = slope[np.argmin(np.abs(grid - 8))]
    bright = slope[np.argmin(np.abs(grid - 76))]
    print(f"  d(phi)/d(gray): {dark:.2f} at gray 8 versus {bright:.2f} at gray 76"
          f"  ->  the dark end is compressed {dark / bright:.1f}x")

    reference = next(c for c in captures if is_flat_tgc(c) and c.tgc_levels[0] == TGC_CENTER_LEVEL)
    alpha, per_row = response_units_per_count(reference, grid, phi)
    print(f"\nBC0 scale from {reference.name}: {len(per_row)} independent depth rows")
    print(f"  d(phi)/d(BC0 count) = {alpha:.5f}"
          f"  (IQR {np.percentile(per_row, 25):.5f}..{np.percentile(per_row, 75):.5f})")
    print(f"  -> {1.0 / alpha:.1f} BC0 counts per TGC slider level")

    dynamic_range = float(reference.dynamic_range_level)
    print("\nper-band gain per slider level")
    for label, kwargs in (("raw gray", dict(gray_per_db=255.0 / dynamic_range)),
                          ("phi corrected", dict(grid=grid, phi=phi))):
        rows = band_slopes(captures, images, **kwargs)
        stacked = np.array([values for _, values in rows])
        mean = stacked.mean(axis=0)
        interior = mean[1:]
        print(f"  {label:<14}" + " ".join(f"{v:7.4f}" for v in mean))
        print(f"  {'':<14}spread {mean.max() / mean.min():.2f}x   "
              f"bands 1-7 CV {interior.std() / interior.mean() * 100:.1f}%")

    print("\nNote: phi is in slider-equivalent units. Converting to dB needs an absolute anchor")
    print("that this sweep does not contain; UIDynamicRangeLevel being literal dB is still an")
    print("untested assumption, which is what acquisition sequence S3 is for.")

    if args.out:
        np.save(args.out, np.vstack([grid, phi]))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
