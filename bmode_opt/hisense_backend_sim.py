"""Offline forward model of the Hisense back-end display chain, driven by raw BC0 data.

Algo_BC0.bin is tapped upstream of the whole back end: changing TGC leaves it unchanged to
within frame noise while the displayed image moves by more than 20 dB. That makes BC0 a
parameter-independent observation of the tissue, and lets every back-end knob be simulated
offline instead of round-tripping through the console.

BC0 is already log compressed, so the chain below is affine in dB:

    display_dB = BC0 / COUNTS_PER_DB + depth_response_dB(z) + tgc_dB(z) + gain_dB
    window_dB  = dr_ui_to_window_db(UIDynamicRangeLevel)
    gray       = clip(GRAY_PIVOT + (display_dB - pivot_dB) / window_dB * 255, 0, 255)

Two things about that mapping were wrong in earlier versions of this model, and both were
found by sweeping the console's dynamic range in 20260904_DR:

  1. The console's "dynamic range" number is NOT the window width in dB. It has to go
     through dr_ui_to_window_db(); feeding the UI number straight in is wrong by 0.72x
     to 2.30x across the 30..400 range.
  2. The window does not hang from a fixed top. Widening it rotates the mapping about a
     fixed mid gray, GRAY_PIVOT: bright pixels barely move while dark ones lift a lot.
     Anchoring at the top instead predicts every pixel racing toward 255, which is off
     by 141 gray levels at the wide end.

depth_response_dB(z) is a measured per-depth correction curve, not a fudge factor. BC0 sits
upstream of processing that is itself depth dependent - spatial compounding carries explicit
depth-varying weights (BSpatialCompWeightCoe, BSpatialCompDepthThreshold) and the near field
is blanked (BDeadPoints) - so BC0 and the display do not share a depth profile. Without this
term a pure affine model misses by up to 26 gray levels; with it, held-out TGC settings are
reproduced to 0.4-1.0 dB. Calibrate it once from a flat-TGC capture.

Unlike the frame-max normalisation used by the USB pipeline's apply_bmode_mapping(), the
reference level here is absolute. A max-normalised mapping would make the rendered image
depend on the single brightest pixel of the frame, which couples every band to a bright
specular target and destroys the repeatability a feedback loop needs.

Calibration constants are fitted by calibrate_tgc_db_per_level(), calibrate_counts_per_db()
and calibrate_depth_response() from the TGC sweep in data/hisense_medical.
"""

import argparse
from pathlib import Path

import numpy as np

from hisense_loader import (
    DEFAULT_DATA_DIR,
    NUM_TGC_BANDS,
    band_centres,
    band_edges,
    crop_image_area,
    find_captures,
    load_capture,
    load_screenshot,
    crop_capture_image,
)


TGC_MIN_LEVEL = 0
TGC_MAX_LEVEL = 255
TGC_CENTER_LEVEL = 127

# TGC slider slope. Measured by tests/measure_actuator_steps.py, which subtracts pairs of
# frames of one static scene that differ in a single knob: counts, depth response and pivot are
# identical between the two frames and cancel, so the reading needs no calibration at all.
# Three sessions agree, over 2497 band readings with a residual of 0.446 dB:
#
#     session        slider dB/level   gain dB/level   residual
#     20260819           0.07770          0.20108       0.285 dB
#     20260901_E3        0.07728          0.19920       0.664 dB
#     20260904_DR        0.07722          0.19958       0.449 dB
#     20260909_GEN       0.08229          0.28649       0.498 dB   <- fundamental, high gain
#
# Refitting all four together with the gain ladder split at level 127 (see GAIN_DB_PER_LEVEL
# below for why it has to be split) gives 0.07940 for the slider over 4473 readings, residual
# 0.508 dB. The slider needs no split of its own: both datasets sweep it over its whole 0 to
# 254 range and the two agree to 6%, inside each session's own scatter.
#
# One scalar is enough: grouping the readings by step size gives 0.0766, 0.0765, 0.0778 and
# 0.0774 dB per level for steps of 25-50, 50-80, 80-120 and 120-300 levels.
#
# Supersedes 0.06559, which was fitted from the 20260814 sweep on PIL luma. The console does
# not display luma - it maps through a colour palette - so every constant fitted that way reads
# low; see display_palette. The old value is 15% below this one.
#
# COUNTS_PER_DB: refitted on 20260904_DR after the window-width and anchor fixes. The old
# 762.5 came from a fit that treated the console's dynamic-range number as dB, so it is not
# comparable. Both constants are session defaults; calibration.fit_group() refits them per
# session and imaging mode.
DEFAULT_DB_PER_LEVEL = 0.07940
DEFAULT_COUNTS_PER_DB = 877.3
# The display-dB level that lands on GRAY_PIVOT, at the calibration gain (BUIGainLevel 75).
# Refitted after the anchor fix; the old 84.58 was a top-of-window value under the
# discarded model and is not comparable.
DEFAULT_PIVOT_DB = 25.09
# Kept as an alias so callers that still say reference_db keep working; it now means the
# pivot level, not the top of the window.
DEFAULT_REFERENCE_DB = DEFAULT_PIVOT_DB
CALIBRATION_GAIN_LEVEL = 75

# Provenance, stated plainly: this bracket was not fitted. In the 20260814 ramp pair, sliders
# 78/114/146/178 reproduced across the two opposite ramps to within 1-2 gray while 6/42/210/242
# disagreed by 4-13, so the bracket is a round interval drawn just outside the four that agreed.
#
# An earlier comment here blamed display clipping. That was wrong: crushed and saturated
# fractions are 0.0 at every excluded point. The 20260819 sweep identified the real cause as a
# nonlinear display response, and it also shows this constant is the wrong shape for the job.
# The two ramps sample each slider at two different depths; extreme sliders happened to land on
# dark bands (gray 2-12) and middle sliders on brighter ones (gray 17-36), and the disagreement
# tracks that brightness, not the slider value (r = -0.82). The real validity limit is a
# brightness bound: against the recovered response, the linear gray-to-dB model holds to 25%
# only above roughly gray 44, and to 10% only above roughly gray 64.
#
# So this bracket is a blunt stand-in that happens to keep the solver near the operating point
# the linear fit was made at. Wiring hisense_display_response into the model replaces it; see
# that module. Widening it without doing so extrapolates a known-wrong slope further.
CALIBRATED_LEVEL_RANGE = (70, 185)

GRAY_MAX = 255.0

# --- Dynamic range ---------------------------------------------------------------
# The console shows a "dynamic range" number from 30 to 400 (100 internal steps; the UI
# value is 3.7373 * step + 29.66). That number is NOT the display window width in dB.
#
# Measured in 20260904_DR by shifting the image with a *known* uniform TGC offset and
# reading the resulting gray change. Using TGC as the ruler cancels the depth response
# exactly, because it is uniform in depth and calibrated to 0.0656 dB/step:
#
#     UI number    30      67      149     400
#     gray per dB  6.10    4.36    2.73    1.47
#     window dB    41.8    58.5    93.4    174.0
#     UI / window  0.72    1.15    1.60    2.30
#
# A straight line fits those four points to within 5.9 dB. Feeding the UI number in
# directly is wrong by a factor spanning 3.2x end to end.
DR_WINDOW_SLOPE = 0.3519
DR_WINDOW_INTERCEPT = 35.09
# The console's own limit, which is also the span the four calibration points cover.
DR_UI_RANGE = (30.0, 400.0)

# --- Gain ------------------------------------------------------------------------
# BUIGainLevel steps, in dB. Not one number: the console's gain ladder is coarser above the
# middle of its range than below it. Measured by tests/measure_actuator_steps.py, which
# subtracts pairs of frames of one static scene differing in a single knob, so the reading
# needs no calibration at all. Fitting all four swept sessions together, 4473 band readings:
#
#     model                              residual sd
#     one constant over the whole range     1.316 dB
#     split at level 127                    0.508 dB
#
# and 0.508 is what each session's own fit gives, so the split accounts for essentially all of
# the discrepancy. The cross-check frame settles it on its own: gain 129 with sliders at 196,
# against gain 169 with sliders at 127, measures -6.212 dB. The split constants predict -5.992;
# a single 0.20004 predicts -2.820.
#
#     gain level    dB per level    measured on
#     0 to 127        0.20359       20260819, 20260901_E3, 20260904_DR (all harmonic)
#     127 to 255      0.28675       20260909_GEN (fundamental)
#
# WHAT IS NOT SEPARATED. Every harmonic capture in the archive sits between gain 59 and 125,
# and every usable fundamental one between 129 and 229. They overlap only at level 75, in
# 20260828/GEN, which is 90% black and holds a single gain level, so it yields no pair. The
# split above is therefore equally consistent with "the ladder is coarser at high gain" and
# with "fundamental mode steps differently from harmonic". It is written as a property of the
# level because a digital gain applied after beamforming has no way to know the transmit mode,
# but that is reasoning, not measurement. Three harmonic frames at gain 129, 169 and 209 at one
# probe position would settle it; see the acquisition protocol.
GAIN_DB_PER_LEVEL_TABLE = ((0.0, 127.0, 0.20359), (127.0, 255.0, 0.28675))

# The slope at the low end, kept for callers that only need an order of magnitude. Anything
# converting a real dB difference into console clicks should use gain_db_to_levels() with the
# level it is at, because a click is worth 41% more above 127 than below it.
GAIN_DB_PER_LEVEL = GAIN_DB_PER_LEVEL_TABLE[0][2]


def gain_db_per_level(level):
    """Local slope of the gain ladder, in dB per level, at one gain level."""
    level = float(level)
    for low, high, slope in GAIN_DB_PER_LEVEL_TABLE:
        if low <= level < high:
            return float(slope)
    return float(GAIN_DB_PER_LEVEL_TABLE[-1][2])


def gain_db_to_levels(delta_db, at_level):
    """Convert a dB difference into console clicks, at the level it is measured from."""
    return float(delta_db) / gain_db_per_level(at_level)

# --- Where the display window is anchored ----------------------------------------
# Changing the dynamic range rotates the dB-to-gray mapping about a fixed gray level,
# it does not slide a window down from a fixed top. Measured in 20260904_DR: at each
# depth, gray is linear in 1/window_dB (median R^2 0.975 over 300 depths) and every
# depth extrapolates to the same intercept.
#
#   intercept over 300 depths: median 43.0 gray, quartiles 42.2 - 43.7
#
# Solving back for (display_dB - pivot_dB) with this pivot reproduces each depth to a
# median spread of 0.62 dB across 13 frames spanning dynamic range 30 to 400.
#
# Only one session has ever swept the dynamic range, so treat this as a machine constant
# measured once rather than one confirmed twice.
GRAY_PIVOT = 43.0


def dr_ui_to_window_db(ui_value):
    """Convert the console's "dynamic range" number to the window width in dB.

    The UI number is a scale, not a physical quantity; see DR_WINDOW_SLOPE for the
    measurement. Values outside 30..400 are extrapolated rather than clamped, so a caller
    passing nonsense gets an obviously odd window instead of a silent clip.
    """
    return DR_WINDOW_SLOPE * float(ui_value) + DR_WINDOW_INTERCEPT


def capture_window_db(capture):
    """Window width in dB for a capture, from its recorded UIDynamicRangeLevel."""
    return dr_ui_to_window_db(capture.dynamic_range_level)


def gain_level_to_db(level, reference_level=CALIBRATION_GAIN_LEVEL):
    """dB the console's gain adds at one level, relative to another.

    Integrates GAIN_DB_PER_LEVEL_TABLE rather than multiplying by one slope: the ladder is
    coarser above level 127 than below, so a span that crosses the boundary is not the level
    difference times any single constant.
    """
    level = float(level)
    reference_level = float(reference_level)
    sign = 1.0 if level >= reference_level else -1.0
    low, high = min(level, reference_level), max(level, reference_level)
    total = 0.0
    for start, stop, slope in GAIN_DB_PER_LEVEL_TABLE:
        span = max(0.0, min(high, stop) - max(low, start))
        total += span * slope
    return sign * total


def bc0_to_db(bc0, counts_per_db=DEFAULT_COUNTS_PER_DB):
    """Convert raw BC0 counts to dB."""
    return np.asarray(bc0, dtype=np.float64) / float(counts_per_db)


def db_to_bc0(db, counts_per_db=DEFAULT_COUNTS_PER_DB):
    """Convert dB back to raw BC0 counts."""
    return np.asarray(db, dtype=np.float64) * float(counts_per_db)


def tgc_level_to_db(levels, db_per_level=DEFAULT_DB_PER_LEVEL, center=TGC_CENTER_LEVEL):
    """Convert TGC slider levels to the additive dB gain they apply to the display."""
    return (np.asarray(levels, dtype=np.float64) - float(center)) * float(db_per_level)


def tgc_db_to_level(db, db_per_level=DEFAULT_DB_PER_LEVEL, center=TGC_CENTER_LEVEL):
    """Convert a desired additive dB gain to the nearest valid TGC slider level."""
    levels = np.asarray(db, dtype=np.float64) / float(db_per_level) + float(center)
    return np.clip(np.round(levels), TGC_MIN_LEVEL, TGC_MAX_LEVEL).astype(np.int32)


def expand_tgc_to_depth(levels, num_points, db_per_level=DEFAULT_DB_PER_LEVEL):
    """Interpolate the eight TGC slider levels onto a per-depth dB gain curve.

    Gains are placed at the band centres and interpolated linearly, held flat outside the
    outermost centres, which matches the smooth curve the console draws over the image.
    """
    levels = np.asarray(levels, dtype=np.float64).reshape(-1)
    centres = band_centres(num_points, levels.size)
    gains_db = tgc_level_to_db(levels, db_per_level)
    return np.interp(np.arange(int(num_points)), centres, gains_db)


def apply_tgc(bc0_db, tgc_levels, db_per_level=DEFAULT_DB_PER_LEVEL):
    """Add the depth-dependent TGC gain to a (depth, line) dB image."""
    bc0_db = np.asarray(bc0_db, dtype=np.float64)
    curve = expand_tgc_to_depth(tgc_levels, bc0_db.shape[0], db_per_level)
    return bc0_db + curve[:, None]


def db_to_gray(db_image, dynamic_range_db, reference_db=DEFAULT_REFERENCE_DB, graymap_lut=None,
               gray_pivot=GRAY_PIVOT):
    """Map an absolute dB image onto 0..255 display gray, optionally through a graymap LUT.

    reference_db is the display-dB level that lands on gray_pivot. Widening
    dynamic_range_db rotates the mapping about that point rather than dropping the floor,
    which is what the console actually does; see GRAY_PIVOT.
    """
    dynamic_range_db = max(1.0, float(dynamic_range_db))
    gray = float(gray_pivot) + (
        np.asarray(db_image, dtype=np.float64) - float(reference_db)
    ) / dynamic_range_db * GRAY_MAX
    gray = np.clip(np.round(gray), 0.0, GRAY_MAX).astype(np.uint8)
    return gray if graymap_lut is None else np.asarray(graymap_lut, dtype=np.uint8)[gray]


def scan_convert_linear(image, out_height, out_width):
    """Resample a (depth, line) image onto the display raster.

    L15-4 is a linear array (ProbeType 1, ProbeRadius 0, BShape 1), so scan conversion is a
    plain separable resample; no polar-to-Cartesian mapping is involved.
    """
    from PIL import Image

    array = np.asarray(image, dtype=np.float32)
    resized = Image.fromarray(array, mode="F").resize(
        (int(round(out_width)), int(round(out_height))), Image.BILINEAR
    )
    return np.asarray(resized, dtype=np.float64)


def render_db(
    bc0=None,
    tgc_levels=None,
    gain_db=0.0,
    depth_response_db=None,
    counts_per_db=DEFAULT_COUNTS_PER_DB,
    db_per_level=DEFAULT_DB_PER_LEVEL,
    db_image=None,
):
    """Render to an absolute dB image without the display clip, for metric work.

    Takes either raw BC0 counts or an image that is already in dB. Field II shards are
    floating-point envelopes (see fieldii_loader), so pushing them through a counts scale
    and straight back out again would be a detour; pass db_image instead.
    """
    if (bc0 is None) == (db_image is None):
        raise ValueError("Pass exactly one of bc0 or db_image")
    if db_image is None:
        db_image = bc0_to_db(bc0, counts_per_db)
    else:
        db_image = np.array(db_image, dtype=np.float64, copy=True)
    if depth_response_db is not None:
        response = np.asarray(depth_response_db, dtype=np.float64).reshape(-1)
        if response.size != db_image.shape[0]:
            raise ValueError(
                f"depth_response_db length {response.size} does not match BC0 depth {db_image.shape[0]}"
            )
        db_image = db_image + response[:, None]
    if tgc_levels is not None:
        db_image = apply_tgc(db_image, tgc_levels, db_per_level)
    return db_image + float(gain_db)


def render(
    bc0=None,
    tgc_levels=None,
    gain_db=0.0,
    dynamic_range_db=67.0,
    depth_response_db=None,
    reference_db=DEFAULT_REFERENCE_DB,
    counts_per_db=DEFAULT_COUNTS_PER_DB,
    db_per_level=DEFAULT_DB_PER_LEVEL,
    graymap_lut=None,
    out_shape=None,
    db_image=None,
):
    """Render raw BC0 through the simulated back end into a uint8 display image.

    gain_db is an absolute dB offset relative to the calibration gain (BUIGainLevel 75, which
    is what reference_db corresponds to); use gain_level_to_db() to convert a console gain
    level. dynamic_range_db is the window *width in dB*, not the console's UI number - pass
    dr_ui_to_window_db(UIDynamicRangeLevel) when starting from a capture.

    Pass db_image instead of bc0 for a source that is already in dB; see render_db().
    """
    db_image = render_db(
        bc0,
        tgc_levels,
        gain_db,
        depth_response_db,
        counts_per_db,
        db_per_level,
        db_image=db_image,
    )
    if out_shape is not None:
        db_image = scan_convert_linear(db_image, out_shape[0], out_shape[1])
    return db_to_gray(db_image, dynamic_range_db, reference_db, graymap_lut)


def gray_to_db(gray, dynamic_range_db, reference_db=DEFAULT_REFERENCE_DB, gray_pivot=GRAY_PIVOT):
    """Invert db_to_gray for unclipped pixels."""
    dynamic_range_db = max(1.0, float(dynamic_range_db))
    return (np.asarray(gray, dtype=np.float64) - float(gray_pivot)) * (
        dynamic_range_db / GRAY_MAX
    ) + float(reference_db)


def calibrate_depth_response(
    capture,
    reference_db=None,
    counts_per_db=None,
    gray_low=10,
    gray_high=245,
    min_valid_pixels=30,
    db_per_level=DEFAULT_DB_PER_LEVEL,
):
    """Measure the per-depth dB correction between BC0 and the displayed image.

    The capture must have a flat TGC curve, so that the TGC contributes one constant instead
    of a depth-dependent term. That constant is then subtracted, which matters because the
    console has no numeric TGC entry: the operator drags sliders or taps a preset, so the flat
    reference will not necessarily sit at the neutral level. Without the subtraction the
    reference's own gain is folded into the curve and then applied a second time by render().

    Rows whose display pixels are clipped carry no information and are interpolated across;
    the returned mask flags which BC0 depth samples were measured directly.

    Returns (depth_response_db, measured_mask), both on the BC0 depth grid.
    """
    if not is_flat_tgc(capture):
        raise ValueError("Depth-response calibration needs a capture with a flat TGC curve")

    if counts_per_db is None or reference_db is None:
        fitted_counts, fitted_reference, _ = calibrate_counts_per_db(capture)
        counts_per_db = fitted_counts if counts_per_db is None else counts_per_db
        reference_db = fitted_reference if reference_db is None else reference_db

    screen = crop_capture_image(capture)[0]
    dynamic_range_db = capture_window_db(capture)
    screen_db = gray_to_db(screen, dynamic_range_db, reference_db)
    bc0_db = bc0_to_db(scan_convert_linear(capture.bc0, screen.shape[0], screen.shape[1]), counts_per_db)

    usable = (screen > gray_low) & (screen < gray_high)
    reference_gain_db = float(tgc_level_to_db(capture.tgc_levels[0], db_per_level))
    residual = screen_db - bc0_db - reference_gain_db
    rows = np.arange(screen.shape[0])
    measured = np.array([usable[row].sum() >= min_valid_pixels for row in rows])
    if measured.sum() < 2:
        raise ValueError("Too few unclipped rows to calibrate the depth response")

    curve = np.full(screen.shape[0], np.nan)
    for row in rows[measured]:
        curve[row] = np.median(residual[row][usable[row]])
    curve = np.interp(rows, rows[measured], curve[measured])

    # Move from the display raster onto the BC0 depth grid.
    num_points = capture.bc0.shape[0]
    bc0_rows = np.linspace(0, screen.shape[0] - 1, num_points)
    response = np.interp(bc0_rows, rows, curve)
    response_mask = np.interp(bc0_rows, rows, measured.astype(float)) > 0.5
    return response, response_mask


def _band_medians(image, num_bands=NUM_TGC_BANDS):
    """Median of each depth band of a (depth, ...) image."""
    edges = band_edges(image.shape[0], num_bands)
    return np.array([np.median(image[edges[k]:edges[k + 1]]) for k in range(num_bands)])


def is_flat_tgc(capture):
    """True if every TGC slider of the capture sits at the same level."""
    return len(set(np.asarray(capture.tgc_levels).tolist())) == 1


def calibrate_tgc_db_per_level(
    captures, level_range=CALIBRATED_LEVEL_RANGE, db_per_level=DEFAULT_DB_PER_LEVEL
):
    """Fit dB-per-slider-level from one or more TGC sweeps.

    Captures are grouped by (gain level, dynamic range), and each group is referenced to its
    own flat-TGC capture. That is what lets a low-end sweep be acquired at a raised gain: the
    global gain offset cancels exactly in the flat-referenced delta, so groups acquired at
    different gains can be pooled into a single fit. Groups without a flat reference, and
    groups that are entirely flat, contribute nothing and are skipped.

    Where a group holds several flat captures (a repeated baseline), their mean is used as the
    reference so the repeats improve it rather than being treated as measurements. Each flat
    carries its own constant gain, which is removed before averaging: the console has no
    numeric TGC entry, so repeated flat references set by dragging sliders will not land on
    exactly the same level. That removal uses the db_per_level argument as a first-order
    correction; it is a fraction of a dB and does not meaningfully feed back into the fit.

    Returns (db_per_level, intercept_db, rms_residual_db, num_points). intercept_db is the
    fitted gain at slider TGC_CENTER_LEVEL; a value near zero means the swept measurements
    extrapolate linearly back to zero gain at the centre, i.e. it is a linearity and
    consistency check on the anchor, not independent proof that 127 is the neutral point.
    """
    captures = list(captures)
    if not captures:
        raise ValueError("TGC calibration needs at least one capture")

    groups = {}
    for capture in captures:
        groups.setdefault((capture.gain_level, capture.dynamic_range_level), []).append(capture)

    levels, gains_db = [], []
    for (_, dynamic_range_level), group in sorted(groups.items()):
        flats = [capture for capture in group if is_flat_tgc(capture)]
        swept = [capture for capture in group if not is_flat_tgc(capture)]
        if not flats or not swept:
            continue

        gray_per_db = GRAY_MAX / float(dynamic_range_level)
        ref_bands = np.mean(
            [
                _band_medians(crop_capture_image(flat)[0])
                - gray_per_db * float(tgc_level_to_db(flat.tgc_levels[0], db_per_level))
                for flat in flats
            ],
            axis=0,
        )
        for capture in swept:
            bands = _band_medians(crop_capture_image(capture)[0])
            for level, delta_gray in zip(capture.tgc_levels, bands - ref_bands):
                if level_range[0] <= level <= level_range[1]:
                    levels.append(float(level))
                    gains_db.append(delta_gray / gray_per_db)

    if len(levels) < 2:
        raise ValueError(
            "Not enough unsaturated TGC samples to fit; check that each gain group has both a "
            "flat-TGC reference and a swept capture, or widen level_range"
        )

    centred = np.asarray(levels) - TGC_CENTER_LEVEL
    design = np.vstack([centred, np.ones_like(centred)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, np.asarray(gains_db), rcond=None)
    residual = np.asarray(gains_db) - design @ [slope, intercept]
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual**2))), len(levels)


def calibrate_counts_per_db(capture, gray_low=5, gray_high=250):
    """Fit BC0 counts per dB, and the display reference level, from one capture.

    Uses the unsaturated middle half of the image, where displayed gray is affine in BC0.
    Returns (counts_per_db, reference_db, slope_gray_per_count).
    """
    screen = crop_capture_image(capture)[0]
    resampled = scan_convert_linear(capture.bc0, screen.shape[0], screen.shape[1])

    start, stop = 3 * screen.shape[0] // 8, 5 * screen.shape[0] // 8
    counts = resampled[start:stop].ravel()
    gray = screen[start:stop].ravel()
    usable = (gray > gray_low) & (gray < gray_high)
    if usable.sum() < 100:
        raise ValueError("Not enough unsaturated pixels to fit the display reference")

    slope, offset = np.polyfit(counts[usable], gray[usable], 1)
    dynamic_range_db = capture_window_db(capture)
    counts_per_db = (GRAY_MAX / dynamic_range_db) / slope
    # gray == GRAY_PIVOT marks the level the dynamic-range control pivots about.
    reference_db = ((GRAY_PIVOT - offset) / slope) / counts_per_db
    return float(counts_per_db), float(reference_db), float(slope)


def flat_tgc_capture(captures):
    """Return the capture whose TGC curve is flat, which is the calibration reference."""
    for capture in captures:
        if is_flat_tgc(capture):
            return capture
    raise ValueError("No flat-TGC capture available to use as the calibration reference")


def validate_capture(capture, depth_response_db, reference_db, counts_per_db, db_per_level=DEFAULT_DB_PER_LEVEL,
                     calibration_gain_level=CALIBRATION_GAIN_LEVEL):
    """Compare a simulated render of one capture against its own console screenshot.

    The capture's own gain is applied, relative to the gain the reference_db was fitted at.
    This term used to be missing because BUIGainLevel had no dB calibration, which made
    every capture at a different gain look like a model failure rather than a missing
    term; GAIN_DB_PER_LEVEL fixes that.

    Returns (actual_bands, predicted_bands) as per-TGC-band median gray levels.
    """
    actual = crop_capture_image(capture)[0]
    predicted = render(
        capture.bc0,
        tgc_levels=capture.tgc_levels,
        gain_db=gain_level_to_db(capture.gain_level, calibration_gain_level),
        dynamic_range_db=capture_window_db(capture),
        depth_response_db=depth_response_db,
        reference_db=reference_db,
        counts_per_db=counts_per_db,
        db_per_level=db_per_level,
        out_shape=actual.shape,
    ).astype(np.float64)
    return _band_medians(actual), _band_medians(predicted)


def build_parser():
    parser = argparse.ArgumentParser(description="Fit Hisense back-end model constants from a TGC sweep.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory holding capture folders.")
    return parser


def main():
    args = build_parser().parse_args()
    captures = [load_capture(path) for path in find_captures(args.data_dir)]
    if not captures:
        raise SystemExit(f"No captures found under {args.data_dir}")

    slope, intercept, rms, count = calibrate_tgc_db_per_level(captures)
    print("TGC slider calibration")
    print(f"  dB per level    : {slope:.5f}  (shipped default {DEFAULT_DB_PER_LEVEL:.5f})")
    print(f"  span 0..255     : {slope * 255:.2f} dB, centre +/-{slope * 127.5:.2f} dB")
    print(f"  intercept at {TGC_CENTER_LEVEL}: {intercept:+.3f} dB  (near zero = linear back to the anchor)")
    print(f"  RMS residual    : {rms:.3f} dB over {count} samples")

    flat = flat_tgc_capture(captures)
    counts_per_db, reference_db, slope_gray = calibrate_counts_per_db(flat)
    print(f"\nDisplay calibration from {flat.name} (gain {flat.gain_level}, DR UI {flat.dynamic_range_level} -> {capture_window_db(flat):.1f} dB)")
    print(f"  counts per dB   : {counts_per_db:.1f}  (shipped default {DEFAULT_COUNTS_PER_DB:.1f})")
    print(f"  reference dB    : {reference_db:.2f}  (shipped default {DEFAULT_REFERENCE_DB:.2f})")
    print(f"  gray per count  : {slope_gray:.6f}")
    print(f"  BC0 span        : {(flat.bc0.max() - flat.bc0.min()) / counts_per_db:.1f} dB")

    response, measured = calibrate_depth_response(flat, reference_db, counts_per_db)
    print(f"\nDepth response from {flat.name}")
    print(f"  measured directly: {measured.sum()}/{measured.size} depth samples")
    print(f"  range            : {response.min():+.1f} .. {response.max():+.1f} dB")

    print("\nSimulated vs console screenshot, per TGC band (gray levels)")
    gray_per_db = GRAY_MAX / capture_window_db(flat)
    for capture in captures:
        actual, predicted = validate_capture(
            capture, response, reference_db, counts_per_db, slope, flat.gain_level
        )
        error = np.abs(predicted - actual)
        held_out = "reference" if capture.path == flat.path else "held out"
        print(f"  {capture.name} ({held_out})")
        print(f"    gain     : {capture.gain_level} "
              f"({gain_level_to_db(capture.gain_level, flat.gain_level):+.2f} dB vs reference)")
        print(f"    sliders  : {capture.tgc_levels.tolist()}")
        print(f"    actual   : {[round(v) for v in actual]}")
        print(f"    predicted: {[round(v) for v in predicted]}")
        print(f"    mean |error| = {error.mean():.1f} gray ({error.mean() / gray_per_db:.2f} dB)")


if __name__ == "__main__":
    main()
