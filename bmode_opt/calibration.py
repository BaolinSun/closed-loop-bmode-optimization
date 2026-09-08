"""Per-session, per-mode calibration of the constants the back-end simulator runs on.

    Why one set of constants is not enough

hisense_backend_sim ships DEFAULT_COUNTS_PER_DB = 877.3 and DEFAULT_PIVOT_DB = 25.09, both
fitted on one session in harmonic mode. Predicting each group's own screenshots with them, as
mean absolute per-band gray error:

    session                       mode        with the defaults    with this module
    20260903                      general           30.33                3.79
    20260903_GEN                  general           26.91                6.82
    20260904                      general           29.05                5.02
    20260903                      harmonic           9.94                3.93
    20260903_replication_check    harmonic           9.65                3.12
    20260904                      harmonic          10.90                1.75
    20260904_DR                   harmonic           8.92                1.78

General mode is off by around 30 gray levels under the harmonic-fitted constants. That is a
large enough error to move every threshold in the objective, and it is the reason this exists.

    Why the two scalars have to be fitted with the depth response, not before it

The per-frame fit calibrate_counts_per_db() does regresses displayed gray on BC0 counts and
derives both constants from one slope and one offset. Within a group the two come out strongly
anti-correlated - on 20260903 harmonic at 25.1 mm the pairs run (713, 37.6), (795, 32.4),
(827, 31.9), (1046, 25.1) - so the pair is constrained and neither member is. It also drifts
with a setting it cannot physically depend on: over the 20260904_DR sweep, where only the
display window moves, it reads 818, 843, 866, 877, 886, 899, 931 counts per dB.

Fitting the two scalars against screenshot error but with the depth response held at zero does
not fix it either. They then have to absorb a depth-dependent term they cannot represent, and
the harmonic groups land anywhere between 544 and 1459 counts per dB with 9 to 15 gray of
residual. Solving both together, alternating, brings the residual down to the table above.

    What is still not separated

Even solved jointly, counts_per_db and pivot_db stay anti-correlated: general reads 579, 772,
603 across three sessions and harmonic 884, 819, 679, 923 across four, about 30% either way,
while the paired pivot moves the opposite direction each time. The pair predicts screenshots
well; the members individually are not pinned. So use them together, within their own group,
and do not compare dB values across groups - a noise floor of 12.8 dB in one group and 5.3 in
another is not evidence about noise, only about two different dB scales.

The mode gap narrowed from 61% to 27% once the depth response was included, which is inside the
within-mode scatter. So the earlier reading that counts per dB is systematically mode-dependent
does not survive; what does survive is that the constants are group-specific.

The recovered depth responses do reproduce. Sampled every 5 mm, the four harmonic sessions give
(1.1, -3.6, -5.6, -2.8, 0.3, 1.7, ...), (-0.4, -3.8, -6.7, -3.3, -0.1, 1.0, ...),
(-2.7, -5.0, -7.5, -3.4, 0.4, 1.1, ...) and (1.1, -5.3, -6.6, -3.1, 0.7, 1.3, ...) - the same
curve, independently fitted on four sessions.
"""

from dataclasses import dataclass, field

import numpy as np

from hisense_backend_sim import (
    CALIBRATION_GAIN_LEVEL,
    GRAY_MAX,
    GRAY_PIVOT,
    DEFAULT_COUNTS_PER_DB,
    DEFAULT_DB_PER_LEVEL,
    DEFAULT_PIVOT_DB,
    bc0_to_db,
    capture_window_db,
    gain_level_to_db,
    gray_to_db,
    is_flat_tgc,
    render,
    scan_convert_linear,
    tgc_level_to_db,
)
from hisense_loader import NUM_TGC_BANDS, crop_capture_image


# Gray levels outside this band carry no information about the mapping, because the display has
# clipped there and the relation between dB and gray is flat.
USABLE_GRAY = (10, 245)

# Frames used to fit one group. The fit is over two scalars, so a dozen frames spanning the
# group's display depths and gains is plenty, and it keeps a full sweep affordable.
DEFAULT_FIT_FRAMES = 12


@dataclass
class GroupCalibration:
    """Constants for one (session, imaging mode) group, and how well they predict screenshots."""

    counts_per_db: float
    pivot_db: float
    gray_error: float
    num_fit_frames: int
    depth_axis_mm: np.ndarray = field(default=None)
    depth_response_db: np.ndarray = field(default=None)

    def db_image(self, capture):
        """The capture's BC0 in dB under this group's calibration."""
        return bc0_to_db(capture.bc0, self.counts_per_db)


def _select_fit_frames(captures, limit=DEFAULT_FIT_FRAMES):
    """A spread of frames over display depth and gain, so the fit is not one operating point."""
    usable = [c for c in captures if is_flat_tgc(c)]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda c: (c.geometry.depth_mm, c.gain_level,
                                            c.dynamic_range_level, c.name))
    if len(ordered) <= limit:
        return ordered
    picks = np.linspace(0, len(ordered) - 1, limit).round().astype(int)
    return [ordered[index] for index in sorted(set(picks.tolist()))]


def screenshot_gray_error(capture, counts_per_db, pivot_db, depth_response_db=None,
                          num_bands=NUM_TGC_BANDS,
                          calibration_gain_level=CALIBRATION_GAIN_LEVEL):
    """Mean absolute per-band gray error between a simulated render and the real screenshot."""
    actual = crop_capture_image(capture)[0]
    predicted = render(
        capture.bc0,
        tgc_levels=capture.tgc_levels,
        gain_db=gain_level_to_db(capture.gain_level, calibration_gain_level),
        dynamic_range_db=capture_window_db(capture),
        depth_response_db=depth_response_db,
        reference_db=pivot_db,
        counts_per_db=counts_per_db,
        out_shape=actual.shape,
    ).astype(np.float64)
    edges = np.linspace(0, actual.shape[0], num_bands + 1).round().astype(int)
    errors = [
        abs(np.median(actual[edges[k]:edges[k + 1]]) - np.median(predicted[edges[k]:edges[k + 1]]))
        for k in range(num_bands)
    ]
    return float(np.mean(errors))


# Depth bands used inside the fit. Finer than the eight TGC bands, because these also carry the
# only samples of the depth response, and eight points per frame is thin for a curve.
FIT_BANDS = 32


def band_summary(capture, num_bands=FIT_BANDS, db_per_level=DEFAULT_DB_PER_LEVEL):
    """Everything about one frame the fit needs, reduced to a few dozen numbers.

    Scan conversion is bilinear and the counts-to-dB step is a division, so they commute, and a
    flat TGC curve plus the gain are constants that pass through both. The display map is
    monotone, so the median of a rendered band is the rendered value of that band's median.
    That collapses a full render per trial into arithmetic on a short vector, which is what
    makes the search affordable.

    Band centres are carried in millimetres, not row index. Frames in a group sit at several
    display depths, and the console's own depth gain follows depth; without the physical axis
    the same band index from two frames would be averaged as though it were the same place.
    """
    actual = crop_capture_image(capture)[0]
    counts = scan_convert_linear(capture.bc0, actual.shape[0], actual.shape[1])
    edges = np.linspace(0, actual.shape[0], num_bands + 1).round().astype(int)
    centres = (0.5 * (edges[:-1] + edges[1:]) / actual.shape[0]) * capture.geometry.depth_mm
    return {
        "actual": np.array([np.median(actual[edges[k]:edges[k + 1]]) for k in range(num_bands)]),
        "counts": np.array([np.median(counts[edges[k]:edges[k + 1]]) for k in range(num_bands)]),
        "centres_mm": centres,
        "offset_db": float(tgc_level_to_db(capture.tgc_levels[0], db_per_level))
                     + gain_level_to_db(capture.gain_level),
        "window_db": capture_window_db(capture),
        "depth_mm": float(capture.geometry.depth_mm),
    }


def _response_at(summary, axis_mm, response_db):
    if response_db is None:
        return 0.0
    return np.interp(summary["centres_mm"], axis_mm, response_db,
                     left=response_db[0], right=response_db[-1])


def _predicted_bands(summary, counts_per_db, pivot_db, axis_mm=None, response_db=None):
    level = (summary["counts"] / float(counts_per_db)
             + _response_at(summary, axis_mm, response_db)
             + summary["offset_db"] - float(pivot_db))
    raw = GRAY_PIVOT + level / summary["window_db"] * GRAY_MAX
    return np.clip(np.round(raw), 0.0, GRAY_MAX)


def group_cost(summaries, counts_per_db, pivot_db, axis_mm=None, response_db=None):
    """Mean absolute per-band gray error over a group's frames."""
    return float(np.mean([
        np.mean(np.abs(s["actual"] - _predicted_bands(s, counts_per_db, pivot_db,
                                                      axis_mm, response_db)))
        for s in summaries]))


def _solve_response(summaries, counts_per_db, pivot_db, axis_mm, smooth_mm=2.0):
    """The depth-dependent dB the two scalars cannot account for.

    Inverted from each band's own screenshot level, then pooled by physical depth across the
    group. Bands whose display gray has clipped are dropped: there the mapping is flat and the
    inversion has no information in it.
    """
    depths, values = [], []
    for summary in summaries:
        usable = (summary["actual"] > USABLE_GRAY[0]) & (summary["actual"] < USABLE_GRAY[1])
        if not usable.any():
            continue
        wanted = ((summary["actual"][usable] - GRAY_PIVOT) / GRAY_MAX * summary["window_db"]
                  + float(pivot_db) - summary["offset_db"]
                  - summary["counts"][usable] / float(counts_per_db))
        depths.append(summary["centres_mm"][usable])
        values.append(wanted)
    if not depths:
        return np.zeros_like(axis_mm)
    depths = np.concatenate(depths)
    values = np.concatenate(values)
    curve = np.empty_like(axis_mm)
    for index, depth in enumerate(axis_mm):
        near = np.abs(depths - depth) <= float(smooth_mm)
        curve[index] = np.median(values[near]) if near.sum() >= 3 else np.nan
    good = np.isfinite(curve)
    if good.sum() < 2:
        return np.zeros_like(axis_mm)
    return np.interp(axis_mm, axis_mm[good], curve[good])


def fit_group(captures, counts_range=(300.0, 1600.0), pivot_range=(5.0, 50.0),
              coarse=41, refinements=4, limit=DEFAULT_FIT_FRAMES, iterations=5,
              num_axis_points=128):
    """Fit counts_per_db, pivot_db and the depth response for one group, against screenshots.

    Alternates: solve the two scalars on a grid with the current depth response held, then
    re-solve the depth response as the residual. Fitting the scalars alone does not work - with
    the depth response forced to zero they have to absorb a depth-dependent term they cannot
    represent, and the harmonic groups then land anywhere between 544 and 1459 counts per dB.

    Grid search rather than a gradient method: the criterion goes through a round and a clip, so
    it is piecewise constant in places and not differentiable.
    """
    frames = _select_fit_frames(captures, limit)
    if not frames:
        return None
    summaries = [band_summary(c) for c in frames]
    axis_mm = np.linspace(0.0, max(s["depth_mm"] for s in summaries), int(num_axis_points))
    response = np.zeros_like(axis_mm)
    best = None

    for _ in range(int(iterations)):
        low_counts, high_counts = counts_range
        low_pivot, high_pivot = pivot_range
        best = None
        for _ in range(int(refinements) + 1):
            for counts_per_db in np.linspace(low_counts, high_counts, coarse):
                for pivot_db in np.linspace(low_pivot, high_pivot, coarse):
                    value = group_cost(summaries, counts_per_db, pivot_db, axis_mm, response)
                    if best is None or value < best[0]:
                        best = (value, float(counts_per_db), float(pivot_db))
            _, counts_per_db, pivot_db = best
            counts_step = (high_counts - low_counts) / (coarse - 1)
            pivot_step = (high_pivot - low_pivot) / (coarse - 1)
            low_counts, high_counts = counts_per_db - counts_step, counts_per_db + counts_step
            low_pivot, high_pivot = pivot_db - pivot_step, pivot_db + pivot_step
        response = _solve_response(summaries, best[1], best[2], axis_mm)

    return GroupCalibration(
        counts_per_db=best[1], pivot_db=best[2],
        gray_error=group_cost(summaries, best[1], best[2], axis_mm, response),
        num_fit_frames=len(frames), depth_axis_mm=axis_mm, depth_response_db=response)


def depth_response_for(capture, calibration):
    """The group's depth response resampled onto one capture's BC0 depth grid."""
    if calibration.depth_response_db is None:
        return None
    rows_mm = np.linspace(capture.geometry.min_depth_mm, capture.geometry.depth_mm,
                          capture.bc0.shape[0])
    return np.interp(rows_mm, calibration.depth_axis_mm, calibration.depth_response_db,
                     left=calibration.depth_response_db[0],
                     right=calibration.depth_response_db[-1])
