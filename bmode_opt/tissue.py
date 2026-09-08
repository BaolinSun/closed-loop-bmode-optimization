"""Deciding which pixels hold tissue, from evidence rather than from a per-frame guess.

Every term of the back-end objective rests on this decision. A pixel with echo that renders
black is an exposure failure; a pixel with no echo that renders black is correct, and a cyst
is supposed to be black. Get the split wrong and the objective either punishes correct images
or, as it turned out, stops seeing failures at all.

    What was wrong

objective.signal_mask() took the frame's noise floor to be the median of its deepest ten
percent of rows, and called anything three dB above that signal. That is a guess, and it is
wrong on both data sources, in the same direction: it lands too high, so real tissue is
discarded as noise.

On Field II there is no floor to find. Every one of the 4560 shards carries noise_enabled=0,
and the depth trend is negative from the first row to the last - measured over three phantom
types and four settings, the local slope in the deepest tenth runs -0.11 to -0.98 dB/mm, never
flattening. The "floor" the estimator returns is deep tissue, and 18.9% of the frame goes with
it.

On the console there is a real floor, but a single frame is usually too shallow to see it.
Comparing the per-frame estimate against the floor measured from that session's deepest frames:

    display depth mm    harmonic overestimate dB    general overestimate dB
          25.1                  +11.63                      +15.69
          33.5                   +8.56                       +9.98
          41.9                   +4.71                    +5.08, +6.58
          50.2                   +1.27                       +0.87
          58.6                   +0.18                    +0.56, 0.00
          67.0                   -0.12                       -0.04

The estimate only converges once the display depth is well past the penetration limit, which
is 31.0 mm in harmonic mode and 40.6 mm at 11.4 MHz in general mode. Most of the data sits at
41.9 mm, where it is five to seven dB high.

    What is done instead

Field II: read the structure off the phantom. truth_mask is zero for tissue and carries the
cyst index elsewhere; point targets come with their coordinates. Nothing is estimated, and no
noise floor is applied, because there is no noise.

Console: measure the floor once per session and imaging mode, from the frames whose display
depth clears the penetration limit by a margin, then apply it to every frame in that group.
The two general-mode sessions that can be measured independently agree to 0.32 dB, and within
a group the frame-to-frame spread is 0.44 to 0.85 dB.

Where a group has no frame deep enough, this module says so rather than guessing. Borrowing a
floor from another session is offered but flagged, because counts_per_db has been seen to
differ by 20% between sessions, which is about 2.6 dB on a 13 dB floor.

    A note on the protocol's 15.85 dB

Section 7.8.5 of the acquisition protocol uses 15.85 dB as the 20260903 noise floor. Measured
from that session's deep harmonic frames the answer is 12.86 dB with a standard deviation of
0.44. Pooling all 74 frames regardless of display depth instead gives 16.45 dB with a standard
deviation of 4.21 - which reproduces both the magnitude and the scatter of the published
figure, and identifies the cause: shallow frames whose deepest rows are still tissue were
averaged in. The deep-frame value is the one to use.
"""

import numpy as np

from hisense_loader import get_leaf


# A frame can only show its own noise floor if the display window reaches well past where the
# echo dies. Penetration was measured at 31.0 mm for harmonic and 40.6 mm at 11.4 MHz for
# general mode, so 58 mm leaves a margin of at least 17 mm on the worse case; it also happens
# to be the second-deepest rung of the console's depth ladder.
NOISE_FLOOR_MIN_DEPTH_MM = 58.0

# Deepest fraction of a qualifying frame used for the measurement. Widening this to a full
# eighth moves the answer by 0.04 dB, so the choice is not load-bearing.
NOISE_FLOOR_ROWS_FRACTION = 0.10

# How far above the floor a console pixel has to sit to count as tissue. Three dB is the
# margin the protocol already uses when it reports penetration depth.
DEFAULT_TISSUE_MARGIN_DB = 3.0

# Radius excluded around each Field II point target. The targets are specular reflectors at
# amplitude 100 against speckle at 0.15, so they are not background tissue and would drag any
# level statistic upwards; a millimetre is several resolution cells at every frequency in the
# grid.
POINT_TARGET_EXCLUSION_MM = 1.0

IMAGE_MODE_NAMES = {0: "general", 1: "harmonic"}


def capture_image_mode(capture):
    """Imaging mode of a console capture, as the BImageMode integer."""
    return int(get_leaf(capture.fe_params, "BImageMode"))


def fieldii_tissue_mask(capture, exclusion_mm=POINT_TARGET_EXCLUSION_MM):
    """Tissue pixels of a Field II shard, from the phantom's own structure.

    Tissue is everything the truth mask marks zero, minus a neighbourhood of each point
    target. No noise floor enters: the shards are generated with noise_enabled=0, so every
    row down to the last one carries echo, and thresholding would only remove deep tissue.
    """
    mask = np.asarray(capture.truth_mask, dtype=np.uint8) == 0
    targets = np.asarray(capture.point_targets_mm, dtype=np.float64)
    if targets.size == 0:
        return mask

    geometry = capture.geometry
    rows = np.arange(mask.shape[0], dtype=np.float64) * geometry.mm_per_point + geometry.min_depth_mm
    columns = (np.arange(mask.shape[1], dtype=np.float64) - 0.5 * (mask.shape[1] - 1)) * geometry.mm_per_line
    depth_grid, lateral_grid = np.meshgrid(rows, columns, indexing="ij")
    for depth_mm, lateral_mm in targets.reshape(-1, 2):
        near = np.hypot(depth_grid - depth_mm, lateral_grid - lateral_mm) <= float(exclusion_mm)
        mask &= ~near
    return mask


def measure_noise_floor(captures, bc0_to_db, min_depth_mm=NOISE_FLOOR_MIN_DEPTH_MM,
                        rows_fraction=NOISE_FLOOR_ROWS_FRACTION):
    """Noise floor in dB for console captures sharing one session and imaging mode.

    bc0_to_db converts a capture's raw counts to dB and must carry that session's own
    calibration, since counts_per_db is a per-session quantity.

    Returns None when no capture reaches min_depth_mm. That is the honest outcome, not a
    failure: without a frame that sees past the echo there is nothing in the data to measure.
    """
    qualifying = [c for c in captures if c.geometry.depth_mm >= float(min_depth_mm)]
    if not qualifying:
        return None
    values = []
    for capture in qualifying:
        db_image = bc0_to_db(capture)
        start = int(db_image.shape[0] * (1.0 - float(rows_fraction)))
        values.append(float(np.median(db_image[start:])))
    values = np.asarray(values, dtype=np.float64)
    return {
        "floor_db": float(np.median(values)),
        "std_db": float(np.std(values)),
        "range_db": float(values.max() - values.min()),
        "num_frames": int(values.size),
        "min_depth_mm": float(min_depth_mm),
        "borrowed_from": None,
    }


def measure_session_noise_floors(captures_by_group, bc0_to_db, allow_borrowing=True,
                                 min_depth_mm=NOISE_FLOOR_MIN_DEPTH_MM):
    """Noise floors for every (session, imaging mode) group, borrowing where allowed.

    captures_by_group maps a (session, mode) key to its captures. Groups with no deep enough
    frame get the floor of another group in the same imaging mode, marked with borrowed_from so
    a caller can treat those frames differently - the borrow is across a calibration boundary
    and has not been checked.
    """
    measured = {}
    for key, captures in captures_by_group.items():
        result = measure_noise_floor(captures, bc0_to_db, min_depth_mm)
        if result is not None:
            measured[key] = result

    floors = dict(measured)
    if allow_borrowing:
        for key in captures_by_group:
            if key in floors:
                continue
            mode = key[1]
            donors = [(k, v) for k, v in measured.items() if k[1] == mode]
            if not donors:
                continue
            donor_key, donor = min(donors, key=lambda item: -item[1]["num_frames"])
            floors[key] = dict(donor, borrowed_from=donor_key)
    return floors


def console_tissue_mask(db_image, floor_db, margin_db=DEFAULT_TISSUE_MARGIN_DB):
    """Tissue pixels of a console frame, given its group's measured noise floor.

    floor_db must come from measure_noise_floor(), not from the frame itself; a frame shallower
    than the penetration limit cannot see its own floor and estimating it there discards
    tissue, by five to sixteen dB worth on the sessions acquired so far.
    """
    if floor_db is None:
        raise ValueError(
            "No measured noise floor for this frame's group. Acquire a frame at "
            f"{NOISE_FLOOR_MIN_DEPTH_MM:.0f} mm or deeper in the same session and mode, or "
            "drop the frame as undecidable - do not estimate the floor from the frame itself."
        )
    return np.asarray(db_image, dtype=np.float64) > float(floor_db) + float(margin_db)
