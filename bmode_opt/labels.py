"""One label schema for both data sources, with the uncertainty that belongs to each.

Field II frames pretrain and console frames fine-tune, so they have to carry the same fields.
What they cannot carry is the same confidence, and pretending otherwise would hide the one
thing a fine-tuning stage most needs to know.

    Why the two sources differ, and by how much

On a Field II frame the render *is* the ground truth: there is no console screenshot it has to
agree with, the phantom's structure is known exactly, and the objective scores exactly the
image the simulator produced. On a console frame the objective scores a rebuild, and the
rebuild is not the screenshot.

Measured on the three sessions that swept the back end at a fixed probe position, the error
J(rebuild) - J(screenshot) splits into two parts:

    session          settings   bias    residual   J span over settings   residual / span
    20260819            10     -0.038    0.036            0.916                3.9%
    20260901_E3          7     -0.032    0.089            0.781               11.3%
    20260904_DR         28     +0.022    0.238            3.391                7.0%

The bias is a constant and cannot move an argmin, so it is not an error in the label at all.
The residual is what can. It runs 4 to 11 percent of the range the objective spans across the
settings being chosen between, and the rank correlation between deciding by screenshot and
deciding by rebuild is 0.82 to 0.97. Where the two disagreed on the single best setting, they
disagreed between candidates whose true objectives differ by less than the residual - on
20260901_E3 by 0.030 against a residual of 0.089, on 20260819 by 0.010 against 0.036. That is a
tie being broken differently, not a wrong answer.

So a console label is usable, and its honest form is an optimum plus an equivalence band wide
enough to hold the simulator's own residual. That band is what label_uncertainty carries into
the tolerance, and it is why the direction fields here go through the band rather than through
a fixed threshold.

    What is not measured

Both modes now have a session that swept the back end at one probe position, so nothing is
borrowed across the mode boundary any more: 20260909_GEN measures 0.1102 for fundamental and
the three harmonic sessions give 0.046 to 0.309. Groups without their own sweep still borrow,
but from a session in their own mode.

What remains open is the gain ladder, not the uncertainty. Every harmonic capture sits between
gain 59 and 125 and every usable fundamental one between 129 and 229, so the 41% step in dB per
level at the boundary is equally consistent with the ladder being coarser at high gain and with
the two modes stepping differently. See hisense_backend_sim.GAIN_DB_PER_LEVEL_TABLE.

Dynamic range is reported unchanged with a zero delta on every frame; see backend_solver for
why the data cannot determine it.
"""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

import numpy as np

from hisense_backend_sim import (
    CALIBRATION_GAIN_LEVEL,
    DEFAULT_IMAGE_MODE,
    GAIN_DB_PER_LEVEL,
    gain_db_per_level,
    gain_db_to_levels,
    tgc_db_per_level,
    DEFAULT_DB_PER_LEVEL,
    TGC_CENTER_LEVEL,
    TGC_MAX_LEVEL,
    TGC_MIN_LEVEL,
)
from hisense_loader import NUM_TGC_BANDS
import backend_solver as BS


# The three slider groups the direction labels report. Eight numbers is more than a controller
# or an operator reasons about; near, mid and far is the vocabulary the acquisition protocol
# and the console's own presets already use.
SLIDER_GROUPS = {"near": (0, 3), "mid": (3, 5), "far": (5, 8)}

GAIN_DIRECTIONS = ("dark", "correct", "bright")
SLIDER_DIRECTIONS = ("low", "correct", "high")
DYNAMIC_RANGE_DIRECTIONS = ("narrow", "correct", "wide")


@dataclass
class FrameLabel:
    """Everything one frame contributes to training, in the same shape for both sources."""

    # Identity and split
    source: str                       # "fieldii" or "console"
    frame_id: str
    group_id: str                     # scene seed, or session and imaging mode
    split: Optional[str]

    # Front-end state the frame was acquired at
    depth_mm: float
    frequency_mhz: Optional[float]
    focus_mm: Optional[float]
    imaging_mode: str

    # Back-end state the frame was acquired at
    gain_db: float
    tgc_levels: List[int]
    dr_ui: float

    # The optimum, in the same units
    optimal_gain_db: float
    optimal_tgc_levels: List[int]
    optimal_dr_ui: float

    # Form B: relative direction, then the magnitude
    gain_direction: str
    slider_directions: dict
    dr_direction: str
    delta_gain_levels: float
    delta_tgc_levels: List[float]
    delta_dr_ui: float
    # dB per console click at the level this frame sits at. The gain ladder is coarser above
    # level 127 than below, so a delta in clicks cannot be converted back to dB without it.
    gain_db_per_level: float

    # How much of this to believe
    objective: float
    tolerance: float
    label_uncertainty: float
    deadband_gain_levels: float
    equivalent_count: int
    dr_determined: bool
    at_gain_edge: bool
    calibration_borrowed: bool
    notes: List[str] = field(default_factory=list)

    def as_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------------------
#   Where a frame's starting point comes from
#
# The label says how far it is from the current setting to the optimum, so every frame needs a
# current setting. A console frame has one already: wherever the operator left the knobs. A
# Field II frame does not, and the first version of this file used the neutral setting - gain 0,
# sliders flat at 127 - for all 4560 of them. The optimum is a property of the scene, so
# subtracting one fixed point from it produced one fixed answer: the gain label came out 'dark'
# on 4560 frames out of 4560, with a median delta of 34.99 clicks and a 5th-to-95th spread of
# only 17 clicks around it. That is not a label.
#
# So a starting point is drawn instead, over the span the console's own frames turned out to
# need: their gain deltas ran -17.5 to +25.0 clicks between the 5th and 95th percentiles. A
# Field II frame perturbed over that range yields labels on the same scale as a console frame's,
# which is what pretraining on one and fine-tuning on the other requires.
#
# The slider perturbation is zero-mean by construction. An offset would only be a gain change
# wearing a different hat - backend_solver's zero-mean constraint hands the overall level to
# gain - so what is left is the shape, and the near/mid/far triple can report two things about
# a shape: whether it leans, and whether it bows. A tilt alone leaves the mid group at exactly
# zero, because mid straddles the centre the tilt pivots about, and the first version of this
# drew only a tilt: 100% of Field II frames came back with slider_directions[mid] == 'correct'.
# So an arch is drawn alongside it.
#
# The ranges are set to cover what the console's own frames need, not to match it: console
# slider deltas reach 128 levels because its optimum often rails at an end while the operator
# left the sliders near neutral. Field II should span at least that, so the pretrained model
# sees the whole range it will later be asked about.
FIELDII_GAIN_DELTA_CLICKS = (-20.0, 25.0)
FIELDII_SLIDER_TILT_LEVELS = (-110.0, 110.0)
FIELDII_SLIDER_ARCH_LEVELS = (-70.0, 70.0)


def slider_shape_basis(num_bands=NUM_TGC_BANDS):
    """The tilt and arch a slider perturbation is built from, both zero-mean over the bands."""
    positions = (np.arange(num_bands) - 0.5 * (num_bands - 1)) / (0.5 * (num_bands - 1))
    tilt = positions
    arch = positions ** 2
    arch = arch - arch.mean()
    return tilt, arch / np.abs(arch).max()


def draw_start(rng, optimal_gain_db, optimal_tgc_levels,
               image_mode=DEFAULT_IMAGE_MODE,
               gain_delta_clicks=FIELDII_GAIN_DELTA_CLICKS,
               slider_tilt_levels=FIELDII_SLIDER_TILT_LEVELS,
               slider_arch_levels=FIELDII_SLIDER_ARCH_LEVELS,
               num_bands=NUM_TGC_BANDS):
    """A synthetic starting point for a frame that has no operator behind it.

    Returns (gain_db, tgc_levels). The delta is drawn first and the start placed at optimum
    minus delta, so the label distribution is the one that was chosen rather than whatever a
    distribution over starting points happens to induce. Clipping the start to the slider range
    can shorten a delta on frames whose optimum already sits near an end, which is not a defect
    - a slider that is already at 255 cannot be started above it either.
    """
    tilt_basis, arch_basis = slider_shape_basis(num_bands)
    delta_gain_clicks = rng.uniform(*gain_delta_clicks)
    delta_levels = (rng.uniform(*slider_tilt_levels) * tilt_basis
                    + rng.uniform(*slider_arch_levels) * arch_basis)

    start_gain_db = float(optimal_gain_db) - delta_gain_clicks * gain_db_per_level(image_mode)
    start_levels = np.clip(np.asarray(optimal_tgc_levels, dtype=np.float64) - delta_levels,
                           TGC_MIN_LEVEL, TGC_MAX_LEVEL)
    return start_gain_db, start_levels


def _direction(delta, deadband, names):
    """Form B on one axis: below, at, or above the optimum, with the band in the middle."""
    if abs(float(delta)) <= float(deadband):
        return names[1]
    return names[0] if float(delta) > 0 else names[2]


def gain_deadband_levels(sweep, gain_db, window_db, reference_db, tolerance,
                         limit_db=8.0, step_db=0.05, image_mode=DEFAULT_IMAGE_MODE):
    """How far gain can move before the objective moves by more than the tolerance.

    Expressed in console clicks, because that is the unit the label is emitted in and the unit
    an operator would act in. Derived per frame rather than fixed: the objective's slope in
    gain depends on how much of the image is near a clip, which varies a lot across the grid.
    """
    base = sweep.evaluate(gain_db, window_db, reference_db)
    for offset in np.arange(step_db, float(limit_db), step_db):
        moved = max(abs(sweep.evaluate(gain_db + offset, window_db, reference_db) - base),
                    abs(sweep.evaluate(gain_db - offset, window_db, reference_db) - base))
        if moved >= float(tolerance):
            return float(offset / gain_db_per_level(image_mode))
    return float(limit_db / gain_db_per_level(image_mode))


def label_frame(db_image, valid_mask, dr_ui, reference_db, current, target_gray,
                source, frame_id, group_id, imaging_mode, depth_mm,
                frequency_mhz=None, focus_mm=None, split=None,
                label_uncertainty=0.0, void_mask=None, calibration_borrowed=False,
                notes=None, rng=None, image_mode=DEFAULT_IMAGE_MODE, **solver_kwargs):
    """Solve one frame and express the answer as a label.

    current is the setting the frame sits at. Pass None for a frame that has no operator behind
    it - a starting point is then drawn from draw_start() using rng, after the optimum is known,
    so the delta lands in the range the console's own frames turned out to need. Passing None
    without an rng is an error rather than a silent default, because an unseeded label set
    cannot be reproduced.

    label_uncertainty widens the equivalence band by the simulator's own residual, so a console
    frame's "correct" covers everything the rebuild cannot tell apart from the optimum. On a
    Field II frame it is zero and the band narrows to the objective's response to the 0.22 dB
    that repeat captures of one scene already disagree by.
    """
    drawn = current is None
    if drawn and rng is None:
        raise ValueError("current=None needs an rng so the drawn start is reproducible")
    solver_current = current if current is not None else (
        0.0, np.full(NUM_TGC_BANDS, TGC_CENTER_LEVEL, dtype=np.float64), float(dr_ui))

    result = BS.solve_backend(
        db_image, valid_mask, dr_ui=dr_ui, reference_db=reference_db, current=solver_current,
        void_mask=void_mask, target_gray=target_gray, image_mode=image_mode,
        j_uncertainty=float(label_uncertainty), **solver_kwargs)
    slope = gain_db_per_level(image_mode)

    if drawn:
        start_gain_db, start_levels = draw_start(rng, result["gain_db"],
                                                 result["tgc_levels"], image_mode)
        # Round the start to settable levels before taking the difference. The label records
        # the setting a machine would actually be at, so a consumer recomputing the delta from
        # the recorded fields must land on the recorded delta rather than up to a level away.
        start_levels = np.round(start_levels)
        current = (start_gain_db, start_levels, float(dr_ui))
        # The optimum is a property of the scene and does not move with the starting point, so
        # only the deltas are restated here.
        result["delta_gain_levels"] = gain_db_to_levels(
            result["gain_db"] - start_gain_db, image_mode)
        result["delta_tgc_levels"] = (result["tgc_levels"].astype(np.float64)
                                      - start_levels)

    from hisense_backend_sim import dr_ui_to_window_db
    sweep = BS.GainSweep(db_image, result["tgc_levels"], valid_mask,
                         void_mask if void_mask is not None else ~np.asarray(valid_mask, bool),
                         target_gray=target_gray)
    deadband = gain_deadband_levels(sweep, result["gain_db"], dr_ui_to_window_db(dr_ui),
                                    reference_db, result["tolerance"], image_mode=image_mode)

    delta_tgc = np.asarray(result["delta_tgc_levels"], dtype=np.float64)
    slider_deadband = deadband * slope / tgc_db_per_level(image_mode)
    slider_directions = {
        name: _direction(delta_tgc[lo:hi].mean(), slider_deadband, SLIDER_DIRECTIONS)
        for name, (lo, hi) in SLIDER_GROUPS.items()
    }

    return FrameLabel(
        source=source, frame_id=frame_id, group_id=group_id, split=split,
        depth_mm=float(depth_mm), frequency_mhz=frequency_mhz, focus_mm=focus_mm,
        imaging_mode=imaging_mode,
        gain_db=float(current[0]),
        tgc_levels=[int(v) for v in np.asarray(current[1])],
        dr_ui=float(current[2]),
        optimal_gain_db=result["gain_db"],
        optimal_tgc_levels=[int(v) for v in result["tgc_levels"]],
        optimal_dr_ui=result["dr_ui"],
        gain_direction=_direction(result["delta_gain_levels"], deadband, GAIN_DIRECTIONS),
        slider_directions=slider_directions,
        dr_direction=DYNAMIC_RANGE_DIRECTIONS[1],
        delta_gain_levels=float(result["delta_gain_levels"]),
        gain_db_per_level=float(slope),
        delta_tgc_levels=[float(v) for v in delta_tgc],
        delta_dr_ui=0.0,
        objective=result["objective"],
        tolerance=result["tolerance"],
        label_uncertainty=float(label_uncertainty),
        deadband_gain_levels=float(deadband),
        equivalent_count=result["equivalent_count"],
        dr_determined=False,
        at_gain_edge=result["at_gain_edge"],
        calibration_borrowed=bool(calibration_borrowed),
        notes=list(notes or []),
    )
