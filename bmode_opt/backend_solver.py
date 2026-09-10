"""Solving for the back end's three knobs, with the gain/TGC degeneracy removed.

    The problem this module exists to fix

Gain and TGC are not independent actuators as far as the rendered image is concerned. Gain
adds a constant number of dB everywhere. The eight TGC sliders are interpolated onto the
depth axis with weights that sum to one at every row, so moving all eight by the same amount
*also* adds a constant number of dB everywhere. Shifting the sliders by d levels and the gain
by -d * DEFAULT_DB_PER_LEVEL dB therefore produces a bit-identical uint8 image.

Measured on a Field II uniform phantom at 42 mm / 5 MHz / 15 mm focus, sliders moved over an
80-level span with gain compensating:

    slider shift   TGC dB    compensating gain   max gray difference   objective
        -40        -2.624         +2.624                  0             0.8375
        -20        -1.312         +1.312                  0             0.8375
          0         0.000          0.000                  0             0.8375
        +20        +1.312         -1.312                  0             0.8375
        +40        +2.624         -2.624                  0             0.8375

So an unconstrained argmin over (gain, TGC, dynamic range) does not have a solution: it has a
one-dimensional ridge of solutions, every point of which renders the same image and scores the
same objective. Which point a search stops on is decided by iteration order. Labels generated
that way disagree with each other while describing identical images, which is worse than
useless for training.

    The fix

Split the ridge by convention, the way the acquisition protocol's own reasoning already
implies: the sliders own the *shape* of the depth correction and gain owns the *level*. That
is imposed here as a constraint on the tissue-weighted mean of the TGC curve,

    sum_z w(z) * tgc_curve_db(z) / sum_z w(z) == 0,

with w(z) the confidence that row z holds usable tissue. Any action can be moved onto that
constraint surface without changing its rendering, by canonical_action(); the search then runs
entirely inside the surface, so it never spends effort exploring a direction that does nothing.

Two further rules remove what is left:

  * A second-difference penalty on the eight slider values. Without it many slider vectors
    interpolate to nearly the same depth curve and the fit is ill-conditioned in the
    high-frequency directions; the penalty also keeps the answer physically settable rather
    than a saw-tooth.
  * A fixed rule for exact numerical ties - least gain, then least slider travel, then least
    dynamic range - rather than whichever the loop happened to reach first.

The tie-break deliberately does *not* look at the setting the frame was acquired at. It is
tempting, because it would make every label a small correction, but it would also make the
answer a property of where the operator happened to leave the knobs rather than a property of
the scene, and then the delta a model is asked to predict would be partly noise. The optimum
is computed once per frame and the delta is formed against the current setting afterwards.

What *is* reported alongside it is the equivalent set: every action scoring within a
per-frame tolerance of the best. Down that road a model should not be punished for choosing a
different member of it, and a controller sitting inside it should not move.

    Why dynamic range is held rather than searched

It is not determined by the data now in hand, and every criterion that could determine it was
tried and measured:

  * Fitting the window to the signal. Gray span is signal dB span over window width, so the
    old utilisation term was already this comparison written differently - the two agree to
    0.003 across the whole ladder. On console frames the tissue spans only 19 to 21 dB in
    harmonic mode and 27 to 31 dB in fundamental mode, against a narrowest settable window of
    45.6 dB, so the signal never fills even the narrowest window and the term is a constant
    vote for the minimum whatever the image looks like.

  * Widening the window to reach the brightest structure. From the measured noise floor up to
    the 99.9th percentile of the frame spans 26.5 to 28.2 dB in harmonic mode and 36.8 to 39.6
    in fundamental mode. Still below 45.6, so this criterion also says minimum. Meanwhile the
    frames were actually acquired at dynamic range 67, a 58.7 dB window - nearly twice the
    signal - on 188 of 207 captures.

  * Lesion separability. gCNR between a cyst and its same-depth background moves from 0.6996
    to 0.6979 across the entire ladder from 30 to 400. That flatness is not a defect; gCNR is
    invariant to monotone transforms by construction, which is exactly why it is safe to
    compare across settings, and exactly why it cannot choose between them.

The reason none of them bite is that dynamic range governs the discrimination of structures a
few dB apart, and there are none in the data. The Field II generator sets a cyst's scatterer
amplitudes to zero, so the two measurable lesions sit at 26.1 and 39.5 dB of contrast; an
anechoic void is a black hole against gray tissue at any window width.

So dynamic range is reported back unchanged with dr_determined False. Determining it needs
low-contrast lesions - a few dB, not twenty - which is a phantom to simulate or acquire, not
a weight to tune.

    What still depends on the per-session calibration

target_gray comes from tissue.measure_accepted_brightness() and has to be measured per session
and imaging mode. Within a mode it reproduces well across sessions - fundamental mode reads 92, 90
and 89 gray on three sessions, harmonic 34, 34, 36 and 36 on four - but the two modes differ by
a factor of 2.6, and that gap cannot yet be attributed. DEFAULT_PIVOT_DB and
DEFAULT_COUNTS_PER_DB were calibrated on one session in one mode, and counts_per_db has been
seen to differ by 20% between sessions, which moves rendered gray without anyone touching a
knob. Until that is settled, brightness targets are usable within a group and not across one.
"""

import numpy as np

from hisense_backend_sim import (
    CALIBRATION_GAIN_LEVEL,
    DEFAULT_DB_PER_LEVEL,
    gain_db_to_levels,
    DEFAULT_PIVOT_DB,
    GAIN_DB_PER_LEVEL,
    GRAY_MAX,
    GRAY_PIVOT,
    TGC_CENTER_LEVEL,
    TGC_MAX_LEVEL,
    TGC_MIN_LEVEL,
    apply_tgc,
    dr_ui_to_window_db,
    render,
)
from hisense_loader import NUM_TGC_BANDS, band_centres, band_edges
import objective as OBJ


# Dynamic range values to enumerate. The console accepts 30..400 in steps of three or four, so
# this is a coarse ladder over that span rather than the true settable set, which has to come
# off the console itself. Enumerated rather than optimised continuously because a continuous
# prediction is not a settable machine state.
DEFAULT_DR_UI_CANDIDATES = (30, 45, 67, 100, 150, 220, 300, 400)

# Gain offsets to enumerate, in dB relative to whatever reference_db the caller anchors on.
# Stage 2.4 found the objective's optimum sitting at +4..+15 dB with tissue landing on gray
# 73..127, so the grid has to reach well past the +5 dB that bounds *perturbations* of console
# frames; that limit is about predicting real screenshots and does not bound this search.
DEFAULT_GAIN_DB_GRID = np.arange(-8.0, 24.01, 0.5)

# Half-width of the adaptive gain search, in dB, either side of the gain that puts the tissue
# median on the brightness target. Sixteen dB is generous: over the whole archive the optimum
# sits within 9 dB of that centre.
DEFAULT_GAIN_SEARCH_HALF_WIDTH_DB = 16.0
DEFAULT_GAIN_STEP_DB = 0.5

# Weight on the second difference of the eight slider dB values, relative to the weighted
# least-squares fit of the depth trend. Chosen so a smooth attenuation curve is followed
# closely while a single-band spike is not; see the sensitivity report in the verification.
DEFAULT_SMOOTHNESS = 2.0

# Frame-to-frame repeatability of the reference capture S0 was at most 0.220 dB over four THI
# positions and 0.213 dB over two GEN positions. A difference smaller than that is smaller
# than re-acquiring the same frame twice, so it cannot be called an improvement.
DEFAULT_DEADBAND_DB = 0.22


def tgc_basis(num_points, num_bands=NUM_TGC_BANDS):
    """Rows of per-slider weights, so that tgc_curve_db = basis @ slider_db.

    Built by pushing unit vectors through the same linear interpolation
    expand_tgc_to_depth() uses, so it is that function, exactly, written as a matrix. Each row
    sums to one - which is the algebraic statement of the degeneracy this module removes.
    """
    num_points = int(num_points)
    centres = band_centres(num_points, num_bands)
    rows = np.arange(num_points, dtype=np.float64)
    basis = np.empty((num_points, num_bands), dtype=np.float64)
    for band in range(num_bands):
        unit = np.zeros(num_bands, dtype=np.float64)
        unit[band] = 1.0
        basis[:, band] = np.interp(rows, centres, unit)
    return basis


def row_tissue_levels(db_image, valid_mask, min_pixels=8):
    """Per-row median dB over valid pixels, and a per-row confidence weight.

    The median is taken across lines within a row, so a row crossing a cyst reports the level
    of the tissue beside it rather than an average dragged down by the void. The weight is the
    fraction of the row that is valid, which lets a row that is mostly cyst or mostly noise
    contribute proportionally less to the depth-trend fit instead of being in or out.
    """
    db_image = np.asarray(db_image, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    counts = valid_mask.sum(axis=1)
    masked = np.ma.masked_array(db_image, mask=~valid_mask)
    levels = np.ma.median(masked, axis=1).filled(np.nan)
    weights = counts / float(db_image.shape[1])
    thin = counts < int(min_pixels)
    levels[thin] = np.nan
    weights[thin] = 0.0
    return levels, weights


def tissue_weight_from_mask(valid_mask):
    """Per-row weight from a boolean validity mask, for callers holding their own mask."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    return valid_mask.sum(axis=1) / float(valid_mask.shape[1])


def _default_valid_mask(db_image, margin_db=OBJ.DEFAULT_SIGNAL_MARGIN_DB):
    """Fallback tissue mask when the caller supplies none.

    Only a fallback. Callers should pass valid_mask from tissue.fieldii_tissue_mask() or
    tissue.console_tissue_mask(); this estimator discards deep tissue on both data sources.
    """
    return OBJ.signal_mask(db_image, margin_db=margin_db)


def weighted_curve_mean(slider_db, weights, basis=None, num_points=None):
    """Tissue-weighted mean of the depth curve the sliders produce, in dB."""
    slider_db = np.asarray(slider_db, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if basis is None:
        basis = tgc_basis(num_points if num_points is not None else weights.size, slider_db.size)
    total = weights.sum()
    if total <= 0:
        return 0.0
    return float((weights * (basis @ slider_db)).sum() / total)


def canonical_action(gain_db, tgc_levels, weights, basis=None, db_per_level=DEFAULT_DB_PER_LEVEL):
    """Move an action onto the zero-mean-TGC surface without changing what it renders.

    Returns (gain_db, tgc_levels) in float. The shift moved out of the sliders is added to the
    gain, so the depth curve plus gain is unchanged at every row and the rendered image is
    identical - which is the whole point: this picks a representative of an equivalence class,
    it does not choose a different image.
    """
    tgc_levels = np.asarray(tgc_levels, dtype=np.float64).reshape(-1)
    slider_db = (tgc_levels - float(TGC_CENTER_LEVEL)) * float(db_per_level)
    mean_db = weighted_curve_mean(slider_db, weights, basis)
    return float(gain_db) + mean_db, tgc_levels - mean_db / float(db_per_level)


def solve_tgc_shape(
    db_image,
    valid_mask=None,
    tissue_weight=None,
    num_bands=NUM_TGC_BANDS,
    db_per_level=DEFAULT_DB_PER_LEVEL,
    smoothness=DEFAULT_SMOOTHNESS,
    level_bounds=(TGC_MIN_LEVEL, TGC_MAX_LEVEL),
):
    """Slider levels that flatten the tissue's depth trend, zero-mean and smooth.

    Solved analytically rather than searched. The objective's uniformity term is the spread of
    per-band gray medians, and gray is an affine function of dB wherever nothing clips, so
    flattening the trend in dB flattens it in gray as well; the only place the two part company
    is where the display window clips, which the joint search afterwards is left to handle.

    Returns (levels, info). levels is integer and inside the bounds. info["residual_gain_db"]
    is the mean the rounding and clipping pushed back out of the sliders, which the caller must
    add to gain if the canonical form is to survive - solve_backend() does that.
    """
    db_image = np.asarray(db_image, dtype=np.float64)
    num_points = db_image.shape[0]
    if tissue_weight is None:
        mask = _default_valid_mask(db_image) if valid_mask is None else np.asarray(valid_mask, bool)
        levels_db, weights = row_tissue_levels(db_image, mask)
    else:
        weights = np.asarray(tissue_weight, dtype=np.float64).reshape(-1)
        mask = _default_valid_mask(db_image) if valid_mask is None else np.asarray(valid_mask, bool)
        levels_db, _ = row_tissue_levels(db_image, mask)

    basis = tgc_basis(num_points, num_bands)
    usable = np.isfinite(levels_db) & (weights > 0)
    if usable.sum() < num_bands:
        # Not enough tissue to identify a depth trend. Returning neutral sliders is the honest
        # answer; the caller decides whether to keep the frame.
        neutral = np.full(num_bands, TGC_CENTER_LEVEL, dtype=np.int32)
        return neutral, {"residual_gain_db": 0.0, "usable_rows": int(usable.sum()),
                         "fit_rms_db": float("nan"), "solved": False}

    design = basis[usable]
    weight = weights[usable]
    trend = levels_db[usable]
    trend_mean = float((weight * trend).sum() / weight.sum())
    target = -(trend - trend_mean)

    second_difference = np.zeros((num_bands - 2, num_bands))
    for row in range(num_bands - 2):
        second_difference[row, row:row + 3] = (1.0, -2.0, 1.0)

    normal = design.T @ (weight[:, None] * design)
    normal += float(smoothness) * (second_difference.T @ second_difference)
    rhs = design.T @ (weight * target)
    slider_db = np.linalg.solve(normal, rhs)

    # The fit is already close to zero-mean because the target is; make it exact.
    slider_db -= weighted_curve_mean(slider_db, weights, basis)
    fit_rms = float(np.sqrt((weight * (design @ slider_db - target) ** 2).sum() / weight.sum()))

    low, high = level_bounds
    levels = np.clip(np.round(slider_db / float(db_per_level) + TGC_CENTER_LEVEL), low, high)
    realised_db = (levels - float(TGC_CENTER_LEVEL)) * float(db_per_level)
    residual = weighted_curve_mean(realised_db, weights, basis)
    return levels.astype(np.int32), {
        "residual_gain_db": float(residual),
        "usable_rows": int(usable.sum()),
        "fit_rms_db": fit_rms,
        "solved": True,
    }


class GainSweep:
    """Scores a whole gain sweep for one TGC shape and window without re-rendering.

    Rendering a frame takes about 19 ms and a useful sweep is hundreds of settings, so the
    direct route costs tens of seconds a frame and days over the dataset. It is also
    unnecessary: for a fixed TGC shape, gain only shifts the dB image by a constant and the
    window only rescales dB onto gray. Both leave the *order* of the pixels alone, and every
    term of the objective is a threshold count, a fixed constant, or a histogram of gray:

        crushed             tissue pixels below a dB threshold gain and window width place
        saturated           all pixels above another such threshold
        uniformity          the spread of band levels in dB, which neither gain nor the
                            window can touch, so it is computed once per shape
        noise brightening   the mean excess gray of void pixels, taken from their exact
                            per-gray-level histogram rather than from an affine approximation

    So sorting the dB values once turns each evaluation into a handful of binary searches.
    Verified against the renderer to the last bit; see tests/verify_backend_solver_fastpath.py.

    The masks come from the pre-display image and do not move during the sweep. That is
    deliberate: a mask that responded to the setting being searched would let the search
    improve its score by pushing awkward pixels out of the scored region.
    """

    def __init__(self, db_image, tgc_levels, valid_mask, void_mask=None,
                 num_bands=NUM_TGC_BANDS, db_per_level=DEFAULT_DB_PER_LEVEL, weights=None,
                 uniformity_scale_db=OBJ.DEFAULT_UNIFORMITY_SCALE_DB, target_gray=None,
                 brightness_scale_gray=25.0):
        shaped = apply_tgc(np.asarray(db_image, dtype=np.float64), tgc_levels, db_per_level)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if void_mask is None:
            void_mask = ~valid_mask
        void_mask = np.asarray(void_mask, dtype=bool)

        self.weights = OBJ.DEFAULT_WEIGHTS if weights is None else weights
        self.total_pixels = int(shaped.size)
        self.tissue_values = np.sort(shaped[valid_mask])
        self.all_values = np.sort(shaped.ravel())
        self.void_values = np.sort(shaped[void_mask])
        self.num_tissue = int(self.tissue_values.size)
        self.num_void = int(self.void_values.size)

        # Independent of gain and of the window, so it is a property of this shape alone.
        uniformity = OBJ.depth_uniformity_db_cost(shaped, valid_mask, num_bands,
                                                  uniformity_scale_db)
        self.uniformity = 0.0 if not np.isfinite(uniformity) else float(uniformity)

        # Excess gray charged to a void pixel at each of the 256 possible gray levels.
        self.void_excess = np.clip(np.arange(256.0) - OBJ.VOID_GRAY_LIMIT, 0.0, None) \
            / (GRAY_MAX - OBJ.VOID_GRAY_LIMIT)

        # The two order statistics numpy's median interpolates between, so the tissue level
        # can be mapped through the display rather than approximated.
        self.target_gray = target_gray
        self.brightness_scale_gray = float(brightness_scale_gray)
        if self.num_tissue:
            middle = self.num_tissue // 2
            self.tissue_median_pair = (
                (self.tissue_values[middle], self.tissue_values[middle])
                if self.num_tissue % 2
                else (self.tissue_values[middle - 1], self.tissue_values[middle]))
        else:
            self.tissue_median_pair = None

    @staticmethod
    def _db_of_gray(level, gain_db, window_db, reference_db):
        """The dB value that renders to a given (possibly fractional) gray level."""
        return float(reference_db) - float(gain_db) + (level - GRAY_PIVOT) * window_db / GRAY_MAX

    def centring_gain_db(self, target_gray, window_db, reference_db):
        """The gain that puts this shape's tissue median exactly on target_gray.

        The search is centred here rather than on a fixed span of absolute dB, because
        reference_db is a per-group pivot and the two imaging modes sit 20 dB apart in it -
        fundamental pivots land near 45 and harmonic near 30. A fixed window that suits one
        clips the other: with the grid pinned at -8..+24 dB, 53 of 280 console frames came back
        sitting exactly on the ceiling, every one of them fundamental.
        """
        if self.tissue_median_pair is None:
            return 0.0
        low, high = self.tissue_median_pair
        median_db = 0.5 * (low + high)
        return float((float(target_gray) - GRAY_PIVOT) * window_db / GRAY_MAX
                     + float(reference_db) - median_db)

    def evaluate(self, gain_db, window_db, reference_db, return_terms=False):
        """Objective for one gain, identical to rendering at that gain and scoring it."""
        window_db = max(1.0, float(window_db))
        edge = lambda level: self._db_of_gray(level, gain_db, window_db, reference_db)

        # gray <= 2 needs the pre-round value at or below 2.5; numpy rounds 2.5 down to 2.
        if self.num_tissue:
            below = int(np.searchsorted(self.tissue_values, edge(2.5), side="right"))
            crushed = below / float(self.num_tissue)
        else:
            crushed = 0.0
        # gray >= 253 needs it strictly above 252.5; numpy rounds 252.5 down to 252.
        above = self.total_pixels - int(
            np.searchsorted(self.all_values, edge(252.5), side="right"))
        saturated = above / float(self.total_pixels)

        if self.num_void:
            boundaries = np.array([edge(level + 0.5) for level in range(255)])
            cumulative = np.searchsorted(self.void_values, boundaries, side="right")
            counts = np.empty(256, dtype=np.float64)
            counts[0] = cumulative[0]
            counts[1:255] = np.diff(cumulative)
            counts[255] = self.num_void - cumulative[254]
            brightening = float((counts * self.void_excess).sum() / self.num_void)
        else:
            brightening = 0.0

        if self.target_gray is None or self.tissue_median_pair is None or self.num_tissue < 50:
            brightness = 0.0
        else:
            def to_gray(value):
                raw = GRAY_PIVOT + (value + gain_db - reference_db) / window_db * GRAY_MAX
                return float(np.clip(np.round(raw), 0.0, GRAY_MAX))
            low, high = self.tissue_median_pair
            level = 0.5 * (to_gray(low) + to_gray(high))
            brightness = abs(level - float(self.target_gray)) / self.brightness_scale_gray

        terms = {"crushed": crushed, "saturated": saturated,
                 "uniformity": self.uniformity, "noise_brightening": brightening,
                 "brightness": brightness}
        total = float(sum(self.weights[key] * value for key, value in terms.items()))
        if return_terms:
            terms["total"] = total
            return total, terms
        return total


def action_distance(gain_db, tgc_levels, dr_ui, reference,
                    db_per_level=DEFAULT_DB_PER_LEVEL, dr_step=3.5,
                    at_level=CALIBRATION_GAIN_LEVEL):
    """How far an action is from a reference one, counted in console clicks.

    Gain is converted at its own dB per level, the sliders at theirs, and dynamic range at the
    three-to-four-unit step the console moves in, so the three terms are commensurate: each is
    a number of detents the operator would have to turn.
    """
    gain_ref, tgc_ref, dr_ref = reference
    gain_clicks = abs(gain_db_to_levels(float(gain_db) - float(gain_ref), at_level))
    tgc_clicks = float(np.mean(np.abs(np.asarray(tgc_levels, dtype=np.float64)
                                      - np.asarray(tgc_ref, dtype=np.float64))))
    dr_clicks = abs(float(dr_ui) - float(dr_ref)) / float(dr_step)
    return gain_clicks + tgc_clicks + dr_clicks


# How much of the solved depth correction to try. The direction of the correction comes from
# the measured tissue trend; how much of it is worth applying is a trade the objective has to
# make, because the sliders span only +-8.33 dB (127 levels either side of centre at 0.06559 dB
# per level) while a Field II phantom's tissue falls about 25 dB from 16 to 42 mm. Asked to
# flatten that completely the fit rails at both ends. Nought is a flat slider set, so the flat
# case is inside this grid.
DEFAULT_SHAPE_SCALES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def solve_backend(
    db_image,
    valid_mask,
    dr_ui,
    reference_db=DEFAULT_PIVOT_DB,
    current=None,
    void_mask=None,
    tissue_weight=None,
    gain_db_grid=None,
    shape_scales=DEFAULT_SHAPE_SCALES,
    num_bands=NUM_TGC_BANDS,
    db_per_level=DEFAULT_DB_PER_LEVEL,
    smoothness=DEFAULT_SMOOTHNESS,
    deadband_db=DEFAULT_DEADBAND_DB,
    j_uncertainty=0.0,
    weights=None,
    graymap_lut=None,
    target_gray=None,
    gain_level=None,
    fast=True,
):
    """Best gain and TGC for one pre-display dB image, as a unique canonical action.

    dr_ui is held, not searched, and is reported back unchanged so the delta for dynamic range
    is zero by construction. See the module note: nothing in the data now in hand determines
    it, and a label that always reads "set it to the minimum" is not information.

    valid_mask must be supplied - build it with tissue.fieldii_tissue_mask() or
    tissue.console_tissue_mask(). void_mask defaults to its complement.

    The answer depends only on the image, not on where the knobs currently sit. current is used
    solely to express the answer as a delta.
    """
    db_image = np.asarray(db_image, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if void_mask is None:
        void_mask = ~valid_mask
    void_mask = np.asarray(void_mask, dtype=bool)
    if tissue_weight is None:
        tissue_weight = tissue_weight_from_mask(valid_mask)
    tissue_weight = np.asarray(tissue_weight, dtype=np.float64).reshape(-1)
    basis = tgc_basis(db_image.shape[0], num_bands)
    window_db = dr_ui_to_window_db(dr_ui)

    if current is None:
        current = (0.0, np.full(num_bands, TGC_CENTER_LEVEL, dtype=np.float64), float(dr_ui))

    full_levels, shape_info = solve_tgc_shape(
        db_image, valid_mask=valid_mask, tissue_weight=tissue_weight, num_bands=num_bands,
        db_per_level=db_per_level, smoothness=smoothness,
    )
    full_db = (np.asarray(full_levels, dtype=np.float64) - TGC_CENTER_LEVEL) * db_per_level

    candidates = []
    for scale in shape_scales:
        levels = np.clip(np.round(scale * full_db / db_per_level + TGC_CENTER_LEVEL),
                         TGC_MIN_LEVEL, TGC_MAX_LEVEL)
        # Rounding and clipping put a little level back into the sliders; take it out of the
        # gain so the action stays exactly on the zero-mean surface.
        realised_db = (levels - float(TGC_CENTER_LEVEL)) * db_per_level
        offset = -weighted_curve_mean(realised_db, tissue_weight, basis)
        candidates.append((float(scale), levels, offset))

    # The fast path assumes the display map is the affine window alone. A graymap lookup table
    # is monotone too, so the argument still holds, but the thresholds would have to be
    # inverted through it; until that is needed, fall back to rendering.
    fast = bool(fast) and graymap_lut is None
    sweeps = {
        scale: (GainSweep(db_image, levels, valid_mask, void_mask, num_bands, db_per_level,
                          weights, target_gray=target_gray) if fast else None)
        for scale, levels, _ in candidates
    }

    def score(scale, levels, gain_db):
        if fast:
            return sweeps[scale].evaluate(gain_db, window_db, reference_db)
        gray = render(
            db_image=db_image, tgc_levels=levels, gain_db=gain_db,
            dynamic_range_db=window_db, reference_db=reference_db,
            depth_response_db=None, db_per_level=db_per_level, graymap_lut=graymap_lut,
        )
        shaped = apply_tgc(db_image, levels, db_per_level)
        return OBJ.backend_objective(gray, shaped, valid_mask, void_mask, weights=weights,
                                     num_bands=num_bands, target_gray=target_gray)

    if gain_db_grid is None:
        # Centre the search where the tissue median lands on the target, so the span does not
        # have to cover the gap between the two modes' pivot conventions.
        if target_gray is not None and fast:
            centres = [sweeps[scale].centring_gain_db(target_gray, window_db, reference_db)
                       for scale, _, _ in candidates]
            centre = float(np.median(centres))
        else:
            centre = 8.0
        half = DEFAULT_GAIN_SEARCH_HALF_WIDTH_DB
        gain_db_grid = np.arange(centre - half, centre + half + 1e-9, DEFAULT_GAIN_STEP_DB)

    evaluations = []
    for scale, levels, gain_offset in candidates:
        for gain_db in np.asarray(gain_db_grid, dtype=np.float64):
            total = float(gain_db) + gain_offset
            evaluations.append({
                "scale": scale,
                "tgc_levels": levels,
                "gain_db": total,
                "objective": score(scale, levels, total),
            })

    def ordering(item):
        return (
            item["objective"],
            abs(item["gain_db"]),
            float(np.abs(item["tgc_levels"] - TGC_CENTER_LEVEL).sum()),
        )

    best = min(evaluations, key=ordering)

    nudged = [score(best["scale"], best["tgc_levels"], best["gain_db"] + sign * deadband_db)
              for sign in (-1.0, 1.0)]
    # Two things can make a setting indistinguishable from the best one: the scene itself,
    # which repeat captures already disagree about by the deadband, and the simulator, whose
    # own residual is measured per group and passed in. Both belong in the band.
    tolerance = max(abs(value - best["objective"]) for value in nudged) + float(j_uncertainty)
    equivalent = [item for item in evaluations
                  if item["objective"] <= best["objective"] + tolerance]

    gain_db, tgc_levels = canonical_action(
        best["gain_db"], best["tgc_levels"], tissue_weight, basis, db_per_level
    )
    levels = np.clip(np.round(tgc_levels), TGC_MIN_LEVEL, TGC_MAX_LEVEL).astype(np.int32)

    at_gain_edge = (best["gain_db"] <= float(gain_db_grid[0]) + 1e-9
                    or best["gain_db"] >= float(gain_db_grid[-1]) - 1e-9)

    return {
        "gain_db": float(gain_db),
        "tgc_levels": levels,
        "dr_ui": float(dr_ui),
        "dr_determined": False,
        "brightness_anchored": target_gray is not None,
        "shape_scale": best["scale"],
        "objective": float(best["objective"]),
        "tolerance": float(tolerance),
        "equivalent_count": len(equivalent),
        "equivalent_gain_db": (min(i["gain_db"] for i in equivalent),
                               max(i["gain_db"] for i in equivalent)),
        "equivalent_scales": sorted({i["scale"] for i in equivalent}),
        "at_gain_edge": bool(at_gain_edge),
        "shape_fit_rms_db": shape_info["fit_rms_db"],
        "shape_usable_rows": shape_info["usable_rows"],
        "delta_gain_levels": gain_db_to_levels(
            float(gain_db) - float(current[0]),
            CALIBRATION_GAIN_LEVEL if gain_level is None else gain_level),
        "delta_tgc_levels": levels.astype(np.float64) - np.asarray(current[1], dtype=np.float64),
        "delta_dr_ui": 0.0,
    }
