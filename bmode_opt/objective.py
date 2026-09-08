"""Objective for choosing back-end settings, and the scope rules that go with it.

theta_optimal in the synthesis plan is an argmin over gain, TGC and dynamic range. All three
are display operations: they move levels around, they do not change what the beamformer
resolved. So resolution has no place in this objective - it would be constant across the
entire search and only add noise.

The three terms here are exactly the things gain, TGC and dynamic range do control:

    exposure       is anything that carries signal actually visible, and is anything saturated
    uniformity     does tissue sit at a similar level across depth
    utilisation    how much of the 0..255 range the image occupies

    Why exposure is not measured as "fraction of black pixels"

The acquisition protocol's exposure rule counts pixels at gray <= 2. That rule misfires on
any frame with a real anechoic structure: a cyst is supposed to be black, and a correctly
exposed image of one gets penalised for it. On the Field II cyst phantoms the anechoic lesion
alone covers about 5% of the frame, and rendering at the tissue's own level already puts the
crushed fraction at 31.5%, over the protocol's 30% threshold, with nothing wrong.

So the crushed fraction here is taken only over pixels that carry signal in the first place,
decided in the dB domain against the frame's own noise floor. A pixel with echo that renders
black is an exposure failure; a pixel with no echo that renders black is correct.

    What this objective deliberately leaves out

Speckle SNR. Stage 2.1 measured it across both data sources and it does not survive as an
absolute quality term: fully developed speckle sits at 1.913, the Field II phantoms read
1.59-1.82 because they are under-seeded, and the console reads 3.47-3.90 because it compounds.
Neither deviation means the image is worse. Used absolutely it would mark the highest
frequency - the best resolution available - as the worst image. It is a useful diagnostic
against a per-configuration baseline, not a term in a cost.

Cyst CNR. It does respond to dynamic range, but it needs a lesion mask, which only the
simulation has. Stage 2.2 also showed the value moves by more than a factor of two with the
region convention alone, so it belongs in evaluation, where the convention can be stated,
rather than inside a search.

    Search range versus perturbation range

These are different things, and conflating them pins the search against a boundary. E3's
-3..+5 dB gain limit is how far a *perturbation* may stray from a console frame's own
acquisition point, because that is the span over which the back-end model was checked
against real screenshots. The search for theta_optimal is not bounded by it, and on Field
II frames the limit does not apply at all: nothing is being predicted there, the render is
the ground truth by construction.

Searched without an artificial bound, this objective puts tissue at gray 73-127 across the
phantom types and settings tried, which is +4 to +14 dB above where a frame sits when its
tissue median is placed on GRAY_PIVOT. Clamped to +5 dB instead, nearly every frame lands
on the edge, and a label that always reads "turn it up as far as allowed" carries nothing.
"""

import numpy as np

from hisense_loader import band_edges, NUM_TGC_BANDS

GRAY_MAX = 255.0
# A console screenshot never quite reaches 0, so "crushed" allows a little headroom; the
# saturated end is taken tight because real saturation is unambiguous.
CRUSHED_GRAY = 2
SATURATED_GRAY = 253

# How far above the noise floor a pixel has to sit before its being black counts against the
# exposure. Three dB is the same margin the protocol uses when it reports penetration depth.
DEFAULT_SIGNAL_MARGIN_DB = 3.0

# Deepest fraction of the image used to estimate the noise floor. The console's own penetration
# measurements put the floor around 15.8 dB with the deepest band well past it, so the bottom
# tenth is safely below signal for every configuration in the grid.
DEFAULT_NOISE_ROWS_FRACTION = 0.10


def noise_floor_db(db_image, rows_fraction=DEFAULT_NOISE_ROWS_FRACTION):
    """Estimate the frame's noise floor from its deepest rows.

    Per-frame rather than a shipped constant: the floor moves with frequency and with which
    data source the frame came from, and it is cheap to measure.
    """
    db_image = np.asarray(db_image, dtype=np.float64)
    start = int(db_image.shape[0] * (1.0 - rows_fraction))
    return float(np.median(db_image[start:]))


def signal_mask(db_image, floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB):
    """Pixels carrying echo rather than noise. Anechoic structure is excluded by construction."""
    db_image = np.asarray(db_image, dtype=np.float64)
    if floor_db is None:
        floor_db = noise_floor_db(db_image)
    return db_image > floor_db + margin_db


def _resolve_mask(db_image, valid_mask, floor_db, margin_db):
    """The tissue mask to score against: the caller's if given, else the estimated one.

    Passing valid_mask is the supported path; see tissue.py, which derives it from the Field II
    truth mask or from a console noise floor measured per session and imaging mode. The
    fallback exists so older callers keep working, and it is wrong on both data sources - it
    discards 18.9% of a Field II frame as noise when the simulation has none, and it sits five
    to sixteen dB high on console frames shallower than the penetration limit.
    """
    if valid_mask is not None:
        return np.asarray(valid_mask, dtype=bool)
    return signal_mask(db_image, floor_db, margin_db)


def exposure_cost(gray_image, db_image, floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB,
                  valid_mask=None):
    """(crushed, saturated) fractions, with crushed counted only where there is signal.

    Both are fractions in 0..1 and both are costs, so either one rising is a worse image.
    """
    gray = np.asarray(gray_image)
    has_signal = _resolve_mask(db_image, valid_mask, floor_db, margin_db)
    total = int(has_signal.sum())
    crushed = float((gray[has_signal] <= CRUSHED_GRAY).mean()) if total else 0.0
    saturated = float((gray >= SATURATED_GRAY).mean())
    return crushed, saturated


def depth_uniformity_cost(gray_image, db_image, num_bands=NUM_TGC_BANDS,
                          floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB,
                          valid_mask=None):
    """Spread of the per-band tissue level, in gray levels, normalised to the full range.

    Bands are summarised over signal-carrying pixels only, for the same reason exposure is:
    a band that happens to contain a large cyst should not read as under-amplified. Bands with
    too little signal left are dropped rather than counted as dark.
    """
    gray = np.asarray(gray_image, dtype=np.float64)
    has_signal = _resolve_mask(db_image, valid_mask, floor_db, margin_db)
    edges = band_edges(gray.shape[0], num_bands)
    levels = []
    for k in range(num_bands):
        band = has_signal[edges[k]:edges[k + 1]]
        if band.sum() < 50:
            continue
        levels.append(np.median(gray[edges[k]:edges[k + 1]][band]))
    if len(levels) < 2:
        return float("nan")
    return float(np.std(levels) / GRAY_MAX)


def utilisation_cost(gray_image, db_image, low_pct=1.0, high_pct=99.0,
                     floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB, valid_mask=None):
    """Fraction of the 0..255 range the tissue does *not* occupy.

    Pushes against the exposure term: widening the dynamic range removes clipping but squeezes
    everything toward mid gray, and this is the cost of that squeeze.
    """
    gray = np.asarray(gray_image, dtype=np.float64)
    has_signal = _resolve_mask(db_image, valid_mask, floor_db, margin_db)
    if has_signal.sum() < 50:
        return 1.0
    values = gray[has_signal]
    span = np.percentile(values, high_pct) - np.percentile(values, low_pct)
    return float(np.clip(1.0 - span / GRAY_MAX, 0.0, 1.0))


# ---------------------------------------------------------------------------------------
#   Terms added in step 3, replacing two of the four above
#
# Running the solver over the dynamic range ladder showed the two terms above that respond to
# it were both measuring the same thing - how compressed the dB-to-gray mapping is - in
# opposite directions, and neither was measuring image quality:
#
#   depth_uniformity_cost is a spread of gray levels, so widening the window improves it for
#   free by squeezing every band toward mid gray. It scored 0.1916 at dynamic range 30 and
#   0.0489 at 400 on a frame whose actual tissue trend never changed.
#
#   utilisation_cost is a gray percentile span taken over individual pixels, and individual
#   pixels of speckle span 45.4 dB on that frame against 29.8 dB for the tissue level itself.
#   Speckle fills whatever window it is given, so the term read 0.0039 at dynamic range 30 and
#   only ever rewarded a narrower window. Recomputing it on the tissue level rather than on
#   pixels does not help: it still falls monotonically, 0.3471 to 0.8294 over the same ladder.
#
# The replacements below put each term in the domain where it means something. Uniformity moves
# to dB, where it depends on the TGC sliders alone and the display window cannot touch it.
# Window fit compares the window width against the signal's own dB span, which has an interior
# optimum where the window just covers the signal, instead of a monotone preference.
# ---------------------------------------------------------------------------------------

# Divisor that turns the band spread in dB into a number around one. Ten dB is roughly the
# spread an uncompensated Field II phantom shows across its eight bands, so a frame with no
# depth correction at all scores about 1.0 and a well flattened one scores near 0.
DEFAULT_UNIFORMITY_SCALE_DB = 10.0

# Gray level above which a pixel with no true echo counts as brightened. Sitting at 40 puts it
# just under GRAY_PIVOT, so tissue rendered at its own level is unaffected while noise or cyst
# interior lifted into visible gray is charged for.
VOID_GRAY_LIMIT = 40.0


def band_levels_db(db_image, valid_mask, num_bands=NUM_TGC_BANDS, min_pixels=50):
    """Median dB of the tissue in each depth band, skipping bands with too little of it."""
    db_image = np.asarray(db_image, dtype=np.float64)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    edges = band_edges(db_image.shape[0], num_bands)
    levels = []
    for index in range(num_bands):
        band = valid_mask[edges[index]:edges[index + 1]]
        if band.sum() < int(min_pixels):
            continue
        levels.append(float(np.median(db_image[edges[index]:edges[index + 1]][band])))
    return np.asarray(levels, dtype=np.float64)


def depth_uniformity_db_cost(db_image, valid_mask, num_bands=NUM_TGC_BANDS,
                             scale_db=DEFAULT_UNIFORMITY_SCALE_DB):
    """Spread of the tissue level across depth, in dB.

    Taken on the pre-display dB image, so gain shifts every band equally and cancels, and the
    display window does not enter at all. What is left is exactly what the TGC sliders control,
    which is the point: this term should ask the sliders to flatten the tissue, not reward the
    window for compressing it.
    """
    levels = band_levels_db(db_image, valid_mask, num_bands)
    if levels.size < 2:
        return float("nan")
    return float(np.std(levels) / float(scale_db))


def signal_span_db(db_image, valid_mask, low_pct=1.0, high_pct=99.0):
    """The dB range the tissue actually occupies, from its own percentiles."""
    db_image = np.asarray(db_image, dtype=np.float64)
    values = db_image[np.asarray(valid_mask, dtype=bool)]
    if values.size < 50:
        return float("nan")
    return float(np.percentile(values, high_pct) - np.percentile(values, low_pct))


def window_fit_cost(span_db, window_db):
    """How much of the display window is spent on dB values that do not occur.

    Zero once the window is no wider than the signal, rising as it is opened past it. Below
    that point the clipping terms take over, so the two together have an optimum where the
    window just covers the signal rather than a preference for one end of the ladder.
    """
    if not np.isfinite(span_db) or window_db <= 0:
        return float("nan")
    return float(np.clip(1.0 - span_db / float(window_db), 0.0, 1.0))


def noise_brightening_cost(gray_image, void_mask, limit=VOID_GRAY_LIMIT):
    """How far pixels with no true echo have been lifted into visible gray.

    void_mask is where there is nothing to show: below the console's measured noise floor, or
    inside an anechoic structure. Amplifying those is the failure mode that a pure exposure
    objective would otherwise reward, since brightening them fills the histogram.
    """
    gray = np.asarray(gray_image, dtype=np.float64)
    void_mask = np.asarray(void_mask, dtype=bool)
    if void_mask.sum() == 0:
        return 0.0
    excess = np.clip(gray[void_mask] - float(limit), 0.0, None)
    return float(excess.mean() / (GRAY_MAX - float(limit)))


def gcnr(values_a, values_b, num_bins=256, value_range=(0.0, 255.0)):
    """Generalised contrast-to-noise ratio between two sets of gray values.

    One minus the overlap of the two histograms, so it runs 0 (indistinguishable) to 1
    (perfectly separable). Unlike a plain contrast-to-noise ratio it is invariant to any
    monotone transform of the gray scale, which is what makes it safe to compare across
    dynamic range settings - the setting being judged cannot inflate it by stretching
    contrast, only by actually destroying or preserving separability.
    """
    values_a = np.asarray(values_a, dtype=np.float64).ravel()
    values_b = np.asarray(values_b, dtype=np.float64).ravel()
    if values_a.size < 20 or values_b.size < 20:
        return float("nan")
    edges = np.linspace(value_range[0], value_range[1], int(num_bins) + 1)
    hist_a, _ = np.histogram(values_a, bins=edges)
    hist_b, _ = np.histogram(values_b, bins=edges)
    density_a = hist_a / float(values_a.size)
    density_b = hist_b / float(values_b.size)
    return float(1.0 - np.minimum(density_a, density_b).sum())


def visibility_cost(gray_image, lesion_mask, background_mask, num_bins=256):
    """One minus the gCNR between a lesion and its same-depth background, on the render.

    Returns NaN when the frame has no lesion to measure, which is the honest answer for a
    uniform phantom rather than a zero that would read as a perfect image.
    """
    gray = np.asarray(gray_image, dtype=np.float64)
    lesion_mask = np.asarray(lesion_mask, dtype=bool)
    background_mask = np.asarray(background_mask, dtype=bool)
    value = gcnr(gray[lesion_mask], gray[background_mask], num_bins)
    if not np.isfinite(value):
        return float("nan")
    return float(1.0 - value)


# Weights. Three failures destroy information rather than merely presenting it poorly - a
# crushed pixel, a saturated pixel, and a void lifted into visible gray, which puts something
# on the screen that is not in the patient - so those carry the larger weight. Uniformity is a
# presentation preference and carries one.
DEFAULT_WEIGHTS = {
    "crushed": 3.0,
    "saturated": 3.0,
    "uniformity": 1.0,
    "noise_brightening": 3.0,
    "brightness": 1.0,
}


def backend_objective(gray_image, shaped_db_image, valid_mask, void_mask=None, weights=None,
                      num_bands=NUM_TGC_BANDS, uniformity_scale_db=DEFAULT_UNIFORMITY_SCALE_DB,
                      target_gray=None, return_terms=False):
    """Cost of one rendered image, for searching over gain and TGC. Lower is better.

    shaped_db_image is the dB image the render came from, after the TGC curve has been added -
    the uniformity term has to see the sliders it is judging. Gain may or may not be included;
    it shifts every band equally and cancels in the spread.

    valid_mask says which pixels hold tissue and is required, not estimated; build it with
    tissue.fieldii_tissue_mask() or tissue.console_tissue_mask(), from the image *before* the
    sliders are applied so that the mask stays fixed across a search. void_mask says which
    pixels hold nothing - below the console's measured noise floor, or inside an anechoic
    structure - and defaults to the complement of valid_mask.

    Note what is *not* here. The utilisation term of the earlier formulation is gone: gray span
    equals signal dB span over window width, so that term was algebraically identical to
    comparing the window against the signal, and on console frames the signal spans only 19 to
    31 dB against a narrowest settable window of 45.6 dB, making it a constant vote for the
    minimum regardless of the image. Uniformity moved from gray to dB for the matching reason -
    in gray it improved for free whenever the window was widened.

    That leaves nothing in this objective that responds to dynamic range except clipping, which
    is deliberate: see the module note on why dynamic range is not determined by the data now
    in hand. Search gain and TGC here and hold dynamic range at the acquired value.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if void_mask is None:
        void_mask = ~valid_mask

    crushed, saturated = exposure_cost(gray_image, shaped_db_image, valid_mask=valid_mask)
    uniformity = depth_uniformity_db_cost(shaped_db_image, valid_mask, num_bands,
                                          uniformity_scale_db)
    brightening = noise_brightening_cost(gray_image, void_mask)
    brightness = brightness_cost(gray_image, valid_mask, target_gray)

    terms = {
        "crushed": crushed,
        "saturated": saturated,
        "uniformity": 0.0 if not np.isfinite(uniformity) else uniformity,
        "noise_brightening": brightening,
        "brightness": 0.0 if not np.isfinite(brightness) else brightness,
    }
    total = float(sum(weights[key] * value for key, value in terms.items()))
    if return_terms:
        terms["total"] = total
        return total, terms
    return total


def brightness_cost(gray_image, valid_mask, target_gray,
                    scale_gray=25.0):
    """How far the tissue level sits from where the operator puts it.

    Without this the objective has no opinion about overall brightness at all: gain is bounded
    only by crushing below and saturation above, and everything between is flat, so the answer
    is decided by whatever the tie-break happens to prefer. The utilisation term used to supply
    an accidental anchor; removing it made the gap visible.

    target_gray must be measured per session and imaging mode - see
    tissue.measure_accepted_brightness(). Passing None leaves the term out and leaves gain
    under-determined, which is the correct state before that measurement exists.
    """
    if target_gray is None:
        return float("nan")
    gray = np.asarray(gray_image, dtype=np.float64)
    values = gray[np.asarray(valid_mask, dtype=bool)]
    if values.size < 50:
        return float("nan")
    return float(abs(np.median(values) - float(target_gray)) / float(scale_gray))
