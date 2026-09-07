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


def exposure_cost(gray_image, db_image, floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB):
    """(crushed, saturated) fractions, with crushed counted only where there is signal.

    Both are fractions in 0..1 and both are costs, so either one rising is a worse image.
    """
    gray = np.asarray(gray_image)
    has_signal = signal_mask(db_image, floor_db, margin_db)
    total = int(has_signal.sum())
    crushed = float((gray[has_signal] <= CRUSHED_GRAY).mean()) if total else 0.0
    saturated = float((gray >= SATURATED_GRAY).mean())
    return crushed, saturated


def depth_uniformity_cost(gray_image, db_image, num_bands=NUM_TGC_BANDS,
                          floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB):
    """Spread of the per-band tissue level, in gray levels, normalised to the full range.

    Bands are summarised over signal-carrying pixels only, for the same reason exposure is:
    a band that happens to contain a large cyst should not read as under-amplified. Bands with
    too little signal left are dropped rather than counted as dark.
    """
    gray = np.asarray(gray_image, dtype=np.float64)
    has_signal = signal_mask(db_image, floor_db, margin_db)
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
                     floor_db=None, margin_db=DEFAULT_SIGNAL_MARGIN_DB):
    """Fraction of the 0..255 range the tissue does *not* occupy.

    Pushes against the exposure term: widening the dynamic range removes clipping but squeezes
    everything toward mid gray, and this is the cost of that squeeze.
    """
    gray = np.asarray(gray_image, dtype=np.float64)
    has_signal = signal_mask(db_image, floor_db, margin_db)
    if has_signal.sum() < 50:
        return 1.0
    values = gray[has_signal]
    span = np.percentile(values, high_pct) - np.percentile(values, low_pct)
    return float(np.clip(1.0 - span / GRAY_MAX, 0.0, 1.0))


# Weights. Clipping is the one failure that destroys information rather than merely presenting
# it poorly - a crushed pixel cannot be recovered by any later processing - so it carries the
# largest weight. Uniformity and utilisation trade off against each other and against it.
DEFAULT_WEIGHTS = {
    "crushed": 3.0,
    "saturated": 3.0,
    "uniformity": 1.0,
    "utilisation": 1.0,
}


def backend_objective(gray_image, db_image, weights=None, num_bands=NUM_TGC_BANDS,
                      margin_db=DEFAULT_SIGNAL_MARGIN_DB, return_terms=False):
    """Cost of one rendered image, for searching over gain, TGC and dynamic range.

    Lower is better. db_image is the pre-display dB image the render came from; it decides
    which pixels are supposed to be visible, so the same rendering cannot be scored without it.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    floor = noise_floor_db(db_image)
    crushed, saturated = exposure_cost(gray_image, db_image, floor, margin_db)
    uniformity = depth_uniformity_cost(gray_image, db_image, num_bands, floor, margin_db)
    utilisation = utilisation_cost(gray_image, db_image, floor_db=floor, margin_db=margin_db)

    terms = {
        "crushed": crushed,
        "saturated": saturated,
        "uniformity": 0.0 if np.isnan(uniformity) else uniformity,
        "utilisation": utilisation,
    }
    total = float(sum(weights[key] * value for key, value in terms.items()))
    if return_terms:
        terms["total"] = total
        terms["noise_floor_db"] = floor
        return total, terms
    return total
