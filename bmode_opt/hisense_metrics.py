"""Objective phantom image-quality metrics for Hisense B-mode captures.

Metrics operate on the absolute dB image produced by hisense_backend_sim.render_db(), on the
native BC0 grid of (depth, line). Working in dB on BC0 rather than on the console screenshot
keeps every metric independent of the display knobs and free of the burnt-in overlay.

Only numpy is used; connected-component labelling and the -6 dB width search are implemented
here rather than pulled from scipy/skimage, which are not installed in this project.
"""

import numpy as np

from hisense_loader import NUM_TGC_BANDS, band_edges


DEFAULT_CYST_CONTRAST_DB = 10.0
DEFAULT_MIN_CYST_AREA_MM2 = 1.0
DEFAULT_BACKGROUND_RING_MM = 1.5
FWHM_DROP_DB = 6.0


def depth_band_levels(db_image, num_bands=NUM_TGC_BANDS):
    """Median dB level of each depth band; this is the feedback observable for TGC.

    The median rather than the mean is used so that anechoic cysts and bright wire targets
    do not drag a band level away from the surrounding speckle.
    """
    db_image = np.asarray(db_image, dtype=np.float64)
    edges = band_edges(db_image.shape[0], num_bands)
    return np.array([np.median(db_image[edges[k]:edges[k + 1]]) for k in range(num_bands)])


def uniformity_cost(band_levels, target_db=None):
    """RMS deviation of the depth band levels from the target, in dB."""
    band_levels = np.asarray(band_levels, dtype=np.float64)
    if target_db is None:
        target_db = float(np.median(band_levels))
    return float(np.sqrt(np.mean((band_levels - target_db) ** 2)))


def speckle_snr(roi_db, detrend=True, valid_mask=None):
    """Speckle SNR (mean/std) of the amplitude fluctuation in a homogeneous region.

    Fully developed speckle has an amplitude SNR near 1.91. Values well below it indicate
    excess noise or residual structure; values above it indicate speckle-reduction smoothing.

    With detrend set, the per-depth median is removed first (a division in the amplitude
    domain). Without it, the depth-dependent echo trend dominates the variance and the result
    measures attenuation rather than speckle.
    """
    roi_db = np.asarray(roi_db, dtype=np.float64)
    if detrend and roi_db.ndim == 2:
        roi_db = roi_db - np.median(roi_db, axis=1, keepdims=True)

    amplitude = np.power(10.0, roi_db / 20.0)
    values = amplitude[valid_mask] if valid_mask is not None else amplitude.reshape(-1)
    if values.size < 2:
        return float("nan")
    std = float(values.std())
    return float(values.mean() / std) if std > 0 else float("inf")


def clip_fractions(gray_image, low=0, high=255):
    """Fraction of displayed pixels crushed to black and saturated to white."""
    gray = np.asarray(gray_image)
    return float((gray <= low).mean()), float((gray >= high).mean())


def _flood_fill(mask, start, labels, label):
    """Flood fill one 4-connected component of mask starting at start, writing into labels."""
    height, width = mask.shape
    stack = [start]
    labels[start] = label
    pixels = []
    while stack:
        row, col = stack.pop()
        pixels.append((row, col))
        for next_row, next_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= next_row < height and 0 <= next_col < width:
                if mask[next_row, next_col] and labels[next_row, next_col] == 0:
                    labels[next_row, next_col] = label
                    stack.append((next_row, next_col))
    return pixels


def label_components(mask):
    """Label 4-connected True regions of a boolean mask, returning (labels, count)."""
    mask = np.asarray(mask, dtype=bool)
    labels = np.zeros(mask.shape, dtype=np.int32)
    count = 0
    for row, col in zip(*np.nonzero(mask)):
        if labels[row, col] == 0:
            count += 1
            _flood_fill(mask, (row, col), labels, count)
    return labels, count


def _dilate(mask, radius_rows, radius_cols):
    """Dilate a boolean mask by a rectangular structuring element using shifts."""
    grown = np.zeros_like(mask, dtype=bool)
    for row_shift in range(-radius_rows, radius_rows + 1):
        for col_shift in range(-radius_cols, radius_cols + 1):
            grown |= np.roll(np.roll(mask, row_shift, axis=0), col_shift, axis=1)
    return grown


def detect_anechoic_targets(
    db_image,
    geometry,
    contrast_db=DEFAULT_CYST_CONTRAST_DB,
    min_area_mm2=DEFAULT_MIN_CYST_AREA_MM2,
    max_area_mm2=200.0,
):
    """Find anechoic cyst targets as dark connected regions below the local speckle level.

    Returns a list of dicts with the component mask, centroid in mm, and area in mm^2,
    sorted by descending area.
    """
    db_image = np.asarray(db_image, dtype=np.float64)
    # Compare against a per-depth reference so the search does not favour shallow depths.
    reference = np.median(db_image, axis=1, keepdims=True)
    labels, count = label_components(db_image < reference - contrast_db)

    pixel_area_mm2 = geometry.mm_per_point * geometry.mm_per_line
    targets = []
    for label in range(1, count + 1):
        mask = labels == label
        area_mm2 = float(mask.sum()) * pixel_area_mm2
        if not min_area_mm2 <= area_mm2 <= max_area_mm2:
            continue
        rows, cols = np.nonzero(mask)
        targets.append(
            {
                "mask": mask,
                "area_mm2": area_mm2,
                "depth_mm": float(rows.mean()) * geometry.mm_per_point,
                "lateral_mm": float(cols.mean()) * geometry.mm_per_line,
                "equivalent_diameter_mm": float(2.0 * np.sqrt(area_mm2 / np.pi)),
            }
        )
    return sorted(targets, key=lambda target: target["area_mm2"], reverse=True)


def cyst_cnr(db_image, target_mask, geometry, ring_mm=DEFAULT_BACKGROUND_RING_MM):
    """Contrast-to-noise ratio of one anechoic target against a surrounding background ring."""
    db_image = np.asarray(db_image, dtype=np.float64)
    radius_rows = max(1, int(round(ring_mm / geometry.mm_per_point)))
    radius_cols = max(1, int(round(ring_mm / geometry.mm_per_line)))
    background = _dilate(target_mask, radius_rows, radius_cols) & ~_dilate(target_mask, 1, 1)
    if background.sum() < 10 or target_mask.sum() < 10:
        return float("nan")

    inside, outside = db_image[target_mask], db_image[background]
    spread = np.sqrt((inside.var() + outside.var()) / 2.0)
    return float(abs(inside.mean() - outside.mean()) / spread) if spread > 0 else float("inf")


def _minus_width(profile, axis_mm, peak_index, drop_db=FWHM_DROP_DB):
    """Width of a peak at drop_db below its maximum, by linear interpolation on each side."""
    peak = profile[peak_index]
    threshold = peak - drop_db

    def crossing(indices):
        previous = peak_index
        for index in indices:
            if profile[index] <= threshold:
                span = profile[previous] - profile[index]
                fraction = 0.0 if span == 0 else (profile[previous] - threshold) / span
                return axis_mm[previous] + (axis_mm[index] - axis_mm[previous]) * fraction
            previous = index
        return None

    left = crossing(range(peak_index - 1, -1, -1))
    right = crossing(range(peak_index + 1, profile.size))
    if left is None or right is None:
        return float("nan")
    return float(right - left)


def point_target_fwhm(db_image, geometry, centre, search_mm=2.0):
    """Axial and lateral -6 dB widths of a wire target near centre=(depth_mm, lateral_mm)."""
    db_image = np.asarray(db_image, dtype=np.float64)
    row = int(round(geometry.row_of_depth(centre[0])))
    col = int(round(centre[1] / geometry.mm_per_line))
    row_span = max(1, int(round(search_mm / geometry.mm_per_point)))
    col_span = max(1, int(round(search_mm / geometry.mm_per_line)))

    row0, row1 = max(0, row - row_span), min(db_image.shape[0], row + row_span + 1)
    col0, col1 = max(0, col - col_span), min(db_image.shape[1], col + col_span + 1)
    window = db_image[row0:row1, col0:col1]
    local = np.unravel_index(int(np.argmax(window)), window.shape)
    peak_row, peak_col = row0 + local[0], col0 + local[1]

    axial_axis = (np.arange(db_image.shape[0]) * geometry.mm_per_point
                  + geometry.min_depth_mm)
    lateral_axis = np.arange(db_image.shape[1]) * geometry.mm_per_line
    axial = _minus_width(db_image[:, peak_col], axial_axis, peak_row)
    lateral = _minus_width(db_image[peak_row, :], lateral_axis, peak_col)
    return {
        "depth_mm": float(peak_row * geometry.mm_per_point + geometry.min_depth_mm),
        "lateral_mm": float(peak_col * geometry.mm_per_line),
        "axial_fwhm_mm": axial,
        "lateral_fwhm_mm": lateral,
    }


def summarise(db_image, geometry, gray_image=None, num_bands=NUM_TGC_BANDS):
    """Collect the headline phantom metrics for one rendered frame."""
    bands = depth_band_levels(db_image, num_bands)
    targets = detect_anechoic_targets(db_image, geometry)

    # Speckle statistics are only meaningful away from the cysts and wire targets.
    speckle_mask = np.ones(db_image.shape, dtype=bool)
    for target in targets:
        speckle_mask &= ~_dilate(target["mask"], 2, 2)

    report = {
        "band_levels_db": bands,
        "uniformity_db": uniformity_cost(bands),
        "speckle_snr": speckle_snr(db_image, valid_mask=speckle_mask),
        "dynamic_span_db": float(np.percentile(db_image, 99) - np.percentile(db_image, 1)),
        "num_targets": len(targets),
    }
    if gray_image is not None:
        crushed, saturated = clip_fractions(gray_image)
        report["crushed_fraction"] = crushed
        report["saturated_fraction"] = saturated

    if targets:
        cnrs = [cyst_cnr(db_image, target["mask"], geometry) for target in targets[:5]]
        cnrs = [value for value in cnrs if np.isfinite(value)]
        report["mean_cyst_cnr"] = float(np.mean(cnrs)) if cnrs else float("nan")
        report["largest_target_mm"] = targets[0]["equivalent_diameter_mm"]
    return report
