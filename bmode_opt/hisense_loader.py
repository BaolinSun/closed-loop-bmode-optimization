"""Loader for Hisense ultrasound console export directories."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Resolved from this file so the project can be moved or cloned without editing paths.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from . import data_paths as DP_PATHS
except ImportError:
    import data_paths as DP_PATHS
# Phantom sessions live on the dataset drive (bmode_opt/data_paths.py), not in the project.
DEFAULT_DATA_DIR = DP_PATHS.PHANTOM_ROOT

FE_PARAM_FILE = "Algo_FeParam.pdt"
BE_PARAM_FILE = "BEParam.pdt"
PARTITION_FILE = "Algo_PartitionInfo.pdt"
SCREENSHOT_FILE = "Screenthum.bmp"
BC0_FILE = "Algo_BC0.bin"

NUM_TGC_BANDS = 8
BC0_DTYPE = np.uint16
IMAGE_AREA_THRESHOLD = 0.15
WINDOW_OCCUPANCY = 0.3

# Screenshot row where the console's image panel starts: the title bar above it (logo, date,
# probe) is 80 px tall on the 1600 x 940 Screenthum.bmp. BEParam's WinTop / BImageTop are
# relative to that panel, WinLeft is already a screen column. See screen_image_bounds().
SCREEN_IMAGE_PANEL_TOP_PX = 80

_MISSING = object()


def parse_pdt(path):
    """Parse an indented key:value .pdt file into a flat dict keyed by dotted section path."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDT file not found: {path}")

    params = {}
    stack = []
    # Line terminators in these exports are CR CR LF; normalising every CR to LF is enough.
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(" Begin"):
            stack.append(line[: -len(" Begin")].strip())
        elif line.endswith(" End"):
            if stack:
                stack.pop()
        elif ":" in line:
            # Values may themselves contain a colon (FILEPATH:D:/PatientData/...), so split once.
            key, value = line.split(":", 1)
            params[".".join(stack + [key.strip()])] = value.strip()
    return params


def get_leaf(params, name, default=_MISSING):
    """Look up a parameter by its leaf name, ignoring the enclosing section path."""
    suffix = "." + name
    hits = {key: value for key, value in params.items() if key == name or key.endswith(suffix)}
    if not hits:
        if default is _MISSING:
            raise KeyError(f"Parameter not found: {name}")
        return default
    if len(set(hits.values())) > 1:
        raise KeyError(f"Ambiguous parameter {name!r} with differing values: {sorted(hits)}")
    return next(iter(hits.values()))


def leaf_int(params, name, default=_MISSING):
    """Look up a leaf parameter and convert it to int."""
    value = get_leaf(params, name, default)
    return int(value) if isinstance(value, str) else value


def leaf_float(params, name, default=_MISSING):
    """Look up a leaf parameter and convert it to float."""
    value = get_leaf(params, name, default)
    return float(value) if isinstance(value, str) else value


def leaf_floats(params, name, default=_MISSING):
    """Look up a leaf parameter and convert its comma separated items to a float array."""
    value = get_leaf(params, name, default)
    if not isinstance(value, str):
        return np.asarray(value, dtype=np.float64)
    return np.array([float(item) for item in value.split(",") if item.strip()], dtype=np.float64)


class CineCaptureError(ValueError):
    """A multi-page (cine) export. Its PartitionInfo fields are per-page comma lists and its BC0
    holds hundreds of frames; nothing downstream handles that, so it is refused by name."""


def capture_page_num(capture_dir):
    """PageNum of a capture: 1 for a single frame, the number of frames for a cine export.

    Reads only Algo_PartitionInfo.pdt, so it is cheap enough to filter a whole folder with.
    """
    return _page_num(parse_pdt(Path(capture_dir) / PARTITION_FILE))


def _page_num(partition):
    return int(str(get_leaf(partition, "PageNum")).split(",")[0])


def load_bc0(capture_dir):
    """Load Algo_BC0.bin as a (depth, line) log-domain envelope array."""
    capture_dir = Path(capture_dir)
    partition = parse_pdt(capture_dir / PARTITION_FILE)
    # 电影回放导出（PageNum > 1）时 Line / Point / PageSize 都是逐页列表，下面的 int() 会
    # 以一个看不出原因的 ValueError 崩掉。
    if "," in str(get_leaf(partition, "Line")) or _page_num(partition) > 1:
        raise CineCaptureError("%s is a cine export (PageNum %s); only single frames load"
                               % (capture_dir.name, get_leaf(partition, "PageNum")))
    num_lines = leaf_int(partition, "Line")
    num_points = leaf_int(partition, "Point")
    page_size = leaf_int(partition, "PageSize")
    page_num = leaf_int(partition, "PageNum")

    itemsize = np.dtype(BC0_DTYPE).itemsize
    if num_lines * num_points * itemsize != page_size:
        raise ValueError(
            f"PartitionInfo is inconsistent: Line({num_lines}) * Point({num_points}) * {itemsize} "
            f"!= PageSize({page_size})"
        )

    bc0_path = capture_dir / BC0_FILE
    if not bc0_path.exists():
        raise FileNotFoundError(f"BC0 file not found: {bc0_path}")
    actual_size = bc0_path.stat().st_size
    if actual_size != page_size * page_num:
        raise ValueError(
            f"Bad BC0 size for {bc0_path}: got {actual_size}, "
            f"expected PageSize({page_size}) * PageNum({page_num})"
        )

    raw = np.fromfile(bc0_path, dtype=BC0_DTYPE)
    # Stored line-major; transpose so the array is indexed (depth, line).
    return raw.reshape(num_lines, num_points).T.astype(np.float32)


def get_tgc_levels(be_params):
    """Return the eight BTgc*Level slider values as an int array, shallow band first."""
    return np.array(
        [leaf_int(be_params, f"BTgc{band + 1}Level") for band in range(NUM_TGC_BANDS)],
        dtype=np.int32,
    )


def band_edges(num_points, num_bands=NUM_TGC_BANDS):
    """Return the num_bands+1 depth sample indices that delimit the TGC bands."""
    return np.linspace(0, int(num_points), int(num_bands) + 1).round().astype(int)


def band_centres(num_points, num_bands=NUM_TGC_BANDS):
    """Return the depth sample index at the centre of each TGC band."""
    edges = band_edges(num_points, num_bands)
    return (edges[:-1] + edges[1:]) / 2.0


def _longest_run(mask):
    """Return (start, stop) of the longest contiguous True run in a 1-D boolean mask."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, stops = edges[0::2], edges[1::2]
    if starts.size == 0:
        raise ValueError("No above-threshold region found")
    longest = int(np.argmax(stops - starts))
    return int(starts[longest]), int(stops[longest])


def crop_image_area(gray, image_width_px=None, threshold=IMAGE_AREA_THRESHOLD, occupancy=0.5,
                    window_occupancy=WINDOW_OCCUPANCY):
    """Locate the B-mode image rectangle in a console screenshot, returning (image, bounds).

    Pass image_width_px (BImageWidth) whenever it is available. The columns are then derived
    from geometry: pixel occupancy locates the B window - which reproduces WinLeft..WinLeft+
    WinWidth exactly and does not depend on how bright the image is - and the image is taken
    centred in that window with the width the console reports.

    Without image_width_px the columns fall back to the longest contiguous run of
    above-threshold column means. That heuristic breaks on dark images: the threshold is set
    by the brightest column, which is the graymap bar, and a fundamental-mode image can sit
    entirely below it, leaving the Hisense logo as the longest surviving run.

    Rows always come from pixel occupancy rather than brightness, because an extreme TGC
    setting can crush a whole depth band to black and would otherwise split the run.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if image_width_px is not None:
        column_occupancy = (gray > 0).mean(axis=0)
        inside = np.flatnonzero(column_occupancy > window_occupancy)
        if inside.size == 0:
            raise ValueError("No B window found in the screenshot")
        centre = (inside[0] + inside[-1] + 1) / 2.0
        col0 = max(0, int(round(centre - float(image_width_px) / 2.0)))
        col1 = min(gray.shape[1], int(round(centre + float(image_width_px) / 2.0)))
    else:
        col_mean = gray.mean(axis=0)
        col0, col1 = _longest_run(col_mean > col_mean.max() * threshold)

    row_occupancy = (gray[:, col0:col1] > 0).mean(axis=1)
    row0, row1 = _longest_run(row_occupancy > occupancy)
    return gray[row0:row1, col0:col1], (row0, row1, col0, col1)


def _displayed_image_px(geometry):
    """(rows, cols) the image occupies on screen: physical size over PixelGap (mm per pixel)."""
    return ((geometry.depth_mm - geometry.min_depth_mm) / geometry.pixel_gap_mm,
            geometry.width_mm / geometry.pixel_gap_mm)


def _has_window(geometry):
    return all(value is not None for value in (geometry.win_left_px, geometry.win_top_px,
                                               geometry.win_width_px, geometry.win_height_px,
                                               geometry.image_top_px))


def image_fits_window(geometry):
    """False when the display zoom makes the image larger than the B window, so the screenshot
    shows only part of it. Such a screenshot cannot be compared with the whole BC0. True when
    the export carries no window fields (nothing to check against)."""
    if not _has_window(geometry):
        return True
    rows, cols = _displayed_image_px(geometry)
    return rows <= geometry.win_height_px + 1.0 and cols <= geometry.win_width_px + 1.0


def screen_image_bounds(geometry, screen_shape):
    """(row0, row1, col0, col1) of the B-mode image in the screenshot, from exported geometry.

    The console draws the image at PixelGap mm per screen pixel, centred in the B window
    (screen rows 80 + WinTop .. + WinHeight, columns WinLeft .. + WinWidth). PixelGap already
    carries the display zoom ("Z" on screen, BZoomRatio): on the 2026-10-09/10 carotid captures
    zoom 1.1 / 1.2 / 1.4 / 1.6 / 1.8 all occur, and the union of the frames at each zoom fills
    exactly depth / PixelGap rows by width / PixelGap columns around the window centre
    (1.2: 763 x 695 measured, 762.9 x 694.2 predicted; 1.1: 700 x 637 vs 699.4 x 636.4).
    BImageWidth / BImageHeight do not follow the zoom, so they cannot be used here.

    Returns None when the export carries no window fields (callers fall back to
    crop_image_area()) or when the zoom has pushed the image past the window edges (see
    image_fits_window()). Unlike crop_image_area() this does not look at pixel values, so a
    dark image, a crushed depth band or background noise rejection cannot move the crop - the
    occupancy crop came out a median 96 rows short on the 2026-10 firmware.
    """
    if not _has_window(geometry) or not image_fits_window(geometry):
        return None
    rows, cols = _displayed_image_px(geometry)
    centre_row = (SCREEN_IMAGE_PANEL_TOP_PX + geometry.win_top_px + geometry.image_top_px
                  + geometry.win_height_px / 2.0)
    centre_col = geometry.win_left_px + geometry.win_width_px / 2.0
    row0 = max(0, int(round(centre_row - rows / 2.0)))
    row1 = min(screen_shape[0], int(round(centre_row + rows / 2.0)))
    col0 = max(0, int(round(centre_col - cols / 2.0)))
    col1 = min(screen_shape[1], int(round(centre_col + cols / 2.0)))
    return row0, row1, col0, col1


def crop_capture_image(capture, **kwargs):
    """Load a capture's screenshot and crop it to the B-mode image using its own geometry."""
    screen = load_screenshot(capture.path)
    if not image_fits_window(capture.geometry):
        raise ValueError("%s: display zoom puts the image past the window edges; the "
                         "screenshot shows only part of it" % capture.name)
    bounds = screen_image_bounds(capture.geometry, screen.shape)
    if bounds is None or kwargs:
        return crop_image_area(screen, capture.geometry.image_width_px, **kwargs)
    row0, row1, col0, col1 = bounds
    return screen[row0:row1, col0:col1], bounds


def load_screenshot(capture_dir):
    """Load Screenthum.bmp as a float grayscale array, via PIL's luma weights.

    Careful: the console's B-mode image is not neutral. It is drawn through a tinted palette,
    so this luma projection reads 3 to 16 levels below the gray index the console actually
    computed, and the shortfall depends on the level. Anything comparing against the console's
    own gray - calibration, fidelity, the objective's gray-domain terms - should go through
    display_palette.capture_display_gray() instead. This function is kept because the earlier
    display-response and TGC work was measured on it, and those numbers refer to luma.
    """
    from PIL import Image

    path = Path(capture_dir) / SCREENSHOT_FILE
    if not path.exists():
        raise FileNotFoundError(f"Screenshot not found: {path}")
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


@dataclass
class Geometry:
    """Physical and pixel geometry of one B-mode capture."""

    depth_mm: float
    width_mm: float
    pixel_gap_mm: float
    image_width_px: float
    image_height_px: float
    num_points: int
    num_lines: int
    # Depth of the first sample. Zero for Hisense exports, which start at the probe
    # face; Field II shards start at their configured min_depth_mm instead.
    min_depth_mm: float = 0.0
    # Where the B window sits on the screenshot (BEParam WinLeft / WinTop / WinWidth /
    # BImageTop). None when not exported; see screen_image_bounds().
    win_left_px: float = None
    win_top_px: float = None
    win_width_px: float = None
    win_height_px: float = None
    image_top_px: float = None

    @property
    def mm_per_point(self):
        """Axial spacing of one BC0 depth sample, in mm."""
        return (self.depth_mm - self.min_depth_mm) / self.num_points

    def row_of_depth(self, depth_mm):
        """Row index of a physical depth, accounting for min_depth_mm."""
        return (float(depth_mm) - self.min_depth_mm) / self.mm_per_point

    @property
    def mm_per_line(self):
        """Lateral spacing of one BC0 scan line, in mm."""
        return self.width_mm / self.num_lines


@dataclass
class Capture:
    """One Hisense export directory: raw BC0 plus front-end and back-end parameters."""

    path: Path
    bc0: np.ndarray
    fe_params: dict
    be_params: dict
    partition: dict
    geometry: Geometry
    tgc_levels: np.ndarray
    gain_level: int
    dynamic_range_level: int
    # 发射聚焦深度，单位 mm。BFocusArrayPos 有 16 个槽位，前 BFocusNumValue 个有效。
    # 目前 291 帧全是单焦点，但保留元组，因为多焦点预设会填进更多槽位，而那会
    # 让「聚焦该往哪调」这个标签的含义完全不同。
    focus_depths_mm: tuple = ()

    @property
    def focus_mm(self):
        """单发射聚焦的深度。多焦点时抛错，而不是悄悄取第一个。"""
        if len(self.focus_depths_mm) != 1:
            raise ValueError(
                f"{self.name} has {len(self.focus_depths_mm)} transmit foci "
                f"{self.focus_depths_mm}; focus_mm is only defined for one")
        return self.focus_depths_mm[0]

    @property
    def name(self):
        """Capture directory name, which is also the console timestamp id."""
        return self.path.name


def _optional_float(params, name):
    """A leaf as float, or None when it is missing or ambiguous."""
    try:
        return leaf_float(params, name, None)
    except (KeyError, ValueError):
        return None


def load_capture(capture_dir):
    """Load one Hisense export directory into a Capture."""
    capture_dir = Path(capture_dir)
    if not capture_dir.is_dir():
        raise NotADirectoryError(f"Capture directory not found: {capture_dir}")

    fe_params = parse_pdt(capture_dir / FE_PARAM_FILE)
    be_params = parse_pdt(capture_dir / BE_PARAM_FILE)
    partition = parse_pdt(capture_dir / PARTITION_FILE)
    bc0 = load_bc0(capture_dir)

    geometry = Geometry(
        depth_mm=leaf_float(fe_params, "BDisplayDepth"),
        width_mm=leaf_float(fe_params, "BTotalScanWidth"),
        pixel_gap_mm=leaf_float(be_params, "PixelGap"),
        image_width_px=leaf_float(be_params, "BImageWidth"),
        image_height_px=leaf_float(be_params, "BImageHeight"),
        num_points=bc0.shape[0],
        num_lines=bc0.shape[1],
        win_left_px=_optional_float(be_params, "WinLeft"),
        win_top_px=_optional_float(be_params, "WinTop"),
        win_width_px=_optional_float(be_params, "WinWidth"),
        win_height_px=_optional_float(be_params, "WinHeight"),
        image_top_px=_optional_float(be_params, "BImageTop"),
    )

    return Capture(
        path=capture_dir,
        bc0=bc0,
        fe_params=fe_params,
        be_params=be_params,
        partition=partition,
        geometry=geometry,
        tgc_levels=get_tgc_levels(be_params),
        gain_level=leaf_int(be_params, "BUIGainLevel"),
        dynamic_range_level=leaf_int(be_params, "UIDynamicRangeLevel"),
        focus_depths_mm=tuple(
            leaf_floats(fe_params, "BFocusArrayPos")
            [:leaf_int(fe_params, "BFocusNumValue", 1)]),
    )


def find_captures(data_dir=DEFAULT_DATA_DIR):
    """Return every sub-directory of data_dir holding a BC0 file, sorted by name.

    Only direct children are searched. Captures are grouped into per-session folders, and a
    depth response calibrated in one session does not transfer to another, so silently
    pooling sessions would corrupt the calibration rather than merely widen it. When a
    directory holds sessions instead of captures, the error names them so the caller can pick.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    captures = sorted(
        child for child in data_dir.iterdir() if child.is_dir() and (child / BC0_FILE).exists()
    )
    if captures:
        return captures

    sessions = sorted(
        child.name
        for child in data_dir.iterdir()
        if child.is_dir() and any(sub.is_dir() and (sub / BC0_FILE).exists() for sub in child.iterdir())
    )
    if sessions:
        raise FileNotFoundError(
            f"No captures directly under {data_dir}, but it holds capture sessions: "
            f"{', '.join(sessions)}. Point --data-dir at one session, e.g. {data_dir / sessions[-1]}"
        )
    return captures


def build_parser():
    parser = argparse.ArgumentParser(description="Summarise Hisense ultrasound export directories.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Directory holding capture folders.")
    parser.add_argument("--capture", type=Path, default=None, help="Summarise a single capture directory.")
    return parser


def main():
    args = build_parser().parse_args()
    capture_dirs = [args.capture] if args.capture else find_captures(args.data_dir)
    if not capture_dirs:
        raise SystemExit(f"No captures found under {args.data_dir}")

    for capture_dir in capture_dirs:
        capture = load_capture(capture_dir)
        geometry = capture.geometry
        print(capture.name)
        print(f"  BC0        : {geometry.num_points} points x {geometry.num_lines} lines, "
              f"range [{capture.bc0.min():.0f}, {capture.bc0.max():.0f}]")
        print(f"  geometry   : {geometry.depth_mm:.2f} mm deep x {geometry.width_mm:.2f} mm wide, "
              f"{geometry.mm_per_point * 1000:.1f} um/point")
        print(f"  probe      : ID {get_leaf(capture.fe_params, 'ProbeID')}, "
              f"freq {get_leaf(capture.fe_params, 'BFreqValue')} MHz, "
              f"MI {get_leaf(capture.fe_params, 'MI')}")
        print(f"  gain / DR  : {capture.gain_level} / {capture.dynamic_range_level}")
        print(f"  TGC levels : {capture.tgc_levels.tolist()}")


if __name__ == "__main__":
    main()
