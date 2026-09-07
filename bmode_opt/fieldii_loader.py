"""Load Field II simulation shards as captures the Hisense back-end model can render.

data/field_ii/full holds 4560 shards written by fieldii_generate_dataset.m: three phantom
types (uniform, cyst, point) x 20 seeds x 76 (depth, frequency, focus) settings. They exist
because the console data is scene-poor - 126 grid frames from four probe placements - while
the simulation carries 60 independent phantoms and, unlike the console, ground truth.

    Which dataset is the console's Algo_BC0.bin?

The shard's /algo_bc0 is NOT it, despite the name. The generator applies gain and TGC first
and encodes the result:

    display_envelope = envelope * 10^((gain_db + tgc_curve_db)/20)
    algo_bc0         = 65535 * clip((20*log10(display_envelope/reference) + DR)/DR, 0, 1)

Recomputing /algo_bc0 that way reproduces it to 1 count on 99.98% of pixels, while using
/envelope directly disagrees on 96.5%. So /algo_bc0 sits *after* the back end, whereas the
console's tap sits *before* it - the whole premise of this project. The dataset that matches
the console is /envelope, and it is the better one anyway: 103.6 dB of span against the 67 dB
/algo_bc0 keeps, with 3.45% of /algo_bc0 already crushed to zero.

    Levels are absolute, and they range far wider than the gain axis

Every shard shares one display_reference, so dB values are comparable across the whole set.
They are also spread out: over a 30-shard sample the mid-depth tissue median ranges from
-74.5 to -15.9 dB, a 58.6 dB span driven by depth and frequency. The synthesis gain axis is
8 dB wide (E3), so it cannot by itself bring every shard to a sensible exposure. Callers that
need a standard operating point should use tissue_median_db per capture - that is the
simulated equivalent of an operator setting gain per the protocol's exposure rule - rather
than assuming one global offset works.

    TGC

Shards carry a four-knot TGC curve in /tgc_curve_db, already folded into /algo_bc0. Working
from /envelope makes it irrelevant: this loader reports flat sliders and lets the Hisense
eight-slider model supply the whole curve, so both data sources share one action space with
no interpolation between four knots and eight.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hisense_loader import Geometry

DEFAULT_FIELDII_DIR = (Path(__file__).resolve().parent.parent
                       / "data" / "field_ii" / "full")
HDF5_SUBDIR = "hdf5"

# Counts per dB for the synthetic bc0 view. Matched to the console's fitted constant so the
# two sources feed render() on the same scale; the choice is free here because /envelope is
# raw floats, not a pre-encoded integer image.
FIELDII_COUNTS_PER_DB = 877.3

# Reported on every capture so downstream code treats these like a console capture at the
# calibration gain, flat TGC and the console's default dynamic-range number. None of the
# three is a property of the simulation; they are the operating point the synthesis
# perturbs away from.
NOMINAL_GAIN_LEVEL = 75
NOMINAL_TGC_LEVEL = 127
NOMINAL_DR_UI = 67

_NAME_RE = re.compile(
    r"^(?P<split>[a-z]+)_(?P<phantom>[a-z]+)_s(?P<seed>\d+)"
    r"_d(?P<depth>\d+p\d+)_f(?P<freq>\d+p\d+)_zf(?P<focus>\d+p\d+)$"
)


def _num(token):
    """'04p5' -> 4.5"""
    return float(token.replace("p", "."))


@dataclass
class FieldIICapture:
    """One Field II shard, shaped like hisense_loader.Capture where that matters.

    bc0, geometry, tgc_levels, gain_level and dynamic_range_level exist so the same
    render()/metrics code paths work unchanged. db_image, truth_mask and point_targets_mm
    are the parts the console cannot provide.
    """

    path: Path
    bc0: np.ndarray                 # db_image * counts_per_db, float64, (depth, lines)
    db_image: np.ndarray            # 20*log10(envelope / display_reference)
    geometry: Geometry
    tgc_levels: np.ndarray
    gain_level: int
    dynamic_range_level: int
    counts_per_db: float
    phantom_type: str
    split: str
    seed: int
    frequency_mhz: float
    focus_mm: float
    truth_mask: np.ndarray          # (depth, lines) uint8; nonzero only for cyst phantoms
    point_targets_mm: np.ndarray    # (N, 2) as [depth_mm, lateral_mm]; empty for others
    attrs: dict = field(repr=False, default_factory=dict)

    @property
    def name(self):
        return self.path.stem

    @property
    def tissue_median_db(self):
        """Median dB over the middle half of the depth range.

        The exposure handle: passing this as reference_db puts the tissue median on the
        display pivot, which normalises frames against each other. It is not a good
        operating point on its own - GRAY_PIVOT is 43, while the objective prefers tissue
        nearer gray 115 - so expect the gain search to land well above zero from here.

        Computed over the middle half so the near-field coupling rows and the noise-floor
        tail stay out of it.
        """
        lo, hi = self.db_image.shape[0] // 4, 3 * self.db_image.shape[0] // 4
        return float(np.median(self.db_image[lo:hi]))

    def targets_in_view(self, margin_mm=1.0):
        """Point targets that fall inside this shard's depth range."""
        if self.point_targets_mm.size == 0:
            return self.point_targets_mm
        z = self.point_targets_mm[:, 0]
        keep = (z >= self.geometry.min_depth_mm + margin_mm) & (z <= self.geometry.depth_mm - margin_mm)
        return self.point_targets_mm[keep]


def parse_shard_name(path):
    """Pull split / phantom / seed / depth / frequency / focus out of a shard filename."""
    match = _NAME_RE.match(Path(path).stem)
    if match is None:
        raise ValueError(f"Unrecognised Field II shard name: {Path(path).name}")
    groups = match.groupdict()
    return {
        "split": groups["split"],
        "phantom_type": groups["phantom"],
        "seed": int(groups["seed"]),
        "depth_mm": _num(groups["depth"]),
        "frequency_mhz": _num(groups["freq"]),
        "focus_mm": _num(groups["focus"]),
    }


def find_shards(data_dir=DEFAULT_FIELDII_DIR, split=None, phantom_type=None,
                depth_mm=None, frequency_mhz=None, focus_mm=None):
    """Shard paths under data_dir/hdf5, filtered on the fields encoded in the filename.

    Filtering by name avoids opening 4560 files to select a handful.
    """
    root = Path(data_dir)
    hdf5_dir = root / HDF5_SUBDIR if (root / HDF5_SUBDIR).is_dir() else root
    if not hdf5_dir.is_dir():
        raise NotADirectoryError(f"Field II directory not found: {hdf5_dir}")

    wanted = {
        "split": split,
        "phantom_type": phantom_type,
        "depth_mm": depth_mm,
        "frequency_mhz": frequency_mhz,
        "focus_mm": focus_mm,
    }
    out = []
    for path in sorted(hdf5_dir.glob("*.h5")):
        try:
            meta = parse_shard_name(path)
        except ValueError:
            continue
        if all(want is None or meta[key] == want for key, want in wanted.items()):
            out.append(path)
    if not out:
        raise FileNotFoundError(f"No Field II shards under {hdf5_dir} matching {wanted}")
    return out


def _scalar(attrs, key, default=None):
    if key not in attrs:
        return default
    value = attrs[key]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    array = np.ravel(value)
    return array[0] if array.size else default


def load_shard(path, counts_per_db=FIELDII_COUNTS_PER_DB):
    """Load one shard into a FieldIICapture.

    Needs h5py, which the project's default interpreter does not have; use the cubdl conda
    environment. Arrays arrive from MATLAB transposed, so everything is flipped to the
    (depth, lines) layout the rest of the code assumes.
    """
    import h5py

    path = Path(path)
    with h5py.File(path, "r") as handle:
        envelope = np.asarray(handle["/envelope"][()], dtype=np.float64).T
        truth_mask = np.asarray(handle["/truth_mask"][()], dtype=np.uint8).T
        depth_axis = np.asarray(handle["/depth_axis_mm"][()], dtype=np.float64).ravel()
        lateral_axis = np.asarray(handle["/lateral_axis_mm"][()], dtype=np.float64).ravel()
        raw_targets = np.asarray(handle["/point_targets_mm"][()], dtype=np.float64)
        attrs = {key: handle.attrs[key] for key in handle.attrs}

    reference = float(_scalar(attrs, "display_reference"))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"{path.name}: display_reference is not usable ({reference})")

    # A floor at the smallest positive envelope value keeps log10 finite without inventing
    # a dynamic range the shard does not have.
    positive = envelope[envelope > 0]
    floor = positive.min() if positive.size else 1e-300
    db_image = 20.0 * np.log10(np.maximum(envelope, floor) / reference)

    # /point_targets_mm is [x; z] in MATLAB order and reads back as (2, N); an empty target
    # list is written as a single zero row, which point_target_count flags.
    target_count = int(_scalar(attrs, "point_target_count", 0) or 0)
    if target_count > 0 and raw_targets.size:
        lateral, depth = (raw_targets if raw_targets.shape[0] == 2 else raw_targets.T)[:2]
        targets = np.column_stack([depth, lateral])          # (N, 2) as [depth, lateral]
    else:
        targets = np.zeros((0, 2), dtype=np.float64)

    meta = parse_shard_name(path)
    geometry = Geometry(
        depth_mm=float(depth_axis[-1]),
        width_mm=float(lateral_axis[-1] - lateral_axis[0]),
        pixel_gap_mm=float("nan"),      # no scan-converted raster exists for a shard
        image_width_px=float("nan"),
        image_height_px=float("nan"),
        num_points=db_image.shape[0],
        num_lines=db_image.shape[1],
        min_depth_mm=float(depth_axis[0]),
    )

    return FieldIICapture(
        path=path,
        bc0=db_image * float(counts_per_db),
        db_image=db_image,
        geometry=geometry,
        tgc_levels=np.full(8, NOMINAL_TGC_LEVEL, dtype=np.int64),
        gain_level=NOMINAL_GAIN_LEVEL,
        dynamic_range_level=NOMINAL_DR_UI,
        counts_per_db=float(counts_per_db),
        phantom_type=meta["phantom_type"],
        split=meta["split"],
        seed=meta["seed"],
        frequency_mhz=meta["frequency_mhz"],
        focus_mm=meta["focus_mm"],
        truth_mask=truth_mask,
        point_targets_mm=targets,
        attrs=attrs,
    )
