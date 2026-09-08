"""Recovering the console's gray index from its screenshot, which is not a gray image.

    What was wrong

hisense_loader.load_screenshot() opens Screenthum.bmp and converts it with PIL's "L" mode,
which applies the ITU-R 601-2 luma weights 0.299 R + 0.587 G + 0.114 B. That is correct only
if the image is neutral. It is not: inside the B-mode rectangle only 2.4% to 21% of pixels
have R equal to G equal to B, and the largest channel spread reaches 161.

The console renders B-mode through a *coloured* map - a cool blue-white tint, blue above green
above red, increasingly so toward the bright end:

    palette entry        R     G     B      PIL luma
    brightest          231   239   250        237.9
                       190   196   205        195.2
                       127   131   137        130.5
                        65    67    70         66.7
    darkest              4     4     4          4.0

So the brightest level the console can display comes back as luma 238, not 255, and the error
is level-dependent. Every gray-domain comparison and every gray-domain term of the objective
has been reading a luma projection of a tinted image rather than the index the console
actually computed.

    Why a lookup table is the right model

The B-mode rectangle holds 534,668 pixels and only 217 to 246 distinct RGB triples, so the
display is a one-dimensional palette indexed by a gray level, not a per-channel transform of
something richer. The console also draws that palette on screen as a vertical bar, which makes
it directly readable rather than something to infer.

The bar is identical on every session and mode checked - same rows, same 229 distinct colours,
same endpoints - so one palette serves the whole dataset. It is read per capture anyway, since
that costs nothing and would catch a preset change.
"""

import numpy as np

# Columns the colour bar occupies on the 1600x940 console screen. Found by scanning for a
# narrow vertical strip whose luminance is monotone over a long run; it came out at 216-224 on
# every capture, and these five columns sit safely inside it.
BAR_COLUMNS = (218, 223)

# The strip also carries a label above the ramp, so the ramp is taken as the longest monotone
# run rather than the whole lit part of the column.
MIN_RAMP_ROWS = 120

PALETTE_SIZE = 256


def _longest_monotone_run(values, tolerance=1.0):
    """Start and stop of the longest run over which values only move one way."""
    best = (0, 0)
    for sign in (1.0, -1.0):
        start = 0
        for index in range(1, values.size):
            if sign * (values[index] - values[index - 1]) < -tolerance:
                if index - start > best[1] - best[0]:
                    best = (start, index)
                start = index
        if values.size - start > best[1] - best[0]:
            best = (start, values.size)
    return best


def extract_palette(screen_rgb, columns=BAR_COLUMNS, size=PALETTE_SIZE):
    """The console's display palette, as a size x 3 array ordered dark to bright.

    Read off the ramp the console draws on screen. Resampled to size entries because the bar is
    about 233 pixels tall and so cannot show every level; the resampling is linear in position,
    which is what the bar itself is.
    """
    screen_rgb = np.asarray(screen_rgb, dtype=np.float64)
    strip = screen_rgb[:, columns[0]:columns[1]].mean(axis=1)
    luminance = strip.sum(axis=1)
    lit = np.flatnonzero(luminance > 8)
    if lit.size < MIN_RAMP_ROWS:
        raise ValueError("No colour bar found in this screenshot")

    segment = strip[lit[0]:lit[-1] + 1]
    start, stop = _longest_monotone_run(segment.sum(axis=1))
    ramp = segment[start:stop]
    if ramp.shape[0] < MIN_RAMP_ROWS:
        raise ValueError("Colour bar ramp too short to be the display palette")
    if ramp.sum(axis=1)[0] > ramp.sum(axis=1)[-1]:
        ramp = ramp[::-1]                      # the bar is drawn bright at the top

    source = np.linspace(0.0, 1.0, ramp.shape[0])
    target = np.linspace(0.0, 1.0, int(size))
    return np.stack([np.interp(target, source, ramp[:, channel]) for channel in range(3)],
                    axis=1)


def invert_palette(screen_rgb, palette):
    """Map an RGB image back onto the palette's index, giving the console's own gray levels.

    Nearest colour in Euclidean RGB. The palette is monotone and well separated, so nearest
    colour is exact for any pixel the console actually drew; annotation overlays in other hues
    land on whichever entry is closest, which is why callers should crop to the B-mode
    rectangle first.
    """
    screen_rgb = np.asarray(screen_rgb, dtype=np.float64)
    shape = screen_rgb.shape[:-1]
    flat = screen_rgb.reshape(-1, 3)
    palette = np.asarray(palette, dtype=np.float64)

    index = np.empty(flat.shape[0], dtype=np.int32)
    residual = np.empty(flat.shape[0], dtype=np.float64)
    step = 40000                                # keeps the distance matrix a few hundred MB
    for start in range(0, flat.shape[0], step):
        chunk = flat[start:start + step]
        distance = ((chunk[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
        index[start:start + step] = distance.argmin(axis=1)
        residual[start:start + step] = np.sqrt(distance.min(axis=1))
    return index.reshape(shape).astype(np.float64), residual.reshape(shape)


def capture_screen_rgb(capture, screenshot_file="Screenthum.bmp"):
    """The capture's screenshot as RGB, without the luma collapse load_screenshot applies.

    Screenthum.bmp and the DICOM's pixel data were checked to be bit-identical on a capture at
    1600x940 RGB, so the DICOM offers no extra fidelity here and the bitmap is the cheaper read.
    """
    from PIL import Image
    return np.asarray(Image.open(capture.path / screenshot_file).convert("RGB"),
                      dtype=np.float64)


def invert_palette_fast(screen_rgb, palette):
    """invert_palette by luminance rather than a full nearest-colour search.

    The palette is monotone in luminance, so the nearest entry to a pixel the console drew is
    the one whose luminance matches - a binary search rather than 256 distance evaluations per
    pixel. Verified against invert_palette on the B-mode rectangle; use that one when a pixel
    might not come from the palette at all.
    """
    screen_rgb = np.asarray(screen_rgb, dtype=np.float64)
    palette = np.asarray(palette, dtype=np.float64)
    bounds = 0.5 * (palette[1:].sum(axis=1) + palette[:-1].sum(axis=1))
    return np.searchsorted(bounds, screen_rgb.sum(axis=-1)).astype(np.float64)


def capture_display_gray(capture, crop=True):
    """The B-mode rectangle of a capture, in the console's own gray index rather than luma.

    This is what every gray-domain comparison should run on. hisense_loader.load_screenshot()
    returns PIL's luma instead, which reads 3 to 16 levels low across the range because the
    display map is tinted; see the module docstring.
    """
    from hisense_loader import crop_image_area

    screen = capture_screen_rgb(capture)
    palette = extract_palette(screen)
    if not crop:
        return invert_palette_fast(screen, palette), palette
    luma = 0.299 * screen[..., 0] + 0.587 * screen[..., 1] + 0.114 * screen[..., 2]
    _, (row0, row1, col0, col1) = crop_image_area(luma, capture.geometry.image_width_px)
    return invert_palette_fast(screen[row0:row1, col0:col1], palette), palette
