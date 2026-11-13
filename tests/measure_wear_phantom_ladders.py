"""体模 T6：频率、聚焦梯级在新板上是否逐档有区别（docs/wear_phantom_pilot_protocol.md §8.2）。

读一个会话的 model_data/（研究模式“导出数据”的目录，按 acquisition_step 分块、分步）。用 BC0（每 dB 400 计数，
第 i 行深度 i·末深/870，256 列等分 x_min..x_max），每块单独分析（块与块之间探头可能移动过）：

- 运动：R′、R1、R2、R3、R4 两两在 10–35 mm 的相关和平均电平差。
- 频率（F1–F5，焦点为 R′ 的焦点）：每 5 mm 段平均 dB；噪声底（55–67 mm 平均）；穿透深度（从 10 mm 往深，
  段平均高出噪声底 >= 6 dB 的最深处）；10–30 mm 的电平斜率；10–20 mm 斑点的轴向、侧向相关长度（-6 dB 宽，分辨率的代理）。
- 聚焦（Z1–Z5，频率为 R′ 的频率）：每段对 5 档平均归一后，哪一档最亮；斑点侧向相关长度随深度。
- 钢丝（若这一块有）：在 R′ 上找比局部背景高 >= 15 dB 的点，逐档量侧向、轴向 -6 dB 宽度与峰值。

注意：包络小、噪声底是量化地板（见 §8.4），所以“噪声底 + 6 dB”是相对这台设备的量化地板而言。
用法：python tests/measure_wear_phantom_ladders.py <会话目录>
报告（英文）写到本脚本旁的 measure_wear_phantom_ladders.txt，并打印同样内容。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

COUNTS_PER_DB = 400.0
POINTS, LINES = 870, 256
BAND = 5.0
TOP_MM = 10.0                 # skip the reverberation under the probe face
# Frequency of each F step: read from the export (acquisition_step.frequency_step_mhz); the ladder was
# fundamental 5.0-11.4 until 2026-11-06 and harmonic 4.4-5.7 since.  Fallback only when the field is missing.
FREQ = {"F1": 4.4, "F2": 4.7, "F3": 5.0, "F4": 5.3, "F5": 5.7}
FOCUS = {"Z1": 10, "Z2": 15, "Z3": 20, "Z4": 25, "Z5": 30}


def step_mhz(frames, step):
    """The Hisense ladder value an F frame was exported with."""
    value = frames[step][0].get("frequency_step_mhz") if step in frames else None
    return float(value) if value is not None else FREQ[step]


def load(directory):
    meta = json.loads((directory / "wear_model_meta.json").read_text(encoding="utf-8"))
    bc = np.fromfile(directory / "Algo_BC0.bin", dtype=np.uint16).reshape(LINES, POINTS).T.astype(np.float64)
    image = meta["image"]
    z = np.arange(POINTS) * float(image["end_depth_mm"]) / POINTS
    x = np.linspace(float(image["x_min_mm"]), float(image["x_max_mm"]), LINES)
    return meta["acquisition_step"], bc / COUNTS_PER_DB, bc > 0, z, x


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def band_means(db, valid, z, top=TOP_MM, end=None):
    end = end or z[-1]
    out = []
    for lo in np.arange(top, end - 1e-9, BAND):
        sel = valid & ((z >= lo) & (z < lo + BAND))[:, None]
        out.append((lo, float(db[sel].mean()) if sel.any() else float("nan")))
    return out


def speckle_width(db, valid, z, x, lo, hi):
    """-6 dB (half of the normalised autocorrelation peak) widths of the dB speckle, axial and lateral, in mm."""
    rows = (z >= lo) & (z < hi)
    cols = valid[rows].all(axis=0)
    w = db[rows][:, cols]
    w = w - w.mean()
    f = np.fft.fft2(w, s=(2 * w.shape[0], 2 * w.shape[1]))
    ac = np.real(np.fft.ifft2(f * np.conj(f)))
    ac /= ac[0, 0]
    def width(profile, step):
        below = np.flatnonzero(profile < 0.5)
        if not below.size:
            return float("nan")
        k = below[0]
        frac = (profile[k - 1] - 0.5) / (profile[k - 1] - profile[k])
        return 2 * (k - 1 + frac) * step
    return width(ac[:, 0], z[1] - z[0]), width(ac[0, :], x[1] - x[0])


def find_wires(db, valid, z, x):
    from scipy.ndimage import maximum_filter, median_filter  # noqa: F401  (scipy is in cubdl)
    background = median_filter(db, size=(41, 25))
    peaks = (db == maximum_filter(db, size=(25, 15))) & (db - background >= 15) & valid
    peaks &= ((z >= TOP_MM + 2) & (z <= 45))[:, None]
    found = [(int(r), int(c)) for r, c in zip(*np.nonzero(peaks))]
    found.sort(key=lambda rc: rc[0])
    return found


def wire_widths(db, r, c, z, x, search=6):
    """Peak, lateral and axial -6 dB widths (mm) of the brightest point near (r, c)."""
    r0, r1, c0, c1 = max(0, r - search), r + search + 1, max(0, c - search), c + search + 1
    sub = db[r0:r1, c0:c1]
    rr, cc = np.unravel_index(np.argmax(sub), sub.shape)
    r, c = r0 + rr, c0 + cc
    amp = 10 ** (db / 20.0)
    def width(profile, centre, step):
        half = profile[centre] / 2.0
        left = centre
        while left > 0 and profile[left] > half:
            left -= 1
        right = centre
        while right < len(profile) - 1 and profile[right] > half:
            right += 1
        def cross(i, j):
            return i + (profile[i] - half) / (profile[i] - profile[j]) * (j - i) if profile[i] != profile[j] else i
        return (cross(right - 1, right) - cross(left + 1, left)) * step
    return float(db[r, c]), width(amp[r, :], c, x[1] - x[0]), width(amp[:, c], r, z[1] - z[0])


def analyse_block(block, frames):
    out = [f"== block {block}  (R' focus {FOCUS.get('Z2') if False else ''})".rstrip()]
    r_prime = frames.get("R'")
    st, db0, v0, z, x = r_prime
    expert = st.get("expert") or {}
    out[0] = f"== block {block}: R' = {expert.get('frequency_mhz')} step, focus {expert.get('focus_mm')} mm, depth {expert.get('depth_mm')} mm"
    # motion
    sel = ((z >= TOP_MM) & (z < 35))[:, None] & v0
    refs = [k for k in ("R'", "R1", "R2", "R3", "R4") if k in frames]
    pairs = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            a, b = frames[refs[i]][1], frames[refs[j]][1]
            pairs.append(f"{refs[i]}~{refs[j]} r {corr(a[sel], b[sel]):.3f} d {a[sel].mean() - b[sel].mean():+.2f} dB")
    out.append("   motion (10-35 mm): " + "; ".join(pairs))
    # frequency
    out.append("   frequency ladder (focus of R'):")
    out.append("     step   " + " ".join(f"{lo:4.0f}+" for lo, _ in band_means(db0, v0, z, end=55)) +
               "   noise  penetr  slope10-30  speckle ax/lat mm (10-20)")
    for step in FREQ:
        if step not in frames:
            continue
        _st, db, v, zz, xx = frames[step]
        mhz = step_mhz(frames, step)
        bands = band_means(db, v, zz, end=55)
        noise = float(db[v & (zz >= 55)[:, None]].mean())
        pen = None
        for lo, m in bands:
            if m - noise >= 6:
                pen = lo + BAND
            else:
                break
        fit_sel = v & ((zz >= 10) & (zz < 30))[:, None]
        zz2 = np.broadcast_to(zz[:, None], db.shape)[fit_sel]
        slope = float(np.polyfit(zz2, db[fit_sel], 1)[0])
        ax, lat = speckle_width(db, v, zz, xx, 10, 20)
        out.append(f"     {mhz:4.1f}   " + " ".join(f"{m:5.1f}" for _lo, m in bands) +
                   f"   {noise:5.1f}  {('%3.0f mm' % pen) if pen else '  none'}  {slope:+8.3f}    {ax:.3f} / {lat:.3f}")
    # focus
    zsteps = [s for s in FOCUS if s in frames]
    if zsteps:
        out.append("   focus ladder (frequency of R'): band mean dB minus the mean over the 5 focus steps; * = brightest")
        table = {s: band_means(frames[s][1], frames[s][2], frames[s][3], end=50) for s in zsteps}
        los = [lo for lo, _ in table[zsteps[0]]]
        out.append("     focus  " + " ".join(f"{lo:5.0f}+" for lo in los) + "   lateral speckle mm 10-20 / 20-30 / 30-40")
        mean = np.mean([[m for _lo, m in table[s]] for s in zsteps], axis=0)
        best = np.argmax([[m for _lo, m in table[s]] for s in zsteps], axis=0)
        for k, s in enumerate(zsteps):
            cells = []
            for i, (_lo, m) in enumerate(table[s]):
                cells.append(f"{m - mean[i]:+5.1f}{'*' if best[i] == k else ' '}")
            _st, db, v, zz, xx = frames[s]
            lats = [speckle_width(db, v, zz, xx, a, a + 10)[1] for a in (10, 20, 30)]
            out.append(f"     {FOCUS[s]:3d} mm " + " ".join(cells) + "   " + " / ".join(f"{w:.3f}" for w in lats))
        out.append("     brightest focus per band: " + ", ".join(f"{lo:.0f}-{lo + BAND:.0f} mm -> {FOCUS[zsteps[b]]}"
                                                         for lo, b in zip(los, best)))
    # wires
    wires = find_wires(db0, v0, z, x)
    if wires:
        out.append(f"   wires found on R': {len(wires)} at " + ", ".join(f"({x[c]:+.1f} mm, {z[r]:.1f} mm)" for r, c in wires))
        freq_labels = {s: step_mhz(frames, s) for s in FREQ}
        for label, steps in (("frequency", freq_labels), ("focus", FOCUS)):
            out.append(f"   wires vs {label}: lateral / axial -6 dB width (mm), peak dB")
            for r, c in wires:
                cells = []
                for s, v in steps.items():
                    if s in frames:
                        pk, lat, ax = wire_widths(frames[s][1], r, c, frames[s][3], frames[s][4])
                        cells.append(f"{v:>4}: {lat:.2f}/{ax:.2f} {pk:4.1f}")
                out.append(f"     wire at {z[r]:4.1f} mm, x {x[c]:+5.1f}:  " + "  ".join(cells))
    else:
        out.append("   no wires on R' (uniform region)")
    return out


def main():
    if len(sys.argv) != 2:
        print("usage: python tests/measure_wear_phantom_ladders.py <session directory>")
        return 2
    session = Path(sys.argv[1])
    blocks = defaultdict(dict)
    for d in sorted((session / "model_data").iterdir()):
        if not (d / "wear_model_meta.json").exists():
            continue
        st, db, valid, z, x = load(d)
        if not st or st.get("step") == "C":
            continue
        blocks[(st["view"], st["side"], st["block"])][st["step"]] = (st, db, valid, z, x)
    lines = [f"T6 ladders: {session.name}", ""]
    for key in sorted(blocks, key=lambda k: k[2]):
        lines += analyse_block(key[2], blocks[key]) + [""]
    text = "\n".join(lines) + "\n"
    Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
