"""体模 T3：同一设置下帧间的重复性（docs/wear_phantom_pilot_protocol.md §5）。

读 us_desktop_tools 自动采集写的 pilot/<时间>_T3_<位置>/ 片段（.npz + .json），每段约 5 秒。
后面判断“有区别”都以这里量出的重复性为尺子（差别要大于它的 3 倍）。

每个 5 mm 深度段：
- 每帧在该段的平均 dB（对数域平均，全部线）→ 跨帧的标准差、极差、线性漂移（dB/秒）。
  不用中位数：包络值只有几个计数时，中位数只能取 20·log10(整数)，一跳就是 1–2 dB，统计不连续；
- 相邻两帧的 dB 图相关（体模不动时：有信号接近 1，纯噪声接近 0），用来标出“有信号”的段；
- 每个像素跨帧的 dB 标准差的中位数（斑点本身的帧间抖动）。

用法：python tests/measure_wear_phantom_stability.py <T3 目录> [更多 T3 目录]
报告（英文）写到本脚本旁的 measure_wear_phantom_stability.txt，并打印同样内容。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

BAND_MM = 5.0
SIGNAL_CORR = 0.5


def load_burst(json_path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(json_path.with_suffix(".npz"))
    frames = meta["frames"]
    rows = {f["rows"] for f in frames}
    if len(rows) != 1:
        raise ValueError(f"{json_path.name}: frame length changes inside the burst {sorted(rows)}")
    db = np.stack([20.0 * np.log10(np.maximum(data[f["array"]].astype(np.float64), 1.0)) for f in frames])
    env = np.stack([data[f["array"]] for f in frames])
    g = frames[0]["geometry"]
    z = g[0] + np.arange(db.shape[1]) * g[1]
    t = np.array([f["stamp"]["host_receive_monotonic_ns"] for f in frames], dtype=np.float64) / 1e9
    seq = [f["stamp"]["frame_seq"] for f in frames]
    return meta, db, env, z, t - t[0], seq


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def analyse(json_path):
    meta, db, env, z, t, seq = load_burst(json_path)
    out = [f"== {meta['name']}  frames {db.shape[0]}  rows {db.shape[1]}  lines {db.shape[2]}  "
           f"span {t[-1]:.2f} s  frame-seq gaps {sum(1 for a, b in zip(seq, seq[1:]) if b - a != 1)}",
           f"   samples <= 2 counts: {np.mean(env <= 2):.2f}   max envelope {int(env.max())}",
           "   band(mm)     mean dB  frame-to-frame std  range   drift dB/s  neighbour corr  pixel std(dB)  signal"]
    bands = []
    start = np.floor(z[0] / BAND_MM) * BAND_MM
    for lo in np.arange(start, z[-1], BAND_MM):
        sel = (z >= lo) & (z < lo + BAND_MM)
        if sel.sum() < 10:
            continue
        per_frame = db[:, sel, :].reshape(db.shape[0], -1).mean(axis=1)
        slope = float(np.polyfit(t, per_frame, 1)[0]) if len(t) > 2 else float("nan")
        nb = float(np.median([corr(db[i, sel], db[i + 1, sel]) for i in range(db.shape[0] - 1)]))
        pix = float(np.median(db[:, sel, :].std(axis=0)))
        signal = nb >= SIGNAL_CORR
        bands.append(dict(lo=lo, median=float(np.mean(per_frame)), std=float(per_frame.std(ddof=1)),
                          range=float(np.ptp(per_frame)), slope=slope, corr=nb, pix=pix, signal=signal))
        out.append(f"   {lo:4.0f}-{lo + BAND_MM:<4.0f}  {np.mean(per_frame):8.2f}  {per_frame.std(ddof=1):17.3f}  "
                   f"{np.ptp(per_frame):6.3f}  {slope:+10.4f}  {nb:14.2f}  {pix:12.2f}   {'yes' if signal else 'no'}")
    sig = [b for b in bands if b["signal"]]
    if sig:
        whole = db[:, (z >= sig[0]["lo"]) & (z < sig[-1]["lo"] + BAND_MM), :].reshape(db.shape[0], -1).mean(axis=1)
        out.append(f"   signal bands {sig[0]['lo']:.0f}-{sig[-1]['lo'] + BAND_MM:.0f} mm: largest band std {max(b['std'] for b in sig):.3f} dB, "
                   f"largest range {max(b['range'] for b in sig):.3f} dB; whole signal region std {whole.std(ddof=1):.3f} dB, "
                   f"drift {np.polyfit(t, whole, 1)[0]:+.4f} dB/s")
    return out, bands


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/measure_wear_phantom_stability.py <T3 directory> [more]")
        return 2
    lines = ["T3 stability: per 5 mm depth band, statistics of the per-frame mean dB (log-domain mean) over one burst",
             "signal = neighbouring frames correlate >= 0.5 (static phantom); noise bands are listed but not used", ""]
    summary = []
    for directory in map(Path, sys.argv[1:]):
        for path in sorted(directory.glob("*.json")):
            if path.name == "report.json":
                continue
            text, bands = analyse(path)
            lines += text + [""]
            sig = [b for b in bands if b["signal"]]
            if sig:
                summary.append((path.stem, max(b["std"] for b in sig), max(b["range"] for b in sig)))
    lines.append("summary (signal bands only)")
    for name, std, rng in summary:
        lines.append(f"   {name}: largest band std {std:.3f} dB, largest band range {rng:.3f} dB")
    if summary:
        ref = max(s for _n, s, _r in summary)
        lines.append(f"   repeatability yardstick (largest std): {ref:.3f} dB -> a difference counts if > 3x = {3 * ref:.2f} dB")
    text = "\n".join(lines) + "\n"
    Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
