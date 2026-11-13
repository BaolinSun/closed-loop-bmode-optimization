"""体模 T8：前端接收增益（AFE 通道增益、硬件 TGC）能否把噪声抬出量化地板（docs/wear_phantom_pilot_protocol.md §8.4）。

读 us_desktop_tools 自动采集“体模 T8”写的 pilot/<时间>_T8_<设置名>/ 片段：每组设置 5 个频率档各一段。
每段（对数域按帧平均前先逐帧统计）：
- 每个 5 mm 深度段：平均 dB、包络 <= 2 个计数的采样比例、相邻两帧相关（体模不动：有信号接近 1）；
- 有信号的最深深度：从浅到深第一个相邻帧相关 < 0.5 的段之前；
- 噪声底：相邻帧相关 < 0.1 的段的平均 dB 与 <= 2 计数比例（比例高 = 量化地板，不是物理噪声）；
- 浅处是否削顶：与名为 base 的那组相比，5–10 mm 的电平增量明显小于 20–30 mm 的增量，说明浅处不再线性。

用法：python tests/measure_wear_phantom_gain.py [--base 设置名] <pilot 目录或若干 T8 目录>（默认以 base 为基准）
报告（英文）写到本脚本旁的 measure_wear_phantom_gain.txt，并打印同样内容。
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BAND_MM = 5.0
SIGNAL_CORR = 0.5
NOISE_CORR = 0.1


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def analyse(json_path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(json_path.with_suffix(".npz"))
    frames = meta["frames"]
    env = np.stack([data[f["array"]] for f in frames]).astype(np.float64)
    db = 20.0 * np.log10(np.maximum(env, 1.0))
    g = frames[0]["geometry"]
    z = g[0] + np.arange(db.shape[1]) * g[1]
    bands = []
    for lo in np.arange(np.floor(z[0] / BAND_MM) * BAND_MM, z[-1], BAND_MM):
        sel = (z >= lo) & (z < lo + BAND_MM)
        if sel.sum() < 10:
            continue
        c = float(np.median([corr(db[i, sel], db[i + 1, sel]) for i in range(min(10, db.shape[0] - 1))]))
        bands.append(dict(lo=float(lo), mean=float(db[:, sel].mean()), q=float(np.mean(env[:, sel] <= 2)), corr=c))
    deepest = None
    for b in bands:
        if b["corr"] < SIGNAL_CORR:
            break
        deepest = b["lo"] + BAND_MM
    noise = [b for b in bands if b["corr"] < NOISE_CORR]
    def level(lo, hi):
        sel = (z >= lo) & (z < hi)
        return float(db[:, sel].mean()) if sel.any() else float("nan")
    return dict(frequency=meta["details"].get("frequency"), gains=meta["details"].get("gains") or {},
                frames=len(frames), bands=bands, deepest=deepest,
                noise_db=float(np.mean([b["mean"] for b in noise])) if noise else float("nan"),
                noise_q=float(np.mean([b["q"] for b in noise])) if noise else float("nan"),
                q_15_40=float(np.mean(env[:, (z >= 15) & (z < 40)] <= 2)),
                near=level(5, 10), mid=level(20, 30), p999=float(np.percentile(env[:, z < 15], 99.9)),
                peak=float(env.max()))


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/measure_wear_phantom_gain.py <pilot directory or T8 directories>")
        return 2
    args = sys.argv[1:]
    base_tag = "base"
    if len(args) >= 2 and args[0] == "--base":
        base_tag, args = args[1], args[2:]
    paths = []
    for arg in map(Path, args):
        paths += sorted(arg.rglob("T8_*.json")) if arg.is_dir() else []
    runs = defaultdict(dict)
    for path in paths:
        m = re.match(r"T8_(?P<tag>.+)_(?P<f>[0-9.]+)$", path.stem)
        if m:
            runs[m["tag"]][float(m["f"])] = analyse(path)
    lines = ["T8 receive gain: per setting (tag) and frequency step", ""]
    for tag, by_f in runs.items():
        gains = next(iter(by_f.values()))["gains"]
        lines.append(f"== {tag}: " + "  ".join(f"{k}={v}" for k, v in gains.items()))
        lines.append("   step  deepest signal  <=2 counts 15-40mm  noise floor dB (<=2 share)  5-10mm dB  20-30mm dB  p99.9 <15mm  peak")
        for f in sorted(by_f):
            r = by_f[f]
            lines.append(f"   {f:4.1f}  {('%5.0f mm' % r['deepest']) if r['deepest'] else '   none':>14}  {r['q_15_40']:18.2f}  "
                         f"{r['noise_db']:8.2f} ({r['noise_q']:.2f})          {r['near']:9.2f}  {r['mid']:10.2f}  "
                         f"{r['p999']:11.0f}  {r['peak']:6.0f}")
        lines.append("")
    base = runs.get(base_tag)
    if base:
        lines.append(f"change against '{base_tag}' (dB): near = 5-10 mm, mid = 20-30 mm; near rising much less than mid hints at clipping")
        for tag, by_f in runs.items():
            if tag == base_tag:
                continue
            parts = []
            for f in sorted(by_f):
                if f in base:
                    dn = by_f[f]["near"] - base[f]["near"]
                    dm = by_f[f]["mid"] - base[f]["mid"]
                    flag = "  <- near compressed" if dm - dn > 3.0 else ""
                    parts.append(f"{f:g}: near {dn:+.1f} mid {dm:+.1f}{flag}")
            lines.append(f"   {tag}: " + "; ".join(parts))
        lines.append("")
    lines.append("choose: deepest signal at the highest steps (fundamental 8.0 / 11.4, harmonic 5.3 / 5.7) as large as possible, "
                 "<=2-count share low, near not compressed")
    text = "\n".join(lines) + "\n"
    Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
