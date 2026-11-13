"""体模 T5：在深度 X 真实采集的帧，与在 67.0 mm 采集后裁到 X 的帧是否相同（docs/wear_phantom_pilot_protocol.md §7）。

这是“深度标签靠离线裁剪”的前提：研究模式里只有 R′ 在专家深度实采，其余帧都在 67.0 mm 采集。

一、原始帧片段（pilot/<时间>_T5_<位置>/）：每个 X、每遍有三段——67a（之前 67 mm）、real（X 实采）、
67b（之后 67 mm）。每段先在对数域对全部帧取平均，67 mm 的两段裁到 real 的行数（两种长度的行几何必须一致）。
- 从第一个采样到 X 等分 8 段，每段的平均 dB：real − (67a+67b)/2，对照是 67a − 67b；
- 有信号的区域内，real 与 67 平均图的相关，对照是 67a 与 67b 的相关；
- 轴向错位：real 与 67 平均图在 ±4 行（每行 0.0616 mm）内的最大相关所在的行偏移。
二、导出对照（会话 model_data/ 里成对的 R′ 与 R1）：R′ 的 BC0 覆盖 0..X，R1 的覆盖 0..67，
把 R1 按深度插值到 R′ 的行上（第 i 行深度 i·末深/870），逐段比较 BC0 的平均 dB；对照是不同对之间的 R′ 与 R′。

通过（协议 §7.3）：每段差 ≤ 0.5 dB，且不大于对照的 3 倍；轴向错位 ≤ 0.1 mm；相关不低于对照减 0.02。
用法：python tests/measure_wear_phantom_depth_crop.py <T5 目录> [会话 model_data 目录]
报告（英文）写到本脚本旁的 measure_wear_phantom_depth_crop.txt，并打印同样内容。
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BANDS = 8
MAX_LAG_ROWS = 4
SIGNAL_CORR = 0.5
COUNTS_PER_DB = 400.0
POINTS, LINES = 870, 256


def burst_mean(json_path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(json_path.with_suffix(".npz"))
    frames = meta["frames"]
    db = np.stack([20.0 * np.log10(np.maximum(data[f["array"]].astype(np.float64), 1.0)) for f in frames])
    geometry = frames[0]["geometry"]
    # neighbouring-frame correlation per row block, to mark rows that carry signal
    return db.mean(axis=0), db, geometry


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def signal_rows(db_frames):
    """Rows whose 0.5 mm neighbourhood correlates between neighbouring frames (static phantom)."""
    n = db_frames.shape[1]
    keep = np.zeros(n, bool)
    for r0 in range(0, n, 8):
        sl = slice(r0, min(n, r0 + 8))
        c = np.median([corr(db_frames[i, sl], db_frames[i + 1, sl]) for i in range(min(10, db_frames.shape[0] - 1))])
        keep[sl] = c >= SIGNAL_CORR
    return keep


def best_lag(a, b, rows):
    best = (-2.0, 0)
    for lag in range(-MAX_LAG_ROWS, MAX_LAG_ROWS + 1):
        lo, hi = max(0, lag), min(rows, rows + lag)
        if hi - lo < 50:
            continue
        c = corr(a[lo:hi], b[lo - lag:hi - lag])
        best = max(best, (c, lag))
    return best[1]


def compare_bursts(directory):
    groups = defaultdict(dict)
    for path in directory.glob("T5_*.json"):
        m = re.match(r"T5_(?P<tag>.+?)_(?P<x>[0-9.]+)_r(?P<rep>\d+)_(?P<part>67a|67b|real)$", path.stem)
        if m:
            groups[(float(m["x"]), int(m["rep"]))][m["part"]] = path
    lines, rows_out = [], []
    for (x, rep), parts in sorted(groups.items()):
        if set(parts) != {"67a", "67b", "real"}:
            lines.append(f"== X {x:g} r{rep}: incomplete {sorted(parts)}")
            continue
        real, real_frames, g_real = burst_mean(parts["real"])
        a, a_frames, g_a = burst_mean(parts["67a"])
        b, b_frames, g_b = burst_mean(parts["67b"])
        n = real.shape[0]
        if abs(g_real[0] - g_a[0]) > 1e-6 or abs(g_real[1] - g_a[1]) > 1e-9:
            lines.append(f"== X {x:g} r{rep}: row geometry differs {g_real[:2]} vs {g_a[:2]}")
            continue
        a, b = a[:n], b[:n]
        full = 0.5 * (a + b)
        z = g_real[0] + np.arange(n) * g_real[1]
        sig = signal_rows(real_frames) & signal_rows(a_frames[:, :n])
        edges = np.linspace(z[0], x, BANDS + 1)
        diff, ctrl, has_sig = [], [], []
        for k in range(BANDS):
            sel = (z >= edges[k]) & (z < edges[k + 1] + (1e-9 if k == BANDS - 1 else 0))
            diff.append(float(real[sel].mean() - full[sel].mean()))
            ctrl.append(float(a[sel].mean() - b[sel].mean()))
            has_sig.append(bool(sig[sel].mean() > 0.5))
        c_real = corr(real[sig], full[sig]) if sig.any() else float("nan")
        c_ctrl = corr(a[sig], b[sig]) if sig.any() else float("nan")
        rows_sig = np.flatnonzero(sig)
        lag = best_lag(real[rows_sig[0]:rows_sig[-1] + 1], full[rows_sig[0]:rows_sig[-1] + 1], len(rows_sig)) if sig.any() else 0
        shift_mm = lag * g_real[1]
        worst = max(abs(d) for d in diff)
        worst_sig = max((abs(d) for d, s in zip(diff, has_sig) if s), default=float("nan"))
        # A band passes at <= 0.5 dB, or within 3x its own control when 67a and 67b already differ that much.
        ok = (all(abs(d) <= max(3 * abs(c), 0.5) for d, c in zip(diff, ctrl))
              and abs(shift_mm) <= 0.1 and (c_real >= c_ctrl - 0.02 if sig.any() else True))
        rows_out.append(dict(x=x, rep=rep, worst=worst, worst_sig=worst_sig, ctrl=max(abs(c) for c in ctrl),
                             c_real=c_real, c_ctrl=c_ctrl, shift=shift_mm, ok=ok,
                             mean_diff=float(np.mean(diff))))
        lines.append(f"== X {x:g} mm  r{rep}  rows {n}  frames real/67a/67b {real_frames.shape[0]}/{a_frames.shape[0]}/{b_frames.shape[0]}  "
                     f"signal to {z[rows_sig[-1]] if sig.any() else float('nan'):.1f} mm")
        lines.append("   band ends (mm): " + " ".join(f"{e:5.1f}" for e in edges[1:]))
        lines.append("   real - 67 (dB): " + " ".join(f"{d:+5.2f}" for d in diff))
        lines.append("   67a - 67b (dB): " + " ".join(f"{c:+5.2f}" for c in ctrl))
        lines.append("   signal band:    " + " ".join(f"{'  yes' if s else '   no'}" for s in has_sig))
        lines.append(f"   corr real~67 {c_real:.3f}  control 67a~67b {c_ctrl:.3f}  axial shift {shift_mm:+.3f} mm  "
                     f"-> {'PASS' if ok else 'FAIL'}")
    return lines, rows_out


def read_bc0(directory):
    raw = np.fromfile(directory / "Algo_BC0.bin", dtype=np.uint16)
    return raw.reshape(LINES, POINTS).T.astype(np.float64) / COUNTS_PER_DB


def compare_exports(model_dir):
    items = []
    for d in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        meta_path = d / "wear_model_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        step = (meta.get("acquisition_step") or {}).get("step")
        image = meta["image"]
        items.append((d.name, step, float(image["end_depth_mm"]), read_bc0(d),
                      float(image["x_min_mm"]), float(image["x_max_mm"])))
    pairs = []
    for i in range(len(items) - 1):
        if items[i][1] == "R'" and items[i + 1][1] == "R1":
            pairs.append((items[i], items[i + 1]))
    lines = [f"exported pairs found: {len(pairs)} (R' then R1)"]
    out = []
    rprimes = []
    for (n1, _s1, x, bc_r, xr0, xr1), (n2, _s2, full_end, bc_f, xf0, xf1) in pairs:
        # The exported width depends on the export depth (edge steering fans out with depth), so the
        # 256 columns of R' and R1 are at different x: map R1 onto R' by position on both axes.
        z_r = np.arange(POINTS) * x / POINTS
        z_f = np.arange(POINTS) * full_end / POINTS
        x_r = np.linspace(xr0, xr1, LINES)
        x_f = np.linspace(xf0, xf1, LINES)
        rows = np.stack([np.interp(z_r, z_f, bc_f[:, j]) for j in range(LINES)], axis=1)
        crop = np.stack([np.interp(x_r, x_f, rows[i, :]) for i in range(POINTS)], axis=0)
        lines.append(f"   widths: R' {xr1 - xr0:.2f} mm, R1 {xf1 - xf0:.2f} mm")
        valid = (bc_r > 0) & (crop > 0)
        edges = np.linspace(z_r[valid.any(axis=1)][0], x, BANDS + 1)
        diff = []
        for k in range(BANDS):
            sel = (z_r >= edges[k]) & (z_r <= edges[k + 1])
            m = valid & sel[:, None]
            diff.append(float(bc_r[m].mean() - crop[m].mean()))
        rprimes.append((n1, bc_r, valid))
        out.append(max(abs(d) for d in diff))
        lines.append(f"   R' {n1} ({x:g} mm) vs R1 {n2} cropped: " + " ".join(f"{d:+5.2f}" for d in diff)
                     + f"   (corr {corr(bc_r[valid], crop[valid]):.3f})")
    for (na, a, va), (nb, b, vb) in zip(rprimes, rprimes[1:]):
        m = va & vb
        lines.append(f"   control R' {na} vs R' {nb}: mean {float(a[m].mean() - b[m].mean()):+.2f} dB, corr {corr(a[m], b[m]):.3f}")
    return lines, out


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/measure_wear_phantom_depth_crop.py <T5 directory> [model_data directory]")
        return 2
    lines = ["T5 depth: real acquisition at X vs 67.0 mm acquisition cropped to X", ""]
    burst_lines, rows = compare_bursts(Path(sys.argv[1]))
    lines += burst_lines
    if len(sys.argv) > 2:
        lines += ["", "export path (BC0 of exported R' vs exported R1 cropped)"]
        export_lines, _ = compare_exports(Path(sys.argv[2]))
        lines += export_lines
    lines += ["", "summary (raw bursts)"]
    for x in sorted({r["x"] for r in rows}):
        rs = [r for r in rows if r["x"] == x]
        lines.append(f"   X {x:5.1f}: worst band |diff| {max(r['worst'] for r in rs):.2f} dB "
                     f"(signal bands {max(r['worst_sig'] for r in rs):.2f}), mean diff "
                     f"{np.mean([r['mean_diff'] for r in rs]):+.2f} dB, control {max(r['ctrl'] for r in rs):.2f} dB, "
                     f"corr {min(r['c_real'] for r in rs):.3f} vs control {min(r['c_ctrl'] for r in rs):.3f}, "
                     f"shift {max(abs(r['shift']) for r in rs):.3f} mm, {sum(r['ok'] for r in rs)}/{len(rs)} pass")
    text = "\n".join(lines) + "\n"
    Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
