"""新板颈动脉研究模式：一个人体会话的采集核对与梯级分析（采集规范 docs/wear_carotid_acquisition_protocol.md v0.4）。

读会话的 model_data/（“导出数据”的目录，按 acquisition_step 分块、分步）。BC0 每 dB 400 计数，
dB = 20·log10(包络原始计数)，所以 0 dB 就是原始计数 ≤ 1、6.02 dB 就是 ≤ 2（量化地板）。

扫描转换后图像两侧是扇形以外的空白（边缘偏转），这些像素也是 0，所以一律只在扇形内统计：
|x| <= 31.5 * 阵元间距 + z * tan(边缘偏转)，且 z 不浅于第一个采样点。

报告：
1. 每块的步骤核对：频率档、焦点、导出深度、实际采集末深、切档后帧数 / 毫秒、成像模式与锁定设置（PI.json）、两次保存的间隔。
2. 信号电平与量化：每 5 mm 段的扇形内中位 dB、原始计数 ≤ 1 和 ≤ 2 的比例。
3. 运动：R′、R1–R4 两两在 8–35 mm 的归一化相关与平均电平差；电影回放相邻帧的相关（心动、呼吸的基线）。
4. 频率（F1–F5）、聚焦（Z1–Z5）：每段电平、噪声段电平（50–67 mm）、斑点轴向 / 侧向相关长度。
5. 深度（D1–D5 实采）对 R3、R4 裁剪到同一深度：逐段电平差与相关；R3~R4 作为运动基线。

电平差一律按能量平均（10·log10 平均能量之比）：各深度档导出时每个像素平均的原始样本数不同，
直接平均 dB 值会差出约 1 dB 的统计偏差，能量平均不受影响。

用法：python tests/measure_wear_carotid_session.py <会话目录>
报告（英文）写到本脚本旁的 measure_wear_carotid_session.txt，并打印同样内容。
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

COUNTS_PER_DB = 400.0
POINTS, LINES = 870, 256
BAND = 5.0
DB_ONE_COUNT = 0.0
DB_TWO_COUNTS = 20 * math.log10(2.0)

VIEW_NAMES = {"颈总动脉横切": "CCA transverse", "颈总动脉纵切": "CCA longitudinal", "分叉部横切": "bulb transverse",
              "分叉部纵切": "bulb longitudinal", "颈内动脉近段纵切": "proximal ICA longitudinal",
              "椎动脉椎间段纵切": "vertebral longitudinal", "斑块纵切": "plaque longitudinal",
              "斑块横切": "plaque transverse", "甲状腺横切": "thyroid transverse"}
SIDE_NAMES = {"左": "left", "右": "right"}

# config_json/PI.json values the study locks (parameter id -> (name, value)).
LOCKED = {83: ("IMAGING_MODE", 2), 102: ("EDGE_STEERING_MDEG", 4000), 103: ("RX_F_NUMBER_X1000", 2100),
          16: ("HV_CODE", 110), 32: ("AFE_LNA_CODE", 1), 33: ("AFE_PGA_CODE", 0), 34: ("AFE_CHANNEL_GAIN_CODE", 0),
          48: ("TGC_START_CODE", 50), 49: ("TGC_STOP_CODE", 130), 50: ("TGC_POS_STEP_CODE", 3),
          51: ("TGC_POS_FREQ_CODE", 1), 54: ("TGC_SLOPE_DELAY_CYCLES", 800), 64: ("FIR_PROFILE_ID", 1),
          65: ("RX_APODIZATION_PROFILE", 1), 2: ("LINE_COUNT", 96), 4: ("DOWNSAMPLE_FACTOR", 4)}


class Frame:
    def __init__(self, directory):
        self.dir = directory
        self.meta = json.loads((directory / "wear_model_meta.json").read_text(encoding="utf-8"))
        self.st = self.meta.get("acquisition_step") or {}   # absent in old exports (batch 0)
        image = self.meta["image"]
        bc = np.fromfile(directory / "Algo_BC0.bin", dtype=np.uint16).reshape(LINES, POINTS).T
        self.db = bc.astype(np.float64) / COUNTS_PER_DB
        self.end = float(image["end_depth_mm"])
        self.z = np.arange(POINTS) * self.end / POINTS
        self.x = np.linspace(float(image["x_min_mm"]), float(image["x_max_mm"]), LINES)
        source = self.meta["source"]
        known = {int(k): v for k, v in (source["capture_observation"].get("known_device_values") or {}).items()}
        self.known = known
        pitch = float(source["capture_observation"]["display_geometry"][2])
        theta = math.radians(known.get(102, 4000) / 1000.0)
        first = float(image["first_sample_depth_mm"])
        half = 31.5 * pitch + self.z * math.tan(theta)
        self.fan = (np.abs(self.x)[None, :] <= half[:, None] - 0.5 * (self.x[1] - self.x[0])) & (self.z >= first)[:, None]
        self.seq = source.get("frame_seq")
        self.epoch = (source.get("receiver_id"), source.get("receiver_epoch"))
        self.t = source.get("host_receive_monotonic_ns")

    @property
    def step(self):
        return self.st.get("step")


def band_rows(z, lo, hi):
    return (z >= lo) & (z < hi)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def speckle_width(f, lo, hi):
    """-6 dB widths (mm) of the dB speckle autocorrelation in the fan's full-width rectangle of lo..hi."""
    rows = band_rows(f.z, lo, hi)
    cols = f.fan[rows].all(axis=0)
    if rows.sum() < 8 or cols.sum() < 8:
        return float("nan"), float("nan")
    w = f.db[rows][:, cols]
    w = w - w.mean()
    spec = np.fft.fft2(w, s=(2 * w.shape[0], 2 * w.shape[1]))
    ac = np.real(np.fft.ifft2(spec * np.conj(spec)))
    if ac[0, 0] <= 0:
        return float("nan"), float("nan")
    ac /= ac[0, 0]

    def width(profile, step):
        below = np.flatnonzero(profile < 0.5)
        if not below.size:
            return float("nan")
        k = below[0]
        frac = (profile[k - 1] - 0.5) / (profile[k - 1] - profile[k])
        return 2 * (k - 1 + frac) * step
    return width(ac[:, 0], f.z[1] - f.z[0]), width(ac[0, :], f.x[1] - f.x[0])


def bands(f, top=6.0, end=None):
    end = min(end or f.end, f.end)
    out = []
    for lo in np.arange(top, end - 1e-9, BAND):
        sel = f.fan & band_rows(f.z, lo, lo + BAND)[:, None]
        v = f.db[sel]
        out.append((lo, float(np.median(v)) if v.size else float("nan"), float(v.mean()) if v.size else float("nan"),
                    float((v <= DB_ONE_COUNT).mean()) if v.size else float("nan"),
                    float((v <= DB_TWO_COUNTS + 1e-9).mean()) if v.size else float("nan")))
    return out


def resample_to(src, dst):
    """src's dB image on dst's pixel grid (rows by depth, columns by x), with the joint fan mask."""
    rows = np.interp(dst.z, src.z, np.arange(POINTS))
    cols = np.interp(dst.x, src.x, np.arange(LINES), left=-1, right=-1)
    r0 = np.clip(np.floor(rows).astype(int), 0, POINTS - 2)
    fr = rows - r0
    out = np.zeros_like(dst.db)
    mask = np.zeros_like(dst.fan)
    ok = cols >= 0
    c0 = np.clip(np.floor(cols).astype(int), 0, LINES - 2)
    fc = cols - c0
    # intensity-domain bilinear interpolation
    amp = 10 ** (src.db / 20.0)
    a = (amp[r0][:, c0] * (1 - fc) + amp[r0][:, c0 + 1] * fc)
    b = (amp[r0 + 1][:, c0] * (1 - fc) + amp[r0 + 1][:, c0 + 1] * fc)
    val = a * (1 - fr)[:, None] + b * fr[:, None]
    out[:, ok] = 20 * np.log10(np.maximum(val[:, ok], 1.0))
    fan = src.fan[r0][:, c0] & src.fan[r0 + 1][:, c0 + 1]
    mask[:, ok] = fan[:, ok]
    return out, mask & dst.fan


def compare(a, b, lo, hi, *, crop_from=None):
    """Mean dB difference and correlation of a vs b over lo..hi mm inside both fans.

    With ``crop_from`` b is resampled onto a's grid first (b deeper than a)."""
    if crop_from is not None:
        bdb, mask = resample_to(b, a)
    else:
        bdb, mask = b.db, a.fan & b.fan
    sel = mask & band_rows(a.z, lo, hi)[:, None]
    if sel.sum() < 50:
        return float("nan"), float("nan")
    # Level difference of the mean *energy*: bin means and interpolation (the export's resampling, and the
    # crop here) keep it, while the mean of dB values shifts with how many speckle samples each pixel averages.
    ea = float(np.mean(10 ** (a.db[sel] / 10.0)))
    eb = float(np.mean(10 ** (bdb[sel] / 10.0)))
    return 10 * math.log10(ea / eb), corr(a.db[sel], bdb[sel])


def fmt(v, spec="5.1f"):
    return "  nan" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def analyse_block(key, frames, cine, out):
    view, side, block = key
    out.append(f"== block {block}: {VIEW_NAMES.get(view, 'view?')} / {SIDE_NAMES.get(side, 'side?')}")
    r = frames.get("R'")
    if r is not None:
        ex = r.st.get("expert") or {}
        out.append(f"   R' (expert): depth {ex.get('depth_mm')} mm, step {ex.get('frequency_mhz')}, focus {ex.get('focus_mm')} mm, "
                   f"gain {ex.get('display_gain_db')} dB, compensation {'on' if ex.get('compensation_enabled') else 'off'} "
                   f"{ex.get('compensation_gains_db') if ex.get('compensation_enabled') else ''}")
    # 1. step check
    out.append("   step   MHz  focus  export  acquired  settled        mode  locked  dt_s  seq")
    order = [s for s in STEP_ORDER if s in frames]
    previous = None
    for s in order:
        f = frames[s]
        st = f.st
        bad = [name for pid, (name, want) in LOCKED.items() if f.known.get(pid) is not None and int(f.known[pid]) != want]
        missing = [name for pid, (name, _w) in LOCKED.items() if f.known.get(pid) is None]
        dt = (f.t - previous.t) / 1e9 if previous is not None and f.t and previous.t else float("nan")
        out.append(f"   {s:<5} {fmt(st.get('frequency_step_mhz'), '4.1f')}  {fmt(st.get('focus_step_mm'), '4.0f')}  "
                   f"{st.get('export_depth_mm'):6.1f}  {st.get('acquired_end_depth_mm'):7.2f}  "
                   f"{st.get('settled_frames'):3d} {st.get('settled_ms') or 0:6.0f} ms  {f.known.get(83)}  "
                   f"{'ok' if not bad and not missing else 'BAD ' + ','.join(bad + missing)}  {fmt(dt, '4.1f')}  {f.seq}")
        previous = f
    if cine:
        seqs = [c.seq for c in cine]
        span = (cine[-1].t - cine[0].t) / 1e9 if cine[-1].t and cine[0].t else float("nan")
        rate = (len(cine) - 1) / span if span and span > 0 else float("nan")
        out.append(f"   C: {len(cine)} frames, seq {seqs[0]}..{seqs[-1]} ({'consecutive' if seqs == list(range(seqs[0], seqs[0] + len(seqs))) else 'gaps'}), "
                   f"{span:.2f} s -> {rate:.2f} frames/s")
    if r is None:
        return
    # 2. level and quantisation on R'
    out.append("   R' level by 5 mm band inside the fan: median dB | share <=1 count | share <=2 counts")
    for lo, med, _mean, z1, z2 in bands(r):
        out.append(f"     {lo:4.0f}-{lo + BAND:<4.0f} {med:6.1f} dB   {z1 * 100:5.1f} %   {z2 * 100:5.1f} %")
    # 3. motion
    refs = [s for s in ("R'", "R1", "R2", "R3", "R4") if s in frames and frames[s].end == r.end]
    parts = []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            d, c = compare(frames[refs[i]], frames[refs[j]], 8, 35)
            parts.append(f"{refs[i]}~{refs[j]} r {c:.3f} d {d:+.2f}")
    out.append("   motion 8-35 mm: " + "; ".join(parts))
    if len(cine) > 1:
        cc = [compare(cine[k], cine[k + 1], 8, 35)[1] for k in range(len(cine) - 1)]
        cr = [compare(c, frames["R4"], 8, 35)[1] for c in cine] if "R4" in frames else []
        out.append(f"   cine neighbours r: median {np.median(cc):.3f}, min {np.min(cc):.3f}"
                   + (f"; cine vs R4 r: median {np.median(cr):.3f}, min {np.min(cr):.3f}" if cr else ""))
        # Heart cycle: level of each cine frame against the cine median, per band (no probe change within ~1 s).
        for lo, hi in ((8, 25), (25, 35), (8, 35)):
            levels = []
            for c in cine:
                sel = c.fan & band_rows(c.z, lo, hi)[:, None]
                levels.append(10 * math.log10(float(np.mean(10 ** (c.db[sel] / 10.0)))))
            levels = np.asarray(levels) - np.median(levels)
            out.append(f"   cine level {lo}-{hi} mm vs its median (heart cycle): range {levels.min():+.2f} .. "
                       f"{levels.max():+.2f} dB, sd {levels.std():.2f} dB")
    # 4. frequency and focus ladders
    for label, steps, key_name in (("frequency (focus of R')", [f"F{k}" for k in range(1, 6)], "frequency_step_mhz"),
                                   ("focus (frequency of R')", [f"Z{k}" for k in range(1, 6)], "focus_step_mm")):
        present = [s for s in steps if s in frames]
        if not present:
            continue
        out.append(f"   {label}: band mean dB inside the fan; noise = 50-67 mm mean; speckle ax/lat -6 dB (mm)")
        lows = [lo for lo, *_ in bands(frames[present[0]], end=50)]
        out.append("     value " + " ".join(f"{lo:5.0f}+" for lo in lows) + "  noise  <=2cnt 8-35   speckle 8-20     20-35")
        for s in present:
            f = frames[s]
            b = bands(f, end=50)
            sel = f.fan & band_rows(f.z, 50, 67)[:, None]
            noise = float(f.db[sel].mean())
            sel2 = f.fan & band_rows(f.z, 8, 35)[:, None]
            q = float((f.db[sel2] <= DB_TWO_COUNTS + 1e-9).mean())
            w1 = speckle_width(f, 8, 20)
            w2 = speckle_width(f, 20, 35)
            out.append(f"     {f.st.get(key_name):5}  " + " ".join(f"{mean:6.1f}" for _lo, _med, mean, *_ in b) +
                       f"  {noise:5.1f}   {q * 100:4.0f} %    {fmt(w1[0], '.2f')}/{fmt(w1[1], '.2f')}  {fmt(w2[0], '.2f')}/{fmt(w2[1], '.2f')}")
    # 5. depth: D real vs R3 / R4 cropped
    dsteps = [s for s in (f"D{k}" for k in range(1, 6)) if s in frames]
    if dsteps:
        out.append("   depth: D (real) vs R3 / R4 cropped to the same depth, 8 mm to the D end (band means; r = correlation)")
        base = [compare(frames["R3"], frames["R4"], 8, 35)] if "R3" in frames and "R4" in frames else []
        if base:
            out.append(f"     R3 vs R4 (motion baseline, 8-35 mm): d {base[0][0]:+.2f} dB, r {base[0][1]:.3f}")
        for s in dsteps:
            f = frames[s]
            cells = []
            for ref in ("R3", "R4"):
                if ref in frames:
                    d, c = compare(f, frames[ref], 8, f.end, crop_from=True)
                    cells.append(f"vs {ref}: d {d:+.2f} dB r {c:.3f}")
            per_band = []
            if "R3" in frames:
                for lo in np.arange(8, f.end - 1e-9, BAND):
                    d, _c = compare(f, frames["R3"], lo, min(lo + BAND, f.end), crop_from=True)
                    per_band.append(f"{d:+.1f}")
            sel = f.fan & band_rows(f.z, 8, f.end)[:, None]
            out.append(f"     {s} {f.end:5.1f} mm  " + "; ".join(cells) +
                       f"; <=1 count {float((f.db[sel] <= 0).mean()) * 100:4.1f} %  bands vs R3 from 8 mm: " + " ".join(per_band))


STEP_ORDER = ("R'", "R1", "F1", "F2", "F3", "F4", "F5", "R2", "Z1", "Z2", "Z3", "Z4", "Z5", "R3",
              "D1", "D2", "D3", "D4", "D5", "R4")


def energy_db(f, lo, hi):
    sel = f.fan & band_rows(f.z, lo, hi)[:, None]
    return 10 * math.log10(float(np.mean(10 ** (f.db[sel] / 10.0)))) if sel.any() else float("nan")


def quantised_share(f, lo=10, hi=30):
    sel = f.fan & band_rows(f.z, lo, hi)[:, None]
    return float((f.db[sel] <= DB_TWO_COUNTS + 1e-9).mean()) if sel.any() else float("nan")


CONTACT_LIMIT = 0.35   # share of <=2-count pixels at 10-30 mm above which the probe was not coupled (in vivo PI tissue: 5-27 %)


def summarise_block(tag, key, frames, cine):
    """One summary row: contact, motion, ladder effects, depth agreement."""
    singles = [s for s in STEP_ORDER if s in frames]
    any_frame = frames[singles[0]]
    mode = any_frame.known.get(83)
    q = {s: quantised_share(frames[s]) for s in singles}
    lost = [s for s in singles if q[s] > CONTACT_LIMIT]
    refs = [s for s in ("R'", "R1", "R2", "R3", "R4") if s in frames and frames[s].end == FULL_MM and s not in lost]
    rs, ds = [], []
    for i in range(len(refs)):
        for j in range(i + 1, len(refs)):
            d, c = compare(frames[refs[i]], frames[refs[j]], 8, 35)
            rs.append(c)
            ds.append(abs(d))
    fsteps = [f"F{k}" for k in range(1, 6) if f"F{k}" in frames]
    f_gap = (energy_db(frames[fsteps[0]], 8, 35) - energy_db(frames[fsteps[-1]], 8, 35)) if len(fsteps) == 5 else float("nan")
    z_near = (energy_db(frames["Z1"], 8, 15) - energy_db(frames["Z5"], 8, 15)) if "Z1" in frames and "Z5" in frames else float("nan")
    z_deep = (energy_db(frames["Z5"], 25, 35) - energy_db(frames["Z1"], 25, 35)) if "Z1" in frames and "Z5" in frames else float("nan")
    dd = {}
    for s in (f"D{k}" for k in range(1, 6)):
        if s in frames and "R3" in frames:
            dd[s] = compare(frames[s], frames["R3"], 8, frames[s].end, crop_from=True)[0]
    shallow = np.mean([dd[s] for s in ("D1", "D2", "D3") if s in dd]) if dd else float("nan")
    deep = np.mean([dd[s] for s in ("D4", "D5") if s in dd]) if dd else float("nan")
    ok = not lost and (min(rs) if rs else 0) >= 0.6
    return (f"{tag:<16} {key[2]:3d}  {mode}  {len(singles):2d}+{len(cine):2d}  "
            f"{max(q.values()) * 100:4.0f}%  {','.join(lost) if lost else '-':<14} "
            f"{(min(rs) if rs else float('nan')):5.2f} {(max(ds) if ds else float('nan')):5.2f}  "
            f"{f_gap:+5.1f}  {z_near:+5.1f} {z_deep:+5.1f}  {shallow:+5.2f} {deep:+5.2f}  {'ok' if ok else 'CHECK'}")


SUMMARY_HEAD = [
    "summary per block (mode 0 fundamental, 2 PI harmonic)",
    "  quant = worst single frame's share of <=2-count pixels at 10-30 mm; lost = frames above 35 % (probe not coupled)",
    "  r_min / |d|max = R frames pairwise at 8-35 mm (correlation, energy dB), lost frames left out",
    "  F lo-hi = energy 8-35 mm of the lowest minus the highest frequency step",
    "  Z near = focus 10 minus 30 at 8-15 mm; Z deep = focus 30 minus 10 at 25-35 mm (both > 0 = focus works)",
    "  D1-3 / D4-5 = mean energy dB of real-depth D frames minus R3 cropped to the same depth",
    "session          blk mode  n     quant lost           r_min |d|max F lo-hi  Znear Zdeep  D1-3  D4-5   verdict",
]
FULL_MM = 67.0


def load_session(session):
    blocks = defaultdict(dict)
    cines = defaultdict(list)
    for d in sorted((session / "model_data").iterdir()):
        if not (d / "wear_model_meta.json").exists():
            continue
        f = Frame(d)
        if not f.st:
            continue
        key = (f.st["view"], f.st["side"], f.st["block"])
        if f.step == "C":
            cines[key].append(f)
        else:
            if f.step in blocks[key]:
                blocks[key][f.step + "_dup"] = f
            blocks[key][f.step] = f
    return blocks, cines


def ascii_tag(name):
    parts = [p for p in name.split("_") if p.isascii()]
    return "_".join(parts[:2])[:16]


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/measure_wear_carotid_session.py <session directory | folder of sessions> ...")
        return 2
    sessions = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if (p / "model_data").exists():
            sessions.append(p)
        else:
            sessions += [s for s in sorted(p.iterdir()) if (s / "model_data").exists() and any((s / "model_data").iterdir())]
    if len(sessions) > 1:
        summary, details = list(SUMMARY_HEAD), []
        for session in sessions:
            blocks, cines = load_session(session)
            tag = ascii_tag(session.name)
            details.append(f"######## {tag} ({session.name.encode('ascii', 'replace').decode()})")
            for key in sorted(blocks, key=lambda k: k[2]):
                cine = sorted(cines.get(key, []), key=lambda c: c.st.get("cine_frame_index") or 0)
                summary.append(summarise_block(tag, key, blocks[key], cine))
                analyse_block(key, blocks[key], cine, details)
                details.append("")
        text = "\n".join(summary + [""] + details) + "\n"
        Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
        print(text)
        return 0
    session = sessions[0]
    blocks = defaultdict(dict)
    cines = defaultdict(list)
    for d in sorted((session / "model_data").iterdir()):
        if not (d / "wear_model_meta.json").exists():
            continue
        f = Frame(d)
        if not f.st:
            continue
        key = (f.st["view"], f.st["side"], f.st["block"])
        if f.step == "C":
            cines[key].append(f)
        else:
            if f.step in blocks[key]:
                blocks[key][f.step + "_dup"] = f
            blocks[key][f.step] = f
    lines = [f"session: {session.name.encode('ascii', 'replace').decode()}",
             f"blocks: {len(blocks)}; exports: {sum(len(v) for v in blocks.values())} single + "
             f"{sum(len(v) for v in cines.values())} cine",
             "dB = 20 log10(raw envelope counts): 0 dB is <= 1 count, 6.0 dB is 2 counts (quantisation floor)", ""]
    for key in sorted(blocks, key=lambda k: k[2]):
        frames = blocks[key]
        dups = [s for s in frames if s.endswith("_dup")]
        if dups:
            lines.append(f"   note: repeated steps (latest kept): {', '.join(d[:-4] for d in dups)}")
        cine = sorted(cines.get(key, []), key=lambda c: c.st.get("cine_frame_index") or 0)
        analyse_block(key, frames, cine, lines)
        lines.append("")
    text = "\n".join(lines) + "\n"
    Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
