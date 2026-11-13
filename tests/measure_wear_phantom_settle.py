"""体模 T2：切档后第几帧才真正是新参数下采的（docs/wear_phantom_pilot_protocol.md §4.3）。

读 us_desktop_tools 自动采集写的 pilot/<时间>_T2/ 片段（每段 .npz + .json）。对每段：

1. 第 0 帧 = 帧记录里“设备已确认的值”第一次变成新档的那一帧（频率看发射序列首段、
   焦点看 TX_FOCUS_DEPTH_UM、深度看帧布局的 output_samples）。
2. 旧图 = 第 0 帧前第 2 帧及更早的平均；新图 = 第 +5 帧及以后的平均（dB 域，按 8 行块平均）。
3. 每帧按线分成 4 组，每组分别算与旧图、新图的相关；4 组都更像新图才算“已生效”，
   否则算未生效（帧内一半新一半旧也算未生效）。深度切换只比两种长度共有的行。
4. 对照：新图内各帧与新图（留一）相关的最小值，以及旧图与新图之间的相关。两者差得开，
   这次切换才“可判”。

结论：现行规则是“第 0 帧之后至少 3 帧且至少 350 毫秒”才允许保存。对每次切换算出第一张新图比第 0 帧
晚多少毫秒（负数表示设备比上位机记录还早），以及最早允许保存的时刻比它晚多少（余量）。余量全为正，规则成立。
可以给多个 T2 目录，一起汇总。

用法：python tests/measure_wear_phantom_settle.py <pilot/..._T2 目录>
报告（英文）写到本脚本旁的 measure_wear_phantom_settle.txt，并打印同样内容。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PID = {"TX_SEGMENT_0": "113", "TX_FOCUS_DEPTH_UM": "101", "RX_DEMOD": "84"}
ROW_BLOCK = 8
LINE_GROUPS = 4
NEW_FROM = 5          # new reference starts at frame +5
SETTLE_FRAMES_RULE = 3     # us_desktop_tools study_protocol since 2026-11-06
SETTLE_MIN_MS_RULE = 350.0


def known(frame):
    obs = frame["stamp"]["capture_observation"]
    merged = {}
    for source in ("readback_observation", "known_device_values"):
        for k, v in (obs.get(source) or {}).items():
            if v is not None:
                merged[str(k)] = v
    return merged, (obs.get("frame_layout") or {}).get("output_samples")


def config_key(frame, kind):
    values, samples = known(frame)
    if kind == "frequency":
        return (values.get(PID["TX_SEGMENT_0"]), values.get(PID["RX_DEMOD"]))
    if kind == "focus":
        return values.get(PID["TX_FOCUS_DEPTH_UM"])
    return samples


def to_db(env, rows):
    env = np.asarray(env[:rows], dtype=np.float64)
    db = 20.0 * np.log10(np.maximum(env, 1.0))
    n = (db.shape[0] // ROW_BLOCK) * ROW_BLOCK
    return db[:n].reshape(n // ROW_BLOCK, ROW_BLOCK, db.shape[1]).mean(axis=1)


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def groups(lines):
    edges = np.linspace(0, lines, LINE_GROUPS + 1).round().astype(int)
    return [slice(edges[i], edges[i + 1]) for i in range(LINE_GROUPS)]


def analyse(json_path):
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    kind = meta["details"]["kind"]
    frames = meta["frames"]
    data = np.load(json_path.with_suffix(".npz"))
    keys = [config_key(f, kind) for f in frames]
    k0 = next((i for i in range(1, len(keys)) if keys[i] != keys[0]), None)
    submitted = next(e["host_monotonic_ns"] for e in meta["events"] if e["event"] == "switch_submitted")
    times = [f["stamp"]["host_receive_monotonic_ns"] for f in frames]
    seqs = [f["stamp"]["frame_seq"] for f in frames]
    out = {"name": meta["name"], "kind": kind, "frames": len(frames), "k0": k0,
           "seq_gaps": int(sum(1 for a, b in zip(seqs, seqs[1:]) if b - a != 1))}
    if k0 is None or k0 < 3 or len(frames) - (k0 + NEW_FROM) < 3:
        out["error"] = "switch not found or too few frames on one side"
        return out
    rows_all = [f["rows"] for f in frames]
    common = min(rows_all)
    lines = frames[0]["lines"]
    db = [to_db(data[f["array"]], common) for f in frames]
    old_idx = list(range(0, k0 - 1))
    new_idx = list(range(k0 + NEW_FROM, len(frames)))
    old_ref = np.mean([db[i] for i in old_idx], axis=0)
    new_ref = np.mean([db[i] for i in new_idx], axis=0)
    gs = groups(lines)

    def loo(idx, i):
        rest = [db[j] for j in idx if j != i]
        return np.mean(rest, axis=0)

    cross = max(corr(old_ref[:, g], new_ref[:, g]) for g in gs)
    states = []
    for i in range(len(frames)):
        # A frame that is part of a reference is compared with that reference without itself.
        o_ref = loo(old_idx, i) if i in old_idx and len(old_idx) > 1 else old_ref
        n_ref = loo(new_idx, i) if i in new_idx and len(new_idx) > 1 else new_ref
        c_old = [corr(db[i][:, g], o_ref[:, g]) for g in gs]
        c_new = [corr(db[i][:, g], n_ref[:, g]) for g in gs]
        votes = sum(1 for a, b in zip(c_old, c_new) if b > a)
        state = "N" if votes == LINE_GROUPS else ("O" if votes == 0 else "M")
        states.append((i - k0, state, rows_all[i], min(c_old), min(c_new), (times[i] - submitted) / 1e6))
    # Separable: every steady old frame reads O and every steady new frame reads N (leave one out).
    # A depth switch is never separable by image (the shared rows show the same phantom); there the
    # frame length itself, which comes from the device frame, marks the switch.
    separable = (kind != "depth" and all(states[i][1] == "O" for i in old_idx)
                 and all(states[i][1] == "N" for i in new_idx))
    out.update(cross=cross, separable=separable, level_change_db=float(np.median(new_ref - old_ref)))
    out["states"] = states
    first = next((j for j in range(len(states)) if all(t[1] == "N" for t in states[j:])), None)
    if kind == "depth":
        first = k0 if rows_all[k0] != rows_all[k0 - 1] else None
    out["first_new_index"] = first
    if first is not None:
        # In received frames (frame_seq), which is what the SETTLE_FRAMES rule counts: a frame may be
        # saved once frame_seq - seq(frame 0) >= SETTLE_FRAMES.  lag <= 0: the device was there first.
        out["lag_seq"] = int(seqs[first] - seqs[k0])
        out["n_unsettled"] = max(0, first - k0)
        out["latency_ms"] = (times[first] - submitted) / 1e6
        out["delay_ms"] = (times[first] - times[k0]) / 1e6
    else:
        out["lag_seq"] = out["n_unsettled"] = None
    out["early_new"] = sum(1 for s in states if s[0] < 0 and s[1] == "N")
    out["frame_interval_ms"] = float(np.median(np.diff(times)) / 1e6)
    out["receive_interval_ms"] = float((times[-1] - times[0]) / max(1, seqs[-1] - seqs[0]) / 1e6)
    out["k0_latency_ms"] = (times[k0] - submitted) / 1e6
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: python tests/measure_wear_phantom_settle.py <pilot T2 directory> [more T2 directories]")
        return 2
    lines = [f"T2 settle analysis; rule: save at >= {SETTLE_FRAMES_RULE} received frames AND >= {SETTLE_MIN_MS_RULE:g} ms after frame 0",
             "frame 0 = first frame whose known (host-recorded) values are new; state N/O/M = new/old/mixed by line groups",
             "delay = first frame from which every frame is new, minus frame 0 (ms; negative = device ahead of the host record)",
             "margin = earliest allowed save (max of N frames, min ms) minus that frame (ms); must be > 0", ""]
    summary = []
    for directory in map(Path, sys.argv[1:]):
        results = [analyse(p) for p in sorted(directory.glob("T2_*.json"))]
        lines.append(f"######## {directory}")
        for r in results:
            lines.append(f"== {r['name']}  shown frames {r['frames']}  frame-seq gaps {r['seq_gaps']}")
            if "error" in r:
                lines.append(f"   ERROR: {r['error']}")
                continue
            lines.append(f"   separable {r['separable']} (old-vs-new reference corr {r['cross']:.3f}); "
                         f"median level change {r['level_change_db']:+.2f} dB")
            lines.append(f"   received every {r['receive_interval_ms']:.0f} ms; frame 0 arrives {r['k0_latency_ms']:.0f} ms after submission")
            seq = "  ".join(f"{k:+d}:{s}" + (f"({rows})" if r["kind"] == "depth" else "")
                            for k, s, rows, _co, _cn, _t in r["states"] if -3 <= k <= 4)
            lines.append(f"   k-3..+4: {seq}")
            if r["lag_seq"] is None:
                continue
            allowed = max(SETTLE_FRAMES_RULE * r["receive_interval_ms"], SETTLE_MIN_MS_RULE)
            r["margin_ms"] = allowed - r["delay_ms"]
            lines.append(f"   first all-new frame: lag {r['lag_seq']} frames, delay {r['delay_ms']:.0f} ms "
                         f"({r['latency_ms']:.0f} ms after submission); margin {r['margin_ms']:.0f} ms")
            if r["separable"] or r["kind"] == "depth":
                summary.append((directory.name, r))
        lines.append("")
    lines.append("summary (separable frequency/focus switches, and depth judged by frame length)")
    for kind in ("frequency", "focus", "depth"):
        rs = [r for _d, r in summary if r["kind"] == kind]
        if rs:
            lines.append(f"   {kind:9s} n={len(rs):2d}  lag {min(r['lag_seq'] for r in rs)}..{max(r['lag_seq'] for r in rs)} frames  "
                         f"delay {min(r['delay_ms'] for r in rs):.0f}..{max(r['delay_ms'] for r in rs):.0f} ms  "
                         f"device switch {min(r['latency_ms'] for r in rs):.0f}..{max(r['latency_ms'] for r in rs):.0f} ms after submission  "
                         f"min margin {min(r['margin_ms'] for r in rs):.0f} ms")
    worst = min((r["margin_ms"] for _d, r in summary), default=None)
    lines.append("   verdict: " + ("no usable switch" if worst is None else
                                     f"rule holds, smallest margin {worst:.0f} ms" if worst > 0 else
                                     f"rule FAILS, margin {worst:.0f} ms"))
    text = "\n".join(lines) + "\n"
    Path(__file__).with_suffix(".txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
