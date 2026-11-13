# -*- coding: utf-8 -*-
"""人体颈动脉采集清单：按受试者分组，剔除不进训练的存储，写出 data/invivo_carotid_manifest.json。

颈动脉/原始数据 是扁平的，一次存储一个目录（17 位时间戳）。受试者只能由
Algo_PartitionInfo.pdt 里 FILEPATH 的场次号（如 20261009_142958_00001）还原。各工具原本按
<场次>/<采集>/Algo_BC0.bin 找场次，这里不动 E 盘上的文件，而是写一份清单，
bmode_opt/console_sessions.py 读它。

剔除规则（按顺序，先命中先算）：

  error       参数文件缺失或读不出场次号
  cine        电影回放导出（PageNum > 1）：Line / Point / PageSize 是逐页列表，
              hisense_loader.load_capture 不收
  preset      检查预设不是 Carotid（Thyroid、L-Ext V）
  probe       探头不是 L15-4（20261009_140742_00001 整个用的 L12-3S）
  short       剩下的单帧不到 MIN_FRAMES 张的场次：试机或中断，不算受试者

清单里每帧带上前端与后端设置（不读 BC0），场景族、阶梯、底噪「够不够深」都直接用它。

用法：python tools_build_invivo_manifest.py [--root E:/数据/hisense_medical/颈动脉/原始数据]
报告写在脚本旁边的 tools_build_invivo_manifest.txt。
"""

import argparse
import collections
import concurrent.futures
import io
import json
import os
import re
import sys
import time

sys.path.insert(0, "bmode_opt")

from hisense_loader import BC0_FILE, BE_PARAM_FILE, FE_PARAM_FILE, PARTITION_FILE, parse_pdt
import console_sessions as CS
import data_paths as DPATH

DEFAULT_ROOT = str(DPATH.CAROTID_RAW_ROOT)
REPORT_PATH = "tools_build_invivo_manifest.txt"
PRESET = "Carotid"
PROBE = "L15-4"
# 与 tests/measure_invivo_acquisition.py 的受试者门槛相同。
MIN_FRAMES = 20


def leaf(params, name):
    hits = [v for k, v in params.items() if k == name or k.endswith("." + name)]
    return hits[0] if hits else ""


def decode_chars(value):
    """ProbeNameActual / ExamMode 这类以逗号分隔的字符码。"""
    try:
        return "".join(chr(int(c)) for c in value.split(",") if c.strip() and int(c) > 0)
    except ValueError:
        return value


def scan(path):
    name = os.path.basename(path)
    try:
        part = parse_pdt(os.path.join(path, PARTITION_FILE))
        fe = parse_pdt(os.path.join(path, FE_PARAM_FILE))
        be = parse_pdt(os.path.join(path, BE_PARAM_FILE))
        if not os.path.exists(os.path.join(path, BC0_FILE)):
            raise FileNotFoundError(BC0_FILE)
    except Exception as exc:  # 缺文件也记下来，不中断整次扫描
        return {"name": name, "session": "", "error": "%s" % type(exc).__name__}
    match = re.search(r"/(\d{8}_\d{6}_\d{5})/", leaf(part, "FILEPATH"))
    row = {
        "name": name,
        "session": match.group(1) if match else "",
        "error": "" if match else "no session id in FILEPATH",
        "page_num": int(leaf(part, "PageNum").split(",")[0] or 1),
        "exam": decode_chars(leaf(fe, "ExamMode")),
        "probe": decode_chars(leaf(fe, "ProbeNameActual")),
    }
    if row["page_num"] > 1 or row["error"]:
        return row
    try:
        return dict(row, **settings(fe, be))
    except ValueError as exc:
        return dict(row, error="settings: %s" % exc)


def settings(fe, be):
    num_foci = int(leaf(fe, "BFocusNumValue") or 1)
    foci = [float(v) for v in leaf(fe, "BFocusArrayPos").split(",") if v.strip()][:num_foci]
    return {
        "image_mode": int(leaf(fe, "BImageMode")),
        "depth_mm": round(float(leaf(fe, "BDisplayDepth")), 4),
        # 与 calibration.capture_frequency 相同：主机按 0.1 MHz 报。
        "frequency_mhz": round(float(leaf(fe, "BFreqValue")), 1),
        "focus_mm": foci[0] if len(foci) == 1 else foci,
        "gain_level": int(leaf(be, "BUIGainLevel")),
        "dynamic_range_level": int(leaf(be, "UIDynamicRangeLevel")),
        "tgc_levels": [int(leaf(be, "BTgc%dLevel" % (i + 1))) for i in range(8)],
    }


FRAME_KEYS = ("name", "image_mode", "depth_mm", "frequency_mhz", "focus_mm", "gain_level",
              "dynamic_range_level", "tgc_levels")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--out", default=str(CS.INVIVO_MANIFEST))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    lines = []
    emit = lines.append

    dirs = sorted(os.path.join(args.root, d) for d in os.listdir(args.root))
    print("scanning %d capture dirs under %s ..." % (len(dirs), args.root.encode("ascii", "replace").decode()),
          flush=True)
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(scan, dirs, chunksize=32))

    excluded = []
    kept = collections.defaultdict(list)
    for row in rows:
        if row["error"]:
            reason = "error: " + row["error"]
        elif row["page_num"] > 1:
            reason = "cine"
        elif row["exam"] != PRESET:
            reason = "preset: " + row["exam"]
        elif row["probe"] != PROBE:
            reason = "probe: " + row["probe"]
        else:
            kept[row["session"]].append(row)
            continue
        excluded.append({"name": row["name"], "session": row["session"], "reason": reason})

    sessions, short = {}, {}
    for sid, frames in sorted(kept.items()):
        if len(frames) < MIN_FRAMES:
            short[sid] = len(frames)
            excluded.extend({"name": f["name"], "session": sid, "reason": "short session"}
                            for f in frames)
            continue
        sessions[sid] = [{k: f[k] for k in FRAME_KEYS} for f in sorted(frames, key=lambda f: f["name"])]

    all_sessions = sorted({r["session"] for r in rows if r["session"]})
    dropped = [s for s in all_sessions if s not in sessions]
    manifest = {
        "note": ("In-vivo carotid captures grouped by the session id in Algo_PartitionInfo.pdt "
                 "FILEPATH. Files stay where they are under root; each frame is root/name. "
                 "Excluded: cine exports, non-%s presets, probes other than %s, and sessions "
                 "with fewer than %d frames left." % (PRESET, PROBE, MIN_FRAMES)),
        "root": args.root.replace("\\", "/"),
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "sessions": sessions,
        "excluded": excluded,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=0)

    reasons = collections.Counter(e["reason"] for e in excluded)
    by_reason = collections.defaultdict(collections.Counter)
    for e in excluded:
        by_reason[e["reason"]][e["session"]] += 1
    frames = [f for v in sessions.values() for f in v]
    emit("=========== in-vivo carotid manifest ===========")
    emit("  capture dirs scanned        %d" % len(rows))
    emit("  console sessions            %d" % len(all_sessions))
    emit("  subjects kept               %d  (%d single frames)" % (len(sessions), len(frames)))
    emit("  frames per subject          min %d / median %d / max %d" % (
        min(map(len, sessions.values())), sorted(map(len, sessions.values()))[len(sessions) // 2],
        max(map(len, sessions.values()))))
    emit("")
    emit("  excluded captures by reason")
    for reason, n in reasons.most_common():
        emit("    %-28s %5d  in %d session(s)" % (reason, n, len(by_reason[reason])))
    for reason in sorted(by_reason):
        if reason == "cine":
            continue
        emit("    %s: %s" % (reason, ", ".join("%s(%d)" % kv for kv in sorted(by_reason[reason].items()))))
    emit("")
    emit("  sessions dropped entirely   %d" % len(dropped))
    for sid in dropped:
        emit("    %s  %s" % (sid, dict(collections.Counter(
            e["reason"] for e in excluded if e["session"] == sid))))
    emit("")
    emit("  kept frames: image mode %s" % dict(collections.Counter(f["image_mode"] for f in frames)))
    emit("  kept frames: depth mm   %s" % dict(sorted(collections.Counter(
        round(f["depth_mm"], 1) for f in frames).items())))
    emit("  kept frames: freq MHz   %s" % dict(sorted(collections.Counter(
        f["frequency_mhz"] for f in frames).items())))
    emit("  kept frames: focus mm   %s" % dict(sorted(collections.Counter(
        str(f["focus_mm"]) for f in frames).items())))
    emit("")
    emit("  wrote %s in %.0f s" % (args.out, time.time() - started))

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
