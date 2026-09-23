# -*- coding: utf-8 -*-
"""一次新采集 -> 六参数调整建议：读一个海信采集目录，按实机模型说明该怎么调。

    用法

    python tools_suggest_six_param.py --capture data/hisense_medical/20260910/20260910143012345
    python tools_suggest_six_param.py --capture data/hisense_medical/20260910        # 整个场次

默认模型是 runs/console_final_v2/all_data 的 frontend.pt（前端三轴）+ last.pt（增益 / TGC），
即实机微调的最终模型；--frontend / --backend 指向别的检查点，--checkpoint 用单个检查点。

    与训练时同一条预处理

网络看到的必须与训练缓存里的一模一样，所以图像这一路直接调用
tools_build_console_training_cache.pool_rows / pool_lines，标量这一路照抄那个脚本写缓存时的取值：

    图像      BC0 / 本组 counts_per_db + 本组深度响应，按强度分块平均到 512 x 128
    曝光参考  本组 pivot_db
    底噪      本组底噪（floors_in_counts 按计数换算过）+ 深度响应，两端的值另作标量
    当前设置  显示深度、发射频率、发射聚焦、增益 dB、8 段 TGC 档位、动态范围 UI、成像模式

--self-check data/console_dl_cache 把这条预处理的结果与训练缓存逐帧对照（能对上的帧），
用来确认换了机器、换了标定文件之后，推理与训练看到的仍是同一幅图。

    标定从哪来

新场次在 console_calibration.json 里没有条目时，向同一成像模式、底噪实测的场次借
（--calibration-group 指定借谁），报告里标明借了谁。借标定直接影响 dB 刻度：谐波场次之间
counts_per_db 相差过 44%，借来的增益建议就差同样的比例。所以新场次采完应当先跑
tools_refit_calibration.py 定标，再用这里的建议；借用只是没有标定时的下策。

    档位

每根轴的可选档按成像模式取自 --labels（默认 data/labels_console.jsonl）：基波 5.0-11.4 MHz、
谐波 4.4-5.7 MHz，两套几乎不重叠，不按模式限制网络就会在另一种模式的频率里挑。聚焦的可选档
随显示深度走，取档位表里不超过当前显示深度的那些。当前档永远算可选。

    建议的执行顺序

与闭环仿真同一套协议：前端三轴有改动时先改前端、重新采一帧，再回来做后端修正——后端修正是在
当前这幅图上算出来的，前端一换图就变了。后端的死区（--stop-deadband-levels，默认 0.5 级）
与单步限幅（--max-gain-step-clicks，默认 80 级）也与闭环一致。

动态范围没有判据（gCNR 对单调变换不变，标签里 dr_determined 全为假），这根轴没有训练，
报告里只重复当前值。

终端输出与写在脚本旁边的 tools_suggest_six_param.txt 同内容（ASCII）；--json 另写一份
每帧一行的 jsonl，供上位机脚本直接读。
"""

import argparse
import io
import json
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np
import torch

import bmode_dl.constants as K
import tools_build_console_training_cache as CACHE
from bmode_dl.checkpoint import load_checkpoint, load_combined
from bmode_dl.dataset import InputBuilder
from bmode_dl.metrics import decode

REPORT_PATH = "tools_suggest_six_param.txt"
AXES = (("depth", "depth_mm", "mm"), ("frequency", "frequency_mhz", "MHz"), ("focus", "focus_mm", "mm"))
# 各轴「档位变大」在主机上的说法（下标 0 = 应当增大）
AXIS_WORDS = {"depth": ("deeper", "shallower"), "frequency": ("higher", "lower"),
              "focus": ("deeper", "shallower")}
DEFAULT_RUN = os.path.join("runs", "console_final_v2", "all_data")
# 主机增益档位的范围。实测采集里见过 59 - 229 级，闭环的累计限幅也按 255 级全程算
# （bmode_dl/closed_loop.py 的 max_gain_total_clicks）。
GAIN_MIN_LEVEL, GAIN_MAX_LEVEL = 0, 255


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Six-parameter suggestions for one console capture")
    p.add_argument("--capture", required=True,
                   help="a capture directory (holding Algo_BC0.bin) or a session directory of captures")
    p.add_argument("--frontend", default=os.path.join(DEFAULT_RUN, "frontend.pt"),
                   help="checkpoint used for depth / frequency / focus")
    p.add_argument("--backend", default=os.path.join(DEFAULT_RUN, "last.pt"),
                   help="checkpoint used for gain / TGC / dynamic range")
    p.add_argument("--checkpoint", default=None,
                   help="one checkpoint for all six axes, instead of --frontend + --backend")
    p.add_argument("--labels", default="data/labels_console.jsonl",
                   help="labels the per-mode ladders are read from; without it every ladder step is offered")
    p.add_argument("--calibration-group", default=None,
                   help="borrow this group's calibration, as session/mode (e.g. 20260910/1); "
                        "by default the capture's own group, or the newest group of the same mode")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stop-deadband-levels", type=float, default=0.5,
                   help="a gain or slider correction smaller than this many console clicks reads as 'keep'")
    p.add_argument("--max-gain-step-clicks", type=int, default=80,
                   help="largest gain correction suggested at once, in clicks (0 = no limit)")
    p.add_argument("--limit", type=int, default=None, help="only the first N captures of a session")
    p.add_argument("--json", default=None, help="also write one JSON object per capture to this file")
    p.add_argument("--self-check", default=None,
                   help="a training cache directory: compare this script's preprocessing against it, "
                        "frame by frame, for the captures that are in both")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------
#   采集与标定

def find_capture_dirs(path):
    """一个采集目录 -> [它自己]；一个场次目录 -> 里面的全部采集。"""
    from hisense_loader import BC0_FILE, find_captures
    if os.path.isfile(os.path.join(path, BC0_FILE)):
        return [path]
    return [str(p) for p in find_captures(path)]


def session_name(capture_dir):
    """采集目录相对 data/hisense_medical 的场次名（20260910、20260828/GEN 这样）。"""
    from hisense_loader import DEFAULT_DATA_DIR
    parent = os.path.abspath(os.path.join(capture_dir, os.pardir))
    try:
        rel = os.path.relpath(parent, str(DEFAULT_DATA_DIR))
    except ValueError:
        return os.path.basename(parent)
    return rel.replace(os.sep, "/")


def resolve_calibration(cal_by_group, session, mode, override=None):
    """返回 (entry, 用的是谁的标定, 是否借来的)。

    借用规则：同一成像模式、底噪实测过的场次里最新的一个（场次名按字典序，名字以日期起头）。
    没有实测底噪的就退回同模式任意一个；一个都没有直接报错，不猜。
    """
    if override:
        if "/" not in override:
            raise SystemExit("--calibration-group must look like session/mode, e.g. 20260910/1")
        name, mode_text = override.rsplit("/", 1)
        key = (name, int(mode_text))
        if key not in cal_by_group:
            raise SystemExit("no calibration for %s; known groups: %s"
                             % (override, ", ".join("%s/%d" % k for k in sorted(cal_by_group))))
        return cal_by_group[key], key, key != (session, mode)
    key = (session, mode)
    if key in cal_by_group:
        return cal_by_group[key], key, False
    donors = [k for k in cal_by_group if k[1] == mode and cal_by_group[k].get("floor_measured")]
    if not donors:
        donors = [k for k in cal_by_group if k[1] == mode]
    if not donors:
        raise SystemExit("no calibration group in imaging mode %d; run tools_refit_calibration.py first" % mode)
    key = sorted(donors)[-1]
    return cal_by_group[key], key, True


# ---------------------------------------------------------------------------------------
#   预处理（与 tools_build_console_training_cache.py 同一套）

def prepare_capture(capture, entry, rows, lines):
    """一帧 -> 网络输入需要的全部数组与当前设置。"""
    import calibration as CAL
    import hisense_backend_sim as S
    import tissue as T

    cal = entry["cal"]
    response = CAL.depth_response_for(capture, cal)
    if response is None:
        response = np.zeros(capture.bc0.shape[0])
    db = S.bc0_to_db(capture.bc0, cal.counts_per_db) + response[:, None]
    floor = float(entry["floor"]) + response
    mode = S.capture_image_mode(capture)
    focus = capture.focus_depths_mm
    if len(focus) != 1:
        raise ValueError("%s has %d transmit foci %s; the focus axis is only defined for one"
                         % (capture.name, len(focus), tuple(focus)))
    return {
        "frame_id": capture.name,
        "source_rows": int(db.shape[0]),
        "db": CACHE.pool_lines(CACHE.pool_rows(db, rows), lines).astype(np.float32),
        "floor": CACHE.pool_rows(floor, rows).astype(np.float32),
        "min_depth_mm": float(capture.geometry.min_depth_mm),
        "max_depth_mm": float(capture.geometry.depth_mm),      # 图像深度轴：主机原始值，与缓存一致
        "reference_db": float(cal.pivot_db),
        "noise_floor_top_db": float(floor[0]),
        "noise_floor_bottom_db": float(floor[-1]),
        "mode": int(mode),
        "mode_name": T.IMAGE_MODE_NAMES[mode],
        # 标量与档位用一位小数：tools_generate_console_labels.py 在合并前端标签时就是这样统一的
        # （主机原始值 41.871418，档位与标签里是 41.9），不跟着round 当前档就不在档位表上。
        "depth_mm": round(float(capture.geometry.depth_mm), 1),
        "frequency_mhz": float(CAL.capture_frequency(capture)),
        "focus_mm": float(focus[0]),
        "gain_level": int(capture.gain_level),
        "gain_db": float(S.capture_gain_db(capture)),
        "tgc_levels": np.asarray(capture.tgc_levels, np.float64),
        "dr_ui": float(capture.dynamic_range_level),
    }


# ---------------------------------------------------------------------------------------
#   档位

def mode_ladders(labels_path, ladders):
    """{(成像模式, 轴): 该模式下见过的档位}。读不到标签就返回空字典（全部档位都算可选）。"""
    if not labels_path or not os.path.exists(labels_path):
        return {}
    out = {}
    with io.open(labels_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("source") != "console":
                continue
            mode = K.MODE_HARMONIC if row.get("imaging_mode") == "harmonic" else K.MODE_FUNDAMENTAL
            for _, key, _ in AXES:
                out.setdefault((mode, key), set()).add(float(row[key]))
    return {k: sorted(v) for k, v in out.items()}


def valid_mask(ladder, allowed, current_value):
    """(可选档的多热向量, 当前档下标, 当前值是否正好在档位表上)。

    allowed 为 None 表示整条档位表都可选。当前档永远算可选，否则网络只能在别的档里挑。
    """
    values = np.asarray(ladder, np.float64)
    mask = np.ones(len(values), np.float32) if allowed is None else np.array(
        [1.0 if any(abs(v - a) <= 1e-3 for a in allowed) else 0.0 for v in values], np.float32)
    distance = np.abs(values - float(current_value))
    current_idx = int(distance.argmin())
    exact = bool(distance[current_idx] <= 1e-3)
    mask[current_idx] = 1.0
    return mask, current_idx, exact


def steps_between(mask, from_idx, to_idx):
    """两档之间隔了几个【可选】档（正数 = 往大调）。"""
    selectable = [i for i in range(len(mask)) if mask[i] > 0]
    if from_idx not in selectable or to_idx not in selectable:
        return to_idx - from_idx
    return selectable.index(to_idx) - selectable.index(from_idx)


# ---------------------------------------------------------------------------------------
#   建议

def suggest_batch(model, builder, device, frames, ladders, per_mode, args):
    """一批帧 -> 每帧一个建议字典。"""
    def column(key, dtype=np.float32):
        return torch.as_tensor(np.asarray([f[key] for f in frames], dtype), device=device)

    masks = {}
    for axis, key, _ in AXES:
        rows = []
        for f in frames:
            allowed = per_mode.get((f["mode"], key))
            if key == "focus_mm":
                pool = allowed if allowed is not None else ladders[key]
                allowed = [v for v in pool if v <= f["max_depth_mm"] + 1e-6]
            mask, idx, exact = valid_mask(ladders[key], allowed, f[key])
            f["_%s_idx" % axis], f["_%s_exact" % axis] = idx, exact
            rows.append(mask)
        masks[axis] = torch.as_tensor(np.stack(rows), device=device)

    inputs = builder(torch.as_tensor(np.stack([f["db"] for f in frames]), device=device),
                     torch.as_tensor(np.stack([f["floor"] for f in frames]), device=device),
                     column("min_depth_mm"), column("max_depth_mm"), column("depth_mm"),
                     column("frequency_mhz"), column("focus_mm"), column("gain_db"),
                     torch.as_tensor(np.stack([f["tgc_levels"] for f in frames]).astype(np.float32),
                                     device=device),
                     column("dr_ui"), column("reference_db"), column("mode"),
                     column("noise_floor_top_db"), column("noise_floor_bottom_db"))
    with torch.no_grad():
        out = model(inputs)
        dec = decode(out, {"depth_valid": masks["depth"], "frequency_valid": masks["frequency"],
                           "focus_valid": masks["focus"]})
    dec = {k: (v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy()) for k, v in dec.items()}

    results = []
    for j, f in enumerate(frames):
        results.append(one_suggestion(f, j, dec, masks, ladders, args))
    return results


def one_suggestion(f, j, dec, masks, ladders, args):
    """一帧的六根轴建议。前端按档位表给目标档，后端按主机的级给点击数。"""
    mode = f["mode"]
    gain_slope = K.GAIN_DB_PER_LEVEL[mode]
    tgc_slope = K.TGC_DB_PER_LEVEL[mode]
    out = {"frame_id": f["frame_id"], "imaging_mode": f["mode_name"],
           "current": {"depth_mm": f["depth_mm"], "frequency_mhz": f["frequency_mhz"],
                       "focus_mm": f["focus_mm"], "gain_level": f["gain_level"], "gain_db": f["gain_db"],
                       "tgc_levels": [int(v) for v in f["tgc_levels"]], "dr_ui": f["dr_ui"]},
           "frontend": {}, "backend": {}}

    frontend_changed = False
    for axis, key, unit in AXES:
        mask = masks[axis][j].cpu().numpy()
        current_idx = f["_%s_idx" % axis]
        wanted_idx = int(dec["%s_idx" % axis][j])
        prob = dec["%s_prob" % axis][j]
        steps = steps_between(mask, current_idx, wanted_idx)
        value = float(ladders[key][wanted_idx])
        frontend_changed = frontend_changed or steps != 0
        out["frontend"][axis] = {
            "current": f[key], "suggested": value, "steps": int(steps), "unit": unit,
            "direction": "keep" if steps == 0 else AXIS_WORDS[axis][0 if steps > 0 else 1],
            "probability": float(prob[wanted_idx]), "probability_current": float(prob[current_idx]),
            "expected_step": float(ladders[key][int(dec["%s_idx_expected" % axis][j])]),
            "current_on_ladder": bool(f["_%s_exact" % axis]),
            "selectable": [float(v) for v, m in zip(ladders[key], mask) if m > 0],
        }

    delta_db = float(dec["gain_delta_db"][j])
    clicks = float(np.round(delta_db / gain_slope))
    if abs(delta_db / gain_slope) <= args.stop_deadband_levels:
        clicks = 0.0
    clamped = False
    if args.max_gain_step_clicks and abs(clicks) > args.max_gain_step_clicks:
        clicks = float(np.sign(clicks) * args.max_gain_step_clicks)
        clamped = True
    new_level = int(np.clip(f["gain_level"] + clicks, GAIN_MIN_LEVEL, GAIN_MAX_LEVEL))
    out["backend"]["gain"] = {
        "delta_db": delta_db, "clicks": int(clicks), "clamped": clamped,
        "level": f["gain_level"], "suggested_level": new_level,
        "level_clipped": bool(f["gain_level"] + clicks != new_level),
    }

    delta_levels = dec["tgc_delta_db"][j] / tgc_slope
    deadband = args.stop_deadband_levels * gain_slope / tgc_slope
    group_delta = [float(delta_levels[lo:hi].mean()) for _, lo, hi in K.SLIDER_GROUPS]
    sliders_ok = all(abs(v) <= deadband for v in group_delta)
    new_levels = np.clip(np.round(f["tgc_levels"] + delta_levels), K.TGC_MIN_LEVEL, K.TGC_MAX_LEVEL)
    out["backend"]["tgc"] = {
        "delta_levels": [float(v) for v in delta_levels],
        "delta_db": [float(v) for v in dec["tgc_delta_db"][j]],
        "group_delta_levels": dict(zip([g[0] for g in K.SLIDER_GROUPS], group_delta)),
        "within_deadband": bool(sliders_ok),
        "suggested_levels": ([int(v) for v in f["tgc_levels"]] if sliders_ok else
                             [int(v) for v in new_levels]),
    }
    out["backend"]["dynamic_range"] = {"current_ui": f["dr_ui"], "suggested_ui": f["dr_ui"],
                                       "trained": False,
                                       "note": "no criterion in the labels (dr_determined is false "
                                               "everywhere); this axis is not trained"}
    out["deadband_levels"] = float(args.stop_deadband_levels)
    out["frontend_changed"] = bool(frontend_changed)
    out["settled"] = bool(not frontend_changed and clicks == 0 and sliders_ok)
    return out


# ---------------------------------------------------------------------------------------
#   报告

def format_suggestion(result, emit):
    cur = result["current"]
    emit("  ----- %s -----" % result["frame_id"])
    emit("    now        %s   depth %.1f mm   frequency %.1f MHz   focus %.1f mm"
         % (result["imaging_mode"], cur["depth_mm"], cur["frequency_mhz"], cur["focus_mm"]))
    emit("               gain level %d (%.2f dB)   dynamic range UI %.0f   TGC %s"
         % (cur["gain_level"], cur["gain_db"], cur["dr_ui"],
            " ".join("%3d" % v for v in cur["tgc_levels"])))
    if result["settled"]:
        emit("    -> every axis is already where the model would put it (no change suggested)")
    order = ("front end first, then capture again and re-run for the back end"
             if result["frontend_changed"] else "front end unchanged; the back-end correction below applies")
    emit("    front end (%s)" % order)
    for axis, _, unit in AXES:
        s = result["frontend"][axis]
        if s["steps"] == 0:
            action = "keep %.1f %s" % (s["current"], unit)
        else:
            action = "%.1f -> %.1f %s (%s by %d step%s)" % (
                s["current"], s["suggested"], unit, s["direction"], abs(s["steps"]),
                "" if abs(s["steps"]) == 1 else "s")
        note = "" if s["current_on_ladder"] else "   [current value is not on the ladder]"
        emit("      %-10s %-44s p %.2f (current %.2f)%s"
             % (axis, action, s["probability"], s["probability_current"], note))
    gain = result["backend"]["gain"]
    tgc = result["backend"]["tgc"]
    emit("    back end")
    if gain["clicks"] == 0:
        emit("      %-10s keep level %d (the model wants %+.2f dB, inside the %.1f-click deadband)"
             % ("gain", gain["level"], gain["delta_db"], result["deadband_levels"]))
    else:
        emit("      %-10s %+d clicks: level %d -> %d (%+.2f dB)%s%s"
             % ("gain", gain["clicks"], gain["level"], gain["suggested_level"], gain["delta_db"],
                "   [clamped to the single-step limit]" if gain["clamped"] else "",
                "   [clipped at the console's gain range]" if gain["level_clipped"] else ""))
        # 借来的标定把 dB 刻度整体搬了家，增益建议会顶到限幅上——那是标定的问题，不是图像的问题
        if gain["clamped"] and result.get("calibration", {}).get("borrowed"):
            emit("                 this capture uses a borrowed calibration; a correction this large is "
                 "usually the dB scale, not the image. Calibrate the session first.")
    if tgc["within_deadband"]:
        emit("      %-10s keep (near %+.1f, mid %+.1f, far %+.1f levels, all inside the deadband)"
             % ("TGC", tgc["group_delta_levels"]["near"], tgc["group_delta_levels"]["mid"],
                tgc["group_delta_levels"]["far"]))
    else:
        emit("      %-10s near %+.1f   mid %+.1f   far %+.1f  (levels)"
             % ("TGC", tgc["group_delta_levels"]["near"], tgc["group_delta_levels"]["mid"],
                tgc["group_delta_levels"]["far"]))
        emit("        band      %s" % " ".join("%4d" % (i + 1) for i in range(K.NUM_TGC_BANDS)))
        emit("        now       %s" % " ".join("%4d" % v for v in cur["tgc_levels"]))
        emit("        set to    %s" % " ".join("%4d" % v for v in tgc["suggested_levels"]))
    emit("      %-10s keep %.0f (%s)" % ("dyn range", result["backend"]["dynamic_range"]["current_ui"],
                                         result["backend"]["dynamic_range"]["note"]))


# ---------------------------------------------------------------------------------------
#   与训练缓存对照

def self_check(cache_dir, labels_path, frames, emit):
    """能在训练缓存 / 标签里找到的帧，逐项比预处理的结果。

    缓存管图像这一路（db、底噪、深度两端、曝光参考），标签管标量这一路（当前的六个设置）。
    两边都对上，才说明推理看到的与训练时看到的是同一帧。
    """
    index, arrays = None, None
    try:
        from bmode_dl.dataset import load_cache
        index, arrays = load_cache(cache_dir)
    except Exception as error:
        emit("  self-check skipped: %s" % error)
        return
    where = {str(f): i for i, f in enumerate(arrays["frame_id"])}
    checked, worst = 0, {}
    for f in frames:
        i = where.get(f["frame_id"])
        if i is None:
            continue
        checked += 1
        pairs = [("db", f["db"], arrays["db"][i]), ("floor", f["floor"], arrays["floor"][i])]
        for key in ("min_depth_mm", "max_depth_mm", "reference_db", "noise_floor_top_db",
                    "noise_floor_bottom_db"):
            if key in arrays:
                pairs.append((key, np.asarray(f[key], np.float64), np.asarray(arrays[key][i], np.float64)))
        for name, mine, theirs in pairs:
            diff = float(np.max(np.abs(np.asarray(mine, np.float64) - np.asarray(theirs, np.float64))))
            worst[name] = max(worst.get(name, 0.0), diff)
    if not checked:
        emit("  self-check: none of these captures is in %s" % cache_dir)
    else:
        ok = all(v <= 1e-3 for v in worst.values())
        emit("  self-check against %s: %d frame(s), %s" % (cache_dir, checked, "PASS" if ok else "FAIL"))
        for name in sorted(worst):
            emit("    max |mine - cache|  %-22s %.3e" % (name, worst[name]))
    self_check_settings(labels_path, frames, emit)


def self_check_settings(labels_path, frames, emit):
    """当前设置与标签行对照（标签是教师求解时用的那一组值）。"""
    if not labels_path or not os.path.exists(labels_path):
        return
    rows = {}
    with io.open(labels_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["frame_id"]] = row
    scalar = ("depth_mm", "frequency_mhz", "focus_mm", "gain_db", "dr_ui")
    checked, worst, mismatch = 0, {}, []
    for f in frames:
        row = rows.get(f["frame_id"])
        if row is None or "gain_db" not in row:
            continue
        checked += 1
        for key in scalar:
            worst[key] = max(worst.get(key, 0.0), abs(float(f[key]) - float(row[key])))
        worst["tgc_levels"] = max(worst.get("tgc_levels", 0.0),
                                  float(np.max(np.abs(f["tgc_levels"] - np.asarray(row["tgc_levels"], np.float64)))))
        if f["mode_name"] != row["imaging_mode"]:
            mismatch.append("%s: imaging mode %s vs %s" % (f["frame_id"], f["mode_name"], row["imaging_mode"]))
    if not checked:
        emit("  self-check: none of these captures is in %s" % labels_path)
        return
    ok = all(v <= 1e-3 for v in worst.values()) and not mismatch
    emit("  self-check against %s: %d frame(s), %s" % (labels_path, checked, "PASS" if ok else "FAIL"))
    emit("    max |mine - label|    %s" % "  ".join("%s %.1e" % (k, worst[k]) for k in sorted(worst)))
    for text in mismatch:
        emit("    %s" % text)


# ---------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    lines = []

    def emit(text=""):
        lines.append(text)

    import hisense_backend_sim as S
    from hisense_loader import load_capture
    import tools_generate_labels as TG
    import tools_generate_console_labels as G

    if args.checkpoint:
        model, builder, payload = load_checkpoint(args.checkpoint, args.device)
        front_payload = payload
        model_text = args.checkpoint
    else:
        model, builder, payload, front_payload = load_combined(args.frontend, args.backend, args.device)
        model_text = "%s (front end) + %s (gain / TGC)" % (args.frontend, args.backend)
    ladders = payload["ladders"]
    shape = payload["cache_shape"]
    rows, lines_out = int(shape["rows"]), int(shape["lines"])

    capture_dirs = find_capture_dirs(args.capture)
    if args.limit:
        capture_dirs = capture_dirs[:args.limit]
    if not capture_dirs:
        raise SystemExit("no capture under %s" % args.capture)

    emit("=========== six-parameter suggestions ===========")
    emit("  model      %s" % model_text)
    emit("             trained on %s, epoch %s (front end epoch %s), scalar_norm %s"
         % (payload["config"].get("labels", "?"), payload.get("epoch"), front_payload.get("epoch"),
            payload["config"].get("scalar_norm", "fixed")))
    emit("  captures   %d under %s" % (len(capture_dirs), args.capture))

    cal_by_group = TG.load_calibration()
    G.floors_in_counts(cal_by_group)
    per_mode = mode_ladders(args.labels, ladders)
    if per_mode:
        emit("  ladders    per imaging mode from %s" % args.labels)
    else:
        emit("  ladders    every step of the checkpoint's ladder is offered (no labels file)")
    for key in ("depth_mm", "frequency_mhz", "focus_mm"):
        emit("    %-14s %s" % (key, " ".join("%g" % v for v in ladders[key])))

    frames, skipped, groups = [], [], {}
    for path in capture_dirs:
        try:
            capture = load_capture(path)
            mode = S.capture_image_mode(capture)
            session = session_name(path)
            entry, used, borrowed = resolve_calibration(cal_by_group, session, mode, args.calibration_group)
            if entry["floor"] is None:
                raise ValueError("the calibration group %s/%d has no noise floor" % used)
            frame = prepare_capture(capture, entry, rows, lines_out)
            frame["session"] = session
            frame["calibration"] = {"group": "%s/%d" % used, "borrowed": bool(borrowed),
                                    "floor_measured": bool(entry.get("floor_measured")),
                                    "counts_per_db": float(entry["cal"].counts_per_db),
                                    "pivot_db": float(entry["cal"].pivot_db),
                                    "floor_db": float(entry["floor"])}
            groups.setdefault(frame["calibration"]["group"], frame["calibration"])
            frames.append(frame)
        except Exception as error:
            skipped.append((os.path.basename(path), "%s: %s" % (type(error).__name__, error)))

    emit("  calibration")
    for name in sorted(groups):
        c = groups[name]
        emit("    %-18s counts/dB %7.1f  pivot %6.2f dB  floor %5.2f dB%s%s"
             % (name, c["counts_per_db"], c["pivot_db"], c["floor_db"],
                "" if c["floor_measured"] else "  [floor borrowed within the mode]",
                "  [BORROWED FROM ANOTHER SESSION: run tools_refit_calibration.py for this one]"
                if c["borrowed"] else ""))
    for name, why in skipped:
        emit("    skipped %s (%s)" % (name, why))
    if not frames:
        raise SystemExit("no capture could be prepared")

    source_rows = sorted(set(f["source_rows"] for f in frames))
    if len(source_rows) != 1:
        raise SystemExit("captures have different depth sample counts: %s" % source_rows)
    if source_rows[0] != int(shape["source_rows"]):
        emit("  NOTE: these captures have %d depth samples, the training cache had %d; the TGC curve and the "
             "depth axis are rebuilt for %d" % (source_rows[0], int(shape["source_rows"]), source_rows[0]))
    builder = InputBuilder(rows, lines_out, source_rows[0], payload["norm"],
                           use_noise_floor=not payload["config"].get("no_noise_floor", False),
                           scalar_norm=payload["config"].get("scalar_norm", "fixed")).to(args.device)
    emit("")

    if args.self_check:
        self_check(args.self_check, args.labels, frames, emit)
        emit("")

    results = []
    for s in range(0, len(frames), args.batch_size):
        results.extend(suggest_batch(model, builder, args.device, frames[s:s + args.batch_size],
                                     ladders, per_mode, args))
    for frame, result in zip(frames, results):
        result["session"] = frame["session"]
        result["calibration"] = frame["calibration"]
        format_suggestion(result, emit)

    settled = sum(1 for r in results if r["settled"])
    front = sum(1 for r in results if r["frontend_changed"])
    emit("")
    emit("  %d capture(s): %d need a front-end change, %d need only a back-end correction, %d are settled"
         % (len(results), front, len(results) - front - settled, settled))
    emit("  dynamic range is never suggested: there is no criterion in the labels, so it was not trained")
    emit("  done in %.1f s" % (time.time() - started))

    if args.json:
        with io.open(args.json, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        emit("  wrote %s" % args.json)

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return results


if __name__ == "__main__":
    main()
