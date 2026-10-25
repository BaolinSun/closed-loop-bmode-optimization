# -*- coding: utf-8 -*-
"""上机闭环实测的记录工具：把一次真机上「按建议调 -> 重新采集」的多帧串成一条轨迹。

    要回答的问题

闭环仿真只能在 Field II 的设置网格上跑（每个体模缓存了全部 深度 x 频率 x 聚焦 组合，前端改档
可以真的换一幅图）。实机没有这张网格，闭环是不是真的停得下来、会不会来回打转，只能上机做：
按模型的建议调一次、重新采一帧、再问一次模型，直到它说不用再动。本脚本把这一串采集读进来，
逐步打印「模型说什么、人实际做了什么、两者是否一致、图像变成什么样」，并写出与仿真同一格式的
轨迹记录，交给 tests/measure_loop_stop_points.py 与仿真轨迹放在一起比较。

    用法

    # 按时间戳排序的一串采集（同一个探头位置，中间只改参数）
    python tools_track_console_loop.py --session data/hisense_medical/20260930_loop1

    # 或显式给出顺序（目录名不是时间戳、或一个场次里有多条轨迹时）
    python tools_track_console_loop.py --steps data/.../cap1 data/.../cap2 data/.../cap3

模型、标定、档位与预处理全部复用 tools_suggest_six_param.py 的函数，两个脚本看到的输入一致。

    每一步的判读

协议与仿真一致：前端三轴有改动时这一步只改前端、重新采一帧，不动后端（后端修正是在当前这幅图上
算出来的，前端一换图就作废）；前端不动时才做后端修正。所以每一步的「应做动作」是确定的，
脚本据此检查人是否照做（compliance）：

    前端  三轴是否都调到了建议档
    增益  档位变化是否等于建议的点击数（容差 --gain-tolerance-clicks，默认 1 级）
    TGC   8 段是否调到了建议值（平均差容差 --tgc-tolerance-levels，默认 2 级）

两者不一致不算错误，但会标出来：轨迹如果没按建议走，它就不能用来判断模型的闭环行为。
一步里前端与后端同时改动也会标出来（与仿真不可比）。

    图像那一行

每一步另报三个不需要标签就能看的量，用来判断图像是不是真的在变好：

    median gray      整幅图灰阶中位数（目标是组织落在中灰，不是越亮越好）
    crushed / sat    压到 0 与顶到 255 的像素比例
    bottom 2 mm      图像最底部 2 mm 的 dB 中位数高出噪声底多少（穿透余量，与频率、深度标签
                     用的是同一个量；tools_generate_console_labels.py 的门槛是 6 dB）

    停下与打转

模型对最后一帧给出「六根轴都不用动」即为停下（converged）。走回一个已经走过的前端设置记为打转
（oscillated），与仿真里的判定相同。轨迹记录字段与 bmode_dl/closed_loop.py 的 records 对齐，
另加 applied / matched / image 三组实测字段；教师标签相关的字段（frontend_correct、
gain_error_db）只有在这些帧恰好在 --labels 里有标签时才写（用已采数据做干跑时会有）。

终端输出与写在脚本旁边的 tools_track_console_loop.txt 同内容（ASCII）。
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
import tools_suggest_six_param as SG
from bmode_dl.render import BackendRenderer

REPORT_PATH = "tools_track_console_loop.txt"
AXES = ("depth", "frequency", "focus")
AXIS_KEY = {"depth": "depth_mm", "frequency": "frequency_mhz", "focus": "focus_mm"}
AXIS_UNIT = {"depth": "mm", "frequency": "MHz", "focus": "mm"}
# 底部穿透余量的窗口，与 tools_generate_console_labels.BOTTOM_WINDOW_MM 相同
BOTTOM_WINDOW_MM = 2.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Record one on-console closed-loop trajectory")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--session", help="a session directory whose captures, sorted by name, are the trajectory")
    g.add_argument("--steps", nargs="+", help="capture directories in the order they were acquired")
    p.add_argument("--frontend", default=os.path.join(SG.DEFAULT_RUN, "frontend.pt"))
    p.add_argument("--backend", default=os.path.join(SG.DEFAULT_RUN, "last.pt"))
    p.add_argument("--checkpoint", default=None, help="one checkpoint for all six axes")
    p.add_argument("--labels", default="data/labels_console.jsonl",
                   help="labels the per-mode ladders are read from; also supplies the teacher's answer "
                        "for any step that happens to be a labelled frame (dry runs on existing data)")
    p.add_argument("--calibration-group", default=None, help="borrow this group's calibration, as session/mode")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stop-deadband-levels", type=float, default=0.5)
    p.add_argument("--max-gain-step-clicks", type=int, default=80)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gain-tolerance-clicks", type=float, default=1.0,
                   help="how far the applied gain may differ from the suggested clicks and still count as followed")
    p.add_argument("--tgc-tolerance-levels", type=float, default=2.0,
                   help="mean absolute slider difference that still counts as followed")
    p.add_argument("--name", default=None, help="trajectory name in the record (default: the session name)")
    p.add_argument("--json", default=None,
                   help="write the trajectory record here (default: data/console_loops/<name>_trajectory.jsonl)")
    p.add_argument("--self-check", default=None, help="a training cache to compare the preprocessing against")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------
#   图像那一行

def image_stats(frame):
    """当前设置下渲染出的灰阶与底部穿透余量。不需要标签，人在屏幕上看到的就是这些。"""
    rows = frame["db"].shape[0]
    renderer = BackendRenderer(rows, frame["source_rows"])
    one = lambda v: torch.as_tensor(np.asarray([v], np.float32))
    gray = renderer(torch.as_tensor(frame["db"][None]), one(frame["gain_db"]),
                    torch.as_tensor(frame["tgc_levels"][None].astype(np.float32)),
                    one(frame["reference_db"]), one(frame["dr_ui"]), one(float(frame["mode"])))
    gray = gray[0].numpy()
    z = np.linspace(frame["min_depth_mm"], frame["max_depth_mm"], rows)
    bottom = z >= frame["max_depth_mm"] - BOTTOM_WINDOW_MM
    if not bottom.any():
        bottom = np.zeros(rows, bool)
        bottom[-1] = True
    excess = float(np.median(frame["db"][bottom], axis=1).mean() - frame["floor"][bottom].mean())
    return {"median_gray": float(np.median(gray)),
            "crushed": float((gray <= 0.0).mean()), "saturated": float((gray >= K.GRAY_MAX).mean()),
            "bottom_excess_db": excess}


# ---------------------------------------------------------------------------------------
#   一步里人实际做了什么

def setting_of(frame):
    return tuple(round(float(frame[AXIS_KEY[a]]), 3) for a in AXES)


def applied_between(before, after):
    """两帧之间实际发生的改动。"""
    out = {"frontend": {}, "frontend_changed": False}
    for axis in AXES:
        key = AXIS_KEY[axis]
        moved = abs(float(after[key]) - float(before[key])) > 1e-3
        out["frontend"][axis] = {"from": float(before[key]), "to": float(after[key]), "changed": bool(moved)}
        out["frontend_changed"] = out["frontend_changed"] or moved
    clicks = int(after["gain_level"]) - int(before["gain_level"])
    tgc_delta = np.asarray(after["tgc_levels"], np.float64) - np.asarray(before["tgc_levels"], np.float64)
    out["gain_clicks"] = clicks
    out["tgc_delta_levels"] = [float(v) for v in tgc_delta]
    out["backend_changed"] = bool(clicks != 0 or np.any(np.abs(tgc_delta) > 0))
    out["dr_changed"] = bool(abs(float(after["dr_ui"]) - float(before["dr_ui"])) > 1e-6)
    return out


def compare(result, applied, args):
    """模型建议 vs 实际动作。返回 (是否照做, 说明行)。"""
    expected = "frontend" if result["frontend_changed"] else ("stop" if result["settled"] else "backend")
    notes, ok = [], True
    if expected == "frontend":
        for axis in AXES:
            want = result["frontend"][axis]
            got = applied["frontend"][axis]
            if abs(got["to"] - want["suggested"]) > 1e-3:
                ok = False
                notes.append("%s went to %.1f, the model asked for %.1f" % (axis, got["to"], want["suggested"]))
        if applied["backend_changed"]:
            notes.append("the back end was changed in the same step (the simulation never does this: "
                         "the correction was computed on the previous image)")
    elif expected == "backend":
        if applied["frontend_changed"]:
            ok = False
            notes.append("the front end moved although the model asked to keep it")
        gain = result["backend"]["gain"]
        if abs(applied["gain_clicks"] - gain["clicks"]) > args.gain_tolerance_clicks:
            ok = False
            notes.append("gain moved %+d clicks, the model asked for %+d"
                         % (applied["gain_clicks"], gain["clicks"]))
        tgc = result["backend"]["tgc"]
        if not tgc["within_deadband"]:
            want = np.asarray(tgc["suggested_levels"], np.float64)
            got = np.asarray(applied["tgc_delta_levels"], np.float64) + np.asarray(
                result["current"]["tgc_levels"], np.float64)
            diff = float(np.abs(got - want).mean())
            if diff > args.tgc_tolerance_levels:
                ok = False
                notes.append("TGC differs from the suggestion by %.1f levels on average" % diff)
    else:
        if applied["frontend_changed"] or applied["backend_changed"]:
            ok = False
            notes.append("something was changed although the model had stopped")
    return expected, ok, notes


# ---------------------------------------------------------------------------------------
#   教师标签（只有干跑在已采数据上时才有）

def teacher_rows(labels_path):
    if not labels_path or not os.path.exists(labels_path):
        return {}
    out = {}
    with io.open(labels_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["frame_id"]] = row
    return out


def teacher_verdict(row, frame):
    """最后一帧若有标签，报出教师对这一帧的评判。没有标签返回 None。"""
    if row is None:
        return None
    mode = int(frame["mode"])
    gain_slope = K.GAIN_DB_PER_LEVEL[mode]
    tgc_slope = K.TGC_DB_PER_LEVEL[mode]
    out = {}
    if "optimal_gain_db" in row:
        error = float(row["optimal_gain_db"]) - float(frame["gain_db"])
        out["gain_error_db"] = error
        out["gain_within_deadband"] = bool(abs(error) <= float(row["deadband_gain_levels"]) * gain_slope)
        out["tgc_mae_db"] = float(np.abs(np.asarray(row["optimal_tgc_levels"], np.float64)
                                         - np.asarray(frame["tgc_levels"], np.float64)).mean() * tgc_slope)
    correct, steps = {}, {}
    for axis in AXES:
        if not row.get("%s_determined" % axis):
            continue
        correct[axis] = bool(row.get("%s_direction" % axis) == "correct")
        delta = row.get("delta_%s_steps" % axis)
        if delta is not None:
            steps[axis] = int(delta)
    if correct:
        out["frontend_correct"] = correct
    if steps:
        out["frontend_steps_to_optimum"] = steps
    return out or None


# ---------------------------------------------------------------------------------------
#   报告

def format_step(number, frame, stats, result, emit):
    cur = result["current"]
    emit("  ----- step %d: %s -----" % (number, frame["frame_id"]))
    emit("    setting    %s  depth %.1f mm  frequency %.1f MHz  focus %.1f mm | gain level %d  DR %.0f"
         % (result["imaging_mode"], cur["depth_mm"], cur["frequency_mhz"], cur["focus_mm"],
            cur["gain_level"], cur["dr_ui"]))
    emit("               TGC %s" % " ".join("%3d" % v for v in cur["tgc_levels"]))
    emit("    image      median gray %3.0f   crushed %4.1f%%   saturated %4.1f%%   bottom 2 mm %+5.1f dB over the floor"
         % (stats["median_gray"], 100 * stats["crushed"], 100 * stats["saturated"], stats["bottom_excess_db"]))
    wants = []
    for axis in AXES:
        s = result["frontend"][axis]
        if s["steps"]:
            wants.append("%s %.1f -> %.1f %s (%s %d)" % (axis, s["current"], s["suggested"], AXIS_UNIT[axis],
                                                        s["direction"], abs(s["steps"])))
    gain, tgc = result["backend"]["gain"], result["backend"]["tgc"]
    back = []
    if gain["clicks"]:
        back.append("gain %+d clicks (level %d -> %d)" % (gain["clicks"], gain["level"], gain["suggested_level"]))
    if not tgc["within_deadband"]:
        back.append("TGC near %+.1f mid %+.1f far %+.1f levels"
                    % (tgc["group_delta_levels"]["near"], tgc["group_delta_levels"]["mid"],
                       tgc["group_delta_levels"]["far"]))
    if wants:
        emit("    model      front end: %s" % ";  ".join(wants))
        if back:
            emit("               (back end held: %s -- recompute it after the front end moves)" % ";  ".join(back))
    elif back:
        emit("    model      back end: %s" % ";  ".join(back))
    else:
        emit("    model      nothing to change: this is a stop")


def format_applied(applied, expected, ok, notes, emit):
    moves = []
    for axis in AXES:
        got = applied["frontend"][axis]
        if got["changed"]:
            moves.append("%s %.1f -> %.1f" % (axis, got["from"], got["to"]))
    if applied["gain_clicks"]:
        moves.append("gain %+d clicks" % applied["gain_clicks"])
    tgc = np.asarray(applied["tgc_delta_levels"], np.float64)
    if np.any(np.abs(tgc) > 0):
        moves.append("TGC mean %+.1f levels (max %+.0f)" % (tgc.mean(), tgc[np.argmax(np.abs(tgc))]))
    if applied["dr_changed"]:
        moves.append("dynamic range changed (the model never asks for this)")
    emit("    operator   %s   [%s]" % (";  ".join(moves) if moves else "nothing changed",
                                       "followed the %s suggestion" % expected if ok else "DIFFERS"))
    for text in notes:
        emit("               note: %s" % text)


# ---------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    lines = []

    def emit(text=""):
        lines.append(text)

    if args.steps:
        capture_dirs = list(args.steps)
        source = "%d capture(s) given on the command line" % len(capture_dirs)
        name = args.name or os.path.basename(os.path.dirname(os.path.abspath(capture_dirs[0])))
    else:
        capture_dirs = SG.find_capture_dirs(args.session)
        source = args.session
        name = args.name or os.path.basename(os.path.abspath(args.session))
    if args.limit:
        capture_dirs = capture_dirs[:args.limit]
    if len(capture_dirs) < 2:
        raise SystemExit("a trajectory needs at least two captures; got %d" % len(capture_dirs))

    model, payload, front_payload, model_text = SG.load_model(args)
    ladders = payload["ladders"]

    emit("=========== on-console closed-loop trajectory ===========")
    SG.describe_model(payload, front_payload, model_text, emit)
    emit("  trajectory %s: %d capture(s) from %s" % (name, len(capture_dirs), source))
    frames, builder, per_mode = SG.prepare_frames(capture_dirs, payload, args, emit)
    emit("")
    if len(frames) != len(capture_dirs):
        emit("  WARNING: %d capture(s) could not be prepared; the trajectory below skips them"
             % (len(capture_dirs) - len(frames)))
        emit("")
    if args.self_check:
        SG.self_check(args.self_check, args.labels, frames, emit)
        emit("")

    results = []
    for s in range(0, len(frames), args.batch_size):
        results.extend(SG.suggest_batch(model, builder, args.device, frames[s:s + args.batch_size],
                                        ladders, per_mode, args))
    stats = [image_stats(f) for f in frames]
    rows = teacher_rows(args.labels)

    visited = {setting_of(frames[0])}
    path = [{"step": 0, "action": "start", "setting_mm": list(setting_of(frames[0])),
             "frame": frames[0]["frame_id"]}]
    steps_record, frontend_moves, backend_moves, total_clicks = [], 0, 0, 0
    axis_changes = dict((a, 0) for a in AXES)
    oscillated, followed, checked, protocol_breaks = False, 0, 0, 0

    for i, (frame, result, stat) in enumerate(zip(frames, results, stats)):
        format_step(i + 1, frame, stat, result, emit)
        if i + 1 >= len(frames):
            if not result["settled"]:
                emit("    -> the trajectory ends here but the model still wants a change (not converged)")
            else:
                emit("    -> the model stops here: every axis is where it would put it")
            emit("")
            break
        applied = applied_between(frame, frames[i + 1])
        expected, ok, notes = compare(result, applied, args)
        format_applied(applied, expected, ok, notes, emit)
        checked += 1
        followed += int(ok)
        protocol_breaks += int(applied["frontend_changed"] and applied["backend_changed"])
        setting = setting_of(frames[i + 1])
        if applied["frontend_changed"]:
            frontend_moves += 1
            for axis in AXES:
                axis_changes[axis] += int(applied["frontend"][axis]["changed"])
            if setting in visited:
                oscillated = True
                emit("               note: this setting has already been visited (oscillation)")
            visited.add(setting)
            path.append({"step": i + 1, "action": "frontend", "setting_mm": list(setting),
                         "frame": frames[i + 1]["frame_id"]})
        if applied["backend_changed"]:
            backend_moves += 1
            total_clicks += applied["gain_clicks"]
            path.append({"step": i + 1, "action": "backend", "setting_mm": list(setting_of(frame)),
                         "gain_clicks": float(applied["gain_clicks"]),
                         "frame": frames[i + 1]["frame_id"]})
        if not applied["frontend_changed"] and not applied["backend_changed"]:
            path.append({"step": i + 1, "action": "no-change", "setting_mm": list(setting),
                         "frame": frames[i + 1]["frame_id"]})
        steps_record.append({"step": i + 1, "frame": frame["frame_id"], "expected": expected,
                             "followed": bool(ok), "notes": notes, "applied": applied,
                             "suggestion": result, "image": stat})
        emit("")

    last, last_result = frames[-1], results[-1]
    path.append({"step": len(frames) - 1, "action": "stop" if last_result["settled"] else "end",
                 "setting_mm": list(setting_of(last)), "frame": last["frame_id"]})
    record = {
        "trajectory": name, "source": "console",
        "start_frame": frames[0]["frame_id"], "final_frame": last["frame_id"],
        "converged": bool(last_result["settled"]), "oscillated": bool(oscillated), "frozen": False,
        "steps": len(frames) - 1, "frontend_moves": frontend_moves, "backend_moves": backend_moves,
        "axis_changes": axis_changes, "gain_total_clicks": float(total_clicks),
        "gain_clamped": bool(any(s["suggestion"]["backend"]["gain"]["clamped"] for s in steps_record)),
        "followed": followed, "checked": checked, "protocol_breaks": protocol_breaks,
        "start_setting_mm": list(setting_of(frames[0])), "final_setting_mm": list(setting_of(last)),
        "start_image": stats[0], "final_image": stats[-1],
        "path": path, "step_details": steps_record,
    }
    verdict = teacher_verdict(rows.get(last["frame_id"]), last)
    if verdict:
        record.update(verdict)

    emit("  ----- trajectory -----")
    emit("    %d step(s); %s" % (record["steps"],
                                 "stopped" if record["converged"] else "NOT stopped (still asking for a change)"))
    emit("    front-end moves %d (depth %d, frequency %d, focus %d), back-end moves %d, net gain %+d clicks"
         % (frontend_moves, axis_changes["depth"], axis_changes["frequency"], axis_changes["focus"],
            backend_moves, total_clicks))
    emit("    setting   %s -> %s (depth mm, frequency MHz, focus mm)"
         % (record["start_setting_mm"], record["final_setting_mm"]))
    emit("    operator followed the suggestion in %d / %d step(s)%s"
         % (followed, checked, "" if not protocol_breaks else
            "; %d step(s) changed front and back end together" % protocol_breaks))
    if oscillated:
        emit("    OSCILLATION: the trajectory returned to a front-end setting it had already visited")
    emit("    image     median gray %.0f -> %.0f   crushed %.1f%% -> %.1f%%   bottom 2 mm %+.1f -> %+.1f dB"
         % (stats[0]["median_gray"], stats[-1]["median_gray"], 100 * stats[0]["crushed"],
            100 * stats[-1]["crushed"], stats[0]["bottom_excess_db"], stats[-1]["bottom_excess_db"]))
    if verdict:
        emit("    teacher (the last frame is a labelled frame, so this is a dry run on existing data)")
        if "gain_error_db" in verdict:
            emit("      gain still %+.2f dB from the teacher's optimum (%s the deadband), TGC %.2f dB"
                 % (verdict["gain_error_db"], "inside" if verdict["gain_within_deadband"] else "outside",
                    verdict["tgc_mae_db"]))
        if "frontend_correct" in verdict:
            emit("      front end at the stop point: %s"
                 % ", ".join("%s %s" % (a, "correct" if v else "wrong")
                             for a, v in sorted(verdict["frontend_correct"].items())))
    else:
        emit("    no teacher label for the last frame: this is a real on-console run, so the verdict is the "
             "image and the operator's judgement, not a number")
    emit("  done in %.1f s" % (time.time() - started))

    out_path = args.json or os.path.join("data", "console_loops", "%s_trajectory.jsonl" % name)
    directory = os.path.dirname(os.path.abspath(out_path))
    if not os.path.isdir(directory):
        os.makedirs(directory)
    with io.open(out_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    emit("  wrote %s (same schema as the simulated trajectories; "
         "tests/measure_loop_stop_points.py reads both)" % out_path)

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return record


if __name__ == "__main__":
    main()
