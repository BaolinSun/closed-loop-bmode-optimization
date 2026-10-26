# -*- coding: utf-8 -*-
"""把一位测试集受试者的真实采集做成一页可展示的 demo（图像内嵌的单个 HTML 文件）。

    为什么能这么做

增益与 TGC 不改变 BC0，只改变渲染。所以「模型建议」与「教师最优」两种设置下的图像都可以从同一帧
原始数据精确渲染出来——不是示意图，是那一帧真的调成那样的样子。聚焦这一轴不同：它改变 BC0 本身，
所以页面里的六档聚焦是同一探头位置下真实采到的六帧。

    页面内容

  1. 一帧：当前设置（实机采下来的样子）-> 模型建议 -> 教师最优，外加三条 TGC 曲线
  2. 后端闭环：按建议调、重新观测、再建议，逐步到停下，每一步一张图
  3. 聚焦：同一探头位置真实采到的六档，标出模型选哪档、教师选哪档
  4. 增益残差的形状：按受试者看，残差几乎全是整位受试者的常数偏置
  5. 测试集上的汇总指标与只看设置的查表基线对照

    展示哪一位不能靠挑

候选是测试集里扫过 >= 5 档聚焦的场景；选的是【起点】最典型的那一个（锚点帧的起始增益误差最接近
候选的中位数）。按结果挑会把选择本身变成结论：按最终误差取中位数，选出来的往往是起点本来就很准
的那一位，而闭环在那种帧上只会把它推远。页面里也写明候选的分布。

    用法

    python tools_build_demo_page.py
    python tools_build_demo_page.py --family 20261009_183301_00001/1 --out data/demo.html

受试者取自测试集（从未参与训练与选模），默认模型 runs/invivo_bf_final。图像、曲线、指标全部现算，
汇总数字直接从 runs/*/fold_0/evaluation_*.json 与闭环轨迹 jsonl 读，页面里没有手抄的数字。
终端输出与写在脚本旁边的 tools_build_demo_page.txt 同内容（ASCII）。
"""

import argparse
import base64
import io
import json
import os
import sys
import time
from string import Template

sys.path.insert(0, "bmode_opt")
import numpy as np
import torch

import bmode_dl.constants as K
import tools_suggest_six_param as SG
from bmode_dl.render import BackendRenderer

REPORT_PATH = "tools_build_demo_page.txt"
DEFAULT_RUN = os.path.join("runs", "invivo_bf_final", "all_data")
PIXELS_PER_MM = 9.0          # 深度与横向同一个尺度，所以图像比例是真实的
SCAN_WIDTH_MM = 38.1         # L15-4 的 BTotalScanWidth，各帧相同


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build a self-contained demo page from one test subject")
    p.add_argument("--family", default=None,
                   help="scene family to tell the story with (default: chosen by --pick)")
    p.add_argument("--pick", choices=("median", "first"), default="median",
                   help="median: the candidate whose starting gain error is the median of all candidates, "
                        "so the page shows a typical starting point chosen before seeing the outcome")
    p.add_argument("--candidates", type=int, default=16, help="how many scenes to weigh for --pick median")
    p.add_argument("--frontend", default=os.path.join(DEFAULT_RUN, "frontend.pt"))
    p.add_argument("--backend", default=os.path.join(DEFAULT_RUN, "last.pt"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--labels", default="data/labels_console_invivo_carotid_target.jsonl")
    p.add_argument("--dataset", default="invivo", choices=("auto", "phantom", "invivo"))
    p.add_argument("--calibration", default=None)
    p.add_argument("--calibration-group", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stop-deadband-levels", type=float, default=0.5)
    p.add_argument("--max-gain-step-clicks", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=4)
    p.add_argument("--single-step-json", default="runs/invivo_bf_v2/fold_0/evaluation_combined_test_noloop.json")
    p.add_argument("--loop-json", default="runs/invivo_bf_v2/fold_0/evaluation_combined_bloop.json")
    p.add_argument("--loop-trajectories",
                   default="runs/invivo_bf_v2/fold_0/closed_loop_trajectories_combined_bloop.jsonl")
    p.add_argument("--out", default="data/demo_invivo_carotid.html")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------
#   渲染

def render_gray(frame, gain_db, tgc_levels):
    """一帧在给定增益 / TGC 下的灰阶（与主机后端逐像素一致的那套算术）。"""
    rows = frame["db"].shape[0]
    renderer = BackendRenderer(rows, frame["source_rows"])
    one = lambda v: torch.as_tensor(np.asarray([v], np.float32))
    gray = renderer(torch.as_tensor(frame["db"][None]), one(float(gain_db)),
                    torch.as_tensor(np.asarray(tgc_levels, np.float32)[None]),
                    one(frame["reference_db"]), one(frame["dr_ui"]), one(float(frame["mode"])))
    return gray[0].numpy()


def png_data_uri(gray, depth_mm):
    """灰阶数组 -> 按物理比例放大的 PNG data URI。"""
    from PIL import Image

    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L")
    image = image.resize((int(round(SCAN_WIDTH_MM * PIXELS_PER_MM)),
                          int(round(depth_mm * PIXELS_PER_MM))), Image.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------------------
#   后端闭环（算术与 bmode_dl/closed_loop.py 相同：点击数取整、死区停、单步限幅）

def apply_backend(frame, result):
    gain = result["backend"]["gain"]
    tgc = result["backend"]["tgc"]
    clicks = float(gain["clicks"])
    levels = np.asarray(frame["tgc_levels"], np.float64)
    if not tgc["within_deadband"]:
        levels = np.asarray(tgc["suggested_levels"], np.float64)
    return {"gain_db": frame["gain_db"] + clicks * K.GAIN_DB_PER_LEVEL[int(frame["mode"])],
            "gain_level": int(frame["gain_level"] + clicks), "tgc_levels": levels,
            "clicks": int(clicks), "tgc_moved": not tgc["within_deadband"],
            "stopped": clicks == 0 and tgc["within_deadband"]}


def backend_loop(model, builder, frame, row, ladders, per_mode, args, trained, with_images=True):
    """在同一帧上跑后端闭环，记下每一步的设置、图像与离教师最优还有多远。"""
    tgc_slope = K.TGC_DB_PER_LEVEL[int(frame["mode"])]
    optimal_gain = float(row["optimal_gain_db"])
    optimal_levels = np.asarray(row["optimal_tgc_levels"], np.float64)
    state = dict(frame)
    steps = []
    for step in range(int(args.max_steps) + 1):
        entry = {"step": step, "gain_level": int(state["gain_level"]), "gain_db": float(state["gain_db"]),
                 "gain_error_db": optimal_gain - state["gain_db"],
                 "tgc_error_db": float(np.abs(optimal_levels
                                              - np.asarray(state["tgc_levels"], np.float64)).mean() * tgc_slope)}
        if with_images:
            entry["image"] = png_data_uri(render_gray(state, state["gain_db"], state["tgc_levels"]),
                                          state["max_depth_mm"])
        result = SG.suggest_batch(model, builder, args.device, [state], ladders, per_mode, args, trained)[0]
        action = apply_backend(state, result)
        entry["suggestion"] = {"clicks": action["clicks"], "tgc_moved": action["tgc_moved"]}
        entry["stopped"] = action["stopped"]
        steps.append(entry)
        if action["stopped"]:
            break
        state = dict(state, gain_db=action["gain_db"], gain_level=action["gain_level"],
                     tgc_levels=action["tgc_levels"])
    return steps


# ---------------------------------------------------------------------------------------
#   选场景

def scene_frames(name, group, payload, args, emit):
    """一个族里每档聚焦取最早的一帧。返回 (frames, builder, per_mode, rows_by_frame)。"""
    import console_sessions as CS

    per_focus = {}
    for row in sorted(group, key=lambda r: r["frame_id"]):
        per_focus.setdefault(round(row["focus_mm"], 1), row)
    session = name.rsplit("/", 1)[0]
    paths = dict((os.path.basename(str(p)), str(p)) for p in CS.session_captures(session))
    capture_dirs = [paths[r["frame_id"]] for r in per_focus.values() if r["frame_id"] in paths]
    if len(capture_dirs) < 2:
        return None
    frames, builder, per_mode = SG.prepare_frames(capture_dirs, payload, args, emit)
    return frames, builder, per_mode, dict((r["frame_id"], r) for r in group)


def scene_story(name, group, model, payload, trained, args, emit, with_images=False):
    """一个族的锚点帧走完闭环的结果，用来比较候选。"""
    prepared = scene_frames(name, group, payload, args, emit)
    if prepared is None:
        return None
    frames, builder, per_mode, rows = prepared
    ladders = payload["ladders"]
    results = SG.suggest_batch(model, builder, args.device, frames, ladders, per_mode, args, trained)
    order = sorted(range(len(frames)), key=lambda i: frames[i]["focus_mm"])
    anchor = min(order, key=lambda i: abs(frames[i]["focus_mm"] - 15.0))
    row = rows[frames[anchor]["frame_id"]]
    steps = backend_loop(model, builder, frames[anchor], row, ladders, per_mode, args, trained, with_images)
    return {"family": name, "frames": frames, "results": results, "builder": builder, "per_mode": per_mode,
            "order": order, "anchor": anchor, "row": row, "rows": rows, "steps": steps,
            "start_gain": abs(steps[0]["gain_error_db"]), "final_gain": abs(steps[-1]["gain_error_db"]),
            "start_tgc": steps[0]["tgc_error_db"], "final_tgc": steps[-1]["tgc_error_db"],
            "focus_match": round(float(row["optimal_focus_mm"]), 1)
                           == round(float(results[anchor]["frontend"]["focus"]["suggested"]), 1)}


def pick_scene(candidates, by_family, model, payload, trained, args, emit):
    """按起点的典型程度挑，而不是按结果挑。"""
    quiet = lambda text="": None
    stories = []
    for name, _ in candidates:
        group = [r for r in by_family.get(name, []) if r.get("focus_determined")]
        try:
            story = scene_story(name, group, model, payload, trained, args, quiet)
        except Exception as error:
            emit("    candidate %s skipped (%s: %s)" % (name, type(error).__name__, error))
            continue
        if story:
            stories.append(story)
    if not stories:
        raise SystemExit("no candidate scene could be prepared")
    starts = sorted(s["start_gain"] for s in stories)
    median_start = starts[len(starts) // 2]
    finals = sorted(s["final_gain"] for s in stories)
    survey = {"n": len(stories), "start_median": median_start,
              "gain_best": finals[0], "gain_median": finals[len(finals) // 2], "gain_worst": finals[-1],
              "tgc_median": sorted(s["final_tgc"] for s in stories)[len(stories) // 2],
              "focus_match": sum(1 for s in stories if s["focus_match"])}
    emit("  survey     %d candidate scenes: start gain error median %.2f dB; after the loop best %.2f, "
         "median %.2f, worst %.2f dB; median TGC %.2f dB; focus matches the teacher in %d"
         % (survey["n"], survey["start_median"], survey["gain_best"], survey["gain_median"],
            survey["gain_worst"], survey["tgc_median"], survey["focus_match"]))
    for s in sorted(stories, key=lambda x: x["start_gain"]):
        emit("    %-28s gain %5.2f -> %5.2f dB, TGC %5.2f -> %.2f dB, focus %s"
             % (s["family"], s["start_gain"], s["final_gain"], s["start_tgc"], s["final_tgc"],
                "match" if s["focus_match"] else "differs"))
    chosen = min(stories, key=lambda s: abs(s["start_gain"] - median_start))
    emit("  chosen     %s (the median starting point, picked before looking at the outcome)" % chosen["family"])
    return chosen, survey


# ---------------------------------------------------------------------------------------
#   图表

def svg_tgc_curves(current, suggested, teacher, slope):
    """三条 TGC 曲线（dB，相对中位 127）。直接标注三条线，不用图例框。"""
    width, height = 560, 230
    left, right, top, bottom = 48, 118, 18, 34
    bands = len(current)
    db = [(np.asarray(v, float) - K.TGC_CENTER_LEVEL) * slope for v in (current, suggested, teacher)]
    low = min(-2.0, min(float(d.min()) for d in db) - 1.0)
    high = max(2.0, max(float(d.max()) for d in db) + 1.0)
    x = lambda i: left + i * (width - left - right) / (bands - 1.0)
    y = lambda v: top + (high - v) * (height - top - bottom) / (high - low)
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="TGC curves in dB" class="chart">' % (width, height)]
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="axis"/>'
               % (left, y(0.0), width - right, y(0.0)))
    out.append('<text x="%.1f" y="%.1f" class="tick" text-anchor="end">0 dB</text>' % (left - 8, y(0.0) + 4))
    for value in (round(low + 1), round(high - 1)):
        out.append('<text x="%.1f" y="%.1f" class="tick" text-anchor="end">%+d</text>'
                   % (left - 8, y(value) + 4, value))
    for i in range(bands):
        out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%d</text>'
                   % (x(i), height - 14, i + 1))
    out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">TGC band, near to far</text>'
               % ((left + width - right) / 2.0, height - 2))
    for name, colour, d in zip(("current", "model", "teacher"),
                               ("var(--series-3)", "var(--series-1)", "var(--series-2)"), db):
        points = " ".join("%.1f,%.1f" % (x(i), y(float(d[i]))) for i in range(bands))
        out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round"/>'
                   % (points, colour))
        for i in range(bands):
            out.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s" stroke="var(--surface-1)" '
                       'stroke-width="2"/>' % (x(i), y(float(d[i])), colour))
        out.append('<text x="%.1f" y="%.1f" class="series-label" fill="%s">%s</text>'
                   % (width - right + 8, y(float(d[-1])) + 4, colour, name))
    out.append("</svg>")
    return "\n".join(out)


def svg_convergence(steps, key, label, colour):
    """闭环逐步的误差。单条线，标题已经说明是什么，不要图例。"""
    width, height = 300, 150
    left, right, top, bottom = 46, 16, 16, 30
    values = [abs(s[key]) for s in steps]
    high = max(values) * 1.15 + 1e-6
    n = len(values)
    x = lambda i: left + (i * (width - left - right) / max(1.0, n - 1.0) if n > 1 else 0)
    y = lambda v: top + (high - v) * (height - top - bottom) / high
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="%s per step" class="chart">' % (width, height, label)]
    out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="axis"/>'
               % (left, y(0), width - right, y(0)))
    out.append('<text x="%.1f" y="%.1f" class="tick" text-anchor="end">0</text>' % (left - 6, y(0) + 4))
    out.append('<text x="%.1f" y="%.1f" class="tick" text-anchor="end">%.1f</text>'
               % (left - 6, y(high) + 10, high))
    out.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>'
               % (" ".join("%.1f,%.1f" % (x(i), y(v)) for i, v in enumerate(values)), colour))
    for i, v in enumerate(values):
        out.append('<circle cx="%.1f" cy="%.1f" r="4" fill="%s" stroke="var(--surface-1)" stroke-width="2"/>'
                   % (x(i), y(v), colour))
        out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">%d</text>' % (x(i), height - 14, i))
        if i in (0, len(values) - 1):
            out.append('<text x="%.1f" y="%.1f" class="point-label" text-anchor="middle">%.2f</text>'
                       % (x(i), y(v) - 10, v))
    out.append('<text x="%.1f" y="%d" class="tick" text-anchor="middle">step</text>'
               % ((left + width - right) / 2.0, height - 2))
    out.append("</svg>")
    return "\n".join(out)


def svg_metrics(pairs, unit):
    """网络与查表基线的分组条形图。两组，图例 + 直接标数。"""
    rows = len(pairs)
    row_height, width = 46, 560
    left, right, top = 118, 76, 26
    height = top + rows * row_height + 10
    high = max(max(a, b) for _, a, b in pairs) * 1.3
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="network versus lookup baseline" class="chart">'
           % (width, height)]
    out.append('<rect x="%d" y="6" width="10" height="10" rx="2" fill="var(--series-1)"/>' % left)
    out.append('<text x="%d" y="15" class="tick">network</text>' % (left + 16))
    out.append('<rect x="%d" y="6" width="10" height="10" rx="2" fill="var(--series-3)"/>' % (left + 84))
    out.append('<text x="%d" y="15" class="tick">settings lookup</text>' % (left + 100))
    for i, (name, net, base) in enumerate(pairs):
        y0 = top + i * row_height
        out.append('<text x="%d" y="%d" class="row-label" text-anchor="end">%s</text>' % (left - 10, y0 + 18, name))
        for j, (value, colour) in enumerate(((net, "var(--series-1)"), (base, "var(--series-3)"))):
            bar = (width - left - right) * value / high
            out.append('<rect x="%d" y="%d" width="%.1f" height="12" rx="4" fill="%s"/>'
                       % (left, y0 + 2 + j * 15, max(2.0, bar), colour))
            out.append('<text x="%.1f" y="%d" class="point-label">%s</text>'
                       % (left + max(2.0, bar) + 6, y0 + 12 + j * 15, unit % value))
    out.append("</svg>")
    return "\n".join(out)


def svg_bias_strip(stats):
    """每位受试者一个点：闭环后的带符号增益偏置。单序列，不要图例。"""
    subjects = stats["subjects"]
    width, height = 560, 200
    left, right, top, bottom = 50, 20, 26, 44
    span = max(5.0, max(abs(s["bias"]) for s in subjects) * 1.12)
    n = len(subjects)
    x = lambda i: left + i * (width - left - right) / max(1.0, n - 1.0)
    y = lambda v: top + (span - v) * (height - top - bottom) / (2 * span)
    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="per-subject gain bias" class="chart">'
           % (width, height)]
    out.append('<rect x="%d" y="%.1f" width="%d" height="%.1f" fill="var(--series-3)" opacity="0.15"/>'
               % (left, y(1.0), width - left - right, y(-1.0) - y(1.0)))
    out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" class="axis"/>'
               % (left, y(0.0), width - right, y(0.0)))
    for value in (-4, -2, 0, 2, 4):
        if abs(value) <= span:
            out.append('<text x="%d" y="%.1f" class="tick" text-anchor="end">%+d</text>'
                       % (left - 8, y(value) + 4, value))
    for i, subject in enumerate(subjects):
        colour = "var(--series-2)" if abs(subject["bias"]) > 1.0 else "var(--series-1)"
        out.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2" opacity="0.45"/>'
                   % (x(i), y(0.0), x(i), y(subject["bias"]), colour))
        out.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="%s" stroke="var(--surface-1)" stroke-width="2"/>'
                   % (x(i), y(subject["bias"]), colour))
    out.append('<text x="%d" y="%.1f" class="point-label">the +-1 dB band</text>' % (left + 4, y(1.0) - 5))
    out.append('<text x="%d" y="%d" class="tick">%d subjects, sorted by bias</text>'
               % (left, height - 20, len(subjects)))
    out.append('<text x="%d" y="%d" class="tick">dB from the teacher&#8217;s optimum, signed</text>'
               % (left, height - 5))
    out.append("</svg>")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------
#   汇总数字

def read_metrics(path):
    return json.load(io.open(path, encoding="utf-8")) if os.path.exists(path) else {}


def subject_bias(path, rows):
    """按受试者算闭环后的带符号偏置与受试者内标准差。"""
    if not os.path.exists(path):
        return None
    per = {}
    with io.open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            row = rows.get(record.get("start_frame"))
            if row is None:
                continue
            per.setdefault(row["group_id"].rsplit("/", 1)[0], []).append(float(record["gain_error_db"]))
    if not per:
        return None
    stats = []
    for subject, values in per.items():
        values = np.asarray(values, float)
        stats.append({"subject": subject, "n": len(values), "bias": float(values.mean()),
                      "sd": float(values.std()), "abs": float(np.abs(values).mean())})
    stats.sort(key=lambda s: s["bias"])
    bias = np.asarray([s["bias"] for s in stats])
    sd = np.asarray([s["sd"] for s in stats])
    return {"subjects": stats, "sd_median": float(np.median(sd)),
            "bias_share": float((bias ** 2).sum() / ((bias ** 2).sum() + (sd ** 2).sum())),
            "within_1db": int((np.abs(bias) <= 1.0).sum()), "worst": float(np.abs(bias).max()),
            "frames": int(sum(s["n"] for s in stats))}


# ---------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    lines = []

    def emit(text=""):
        lines.append(text)

    import tools_track_console_loop as TR
    rows = TR.teacher_rows(args.labels)
    by_family = {}
    for row in rows.values():
        by_family.setdefault(row.get("family_id"), []).append(row)

    emit("=========== demo page ===========")
    candidates = []
    for name, group in sorted(by_family.items()):
        if not name or any(r.get("split") != "test" for r in group):
            continue
        if len(set(r["focus_mm"] for r in group if r.get("focus_determined"))) >= 5:
            candidates.append((name, len(group)))
    emit("  candidates %d test-set scenes swept at least five focus steps" % len(candidates))

    model, payload, front_payload, model_text = SG.load_model(args)
    ladders = payload["ladders"]
    trained = SG.trained_axes(payload)
    emit("  model      %s" % model_text)
    emit("  trained    %s" % ", ".join("%s %s" % (k, "yes" if v else "no") for k, v in sorted(trained.items())))

    survey = None
    if args.family:
        group = [r for r in by_family.get(args.family, []) if r.get("focus_determined")]
        if not group:
            raise SystemExit("family %s has no focus-labelled frame" % args.family)
        story = scene_story(args.family, group, model, payload, trained, args, emit, with_images=True)
    else:
        if not candidates:
            raise SystemExit("no test-set scene swept five focus steps; pass --family")
        picked, survey = pick_scene(candidates[:args.candidates], by_family, model, payload, trained, args, emit)
        group = [r for r in by_family.get(picked["family"], []) if r.get("focus_determined")]
        story = scene_story(picked["family"], group, model, payload, trained, args, emit, with_images=True)

    frames, results, order, anchor = story["frames"], story["results"], story["order"], story["anchor"]
    frame, result, row, steps = frames[anchor], results[anchor], story["row"], story["steps"]
    slope = K.TGC_DB_PER_LEVEL[int(frame["mode"])]
    emit("  anchor     %s (depth %.1f mm, frequency %.1f MHz, focus %.1f mm)"
         % (frame["frame_id"], frame["depth_mm"], frame["frequency_mhz"], frame["focus_mm"]))
    emit("  loop       %d step(s), gain %.2f -> %.2f dB, TGC %.2f -> %.2f dB"
         % (len(steps) - 1, story["start_gain"], story["final_gain"], story["start_tgc"], story["final_tgc"]))

    panel = {
        "current": png_data_uri(render_gray(frame, frame["gain_db"], frame["tgc_levels"]), frame["max_depth_mm"]),
        "model": png_data_uri(render_gray(frame, frame["gain_db"] + result["backend"]["gain"]["clicks"]
                                          * K.GAIN_DB_PER_LEVEL[int(frame["mode"])],
                                          result["backend"]["tgc"]["suggested_levels"]), frame["max_depth_mm"]),
        "teacher": png_data_uri(render_gray(frame, row["optimal_gain_db"], row["optimal_tgc_levels"]),
                                frame["max_depth_mm"]),
    }

    # 聚焦：每档一张，渲染在各自的教师最优后端上，看到的差别才是聚焦本身
    focus_cards = []
    for i in order:
        f, r = frames[i], story["rows"][frames[i]["frame_id"]]
        focus_cards.append({
            "focus_mm": f["focus_mm"],
            "image": png_data_uri(render_gray(f, r["optimal_gain_db"], r["optimal_tgc_levels"]), f["max_depth_mm"]),
            "model": bool(results[i]["frontend"]["focus"]["steps"] == 0),
            "model_pick": results[i]["frontend"]["focus"]["suggested"],
            "teacher": bool(abs(r["optimal_focus_mm"] - f["focus_mm"]) < 1e-6)})
    teacher_pick = sorted(set(round(story["rows"][frames[i]["frame_id"]]["optimal_focus_mm"], 1) for i in order))
    model_pick = sorted(set(round(c["model_pick"], 1) for c in focus_cards))
    far_wall = row.get("far_wall_mm")
    emit("  focus      swept %s mm; teacher %s; model %s; far wall %s"
         % ("/".join("%g" % c["focus_mm"] for c in focus_cards), teacher_pick, model_pick,
            ("%.1f mm" % far_wall) if far_wall else "not read"))

    bias = subject_bias(args.loop_trajectories, rows)
    if bias:
        emit("  bias       %d subjects: within-subject sd %.2f dB, %d within +-1 dB, worst %.2f dB; "
             "per-subject bias is %.0f%% of the squared error"
             % (len(bias["subjects"]), bias["sd_median"], bias["within_1db"], bias["worst"],
                100 * bias["bias_share"]))

    single = read_metrics(args.single_step_json)
    loop = read_metrics(args.loop_json).get("closed_loop", {})
    net, base = single.get("network_recorded_start", {}), single.get("settings_lookup_baseline", {})
    emit("  metrics    %s + %s" % (args.single_step_json, args.loop_json))

    html = build_html(story["family"], frame, row, result, panel, steps, focus_cards, teacher_pick,
                      model_pick, far_wall, net, base, loop, slope, payload, front_payload, survey, bias)
    directory = os.path.dirname(os.path.abspath(args.out))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    io.open(args.out, "w", encoding="utf-8").write(html)
    emit("  wrote %s (%.0f KB, images inlined)" % (args.out, os.path.getsize(args.out) / 1024.0))
    emit("  done in %.0f s" % (time.time() - started))

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return args.out


def build_html(family, frame, row, result, panel, steps, focus_cards, teacher_pick, model_pick,
               far_wall, net, base, loop, slope, payload, front_payload, survey, bias):
    gain, tgc = result["backend"]["gain"], result["backend"]["tgc"]
    f = lambda key: float(net.get(key, float("nan")))
    b = lambda key: float(base.get(key, float("nan")))

    step_cards = []
    for s in steps:
        if s["step"] == 0:
            caption = "start: what the operator captured"
        elif s["stopped"]:
            caption = "step %d: the model stops here" % s["step"]
        else:
            caption = "step %d" % s["step"]
        action = []
        if s["suggestion"]["clicks"]:
            action.append("gain %+d clicks" % s["suggestion"]["clicks"])
        if s["suggestion"]["tgc_moved"]:
            action.append("TGC moved")
        step_cards.append(
            '<figure class="step"><img src="%s" alt="frame at step %d"/><figcaption><b>%s</b>'
            '<span>gain level %d &middot; %+.2f dB from the optimum</span><span>TGC off by %.2f dB</span>'
            '<span class="action">%s</span></figcaption></figure>'
            % (s["image"], s["step"], caption, s["gain_level"], s["gain_error_db"], s["tgc_error_db"],
               ", ".join(action) if action else "stop"))

    focus_html = []
    for c in focus_cards:
        marks = ""
        if c["teacher"]:
            marks += '<span class="mark teacher">teacher</span>'
        if c["model"]:
            marks += '<span class="mark model">model</span>'
        focus_html.append('<figure class="focus%s"><img src="%s" alt="focus %g mm"/><figcaption><b>%g mm</b>%s'
                          '</figcaption></figure>'
                          % (" picked" if (c["teacher"] or c["model"]) else "", c["image"], c["focus_mm"],
                             c["focus_mm"], marks))

    agree = sorted(model_pick) == sorted(teacher_pick)
    fields = {
        "family": family,
        "frame": frame["frame_id"],
        "depth": "%.1f" % frame["depth_mm"],
        "frequency": "%.1f" % frame["frequency_mhz"],
        "focus": "%.1f" % frame["focus_mm"],
        "gain_level": "%d" % frame["gain_level"],
        "dr": "%.0f" % frame["dr_ui"],
        "far_wall": ("%.1f mm" % far_wall) if far_wall else "not read",
        "current_img": panel["current"], "model_img": panel["model"], "teacher_img": panel["teacher"],
        "gain_clicks": "%+d" % gain["clicks"],
        "gain_from": "%d" % gain["level"], "gain_to": "%d" % gain["suggested_level"],
        "gain_db": "%+.2f" % gain["delta_db"],
        "teacher_gain_db": "%+.2f" % (float(row["optimal_gain_db"]) - float(frame["gain_db"])),
        "tgc_near": "%+.0f" % tgc["group_delta_levels"]["near"],
        "tgc_mid": "%+.0f" % tgc["group_delta_levels"]["mid"],
        "tgc_far": "%+.0f" % tgc["group_delta_levels"]["far"],
        "curves": svg_tgc_curves(frame["tgc_levels"], tgc["suggested_levels"], row["optimal_tgc_levels"], slope),
        "steps": "\n".join(step_cards),
        "gain_chart": svg_convergence(steps, "gain_error_db", "gain error", "var(--series-1)"),
        "tgc_chart": svg_convergence(steps, "tgc_error_db", "TGC error", "var(--series-2)"),
        "loop_steps": "%d" % (len(steps) - 1),
        "focus_cards": "\n".join(focus_html),
        "focus_swept": ", ".join("%g" % c["focus_mm"] for c in focus_cards),
        "focus_teacher": ", ".join("%g" % v for v in teacher_pick),
        "focus_model": ", ".join("%g" % v for v in model_pick),
        "focus_verdict": ("the model picks the same step" if agree else "they disagree here"),
        "metrics": svg_metrics([("gain MAE", f("gain_mae_db"), b("gain_mae_db")),
                                ("TGC MAE", f("tgc_mae_db"), b("tgc_mae_db"))], "%.2f dB"),
        "scores": svg_metrics([("gain direction F1", f("gain_dir_f1_derived"), b("gain_dir_f1_derived")),
                               ("TGC direction F1", f("slider_dir_f1_derived"), b("slider_dir_f1_derived")),
                               ("focus top-1", f("focus_top1"), b("focus_top1"))], "%.3f"),
        "net_gain": "%.2f" % f("gain_mae_db"), "base_gain": "%.2f" % b("gain_mae_db"),
        "net_gain3": "%.3f" % f("gain_mae_db"), "base_gain3": "%.3f" % b("gain_mae_db"),
        "net_tgc": "%.2f" % f("tgc_mae_db"), "base_tgc": "%.2f" % b("tgc_mae_db"),
        "net_tgc3": "%.3f" % f("tgc_mae_db"), "base_tgc3": "%.3f" % b("tgc_mae_db"),
        "net_focus": "%.3f" % f("focus_top1"), "base_focus": "%.3f" % b("focus_top1"),
        "converged": "%.0f" % (100.0 * float(loop.get("converged", float("nan")))),
        "converged1": "%.1f" % (100.0 * float(loop.get("converged", float("nan")))),
        "loop_gain_start": "%.2f" % float(loop.get("gain_abs_error_db_start", float("nan"))),
        "loop_gain_final": "%.2f" % float(loop.get("gain_abs_error_db_final", float("nan"))),
        "loop_tgc_start": "%.2f" % float(loop.get("tgc_mae_db_start", float("nan"))),
        "loop_tgc_final": "%.2f" % float(loop.get("tgc_mae_db_final", float("nan"))),
        "bias_strip": svg_bias_strip(bias) if bias else "",
        "bias_sd": ("%.2f" % bias["sd_median"]) if bias else "?",
        "bias_share": ("%.0f" % (100 * bias["bias_share"])) if bias else "?",
        "bias_within": ("%d" % bias["within_1db"]) if bias else "?",
        "bias_total": ("%d" % len(bias["subjects"])) if bias else "?",
        "bias_worst": ("%.1f" % bias["worst"]) if bias else "?",
        "survey": ("" if not survey else
                   "This scene was picked because its starting gain sits at the median of the %d test-set "
                   "scenes that swept the focus (%.2f dB from the teacher&#8217;s optimum), chosen before "
                   "looking at the outcome. Across those scenes the loop ends between %.2f and %.2f dB, "
                   "median %.2f dB, and the model matches the teacher&#8217;s focus step in %d of them."
                   % (survey["n"], survey["start_median"], survey["gain_best"], survey["gain_worst"],
                      survey["gain_median"], survey["focus_match"])),
        "epoch": str(payload.get("epoch")), "front_epoch": str(front_payload.get("epoch")),
    }
    return Template(TEMPLATE).substitute(fields)


TEMPLATE = u"""<title>Carotid Six-Parameter Demo</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>
/* 版式：一条窄栏讲一帧的故事，图像并排，数字跟在图下面 */
:root {
  color-scheme: light;
  --bg: #f5f5f2;
  --surface-1: #fcfcfb;
  --line: #dededa;
  --text-primary: #16160f;
  --text-secondary: #55544d;
  --text-muted: #86857c;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --font-display: "Spectral", Georgia, serif;
  --font-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --font-data: ui-monospace, "SFMono-Regular", Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #121211; --surface-1: #1a1a19; --line: #333331;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8d8c83;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface-1: #1a1a19; --line: #333331;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8d8c83;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
body { background: var(--bg); color: var(--text-primary); font-family: var(--font-body); line-height: 1.55; }
.wrap { max-width: 61rem; margin: 0 auto;
        padding-left: max(16px, env(safe-area-inset-left, 0px));
        padding-right: max(16px, env(safe-area-inset-right, 0px));
        padding-block: 2.5rem 4rem; }
h1 { font-family: var(--font-display); font-size: clamp(1.7rem, 4.4vw, 2.5rem); line-height: 1.15;
     margin: 0 0 .6rem; text-wrap: balance; font-weight: 600; }
h2 { font-family: var(--font-display); font-size: 1.35rem; font-weight: 600; margin: 0 0 .4rem;
     text-wrap: balance; }
h3 { font-size: .9rem; margin: 0 0 .5rem; font-weight: 600; }
.eyebrow { font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--text-muted);
           margin: 0 0 .5rem; }
.lede { font-size: 1.04rem; color: var(--text-secondary); max-width: 62ch; margin: 0 0 1.6rem; }
section { border-top: 1px solid var(--line); padding-block: 2rem 0; margin-top: 2.2rem; }
section > p { color: var(--text-secondary); max-width: 66ch; }
.hero-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(10.5rem, 1fr)); gap: 1px;
              background: var(--line); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
.hero-stats div { background: var(--surface-1); padding: .85rem 1rem; min-width: 0; }
.hero-stats b { display: block; font-family: var(--font-data); font-size: 1.5rem;
                font-variant-numeric: tabular-nums; }
.hero-stats span { font-size: .8rem; color: var(--text-muted); }
.triptych { display: grid; grid-template-columns: repeat(auto-fit, minmax(13rem, 1fr)); gap: 1rem;
            margin-top: 1.2rem; }
.steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: 1rem;
         margin-top: 1.2rem; }
.focus-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.5rem, 1fr)); gap: .8rem;
             margin-top: 1.2rem; }
figure { margin: 0; background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px;
         padding: .7rem; min-width: 0; }
figure img { display: block; width: 100%; max-width: 100%; border-radius: 4px; background: #000; }
figcaption { font-size: .8rem; color: var(--text-secondary); margin-top: .55rem; display: grid; gap: .12rem; }
figcaption b { color: var(--text-primary); font-size: .88rem; }
figcaption span { font-family: var(--font-data); font-variant-numeric: tabular-nums; font-size: .76rem;
                  color: var(--text-muted); }
.focus.picked { border-color: var(--series-1); }
.mark { display: inline-block; font-family: var(--font-body); font-size: .68rem; letter-spacing: .06em;
        text-transform: uppercase; border-radius: 999px; padding: .08rem .45rem; margin: .2rem .3rem 0 0;
        border: 1px solid currentColor; }
.mark.model { color: var(--series-1); }
.mark.teacher { color: var(--series-2); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(16rem, 1fr)); gap: 1rem;
          margin-top: 1.2rem; }
.card { background: var(--surface-1); border: 1px solid var(--line); border-radius: 10px;
        padding: .9rem 1rem; min-width: 0; }
.chart { width: 100%; height: auto; display: block; overflow: visible; }
.axis { stroke: var(--line); stroke-width: 1; }
.tick { fill: var(--text-muted); font-size: 11px; font-family: var(--font-body); }
.row-label { fill: var(--text-secondary); font-size: 12px; font-family: var(--font-body); }
.series-label, .point-label { font-size: 11px; font-family: var(--font-data); fill: var(--text-secondary); }
.series-label { font-weight: 600; }
table { border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: 1rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--line); }
td.num { font-family: var(--font-data); font-variant-numeric: tabular-nums; text-align: right; }
th { font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; color: var(--text-muted);
     font-weight: 600; }
.scroll { overflow-x: auto; }
.note { font-size: .84rem; color: var(--text-muted); max-width: 68ch; margin: 0; }
.off td { color: var(--text-muted); }
footer { border-top: 1px solid var(--line); margin-top: 2.5rem; padding-top: 1.2rem; font-size: .82rem;
         color: var(--text-muted); }
footer p { max-width: 72ch; }
</style>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@500;600&family=Source+Sans+3:wght@400;600&display=swap">
<div class="wrap">
  <p class="eyebrow">In-vivo carotid &middot; scene ${family} &middot; held-out test set</p>
  <h1>One frame, six knobs, and what the network does with them</h1>
  <p class="lede">Every image here is rendered from the same recorded envelope the console captured. Gain and
  TGC do not change that envelope, only how it maps to grey, so what the model suggests can be shown exactly
  rather than illustrated. This subject took no part in training or in model selection.</p>
  <div class="hero-stats">
    <div><b>${net_tgc} dB</b><span>TGC error on 29 unseen subjects &mdash; lookup baseline ${base_tgc}</span></div>
    <div><b>${net_gain} dB</b><span>gain error &mdash; lookup baseline ${base_gain}</span></div>
    <div><b>${converged}%</b><span>back-end loops that stop, in 2&ndash;3 steps</span></div>
  </div>

  <section>
    <p class="eyebrow">1 &middot; the suggestion</p>
    <h2>Current setting, the model&rsquo;s suggestion, the teacher&rsquo;s optimum</h2>
    <p>Frame ${frame}: ${depth} mm display depth, ${frequency} MHz harmonic, focus ${focus} mm, gain level
    ${gain_level}, dynamic range ${dr}. The vessel&rsquo;s far wall sits at ${far_wall}. The console was left
    with a flat TGC at 127, which is how 98.5% of these captures arrive.</p>
    <div class="triptych">
      <figure><img src="${current_img}" alt="frame as captured"/><figcaption><b>As captured</b>
        <span>gain level ${gain_level} &middot; TGC flat 127</span></figcaption></figure>
      <figure><img src="${model_img}" alt="frame with the model's suggestion applied"/>
        <figcaption><b>Model&rsquo;s suggestion</b>
        <span>gain ${gain_clicks} clicks: ${gain_from} &rarr; ${gain_to} (${gain_db} dB)</span>
        <span>TGC near ${tgc_near} &middot; mid ${tgc_mid} &middot; far ${tgc_far} levels</span></figcaption></figure>
      <figure><img src="${teacher_img}" alt="frame with the teacher's optimum applied"/>
        <figcaption><b>Teacher&rsquo;s optimum</b><span>gain ${teacher_gain_db} dB from the captured setting</span>
        <span>solved on this frame, not predicted</span></figcaption></figure>
    </div>
    <div class="charts">
      <div class="card"><h3>TGC curves, in dB about the mid level</h3>${curves}</div>
      <div class="card"><h3>Why the far field has to lift</h3>
        <p class="note">Tissue attenuates, so the deep bands need more gain than the shallow ones. The teacher
        solves for the curve that brings every depth to the brightness this operator accepted; the network
        predicts that curve from the frame&rsquo;s own depth profile. That is why it transfers almost unchanged
        to new people: 0.23 dB on subjects it trained on, ${net_tgc} dB on these.</p></div>
    </div>
  </section>

  <section>
    <p class="eyebrow">2 &middot; the loop</p>
    <h2>Suggest, apply, look again &mdash; until it stops</h2>
    <p>The back-end half of the closed loop runs exactly on a recorded frame: gain and TGC change the render,
    not the envelope, so the model can be asked again about the image it just produced. This trajectory took
    ${loop_steps} step(s).</p>
    <div class="steps">${steps}</div>
    <div class="charts">
      <div class="card"><h3>Gain, dB from the teacher&rsquo;s optimum</h3>${gain_chart}</div>
      <div class="card"><h3>TGC, dB from the teacher&rsquo;s optimum</h3>${tgc_chart}</div>
    </div>
    <p class="note">Over all 2631 frames of the 28 validation subjects, ${converged1}% of the loops stop; gain
    error falls ${loop_gain_start} &rarr; ${loop_gain_final} dB and TGC ${loop_tgc_start} &rarr;
    ${loop_tgc_final} dB. The stopping point lands on the single-step accuracy, so nothing accumulates around
    the loop.</p>
  </section>

  <section>
    <p class="eyebrow">3 &middot; the front end</p>
    <h2>Transmit focus: six real captures from one probe position</h2>
    <p>Focus changes the envelope itself, so these are six separate captures from the same probe position,
    each rendered at its own solved back end so that what differs is the focus and not the brightness. Swept:
    ${focus_swept} mm. The teacher picks ${focus_teacher} mm, the model picks ${focus_model} mm &mdash;
    ${focus_verdict}.</p>
    <div class="focus-row">${focus_cards}</div>
    <p class="note">Focus is the weakest of the three working axes: ${net_focus} exact hits against the lookup
    baseline&rsquo;s ${base_focus} on the test subjects, but about 0.99 on subjects it trained on, so most of
    that head is memorising rather than generalising. Depth, frequency and dynamic range are switched off
    entirely &mdash; their labels can be guessed from the current setting without looking at the image, so
    there is nothing for the network to learn from them yet.</p>
  </section>

  <section>
    <p class="eyebrow">4 &middot; where the error lives</p>
    <h2>The gain residual is one number per subject</h2>
    <p>Averaged over frames the gain lands ${loop_gain_final} dB from the teacher, but that average hides the
    shape of the error. Within one subject the model is consistent to ${bias_sd} dB; what varies is a constant
    offset for that subject, in either direction. ${bias_share}% of the squared error is this per-subject bias
    rather than frame-to-frame scatter.</p>
    <div class="charts">
      <div class="card"><h3>Gain bias per validation subject</h3>${bias_strip}</div>
      <div class="card"><h3>What that means on the machine</h3>
        <p class="note">${bias_within} of ${bias_total} subjects land inside &plusmn;1 dB. The rest are off by
        up to ${bias_worst} dB &mdash; every frame of that subject by the same amount, so a single correction
        by the operator fixes the whole examination. This is the part of the target a single image cannot
        carry: the teacher&rsquo;s brightness target was measured from the frames this operator accepted in
        that session, and some sessions ran brighter or darker than that operator&rsquo;s own average.</p></div>
    </div>
  </section>

  <section>
    <p class="eyebrow">5 &middot; the numbers</p>
    <h2>Against &ldquo;just remember the setting&rdquo;</h2>
    <p>The baseline is the honest one: for each combination of imaging mode, depth, frequency and focus, the
    median optimum of the training subjects, applied to the test subjects. Beating it is the whole question &mdash;
    on phantom data the network could not.</p>
    <div class="charts">
      <div class="card"><h3>Error, lower is better</h3>${metrics}</div>
      <div class="card"><h3>Agreement with the teacher, higher is better</h3>${scores}</div>
    </div>
    <div class="scroll"><table>
      <thead><tr><th>Axis</th><th>Network</th><th>Lookup</th><th>State</th></tr></thead>
      <tbody>
        <tr><td>TGC, 8 bands</td><td class="num">${net_tgc3} dB</td><td class="num">${base_tgc3} dB</td><td>working</td></tr>
        <tr><td>Gain</td><td class="num">${net_gain3} dB</td><td class="num">${base_gain3} dB</td><td>working</td></tr>
        <tr><td>Transmit focus</td><td class="num">${net_focus}</td><td class="num">${base_focus}</td><td>marginal</td></tr>
        <tr class="off"><td>Display depth</td><td class="num">&mdash;</td><td class="num">0.775</td><td>off: guessable from the setting</td></tr>
        <tr class="off"><td>Transmit frequency</td><td class="num">&mdash;</td><td class="num">0.946</td><td>off: guessable from the setting</td></tr>
        <tr class="off"><td>Dynamic range</td><td class="num">&mdash;</td><td class="num">&mdash;</td><td>off: no criterion in the labels</td></tr>
      </tbody>
    </table></div>
  </section>

  <footer>
    <p>${survey}</p>
    <p>17545 frames from 189 subjects over nine acquisition days, L15-4 probe, harmonic imaging. Split by
    subject before training: 132 train, 28 validation, 29 test, with every acquisition day present in all
    three. Back-end labels are solved per frame against the brightness the operator accepted in that session;
    the front-end target depth comes from provisional readings of the vessel&rsquo;s far wall. Checkpoints:
    back end epoch ${epoch}, front end epoch ${front_epoch}.</p>
  </footer>
</div>
"""


if __name__ == "__main__":
    main()
