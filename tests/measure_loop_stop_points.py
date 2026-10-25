# -*- coding: utf-8 -*-
"""闭环停点为什么比单步差：差在哪一轴、哪个方向、哪一类轨迹。

    问题

用户看到的是闭环停下来时的那幅图，不是单步指标。而这两者在 fieldii_final 的测试集上差得很远：

    聚焦最优档   单步 0.836  ->  停点 0.499
    频率可接受   单步 0.802  ->  停点 0.556
    深度最优档   单步 0.542  ->  停点 0.789
    三轴全对                     停点 0.195

深度反而变好、聚焦与频率变差，说明这不是「模型不准」一句话能解释的。本脚本只读已有的
closed_loop_trajectories_*.jsonl 与 evaluation_*.json，不重跑模型，回答四个问题：

  1. 单步与停点逐轴对照，并按轨迹类别（停下 / 没停下 / 打转 / 限幅）拆开——短板集中在哪一类。
  2. 停点的带符号档差分布：正数 = 教师还要往深 / 往高，即停早了；负数 = 走过头了。
  3. 闭环到底有没有把前端调好：同一条轨迹起点离最优几档 -> 停点离最优几档（要 --labels）。
     这是最要紧的一张表——如果停点并不比起点近，闭环在前端上就是白走。
  4. 「停点分数」（三轴停点正确率的平均）与现在选检查点用的 frontend_score_near 排序是否一致。
     不一致就说明选模指标选错了对象。

    用法

    python tests/measure_loop_stop_points.py --run runs/fieldii_final
    python tests/measure_loop_stop_points.py --run runs/fieldii_v5 --tag heldout_test
    python tests/measure_loop_stop_points.py --trajectories data/console_loops/x_trajectory.jsonl
    python tests/measure_loop_stop_points.py --compare runs/fieldii_v3 runs/fieldii_v5 runs/fieldii_final

--trajectories 也吃 tools_track_console_loop.py 写出的实机轨迹（同一套字段）；实机轨迹没有
教师标签时，第 1、2、3 节自动跳过，只报导航行为（步数、改档次数、打转、是否停下）。

报告写在脚本旁边的 measure_loop_stop_points.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import glob
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
AXES = ("depth", "frequency", "focus")
# 单步指标里与停点正确率对应的那一项：前端三轴的单步命中口径
SINGLE_STEP_KEY = {"depth": "depth_top1", "frequency": "frequency_hit", "focus": "focus_top1"}
LINES = []


def emit(text=""):
    LINES.append(str(text))


def read_jsonl(path):
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------------------
#   找文件

def find_trajectories(run, tag=None):
    """运行目录下的 (折名, 轨迹文件, 评估 json) 列表。"""
    out = []
    for fold in sorted(glob.glob(os.path.join(run, "fold_*"))) + [os.path.join(run, "all_data")]:
        if not os.path.isdir(fold):
            continue
        for path in sorted(glob.glob(os.path.join(fold, "closed_loop_trajectories_*.jsonl"))):
            name = os.path.basename(path)[len("closed_loop_trajectories_"):-len(".jsonl")]
            if tag and tag not in name:
                continue
            evaluation = os.path.join(fold, "evaluation_%s.json" % name)
            out.append((os.path.basename(fold), name, path,
                        evaluation if os.path.exists(evaluation) else None))
    return out


def pick_tag(found):
    """一个运行目录下有多套闭环设置时，默认取文件最多的那一套（其余用 --tag 选）。"""
    counts = collections.Counter(name for _, name, _, _ in found)
    return counts.most_common(1)[0][0] if counts else None


# ---------------------------------------------------------------------------------------
#   起点与停点的距离（需要标签）

def label_index(labels_path):
    """frame_id -> (每轴当前档下标, 每轴最优档下标)。档位表由标签里出现过的值构成。"""
    if not labels_path or not os.path.exists(labels_path):
        return None
    rows = read_jsonl(labels_path)
    ladders = {}
    for axis, key in (("depth", "depth_mm"), ("frequency", "frequency_mhz"), ("focus", "focus_mm")):
        values = set()
        for row in rows:
            values.add(float(row[key]))
            optimal = row.get("optimal_%s" % key)
            if optimal is not None:
                values.add(float(optimal))
            for v in (row.get("%s_ladder" % axis) or []):
                values.add(float(v))
        ladders[axis] = sorted(values)
    index = {}
    for row in rows:
        entry = {}
        for axis, key in (("depth", "depth_mm"), ("frequency", "frequency_mhz"), ("focus", "focus_mm")):
            if not row.get("%s_determined" % axis):
                continue
            optimal = row.get("optimal_%s" % key)
            if optimal is None:
                continue
            ladder = ladders[axis]
            here = int(np.argmin(np.abs(np.asarray(ladder) - float(row[key]))))
            best = int(np.argmin(np.abs(np.asarray(ladder) - float(optimal))))
            entry[axis] = (here, best)
        index[row["frame_id"]] = entry
    return index


# ---------------------------------------------------------------------------------------
#   各节

def section_single_vs_stop(records, evaluation):
    """1. 单步与停点逐轴对照，并按轨迹类别拆开。"""
    emit("=========== 1. single step vs stop point ===========")
    single = {}
    if evaluation:
        data = json.load(io.open(evaluation, encoding="utf-8")).get("network_recorded_start", {})
        single = {axis: data.get(SINGLE_STEP_KEY[axis]) for axis in AXES}
    have = [r for r in records if r.get("frontend_correct")]
    if not have:
        emit("  no trajectory carries teacher labels; skipped")
        emit("")
        return {}
    stop = {}
    emit("    axis        single step   stop point   within 1 step   mean signed steps (+ = the teacher wants more)")
    for axis in AXES:
        values = [r["frontend_correct"][axis] for r in have if axis in r["frontend_correct"]]
        steps = [r["frontend_steps_to_optimum"][axis] for r in have
                 if axis in r.get("frontend_steps_to_optimum", {})]
        if not values:
            continue
        stop[axis] = float(np.mean(values))
        emit("    %-10s %11s   %10.3f   %13.3f   %+.3f"
             % (axis, ("%.3f" % single[axis]) if single.get(axis) is not None else "-", stop[axis],
                float(np.mean([abs(v) <= 1 for v in steps])) if steps else float("nan"),
                float(np.mean(steps)) if steps else float("nan")))
    all_correct = [all(r["frontend_correct"].values()) for r in have]
    emit("    %-10s %11s   %10.3f" % ("all three", "-", float(np.mean(all_correct))))
    emit("")

    groups = [("converged", lambda r: r.get("converged")), ("not converged", lambda r: not r.get("converged")),
              ("oscillated", lambda r: r.get("oscillated")), ("gain clamped", lambda r: r.get("gain_clamped"))]
    emit("    by trajectory type (the deficit is concentrated where the count is large and the score low)")
    emit("    %-14s %6s %8s %8s %8s %9s" % ("type", "n", "depth", "frequency", "focus", "all three"))
    for name, keep in groups:
        subset = [r for r in have if keep(r)]
        if not subset:
            emit("    %-14s %6d" % (name, 0))
            continue
        cells = []
        for axis in AXES:
            values = [r["frontend_correct"][axis] for r in subset if axis in r["frontend_correct"]]
            cells.append(float(np.mean(values)) if values else float("nan"))
        cells.append(float(np.mean([all(r["frontend_correct"].values()) for r in subset])))
        emit("    %-14s %6d %8.3f %8.3f %8.3f %9.3f" % (name, len(subset), cells[0], cells[1], cells[2], cells[3]))
    emit("")
    return stop


def section_signed_steps(records):
    """2. 停点的带符号档差分布。"""
    emit("=========== 2. how far the stop point sits from the teacher's step ===========")
    have = [r for r in records if r.get("frontend_steps_to_optimum")]
    if not have:
        emit("  no trajectory carries teacher labels; skipped")
        emit("")
        return
    emit("    + = the teacher wants a bigger step than the one the loop stopped on (deeper, higher frequency,")
    emit("        deeper focus); - = the loop stopped on a step bigger than the teacher's")
    for axis in AXES:
        steps = [r["frontend_steps_to_optimum"][axis] for r in have if axis in r["frontend_steps_to_optimum"]]
        if not steps:
            continue
        counts = collections.Counter(steps)
        span = sorted(counts)
        histogram = "  ".join("%+d:%d" % (k, counts[k]) for k in span)
        under = float(np.mean([v > 0 for v in steps]))
        over = float(np.mean([v < 0 for v in steps]))
        emit("    %-10s n %4d   mean %+.3f   needs a bigger step %.3f / a smaller one %.3f"
             % (axis, len(steps), float(np.mean(steps)), under, over))
        emit("               %s" % histogram)
    emit("")


def section_start_to_stop(records, index):
    """3. 起点 -> 停点：闭环有没有把前端调近。"""
    emit("=========== 3. did the loop improve the front end at all ===========")
    if index is None:
        emit("  --labels not given or not found; skipped")
        emit("")
        return
    rows = []
    for r in records:
        start = index.get(r.get("start_frame"))
        stop = r.get("frontend_steps_to_optimum")
        if not start or not stop:
            continue
        rows.append((start, stop))
    if not rows:
        emit("  no trajectory could be matched to the labels (frame ids differ?); skipped")
        emit("")
        return
    emit("    distance to the teacher's step, in ladder steps; 'start' is the frame the trajectory began on")
    emit("    %-10s %6s %12s %12s %10s %10s" % ("axis", "n", "start |steps|", "stop |steps|", "better", "worse"))
    for axis in AXES:
        pairs = [(abs(s[axis][1] - s[axis][0]), abs(e[axis])) for s, e in rows if axis in s and axis in e]
        if not pairs:
            continue
        before = np.asarray([p[0] for p in pairs], float)
        after = np.asarray([p[1] for p in pairs], float)
        emit("    %-10s %6d %12.3f %12.3f %10.3f %10.3f"
             % (axis, len(pairs), before.mean(), after.mean(),
                float(np.mean(after < before)), float(np.mean(after > before))))
    emit("    (better / worse = the share of trajectories whose stop point is closer / farther than its start)")
    emit("")


def section_navigation(records):
    """4. 导航行为：步数、改档次数与停点的关系。实机轨迹也能报。"""
    emit("=========== 4. navigation ===========")
    n = len(records)
    emit("    %d trajectory(ies): converged %.3f, oscillated %.3f, frozen %.3f, gain clamped %.3f"
         % (n, float(np.mean([bool(r.get("converged")) for r in records])),
            float(np.mean([bool(r.get("oscillated")) for r in records])),
            float(np.mean([bool(r.get("frozen")) for r in records])),
            float(np.mean([bool(r.get("gain_clamped")) for r in records]))))
    moves = collections.Counter(r.get("frontend_moves", 0) for r in records)
    emit("    front-end moves per trajectory: %s"
         % "  ".join("%d:%d" % (k, moves[k]) for k in sorted(moves)))
    changes = {axis: float(np.mean([r.get("axis_changes", {}).get(axis, 0) for r in records])) for axis in AXES}
    emit("    mean changes per axis: %s" % "  ".join("%s %.2f" % (a, changes[a]) for a in AXES))
    have = [r for r in records if r.get("frontend_correct")]
    if have:
        emit("    stop-point score by how much the front end moved")
        emit("    %-14s %6s %8s %8s %8s %9s" % ("moves", "n", "depth", "frequency", "focus", "all three"))
        for bucket, keep in (("0", lambda m: m == 0), ("1", lambda m: m == 1), ("2", lambda m: m == 2),
                             ("3 or more", lambda m: m >= 3)):
            subset = [r for r in have if keep(r.get("frontend_moves", 0))]
            if not subset:
                continue
            cells = []
            for axis in AXES:
                values = [r["frontend_correct"][axis] for r in subset if axis in r["frontend_correct"]]
                cells.append(float(np.mean(values)) if values else float("nan"))
            cells.append(float(np.mean([all(r["frontend_correct"].values()) for r in subset])))
            emit("    %-14s %6d %8.3f %8.3f %8.3f %9.3f"
                 % (bucket, len(subset), cells[0], cells[1], cells[2], cells[3]))
    # 一步也没做后端修正就走到头的轨迹：曝光整条没调过，最终增益误差全靠运气
    no_backend = [r for r in records if r.get("backend_moves", 0) == 0]
    if no_backend and "gain_error_db" in (no_backend[0] or {}):
        errors = [abs(r["gain_error_db"]) for r in no_backend if "gain_error_db" in r]
        others = [abs(r["gain_error_db"]) for r in records if r.get("backend_moves", 0) > 0
                  and "gain_error_db" in r]
        emit("    trajectories that never corrected the back end: %d of %d, final |gain error| %.2f dB "
             "(the rest %.2f dB)" % (len(no_backend), n, float(np.mean(errors)) if errors else float("nan"),
                                     float(np.mean(others)) if others else float("nan")))
    emit("")


def stop_point_score(records):
    have = [r for r in records if r.get("frontend_correct")]
    if not have:
        return float("nan")
    per_axis = []
    for axis in AXES:
        values = [r["frontend_correct"][axis] for r in have if axis in r["frontend_correct"]]
        if values:
            per_axis.append(float(np.mean(values)))
    return float(np.mean(per_axis)) if per_axis else float("nan")


def section_compare(runs, tag):
    """5. 停点分数与 frontend_score_near 的排序是否一致。

    一个检查点一行：同一个折下若有几套闭环设置（限幅、滞回），它们共用同一个检查点，选模上是
    同一行，取名字最靠前的那一套并注明折叠了几套。不同折、不同运行评估的体模不同，排名只在
    「同一套评估协议」内部才有意义，所以下面还打印每行的轨迹条数与标签口径。
    """
    emit("=========== 5. would the stop point pick a different checkpoint ===========")
    emit("    one row per checkpoint; rows evaluated on different groups are not comparable with each other")
    emit("    %-26s %-24s %6s %10s %10s %10s" % ("run / fold", "tag", "n", "near score", "stop score", "converged"))
    by_checkpoint = collections.OrderedDict()
    for run in runs:
        for fold, name, path, evaluation in find_trajectories(run, tag):
            key = (os.path.basename(run.rstrip("/\\")), fold)
            by_checkpoint.setdefault(key, []).append((name, path, evaluation))
    table = []
    for key, entries in by_checkpoint.items():
        name, path, evaluation = sorted(entries)[0]
        records = read_jsonl(path)
        near = float("nan")
        if evaluation:
            data = json.load(io.open(evaluation, encoding="utf-8")).get("network_recorded_start", {})
            near = float(data.get("frontend_score_near", float("nan")))
        score = stop_point_score(records)
        converged = float(np.mean([bool(r.get("converged")) for r in records])) if records else float("nan")
        label = "%s/%s" % key
        table.append((label, near, score))
        emit("    %-26s %-24s %6d %10.3f %10.3f %10.3f%s"
             % (label, name[:24], len(records), near, score, converged,
                "" if len(entries) == 1 else "   [%d loop settings collapsed]" % len(entries)))
    usable = [t for t in table if np.isfinite(t[1]) and np.isfinite(t[2])]
    near = np.asarray([t[1] for t in usable])
    stop = np.asarray([t[2] for t in usable])
    if len(usable) < 3:
        emit("    fewer than 3 checkpoints; nothing to rank")
    elif near.std() < 1e-6:
        emit("    every row has the same near score (one checkpoint); nothing to rank")
    else:
        best_near = usable[int(np.argmax(near))][0]
        best_stop = usable[int(np.argmax(stop))][0]
        emit("    %d checkpoints: correlation %.3f; best under the near score %s, best under the stop score %s%s"
             % (len(usable), float(np.corrcoef(near, stop)[0, 1]), best_near, best_stop,
                "  (same)" if best_near == best_stop else "  <- DIFFERENT"))
    emit("")


# ---------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Why the closed-loop stop point differs from the single step")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="a run directory (fold_*/ or all_data/ under it)")
    source.add_argument("--trajectories", nargs="+", help="trajectory jsonl files, simulated or on-console")
    parser.add_argument("--tag", default=None, help="which closed-loop setting to read when a run has several")
    parser.add_argument("--labels", default=os.path.join(ROOT, "data", "labels_fieldii.jsonl"),
                        help="labels used for section 3 (the start point's distance to the optimum)")
    parser.add_argument("--compare", nargs="*", default=None,
                        help="extra run directories for section 5 (defaults to --run alone)")
    parser.add_argument("--compare-tag", default=None,
                        help="restrict section 5 to this closed-loop tag; by default --tag is used only when "
                             "--compare is not given, because runs rarely share a tag")
    args = parser.parse_args(argv)
    os.chdir(ROOT)

    records, evaluation, title = [], None, ""
    if args.run:
        found = find_trajectories(args.run, args.tag)
        if not found:
            raise SystemExit("no closed_loop_trajectories_*.jsonl under %s" % args.run)
        tag = args.tag or pick_tag(found)
        found = [f for f in found if tag in f[1]]
        for fold, name, path, ev in found:
            records.extend(read_jsonl(path))
            evaluation = evaluation or ev
        title = "%s, tag %s, %d fold(s)" % (args.run, tag, len(found))
    else:
        for path in args.trajectories:
            records.extend(read_jsonl(path))
        title = ", ".join(args.trajectories)

    emit("=========== closed-loop stop points ===========")
    emit("  source     %s" % title)
    emit("  records    %d trajectory(ies), %d with teacher labels"
         % (len(records), sum(1 for r in records if r.get("frontend_correct"))))
    if evaluation:
        emit("  single-step metrics from %s" % os.path.relpath(evaluation, ROOT))
    emit("")

    section_single_vs_stop(records, evaluation)
    section_signed_steps(records)
    section_start_to_stop(records, label_index(args.labels))
    section_navigation(records)
    runs = args.compare if args.compare else ([args.run] if args.run else [])
    if runs:
        section_compare(runs, args.compare_tag or (None if args.compare else args.tag))

    text = "\n".join(LINES)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return records


if __name__ == "__main__":
    main()
