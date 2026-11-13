# -*- coding: utf-8 -*-
"""聚焦头对照实验的汇总：有序回归（CORAL）与远壁深度先验各自带来了什么。

把若干个 run 目录放在一起比：先核对训练参数（只要有一项不是刻意改的，
那条对比就不干净，报告里必须点出来），再看测试集单步指标、闭环指标，
最后看验证曲线的平台均值与形状——挑最优 epoch 会把选择噪声当成效果，
所以平台均值才是能拿去比的数。

用法：
    python tests/measure_focus_head_ablation.py --runs runs/invivo_bf_v2 runs/invivo_bf_v6_ordinal ...
不给 --runs 时用下面 DEFAULT_RUNS 里那组。
"""
import argparse
import csv
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

REPORT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "measure_focus_head_ablation.txt")

DEFAULT_RUNS = [
    "runs/invivo_bf_v2",
    "runs/invivo_bf_v5_ordinal_only",
    "runs/invivo_bf_v5_farwall_only",
    "runs/invivo_bf_v5_both",
    "runs/invivo_bf_v6_ordinal",
    "runs/invivo_bf_v6_farwall",
    "runs/invivo_bf_v6_both",
]

# 刻意改的开关放前面；其余任何一项不一致都是混淆，要报出来
INTENDED = ("focus_head", "far_wall")
CONTROLLED = ("scalar_norm", "batch", "amp", "lr", "epochs", "group_by", "seed",
              "frontend_select", "recorded_start_prob", "frontend_dropout",
              "frontend_label_smoothing", "d_model", "transformer_layers",
              "invariance_weight", "backend_output", "input_mode")

SINGLE_STEP = ("focus_top1", "focus_top1_near", "focus_within1", "focus_dir_f1_derived",
               "gain_mae_db", "tgc_mae_db", "gain_feedback_slope", "backend_score",
               "far_wall_mae_mm")
# --closed-loop backend 只把增益/TGC 放进环里，前端档位不步进，所以 final_depth_*
# final_frequency_* final_focus_* 在所有 run 里都等于记录起点的那一份，一字不差，
# 不是可比的量，别放进表里
CLOSED_LOOP = ("converged", "gain_abs_error_db_final", "gain_within_deadband_final",
               "tgc_mae_db_final", "oscillated", "gain_clamped")
PLATEAU = ("val_focus_top1_near", "val_focus_top1", "val_focus_within1",
           "val_gain_mae_db", "val_tgc_mae_db", "val_far_wall_mae_mm")

LINES = []


def emit(text=""):
    LINES.append(text)
    print(text)


def table(header, rows, indent="  "):
    cols = [header] + rows
    width = [max(len(str(r[i])) for r in cols) for i in range(len(header))]
    emit(indent + "  ".join(str(c).ljust(width[i]) if i == 0 else str(c).rjust(width[i])
                            for i, c in enumerate(header)))
    for row in rows:
        emit(indent + "  ".join(str(c).ljust(width[i]) if i == 0 else str(c).rjust(width[i])
                                for i, c in enumerate(row)))


def read_args(run):
    """训练参数从 train_report.txt 里那行 JSON 取。"""
    path = os.path.join(run, "train_report.txt")
    if not os.path.exists(path):
        return {}
    text = io.open(path, encoding="utf-8", errors="replace").read()
    found = re.search(r"^\s*args (\{.*\})\s*$", text, re.M)
    return json.loads(found.group(1)) if found else {}


def read_eval(run, tag):
    path = os.path.join(run, "fold_0", "evaluation_combined_%s.json" % tag)
    return json.load(io.open(path, encoding="utf-8")) if os.path.exists(path) else {}


def read_metrics(run):
    path = os.path.join(run, "fold_0", "metrics.csv")
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(io.open(path, encoding="utf-8")))
    return [r for r in rows if r.get("val_score")]


def fmt(value, places=4):
    if value is None:
        return "-"
    return ("%." + str(places) + "f") % value


def check_confounds(runs, args_by_run):
    emit("=========== 1. are the comparisons clean? ===========")
    emit("  the switches under test:")
    rows = []
    for run in runs:
        a = args_by_run[run]
        rows.append([os.path.basename(run)] + [str(a.get(k, "-")) for k in INTENDED])
    table(["run"] + list(INTENDED), rows)
    emit("")

    differing = []
    for key in CONTROLLED:
        values = set(json.dumps(args_by_run[r].get(key)) for r in runs)
        if len(values) > 1:
            differing.append(key)
    if not differing:
        emit("  every other training setting is identical across these runs; the")
        emit("  differences can be read as the effect of the switches above.")
        return []
    emit("  WARNING: these settings are NOT identical across the runs, so any")
    emit("  difference between the groups they separate mixes several changes:")
    rows = []
    for run in runs:
        a = args_by_run[run]
        rows.append([os.path.basename(run)] + [str(a.get(k, "-")) for k in differing])
    table(["run"] + differing, rows)
    return differing


def single_step(runs, tag):
    emit("")
    emit("=========== 2. test split, single step from the recorded start ===========")
    rows = []
    for run in runs:
        section = read_eval(run, tag).get("network_recorded_start", {})
        rows.append([os.path.basename(run)] + [fmt(section.get(k)) for k in SINGLE_STEP])
    table(["run"] + list(SINGLE_STEP), rows)
    emit("  focus_within1 is the metric an ordering constraint should move first:")
    emit("  it says the prediction is at most one rung away, which is what the")
    emit("  ordinal head is built to get right. top-1 alone can move on luck.")


def closed_loop(runs, tag):
    emit("")
    emit("=========== 3. test split, closed loop (back end, stop dead band 2) ===========")
    rows = []
    for run in runs:
        section = read_eval(run, tag).get("closed_loop", {})
        rows.append([os.path.basename(run)] + [fmt(section.get(k)) for k in CLOSED_LOOP])
    table(["run"] + list(CLOSED_LOOP), rows)
    emit("  the front-end rungs are not stepped in backend mode, so the loop's")
    emit("  final depth/frequency/focus figures are the recorded start's and are")
    emit("  identical in every run; only gain and TGC are being compared here.")


def plateau(runs, keep=5):
    emit("")
    emit("=========== 4. validation, mean of the last %d evaluation points ===========" % keep)
    rows = []
    for run in runs:
        tail = read_metrics(run)[-keep:]
        out = []
        for key in PLATEAU:
            values = [float(r[key]) for r in tail if r.get(key)]
            out.append(fmt(sum(values) / len(values)) if values else "-")
        rows.append([os.path.basename(run)] + out)
    table(["run"] + [k[4:] for k in PLATEAU], rows)
    emit("  the plateau mean, not the best epoch: inside one run neighbouring")
    emit("  evaluation points differ by as much as the effect being measured, so")
    emit("  the best epoch is partly the checkpoint lottery.")

    emit("")
    emit("  best epoch vs plateau, on focus_top1_near:")
    rows = []
    for run in runs:
        points = [(float(r["val_focus_top1_near"]), int(r["epoch"])) for r in read_metrics(run)]
        if not points:
            continue
        best = max(points)
        tail = [p[0] for p in points[-keep:]]
        rows.append([os.path.basename(run), fmt(best[0]), best[1], fmt(min(tail)),
                     fmt(max(tail)), fmt(max(tail) - min(tail))])
    table(["run", "best", "epoch", "plateau_min", "plateau_max", "spread"], rows)


def curves(runs):
    emit("")
    emit("=========== 5. the shape of the validation curve ===========")
    emit("  focus_top1_near by epoch; a head that helps should keep rising and")
    emit("  then sit flat, not peak in the first few epochs and decay.")
    for run in runs:
        points = read_metrics(run)
        if not points:
            continue
        text = " ".join("%d:%.3f" % (int(r["epoch"]), float(r["val_focus_top1_near"]))
                        for r in points)
        emit("  %-26s %s" % (os.path.basename(run), text))
    emit("")
    emit("  focus_within1 by epoch:")
    for run in runs:
        points = read_metrics(run)
        if not points:
            continue
        text = " ".join("%d:%.3f" % (int(r["epoch"]), float(r["val_focus_within1"]))
                        for r in points)
        emit("  %-26s %s" % (os.path.basename(run), text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--tag", default="test_bloop_db2")
    parser.add_argument("--out", default=REPORT_PATH)
    opts = parser.parse_args()

    runs = [r if os.path.isabs(r) else os.path.join(ROOT, r) for r in opts.runs]
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        print("none of the given run directories exist")
        return 1

    args_by_run = dict((r, read_args(r)) for r in runs)
    emit("=========== focus head ablation ===========")
    emit("  runs %s" % ", ".join(os.path.basename(r) for r in runs))
    emit("  evaluation tag %s" % opts.tag)
    emit("")
    check_confounds(runs, args_by_run)
    single_step(runs, opts.tag)
    closed_loop(runs, opts.tag)
    plateau(runs)
    curves(runs)

    io.open(opts.out, "w", encoding="utf-8", newline="\n").write("\n".join(LINES) + "\n")
    print("")
    print("report written to %s" % opts.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
