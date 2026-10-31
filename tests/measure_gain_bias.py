# -*- coding: utf-8 -*-
"""增益残差的归因：1.37 dB 里，哪一部分是标签漂移，哪一部分是模型没学到的。

    问题

留出测试集上增益单步误差 1.37 dB，查表基线 1.94 dB。已知残差以受试者为单位呈常数偏置
（受试者内一致性 0.74 dB，平方误差的 82% 来自受试者间），但一直没查过**偏置与什么相关**。
这决定了后面的工作往哪投：

  标签侧  后端目标是「本场次操作者所接受的组织亮度」。不同场次的主观基准本来就飘，
          而网络只看图像，看不出这一场的操作者偏亮还是偏暗 —— 这部分是学不出来的。
  模型侧  若偏置与图像可见的量（亮度参考、底噪、深度）相关，那是模型没用好已有输入，
          是可以改的。

    怎么判

决定性的一步在第 5 节：把每个（场次, 成像模式）组**被接受的组织亮度**从已有数字里反解出来。
后端求解器的定心增益是

    optimal_gain_db = (target_gray - 43) * window_db / 255 + reference_db - tissue_median_db

三项里 optimal_gain_db 在标签行、reference_db 与 db 图在训练缓存、window_db 由 dr_ui 换算，
于是

    target_gray = 43 + 255 / window_db * (optimal_gain_db - reference_db + tissue_median_db)

一组之内 target_gray 是常数，所以组内散布量的是反解噪声（应当只有几个灰阶），组间散布量的
才是场次之间的目标漂移。再看模型的受试者偏置与这个目标的相关：网络看不见目标，若两者强相关，
那 1.37 dB 里的大头就是标签侧的漂移，模型再调也降不下去。

缓存里的 db 是 BC0/counts_per_db 加本组深度响应（正是解后端标签时用的那张 dB 图），floor 是
本组底噪加同一条深度响应，因此组织掩膜按 db > floor + 3 dB 取，与标注时同口径（缓存按行存
floor，标注时是常数加在 db 上，两者差一个深度响应项；这点只影响反解噪声，不影响组间比较）。

    用法

    python tests/measure_gain_bias.py --run runs/invivo_bf_v2
    python tests/measure_gain_bias.py --run runs/invivo_bf_v2 --frames-per-group 20

报告写在脚本旁边的 measure_gain_bias.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import glob
import io
import json
import math
import os
import struct
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))

import numpy as np

import data_paths as DPATH
import hisense_backend_sim as S

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
GRAY_PIVOT = 43.0
GRAY_MAX = 255.0
TISSUE_MARGIN_DB = 3.0
MIN_TISSUE_PIXELS = 1000
LINES = []


def emit(text=""):
    LINES.append(str(text))


# ---------------------------------------------------------------------------------------
#   小统计

def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan"), len(x)
    return float(np.corrcoef(x, y)[0, 1]), len(x)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    return pearson(rankdata(x[keep]), rankdata(y[keep]))[0]


def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), float)
    ranks[order] = np.arange(1, len(values) + 1)
    # 并列取平均秩
    values = np.asarray(values, float)
    for value in np.unique(values):
        same = values == value
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    return ranks


def fisher_p(r, n):
    """相关系数的近似双侧 p（Fisher z 正态近似），n 小的时候只当作量级看。"""
    if not np.isfinite(r) or n < 4 or abs(r) >= 1.0:
        return float("nan")
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


def ols_r2(features, target):
    """最小二乘的 R2 与调整 R2；features 不含常数列。"""
    target = np.asarray(target, float)
    design = np.column_stack([np.ones(len(target))] + [np.asarray(f, float) for f in features])
    coef, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design.dot(coef)
    ss_res = float((residual ** 2).sum())
    ss_tot = float(((target - target.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n, p = len(target), design.shape[1] - 1
    adj = 1.0 - (1.0 - r2) * (n - 1) / float(n - p - 1) if n - p - 1 > 0 else float("nan")
    return r2, adj, p


def one_hot(labels):
    """类别变量的哑变量（丢掉最后一列，常数列由 ols_r2 补）。"""
    names = sorted(set(labels))
    return [np.asarray([1.0 if l == name else 0.0 for l in labels]) for name in names[:-1]]


# ---------------------------------------------------------------------------------------
#   缓存：db.npy 以 STORED 方式存在 npz 里，按偏移 memmap，只读要用的帧

def cache_arrays(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        small = {}
        with archive.open("frame_id.npy") as handle:
            small["frame_id"] = np.load(handle, allow_pickle=False)
        for name in ("floor", "reference_db", "noise_floor_top_db", "noise_floor_bottom_db",
                     "min_depth_mm", "max_depth_mm"):
            if name + ".npy" in names:
                with archive.open(name + ".npy") as handle:
                    small[name] = np.load(handle, allow_pickle=False)
        info = archive.getinfo("db.npy")
    if info.compress_type != zipfile.ZIP_STORED:
        raise SystemExit("db.npy is compressed in the npz; cannot memory-map it")
    with io.open(path, "rb") as handle:
        handle.seek(info.header_offset)
        header = handle.read(30)
        name_len, extra_len = struct.unpack("<HH", header[26:30])
        handle.seek(info.header_offset + 30 + name_len + extra_len)
        version = np.lib.format.read_magic(handle)
        shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
        offset = handle.tell()
    if fortran:
        raise SystemExit("db.npy is fortran-ordered; not handled")
    small["db"] = np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=shape)
    return small


def tissue_median_db(db_frame, floor_row):
    mask = db_frame > (floor_row[:, None] + TISSUE_MARGIN_DB)
    if mask.sum() < MIN_TISSUE_PIXELS:
        return float("nan")
    return float(np.median(db_frame[mask]))


# ---------------------------------------------------------------------------------------
#   读入

def read_trajectories(path):
    out = {}
    with io.open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                out[record["start_frame"]] = record
    return out


def read_labels(path):
    out = {}
    with io.open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["frame_id"]] = row
    return out


def find_run(run, tag):
    folds = sorted(glob.glob(os.path.join(run, "fold_*")) + glob.glob(os.path.join(run, "all_data")))
    if not folds:
        raise SystemExit("no fold_*/ under %s" % run)
    fold = folds[0]
    pattern = os.path.join(fold, "closed_loop_trajectories_*%s*.jsonl" % (tag or ""))
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit("no closed-loop trajectories under %s" % fold)
    return fold, files[0]


def subject_of(row):
    return row["group_id"].rsplit("/", 1)[0]


# ---------------------------------------------------------------------------------------

def section_subjects(per_subject):
    emit("--- 1. per-subject gain bias on the held-out test set ---")
    bias = np.asarray([s["bias"] for s in per_subject])
    sd = np.asarray([s["sd"] for s in per_subject])
    frames = int(sum(s["n"] for s in per_subject))
    share = float((bias ** 2).sum() / ((bias ** 2).sum() + (sd ** 2).sum()))
    emit("  subjects %d, frames %d" % (len(per_subject), frames))
    emit("  bias          mean %+.2f dB, sd %.2f dB, median |bias| %.2f dB, worst %.2f dB"
         % (bias.mean(), bias.std(ddof=1), np.median(np.abs(bias)), np.abs(bias).max()))
    emit("  within-subject sd   median %.2f dB (this is the floor if every bias were removed)"
         % np.median(sd))
    emit("  squared error from between-subject bias  %.0f%%" % (100 * share))
    emit("  %d of %d subjects within +-1 dB" % (int((np.abs(bias) <= 1.0).sum()), len(bias)))
    emit()
    emit("  %-22s %-9s %5s %8s %7s %8s" % ("subject", "day", "n", "bias dB", "sd dB", "mode"))
    for s in sorted(per_subject, key=lambda s: s["bias"]):
        emit("  %-22s %-9s %5d %+8.2f %7.2f %8s"
             % (s["subject"], s["day"], s["n"], s["bias"], s["sd"], s["mode"]))
    emit()
    return share


def section_days(per_subject):
    emit("--- 2. variance decomposition over acquisition days ---")
    by_day = collections.defaultdict(list)
    for s in per_subject:
        by_day[s["day"]].append(s)
    day_means = np.asarray([np.mean([s["bias"] for s in group]) for group in by_day.values()])
    within_day = np.concatenate([[s["bias"] - np.mean([t["bias"] for t in group])
                                  for s in group] for group in by_day.values()])
    within_subject = np.asarray([s["sd"] for s in per_subject])
    emit("  days %d (subjects per day %s)"
         % (len(by_day), ", ".join("%d" % len(g) for g in by_day.values())))
    emit("  between-day sd                 %.2f dB" % day_means.std(ddof=1))
    emit("  between-subject within day sd  %.2f dB" % within_day.std(ddof=1))
    emit("  within-subject sd (median)     %.2f dB" % np.median(within_subject))
    emit()
    emit("  %-10s %6s %9s" % ("day", "n", "mean bias"))
    for day in sorted(by_day):
        group = by_day[day]
        emit("  %-10s %6d %+9.2f" % (day, len(group), np.mean([s["bias"] for s in group])))
    emit()
    return by_day


def section_features(per_subject, features):
    emit("--- 3. per-subject features vs bias (n=%d) ---" % len(per_subject))
    bias = [s["bias"] for s in per_subject]
    rows = []
    for name, values in features:
        r, n = pearson(values, bias)
        rows.append((name, r, spearman(values, bias), fisher_p(r, n), n))
    rows.sort(key=lambda row: -abs(row[1]) if np.isfinite(row[1]) else 0.0)
    emit("  %-30s %7s %7s %10s %5s" % ("feature", "r", "rho", "p(approx)", "n"))
    for name, r, rho, p, n in rows:
        emit("  %-30s %+7.2f %+7.2f %10.4f %5d" % (name, r, rho, p, n))
    emit()
    emit("  p is a normal approximation; with %d features the Bonferroni 0.05 threshold is %.4f"
         % (len(rows), 0.05 / max(1, len(rows))))
    emit()
    return rows


def section_models(per_subject, features, days):
    emit("--- 4. multivariate explanatory power ---")
    bias = np.asarray([s["bias"] for s in per_subject])
    table = [("acquisition day (one-hot)", one_hot([s["day"] for s in per_subject]))]
    lookup = dict(features)
    for name in ("accepted_target_gray", "reference_db", "tissue_median_db", "harmonic_fraction"):
        if name in lookup and np.isfinite(np.asarray(lookup[name], float)).all():
            table.append((name, [lookup[name]]))
    if "accepted_target_gray" in lookup and "reference_db" in lookup:
        table.append(("accepted_target_gray + reference_db",
                      [lookup["accepted_target_gray"], lookup["reference_db"]]))
    emit("  %-38s %7s %9s %4s" % ("predictors", "R2", "adj R2", "k"))
    for name, columns in table:
        r2, adj, k = ols_r2(columns, bias)
        emit("  %-38s %7.3f %9.3f %4d" % (name, r2, adj, k))
    emit()


def section_target(groups, per_subject, features):
    emit("--- 5. label side: accepted tissue brightness recovered per group ---")
    within = np.asarray([g["sd_gray"] for g in groups if np.isfinite(g["sd_gray"])])
    target = np.asarray([g["target_gray"] for g in groups])
    db_per_gray = np.asarray([g["db_per_gray"] for g in groups])
    emit("  groups %d (session x imaging mode), frames used %d"
         % (len(groups), int(sum(g["n"] for g in groups))))
    emit("  within-group sd of the recovered target   median %.1f gray  (recovery noise;"
         % np.median(within))
    emit("                                             a group's target is a constant by"
         " construction)")
    emit("  between-group sd of the target            %.1f gray = %.2f dB"
         % (target.std(ddof=1), target.std(ddof=1) * np.median(db_per_gray)))
    for mode in sorted(set(g["mode"] for g in groups)):
        values = np.asarray([g["target_gray"] for g in groups if g["mode"] == mode])
        emit("    %-11s groups %3d, target %5.1f +- %4.1f gray (range %.0f - %.0f)"
             % (mode, len(values), values.mean(), values.std(ddof=1), values.min(), values.max()))
    by_day = collections.defaultdict(list)
    for g in groups:
        by_day[g["day"]].append(g["target_gray"])
    day_means = np.asarray([np.mean(v) for v in by_day.values()])
    emit("  between-day sd of the target              %.1f gray = %.2f dB"
         % (day_means.std(ddof=1), day_means.std(ddof=1) * np.median(db_per_gray)))
    emit()
    lookup = dict(features)
    if "accepted_target_gray" in lookup:
        bias = [s["bias"] for s in per_subject]
        r, n = pearson(lookup["accepted_target_gray"], bias)
        emit("  model bias vs the session's own accepted target:  r = %+.2f (n = %d, p ~ %.4f)"
             % (r, n, fisher_p(r, n)))
        emit("  the network only sees the image, so it cannot know this target; a strong positive")
        emit("  correlation means that much of the residual is label-side drift, not model error.")
    emit()


# ---------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default=os.path.join(ROOT, "runs", "invivo_bf_v2"))
    parser.add_argument("--tag", default="bloop", help="which closed-loop file to read")
    parser.add_argument("--labels", default=os.path.join(
        ROOT, "data", "labels_console_invivo_carotid_target.jsonl"))
    parser.add_argument("--cache", default=str(
        DPATH.output_dir("console_dl_cache_invivo_carotid") / "cache.npz"))
    parser.add_argument("--frames-per-group", type=int, default=12,
                        help="frames sampled per group for the label-side section")
    parser.add_argument("--out", default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    fold, trajectory_path = find_run(args.run, args.tag)
    trajectories = read_trajectories(trajectory_path)
    labels = read_labels(args.labels)
    cache = cache_arrays(args.cache)
    index = dict((str(f), i) for i, f in enumerate(cache["frame_id"]))

    emit("=========== gain bias attribution ===========")
    emit("  run          %s" % fold)
    emit("  trajectories %s (%d frames)" % (os.path.basename(trajectory_path), len(trajectories)))
    emit("  labels       %s (%d rows)" % (os.path.basename(args.labels), len(labels)))
    emit("  cache        %s (%d frames)" % (args.cache, len(index)))
    emit()

    # ---- 测试集：每帧残差 + 每帧的组织中位 dB
    per_frame = collections.defaultdict(list)
    for frame_id, record in trajectories.items():
        row = labels.get(frame_id)
        if row is None or frame_id not in index:
            continue
        per_frame[subject_of(row)].append((frame_id, float(record["gain_error_db"]), row))
    if not per_frame:
        raise SystemExit("no trajectory frame matched the labels and the cache")

    needed = set()
    for values in per_frame.values():
        needed.update(frame_id for frame_id, _, _ in values)

    # ---- 标签侧：每组抽若干帧
    by_group = collections.defaultdict(list)
    for frame_id, row in labels.items():
        if frame_id in index:
            by_group[row["group_id"]].append(frame_id)
    for group, frames in by_group.items():
        frames.sort()
        step = max(1, len(frames) // max(1, args.frames_per_group))
        needed.update(frames[::step][:args.frames_per_group])

    emit("  reading %d frames from the cache for tissue medians ..." % len(needed))
    medians = {}
    for frame_id in sorted(needed):
        i = index[frame_id]
        medians[frame_id] = tissue_median_db(np.asarray(cache["db"][i], np.float64),
                                             np.asarray(cache["floor"][i], np.float64))
    emit("  done in %.0f s" % (time.time() - started))
    emit()

    def target_gray(frame_id, row):
        median = medians.get(frame_id, float("nan"))
        if not np.isfinite(median) or row.get("optimal_gain_db") is None:
            return float("nan"), float("nan")
        window = S.dr_ui_to_window_db(float(row["dr_ui"]))
        reference = float(cache["reference_db"][index[frame_id]])
        gray = GRAY_PIVOT + GRAY_MAX / window * (float(row["optimal_gain_db"]) - reference + median)
        return gray, window / GRAY_MAX

    # ---- 每位受试者
    per_subject = []
    for subject, values in per_frame.items():
        errors = np.asarray([e for _, e, _ in values], float)
        rows = [row for _, _, row in values]
        i = [index[frame_id] for frame_id, _, _ in values]
        grays = np.asarray([target_gray(frame_id, row)[0] for frame_id, _, row in values], float)
        modes = [row["imaging_mode"] for row in rows]
        per_subject.append({
            "subject": subject, "day": subject[:8], "n": len(errors),
            "bias": float(errors.mean()), "sd": float(errors.std(ddof=1) if len(errors) > 1 else 0.0),
            "mode": "harmonic" if modes.count("harmonic") >= len(modes) / 2.0 else "fundamental",
            "harmonic_fraction": float(np.mean([m == "harmonic" for m in modes])),
            "hour": float(int(subject[9:11]) + int(subject[11:13]) / 60.0),
            "reference_db": float(np.mean(cache["reference_db"][i])),
            "noise_floor_top_db": float(np.mean(cache["noise_floor_top_db"][i])),
            "noise_floor_bottom_db": float(np.mean(cache["noise_floor_bottom_db"][i])),
            "tissue_median_db": float(np.nanmean([medians[f] for f, _, _ in values])),
            "accepted_target_gray": float(np.nanmean(grays)),
            "depth_mm": float(np.mean([row["depth_mm"] for row in rows])),
            "frequency_mhz": float(np.mean([row["frequency_mhz"] for row in rows])),
            "focus_mm": float(np.mean([row["focus_mm"] for row in rows])),
            "dr_ui": float(np.mean([row["dr_ui"] for row in rows])),
            "optimal_gain_db": float(np.mean([row["optimal_gain_db"] for row in rows])),
            "start_gain_db": float(np.mean([row["gain_db"] for row in rows])),
            "label_uncertainty": float(np.mean([row.get("label_uncertainty") or 0.0 for row in rows])),
            "calibration_borrowed": float(np.mean([bool(row.get("calibration_borrowed")) for row in rows])),
            "motion_flag": float(np.mean([bool(row.get("family_motion_flag")) for row in rows])),
            "deadband_gain_levels": float(np.mean([row["deadband_gain_levels"] for row in rows])),
        })

    keys = ("accepted_target_gray", "reference_db", "tissue_median_db", "noise_floor_top_db",
            "noise_floor_bottom_db", "optimal_gain_db", "start_gain_db", "harmonic_fraction",
            "depth_mm", "frequency_mhz", "focus_mm", "dr_ui", "label_uncertainty",
            "calibration_borrowed", "motion_flag", "deadband_gain_levels", "hour", "n")
    features = [(key, [s[key] for s in per_subject]) for key in keys]

    section_subjects(per_subject)
    days = section_days(per_subject)
    section_features(per_subject, features)
    section_models(per_subject, features, days)

    # ---- 标签侧：每组的目标
    groups = []
    for group, frames in sorted(by_group.items()):
        values, per_db = [], []
        for frame_id in frames:
            if frame_id not in medians:
                continue
            gray, db_per_gray = target_gray(frame_id, labels[frame_id])
            if np.isfinite(gray):
                values.append(gray)
                per_db.append(db_per_gray)
        if len(values) < 2:
            continue
        values = np.asarray(values, float)
        row = labels[frames[0]]
        groups.append({"group": group, "day": group[:8], "mode": row["imaging_mode"],
                       "n": len(values), "target_gray": float(np.median(values)),
                       "sd_gray": float(values.std(ddof=1)),
                       "db_per_gray": float(np.mean(per_db))})
    section_target(groups, per_subject, features)

    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(LINES)
    io.open(args.out, "w", encoding="utf-8", newline=chr(10)).write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
