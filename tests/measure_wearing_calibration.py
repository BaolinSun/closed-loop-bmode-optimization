# -*- coding: utf-8 -*-
"""每次佩戴校正一次，能把增益误差降到多少：留出帧上的离线验证。

    为什么做这件事

measure_gain_bias.py 的结论是：增益的受试者间偏置与本场次被接受的组织亮度相关 r = +0.94，
与图像可见的那部分相关 -0.07。网络看不见本场次的主观基准，这部分不是模型误差，训练不掉。
剩下的出路之一是「每次佩戴校正一次」：贴上探头、闭环停下来之后，让人把增益确认一次，
把这一次的差值当作本次佩戴的常数，之后整段监测沿用。

    怎么验才算数

关键是**不能在用来估偏置的那几帧上评估**。本脚本对每位受试者：

    1. 取 k 帧作为校正帧（按采集顺序的前 k 帧，或随机抽 k 帧重复 200 次取平均）；
    2. 用这 k 帧的残差估出本次佩戴的常数（均值或中位数）；
    3. 只在**其余帧**上评估修正后的误差。

同时给出三条参照：不校正、去掉全体平均偏置（对照组，应当没有效果）、以及用全部帧算出的
受试者常数（oracle，泄漏，只作上界）。第 3 节再按帧距校正帧的远近分段，看这个常数在一个
场次里撑不撑得住 —— 长时程佩戴关心的正是这一点。

    数据

读闭环轨迹里的停点残差 gain_error_db（教师最优值 - 停下来时的增益），即用户最后看到的那幅图
的误差。接受判据用标签行的 deadband_gain_levels x gain_db_per_level（约 0.85 dB）。

    用法

    python tests/measure_wearing_calibration.py --run runs/invivo_bf_v2
    python tests/measure_wearing_calibration.py --run runs/invivo_bf_v2 --draws 500

报告写在脚本旁边的 measure_wearing_calibration.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import glob
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
KS = (1, 2, 3, 5, 10)
COMMON_FROM = 10          # 公共留出集：第 11 帧起，各个 k 在同一批帧上比较
OPERATING_BAND_DB = 1.0   # 「同一工作设置」的宽度：起点离本受试者中位起点多远还算同一档
LINES = []


def emit(text=""):
    LINES.append(str(text))


def read_jsonl(path):
    out = []
    with io.open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
    return out


def find_trajectories(run, tag):
    folds = sorted(glob.glob(os.path.join(run, "fold_*")) + glob.glob(os.path.join(run, "all_data")))
    if not folds:
        raise SystemExit("no fold_*/ under %s" % run)
    files = sorted(glob.glob(os.path.join(folds[0], "closed_loop_trajectories_*%s*.jsonl" % (tag or ""))))
    if not files:
        raise SystemExit("no closed-loop trajectories under %s" % folds[0])
    return folds[0], files[0]


def subjects_from(trajectory_path, labels_path):
    """每位受试者：按采集顺序的停点残差、接受死区、所属划分。"""
    labels = dict((row["frame_id"], row) for row in read_jsonl(labels_path))
    per = {}
    for record in read_jsonl(trajectory_path):
        frame_id = record["start_frame"]
        row = labels.get(frame_id)
        if row is None:
            continue
        subject = row["group_id"].rsplit("/", 1)[0]
        per.setdefault(subject, []).append(
            (frame_id, float(record["gain_error_db"]),
             float(row["deadband_gain_levels"]) * float(row["gain_db_per_level"]),
             row.get("split"), float(record.get("start_gain_error_db", float("nan"))),
             bool(record.get("converged"))))
    out = []
    for subject, values in sorted(per.items()):
        values.sort()                                   # frame_id 是时间戳，排序即采集顺序
        out.append({"subject": subject, "day": subject[:8],
                    "frames": [v[0] for v in values],
                    "error": np.asarray([v[1] for v in values], float),
                    "deadband": np.asarray([v[2] for v in values], float),
                    "split": values[0][3],
                    "start_error": np.asarray([v[4] for v in values], float),
                    "converged": np.asarray([v[5] for v in values], bool)})
    return out


# ---------------------------------------------------------------------------------------
#   指标：都在「留出帧」上算

def score(subjects, estimate, mask):
    """estimate(subject) -> 本次佩戴的常数；mask(subject) -> 参与评估的帧。"""
    per_subject, frames, errors, hits = [], 0, [], []
    for s in subjects:
        keep = mask(s)
        if not keep.any():
            continue
        corrected = s["error"][keep] - estimate(s)
        per_subject.append(float(np.abs(corrected).mean()))
        frames += int(keep.sum())
        errors.append(corrected)
        hits.append(np.abs(corrected) <= s["deadband"][keep])
    errors = np.concatenate(errors)
    hits = np.concatenate(hits)
    return {"subject_mae": float(np.mean(per_subject)), "frame_mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt((errors ** 2).mean())), "within": float(hits.mean()),
            "worst_subject": float(np.max(per_subject)), "frames": frames,
            "subjects": len(per_subject)}


def operating_pool(s):
    """起点落在本受试者中位起点附近的帧：最接近「佩戴时只有一个工作设置」的那一批。"""
    median = float(np.median(s["start_error"]))
    return np.where(np.abs(s["start_error"] - median) <= OPERATING_BAND_DB)[0]


def head_mask(s, k):
    keep = np.ones(len(s["error"]), bool)
    keep[:k] = False
    return keep


def common_mask(s):
    keep = np.zeros(len(s["error"]), bool)
    keep[COMMON_FROM:] = True
    return keep


def row(name, result):
    emit("  %-34s %9.2f %9.2f %8.2f %8.1f%% %9.2f %7d"
         % (name, result["subject_mae"], result["frame_mae"], result["rmse"],
            100 * result["within"], result["worst_subject"], result["frames"]))


def header():
    emit("  %-34s %9s %9s %8s %9s %9s %7s"
         % ("protocol", "MAE/subj", "MAE/frame", "RMSE", "in dead-", "worst", "frames"))
    emit("  %-34s %9s %9s %8s %9s %9s %7s"
         % ("", "dB", "dB", "dB", "band", "subj dB", ""))


# ---------------------------------------------------------------------------------------

def section_reference(subjects):
    emit("--- 1. reference points (all frames, no held-out split) ---")
    everything = lambda s: np.ones(len(s["error"]), bool)
    header()
    row("no calibration", score(subjects, lambda s: 0.0, everything))
    overall = float(np.mean([s["error"].mean() for s in subjects]))
    row("remove the split-wide mean (control)", score(subjects, lambda s: overall, everything))
    row("oracle: each subject's own mean", score(subjects, lambda s: s["error"].mean(), everything))
    emit()
    emit("  the control removes %+.2f dB from every frame and should do nothing; the oracle uses"
         % overall)
    emit("  every frame of a subject to estimate that subject's constant, so it leaks and is only")
    emit("  an upper bound on what a real calibration can buy.")
    emit()


def section_curve(subjects, draws, seed):
    emit("--- 2. calibrate on k frames, score on the frames left over ---")
    emit("  policy A: the first k frames of the wearing (what a real session would do)")
    emit("  scored on the rest of that subject's frames")
    header()
    row("no calibration (frames 2+)", score(subjects, lambda s: 0.0, lambda s: head_mask(s, 1)))
    for k in KS:
        for name, estimator in (("mean", np.mean), ("median", np.median)):
            if k < 3 and name == "median":
                continue
            row("first %-2d frames, %s" % (k, name),
                score(subjects, lambda s, k=k, e=estimator: e(s["error"][:k]),
                      lambda s, k=k: head_mask(s, k)))
    emit()
    emit("  policy B: k frames drawn at random, averaged over %d draws" % draws)
    emit("  scored on the common held-out set (frame %d onwards) so every row sees the same frames"
         % (COMMON_FROM + 1))
    header()
    row("no calibration", score(subjects, lambda s: 0.0, common_mask))
    rng = np.random.RandomState(seed)
    for k in KS:
        for name, estimator in (("mean", np.mean), ("median", np.median)):
            if k < 3 and name == "median":
                continue
            totals = None
            for _ in range(draws):
                picks = {}
                for s in subjects:
                    pool = np.arange(COMMON_FROM)          # 只从公共留出集之外抽
                    picks[s["subject"]] = estimator(s["error"][rng.choice(pool, k, replace=False)])
                result = score(subjects, lambda s: picks[s["subject"]], common_mask)
                totals = ({key: 0.0 for key in result} if totals is None else totals)
                for key in result:
                    totals[key] += result[key] / float(draws)
            row("random %-2d frames, %s" % (k, name), totals)
    emit()
    head = np.concatenate([s["start_error"][:3] for s in subjects])
    rest = np.concatenate([s["start_error"][3:] for s in subjects])
    inside = np.median([np.corrcoef(s["error"], s["start_error"])[0, 1] for s in subjects])
    emit("  why policy A is the pessimistic one here: these sessions swept the back end, and the")
    emit("  first frames sit far below the operating point (start error %+.2f dB over the first 3"
         % head.mean())
    emit("  frames against %+.2f dB over the rest). The stop point still carries part of the start"
         % rest.mean())
    emit("  error (within-subject r = %+.2f), so a calibration taken there inherits that offset."
         % inside)
    emit("  A wearing has one operating setting and no sweep, so policy B is the closer analogue.")
    emit()
    emit("  policy C: only the frames whose start sits within %.1f dB of that subject's median"
         % OPERATING_BAND_DB)
    emit("  start - the closest stand-in for a wearing, where there is one operating setting")
    header()
    pools = dict((s["subject"], operating_pool(s)) for s in subjects)
    usable = [s for s in subjects if len(pools[s["subject"]]) >= 12]
    emit("  (%d of %d subjects have at least 12 such frames)" % (len(usable), len(subjects)))
    for k in (1, 3, 5):
        totals = None
        for _ in range(draws):
            picks, masks = {}, {}
            for s in usable:
                pool = pools[s["subject"]]
                chosen = rng.choice(pool, k, replace=False)
                picks[s["subject"]] = float(np.mean(s["error"][chosen]))
                keep = np.zeros(len(s["error"]), bool)
                keep[pool] = True
                keep[chosen] = False
                masks[s["subject"]] = keep
            result = score(usable, lambda s: picks[s["subject"]], lambda s: masks[s["subject"]])
            totals = ({key: 0.0 for key in result} if totals is None else totals)
            for key in result:
                totals[key] += result[key] / float(draws)
        if k == 1:
            row("no calibration", score(usable, lambda s: 0.0,
                                        lambda s: np.isin(np.arange(len(s["error"])),
                                                          pools[s["subject"]])))
        row("operating band, %d frames, mean" % k, totals)
    row("operating band, oracle",
        score(usable, lambda s: float(np.mean(s["error"][pools[s["subject"]]])),
              lambda s: np.isin(np.arange(len(s["error"])), pools[s["subject"]])))
    emit()


def section_distance(subjects, k):
    emit("--- 3. does the constant hold up through the wearing? (first %d frames calibrate) ---" % k)
    emit("  held-out frames split into quartiles by how far they sit from the calibration frames")
    emit("  %-12s %7s %11s %11s %9s" % ("quartile", "frames", "MAE raw dB", "MAE cal dB", "in band"))
    buckets = [[] for _ in range(4)]
    bands = [[] for _ in range(4)]
    for s in subjects:
        rest = np.arange(k, len(s["error"]))
        if len(rest) < 8:
            continue
        constant = float(np.mean(s["error"][:k]))
        edges = np.array_split(rest, 4)
        for q, part in enumerate(edges):
            buckets[q].append(np.column_stack([s["error"][part], s["error"][part] - constant]))
            bands[q].append(np.abs(s["error"][part] - constant) <= s["deadband"][part])
    for q in range(4):
        values = np.concatenate(buckets[q])
        hits = np.concatenate(bands[q])
        emit("  %-12s %7d %11.2f %11.2f %8.1f%%"
             % ("Q%d" % (q + 1), len(values), np.abs(values[:, 0]).mean(),
                np.abs(values[:, 1]).mean(), 100 * hits.mean()))
    emit()
    emit("  a flat MAE across quartiles means one calibration covers the whole session;")
    emit("  a rising one would mean the constant drifts and the wearing needs re-calibration.")
    emit()


def section_subjects(subjects, k):
    emit("--- 4. per subject (first %d frames calibrate, scored on the rest) ---" % k)
    emit("  %-22s %6s %9s %10s %10s %9s"
         % ("subject", "frames", "bias dB", "MAE raw", "MAE cal", "in band"))
    rows = []
    for s in subjects:
        keep = head_mask(s, k)
        constant = float(np.mean(s["error"][:k]))
        raw = float(np.abs(s["error"][keep]).mean())
        cal = float(np.abs(s["error"][keep] - constant).mean())
        hits = float((np.abs(s["error"][keep] - constant) <= s["deadband"][keep]).mean())
        rows.append((s["subject"], int(keep.sum()), float(s["error"].mean()), raw, cal, hits))
    for name, n, bias, raw, cal, hits in sorted(rows, key=lambda r: -abs(r[2])):
        emit("  %-22s %6d %+9.2f %10.2f %10.2f %8.1f%%" % (name, n, bias, raw, cal, 100 * hits))
    emit()
    better = sum(1 for r in rows if r[4] < r[3])
    emit("  calibration helps %d of %d subjects" % (better, len(rows)))
    emit()


def section_verdict(subjects, k, draws, seed):
    emit("--- 5. verdict ---")
    raw = score(subjects, lambda s: 0.0, lambda s: head_mask(s, k))
    cal = score(subjects, lambda s: float(np.mean(s["error"][:k])), lambda s: head_mask(s, k))
    oracle = score(subjects, lambda s: s["error"].mean(), common_mask)
    rng = np.random.RandomState(seed + 1)
    random_raw = score(subjects, lambda s: 0.0, common_mask)
    random_cal = None
    for _ in range(draws):
        picks = dict((s["subject"], float(np.mean(s["error"][rng.choice(COMMON_FROM, k, replace=False)])))
                     for s in subjects)
        result = score(subjects, lambda s: picks[s["subject"]], common_mask)
        random_cal = ({key: 0.0 for key in result} if random_cal is None else random_cal)
        for key in result:
            random_cal[key] += result[key] / float(draws)
    emit("  on the frames that never took part in the calibration (k = %d):" % k)
    emit("    policy A, first %d frames   MAE %.2f -> %.2f dB (-%.0f%%), in band %.1f%% -> %.1f%%,"
         " worst %.2f -> %.2f dB"
         % (k, raw["frame_mae"], cal["frame_mae"],
            100 * (1 - cal["frame_mae"] / raw["frame_mae"]),
            100 * raw["within"], 100 * cal["within"],
            raw["worst_subject"], cal["worst_subject"]))
    emit("    policy B, random %d frames  MAE %.2f -> %.2f dB (-%.0f%%), in band %.1f%% -> %.1f%%,"
         " worst %.2f -> %.2f dB"
         % (k, random_raw["frame_mae"], random_cal["frame_mae"],
            100 * (1 - random_cal["frame_mae"] / random_raw["frame_mae"]),
            100 * random_raw["within"], 100 * random_cal["within"],
            random_raw["worst_subject"], random_cal["worst_subject"]))
    emit("    oracle ceiling             MAE %.2f dB, in band %.1f%%"
         % (oracle["frame_mae"], 100 * oracle["within"]))
    pools = dict((x["subject"], operating_pool(x)) for x in subjects)
    usable = [x for x in subjects if len(pools[x["subject"]]) >= 12]
    band = lambda x: np.isin(np.arange(len(x["error"])), pools[x["subject"]])
    band_raw = score(usable, lambda x: 0.0, band)
    band_cal = None
    for _ in range(draws):
        picks, masks = {}, {}
        for x in usable:
            pool = pools[x["subject"]]
            chosen = rng.choice(pool, k, replace=False)
            picks[x["subject"]] = float(np.mean(x["error"][chosen]))
            keep = np.zeros(len(x["error"]), bool)
            keep[pool] = True
            keep[chosen] = False
            masks[x["subject"]] = keep
        result = score(usable, lambda x: picks[x["subject"]], lambda x: masks[x["subject"]])
        band_cal = ({key: 0.0 for key in result} if band_cal is None else band_cal)
        for key in result:
            band_cal[key] += result[key] / float(draws)
    emit()
    emit("  and on the stand-in for an actual wearing - frames at one operating setting"
         " (%d of %d subjects):" % (len(usable), len(subjects)))
    emit("    policy C, %d frames         MAE %.2f -> %.2f dB (-%.0f%%), in band %.1f%% -> %.1f%%,"
         " worst %.2f -> %.2f dB"
         % (k, band_raw["frame_mae"], band_cal["frame_mae"],
            100 * (1 - band_cal["frame_mae"] / band_raw["frame_mae"]),
            100 * band_raw["within"], 100 * band_cal["within"],
            band_raw["worst_subject"], band_cal["worst_subject"]))
    emit("    this is the number to quote for the wearable case; it sits within %.2f dB of that"
         % (band_cal["frame_mae"] - score(usable, lambda x: float(np.mean(x["error"][pools[x["subject"]]])),
                                          band)["frame_mae"]))
    emit("    subset's own oracle, so three confirmed frames are already close to all one constant")
    emit("    per wearing can give.")
    emit()
    emit("  one to three confirmed frames per wearing collect most of what is available; section 2")
    emit("  shows k beyond 3 buys little. What is left is the within-subject scatter, and that is")
    emit("  what a better model would have to attack.")
    emit()


# ---------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default=os.path.join(ROOT, "runs", "invivo_bf_v2"))
    parser.add_argument("--tag", default="bloop")
    parser.add_argument("--labels", default=os.path.join(
        ROOT, "data", "labels_console_invivo_carotid_target.jsonl"))
    parser.add_argument("--k", type=int, default=3, help="calibration frames for sections 3-5")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    fold, trajectory_path = find_trajectories(args.run, args.tag)
    subjects = subjects_from(trajectory_path, args.labels)
    if not subjects:
        raise SystemExit("no trajectory frame matched the labels")
    splits = sorted(set(s["split"] or "?" for s in subjects))
    counts = [len(s["error"]) for s in subjects]

    emit("=========== one calibration per wearing: offline validation ===========")
    emit("  run          %s" % fold)
    emit("  trajectories %s" % os.path.basename(trajectory_path))
    emit("  split        %s: %d subjects, %d frames (%d - %d per subject)"
         % ("/".join(splits), len(subjects), sum(counts), min(counts), max(counts)))
    emit("  deadband     %.2f dB median (deadband_gain_levels x gain_db_per_level)"
         % float(np.median(np.concatenate([s["deadband"] for s in subjects]))))
    emit("  residual     closed-loop stop point: teacher optimum - settled gain")
    emit()

    section_reference(subjects)
    section_curve(subjects, args.draws, args.seed)
    section_distance(subjects, args.k)
    section_subjects(subjects, args.k)
    section_verdict(subjects, args.k, args.draws, args.seed)

    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(LINES)
    io.open(args.out, "w", encoding="utf-8", newline=chr(10)).write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
