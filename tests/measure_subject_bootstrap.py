# -*- coding: utf-8 -*-
"""按受试者自助重抽的置信区间：网络与设置查表基线的差距，哪些站得住。

    问题

留出测试集只有 29 位受试者。报告里都是点估计：聚焦 0.330 对基线 0.227、增益 1.37 对 1.94、
时间增益补偿 0.26 对 2.80。同一位受试者的几十帧不是独立样本（残差以受试者为单位呈常数偏置，
见 measure_gain_bias.py），所以按帧算区间会把区间算得太窄。答辩时一定会被问「29 个人，
聚焦高 0.10 算不算真的高」。

    怎么做

按**受试者**整群自助重抽（cluster bootstrap）：每次有放回地抽 29 位受试者，把他们的全部帧
汇总起来重算指标，重复 B 次，取 2.5% 与 97.5% 分位。配对差值（同一次重抽里网络减基线）单独
给区间——这才是「谁更好」的直接答案。再补一个与分布无关的符号检验：29 位里网络赢了几位。

两种方法在完全相同的帧、相同的起点（recorded start）上算，指标定义直接沿用
bmode_dl.metrics.compute_metrics 的逐帧口径，点估计应当与评估报告里的数字一致（脚本会核对）。

    用法

    python tests/measure_subject_bootstrap.py --run runs/invivo_bf_v2 --groups test
    python tests/measure_subject_bootstrap.py --run runs/invivo_bf_v2 --groups test --draws 20000

要读训练缓存与检查点，CPU 上跑一次前向大约几分钟。报告写在脚本旁边的
measure_subject_bootstrap.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import io
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))

import numpy as np
import torch

import data_paths as DPATH
from bmode_dl import constants as K
from bmode_dl.checkpoint import load_combined
from bmode_dl.dataset import FieldIIData
from bmode_dl.metrics import predict, settings_lookup_baseline

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
LINES = []


def emit(text=""):
    LINES.append(str(text))


# ---------------------------------------------------------------------------------------
#   逐帧量：口径与 compute_metrics 一致

def per_frame(preds, tg):
    """返回 {指标: (逐帧值, 该指标计入哪些帧)}。"""
    backend = tg["backend_mask"] > 0
    out = {}
    err = np.abs(preds["gain_delta_db"] - tg["delta_gain_db"])
    out["gain_mae_db"] = (err, backend)
    out["gain_within_deadband"] = ((err <= tg["gain_deadband_db"]).astype(float), backend)
    out["tgc_mae_db"] = (np.abs(preds["tgc_delta_db"] - tg["delta_tgc_db"]).mean(axis=1), backend)
    for axis in ("focus", "depth", "frequency"):
        mask = (tg["%s_mask" % axis] > 0) & (tg["optimal_%s_idx" % axis] >= 0)
        pred, target = preds["%s_idx" % axis], tg["optimal_%s_idx" % axis]
        out["%s_top1" % axis] = ((pred == target).astype(float), mask)
        out["%s_within1" % axis] = ((np.abs(pred - target) <= 1).astype(float), mask)
    return out


def by_subject(values, mask, subjects, order):
    """每位受试者的 (求和, 计数)，顺序与 order 一致。"""
    total = np.zeros(len(order))
    count = np.zeros(len(order))
    position = dict((name, i) for i, name in enumerate(order))
    for value, keep, name in zip(values, mask, subjects):
        if not keep:
            continue
        i = position[name]
        total[i] += value
        count[i] += 1
    return total, count


def bootstrap(net_total, net_count, base_total, base_count, draws, rng):
    """整群重抽：返回 (网络, 基线, 差值) 三组分位数。"""
    n = len(net_count)
    picks = rng.randint(0, n, size=(draws, n))
    net = net_total[picks].sum(axis=1) / np.maximum(1e-9, net_count[picks].sum(axis=1))
    base = base_total[picks].sum(axis=1) / np.maximum(1e-9, base_count[picks].sum(axis=1))
    return net, base, net - base


def interval(values):
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


# ---------------------------------------------------------------------------------------

LOWER_IS_BETTER = ("gain_mae_db", "tgc_mae_db")
SHOWN = ("gain_mae_db", "gain_within_deadband", "tgc_mae_db", "focus_top1", "focus_within1",
         "depth_top1", "frequency_top1")


def section_table(stats, draws):
    emit("--- 1. metric by metric, 95%% cluster bootstrap over subjects (%d draws) ---" % draws)
    emit("  %-22s %18s %18s %20s %7s" % ("metric", "network", "settings lookup",
                                         "difference", "favours"))
    for name in SHOWN:
        if name not in stats:
            continue
        s = stats[name]
        better = "lower" if name in LOWER_IS_BETTER else "higher"
        sign = -1.0 if name in LOWER_IS_BETTER else 1.0
        share = float((sign * s["diff"] > 0).mean())
        emit("  %-22s %7.3f [%6.3f,%6.3f] %7.3f [%6.3f,%6.3f] %+7.3f [%+6.3f,%+6.3f] %6.0f%%"
             % (name, s["net_point"], s["net_ci"][0], s["net_ci"][1],
                s["base_point"], s["base_ci"][0], s["base_ci"][1],
                s["diff_point"], s["diff_ci"][0], s["diff_ci"][1], 100 * share))
    emit()
    emit("  'difference' is network minus baseline, resampled in pairs on the same subjects;")
    emit("  'favours' is the share of resamples in which the network is the better of the two")
    emit("  (%s is better for %s). A difference interval that straddles zero is not resolved"
         % ("lower", ", ".join(LOWER_IS_BETTER)))
    emit("  by %d subjects, however clean the point estimate looks." % stats["subjects"])
    emit("  depth and frequency are not trained on this data (their labels follow from the current")
    emit("  settings), so the baseline winning those two is expected, not a finding. frequency_top1")
    emit("  here is the strict rung match, not the acceptable-set hit the evaluation report uses.")
    emit()


def section_sign(stats):
    emit("--- 2. per-subject sign test (distribution free) ---")
    emit("  %-22s %12s %10s %12s" % ("metric", "net better", "ties", "two-sided p"))
    for name in SHOWN:
        if name not in stats:
            continue
        s = stats[name]
        wins, ties, n = s["wins"], s["ties"], s["n_subjects"]
        trials = n - ties
        p = sign_p(wins, trials)
        emit("  %-22s %6d of %-5d %10d %12s"
             % (name, wins, trials, ties, "%.4f" % p if np.isfinite(p) else "-"))
    emit()


def sign_p(wins, trials):
    """双侧符号检验的精确 p（二项，n 小，直接枚举）。"""
    if trials <= 0:
        return float("nan")
    from math import comb
    tail = sum(comb(trials, i) for i in range(0, min(wins, trials - wins) + 1))
    return min(1.0, 2.0 * tail / float(2 ** trials))


def section_focus(stats, draws):
    emit("--- 3. the focus axis, which is the one in question ---")
    s = stats.get("focus_top1")
    if s is None:
        emit("  no focus labels in this split")
        emit()
        return
    emit("  network %.3f [%.3f, %.3f], baseline %.3f [%.3f, %.3f] over %d subjects"
         % (s["net_point"], s["net_ci"][0], s["net_ci"][1],
            s["base_point"], s["base_ci"][0], s["base_ci"][1], s["n_subjects"]))
    emit("  difference %+.3f [%+.3f, %+.3f]; the network is ahead in %.0f%% of resamples"
         % (s["diff_point"], s["diff_ci"][0], s["diff_ci"][1], 100 * float((s["diff"] > 0).mean())))
    resolved = s["diff_ci"][0] > 0
    emit("  => the advantage %s at 95%% with %d subjects."
         % ("holds" if resolved else "is NOT resolved", s["n_subjects"]))
    wins, ties = s["wins"], s["ties"]
    trials = s["n_subjects"] - ties
    emit("  the distribution-free sign test is weaker: the network wins %d of %d subjects,"
         % (wins, trials))
    emit("  two-sided p = %.4f. Treat the advantage as real but marginal - it is at the edge of"
         % sign_p(wins, trials))
    emit("  what %d subjects can resolve, and a claim stronger than that is not supported."
         % s["n_subjects"])
    if not resolved:
        need = int(np.ceil(s["n_subjects"] * (np.std(s["diff"]) * 1.96 / max(1e-9, s["diff_point"])) ** 2))
        emit("  a rough extrapolation (interval shrinking as 1/sqrt(subjects)) puts the number of")
        emit("  subjects needed for a resolved difference at about %d." % need)
    emit()


def section_subjects(stats):
    emit("--- 4. per-subject focus top-1 ---")
    s = stats.get("focus_top1")
    if s is None:
        return
    emit("  %-24s %7s %10s %10s" % ("subject", "frames", "network", "baseline"))
    rows = sorted(zip(stats["order"], s["net_count"], s["net_total"], s["base_total"]),
                  key=lambda r: (r[2] - r[3]))
    for name, count, net, base in rows:
        if count <= 0:
            continue
        emit("  %-24s %7d %10.3f %10.3f" % (name, int(count), net / count, base / count))
    emit()


# ---------------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", default=os.path.join(ROOT, "runs", "invivo_bf_v2"))
    parser.add_argument("--groups", choices=("val", "test"), default="test")
    parser.add_argument("--cache", default=str(DPATH.output_dir("console_dl_cache_invivo_carotid")))
    parser.add_argument("--labels", default=os.path.join(
        ROOT, "data", "labels_console_invivo_carotid_target.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=REPORT_PATH)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    fold = os.path.join(args.run, "fold_0")
    model, builder, payload, front = load_combined(os.path.join(fold, "best_frontend.pt"),
                                                   os.path.join(fold, "best_backend.pt"),
                                                   args.device)
    group_by = payload["config"].get("group_by", "group")
    data = FieldIIData(args.cache, args.labels, device=args.device, ladders=payload["ladders"],
                       group_by=group_by, log=lambda *a: None)
    eval_idx = np.array([i for i in range(data.n) if data.splits[i] == args.groups], np.int64)
    train_idx = np.array([i for i in range(data.n) if data.splits[i] == "train"], np.int64)
    if not len(eval_idx):
        raise SystemExit("no label row has split=%s" % args.groups)

    emit("=========== subject-level bootstrap ===========")
    emit("  run       %s" % fold)
    emit("  split     %s: %d frames, %d subjects"
         % (args.groups, len(eval_idx), len(set(data.group_ids[i] for i in eval_idx))))
    emit("  model     front end epoch %s + back end epoch %s, on %s"
         % (front.get("epoch"), payload.get("epoch"), args.device))
    emit("  running the network over the split ...")

    with torch.no_grad():
        preds, tg = predict(model, data, builder, eval_idx, start="label", batch_size=args.batch_size)
    base_preds, base_tg = settings_lookup_baseline(data, train_idx, eval_idx, start="label")
    emit("  forward pass done in %.0f s" % (time.time() - started))
    emit()

    net_frames = per_frame(preds, tg)
    base_frames = per_frame(base_preds, base_tg)
    subjects = [data.group_ids[i] for i in eval_idx]
    order = sorted(set(subjects))
    rng = np.random.RandomState(args.seed)
    stats = {"subjects": len(order), "order": order}
    for name in net_frames:
        net_total, net_count = by_subject(net_frames[name][0], net_frames[name][1], subjects, order)
        base_total, base_count = by_subject(base_frames[name][0], base_frames[name][1], subjects, order)
        if net_count.sum() == 0:
            continue
        net, base, diff = bootstrap(net_total, net_count, base_total, base_count, args.draws, rng)
        keep = net_count > 0
        per_net = net_total[keep] / net_count[keep]
        per_base = base_total[keep] / base_count[keep]
        sign = -1.0 if name in LOWER_IS_BETTER else 1.0
        stats[name] = {
            "net_point": float(net_total.sum() / net_count.sum()),
            "base_point": float(base_total.sum() / base_count.sum()),
            "diff_point": float(net_total.sum() / net_count.sum()
                                - base_total.sum() / base_count.sum()),
            "net_ci": interval(net), "base_ci": interval(base), "diff_ci": interval(diff),
            "diff": diff, "n_subjects": int(keep.sum()),
            "net_total": net_total, "base_total": base_total, "net_count": net_count,
            "wins": int((sign * (per_net - per_base) > 0).sum()),
            "ties": int((per_net == per_base).sum()),
        }

    section_table(stats, args.draws)
    section_sign(stats)
    section_focus(stats, args.draws)
    section_subjects(stats)

    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(LINES)
    io.open(args.out, "w", encoding="utf-8", newline=chr(10)).write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
