# -*- coding: utf-8 -*-
"""零样本评估：海信人体颈动脉上训练的六参数模型，直接用在可穿戴线阵（新板，谐波）数据上。

    数据

tools_build_wear_dataset.py 生成的标签与缓存（4 位受试者、19 个采集块、768 帧）：
后端标签由同一个求解器在可穿戴帧上解出，目标亮度借海信人体的中位 62 灰；聚焦标签按人体暂定规则，
远壁深度是 Claude 读的暂定值。深度、频率两个头在人体模型里没有训练，不评。

    模型

    invivo_bf_v2/fold_0   海信测试集（29 人）上有现成成绩的那一折，用来与域内成绩直接对比
    invivo_bf_final       全部海信数据训练的最终模型（all_data：frontend.pt + last.pt）

    两种输入

    raw      可穿戴的 dB 原样输入（counts_per_db = 400 的精确刻度）
    aligned  图像 dB、曝光参考、底噪一起平移，使可穿戴的曝光参考（pivot）等于海信人体的中位 28.07 dB。
             三者同移，渲染出的灰度完全不变，只去掉两台设备 dB 原点的不同；剩下的差别（底噪相对
             pivot 高 9-10 dB、成像宽度、探头）才是真正的域差。

    指标（与 measure_subject_bootstrap.py 同一口径，记录起点）

    gain_mae_db / gain_within_deadband / tgc_mae_db   后端，全部帧
    focus_top1 / focus_within1                         聚焦，有聚焦标签的帧
    基线 keep：什么都不改（增益、TGC 改动为 0，聚焦留在当前档）

域内参照（海信测试集 29 人，invivo_bf_v2 fold_0，tests/measure_subject_bootstrap.txt）：
增益 1.366 dB、死区内 0.497、TGC 0.262 dB、聚焦 top1 0.330、差一档以内 0.731。

用法：python tests/measure_wear_zero_shot.py
报告写在脚本旁边的 measure_wear_zero_shot.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
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
from bmode_dl.checkpoint import load_combined
from bmode_dl.dataset import FieldIIData
from bmode_dl.metrics import predict

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
HISENSE_PIVOT_DB = 28.068          # median pivot_db of the 189 Hisense in-vivo groups
IN_DOMAIN = {"gain_mae_db": 1.366, "gain_within_deadband": 0.497, "tgc_mae_db": 0.262,
             "focus_top1": 0.330, "focus_within1": 0.731}
MODELS = {
    "v2_fold0": ("runs/invivo_bf_v2/fold_0/best_frontend.pt", "runs/invivo_bf_v2/fold_0/best_backend.pt"),
    "final": ("runs/invivo_bf_final/all_data/frontend.pt", "runs/invivo_bf_final/all_data/last.pt"),
}
METRICS = ("gain_mae_db", "gain_within_deadband", "tgc_mae_db", "focus_top1", "focus_within1")
LINES = []


def emit(text=""):
    LINES.append(str(text))


def per_frame(preds, tg):
    """{metric: (values, mask)}, the definitions of measure_subject_bootstrap.per_frame."""
    backend = tg["backend_mask"] > 0
    err = np.abs(preds["gain_delta_db"] - tg["delta_gain_db"])
    out = {"gain_mae_db": (err, backend),
           "gain_within_deadband": ((err <= tg["gain_deadband_db"]).astype(float), backend),
           "tgc_mae_db": (np.abs(preds["tgc_delta_db"] - tg["delta_tgc_db"]).mean(axis=1), backend)}
    mask = (tg["focus_mask"] > 0) & (tg["optimal_focus_idx"] >= 0)
    pred, target = preds["focus_idx"], tg["optimal_focus_idx"]
    out["focus_top1"] = ((pred == target).astype(float), mask)
    out["focus_within1"] = ((np.abs(pred - target) <= 1).astype(float), mask)
    return out


def keep_baseline(preds, data, idx):
    """Change nothing: zero back-end deltas, focus left at the current rung."""
    base = dict(preds)
    base["gain_delta_db"] = np.zeros_like(preds["gain_delta_db"])
    base["tgc_delta_db"] = np.zeros_like(preds["tgc_delta_db"])
    base["focus_idx"] = data.encoded["focus_idx"][idx]
    return base


def mean_of(values, mask, select=None):
    keep = mask if select is None else (mask & select)
    return float(values[keep].mean()) if keep.any() else float("nan"), int(keep.sum())


def table(title, frames, base_frames, groups):
    """Rows = metric; columns = all frames, then each group (session or step kind)."""
    emit("  %s" % title)
    names = ["all"] + list(groups)
    emit("    %-22s" % "metric" + "".join("%14s" % n for n in names))
    for metric in METRICS:
        values, mask = frames[metric]
        cells = []
        for n in names:
            sel = None if n == "all" else groups[n]
            v, k = mean_of(values, mask, sel)
            b, _ = mean_of(base_frames[metric][0], base_frames[metric][1], sel)
            cells.append("   nan        " if k == 0 else "%6.3f/%-6.3f " % (v, b))
        emit("    %-22s" % metric + "".join("%14s" % c for c in cells))
    emit("    (each cell: network / keep-current baseline)")


def run(name, data, idx, args, offset):
    front, back = (os.path.join(ROOT, p) for p in MODELS[name])
    model, builder, payload, _front = load_combined(front, back, args.device)
    if offset:
        data.db += offset
        data.floor += offset
        for key in ("reference_db", "noise_floor_top_db", "noise_floor_bottom_db"):
            data.t[key] += offset
    try:
        with torch.no_grad():
            preds, tg = predict(model, data, builder, idx, start="label", batch_size=args.batch_size)
    finally:
        if offset:
            data.db -= offset
            data.floor -= offset
            for key in ("reference_db", "noise_floor_top_db", "noise_floor_bottom_db"):
                data.t[key] -= offset
    return preds, tg, payload


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(DPATH.output_dir("wear_dl_cache_carotid")))
    p.add_argument("--labels", default=os.path.join(ROOT, "data", "labels_wear_carotid.jsonl"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args(argv)
    started = time.time()

    _m, _b, payload, _f = load_combined(*(os.path.join(ROOT, q) for q in MODELS["v2_fold0"]), args.device)
    data = FieldIIData(args.cache, args.labels, device=args.device, ladders=payload["ladders"],
                       group_by="subject", log=lambda *a: None)
    idx = np.arange(data.n, dtype=np.int64)
    rows = data.rows
    sessions = collections.OrderedDict(
        (s, np.array([r["wear_session"] == s for r in rows])) for s in sorted({r["wear_session"] for r in rows}))
    kind = lambda r: "R'" if r["wear_step"] == "R'" else r["wear_step"][0]
    kinds = collections.OrderedDict(
        (k, np.array([kind(r) == k for r in rows])) for k in ("R'", "R", "F", "Z", "D", "C"))
    low = np.array([r.get("wear_wall_confidence") == "low" for r in rows])
    wear_pivot = float(np.median(data.t["reference_db"].cpu().numpy()))
    offset = HISENSE_PIVOT_DB - wear_pivot

    emit("=========== zero-shot: Hisense in-vivo model on wearable carotid data ===========")
    emit("  data      %d frames, %d subjects (%s); focus labels on %d frames"
         % (data.n, len(sessions), ", ".join(sessions), int((data.encoded["focus_mask"] > 0).sum())))
    emit("  input     wear pivot %.2f dB, Hisense median pivot %.2f dB -> aligned offset %+.2f dB"
         % (wear_pivot, HISENSE_PIVOT_DB, offset))
    floor = float(np.median(data.t["noise_floor_top_db"].cpu().numpy()))
    emit("            noise floor minus pivot: wear %+.1f dB, Hisense in vivo about %+.1f dB (median floor 12.1)"
         % (floor - wear_pivot, 12.117 - HISENSE_PIVOT_DB))
    emit("            image dB: wear p50 %.1f (raw) / %.1f (aligned); Hisense training mean %.1f +- %.1f"
         % (float(data.db.median()), float(data.db.median()) + offset, payload["norm"]["db_mean"],
            payload["norm"]["db_std"]))
    emit("  in-domain reference (Hisense test, 29 subjects, v2 fold 0): " +
         ", ".join("%s %.3f" % kv for kv in IN_DOMAIN.items()))
    emit()

    summary = []
    for name in MODELS:
        for label, shift in (("raw", 0.0), ("aligned", offset)):
            preds, tg, pl = run(name, data, idx, args, shift)
            frames = per_frame(preds, tg)
            base = per_frame(keep_baseline(preds, data, idx), tg)
            emit("--- model %s, input %s ---" % (name, label))
            table("by subject", frames, base, sessions)
            table("by step kind", frames, base, kinds)
            # back-end bias: is the network systematically darker / brighter than the teacher?
            backend = tg["backend_mask"] > 0
            bias = float((preds["gain_delta_db"] - tg["delta_gain_db"])[backend].mean())
            tgc_bias = (preds["tgc_delta_db"] - tg["delta_tgc_db"])[backend].mean(axis=0)
            emit("  gain: network delta %+.2f +- %.2f dB, teacher delta %+.2f +- %.2f dB, bias %+.2f dB"
                 % (preds["gain_delta_db"][backend].mean(), preds["gain_delta_db"][backend].std(),
                    tg["delta_gain_db"][backend].mean(), tg["delta_gain_db"][backend].std(), bias))
            emit("  TGC bias per band (network - teacher, dB): " + " ".join("%+.2f" % v for v in tgc_bias))
            r = np.corrcoef(preds["gain_delta_db"][backend], tg["delta_gain_db"][backend])[0, 1]
            emit("  gain: correlation network vs teacher delta r = %+.2f (a constant guess has r = 0)" % r)
            band_err = np.abs(preds["tgc_delta_db"] - tg["delta_tgc_db"])[backend]
            full = np.array([abs(rr["depth_mm"] - 67.0) < 0.05 for rr in rows])[backend]
            emit("  TGC MAE per band, 67 mm frames (band k covers %s mm): %s"
                 % ("/".join("%.0f" % (67.0 * (k + 1) / 8) for k in range(8)),
                    " ".join("%.2f" % v for v in band_err[full].mean(axis=0))))
            emit("  TGC MAE bands 1-4 (0-33 mm, the carotid) on 67 mm frames: %.2f dB; teacher delta there %+.2f dB mean"
                 % (band_err[full][:, :4].mean(), tg["delta_tgc_db"][backend][full][:, :4].mean()))
            fmask = (tg["focus_mask"] > 0) & (tg["optimal_focus_idx"] >= 0)
            ladder = payload["ladders"]["focus_mm"]
            if fmask.any():
                pred_c = collections.Counter(ladder[i] for i in preds["focus_idx"][fmask])
                true_c = collections.Counter(ladder[i] for i in tg["optimal_focus_idx"][fmask])
                emit("  focus predicted %s" % dict(sorted(pred_c.items())))
                emit("  focus labelled  %s" % dict(sorted(true_c.items())))
                ok = fmask & ~low
                v, k = mean_of(frames["focus_top1"][0], ok)
                w, _ = mean_of(frames["focus_within1"][0], ok)
                emit("  focus on blocks with a clear far wall only: top1 %.3f, within1 %.3f (%d frames)" % (v, w, k))
            emit()
            summary.append((name, label) + tuple(mean_of(frames[m][0], frames[m][1])[0] for m in METRICS)
                           + tuple(mean_of(base[m][0], base[m][1])[0] for m in METRICS))

    emit("=========== summary (all frames) ===========")
    emit("  %-10s %-8s" % ("model", "input") + "".join("%22s" % m for m in METRICS))
    emit("  %-19s" % "Hisense test (ref)" + "".join("%22.3f" % IN_DOMAIN[m] for m in METRICS))
    for row in summary:
        emit("  %-10s %-8s" % row[:2] + "".join("%22.3f" % v for v in row[2:2 + len(METRICS)]))
    emit("  %-19s" % "keep current" + "".join("%22.3f" % v for v in summary[0][2 + len(METRICS):]))
    emit("  (for gain_mae_db and tgc_mae_db lower is better)")
    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(LINES)
    io.open(REPORT_PATH, "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
