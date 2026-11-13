# -*- coding: utf-8 -*-
"""可穿戴数据上后端（增益、TGC）微调的留一人评估：微调真的学到了逐帧信息吗。

    比较四样（都在留出的那个人上算，记录起点，口径同 measure_wear_zero_shot.py）

    zero-shot   海信人体最终模型（invivo_bf_final/all_data），不微调
    fine-tuned  runs/<run>/fold_k/last.pt（默认，固定最后一轮；best_backend.pt 是按留出者挑的轮次，偏乐观）：从同一个模型出发，在另外 3 人上只训练后端
    train mean  不看图：直接给训练的 3 人的平均最优增益与平均最优 TGC 曲线
    keep        什么都不改

微调时输出层的反标准化换成了可穿戴训练集的平均最优值（tools_train_six_param 的 “kept new-domain
normalisation”），这一步本身就把系统偏差吸收掉了。所以要看的是 fine-tuned 能不能赢 train mean：
赢了才说明网络在逐帧地看图，而不只是学到一条平均曲线。

用法：python tests/measure_wear_backend_finetune.py [--run runs/wear_bf_ft_v1] [--pick last|best_backend]
报告写在脚本旁边的 measure_wear_backend_finetune.txt（UTF-8），终端输出同内容（ASCII）。
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
from bmode_dl import constants as K
from bmode_dl.checkpoint import load_combined
from bmode_dl.dataset import FieldIIData
from bmode_dl.metrics import predict

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
ZERO_SHOT = ("runs/invivo_bf_final/all_data/frontend.pt", "runs/invivo_bf_final/all_data/last.pt")
METHODS = ("zero-shot", "fine-tuned", "train mean", "keep")
LINES = []


def emit(text=""):
    LINES.append(str(text))


def scores(pred_gain, pred_tgc, tg, rows, select):
    """gain MAE, share within the dead band, TGC MAE (all bands), TGC MAE bands 1-4 on 67 mm frames, gain r."""
    keep = (tg["backend_mask"] > 0) & select
    err = np.abs(pred_gain - tg["delta_gain_db"])[keep]
    band = np.abs(pred_tgc - tg["delta_tgc_db"])[keep]
    full = np.array([abs(r["depth_mm"] - 67.0) < 0.05 for r in rows])[keep]
    within = (np.abs(pred_gain - tg["delta_gain_db"]) <= tg["gain_deadband_db"])[keep]
    r = (np.corrcoef(pred_gain[keep], tg["delta_gain_db"][keep])[0, 1]
         if keep.sum() > 2 and np.std(pred_gain[keep]) > 1e-9 else float("nan"))
    return (float(err.mean()), float(within.mean()), float(band.mean()),
            float(band[full][:, :4].mean()) if full.any() else float("nan"), float(r), int(keep.sum()))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join(ROOT, "runs", "wear_bf_ft_v1"))
    p.add_argument("--cache", default=str(DPATH.output_dir("wear_dl_cache_carotid")))
    p.add_argument("--labels", default=os.path.join(ROOT, "data", "labels_wear_carotid.jsonl"))
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--pick", choices=("last", "best_backend"), default="last",
                   help="last = the fixed final epoch (no selection on the held-out subject); best_backend = the "
                        "epoch the trainer picked on that same subject, which leaks and is optimistic")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    started = time.time()

    zs_model, zs_builder, zs_payload, _ = load_combined(*(os.path.join(ROOT, q) for q in ZERO_SHOT), args.device)
    zs_data = FieldIIData(args.cache, args.labels, device=args.device, ladders=zs_payload["ladders"],
                          group_by="subject", log=lambda *a: None)
    data = FieldIIData(args.cache, args.labels, device=args.device, group_by="subject", log=lambda *a: None)
    if [r["frame_id"] for r in zs_data.rows] != [r["frame_id"] for r in data.rows]:
        raise SystemExit("the two label views are not in the same frame order")
    rows = data.rows
    slope = K.TGC_DB_PER_LEVEL[K.MODE_HARMONIC]
    kind = lambda r: "R'" if r["wear_step"] == "R'" else r["wear_step"][0]

    emit("=========== back-end fine-tuning on wearable data, leave one subject out ===========")
    emit("  run   %s, checkpoint %s.pt per fold" % (os.path.relpath(args.run, ROOT), args.pick))
    emit("  data  %d frames, %d subjects" % (data.n, len(set(data.group_ids))))
    emit("  metrics on the held-out subject, recorded starts: gain MAE dB | within dead band | "
         "TGC MAE dB (8 bands) | TGC MAE dB bands 1-4 (0-33 mm, 67 mm frames) | gain r")
    emit()

    pooled = {m: {"gain": [], "tgc": [], "tg": [], "idx": []} for m in METHODS}
    per_fold = []
    for fold in range(args.folds):
        train_idx, val_idx, val_groups = data.split_indices(args.folds, fold, 0)
        checkpoint = os.path.join(args.run, "fold_%d" % fold, args.pick + ".pt")
        model, builder, payload, _ = load_combined(checkpoint, checkpoint, args.device)
        with torch.no_grad():
            ft, tg = predict(model, data, builder, val_idx, start="label", batch_size=32)
            zs, _tg2 = predict(zs_model, zs_data, zs_builder, val_idx, start="label", batch_size=32)
        e = data.encoded
        mean_gain = float(e["optimal_gain_db"][train_idx].mean())
        mean_levels = e["optimal_tgc_levels"][train_idx].mean(axis=0)
        mean_pred_gain = mean_gain - e["gain_db"][val_idx]
        mean_pred_tgc = (mean_levels[None, :] - e["tgc_levels"][val_idx]) * slope
        preds = {"zero-shot": (zs["gain_delta_db"], zs["tgc_delta_db"]),
                 "fine-tuned": (ft["gain_delta_db"], ft["tgc_delta_db"]),
                 "train mean": (mean_pred_gain, mean_pred_tgc),
                 "keep": (np.zeros(len(val_idx)), np.zeros((len(val_idx), K.NUM_TGC_BANDS)))}
        subject = rows[val_idx[0]]["wear_session"]
        fold_rows = [rows[i] for i in val_idx]
        emit("--- fold %d: held out %s (%d frames), epoch %s ---"
             % (fold, subject, len(val_idx), payload.get("epoch")))
        for method in METHODS:
            g, t = preds[method]
            s = scores(g, t, tg, fold_rows, np.ones(len(val_idx), bool))
            emit("  %-11s %6.3f  %5.3f  %6.3f  %6.3f  %+5.2f" % ((method,) + s[:5]))
            pooled[method]["gain"].append(g)
            pooled[method]["tgc"].append(t)
            pooled[method]["tg"].append(tg)
            pooled[method]["idx"].append(val_idx)
            per_fold.append((fold, subject, method) + s[:4])
        emit()

    emit("=========== pooled over the four held-out subjects ===========")
    all_idx = np.concatenate(pooled["keep"]["idx"])
    all_rows = [rows[i] for i in all_idx]
    tg_all = {k: np.concatenate([t[k] for t in pooled["keep"]["tg"]]) for k in pooled["keep"]["tg"][0]}
    kinds = collections.OrderedDict((k, np.array([kind(r) == k for r in all_rows])) for k in ("R'", "R", "F", "Z", "D", "C"))
    emit("  %-11s %8s %8s %8s %8s %7s" % ("method", "gain", "in-band", "TGC", "TGC1-4", "gain r"))
    for method in METHODS:
        g = np.concatenate(pooled[method]["gain"])
        t = np.concatenate(pooled[method]["tgc"])
        s = scores(g, t, tg_all, all_rows, np.ones(len(all_idx), bool))
        emit("  %-11s %8.3f %8.3f %8.3f %8.3f %+7.2f" % ((method,) + s[:5]))
    emit("  (Hisense test, in domain: gain 1.366, in-band 0.497, TGC 0.262)")
    emit()
    emit("  TGC MAE (dB, 8 bands) by step kind")
    emit("  %-11s" % "method" + "".join("%8s" % k for k in kinds))
    for method in METHODS:
        g = np.concatenate(pooled[method]["gain"])
        t = np.concatenate(pooled[method]["tgc"])
        emit("  %-11s" % method + "".join("%8.3f" % scores(g, t, tg_all, all_rows, sel)[2] for sel in kinds.values()))
    emit()
    emit("  per-band TGC MAE on 67 mm frames (bands of 8.4 mm from the top)")
    full = np.array([abs(r["depth_mm"] - 67.0) < 0.05 for r in all_rows]) & (tg_all["backend_mask"] > 0)
    for method in METHODS:
        t = np.concatenate(pooled[method]["tgc"])
        emit("  %-11s " % method + " ".join("%5.2f" % v for v in np.abs(t - tg_all["delta_tgc_db"])[full].mean(axis=0)))
    emit()
    # does the network track frame-to-frame changes, beyond the subject-level mean?
    emit("  frame-to-frame: residual after removing each block's own mean (gain dB MAE, TGC dB MAE)")
    blocks = np.array(["%s/%s" % (r["wear_session"], r["wear_block"]) for r in all_rows])
    for method in METHODS:
        g = np.concatenate(pooled[method]["gain"])
        t = np.concatenate(pooled[method]["tgc"])
        eg, et = [], []
        for b in sorted(set(blocks)):
            m = (blocks == b) & (tg_all["backend_mask"] > 0)
            dg = (g - tg_all["delta_gain_db"])[m]
            dt = (t - tg_all["delta_tgc_db"])[m]
            eg.append(np.abs(dg - dg.mean()))
            et.append(np.abs(dt - dt.mean(axis=0)).mean(axis=1))
        emit("  %-11s gain %.3f  TGC %.3f" % (method, float(np.concatenate(eg).mean()), float(np.concatenate(et).mean())))
    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(LINES)
    io.open(REPORT_PATH, "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
