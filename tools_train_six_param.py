# -*- coding: utf-8 -*-
"""训练六参数网络 SixParamNet（Field II 带噪数据预训练）。

输入：tools_build_fieldii_training_cache.py 的缓存 + data/labels_fieldii.jsonl。
输出：runs/<name>/fold_<k>/ 下
          best.pt           综合分数最高的轮次
          best_backend.pt   后端分数（增益方向 F1、滑块方向 F1）最高的轮次
          best_frontend.pt  前端分数（深度、频率、聚焦）最高的轮次
          last.pt           最后一轮
          metrics.csv       每轮一行，全部数值指标（列按所有轮次的并集写）
      runs/<name>/train_report.txt；--fold all 时另有 summary.json。

    验证方式

当前 1848 行全部是 split=train 的 11 个体模，只能按体模留出做交叉验证（--folds 4 --fold k）。
标签里出现 split=val 的行时（仿真全部完成、重跑标签后），自动改用 split 字段。
--fold none：全部体模训练，没有验证集。后端取 last.pt；前端在 --frontend-epoch 指定的轮次另存
frontend.pt（取交叉验证汇总里报告的前端最佳轮次中位数）。

    与 fieldii_v1 相比的三处改动（docs 见训练日志分析）

  1. 后端输出方式默认 optimum：网络预测最优增益 / 最优 TGC，修正量在网络外相减
     （--backend-output delta 恢复旧方式）。
  2. 前端与后端分开选检查点；前端头加大丢弃率（--frontend-dropout，默认 0.3）并做标签平滑
     （--frontend-label-smoothing，默认 0.1）。
  3. metrics.csv 保留全部验证指标（旧版表头按第 1 轮定下，第 1 轮不验证，非主要指标被丢掉）。

每折都打印体模物理量（衰减、电子噪声、声速）的训练/验证覆盖范围，验证集超出训练范围时标出
EXTRAPOLATES：fieldii_v3 的第 2 折就是验证集同时拿走衰减最低与最高的体模，那一折所有前端指标
都低 0.2 以上。

    fieldii_v2 日志分析后的改动

best_frontend.pt 默认改按 frontend_score_near 选：只统计当前设置与最优相差不超过 1 档的帧。
fieldii_v2 的闭环停点全在这一带，而整体单步准确率（聚焦 0.86）远高于停点上的准确率（0.50）。
--frontend-select frontend_score 恢复旧口径。

    每个 epoch

训练批次一律重抽后端起点（bmode_dl.labels.draw_start）并随机左右翻转；验证在
"标签记录的起点"上算指标，另报一次固定种子重抽起点的指标。检查点按前者选。

用法（服务器）：
    python tools_train_six_param.py --fold all --amp --name fieldii_v2
    python tools_train_six_param.py --fold none --epochs 150 --frontend-epoch 30 --amp --name fieldii_v2_all
    python tools_train_six_param.py --fold all --amp --backend-output delta --frontend-dropout 0.1 \
        --frontend-label-smoothing 0 --name fieldii_v1_repro

    实机微调（海信，labels_console.jsonl）

    python tools_train_six_param.py --cache data/console_dl_cache --labels data/labels_console.jsonl \
        --init-checkpoint runs/fieldii_v3_all/all_data/last.pt --freeze encoders --group-by placement \
        --scalar-norm data --fold all --epochs 60 --lr 1e-4 --warmup-epochs 2 --eval-every 1 --amp \
        --name console_ft_v1

--init-checkpoint 装入预训练权重；网络结构（输入方式、宽度、层数、后端输出方式、底噪输入）跟随
检查点。实机的深度 7 档、频率 9 档（谐波 5 档与基波 5 档的并集）与 Field II 不同，前端三个头的
最后一层重新初始化，其余全部装入。最优增益 / TGC 的标准化按实机训练集重新估计。
"""

import argparse
import csv
import io
import json
import math
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from bmode_dl.checkpoint import FREEZE_CHOICES, freeze, load_pretrained, model_from_config, save_checkpoint
from bmode_dl.dataset import GROUP_BY, INPUT_MODES, SCALAR_NORMS, FieldIIData, InputBuilder, make_batch
from bmode_dl.losses import DEFAULT_WEIGHTS, SixParamLoss
from bmode_dl.metrics import (MAIN_KEYS, NEAR_KEYS, compute_metrics, flatten_metrics, format_metrics,
                              predict)
from bmode_dl.model import BACKEND_OUTPUTS, count_parameters

BACKEND_KEYS = ("gain_mae_db", "gain_within_deadband", "gain_dir_f1_derived", "tgc_mae_db", "slider_dir_f1_derived")
FRONTEND_KEYS = ("depth_top1", "depth_within1", "depth_dir_f1_derived", "frequency_hit", "frequency_dir_f1_derived",
                 "focus_top1", "focus_within1", "focus_dir_f1_derived") + NEAR_KEYS
FRONTEND_SELECT = ("frontend_score_near", "frontend_score")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train the six-parameter network on Field II labels")
    p.add_argument("--cache", default="data/fieldii_dl_cache")
    p.add_argument("--labels", default="data/labels_fieldii.jsonl")
    p.add_argument("--runs", default="runs")
    p.add_argument("--name", default=None, help="run name (default: timestamp)")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--fold", default="0", help="fold index, 'all', or 'none' (train on everything)")
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=5.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--amp", action="store_true", help="float16 mixed precision (CUDA only)")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--patience", type=int, default=0,
                   help="stop after this many evaluations without improvement of BOTH sub-scores (0 = off)")
    p.add_argument("--input-mode", choices=INPUT_MODES, default="full")
    p.add_argument("--no-noise-floor", action="store_true")
    p.add_argument("--no-transformer", action="store_true")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--no-redraw", action="store_true", help="train on the recorded starts only (ablation)")
    p.add_argument("--d-model", dest="d_model", type=int, default=256)
    p.add_argument("--transformer-layers", dest="transformer_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--backend-output", dest="backend_output", choices=BACKEND_OUTPUTS, default="optimum",
                   help="optimum: predict optimal gain/TGC and subtract the current setting outside the network; "
                        "delta: predict the correction directly (fieldii_v1)")
    p.add_argument("--frontend-dropout", dest="frontend_dropout", type=float, default=0.3,
                   help="dropout inside the depth/frequency/focus heads")
    p.add_argument("--frontend-label-smoothing", dest="frontend_label_smoothing", type=float, default=0.1)
    p.add_argument("--frontend-select", choices=FRONTEND_SELECT, default="frontend_score_near",
                   help="metric that chooses best_frontend.pt: frontend_score_near counts only frames within one "
                        "ladder step of the optimum, which is where the closed loop actually stops")
    p.add_argument("--frontend-epoch", type=int, default=None,
                   help="with --fold none: also save frontend.pt at this epoch")
    p.add_argument("--borderline-weight", type=float, default=0.5)
    p.add_argument("--uncertainty-weighting", action="store_true")
    p.add_argument("--loss-weight", action="append", default=[], metavar="TERM=VALUE",
                   help="override a loss weight, e.g. --loss-weight aux=0; terms: %s" % ", ".join(DEFAULT_WEIGHTS))
    p.add_argument("--val-redraw-seed", type=int, default=12345)
    # ---- 微调（例如 Field II 预训练 -> 海信实机）
    p.add_argument("--init-checkpoint", default=None,
                   help="start from this pretrained checkpoint; its architecture settings (input mode, width, "
                        "layers, back-end output, noise-floor input) are taken over, ladder-dependent output "
                        "layers are re-initialised")
    p.add_argument("--freeze", choices=FREEZE_CHOICES, default="none",
                   help="encoders: freeze the image and profile encoders; backbone: train the output heads only")
    p.add_argument("--group-by", dest="group_by", choices=GROUP_BY, default="group",
                   help="fold unit: group = label group_id (Field II phantom); placement = console probe "
                        "placement (scene family, merging directories that differ only by a _GEN/_THI suffix), "
                        "so one placement never sits in both training and validation")
    p.add_argument("--scalar-norm", dest="scalar_norm", choices=SCALAR_NORMS, default="fixed",
                   help="fixed = Field II dB constants (fieldii_v1..v3); data = standardise gain, reference and "
                        "noise floor on the training data (use for console fine-tuning)")
    return p.parse_args(argv)


class Report(object):
    """终端与文本文件同内容（ASCII）。"""

    def __init__(self, path=None):
        self.path = path
        self.lines = []

    def __call__(self, text=""):
        text = str(text)
        print(text, flush=True)
        self.lines.append(text)
        if self.path:
            with io.open(self.path, "a", encoding="utf-8") as handle:
                handle.write(text + "\n")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_loss_weights(items):
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if key not in DEFAULT_WEIGHTS:
            raise SystemExit("unknown loss term %r" % key)
        out[key] = float(value)
    return out


def write_csv(path, rows):
    """每轮重写一次：列取所有轮次的并集，验证轮与非验证轮的列不同也不会丢。"""
    fixed = ["epoch", "lr", "seconds"]
    names = sorted(set(k for r in rows for k in r) - set(fixed))
    names = fixed + [k for k in names if k.startswith("train_")] + [k for k in names if not k.startswith("train_")]
    with io.open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def format_norm(norm):
    out = {}
    for k, v in norm.items():
        out[k] = [round(float(x), 3) for x in v] if isinstance(v, (list, tuple)) else round(float(v), 4)
    return json.dumps(out)


# 预训练检查点决定、微调时不能改的结构设置
ARCHITECTURE_KEYS = ("input_mode", "d_model", "transformer_layers", "no_transformer", "backend_output",
                     "no_noise_floor")


def inherit_architecture(args, report):
    """微调时网络结构跟随预训练检查点；命令行给了不同的值就报出来并以检查点为准。"""
    payload = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    defaults = {"input_mode": "full", "d_model": 256, "transformer_layers": 2, "no_transformer": False,
                "backend_output": "delta", "no_noise_floor": False}
    for key in ARCHITECTURE_KEYS:
        value = config.get(key, defaults[key])
        if getattr(args, key) != value:
            report("  --%s %s overridden by the pretrained checkpoint: %s"
                   % (key.replace("_", "-"), getattr(args, key), value))
        setattr(args, key, value)


def train_fold(args, data, fold, out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed + (0 if fold is None else int(fold)))
    device = data.device
    amp = bool(args.amp and device.type == "cuda")

    train_idx, val_idx, val_groups = data.split_indices(args.folds, fold, args.fold_seed)
    train_groups = sorted(set(data.group_ids[i] for i in train_idx))
    report("=========== fold %s ===========" % ("none" if fold is None else fold))
    report("  train: %d frames, %d groups" % (len(train_idx), len(train_groups)))
    report("  val:   %d frames, %d groups %s" % (len(val_idx), len(val_groups), val_groups))
    for line in data.label_summary(train_idx):
        report(line)
    coverage_lines, coverage = ([], {})
    if len(val_idx):
        coverage_lines, coverage = data.physics_coverage(train_idx, val_idx)
        if coverage_lines:
            report("  phantom physics coverage (validation outside the training range = that fold extrapolates)")
            for line in coverage_lines:
                report(line)

    norm = data.normalisation(train_idx, seed=args.seed)
    class_weights = data.class_weights(train_idx, seed=args.seed)
    report("  norm: %s" % format_norm(norm))
    report("  class weights: %s" % json.dumps({k: [round(float(x), 3) for x in v] for k, v in class_weights.items()}))

    builder = InputBuilder(data.rows_out, data.lines, data.source_rows, norm,
                           use_noise_floor=not args.no_noise_floor, scalar_norm=args.scalar_norm).to(device)
    config = vars(args).copy()
    model = model_from_config(config, data.ladders, norm)
    if args.init_checkpoint:
        load_pretrained(model, args.init_checkpoint, log=report)
    frozen_count, trainable_count = freeze(model, args.freeze)
    if args.freeze != "none":
        report("  freeze %s: %.2f M frozen, %.2f M trainable" % (args.freeze, frozen_count / 1e6,
                                                                 trainable_count / 1e6))
    model = model.to(device)
    criterion = SixParamLoss(class_weights, norm, parse_loss_weights(args.loss_weight),
                             borderline_weight=args.borderline_weight,
                             uncertainty_weighting=args.uncertainty_weighting,
                             frontend_label_smoothing=args.frontend_label_smoothing).to(device)
    report("  model: input_mode=%s  backend_output=%s  frontend_dropout=%.2f  label_smoothing=%.2f  parameters=%.2f M"
           % (args.input_mode, args.backend_output, args.frontend_dropout, args.frontend_label_smoothing,
              count_parameters(model) / 1e6))

    params = [p for p in list(model.parameters()) + list(criterion.parameters()) if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                                   {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    steps_per_epoch = max(1, int(math.ceil(len(train_idx) / float(args.batch))))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_rows = []
    cache_shape = {"rows": data.rows_out, "lines": data.lines, "source_rows": data.source_rows}
    extra = {"fold": fold, "val_groups": val_groups, "train_groups": train_groups}
    front_key = args.frontend_select
    selection = ("score", "backend_score", front_key)
    best = {name: {"score": -float("inf"), "epoch": -1, "metrics": None} for name in selection}
    files = {"score": "best.pt", "backend_score": "best_backend.pt", front_key: "best_frontend.pt"}
    marks = {"score": "*", "backend_score": "B", front_key: "F"}
    bad_evals = 0
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_idx)
        sums, batches = {}, 0
        for s in range(0, len(order), args.batch):
            batch_idx = order[s:s + args.batch]
            if len(batch_idx) < 2:
                continue
            flip = (not args.no_flip) and rng.rand() < 0.5
            inputs, targets = make_batch(data, builder, batch_idx, start="label" if args.no_redraw else "redraw",
                                         generator=generator, flip=flip)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(inputs)
            loss, logs = criterion(out, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v
            batches += 1
        train_logs = {k: v / max(1, batches) for k, v in sums.items()}

        row = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.time() - started, 1)}
        row.update({"train_" + k: round(v, 5) for k, v in train_logs.items()})
        line = "  epoch %3d  lr %.2e  loss %.4f" % (epoch, row["lr"], train_logs.get("total", float("nan")))

        evaluate_now = len(val_idx) > 0 and (epoch % args.eval_every == 0 or epoch == args.epochs)
        if evaluate_now:
            preds, tg = predict(model, data, builder, val_idx, start="label", amp=amp)
            metrics = compute_metrics(preds, tg, norm)
            preds_r, tg_r = predict(model, data, builder, val_idx, start="redraw", seed=args.val_redraw_seed, amp=amp)
            metrics_r = compute_metrics(preds_r, tg_r, norm)
            row.update(flatten_metrics(metrics, "val_"))
            row.update(flatten_metrics(metrics_r, "valredraw_"))
            line += "  | val %s" % format_metrics(metrics, ("score", "backend_score", front_key, "gain_mae_db",
                                                              "tgc_mae_db", "depth_top1", "frequency_hit",
                                                              "focus_top1"))
            improved = ""
            for name in selection:
                value = metrics.get(name, float("nan"))
                if np.isfinite(value) and value > best[name]["score"]:
                    best[name] = {"score": value, "epoch": epoch, "metrics": metrics}
                    save_checkpoint(os.path.join(out_dir, files[name]), model, config, data.ladders, norm,
                                    class_weights, cache_shape,
                                    dict(extra, epoch=epoch, selected_by=name, metrics=metrics,
                                         metrics_redraw=metrics_r))
                    improved += marks[name]
            if improved:
                line += "  " + improved
            # 早停只看两个分项：两个都没提高才算一次
            bad_evals = 0 if ("B" in improved or "F" in improved) else bad_evals + 1
        elif len(val_idx) == 0 and args.frontend_epoch and epoch == args.frontend_epoch:
            save_checkpoint(os.path.join(out_dir, "frontend.pt"), model, config, data.ladders, norm, class_weights,
                            cache_shape, dict(extra, epoch=epoch, selected_by="frontend_epoch"))
            line += "  saved frontend.pt"
        report(line)

        csv_rows.append(row)
        write_csv(csv_path, csv_rows)

        if args.patience and bad_evals >= args.patience:
            report("  early stop: %d evaluations without improvement of either sub-score" % bad_evals)
            break

    save_checkpoint(os.path.join(out_dir, "last.pt"), model, config, data.ladders, norm, class_weights, cache_shape,
                    dict(extra, epoch=epoch, selected_by="last"))
    if best["score"]["metrics"] is None:
        report("  no validation set; last.pt is the back-end model%s"
               % (", frontend.pt the front-end model" if args.frontend_epoch else ""))
        report("  fold finished in %.0f s" % (time.time() - started))
        report("")
        return None

    for name in selection:
        report("  best %-20s epoch %3d: %s" % (name, best[name]["epoch"], format_metrics(best[name]["metrics"])))
    # 组合：后端指标取 best_backend 那一轮，前端指标取 best_frontend 那一轮（即 CombinedModel 的单步表现）
    combined = {k: best["backend_score"]["metrics"].get(k) for k in BACKEND_KEYS + ("backend_score",)}
    combined.update({k: best[front_key]["metrics"].get(k)
                     for k in FRONTEND_KEYS + ("frontend_score", "frontend_score_near")})
    parts = [combined.get("backend_score"), combined.get("frontend_score")]
    combined["score"] = float(np.mean([p for p in parts if p is not None and np.isfinite(p)]))
    report("  combined (backend epoch %d + frontend epoch %d): %s"
           % (best["backend_score"]["epoch"], best[front_key]["epoch"], format_metrics(combined)))
    report("  fold finished in %.0f s" % (time.time() - started))
    report("")
    return {"best": best["score"]["metrics"], "combined": combined, "coverage": coverage,
            "epochs": {("frontend_score" if name == front_key else name): best[name]["epoch"] for name in best}}


def summarise(results, keys, section, report):
    summary = {}
    for key in keys:
        values = [r[section][key] for r in results
                  if isinstance(r[section].get(key), float) and np.isfinite(r[section][key])]
        if values:
            summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
            report("    %-26s %.4f +- %.4f  (n=%d)" % (key, summary[key]["mean"], summary[key]["std"], len(values)))
    return summary


def main(argv=None):
    args = parse_args(argv)
    name = args.name or time.strftime("six_param_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.runs, name)
    os.makedirs(run_dir, exist_ok=True)
    report = Report(os.path.join(run_dir, "train_report.txt"))
    report("=========== six-parameter network training ===========")
    report("  run dir %s" % run_dir)
    report("  torch %s, device %s, amp %s" % (torch.__version__, args.device, args.amp))
    report("  args %s" % json.dumps(vars(args), sort_keys=True))

    if args.init_checkpoint:
        inherit_architecture(args, report)
    data = FieldIIData(args.cache, args.labels, device=args.device, log=report, group_by=args.group_by)
    if data.uses_split_field():
        counts = Counter(s for s in data.splits)
        report("  source %s, split from the labels: %s (folds not used)"
               % (data.source, dict(sorted((str(k), v) for k, v in counts.items()))))
    else:
        report("  source %s, folds grouped by %s" % (data.source, args.group_by))
    report("  data: %d frames, %d groups (%s), cache %dx%d (source rows %d)"
           % (data.n, len(set(data.group_ids)), data.group_by, data.rows_out, data.lines, data.source_rows))
    report("  ladders %s" % json.dumps(data.ladders))
    report("")

    if args.fold == "none":
        folds = [None]
    elif args.fold == "all":
        folds = list(range(args.folds))
    else:
        folds = [int(args.fold)]

    # 标签自带 val 划分时不分折：每一折都会得到同一套训练 / 验证集，跑 4 折只是重复 4 次
    if data.uses_split_field() and folds != [None]:
        if len(folds) > 1:
            report("  NOTE: the labels carry a val split, so folds are ignored; running once instead of %d times"
                   % len(folds))
        folds = [0]

    results = {}
    for fold in folds:
        out_dir = os.path.join(run_dir, "all_data" if fold is None else "fold_%d" % fold)
        results["none" if fold is None else str(fold)] = train_fold(args, data, fold, out_dir, report)

    valid = {k: v for k, v in results.items() if v}
    if len(valid) > 1:
        report("=========== cross-validation summary (%d folds) ===========" % len(valid))
        report("  best.pt (single checkpoint chosen by the overall score)")
        best_summary = summarise(list(valid.values()), MAIN_KEYS, "best", report)
        report("  combined (best_backend.pt for gain/TGC + best_frontend.pt for depth/frequency/focus)")
        combined_summary = summarise(list(valid.values()), MAIN_KEYS, "combined", report)
        epochs = {name: [v["epochs"][name] for v in valid.values()] for name in ("score", "backend_score",
                                                                                 "frontend_score")}
        report("  best epochs per fold: %s" % json.dumps(epochs))
        report("  suggested --frontend-epoch for --fold none: %d (median of best front-end epochs)"
               % int(np.median(epochs["frontend_score"])))
        report("  note: the combined figures pick each epoch on the validation phantoms themselves, so they are "
               "optimistic in the same way best.pt already was")
        with io.open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"folds": valid, "summary_best": best_summary,
                                     "summary_combined": combined_summary, "best_epochs": epochs},
                                    indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
