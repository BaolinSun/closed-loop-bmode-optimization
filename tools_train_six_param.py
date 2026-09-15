# -*- coding: utf-8 -*-
"""训练六参数网络 SixParamNet（Field II 带噪数据预训练）。

输入：tools_build_fieldii_training_cache.py 的缓存 + data/labels_fieldii.jsonl。
输出：runs/<name>/fold_<k>/{best.pt, last.pt, metrics.csv, train_report.txt}，
      --fold all 时另有 runs/<name>/summary.json 与 summary.txt。

    验证方式

当前 1848 行全部是 split=train 的 11 个体模，只能按体模留出做交叉验证（--folds 4 --fold k）。
标签里出现 split=val 的行时（仿真全部完成、重跑标签后），自动改用 split 字段。
--fold none：全部体模训练固定轮数，没有验证集，产出部署/微调起点用的权重。

    每个 epoch

训练批次一律重抽后端起点（bmode_dl.labels.draw_start）并随机左右翻转；验证在
"标签记录的起点"上算指标，另报一次固定种子重抽起点的指标。best.pt 按前者的综合分数选。

用法（服务器）：
    python tools_train_six_param.py --fold all --amp --name fieldii_v1
    python tools_train_six_param.py --fold none --epochs 150 --amp --name fieldii_v1_all
    python tools_train_six_param.py --fold 0 --input-mode no_image --name ablation_no_image
"""

import argparse
import csv
import io
import json
import math
import os
import random
import time

import numpy as np
import torch

from bmode_dl.checkpoint import model_from_config, save_checkpoint
from bmode_dl.dataset import INPUT_MODES, FieldIIData, InputBuilder, make_batch
from bmode_dl.losses import DEFAULT_WEIGHTS, SixParamLoss
from bmode_dl.metrics import MAIN_KEYS, compute_metrics, format_metrics, predict
from bmode_dl.model import count_parameters


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
    p.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without improvement (0 = off)")
    p.add_argument("--input-mode", choices=INPUT_MODES, default="full")
    p.add_argument("--no-noise-floor", action="store_true")
    p.add_argument("--no-transformer", action="store_true")
    p.add_argument("--no-flip", action="store_true")
    p.add_argument("--no-redraw", action="store_true", help="train on the recorded starts only (ablation)")
    p.add_argument("--d-model", dest="d_model", type=int, default=256)
    p.add_argument("--transformer-layers", dest="transformer_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--borderline-weight", type=float, default=0.5)
    p.add_argument("--uncertainty-weighting", action="store_true")
    p.add_argument("--loss-weight", action="append", default=[], metavar="TERM=VALUE",
                   help="override a loss weight, e.g. --loss-weight aux=0; terms: %s" % ", ".join(DEFAULT_WEIGHTS))
    p.add_argument("--val-redraw-seed", type=int, default=12345)
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


def train_fold(args, data, fold, out_dir, report):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(args.seed + (0 if fold is None else int(fold)))
    device = data.device
    amp = bool(args.amp and device.type == "cuda")

    train_idx, val_idx, val_groups = data.split_indices(args.folds, fold, args.fold_seed)
    train_groups = sorted(set(data.group_ids[i] for i in train_idx))
    report("=========== fold %s ===========" % ("none" if fold is None else fold))
    report("  train: %d frames, %d phantoms" % (len(train_idx), len(train_groups)))
    report("  val:   %d frames, %d phantoms %s" % (len(val_idx), len(val_groups), val_groups))
    for line in data.label_summary(train_idx):
        report(line)

    norm = data.normalisation(train_idx, seed=args.seed)
    class_weights = data.class_weights(train_idx, seed=args.seed)
    report("  norm: %s" % json.dumps({k: round(v, 4) for k, v in norm.items()}))
    report("  class weights: %s" % json.dumps({k: [round(float(x), 3) for x in v] for k, v in class_weights.items()}))

    builder = InputBuilder(data.rows_out, data.lines, data.source_rows, norm,
                           use_noise_floor=not args.no_noise_floor).to(device)
    config = vars(args).copy()
    model = model_from_config(config, data.ladders).to(device)
    criterion = SixParamLoss(class_weights, norm, parse_loss_weights(args.loss_weight),
                             borderline_weight=args.borderline_weight,
                             uncertainty_weighting=args.uncertainty_weighting).to(device)
    report("  model: input_mode=%s  parameters=%.2f M" % (args.input_mode, count_parameters(model) / 1e6))

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
    csv_file = io.open(csv_path, "w", encoding="utf-8", newline="")
    writer = None
    cache_shape = {"rows": data.rows_out, "lines": data.lines, "source_rows": data.source_rows}
    best_score, best_epoch, best_metrics, bad_evals = -float("inf"), -1, None, 0
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
            row.update({"val_" + k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            row.update({"valredraw_" + k: v for k, v in metrics_r.items() if isinstance(v, (int, float))})
            line += "  | val %s" % format_metrics(metrics, ("score", "gain_mae_db", "tgc_mae_db", "depth_top1",
                                                              "frequency_hit", "focus_top1"))
            if metrics["score"] > best_score:
                best_score, best_epoch, best_metrics, bad_evals = metrics["score"], epoch, metrics, 0
                save_checkpoint(os.path.join(out_dir, "best.pt"), model, config, data.ladders, norm, class_weights,
                                cache_shape, {"epoch": epoch, "fold": fold, "val_groups": val_groups,
                                              "train_groups": train_groups, "metrics": metrics,
                                              "metrics_redraw": metrics_r})
                line += "  *"
            else:
                bad_evals += 1
        report(line)

        if writer is None:
            fieldnames = sorted(set(row) | {"val_" + k for k in MAIN_KEYS} | {"valredraw_" + k for k in MAIN_KEYS})
            fieldnames = ["epoch", "lr", "seconds"] + [f for f in fieldnames if f not in ("epoch", "lr", "seconds")]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
        writer.writerow(row)
        csv_file.flush()

        if args.patience and bad_evals >= args.patience:
            report("  early stop: %d evaluations without improvement" % bad_evals)
            break

    csv_file.close()
    save_checkpoint(os.path.join(out_dir, "last.pt"), model, config, data.ladders, norm, class_weights, cache_shape,
                    {"epoch": epoch, "fold": fold, "val_groups": val_groups, "train_groups": train_groups})
    if best_metrics is None:
        report("  no validation set; last.pt is the model")
    else:
        report("  best epoch %d: %s" % (best_epoch, format_metrics(best_metrics)))
    report("  fold finished in %.0f s" % (time.time() - started))
    report("")
    return best_metrics


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

    data = FieldIIData(args.cache, args.labels, device=args.device, log=report)
    report("  data: %d frames, %d phantoms, cache %dx%d (source rows %d)"
           % (data.n, len(set(data.group_ids)), data.rows_out, data.lines, data.source_rows))
    report("  ladders %s" % json.dumps(data.ladders))
    report("")

    if args.fold == "none":
        folds = [None]
    elif args.fold == "all":
        folds = list(range(args.folds))
    else:
        folds = [int(args.fold)]

    results = {}
    for fold in folds:
        out_dir = os.path.join(run_dir, "all_data" if fold is None else "fold_%d" % fold)
        results["none" if fold is None else str(fold)] = train_fold(args, data, fold, out_dir, report)

    valid = {k: v for k, v in results.items() if v}
    if len(valid) > 1:
        report("=========== cross-validation summary (%d folds) ===========" % len(valid))
        summary = {}
        for key in MAIN_KEYS:
            values = [v[key] for v in valid.values() if isinstance(v.get(key), float) and np.isfinite(v[key])]
            if values:
                summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}
                report("  %-26s %.4f +- %.4f  (n=%d)" % (key, summary[key]["mean"], summary[key]["std"], len(values)))
        with io.open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"folds": valid, "summary": summary}, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
