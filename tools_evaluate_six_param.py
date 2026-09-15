# -*- coding: utf-8 -*-
"""评估六参数网络：单步指标、只看设置的查表基线、闭环优化仿真。

    --run runs/<name>                 评估其下每个折（fold_*，或 --fold none 的 all_data），并汇总
    --checkpoint runs/<name>/fold_0/best.pt
    --groups all                      评估全部体模（默认只评估检查点记录的验证体模）
    --select auto|combined|best|last  --run 时每个折用哪个模型：
        combined  best_frontend.pt 出前端三轴 + best_backend.pt 出增益/TGC（all_data 下为 frontend.pt + last.pt）
        best      best.pt（单一检查点；fieldii_v1 只有它）
        last      last.pt
        auto      有 combined 所需文件就用 combined，否则 best，再否则 last

写出 <折目录>/evaluation_report_<选择>.txt、evaluation_<选择>.json、closed_loop_trajectories_<选择>.jsonl，
--run 时另写 runs/<name>/evaluation_summary_<选择>.txt。终端输出与报告同内容（ASCII）。

注意：用 --groups all 评估一个折的检查点时包含训练体模，只能看拟合程度，不能当泛化结论。
"""

import argparse
import glob
import io
import json
import os

import numpy as np
import torch

import bmode_dl.constants as K
from bmode_dl.checkpoint import load_checkpoint, load_combined
from bmode_dl.closed_loop import run_closed_loop
from bmode_dl.dataset import FieldIIData
from bmode_dl.metrics import (MAIN_KEYS, compute_metrics, confusion, direction_from_delta_np,
                              direction_from_index_np, format_metrics, predict, settings_lookup_baseline)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate the six-parameter network")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run directory containing fold_*/best.pt")
    g.add_argument("--checkpoint", help="a single checkpoint")
    p.add_argument("--cache", default="data/fieldii_dl_cache")
    p.add_argument("--labels", default="data/labels_fieldii.jsonl")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--groups", default="val", help="'val' (checkpoint's validation phantoms) or 'all'")
    p.add_argument("--select", choices=("auto", "combined", "best", "last"), default="auto",
                   help="which model of each fold to evaluate with --run")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--redraw-seeds", type=int, default=3, help="extra evaluations on re-drawn back-end starts")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--stop-deadband-levels", type=float, default=0.5)
    p.add_argument("--no-closed-loop", action="store_true")
    return p.parse_args(argv)


class Report(object):
    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text, flush=True)
        self.lines.append(str(text))

    def save(self, path):
        with io.open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines) + "\n")


def resolve_fold(fold_dir, select):
    """一个折目录 -> (选择名, 前端文件, 后端文件)；单一检查点时前端文件为 None。"""
    def has(name):
        return os.path.exists(os.path.join(fold_dir, name))

    pairs = [("best_frontend.pt", "best_backend.pt"), ("frontend.pt", "last.pt")]
    pair = next((pr for pr in pairs if has(pr[0]) and has(pr[1])), None)
    if select in ("auto", "combined") and pair is not None:
        return "combined", os.path.join(fold_dir, pair[0]), os.path.join(fold_dir, pair[1])
    if select == "combined":
        raise SystemExit("%s has no best_frontend.pt/best_backend.pt (or frontend.pt/last.pt)" % fold_dir)
    for name, file_name in (("best", "best.pt"), ("last", "last.pt")):
        if select in ("auto", name) and has(file_name):
            return name, None, os.path.join(fold_dir, file_name)
    raise SystemExit("no usable checkpoint in %s for --select %s" % (fold_dir, select))


def evaluate_checkpoint(path, args, data_cache, frontend_path=None, label=None):
    report = Report()
    device = torch.device(args.device)
    if frontend_path is None:
        model, builder, payload = load_checkpoint(path, device)
        label = label or os.path.splitext(os.path.basename(path))[0]
        described = path
    else:
        model, builder, payload, front_payload = load_combined(frontend_path, path, device)
        label = label or "combined"
        described = "front-end %s (epoch %s) + back-end %s (epoch %s)" % (
            frontend_path, front_payload.get("epoch"), path, payload.get("epoch"))
    amp = bool(args.amp and device.type == "cuda")
    key = (args.cache, args.labels)
    if key not in data_cache:
        data_cache[key] = FieldIIData(args.cache, args.labels, device=args.device, ladders=payload["ladders"],
                                      log=report)
    data = data_cache[key]
    if data.ladders != payload["ladders"]:
        raise SystemExit("ladders in %s differ from the data" % path)

    val_groups = payload.get("val_groups") or []
    if args.groups == "all" or not val_groups:
        eval_idx = np.arange(data.n)
        scope = "all phantoms" + ("" if val_groups else " (checkpoint has no validation set)")
    else:
        eval_idx = data.indices_for_groups(val_groups)
        scope = "validation phantoms %s" % val_groups
    train_groups = payload.get("train_groups") or sorted(set(data.group_ids) - set(val_groups))
    train_idx = data.indices_for_groups(train_groups)

    report("=========== %s ===========" % described)
    report("  epoch %s, input_mode %s, backend_output %s, evaluated on %d frames: %s"
           % (payload.get("epoch"), payload["config"].get("input_mode"),
              payload["config"].get("backend_output", "delta"), len(eval_idx), scope))
    report("")

    results = {}
    report("----- single step, recorded starts -----")
    preds, tg = predict(model, data, builder, eval_idx, start="label", amp=amp)
    m = compute_metrics(preds, tg, payload["norm"])
    results["network_recorded_start"] = m
    report("  network  %s" % format_metrics(m))
    report("  TGC per-band MAE dB %s" % m.get("tgc_band_mae_db"))
    for axis in ("gain", "depth", "frequency", "focus"):
        if axis == "gain":
            backend = tg["backend_mask"] > 0
            slope = np.where(tg["mode"] > 0.5, K.GAIN_DB_PER_LEVEL[1], K.GAIN_DB_PER_LEVEL[0])
            derived = direction_from_delta_np(preds["gain_delta_db"] / slope, tg["deadband_gain_levels"])
            cm = confusion(tg["gain_dir"][backend], derived[backend])
        else:
            mask = tg["%s_mask" % axis] > 0
            derived = direction_from_index_np(preds["%s_idx" % axis], tg["%s_idx" % axis])
            cm = confusion(tg["%s_dir" % axis][mask], derived[mask])
        report("  %-9s direction confusion (rows = label increase/correct/decrease, cols = derived): %s"
               % (axis, cm.tolist()))

    if len(train_idx) and args.groups != "all":
        base_preds, base_tg = settings_lookup_baseline(data, train_idx, eval_idx, start="label")
        b = compute_metrics(base_preds, base_tg)
        results["settings_lookup_baseline"] = b
        report("  settings %s" % format_metrics(b))
        report("  (settings = median/mode optimum of training phantoms at the same depth/frequency/focus)")

    for s in range(args.redraw_seeds):
        seed = 1000 + s
        preds_r, tg_r = predict(model, data, builder, eval_idx, start="redraw", seed=seed, amp=amp)
        mr = compute_metrics(preds_r, tg_r, payload["norm"])
        results["network_redraw_seed%d" % seed] = mr
        report("  redraw seed %d  %s" % (seed, format_metrics(mr, ("gain_mae_db", "gain_within_deadband",
                                                                    "gain_dir_f1_derived", "tgc_mae_db",
                                                                    "slider_dir_f1_derived"))))
    report("  dynamic range: %d determined labels (no criterion yet; head is untrained)" % m["dynamic_range_n"])
    report("")

    trajectories = []
    if not args.no_closed_loop:
        report("----- closed loop (max %d steps, stop deadband %.2f gain levels) -----"
               % (args.max_steps, args.stop_deadband_levels))
        summary, trajectories = run_closed_loop(model, data, builder, eval_idx, max_steps=args.max_steps,
                                                amp=amp, stop_deadband_levels=args.stop_deadband_levels)
        results["closed_loop"] = summary
        for k, v in summary.items():
            report("  %-30s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
        report("")

    out_dir = os.path.dirname(os.path.abspath(path))
    report.save(os.path.join(out_dir, "evaluation_report_%s.txt" % label))
    with io.open(os.path.join(out_dir, "evaluation_%s.json" % label), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(results, indent=2, default=float) + "\n")
    if trajectories:
        with io.open(os.path.join(out_dir, "closed_loop_trajectories_%s.jsonl" % label), "w", encoding="utf-8") as handle:
            for record in trajectories:
                handle.write(json.dumps(record) + "\n")
    return results


def main(argv=None):
    args = parse_args(argv)
    data_cache = {}
    if args.checkpoint:
        evaluate_checkpoint(args.checkpoint, args, data_cache)
        return
    fold_dirs = (sorted(glob.glob(os.path.join(args.run, "fold_*")))
                 or sorted(glob.glob(os.path.join(args.run, "all_data"))))
    if not fold_dirs:
        raise SystemExit("no fold directory under %s" % args.run)
    specs = [resolve_fold(d, args.select) for d in fold_dirs]
    labels = sorted(set(spec[0] for spec in specs))
    all_results = [evaluate_checkpoint(back, args, data_cache, frontend_path=front, label=name)
                   for name, front, back in specs]

    report = Report()
    report("=========== summary over %d folds (%s, model: %s) ===========" % (len(specs), args.run, "/".join(labels)))
    for section in ("network_recorded_start", "settings_lookup_baseline"):
        report("  %s" % section)
        for key in MAIN_KEYS:
            values = [r[section][key] for r in all_results
                      if section in r and isinstance(r[section].get(key), float) and np.isfinite(r[section][key])]
            if values:
                report("    %-26s %.4f +- %.4f" % (key, np.mean(values), np.std(values)))
    if not args.no_closed_loop:
        report("  closed_loop")
        keys = [k for k, v in all_results[0]["closed_loop"].items() if isinstance(v, float)]
        for key in keys:
            values = [r["closed_loop"][key] for r in all_results if np.isfinite(r["closed_loop"][key])]
            if values:
                report("    %-30s %.4f +- %.4f" % (key, np.mean(values), np.std(values)))
    report.save(os.path.join(args.run, "evaluation_summary_%s.txt" % "_".join(labels)))


if __name__ == "__main__":
    main()
