# -*- coding: utf-8 -*-
"""六参数网络代码的不变量检查（不需要 HDF5 与 GPU，本地 cubdl 环境可跑）。

  1. bmode_dl.constants 与 bmode_opt 里的原值一致。
  2. torch 渲染与 hisense_backend_sim.render(db_image=...) 逐像素一致（缓存行数 = 原始行数时）；
     降采样行数下 TGC 插值矩阵与 np.interp 一致。
  3. 用 labels_fieldii.jsonl 记录的起点重算 delta 与方向，与记录值一致；重抽起点落在 draw_start 的范围内。
  4. 合成缓存上的端到端冒烟：三种输入模式前向 + 反向、无定出标签时损失为 0、检查点存取、
     训练脚本 1 个 epoch、评估脚本（单步 + 查表基线 + 闭环）。

用法：python tests/verify_six_param_model.py [--labels data/labels_fieldii.jsonl] [--skip-scripts]
"""

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))

import numpy as np
import torch

import bmode_dl.constants as K
from bmode_dl import labels as L
from bmode_dl.checkpoint import load_checkpoint, save_checkpoint, model_from_config
from bmode_dl.closed_loop import run_closed_loop
from bmode_dl.dataset import FieldIIData, InputBuilder, make_batch, read_jsonl
from bmode_dl.losses import SixParamLoss
from bmode_dl.metrics import compute_metrics, predict, settings_lookup_baseline
from bmode_dl.model import count_parameters
from bmode_dl.render import BackendRenderer, row_positions, tgc_interp_matrix
from bmode_dl.render import band_centres as dl_band_centres

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
FAILURES = []
LINES = []


def emit(text=""):
    print(text, flush=True)
    LINES.append(str(text))


def check(name, ok, detail=""):
    emit("  [%s] %s%s" % ("PASS" if ok else "FAIL", name, ("  " + detail) if detail else ""))
    if not ok:
        FAILURES.append(name)


# ---------------------------------------------------------------------------------------
def check_constants():
    emit("=========== 1. constants ===========")
    import hisense_backend_sim as HB
    import hisense_loader as HL
    import labels as BL
    check("gain dB per level", HB.GAIN_DB_PER_LEVEL_BY_MODE == K.GAIN_DB_PER_LEVEL)
    check("TGC dB per level", HB.TGC_DB_PER_LEVEL_BY_MODE == K.TGC_DB_PER_LEVEL)
    check("TGC range/centre", (HB.TGC_MIN_LEVEL, HB.TGC_MAX_LEVEL, HB.TGC_CENTER_LEVEL)
          == (K.TGC_MIN_LEVEL, K.TGC_MAX_LEVEL, K.TGC_CENTER_LEVEL))
    check("TGC bands", HL.NUM_TGC_BANDS == K.NUM_TGC_BANDS)
    check("gray pivot/max", (HB.GRAY_PIVOT, HB.GRAY_MAX) == (K.GRAY_PIVOT, K.GRAY_MAX))
    check("DR window", (HB.DR_WINDOW_SLOPE, HB.DR_WINDOW_INTERCEPT) == (K.DR_WINDOW_SLOPE, K.DR_WINDOW_INTERCEPT))
    check("start ranges", (BL.FIELDII_GAIN_DELTA_CLICKS, BL.FIELDII_SLIDER_TILT_LEVELS, BL.FIELDII_SLIDER_ARCH_LEVELS)
          == (K.START_GAIN_DELTA_CLICKS, K.START_SLIDER_TILT_LEVELS, K.START_SLIDER_ARCH_LEVELS))
    check("slider groups", {n: (lo, hi) for n, lo, hi in K.SLIDER_GROUPS} == BL.SLIDER_GROUPS)
    check("direction names", (BL.GAIN_DIRECTIONS, BL.SLIDER_DIRECTIONS, BL.DYNAMIC_RANGE_DIRECTIONS)
          == (K.GAIN_DIRECTIONS, K.SLIDER_DIRECTIONS, K.DYNAMIC_RANGE_DIRECTIONS))
    tilt_a, arch_a = BL.slider_shape_basis()
    tilt_b, arch_b = L.slider_shape_basis()
    check("slider shape basis", np.allclose(tilt_a, tilt_b) and np.allclose(arch_a, arch_b))
    check("band centres", np.allclose(HL.band_centres(1851), dl_band_centres(1851)))
    emit("")


def check_render():
    emit("=========== 2. render ===========")
    import hisense_backend_sim as HB
    rng = np.random.RandomState(0)
    rows, lines = K.FIELDII_SOURCE_ROWS, 16
    renderer = BackendRenderer(rows, rows)
    worst = 0
    for trial in range(6):
        db = -60.0 + 25.0 * rng.randn(rows, lines)
        mode = trial % 2
        gain = float(rng.uniform(-8, 12))
        levels = rng.randint(0, 256, size=8)
        dr_ui = float(rng.uniform(30, 200))
        ref = float(rng.uniform(-70, -15))
        expected = HB.render(db_image=db, tgc_levels=levels, gain_db=gain, dynamic_range_db=HB.dr_ui_to_window_db(dr_ui),
                             reference_db=ref, db_per_level=HB.TGC_DB_PER_LEVEL_BY_MODE[mode]).astype(np.float64)
        got = renderer(torch.tensor(db[None]), torch.tensor([gain], dtype=torch.float64),
                       torch.tensor(levels[None], dtype=torch.float64), torch.tensor([ref], dtype=torch.float64),
                       torch.tensor([dr_ui], dtype=torch.float64), torch.tensor([float(mode)], dtype=torch.float64))
        worst = max(worst, int(np.abs(got[0].numpy() - expected).max()))
    check("torch render == hisense_backend_sim.render (1851 rows)", worst == 0, "max abs gray diff %d" % worst)

    matrix = tgc_interp_matrix(512)
    levels = rng.randint(0, 256, size=8).astype(np.float64)
    curve_db = (levels - 127) * 0.08226
    import hisense_loader as HL
    expected = np.interp(row_positions(512), HL.band_centres(1851), curve_db)
    check("TGC interpolation matrix at 512 rows", np.allclose(matrix @ curve_db, expected, atol=1e-9))
    full = HB.expand_tgc_to_depth(levels, 1851, 0.08226)
    edges = np.linspace(0, 1851, 513).round().astype(int)
    block_mean = np.add.reduceat(full, edges[:-1]) / np.diff(edges)
    diff = np.abs(block_mean - matrix @ curve_db).max()
    check("block-mean TGC vs block-centre TGC (512 rows)", diff < 0.05, "max %.4f dB" % diff)
    emit("")


def check_labels(label_path):
    emit("=========== 3. label re-derivation (%s) ===========" % label_path)
    rows = read_jsonl(label_path)
    ladders = L.collect_ladders(rows)
    enc = L.encode_rows(rows, ladders)
    emit("  rows %d, ladders %s" % (len(rows), json.dumps(ladders)))
    t = {k: torch.from_numpy(v) for k, v in enc.items()}
    b = L.backend_targets(t["optimal_gain_db"].double(), t["optimal_tgc_levels"].double(), t["gain_db"].double(),
                          t["tgc_levels"].double(), t["deadband_gain_levels"].double(), t["mode"].double())
    m = enc["backend_mask"] > 0
    gd = np.abs(b["delta_gain_db"].numpy()[m] - enc["delta_gain_db"][m]).max()
    td = np.abs(b["delta_tgc_db"].numpy()[m] - enc["delta_tgc_db"][m]).max()
    check("delta_gain_db recomputed", gd < 1e-3, "max diff %.2e dB" % gd)
    check("delta_tgc_db recomputed", td < 1e-3, "max diff %.2e dB" % td)
    ga = (b["gain_dir"].numpy()[m] == enc["gain_dir"][m]).mean()
    sa = (b["slider_dir"].numpy()[m] == enc["slider_dir"][m]).mean()
    check("gain direction recomputed", ga == 1.0, "agreement %.4f" % ga)
    check("slider directions recomputed", sa == 1.0, "agreement %.4f" % sa)

    for axis in ("depth", "focus", "frequency"):
        mask = enc["%s_mask" % axis] > 0
        cur, opt = enc["%s_idx" % axis][mask], enc["optimal_%s_idx" % axis][mask]
        derived = np.where(opt > cur, 0, np.where(opt == cur, 1, 2))
        agree = (derived == enc["%s_dir" % axis][mask]).mean()
        check("%s direction == sign(optimum - current)" % axis, agree == 1.0, "agreement %.4f, n %d" % (agree, mask.sum()))
        valid = enc["%s_valid" % axis][mask]
        check("%s optimum inside valid ladder mask" % axis, bool(valid[np.arange(mask.sum()), opt].all()))
    fmask = enc["frequency_mask"] > 0
    check("optimal frequency inside acceptable set",
          bool(enc["frequency_acceptable"][fmask][np.arange(fmask.sum()), enc["optimal_frequency_idx"][fmask]].all()))

    gen = torch.Generator().manual_seed(1)
    g, lv = L.draw_start(t["optimal_gain_db"], t["optimal_tgc_levels"], t["mode"], gen)
    clicks = (t["optimal_gain_db"] - g) / K.GAIN_DB_PER_LEVEL[0]
    check("redrawn gain clicks within draw_start range",
          bool((clicks >= K.START_GAIN_DELTA_CLICKS[0] - 1e-3).all() and (clicks <= K.START_GAIN_DELTA_CLICKS[1] + 1e-3).all()),
          "min %.2f max %.2f" % (float(clicks.min()), float(clicks.max())))
    check("redrawn sliders are integers in 0..255",
          bool((lv == torch.round(lv)).all() and (lv >= 0).all() and (lv <= 255).all()))
    rb = L.backend_targets(t["optimal_gain_db"], t["optimal_tgc_levels"], g, lv, t["deadband_gain_levels"], t["mode"])
    emit("  redrawn gain direction counts %s (recorded %s)"
         % (np.bincount(rb["gain_dir"].numpy(), minlength=3).tolist(),
            np.bincount(enc["gain_dir"][enc["gain_dir"] >= 0], minlength=3).tolist()))
    emit("")
    return rows


# ---------------------------------------------------------------------------------------
def write_synthetic_cache(rows, out_dir, cache_rows=128, lines=32, seed=0):
    """合成缓存：dB 随深度衰减 + 斑点，底噪常数。只用于冒烟，不代表真实数据。"""
    rng = np.random.RandomState(seed)
    n = len(rows)
    z = np.linspace(0, 1, cache_rows)[None, :, None]
    depth = np.array([r["depth_mm"] for r in rows], np.float32)
    freq = np.array([r["frequency_mhz"] for r in rows], np.float32)
    db = (-20.0 - 0.7 * freq[:, None, None] * depth[:, None, None] / 10.0 * z * 10.0 / 2.0
          + 5.6 * rng.randn(n, cache_rows, lines)).astype(np.float32)
    floor = np.full((n, cache_rows), -90.0, np.float32)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "cache.npz"), db=db, floor=floor,
             frame_id=np.array([r["frame_id"] for r in rows]),
             min_depth_mm=np.full(n, 0.5, np.float32), max_depth_mm=depth)
    with io.open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"rows": cache_rows, "lines": lines, "source_rows": K.FIELDII_SOURCE_ROWS,
                                 "frames": n}) + "\n")


def check_smoke(rows, skip_scripts):
    emit("=========== 4. synthetic end-to-end smoke test (CPU) ===========")
    groups = sorted(set(r["group_id"] for r in rows))
    # 两个均匀 + 一个囊肿体模，保证分折有训练集和验证集
    picked = [g for g in groups if "/uniform/" in g][:2] + [g for g in groups if "/cyst/" in g][:1]
    subset = [r for r in rows if r["group_id"] in picked]
    tmp = tempfile.mkdtemp(prefix="six_param_verify_")
    try:
        cache_dir = os.path.join(tmp, "cache")
        label_path = os.path.join(tmp, "labels.jsonl")
        with io.open(label_path, "w", encoding="utf-8") as handle:
            for r in subset:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_synthetic_cache(subset, cache_dir)
        emit("  synthetic cache: %d frames from %s" % (len(subset), picked))

        data = FieldIIData(cache_dir, label_path, device="cpu", log=emit)
        train_idx, val_idx, val_groups = data.split_indices(num_folds=3, fold=0)
        check("fold split is by phantom", not (set(data.group_ids[i] for i in train_idx) & set(val_groups)),
              "train %d / val %d, val %s" % (len(train_idx), len(val_idx), val_groups))
        norm = data.normalisation(train_idx)
        cw = data.class_weights(train_idx, redraws=2)
        builder = InputBuilder(data.rows_out, data.lines, data.source_rows, norm)
        gen = torch.Generator().manual_seed(0)
        inputs, tg = make_batch(data, builder, train_idx[:6], start="redraw", generator=gen, flip=True)
        check("input shapes", tuple(inputs["image"].shape) == (6, 3, 128, 32) and tuple(inputs["profile"].shape) == (6, 5, 128)
              and tuple(inputs["scalars"].shape) == (6, 17), "%s %s %s" % (tuple(inputs["image"].shape),
                                                                         tuple(inputs["profile"].shape),
                                                                         tuple(inputs["scalars"].shape)))
        check("inputs finite", all(bool(torch.isfinite(v).all()) for v in inputs.values()))

        for mode in ("full", "no_image", "params_only"):
            model = model_from_config({"input_mode": mode}, data.ladders)
            criterion = SixParamLoss(cw, norm, uncertainty_weighting=(mode == "full"))
            out = model(inputs)
            loss, logs = criterion(out, tg)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            ok = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(g).all()) for g in grads) and len(grads) > 0
            check("forward/backward input_mode=%s" % mode, ok,
                  "params %.2f M, loss %.3f, dr terms %.1f/%.1f" % (count_parameters(model) / 1e6, float(loss),
                                                                   logs["dr_reg"], logs["dr_dir"]))
            if mode == "full":
                nd = len(data.ladders["depth_mm"])
                shapes = {k: tuple(v.shape) for k, v in out.items()}
                check("output shapes", shapes["tgc_delta_db"] == (6, 8) and shapes["slider_dir"] == (6, 3, 3)
                      and shapes["depth_logits"] == (6, nd), str(shapes))
                check("dynamic range loss is zero without labels", logs["dr_reg"] == 0.0 and logs["dr_dir"] == 0.0)

        model = model_from_config({"input_mode": "full"}, data.ladders)
        ckpt = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(ckpt, model, {"input_mode": "full"}, data.ladders, norm, cw,
                        {"rows": data.rows_out, "lines": data.lines, "source_rows": data.source_rows},
                        {"val_groups": val_groups})
        model2, builder2, payload = load_checkpoint(ckpt)
        model.eval()
        with torch.no_grad():
            same = torch.allclose(model(inputs)["gain_delta_db"], model2(inputs)["gain_delta_db"], atol=1e-6)
        check("checkpoint round trip", bool(same))

        preds, ptg = predict(model2, data, builder2, val_idx, start="label", batch_size=32)
        metrics = compute_metrics(preds, ptg, norm)
        check("metrics computed", np.isfinite(metrics["score"]), "score %.3f (untrained)" % metrics["score"])
        bp, btg = settings_lookup_baseline(data, train_idx, val_idx)
        bm = compute_metrics(bp, btg)
        check("settings baseline computed", np.isfinite(bm["score"]), "score %.3f" % bm["score"])
        summary, records = run_closed_loop(model2, data, builder2, val_idx[:40], max_steps=3, batch_size=32)
        check("closed loop runs", summary["trajectories"] == 40, json.dumps(summary)[:200])

        if not skip_scripts:
            import tools_train_six_param as TT
            import tools_evaluate_six_param as TE
            runs = os.path.join(tmp, "runs")
            started = time.time()
            TT.main(["--cache", cache_dir, "--labels", label_path, "--runs", runs, "--name", "smoke",
                     "--folds", "3", "--fold", "0", "--epochs", "1", "--batch", "16", "--eval-every", "1",
                     "--device", "cpu", "--d-model", "64", "--transformer-layers", "1"])
            best = os.path.join(runs, "smoke", "fold_0", "best.pt")
            check("training script wrote best.pt", os.path.exists(best), "%.0f s" % (time.time() - started))
            TE.main(["--run", os.path.join(runs, "smoke"), "--cache", cache_dir, "--labels", label_path,
                     "--device", "cpu", "--redraw-seeds", "1", "--max-steps", "2"])
            check("evaluation script wrote reports",
                  os.path.exists(os.path.join(runs, "smoke", "fold_0", "evaluation_report.txt"))
                  and os.path.exists(os.path.join(runs, "smoke", "evaluation_summary.txt")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    emit("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=os.path.join(ROOT, "data", "labels_fieldii.jsonl"))
    parser.add_argument("--skip-scripts", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    check_constants()
    check_render()
    rows = check_labels(args.labels)
    emit("")
    check_smoke(rows, args.skip_scripts)
    emit("=========== result ===========")
    emit("  %d failure(s)%s" % (len(FAILURES), (": " + ", ".join(FAILURES)) if FAILURES else ""))
    captured = list(LINES)
    with io.open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(captured) + "\n")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
