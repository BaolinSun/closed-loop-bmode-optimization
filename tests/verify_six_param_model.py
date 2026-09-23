# -*- coding: utf-8 -*-
"""六参数网络代码的不变量检查（不需要 HDF5 与 GPU，本地 cubdl 环境可跑）。

  1. bmode_dl.constants 与 bmode_opt 里的原值一致。
  2. torch 渲染与 hisense_backend_sim.render(db_image=...) 逐像素一致（缓存行数 = 原始行数时）；
     降采样行数下 TGC 插值矩阵与 np.interp 一致。
  3. 用 labels_fieldii.jsonl 记录的起点重算 delta 与方向，与记录值一致；重抽起点落在 draw_start 的范围内。
  4. 合成缓存上的端到端冒烟：三种输入模式前向 + 反向、无定出标签时损失为 0、检查点存取、
     训练脚本 1 个 epoch、评估脚本（单步 + 查表基线 + 闭环）。
  5. fieldii_v1 日志分析后的改动：optimum 后端输出（最优值 - 当前值的算术、零初始化起点、旧检查点仍能载入）、
     前端标签平滑、前后端分开的检查点与组合模型、metrics.csv 保留全部验证指标。
  6. fieldii_v2 日志分析后的改动：近最优帧指标与期望档决策、闭环的滞回与打转冻结、逐步路径记录。
  7. 实机微调：按探头摆放分组（_GEN/_THI 目录合并）、缓存覆盖标签行的参考与底噪、标量按数据标准化、
     预训练权重部分装入（档位不同的输出层重新初始化、新域标准化保留）、冻结范围、结构跟随预训练。
  8. 增益正反馈的防护：levels 标量刻度、预测最优不随起点变的不变性损失、正反馈斜率指标、
     闭环单步与累计限幅、运行目录防覆盖。
  9. 标签自带 train / val / test 划分时的分折；--fold none 不含测试集。
 10. 推理脚本 tools_suggest_six_param.py 的档位掩膜与后端换算，且死区与限幅跟着闭环走。

用法：python tests/verify_six_param_model.py [--labels data/labels_fieldii.jsonl] [--skip-scripts]
"""

import argparse
import collections
import csv
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
from bmode_dl.checkpoint import (FRONTEND_OUTPUT_KEYS, CombinedModel, load_checkpoint, model_from_config,
                                 save_checkpoint)
from bmode_dl.closed_loop import run_closed_loop
from bmode_dl.dataset import FieldIIData, InputBuilder, make_batch, read_jsonl
from bmode_dl.losses import SixParamLoss
from bmode_dl.metrics import compute_metrics, format_metrics, predict, settings_lookup_baseline
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
def write_synthetic_cache(rows, out_dir, cache_rows=128, lines=32, seed=0, console=False):
    """合成缓存：dB 随深度衰减 + 斑点，底噪常数。只用于冒烟，不代表真实数据。

    console=True 时按实机缓存的格式另写 reference_db 与底噪两端（实机标签行没有这些字段）。
    """
    rng = np.random.RandomState(seed)
    n = len(rows)
    z = np.linspace(0, 1, cache_rows)[None, :, None]
    depth = np.array([r["depth_mm"] for r in rows], np.float32)
    freq = np.array([r["frequency_mhz"] for r in rows], np.float32)
    db = (-20.0 - 0.7 * freq[:, None, None] * depth[:, None, None] / 10.0 * z * 10.0 / 2.0
          + 5.6 * rng.randn(n, cache_rows, lines)).astype(np.float32)
    floor = np.full((n, cache_rows), -90.0, np.float32)
    os.makedirs(out_dir, exist_ok=True)
    extra = {}
    if console:
        db = db + 70.0                         # 实机 dB 刻度（中位约 29 dB），与 Field II 差一个常数
        floor = floor + 100.0
        extra = {"reference_db": (30.0 + 5.0 * rng.rand(n)).astype(np.float32),
                 "noise_floor_top_db": floor[:, 0].copy(), "noise_floor_bottom_db": floor[:, -1].copy()}
    np.savez(os.path.join(out_dir, "cache.npz"), db=db, floor=floor,
             frame_id=np.array([r["frame_id"] for r in rows]),
             min_depth_mm=np.full(n, 0.5, np.float32), max_depth_mm=depth, **extra)
    with io.open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"rows": cache_rows, "lines": lines,
                                 "source_rows": 870 if console else K.FIELDII_SOURCE_ROWS,
                                 "source": "console" if console else "fieldii", "frames": n}) + "\n")


def check_backend_output_and_smoothing(data, norm, cw, inputs, tg):
    """optimum 输出方式的算术、末层零初始化、旧检查点兼容、标签平滑为 0 时与普通交叉熵一致。"""
    import torch.nn.functional as F
    from bmode_dl.losses import ladder_losses
    from bmode_dl.model import masked_logits

    model = model_from_config({"backend_output": "optimum"}, data.ladders, norm).eval()
    with torch.no_grad():
        out = model(inputs)
    current_gain = inputs["scalars"][:, 3] * 10.0
    current_tgc_db = (inputs["scalars"][:, 4:12] * 127.0) * K.TGC_DB_PER_LEVEL[0]
    check("optimum output: gain delta == optimum - current",
          bool(torch.allclose(out["gain_delta_db"], out["gain_optimal_db"] - current_gain, atol=1e-5)))
    check("optimum output: current gain recovered from scalars",
          bool(torch.allclose(current_gain, tg["gain_db"], atol=1e-4)),
          "max diff %.2e dB" % float((current_gain - tg["gain_db"]).abs().max()))
    check("optimum output: TGC delta == optimum - current",
          bool(torch.allclose(out["tgc_delta_db"], out["tgc_optimal_db"] - current_tgc_db, atol=1e-4)))
    check("optimum output: current TGC recovered from scalars",
          bool(torch.allclose(inputs["scalars"][:, 4:12] * 127.0 + 127.0, tg["tgc_levels"], atol=1e-3)))
    starts_at_mean = (bool(torch.allclose(out["gain_optimal_db"],
                                          torch.full_like(out["gain_optimal_db"], norm["opt_gain_mean"]), atol=1e-5))
                      and bool(torch.allclose(out["tgc_optimal_db"][0], torch.tensor(norm["opt_tgc_db_mean"]),
                                              atol=1e-4)))
    check("optimum output starts at the training-set mean (zero-initialised last layer)", starts_at_mean,
          "opt_gain_mean %.3f dB, opt_gain_std %.3f dB" % (norm["opt_gain_mean"], norm["opt_gain_std"]))

    legacy = model_from_config({}, data.ladders)
    check("config without backend_output rebuilds the legacy delta model",
          legacy.backend_output == "delta" and not hasattr(legacy, "gain_opt_mean"))

    logits = torch.randn(6, len(data.ladders["depth_mm"]))
    ce0, _, _ = ladder_losses(logits, tg["depth_valid"], tg["optimal_depth_idx"], tg["depth_mask"], 0.0)
    ref = F.cross_entropy(masked_logits(logits, tg["depth_valid"]), tg["optimal_depth_idx"].clamp(min=0),
                          reduction="none")
    m = tg["depth_mask"] * (tg["optimal_depth_idx"] >= 0).float()
    ref = (ref * m).sum() / (m.sum() + 1e-6)
    check("ladder loss with smoothing 0 == cross entropy", bool(torch.allclose(ce0, ref, atol=1e-5)))
    ce1, _, _ = ladder_losses(logits, tg["depth_valid"], tg["optimal_depth_idx"], tg["depth_mask"], 0.1)
    check("ladder loss with smoothing 0.1 is finite and differs", bool(torch.isfinite(ce1)) and float(ce1) != float(ce0))

    old = os.path.join(ROOT, "runs", "fieldii_v1", "fold_0", "best.pt")
    if os.path.exists(old):
        try:
            legacy_model, _, payload = load_checkpoint(old)
            check("fieldii_v1 checkpoint still loads (strict state dict)", legacy_model.backend_output == "delta",
                  "epoch %s" % payload.get("epoch"))
        except Exception as exc:
            check("fieldii_v1 checkpoint still loads (strict state dict)", False, repr(exc)[:200])


class Oscillator(torch.nn.Module):
    """假模型：深度在最浅两档之间来回要，频率与聚焦保持不变，增益永远说还差 5 dB。

    用来走通闭环的打转分支——真模型在合成缓存上不一定打转。
    """

    def __init__(self, ladders):
        super().__init__()
        self.ladders = ladders

    def _current(self, values, ladder):
        table = torch.tensor(ladder, dtype=torch.float32)
        return (values[:, None] - table[None, :]).abs().argmin(dim=1)

    def forward(self, inputs):
        s = inputs["scalars"].float()
        b = s.shape[0]
        depth = self._current(s[:, 0] * 60.0, self.ladders["depth_mm"])
        frequency = self._current(s[:, 1] * 8.0, self.ladders["frequency_mhz"])
        focus = self._current(s[:, 2] * 40.0, self.ladders["focus_mm"])
        wanted_depth = torch.where(depth == 0, torch.ones_like(depth), torch.zeros_like(depth))
        one_hot = lambda idx, n: torch.nn.functional.one_hot(idx, n).float() * 10.0
        return {
            "gain_delta_db": torch.full((b,), 5.0),
            "gain_dir": torch.zeros(b, 3),
            "tgc_delta_db": torch.zeros(b, K.NUM_TGC_BANDS),
            "slider_dir": torch.zeros(b, len(K.SLIDER_GROUPS), 3),
            "depth_logits": one_hot(wanted_depth, len(self.ladders["depth_mm"])),
            "depth_dir": torch.zeros(b, 3),
            "frequency_logits": one_hot(frequency, len(self.ladders["frequency_mhz"])),
            "frequency_dir": torch.zeros(b, 3),
            "focus_logits": one_hot(focus, len(self.ladders["focus_mm"])),
            "focus_dir": torch.zeros(b, 3),
            "dr_delta_ui": torch.zeros(b),
            "dr_dir": torch.zeros(b, 3),
            "aux": torch.zeros(b, 2),
        }


class ShiftedOptimum(torch.nn.Module):
    """假模型：修正量 = 系数 * (6 - 当前增益) + 常数，即预测最优 = 当前 + 修正。

    系数 1、常数 0 时预测最优恒为 6 dB（正确，斜率 0）；系数 0、常数 c 时预测最优 = 当前 + c
    （正反馈，斜率 1）；用来检验斜率指标、不变性损失与闭环限幅。
    """

    def __init__(self, ladders, pull=1.0, push=0.0):
        super().__init__()
        self.inner = Oscillator(ladders)
        self.pull, self.push = float(pull), float(push)

    def forward(self, inputs):
        out = self.inner(inputs)
        current = inputs["scalars"][:, 3].float() * 10.0          # fixed 刻度下还原当前增益
        out["gain_delta_db"] = self.pull * (6.0 - current) + self.push
        # 最优 TGC 恒为 0 dB（修正量 = -当前），这样 TGC 部分对两种假模型都不随起点变
        out["tgc_delta_db"] = -inputs["scalars"][:, 4:12].float() * 127.0 * K.TGC_DB_PER_LEVEL[0]
        # 前端保持不动，只测后端
        for axis, ladder in (("depth", "depth_mm"), ("frequency", "frequency_mhz"), ("focus", "focus_mm")):
            values = inputs["scalars"][:, {"depth": 0, "frequency": 1, "focus": 2}[axis]].float() \
                * {"depth": 60.0, "frequency": 8.0, "focus": 40.0}[axis]
            here = self.inner._current(values, self.inner.ladders[ladder])
            out["%s_logits" % axis] = torch.nn.functional.one_hot(
                here, len(self.inner.ladders[ladder])).float() * 10.0
        return out


def check_feedback_guards(data, builder, val_idx):
    """levels 刻度、正反馈斜率、不变性损失、闭环限幅。"""
    from bmode_dl.losses import invariance_loss
    from bmode_dl.metrics import gain_feedback_slope
    from bmode_dl.dataset import InputBuilder as IB
    norm = dict(builder.norm)
    levels = IB(data.rows_out, data.lines, data.source_rows, norm, scalar_norm="levels")
    fixed = IB(data.rows_out, data.lines, data.source_rows, norm, scalar_norm="fixed")
    idx = val_idx[:8]
    a, tg = make_batch(data, levels, idx)
    b, _ = make_batch(data, fixed, idx)
    ref = tg["optimal_gain_db"] * 0 + data.t["reference_db"][torch.as_tensor(idx)]
    check("scalar_norm=levels: gain as fixed, reference standardised with the image dB statistics",
          bool(torch.allclose(a["scalars"][:, 3], b["scalars"][:, 3]))
          and bool(torch.allclose(a["scalars"][:, 13], (ref - norm["db_mean"]) / norm["db_std"], atol=1e-5)))

    good, bad = ShiftedOptimum(data.ladders, 1.0, 0.0), ShiftedOptimum(data.ladders, 0.0, 3.0)
    slope_good, _ = gain_feedback_slope(good, data, fixed, val_idx, max_frames=32)
    slope_bad, frac_bad = gain_feedback_slope(bad, data, fixed, val_idx, max_frames=32)
    check("feedback slope: 0 for an optimum that ignores the current gain, 1 for one that tracks it",
          abs(slope_good) < 1e-3 and abs(slope_bad - 1.0) < 1e-3,
          "good %.4f, bad %.4f" % (slope_good, slope_bad))

    gen = torch.Generator().manual_seed(3)
    ia, ta = make_batch(data, fixed, idx, start="redraw", generator=gen)
    ib, tb = make_batch(data, fixed, idx, start="redraw", generator=gen)
    with torch.no_grad():
        loss_good = float(invariance_loss(good(ia), ta, good(ib), tb))
        loss_bad = float(invariance_loss(bad(ia), ta, bad(ib), tb))
    check("invariance loss is zero when the optimum ignores the start and positive when it tracks it",
          loss_good < 1e-5 and loss_bad > 0.1, "good %.2e, bad %.3f" % (loss_good, loss_bad))

    # 限幅：每步都要 +100 dB 的模型，单步不超过 40 级，累计不超过 120 级
    runaway = ShiftedOptimum(data.ladders, 0.0, 100.0)
    summary, records = run_closed_loop(runaway, data, fixed, val_idx[:8], max_steps=6, batch_size=8)
    steps_ok = all(abs(p.get("gain_clicks", 0)) <= 80 for r in records for p in r["path"])
    total_ok = all(abs(r["gain_total_clicks"]) <= 255 + 1e-6 for r in records)
    check("closed loop clamps the gain step (80 clicks) and the net change (255 clicks)",
          steps_ok and total_ok and summary["gain_clamped"] == 1.0,
          "max total %.0f clicks, clamped %.2f" % (max(abs(r["gain_total_clicks"]) for r in records),
                                                    summary["gain_clamped"]))
    _, unbounded = run_closed_loop(runaway, data, fixed, val_idx[:8], max_steps=6, batch_size=8,
                                   max_gain_step_clicks=0, max_gain_total_clicks=0)
    check("without the clamp the same model runs away",
          max(abs(r["gain_total_clicks"]) for r in unbounded) > 1000,
          "max total %.0f clicks" % max(abs(r["gain_total_clicks"]) for r in unbounded))


def check_near_metrics_and_hysteresis(data, builder, model, norm, val_idx, preds, tg):
    """近最优帧指标、期望档决策、闭环滞回与打转冻结、逐步路径。"""
    metrics = compute_metrics(preds, tg, norm)
    near_keys = ["frontend_score_near", "depth_top1_near", "frequency_hit_near", "focus_top1_near"]
    missing = [k for k in near_keys if k not in metrics or not np.isfinite(metrics[k])]
    check("near-optimum metrics computed", not missing,
          "n_near depth %s frequency %s focus %s; missing %s"
          % (metrics.get("depth_n_near"), metrics.get("frequency_n_near"), metrics.get("focus_n_near"), missing))
    # 近最优帧确实是当前设置与最优相差不超过 1 档的那些帧
    hand = {}
    for axis in ("depth", "frequency", "focus"):
        mask = (tg["%s_mask" % axis] > 0) & (tg["optimal_%s_idx" % axis] >= 0)
        near = np.abs(tg["%s_idx" % axis][mask] - tg["optimal_%s_idx" % axis][mask]) <= 1
        hand[axis] = int(near.sum())
    check("near-optimum frame counts match a hand count",
          all(metrics["%s_n_near" % a] == hand[a] for a in hand), str(hand))
    check("expected-step decision reported",
          all(k in metrics for k in ("depth_top1_expected", "frequency_hit_expected", "focus_top1_expected")))
    probs_ok = all(np.allclose(preds["%s_prob" % a].sum(axis=1), 1.0, atol=1e-4)
                   for a in ("depth", "frequency", "focus"))
    valid_ok = all(float(preds["%s_prob" % a][tg["%s_valid" % a] <= 0].max(initial=0.0)) < 1e-3
                   for a in ("depth", "frequency", "focus"))
    check("ladder probabilities sum to 1 and avoid unavailable steps", probs_ok and valid_ok)

    # 滞回作用范围：always 时余量 1.0 挡住每一次换档；revisit 时只挡回头，新设置照常换
    summary, records = run_closed_loop(model, data, builder, val_idx[:24], max_steps=3, batch_size=24,
                                       frontend_margin=1.0, margin_mode="always")
    check("hysteresis 1.0 with margin_mode=always stops every front-end change",
          all(r["frontend_moves"] == 0 for r in records) and summary["oscillated"] == 0.0,
          "backend moves %s" % sorted(set(r["backend_moves"] for r in records)))

    osc_records = run_closed_loop(Oscillator(data.ladders), data, builder, val_idx[:16], max_steps=6, batch_size=16,
                                  frontend_margin=1.0, margin_mode="revisit", freeze_on_revisit=True)[1]
    no_repeat = all(len(set(tuple(p["setting"]) for p in r["path"] if p["action"] in ("start", "frontend")))
                    == 1 + r["frontend_moves"] for r in osc_records)
    check("margin_mode=revisit lets new settings through but blocks the way back",
          no_repeat and all(r["frontend_moves"] > 0 for r in osc_records)
          and all(r["backend_moves"] > 0 for r in osc_records),
          "front-end moves %s, back-end moves %s, oscillated %.2f"
          % (sorted(set(r["frontend_moves"] for r in osc_records)),
             sorted(set(r["backend_moves"] for r in osc_records)),
             float(np.mean([r["oscillated"] for r in osc_records]))))

    always_records = run_closed_loop(Oscillator(data.ladders), data, builder, val_idx[:16], max_steps=6,
                                     batch_size=16, frontend_margin=1.0, margin_mode="always")[1]
    check("margin_mode=always keeps the same model from moving at all",
          all(r["frontend_moves"] == 0 for r in always_records))

    summary, records = run_closed_loop(model, data, builder, val_idx[:24], max_steps=6, batch_size=24,
                                       frontend_margin=0.0, freeze_on_revisit=True)
    path_ok = all(r["path"][0]["action"] == "start" and len(r["path"]) >= 1 for r in records)
    actions = collections.Counter(p["action"] for r in records for p in r["path"])
    check("every trajectory records a step-by-step path", path_ok, str(dict(actions)))
    # 一次前端改动可以同时动两三轴，所以各轴改档次数之和 >= 前端改动次数，每一轴 <= 前端改动次数
    consistent = [(sum(r["axis_changes"].values()) >= r["frontend_moves"])
                  and all(v <= r["frontend_moves"] for v in r["axis_changes"].values())
                  and (r["frontend_moves"] > 0 or sum(r["axis_changes"].values()) == 0)
                  for r in records]
    check("axis_changes is consistent with frontend_moves", all(consistent),
          "axes changed per move %s" % sorted(set(sum(r["axis_changes"].values()) for r in records)))
    frozen = [r for r in records if r["frozen"]]
    after_freeze_ok = True
    for r in frozen:
        seen_freeze = False
        for step in r["path"]:
            if step["action"] == "freeze":
                seen_freeze = True
            elif seen_freeze and step["action"] == "frontend":
                after_freeze_ok = False
    check("no front-end change after the freeze", after_freeze_ok, "frozen %d / %d trajectories" % (len(frozen), len(records)))
    check("closed-loop summary reports the new fields",
          all(k in summary for k in ("frozen_frontend", "final_frontend_all_within1", "final_depth_mean_signed_steps")),
          format_metrics(summary, ("converged", "oscillated", "frozen_frontend", "final_frontend_all_within1")))

    summary_expected, _ = run_closed_loop(model, data, builder, val_idx[:24], max_steps=3, batch_size=24,
                                          decision="expected")
    check("closed loop runs with the expected-step decision", summary_expected["trajectories"] == 24)

    # 真正走一遍打转分支：这个假模型在最浅两档深度之间来回要，并且一直说增益偏暗
    summary, records = run_closed_loop(Oscillator(data.ladders), data, builder, val_idx[:16], max_steps=6,
                                       batch_size=16, frontend_margin=0.1, freeze_on_revisit=True)
    frozen = [r for r in records if r["frozen"]]
    after_freeze = []
    for r in records:
        seen = False
        for step in r["path"]:
            seen = seen or step["action"] == "freeze"
            if seen and step["action"] == "frontend":
                after_freeze.append(r)
                break
    check("an oscillating model is detected and frozen, then only the back end moves",
          len(frozen) == len(records) and summary["oscillated"] == 1.0 and not after_freeze
          and all(r["backend_moves"] > 0 for r in records),
          "frozen %d/%d, oscillated %.2f, backend moves %s"
          % (len(frozen), len(records), summary["oscillated"],
             sorted(set(r["backend_moves"] for r in records))))
    check("oscillating axis is reported", summary.get("oscillating_depth_changes", 0) > 0,
          "depth %.2f frequency %.2f focus %.2f changes per oscillating trajectory"
          % (summary.get("oscillating_depth_changes", float("nan")),
             summary.get("oscillating_frequency_changes", float("nan")),
             summary.get("oscillating_focus_changes", float("nan"))))

    without = run_closed_loop(Oscillator(data.ladders), data, builder, val_idx[:16], max_steps=6, batch_size=16,
                              frontend_margin=0.1, freeze_on_revisit=False)[0]
    # 不冻结时这个假模型每一步都在换前端，一次后端修正也做不了，正是 fieldii_v2 里打转轨迹的样子。
    # 冻结之后后端反而做了 5 步 +5 dB：这是假模型永远要更多增益的结果，真模型的修正量会收敛。
    check("without the freeze the same model never reaches a back-end step",
          without["frozen_frontend"] == 0.0 and without["converged"] == 0.0,
          "backend steps: frozen %s, not frozen %s"
          % (sorted(set(r["backend_moves"] for r in records)), without["frontend_moves_hist"]))


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
        lines, coverage = data.physics_coverage(train_idx, val_idx)
        keys_ok = set(coverage) == {"attenuation_db_cm_mhz", "electronic_noise_db", "sound_speed_mps"}
        att = coverage.get("attenuation_db_cm_mhz", {})
        hand = [r["attenuation_db_cm_mhz"] for i, r in enumerate(data.rows) if i in set(val_idx.tolist())]
        flagged = att.get("extrapolates_below") == (min(hand) < att.get("train_min", 0))
        check("physics coverage per fold", keys_ok and flagged and len(lines) == 3,
              " | ".join(line.strip() for line in lines))

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

        new_config = {"input_mode": "full", "backend_output": "optimum", "frontend_dropout": 0.3}
        for mode in ("full", "no_image", "params_only"):
            model = model_from_config(dict(new_config, input_mode=mode), data.ladders, norm)
            criterion = SixParamLoss(cw, norm, uncertainty_weighting=(mode == "full"), frontend_label_smoothing=0.1)
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

        check_backend_output_and_smoothing(data, norm, cw, inputs, tg)

        model = model_from_config(new_config, data.ladders, norm)
        with torch.no_grad():
            for head in (model.gain_head, model.tgc_band_head):   # 让保存的权重非平凡
                head.net[-1].weight.normal_(0, 0.05)
        ckpt = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(ckpt, model, new_config, data.ladders, norm, cw,
                        {"rows": data.rows_out, "lines": data.lines, "source_rows": data.source_rows},
                        {"val_groups": val_groups})
        model2, builder2, payload = load_checkpoint(ckpt)
        model.eval()
        with torch.no_grad():
            same = (torch.allclose(model(inputs)["gain_delta_db"], model2(inputs)["gain_delta_db"], atol=1e-5)
                    and torch.allclose(model(inputs)["tgc_delta_db"], model2(inputs)["tgc_delta_db"], atol=1e-5))
        check("checkpoint round trip (optimum output, normalisation buffers restored)", bool(same))

        other = model_from_config(dict(new_config, backend_output="delta"), data.ladders, norm).eval()
        combined = CombinedModel(other, model2).eval()
        with torch.no_grad():
            co, fo, bo = combined(inputs), other(inputs), model2(inputs)
        ok = (all(torch.equal(co[k], fo[k]) for k in FRONTEND_OUTPUT_KEYS)
              and torch.equal(co["gain_delta_db"], bo["gain_delta_db"])
              and torch.equal(co["tgc_delta_db"], bo["tgc_delta_db"]))
        check("combined model: front-end outputs from one network, back-end from the other", bool(ok))

        preds, ptg = predict(model2, data, builder2, val_idx, start="label", batch_size=32)
        metrics = compute_metrics(preds, ptg, norm)
        check("metrics computed", np.isfinite(metrics["score"]), "score %.3f (untrained)" % metrics["score"])
        bp, btg = settings_lookup_baseline(data, train_idx, val_idx)
        bm = compute_metrics(bp, btg)
        check("settings baseline computed", np.isfinite(bm["score"]), "score %.3f" % bm["score"])
        summary, records = run_closed_loop(model2, data, builder2, val_idx[:40], max_steps=3, batch_size=32)
        check("closed loop runs", summary["trajectories"] == 40, json.dumps(summary)[:160])
        check_near_metrics_and_hysteresis(data, builder2, model2, norm, val_idx, preds, ptg)
        check_feedback_guards(data, builder2, val_idx)

        if not skip_scripts:
            import tools_train_six_param as TT
            import tools_evaluate_six_param as TE
            runs = os.path.join(tmp, "runs")
            started = time.time()
            TT.main(["--cache", cache_dir, "--labels", label_path, "--runs", runs, "--name", "smoke",
                     "--folds", "3", "--fold", "0", "--epochs", "1", "--batch", "16", "--eval-every", "1",
                     "--device", "cpu", "--d-model", "64", "--transformer-layers", "1"])
            fold_dir = os.path.join(runs, "smoke", "fold_0")
            written = [f for f in ("best.pt", "best_backend.pt", "best_frontend.pt", "last.pt")
                       if os.path.exists(os.path.join(fold_dir, f))]
            check("training script wrote best/best_backend/best_frontend/last", len(written) == 4,
                  "%s, %.0f s" % (written, time.time() - started))
            try:
                TT.main(["--cache", cache_dir, "--labels", label_path, "--runs", runs, "--name", "smoke",
                         "--epochs", "1", "--device", "cpu"])
                refused = False
            except SystemExit:
                refused = True
            check("reusing a run name without --overwrite is refused", refused)
            with io.open(os.path.join(fold_dir, "metrics.csv"), encoding="utf-8") as handle:
                header = next(csv.reader(handle))
            wanted = ["val_aux_attenuation_mae", "val_gain_dir_f1_head", "val_slider_near_dir_f1_derived",
                      "val_tgc_band_mae_db_0", "val_backend_score", "val_frontend_score", "valredraw_tgc_mae_db",
                      "val_frontend_score_near", "val_focus_top1_near", "val_depth_mean_signed_steps",
                      "val_focus_top1_expected", "val_gain_feedback_slope", "train_invariance"]
            missing = [w for w in wanted if w not in header]
            check("metrics.csv keeps every validation metric", not missing,
                  "%d columns, missing %s" % (len(header), missing))
            TE.main(["--run", os.path.join(runs, "smoke"), "--cache", cache_dir, "--labels", label_path,
                     "--device", "cpu", "--redraw-seeds", "1", "--max-steps", "2", "--frontend-margin", "0.1"])
            tag = "combined_m010_rev_frz_c80-255_s2"   # 限幅与 --max-steps 2 也进文件名
            check("evaluation script (combined model) wrote reports named after the loop settings",
                  os.path.exists(os.path.join(fold_dir, "evaluation_report_%s.txt" % tag))
                  and os.path.exists(os.path.join(runs, "smoke", "evaluation_summary_%s.txt" % tag)),
                  tag)
            TE.main(["--run", os.path.join(runs, "smoke"), "--cache", cache_dir, "--labels", label_path,
                     "--device", "cpu", "--redraw-seeds", "1", "--max-steps", "2", "--frontend-margin", "0",
                     "--no-freeze-on-revisit", "--no-closed-loop"])
            check("a second run with different settings does not overwrite the first",
                  os.path.exists(os.path.join(fold_dir, "evaluation_report_%s.txt" % tag))
                  and os.path.exists(os.path.join(fold_dir, "evaluation_report_combined_noloop.txt")),
                  str(sorted(f for f in os.listdir(fold_dir) if f.startswith("evaluation_report"))))
            payload = torch.load(os.path.join(fold_dir, "best_frontend.pt"), map_location="cpu", weights_only=False)
            check("best_frontend.pt is selected on the near-optimum score",
                  payload.get("selected_by") == "frontend_score_near", str(payload.get("selected_by")))
            with io.open(os.path.join(fold_dir, "closed_loop_trajectories_%s.jsonl" % tag), encoding="utf-8") as h:
                first = json.loads(h.readline())
            check("closed-loop trajectories carry the path and the steps to the optimum",
                  "path" in first and "frontend_steps_to_optimum" in first and "axis_changes" in first,
                  json.dumps({k: first[k] for k in ("frozen", "axis_changes", "frontend_steps_to_optimum")}))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    emit("")


def check_split_field(rows):
    """标签带 train / val / test 划分时：分折用 split 字段，--fold none 训练训练 + 验证、不含测试。"""
    emit("=========== 9. split field (train / val / test) ===========")
    splits = {}
    for r in rows:
        splits.setdefault(r.get("split"), []).append(r["group_id"])
    if not all(s in splits for s in ("train", "val", "test")):
        emit("  skipped: labels have no train/val/test split")
        emit("")
        return
    picked = {s: sorted(set(splits[s]))[0] for s in ("train", "val", "test")}
    subset = [r for r in rows if r["group_id"] in picked.values()]
    tmp = tempfile.mkdtemp(prefix="six_param_split_")
    try:
        cache_dir = os.path.join(tmp, "cache")
        label_path = os.path.join(tmp, "labels.jsonl")
        with io.open(label_path, "w", encoding="utf-8") as handle:
            for r in subset:
                handle.write(json.dumps(r, ensure_ascii=False) + "\n")
        write_synthetic_cache(subset, cache_dir)
        data = FieldIIData(cache_dir, label_path, device="cpu", log=lambda text: None)
        train_idx, val_idx, _ = data.split_indices(4, 0)
        check("with a split field, fold 0 trains on split=train and validates on split=val",
              {data.splits[i] for i in train_idx} == {"train"} and {data.splits[i] for i in val_idx} == {"val"})
        all_idx, none_val, _ = data.split_indices(4, None)
        check("--fold none trains on train + val and never on test",
              {data.splits[i] for i in all_idx} == {"train", "val"} and len(none_val) == 0,
              "%d frames, splits %s" % (len(all_idx), sorted({data.splits[i] for i in all_idx})))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    emit("")


def check_finetune(fieldii_rows, console_path, skip_scripts):
    """实机微调的各个环节，用合成缓存（实机标签行 + 合成图像）走一遍。"""
    from bmode_dl.checkpoint import DOMAIN_BUFFERS, HEAD_MODULES, freeze, load_pretrained
    from bmode_dl.dataset import group_key
    emit("=========== 7. console fine-tuning (synthetic images, real console labels) ===========")
    if not os.path.exists(console_path):
        emit("  skipped: %s not found" % console_path)
        emit("")
        return
    console_rows = read_jsonl(console_path)

    # 分组：同一次摆放的 _GEN / _THI 目录合并，Field II 行不受影响
    e8 = {group_key(r, "placement") for r in console_rows if r.get("family_id", "").startswith("20260911_E8")}
    kept = group_key({"source": "console", "group_id": "20260903/0", "family_id": "20260903/0"}, "placement")
    fieldii = group_key(fieldii_rows[0], "placement")
    check("placement grouping merges E8_GEN and E8_THI, keeps other families, leaves Field II alone",
          e8 == {"20260911_E8/0"} and kept == "20260903/0" and fieldii == fieldii_rows[0]["group_id"],
          "E8 -> %s, Field II -> %s" % (sorted(e8), fieldii))
    placements = {group_key(r, "placement") for r in console_rows}
    families = {r.get("family_id") for r in console_rows}
    emit("  console: %d frames, %d families, %d placements" % (len(console_rows), len(families), len(placements)))

    tmp = tempfile.mkdtemp(prefix="six_param_ft_")
    try:
        cache_dir = os.path.join(tmp, "console_cache")
        write_synthetic_cache(console_rows, cache_dir, console=True)
        data = FieldIIData(cache_dir, console_path, device="cpu", group_by="placement", log=lambda text: None)
        arrays = np.load(os.path.join(cache_dir, "cache.npz"))
        order = [list(arrays["frame_id"]).index(f) for f in data.frame_ids]
        check("cache overrides reference_db and noise floor on console rows",
              np.allclose(data.encoded["reference_db"], arrays["reference_db"][order])
              and np.allclose(data.encoded["noise_floor_top_db"], arrays["noise_floor_top_db"][order]))
        leaks = []
        for fold in range(4):
            train_idx, val_idx, _ = data.split_indices(4, fold)
            shared = set(data.group_ids[i] for i in train_idx) & set(data.group_ids[i] for i in val_idx)
            leaks.extend(shared)
        check("no placement is in both training and validation", not leaks, str(sorted(set(leaks))))

        train_idx, val_idx, _ = data.split_indices(4, 0)
        norm = data.normalisation(train_idx)
        builder = InputBuilder(data.rows_out, data.lines, data.source_rows, norm, scalar_norm="data")
        inputs, tg = make_batch(data, builder, train_idx[:8])
        tg_ref = data.t["reference_db"][torch.as_tensor(train_idx[:8])]
        ref_expected = tg_ref - norm["db_mean"]
        check("scalar_norm=data standardises reference with the image dB statistics",
              bool(torch.allclose(inputs["scalars"][:, 13], ref_expected / norm["db_std"], atol=1e-5)))
        fixed = InputBuilder(data.rows_out, data.lines, data.source_rows, norm, scalar_norm="fixed")
        fixed_inputs, _ = make_batch(data, fixed, train_idx[:8])
        check("scalar_norm=fixed keeps the Field II constants (legacy checkpoints)",
              bool(torch.allclose(fixed_inputs["scalars"][:, 13], (tg_ref + 45.0) / 30.0, atol=1e-5)))

        # 预训练（Field II 档位）-> 实机档位：只有前端三个头的最后一层重新初始化，新域标准化保留
        fieldii_ladders = L.collect_ladders(fieldii_rows)
        pretrained = model_from_config({"backend_output": "optimum"}, fieldii_ladders,
                                       {"opt_gain_mean": 7.0, "opt_gain_std": 0.7,
                                        "opt_tgc_db_mean": [0.0] * 8, "opt_tgc_db_std": [1.0] * 8})
        with torch.no_grad():
            for parameter in pretrained.parameters():
                parameter.add_(0.01)
        ckpt = os.path.join(tmp, "pretrained.pt")
        save_checkpoint(ckpt, pretrained, {"backend_output": "optimum", "d_model": 256}, fieldii_ladders,
                        {"db_mean": -41.0, "db_std": 19.0}, {"gain_dir": [1, 1, 1]},
                        {"rows": 128, "lines": 32, "source_rows": K.FIELDII_SOURCE_ROWS})
        target = model_from_config({"backend_output": "optimum"}, data.ladders, norm)
        loaded, shape_skipped, buffer_skipped, missing, _ = load_pretrained(target, ckpt, log=lambda text: None)
        expected_skip = {"%s_head.net.3.%s" % (a, w) for a in ("depth", "frequency", "focus")
                         for w in ("weight", "bias")}
        check("only the three front-end output layers are re-initialised",
              set(shape_skipped) == expected_skip and not missing,
              "%d loaded, skipped %s" % (loaded, sorted(shape_skipped)))
        check("new-domain normalisation buffers are kept",
              set(buffer_skipped) == set(DOMAIN_BUFFERS)
              and abs(float(target.gain_opt_mean) - norm["opt_gain_mean"]) < 1e-5,
              "gain_opt_mean %.3f (console) vs 7.0 (pretrained)" % float(target.gain_opt_mean))
        source_state = pretrained.state_dict()
        copied = all(torch.equal(target.state_dict()[k], v) for k, v in source_state.items()
                     if k not in expected_skip and k not in DOMAIN_BUFFERS)
        check("every other tensor equals the pretrained one", copied)

        _, trainable = freeze(target, "encoders")
        encoder_frozen = all(not p.requires_grad for n, p in target.named_parameters()
                             if n.split(".", 1)[0] in ("image_encoder", "profile_encoder"))
        rest_trainable = all(p.requires_grad for n, p in target.named_parameters()
                             if n.split(".", 1)[0] not in ("image_encoder", "profile_encoder"))
        check("freeze=encoders freezes only the image and profile encoders", encoder_frozen and rest_trainable,
              "%.2f M trainable" % (trainable / 1e6))
        freeze(target, "backbone")
        heads_only = all(p.requires_grad == (n.split(".", 1)[0] in HEAD_MODULES)
                         for n, p in target.named_parameters())
        check("freeze=backbone leaves only the output heads trainable", heads_only)
        freeze(target, "none")
        check("freeze=none trains everything", all(p.requires_grad for p in target.parameters()))

        if not skip_scripts:
            import tools_train_six_param as TT
            import tools_evaluate_six_param as TE
            small = model_from_config({"backend_output": "optimum", "d_model": 64, "transformer_layers": 1},
                                      fieldii_ladders, {"opt_gain_mean": 7.0, "opt_gain_std": 0.7,
                                                        "opt_tgc_db_mean": [0.0] * 8, "opt_tgc_db_std": [1.0] * 8})
            small_ckpt = os.path.join(tmp, "pretrained_small.pt")
            save_checkpoint(small_ckpt, small, {"backend_output": "optimum", "d_model": 64, "transformer_layers": 1},
                            fieldii_ladders, {"db_mean": -41.0, "db_std": 19.0}, {"gain_dir": [1, 1, 1]},
                            {"rows": 128, "lines": 32, "source_rows": K.FIELDII_SOURCE_ROWS})
            runs = os.path.join(tmp, "runs")
            TT.main(["--cache", cache_dir, "--labels", console_path, "--runs", runs, "--name", "ft",
                     "--init-checkpoint", small_ckpt, "--freeze", "encoders", "--group-by", "placement",
                     "--scalar-norm", "data", "--fold", "0", "--epochs", "1", "--batch", "16", "--eval-every", "1",
                     "--device", "cpu"])
            fold_dir = os.path.join(runs, "ft", "fold_0")
            payload = torch.load(os.path.join(fold_dir, "best_backend.pt"), map_location="cpu", weights_only=False)
            check("fine-tuning inherits the pretrained architecture and records the settings",
                  payload["config"]["d_model"] == 64 and payload["config"]["transformer_layers"] == 1
                  and payload["config"]["scalar_norm"] == "data" and payload["config"]["group_by"] == "placement"
                  and payload["ladders"] == data.ladders,
                  "d_model %s, layers %s, scalar_norm %s, group_by %s"
                  % (payload["config"]["d_model"], payload["config"]["transformer_layers"],
                     payload["config"]["scalar_norm"], payload["config"]["group_by"]))
            TE.main(["--run", os.path.join(runs, "ft"), "--cache", cache_dir, "--labels", console_path,
                     "--device", "cpu", "--redraw-seeds", "1"])
            report_path = [f for f in os.listdir(fold_dir) if f.startswith("evaluation_report_combined")]
            text = io.open(os.path.join(fold_dir, report_path[0]), encoding="utf-8").read() if report_path else ""
            check("evaluation on console data runs and skips the closed loop",
                  "closed loop skipped" in text and "settings " in text, str(report_path))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    emit("")


def check_suggestion():
    """推理脚本的档位掩膜与后端换算。不需要采集数据与检查点，纯算术。

    与闭环重复的两件事（死区、单步限幅）在这里对齐：闭环的默认值一改，这个检查就失败。
    """
    emit("=========== 10. suggestion script (one capture -> six suggestions) ===========")
    import inspect

    import tools_suggest_six_param as SG

    ladders = {"depth_mm": [25.1, 33.5, 41.9, 50.2, 58.6, 67.0, 75.4],
               "frequency_mhz": [4.4, 4.7, 5.0, 5.3, 5.7, 6.7, 8.0, 10.0, 11.4],
               "focus_mm": [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]}

    # 频率按成像模式限制：谐波只在 4.4-5.7 里挑，基波的 6.7-11.4 不可选
    mask, idx, exact = SG.valid_mask(ladders["frequency_mhz"], [4.4, 4.7, 5.0, 5.3, 5.7], 5.0)
    check("the frequency mask keeps only this imaging mode's steps",
          list(mask) == [1, 1, 1, 1, 1, 0, 0, 0, 0] and idx == 2 and exact, str(list(mask)))
    check("steps are counted over the selectable steps only",
          SG.steps_between(mask, 2, 0) == -2 and SG.steps_between(mask, 2, 4) == 2)
    # 主机原始显示深度（41.871418）落在档位之间：取最近的一档，并标出当前值不在表上
    off, off_idx, off_exact = SG.valid_mask(ladders["depth_mm"], None, 41.871418)
    check("an off-ladder current value snaps to the nearest step and is flagged",
          off_idx == 2 and not off_exact and off.sum() == len(ladders["depth_mm"]))
    # 当前档永远可选，哪怕不在允许集合里（否则网络只能被迫换档）
    forced, forced_idx, _ = SG.valid_mask(ladders["focus_mm"], [5.0, 10.0], 25.0)
    check("the current step is always selectable", forced[forced_idx] == 1 and forced_idx == 4)

    sig = inspect.signature(run_closed_loop).parameters
    args = SG.parse_args(["--capture", "unused"])
    check("the deadband and the single-step clamp follow the closed loop",
          args.stop_deadband_levels == sig["stop_deadband_levels"].default
          and args.max_gain_step_clicks == sig["max_gain_step_clicks"].default,
          "deadband %.2f clicks, clamp %d clicks" % (args.stop_deadband_levels, args.max_gain_step_clicks))

    mode = K.MODE_HARMONIC
    gain_slope, tgc_slope = K.GAIN_DB_PER_LEVEL[mode], K.TGC_DB_PER_LEVEL[mode]
    frame = {"frame_id": "synthetic", "mode": mode, "mode_name": "harmonic", "depth_mm": 41.9,
             "frequency_mhz": 5.0, "focus_mm": 15.0, "gain_level": 120, "gain_db": 9.2,
             "tgc_levels": np.full(K.NUM_TGC_BANDS, 127.0), "dr_ui": 67.0,
             "_depth_idx": 2, "_frequency_idx": 2, "_focus_idx": 2,
             "_depth_exact": True, "_frequency_exact": True, "_focus_exact": True}
    masks = {"depth": torch.ones(1, len(ladders["depth_mm"])),
             "frequency": torch.tensor(mask)[None, :],
             "focus": torch.ones(1, len(ladders["focus_mm"]))}

    def run(gain_clicks, tgc_levels_delta, wanted=(2, 2, 2), gain_level=120):
        dec = {"gain_delta_db": np.array([gain_clicks * gain_slope], np.float32),
               "tgc_delta_db": np.full((1, K.NUM_TGC_BANDS), tgc_levels_delta * tgc_slope, np.float32)}
        for a, axis in enumerate(("depth", "frequency", "focus")):
            n = masks[axis].shape[1]
            prob = np.full((1, n), 0.1, np.float32)
            prob[0, wanted[a]] = 0.6
            dec["%s_idx" % axis] = np.array([wanted[a]], np.int64)
            dec["%s_idx_expected" % axis] = np.array([wanted[a]], np.int64)
            dec["%s_prob" % axis] = prob
        return SG.one_suggestion(dict(frame, gain_level=gain_level), 0, dec, masks, ladders, args)

    inside = run(0.4, 1.0)
    check("a correction inside the deadband reads as 'keep'",
          inside["backend"]["gain"]["clicks"] == 0
          and inside["backend"]["tgc"]["within_deadband"]
          and inside["backend"]["tgc"]["suggested_levels"] == [127] * K.NUM_TGC_BANDS
          and inside["settled"],
          "deadband is %.1f gain clicks = %.1f slider levels"
          % (args.stop_deadband_levels, args.stop_deadband_levels * gain_slope / tgc_slope))
    moved = run(-3.4, 10.0)
    check("gain rounds to whole clicks and TGC to whole levels",
          moved["backend"]["gain"]["clicks"] == -3
          and moved["backend"]["gain"]["suggested_level"] == 117
          and moved["backend"]["tgc"]["suggested_levels"] == [137] * K.NUM_TGC_BANDS
          and not moved["settled"],
          "gain %d clicks, TGC %s" % (moved["backend"]["gain"]["clicks"],
                                      moved["backend"]["tgc"]["suggested_levels"][0]))
    clamped = run(200.0, 0.0, gain_level=200)
    check("a huge gain correction is clamped, then clipped at the console's range",
          clamped["backend"]["gain"]["clicks"] == args.max_gain_step_clicks
          and clamped["backend"]["gain"]["clamped"] and clamped["backend"]["gain"]["level_clipped"]
          and clamped["backend"]["gain"]["suggested_level"] == SG.GAIN_MAX_LEVEL,
          "level 200 %+d -> %d" % (clamped["backend"]["gain"]["clicks"],
                                   clamped["backend"]["gain"]["suggested_level"]))
    changed = run(0.0, 0.0, wanted=(3, 0, 3))
    steps = {a: changed["frontend"][a]["steps"] for a in ("depth", "frequency", "focus")}
    words = {a: changed["frontend"][a]["direction"] for a in ("depth", "frequency", "focus")}
    check("front-end changes are reported as signed steps in the console's own words",
          steps == {"depth": 1, "frequency": -2, "focus": 1}
          and words == {"depth": "deeper", "frequency": "lower", "focus": "deeper"}
          and changed["frontend_changed"] and not changed["settled"], str(steps))
    emit("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=os.path.join(ROOT, "data", "labels_fieldii.jsonl"))
    parser.add_argument("--console-labels", default=os.path.join(ROOT, "data", "labels_console.jsonl"))
    parser.add_argument("--skip-scripts", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    check_constants()
    check_render()
    rows = check_labels(args.labels)
    emit("")
    check_smoke(rows, args.skip_scripts)
    check_finetune(rows, args.console_labels, args.skip_scripts)
    check_split_field(rows)
    check_suggestion()
    emit("=========== result ===========")
    emit("  %d failure(s)%s" % (len(FAILURES), (": " + ", ".join(FAILURES)) if FAILURES else ""))
    captured = list(LINES)
    with io.open(REPORT_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(captured) + "\n")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
