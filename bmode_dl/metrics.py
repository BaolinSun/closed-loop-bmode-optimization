# -*- coding: utf-8 -*-
"""评估：网络预测解码、单步指标、只看设置的查表基线。

方向有两种读法，都报告：
    head     方向头的 argmax
    derived  由数值输出推出：增益/TGC 的预测 delta 与该行死区比较；前端预测最优档与当前档比较。
             闭环里实际执行的是 derived，所以它是主要指标。

综合分数 score = 平均(增益方向 F1[derived], 三组滑块方向 F1[derived], 深度最优档准确率,
                     频率可接受集合命中率, 聚焦最优档准确率)。动态范围无标签，不计入。
"""

from collections import defaultdict

import numpy as np
import torch

from . import constants as K
from .dataset import make_batch
from .model import masked_logits


# ---------------------------------------------------------------------------------------
#   基本指标

def accuracy(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float((y == p).mean()) if y.size else float("nan")


def macro_f1(y, p, num_classes=3):
    """只对真值里出现过的类别求平均。"""
    y, p = np.asarray(y), np.asarray(p)
    if y.size == 0:
        return float("nan")
    scores = []
    for c in range(num_classes):
        if not (y == c).any():
            continue
        tp = float(((y == c) & (p == c)).sum())
        fp = float(((y != c) & (p == c)).sum())
        fn = float(((y == c) & (p != c)).sum())
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores)) if scores else float("nan")


def confusion(y, p, num_classes=3):
    m = np.zeros((num_classes, num_classes), np.int64)
    for a, b in zip(np.asarray(y), np.asarray(p)):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            m[a, b] += 1
    return m


def direction_from_delta_np(delta, deadband):
    delta, deadband = np.asarray(delta, np.float64), np.asarray(deadband, np.float64)
    return np.where(np.abs(delta) <= deadband, 1, np.where(delta > 0, 0, 2))


def direction_from_index_np(pred_idx, current_idx):
    return np.where(pred_idx > current_idx, 0, np.where(pred_idx == current_idx, 1, 2))


# ---------------------------------------------------------------------------------------
#   网络预测

@torch.no_grad()
def predict(model, data, builder, idx, start="label", seed=None, batch_size=64, amp=False, flip=False):
    """在 idx 上跑网络，返回 (preds, targets) 两个 numpy 字典。"""
    model.eval()
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    preds, targets = defaultdict(list), defaultdict(list)
    for s in range(0, len(idx), batch_size):
        batch_idx = idx[s:s + batch_size]
        inputs, tg = make_batch(data, builder, batch_idx, start=start, generator=generator, flip=flip)
        with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=bool(amp)):
            out = model(inputs)
        decoded = decode(out, tg)
        for k, v in decoded.items():
            preds[k].append(v.detach().float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy())
        for k, v in tg.items():
            targets[k].append(v.detach().cpu().numpy())
    return ({k: np.concatenate(v) for k, v in preds.items()},
            {k: np.concatenate(v) for k, v in targets.items()})


def decode(out, tg):
    """网络输出 -> 可执行的建议。tg 只用到可选档位掩膜（推理时按设备的档位表给出）。"""
    return {
        "gain_delta_db": out["gain_delta_db"].float(),
        "gain_dir_head": out["gain_dir"].argmax(dim=1),
        "tgc_delta_db": out["tgc_delta_db"].float(),
        "slider_dir_head": out["slider_dir"].argmax(dim=2),
        "depth_idx": masked_logits(out["depth_logits"], tg["depth_valid"]).argmax(dim=1),
        "depth_dir_head": out["depth_dir"].argmax(dim=1),
        "frequency_idx": masked_logits(out["frequency_logits"], tg["frequency_valid"]).argmax(dim=1),
        "frequency_dir_head": out["frequency_dir"].argmax(dim=1),
        "focus_idx": masked_logits(out["focus_logits"], tg["focus_valid"]).argmax(dim=1),
        "focus_dir_head": out["focus_dir"].argmax(dim=1),
        "dr_delta_ui": out["dr_delta_ui"].float(),
        "aux": out["aux"].float(),
    }


# ---------------------------------------------------------------------------------------
#   只看设置的查表基线

def settings_lookup_baseline(data, train_idx, eval_idx, start="label", seed=None):
    """同一 (深度, 频率, 聚焦) 设置下训练集的众数最优档 / 中位最优增益与滑块。

    对应 docs/fieldii_labels_20260915.md §4.1 "只用设置猜方向" 的检查；网络要比它好才说明看了图像。
    """
    train_idx = np.asarray(train_idx)
    enc = data.encoded
    table = defaultdict(list)
    for i in train_idx:
        table[(int(enc["depth_idx"][i]), int(enc["frequency_idx"][i]), int(enc["focus_idx"][i]))].append(i)

    def mode_of(values, default):
        values = [v for v in values if v >= 0]
        if not values:
            return default
        return int(np.bincount(values).argmax())

    all_gain = np.median(enc["optimal_gain_db"][train_idx])
    all_tgc = np.median(enc["optimal_tgc_levels"][train_idx], axis=0)
    _, targets = _targets_only(data, eval_idx, start, seed)
    n = len(eval_idx)
    preds = {"gain_delta_db": np.zeros(n, np.float32), "tgc_delta_db": np.zeros((n, K.NUM_TGC_BANDS), np.float32),
             "depth_idx": np.zeros(n, np.int64), "frequency_idx": np.zeros(n, np.int64),
             "focus_idx": np.zeros(n, np.int64)}
    for j, i in enumerate(eval_idx):
        key = (int(enc["depth_idx"][i]), int(enc["frequency_idx"][i]), int(enc["focus_idx"][i]))
        members = table.get(key, [])
        opt_gain = np.median(enc["optimal_gain_db"][members]) if members else all_gain
        opt_tgc = np.median(enc["optimal_tgc_levels"][members], axis=0) if members else all_tgc
        tgc_slope = K.TGC_DB_PER_LEVEL[int(enc["mode"][i])]
        preds["gain_delta_db"][j] = opt_gain - targets["gain_db"][j]
        preds["tgc_delta_db"][j] = (opt_tgc - targets["tgc_levels"][j]) * tgc_slope
        preds["depth_idx"][j] = mode_of([int(enc["optimal_depth_idx"][m]) for m in members], key[0])
        preds["frequency_idx"][j] = mode_of([int(enc["optimal_frequency_idx"][m]) for m in members], key[1])
        preds["focus_idx"][j] = mode_of([int(enc["optimal_focus_idx"][m]) for m in members], key[2])
    return preds, targets


@torch.no_grad()
def _targets_only(data, idx, start, seed, batch_size=64):
    """不跑网络，只取（可能重抽了起点的）标签。与 predict 用同样的种子得到同样的起点。"""
    from .dataset import InputBuilder
    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    targets = defaultdict(list)
    builder = InputBuilder(data.rows_out, data.lines, data.source_rows,
                           {"db_mean": 0.0, "db_std": 1.0}).to(data.device)
    for s in range(0, len(idx), batch_size):
        _, tg = make_batch(data, builder, idx[s:s + batch_size], start=start, generator=generator)
        for k, v in tg.items():
            targets[k].append(v.cpu().numpy())
    return None, {k: np.concatenate(v) for k, v in targets.items()}


# ---------------------------------------------------------------------------------------
#   指标汇总

def compute_metrics(preds, tg, norm=None):
    m = {}
    backend = tg["backend_mask"] > 0
    gain_slope = np.where(tg["mode"] > 0.5, K.GAIN_DB_PER_LEVEL[1], K.GAIN_DB_PER_LEVEL[0])
    tgc_slope = np.where(tg["mode"] > 0.5, K.TGC_DB_PER_LEVEL[1], K.TGC_DB_PER_LEVEL[0])

    # 增益
    if backend.any():
        err = np.abs(preds["gain_delta_db"] - tg["delta_gain_db"])[backend]
        m["gain_mae_db"] = float(err.mean())
        m["gain_within_deadband"] = float((err <= tg["gain_deadband_db"][backend]).mean())
        derived = direction_from_delta_np(preds["gain_delta_db"] / gain_slope, tg["deadband_gain_levels"])
        m["gain_dir_acc_derived"] = accuracy(tg["gain_dir"][backend], derived[backend])
        m["gain_dir_f1_derived"] = macro_f1(tg["gain_dir"][backend], derived[backend])
        if "gain_dir_head" in preds:
            m["gain_dir_f1_head"] = macro_f1(tg["gain_dir"][backend], preds["gain_dir_head"][backend])

        # TGC
        band_err = np.abs(preds["tgc_delta_db"] - tg["delta_tgc_db"])[backend]
        m["tgc_mae_db"] = float(band_err.mean())
        m["tgc_band_mae_db"] = [round(float(v), 3) for v in band_err.mean(axis=0)]
        pred_levels = preds["tgc_delta_db"] / tgc_slope[:, None]
        slider_deadband = tg["deadband_gain_levels"] * gain_slope / tgc_slope
        f1_derived, f1_head = [], []
        for g, (name, lo, hi) in enumerate(K.SLIDER_GROUPS):
            d = direction_from_delta_np(pred_levels[:, lo:hi].mean(axis=1), slider_deadband)
            f1_derived.append(macro_f1(tg["slider_dir"][backend, g], d[backend]))
            m["slider_%s_dir_f1_derived" % name] = f1_derived[-1]
            if "slider_dir_head" in preds:
                f1_head.append(macro_f1(tg["slider_dir"][backend, g], preds["slider_dir_head"][backend, g]))
        m["slider_dir_f1_derived"] = float(np.nanmean(f1_derived))
        if f1_head:
            m["slider_dir_f1_head"] = float(np.nanmean(f1_head))

    # 前端三轴
    for axis in ("depth", "frequency", "focus"):
        mask = (tg["%s_mask" % axis] > 0) & (tg["optimal_%s_idx" % axis] >= 0)
        m["%s_n" % axis] = int(mask.sum())
        if not mask.any():
            continue
        pred = preds["%s_idx" % axis][mask]
        target = tg["optimal_%s_idx" % axis][mask]
        m["%s_top1" % axis] = accuracy(target, pred)
        m["%s_within1" % axis] = float((np.abs(pred - target) <= 1).mean())
        derived = direction_from_index_np(pred, tg["%s_idx" % axis][mask])
        m["%s_dir_acc_derived" % axis] = accuracy(tg["%s_dir" % axis][mask], derived)
        m["%s_dir_f1_derived" % axis] = macro_f1(tg["%s_dir" % axis][mask], derived)
        if "%s_dir_head" % axis in preds:
            m["%s_dir_f1_head" % axis] = macro_f1(tg["%s_dir" % axis][mask], preds["%s_dir_head" % axis][mask])
        if axis == "frequency":
            accept = tg["frequency_acceptable"][mask]
            m["frequency_hit"] = float(accept[np.arange(len(pred)), pred].mean())

    m["dynamic_range_n"] = int((tg["dr_mask"] > 0).sum())
    if "aux" in preds and norm is not None and (tg["aux_mask"] > 0).any():
        a = tg["aux_mask"] > 0
        att = preds["aux"][a, 0] * norm["att_std"] + norm["att_mean"]
        noise = preds["aux"][a, 1] * norm["noise_std"] + norm["noise_mean"]
        m["aux_attenuation_mae"] = float(np.abs(att - tg["attenuation_db_cm_mhz"][a]).mean())
        m["aux_electronic_noise_mae_db"] = float(np.abs(noise - tg["electronic_noise_db"][a]).mean())

    parts = [m.get("gain_dir_f1_derived"), m.get("slider_dir_f1_derived"), m.get("depth_top1"),
             m.get("frequency_hit"), m.get("focus_top1")]
    parts = [p for p in parts if p is not None and np.isfinite(p)]
    m["score"] = float(np.mean(parts)) if parts else float("nan")
    return m


MAIN_KEYS = ("score", "gain_mae_db", "gain_within_deadband", "gain_dir_f1_derived", "tgc_mae_db",
             "slider_dir_f1_derived", "depth_top1", "depth_within1", "depth_dir_f1_derived",
             "frequency_hit", "frequency_dir_f1_derived", "focus_top1", "focus_within1", "focus_dir_f1_derived")


def format_metrics(m, keys=MAIN_KEYS):
    """一行 ASCII。"""
    parts = []
    for k in keys:
        if k in m and m[k] is not None:
            v = m[k]
            parts.append("%s=%.3f" % (k, v) if isinstance(v, float) else "%s=%s" % (k, v))
    return "  ".join(parts)
