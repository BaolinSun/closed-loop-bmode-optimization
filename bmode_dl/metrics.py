# -*- coding: utf-8 -*-
"""评估：网络预测解码、单步指标、只看设置的查表基线。

方向有两种读法，都报告：
    head     方向头的 argmax
    derived  由数值输出推出：增益/TGC 的预测 delta 与该行死区比较；前端预测最优档与当前档比较。
             闭环里实际执行的是 derived，所以它是主要指标。

综合分数 score = 平均(增益方向 F1[derived], 三组滑块方向 F1[derived], 深度最优档准确率,
                     频率可接受集合命中率, 聚焦最优档准确率)。动态范围无标签，不计入。
backend_score = 前两项平均；frontend_score = 后三项平均。

    近最优帧（*_near）

fieldii_v2 的闭环停点几乎都落在判据门槛附近：聚焦单步准确率 0.86，而停点上只有 0.50。原因是
单步指标里多数起点离最优很远、很好判断，闭环停点却总在"再调一档还是就停"的边界上。所以另报一套
只统计当前设置与最优相差不超过 1 档的帧的指标（*_near），frontend_score_near 用来选前端检查点。

    档位决策方式

argmax    取概率最大的档
expected  取概率分布的期望档位、再取最近的可选档。分布偏向一侧时期望值会跟着偏，
          可以看出 fieldii_v2 深度、聚焦系统性偏浅（平均 +0.4 档）是不是纯粹由 argmax 造成的。
两种都算，指标里以 argmax 为准、expected 的结果以 *_expected 报告。
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


def nearest_valid_index(target, valid):
    """离 target（可以是小数）最近的可选档下标。target (B,)，valid (B, K)。"""
    k = torch.arange(valid.shape[1], device=valid.device, dtype=torch.float32)
    distance = (k[None, :] - target[:, None]).abs()
    distance = distance.masked_fill(valid <= 0, float("inf"))
    return distance.argmin(dim=1)


def ladder_decision(logits, valid, mode="argmax"):
    """(预测档下标, 概率)。mode="expected" 时取期望档位再取最近的可选档。"""
    masked = masked_logits(logits, valid)
    prob = masked.softmax(dim=1)
    if mode == "expected":
        k = torch.arange(prob.shape[1], device=prob.device, dtype=prob.dtype)
        return nearest_valid_index((prob * k[None, :]).sum(dim=1).float(), valid), prob
    return masked.argmax(dim=1), prob


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
    """网络输出 -> 可执行的建议。tg 只用到可选档位掩膜（推理时按设备的档位表给出）。

    前端三轴同时给出 argmax 档、期望档（*_idx_expected）与整条概率（*_prob，闭环的滞回要用）。
    """
    decoded = {
        "gain_delta_db": out["gain_delta_db"].float(),
        "gain_dir_head": out["gain_dir"].argmax(dim=1),
        "tgc_delta_db": out["tgc_delta_db"].float(),
        "slider_dir_head": out["slider_dir"].argmax(dim=2),
        "depth_dir_head": out["depth_dir"].argmax(dim=1),
        "frequency_dir_head": out["frequency_dir"].argmax(dim=1),
        "focus_dir_head": out["focus_dir"].argmax(dim=1),
        "dr_delta_ui": out["dr_delta_ui"].float(),
        "aux": out["aux"].float(),
    }
    for axis in ("depth", "frequency", "focus"):
        logits, valid = out["%s_logits" % axis], tg["%s_valid" % axis]
        decoded["%s_idx" % axis], decoded["%s_prob" % axis] = ladder_decision(logits, valid, "argmax")
        decoded["%s_idx_expected" % axis], _ = ladder_decision(logits, valid, "expected")
    return decoded


@torch.no_grad()
def gain_feedback_slope(model, data, builder, idx, delta_db=4.0, max_frames=128, seed=0, amp=False,
                        batch_size=64):
    """预测最优增益对当前增益的斜率（每帧一个），用来发现闭环正反馈。

    同一帧在当前增益 -delta 与 +delta 两处各预测一次最优值，斜率 = 两者之差 / (2 delta)。
    正确的模型斜率约为 0（最优值是图像的属性）；大于 1 时每做一次后端修正，目标跑得更远，闭环
    必然发散。单步指标看不出这个问题（fieldii_v4 的 data 模型单步很好，斜率 1.45）。
    返回 (中位数, 大于 1 的比例)。
    """
    model.eval()
    idx = np.asarray(idx)
    if len(idx) > max_frames:
        idx = np.random.RandomState(seed).choice(idx, max_frames, replace=False)
    slopes = []
    for s in range(0, len(idx), batch_size):
        batch_idx = idx[s:s + batch_size]
        t = torch.as_tensor(batch_idx, device=data.device, dtype=torch.long)
        gain, levels = data.t["gain_db"][t].float(), data.t["tgc_levels"][t].float()
        optima = []
        for shift in (-delta_db, delta_db):
            state = {"gain_db": gain + shift, "tgc_levels": levels}
            inputs, tg = make_batch(data, builder, batch_idx, state=state)
            with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=bool(amp)):
                out = model(inputs)
            optima.append(out["gain_delta_db"].float() + state["gain_db"])
        slopes.append(((optima[1] - optima[0]) / (2.0 * delta_db)).cpu().numpy())
    slopes = np.concatenate(slopes) if slopes else np.zeros(0)
    if not slopes.size:
        return float("nan"), float("nan")
    return float(np.median(slopes)), float((slopes > 1.0).mean())


# ---------------------------------------------------------------------------------------
#   只看设置的查表基线

def settings_lookup_baseline(data, train_idx, eval_idx, start="label", seed=None):
    """同一 (成像模式, 深度, 频率, 聚焦) 设置下训练集的众数最优档 / 中位最优增益与滑块。

    对应 docs/fieldii_labels_20260915.md §4.1 "只用设置猜方向" 的检查；网络要比它好才说明看了图像。

    成像模式必须在键里：实机的基波与谐波增益刻度相差很大（每级 dB 不同、档位区间不同），而 5.0 MHz
    两种模式都有、深度与聚焦档也共用，不分模式时同一格里混着两种模式，没见过的格还会退回两种
    模式合在一起的中位数。console_ft_v1 的首次评估就是这样：查表增益误差 5.24 dB，比只按模式取
    均值（约 2.4 dB）还差，把网络的优势夸大了一倍多。Field II 只有基波，加这一维不影响它的结果。
    没见过的格退回同模式的训练集中位数。
    """
    train_idx = np.asarray(train_idx)
    enc = data.encoded
    table = defaultdict(list)
    by_mode = defaultdict(list)
    for i in train_idx:
        mode = int(enc["mode"][i])
        table[(mode, int(enc["depth_idx"][i]), int(enc["frequency_idx"][i]), int(enc["focus_idx"][i]))].append(i)
        by_mode[mode].append(i)

    def mode_of(values, default):
        values = [v for v in values if v >= 0]
        if not values:
            return default
        return int(np.bincount(values).argmax())

    fallback_gain = {m: np.median(enc["optimal_gain_db"][ix]) for m, ix in by_mode.items()}
    fallback_tgc = {m: np.median(enc["optimal_tgc_levels"][ix], axis=0) for m, ix in by_mode.items()}
    all_gain = np.median(enc["optimal_gain_db"][train_idx])
    all_tgc = np.median(enc["optimal_tgc_levels"][train_idx], axis=0)
    _, targets = _targets_only(data, eval_idx, start, seed)
    n = len(eval_idx)
    preds = {"gain_delta_db": np.zeros(n, np.float32), "tgc_delta_db": np.zeros((n, K.NUM_TGC_BANDS), np.float32),
             "depth_idx": np.zeros(n, np.int64), "frequency_idx": np.zeros(n, np.int64),
             "focus_idx": np.zeros(n, np.int64)}
    for j, i in enumerate(eval_idx):
        mode = int(enc["mode"][i])
        key = (mode, int(enc["depth_idx"][i]), int(enc["frequency_idx"][i]), int(enc["focus_idx"][i]))
        members = table.get(key, [])
        opt_gain = (np.median(enc["optimal_gain_db"][members]) if members
                    else fallback_gain.get(mode, all_gain))
        opt_tgc = (np.median(enc["optimal_tgc_levels"][members], axis=0) if members
                   else fallback_tgc.get(mode, all_tgc))
        tgc_slope = K.TGC_DB_PER_LEVEL[mode]
        preds["gain_delta_db"][j] = opt_gain - targets["gain_db"][j]
        preds["tgc_delta_db"][j] = (opt_tgc - targets["tgc_levels"][j]) * tgc_slope
        preds["depth_idx"][j] = mode_of([int(enc["optimal_depth_idx"][m]) for m in members], key[1])
        preds["frequency_idx"][j] = mode_of([int(enc["optimal_frequency_idx"][m]) for m in members], key[2])
        preds["focus_idx"][j] = mode_of([int(enc["optimal_focus_idx"][m]) for m in members], key[3])
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
        current = tg["%s_idx" % axis][mask]
        m["%s_top1" % axis] = accuracy(target, pred)
        m["%s_within1" % axis] = float((np.abs(pred - target) <= 1).mean())
        m["%s_mean_signed_steps" % axis] = float(np.mean(target - pred))     # 正数 = 网络偏小（偏浅）
        derived = direction_from_index_np(pred, current)
        m["%s_dir_acc_derived" % axis] = accuracy(tg["%s_dir" % axis][mask], derived)
        m["%s_dir_f1_derived" % axis] = macro_f1(tg["%s_dir" % axis][mask], derived)
        if "%s_dir_head" % axis in preds:
            m["%s_dir_f1_head" % axis] = macro_f1(tg["%s_dir" % axis][mask], preds["%s_dir_head" % axis][mask])
        accept = tg["frequency_acceptable"][mask] if axis == "frequency" else None
        if accept is not None:
            m["frequency_hit"] = float(accept[np.arange(len(pred)), pred].mean())

        # 近最优帧：当前设置与最优相差不超过 1 档，闭环停点就落在这一带
        near = np.abs(current - target) <= 1
        m["%s_n_near" % axis] = int(near.sum())
        if near.any():
            if accept is not None:
                m["frequency_hit_near"] = float(accept[near][np.arange(int(near.sum())), pred[near]].mean())
            else:
                m["%s_top1_near" % axis] = accuracy(target[near], pred[near])
            m["%s_mean_signed_steps_near" % axis] = float(np.mean(target[near] - pred[near]))

        # 期望档决策，用来判断偏浅是不是 argmax 造成的
        if "%s_idx_expected" % axis in preds:
            expected = preds["%s_idx_expected" % axis][mask]
            if accept is not None:
                m["frequency_hit_expected"] = float(accept[np.arange(len(expected)), expected].mean())
            else:
                m["%s_top1_expected" % axis] = accuracy(target, expected)
            m["%s_mean_signed_steps_expected" % axis] = float(np.mean(target - expected))

    m["dynamic_range_n"] = int((tg["dr_mask"] > 0).sum())
    if "aux" in preds and norm is not None and (tg["aux_mask"] > 0).any():
        a = tg["aux_mask"] > 0
        att = preds["aux"][a, 0] * norm["att_std"] + norm["att_mean"]
        noise = preds["aux"][a, 1] * norm["noise_std"] + norm["noise_mean"]
        m["aux_attenuation_mae"] = float(np.abs(att - tg["attenuation_db_cm_mhz"][a]).mean())
        m["aux_electronic_noise_mae_db"] = float(np.abs(noise - tg["electronic_noise_db"][a]).mean())

    def mean_of(keys):
        parts = [m.get(k) for k in keys]
        parts = [p for p in parts if p is not None and np.isfinite(p)]
        return float(np.mean(parts)) if parts else float("nan")

    m["score"] = mean_of(BACKEND_SCORE_KEYS + FRONTEND_SCORE_KEYS)
    m["backend_score"] = mean_of(BACKEND_SCORE_KEYS)
    m["frontend_score"] = mean_of(FRONTEND_SCORE_KEYS)
    m["frontend_score_near"] = mean_of(FRONTEND_SCORE_NEAR_KEYS)
    m["frontend_score_expected"] = mean_of(FRONTEND_SCORE_EXPECTED_KEYS)
    return m


# 分项分数：前端与后端的最佳轮次相差很远（fieldii_v1：前端第 10–40 轮，后端第 120–150 轮），
# 训练脚本按这两个分数分别保存 best_frontend.pt / best_backend.pt
BACKEND_SCORE_KEYS = ("gain_dir_f1_derived", "slider_dir_f1_derived")
FRONTEND_SCORE_KEYS = ("depth_top1", "frequency_hit", "focus_top1")
# 近最优帧上的同样三项：闭环停点都在这一带，选前端检查点默认看它
FRONTEND_SCORE_NEAR_KEYS = ("depth_top1_near", "frequency_hit_near", "focus_top1_near")
FRONTEND_SCORE_EXPECTED_KEYS = ("depth_top1_expected", "frequency_hit_expected", "focus_top1_expected")


def flatten_metrics(m, prefix=""):
    """数值指标拍平成一层（列表展开成 _0.._n），写 CSV 用。"""
    flat = {}
    for k, v in m.items():
        if isinstance(v, bool):
            flat[prefix + k] = int(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            flat[prefix + k] = float(v) if isinstance(v, (float, np.floating)) else int(v)
        elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                flat["%s%s_%d" % (prefix, k, i)] = x
    return flat


MAIN_KEYS = ("score", "backend_score", "frontend_score", "frontend_score_near", "gain_feedback_slope", "gain_mae_db", "gain_within_deadband", "gain_dir_f1_derived", "tgc_mae_db",
             "slider_dir_f1_derived", "depth_top1", "depth_top1_near", "depth_within1", "depth_dir_f1_derived",
             "frequency_hit", "frequency_hit_near", "frequency_dir_f1_derived", "focus_top1", "focus_top1_near",
             "focus_within1", "focus_dir_f1_derived")

NEAR_KEYS = ("frontend_score_near", "depth_top1_near", "frequency_hit_near", "focus_top1_near",
             "depth_mean_signed_steps_near", "focus_mean_signed_steps_near")
EXPECTED_KEYS = ("frontend_score_expected", "depth_top1_expected", "frequency_hit_expected",
                 "focus_top1_expected", "depth_mean_signed_steps_expected", "focus_mean_signed_steps_expected")


def format_metrics(m, keys=MAIN_KEYS):
    """一行 ASCII。"""
    parts = []
    for k in keys:
        if k in m and m[k] is not None:
            v = m[k]
            parts.append("%s=%.3f" % (k, v) if isinstance(v, float) else "%s=%s" % (k, v))
    return "  ".join(parts)
