# -*- coding: utf-8 -*-
"""掩膜多任务损失。

每一项都是 Σ m·w·L / (Σ m·w + ε)：m 是该轴是否定出，w 是置信或类别权重。某轴在一个批次里
没有任何定出的行时，该项为 0 且不产生梯度（动态范围目前永远如此）。

    增益    ε-不敏感 Huber（ε = 该行死区 dB）+ 方向加权交叉熵
    TGC     8 段 Huber（dB）+ 预测最优曲线二阶差分超出教师曲线的部分 + 三组方向交叉熵
    深度    最优档交叉熵 + 期望档距离 Σ p_k |k - y| + 方向交叉熵
    频率    可接受集合似然 -log Σ_{k∈可接受} p_k（borderline 行降权）+ 方向交叉熵
    聚焦    同深度，按该行 focus_ladder 掩膜
    动态范围 回归 + 方向（掩膜 dr_mask，现为 0）
    辅助    衰减、电子噪声回归（标准化后 MSE）

前端三轴可加标签平滑（frontend_label_smoothing）：fieldii_v1 里前端训练损失降到 0.001 量级、验证
在第 10–40 轮见顶，是在记训练体模。平滑只摊到可选档上。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import masked_logits

TASKS = ("gain", "tgc", "depth", "frequency", "focus", "dynamic_range", "aux")

DEFAULT_WEIGHTS = {
    "gain_reg": 1.0, "gain_dir": 0.5,
    "tgc_reg": 1.0, "tgc_smooth": 0.02, "tgc_dir": 0.5,
    "depth_cls": 1.0, "depth_ord": 0.5, "depth_dir": 0.5,
    "frequency_cls": 1.0, "frequency_dir": 0.5,
    "focus_cls": 1.0, "focus_ord": 0.5, "focus_dir": 0.5,
    "dr_reg": 1.0, "dr_dir": 0.5,
    "aux": 0.1,
}

TERM_TASK = {"gain_reg": "gain", "gain_dir": "gain", "tgc_reg": "tgc", "tgc_smooth": "tgc", "tgc_dir": "tgc",
             "depth_cls": "depth", "depth_ord": "depth", "depth_dir": "depth",
             "frequency_cls": "frequency", "frequency_dir": "frequency",
             "focus_cls": "focus", "focus_ord": "focus", "focus_dir": "focus",
             "dr_reg": "dynamic_range", "dr_dir": "dynamic_range", "aux": "aux"}


def masked_mean(values, weights, eps=1e-6):
    weights = weights.float()
    total = weights.sum()
    return (values.float() * weights).sum() / (total + eps), total


def huber(x, delta=1.0):
    ax = x.abs()
    return torch.where(ax < delta, 0.5 * ax * ax / delta, ax - 0.5 * delta)


def weighted_ce(logits, target, mask, class_weights, smoothing=0.0):
    """带类别权重的交叉熵；target 为 -1 的行由 mask 排除。smoothing 为标签平滑系数。"""
    safe = target.clamp(min=0)
    nll = F.cross_entropy(logits.float(), safe, reduction="none", label_smoothing=float(smoothing))
    w = mask.float() * (target >= 0).float() * class_weights[safe]
    return masked_mean(nll, w)


def smoothing_target(valid, smoothing):
    """标签平滑只摊到可选档上：不可选档（logits 被置为 -1e4）的目标概率为 0。"""
    valid = (valid > 0).float()
    return valid / valid.sum(dim=1, keepdim=True).clamp(min=1.0)


def ladder_losses(logits, valid, target, mask, smoothing=0.0):
    """最优档交叉熵（可选档内的标签平滑）+ 期望档距离。"""
    logits = masked_logits(logits, valid)
    safe = target.clamp(min=0)
    m = mask.float() * (target >= 0).float()
    logp = F.log_softmax(logits, dim=1)
    nll = -logp.gather(1, safe[:, None]).squeeze(1)
    if smoothing > 0:
        uniform = smoothing_target(valid, smoothing)
        nll = (1.0 - smoothing) * nll - smoothing * (uniform * logp).sum(dim=1)
    ce, total = masked_mean(nll, m)
    prob = logits.softmax(dim=1)
    k = torch.arange(logits.shape[1], device=logits.device, dtype=prob.dtype)
    distance = (prob * (k[None, :] - safe[:, None].to(prob.dtype)).abs()).sum(dim=1)
    ordinal, _ = masked_mean(distance, m)
    return ce, ordinal, total


class SixParamLoss(nn.Module):
    def __init__(self, class_weights, norm, weights=None, borderline_weight=0.5,
                 uncertainty_weighting=False, frontend_label_smoothing=0.0):
        super().__init__()
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)
        for key, value in class_weights.items():
            self.register_buffer("cw_" + key, torch.as_tensor(value, dtype=torch.float32), persistent=False)
        self.norm = dict(norm)
        self.borderline_weight = float(borderline_weight)
        self.smoothing = float(frontend_label_smoothing)
        self.uncertainty_weighting = bool(uncertainty_weighting)
        if self.uncertainty_weighting:
            self.log_vars = nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in TASKS})

    def cw(self, key):
        return getattr(self, "cw_" + key)

    def forward(self, out, tg):
        terms, counts = {}, {}
        backend = tg["backend_mask"]

        # 增益
        err = (out["gain_delta_db"].float() - tg["delta_gain_db"]).abs()
        insensitive = F.relu(err - tg["gain_deadband_db"])
        terms["gain_reg"], counts["gain"] = masked_mean(huber(insensitive), backend)
        terms["gain_dir"], _ = weighted_ce(out["gain_dir"], tg["gain_dir"], backend, self.cw("gain_dir"))

        # TGC
        tgc_err = huber(out["tgc_delta_db"].float() - tg["delta_tgc_db"]).mean(dim=1)
        terms["tgc_reg"], counts["tgc"] = masked_mean(tgc_err, backend)
        # 平滑：预测的最优曲线不应比教师的最优曲线更"弯"（只罚超出教师曲率的部分）
        current_db = (tg["tgc_levels"] - 127.0) * tg["tgc_db_per_level"][:, None]
        teacher_db = (tg["optimal_tgc_levels"] - 127.0) * tg["tgc_db_per_level"][:, None]
        curve = current_db + out["tgc_delta_db"].float()
        second = lambda c: c[:, 2:] - 2.0 * c[:, 1:-1] + c[:, :-2]
        excess = F.relu(second(curve).abs() - second(teacher_db).abs())
        terms["tgc_smooth"], _ = masked_mean((excess ** 2).mean(dim=1), backend)
        groups = tg["slider_dir"].shape[1]
        dir_logits = out["slider_dir"].reshape(-1, 3)
        dir_target = tg["slider_dir"].reshape(-1)
        dir_mask = backend[:, None].expand(-1, groups).reshape(-1)
        terms["tgc_dir"], _ = weighted_ce(dir_logits, dir_target, dir_mask, self.cw("slider_dir"))

        # 深度
        terms["depth_cls"], terms["depth_ord"], counts["depth"] = ladder_losses(
            out["depth_logits"], tg["depth_valid"], tg["optimal_depth_idx"], tg["depth_mask"], self.smoothing)
        terms["depth_dir"], _ = weighted_ce(out["depth_dir"], tg["depth_dir"], tg["depth_mask"], self.cw("depth_dir"),
                                            self.smoothing)

        # 频率：可接受集合
        f_logits = masked_logits(out["frequency_logits"], tg["frequency_valid"])
        accept = masked_logits(out["frequency_logits"], tg["frequency_acceptable"])
        nll = torch.logsumexp(f_logits, dim=1) - torch.logsumexp(accept, dim=1)
        if self.smoothing > 0:
            logp = F.log_softmax(f_logits, dim=1)
            uniform = smoothing_target(tg["frequency_valid"], self.smoothing)
            nll = (1.0 - self.smoothing) * nll - self.smoothing * (uniform * logp).sum(dim=1)
        f_weight = (tg["frequency_mask"] * tg["frequency_weight"]
                    * (1.0 - tg["frequency_borderline"] * (1.0 - self.borderline_weight)))
        terms["frequency_cls"], counts["frequency"] = masked_mean(nll, f_weight)
        terms["frequency_dir"], _ = weighted_ce(out["frequency_dir"], tg["frequency_dir"], tg["frequency_mask"],
                                                self.cw("frequency_dir"), self.smoothing)

        # 聚焦
        terms["focus_cls"], terms["focus_ord"], counts["focus"] = ladder_losses(
            out["focus_logits"], tg["focus_valid"], tg["optimal_focus_idx"], tg["focus_mask"], self.smoothing)
        terms["focus_dir"], _ = weighted_ce(out["focus_dir"], tg["focus_dir"], tg["focus_mask"], self.cw("focus_dir"),
                                            self.smoothing)

        # 动态范围（目前无定出的标签，两项恒为 0）
        terms["dr_reg"], counts["dynamic_range"] = masked_mean(
            huber((out["dr_delta_ui"].float() - tg["delta_dr_ui"]) / 10.0), tg["dr_mask"])
        terms["dr_dir"], _ = weighted_ce(out["dr_dir"], tg["dr_dir"], tg["dr_mask"], self.cw("dr_dir"))

        # 辅助
        aux_target = torch.stack([(tg["attenuation_db_cm_mhz"] - self.norm["att_mean"]) / self.norm["att_std"],
                                  (tg["electronic_noise_db"] - self.norm["noise_mean"]) / self.norm["noise_std"]],
                                 dim=1)
        terms["aux"], counts["aux"] = masked_mean(((out["aux"].float() - aux_target) ** 2).mean(dim=1),
                                                  tg["aux_mask"])

        task_loss = {t: torch.zeros((), device=backend.device) for t in TASKS}
        for name, value in terms.items():
            task_loss[TERM_TASK[name]] = task_loss[TERM_TASK[name]] + self.weights.get(name, 0.0) * value

        total = torch.zeros((), device=backend.device)
        for task, value in task_loss.items():
            if float(counts[task]) <= 0:
                continue                     # 本批无定出的行：不计入（也不更新不确定性参数）
            if self.uncertainty_weighting and task != "aux":
                s = self.log_vars[task]
                total = total + torch.exp(-s) * value + s
            else:
                total = total + value
        logs = {k: float(v.detach()) for k, v in terms.items()}
        logs["total"] = float(total.detach())
        return total, logs
