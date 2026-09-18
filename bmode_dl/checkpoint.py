# -*- coding: utf-8 -*-
"""检查点的保存与恢复：模型权重 + 重建网络与输入所需的一切（档位、标准化、类别权重、配置）。"""

import torch
import torch.nn as nn

from .dataset import InputBuilder
from .model import SixParamNet

FRONTEND_OUTPUT_KEYS = ("depth_logits", "depth_dir", "frequency_logits", "frequency_dir",
                        "focus_logits", "focus_dir")


def save_checkpoint(path, model, config, ladders, norm, class_weights, cache_shape, extra=None):
    payload = {
        "model": model.state_dict(),
        "config": dict(config),
        "ladders": ladders,
        "norm": norm,
        "class_weights": {k: [float(x) for x in v] for k, v in class_weights.items()},
        "cache_shape": dict(cache_shape),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def model_from_config(config, ladders, norm=None):
    """按配置建网络。没有 backend_output 字段的旧检查点（fieldii_v1）按 delta 输出方式重建。"""
    frontend_dropout = config.get("frontend_dropout")
    return SixParamNet(ladders, input_mode=config.get("input_mode", "full"),
                       d_model=int(config.get("d_model", 256)),
                       transformer_layers=int(config.get("transformer_layers", 2)),
                       use_transformer=not config.get("no_transformer", False),
                       dropout=float(config.get("dropout", 0.1)),
                       backend_output=config.get("backend_output", "delta"),
                       backend_norm=norm,
                       frontend_dropout=None if frontend_dropout is None else float(frontend_dropout))


def load_checkpoint(path, device="cpu"):
    """返回 (model, builder, payload)。builder 按检查点记录的缓存形状与标准化重建。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = model_from_config(payload["config"], payload["ladders"], payload["norm"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    shape = payload["cache_shape"]
    builder = InputBuilder(shape["rows"], shape["lines"], shape["source_rows"], payload["norm"],
                           use_noise_floor=not payload["config"].get("no_noise_floor", False),
                           scalar_norm=payload["config"].get("scalar_norm", "fixed")).to(device)
    return model, builder, payload


# 按训练数据重新估计、微调时不能被预训练值覆盖的缓冲区（optimum 输出方式的标准化）
DOMAIN_BUFFERS = ("gain_opt_mean", "gain_opt_std", "tgc_opt_mean", "tgc_opt_std")


def load_pretrained(model, path, log=print):
    """把预训练检查点的权重装进一个新建的网络（微调用）。

    跳过两类参数：
      形状不同    档位表不同的输出层。实机的深度 7 档、频率 9 档（谐波与基波各 5 档的并集），
                  Field II 是 6 / 4 档，前端三个头的最后一层只能重新初始化；隐藏层照常装入。
      域相关缓冲  DOMAIN_BUFFERS：最优增益 / TGC 的均值方差按新数据重新估计，保留新值。
    返回 (装入数, 形状不同跳过的名字, 缓冲跳过的名字, 新网络里预训练没有的名字)。
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["model"]
    target = model.state_dict()
    loadable, shape_skipped, buffer_skipped = {}, [], []
    for name, value in source.items():
        if name not in target:
            continue
        if name in DOMAIN_BUFFERS:
            buffer_skipped.append(name)
        elif tuple(value.shape) != tuple(target[name].shape):
            shape_skipped.append(name)
        else:
            loadable[name] = value
    missing = [n for n in target if n not in loadable and n not in DOMAIN_BUFFERS and n not in shape_skipped]
    model.load_state_dict(loadable, strict=False)
    log("  pretrained %s (epoch %s, backend_output %s): loaded %d tensors"
        % (path, payload.get("epoch"), payload["config"].get("backend_output", "delta"), len(loadable)))
    if shape_skipped:
        log("    re-initialised (ladder size differs): %s" % ", ".join(shape_skipped))
    if buffer_skipped:
        log("    kept new-domain normalisation: %s" % ", ".join(buffer_skipped))
    if missing:
        log("    not in the pretrained model: %s" % ", ".join(missing))
    return len(loadable), shape_skipped, buffer_skipped, missing, payload


# 冻结范围：encoders 冻结图像与剖面编码器（含其中的 FiLM）；backbone 冻结除各输出头以外的全部
HEAD_MODULES = ("gain_head", "tgc_band_head", "slider_dir_head", "depth_head", "frequency_head",
                "focus_head", "dr_head", "aux_head")
FREEZE_CHOICES = ("none", "encoders", "backbone")


def freeze(model, scope):
    """按范围冻结参数，返回 (冻结的参数量, 仍可训练的参数量)。"""
    if scope not in FREEZE_CHOICES:
        raise ValueError("freeze must be one of %s" % (FREEZE_CHOICES,))
    for name, parameter in model.named_parameters():
        top = name.split(".", 1)[0]
        if scope == "encoders":
            frozen = top in ("image_encoder", "profile_encoder")
        elif scope == "backbone":
            frozen = top not in HEAD_MODULES
        else:
            frozen = False
        parameter.requires_grad_(not frozen)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return frozen_count, trainable


class CombinedModel(nn.Module):
    """前端三轴取一个网络的输出，其余（增益、TGC、动态范围、辅助）取另一个网络的输出。

    用于把同一折的 best_frontend.pt（前端验证峰值，早）和 best_backend.pt（后端验证峰值，晚）
    合起来部署；两者输入必须一致（同一折的标准化、同样的底噪开关），load_combined 会检查。
    """

    def __init__(self, frontend, backend):
        super().__init__()
        self.frontend = frontend
        self.backend = backend

    def forward(self, inputs):
        out = dict(self.backend(inputs))
        front = self.frontend(inputs)
        for key in FRONTEND_OUTPUT_KEYS:
            out[key] = front[key]
        return out


def load_combined(frontend_path, backend_path, device="cpu"):
    """返回 (CombinedModel, builder, backend_payload, frontend_payload)。"""
    front_model, _, front = load_checkpoint(frontend_path, device)
    back_model, builder, back = load_checkpoint(backend_path, device)
    if front["ladders"] != back["ladders"]:
        raise ValueError("ladders differ between %s and %s" % (frontend_path, backend_path))
    for key in ("db_mean", "db_std"):
        if abs(float(front["norm"][key]) - float(back["norm"][key])) > 1e-6:
            raise ValueError("input normalisation differs between the two checkpoints (%s)" % key)
    if bool(front["config"].get("no_noise_floor", False)) != bool(back["config"].get("no_noise_floor", False)):
        raise ValueError("noise-floor input setting differs between the two checkpoints")
    model = CombinedModel(front_model, back_model).to(device).eval()
    return model, builder, back, front
