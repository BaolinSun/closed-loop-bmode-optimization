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
                           use_noise_floor=not payload["config"].get("no_noise_floor", False)).to(device)
    return model, builder, payload


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
