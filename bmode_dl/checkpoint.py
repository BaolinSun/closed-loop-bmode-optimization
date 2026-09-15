# -*- coding: utf-8 -*-
"""检查点的保存与恢复：模型权重 + 重建网络与输入所需的一切（档位、标准化、类别权重、配置）。"""

import torch

from .dataset import InputBuilder
from .model import SixParamNet


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


def model_from_config(config, ladders):
    return SixParamNet(ladders, input_mode=config.get("input_mode", "full"),
                       d_model=int(config.get("d_model", 256)),
                       transformer_layers=int(config.get("transformer_layers", 2)),
                       use_transformer=not config.get("no_transformer", False),
                       dropout=float(config.get("dropout", 0.1)))


def load_checkpoint(path, device="cpu"):
    """返回 (model, builder, payload)。builder 按检查点记录的缓存形状与标准化重建。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    model = model_from_config(payload["config"], payload["ladders"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    shape = payload["cache_shape"]
    builder = InputBuilder(shape["rows"], shape["lines"], shape["source_rows"], payload["norm"],
                           use_noise_floor=not payload["config"].get("no_noise_floor", False)).to(device)
    return model, builder, payload
