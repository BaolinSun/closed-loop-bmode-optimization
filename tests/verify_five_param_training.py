# -*- coding: utf-8 -*-
"""五参数训练的不变量检查；只计算合成样本的梯度，不执行优化器或训练入口。"""
import copy
import io
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from bmode_dl import constants as K
from bmode_dl.labels import encode_rows, backend_targets, collect_ladders
from bmode_dl.losses import SixParamLoss, acceptable_losses, acceptable_direction_loss
from bmode_dl.metrics import compute_metrics, decode
from bmode_dl.checkpoint import model_from_config
from tools_train_six_param import parse_args, parse_loss_weights

LINES = []


def check(name, condition):
    assert condition, name
    LINES.append("PASS " + name)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(9)
    ladders = {"depth_mm": [25.1, 33.5, 41.9], "frequency_mhz": [4.4, 5.3, 5.7],
               "focus_mm": [10.0, 15.0, 20.0]}
    base = dict(depth_mm=33.5, frequency_mhz=5.3, focus_mm=15.0, imaging_mode="harmonic",
                backend_determined=False, depth_determined=True, optimal_depth_mm=41.9,
                depth_direction="shallow", depth_acceptable_mm=[33.5, 41.9],
                frequency_determined=True, optimal_frequency_mhz=5.7, frequency_direction="low",
                frequency_acceptable_mhz=[5.3, 5.7], frequency_confidence="borderline",
                focus_determined=True, optimal_focus_mm=20.0, focus_direction="shallow")
    rows = [dict(base) for _ in range(4)]
    rows[1]["depth_acceptable_mm"] = None
    rows[2]["depth_determined"] = False
    enc = encode_rows(rows, ladders)
    check("depth sets survive encoding, with hard-label fallback and missing-label mask",
          enc["depth_acceptable"].tolist() == [[0, 1, 1], [0, 0, 1], [0, 0, 0], [0, 1, 1]])
    tg = {k: torch.from_numpy(v) for k, v in enc.items()}
    tg["backend_mask"][:] = 1
    tg["gain_db"][:] = -1
    tg["optimal_gain_db"][:] = 2
    tg["optimal_tgc_levels"][:] = torch.arange(8) * 10 + 95
    tg["deadband_gain_levels"][:] = 1
    tg.update(backend_targets(tg["optimal_gain_db"], tg["optimal_tgc_levels"], tg["gain_db"],
                              tg["tgc_levels"], tg["deadband_gain_levels"], tg["mode"]))
    tg["tgc_db_per_level"] = torch.full((4,), K.TGC_DB_PER_LEVEL[1])
    valid = torch.tensor([[1., 1., 0.]])
    accept = torch.tensor([[0., 1., 1.]])
    logits = torch.tensor([[0., 3., 100.]], requires_grad=True)
    ce, distance, _ = acceptable_losses(logits, valid, accept, torch.ones(1))
    (ce + distance).backward()
    check("invalid logits never enter set likelihood or ordinal distance", logits.grad[0, 2] == 0 and ce < 0.1)
    empty = acceptable_losses(logits.detach(), valid, torch.zeros_like(accept), torch.ones(1))
    check("empty acceptable set has zero loss and zero support", all(float(v) == 0 for v in empty))
    keep = acceptable_direction_loss(torch.tensor([[-8., 8., -8.]]), torch.tensor([1]),
                                    torch.ones_like(accept), accept, torch.ones(1), torch.ones(3))
    move = acceptable_direction_loss(torch.tensor([[8., -8., -8.]]), torch.tensor([1]),
                                    torch.ones_like(accept), accept, torch.ones(1), torch.ones(3))
    check("current acceptable rung supervises keep, not a move to the canonical target", keep < move)
    norm = dict(att_mean=0., att_std=1., noise_mean=0., noise_std=1.)
    cw = {k: np.ones(3) for k in ("gain_dir", "slider_dir", "depth_dir", "frequency_dir", "focus_dir", "dr_dir")}
    args = parse_args(["--frontend-set-targets", "--frontend-select", "frontend_score_balanced",
                       "--loss-weight", "dr_reg=0", "--loss-weight", "dr_dir=0"])
    weights = parse_loss_weights(args.loss_weight)
    for kind in ("softmax", "ordinal"):
        config = dict(vars(args), d_model=64, transformer_layers=1, focus_head=kind)
        model = model_from_config(config, ladders, norm).eval()
        inputs = {"image": torch.randn(4, 3, 64, 32), "profile": torch.randn(4, 5, 64),
                  "scalars": torch.randn(4, 17)}
        out = model(inputs)
        criterion = SixParamLoss(cw, norm, weights, frontend_set_targets=True)
        loss, _ = criterion(out, tg)
        loss.backward()
        for head in ("depth_head", "frequency_head", "focus_head", "gain_head", "tgc_band_head"):
            grad = sum(float(p.grad.abs().sum()) for p in getattr(model, head).parameters() if p.grad is not None)
            check(kind + ": " + head + " receives finite nonzero gradients", np.isfinite(grad) and grad > 0)
        check(kind + ": dynamic range receives no gradient", all(
            p.grad is None or not bool(p.grad.any()) for p in model.dr_head.parameters()))
        tg_zero = copy.deepcopy(tg)
        tg_zero["frequency_weight"][:] = 0
        _, logs = criterion(out, tg_zero)
        check(kind + ": zero-confidence frequency suppresses both losses",
              logs["frequency_cls"] == 0 and logs["frequency_dir"] == 0)
        pred = {k: v.detach().numpy() for k, v in decode(out, tg).items()}
        targets = {k: v.numpy() for k, v in tg.items()}
        pred["depth_idx"][:] = 1
        metrics = compute_metrics(pred, targets)
        check(kind + ": acceptable depth differs from strict top-1",
              metrics["depth_top1"] == 0 and abs(metrics["depth_hit"] - 2 / 3) < 1e-6)
        check(kind + ": balanced front-end score is finite", np.isfinite(metrics["frontend_score_balanced"]))
        # 多数类不能把宏平均分数抬高。
        targets["optimal_depth_idx"] = np.array([1, 1, 1, 2])
        targets["depth_mask"][:] = 1
        targets["depth_acceptable"] = np.eye(3)[targets["optimal_depth_idx"]]
        metrics = compute_metrics(pred, targets)
        check(kind + ": majority-only predictor has 0.75 hit but 0.5 balanced hit",
              metrics["depth_hit"] == 0.75 and metrics["depth_balanced_hit"] == 0.5)
    # 只读标签检查：不加载大缓存，不训练，也不改变既定划分。
    path = os.path.join(ROOT, "data", "labels_console_invivo_carotid_target.jsonl")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as handle:
            real = [json.loads(line) for line in handle if line.strip()]
        subjects = {}
        for row in real:
            subjects.setdefault(row["group_id"].rsplit("/", 1)[0], set()).add(row["split"])
        check("real labels keep subjects in exactly one split", all(len(v) == 1 for v in subjects.values()))
        real_ladders = collect_ladders(real)
        encoded = encode_rows(real, real_ladders)
        for axis in ("depth", "frequency"):
            mask = encoded[axis + "_mask"] > 0
            check("real " + axis + " labels all retain nonempty acceptable sets",
                  bool((encoded[axis + "_acceptable"][mask].sum(axis=1) > 0).all()))
            LINES.append("Real %s supervision: %d frames" % (axis, int(mask.sum())))
        LINES.append("Real labels: %d frames, %d subjects" % (len(real), len(subjects)))
    LINES.append("No optimizer steps, training entry points or checkpoint writes were executed.")


if __name__ == "__main__":
    main()
    report = "\n".join(LINES) + "\n"
    with io.open(os.path.splitext(os.path.abspath(__file__))[0] + ".txt", "w", encoding="utf-8") as handle:
        handle.write(report)
    print(report, end="")
