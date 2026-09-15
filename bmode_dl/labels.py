# -*- coding: utf-8 -*-
"""标签编码与后端起点重抽。

    编码

jsonl 每行变成定长数组：档位下标、可接受集合多热、聚焦可选掩膜、各轴是否定出的掩膜。
所有方向统一成三类下标：0 = 应当增大（最优值大于当前），1 = 正确，2 = 应当减小。

    后端起点重抽（训练时的主要数据增强）

Field II 标签的后端起点是 bmode_opt/labels.draw_start 随机抽的，最优值只取决于图像，不随
起点变。所以每个批次都可以重新抽一个起点，重算 delta 与方向，再按新起点渲染输入灰阶。
逻辑与 bmode_opt/labels.py 逐项对应：

    delta_gain_clicks ~ U(-20, 25)
    delta_levels      = U(-110, 110) * 倾斜基 + U(-70, 70) * 弓形基
    起点增益 dB       = 最优增益 - delta_gain_clicks * 增益 dB/级     （不取整）
    起点滑块          = round(clip(最优滑块 - delta_levels, 0, 255))
    增益方向          |delta_gain_clicks| <= 死区 -> 正确；> 0 偏暗；< 0 偏亮
    滑块组方向        组内 delta_tgc_levels 平均，死区 = 增益死区 * 增益dB/级 / TGC dB/级

tests/verify_six_param_model.py 用 jsonl 记录的起点重算，检查与记录的 delta、方向一致。
"""

import numpy as np
import torch

from . import constants as K


# ---------------------------------------------------------------------------------------
#   档位

def collect_ladders(rows):
    """所有行的深度 / 频率 / 聚焦档位并集，排好序。写进检查点，推理时按同一套下标解码。"""
    depth, frequency, focus = set(), set(), set()
    for row in rows:
        depth.update(float(v) for v in (row.get("depth_ladder") or []))
        frequency.update(float(v) for v in (row.get("frequency_ladder") or []))
        focus.update(float(v) for v in (row.get("focus_ladder") or []))
        depth.add(float(row["depth_mm"]))
        frequency.add(float(row["frequency_mhz"]))
        focus.add(float(row["focus_mm"]))
    return {"depth_mm": sorted(depth), "frequency_mhz": sorted(frequency), "focus_mm": sorted(focus)}


def ladder_index(ladder, value, tol=1e-3):
    """value 在 ladder 中的下标；找不到返回 -1。"""
    if value is None:
        return -1
    for i, v in enumerate(ladder):
        if abs(float(v) - float(value)) <= tol:
            return i
    return -1


def direction_index(names, name):
    if name is None or name not in names:
        return -1
    return names.index(name)


def slider_shape_basis(num_bands=K.NUM_TGC_BANDS):
    """labels.slider_shape_basis 的复制：零均值的倾斜基与弓形基。"""
    positions = (np.arange(num_bands) - 0.5 * (num_bands - 1)) / (0.5 * (num_bands - 1))
    tilt = positions
    arch = positions ** 2
    arch = arch - arch.mean()
    return tilt, arch / np.abs(arch).max()


# ---------------------------------------------------------------------------------------
#   jsonl 行 -> 数组

def _mode_of(row):
    return K.MODE_HARMONIC if row.get("imaging_mode") == "harmonic" else K.MODE_FUNDAMENTAL


def encode_rows(rows, ladders):
    """把标签行编码成 numpy 数组字典（长度 N）。未定出的轴下标为 -1、掩膜为 0。"""
    n = len(rows)
    nd, nf, nz = len(ladders["depth_mm"]), len(ladders["frequency_mhz"]), len(ladders["focus_mm"])
    out = {
        "mode": np.zeros(n, np.float32),
        "depth_mm": np.zeros(n, np.float32),
        "frequency_mhz": np.zeros(n, np.float32),
        "focus_mm": np.zeros(n, np.float32),
        "depth_idx": np.zeros(n, np.int64),
        "frequency_idx": np.zeros(n, np.int64),
        "focus_idx": np.zeros(n, np.int64),
        # 后端当前与最优
        "gain_db": np.zeros(n, np.float32),
        "tgc_levels": np.zeros((n, K.NUM_TGC_BANDS), np.float32),
        "dr_ui": np.zeros(n, np.float32),
        "reference_db": np.zeros(n, np.float32),
        "optimal_gain_db": np.zeros(n, np.float32),
        "optimal_tgc_levels": np.zeros((n, K.NUM_TGC_BANDS), np.float32),
        "deadband_gain_levels": np.zeros(n, np.float32),
        "delta_gain_db": np.zeros(n, np.float32),
        "delta_tgc_db": np.zeros((n, K.NUM_TGC_BANDS), np.float32),
        "gain_dir": np.full(n, -1, np.int64),
        "slider_dir": np.full((n, len(K.SLIDER_GROUPS)), -1, np.int64),
        "backend_mask": np.zeros(n, np.float32),
        # 动态范围
        "delta_dr_ui": np.zeros(n, np.float32),
        "dr_dir": np.full(n, -1, np.int64),
        "dr_mask": np.zeros(n, np.float32),
        # 前端
        "optimal_depth_idx": np.full(n, -1, np.int64),
        "optimal_focus_idx": np.full(n, -1, np.int64),
        "optimal_frequency_idx": np.full(n, -1, np.int64),
        "frequency_acceptable": np.zeros((n, nf), np.float32),
        "depth_valid": np.zeros((n, nd), np.float32),
        "frequency_valid": np.zeros((n, nf), np.float32),
        "focus_valid": np.zeros((n, nz), np.float32),
        "depth_dir": np.full(n, -1, np.int64),
        "frequency_dir": np.full(n, -1, np.int64),
        "focus_dir": np.full(n, -1, np.int64),
        "depth_mask": np.zeros(n, np.float32),
        "frequency_mask": np.zeros(n, np.float32),
        "focus_mask": np.zeros(n, np.float32),
        "frequency_weight": np.zeros(n, np.float32),
        "frequency_borderline": np.zeros(n, np.float32),
        # 辅助目标（仿真真值，只作监督，不作输入）
        "attenuation_db_cm_mhz": np.zeros(n, np.float32),
        "electronic_noise_db": np.zeros(n, np.float32),
        "aux_mask": np.zeros(n, np.float32),
        "noise_floor_top_db": np.zeros(n, np.float32),
        "noise_floor_bottom_db": np.zeros(n, np.float32),
    }

    for i, row in enumerate(rows):
        mode = _mode_of(row)
        out["mode"][i] = mode
        out["depth_mm"][i] = row["depth_mm"]
        out["frequency_mhz"][i] = row["frequency_mhz"]
        out["focus_mm"][i] = row["focus_mm"]
        out["depth_idx"][i] = ladder_index(ladders["depth_mm"], row["depth_mm"])
        out["frequency_idx"][i] = ladder_index(ladders["frequency_mhz"], row["frequency_mhz"])
        out["focus_idx"][i] = ladder_index(ladders["focus_mm"], row["focus_mm"])

        backend = bool(row.get("backend_determined", "optimal_gain_db" in row))
        if backend:
            out["gain_db"][i] = row["gain_db"]
            out["tgc_levels"][i] = row["tgc_levels"]
            out["dr_ui"][i] = row["dr_ui"]
            out["reference_db"][i] = row["reference_db"]
            out["optimal_gain_db"][i] = row["optimal_gain_db"]
            out["optimal_tgc_levels"][i] = row["optimal_tgc_levels"]
            out["deadband_gain_levels"][i] = row["deadband_gain_levels"]
            out["delta_gain_db"][i] = row["delta_gain_db"]
            out["delta_tgc_db"][i] = row["delta_tgc_db"]
            out["gain_dir"][i] = direction_index(K.GAIN_DIRECTIONS, row["gain_direction"])
            for g, (name, _, _) in enumerate(K.SLIDER_GROUPS):
                out["slider_dir"][i, g] = direction_index(K.SLIDER_DIRECTIONS, row["slider_directions"][name])
            out["backend_mask"][i] = 1.0
        else:
            out["tgc_levels"][i] = K.TGC_CENTER_LEVEL
            out["optimal_tgc_levels"][i] = K.TGC_CENTER_LEVEL
            out["dr_ui"][i] = row.get("dr_ui", 67.0) or 67.0
            out["reference_db"][i] = row.get("reference_db", 0.0) or 0.0

        if row.get("dr_determined"):
            out["delta_dr_ui"][i] = row.get("delta_dr_ui", 0.0)
            out["dr_dir"][i] = direction_index(K.DYNAMIC_RANGE_DIRECTIONS, row.get("dr_direction"))
            out["dr_mask"][i] = 1.0 if out["dr_dir"][i] >= 0 else 0.0

        for v in (row.get("depth_ladder") or ladders["depth_mm"]):
            j = ladder_index(ladders["depth_mm"], v)
            if j >= 0:
                out["depth_valid"][i, j] = 1.0
        for v in (row.get("frequency_ladder") or ladders["frequency_mhz"]):
            j = ladder_index(ladders["frequency_mhz"], v)
            if j >= 0:
                out["frequency_valid"][i, j] = 1.0
        for v in (row.get("focus_ladder") or ladders["focus_mm"]):
            j = ladder_index(ladders["focus_mm"], v)
            if j >= 0:
                out["focus_valid"][i, j] = 1.0

        if row.get("depth_determined"):
            j = ladder_index(ladders["depth_mm"], row.get("optimal_depth_mm"))
            d = direction_index(K.DEPTH_DIRECTIONS, row.get("depth_direction"))
            if j >= 0 and d >= 0:
                out["optimal_depth_idx"][i], out["depth_dir"][i], out["depth_mask"][i] = j, d, 1.0
                out["depth_valid"][i, j] = 1.0
        if row.get("focus_determined"):
            j = ladder_index(ladders["focus_mm"], row.get("optimal_focus_mm"))
            d = direction_index(K.FOCUS_DIRECTIONS, row.get("focus_direction"))
            if j >= 0 and d >= 0:
                out["optimal_focus_idx"][i], out["focus_dir"][i], out["focus_mask"][i] = j, d, 1.0
                out["focus_valid"][i, j] = 1.0
        if row.get("frequency_determined"):
            j = ladder_index(ladders["frequency_mhz"], row.get("optimal_frequency_mhz"))
            d = direction_index(K.FREQUENCY_DIRECTIONS, row.get("frequency_direction"))
            acceptable = row.get("frequency_acceptable_mhz") or [row.get("optimal_frequency_mhz")]
            if j >= 0 and d >= 0:
                out["optimal_frequency_idx"][i], out["frequency_dir"][i], out["frequency_mask"][i] = j, d, 1.0
                for v in acceptable:
                    a = ladder_index(ladders["frequency_mhz"], v)
                    if a >= 0:
                        out["frequency_acceptable"][i, a] = 1.0
                        out["frequency_valid"][i, a] = 1.0
                out["frequency_acceptable"][i, j] = 1.0
                out["frequency_weight"][i] = float(row.get("frequency_loss_weight", 1.0) or 0.0)
                out["frequency_borderline"][i] = 1.0 if row.get("frequency_confidence") == "borderline" else 0.0

        if row.get("attenuation_db_cm_mhz") is not None and row.get("electronic_noise_db") is not None:
            out["attenuation_db_cm_mhz"][i] = row["attenuation_db_cm_mhz"]
            out["electronic_noise_db"][i] = row["electronic_noise_db"]
            out["aux_mask"][i] = 1.0
        out["noise_floor_top_db"][i] = row.get("noise_floor_top_db", 0.0) or 0.0
        out["noise_floor_bottom_db"][i] = row.get("noise_floor_bottom_db", 0.0) or 0.0
    return out


# ---------------------------------------------------------------------------------------
#   后端标签（torch）

def _slopes(mode):
    gain = torch.where(mode > 0.5, torch.full_like(mode, K.GAIN_DB_PER_LEVEL[K.MODE_HARMONIC]),
                       torch.full_like(mode, K.GAIN_DB_PER_LEVEL[K.MODE_FUNDAMENTAL]))
    tgc = torch.where(mode > 0.5, torch.full_like(mode, K.TGC_DB_PER_LEVEL[K.MODE_HARMONIC]),
                      torch.full_like(mode, K.TGC_DB_PER_LEVEL[K.MODE_FUNDAMENTAL]))
    return gain, tgc


def direction_from_delta(delta, deadband):
    """labels._direction：|delta| <= 死区 -> 1；delta > 0 -> 0（应增大）；否则 2。"""
    inside = delta.abs() <= deadband
    return torch.where(inside, torch.ones_like(delta, dtype=torch.long),
                       torch.where(delta > 0, torch.zeros_like(delta, dtype=torch.long),
                                   torch.full_like(delta, 2, dtype=torch.long)))


def backend_targets(optimal_gain_db, optimal_tgc_levels, gain_db, tgc_levels, deadband_gain_levels, mode):
    """由最优值与（任意）当前值算后端标签。全部 (B,) / (B, 8) 张量。"""
    gain_slope, tgc_slope = _slopes(mode)
    delta_gain_levels = (optimal_gain_db - gain_db) / gain_slope
    delta_tgc_levels = optimal_tgc_levels - tgc_levels
    gain_dir = direction_from_delta(delta_gain_levels, deadband_gain_levels)
    slider_deadband = deadband_gain_levels * gain_slope / tgc_slope
    slider_dir = torch.stack([direction_from_delta(delta_tgc_levels[:, lo:hi].mean(dim=1), slider_deadband)
                              for _, lo, hi in K.SLIDER_GROUPS], dim=1)
    return {
        "delta_gain_db": delta_gain_levels * gain_slope,
        "delta_gain_levels": delta_gain_levels,
        "delta_tgc_db": delta_tgc_levels * tgc_slope[:, None],
        "delta_tgc_levels": delta_tgc_levels,
        "gain_dir": gain_dir,
        "slider_dir": slider_dir,
        "gain_deadband_db": deadband_gain_levels * gain_slope,
        "slider_deadband_levels": slider_deadband,
    }


def draw_start(optimal_gain_db, optimal_tgc_levels, mode, generator=None):
    """torch 版 labels.draw_start（起点滑块已取整）。随机数在 CPU 生成器上抽，再搬到数据设备。"""
    b = optimal_gain_db.shape[0]
    device = optimal_gain_db.device
    u = torch.rand((b, 3), generator=generator, dtype=torch.float64).to(device=device, dtype=torch.float32)
    lo, hi = K.START_GAIN_DELTA_CLICKS
    clicks = lo + (hi - lo) * u[:, 0]
    lo, hi = K.START_SLIDER_TILT_LEVELS
    tilt_amp = lo + (hi - lo) * u[:, 1]
    lo, hi = K.START_SLIDER_ARCH_LEVELS
    arch_amp = lo + (hi - lo) * u[:, 2]
    tilt, arch = slider_shape_basis()
    tilt = torch.tensor(tilt, dtype=torch.float32, device=device)
    arch = torch.tensor(arch, dtype=torch.float32, device=device)
    delta_levels = tilt_amp[:, None] * tilt[None, :] + arch_amp[:, None] * arch[None, :]
    gain_slope, _ = _slopes(mode)
    start_gain = optimal_gain_db - clicks * gain_slope
    start_levels = torch.round(torch.clamp(optimal_tgc_levels - delta_levels,
                                           K.TGC_MIN_LEVEL, K.TGC_MAX_LEVEL))
    return start_gain, start_levels
