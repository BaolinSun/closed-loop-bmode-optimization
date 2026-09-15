# -*- coding: utf-8 -*-
"""GPU 上的后端渲染，与 bmode_opt/hisense_backend_sim.render(db_image=...) 一致。

    gray = clip(round(43 + (dB + TGC曲线 + 增益 - reference_db) / 窗宽 * 255), 0, 255)

TGC 曲线：8 个滑块值放在各段中心（hisense_loader.band_centres，按原始 1851 个深度点算），
中间线性插值、两端外保持常数（np.interp 的行为）。缓存把深度方向分块降采样，每个缓存行
取它覆盖的原始行下标的平均位置来插值；TGC 曲线在段中心之间是直线，所以块内平均与块中心
取值相同（跨过段中心的块差别小于一个块）。缓存行数等于原始行数时与原实现逐像素一致。
"""

import numpy as np
import torch

from . import constants as K


def block_edges(out_rows, source_rows=K.FIELDII_SOURCE_ROWS):
    """降采样分块边界（缓存脚本用同一个函数）。"""
    return np.linspace(0, int(source_rows), int(out_rows) + 1).round().astype(np.int64)


def row_positions(out_rows, source_rows=K.FIELDII_SOURCE_ROWS):
    """每个缓存行覆盖的原始行下标的平均位置。"""
    edges = block_edges(out_rows, source_rows)
    return (edges[:-1] + edges[1:] - 1) / 2.0


def band_centres(num_points, num_bands=K.NUM_TGC_BANDS):
    """hisense_loader.band_centres 的复制。"""
    edges = np.linspace(0, int(num_points), int(num_bands) + 1).round().astype(int)
    return (edges[:-1] + edges[1:]) / 2.0


def tgc_interp_matrix(out_rows, source_rows=K.FIELDII_SOURCE_ROWS, num_bands=K.NUM_TGC_BANDS):
    """(out_rows, num_bands) 线性插值矩阵：曲线 = 滑块 dB @ M.T。"""
    centres = band_centres(source_rows, num_bands)
    positions = row_positions(out_rows, source_rows)
    matrix = np.zeros((len(positions), num_bands), dtype=np.float64)
    for band in range(num_bands):
        one_hot = np.zeros(num_bands)
        one_hot[band] = 1.0
        matrix[:, band] = np.interp(positions, centres, one_hot)
    return matrix


class BackendRenderer(torch.nn.Module):
    """按当前增益 / TGC / 动态范围把 dB 图渲染成灰阶。无可学习参数。"""

    def __init__(self, out_rows, source_rows=K.FIELDII_SOURCE_ROWS):
        super().__init__()
        self.register_buffer("tgc_matrix",
                             torch.tensor(tgc_interp_matrix(out_rows, source_rows), dtype=torch.float64),
                             persistent=False)

    def tgc_curve_db(self, tgc_levels, mode):
        """tgc_levels (B, 8) 滑块值，mode (B,) 成像模式 -> (B, rows) dB 曲线。"""
        slope = torch.where(mode > 0.5,
                            torch.full_like(mode, K.TGC_DB_PER_LEVEL[K.MODE_HARMONIC]),
                            torch.full_like(mode, K.TGC_DB_PER_LEVEL[K.MODE_FUNDAMENTAL]))
        band_db = (tgc_levels - K.TGC_CENTER_LEVEL) * slope[:, None]
        return band_db @ self.tgc_matrix.t().to(band_db.dtype)

    def forward(self, db, gain_db, tgc_levels, reference_db, dr_ui, mode, round_gray=True):
        """db (B, rows, lines)，其余 (B,) 或 (B, 8)。返回 0..255 的 float 灰阶。"""
        # 传入 float64 时全程 float64（verify 脚本逐像素对照用），否则 float32
        dtype = torch.float64 if db.dtype == torch.float64 else torch.float32
        db = db.to(dtype)
        curve = self.tgc_curve_db(tgc_levels.to(dtype), mode.to(dtype))
        curve = curve.to(dtype)
        window = K.dr_ui_to_window_db(dr_ui.to(dtype)).clamp(min=1.0)
        shifted = db + curve[:, :, None] + (gain_db.to(dtype) - reference_db.to(dtype))[:, None, None]
        gray = K.GRAY_PIVOT + shifted / window[:, None, None] * K.GRAY_MAX
        if round_gray:
            gray = torch.round(gray)
        return gray.clamp(0.0, K.GRAY_MAX)
