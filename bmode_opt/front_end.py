# -*- coding: utf-8 -*-
"""前端三根轴（显示深度、发射频率、发射聚焦）的判据。

    与后端目标函数的分工

后端那五项（压黑、饱和、深度均匀性、噪声抬亮、亮度）全是关于**显示窗口摆放**的，
一帧就能算，也一帧就能改。前端三个参数改变 BC0 本身，最优只能靠比较同一场景下
真实采到的多帧得到，所以这里的每一项都是**族内比较用的标量**，不是能直接优化的
连续量。

比较必须在各自的后端最优上进行：4.0 MHz 和 8.0 MHz 两帧若用同一增益比较，
8.0 MHz 那帧一定更暗，于是「看起来更差」——那跟频率选得对不对无关。

    每一项都是量出来才留下的

tests/measure_front_end_criteria.py 逐项测了分辨力（沿轴变化要大、跨散斑种子
argmin 要稳）。结果决定了这个模块留下什么、扔掉什么：

  留下  侧向半高全宽    聚焦上有真正的内部极值（25 mm，一致率 0.925）
  留下  深度不均匀度    加上求解出的 TGC 之后仍保留 85% 的跨频率跨度
  留下  取景（本模块）  深度轴上唯一能阻止「越浅越好」的项
  扔掉  轴向半高全宽    对聚焦全平（0.2403-0.2433），一致率 0.750，近乎随机
  扔掉  gCNR            全无回声囊肿上饱和，整个频率阶梯只有 0.66-0.71
  待定  穿透深度        需要噪声底，Field II 的 noise_enabled 全是 0，见 E8

    为什么深度需要取景项

不加取景项时，每一个判据都选最浅的那一档——浅图衰减范围小，什么都更好看。
阻止你一直变浅的唯一原因是**结构会被切到图外**。所以深度的判据不是画质，是
取景：既要装得下感兴趣的结构，又不要浪费大片什么都没有的深处。
"""

import numpy as np

# 结构下方要留出的余量。装到刚好卡住底边不算装下了，操作者需要看到结构周围一圈。
FRAMING_MARGIN_MM = 5.0

# 点靶半高全宽在 -6 dB 处量（dB 图是 20log10(包络)，-6 dB 即半幅）。
FWHM_DROP_DB = 6.0
TARGET_WINDOW_MM = (3.0, 3.0)


def structure_extent_mm(truth_mask, geometry, point_targets_mm=None):
    """本帧视野里最深的结构在哪个深度，单位 mm。没有结构返回 None。

    Field II 的 truth_mask：0 是背景组织，1 是无回声囊肿，2 是高回声夹杂。
    点靶体模的 truth_mask 全是 0，靶点位置另由 point_targets_mm 给出。
    均匀体模两者皆无——那种场景**定不了深度**，因为每个深度看到的都一样，
    这是事实而不是缺陷，标签必须留空。
    """
    deepest = None
    if truth_mask is not None:
        rows = np.where(np.any(np.asarray(truth_mask) > 0, axis=1))[0]
        if rows.size:
            deepest = geometry.min_depth_mm + (rows[-1] + 0.5) * geometry.mm_per_point
    if point_targets_mm is not None and len(point_targets_mm):
        targets = np.asarray(point_targets_mm).reshape(-1, 2)
        in_view = targets[targets[:, 0] <= geometry.depth_mm]
        if in_view.size:
            candidate = float(in_view[:, 0].max())
            deepest = candidate if deepest is None else max(deepest, candidate)
    return deepest


def framing_cost(depth_mm, deepest_structure_mm, margin_mm=FRAMING_MARGIN_MM):
    """显示深度取景得好不好。0 最好。

    两项相加，两边都是相对于显示深度的比例，所以不同深度之间可比：

      截断  结构伸到图外的部分。切掉结构是硬伤，权重给 2。
      浪费  结构（含余量）之下什么都没有的那一段。浪费像素也压低帧率，权重 1。

    结构刚好落在「装得下且底下不空太多」的位置时两项同时为 0，所以这一项在深度
    轴上有内部极值——这正是其余判据都没有的。
    """
    depth_mm = float(depth_mm)
    if deepest_structure_mm is None or depth_mm <= 0:
        return float("nan")
    wanted = float(deepest_structure_mm) + margin_mm
    truncated = max(0.0, wanted - depth_mm) / depth_mm
    wasted = max(0.0, depth_mm - wanted) / depth_mm
    return 2.0 * truncated + 1.0 * wasted


def _width_at_drop(profile, spacing, peak_index):
    """剖面上 -6 dB 处的全宽，两侧交点线性插值。"""
    level = profile[peak_index] - FWHM_DROP_DB
    left = right = None
    for k in range(peak_index, 0, -1):
        if profile[k - 1] <= level <= profile[k]:
            span = profile[k] - profile[k - 1]
            left = (k - 1) + (level - profile[k - 1]) / span if span else k
            break
    for k in range(peak_index, profile.size - 1):
        if profile[k + 1] <= level <= profile[k]:
            span = profile[k] - profile[k + 1]
            right = k + (profile[k] - level) / span if span else k
            break
    if left is None or right is None:
        return float("nan")
    return float((right - left) * spacing)


def lateral_resolution_mm(db_image, geometry, point_targets_mm):
    """视野内点靶侧向半高全宽的中位数，单位 mm。越小越好。

    在 dB 图上量，不过显示——分辨率是前端的性质，显示窗口既不能改善也不能破坏它。
    """
    if point_targets_mm is None or not len(point_targets_mm):
        return float("nan")
    widths = []
    half_rows = max(2, int(TARGET_WINDOW_MM[0] / geometry.mm_per_point))
    half_cols = max(2, int(TARGET_WINDOW_MM[1] / geometry.mm_per_line))
    for depth_mm, offset_mm in np.asarray(point_targets_mm).reshape(-1, 2):
        row = int(round(geometry.row_of_depth(depth_mm)))
        column = int(round((offset_mm + geometry.width_mm / 2.0) / geometry.mm_per_line))
        if not (0 <= row < geometry.num_points and 0 <= column < geometry.num_lines):
            continue
        r0, r1 = max(0, row - half_rows), min(geometry.num_points, row + half_rows + 1)
        c0, c1 = max(0, column - half_cols), min(geometry.num_lines, column + half_cols + 1)
        patch = db_image[r0:r1, c0:c1]
        if patch.size < 9:
            continue
        local = np.unravel_index(np.argmax(patch), patch.shape)
        widths.append(_width_at_drop(patch[local[0], :], geometry.mm_per_line, local[1]))
    if not widths or np.all(np.isnan(widths)):
        return float("nan")
    return float(np.nanmedian(widths))


def penetration_depth_mm(db_image, geometry, noise_floor_db, margin_db=3.0):
    """组织仍高出噪声底 margin_db 的最深处，单位 mm。

    没有噪声底就没有穿透可言：噪声底为 None 时返回 nan，而不是假装信号一直有。
    Field II 数据集的 noise_enabled 全部为 0，所以这一项在那边永远是 nan，
    频率标签也因此定不下来——待 E8 实测逐频率的噪声底后补齐。
    """
    if noise_floor_db is None or not np.isfinite(noise_floor_db):
        return float("nan")
    rows = np.median(np.asarray(db_image, dtype=np.float64), axis=1)
    good = np.where(rows >= noise_floor_db + margin_db)[0]
    if good.size == 0:
        return float(geometry.min_depth_mm)
    return float(geometry.min_depth_mm + (good[-1] + 0.5) * geometry.mm_per_point)
