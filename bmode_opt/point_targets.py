# -*- coding: utf-8 -*-
"""体模线靶（点靶）的半高全宽，以及由它得到的实机逐频率分辨率表。

从 tests/measure_point_target_resolution.py 挪来：频率标签要用分辨率表，正式工具不应
从 tests/ 导入。测量方法与 2026-09-14 的验证完全相同，理由见那个脚本的文档字符串。

    为什么频率标签需要这张表

原规则「覆盖得住的最高频」默认频率越高越锐。点靶验证（docs/console_training_data_20260914.md
1.2 节）表明谐波与基波浅显示深度成立，但基波显示深度 >=41.9 mm 时 10/11.4 MHz 轴向
反而比 8.0 MHz 粗 26-44%，而且这个台阶取决于显示深度，导出参数里查不到原因。所以
「哪一档最锐」不能假设，只能按模式、按显示深度查实测。

    分辨率分数

每根靶在比较集的每个频率下都量到才参与。逐靶把侧向、轴向半高全宽各自除以该靶在
所有频率上的平均，两者取平均，再对所有靶取平均。分数越小越锐，只在同一张表内可比。

  逐靶归一化：深处的靶本来就宽，不归一化会让少数深靶主导。
  侧向与轴向等权：聚焦前侧向只有约 3 个线距，被线密度限制、各频率几乎持平，等权
  不会引入偏向；聚焦后侧向随频率变锐 4-15%，是真实信息，不应丢掉。
"""

import numpy as np
from scipy import ndimage

import calibration as CAL
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

# 没有场次标定时的缺省换算。不是「只影响刻度」：BC0 是对数域，dB = 计数 / counts_per_db，
# 换算系数变了，-6 dB 半高全宽的位置和 25 dB 的靶判据都跟着变。2026-09-15 起 resolution_table
# 按各场次标定值换算（575 / 548 / 559 / 655）；与固定 560 相比最优档只在基波 67 mm 由 6.7
# 变 8.0，两者分数 0.934 与 0.935，本在并列容差内。
COUNTS_PER_DB = 560.0
CALIBRATION_PATH = __import__("pathlib").Path(__file__).resolve().parent / "console_calibration.json"

PIN_EXCESS_DB = 25.0
NEAR_FIELD_MM = 3.0
SECONDARY_MM = 2.0
SECONDARY_LATERAL_MM = 0.6
MATCH_RADIUS_MM = 0.5
FWHM_DROP_DB = 6.0
PLATEAU_DB = 0.05             # 贴着峰值这么近的采样点算平台
PLATEAU_SAMPLES = 3

MODE_CODES = {"fundamental": 0, "harmonic": 1}

# 分辨率表的来源：同一场景族、聚焦 15 mm、只有频率变化的比较集，每种模式三档显示深度。
RESOLUTION_SETS = [
    ("fundamental", "20260903_GEN", 25.1, 15.0),
    ("fundamental", "20260903_GEN", 41.9, 15.0),
    ("fundamental", "20260911_E8_GEN", 67.0, 15.0),
    ("harmonic", "20260903", 25.1, 15.0),
    ("harmonic", "20260903", 41.9, 15.0),
    ("harmonic", "20260911_E8_THI", 67.0, 15.0),
]


def depth_axis(geometry):
    return geometry.min_depth_mm + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point


def lateral_axis(geometry):
    return (np.arange(geometry.num_lines) - (geometry.num_lines - 1) / 2.0) * geometry.mm_per_line


def excess_map(db, geometry):
    """每个像素比周围 3 mm 中位数高出多少 dB。"""
    window = (max(3, int(3.0 / geometry.mm_per_point)), max(3, int(3.0 / geometry.mm_per_line)))
    return db - ndimage.median_filter(db, size=window)


def detect_pins(db, geometry):
    """点靶的 (行, 列, 超出背景 dB)，已剔除近场与次级回波。"""
    excess = excess_map(db, geometry)
    neighbourhood = (max(3, int(1.0 / geometry.mm_per_point)),
                     max(3, int(1.0 / geometry.mm_per_line)))
    peaks = np.argwhere((db == ndimage.maximum_filter(db, size=neighbourhood))
                        & (excess > PIN_EXCESS_DB))
    z, x = depth_axis(geometry), lateral_axis(geometry)
    candidates = [(int(r), int(c), float(excess[r, c])) for r, c in peaks if z[r] >= NEAR_FIELD_MM]
    candidates.sort(key=lambda t: -t[2])
    kept = []
    for r, c, e in candidates:
        secondary = any(abs(x[c] - x[kc]) <= SECONDARY_LATERAL_MM
                        and 0.0 < z[r] - z[kr] <= SECONDARY_MM for kr, kc, _ in kept)
        if not secondary:
            kept.append((r, c, e))
    return kept


def refine_peak(profile, index):
    """抛物线插值的峰值。返回 (峰值 dB, 亚采样位置)。"""
    if index <= 0 or index >= profile.size - 1:
        return float(profile[index]), float(index)
    a, b, c = profile[index - 1], profile[index], profile[index + 1]
    denominator = a - 2.0 * b + c
    if denominator >= 0:
        return float(b), float(index)
    offset = 0.5 * (a - c) / denominator
    return float(b - 0.25 * (a - c) * offset), float(index + offset)


def fwhm(profile, index, spacing):
    """-6 dB 全宽，单位 mm。任一侧找不到交点返回 nan。"""
    peak, _ = refine_peak(profile, index)
    level = peak - FWHM_DROP_DB
    left = right = None
    for k in range(index, 0, -1):
        if profile[k - 1] < level <= profile[k]:
            left = (k - 1) + (level - profile[k - 1]) / (profile[k] - profile[k - 1])
            break
    for k in range(index, profile.size - 1):
        if profile[k + 1] < level <= profile[k]:
            right = k + (profile[k] - level) / (profile[k] - profile[k + 1])
            break
    if left is None or right is None:
        return float("nan")
    return float((right - left) * spacing)


def plateau(profile, index):
    """峰顶贴着峰值的连续采样点数。"""
    top = profile[index]
    count, k = 1, index - 1
    while k >= 0 and top - profile[k] <= PLATEAU_DB:
        count, k = count + 1, k - 1
    k = index + 1
    while k < profile.size and top - profile[k] <= PLATEAU_DB:
        count, k = count + 1, k + 1
    return count


def measure_pin(db, excess, geometry, row, col):
    """在 (row, col) 附近重新找峰，量侧向与轴向半高全宽。"""
    half_rows = max(1, int(MATCH_RADIUS_MM / geometry.mm_per_point))
    half_cols = max(1, int(MATCH_RADIUS_MM / geometry.mm_per_line))
    r0, r1 = max(0, row - half_rows), min(geometry.num_points, row + half_rows + 1)
    c0, c1 = max(0, col - half_cols), min(geometry.num_lines, col + half_cols + 1)
    local = np.unravel_index(np.argmax(db[r0:r1, c0:c1]), (r1 - r0, c1 - c0))
    r, c = r0 + int(local[0]), c0 + int(local[1])
    if excess[r, c] < PIN_EXCESS_DB:
        return None
    return {
        "lateral": fwhm(db[r, :], c, geometry.mm_per_line),
        "axial": fwhm(db[:, c], r, geometry.mm_per_point),
        "plateau": max(plateau(db[r, :], c), plateau(db[:, c], r)),
        "excess": float(excess[r, c]),
    }


def load_set(mode, session, depth, focus):
    """{频率: 帧}。同一频率多帧时取第一帧（同设置重复帧 BC0 几乎一致）。

    必须按成像模式筛。第一版没筛，20260903 场次在 41.9 mm 恰好有一帧 8.0 MHz 的基波，
    被当成谐波组的「最高频率」，把谐波 4.4 MHz 与基波 8.0 MHz 放在一起比，得出侧向
    +48.5%、轴向 +22.4% 的假结果。去掉那一帧后谐波轴向是 -19.4%。
    """
    frames = {}
    for path in sorted(find_captures(DEFAULT_DATA_DIR / session)):
        capture = load_capture(path)
        if (S.capture_image_mode(capture) == MODE_CODES[mode]
                and abs(capture.geometry.depth_mm - depth) < 0.5 and capture.focus_mm == focus):
            frames.setdefault(round(CAL.capture_frequency(capture), 2), capture)
    return frames


def console_db(capture, counts_per_db=COUNTS_PER_DB):
    return S.bc0_to_db(capture.bc0, counts_per_db)


def pin_table(frames, to_db=console_db):
    """[(靶深 mm, 靶横向 mm, {频率: 测量})]，只收每个频率都量得到的靶；以及峰顶平台化的次数。

    to_db 把一帧变成 dB 图。实机默认按 COUNTS_PER_DB 换算；Field II 分片直接给 db_image。
    """
    freqs = sorted(frames)
    reference = frames[freqs[len(freqs) // 2]]
    geometry = reference.geometry
    pins = detect_pins(to_db(reference), geometry)
    z, x = depth_axis(geometry), lateral_axis(geometry)
    prepared = {}
    for f in freqs:
        db = to_db(frames[f])
        prepared[f] = (db, excess_map(db, frames[f].geometry))
    table, clipped = [], 0
    for row, col, _ in pins:
        per_freq = {}
        for f in freqs:
            db, excess = prepared[f]
            m = measure_pin(db, excess, frames[f].geometry, row, col)
            if m is None or not (np.isfinite(m["lateral"]) and np.isfinite(m["axial"])):
                per_freq = None
                break
            if m["plateau"] >= PLATEAU_SAMPLES:
                clipped += 1
            per_freq[f] = m
        if per_freq:
            table.append((float(z[row]), float(x[col]), per_freq))
    table.sort(key=lambda t: t[0])
    return table, clipped


def resolution_scores(table):
    """{频率: 分数}，越小越锐。见模块文档字符串。"""
    freqs = sorted(table[0][2])
    per_pin = []
    for _, _, p in table:
        axial_mean = np.mean([p[f]["axial"] for f in freqs])
        lateral_mean = np.mean([p[f]["lateral"] for f in freqs])
        per_pin.append([0.5 * (p[f]["axial"] / axial_mean + p[f]["lateral"] / lateral_mean)
                        for f in freqs])
    scores = np.mean(per_pin, axis=0)
    return {f: float(s) for f, s in zip(freqs, scores)}


def session_counts_per_db(path=CALIBRATION_PATH):
    """{(场次, 模式代码): counts_per_db}，来自 console_calibration.json。"""
    import json
    with open(path, encoding="utf-8") as handle:
        groups = json.load(handle)["groups"]
    return {(g["session"], g["image_mode"]): float(g["counts_per_db"]) for g in groups}


def resolution_table(sets=RESOLUTION_SETS):
    """{(模式代码, 显示深度): {频率: 分数}}，以及每张表用了几根靶。按场次标定换算 dB。"""
    counts = session_counts_per_db()
    tables, pins = {}, {}
    for mode, session, depth, focus in sets:
        frames = load_set(mode, session, depth, focus)
        if len(frames) < 2:
            continue
        per_db = counts.get((session, MODE_CODES[mode]), COUNTS_PER_DB)
        table, _ = pin_table(frames, lambda capture, k=per_db: console_db(capture, k))
        if not table:
            continue
        tables[(MODE_CODES[mode], depth)] = resolution_scores(table)
        pins[(MODE_CODES[mode], depth)] = len(table)
    return tables, pins


def fieldii_pin_table(frames, half_window_mm=1.5):
    """Field II 版的 pin_table：靶的位置由分片的 point_targets_mm 精确给出，不做检测。

    实机的 pin_table 对整幅图做 3 mm 中值滤波来找靶，浅裁剪分片上窗口有 200 行 x 13 列，
    912 个分片要一个多小时。Field II 靶位已知，只需在每根靶周围 3 mm 见方的小窗里取中位数
    当背景，其余量法（25 dB 判据、抛物线峰值插值、-6 dB 半高全宽、平台检查）与实机相同。
    """
    freqs = sorted(frames)
    reference = frames[freqs[len(freqs) // 2]]
    geometry = reference.geometry
    table, clipped = [], 0
    # 体模声速随机化了 ±60 m/s，波束形成却按 1540 m/s 换算深度，靶在图上的深度是
    # 真实深度 x 1540 / 声速。30 mm 处最多差 1.2 mm，超出 MATCH_RADIUS_MM 就会找不到靶。
    beamform_speed = 1540.0
    speed = float(np.ravel(reference.attrs["sound_speed_mps"])[0])
    for true_depth_mm, lateral_mm in reference.targets_in_view(margin_mm=2.0):
        depth_mm = true_depth_mm * beamform_speed / speed
        if depth_mm < NEAR_FIELD_MM or depth_mm > geometry.depth_mm - 1.0:
            continue
        row = int(round((depth_mm - geometry.min_depth_mm) / geometry.mm_per_point - 0.5))
        col = int(round(lateral_mm / geometry.mm_per_line + (geometry.num_lines - 1) / 2.0))
        per_freq = {}
        for f in freqs:
            shard = frames[f]
            g = shard.geometry
            db = shard.db_image
            hr = int(half_window_mm / g.mm_per_point)
            hc = max(2, int(half_window_mm / g.mm_per_line))
            r0, r1 = max(0, row - hr), min(g.num_points, row + hr + 1)
            c0, c1 = max(0, col - hc), min(g.num_lines, col + hc + 1)
            excess = np.full(db.shape, -np.inf)
            excess[r0:r1, c0:c1] = db[r0:r1, c0:c1] - np.median(db[r0:r1, c0:c1])
            m = measure_pin(db, excess, g, row, col)
            if m is None or not (np.isfinite(m["lateral"]) and np.isfinite(m["axial"])):
                per_freq = None
                break
            if m["plateau"] >= PLATEAU_SAMPLES:
                clipped += 1
            per_freq[f] = m
        if per_freq:
            table.append((float(depth_mm), float(lateral_mm), per_freq))
    table.sort(key=lambda t: t[0])
    return table, clipped


def fieldii_resolution_table(data_dir, split="train"):
    """Field II 的 {(0, 显示深度): {频率: 分数}}，以及每张表用了几根靶、几个比较集。

    用无噪声的旧数据集（data/field_ii/full）里的点靶体模。探头、脉冲、波束形成、频率
    阶梯（4 / 5 / 6.5 / 8 MHz）与带噪正式数据完全相同，点扩散函数与噪声无关，所以在
    新数据的点靶体模生成之前就能用；无噪声还让靶的半高全宽量得更干净。

    同一体模、同一显示深度、同一聚焦、只有频率变化的分片组成一个比较集，与实机的做法
    相同。每个比较集先各自算分数（逐靶归一化），再在同一显示深度下对所有体模、所有聚焦
    取平均。
    """
    import collections
    import re
    from fieldii_loader import find_shards, load_shard, parse_shard_name
    groups = collections.defaultdict(dict)
    for path in find_shards(data_dir):
        meta = parse_shard_name(path)
        if meta["phantom_type"] != "point" or (split and meta["split"] != split):
            continue
        depth = float(re.search(r"_d(\d+)p(\d)", path.name).group(1))
        groups[(meta["seed"], depth, meta["focus_mm"])][meta["frequency_mhz"]] = path
    sums, counts, pins = collections.defaultdict(dict), collections.Counter(), collections.Counter()
    for (seed, depth, focus), by_frequency in sorted(groups.items()):
        if len(by_frequency) < 2:
            continue
        frames = {f: load_shard(p) for f, p in by_frequency.items()}
        table, _ = fieldii_pin_table(frames)
        if not table:
            continue
        for f, s in resolution_scores(table).items():
            sums[depth][f] = sums[depth].get(f, 0.0) + s
        counts[depth] += 1
        pins[depth] += len(table)
    tables = {(0, d): {f: v / counts[d] for f, v in fs.items()} for d, fs in sums.items()}
    return tables, {(0, d): pins[d] for d in pins}, {(0, d): counts[d] for d in counts}
