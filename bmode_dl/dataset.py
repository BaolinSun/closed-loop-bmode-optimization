# -*- coding: utf-8 -*-
"""缓存与标签载入、按体模分折、组批、网络输入特征。

整个缓存（1848 x 512 x 128 float32 约 380 MB）直接放在 GPU 上，批次按下标切片，数据增强
（后端起点重抽、左右翻转）与渲染都在 GPU 上做，不用 DataLoader。

    网络输入（build_inputs）

    image    (B, 3, H, W)  当前增益/TGC/动态范围渲染的灰阶 / 255；绝对 dB 标准化；物理深度 / 60 mm
    profile  (B, 5, H)     逐行 dB 的 10/50/90 分位（标准化）；逐行灰阶中位数 / 255；
                           逐行 dB 中位数减底噪 / 20（关闭底噪输入时为 0）
    scalars  (B, 17)       SCALAR_NAMES

底噪是实机上接收机的已知常数的对应物（实机按场次一个常数），所以可以作输入；衰减、电子噪声、
声速是仿真真值，只作辅助监督目标，不作输入。
"""

import io
import json
import os
from collections import Counter, OrderedDict

import numpy as np
import torch

from . import constants as K
from . import labels as L
from .render import BackendRenderer, row_positions

SCALAR_NAMES = (["depth_mm/60", "frequency_mhz/8", "focus_mm/40", "gain_db/10"]
                + ["tgc%d (level-127)/127" % i for i in range(K.NUM_TGC_BANDS)]
                + ["dr_ui/100", "(reference_db+45)/30", "imaging_mode",
                   "(noise_floor_top_db+90)/10", "(noise_floor_bottom_db+90)/10"])
NUM_SCALARS = len(SCALAR_NAMES)
PROFILE_CHANNELS = 5
IMAGE_CHANNELS = 3
INPUT_MODES = ("full", "no_image", "params_only")


def read_jsonl(path):
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_cache(cache_dir):
    """读 tools_build_fieldii_training_cache.py 的产物。"""
    index = json.load(io.open(os.path.join(cache_dir, "index.json"), encoding="utf-8"))
    arrays = np.load(os.path.join(cache_dir, "cache.npz"), allow_pickle=False)
    return index, {k: arrays[k] for k in arrays.files}


# ---------------------------------------------------------------------------------------
#   分折

def make_group_folds(group_ids, phantom_types, num_folds, seed=0):
    """按体模分折：每种体模类型内部洗牌后轮流分到各折，保证每折类型分布接近。

    返回 [折0的体模列表, 折1的体模列表, ...]。
    """
    by_type = OrderedDict()
    for g, t in sorted(set(zip(group_ids, phantom_types))):
        by_type.setdefault(t, []).append(g)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(num_folds)]
    offset = 0
    for t in sorted(by_type):
        groups = list(by_type[t])
        rng.shuffle(groups)
        for i, g in enumerate(groups):
            folds[(i + offset) % num_folds].append(g)
        offset += len(groups)
    return [sorted(f) for f in folds]


class FieldIIData(object):
    """缓存 + 标签，全部在 device 上。"""

    def __init__(self, cache_dir, labels_path, device="cuda", ladders=None, log=print):
        self.device = torch.device(device)
        index, arrays = load_cache(cache_dir)
        self.index = index
        rows = read_jsonl(labels_path)
        frame_to_cache = {str(f): i for i, f in enumerate(arrays["frame_id"])}
        kept = [r for r in rows if r["frame_id"] in frame_to_cache]
        missing = len(rows) - len(kept)
        if missing:
            log("  WARNING: %d label rows have no cache entry and are skipped" % missing)
        if not kept:
            raise RuntimeError("no label row matches the cache; rebuild the cache for this label file")
        self.rows = kept
        order = np.array([frame_to_cache[r["frame_id"]] for r in kept], dtype=np.int64)

        self.ladders = ladders if ladders is not None else L.collect_ladders(kept)
        self.encoded = L.encode_rows(kept, self.ladders)
        self.group_ids = [r.get("group_id") or r.get("family_id") for r in kept]
        self.phantom_types = [r.get("phantom_type", "unknown") for r in kept]
        self.splits = [r.get("split") for r in kept]
        self.frame_ids = [r["frame_id"] for r in kept]

        self.source_rows = int(index["source_rows"])
        self.rows_out = int(index["rows"])
        self.lines = int(index["lines"])
        self.db = torch.from_numpy(np.ascontiguousarray(arrays["db"][order], dtype=np.float32)).to(self.device)
        self.floor = torch.from_numpy(np.ascontiguousarray(arrays["floor"][order], dtype=np.float32)).to(self.device)
        self.min_depth_mm = torch.from_numpy(arrays["min_depth_mm"][order].astype(np.float32)).to(self.device)
        self.max_depth_mm = torch.from_numpy(arrays["max_depth_mm"][order].astype(np.float32)).to(self.device)
        self.t = {k: torch.from_numpy(v).to(self.device) for k, v in self.encoded.items()}
        self.n = len(kept)

    # -------------------------------------------------------------------------------
    def split_indices(self, num_folds=4, fold=0, seed=0):
        """返回 (train_idx, val_idx, val_groups)。

        标签里有 val/test split 时按 split 字段（train -> 训练，val -> 验证，test 不参与）；
        否则按体模分折。fold=None 表示全部用于训练、无验证集。
        """
        idx = np.arange(self.n)
        split_names = set(s for s in self.splits if s)
        if fold is None:
            return idx, np.zeros(0, np.int64), []
        if "val" in split_names:
            train = np.array([i for i in idx if self.splits[i] == "train"], np.int64)
            val = np.array([i for i in idx if self.splits[i] == "val"], np.int64)
            return train, val, sorted(set(self.group_ids[i] for i in val))
        folds = make_group_folds(self.group_ids, self.phantom_types, num_folds, seed)
        val_groups = set(folds[int(fold)])
        train = np.array([i for i in idx if self.group_ids[i] not in val_groups], np.int64)
        val = np.array([i for i in idx if self.group_ids[i] in val_groups], np.int64)
        return train, val, sorted(val_groups)

    def indices_for_groups(self, groups):
        groups = set(groups)
        return np.array([i for i in range(self.n) if self.group_ids[i] in groups], np.int64)

    # -------------------------------------------------------------------------------
    def normalisation(self, train_idx, max_frames=256, seed=0):
        """训练集上的标准化统计：dB 通道均值方差、辅助目标均值方差。"""
        rng = np.random.RandomState(seed)
        pick = train_idx if len(train_idx) <= max_frames else rng.choice(train_idx, max_frames, replace=False)
        sample = self.db[torch.as_tensor(pick, device=self.device)]
        db_mean = float(sample.mean())
        db_std = float(sample.std().clamp(min=1e-3))
        tr = torch.as_tensor(train_idx, device=self.device)
        att = self.t["attenuation_db_cm_mhz"][tr]
        noise = self.t["electronic_noise_db"][tr]
        return {"db_mean": db_mean, "db_std": db_std,
                "att_mean": float(att.mean()), "att_std": float(att.std().clamp(min=1e-3)) if len(tr) > 1 else 1.0,
                "noise_mean": float(noise.mean()), "noise_std": float(noise.std().clamp(min=1e-3)) if len(tr) > 1 else 1.0}

    def class_weights(self, train_idx, redraws=16, seed=0, power=0.5):
        """方向类别权重 = (频数的倒数)^power，按训练集归一化；未出现的类别权重 0。

        后端方向按重抽起点的分布统计（训练时看到的正是这个分布）。
        """
        tr = torch.as_tensor(train_idx, device=self.device)
        gen = torch.Generator().manual_seed(seed)
        gain_counts = np.zeros(3)
        slider_counts = np.zeros(3)
        backend = self.t["backend_mask"][tr] > 0
        for _ in range(redraws):
            g, lv = L.draw_start(self.t["optimal_gain_db"][tr], self.t["optimal_tgc_levels"][tr],
                                 self.t["mode"][tr], gen)
            tg = L.backend_targets(self.t["optimal_gain_db"][tr], self.t["optimal_tgc_levels"][tr], g, lv,
                                   self.t["deadband_gain_levels"][tr], self.t["mode"][tr])
            gain_counts += np.bincount(tg["gain_dir"][backend].cpu().numpy(), minlength=3)[:3]
            slider_counts += np.bincount(tg["slider_dir"][backend].reshape(-1).cpu().numpy(), minlength=3)[:3]

        def weights(counts):
            counts = np.asarray(counts, np.float64)
            w = np.where(counts > 0, (counts.sum() / np.maximum(counts, 1.0)) ** power, 0.0)
            present = w > 0
            if present.any():
                w[present] = w[present] / w[present].mean()
            return w.astype(np.float32)

        out = {"gain_dir": weights(gain_counts), "slider_dir": weights(slider_counts)}
        for key, mask in (("depth_dir", "depth_mask"), ("frequency_dir", "frequency_mask"),
                          ("focus_dir", "focus_mask"), ("dr_dir", "dr_mask")):
            m = self.t[mask][tr] > 0
            labels = self.t[key][tr][m].cpu().numpy()
            out[key] = weights(np.bincount(labels, minlength=3)[:3] if labels.size else np.zeros(3))
        return out

    def label_summary(self, idx):
        """ASCII 的标签分布摘要。"""
        lines = []
        rows = [self.rows[i] for i in idx]
        for key in ("gain_direction", "depth_direction", "frequency_direction", "focus_direction"):
            lines.append("    %-20s %s" % (key, dict(Counter(r.get(key) for r in rows))))
        return lines


# ---------------------------------------------------------------------------------------
#   批次

class InputBuilder(torch.nn.Module):
    """从缓存切片 + 当前后端设置构造网络输入。无可学习参数。"""

    def __init__(self, rows, lines, source_rows, norm, use_noise_floor=True):
        super().__init__()
        self.renderer = BackendRenderer(rows, source_rows)
        frac = row_positions(rows, source_rows) / max(1.0, float(source_rows - 1))
        self.register_buffer("row_fraction", torch.tensor(frac, dtype=torch.float32), persistent=False)
        self.norm = dict(norm)
        self.use_noise_floor = bool(use_noise_floor)
        self.lines = int(lines)

    @torch.no_grad()
    def forward(self, db, floor, min_depth_mm, max_depth_mm, depth_mm, frequency_mhz, focus_mm,
                gain_db, tgc_levels, dr_ui, reference_db, mode, noise_floor_top_db, noise_floor_bottom_db):
        b, h, w = db.shape
        db = db.float()
        gray = self.renderer(db, gain_db, tgc_levels, reference_db, dr_ui, mode)
        db_n = (db - self.norm["db_mean"]) / self.norm["db_std"]
        z = min_depth_mm[:, None] + self.row_fraction[None, :] * (max_depth_mm - min_depth_mm)[:, None]
        z_img = (z / 60.0)[:, :, None].expand(b, h, w)
        image = torch.stack([gray / K.GRAY_MAX, db_n, z_img], dim=1)

        sorted_db, _ = torch.sort(db, dim=2)
        sorted_gray, _ = torch.sort(gray, dim=2)
        q = lambda p: int(round(p * (w - 1)))
        q10, q50, q90 = sorted_db[:, :, q(0.1)], sorted_db[:, :, q(0.5)], sorted_db[:, :, q(0.9)]
        norm_db = lambda x: (x - self.norm["db_mean"]) / self.norm["db_std"]
        if self.use_noise_floor:
            excess = (q50 - floor.float()) / 20.0
            floor_top = (noise_floor_top_db + 90.0) / 10.0
            floor_bottom = (noise_floor_bottom_db + 90.0) / 10.0
        else:
            excess = torch.zeros_like(q50)
            floor_top = torch.zeros_like(noise_floor_top_db)
            floor_bottom = torch.zeros_like(noise_floor_bottom_db)
        profile = torch.stack([norm_db(q10), norm_db(q50), norm_db(q90),
                               sorted_gray[:, :, q(0.5)] / K.GRAY_MAX, excess], dim=1)

        scalars = torch.cat([
            (depth_mm / 60.0)[:, None], (frequency_mhz / 8.0)[:, None], (focus_mm / 40.0)[:, None],
            (gain_db / 10.0)[:, None], (tgc_levels - K.TGC_CENTER_LEVEL) / float(K.TGC_CENTER_LEVEL),
            (dr_ui / 100.0)[:, None], ((reference_db + 45.0) / 30.0)[:, None], mode[:, None],
            floor_top[:, None], floor_bottom[:, None]], dim=1)
        return {"image": image, "profile": profile, "scalars": scalars}


TARGET_KEYS = ("mode", "depth_idx", "frequency_idx", "focus_idx", "optimal_gain_db", "optimal_tgc_levels",
               "deadband_gain_levels", "backend_mask", "delta_dr_ui", "dr_dir", "dr_mask",
               "optimal_depth_idx", "optimal_focus_idx", "optimal_frequency_idx", "frequency_acceptable",
               "depth_valid", "frequency_valid", "focus_valid", "depth_dir", "frequency_dir", "focus_dir",
               "depth_mask", "frequency_mask", "focus_mask", "frequency_weight", "frequency_borderline",
               "attenuation_db_cm_mhz", "electronic_noise_db", "aux_mask")


def make_batch(data, builder, idx, start="label", generator=None, flip=False, state=None):
    """一个批次的 (inputs, targets)。

    start   "label"  用 jsonl 记录的起点
            "redraw" 重抽后端起点并重算后端标签（训练增强、固定种子的验证）
    state   闭环仿真时直接给定当前后端设置 {"gain_db", "tgc_levels"}，覆盖 start
    flip    左右翻转（所有标签不变）
    """
    idx_t = torch.as_tensor(idx, device=data.device, dtype=torch.long)
    t = {k: v[idx_t] for k, v in data.t.items()}
    db = data.db[idx_t]
    if flip:
        db = torch.flip(db, dims=[2])

    if state is not None:
        gain_db, tgc_levels = state["gain_db"].float(), state["tgc_levels"].float()
    elif start == "redraw":
        gain_db, tgc_levels = L.draw_start(t["optimal_gain_db"], t["optimal_tgc_levels"], t["mode"], generator)
    else:
        gain_db, tgc_levels = t["gain_db"], t["tgc_levels"]

    backend = L.backend_targets(t["optimal_gain_db"], t["optimal_tgc_levels"], gain_db, tgc_levels,
                                t["deadband_gain_levels"], t["mode"])
    inputs = builder(db, data.floor[idx_t], data.min_depth_mm[idx_t], data.max_depth_mm[idx_t],
                     t["depth_mm"], t["frequency_mhz"], t["focus_mm"], gain_db, tgc_levels, t["dr_ui"],
                     t["reference_db"], t["mode"], t["noise_floor_top_db"], t["noise_floor_bottom_db"])
    targets = {k: t[k] for k in TARGET_KEYS}
    targets.update(backend)
    targets["gain_db"] = gain_db
    targets["tgc_levels"] = tgc_levels
    targets["tgc_db_per_level"] = torch.where(t["mode"] > 0.5,
                                              torch.full_like(t["mode"], K.TGC_DB_PER_LEVEL[K.MODE_HARMONIC]),
                                              torch.full_like(t["mode"], K.TGC_DB_PER_LEVEL[K.MODE_FUNDAMENTAL]))
    # 没有定出的后端行，方向下标可能是 -1 以外的数；由掩膜排除
    return inputs, targets
