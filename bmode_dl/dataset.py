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

    实机（海信）缓存

tools_build_console_training_cache.py 写出同一格式的缓存，另带 reference_db（本组 pivot）与
底噪两端；实机标签行里没有这些字段，载入时用缓存里的值覆盖。

    标量的刻度（scalar_norm）

fixed  增益 / 10、(reference_db + 45) / 30、(底噪 + 90) / 10：按 Field II 的 dB 刻度写死，
       fieldii_v1 到 v3 都是这样训练的，旧检查点没有 scalar_norm 字段时按它处理。
data   增益按训练集均值方差标准化；reference_db 与底噪用图像 dB 的均值方差标准化（它们与图像
       同一刻度）。实机 dB 刻度与 Field II 差一个任意常数（实机中位约 29 dB，Field II 约 -41 dB），
       写死的换算会把实机标量推到预训练从没见过的范围（reference 2.3-3.2、底噪 8.6-11.3），
       所以微调用 data。
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
SCALAR_NORMS = ("fixed", "data")
GROUP_BY = ("group", "placement")

# 只差成像模式后缀的实机目录是同一次摆放：E8、E9 把基波与谐波存成 _GEN / _THI 两个目录
_MODE_SUFFIXES = ("_GEN", "_THI", "/GEN", "/THI")


def group_key(row, group_by="group"):
    """分折用的组。

    group      标签的 group_id（Field II 是体模；实机是 "场次/成像模式"）
    placement  实机按探头摆放：独立单位是场景族（docs/console_training_data_20260914.md §2.1，
               "同一族共享探头位置，必须整族进同一个集合"），即 family_id。同一次摆放在部分场次里
               本来就跨两种模式（20260903/0 同时有基波 6 帧、谐波 67 帧），但 E8、E9 把基波与谐波
               存成 _GEN / _THI 两个目录、分成了两个族，所以再把只差模式后缀的目录合并：
               20260911_E8_GEN/0 与 20260911_E8_THI/0 -> 20260911_E8/0。按目录分组拦不住这一点，
               会让同一次摆放的图像同时出现在训练与验证里。Field II 行不受影响，仍按体模。
    """
    group = row.get("group_id") or row.get("family_id")
    if group_by != "placement" or row.get("source") != "console":
        return group
    family = row.get("family_id") or group
    if "/" not in family:
        return family
    session, index = family.rsplit("/", 1)
    for suffix in _MODE_SUFFIXES:
        if session.endswith(suffix):
            session = session[:-len(suffix)]
            break
    return "%s/%s" % (session, index)


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

    # 缓存里可以覆盖标签行的字段（实机标签行没有这些，由缓存脚本从标定写入）
    CACHE_OVERRIDES = ("reference_db", "noise_floor_top_db", "noise_floor_bottom_db")

    def __init__(self, cache_dir, labels_path, device="cuda", ladders=None, log=print, group_by="group"):
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
        self.group_ids = [group_key(r, group_by) for r in kept]
        self.group_by = group_by
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
        overridden = []
        for key in self.CACHE_OVERRIDES:
            if key in arrays:
                self.encoded[key] = arrays[key][order].astype(np.float32)
                overridden.append(key)
        if overridden:
            log("  cache overrides label fields: %s" % ", ".join(overridden))
        lacking = [k for k in self.CACHE_OVERRIDES if k not in arrays and not all(k in r for r in kept)]
        if lacking:
            raise RuntimeError("label rows lack %s and the cache does not provide them; build console caches "
                               "with tools_build_console_training_cache.py" % ", ".join(lacking))
        self.source = index.get("source", "fieldii")
        self.t = {k: torch.from_numpy(v).to(self.device) for k, v in self.encoded.items()}
        self.n = len(kept)

    # -------------------------------------------------------------------------------
    def uses_split_field(self):
        """标签自带 val 划分（仿真跑完验证集后即如此），分折参数不再起作用。"""
        return "val" in set(s for s in self.splits if s)

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
        # 最优增益（dB，相对 reference_db）与最优 TGC 曲线（各段 (档位-127)*dB/级）的均值方差，
        # 供 optimum 输出方式标准化；下限避免体模很少时方差过小把输出放大
        backend = tr[self.t["backend_mask"][tr] > 0]
        if len(backend) > 1:
            opt_gain = self.t["optimal_gain_db"][backend]
            slope = torch.where(self.t["mode"][backend] > 0.5,
                                torch.full_like(opt_gain, K.TGC_DB_PER_LEVEL[K.MODE_HARMONIC]),
                                torch.full_like(opt_gain, K.TGC_DB_PER_LEVEL[K.MODE_FUNDAMENTAL]))
            opt_tgc_db = (self.t["optimal_tgc_levels"][backend] - K.TGC_CENTER_LEVEL) * slope[:, None]
            opt = {"opt_gain_mean": float(opt_gain.mean()), "opt_gain_std": float(opt_gain.std().clamp(min=0.1)),
                   "opt_tgc_db_mean": [float(v) for v in opt_tgc_db.mean(dim=0)],
                   "opt_tgc_db_std": [float(v) for v in opt_tgc_db.std(dim=0).clamp(min=0.5)]}
        else:
            opt = {"opt_gain_mean": 0.0, "opt_gain_std": 1.0,
                   "opt_tgc_db_mean": [0.0] * K.NUM_TGC_BANDS, "opt_tgc_db_std": [1.0] * K.NUM_TGC_BANDS}
        out = {"db_mean": db_mean, "db_std": db_std,
               "att_mean": float(att.mean()), "att_std": float(att.std().clamp(min=1e-3)) if len(tr) > 1 else 1.0,
               "noise_mean": float(noise.mean()), "noise_std": float(noise.std().clamp(min=1e-3)) if len(tr) > 1 else 1.0}
        gain = self.t["gain_db"][tr]
        out["gain_mean"] = float(gain.mean())
        out["gain_std"] = float(gain.std().clamp(min=0.5)) if len(tr) > 1 else 1.0
        out.update(opt)
        return out

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

    def physics_coverage(self, train_idx, val_idx):
        """每折的体模物理量覆盖：验证集落在训练集范围之外时，那一折只能外推。

        fieldii_v3 的第 2 折验证集同时拿走了衰减最低与最高的体模，频率命中 0.62、深度 0.47，
        都比另外三折低 0.2 以上，而这件事当时要手工查标签才看得出来。
        """
        per_phantom = {}
        for i, row in enumerate(self.rows):
            per_phantom[self.group_ids[i]] = row
        lines, summary = [], {}
        for key, unit in (("attenuation_db_cm_mhz", "dB/cm/MHz"), ("electronic_noise_db", "dB"),
                          ("sound_speed_mps", "m/s")):
            train_values = [per_phantom[g].get(key) for g in sorted(set(self.group_ids[i] for i in train_idx))]
            val_values = [per_phantom[g].get(key) for g in sorted(set(self.group_ids[i] for i in val_idx))]
            train_values = [float(v) for v in train_values if v is not None]
            val_values = [float(v) for v in val_values if v is not None]
            if not train_values or not val_values:
                continue
            below = min(val_values) < min(train_values)
            above = max(val_values) > max(train_values)
            flag = ("  EXTRAPOLATES " + " AND ".join([w for w, on in (("BELOW", below), ("ABOVE", above)) if on])
                    if (below or above) else "")
            summary[key] = {"train_min": min(train_values), "train_max": max(train_values),
                            "val_min": min(val_values), "val_max": max(val_values),
                            "val_values": val_values, "extrapolates_below": below, "extrapolates_above": above}
            lines.append("    %-22s train %8.3f - %8.3f   val %8.3f - %8.3f  %s%s"
                         % (key + " (" + unit + ")", min(train_values), max(train_values),
                            min(val_values), max(val_values),
                            " ".join("%.3f" % v for v in val_values), flag))
        return lines, summary

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

    def __init__(self, rows, lines, source_rows, norm, use_noise_floor=True, scalar_norm="fixed"):
        super().__init__()
        self.renderer = BackendRenderer(rows, source_rows)
        frac = row_positions(rows, source_rows) / max(1.0, float(source_rows - 1))
        self.register_buffer("row_fraction", torch.tensor(frac, dtype=torch.float32), persistent=False)
        self.norm = dict(norm)
        self.use_noise_floor = bool(use_noise_floor)
        if scalar_norm not in SCALAR_NORMS:
            raise ValueError("scalar_norm must be one of %s" % (SCALAR_NORMS,))
        self.scalar_norm = scalar_norm
        self.lines = int(lines)

    def _level(self, value, fixed_offset, fixed_scale):
        """与图像同一 dB 刻度的标量（曝光参考、底噪）。"""
        if self.scalar_norm == "data":
            return (value - self.norm["db_mean"]) / self.norm["db_std"]
        return (value + fixed_offset) / fixed_scale

    def _gain(self, gain_db):
        if self.scalar_norm == "data":
            return (gain_db - self.norm["gain_mean"]) / self.norm["gain_std"]
        return gain_db / 10.0

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
            floor_top = self._level(noise_floor_top_db, 90.0, 10.0)
            floor_bottom = self._level(noise_floor_bottom_db, 90.0, 10.0)
        else:
            excess = torch.zeros_like(q50)
            floor_top = torch.zeros_like(noise_floor_top_db)
            floor_bottom = torch.zeros_like(noise_floor_bottom_db)
        profile = torch.stack([norm_db(q10), norm_db(q50), norm_db(q90),
                               sorted_gray[:, :, q(0.5)] / K.GRAY_MAX, excess], dim=1)

        scalars = torch.cat([
            (depth_mm / 60.0)[:, None], (frequency_mhz / 8.0)[:, None], (focus_mm / 40.0)[:, None],
            self._gain(gain_db)[:, None], (tgc_levels - K.TGC_CENTER_LEVEL) / float(K.TGC_CENTER_LEVEL),
            (dr_ui / 100.0)[:, None], self._level(reference_db, 45.0, 30.0)[:, None], mode[:, None],
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
