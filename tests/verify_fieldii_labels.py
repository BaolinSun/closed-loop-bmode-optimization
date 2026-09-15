# -*- coding: utf-8 -*-
"""Field II 带噪标签（data/labels_fieldii.jsonl）特有的检查。

算术与逻辑一致性、物理规律、覆盖由 tests/verify_console_labels.py data/labels_fieldii.jsonl
检查；闭环链由 tests/verify_frontend_chain.py data/labels_fieldii.jsonl 检查。这里只查三件
Field II 数据才能回答、也最关心的事：

  一、【标签依赖图像吗】实机只有一个体模，频率方向不看图就能猜中 95%，模型学不会看图。
     Field II 体模的衰减、噪声、声速都随机，同样的设置下标签应该随体模不同。量法与实机
     相同：只用 (显示深度, 当前频率, 当前聚焦) 按多数投票猜方向，看能猜中多少。

  二、【可用深度随体模的物理量变化吗】同一 (频率, 聚焦) 下，衰减越大、噪声越高，可用
     深度应越浅。用深度判据依据里记下的可用深度，对衰减、噪声电平各求秩相关。

  三、【组织掩膜排除噪声改变了多少】从标签里抽 8 MHz 与 6.5 MHz、60 mm 的分片，分别用
     排除噪声与不排除噪声的掩膜解后端，报告最优增益差。2026-09-13 在试跑数据上量到最多
     8.6 dB；这里确认修复在正式数据上的作用量。

用法：python tests/verify_fieldii_labels.py
"""

import collections
import io
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
os.chdir(ROOT)

import numpy as np

LABELS = "data/labels_fieldii.jsonl"
MASK_SAMPLE = 24
# 两种掩膜用同一个目标灰阶和同一个曝光参考，只比掩膜本身的作用。取实机基波各组的量级。
TARGET_GRAY = 115.0


def spearman(x, y):
    """平均秩的秩相关（有并列时不偏）。"""
    def ranks(v):
        v = np.asarray(v, dtype=np.float64)
        order = np.argsort(v, kind="mergesort")
        r = np.empty(v.size)
        r[order] = np.arange(v.size)
        for value in np.unique(v):
            same = v == value
            r[same] = r[same].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def image_free_accuracy(rows, stem, keys):
    determined = [r for r in rows if r.get("%s_determined" % stem)]
    votes = collections.defaultdict(collections.Counter)
    for r in determined:
        votes[tuple(r[k] for k in keys)][r["%s_direction" % stem]] += 1
    hit = sum(c.most_common(1)[0][1] for c in votes.values())
    return hit, len(determined)


def main():
    lines = []
    emit = lines.append
    failures = []
    rows = [json.loads(l) for l in io.open(LABELS, encoding="utf-8")]
    phantoms = sorted({r["group_id"] for r in rows})
    emit(u"%d rows, %d phantoms in %s" % (len(rows), len(phantoms), LABELS))

    emit(u"")
    emit(u"=========== 1. can the labels be guessed without looking at the image ===========")
    emit(u"  Majority vote over (display depth, current frequency, current focus) alone.")
    emit(u"  Console reference (one phantom): frequency 233/246 = 95%.")
    settings = ["depth_mm", "frequency_mhz", "focus_mm"]
    for stem in ("frequency", "depth", "focus"):
        hit, total = image_free_accuracy(rows, stem, settings)
        emit(u"  %-10s %4d / %4d = %5.1f%%" % (stem, hit, total, 100.0 * hit / max(total, 1)))
    backend = [r for r in rows if r.get("backend_determined")]
    votes = collections.defaultdict(collections.Counter)
    for r in backend:
        votes[tuple(r[k] for k in settings)][r["gain_direction"]] += 1
    hit = sum(c.most_common(1)[0][1] for c in votes.values())
    emit(u"  %-10s %4d / %4d = %5.1f%%  (the start is drawn at random, so near chance is expected)"
         % ("gain", hit, len(backend), 100.0 * hit / max(len(backend), 1)))
    # 同一设置下，最优值在体模之间变不变：这比方向更直接。
    for stem, key in (("frequency", "optimal_frequency_mhz"), ("depth", "optimal_depth_mm"),
                      ("focus", "optimal_focus_mm")):
        spread = collections.defaultdict(set)
        for r in rows:
            if r.get("%s_determined" % stem):
                spread[tuple(r[k] for k in settings)].add(r[key])
        varying = sum(len(v) > 1 for v in spread.values())
        emit(u"  %-10s settings whose optimum differs between phantoms: %d / %d"
             % (stem, varying, len(spread)))

    emit(u"")
    emit(u"=========== 2. usable depth against each phantom's attenuation and noise ===========")
    usable = collections.defaultdict(list)
    for r in rows:
        basis = r.get("depth_basis") or ""
        m = re.search(r"usable to ([0-9.]+) mm", basis)
        if not m:
            continue
        # 深度按 聚焦 -> 频率 -> 深度 的顺序求，依据来自哪个比较集要看 depth_conditioned_on。
        how = r.get("depth_conditioned_on") or ""
        f = r["optimal_frequency_mhz"] if how.startswith("optimal frequency") else r["frequency_mhz"]
        z = r["optimal_focus_mm"] if "optimal focus" in how else r["focus_mm"]
        usable[(f, z)].append(
            (r["group_id"], float(m.group(1)), r["attenuation_db_cm_mhz"], r["electronic_noise_db"]))
    for (f, z) in sorted(usable):
        per_phantom = {}
        for g, u, a, n in usable[(f, z)]:
            per_phantom[g] = (u, a, n)
        if len(per_phantom) < 5:
            continue
        u, a, n = zip(*per_phantom.values())
        rho_a, rho_n = spearman(a, u), spearman(n, u)
        emit(u"  %.1f MHz focus %4.1f mm  phantoms %2d  usable %4.1f-%4.1f mm  rank corr with attenuation %+.2f, with noise %+.2f"
             % (f, z, len(per_phantom), min(u), max(u), rho_a, rho_n))
    all_u = [(v[1], v[2], v[3]) for vs in usable.values() for v in vs]
    if all_u:
        u, a, n = zip(*all_u)
        emit(u"  pooled over settings: rank corr usable depth vs attenuation %+.2f, vs noise %+.2f"
             % (spearman(a, u), spearman(n, u)))

    emit(u"")
    emit(u"=========== 3. what excluding noise from the tissue mask changes ===========")
    import fieldii_loader as FL
    import fieldii_noise as FN
    import labels as LB
    import tissue as T
    rng = np.random.RandomState(0)
    candidates = [r for r in backend if r["depth_mm"] == 60.0 and r["frequency_mhz"] in (6.5, 8.0)]
    sample = [candidates[i] for i in rng.choice(len(candidates), min(MASK_SAMPLE, len(candidates)), replace=False)]
    diffs = []
    for r in sample:
        shard = FL.load_shard(FL.Path(r["data_dir"]) / "hdf5" / (r["frame_id"] + ".h5"))
        with_noise = T.fieldii_tissue_mask(shard)
        without_noise = T.fieldii_tissue_mask(shard, floor_db=FN.noise_floor_db(shard))
        solved = []
        for mask in (with_noise, without_noise):
            solved.append(LB.label_frame(
                shard.db_image, mask, dr_ui=shard.dynamic_range_level,
                reference_db=r["reference_db"], current=None, rng=np.random.RandomState(1),
                target_gray=TARGET_GRAY, source="fieldii", frame_id=r["frame_id"],
                group_id=r["group_id"], imaging_mode="fundamental", depth_mm=60.0))
        diffs.append((r["frequency_mhz"], solved[0].optimal_gain_db - solved[1].optimal_gain_db,
                      float(with_noise.mean()), float(without_noise.mean())))
    for f in (6.5, 8.0):
        d = [x for x in diffs if x[0] == f]
        if d:
            g = np.array([x[1] for x in d])
            emit(u"  %.1f MHz, 60 mm, %2d shards: optimal gain (noise counted as tissue) - (noise excluded)"
                 u"  median %+.2f dB, max |.| %.2f dB; tissue fraction %.2f -> %.2f"
                 % (f, len(d), float(np.median(g)), float(np.abs(g).max()),
                    float(np.mean([x[2] for x in d])), float(np.mean([x[3] for x in d]))))

    emit(u"")
    emit(u"=========== result ===========")
    hit, total = image_free_accuracy(rows, "frequency", settings)
    if total and hit / total > 0.9:
        failures.append("frequency labels are still guessable without the image (%.0f%%)" % (100.0 * hit / total))
    for f in failures:
        emit(u"  FAIL  " + f)
    if not failures:
        emit(u"  all checks pass")
    text = u"\n".join(lines)
    io.open(os.path.join(ROOT, "tests", "verify_fieldii_labels.txt"), "w", encoding="utf-8").write(text)
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
