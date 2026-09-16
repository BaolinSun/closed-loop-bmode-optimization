# -*- coding: utf-8 -*-
"""检查 data/labels_console.jsonl：六根轴的标签在算术、逻辑和物理上是否自洽。

    查三类东西

一、【算术】delta = 最优 - 当前，档数差与方向同号，determined 为假时各字段必须为空。
   这些错了说明代码有 bug，不是判据有问题。

二、【物理】判据是从 E8/E9 测出来的，那么标签必须重现测到的规律，否则就是判据在
   实现中走了样：
     - 频率：显示深度越深，最优频率越低（穿透随频率缩短，E8 实测基波 69.5->41.4 mm）
     - 深度：发射频率越低，最优深度越深（低频穿透更深）
     - 聚焦：显示深度越深，最优聚焦越深（E9 实测逐带最优精确跟随深度）

三、【覆盖】每根轴有多少帧定得出、落在哪些场次，以及后端标签因底噪修正变了多少。

用法：python tests/verify_console_labels.py [标签文件]
      不给参数检查 data/labels_console.jsonl；给 data/labels_fieldii.jsonl 检查 Field II 标签
      （第 4 节只对实机文件有意义，Field II 文件跳过）。
"""

import collections
import io
import json
import os
import sys

import numpy as np

LABELS = "data/labels_console.jsonl"
OLD_LABELS = "data/labels_backend.jsonl"

AXES = [("depth", "depth_mm", ("shallow", "correct", "deep")),
        ("frequency", "frequency_mhz", ("low", "correct", "high")),
        ("focus", "focus_mm", ("shallow", "correct", "deep"))]


def spearman(x, y):
    """秩相关，不引入 scipy。"""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    lines = []
    emit = lines.append
    failures = []
    labels = sys.argv[1] if len(sys.argv) > 1 else LABELS
    rows = [json.loads(l) for l in io.open(labels, encoding="utf-8")]
    emit(u"%d rows in %s" % (len(rows), labels))

    emit(u"")
    emit(u"=========== 1. arithmetic and logic ===========")
    for stem, field, names in AXES:
        bad_delta = bad_dir = bad_empty = bad_ladder = 0
        for r in rows:
            if not r.get("%s_determined" % stem):
                if any(r.get(k) is not None for k in
                       ("optimal_%s" % field, "%s_direction" % stem, "delta_%s" % field)):
                    bad_empty += 1
                continue
            optimal, current = r["optimal_%s" % field], r[field]
            if abs((optimal - current) - r["delta_%s" % field]) > 1e-6:
                bad_delta += 1
            steps = r["delta_%s_steps" % stem]
            expected = names[1] if steps == 0 else (names[0] if steps > 0 else names[2])
            if r["%s_direction" % stem] != expected:
                bad_dir += 1
            ladder = r.get("%s_ladder" % stem) or []
            if optimal not in ladder or current not in ladder:
                bad_ladder += 1
        emit(u"  %-10s delta wrong %d   direction wrong %d   undetermined but filled %d   "
             u"value off its ladder %d" % (stem, bad_delta, bad_dir, bad_empty, bad_ladder))
        if bad_delta or bad_dir or bad_empty or bad_ladder:
            failures.append("%s arithmetic/logic" % stem)

    emit(u"")
    emit(u"=========== 2. do the labels reproduce what E8 and E9 measured ===========")
    emit(u"  One point per comparison set, so a set with many frames does not dominate.")

    def per_set(stem, field, key_fields):
        seen = {}
        for r in rows:
            if not r.get("%s_determined" % stem):
                continue
            key = (r["family_id"],) + tuple(r[k] for k in key_fields)
            seen[key] = r
        return list(seen.values())

    # 频率最优 vs 显示深度，按模式分（两种模式的频率阶梯几乎不重叠）
    for mode in ("fundamental", "harmonic"):
        sets = [r for r in per_set("frequency", "frequency_mhz", ["imaging_mode", "depth_mm", "focus_mm"])
                if r["imaging_mode"] == mode]
        if len(sets) >= 3:
            rho = spearman([r["depth_mm"] for r in sets], [r["optimal_frequency_mhz"] for r in sets])
            emit(u"  frequency optimum vs display depth, %-11s sets %3d   rank corr %+.2f  (expect < 0)"
                 % (mode, len(sets), rho))
            by = collections.defaultdict(list)
            for r in sets:
                by[round(r["depth_mm"], 1)].append(r["optimal_frequency_mhz"])
            emit(u"      " + u"   ".join(u"%g mm -> %s" % (d, sorted(set(v)))
                                         for d, v in sorted(by.items())))
            if np.isfinite(rho) and rho > 0:
                failures.append("frequency optimum rises with depth in %s" % mode)

    # 深度最优 vs 频率。
    #
    # 不直接拿深度最优做相关：深度阶梯一档 8.4 mm，而谐波整条频率阶梯上穿透只差
    # 6.6 mm（E8：46.4 -> 39.8），量化之后本来就几乎持平，秩相关的正负纯属偶然。
    # 真正的物理量是穿透深度，它记在 depth_basis 里，拿它检验。
    #
    # 另外排除在边界上的集：E8 只扫了 67 与 75.4 mm，任何频率都选 67，那是扫描范围
    # 决定的，不是频率决定的，混进来只会稀释。
    import re
    for mode in ("fundamental", "harmonic"):
        sets = [r for r in per_set("depth", "depth_mm", ["imaging_mode", "frequency_mhz", "focus_mm"])
                if r["imaging_mode"] == mode and not r["depth_at_edge"]]
        # 2026-09-14 起深度判据改为「底部余量 >= 6 dB 的最深阶梯」，依据里记的是可用深度。
        #
        # 必须在【同一个族/体模内部】比：深度按 聚焦 -> 频率 -> 深度 求，用的是挑出来的
        # 频率。跨体模混比时，衰减低的体模既被挑到更高频率、可用深度又本来就更深，相关会
        # 变成正的——那是选择效应，不是物理出错（Field II 上混比得到 +0.84，族内比是负的）。
        # 同一个族内体模不变，频率随每行的设置变化，比的才是频率本身。
        def set_frequency(r):
            how = r["depth_conditioned_on"] or ""
            return r["optimal_frequency_mhz"] if how.startswith("optimal frequency") else r["frequency_mhz"]

        def set_focus(r):
            return (r["optimal_focus_mm"] if "optimal focus" in (r["depth_conditioned_on"] or "")
                    else r["focus_mm"])

        # 用所有行，不用上面按 (族, 模式, 当前频率, 聚焦) 去重后的 sets：那个键不含显示
        # 深度，会把同一比较集的六个显示深度压成一行，Field II 上留下的恰好全是 60 mm 那
        # 档，样本退化成「每组只剩一个频率」。这里按条件频率、条件聚焦分组才对得上比较集。
        groups = collections.defaultdict(dict)
        for r in rows:
            if r["imaging_mode"] != mode or not r.get("depth_determined") or r["depth_at_edge"]:
                continue
            if not r["depth_basis"] or not re.search(r"usable to [0-9.]+ mm", r["depth_basis"]):
                continue
            key = (r["family_id"], round(set_focus(r), 1))
            groups[key][set_frequency(r)] = float(
                re.search(r"usable to ([0-9.]+) mm", r["depth_basis"]).group(1))
        rhos, pooled, expected = [], collections.defaultdict(list), []
        for key, by_frequency in groups.items():
            if len(by_frequency) < 2:
                continue
            order = sorted(by_frequency)
            # 低频的可用深度应当不浅于高频。两档的组秩相关只有正负两个值，所以另外报
            # 「最低频不浅于最高频」的比例，它在两档组上也说得通。
            expected.append(by_frequency[order[0]] >= by_frequency[order[-1]] - 0.05)
            if len(order) >= 3:
                rhos.append(spearman(order, [by_frequency[f] for f in order]))
            for f, u in by_frequency.items():
                pooled[f].append(u)
        if expected:
            emit(u"  usable depth vs frequency within a family, %-11s groups %3d   lowest frequency reaches "
                 u"deepest in %d of them%s" % (mode, len(expected), sum(expected),
                                               u"" if not rhos else
                                               u";  median rank corr %+.2f over the %d groups with 3+ frequencies"
                                               % (float(np.median(rhos)), len(rhos))))
            emit(u"      " + u"   ".join(u"%g MHz -> %.1f mm (%d)" % (f, np.mean(v), len(v))
                                         for f, v in sorted(pooled.items())))
            if sum(expected) < 0.5 * len(expected):
                failures.append("usable depth deepens with frequency in %s" % mode)
        if len(sets) >= 3:
            rho = spearman([r["frequency_mhz"] for r in sets], [r["optimal_depth_mm"] for r in sets])
            emit(u"  depth optimum vs frequency (non-edge), %-11s sets %3d   rank corr %+.2f  (informational)"
                 % (mode, len(sets), rho))
            # 谐波这一项在 2026-09-13 读出 +0.25，看着违背物理，其实是聚焦效应漏了进来：
            # 聚焦 15 mm 下谐波所有频率的最优深度都是 50.2 mm（频率影响确实是平的），唯一
            # 的变化来自 5.0 MHz 那几个聚焦不同的集——深聚焦把能量送到深处，穿透从聚焦
            # 10 的约 42 mm 涨到聚焦 30 的约 50 mm，最优深度随之变成 58.6。20260903 与
            # E9_THI 两个独立场次在每个 (频率, 聚焦) 上把穿透复现到 1 mm 以内。

    for mode in ("fundamental", "harmonic"):
        sets = [r for r in per_set("focus", "focus_mm", ["imaging_mode", "depth_mm", "frequency_mhz"])
                if r["imaging_mode"] == mode]
        if len(sets) >= 3:
            rho = spearman([r["depth_mm"] for r in sets], [r["optimal_focus_mm"] for r in sets])
            emit(u"  focus optimum vs display depth,     %-11s sets %3d   rank corr %+.2f  (expect > 0)"
                 % (mode, len(sets), rho))
            by = collections.defaultdict(list)
            for r in sets:
                by[round(r["depth_mm"], 1)].append(r["optimal_focus_mm"])
            emit(u"      " + u"   ".join(u"%g mm -> %s" % (d, sorted(set(v)))
                                         for d, v in sorted(by.items())))
            if np.isfinite(rho) and rho < 0:
                failures.append("focus optimum gets shallower with depth in %s" % mode)

    emit(u"")
    emit(u"=========== 3. coverage by session ===========")
    emit(u"%-28s %6s %6s %6s %6s %6s" % (u"session", u"rows", u"depth", u"freq", u"focus", u"gain"))
    by_session = collections.defaultdict(list)
    for r in rows:
        by_session[r["group_id"].rsplit("/", 1)[0]].append(r)
    for session in sorted(by_session):
        rs = by_session[session]
        emit(u"%-28s %6d %6d %6d %6d %6d"
             % (session, len(rs), sum(r["depth_determined"] for r in rs),
                sum(r["frequency_determined"] for r in rs),
                sum(r["focus_determined"] for r in rs), len(rs)))

    if os.path.exists(OLD_LABELS) and labels == LABELS:
        old = {json.loads(l)["frame_id"]: json.loads(l)
               for l in io.open(OLD_LABELS, encoding="utf-8") if '"console"' in l}
        shared = [r for r in rows if r["frame_id"] in old]
        if shared:
            moved = np.array([r["optimal_gain_db"] - old[r["frame_id"]]["optimal_gain_db"]
                              for r in shared])
            flipped = sum(r["gain_direction"] != old[r["frame_id"]]["gain_direction"]
                          for r in shared)
            emit(u"")
            emit(u"=========== 4. back-end labels versus the previous file ===========")
            emit(u"  %d frames in both. The previous file borrowed noise floors in dB, and"
                 % len(shared))
            emit(u"  tools_refit_calibration marked every group as measured, so up to 5.5 dB")
            emit(u"  of error reached the tissue mask. This is how much that moved the answer.")
            emit(u"  optimal gain change: median %+.2f dB   |change| p90 %.2f dB   max %.2f dB"
                 % (float(np.median(moved)), float(np.percentile(np.abs(moved), 90)),
                    float(np.abs(moved).max())))
            emit(u"  gain direction flipped on %d of %d frames" % (flipped, len(shared)))

    emit(u"")
    emit(u"=========== result ===========")
    if failures:
        for f in failures:
            emit(u"  FAIL  " + f)
    else:
        emit(u"  all checks pass")

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "verify_console_labels.txt" if labels == LABELS else
                       "verify_%s.txt" % os.path.splitext(os.path.basename(labels))[0])
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
