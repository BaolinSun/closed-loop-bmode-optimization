# -*- coding: utf-8 -*-
"""前端标签的前提：把实机帧切成「同一场景」的族。这个脚本量的是切分该用什么判据。

    为什么需要族

后端的 θ_optimal 是算出来的——BC0 对增益/TGC/动态范围不变，一帧就够。前端三个
参数改变 BC0 本身，8 MHz 的采集里没有信息能算出 5 MHz 长什么样，所以最优只能靠
比较同一场景下真实采到的多帧得到。族 = 同一探头位置下的一组帧。

    族不等于场次

20260903 的 15 帧锚点（同为 41.9 mm / 5.00 MHz / 焦点 15 mm）互相的 BC0 相关是
0.98-1.00，但最后一帧对全部只有 0.69——一个场次里含多个探头位置。

    这个脚本回答两个问题

一、【同设置】重复帧的相关有多高，探头动过之后掉到多少。这给出阈值。

二、【跨设置】同一位置、不同频率/聚焦的两帧还相关吗。这决定算法：
    - 若跨设置仍高，可以逐帧按时间顺序切，判据统一；
    - 若跨设置就掉下来，只能拿锚点帧（重复出现的那个设置）当路标切，
      其余帧按时间归入所在区间。
   判断办法：取时间上夹在两帧互相 0.95 以上的锚点之间的扫描帧——几秒之内探头不会
   动开又动回来——再看它对这两个锚点的相关。

深度不同的两帧 BC0 行数和 mm/点都不同，一律重采样到公共物理深度轴再比。

用法：python tests/measure_scene_families.py
"""

import io
import itertools
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import calibration as CAL
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

# 相关只在这段物理深度上算。近场有固定的探头结构，最深处是噪声，两头都不带场景信息。
CORR_DEPTH_MM = (8.0, 30.0)
COMMON_ROWS = 256

# 判定「探头没动」的锚点门槛。两个锚点互相高于它，才认为其间的扫描帧场景未变。
STATIC_ANCHOR_CORR = 0.95

SESSIONS = ["20260814", "20260819", "20260828/GEN", "20260828/THI", "20260831",
            "20260901", "20260901_E2", "20260901_E3", "20260903", "20260903_GEN",
            "20260903_replication_check", "20260904", "20260904_DR",
            "20260909_GEN", "20260910"]

MODE_NAMES = {0: "fundamental", 1: "harmonic"}


def setting(capture):
    """一帧的前端设置：成像模式、显示深度、发射频率、发射聚焦。"""
    return (S.capture_image_mode(capture), round(capture.geometry.depth_mm, 1),
            round(CAL.capture_frequency(capture), 2), capture.focus_mm)


def profile(capture):
    """重采样到公共物理深度轴上的 BC0（dB），跨深度设置也能比。"""
    db = S.bc0_to_db(capture.bc0, 877.3)
    geometry = capture.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
    if depth[-1] < CORR_DEPTH_MM[1]:
        return None
    want = np.linspace(CORR_DEPTH_MM[0], CORR_DEPTH_MM[1], COMMON_ROWS)
    out = np.empty((COMMON_ROWS, db.shape[1]))
    for line in range(db.shape[1]):
        out[:, line] = np.interp(want, depth, db[:, line])
    return out


def correlate(a, b):
    """两帧在公共深度轴上的相关。线数不同则截到较少的那个。"""
    lines = min(a.shape[1], b.shape[1])
    return float(np.corrcoef(a[:, :lines].ravel(), b[:, :lines].ravel())[0, 1])


def describe_difference(a, b):
    """两个设置差在哪几根轴上。"""
    axes = []
    if a[0] != b[0]:
        axes.append(u"mode")
    for name, index in [(u"depth", 1), (u"frequency", 2), (u"focus", 3)]:
        if a[index] != b[index]:
            axes.append(u"%s %g->%g" % (name, b[index], a[index]))
    return axes


def question_one(loaded, emit):
    """同设置重复帧的相关分布。"""
    emit(u"=========== question 1: same setting, repeated frames ===========")
    emit(u"  How high does the correlation sit when the probe held still, and how far")
    emit(u"  does it drop once it moved. This is where the threshold comes from.")
    emit(u"")
    emit(u"%-28s %-11s %6s %6s %6s %4s %8s %8s"
         % (u"session", u"mode", u"depth", u"freq", u"focus", u"n",
            u"min corr", u"max corr"))
    values = []
    for session, entries in loaded.items():
        groups = {}
        for capture, prof in entries:
            if prof is not None:
                groups.setdefault(setting(capture), []).append(prof)
        for key, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            pairs = [correlate(a, b) for a, b in itertools.combinations(members, 2)]
            values.extend(pairs)
            emit(u"%-28s %-11s %6.1f %6.2f %6.1f %4d %8.4f %8.4f"
                 % (session, MODE_NAMES[key[0]], key[1], key[2], key[3],
                    len(members), min(pairs), max(pairs)))

    values = np.array(values)
    emit(u"")
    emit(u"  %d pairs at an identical setting" % values.size)
    emit(u"  histogram")
    counts, edges = np.histogram(values, bins=np.arange(0.0, 1.05, 0.05))
    for k in range(counts.size):
        if counts[k]:
            emit(u"    %.2f - %.2f  %5d  %s"
                 % (edges[k], edges[k + 1], counts[k], u"#" * min(60, counts[k])))
    return values


def question_two(loaded, emit):
    """同一位置、不同前端设置的相关，按变动的轴分组。"""
    emit(u"")
    emit(u"=========== question 2: same placement, different setting ===========")
    emit(u"  A sweep frame whose timestamp falls between two anchor frames that")
    emit(u"  correlate above %.2f with each other. The probe cannot have moved away and"
         % STATIC_ANCHOR_CORR)
    emit(u"  back in those seconds, so whatever correlation this frame shows is the cost")
    emit(u"  of turning the knob, not of moving the probe.")
    emit(u"")
    emit(u"%-24s %-20s %-30s %8s %8s"
         % (u"session", u"frame", u"differs in", u"vs prev", u"vs next"))
    by_axis = {}
    for session, entries in loaded.items():
        usable = [(c, p) for c, p in entries if p is not None]
        if not usable:
            continue
        counts = {}
        for capture, _ in usable:
            counts[setting(capture)] = counts.get(setting(capture), 0) + 1
        anchor = max(counts, key=lambda k: counts[k])
        if counts[anchor] < 3:
            continue
        marks = [i for i, (c, _) in enumerate(usable) if setting(c) == anchor]
        for index, (capture, prof) in enumerate(usable):
            current = setting(capture)
            if current == anchor:
                continue
            before = [m for m in marks if m < index]
            after = [m for m in marks if m > index]
            if not before or not after:
                continue
            low, high = before[-1], after[0]
            if correlate(usable[low][1], usable[high][1]) < STATIC_ANCHOR_CORR:
                continue                  # 这段区间内探头动过，不能当作静止
            axes = describe_difference(current, anchor)
            if not axes:
                continue
            previous = correlate(prof, usable[low][1])
            following = correlate(prof, usable[high][1])
            by_axis.setdefault(axes[0].split()[0], []).append(min(previous, following))
            emit(u"%-24s %-20s %-30s %8.4f %8.4f"
                 % (session, capture.name[:20], u", ".join(axes)[:30],
                    previous, following))

    emit(u"")
    emit(u"  worst of the two correlations, grouped by which axis moved")
    emit(u"%-14s %8s %8s %8s %8s" % (u"axis", u"n", u"min", u"median", u"max"))
    for axis in sorted(by_axis):
        v = np.array(by_axis[axis])
        emit(u"%-14s %8d %8.4f %8.4f %8.4f"
             % (axis, v.size, v.min(), float(np.median(v)), v.max()))
    return by_axis


def verdict(same_setting, by_axis, emit):
    """两个分布重不重叠，决定切分算法。"""
    emit(u"")
    emit(u"=========== what this means for the segmentation rule ===========")
    if not by_axis:
        emit(u"  No usable same-placement pair, cannot decide.")
        return
    worst = min(float(np.min(v)) for v in by_axis.values())
    moved = same_setting[same_setting < 0.85]
    emit(u"  lowest same-placement, different-setting correlation : %.4f" % worst)
    if not moved.size:
        emit(u"  No same-setting pair fell below 0.85, so this run has no moved-probe")
        emit(u"  example to compare against.")
        return
    emit(u"  highest correlation seen after the probe moved       : %.4f" % moved.max())
    if worst > moved.max():
        emit(u"  The two do not overlap. One threshold separates them, so every frame can")
        emit(u"  be segmented against its time neighbour whatever its setting.")
        emit(u"  Suggested threshold: %.2f" % round((worst + moved.max()) / 2, 2))
    else:
        emit(u"  The two overlap: turning a knob can look exactly like moving the probe.")
        emit(u"  Segment on anchor frames only and assign the rest by timestamp.")


def main():
    lines = []
    emit = lines.append

    loaded = {}
    for session in SESSIONS:
        root = DEFAULT_DATA_DIR / session
        if not root.exists():
            continue
        captures = sorted((load_capture(p) for p in find_captures(root)),
                          key=lambda c: c.name)
        loaded[session] = [(c, profile(c)) for c in captures]

    same_setting = question_one(loaded, emit)
    by_axis = question_two(loaded, emit)
    verdict(same_setting, by_axis, emit)

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_scene_families.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
