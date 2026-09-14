# -*- coding: utf-8 -*-
"""频率标签有多靠近判据的门槛：显示深度 41.9 mm 的比较集逐频率看穿透余量。

同一个体模、同一显示深度、同一聚焦，三个基波场景族给出了 8.0 / 10 / 11.4 三个不同的
最优频率。体模衰减不会变，所以怀疑这是穿透读数贴着「覆盖显示深度」的门槛抖动，而
不是物理差异。这里对每帧报告：
  - penetration   标签工具用的穿透深度（行中位数高出底噪 3 dB 的最深处）
  - bottom excess 图像最底 2 mm 的行中位数比底噪高多少 dB——穿透刚好卡在显示深度时，
                  这个数说明离门槛有多远
并把余量从 3 dB 改到 1、6 dB，容差从 1 mm 改到 0、3 mm，看最优频率怎么变。

用法：python tests/measure_frequency_label_margin.py
"""
import collections
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
os.chdir(ROOT)

import numpy as np

import tools_generate_console_labels as G
import hisense_backend_sim as S
import scene_family as SF
import tissue as T
import tools_generate_labels as TG
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

DISPLAY_MM = 41.9
FOCUS_MM = 15.0
BOTTOM_MM = 2.0


def main():
    lines = []
    emit = lines.append
    G.CONSOLE_LADDERS.update(G.console_ladders())
    cal_by_group = TG.load_calibration()
    G.floors_in_counts(cal_by_group)
    for key in sorted(cal_by_group):
        session, mode = key
        entry = cal_by_group[key]
        if entry["floor"] is None:
            continue
        caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        caps = sorted((c for c in caps if S.capture_image_mode(c) == mode), key=lambda c: c.name)
        if not any(abs(c.geometry.depth_mm - DISPLAY_MM) < 0.1 for c in caps):
            continue
        by_name = {c.name: c for c in caps}
        for family in SF.segment_session(caps, session=session):
            sets = G.comparison_sets(family, "frequency_mhz")
            for setting_key, members in sets.items():
                depth, focus = setting_key[1], setting_key[2]
                if abs(depth - DISPLAY_MM) > 0.1:
                    continue
                emit(u"--- %s %s family %s  display %.1f mm  focus %g mm  floor %.2f dB ---"
                     % (session, T.IMAGE_MODE_NAMES[mode], family.family_id,
                        depth, focus, entry["floor"]))
                per_freq = collections.defaultdict(list)
                for name, setting in members:
                    cap = by_name[name]
                    db = S.bc0_to_db(cap.bc0, entry["cal"].counts_per_db)
                    rows = np.median(db, axis=1)
                    n = int(round(BOTTOM_MM / cap.geometry.mm_per_point))
                    per_freq[setting[2]].append((rows, cap.geometry, float(np.median(rows[-n:]) - entry["floor"])))
                emit(u"  %8s %6s %13s %14s" % (u"freq MHz", u"frames", u"penetration", u"bottom excess"))
                for f in sorted(per_freq):
                    pens = [_pen(r, g, entry["floor"], 3.0) for r, g, _ in per_freq[f]]
                    emit(u"  %8g %6d %10.1f mm %11.1f dB"
                         % (f, len(pens), np.mean(pens), np.mean([e for _, _, e in per_freq[f]])))
                for margin in (1.0, 3.0, 6.0):
                    for tol in (0.0, 1.0, 3.0):
                        covers = [f for f in sorted(per_freq)
                                  if np.mean([_pen(r, g, entry["floor"], margin) for r, g, _ in per_freq[f]])
                                  >= depth - tol]
                        pick = max(covers) if covers else min(per_freq)
                        emit(u"    margin %.0f dB  tolerance %.0f mm  -> optimum %g MHz%s"
                             % (margin, tol, pick, u"" if covers else u" (nothing covers)"))
    text = u"\n".join(lines)
    io.open(os.path.join(ROOT, "tests", "measure_frequency_label_margin.txt"), "w", encoding="utf-8").write(text)
    print(text)


def _pen(rows, geometry, floor, margin):
    good = np.where(rows >= floor + margin)[0]
    if good.size == 0:
        return float(geometry.min_depth_mm)
    return float(geometry.min_depth_mm + (good[-1] + 0.5) * geometry.mm_per_point)


if __name__ == "__main__":
    main()
