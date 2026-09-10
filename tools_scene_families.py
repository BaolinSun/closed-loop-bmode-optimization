# -*- coding: utf-8 -*-
"""把实机帧切成同一探头位置下的族，写出 data/scene_families.json。

前端标签（深度、频率、聚焦）的最优值只能靠比较同一场景下真实采到的多帧得到，
所以先得知道哪些帧属于同一个场景。判据与阈值的来历见 bmode_opt/scene_family.py
和 tests/measure_scene_families.py。

这一步同时给出**覆盖清点**：每根前端轴上有多少个族真的扫过两个以上取值。只扫了
一根轴的族，另外两根轴没有可比对象，那两个标签必须留空——这就是后面标签里逐轴
`*_determined` 标志的依据。

用法：python tools_scene_families.py
"""

import io
import json
import os
import sys

sys.path.insert(0, "bmode_opt")

import scene_family as SF
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

OUT_JSON = "data/scene_families.json"
MODE_NAMES = {0: "fundamental", 1: "harmonic"}


def sessions():
    """全部采集场次，含 20260828/GEN 这类嵌套目录。"""
    found = []
    for path in sorted(DEFAULT_DATA_DIR.rglob("Algo_BC0.bin")):
        label = str(path.parent.parent.relative_to(DEFAULT_DATA_DIR)).replace("\\", "/")
        if label not in found:
            found.append(label)
    return found


def main():
    lines = []
    emit = lines.append

    all_families, records = [], []
    for label in sessions():
        captures = sorted((load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / label)),
                          key=lambda c: c.name)
        families = SF.segment_session(captures, session=label)
        all_families.extend(families)

    emit(u"=========== scene families ===========")
    emit(u"  A family is one probe placement. Frames in different families are of")
    emit(u"  different scenes and must never be compared to pick a front-end setting.")
    emit(u"")
    emit(u"%-26s %6s %7s %7s %-28s %s"
         % (u"family", u"frames", u"anchors", u"loose", u"swept axes", u"flag"))
    for family in all_families:
        emit(u"%-26s %6d %7d %7d %-28s %s"
             % (family.family_id, len(family), len(family.anchor_names),
                len(family.unbracketed), u", ".join(family.swept_axes()) or u"-",
                u"anchor starved" if family.anchor_starved else u""))
        records.append({
            "family_id": family.family_id,
            "session": family.session,
            "frame_names": family.frame_names,
            "anchor_names": family.anchor_names,
            "unbracketed": family.unbracketed,
            "anchor_starved": family.anchor_starved,
            "swept_axes": list(family.swept_axes()),
            "settings": [list(s) for s in family.settings],
        })

    emit(u"")
    emit(u"=========== how many families can label each axis ===========")
    emit(u"  An axis is labelable in a family only if that family holds two or more")
    emit(u"  values of it in one imaging mode. Everything else has nothing to compare.")
    emit(u"")
    emit(u"%-14s %10s %10s %12s" % (u"axis", u"families", u"frames", u"values seen"))
    for axis in SF.FRONT_AXES:
        useful = [f for f in all_families if axis in f.swept_axes()]
        seen = set()
        for family in useful:
            for values in family.axis_values(axis).values():
                seen.update(values)
        emit(u"%-14s %10d %10d %12s"
             % (axis, len(useful), sum(len(f) for f in useful),
                u", ".join("%g" % v for v in sorted(seen))))

    emit(u"")
    emit(u"=========== per-mode ladders inside labelable families ===========")
    for axis in SF.FRONT_AXES:
        per_mode = {}
        for family in all_families:
            if axis not in family.swept_axes():
                continue
            for mode, values in family.axis_values(axis).items():
                if len(values) > 1:
                    per_mode.setdefault(mode, set()).update(values)
        for mode in sorted(per_mode):
            emit(u"  %-14s %-12s %s"
                 % (axis, MODE_NAMES[mode],
                    u", ".join("%g" % v for v in sorted(per_mode[mode]))))

    emit(u"")
    emit(u"=========== the comparison sets that actually decide a label ===========")
    emit(u"  Sweeping an axis inside a family is not enough. To pick the best depth the")
    emit(u"  frames compared must agree on frequency and focus, otherwise the winner")
    emit(u"  differs on more than one knob and the label is confounded. A comparison set")
    emit(u"  is the frames of one family sharing mode and the other two axes.")
    emit(u"")
    emit(u"%-14s %8s %10s %10s %12s %10s"
         % (u"axis", u"sets", u"frames", u"labelable", u"members", u"largest"))
    for axis in SF.FRONT_AXES:
        others = [1 + SF.FRONT_AXES.index(a) for a in SF.FRONT_AXES if a != axis]
        sets, labelable, sizes = 0, 0, []
        for family in all_families:
            groups = {}
            for name, setting in zip(family.frame_names, family.settings):
                key = (setting[0],) + tuple(setting[i] for i in others)
                groups.setdefault(key, set()).add(setting[1 + SF.FRONT_AXES.index(axis)])
            counts = {}
            for name, setting in zip(family.frame_names, family.settings):
                key = (setting[0],) + tuple(setting[i] for i in others)
                counts[key] = counts.get(key, 0) + 1
            for key, values in groups.items():
                sets += 1
                if len(values) > 1:
                    labelable += counts[key]
                    sizes.append(len(values))
        emit(u"%-14s %8d %10d %10d %12.1f %10d"
             % (axis, sets, sum(len(f) for f in all_families), labelable,
                (sum(sizes) / float(len(sizes)) if sizes else 0.0),
                max(sizes) if sizes else 0))
    emit(u"")
    emit(u"  members = how many settings of that axis a comparison set holds on average,")
    emit(u"  i.e. how many candidates the argmin gets to choose between.")

    emit(u"")
    emit(u"  %d families over %d frames"
         % (len(all_families), sum(len(f) for f in all_families)))
    loose = sum(len(f.unbracketed) for f in all_families)
    emit(u"  %d frames sit outside any anchor pair and carry lower confidence" % loose)

    os.makedirs("data", exist_ok=True)
    io.open(OUT_JSON, "w", encoding="utf-8").write(
        json.dumps({"same_scene_corr": SF.SAME_SCENE_CORR,
                    "corr_depth_mm": list(SF.CORR_DEPTH_MM),
                    "families": records}, ensure_ascii=False, indent=1))

    text = "\n".join(lines)
    io.open("tools_scene_families.txt", "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s and tools_scene_families.txt" % OUT_JSON)


if __name__ == "__main__":
    main()
