# -*- coding: utf-8 -*-
"""把实机帧切成同一探头位置下的族，写出 data/scene_families.json。

前端标签（深度、频率、聚焦）的最优值只能靠比较同一场景下真实采到的多帧得到，
所以先得知道哪些帧属于同一个场景。判据与阈值的来历见 bmode_opt/scene_family.py
和 tests/measure_scene_families.py。

这一步同时给出**覆盖清点**：每根前端轴上有多少个族真的扫过两个以上取值。只扫了
一根轴的族，另外两根轴没有可比对象，那两个标签必须留空——这就是后面标签里逐轴
`*_determined` 标志的依据。

场次由 bmode_opt/console_sessions.py 发现。两套数据切法不同（见 scene_family 模块文档）：

  --dataset phantom  锚点相关切（segment_session），写 data/scene_families.json
  --dataset invivo   按采集方案切（segment_protocol），相关只作质量标记，
                     写 data/scene_families_invivo_carotid.json

用法：python tools_scene_families.py [--dataset invivo] [--workers N]
"""

import argparse
import collections
import concurrent.futures
import io
import json
import os
import sys

sys.path.insert(0, "bmode_opt")

import console_sessions as CS
import scene_family as SF
from hisense_loader import load_capture

MODE_NAMES = {0: "fundamental", 1: "harmonic"}
METHOD = {"phantom": "anchor_correlation", "invivo": "protocol"}


def session_families(label, method):
    """工作进程：读一个场次、切族。"""
    captures = sorted((load_capture(p) for p in CS.session_captures(label)),
                      key=lambda c: c.name)
    return SF.segment(captures, method, session=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CS.DATASETS), default="phantom")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    out_json = args.out or CS.output_path(args.dataset, "families")
    method = METHOD[args.dataset]
    lines = []
    emit = lines.append

    labels = CS.sessions(args.dataset)
    all_families, records, failures = [], [], {}
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        futures = {label: pool.submit(session_families, label, method) for label in labels}
        for label in labels:
            try:
                all_families.extend(futures[label].result())
            except Exception as exc:
                failures[label] = "%s: %s" % (type(exc).__name__, exc)

    emit(u"=========== scene families ===========")
    emit(u"  A family is one probe placement. Frames in different families are of")
    emit(u"  different scenes and must never be compared to pick a front-end setting.")
    emit(u"")
    emit(u"%-30s %6s %7s %7s %8s %-28s %s"
         % (u"family", u"frames", u"anchors", u"loose", u"min corr", u"swept axes", u"flag"))
    for family in all_families:
        flags = [u"anchor starved"] if family.anchor_starved else []
        if family.motion_flag:
            flags.append(u"motion")
        if family.method != method:
            flags.append(family.method)
        emit(u"%-30s %6d %7d %7d %8s %-28s %s"
             % (family.family_id, len(family), len(family.anchor_names),
                len(family.unbracketed),
                u"-" if family.anchor_min_corr is None else u"%.3f" % family.anchor_min_corr,
                u", ".join(family.swept_axes()) or u"-", u", ".join(flags)))
        records.append({
            "family_id": family.family_id,
            "session": family.session,
            "method": family.method,
            "frame_names": family.frame_names,
            "anchor_names": family.anchor_names,
            "unbracketed": family.unbracketed,
            "anchor_starved": family.anchor_starved,
            "anchor_min_corr": family.anchor_min_corr,
            "motion_flag": family.motion_flag,
            "sweeps": [[axis, list(values)] for axis, values in family.sweeps],
            "swept_axes": list(family.swept_axes()),
            "settings": [list(s) for s in family.settings],
        })

    emit(u"")
    emit(u"=========== families per session ===========")
    per_session = collections.Counter(f.session for f in all_families)
    full = collections.Counter(f.session for f in all_families
                               if len(f.swept_axes()) == len(SF.FRONT_AXES))
    emit(u"  sessions %d   families %d   families sweeping all three axes %d"
         % (len(per_session), len(all_families), sum(full.values())))
    emit(u"  families per session            %s"
         % dict(sorted(collections.Counter(per_session.values()).items())))
    emit(u"  all-three-axes families/session %s"
         % dict(sorted(collections.Counter(full.get(s, 0) for s in per_session).items())))
    emit(u"  method                          %s"
         % dict(collections.Counter(f.method for f in all_families)))
    corr = [f.anchor_min_corr for f in all_families if f.anchor_min_corr is not None]
    emit(u"  motion flag (adjacent anchors < %.2f): %d of %d families with >= 2 anchors"
         % (SF.SAME_SCENE_CORR, sum(f.motion_flag for f in all_families), len(corr)))
    if failures:
        emit(u"  FAILED sessions: %d" % len(failures))
        for label, error in sorted(failures.items()):
            emit(u"    %s  %s" % (label, error))

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

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    io.open(out_json, "w", encoding="utf-8").write(
        json.dumps({"dataset": args.dataset, "method": method,
                    "same_scene_corr": SF.SAME_SCENE_CORR,
                    "corr_depth_mm": list(SF.CORR_DEPTH_MM),
                    "families": records}, ensure_ascii=False, indent=1))

    report = CS.report_path(__file__, args.dataset, args.out)
    text = "\n".join(lines)
    io.open(report, "w", encoding="utf-8").write(text)
    print(text.encode("ascii", "replace").decode("ascii"))
    print("\nwrote %s and %s" % (out_json, os.path.basename(report)))


if __name__ == "__main__":
    main()
