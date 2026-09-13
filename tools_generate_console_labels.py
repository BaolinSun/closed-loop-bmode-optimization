# -*- coding: utf-8 -*-
"""实机侧 6 参数标签：后端（增益、TGC、动态范围）+ 前端（显示深度、发射频率、发射聚焦）。

写出 data/labels_console.jsonl，每帧一行，六根轴同在一行里。

    后端三轴怎么来

沿用 tools_generate_labels.py 的做法，直接复用它的函数：BC0 对增益、TGC、动态范围
不变，一帧就能在连续空间里解出最优。唯一的改动在噪声底的借用，见 floors_in_counts。

动态范围仍然定不下来（dr_determined=False）：gCNR 对单调变换不变，而动态范围就是
单调变换，整个阶梯上 0.6996 -> 0.6979 纹丝不动；要低对比度靶才有判据，E10 未采。
标签照常输出，但 optimal_dr_ui 等于当前值，方向恒为 correct，下游必须看 dr_determined。

    前端三轴怎么来

前端参数改变 BC0 本身，最优只能靠比较【同一场景下真实采到的多帧】得到。所以：

  1. 用 scene_family 把每个场次切成探头位置不变的族。
  2. 在族内对每根轴取【比较集】：成像模式与其余两根轴都相同、只有这一根轴变化的帧。
  3. 在比较集内求这一根轴的【条件最优】——在其余两根轴给定的前提下，这根旋钮该拧到
     哪一档。这正好是 Form B 方向标签的含义。

每根轴的判据都是测出来的，不是假设的：

  发射频率  在穿透仍能覆盖当前显示深度的频率里，挑最高的那个（分辨率最好）。
            E8 实测穿透随频率大幅缩短（基波 5.0 -> 11.4 MHz：69.5 -> 41.4 mm），
            所以这条约束真的会咬合。浅层信噪比间隔不能用——基波的间隔在整条阶梯上
            只变 -0.4 dB/MHz，主机在接收链里按频率做了补偿，但那不改变衰减斜率。

  发射聚焦  比较集内各聚焦档的侧向散斑宽度，逐深度带归一化后取平均，最小者胜。
            E9 实测最优聚焦精确跟随深度带（28/32 落在 ±2.5 mm 内），所以对整幅可用
            图像求平均，胜出的是聚焦在可用深度中段的那一档。

  显示深度  取景：图像要装得下有信号的区域，又不要在穿透以下浪费大片噪声。
            这是【体模上】的判据。体模没有「感兴趣的解剖结构」，操作者在均匀组织上
            的做法就是把深度拉到图像变成噪声的地方为止。到人体数据上应当换成按解剖
            取景，标签字段不变——这与后端两个数据源「标签种类相同、生成算法可以不同」
            的约定一致。

逐轴可选：一帧所在的族若没有扫过某根轴，那根轴的标签留空（*_determined=False），
而不是填一个默认值。

    底噪按计数借用，不按 dB 借用

没有深帧的场次量不到底噪，要向同模式的其他场次借。原代码借的是 dB 值，但 dB 值
要除以本场次的 counts_per_db，而谐波场次之间 counts_per_db 相差 44%。同一天两个独立
测量证明底噪以【BC0 计数】计才是接收机常数：E8_THI 17.04 dB x 655.1 = 11163，
E9_THI 20.67 dB x 551.8 = 11406，差 2.2%；dB 值却差 3.6 dB。所以借的是计数，再按
本场次的 counts_per_db 换回 dB。

用法：python tools_generate_console_labels.py
"""

import argparse
import collections
import io
import json
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np

import front_end as FE
import hisense_backend_sim as S
import labels as LB
import scene_family as SF
import tissue as T
import tools_generate_labels as TG
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

OUT_PATH = "data/labels_console.jsonl"
CAL_PATH = "bmode_opt/console_calibration.json"

# 穿透仍覆盖到显示深度往上这么多毫米以内，就算「装得下整幅图」。图像最底几行有边缘
# 效应，要求逐行覆盖到最后一行会把本该算覆盖的帧判成不覆盖。
COVER_TOLERANCE_MM = 1.0

DEPTH_DIRECTIONS = ("shallow", "correct", "deep")
FREQUENCY_DIRECTIONS = ("low", "correct", "high")
FOCUS_DIRECTIONS = ("shallow", "correct", "deep")

AXIS_INDEX = {"depth_mm": 1, "frequency_mhz": 2, "focus_mm": 3}

# 主机上每根轴实际存在的档位，按成像模式分，由 console_ladders() 从全部采集里读出。
#
# 用来判断「在边界上」是否真的意味着不确定。第一版只看比较集自己扫过的范围，于是谐波
# 在浅深度下选中 5.7 MHz 就被判「真实最优可能在范围外」——但主机上根本没有比 5.7 更
# 高的谐波频率。那样有 188/242 条频率标签落在边界上，绝大多数是假警报。
CONSOLE_LADDERS = {}


def console_ladders():
    """{(模式, 轴): 排好序的档位}。只收深度与频率：聚焦档位随显示深度移动，一个模式里
    合并起来没有意义，聚焦的边界仍按比较集自己的范围判断。"""
    ladders = collections.defaultdict(set)
    for path in DEFAULT_DATA_DIR.rglob("Algo_BC0.bin"):
        capture = load_capture(path.parent)
        mode = S.capture_image_mode(capture)
        setting = SF.front_setting(capture)
        ladders[(mode, "depth_mm")].add(setting[AXIS_INDEX["depth_mm"]])
        ladders[(mode, "frequency_mhz")].add(setting[AXIS_INDEX["frequency_mhz"]])
    return {k: sorted(v) for k, v in ladders.items()}


def beyond(mode, axis, value, direction):
    """主机阶梯上在 value 之外（direction=+1 更大，-1 更小）是否还有档位。"""
    ladder = CONSOLE_LADDERS.get((mode, axis))
    if not ladder:
        return True                     # 不知道就当作还有，宁可多报不确定
    return any(v > value for v in ladder) if direction > 0 else any(v < value for v in ladder)


def floors_in_counts(cal_by_group):
    """把没量到底噪的组，按同模式底噪计数的中位数换算回本组的 dB。

    「量到」由数据本身判断：组里至少有一帧显示深度够深（>= tissue.NOISE_FLOOR_MIN_DEPTH_MM），
    图像最深处才可能已经超出穿透、露出底噪。

    不能用 console_calibration.json 里的 noise_floor_measured。tools_refit_calibration.py
    写的是 `floors.get(key) is not None`，而 floors 是在允许借用的情况下算出来的，借来的
    值也非空，于是 21 个组全标成 True。后果换算成计数就看得出来：真正量到的谐波组
    （20260903 / 20260901 / E8_THI / E9_THI）落在 11164-11404 计数，差 2.1%；借了
    18.62 dB 的组被放大到 13227-15906，例如 20260814 应为 13.1 dB 却用了 18.62，
    高 5.5 dB，会把真实组织当噪声剔除。

    量到了的组保留自己的测量值，只替换借来的。
    """
    deep_enough = {}
    for key in cal_by_group:
        try:
            paths = find_captures(DEFAULT_DATA_DIR / key[0])
        except Exception:
            deep_enough[key] = False
            continue
        captures = [load_capture(p) for p in paths]
        deep_enough[key] = any(
            S.capture_image_mode(c) == key[1]
            and c.geometry.depth_mm >= T.NOISE_FLOOR_MIN_DEPTH_MM for c in captures)

    per_mode = collections.defaultdict(list)
    for key, entry in cal_by_group.items():
        if deep_enough[key] and entry["floor"] is not None:
            per_mode[key[1]].append(entry["floor"] * entry["cal"].counts_per_db)
    changed = []
    for key, entry in cal_by_group.items():
        if deep_enough[key] or not per_mode.get(key[1]):
            continue
        counts = float(np.median(per_mode[key[1]]))
        new = counts / entry["cal"].counts_per_db
        changed.append((key, entry["floor"], new))
        entry["floor"] = new
        entry["floor_measured"] = False
    return changed, {m: (float(np.median(v)), len(v)) for m, v in per_mode.items()}


def measure_frame(capture, calibration, floor_db):
    """前端判据需要的逐帧量：穿透深度，以及逐深度带的侧向散斑宽度。

    都在未加深度响应的 BC0 dB 上算：底噪就是在那上面量的，两者必须同一刻度；侧向
    宽度则与逐行增益无关。
    """
    db = S.bc0_to_db(capture.bc0, calibration.counts_per_db)
    return {
        "penetration_mm": FE.penetration_depth_mm(db, capture.geometry, floor_db),
        "widths": FE.band_speckle_widths(db, capture.geometry, floor_db),
        "display_depth_mm": float(capture.geometry.depth_mm),
    }


def comparison_sets(family, axis):
    """族内某根轴的比较集：模式与其余两根轴相同的帧。只保留该轴取到两个以上值的集。"""
    index = AXIS_INDEX[axis]
    others = [i for i in (1, 2, 3) if i != index]
    sets = collections.defaultdict(list)
    for name, setting in zip(family.frame_names, family.settings):
        sets[(setting[0],) + tuple(setting[i] for i in others)].append((name, setting))
    return {k: v for k, v in sets.items()
            if len({s[index] for _, s in v}) >= 2}


def optimum_frequency(members, measured):
    """穿透仍覆盖显示深度的最高频率。一个都覆盖不了就取穿透最深的最低频率。"""
    mode = members[0][1][0]
    by_value = collections.defaultdict(list)
    for name, setting in members:
        by_value[setting[AXIS_INDEX["frequency_mhz"]]].append(measured[name])
    values = sorted(by_value)
    depth = members[0][1][AXIS_INDEX["depth_mm"]]
    covers = [v for v in values
              if np.mean([m["penetration_mm"] for m in by_value[v]])
              >= depth - COVER_TOLERANCE_MM]
    if covers:
        best = max(covers)
        # 选中的是扫过的最高档，而主机上还有更高的档没扫：那一档也许同样覆盖得住。
        at_edge = best == values[-1] and beyond(mode, "frequency_mhz", best, +1)
        return best, values, at_edge, "covers the %.1f mm image" % depth
    at_edge = beyond(mode, "frequency_mhz", values[0], -1)
    return values[0], values, at_edge, "no swept frequency covers the %.1f mm image" % depth


def optimum_depth(members, measured):
    """取景：显示深度要装得下穿透深度并留出余量，又不要在噪声里浪费。

    【在主机的完整深度阶梯上求，不只在扫过的深度里求。】这是深度与另外两根轴的
    关键区别：深度的判据只需要穿透深度，而穿透是一次物理测量，与扫了哪些深度无关。
    频率与聚焦做不到——没扫到的档位根本没有测量值。

    第一版只在扫过的深度里选，出了两类假答案。E9 只扫了 25.1/41.9/58.6 mm，谐波
    5 MHz 穿透约 46 mm、想要约 51 mm，只能在 41.9（截断）与 58.6（浪费）之间挑，
    选了 58.6——而主机上明明有 50.2。E8 只扫了 67/75.4 mm，于是任何频率都选 67。
    at_edge 抓得住「超出两端」，抓不住「跳过了中间一档」。

    穿透要从【显示深度大于穿透】的帧上才量得到——浅的帧整幅都覆盖，读出来只是显示
    深度本身。取未被整幅覆盖的帧里读数最深的那个。若每一帧都被整幅覆盖，只知道穿透
    至少有扫过的最深那么深，这时按下界求，并标记在边界上。
    """
    mode = members[0][1][0]
    swept = sorted({s[AXIS_INDEX["depth_mm"]] for _, s in members})
    ladder = CONSOLE_LADDERS.get((mode, "depth_mm")) or swept
    readings = [measured[name]["penetration_mm"] for name, setting in members
                if measured[name]["penetration_mm"]
                < setting[AXIS_INDEX["depth_mm"]] - COVER_TOLERANCE_MM]
    if readings:
        penetration, bound = float(max(readings)), False
    else:
        penetration, bound = float(swept[-1]), True

    costs = [FE.framing_cost(v, penetration) for v in ladder]
    best = ladder[int(np.argmin(costs))]
    wanted = penetration + FE.FRAMING_MARGIN_MM
    # 不确定有两种来源：穿透只知道下界，真实值可能更深；或者想要的深度超出主机阶梯两端。
    at_edge = bound or wanted > ladder[-1] or wanted < ladder[0]
    basis = ("penetration at least %.1f mm (every swept frame fully covered)" % penetration
             if bound else "penetration %.1f mm" % penetration)
    return best, ladder, at_edge, basis


def optimum_focus(members, measured):
    """逐深度带找最锐利的聚焦，取这些逐带最优的中位数。

    E9 在实机上证明的是【逐带】的规律：每个深度带里侧向散斑宽度最小的聚焦档，落在
    该带中心 ±2.5 mm 内（28/32，档间差异是帧间离散的 39-1231 倍）。所以这里直接用它：

      1. 每个带只在【该带量得到的】聚焦档之间比，取最锐的那个。
      2. 对这幅图里所有确有组织信号的带，取逐带最优的（下）中位数。

    结果就是服务于可用深度中段的那一档。体模没有感兴趣的解剖结构，可用图像的中段就是
    默认的感兴趣区；到人体数据上换成按解剖定，字段不变。

    前两版都错了，记下以免重蹈：

    第一版只比「每个聚焦都量得到的公共带」。聚焦把能量集中在焦深附近，同一个带在不同
    聚焦下亮度不同——聚焦 10 mm 时 27.5 mm 以深跌破噪声门槛，聚焦 30 mm 时 7.5 mm
    跌破——于是公共带恰好剩下最不能区分聚焦的中间几个。E9_THI 在 41.9 与 58.6 mm
    下只剩三个带，25.1 mm 下只剩一个，10 帧定不出来。

    第二版改成「量不到的带给该聚焦记最差比值」。这引入了亮度偏置：深聚焦把深部照亮、
    越过门槛的带更多，吃的惩罚更少，于是赢。谐波底噪高（20.67 dB）剔除多，被推到
    30 mm 顶在边界；基波底噪低（6.79 dB）几乎不剔除，结果是 15 mm。同一个判据在两种
    模式下给出相反倾向，说明它量的是亮度而不是锐利度。

    逐带取最优则不需要惩罚：每个带内部只比量得到的档位，门槛高低只影响哪些带参与，
    不影响带内谁胜。
    """
    by_value = collections.defaultdict(list)
    for name, setting in members:
        by_value[setting[AXIS_INDEX["focus_mm"]]].append(measured[name]["widths"])
    values = sorted(by_value)
    pooled = {}
    for value in values:
        bands = collections.defaultdict(list)
        for widths in by_value[value]:
            for band, width in widths.items():
                bands[band].append(width)
        pooled[value] = {b: float(np.mean(w)) for b, w in bands.items()}

    per_band = []
    for band in sorted(set().union(*(set(p) for p in pooled.values()))):
        here = {v: pooled[v][band] for v in values if band in pooled[v]}
        if len(here) < 2:
            continue                    # 只有一个档位量得到，带内无从比较
        per_band.append((band, min(here, key=here.get)))
    if len(per_band) < 2:
        return None, values, False, "fewer than two depth bands comparable across focus"
    bests = sorted(b for _, b in per_band)
    best = bests[(len(bests) - 1) // 2]
    return (best, values, best in (values[0], values[-1]),
            "per-band best %s" % u" ".join("%g@%g" % (f, b) for b, f in per_band))


SOLVERS = {"depth_mm": (optimum_depth, DEPTH_DIRECTIONS),
           "frequency_mhz": (optimum_frequency, FREQUENCY_DIRECTIONS),
           "focus_mm": (optimum_focus, FOCUS_DIRECTIONS)}


def frontend_labels(family, measured):
    """一个族里每帧的前端标签字段。"""
    fields = collections.defaultdict(dict)
    for name in family.frame_names:
        for axis in AXIS_INDEX:
            stem = axis.split("_")[0]
            fields[name].update({"optimal_%s" % axis: None, "%s_direction" % stem: None,
                                 "delta_%s" % axis: None, "delta_%s_steps" % stem: None,
                                 "%s_determined" % stem: False, "%s_at_edge" % stem: False,
                                 "%s_basis" % stem: None})
        fields[name]["family_id"] = family.family_id
        fields[name]["family_unbracketed"] = name in family.unbracketed
        fields[name]["family_anchor_starved"] = family.anchor_starved

    for axis, (solve, names) in SOLVERS.items():
        stem = axis.split("_")[0]
        index = AXIS_INDEX[axis]
        for members in comparison_sets(family, axis).values():
            if any(name not in measured for name, _ in members):
                continue
            best, ladder, at_edge, basis = solve(members, measured)
            if best is None:
                continue
            for name, setting in members:
                current = setting[index]
                steps = ladder.index(best) - ladder.index(current)
                fields[name].update({
                    "optimal_%s" % axis: float(best),
                    "%s_direction" % stem: LB._direction(steps, 0.5, names),
                    "delta_%s" % axis: float(best - current),
                    "delta_%s_steps" % stem: int(steps),
                    "%s_determined" % stem: True,
                    "%s_at_edge" % stem: bool(at_edge),
                    "%s_basis" % stem: basis,
                    "%s_ladder" % stem: [float(v) for v in ladder],
                })
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    started = time.time()
    lines = []
    emit = lines.append

    CONSOLE_LADDERS.update(console_ladders())
    cal_by_group = TG.load_calibration()
    changed, mode_counts = floors_in_counts(cal_by_group)
    emit(u"=========== noise floor borrowed in counts, not in dB ===========")
    for mode, (counts, n) in sorted(mode_counts.items()):
        emit(u"  %-12s receiver floor %.0f BC0 counts (median of %d groups with deep frames)"
             % (T.IMAGE_MODE_NAMES[mode], counts, n))
    emit(u"")
    emit(u"=========== console ladders (edge means the console could still go further) ===========")
    for (mode, axis), ladder in sorted(CONSOLE_LADDERS.items()):
        emit(u"  %-12s %-14s %s" % (T.IMAGE_MODE_NAMES[mode], axis,
                                     u" ".join("%g" % v for v in ladder)))
    emit(u"")
    for key, old, new in sorted(changed):
        emit(u"  %-28s %-12s borrowed floor %6.2f -> %6.2f dB"
             % (key[0], T.IMAGE_MODE_NAMES[key[1]], old, new))

    targets = TG.console_targets(cal_by_group)
    backend = TG.label_console(cal_by_group, targets)
    rows = {row["frame_id"]: row for row in backend}
    emit(u"")
    emit(u"back-end labels: %d frames  (%.0f s)" % (len(rows), time.time() - started))

    front = {}
    for key in sorted(cal_by_group):
        session, mode = key
        entry = cal_by_group[key]
        if entry["floor"] is None:
            continue
        captures = sorted((load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)),
                          key=lambda c: c.name)
        captures = [c for c in captures if S.capture_image_mode(c) == mode]
        measured = {c.name: measure_frame(c, entry["cal"], entry["floor"]) for c in captures}
        for family in SF.segment_session(captures, session=session):
            front.update(frontend_labels(family, measured))
    emit(u"front-end fields: %d frames  (%.0f s)" % (len(front), time.time() - started))

    merged, backend_only = [], 0
    for frame_id, row in rows.items():
        # 后端写入的是主机原始显示深度（41.87、66.99），前端比较集按 scene_family 的约定
        # 取一位小数（41.9、67.0）。不统一的话 optimal_depth_mm - depth_mm 与
        # delta_depth_mm 对不上，当前值也不在阶梯里。差的只有 0.01-0.03 mm，统一成一位小数。
        row["depth_mm"] = round(float(row["depth_mm"]), 1)
        if frame_id in front:
            row.update(front[frame_id])
        else:
            backend_only += 1
        merged.append(row)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    emit(u"")
    emit(u"=========== coverage: which axes each frame carries ===========")
    emit(u"  %d rows written to %s" % (len(merged), args.out))
    if backend_only:
        emit(u"  %d rows have back-end labels but no front-end fields" % backend_only)
    for stem, names in [("depth", DEPTH_DIRECTIONS), ("frequency", FREQUENCY_DIRECTIONS),
                        ("focus", FOCUS_DIRECTIONS)]:
        determined = [r for r in merged if r.get("%s_determined" % stem)]
        counts = collections.Counter(r["%s_direction" % stem] for r in determined)
        edge = sum(r["%s_at_edge" % stem] for r in determined)
        emit(u"  %-10s determined %3d / %d   %s   at edge %d"
             % (stem, len(determined), len(merged),
                u"  ".join(u"%s %d" % (n, counts.get(n, 0)) for n in names), edge))
    for axis, key in [("gain", "gain_direction"), ("dynamic range", "dr_direction")]:
        emit(u"  %-10s %s" % (axis, dict(collections.Counter(r[key] for r in merged))))
    emit(u"  %-10s determined %d / %d" % ("dyn range", sum(r["dr_determined"] for r in merged),
                                          len(merged)))
    emit(u"")
    emit(u"done in %.0f s" % (time.time() - started))

    text = "\n".join(lines)
    io.open("tools_generate_console_labels.txt", "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
