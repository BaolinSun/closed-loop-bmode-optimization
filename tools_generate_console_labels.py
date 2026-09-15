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

  发射频率  图像最底部 2 mm 仍高出底噪 FREQUENCY_MARGIN_DB 的频率里，按实机点靶
            分辨率表挑最锐的一档；标签同时给出可接受集合。详见 optimum_frequency。
            （2026-09-14 前是「穿透 3 dB 覆盖得住的最高频」，三处问题见那里。）

  发射聚焦  比较集内各聚焦档的侧向散斑宽度，逐深度带归一化后取平均，最小者胜。
            E9 实测最优聚焦精确跟随深度带（28/32 落在 ±2.5 mm 内），所以对整幅可用
            图像求平均，胜出的是聚焦在可用深度中段的那一档。

  显示深度  主机深度阶梯上最深的一档，要求图像底部仍高出底噪 FREQUENCY_MARGIN_DB——
            与频率用同一个门槛，否则两根轴互相推（见 optimum_depth）。
            这是【体模上】的判据。体模没有「感兴趣的解剖结构」，取景就是把深度拉到
            信号仍可用的最深处为止。到人体数据上应当换成按解剖
            取景，标签字段不变——这与后端两个数据源「标签种类相同、生成算法可以不同」
            的约定一致。

逐轴可选：一帧所在的族若没有扫过某根轴，那根轴的标签留空（*_determined=False），
而不是填一个默认值。

    三根轴的求解顺序：聚焦 -> 频率 -> 深度

三根轴都通过穿透互相影响：聚焦浅了深部变暗、穿透变短（20260901_E2 谐波 5.7 MHz、显示
41.9 mm：聚焦 10 mm 穿透 28.6 mm，聚焦 20 mm 覆盖整幅），频率决定穿透，深度又按穿透
取景。各自在「另外两根保持当前值」下求最优，会把聚焦太浅的帧标成「频率降低」。

所以按顺序求：先定聚焦；频率在【最优聚焦】的比较集里求；深度在【最优频率、最优聚焦】
的比较集里求。族里没有采到那个组合时，退回当前设置的比较集，并在 *_conditioned_on
里写明，下游据此判断这个标签是否已经考虑了前一根轴的调整。

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
import point_targets as PT
import scene_family as SF
import tissue as T
import tools_generate_labels as TG
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

OUT_PATH = "data/labels_console.jsonl"
CAL_PATH = "bmode_opt/console_calibration.json"

# ---- 频率与深度共用的判据（2026-09-14 改）----
#
# 图像最底部 2 mm 的行中位数要高出底噪这么多，才算这个频率「覆盖得住」这幅图，
# 也才算这个显示深度「可用」。旧判据是穿透 3 dB：高出底噪 3 dB 意味着组织回波功率
# 约等于噪声功率，信噪比约 0 dB，图像上勉强能和噪声分开，而且 41.9 mm 显示时基波 10/11.4 MHz
# 与谐波 5.3/5.7 MHz 的底部正好只高出 2.6-4.1 dB，三个基波族因 0.3-0.8 dB 的读数差
# 给出 8.0/10/11.4 三个答案（tests/measure_frequency_label_margin.py）。6 dB 时组织功率
# 约为噪声的 3 倍（信噪比约 4.8 dB），三个族一致落在 8.0 MHz 附近。
FREQUENCY_MARGIN_DB = 6.0
# 门槛上下各这么多 dB 以内算「贴门槛」：同一频率、同一显示深度在不同场景族间，最弱
# 底部读数相差约 1 dB（8.0 MHz：6.0/6.4/7.2 dB；谐波 5.7 MHz：2.6-3.3 dB）。门槛取
# MARGIN-BORDER、MARGIN、MARGIN+BORDER 三个值各选一次，选出的频率都进可接受集合。
FREQUENCY_BORDER_DB = 1.0
# 只看最底部，不看整幅图最弱处。第一版取「5 mm 以下最弱的 2 mm」，结果谐波 25.1 mm
# 显示时 5.7 MHz 只高出底噪 1.3 dB——最弱处全落在 6 mm：谐波信号要传播一段距离才建立
# 起来，近场本来就暗，而且 4.4/5.3/5.7 MHz 暗、4.7/5.0 MHz 亮，与穿透无关。
BOTTOM_WINDOW_MM = 2.0
# 分辨率分数相差不到这个比例算并列，一并进可接受集合。同一设置的重复帧之间分数最多
# 相差 2.5%（谐波 20260903 41.9 mm 5.0 MHz，15 帧；基波 1.4%、0.8%）。
RESOLUTION_TIE = 0.03
# 扫过的频率在宽松门槛（MARGIN-BORDER）下也都覆盖不了：这时真正该调的是深度，频率
# 标签只是兜底，训练时降权。数值是建议值，下游可以改。
#
# 必须用宽松门槛判「无解」。第一版用 MARGIN 本身，谐波 41.9 mm 显示时最低几档底部正好
# 5.7-7.5 dB，四个族判「无解」、三个族判「有解」，只是把 3 dB 门槛上的抖动搬到了 6 dB。
INFEASIBLE_WEIGHT = 0.25
# {(模式代码, 显示深度): {频率: 分辨率分数}}，main() 里由 point_targets 填入。
RESOLUTION = {}

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
    """前端判据需要的逐帧量：逐行高出底噪多少（频率与深度用），以及逐深度带的侧向散斑宽度。

    都在未加深度响应的 BC0 dB 上算：底噪就是在那上面量的，两者必须同一刻度；侧向
    宽度则与逐行增益无关。
    """
    db = S.bc0_to_db(capture.bc0, calibration.counts_per_db)
    return {
        "row_excess_db": row_excess_db(db, floor_db),
        "mm_per_point": float(capture.geometry.mm_per_point),
        "min_depth_mm": float(capture.geometry.min_depth_mm),
        "widths": FE.band_speckle_widths(db, capture.geometry, floor_db),
        "display_depth_mm": float(capture.geometry.depth_mm),
    }


def row_excess_db(db, floor_db):
    """逐行：行中位数高出底噪多少 dB。excess_at 在任意深度处取最底 BOTTOM_WINDOW_MM 的平均。

    频率判据在显示深度处读它，深度判据在各个阶梯深度处读它。两根轴读的是同一个量，
    这是二者一致的前提。
    """
    return np.median(np.asarray(db, dtype=np.float64), axis=1) - floor_db


def excess_at(measure, depth_mm):
    """一帧在 depth_mm 处的底部余量：以该深度为图像底部时，最底 BOTTOM_WINDOW_MM 的平均。"""
    rows = measure["row_excess_db"]
    spacing = measure["mm_per_point"]
    end = int(np.floor((depth_mm - measure["min_depth_mm"]) / spacing + 1e-6))
    end = min(max(end, 1), rows.size)
    window = max(1, int(round(BOTTOM_WINDOW_MM / spacing)))
    return float(rows[max(0, end - window):end].mean())


def comparison_sets(family, axis):
    """族内某根轴的比较集：模式与其余两根轴相同的帧。只保留该轴取到两个以上值的集。"""
    index = AXIS_INDEX[axis]
    others = [i for i in (1, 2, 3) if i != index]
    sets = collections.defaultdict(list)
    for name, setting in zip(family.frame_names, family.settings):
        sets[(setting[0],) + tuple(setting[i] for i in others)].append((name, setting))
    return {k: v for k, v in sets.items()
            if len({s[index] for _, s in v}) >= 2}


def resolution_for(mode, depth):
    """该模式、该显示深度下的分辨率分数：[(权重, {频率: 分数}, 来源显示深度)]。

    分辨率表只在 25.1 / 41.9 / 67.0 mm 三档显示深度实测过。其余深度在相邻两档之间按
    显示深度线性插值；两端以外用最近一档。插值是假设——基波的轴向台阶在 25.1 与 41.9
    之间何处出现并没有测过——所以调用方把两张相邻表各自选出的档也放进可接受集合。
    """
    measured = sorted(d for m, d in RESOLUTION if m == mode)
    if not measured:
        return []
    for d in measured:
        if abs(depth - d) < 0.05:
            return [(1.0, RESOLUTION[(mode, d)], d)]
    if depth < measured[0]:
        return [(1.0, RESOLUTION[(mode, measured[0])], measured[0])]
    if depth > measured[-1]:
        return [(1.0, RESOLUTION[(mode, measured[-1])], measured[-1])]
    for lo, hi in zip(measured, measured[1:]):
        if lo < depth < hi:
            w = (depth - lo) / (hi - lo)
            return [(1.0 - w, RESOLUTION[(mode, lo)], lo), (w, RESOLUTION[(mode, hi)], hi)]
    return []


def sharpest(candidates, tables):
    """候选频率里分数（按权重合成）最低的一档，以及各候选的分数。分辨率表里没有的频率不参与。"""
    scored = {}
    for f in candidates:
        if tables and all(f in table for _, table, _ in tables):
            scored[f] = sum(w * table[f] for w, table, _ in tables)
    if not scored:
        return None, {}
    return min(scored, key=scored.get), scored


def optimum_frequency(members, measured):
    """图像底部仍高出底噪 FREQUENCY_MARGIN_DB 的频率里，分辨率分数最低的一档。

    2026-09-14 前的规则是「穿透 3 dB 覆盖得住的最高频」，有三处问题：
      一、默认频率越高越锐。点靶显示基波显示深度 >=41.9 mm 时 10/11.4 MHz 反而比
          8.0 MHz 粗 26-44%。改为查实机分辨率表（point_targets.resolution_table）。
      二、3 dB 门槛太松且贴着读数抖动（见 FREQUENCY_MARGIN_DB 的注释）。改为 6 dB，
          并把门槛上下 1 dB 内会改变答案的档都列进可接受集合。
      三、谁都覆盖不了时悄悄取最低频。那时该调的是深度，改为标 infeasible 并降权。
          只有宽松门槛下也覆盖不了才算；正常门槛不行、宽松门槛行的，按宽松门槛选并
          标 borderline，同时把最低频放进可接受集合。

    返回 (最优, 主机频率阶梯, 在边界上, 依据, 附加字段)。附加字段：
      frequency_acceptable_mhz  可接受集合：三个门槛各自选出的档，加上与最优分数并列
                                （RESOLUTION_TIE 以内）且在宽松门槛下覆盖得住的档，加上
                                插值时相邻两张分辨率表各自选出的档。训练时当前频率落在
                                集合内即视为正确。
      frequency_label_kind      measured / infeasible
      frequency_confidence      firm（集合只有最优一档）/ borderline
      frequency_loss_weight     measured 为 1，infeasible 为 INFEASIBLE_WEIGHT
      frequency_swept_mhz       比较集实际扫过的频率
    """
    mode = members[0][1][0]
    depth = members[0][1][AXIS_INDEX["depth_mm"]]
    by_value = collections.defaultdict(list)
    for name, setting in members:
        by_value[setting[AXIS_INDEX["frequency_mhz"]]].append(
            excess_at(measured[name], measured[name]["display_depth_mm"]))
    swept = sorted(by_value)
    excess = {f: float(np.mean(v)) for f, v in by_value.items()}
    ladder = CONSOLE_LADDERS.get((mode, "frequency_mhz")) or swept
    tables = resolution_for(mode, depth)

    def feasible(threshold):
        return [f for f in swept if excess[f] >= threshold]

    lenient = feasible(FREQUENCY_MARGIN_DB - FREQUENCY_BORDER_DB)
    nominal = feasible(FREQUENCY_MARGIN_DB)
    strict = feasible(FREQUENCY_MARGIN_DB + FREQUENCY_BORDER_DB)
    excess_text = u" ".join(u"%g:%.1f" % (f, excess[f]) for f in swept)
    table_text = u"/".join(u"%g" % d for _, _, d in tables) or u"none"

    if not lenient:
        best = swept[0]
        acceptable = {best}
        extra = {"frequency_label_kind": "infeasible",
                 "frequency_loss_weight": INFEASIBLE_WEIGHT}
        at_edge = best != ladder[0]
        basis = (u"no swept frequency keeps %.0f dB over the floor at the bottom of the %.1f mm "
                 u"image (bottom excess dB %s)"
                 % (FREQUENCY_MARGIN_DB - FREQUENCY_BORDER_DB, depth, excess_text))
    else:
        # 正常门槛下没有覆盖得住、宽松门槛下有的，按宽松门槛选；这时按正常门槛本该
        # 是「无解取最低频」，所以最低频也放进可接受集合。
        chosen = nominal or lenient
        best, scored = sharpest(chosen, tables)
        if best is None:
            return None, ladder, False, u"no resolution table for this mode", {}
        acceptable = {best}
        for group in (strict, lenient):
            pick, _ = sharpest(group, tables)
            if pick is not None:
                acceptable.add(pick)
        _, lenient_scores = sharpest(lenient, tables)
        acceptable.update(f for f, s in lenient_scores.items()
                          if s <= scored[best] * (1.0 + RESOLUTION_TIE))
        if len(tables) == 2:
            for _, table, d in tables:
                pick, _ = sharpest(chosen, [(1.0, table, d)])
                if pick is not None:
                    acceptable.add(pick)
        if not nominal:
            acceptable.add(swept[0])
        extra = {"frequency_label_kind": "measured", "frequency_loss_weight": 1.0}
        # 选中的是扫过的最高档而主机上还有更高的档没扫：那一档也许同样覆盖得住且更锐。
        at_edge = best == swept[-1] and best != ladder[-1]
        basis = (u"sharpest of %s by resolution table %s mm%s (bottom excess dB %s)"
                 % (u"/".join(u"%g" % f for f in chosen), table_text,
                    u"" if nominal else u", only under the lenient %.0f dB margin"
                    % (FREQUENCY_MARGIN_DB - FREQUENCY_BORDER_DB), excess_text))
    extra.update({
        "frequency_acceptable_mhz": sorted(float(f) for f in acceptable),
        "frequency_confidence": "firm" if acceptable == {best} else "borderline",
        "frequency_swept_mhz": [float(f) for f in swept],
    })
    return best, ladder, at_edge, basis, extra


def optimum_depth(members, measured):
    """主机深度阶梯上最深的一档，要求以它为图像底部时底部余量仍 >= FREQUENCY_MARGIN_DB。

    比较集里频率、聚焦固定，只有显示深度变。每个阶梯深度 v 用显示深度 >= v 的帧读
    「以 v 为底」的底部余量（excess_at），从浅往深，第一次跌破门槛的前一档就是最优。

    【为什么与频率用同一个门槛】2026-09-14 之前深度按「穿透（高出底噪 3 dB）+ 5 mm
    余量」取景，即图像底部故意放在噪声里。而频率要求底部高出底噪 6 dB。于是按深度
    标签取景后的任何一幅图，频率标签都说「降频」；降频穿透变深，深度标签又说「加深」，
    一路推到最低频、最深、频率无解。tests/verify_frontend_chain.py 在旧标签上看到
    58 次「走一步后还要再走」，典型的是谐波 (25.1 mm, 5.7 MHz) -> (50.2 mm, 5.7 MHz)
    -> (50.2 mm, 4.4 MHz)。同一门槛下，按深度标签取景后当前频率一定覆盖得住；频率若再
    换到更锐的档，那一档也覆盖得住当前深度，深度只会不变或加深，所以沿标签走必然停下。
    不同起点可能停在不同的点（高频浅取景、低频深取景），那是体模上无从裁决的取舍。

    【在主机的完整深度阶梯上求】只要某帧显示深度够深，就能读出任一更浅阶梯处的余量，
    与扫了哪些深度无关。比最深的帧还深的阶梯读不到：若读到的每一档都覆盖得住，只知道
    最优至少这么深，标记在边界上。

    返回的依据里写了「usable to X mm」：最深那帧上余量仍 >= 门槛的最深处，只作报告。
    """
    mode = members[0][1][0]
    swept = sorted({s[AXIS_INDEX["depth_mm"]] for _, s in members})
    ladder = CONSOLE_LADDERS.get((mode, "depth_mm")) or swept
    # 聚焦不能比显示深度深：主机的聚焦阶梯随显示深度移动（25.1 mm 下只到 25 mm），Field II
    # 也只生成聚焦 <= 深度的组合。比这更浅的深度档对当前聚焦不存在，不能当候选。
    focus = members[0][1][AXIS_INDEX["focus_mm"]]
    ladder = [v for v in ladder if v >= focus - 0.05] or ladder
    frames = [measured[name] for name, _ in members]
    evaluated = []
    for value in ladder:
        covering = [m for m in frames if m["display_depth_mm"] >= value - 0.05]
        if not covering:
            break
        evaluated.append((value, float(np.mean([excess_at(m, value) for m in covering]))))
    usable = []
    for value, excess in evaluated:
        if excess < FREQUENCY_MARGIN_DB:
            break
        usable.append(value)

    deepest = max(frames, key=lambda m: m["display_depth_mm"])
    rows = deepest["row_excess_db"]
    window = max(1, int(round(BOTTOM_WINDOW_MM / deepest["mm_per_point"])))
    smooth = np.convolve(rows, np.ones(window) / window, mode="valid")
    good = np.where(smooth >= FREQUENCY_MARGIN_DB)[0]
    usable_mm = (deepest["min_depth_mm"] + (good[-1] + window) * deepest["mm_per_point"]
                 if good.size else float("nan"))
    ladder_text = u" ".join(u"%g:%.1f" % (v, e) for v, e in evaluated)

    if not usable:
        best, at_edge = ladder[0], True
        basis = u"even %g mm leaves under %.0f dB at the bottom (excess dB %s)" % (
            ladder[0], FREQUENCY_MARGIN_DB, ladder_text)
    elif len(usable) == len(evaluated) and evaluated[-1][0] < ladder[-1]:
        best, at_edge = usable[-1], True
        basis = (u"usable at least to %.1f mm (every readable ladder depth keeps %.0f dB; "
                 u"excess dB %s)" % (evaluated[-1][0], FREQUENCY_MARGIN_DB, ladder_text))
    else:
        best, at_edge = usable[-1], False
        basis = u"usable to %.1f mm (excess dB %s)" % (usable_mm, ladder_text)
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


def solve_axis(axis, members, measured):
    """统一返回 (最优, 阶梯, 在边界上, 依据, 附加字段)。"""
    result = SOLVERS[axis][0](members, measured)
    return tuple(result) if len(result) == 5 else tuple(result) + ({},)


def frontend_labels(family, measured):
    """一个族里每帧的前端标签字段。按 聚焦 -> 频率 -> 深度 的顺序求，见模块文档字符串。"""
    fields = collections.defaultdict(dict)
    setting_of = dict(zip(family.frame_names, family.settings))
    for name in family.frame_names:
        for axis in AXIS_INDEX:
            stem = axis.split("_")[0]
            fields[name].update({"optimal_%s" % axis: None, "%s_direction" % stem: None,
                                 "delta_%s" % axis: None, "delta_%s_steps" % stem: None,
                                 "%s_determined" % stem: False, "%s_at_edge" % stem: False,
                                 "%s_basis" % stem: None})
        fields[name].update({"frequency_acceptable_mhz": None, "frequency_label_kind": None,
                             "frequency_confidence": None, "frequency_loss_weight": None,
                             "frequency_swept_mhz": None, "frequency_conditioned_on": None,
                             "depth_conditioned_on": None})
        fields[name]["family_id"] = family.family_id
        fields[name]["family_unbracketed"] = name in family.unbracketed
        fields[name]["family_anchor_starved"] = family.anchor_starved

    def write(name, axis, result):
        best, ladder, at_edge, basis, extra = result
        stem = axis.split("_")[0]
        current = setting_of[name][AXIS_INDEX[axis]]
        steps = ladder.index(best) - ladder.index(current)
        fields[name].update({
            "optimal_%s" % axis: float(best),
            "%s_direction" % stem: LB._direction(steps, 0.5, SOLVERS[axis][1]),
            "delta_%s" % axis: float(best - current),
            "delta_%s_steps" % stem: int(steps),
            "%s_determined" % stem: True,
            "%s_at_edge" % stem: bool(at_edge),
            "%s_basis" % stem: basis,
            "%s_ladder" % stem: [float(v) for v in ladder],
        })
        fields[name].update(extra)

    def solved(axis, sets, key, cache):
        if key not in sets:
            return None
        if key not in cache:
            members = sets[key]
            if any(n not in measured for n, _ in members):
                cache[key] = None
            else:
                result = solve_axis(axis, members, measured)
                cache[key] = None if result[0] is None else result
        return cache[key]

    # 1. 聚焦：在当前深度、当前频率下求。比较集的键是 (模式, 深度, 频率)。
    optimal_focus = {}
    cache = {}
    focus_sets = comparison_sets(family, "focus_mm")
    for name, setting in setting_of.items():
        result = solved("focus_mm", focus_sets, (setting[0], setting[1], setting[2]), cache)
        if result is not None:
            write(name, "focus_mm", result)
            optimal_focus[name] = result[0]

    # 2. 频率：优先在最优聚焦的比较集里求，没采到就退回当前聚焦。键是 (模式, 深度, 聚焦)。
    optimal_frequency = {}
    cache = {}
    frequency_sets = comparison_sets(family, "frequency_mhz")
    for name, setting in setting_of.items():
        mode, depth, _, focus = setting
        if name not in optimal_focus:
            tries = [((mode, depth, focus), "current focus (focus not swept)")]
        elif optimal_focus[name] == focus:
            tries = [((mode, depth, focus), "current focus (already optimal)")]
        else:
            tries = [((mode, depth, optimal_focus[name]), "optimal focus"),
                     ((mode, depth, focus), "current focus (no frames at optimal focus)")]
        for key, how in tries:
            result = solved("frequency_mhz", frequency_sets, key, cache)
            if result is not None:
                write(name, "frequency_mhz", result)
                fields[name]["frequency_conditioned_on"] = how
                optimal_frequency[name] = result[0]
                break

    # 3. 深度：优先在最优频率、最优聚焦的比较集里求，逐步退回当前值。键是 (模式, 频率, 聚焦)。
    cache = {}
    depth_sets = comparison_sets(family, "depth_mm")
    for name, setting in setting_of.items():
        mode, _, frequency, focus = setting
        f_opt = optimal_frequency.get(name, frequency)
        z_opt = optimal_focus.get(name, focus)
        f_text = ("optimal frequency" if name in optimal_frequency
                  else "current frequency (frequency not swept)")
        z_text = "optimal focus" if name in optimal_focus else "current focus (focus not swept)"
        tries, seen = [], set()
        # 最优与当前相同时键重复，保留第一个（写「最优」）的说法。
        for key, how in [((mode, f_opt, z_opt), "%s, %s" % (f_text, z_text)),
                         ((mode, frequency, z_opt), "current frequency, %s" % z_text),
                         ((mode, f_opt, focus), "%s, current focus" % f_text),
                         ((mode, frequency, focus), "current frequency and focus")]:
            if key not in seen:
                seen.add(key)
                tries.append((key, how))
        for key, how in tries:
            result = solved("depth_mm", depth_sets, key, cache)
            if result is not None:
                write(name, "depth_mm", result)
                fields[name]["depth_conditioned_on"] = how
                break
    return fields


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()
    started = time.time()
    lines = []
    emit = lines.append

    CONSOLE_LADDERS.update(console_ladders())
    tables, pins = PT.resolution_table()
    RESOLUTION.update(tables)
    emit(u"=========== console resolution table from point targets (lower = sharper) ===========")
    for (mode, depth), scores in sorted(tables.items()):
        emit(u"  %-12s display %5.1f mm  %2d pins   %s" % (
            T.IMAGE_MODE_NAMES[mode], depth, pins[(mode, depth)],
            u"  ".join(u"%g:%.3f" % (f, s) for f, s in sorted(scores.items()))))
    emit(u"")
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
    determined = [r for r in merged if r.get("frequency_determined")]
    emit(u"  frequency  kind %s   confidence %s"
         % (dict(collections.Counter(r["frequency_label_kind"] for r in determined)),
            dict(collections.Counter(r["frequency_confidence"] for r in determined))))
    emit(u"  frequency  conditioned on %s"
         % dict(collections.Counter(r["frequency_conditioned_on"] for r in determined)))
    emit(u"  depth      conditioned on %s"
         % dict(collections.Counter(r["depth_conditioned_on"] for r in merged
                                    if r.get("depth_determined"))))
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
