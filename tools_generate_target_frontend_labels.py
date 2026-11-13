# -*- coding: utf-8 -*-
"""人体颈动脉前端标签（暂定版）：以目标为准的深度、频率、聚焦规则。

    为什么要新规则

体模时期的前端规则以穿透为准：深度取图像底部仍高出底噪 6 dB 的最深一档，频率在当前图像
底部判断覆盖不覆盖得住，聚焦取整幅可用图像中段最锐的一档。人体 42 mm 预设下画面底部大多
已是噪声，于是频率 97% 判「无解」兜底成 4.4 MHz，深度只看穿透、不看血管在哪，聚焦在大半是
噪声的图像上平均、结果不稳。三根轴都应当围绕检查目标——颈动脉远壁——来定。

    三条规则（远壁深度 W 来自阅片）

  深度  盖住 W + m 的最浅一档（m 默认 14 mm）。可接受集合是 m 取 11-14 mm 时各自选出的档：
        两次模拟阅片拟合出的 m 分别是 11 与 14，这一段内的差别还定不下来。
        与体模规则一样，候选只取不浅于当前聚焦的档（聚焦不能比显示深度深）。
  频率  在 W 下方 2-4 mm（excess_at 在 W + 4 mm 处向上取 2 mm 平均）仍高出底噪 6 dB 的频率里，
        按实机点靶分辨率表取最锐的一档；门槛上下 1 dB、分数并列 3% 以内的档进可接受集合；
        都覆盖不住时标 infeasible 并降权。这些与体模规则相同，只是判断的深度从图像底部
        换成了远壁下方（tools_generate_console_labels.optimum_frequency 的 excess_depth_mm）。
  聚焦  只在远壁附近的深度带（带中心在 W - 5 到 W + 5 mm 之间，5 mm 一带）里比侧向散斑宽度，
        逐带取最锐的聚焦，再取这些逐带最优的中位数。没有可比的目标带时，取扫过的聚焦里
        离 W 最近的一档（basis 里写明 geometric）。

深度与目标的关系不依赖频率和聚焦，所以深度标签给到族内每一帧；频率、聚焦仍只给到比较集
（同一族内只有这一根轴在变的帧），与体模规则的口径一致。

    远壁从哪来、哪些不用

远壁取阅片结果（阅片页面导出格式的 json）里该切面全部有效读数的中位数。阅片切面按其参考帧
与深度扫描帧的采集名对应到 scene_family 切出的族。读数少于 2 个、或读数之间相差超过
--max-spread-mm（默认 3 mm，多半是把颈内静脉认成了颈动脉）的切面，前端三轴一律留空，
远壁字段仍写出供核查。没有阅片的族前端留空。

现在的远壁来自子智能体模拟阅片，不是医生，所以这是暂定版：输出另起文件，
每行带 label_rules = "target_provisional" 与 far_wall_source。真实医生阅片后，用
--far-walls 指向医生的结果、--margin-mm 换成拟合值重跑即可。

    用法

    python tools_generate_target_frontend_labels.py                         # 全部受试者
    python tools_generate_target_frontend_labels.py --sessions 20261011_161332_00001
    python tools_generate_target_frontend_labels.py --far-walls a.json b.json --margin-mm 12

后端字段原样取自 --base（默认人体标签 data/labels_console_invivo_carotid.jsonl）。
报告写在脚本旁边的 tools_generate_target_frontend_labels.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import concurrent.futures
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np

import console_sessions as CS
import data_paths as DPATH
import labels as LB
import point_targets as PT
import scene_family as SF
import tools_generate_console_labels as G
import tools_generate_labels as TG
import measure_depth_reading as MDR

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
OUT_PATH = "data/labels_console_invivo_carotid_target.jsonl"
READING_DIR = str(DPATH.output_dir("depth_reading"))
SIM_RUN = DPATH.output_dir("depth_reading_simulation_20261023") / "run_20261023_161651" / "final_results"
DEFAULT_FAR_WALLS = [
    str(SIM_RUN / "SIM_VASCULAR_AI_20261023_161651.json"),
    str(SIM_RUN / "SIM_GENERAL_AI_20261023_161651.json"),
    os.path.join(READING_DIR, "results_simulated", "SIM_vascular.json"),
    os.path.join(READING_DIR, "results_simulated", "SIM_general.json"),
]
FREQ_BELOW_WALL_MM = 4.0      # excess_at 在 W + 4 mm 向上取 2 mm：远壁下方 2-4 mm
FOCUS_WINDOW_MM = 5.0
# 旧前端字段：重新生成前从基础行里删掉。后端字段（optimal_gain_db、optimal_tgc_levels、
# optimal_dr_ui 等）与当前设置 depth_mm / frequency_mhz / focus_mm 保留。
FRONT_FIELDS_DROP = ("optimal_depth", "optimal_frequency", "optimal_focus", "delta_depth", "delta_frequency",
                     "delta_focus", "depth_", "frequency_", "focus_", "family_")
LINES = []


def emit(text=""):
    LINES.append(str(text))


# ---------------------------------------------------------------- 远壁

def load_far_walls(paths, manifest):
    """{切面: {"wall": 中位数, "spread": 极差, "n": 读数, "sources": [...]}}，只用有效读数。"""
    readings = collections.defaultdict(list)
    sources = collections.defaultdict(set)
    used = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        used.append((os.path.basename(path), data.get("reader"), len(data["results"])))
        for r in data["results"]:
            if r.get("invalid") or r.get("far_wall_mm") is None:
                continue
            readings[r["view"]].append(float(r["far_wall_mm"]))
            sources[r["view"]].add(data.get("reader") or os.path.basename(path))
    walls = {}
    for view, values in readings.items():
        walls[view] = {"wall": float(np.median(values)), "spread": float(max(values) - min(values)),
                       "n": len(values), "sources": sorted(sources[view])}
    views = {v["id"]: v for v in manifest["views"]}
    return walls, views, used


def wall_for_family(family, view_frames, walls):
    """族对应的阅片切面：参考帧或深度扫描帧落在这个族里的那个切面。"""
    names = set(family.frame_names)
    hits = collections.Counter()
    for view_id, frames in view_frames.items():
        n = len(names & frames)
        if n:
            hits[view_id] = n
    if not hits:
        return None, None
    view_id = hits.most_common(1)[0][0]
    return view_id, walls.get(view_id)


# ---------------------------------------------------------------- 规则

def depth_rule(wall, ladder, margin):
    for v in ladder:
        if v >= wall + margin - 1e-6:
            return v, False
    return ladder[-1], True


def optimum_depth_target(wall, focus, mode, args):
    ladder = G.CONSOLE_LADDERS.get((mode, "depth_mm"))
    candidates = [v for v in ladder if v >= focus - 0.05] or ladder
    best, short = depth_rule(wall, candidates, args.margin_mm)
    acceptable = {depth_rule(wall, candidates, m)[0]
                  for m in np.arange(args.margin_range[0], args.margin_range[1] + 0.01, 0.5)}
    acceptable.add(best)
    at_edge = short and G.beyond(mode, "depth_mm", best, +1)
    basis = (u"cover far wall %.1f mm + %g mm (acceptable for margin %g-%g mm: %s)%s"
             % (wall, args.margin_mm, args.margin_range[0], args.margin_range[1],
                u"/".join(u"%g" % v for v in sorted(acceptable)),
                u", limited to depths >= focus %g mm" % focus if candidates != ladder else u""))
    return best, ladder, at_edge, basis, {"depth_acceptable_mm": sorted(float(v) for v in acceptable)}


def optimum_focus_target(members, measured, wall):
    by_value = collections.defaultdict(list)
    for name, setting in members:
        by_value[setting[G.AXIS_INDEX["focus_mm"]]].append(measured[name]["widths"])
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
        if not (wall - FOCUS_WINDOW_MM < band <= wall + FOCUS_WINDOW_MM):
            continue
        here = {v: pooled[v][band] for v in values if band in pooled[v]}
        if len(here) >= 2:
            per_band.append((band, min(here, key=here.get)))
    if per_band:
        bests = sorted(b for _, b in per_band)
        best = bests[(len(bests) - 1) // 2]
        basis = u"far wall %.1f mm, per-band best %s" % (
            wall, u" ".join(u"%g@%g" % (f, b) for b, f in per_band))
    else:
        best = min(values, key=lambda v: abs(v - wall))
        basis = u"far wall %.1f mm, no comparable speckle band near it; geometric: nearest focus" % wall
    return best, values, best in (values[0], values[-1]), basis, {}


def frontend_target(family, measured, wall_info, view_id, args):
    """一个族里每帧的前端字段（以目标为准）。字段名与体模规则相同，另加远壁与规则标记。"""
    setting_of = dict(zip(family.frame_names, family.settings))
    fields = {}
    usable = (wall_info is not None and wall_info["n"] >= 2
              and wall_info["spread"] <= args.max_spread_mm)
    for name in family.frame_names:
        f = {}
        for axis in G.AXIS_INDEX:
            stem = axis.split("_")[0]
            f.update({"optimal_%s" % axis: None, "%s_direction" % stem: None, "delta_%s" % axis: None,
                      "delta_%s_steps" % stem: None, "%s_determined" % stem: False,
                      "%s_at_edge" % stem: False, "%s_basis" % stem: None})
        f.update({"frequency_acceptable_mhz": None, "frequency_label_kind": None,
                  "frequency_confidence": None, "frequency_loss_weight": None,
                  "frequency_swept_mhz": None, "frequency_conditioned_on": None,
                  "depth_conditioned_on": None, "depth_acceptable_mm": None,
                  "family_id": family.family_id, "family_unbracketed": name in family.unbracketed,
                  "family_anchor_starved": family.anchor_starved,
                  "family_motion_flag": family.motion_flag,
                  "family_anchor_min_corr": family.anchor_min_corr,
                  "label_rules": "target_provisional", "reading_view": view_id,
                  "far_wall_mm": None if wall_info is None else round(wall_info["wall"], 2),
                  "far_wall_spread_mm": None if wall_info is None else round(wall_info["spread"], 2),
                  "far_wall_readings": 0 if wall_info is None else wall_info["n"],
                  "far_wall_source": None if wall_info is None else wall_info["sources"],
                  "far_wall_usable": bool(usable)})
        fields[name] = f
    if not usable:
        return fields
    wall = wall_info["wall"]

    def write(name, axis, result, how=None):
        best, ladder, at_edge, basis, extra = result
        stem = axis.split("_")[0]
        current = setting_of[name][G.AXIS_INDEX[axis]]
        steps = ladder.index(best) - ladder.index(current)
        fields[name].update({
            "optimal_%s" % axis: float(best),
            "%s_direction" % stem: LB._direction(steps, 0.5, G.SOLVERS[axis][1]),
            "delta_%s" % axis: float(best - current), "delta_%s_steps" % stem: int(steps),
            "%s_determined" % stem: True, "%s_at_edge" % stem: bool(at_edge),
            "%s_basis" % stem: basis, "%s_ladder" % stem: [float(v) for v in ladder]})
        fields[name].update(extra)
        if how is not None:
            fields[name]["%s_conditioned_on" % stem] = how

    # 深度：只看远壁，族内每帧都给
    for name, setting in setting_of.items():
        mode, depth, _, focus = setting
        result = optimum_depth_target(wall, focus, mode, args)
        if depth in result[1]:
            write(name, "depth_mm", result, "far wall only (independent of frequency and focus)")

    # 聚焦：比较集 (模式, 深度, 频率)
    optimal_focus = {}
    for key, members in G.comparison_sets(family, "focus_mm").items():
        if any(n not in measured for n, _ in members):
            continue
        result = optimum_focus_target(members, measured, wall)
        for name, _ in members:
            write(name, "focus_mm", result)
            optimal_focus[name] = result[0]

    # 频率：比较集 (模式, 深度, 聚焦)，优先最优聚焦的集，没采到退回当前聚焦
    frequency_sets = G.comparison_sets(family, "frequency_mhz")
    cache = {}
    target_mm = wall + FREQ_BELOW_WALL_MM
    for name, setting in setting_of.items():
        mode, depth, _, focus = setting
        if target_mm > depth - 0.05:
            continue                     # 远壁下方已在画面之外，这个深度下判断不了
        if name not in optimal_focus:
            tries = [((mode, depth, focus), "current focus (focus not swept)")]
        elif optimal_focus[name] == focus:
            tries = [((mode, depth, focus), "current focus (already optimal)")]
        else:
            tries = [((mode, depth, optimal_focus[name]), "optimal focus"),
                     ((mode, depth, focus), "current focus (no frames at optimal focus)")]
        for key, how in tries:
            if key not in frequency_sets:
                continue
            if key not in cache:
                members = frequency_sets[key]
                cache[key] = (None if any(n not in measured for n, _ in members)
                              else G.optimum_frequency(members, measured, excess_depth_mm=target_mm))
            result = cache[key]
            if result is None or result[0] is None:
                continue
            write(name, "frequency_mhz", result, how)
            break
    return fields


# ---------------------------------------------------------------- 并行

_WALLS, _VIEW_FRAMES, _ARGS = {}, {}, None


def _init(ladders, resolution, walls, view_frames, args):
    G._init_worker(ladders, resolution)
    _WALLS.update(walls)
    _VIEW_FRAMES.update(view_frames)
    global _ARGS
    _ARGS = args


def session_job(key, entry):
    try:
        captures = sorted(TG.group_captures(key), key=lambda c: c.name)
        measured = {c.name: G.measure_frame(c, entry["cal"], entry["floor"]) for c in captures}
        view_frames = {v: f for v, f in _VIEW_FRAMES.items() if v.startswith(key[0] + "_v")}
        front, families = {}, []
        for family in SF.segment(captures, "protocol", session=key[0]):
            view_id, wall_info = wall_for_family(family, view_frames, _WALLS)
            front.update(frontend_target(family, measured, wall_info, view_id, _ARGS))
            families.append({"family_id": family.family_id, "view": view_id,
                             "has_wall": wall_info is not None, "frames": len(family),
                             "swept": len(family.swept_axes())})
        return {"front": front, "families": families}
    except Exception as exc:
        return {"error": "%s: %s" % (type(exc).__name__, exc)}


def main():
    parser = argparse.ArgumentParser(description="Provisional target-based front-end labels (in-vivo carotid).")
    parser.add_argument("--base", default=CS.output_path("invivo", "labels"))
    parser.add_argument("--calibration", default=CS.output_path("invivo", "calibration"))
    parser.add_argument("--far-walls", nargs="+", default=DEFAULT_FAR_WALLS)
    parser.add_argument("--margin-mm", type=float, default=14.0)
    parser.add_argument("--margin-range", type=float, nargs=2, default=(11.0, 14.0))
    parser.add_argument("--max-spread-mm", type=float, default=3.0)
    parser.add_argument("--out", default=OUT_PATH)
    parser.add_argument("--sessions", default=None)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    started = time.time()

    manifest = MDR.load_manifest(READING_DIR)
    walls, views, used = load_far_walls(args.far_walls, manifest)
    view_frames = {vid: {v["ref"]} | {f["capture"] for f in v["frames"]} for vid, v in views.items()}
    emit("far-wall sources: " + "; ".join("%s (%s, %d rows)" % u for u in used))
    spreads = [w["spread"] for w in walls.values()]
    emit("reading views with a far wall: %d of %d; readings per view %s; spread median %.2f mm, > %g mm: %d"
         % (len(walls), len(views), dict(sorted(collections.Counter(w["n"] for w in walls.values()).items())),
            float(np.median(spreads)), args.max_spread_mm, sum(s > args.max_spread_mm for s in spreads)))
    emit("rules: depth = cover far wall + %g mm (acceptable %g-%g), frequency judged %g mm below the far wall, "
         "focus bands within +-%g mm of the far wall" % (args.margin_mm, args.margin_range[0], args.margin_range[1],
                                                         FREQ_BELOW_WALL_MM, FOCUS_WINDOW_MM))

    sessions = CS.sessions("invivo")
    G.CONSOLE_LADDERS.update(G.console_ladders(sessions))
    tables, _ = PT.resolution_table()
    G.RESOLUTION.update(tables)
    emit("resolution tables: " + ", ".join("mode %d %.1f mm" % k for k in sorted(tables)))
    cal_by_group = TG.load_calibration(args.calibration)
    if args.sessions:
        wanted = set(args.sessions.split(","))
        cal_by_group = {k: v for k, v in cal_by_group.items() if k[0] in wanted}
    G.floors_in_counts(cal_by_group)
    jobs = [k for k in sorted(cal_by_group) if cal_by_group[k]["floor"] is not None]

    front, families, failures = {}, [], {}
    print("labelling %d group(s) with %d worker(s) ..." % (len(jobs), args.workers), flush=True)
    with concurrent.futures.ProcessPoolExecutor(
            args.workers, initializer=_init,
            initargs=(dict(G.CONSOLE_LADDERS), dict(G.RESOLUTION), walls, view_frames, args)) as pool:
        futures = {pool.submit(session_job, k, cal_by_group[k]): k for k in jobs}
        for done, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            if "error" in res:
                failures[futures[fut]] = res["error"]
            else:
                front.update(res["front"])
                families.extend(res["families"])
            if done % 20 == 0 or done == len(futures):
                print("  %d / %d groups (%.0f s)" % (done, len(futures), time.time() - started), flush=True)

    wanted_sessions = {k[0] for k in jobs}
    rows = []
    with io.open(args.base, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row["group_id"].rsplit("/", 1)[0] not in wanted_sessions:
                continue
            # 当前频率取一位小数（新固件 BFreqValue 是 4.400000095 这样的 9 位小数），
            # 否则训练时 collect_ladders 会把它和前端的 4.4 当成两档。
            row["frequency_mhz"] = round(float(row["frequency_mhz"]), 1)
            for k in list(row):
                if k.startswith(FRONT_FIELDS_DROP) and k not in ("depth_mm", "frequency_mhz", "focus_mm"):
                    del row[k]
            if row["frame_id"] in front:
                row.update(front[row["frame_id"]])
            else:
                row.update({"label_rules": "target_provisional", "far_wall_usable": False,
                            "depth_determined": False, "frequency_determined": False, "focus_determined": False})
            rows.append(row)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    emit("")
    emit("families %d; with a reading view %d; with a usable far wall %d"
         % (len(families), sum(f["view"] is not None for f in families), sum(f["has_wall"] for f in families)))
    emit("rows %d written to %s; far wall usable on %d rows" % (
        len(rows), args.out, sum(bool(r.get("far_wall_usable")) for r in rows)))
    for stem, names in [("depth", G.DEPTH_DIRECTIONS), ("frequency", G.FREQUENCY_DIRECTIONS),
                        ("focus", G.FOCUS_DIRECTIONS)]:
        det = [r for r in rows if r.get("%s_determined" % stem)]
        emit("%-10s determined %5d / %d   %s   optimal %s   at edge %d"
             % (stem, len(det), len(rows),
                "  ".join("%s %d" % (n, sum(r["%s_direction" % stem] == n for r in det)) for n in names),
                dict(sorted(collections.Counter(r["optimal_%s_mm" % stem if stem != "frequency" else "optimal_frequency_mhz"]
                                                for r in det).items())),
                sum(bool(r.get("%s_at_edge" % stem)) for r in det)))
    det = [r for r in rows if r.get("frequency_determined")]
    emit("frequency  kind %s   confidence %s   acceptable sets %s"
         % (dict(collections.Counter(r["frequency_label_kind"] for r in det)),
            dict(collections.Counter(r["frequency_confidence"] for r in det)),
            dict(collections.Counter(tuple(r["frequency_acceptable_mhz"]) for r in det).most_common(6))))
    det = [r for r in rows if r.get("focus_determined")]
    emit("focus      geometric fallback on %d rows" % sum("geometric" in (r["focus_basis"] or "") for r in det))
    det = [r for r in rows if r.get("depth_determined")]
    emit("depth      acceptable sets %s" % dict(collections.Counter(tuple(r["depth_acceptable_mm"]) for r in det).most_common(6)))
    for key, error in sorted(failures.items()):
        emit("FAILED %s/%d  %s" % (key[0], key[1], error))
    emit("done in %.0f s" % (time.time() - started))

    text = "\n".join(LINES)
    report = REPORT_PATH if not args.sessions else os.path.splitext(args.out)[0] + "_report.txt"
    with io.open(report, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
