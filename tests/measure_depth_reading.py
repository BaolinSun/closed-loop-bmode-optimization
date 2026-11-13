# -*- coding: utf-8 -*-
"""深度阅片结果 -> 以目标为准的深度规则：医生怎么选，哪条规则最接近医生。

    问题

以目标为准的深度规则要定一个自由参数：远壁下方留多少。医生第 1 帧 93% 停在预设上，标定
不了，所以用 tools_build_depth_reading.py 的离线阅片：医生在参考帧上点远壁，再从同一切面的
6 档深度里选最合适的一档、勾选可接受的。本脚本读阅片结果，回答：

  1. 阅片本身靠不靠得住：每人做了多少、无法判断的比例；同一医生前后两次（复查条目）、
     不同医生之间，远壁位置差多少毫米，最合适深度一致的比例。医生之间的一致率就是任何
     规则能达到的上限。
  2. 医生选的深度与远壁的关系：深度 - 远壁（余量，毫米）与 远壁 / 深度（远壁落在画面
     的几成高度）的分布。
  3. 两种规则形式各取最优参数，按受试者分组交叉验证后的命中率：
       A. 余量规则：盖住「远壁 + m 毫米」的最浅一档
       B. 比例规则：远壁不超过画面高度 r 倍的最浅一档
     对照：一律 41.9 mm（预设）；体模时期的穿透规则（底部高出底噪 6 dB 的最深一档，
     --penetration 时从 E 盘读参考帧的 BC0 现算）。
     命中分两种口径：与医生的最合适一档完全相同；落在医生勾选的可接受集合里。

拟合出的参数写进阅片包目录（E:/数据/closed_loop_bmode_optimization/depth_reading/）下的
fitted_depth_rule.json，供人体标签生成使用。结果文件带 "simulated": true（子智能体模拟阅片，
tools_simulated_depth_reading.py）时，报告里标出「(simulated)」，参数写到
fitted_depth_rule_simulated.json，不占用真实医生结果的文件名。

    用法

    python tests/measure_depth_reading.py                          # 读阅片包 results/*.json
    python tests/measure_depth_reading.py --results 某目录 --penetration

报告写在脚本旁边的 measure_depth_reading.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))

import numpy as np

import data_paths as DPATH

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
READING_DIR = str(DPATH.output_dir("depth_reading"))
SRC = str(DPATH.CAROTID_RAW_ROOT)
# 穿透规则用的新固件常数：试跑三名受试者的标定 counts_per_db 499-568、67 mm 底部底噪
# 6 230-6 320 计数（见 docs/invivo_acquisition_summary_20261018.md 之后的试跑记录）
COUNTS_PER_DB = 530.0
FLOOR_COUNTS = 6270.0
PENETRATION_MARGIN_DB = 6.0
MARGIN_GRID = np.arange(0.0, 30.01, 0.5)
RATIO_GRID = np.arange(0.30, 0.951, 0.01)
FOLDS = 5
LINES = []


def emit(text=""):
    LINES.append(str(text))


def load_manifest(reading_dir):
    with open(os.path.join(reading_dir, "manifest.js"), encoding="utf-8") as fh:
        text = fh.read()
    return json.loads(text[text.index("{"):text.rstrip().rstrip(";").rindex("}") + 1])


def load_results(results_dir):
    readings = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        for r in data["results"]:
            r["reader"] = data["reader"]
            r["simulated"] = bool(data.get("simulated"))
            readings.append(r)
    return readings


def q(values, fmt="%.1f"):
    a = np.asarray([v for v in values if v is not None], float)
    if a.size == 0:
        return "n/a"
    p = np.percentile(a, [10, 25, 50, 75, 90])
    return ("p10 " + fmt + " / p25 " + fmt + " / median " + fmt + " / p75 " + fmt + " / p90 " + fmt
            + "  (n=%d)") % (tuple(p) + (a.size,))


def rule_margin(far_wall, ladder, m):
    """盖住远壁 + m 的最浅一档；都盖不住时取最深一档。"""
    for d in ladder:
        if d >= far_wall + m:
            return d
    return ladder[-1]


def rule_ratio(far_wall, ladder, r):
    """远壁不超过画面高度 r 倍的最浅一档。"""
    for d in ladder:
        if far_wall <= r * d:
            return d
    return ladder[-1]


def hits(preds, rows):
    exact = np.mean([p == r["best_depth_mm"] for p, r in zip(preds, rows)])
    accept = np.mean([p in r["acceptable_depth_mm"] for p, r in zip(preds, rows)])
    return exact, accept


def fit(rule, grid, rows, ladder):
    """按可接受命中率挑参数，并列时按完全命中率。"""
    best = None
    for value in grid:
        exact, accept = hits([rule(r["far_wall_mm"], ladder, value) for r in rows], rows)
        key = (round(accept, 6), round(exact, 6))
        if best is None or key > best[0]:
            best = (key, float(value))
    return best[1]


def cross_validate(rule, grid, rows, ladder, folds=FOLDS):
    """按受试者分组的 k 折：参数只在训练折上拟合。"""
    subjects = sorted({r["subject"] for r in rows})
    rng = np.random.RandomState(0)
    rng.shuffle(subjects)
    fold_of = {s: i % folds for i, s in enumerate(subjects)}
    preds, chosen = {}, []
    for k in range(folds):
        train = [r for r in rows if fold_of[r["subject"]] != k]
        test = [r for r in rows if fold_of[r["subject"]] == k]
        if not train or not test:
            continue
        value = fit(rule, grid, train, ladder)
        chosen.append(value)
        for r in test:
            preds[id(r)] = rule(r["far_wall_mm"], ladder, value)
    ordered = [r for r in rows if id(r) in preds]
    return hits([preds[id(r)] for r in ordered], ordered), chosen


def penetration_depths(views, ladder):
    """体模时期的深度规则：参考帧上底部仍高出底噪 6 dB 的最深一档。"""
    from hisense_loader import load_capture
    out = {}
    for v in views:
        try:
            cap = load_capture(os.path.join(SRC, v["ref"]))
        except Exception:
            continue
        rows = np.median(cap.bc0, axis=1) / COUNTS_PER_DB - FLOOR_COUNTS / COUNTS_PER_DB
        mm = (np.arange(rows.size) + 1) * cap.geometry.mm_per_point
        window = max(1, int(round(2.0 / cap.geometry.mm_per_point)))
        choice = ladder[0]
        for d in ladder:
            end = int(np.searchsorted(mm, d))
            if end > rows.size:
                break
            if rows[max(0, end - window):end].mean() >= PENETRATION_MARGIN_DB:
                choice = d
        out[v["id"]] = choice
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reading-dir", default=READING_DIR)
    parser.add_argument("--results", default=None, help="folder of exported json (default <reading-dir>/results)")
    parser.add_argument("--penetration", action="store_true", help="also score the phantom-era penetration rule")
    args = parser.parse_args()

    manifest = load_manifest(args.reading_dir)
    ladder = sorted(manifest["ladder_mm"])
    views = {v["id"]: v for v in manifest["views"]}
    readings = load_results(args.results or os.path.join(args.reading_dir, "results"))
    if not readings:
        print("no results found")
        return

    emit("Depth reading analysis")
    emit("ladder (mm): " + " ".join("%g" % d for d in ladder))
    emit("")
    emit("== 1. reading volume and reliability")
    by_reader = collections.defaultdict(list)
    for r in readings:
        by_reader[r["reader"]].append(r)
    for reader, rs in sorted(by_reader.items()):
        firsts = [r for r in rs if not r["repeat"]]
        emit("reader %-8s%s items %4d (repeats %3d)  invalid %4.1f%%  no far wall %4.1f%%  seconds/item %s"
             % (reader, " (simulated)" if rs[0]["simulated"] else "", len(firsts), len(rs) - len(firsts),
                100 * np.mean([r["invalid"] for r in firsts]),
                100 * np.mean([r["no_far_wall"] for r in firsts if not r["invalid"]] or [0]),
                q([r["seconds"] for r in rs], "%.0f")))

    usable = [r for r in readings if not r["invalid"] and r["best_depth_mm"] is not None]
    # 同一阅片者的复查：与第一次比
    first = {(r["reader"], r["view"]): r for r in usable if not r["repeat"]}
    pairs = [(first[(r["reader"], r["view"])], r) for r in usable
             if r["repeat"] and (r["reader"], r["view"]) in first]
    if pairs:
        emit("intra-reader (repeat items, n=%d): best depth identical %.0f%%, second best inside first acceptable %.0f%%, "
             "far wall |diff| %s mm"
             % (len(pairs), 100 * np.mean([a["best_depth_mm"] == b["best_depth_mm"] for a, b in pairs]),
                100 * np.mean([b["best_depth_mm"] in a["acceptable_depth_mm"] for a, b in pairs]),
                q([abs(a["far_wall_mm"] - b["far_wall_mm"]) for a, b in pairs
                   if a["far_wall_mm"] is not None and b["far_wall_mm"] is not None])))
    # 不同阅片者之间
    per_view = collections.defaultdict(dict)
    for r in usable:
        if not r["repeat"]:
            per_view[r["view"]][r["reader"]] = r
    inter = [(a, b) for v in per_view.values() for i, a in enumerate(v.values())
             for b in list(v.values())[i + 1:]]
    if inter:
        emit("inter-reader (pairs, n=%d): best depth identical %.0f%%, one's best inside the other's acceptable %.0f%%, "
             "far wall |diff| %s mm"
             % (len(inter), 100 * np.mean([a["best_depth_mm"] == b["best_depth_mm"] for a, b in inter]),
                100 * np.mean([b["best_depth_mm"] in a["acceptable_depth_mm"] for a, b in inter]),
                q([abs(a["far_wall_mm"] - b["far_wall_mm"]) for a, b in inter
                   if a["far_wall_mm"] is not None and b["far_wall_mm"] is not None])))
    else:
        emit("inter-reader: only one reader so far - no ceiling estimate yet")

    emit("")
    emit("== 2. what the doctors chose")
    firsts = [r for r in usable if not r["repeat"]]
    emit("best depth: %s" % dict(sorted(collections.Counter(r["best_depth_mm"] for r in firsts).items())))
    emit("acceptable set size: %s" % dict(sorted(collections.Counter(len(r["acceptable_depth_mm"]) for r in firsts).items())))
    walls = [r for r in firsts if r["far_wall_mm"] is not None]
    emit("far wall (mm): %s" % q([r["far_wall_mm"] for r in walls]))
    emit("margin = best depth - far wall (mm): %s" % q([r["best_depth_mm"] - r["far_wall_mm"] for r in walls]))
    emit("far wall / best depth: %s" % q([r["far_wall_mm"] / r["best_depth_mm"] for r in walls], "%.2f"))
    emit("shallowest acceptable - far wall (mm): %s"
         % q([min(r["acceptable_depth_mm"]) - r["far_wall_mm"] for r in walls]))

    emit("")
    emit("== 3. rules against the doctors (each reading scored separately; CV folds by subject)")
    if len(walls) < 10:
        emit("fewer than 10 readings with a far wall - not fitting yet")
    else:
        emit("%-52s %10s %12s   %s" % ("rule", "exact", "acceptable", "parameter"))
        preset = [41.9 if 41.9 in ladder else min(ladder, key=lambda d: abs(d - 41.9))] * len(walls)
        e, a = hits(preset, walls)
        emit("%-52s %9.0f%% %11.0f%%" % ("preset 41.9 mm always", 100 * e, 100 * a))
        if args.penetration:
            pen = penetration_depths([views[r["view"]] for r in walls], ladder)
            scored = [r for r in walls if r["view"] in pen]
            e, a = hits([pen[r["view"]] for r in scored], scored)
            emit("%-52s %9.0f%% %11.0f%%   n=%d" % ("penetration 6 dB (phantom-era rule)", 100 * e, 100 * a, len(scored)))
        fitted = {}
        for name, rule, grid, unit in [("A. cover far wall + m mm", rule_margin, MARGIN_GRID, "mm"),
                                       ("B. far wall <= r x display depth", rule_ratio, RATIO_GRID, "")]:
            value = fit(rule, grid, walls, ladder)
            e_in, a_in = hits([rule(r["far_wall_mm"], ladder, value) for r in walls], walls)
            (e_cv, a_cv), chosen = cross_validate(rule, grid, walls, ladder)
            emit("%-52s %9.0f%% %11.0f%%   %g %s (in-sample %.0f%% / %.0f%%; fold values %s)"
                 % (name + " [cross-validated]", 100 * e_cv, 100 * a_cv, value, unit, 100 * e_in, 100 * a_in,
                    " ".join("%g" % c for c in chosen)))
            fitted[name] = {"parameter": value, "cv_exact": e_cv, "cv_acceptable": a_cv,
                            "in_sample_exact": e_in, "in_sample_acceptable": a_in}
        out = {"readings": len(walls), "readers": sorted(by_reader), "ladder_mm": ladder,
               "margin_rule_mm": fitted["A. cover far wall + m mm"]["parameter"],
               "ratio_rule": fitted["B. far wall <= r x display depth"]["parameter"], "scores": fitted}
        simulated = any(r["simulated"] for r in readings)
        fitted_name = "fitted_depth_rule_simulated.json" if simulated else "fitted_depth_rule.json"
        out["simulated"] = simulated
        with open(os.path.join(args.reading_dir, fitted_name), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        emit("")
        emit("wrote %s" % fitted_name)

    text = "\n".join(LINES)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
