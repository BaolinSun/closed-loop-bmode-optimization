# -*- coding: utf-8 -*-
"""模拟阅片：把深度阅片包切成盲法批次给子智能体看，再把答案还原成阅片结果文件。

    用途与局限

真实医生阅片之前，用子智能体扮演超声科医生，把「阅片 -> 结果文件 ->
tests/measure_depth_reading.py 拟合规则」整条链路在真实图像上跑通，并得到深度余量的初步估计。
模拟结果只是预演：不能代替真实医生，不直接用作训练标签；两位模拟阅片者是同一个模型，
彼此一致性偏高，不能当作医生之间一致性的上限。结果单独存放在 results_simulated/，
文件里带 "simulated": true，绝不写进真实医生的 results/。

    prepare：盲法批次

从阅片包 manifest.js 按受试者轮转抽 N 个切面（同一受试者最多 2 个），每位阅片者看同一批切面、
各自打乱顺序，另加约 10% 复查条目（放在后半段，换字母、换顺序）。每批 20 条，每个条目一个文件夹：

    ref_grid.png   参考帧，左侧加每 5 毫米一格的深度刻度（主机屏幕上本来就有刻度），用来估计远壁深度
    A.jpg - F.jpg  6 档深度打乱后重命名，文件名不含深度，图像不加刻度

字母与深度的对应关系只写在 _keys/<阅片者>.json，子智能体只读自己的批次目录。

    collect：还原

每批的 answers.json（子智能体写）按 _keys 还原成深度，写成与阅片页面「导出结果」相同格式的
json：results_simulated/<阅片者>.json。同时检查格式：字母是否有效、远壁深度是否在参考帧范围内、
有无遗漏条目。

    用法

    python tools_simulated_depth_reading.py prepare [--views 200] [--readers SIM_vascular SIM_general]
    python tools_simulated_depth_reading.py collect [--partial]

报告写在脚本旁边的 tools_simulated_depth_reading.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import datetime
import json
import os
import random
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from PIL import Image, ImageDraw, ImageFont

import data_paths as DPATH
import measure_depth_reading as MDR

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
READING_DIR = str(DPATH.output_dir("depth_reading"))
SIM_DIR = str(DPATH.output_dir("depth_reading_sim"))
RESULTS_DIR = os.path.join(READING_DIR, "results_simulated")
DEFAULT_READERS = ("SIM_vascular", "SIM_general")
LETTERS = "ABCDEF"
BATCH_SIZE = 20
REPEAT_FRACTION = 0.10
MAX_PER_SUBJECT = 2
GRID_MM = 5.0
MARGIN_PX = 58
LINES = []


def emit(text=""):
    LINES.append(str(text))


def finish():
    text = "\n".join(LINES)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------- prepare

def sample_views(views, n, seed):
    """按受试者轮转抽样：先每人一个，再每人第二个，直到 n 个。"""
    rng = random.Random(seed)
    by_subject = collections.defaultdict(list)
    for v in views:
        by_subject[v["subject"]].append(v)
    subjects = sorted(by_subject)
    rng.shuffle(subjects)
    for s in subjects:
        rng.shuffle(by_subject[s])
    picked = []
    for round_index in range(MAX_PER_SUBJECT):
        for s in subjects:
            if len(picked) >= n:
                return picked
            if round_index < len(by_subject[s]):
                picked.append(by_subject[s][round_index])
    return picked


def reader_order(view_ids, reader, seed):
    """一位阅片者的条目顺序：打乱的全部切面，复查条目插在后半段、且在第一次出现之后。"""
    rng = random.Random("%s:%s" % (seed, reader))
    order = list(view_ids)
    rng.shuffle(order)
    n_repeat = int(round(REPEAT_FRACTION * len(order)))
    half = len(order) // 2
    repeats = rng.sample(order[:half], n_repeat)
    items = [(v, False) for v in order]
    for v in repeats:
        at = rng.randint(half + 1, len(items))
        items.insert(at, (v, True))
    return items


def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def ref_with_scale(src, ref_depth_mm, dst):
    """参考帧左侧加深度刻度：每 5 毫米一格，标注毫米数，像主机屏幕上的刻度。"""
    image = Image.open(src).convert("RGB")
    canvas = Image.new("RGB", (image.width + MARGIN_PX, image.height), (0, 0, 0))
    canvas.paste(image, (MARGIN_PX, 0))
    draw = ImageDraw.Draw(canvas)
    f = font(13)
    mm = 0.0
    while mm <= ref_depth_mm + 1e-6:
        y = int(round(mm / ref_depth_mm * (image.height - 1)))
        major = abs(mm % 10.0) < 1e-6
        draw.line([(MARGIN_PX - (16 if major else 9), y), (MARGIN_PX - 1, y)], fill=(230, 230, 230), width=1)
        if major or mm == 0:
            label = "%d" % mm
            draw.text((4, max(0, min(image.height - 14, y - 7))), label, fill=(230, 230, 230), font=f)
        mm += GRID_MM
    draw.text((4, image.height - 14), "mm", fill=(180, 180, 180), font=f)
    canvas.save(dst)


def prepare(args):
    manifest = MDR.load_manifest(READING_DIR)
    views = {v["id"]: v for v in manifest["views"]}
    picked = sample_views(manifest["views"], args.views, args.seed)
    emit("sampled %d views from %d subjects (of %d views)"
         % (len(picked), len({v["subject"] for v in picked}), len(views)))
    if os.path.exists(SIM_DIR) and os.listdir(SIM_DIR) and not args.force:
        raise SystemExit("%s already exists; pass --force to rebuild (answers there would be lost)"
                         % SIM_DIR.encode("ascii", "backslashreplace").decode())
    os.makedirs(os.path.join(SIM_DIR, "_keys"), exist_ok=True)
    for reader in args.readers:
        rng = random.Random("%s:%s:letters" % (args.seed, reader))
        items = reader_order([v["id"] for v in picked], reader, args.seed)
        keys = {"reader": reader, "seed": args.seed, "batches": {}}
        for b in range(0, len(items), BATCH_SIZE):
            batch = "batch_%02d" % (b // BATCH_SIZE + 1)
            bdir = os.path.join(SIM_DIR, reader, batch)
            os.makedirs(bdir, exist_ok=True)
            listing, bkeys = [], {}
            for k, (view_id, repeat) in enumerate(items[b:b + BATCH_SIZE]):
                view = views[view_id]
                item = "item_%02d" % (k + 1)
                idir = os.path.join(bdir, item)
                os.makedirs(idir, exist_ok=True)
                src = os.path.join(READING_DIR, "images", view_id)
                ref_with_scale(os.path.join(src, "ref.jpg"), view["ref_depth_mm"],
                               os.path.join(idir, "ref_grid.png"))
                frames = list(view["frames"])
                rng.shuffle(frames)
                letters = {}
                for letter, frame in zip(LETTERS, frames):
                    shutil.copyfile(os.path.join(src, frame["file"]), os.path.join(idir, letter + ".jpg"))
                    letters[letter] = frame["depth_mm"]
                listing.append({"item": item})
                bkeys[item] = {"view": view_id, "subject": view["subject"], "repeat": repeat,
                               "ref_depth_mm": view["ref_depth_mm"], "letters": letters}
            with open(os.path.join(bdir, "items.json"), "w", encoding="utf-8") as fh:
                json.dump({"reader": reader, "batch": batch, "items": listing}, fh, indent=1)
            keys["batches"][batch] = bkeys
        with open(os.path.join(SIM_DIR, "_keys", reader + ".json"), "w", encoding="utf-8") as fh:
            json.dump(keys, fh, ensure_ascii=False, indent=1)
        emit("%s: %d items (%d repeats) in %d batches"
             % (reader, len(items), sum(r for _, r in items), len(keys["batches"])))
    emit("batches under %s" % SIM_DIR)


# ---------------------------------------------------------------- collect

def collect(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for reader_file in sorted(os.listdir(os.path.join(SIM_DIR, "_keys"))):
        with open(os.path.join(SIM_DIR, "_keys", reader_file), encoding="utf-8") as fh:
            keys = json.load(fh)
        reader = keys["reader"]
        results, problems, missing = [], [], []
        for batch, bkeys in sorted(keys["batches"].items()):
            path = os.path.join(SIM_DIR, reader, batch, "answers.json")
            if not os.path.exists(path):
                missing.append(batch)
                continue
            with open(path, encoding="utf-8") as fh:
                answers = json.load(fh)
            answers = answers.get("answers", answers) if isinstance(answers, dict) else answers
            by_item = {a.get("item"): a for a in answers}
            for item, key in sorted(bkeys.items()):
                a = by_item.get(item)
                tag = "%s/%s/%s" % (reader, batch, item)
                if a is None:
                    problems.append(tag + " no answer")
                    continue
                invalid = bool(a.get("invalid"))
                best = a.get("best")
                acceptable = [x for x in (a.get("acceptable") or []) if x in LETTERS]
                if not invalid and best not in key["letters"]:
                    problems.append("%s bad best letter %r" % (tag, best))
                    continue
                wall = a.get("far_wall_mm")
                no_wall = bool(a.get("no_far_wall")) or wall is None
                if wall is not None and not (0.0 < float(wall) < key["ref_depth_mm"]):
                    problems.append("%s far wall %s outside 0-%s mm" % (tag, wall, key["ref_depth_mm"]))
                    wall, no_wall = None, True
                best_mm = None if invalid else key["letters"][best]
                accept_mm = [] if invalid else sorted({key["letters"][x] for x in acceptable} | {best_mm})
                results.append({
                    "view": key["view"], "subject": key["subject"], "repeat": key["repeat"],
                    "invalid": invalid,
                    "far_wall": None if (invalid or no_wall) else {"x": None, "y": float(wall) / key["ref_depth_mm"]},
                    "far_wall_mm": None if (invalid or no_wall) else round(float(wall), 2),
                    "no_far_wall": (not invalid) and no_wall,
                    "ref_depth_mm": key["ref_depth_mm"],
                    "best_depth_mm": best_mm, "acceptable_depth_mm": accept_mm,
                    "shown_order_mm": [key["letters"][x] for x in LETTERS],
                    "seconds": None, "saved_at": None,
                    "rationale": a.get("rationale", ""),
                })
        total = sum(len(b) for b in keys["batches"].values())
        emit("%s: %d / %d items collected, %d batches missing %s, %d problems"
             % (reader, len(results), total, len(missing), missing, len(problems)))
        for p in problems[:20]:
            emit("  " + p)
        if missing and not args.partial:
            emit("  not written (missing batches; pass --partial to write anyway)")
            continue
        out = {"reader": reader, "simulated": True, "manifest_seed": keys["seed"],
               "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
               "completed": len(results), "total": total, "results": results}
        with open(os.path.join(RESULTS_DIR, reader + ".json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
        emit("  wrote results_simulated/%s.json" % reader)


def main():
    parser = argparse.ArgumentParser(description="Blinded batches for simulated depth reading.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--views", type=int, default=200)
    p.add_argument("--readers", nargs="+", default=list(DEFAULT_READERS))
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--force", action="store_true")
    c = sub.add_parser("collect")
    c.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    else:
        collect(args)
    finish()


if __name__ == "__main__":
    main()
