# -*- coding: utf-8 -*-
"""人体颈动脉深度阅片包：用已有的深度扫描，让医生选最合适的显示深度，并点出远壁。

    为什么要阅片

以目标为准的深度规则是「盖住远壁 + 余量」，余量是自由参数。方案原想用医生第 1 帧的深度
分布来标定它，但 93% 的切面第 1 帧停在预设 41.9 mm 上，没有信息。每个切面都采了同一位置、
其余参数不变的 6 档深度（25.1-67.0 mm），所以可以离线把这 6 帧拿给医生选，不用再扫。

    每个阅片条目（一个切面）

  1. 在医生参考帧（回到第 1 帧设置的那一帧，通常是 41.9 mm 预设）上点一下颈动脉远壁内中膜。
     这一下就是远壁深度的标注，深度规则拟合要用它。
  2. 同一切面的 6 档深度帧打乱顺序、隐去深度数字并排显示，医生选最合适的一帧，另外勾选
     可以接受的帧。也可以整组标「无法判断」（探头移动、不是颈动脉切面等）。

图像直接取主机截图 Screenthum.bmp 的图像区（彩色，与医生在主机上看到的一致），用
hisense_loader.screen_image_bounds 按导出的几何定位（已处理显示缩放）。截图的文字区
（受试者编号、日期、参数）不在图像区内，不进阅片包。

    输出（E:/数据/closed_loop_bmode_optimization/depth_reading/，只在本机）

    index.html          阅片页面，双击用浏览器打开即可，不需要联网
    manifest.js         条目清单（window.MANIFEST），页面直接读取，避免 file:// 下的跨域限制
    images/<条目>/      ref.jpg（参考帧）与 d25.jpg ... d67.jpg（6 档深度）
    results/            医生导出的结果文件放这里，tests/measure_depth_reading.py 读取

阅片结果保存在浏览器本地（localStorage，按阅片者编号分开），页面上的「导出结果」按钮
下载一个 json 文件。约 10% 的条目在后面重复出现一次，用来量同一医生自身的一致性。

    用法

    python tools_build_depth_reading.py                 # 全部完整切面
    python tools_build_depth_reading.py --limit 5       # 只做 5 个条目试看
    python tools_build_depth_reading.py --height 420    # 图像显示高度（像素）

依赖 tests/measure_invivo_acquisition.py 写出的 data/invivo_inventory/carotid_captures.csv
（逐帧参数缓存）；E 盘数据有变动时先跑 python tests/measure_invivo_acquisition.py --rescan carotid。
报告写在脚本旁边的 tools_build_depth_reading.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import concurrent.futures
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import numpy as np
from PIL import Image

import data_paths as DPATH
from hisense_loader import load_capture, screen_image_bounds
import measure_invivo_acquisition as INV

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
DEFAULT_SRC = str(DPATH.CAROTID_RAW_ROOT)
OUT_DIR = str(DPATH.output_dir("depth_reading"))
LADDER = INV.CAROTID_DEPTH_LADDER          # 25 33 42 50 59 67（四舍五入后的毫米）
REPEAT_FRACTION = 0.10
LINES = []


def emit(text=""):
    LINES.append(str(text))


def find_views(rows):
    """[(条目编号, 受试者, 参考帧, {深度: 帧})]：每组完整深度扫描是一个条目。

    深度扫描窗口里 41.9 mm 会出现两次：窗口第一帧是回到医生设置的参考帧，扫描中途的
    41.9 mm 那帧才是扫描帧。参考帧取窗口第一帧（其深度即医生深度），扫描帧每档取最后出现的那帧。
    """
    sessions = collections.defaultdict(list)
    for r in rows:
        if r["page_num"] != "1" or r["exam"] != "Carotid" or r["probe"] != "L15-4" or r["error"]:
            continue
        r["depth"] = round(float(r["depth_mm"]))
        r["focus"] = round(float(r["focus_mm"]))
        r["freq"] = INV.carotid_freq_name(r)
        r["g"] = int(float(r["gain"]))
        r["d"] = int(float(r["dr"]))
        sessions[r["session"]].append(r)
    views = []
    for sid, seq in sorted(sessions.items()):
        seq.sort(key=lambda r: r["capture"])
        sweeps = INV.find_sweeps(seq, "depth", ("freq", "focus", "g", "d"), set(LADDER), 6)
        same_back = lambda x, y: all(x[f] == y[f] for f in ("freq", "focus", "g", "d"))
        for k, (a, b, _) in enumerate(sweeps):
            window = seq[a:b + 1]
            # 窗口第一帧的深度在后面又出现了，说明它是回到医生设置的参考帧；否则窗口直接从
            # 扫描帧开始，参考帧取窗口前一帧（前端其余参数与后端相同时）
            # 窗口开头可能连着好几帧医生深度：上一个切面最后的回位帧与本切面的医生帧设置完全相同，
            # 都会被吸进窗口。参考帧必须取紧挨第一张扫描帧之前的那一帧，否则拿到的是上一个切面
            # （模拟阅片试跑中 20 条里有 4 条参考帧是横切、扫描帧是纵切，就是这个原因）。
            if window[0]["depth"] in {r["depth"] for r in window[1:]}:
                lead = 1
                while lead < len(window) and window[lead]["depth"] == window[0]["depth"]:
                    lead += 1
                ref, sweep = window[lead - 1], window[lead:]
            elif a > 0 and same_back(seq[a - 1], window[0]):
                ref, sweep = seq[a - 1], window
            else:
                continue
            frames = {}
            for r in sweep:
                frames[r["depth"]] = r["capture"]
            if set(frames) != set(LADDER):
                continue
            views.append(("%s_v%d" % (sid, k + 1), sid, ref["capture"], frames))
    return views


def crop_rgb(capture_dir, height):
    """截图图像区（彩色），缩放到给定高度。返回 (PIL 图像, 图像区行数, 显示深度 mm)。"""
    capture = load_capture(capture_dir)
    screen = Image.open(os.path.join(capture_dir, "Screenthum.bmp")).convert("RGB")
    bounds = screen_image_bounds(capture.geometry, (screen.height, screen.width))
    if bounds is None:
        raise ValueError("no screen bounds (zoom past window or missing window fields)")
    row0, row1, col0, col1 = bounds
    image = screen.crop((col0, row0, col1, row1))
    width = max(1, int(round(image.width * height / float(image.height))))
    return image.resize((width, height), Image.LANCZOS), row1 - row0, float(capture.geometry.depth_mm)


def build_view(view, src, out_dir, height):
    view_id, sid, ref, frames = view
    folder = os.path.join(out_dir, "images", view_id)
    os.makedirs(folder, exist_ok=True)
    image, _, ref_depth = crop_rgb(os.path.join(src, ref), height)
    image.save(os.path.join(folder, "ref.jpg"), quality=88)
    entry = {"id": view_id, "subject": sid, "ref": ref, "ref_depth_mm": round(ref_depth, 2),
             "ref_size": [image.width, image.height], "frames": []}
    for depth in LADDER:
        img, _, depth_mm = crop_rgb(os.path.join(src, frames[depth]), height)
        name = "d%d.jpg" % depth
        img.save(os.path.join(folder, name), quality=88)
        entry["frames"].append({"depth_mm": round(depth_mm, 1), "file": name, "capture": frames[depth],
                                "size": [img.width, img.height]})
    return entry


def main():
    parser = argparse.ArgumentParser(description="Build the offline depth-reading package.")
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--limit", type=int, default=0, help="only the first N views (trial)")
    parser.add_argument("--height", type=int, default=420, help="display height of each image, px")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()

    rows = INV.read_csv(os.path.join(INV.CACHE_DIR, "carotid_captures.csv"))
    views = find_views(rows)
    emit("complete depth sweeps found: %d views in %d subjects"
         % (len(views), len({v[1] for v in views})))
    if args.limit:
        random.Random(args.seed).shuffle(views)
        views = sorted(views[:args.limit])

    entries, failed = [], []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(build_view, v, args.src, args.out, args.height): v for v in views}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            try:
                entries.append(fut.result())
            except Exception as exc:
                failed.append((futures[fut][0], "%s: %s" % (type(exc).__name__, exc)))
            if (i + 1) % 50 == 0:
                print("  %d/%d" % (i + 1, len(views)), flush=True)
    entries.sort(key=lambda e: e["id"])

    rng = random.Random(args.seed)
    repeats = sorted(rng.sample([e["id"] for e in entries], int(round(REPEAT_FRACTION * len(entries)))))
    manifest = {"version": 1, "seed": args.seed, "ladder_mm": [f["depth_mm"] for f in entries[0]["frames"]]
                if entries else [], "repeats": repeats, "views": entries}
    with open(os.path.join(args.out, "manifest.js"), "w", encoding="utf-8") as fh:
        fh.write("window.MANIFEST = " + json.dumps(manifest, ensure_ascii=False) + ";\n")
    with open(os.path.join(ROOT, "tools_depth_reading_page.html"), encoding="utf-8") as fh:
        page = fh.read()
    with open(os.path.join(args.out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(page)
    os.makedirs(os.path.join(args.out, "results"), exist_ok=True)

    emit("views written: %d   failed: %d   repeated for consistency: %d"
         % (len(entries), len(failed), len(repeats)))
    for view_id, why in failed[:30]:
        emit("  failed %s  %s" % (view_id, why))
    emit("reference frame depth (mm): %s" % dict(collections.Counter(e["ref_depth_mm"] for e in entries)))
    emit("open %s in a browser" % os.path.join(args.out, "index.html"))

    text = "\n".join(LINES)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
