# -*- coding: utf-8 -*-
"""六参数网络的实机（海信）微调缓存：把 labels_console.jsonl 涉及的采集压成一个 npz。

与 tools_build_fieldii_training_cache.py 同一格式，训练代码不区分来源。三处与 Field II 不同，
每一处都照搬生成标签时的做法，否则网络看到的图像与教师求解时看到的不是同一幅：

    图像      BC0 / 本组 counts_per_db + 本组深度响应（calibration.depth_response_for）。
              tools_generate_labels.label_console 就是在这个 dB 上求后端最优的。
    曝光参考  本组的 pivot_db（label_console 传给 label_frame 的 reference_db）。标签行里没有
              这个字段，所以写进缓存（reference_db），训练时覆盖标签行。
    底噪      本组的底噪常数，先经 tools_generate_console_labels.floors_in_counts 把借来的值按
              计数换算（与生成标签时同一步），再加上逐行深度响应，与图像同一刻度。

BC0 为 (870 深度点, 256 线)。深度方向按强度分块平均到 512 行（与 Field II 缓存相同的分块
规则），横向相邻两线按强度平均成 128 线，与 Field II 预训练时的宽度一致。

用法：python tools_build_console_training_cache.py [--dataset invivo]
      [--labels ...] [--out ...] [--rows 512] [--lines 128] [--workers N]
--dataset 决定标签、标定与输出目录的默认值（phantom：data/labels_console.jsonl ->
data/console_dl_cache；invivo：data/labels_console_invivo_carotid.jsonl ->
data/console_dl_cache_invivo_carotid）。场次目录由 bmode_opt/console_sessions.py 找。
生成后把输出目录整个拷到服务器。
"""

import argparse
import collections
import concurrent.futures
import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np

import console_sessions as CS

REPORT_PATH = "tools_build_console_training_cache.txt"
SCALARS = ("min_depth_mm", "max_depth_mm", "reference_db", "noise_floor_top_db",
           "noise_floor_bottom_db")


def block_edges(out_rows, source_rows):
    """与 bmode_dl.render.block_edges 相同。"""
    return np.linspace(0, int(source_rows), int(out_rows) + 1).round().astype(np.int64)


def pool_rows(values_db, out_rows):
    """深度方向按强度分块平均。values_db (rows, lines) 或 (rows,)。"""
    edges = block_edges(out_rows, values_db.shape[0])
    counts = np.diff(edges).astype(np.float64)
    intensity = 10.0 ** (np.asarray(values_db, np.float64) / 10.0)
    pooled = np.add.reduceat(intensity, edges[:-1], axis=0)
    pooled = pooled / (counts[:, None] if pooled.ndim == 2 else counts)
    return 10.0 * np.log10(np.maximum(pooled, 1e-30))


def pool_lines(values_db, out_lines):
    """横向按强度把相邻线平均到 out_lines（线数须能整除）。"""
    lines = values_db.shape[1]
    if lines == out_lines:
        return values_db
    if lines % out_lines:
        raise ValueError("%d lines cannot be pooled evenly to %d" % (lines, out_lines))
    k = lines // out_lines
    intensity = 10.0 ** (values_db / 10.0)
    pooled = intensity.reshape(values_db.shape[0], out_lines, k).mean(axis=2)
    return 10.0 * np.log10(np.maximum(pooled, 1e-30))


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        h.update(handle.read())
    return h.hexdigest()


def session_job(session, items, cal_by_group, out_rows, out_lines):
    """工作进程：一个场次里要缓存的帧。items 为 [(frame_id, mode)]。

    返回 {frame_id: (db, floor, 标量...)}、缺失的 frame_id 与源深度点数。
    """
    import calibration as CAL
    import hisense_backend_sim as S
    from hisense_loader import load_capture

    wanted = {frame_id for frame_id, _ in items}
    captures = {}
    for path in CS.session_captures(session):
        if path.name in wanted:
            captures[path.name] = load_capture(path)
    done, missing, source_rows = {}, [], set()
    for frame_id, mode in items:
        capture = captures.get(frame_id)
        entry = cal_by_group.get((session, mode))
        if capture is None or entry is None or entry["floor"] is None:
            missing.append(frame_id)
            continue
        cal = entry["cal"]
        response = CAL.depth_response_for(capture, cal)
        if response is None:
            response = np.zeros(capture.bc0.shape[0])
        db = S.bc0_to_db(capture.bc0, cal.counts_per_db) + response[:, None]
        floor = float(entry["floor"]) + response
        source_rows.add(db.shape[0])
        done[frame_id] = (
            pool_lines(pool_rows(db, out_rows), out_lines).astype(np.float32),
            pool_rows(floor, out_rows).astype(np.float32),
            float(capture.geometry.min_depth_mm), float(capture.geometry.depth_mm),
            float(cal.pivot_db), float(floor[0]), float(floor[-1]))
    return done, missing, source_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(CS.DATASETS), default="phantom")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--lines", type=int, default=128)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    args = parser.parse_args()
    args.labels = args.labels or CS.output_path(args.dataset, "labels")
    args.out = args.out or CS.output_path(args.dataset, "cache")
    cal_path = args.calibration or CS.output_path(args.dataset, "calibration")
    started = time.time()
    lines = []
    emit = lines.append

    import tools_generate_labels as TG
    import tools_generate_console_labels as G

    with io.open(args.labels, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    # 与生成标签时同一顺序：读标定 -> 把借来的底噪按计数换算
    cal_by_group = TG.load_calibration(cal_path)
    G.floors_in_counts(cal_by_group)

    by_session = collections.defaultdict(list)
    for row in rows:
        session, mode = row["group_id"].rsplit("/", 1)
        by_session[session].append((row["frame_id"], int(mode)))

    # 预先分配，几万帧时不必先攒一份列表再 np.stack 复制一遍。
    order = [frame_id for session in sorted(by_session) for frame_id, _ in by_session[session]]
    position = {frame_id: i for i, frame_id in enumerate(order)}
    db_all = np.empty((len(order), args.rows, args.lines), np.float32)
    floor_all = np.empty((len(order), args.rows), np.float32)
    scalars = np.empty((len(order), len(SCALARS)), np.float32)
    filled = np.zeros(len(order), bool)
    missing, source_rows = [], set()
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(session_job, session, by_session[session],
                               {k: v for k, v in cal_by_group.items() if k[0] == session},
                               args.rows, args.lines)
                   for session in sorted(by_session)]
        for count, future in enumerate(concurrent.futures.as_completed(futures), 1):
            done, lost, seen = future.result()
            missing.extend(lost)
            source_rows |= seen
            for frame_id, (db, floor, *values) in done.items():
                i = position[frame_id]
                db_all[i], floor_all[i], scalars[i] = db, floor, values
                filled[i] = True
            if count % 20 == 0 or count == len(futures):
                print("  %d / %d sessions" % (count, len(futures)), flush=True)

    if len(source_rows) != 1:
        raise SystemExit("captures have different depth sample counts: %s" % sorted(source_rows))
    source = source_rows.pop()

    os.makedirs(args.out, exist_ok=True)
    keep = np.flatnonzero(filled)
    arrays = {"frame_id": np.asarray(order)[keep],
              "db": db_all[keep] if keep.size < len(order) else db_all,
              "floor": floor_all[keep] if keep.size < len(order) else floor_all}
    for k, name in enumerate(SCALARS):
        arrays[name] = scalars[keep, k]
    np.savez(os.path.join(args.out, "cache.npz"), **arrays)
    index = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "source": "console",
        "dataset": args.dataset,
        "calibration": cal_path,
        "labels": args.labels,
        "labels_sha1": file_sha1(args.labels),
        "frames": int(len(arrays["frame_id"])),
        "rows": int(args.rows),
        "lines": int(args.lines),
        "source_rows": int(source),
        "db": "BC0 / counts_per_db + depth response (the dB the back-end labels were solved on), "
              "pooled in intensity to rows x lines",
        "floor": "group noise floor after floors_in_counts, plus the depth response, pooled",
        "overrides": ["reference_db", "noise_floor_top_db", "noise_floor_bottom_db"],
    }
    io.open(os.path.join(args.out, "index.json"), "w", encoding="utf-8").write(json.dumps(index, indent=2) + "\n")

    emit("=========== console cache ===========")
    emit("  labels %s: %d rows; cached %d, missing %d" % (args.labels, len(rows), len(arrays["frame_id"]),
                                                          len(missing)))
    for name in missing[:10]:
        emit("    missing: %s" % name)
    emit("  wrote %s/cache.npz: db %s (source rows %d), floor %s, %.0f MB"
         % (args.out, arrays["db"].shape, source, arrays["floor"].shape,
            (arrays["db"].nbytes + arrays["floor"].nbytes) / 1e6))
    # 几万帧时按帧抽样算分位数，免得整块再复制一份。
    sample = arrays["db"][::max(1, len(arrays["db"]) // 2000)]
    emit("  db range p1 %.1f  p50 %.1f  p99 %.1f dB" % tuple(np.percentile(sample, [1, 50, 99])))
    emit("  reference_db (group pivot) %.2f - %.2f dB; noise floor top %.1f - %.1f dB"
         % (arrays["reference_db"].min(), arrays["reference_db"].max(),
            arrays["noise_floor_top_db"].min(), arrays["noise_floor_top_db"].max()))
    emit("  display depth %s mm" % sorted(set(np.round(arrays["max_depth_mm"], 1).tolist())))
    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(lines)
    io.open(CS.report_path(REPORT_PATH, args.dataset,
                                 args.out if args.out != CS.output_path(args.dataset, "cache") else None), "w", encoding="utf-8").write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
