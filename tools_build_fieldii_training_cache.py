# -*- coding: utf-8 -*-
"""六参数网络的训练缓存：把标签涉及的 Field II 分片压成一个 npz。

    为什么要缓存

原始分片 /envelope 是 (1851 深度点, 128 线)，1848 片共 2.7 GB 的 HDF5，每个 epoch 逐个打开
太慢，而且服务器训练不想依赖 h5py 与 bmode_opt。缓存只留网络要用的东西：

    db            (N, 512, 128) float32   20*log10(包络/display_reference)，深度方向按强度分块平均
    floor         (N, 512)      float32   fieldii_noise.noise_floor_db 逐行底噪，同样分块（按 dB 平均）
    min_depth_mm  (N,)                    深度轴首点
    max_depth_mm  (N,)                    深度轴末点（显示深度）
    frame_id      (N,)                    与 labels_fieldii.jsonl 的 frame_id 对应

分块边界与 bmode_dl.render.block_edges 相同（np.linspace(0, 1851, 513).round()），渲染 TGC 曲线
时按块内平均行位置插值，二者必须一致。强度域平均再取 dB：斑点的能量不因降采样丢失。

用法：python tools_build_fieldii_training_cache.py [--labels data/labels_fieldii.jsonl]
      [--out data/fieldii_dl_cache] [--rows 512] [--workers 8]
需要 h5py（本地 cubdl 环境有）；可以在本地生成后把 data/fieldii_dl_cache 整个目录拷到服务器。
"""

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "bmode_opt")
import numpy as np

REPORT_PATH = "tools_build_fieldii_training_cache.txt"
SOURCE_ROWS = 1851


def block_edges(out_rows, source_rows):
    """与 bmode_dl.render.block_edges 相同（此处复制，缓存脚本不导入 torch）。"""
    return np.linspace(0, int(source_rows), int(out_rows) + 1).round().astype(np.int64)


def shard_path(row, default_dir):
    data_dir = row.get("data_dir") or default_dir
    return Path(data_dir) / "hdf5" / (row["frame_id"] + ".h5")


def process(args):
    """一个分片 -> 降采样的 dB 图与底噪。在子进程里跑。"""
    path, out_rows = args
    import fieldii_noise as FN
    from fieldii_loader import load_shard
    shard = load_shard(path)
    db = shard.db_image                                        # (1851, 128)
    if db.shape[0] != SOURCE_ROWS:
        raise ValueError("%s has %d depth rows, expected %d" % (path.name, db.shape[0], SOURCE_ROWS))
    floor = FN.noise_floor_db(shard)
    edges = block_edges(out_rows, db.shape[0])
    counts = np.diff(edges).astype(np.float64)
    intensity = 10.0 ** (db / 10.0)
    pooled = np.add.reduceat(intensity, edges[:-1], axis=0) / counts[:, None]
    db_small = 10.0 * np.log10(np.maximum(pooled, 1e-30))
    floor_small = np.add.reduceat(floor, edges[:-1]) / counts
    return (path.stem, db_small.astype(np.float32), floor_small.astype(np.float32),
            float(shard.geometry.min_depth_mm), float(shard.geometry.depth_mm))


def file_sha1(path, limit_bytes=None):
    h = hashlib.sha1()
    with open(path, "rb") as handle:
        h.update(handle.read() if limit_bytes is None else handle.read(limit_bytes))
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="data/labels_fieldii.jsonl")
    parser.add_argument("--data-dir", default="data/field_ii/full_noise",
                        help="used when a label row has no data_dir field")
    parser.add_argument("--out", default="data/fieldii_dl_cache")
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows (for a quick test)")
    args = parser.parse_args()
    started = time.time()
    lines = []
    emit = lines.append

    with io.open(args.labels, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    jobs = []
    missing = []
    for row in rows:
        path = shard_path(row, args.data_dir)
        if path.exists():
            jobs.append((path, args.rows))
        else:
            missing.append(row["frame_id"])
    emit("=========== inputs ===========")
    emit("  labels %s: %d rows; shards found %d, missing %d" % (args.labels, len(rows), len(jobs), len(missing)))
    for name in missing[:10]:
        emit("    missing: %s" % name)

    results = []
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, r in enumerate(pool.imap(process, jobs, chunksize=4)):
                results.append(r)
                if (i + 1) % 200 == 0:
                    print("  %d / %d shards  (%.0f s)" % (i + 1, len(jobs), time.time() - started))
    else:
        for i, job in enumerate(jobs):
            results.append(process(job))
            if (i + 1) % 200 == 0:
                print("  %d / %d shards  (%.0f s)" % (i + 1, len(jobs), time.time() - started))

    if not results:
        raise SystemExit("no shard processed")
    frame_id = np.array([r[0] for r in results])
    db = np.stack([r[1] for r in results])
    floor = np.stack([r[2] for r in results])
    min_depth = np.array([r[3] for r in results], np.float32)
    max_depth = np.array([r[4] for r in results], np.float32)

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "cache.npz"), db=db, floor=floor, frame_id=frame_id,
             min_depth_mm=min_depth, max_depth_mm=max_depth)
    index = {
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "labels": args.labels,
        "labels_sha1": file_sha1(args.labels),
        "frames": int(len(frame_id)),
        "rows": int(args.rows),
        "lines": int(db.shape[2]),
        "source_rows": SOURCE_ROWS,
        "db": "10*log10(block mean of intensity) of 20*log10(envelope/display_reference)",
        "floor": "block mean of fieldii_noise.noise_floor_db",
    }
    io.open(os.path.join(args.out, "index.json"), "w", encoding="utf-8").write(json.dumps(index, indent=2) + "\n")

    emit("")
    emit("=========== cache ===========")
    emit("  wrote %s/cache.npz: db %s, floor %s, %.0f MB"
         % (args.out, db.shape, floor.shape, (db.nbytes + floor.nbytes) / 1e6))
    emit("  db range p1 %.1f  p50 %.1f  p99 %.1f dB" % tuple(np.percentile(db[:: max(1, len(db) // 64)], [1, 50, 99])))
    emit("  display depth %s mm" % sorted(set(np.round(max_depth, 1).tolist())))
    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
