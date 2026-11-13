# -*- coding: utf-8 -*-
"""可穿戴线阵（新板，谐波）颈动脉数据 -> 与海信人体训练集同一格式的标签与缓存，供零样本评估。

输入是 E:/数据/wearable_device/颈动脉/原始数据/<会话号>/model_data/ 下研究模式“导出数据”的目录
（格式与海信采集目录相同，hisense_loader 直接能读）。输出两样：

    data/labels_wear_carotid.jsonl                          每帧一行，字段与人体标签相同
    E:/数据/closed_loop_bmode_optimization/wear_dl_cache_carotid/   cache.npz + index.json

缓存与 tools_build_console_training_cache.py 同一条预处理（BC0 / counts_per_db + 深度响应，
按强度分块到 512 x 128），bmode_dl.dataset.FieldIIData 照常读取。

    标定：不拟合，直接用导出里的精确值

可穿戴的显示链是精确公式：counts_per_db = 400、pivot_db 由参考与动态范围算出、深度响应为 0
（导出 meta 的 calibration_entry 写明了推导）。海信要从截图拟合的三样这里都是已知的。

    底噪：按会话实测

取本会话 67 mm 帧最深 15 % 行的中位 dB（tissue.measure_noise_floor）。谐波下这基本是量化地板
（原始包络 1-3 个计数），不是热噪声；报告里写明数值。导出 meta 里的 noise_floor_db 取自上位机
噪声抑制的设定值，不是实测，不用。

    后端标签：海信人体的目标亮度

求解器的目标亮度在海信数据上按场次取“操作者接受的组织灰度”。可穿戴这批的 R′ 都是默认设置
（增益 0、补偿关），没有操作者调过，量出来的“接受亮度”没有意义。零样本评估问的是“海信教师
在可穿戴图像上会怎么调”，所以用海信人体 189 组的中位目标 62 灰（tests/measure_gain_bias.txt：
61.6 +- 9.1）。标签不确定度同样借海信人体的常数 0.1278。

    前端标签：只有聚焦

人体模型只训练了聚焦头（深度、频率的标签不看图就能猜中，训练时权重置 0）。聚焦按人体暂定规则
（tools_generate_target_frontend_labels.optimum_focus_target）：在远壁 ±5 mm 的深度带里比侧向
散斑宽度。比较集是本块里深度 67 mm、频率等于 R′ 频率的帧（R、F3、Z1-Z5，以及同设置的电影回放帧）。
远壁深度来自 data/wear_far_walls_provisional.json（Claude 读的暂定值，不是医生阅片）。

用法：python tools_build_wear_dataset.py [--root ...] [--target-gray 62]
报告写在脚本旁边的 tools_build_wear_dataset.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "bmode_opt"))
sys.path.insert(0, ROOT)

import numpy as np

import calibration as CAL
import data_paths as DPATH
import hisense_backend_sim as S
import labels as LB
import tissue as T
import tools_build_console_training_cache as CACHE
import tools_generate_console_labels as G
import tools_generate_target_frontend_labels as TF
from hisense_loader import load_capture

REPORT_PATH = os.path.join(ROOT, "tools_build_wear_dataset.txt")
HARMONIC = 1                      # Hisense BImageMode of the wear PI harmonic exports
HISENSE_TARGET_GRAY = 62.0        # median accepted tissue gray of the 189 Hisense in-vivo groups
HISENSE_LABEL_UNCERTAINTY = 0.1278
FULL_DEPTH_MM = 67.0
LINES = []


def emit(text=""):
    LINES.append(str(text))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Wearable carotid exports -> labels + cache (Hisense format)")
    p.add_argument("--root", default=str(DPATH.DATASET_ROOT / "wearable_device" / "颈动脉" / "原始数据"))
    p.add_argument("--far-walls", default=os.path.join(ROOT, "data", "wear_far_walls_provisional.json"))
    p.add_argument("--labels-out", default=os.path.join(ROOT, "data", "labels_wear_carotid.jsonl"))
    p.add_argument("--cache-out", default=str(DPATH.output_dir("wear_dl_cache_carotid")))
    p.add_argument("--target-gray", type=float, default=HISENSE_TARGET_GRAY)
    p.add_argument("--rows", type=int, default=512)
    p.add_argument("--lines", type=int, default=128)
    return p.parse_args(argv)


def load_session(session_dir):
    """[(capture, meta)] of one session's study exports, harmonic only."""
    out = []
    for d in sorted((session_dir / "model_data").iterdir()):
        meta_path = d / "wear_model_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("acquisition_step"):
            continue
        capture = load_capture(d)
        if T.capture_image_mode(capture) != HARMONIC:
            continue
        out.append((capture, meta))
    return out


def session_calibration(meta):
    entry = meta["calibration_entry"]
    axis = np.asarray(entry["depth_axis_mm"], np.float64)
    return CAL.GroupCalibration(float(entry["counts_per_db"]), float(entry["pivot_db"]), 0.0, 0,
                                axis, np.zeros_like(axis), {})


def focus_fields(rows_by_name, members, measured, wall, wall_info):
    """Write the target-rule focus label onto every member row (same fields as the in-vivo labels)."""
    result = TF.optimum_focus_target(members, measured, wall)
    best, ladder, at_edge, basis, _extra = result
    for name, setting in members:
        current = setting[3]
        steps = ladder.index(best) - ladder.index(current)
        rows_by_name[name].update({
            "optimal_focus_mm": float(best),
            "focus_direction": LB._direction(steps, 0.5, G.FOCUS_DIRECTIONS),
            "delta_focus_mm": float(best - current), "delta_focus_steps": int(steps),
            "focus_determined": True, "focus_at_edge": bool(at_edge), "focus_basis": basis,
            "focus_ladder": [float(v) for v in ladder]})
    return best, basis


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    walls_doc = json.loads(io.open(args.far_walls, encoding="utf-8").read())
    walls = {(r["session"], int(r["block"])): r for r in walls_doc["readings"]}
    excluded = {(r["session"], int(r["block"])): r["reason"] for r in walls_doc.get("excluded_blocks", [])}

    emit("=========== wearable carotid dataset (zero-shot evaluation) ===========")
    emit("  root          %s" % args.root.encode("ascii", "replace").decode())
    emit("  target gray   %.1f (Hisense in-vivo median accepted tissue gray)" % args.target_gray)
    emit("  far walls     %s (%d readings, provisional, not a sonographer)"
         % (os.path.basename(args.far_walls), len(walls)))
    emit()

    rows, cache_items = [], []
    emit("  session              floor dB  floor src  frames  labelled  blocks  focus-labelled blocks")
    for session_dir in sorted(p for p in __import__("pathlib").Path(args.root).iterdir() if p.is_dir()):
        items = load_session(session_dir)
        if not items:
            continue
        session = session_dir.name
        tag = session.split("_")[0]
        cal = session_calibration(items[0][1])
        to_db = lambda c, cal=cal: S.bc0_to_db(c.bc0, cal.counts_per_db)
        deep = [c for c, _ in items if c.geometry.depth_mm >= FULL_DEPTH_MM - 0.5]
        floor_info = T.measure_noise_floor(deep, to_db, min_depth_mm=FULL_DEPTH_MM - 0.5)
        floor = float(floor_info["floor_db"])
        group_id = "%s/%d" % (session, HARMONIC)
        rows_by_name, by_block = {}, collections.defaultdict(list)
        skipped = 0
        for capture, meta in items:
            st = meta["acquisition_step"]
            key = (session, int(st["block"]))
            if key in excluded:
                skipped += 1
                continue
            db = to_db(capture)
            vm = T.console_tissue_mask(db, floor)
            if vm.sum() < 1000:
                skipped += 1
                continue
            frequency = float(CAL.capture_frequency(capture))
            focus = float(capture.focus_depths_mm[0])
            row = LB.label_frame(
                db, vm, dr_ui=capture.dynamic_range_level, reference_db=cal.pivot_db,
                current=(S.capture_gain_db(capture), np.asarray(capture.tgc_levels, np.float64),
                         float(capture.dynamic_range_level)),
                target_gray=args.target_gray, source="console", frame_id=capture.name, group_id=group_id,
                imaging_mode=T.IMAGE_MODE_NAMES[HARMONIC], depth_mm=round(float(capture.geometry.depth_mm), 1),
                frequency_mhz=frequency, focus_mm=focus, split="test", image_mode=HARMONIC,
                label_uncertainty=HISENSE_LABEL_UNCERTAINTY, calibration_borrowed=False,
                notes=["wearable PI harmonic; target gray borrowed from Hisense in vivo"]).as_dict()
            wall = walls.get(key)
            row.update({"family_id": "%s/b%d" % (session, key[1]), "label_rules": "target_provisional",
                        "far_wall_mm": None if wall is None else wall["far_wall_mm"],
                        "far_wall_readings": 0 if wall is None else 1,
                        "far_wall_source": None if wall is None else "claude_provisional",
                        "far_wall_usable": wall is not None,
                        "wear_session": tag, "wear_block": key[1], "wear_step": st["step"],
                        "wear_view_by_image": None if wall is None else wall["view"],
                        "wear_wall_confidence": None if wall is None else wall["confidence"],
                        "wear_noise_floor_db": floor})
            rows_by_name[capture.name] = row
            by_block[key].append((capture, st))
            cache_items.append((capture.name, capture, cal, floor))
        focus_blocks = 0
        for key, members_raw in by_block.items():
            wall = walls.get(key)
            if wall is None:
                continue
            expert_freq = next(float(st["expert"]["frequency_mhz"]) for _, st in members_raw if st.get("expert"))
            members, measured = [], {}
            for capture, st in members_raw:
                setting = (HARMONIC, round(float(capture.geometry.depth_mm), 1),
                           float(CAL.capture_frequency(capture)), float(capture.focus_depths_mm[0]))
                if abs(setting[1] - FULL_DEPTH_MM) > 0.05 or abs(setting[2] - expert_freq) > 0.05:
                    continue
                members.append((capture.name, setting))
                if st["step"] != "C":          # measure on single frames; Cine frames get the label too
                    measured[capture.name] = G.measure_frame(capture, cal, floor)
            measured_members = [m for m in members if m[0] in measured]
            if len({s[3] for _, s in measured_members}) < 2:
                continue
            best, basis = focus_fields(rows_by_name, measured_members, measured, float(wall["far_wall_mm"]), wall)
            for name, setting in members:          # Cine frames at the R setting share the label
                if name not in measured:
                    steps = sorted({s[3] for _, s in measured_members})
                    rows_by_name[name].update({
                        "optimal_focus_mm": float(best),
                        "focus_direction": LB._direction(steps.index(best) - steps.index(setting[3]), 0.5,
                                                         G.FOCUS_DIRECTIONS),
                        "delta_focus_mm": float(best - setting[3]),
                        "delta_focus_steps": int(steps.index(best) - steps.index(setting[3])),
                        "focus_determined": True, "focus_at_edge": best in (steps[0], steps[-1]),
                        "focus_basis": basis, "focus_ladder": steps})
            focus_blocks += 1
        rows.extend(rows_by_name.values())
        emit("  %-20s %7.2f  %-9s  %6d  %8d  %6d  %d"
             % (tag, floor, "measured", len(items), len(rows_by_name), len(by_block), focus_blocks))

    os.makedirs(os.path.dirname(args.labels_out), exist_ok=True)
    with io.open(args.labels_out, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # cache: the same preprocessing as tools_build_console_training_cache.session_job
    n = len(cache_items)
    db_all = np.empty((n, args.rows, args.lines), np.float32)
    floor_all = np.empty((n, args.rows), np.float32)
    scalars = np.empty((n, len(CACHE.SCALARS)), np.float32)
    source_rows = set()
    for i, (name, capture, cal, floor) in enumerate(cache_items):
        response = CAL.depth_response_for(capture, cal)
        if response is None:
            response = np.zeros(capture.bc0.shape[0])
        db = S.bc0_to_db(capture.bc0, cal.counts_per_db) + response[:, None]
        floor_row = floor + response
        source_rows.add(db.shape[0])
        db_all[i] = CACHE.pool_lines(CACHE.pool_rows(db, args.rows), args.lines)
        floor_all[i] = CACHE.pool_rows(floor_row, args.rows)
        scalars[i] = (float(capture.geometry.min_depth_mm), float(capture.geometry.depth_mm), float(cal.pivot_db),
                      float(floor_row[0]), float(floor_row[-1]))
    if len(source_rows) != 1:
        raise SystemExit("captures have different depth sample counts: %s" % sorted(source_rows))
    os.makedirs(args.cache_out, exist_ok=True)
    arrays = {"frame_id": np.asarray([c[0] for c in cache_items]), "db": db_all, "floor": floor_all}
    for k, name in enumerate(CACHE.SCALARS):
        arrays[name] = scalars[:, k]
    np.savez(os.path.join(args.cache_out, "cache.npz"), **arrays)
    index = {"created": time.strftime("%Y-%m-%d %H:%M"), "source": "console", "dataset": "wear_carotid",
             "labels": args.labels_out, "labels_sha1": CACHE.file_sha1(args.labels_out),
             "frames": n, "rows": args.rows, "lines": args.lines, "source_rows": int(source_rows.pop()),
             "db": "BC0 / 400 counts per dB (exact, wear display chain), depth response 0, pooled in intensity",
             "floor": "session noise floor measured on the deepest 15 % rows of 67 mm frames",
             "overrides": ["reference_db", "noise_floor_top_db", "noise_floor_bottom_db"]}
    io.open(os.path.join(args.cache_out, "index.json"), "w", encoding="utf-8").write(json.dumps(index, indent=2) + "\n")

    emit()
    kinds = collections.Counter(r["wear_step"][0] if r["wear_step"] != "R'" else "R'" for r in rows)
    emit("  label rows %d (%s); focus labels on %d rows"
         % (len(rows), ", ".join("%s %d" % kv for kv in sorted(kinds.items())),
            sum(1 for r in rows if r.get("focus_determined"))))
    emit("  gain direction  %s" % dict(collections.Counter(r["gain_direction"] for r in rows)))
    emit("  focus direction %s" % dict(collections.Counter(r.get("focus_direction") for r in rows)))
    opt = np.asarray([r["optimal_gain_db"] for r in rows])
    emit("  teacher optimal gain %.2f +- %.2f dB (current 0 dB everywhere)" % (opt.mean(), opt.std()))
    sample = db_all[::max(1, n // 200)]
    emit("  cache %d frames, db p1 %.1f  p50 %.1f  p99 %.1f dB; reference_db %.2f; floor %.2f - %.2f dB"
         % (n, *np.percentile(sample, [1, 50, 99]), float(arrays["reference_db"].mean()),
            float(arrays["noise_floor_top_db"].min()), float(arrays["noise_floor_top_db"].max())))
    emit("  wrote %s" % os.path.basename(args.labels_out))
    emit("  wrote %s" % os.path.join(args.cache_out, "cache.npz").encode("ascii", "replace").decode())
    emit("  done in %.0f s" % (time.time() - started))
    text = "\n".join(LINES)
    io.open(REPORT_PATH, "w", encoding="utf-8", newline="\n").write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
