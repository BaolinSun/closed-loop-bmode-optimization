# -*- coding: utf-8 -*-
"""人体采集（2026-10-08 起）盘点：采了多少、按方案采齐了没有、能给闭环用多少。

    问题

E:\\数据\\hisense_medical 下有四个部位的人体数据：颈动脉、心脏（闭环用）与乳腺、下肢静脉
（其他任务用）。向导师汇报前要回答：

  1. 每个部位多少名受试者、多少次存储、多大、哪几天采的。
  2. 颈动脉、心脏：每名受试者按逐帧列表采齐了没有——频率 / 焦点 / 深度三组扫描各做了几组
     （≈ 切面数），后端阶梯（表三 / 心脏二）做了多少人；成像模式是否全是谐波；
     专家调优帧的六个参数分布在哪里。
  3. 数据完整性：缺文件、BC0 大小不符、全黑帧、受试者之间的帧数离群。
  4. 乳腺、下肢静脉：只盘点受试者数、视频数、时长，不进闭环。

    做法

第一遍扫描把每次存储的参数写进 data/invivo_inventory/*.csv（缓存，--rescan 重扫），
之后的分析只读缓存。颈动脉原始数据是扁平的采集目录，受试者由 Algo_PartitionInfo.pdt 里
FILEPATH 的场次号（如 20261009_142958_00001）还原；心脏原始数据按场次目录存放，参数在
ImageParam.txt / PatientParam.txt（JSON）里。

扫描块的识别：在按时间排好的帧序列上找连续窗口，其中只有被扫的那一项在变、其余前端参数
不变，且取遍方案规定的全部档位，记为一组完整扫描。深度扫描不要求焦点不变（主机会随深度
自动挪焦点）。

    用法

    python tests/measure_invivo_acquisition.py              # 用缓存
    python tests/measure_invivo_acquisition.py --rescan     # 重扫 E 盘（约十几分钟）

报告写在脚本旁边的 measure_invivo_acquisition.txt（UTF-8），终端输出同内容（ASCII）。
"""

import argparse
import collections
import concurrent.futures
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from bmode_opt import data_paths as DPATH
from bmode_opt.hisense_loader import parse_pdt

REPORT_PATH = os.path.splitext(os.path.abspath(__file__))[0] + ".txt"
DEFAULT_ROOT = str(DPATH.HISENSE_ROOT)
CACHE_DIR = os.path.join(ROOT, "data", "invivo_inventory")
LINES = []

# 新版主机屏幕上的谐波频率档：(下限, 上限) MHz -> 档名，见 carotid_invivo_acquisition_checklist.md
CAROTID_FREQ_NAMES = {
    (6.2, 11.4): "HPen", (6.6, 12.2): "HPenMid", (7.0, 13.0): "FHNor",
    (7.4, 13.8): "PHNor", (8.0, 17.0): "HRes",
}
CAROTID_FOCUS_LADDER = (10, 15, 20, 25, 30, 35)
CAROTID_DEPTH_LADDER = (25, 33, 42, 50, 59, 67)
TGC_FLAT = 127


def emit(text=""):
    LINES.append(str(text))


# ---------------------------------------------------------------- 扫描：颈动脉

def _decode_chars(value):
    """ProbeNameActual / ExamMode 这类以逗号分隔的字符码。"""
    try:
        return "".join(chr(int(c)) for c in value.split(",") if c.strip() and int(c) > 0)
    except ValueError:
        return value


def scan_carotid_capture(path):
    row = {"capture": os.path.basename(path), "error": ""}
    try:
        part = parse_pdt(os.path.join(path, "Algo_PartitionInfo.pdt"))
        fe = parse_pdt(os.path.join(path, "Algo_FeParam.pdt"))
        be = parse_pdt(os.path.join(path, "BEParam.pdt"))
    except Exception as exc:  # 缺文件也要记下来，不中断整次扫描
        row["error"] = f"param:{type(exc).__name__}"
        return row

    def leaf(params, name):
        hits = [v for k, v in params.items() if k == name or k.endswith("." + name)]
        return hits[0] if hits else ""

    filepath = leaf(part, "FILEPATH")
    m = re.search(r"/(\d{8}_\d{6}_\d{5})/", filepath)
    row["session"] = m.group(1) if m else ""
    # 电影回放导出（PageNum > 1）时 Line / Point / PageSize / TimeStamp 都是逐页列表
    row["line"] = leaf(part, "Line").split(",")[0]
    row["point"] = leaf(part, "Point").split(",")[0]
    row["page_num"] = leaf(part, "PageNum")
    page_size = leaf(part, "PageSize").split(",")[0]
    stamps = [s for s in leaf(part, "TimeStamp").split(",") if s.strip()]
    row["ts_span"] = int(stamps[-1]) - int(stamps[0]) if len(stamps) > 1 else 0
    row["depth_mm"] = leaf(fe, "BDisplayDepth")
    foci = leaf(fe, "BFocusArrayPos").split(",")
    row["focus_num"] = leaf(fe, "BFocusNumValue")
    row["focus_mm"] = foci[0] if foci else ""
    row["freq_lo"] = leaf(fe, "BFreqValueLowUI")
    row["freq_hi"] = leaf(fe, "BFreqValueHighUI")
    row["image_mode"] = leaf(fe, "BImageMode")
    row["probe"] = _decode_chars(leaf(fe, "ProbeNameActual"))
    row["exam"] = _decode_chars(leaf(fe, "ExamMode"))
    row["frs"] = leaf(fe, "FRS")
    row["mi"] = leaf(fe, "MI")
    row["gain"] = leaf(be, "BUIGainLevel")
    row["dr"] = leaf(be, "UIDynamicRangeLevel")
    row["tgc"] = ",".join(leaf(be, f"BTgc{i}Level") for i in range(1, 9))

    bc0 = os.path.join(path, "Algo_BC0.bin")
    try:
        size = os.path.getsize(bc0)
        row["bc0_bytes"] = size
        expected = int(page_size) * int(row["page_num"] or 1)
        if size != expected:
            row["error"] = "bc0_size"
        raw = np.fromfile(bc0, dtype=np.uint16, count=int(page_size) // 2)
        row["bc0_mean"] = f"{raw.mean():.1f}"
        row["bc0_p99"] = f"{np.percentile(raw, 99):.0f}"
    except Exception as exc:
        row["error"] = row["error"] or f"bc0:{type(exc).__name__}"
    row["has_screenshot"] = int(os.path.exists(os.path.join(path, "Screenthum.bmp")))
    return row


CAROTID_FIELDS = ["capture", "session", "error", "line", "point", "page_num", "ts_span", "bc0_bytes", "depth_mm",
                  "focus_num", "focus_mm", "freq_lo", "freq_hi", "image_mode", "probe", "exam",
                  "frs", "mi", "gain", "dr", "tgc", "bc0_mean", "bc0_p99", "has_screenshot"]


# ---------------------------------------------------------------- 扫描：心脏

def _json(path):
    with open(path, "rb") as fh:
        text = fh.read().decode("utf-8", errors="replace")
    return json.loads(text)


def _first_key(obj, key):
    """在嵌套 JSON 里找第一个名为 key 的字段。"""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if key in cur:
                return cur[key]
            stack.extend(reversed(list(cur.values())))
        elif isinstance(cur, list):
            stack.extend(reversed(cur))
    return None


def scan_cardiac_capture(args):
    session, path = args
    row = {"session": session, "capture": os.path.basename(path), "error": ""}
    files = set(os.listdir(path))
    row["has_cine"] = int("B_CinePart0.bin" in files)
    row["has_avi"] = int("UltraSound.avi" in files)
    try:
        pat = _json(os.path.join(path, "PatientParam.txt"))["PatientInfo"]
        row["age"] = pat.get("age", "")
        row["sex"] = pat.get("sex", "")
        row["probe"] = pat.get("probeName", "")
        row["exam"] = pat.get("subExamTypeEn", "")
        row["patient_id"] = pat.get("patientId", "")
        var = _json(os.path.join(path, "VariantParam.txt"))["VariantParam"]
        row["frames"] = var.get("totalFrame", [0])[0]
        img = _json(os.path.join(path, "ImageParam.txt"))
        row["depth_cm"] = _first_key(img, "RxDepth_Temp")
        # 频率扫描时变的是 TxFreq_Temp / TxFreq_Phase_DB，TxFreq_DB 全程不变
        row["freq_idx"] = _first_key(img, "TxFreq_DB")
        row["tx_freq"] = _first_key(img, "TxFreq_Temp")
        row["tx_phase"] = _first_key(img, "TxFreq_Phase_DB")
        row["mi"] = _first_key(img, "MI_Temp")
        row["thi"] = _first_key(img, "THIMode_DB")
        focus = _first_key(img, "TxFocus_DB") or [""]
        row["focus_cm"] = focus[0]
        row["focus_num"] = sum(1 for f in focus if f)
        row["gain"] = _first_key(img, "GainRotary_Show")
        row["dr"] = _first_key(img, "Dynamic_Log_DB")
        row["tgc"] = ",".join(str(v) for v in (_first_key(img, "TGC_DB") or []))
        row["fps"] = _first_key(img, "FPS_Temp")
    except Exception as exc:
        row["error"] = f"param:{type(exc).__name__}"
    try:
        row["cine_bytes"] = os.path.getsize(os.path.join(path, "B_CinePart0.bin"))
    except OSError:
        row["cine_bytes"] = 0
    return row


CARDIAC_FIELDS = ["session", "capture", "error", "patient_id", "age", "sex", "probe", "exam",
                  "frames", "depth_cm", "freq_idx", "tx_freq", "tx_phase", "mi", "thi", "focus_cm", "focus_num", "gain", "dr",
                  "tgc", "fps", "has_cine", "has_avi", "cine_bytes"]


# ---------------------------------------------------------------- 扫描：视频类

def video_info(path):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"frames": 0, "fps": 0, "width": 0, "height": 0}
    info = {"frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def scan_files(top, subject_of):
    """列出 top 下全部文件；subject_of(relpath) 给出受试者目录名。"""
    rows = []
    for dirpath, _, filenames in os.walk(top):
        for name in filenames:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, top).replace("\\", "/")
            row = {"subject": subject_of(rel), "file": rel, "ext": os.path.splitext(name)[1].lower(),
                   "bytes": os.path.getsize(full)}
            if row["ext"] == ".avi":
                row.update(video_info(full))
            rows.append(row)
    return rows


FILE_FIELDS = ["subject", "file", "ext", "bytes", "frames", "fps", "width", "height"]


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_scan(root, workers, parts):
    if "carotid" in parts:
        _scan_carotid(root, workers)
    if "cardiac" in parts:
        _scan_cardiac(root, workers)
    video_parts = {"files"} | set(VIDEO_DIRS)
    if parts & video_parts:
        _scan_video_dirs(root, None if "files" in parts else parts & set(VIDEO_DIRS))


def _scan_carotid(root, workers):
    carotid_root = os.path.join(root, "颈动脉", "原始数据")
    dirs = sorted(os.path.join(carotid_root, d) for d in os.listdir(carotid_root))
    print(f"scanning {len(dirs)} carotid captures ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        rows = []
        for i, row in enumerate(pool.map(scan_carotid_capture, dirs, chunksize=16)):
            rows.append(row)
            if (i + 1) % 2000 == 0:
                print(f"  {i + 1}/{len(dirs)}", flush=True)
    write_csv(os.path.join(CACHE_DIR, "carotid_captures.csv"), rows, CAROTID_FIELDS)


def _scan_cardiac(root, workers):
    cardiac_root = os.path.join(root, "心脏", "原始数据")
    jobs = []
    for session in sorted(os.listdir(cardiac_root)):
        for dirpath, dirnames, filenames in os.walk(os.path.join(cardiac_root, session)):
            if "ImageParam.txt" in filenames or "PatientParam.txt" in filenames:
                jobs.append((session, dirpath))
    print(f"scanning {len(jobs)} cardiac captures ...", flush=True)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(scan_cardiac_capture, jobs, chunksize=8))
    write_csv(os.path.join(CACHE_DIR, "cardiac_captures.csv"), rows, CARDIAC_FIELDS)


# 视频类目录：缓存名 -> 相对 root 的路径；受试者为第一级子目录
VIDEO_DIRS = {
    "carotid_video": ("颈动脉", "视频"),
    "carotid_dicom": ("颈动脉", "dicom"),
    "cardiac_video": ("心脏", "视频"),
    "breast": ("乳腺",),
    "vein": ("下肢静脉", "视频"),
}


def _scan_video_dirs(root, names=None):
    """names 为 None 时扫全部视频类目录，否则只扫给定的几个（如只更新了下肢静脉）。"""
    first = lambda rel: rel.split("/")[0]
    for name, rel in VIDEO_DIRS.items():
        if names is not None and name not in names:
            continue
        print(f"scanning {name} ...", flush=True)
        write_csv(os.path.join(CACHE_DIR, f"{name}_files.csv"),
                  scan_files(os.path.join(root, *rel), first), FILE_FIELDS)


# ---------------------------------------------------------------- 分析工具

def fnum(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def capture_day(name):
    return name[:8]


def dist(values, fmt="{:g}", top=12):
    """取值分布：值(次数)，按次数降序。"""
    counts = collections.Counter(values)
    items = counts.most_common(top)
    text = ", ".join(f"{fmt.format(k) if isinstance(k, (int, float)) else k}({n})" for k, n in items)
    if len(counts) > top:
        text += f", ... {len(counts) - top} more"
    return text


def quantiles(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return "n/a"
    q = np.percentile(a, [0, 25, 50, 75, 100])
    return (f"min {q[0]:g} / p25 {q[1]:g} / median {q[2]:g} / p75 {q[3]:g} / max {q[4]:g}"
            f"  (mean {a.mean():.1f}, n={a.size})")


def find_sweeps(seq, key, fixed, ladder, min_distinct):
    """在帧序列里找扫描窗口：key 取 >= min_distinct 个档位，fixed 中各项不变。

    贪心、不重叠，返回 [(起点, 终点, 取到的档位集合)]。ladder 为 None 时不限定档位。
    """
    out = []
    i, n = 0, len(seq)
    while i < n:
        best = None
        j = i
        values = []
        while j < n and all(seq[j][f] == seq[i][f] for f in fixed):
            v = seq[j][key]
            if ladder is not None and v not in ladder:
                break
            values.append(v)
            if len(set(values)) >= min_distinct:
                best = j
                if ladder is not None and set(values) >= set(ladder):
                    break
            j += 1
        if best is not None:
            out.append((i, best, frozenset(values[: best - i + 1])))
            i = best + 1
        else:
            i += 1
    return out


# ---------------------------------------------------------------- 分析：颈动脉

def carotid_freq_name(row):
    lo, hi = fnum(row["freq_lo"]), fnum(row["freq_hi"])
    if lo is None:
        return "?"
    return CAROTID_FREQ_NAMES.get((round(lo, 1), round(hi, 1)), f"{lo:g}-{hi:g}")


MIN_SINGLE_FRAMES = 20  # 少于这么多单帧的场次当作试机 / 中断，不算受试者
PRESET_REF = (42, 15, "FHNor")  # 颈动脉预设：显示深度 41.9 mm、焦点 15 mm、FHNor


def capture_minutes(name):
    """采集目录名的前 12 位 YYYYMMDDHHMM 加秒，换成当天的分钟数。"""
    return int(name[8:10]) * 60 + int(name[10:12]) + int(name[12:14]) / 60


def tgc_shape(levels):
    """表三 / 心脏二的两种 TGC 改法：深处拉高、浅处拉低。"""
    lv = [int(fnum(t, TGC_FLAT)) for t in levels]
    deep_up = all(v > TGC_FLAT + 50 for v in lv[4:])
    shallow_down = all(v < TGC_FLAT - 50 for v in lv[:4])
    if deep_up and shallow_down:
        return "deep up + shallow down"
    if deep_up:
        return "deep up"
    if shallow_down:
        return "shallow down"
    return "flat" if all(v == TGC_FLAT for v in lv) else "other"


def backend_frames(seq, require_return):
    """后端阶梯帧：前端（深度、焦点、频率）与锚帧相同、后端（增益、动态范围、TGC）不同。

    锚帧是最近一帧后端未改动的帧。require_return 时还要求下一帧回到锚帧的后端
    （颈动脉表三是「改一项、存、拨回、存」），这样医生换切面时顺手改增益不会被算进来。
    返回 [(帧, 锚帧)]。
    """
    out, anchor = [], None
    front = lambda r: (r["depth"], r["focus"], r["freq"])
    back = lambda r: (r["g"], r["d"], r["tgc"])
    for i, r in enumerate(seq):
        if anchor is not None and front(r) == front(anchor) and back(r) != back(anchor):
            nxt = seq[i + 1] if i + 1 < len(seq) else None
            if not require_return or (nxt is not None and back(nxt) == back(anchor)):
                out.append((r, anchor))
                continue
        anchor = r
    return out


def analyse_carotid(rows):
    emit("=" * 78)
    emit("CAROTID (closed-loop task) -- raw console exports, 颈动脉/原始数据")
    emit("=" * 78)
    errors = collections.Counter(r["error"] for r in rows if r["error"])
    emit(f"captures (export dirs): {len(rows)}   with errors: {sum(errors.values())} {dict(errors)}")

    good = [r for r in rows if not r["error"].startswith("param")]
    for r in good:
        r["depth"] = round(fnum(r["depth_mm"], 0))
        r["focus"] = round(fnum(r["focus_mm"], 0))
        r["freq"] = carotid_freq_name(r)
        r["g"] = int(fnum(r["gain"], -1))
        r["d"] = int(fnum(r["dr"], -1))
        r["tgc_flat"] = all(int(fnum(t, -1)) == TGC_FLAT for t in r["tgc"].split(","))
        r["mean"] = fnum(r.get("bc0_mean"))
        r["pages"] = int(fnum(r["page_num"], 1))
        r["bytes"] = int(fnum(r.get("bc0_bytes"), 0))
    singles = [r for r in good if r["pages"] == 1]
    cines = [r for r in good if r["pages"] > 1]

    sessions = collections.defaultdict(list)
    for r in good:
        sessions[r["session"] or "UNKNOWN"].append(r)
    for seq in sessions.values():
        seq.sort(key=lambda r: r["capture"])
    main = {s: v for s, v in sessions.items() if sum(r["pages"] == 1 for r in v) >= MIN_SINGLE_FRAMES}
    short = {s: v for s, v in sessions.items() if s not in main}
    emit(f"console sessions: {len(sessions)}  -> subjects (>= {MIN_SINGLE_FRAMES} single frames): {len(main)}")
    emit("short sessions (test / aborted / extra exam): " + ", ".join(
        f"{s}[{len(v)} caps; {dist([r['exam'] for r in v])}; {dist([r['probe'] for r in v])}]"
        for s, v in sorted(short.items())))
    subj_days = collections.Counter(s[:8] for s in main)
    emit("subjects per day: " + ", ".join(f"{d[4:6]}-{d[6:]}:{n}" for d, n in sorted(subj_days.items())))
    days = collections.Counter(capture_day(r["capture"]) for r in good)
    emit("captures per day: " + ", ".join(f"{d[4:6]}-{d[6:]}:{n}" for d, n in sorted(days.items())))
    emit(f"single-frame captures: {len(singles)}   cine captures (multi-page BC0): {len(cines)}")
    emit(f"BC0 volume: single {sum(r['bytes'] for r in singles) / 1e9:.1f} GB, "
         f"cine {sum(r['bytes'] for r in cines) / 1e9:.1f} GB "
         f"(+ ~6 MB Screenthum.bmp per capture = {len(good) * 6.1e6 / 1e9:.0f} GB)")
    emit("")
    emit("-- acquisition settings over all single-frame captures")
    emit(f"probe          : {dist([r['probe'] for r in good])}")
    emit(f"exam preset    : {dist([r['exam'] for r in good])}")
    emit(f"image mode     : {dist([('harmonic' if r['image_mode'] == '1' else 'fundamental' if r['image_mode'] == '0' else r['image_mode']) for r in good])}")
    emit(f"BC0 lines x pts: {dist([r['line'] + 'x' + r['point'] for r in good])}")
    emit(f"focus count    : {dist([r['focus_num'] for r in good])}")
    emit(f"depth (mm)     : {dist([r['depth'] for r in singles], top=16)}")
    emit(f"focus (mm)     : {dist([r['focus'] for r in singles], top=16)}")
    emit(f"frequency band : {dist([r['freq'] for r in singles])}")
    emit(f"TGC all flat   : {sum(r['tgc_flat'] for r in singles)} / {len(singles)}")
    emit(f"dynamic range  : {dist([r['d'] for r in singles])}")
    emit(f"gain           : {dist([r['g'] for r in singles], top=16)}")
    emit(f"screenshot     : {sum(int(r['has_screenshot']) for r in good)} / {len(good)} have Screenthum.bmp")
    emit(f"BC0 mean level : {quantiles([r['mean'] for r in singles if r['mean'] is not None])}")
    emit(f"MI             : {dist([round(fnum(r['mi'], 0), 2) for r in singles])}")

    emit("")
    emit("-- cine captures (multi-frame BC0, pre gain / TGC / DR)")
    emit(f"cines per subject : {quantiles([sum(r['pages'] > 1 for r in v) for v in main.values()])}")
    emit(f"frames per cine   : {quantiles([r['pages'] for r in cines])}")
    dur = [r["pages"] / fnum(r["frs"], 44) for r in cines if fnum(r["frs"], 0)]
    emit(f"seconds per cine  : {quantiles([round(d, 1) for d in dur])}  total {sum(dur) / 60:.0f} min")
    emit(f"cine exam preset  : {dist([r['exam'] for r in cines])}")

    emit("")
    emit("-- per subject: sweep blocks (checklist table 2)")
    per = []
    for sid, allseq in sorted(main.items()):
        seq = [r for r in allseq if r["pages"] == 1 and r["exam"] == "Carotid"]
        freq_sw = find_sweeps(seq, "freq", ("depth", "focus", "g", "d"), set(CAROTID_FREQ_NAMES.values()), 5)
        focus_sw = find_sweeps(seq, "focus", ("depth", "freq", "g", "d"), set(CAROTID_FOCUS_LADDER), 6)
        # 颈动脉深度扫描时焦点不动；不固定焦点的话窗口会把焦点扫描帧一起吞进去
        depth_sw = find_sweeps(seq, "depth", ("freq", "focus", "g", "d"), set(CAROTID_DEPTH_LADDER), 6)
        order = "".join(t for _, t in sorted([(a, "F") for a, _, _ in freq_sw] + [(a, "Z") for a, _, _ in focus_sw]
                                            + [(a, "D") for a, _, _ in depth_sw]))
        # 医生参考帧：频率扫描窗口的第一帧（回到第 1 帧设置的那一帧），其深度、焦点、增益、动态范围
        # 即医生值；频率与窗口后的回位帧一致时才采信
        refs = []
        for a, b, _ in freq_sw:
            ref = seq[a]
            after = seq[b + 1] if b + 1 < len(seq) else None
            refs.append((ref, after is not None and after["freq"] == ref["freq"]))
        ref_g = collections.Counter(r["g"] for r, _ in refs).most_common(1)[0][0] if refs else None
        ref_d = collections.Counter(r["d"] for r, _ in refs).most_common(1)[0][0] if refs else None
        backend = backend_frames(seq, require_return=True)
        times = [capture_minutes(r["capture"]) for r in allseq]
        per.append({
            "sid": sid, "n": len(allseq), "n_single": len(seq), "freq_sw": len(freq_sw),
            "focus_sw": len(focus_sw), "depth_sw": len(depth_sw), "order": order, "refs": refs,
            "ref_g": ref_g, "backend": backend, "minutes": max(times) - min(times),
            "thyroid": sum(r["exam"] == "Thyroid" for r in allseq),
            "probes": {r["probe"] for r in allseq},
        })

    emit(f"captures per subject        : {quantiles([p['n'] for p in per])}")
    emit(f"carotid single frames/subj. : {quantiles([p['n_single'] for p in per])}")
    emit(f"exam duration (min, first..last save): {quantiles([round(p['minutes']) for p in per])}")
    emit(f"complete freq sweeps (5 bands) / subject : {quantiles([p['freq_sw'] for p in per])}")
    emit(f"complete focus sweeps (10..35) / subject : {quantiles([p['focus_sw'] for p in per])}")
    emit(f"complete depth sweeps (25..67) / subject : {quantiles([p['depth_sw'] for p in per])}")
    views = [min(p["freq_sw"], p["focus_sw"], p["depth_sw"]) for p in per]
    emit(f"complete views (all three sweeps) / subject: {dist(views, top=20)}   total {sum(views)}")
    emit(f"sweep order within a view (first 3 letters; D=depth F=freq Z=focus): {dist([p['order'][:3] for p in per])}")
    emit(f"subjects with 8 or more complete views (protocol: 8 core views, both sides): {sum(v >= 8 for v in views)}")
    emit("subjects with fewer than 4 complete views: " + ", ".join(
        f"{p['sid']}(F{p['freq_sw']}/Z{p['focus_sw']}/D{p['depth_sw']})" for p, v in zip(per, views) if v < 4))
    emit(f"table 1 (probe in air, 67 mm, 5 bands) frames: none found; lowest BC0 mean of any single frame = "
         f"{min(r['mean'] for r in singles if r['mean'] is not None):.0f} (tissue median ~15000)")

    emit("")
    emit("-- doctor reference frames (first frame of each freq sweep = 'back to frame 1 settings')")
    refs = [r for p in per for r, _ in p["refs"]]
    emit(f"reference frames: {len(refs)}   frequency confirmed by the return frame: "
         f"{sum(ok for p in per for _, ok in p['refs'])}")
    emit(f"  depth (mm)     : {dist([r['depth'] for r in refs], top=16)}")
    emit(f"  focus (mm)     : {dist([r['focus'] for r in refs], top=16)}")
    emit(f"  frequency band : {dist([r['freq'] for r in refs])}")
    emit(f"  gain           : {dist([r['g'] for r in refs], top=16)}")
    emit(f"  dynamic range  : {dist([r['d'] for r in refs])}")
    emit(f"  TGC flat       : {sum(r['tgc_flat'] for r in refs)} / {len(refs)}")
    at_preset = [(r["depth"], r["focus"], r["freq"]) == PRESET_REF for r in refs]
    emit(f"  depth/focus/freq all equal to preset (42 mm, 15 mm, FHNor): {sum(at_preset)} / {len(refs)}")
    emit(f"  subjects whose every view sits on the preset front end: "
         f"{sum(all((r['depth'], r['focus'], r['freq']) == PRESET_REF for r, _ in p['refs']) for p in per if p['refs'])}"
         f" / {sum(1 for p in per if p['refs'])}")
    emit(f"  per-subject doctor gain: {dist([p['ref_g'] for p in per if p['ref_g'] is not None], top=16)}")

    emit("")
    emit("-- back-end ladder (checklist table 3, planned for ~30 subjects)")
    bk = [p for p in per if len(p["backend"]) >= 4]
    emit(f"subjects with >= 4 back-end frames (change one item, save, change back): {len(bk)}")
    emit(f"  dates: {dist([p['sid'][:8] for p in bk], top=20)}")
    emit(f"  back-end frames per such subject: {dist([len(p['backend']) for p in bk])}  total {sum(len(p['backend']) for p in bk)}")
    bf = [(r, a) for p in bk for r, a in p["backend"]]
    emit(f"  gain offset from anchor : {dist([r['g'] - a['g'] for r, a in bf if r['g'] != a['g']])}")
    emit(f"  dynamic range values    : {dist([r['d'] for r, a in bf if r['d'] != a['d']])}")
    emit(f"  TGC patterns            : {dist([tgc_shape(r['tgc'].split(',')) for r, a in bf if r['tgc'] != a['tgc']])}")
    emit(f"  ladders (6 changed frames) per subject: {dist([round(len(p['backend']) / 6, 1) for p in bk])}")
    emit(f"subjects with 1-3 stray back-end frames: {sum(1 for p in per if 0 < len(p['backend']) < 4)}")

    emit("")
    emit("-- off-protocol content")
    emit(f"Thyroid-preset captures: {sum(p['thyroid'] for p in per)} in {sum(p['thyroid'] > 0 for p in per)} subjects")
    emit(f"subjects using a probe other than L15-4: "
         + ", ".join(f"{p['sid']}{sorted(p['probes'])}" for p in per if p["probes"] != {"L15-4"}))
    return per


# ---------------------------------------------------------------- 分析：心脏

def analyse_cardiac(rows):
    emit("")
    emit("=" * 78)
    emit("CARDIAC (closed-loop task) -- raw console exports, 心脏/原始数据")
    emit("=" * 78)
    errors = collections.Counter(r["error"] for r in rows if r["error"])
    emit(f"captures: {len(rows)}   errors: {dict(errors)}")
    good = [r for r in rows if not r["error"]]
    for r in good:
        r["depth"] = fnum(r["depth_cm"])
        r["focus"] = fnum(r["focus_cm"])
        r["freq"] = r["tx_freq"]
        r["g"] = int(fnum(r["gain"], -1))
        r["d"] = int(fnum(r["dr"], -1))
        r["nframes"] = int(fnum(r["frames"], 0))
        r["tgc_flat"] = len(set(r["tgc"].split(","))) == 1
    sessions = collections.defaultdict(list)
    for r in good:
        sessions[r["session"]].append(r)
    for seq in sessions.values():
        seq.sort(key=lambda r: r["capture"])
    main = {s: v for s, v in sessions.items() if len(v) >= MIN_SINGLE_FRAMES}
    short = {s: v for s, v in sessions.items() if s not in main}
    emit(f"session dirs: {len(sessions)}  -> subjects (>= {MIN_SINGLE_FRAMES} captures): {len(main)}")
    emit("short sessions: " + ", ".join(
        f"{s}[{len(v)} caps; {dist([r['probe'] for r in v])}; {dist([r['exam'] for r in v])}]"
        for s, v in sorted(short.items())))
    subj_days = collections.Counter(s[:8] for s in main)
    emit("subjects per day: " + ", ".join(f"{d[4:6]}-{d[6:]}:{n}" for d, n in sorted(subj_days.items())))
    first = [seq[0] for seq in main.values()]
    emit(f"age filled on console: {sum(fnum(r['age'], -1) > 0 for r in first)} / {len(first)} "
         f"(values: {dist([r['age'] for r in first if fnum(r['age'], -1) > 0])})")
    emit(f"probe: {dist([r['probe'] for r in good])}   exam: {dist([r['exam'] for r in good])}")
    emit(f"harmonic flag THIMode_DB: {dist([r['thi'] for r in good])}")
    singles = [r for r in good if r["nframes"] == 1]
    cines = [r for r in good if r["nframes"] > 1]
    emit(f"single frames: {len(singles)}   cines: {len(cines)}")
    emit(f"cine length (frames): {quantiles([r['nframes'] for r in cines])}")
    dur = [r["nframes"] / fnum(r["fps"], 30) for r in cines if fnum(r["fps"], 0)]
    emit(f"cine length (s, frames / FPS): {quantiles([round(d, 1) for d in dur])}  total {sum(dur) / 60:.0f} min")
    emit(f"  short cines (< 200 frames): {sum(r['nframes'] < 200 for r in cines)}   long: {sum(r['nframes'] >= 200 for r in cines)}")
    emit(f"raw cine volume (B_CinePart0.bin): single {sum(int(fnum(r['cine_bytes'], 0)) for r in singles) / 1e9:.1f} GB, "
         f"cine {sum(int(fnum(r['cine_bytes'], 0)) for r in cines) / 1e9:.1f} GB")
    emit(f"depth (cm)   : {dist([r['depth'] for r in singles], top=20)}")
    emit(f"focus (cm)   : {dist([r['focus'] for r in singles], top=20)}")
    emit(f"tx freq code : {dist([r['freq'] for r in singles])}   (TxFreq_Temp)")
    emit(f"gain (screen): {quantiles([r['g'] for r in singles])}   at 255: {sum(r['g'] == 255 for r in singles)}")
    emit(f"dyn range dB : {dist([r['d'] for r in singles])}")
    emit(f"TGC flat     : {sum(r['tgc_flat'] for r in singles)} / {len(singles)}")
    emit(f"frame rate   : {quantiles([fnum(r['fps']) for r in good])}")

    per = []
    for sid, allseq in sorted(main.items()):
        seq = [r for r in allseq if r["nframes"] == 1]
        freq_sw = find_sweeps(seq, "freq", ("depth", "focus", "g", "d"), None, 3)
        focus_sw = find_sweeps(seq, "focus", ("depth", "freq", "g", "d"), None, 3)
        depth_sw = find_sweeps(seq, "depth", ("freq", "g", "d"), None, 3)
        refs = [seq[a] for a, _, _ in freq_sw]
        ref_g = collections.Counter(r["g"] for r in refs).most_common(1)[0][0] if refs else None
        ref = refs[0] if refs else None
        backend = backend_frames(seq, require_return=False)
        # 频率扫描之后（焦点、深度扫描时）停在哪一档；方案改动一要求拨回医生的那一档
        after = [seq[b + 1]["freq"] for a, b, _ in freq_sw if b + 1 < len(seq)]
        # 医生的临床设置：频率扫描前最近一段电影
        prev_cine = []
        for a, _, _ in freq_sw:
            t = seq[a]["capture"]
            before = [r for r in allseq if r["nframes"] > 1 and r["capture"] < t]
            if before:
                prev_cine.append(before[-1])
        cin = [r for r in allseq if r["nframes"] > 1]
        times = [capture_minutes(r["capture"]) for r in allseq]
        per.append({"sid": sid, "n": len(allseq), "n_single": len(seq), "freq_sw": freq_sw,
                    "focus_sw": focus_sw, "depth_sw": depth_sw, "refs": refs, "ref_g": ref_g,
                    "backend": backend, "after": after, "prev_cine": prev_cine, "cines": len(cin),
                    "short_cines": sum(r["nframes"] < 200 for r in cin),
                    "minutes": max(times) - min(times)})
    emit("")
    emit("-- per subject")
    emit(f"captures per subject     : {quantiles([p['n'] for p in per])}")
    emit(f"single frames per subject: {quantiles([p['n_single'] for p in per])}")
    emit(f"cines per subject        : {quantiles([p['cines'] for p in per])}  (short {quantiles([p['short_cines'] for p in per])})")
    emit(f"exam duration (min)      : {quantiles([round(p['minutes']) for p in per])}")
    emit(f"freq sweeps (>=3 codes)  / subject: {dist([len(p['freq_sw']) for p in per])}")
    emit(f"  codes per sweep: {dist([len(s[2]) for p in per for s in p['freq_sw']])}; code sets: {dist([','.join(sorted(s[2])) for p in per for s in p['freq_sw']], top=6)}")
    emit(f"focus sweeps (>=3 foci)  / subject: {dist([len(p['focus_sw']) for p in per])}")
    emit(f"  foci per sweep : {dist([len(s[2]) for p in per for s in p['focus_sw']])}")
    emit(f"depth sweeps (>=3 depths)/ subject: {dist([len(p['depth_sw']) for p in per])}")
    emit(f"  depths per sweep: {dist([len(s[2]) for p in per for s in p['depth_sw']])}")
    views = [min(len(p["freq_sw"]), len(p["focus_sw"]), len(p["depth_sw"])) for p in per]
    emit(f"complete sweep blocks (all three) / subject: {dist(views, top=20)}   total {sum(views)}")

    emit("")
    emit("-- sweep centre (settings held during each freq sweep) and the doctor's clinical setting")
    refs = [r for p in per for r in p["refs"]]
    emit(f"freq sweeps: {len(refs)}")
    emit(f"  depth (cm)   : {dist([r['depth'] for r in refs], top=20)}")
    emit(f"  focus (cm)   : {dist([r['focus'] for r in refs], top=20)}")
    emit(f"  gain         : {quantiles([r['g'] for r in refs])}   at 255: {sum(r['g'] == 255 for r in refs)}")
    emit(f"  dyn range    : {dist([r['d'] for r in refs])}")
    emit(f"  TGC flat     : {sum(r['tgc_flat'] for r in refs)} / {len(refs)}")
    emit(f"  freq code held for the following focus / depth sweeps: {dist([c for p in per for c in p['after']])}")
    pc = [(c, p) for p in per for c in p["prev_cine"]]
    emit(f"  doctor's freq code (cine just before the sweep)      : {dist([c['freq'] for c, _ in pc])}")
    same = sum(1 for p in per for c, f in zip(p["prev_cine"], p["after"]) if c["freq"] == f)
    emit(f"  sweeps whose later focus / depth sweeps ran at the doctor's freq: {same} / {len(pc)}")
    by_day = []
    for d in sorted({p["sid"][:8] for p in per}):
        ok = sum(1 for p in per if p["sid"][:8] == d for c, f in zip(p["prev_cine"], p["after"]) if c["freq"] == f)
        tot = sum(min(len(p["prev_cine"]), len(p["after"])) for p in per if p["sid"][:8] == d)
        by_day.append(f"{d[4:6]}-{d[6:]}:{ok}/{tot}")
    emit("  by date, returned to doctor's freq: " + ", ".join(by_day))
    same_depth = sum(c["depth"] == p["refs"][i]["depth"] for p in per for i, c in enumerate(p["prev_cine"]))
    emit(f"  doctor's depth (cine before sweep) == sweep depth: {same_depth} / {len(pc)}")
    lc = [r for r in good if r["nframes"] >= 200]
    emit(f"long clinical cines: freq code {dist([r['freq'] for r in lc])}; gain {quantiles([r['g'] for r in lc])}; at 255: {sum(r['g'] == 255 for r in lc)}")

    emit("")
    emit("-- back-end ladder (part two, planned for ~20 subjects)")
    bk = [p for p in per if len(p["backend"]) >= 3]
    emit(f"subjects with >= 3 back-end frames at an unchanged front end: {len(bk)}  dates: {dist([p['sid'][:8] for p in bk], top=20)}")
    emit(f"  back-end frames per such subject: {dist([len(p['backend']) for p in bk])}  total {sum(len(p['backend']) for p in bk)}")
    bf = [(r, a) for p in bk for r, a in p["backend"]]
    emit(f"  gain offset from anchor: {dist([r['g'] - a['g'] for r, a in bf if r['g'] != a['g']])}")
    emit(f"  DR values              : {dist([r['d'] for r, a in bf if r['d'] != a['d']])}")
    emit(f"  TGC changed            : {sum(r['tgc'] != a['tgc'] for r, a in bf)} frames")
    emit(f"subjects with 1-2 stray back-end frames: {sum(1 for p in per if 0 < len(p['backend']) < 3)}")
    small = sorted((p["n"], p["sid"]) for p in per if p["n"] < 45)
    emit(f"subjects with < 45 captures ({len(small)}): " + ", ".join(f"{s}:{k}" for k, s in small))
    return per


# ---------------------------------------------------------------- 分析：视频类

def analyse_files(name, rows, title):
    emit("")
    emit("=" * 78)
    emit(title)
    emit("=" * 78)
    subjects = collections.defaultdict(list)
    for r in rows:
        subjects[r["subject"]].append(r)
    total = sum(int(r["bytes"]) for r in rows)
    emit(f"subject dirs: {len(subjects)}   files: {len(rows)}   size: {total / 1e9:.1f} GB")
    emit(f"file types  : {dist([r['ext'] for r in rows])}")
    avis = [r for r in rows if r["ext"] == ".avi"]
    if avis:
        dur = [int(fnum(r["frames"], 0)) / fnum(r["fps"], 1) for r in avis if fnum(r["fps"], 0)]
        emit(f"videos      : {len(avis)}; per subject {quantiles([sum(1 for x in v if x['ext'] == '.avi') for v in subjects.values()])}")
        emit(f"duration (s): {quantiles([round(d, 1) for d in dur])}  total {sum(dur) / 60:.1f} min")
        emit(f"resolution  : {dist([r['width'] + 'x' + r['height'] for r in avis])}   fps: {dist([round(fnum(r['fps'], 0)) for r in avis])}")
    days = collections.Counter(s[:8] for s in subjects)
    emit("subjects per day: " + ", ".join(f"{d}:{n}" for d, n in sorted(days.items())))
    other = [r for r in rows if r["ext"] not in (".avi",)]
    if other and name in ("breast",):
        for r in other:
            emit(f"  non-video file: {r['file']}  {int(r['bytes']) / 1e9:.2f} GB")
    return subjects


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--rescan", nargs="*", choices=("carotid", "cardiac", "files") + tuple(VIDEO_DIRS),
                        help="rescan E: (all parts when given without names)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.rescan is not None or not os.path.exists(os.path.join(CACHE_DIR, "carotid_captures.csv")):
        run_scan(args.root, args.workers, set(args.rescan or ("carotid", "cardiac", "files")))

    carotid = read_csv(os.path.join(CACHE_DIR, "carotid_captures.csv"))
    cardiac = read_csv(os.path.join(CACHE_DIR, "cardiac_captures.csv"))
    emit("In-vivo acquisition inventory, root: " + args.root)
    car_per = analyse_carotid(carotid)
    card_per = analyse_cardiac(cardiac)

    cv = read_csv(os.path.join(CACHE_DIR, "carotid_video_files.csv"))
    vids = analyse_files("carotid_video", cv, "CAROTID side data -- 颈动脉/视频 (screenshots / clips)")
    raw_ids = {p["sid"] for p in car_per}
    vid_ids = {s[:21] for s in vids}
    emit(f"video subjects matching a raw-data subject: {len(vid_ids & raw_ids)}; "
         f"video only: {sorted(vid_ids - raw_ids)}; raw only: {len(raw_ids - vid_ids)} {sorted(raw_ids - vid_ids)[:20]}")
    analyse_files("carotid_dicom", read_csv(os.path.join(CACHE_DIR, "carotid_dicom_files.csv")),
                  "CAROTID side data -- 颈动脉/dicom")
    cdv = read_csv(os.path.join(CACHE_DIR, "cardiac_video_files.csv"))
    cvids = analyse_files("cardiac_video", cdv, "CARDIAC side data -- 心脏/视频")
    craw = {p["sid"] for p in card_per}
    emit(f"cardiac video subjects matching raw: {len(set(cvids) & craw)}; video only: "
         f"{sorted(set(cvids) - craw)}; raw only: {sorted(craw - set(cvids))}")
    analyse_files("breast", read_csv(os.path.join(CACHE_DIR, "breast_files.csv")),
                  "BREAST (other task) -- 乳腺")
    analyse_files("vein", read_csv(os.path.join(CACHE_DIR, "vein_files.csv")),
                  "LOWER-LIMB VEIN (other task) -- 下肢静脉/视频 (168 acquired)")

    text = "\n".join(LINES)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
