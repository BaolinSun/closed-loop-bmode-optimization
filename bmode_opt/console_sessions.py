# -*- coding: utf-8 -*-
"""实机场次在哪里、每个场次有哪些采集。各工具都从这里找场次，不再各自写死列表。

    两套数据

体模（phantom）  <根>/<场次>/<采集>/Algo_BC0.bin，场次可以嵌套（20260828/GEN）。根依次是
                 E:/数据/hisense_medical/体模（data_paths.PHANTOM_ROOT）与项目里残留的
                 data/hisense_medical，同名场次取先找到的；环境变量 BMODE_PHANTOM_ROOTS
                 （os.pathsep 分隔）可以整体替换。
                 E 盘上的体模目录改过名，按 PHANTOM_ALIASES 还原成标定文件与标签里用的场次名，
                 否则 console_calibration.json 与 point_targets.RESOLUTION_SETS 都对不上。

人体颈动脉（invivo）  E:/数据/hisense_medical/颈动脉/原始数据 是扁平的，一次存储一个目录，
                 受试者只能由 Algo_PartitionInfo.pdt 里 FILEPATH 的场次号还原。
                 tools_build_invivo_manifest.py 把分组与剔除写进 data/invivo_carotid_manifest.json，
                 这里只读清单。E 盘上的文件不移动、不复制。

两套数据的输出文件分开（DATASETS），互不覆盖。电影回放（PageNum > 1）两边都不收。
训练缓存占几个 GB，写在 E:/数据/closed_loop_bmode_optimization/ 下（data_paths.OUTPUT_ROOT）；
标定、族、标签体积小，仍在项目里。
"""

import json
import os
from pathlib import Path

import data_paths as DPATH
from hisense_loader import BC0_FILE, PROJECT_ROOT, capture_page_num

INVIVO_MANIFEST = PROJECT_ROOT / "data" / "invivo_carotid_manifest.json"
DEFAULT_PHANTOM_ROOTS = (DPATH.PHANTOM_ROOT, PROJECT_ROOT / "data" / "hisense_medical")

# E 盘体模目录名 -> 场次名。逐一核对过：目录里的采集名与已提交标签的 frame_id 完全重合
# （0903_THI 74 帧 = 20260903 的谐波 68 + 基波 6；0903 36；0904 25；0904_DR 28）。
PHANTOM_ALIASES = {
    "PatientData_Temp_0903_THI": "20260903",
    "PatientData_Temp_0903": "20260903_replication_check",
    "PatientData_Temp_0904": "20260904",
    "PatientData_Temp_0904_DR": "20260904_DR",
}

# 每套数据的默认输出。体模沿用原来的文件名，人体全部另起文件。
DATASETS = {
    "phantom": {
        "calibration": "bmode_opt/console_calibration.json",
        "families": "data/scene_families.json",
        "labels": "data/labels_console.jsonl",
        "cache": str(DPATH.output_dir("console_dl_cache")),
        "report_suffix": "",
    },
    "invivo": {
        "calibration": "bmode_opt/console_calibration_invivo_carotid.json",
        "families": "data/scene_families_invivo_carotid.json",
        "labels": "data/labels_console_invivo_carotid.jsonl",
        "cache": str(DPATH.output_dir("console_dl_cache_invivo_carotid")),
        "report_suffix": "_invivo_carotid",
    },
}

_PHANTOM = None
_INVIVO = None


def phantom_roots():
    override = os.environ.get("BMODE_PHANTOM_ROOTS")
    roots = ([Path(p) for p in override.split(os.pathsep) if p] if override
             else list(DEFAULT_PHANTOM_ROOTS))
    return [root for root in roots if root.is_dir()]


def _phantom_index():
    """{场次名: [采集目录]}，只收单帧。"""
    global _PHANTOM
    if _PHANTOM is None:
        index = {}
        for root in phantom_roots():
            found = {}
            for path in sorted(root.rglob(BC0_FILE)):
                capture = path.parent
                label = str(capture.parent.relative_to(root)).replace("\\", "/")
                label = PHANTOM_ALIASES.get(label, label)
                if label in index or capture_page_num(capture) > 1:
                    continue
                found.setdefault(label, []).append(capture)
            index.update(found)
        _PHANTOM = {label: sorted(paths, key=lambda p: p.name) for label, paths in index.items()}
    return _PHANTOM


def load_manifest(path=INVIVO_MANIFEST):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _invivo_index():
    """{受试者场次号: (根目录, [逐帧元数据])}。清单不存在时为空。"""
    global _INVIVO
    if _INVIVO is None:
        _INVIVO = {}
        if INVIVO_MANIFEST.exists():
            manifest = load_manifest()
            root = Path(manifest["root"])
            _INVIVO = {sid: (root, frames) for sid, frames in manifest["sessions"].items()}
    return _INVIVO


def sessions(dataset):
    """一套数据里的全部场次名，排好序。"""
    if dataset == "phantom":
        return sorted(_phantom_index())
    if dataset == "invivo":
        if not INVIVO_MANIFEST.exists():
            raise FileNotFoundError("%s is missing; run tools_build_invivo_manifest.py first"
                                    % INVIVO_MANIFEST)
        return sorted(_invivo_index())
    raise ValueError("unknown dataset %r" % dataset)


def session_source(label):
    return "invivo" if label in _invivo_index() else "phantom"


def session_captures(label):
    """一个场次的全部单帧采集目录，按目录名（即时间戳）排序。"""
    invivo = _invivo_index()
    if label in invivo:
        root, frames = invivo[label]
        return [root / frame["name"] for frame in frames]
    phantom = _phantom_index()
    if label in phantom:
        return list(phantom[label])
    raise KeyError("no console session %r under %s or in %s"
                   % (label, ", ".join(str(r) for r in phantom_roots()), INVIVO_MANIFEST))


def session_metadata(label):
    """清单里逐帧的设置（不读 BC0）；体模场次没有清单，返回 None。"""
    entry = _invivo_index().get(label)
    return None if entry is None else entry[1]


def output_path(dataset, kind):
    return DATASETS[dataset][kind]


def report_path(script, dataset, out=None):
    """脚本旁边的报告文件；人体数据另起一个，不覆盖体模的报告。

    给了 out（输出改道或只跑了部分场次）时报告写在 out 旁边，免得一次试跑覆盖正式报告。
    """
    if out:
        return os.path.splitext(out)[0] + "_report.txt"
    stem = os.path.splitext(script)[0]
    return stem + DATASETS[dataset]["report_suffix"] + ".txt"
