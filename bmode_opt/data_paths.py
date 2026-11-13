# -*- coding: utf-8 -*-
"""数据集与大体积生成数据的位置，各工具的默认路径都从这里取。

数据集根目录是 E:/数据（硬盘容量所限，原始数据和占空间大的生成物都不放在项目里）：

    E:/数据/hisense_medical/体模                  体模场次（PatientData_Temp_*）
    E:/数据/hisense_medical/颈动脉/原始数据       人体颈动脉，一次存储一个目录
    E:/数据/field_ii/full_noise                   Field II 带噪仿真
    E:/数据/closed_loop_bmode_optimization/       本项目生成的大文件：训练缓存、阅片包等

小的生成物（标签 jsonl、族 json、清单、逐帧参数缓存）仍在项目的 data/ 下，便于查看和比对。
换机器时设环境变量 BMODE_DATASET_ROOT 即可整体搬走，不用改代码。
"""

import os
from pathlib import Path

DATASET_ROOT = Path(os.environ.get("BMODE_DATASET_ROOT", "E:/数据"))
HISENSE_ROOT = DATASET_ROOT / "hisense_medical"
PHANTOM_ROOT = HISENSE_ROOT / "体模"
CAROTID_RAW_ROOT = HISENSE_ROOT / "颈动脉" / "原始数据"
FIELDII_DATA_DIR = DATASET_ROOT / "field_ii" / "full_noise"
OUTPUT_ROOT = DATASET_ROOT / "closed_loop_bmode_optimization"


def output_dir(name):
    """大体积生成物的目录（训练缓存、阅片包等）。"""
    return OUTPUT_ROOT / name
