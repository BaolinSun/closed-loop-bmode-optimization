# -*- coding: utf-8 -*-
"""从 bmode_opt 复制过来的常数，训练包因此不必导入 bmode_opt。

每个值后面写明出处；tests/verify_six_param_model.py 逐项断言与 bmode_opt 一致，
bmode_opt 里的值一旦改动，那个检查会失败。
"""

# 成像模式：0 基波成像，1 谐波成像（BImageMode）
MODE_FUNDAMENTAL = 0
MODE_HARMONIC = 1

# hisense_backend_sim.GAIN_DB_PER_LEVEL_BY_MODE / TGC_DB_PER_LEVEL_BY_MODE
GAIN_DB_PER_LEVEL = {MODE_FUNDAMENTAL: 0.28778, MODE_HARMONIC: 0.20514}
TGC_DB_PER_LEVEL = {MODE_FUNDAMENTAL: 0.08226, MODE_HARMONIC: 0.07810}

# hisense_backend_sim.TGC_MIN_LEVEL / TGC_MAX_LEVEL / TGC_CENTER_LEVEL；hisense_loader.NUM_TGC_BANDS
TGC_MIN_LEVEL = 0
TGC_MAX_LEVEL = 255
TGC_CENTER_LEVEL = 127
NUM_TGC_BANDS = 8

# hisense_backend_sim.GRAY_PIVOT / GRAY_MAX / DR_WINDOW_SLOPE / DR_WINDOW_INTERCEPT
GRAY_PIVOT = 43.0
GRAY_MAX = 255.0
DR_WINDOW_SLOPE = 0.3519
DR_WINDOW_INTERCEPT = 35.09

# labels.FIELDII_GAIN_DELTA_CLICKS / FIELDII_SLIDER_TILT_LEVELS / FIELDII_SLIDER_ARCH_LEVELS
START_GAIN_DELTA_CLICKS = (-20.0, 25.0)
START_SLIDER_TILT_LEVELS = (-110.0, 110.0)
START_SLIDER_ARCH_LEVELS = (-70.0, 70.0)

# labels.SLIDER_GROUPS
SLIDER_GROUPS = (("near", 0, 3), ("mid", 3, 5), ("far", 5, 8))

# 方向名称（labels.GAIN_DIRECTIONS 等，tools_generate_console_labels.DEPTH_DIRECTIONS 等）。
# 所有轴统一约定：下标 0 = 应当增大（最优值大于当前），1 = 正确，2 = 应当减小。
GAIN_DIRECTIONS = ("dark", "correct", "bright")
SLIDER_DIRECTIONS = ("low", "correct", "high")
DYNAMIC_RANGE_DIRECTIONS = ("narrow", "correct", "wide")
DEPTH_DIRECTIONS = ("shallow", "correct", "deep")
FREQUENCY_DIRECTIONS = ("low", "correct", "high")
FOCUS_DIRECTIONS = ("shallow", "correct", "deep")

# Field II 分片 /envelope 的深度点数（全部分片相同，缓存脚本会检查）
FIELDII_SOURCE_ROWS = 1851


def dr_ui_to_window_db(ui_value):
    """动态范围 UI 数值 -> 显示窗宽 dB（hisense_backend_sim.dr_ui_to_window_db）。"""
    return DR_WINDOW_SLOPE * ui_value + DR_WINDOW_INTERCEPT
