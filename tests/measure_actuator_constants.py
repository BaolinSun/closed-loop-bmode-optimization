# -*- coding: utf-8 -*-
"""滑块与增益的 dB/档常数，在调色板反解后的灰度域上重新测。

    为什么要重测

DEFAULT_DB_PER_LEVEL = 0.06559（滑块每档多少 dB）与 GAIN_DB_PER_LEVEL = 0.1705
（增益每档多少 dB），都是在发现调色板之前、用 PIL 亮度测出来的。而主机截图
不是灰度图：它经过一张一维彩色查找表显示，反解出的灰度比 PIL 亮度高 3–16 个
灰阶，且随亮度段变化（暗端 +3、中段 +16、亮端 +7）。凡是按亮度拟合的常数都要
重拟，这两个也不例外。

    为什么只有一个场次能看出来

tools_refit_calibration.py 重拟后，15 个组里 14 个的逐带误差落在 1.0–11.6，
只有 20260819 是 27.22。那不是数据坏了，是它【唯一有资格检验这两个常数】：

    场次              滑块档取值                          能否暴露滑块常数的错误
    20260819         0 38 77 127 174 214 254           能，跨度 254 档
    20260901_E3      6 127 242                         能，但只有 3 个点
    20260904_DR      3 69 127 176 254                  能
    其余 11 组        几乎全是 127                        不能

组内滑块不动时，滑块常数错了会被 pivot_db 整体吸收，残差看不出来；增益同理
（谐波固定 75、通用固定 167/169）。所以只有真正扫过这两个旋钮的场次会把错误
顶到残差上，而 20260819 的滑块跨度最大，误差也最大 —— 这本身就是证据。

    这个脚本做什么

对三个扫过后端的场次各拟合三次：沿用现有常数、只重拟滑块常数、滑块与增益都
重拟。判据与 calibration.fit_group 一致（与主机截图的逐带灰阶误差），并同样
交替求解深度响应，因为不带深度响应拟标量是不适定的。

假设成立的话，20260819 的误差会从 27 大幅下降，且三个场次解出的滑块常数应当
彼此接近 —— 那就是新的常数值。若三者互不相同，说明这两个常数在不同场次并不
通用，那是另一件事，得单独记下来。

用法：python tests/measure_actuator_constants.py
"""

import io
import os
import sys
import time

sys.path.insert(0, "bmode_opt")
import numpy as np

import hisense_backend_sim as S
import calibration as CAL
import display_palette as DP
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

FIT_BANDS = CAL.FIT_BANDS

# 三个在固定探头位置上扫过后端的场次。其余场次组内后端不变，对这两个常数没有
# 约束力，放进来只会让拟合以为自己被约束住了。
SWEPT_SESSIONS = [
    ("20260819", u"滑块 0–254 大范围 + 增益 75/125"),
    ("20260901_E3", u"滑块 6/127/242 + 增益 59/75/91/105"),
    ("20260904_DR", u"滑块 3–254 + 增益 75/115 + 动态范围扫描"),
]


def summarise(capture):
    """与 calibration.band_summary 相同，但把滑块档与增益档分开留着当自由参数。"""
    actual = DP.capture_display_gray(capture)[0]
    counts = S.scan_convert_linear(capture.bc0, actual.shape[0], actual.shape[1])
    edges = np.linspace(0, actual.shape[0], FIT_BANDS + 1).round().astype(int)
    centres = (0.5 * (edges[:-1] + edges[1:]) / actual.shape[0]) * capture.geometry.depth_mm
    return {
        "actual": np.array([np.median(actual[edges[k]:edges[k + 1]])
                            for k in range(FIT_BANDS)]),
        "counts": np.array([np.median(counts[edges[k]:edges[k + 1]])
                            for k in range(FIT_BANDS)]),
        "centres_mm": centres,
        "tgc_level": float(capture.tgc_levels[0]),
        "gain_level": float(capture.gain_level),
        "window_db": S.capture_window_db(capture),
        "depth_mm": float(capture.geometry.depth_mm),
        "name": capture.name,
    }


def predict(summary, counts_per_db, pivot_db, tgc_per_level, gain_per_level,
            axis_mm=None, response_db=None):
    offset = ((summary["tgc_level"] - S.TGC_CENTER_LEVEL) * tgc_per_level
              + (summary["gain_level"] - S.CALIBRATION_GAIN_LEVEL) * gain_per_level)
    response = 0.0 if response_db is None else np.interp(
        summary["centres_mm"], axis_mm, response_db,
        left=response_db[0], right=response_db[-1])
    level = summary["counts"] / counts_per_db + response + offset - pivot_db
    raw = S.GRAY_PIVOT + level / summary["window_db"] * S.GRAY_MAX
    return np.clip(np.round(raw), 0.0, S.GRAY_MAX)


def group_cost(summaries, counts_per_db, pivot_db, tgc_per_level, gain_per_level,
               axis_mm=None, response_db=None):
    return float(np.mean([
        np.mean(np.abs(s["actual"] - predict(s, counts_per_db, pivot_db, tgc_per_level,
                                             gain_per_level, axis_mm, response_db)))
        for s in summaries]))


def solve_response(summaries, counts_per_db, pivot_db, tgc_per_level, gain_per_level,
                   axis_mm, smooth_mm=2.0):
    """两个标量解释不掉的、随深度变化的那部分 dB。"""
    depths, values = [], []
    for summary in summaries:
        usable = ((summary["actual"] > CAL.USABLE_GRAY[0])
                  & (summary["actual"] < CAL.USABLE_GRAY[1]))
        if not usable.any():
            continue
        offset = ((summary["tgc_level"] - S.TGC_CENTER_LEVEL) * tgc_per_level
                  + (summary["gain_level"] - S.CALIBRATION_GAIN_LEVEL) * gain_per_level)
        wanted = ((summary["actual"][usable] - S.GRAY_PIVOT) / S.GRAY_MAX
                  * summary["window_db"] + pivot_db - offset
                  - summary["counts"][usable] / counts_per_db)
        depths.append(summary["centres_mm"][usable])
        values.append(wanted)
    if not depths:
        return np.zeros_like(axis_mm)
    depths = np.concatenate(depths)
    values = np.concatenate(values)
    curve = np.array([
        np.median(values[np.abs(depths - z) <= smooth_mm])
        if (np.abs(depths - z) <= smooth_mm).sum() >= 3 else np.nan
        for z in axis_mm])
    good = np.isfinite(curve)
    return np.interp(axis_mm, axis_mm[good], curve[good]) if good.sum() >= 2 \
        else np.zeros_like(axis_mm)


def fit(summaries, fit_tgc, fit_gain, iterations=4, refinements=4, num_axis_points=128):
    """交替求解：网格解标量，残差解深度响应，反复。"""
    axis_mm = np.linspace(0.0, max(s["depth_mm"] for s in summaries), num_axis_points)
    response = np.zeros_like(axis_mm)
    tgc_span = (0.02, 0.16) if fit_tgc else (S.DEFAULT_DB_PER_LEVEL,) * 2
    gain_span = (0.05, 0.40) if fit_gain else (S.GAIN_DB_PER_LEVEL,) * 2
    best = None

    for _ in range(int(iterations)):
        counts_span, pivot_span = (300.0, 1600.0), (5.0, 55.0)
        tgc, gain = tgc_span, gain_span
        best = None
        for _ in range(int(refinements)):
            for counts_per_db in np.linspace(*counts_span, 15):
                for pivot_db in np.linspace(*pivot_span, 15):
                    for tgc_per_level in (np.linspace(*tgc, 9) if fit_tgc else [tgc[0]]):
                        for gain_per_level in (np.linspace(*gain, 9) if fit_gain
                                               else [gain[0]]):
                            value = group_cost(summaries, counts_per_db, pivot_db,
                                               tgc_per_level, gain_per_level,
                                               axis_mm, response)
                            if best is None or value < best[0]:
                                best = (value, counts_per_db, pivot_db,
                                        tgc_per_level, gain_per_level)
            _, counts_per_db, pivot_db, tgc_per_level, gain_per_level = best
            counts_step = (counts_span[1] - counts_span[0]) / 14
            pivot_step = (pivot_span[1] - pivot_span[0]) / 14
            counts_span = (counts_per_db - counts_step, counts_per_db + counts_step)
            pivot_span = (pivot_db - pivot_step, pivot_db + pivot_step)
            if fit_tgc:
                step = (tgc[1] - tgc[0]) / 8
                tgc = (max(0.005, tgc_per_level - step), tgc_per_level + step)
            if fit_gain:
                step = (gain[1] - gain[0]) / 8
                gain = (max(0.01, gain_per_level - step), gain_per_level + step)
        response = solve_response(summaries, best[1], best[2], best[3], best[4], axis_mm)

    return best + (axis_mm, response)


def main():
    lines = []
    emit = lines.append
    for session, note in SWEPT_SESSIONS:
        started = time.time()
        captures = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        summaries = [summarise(c) for c in captures]
        emit(u"")
        emit(u"=========== %s（%s，%d 帧）===========" % (session, note, len(summaries)))
        emit(u"  滑块档取值 %s" % sorted({int(s["tgc_level"]) for s in summaries}))
        emit(u"  增益档取值 %s" % sorted({int(s["gain_level"]) for s in summaries}))
        emit(u"")
        emit(u"%-26s %11s %10s %13s %13s %11s" % (
            u"拟合方式", u"counts/dB", u"pivot_dB", u"滑块dB/档", u"增益dB/档", u"逐带误差"))
        for label, fit_tgc, fit_gain in [(u"沿用现有常数", False, False),
                                         (u"只重拟滑块常数", True, False),
                                         (u"滑块+增益都重拟", True, True)]:
            result = fit(summaries, fit_tgc, fit_gain)
            emit(u"%-26s %11.1f %10.2f %13.5f %13.5f %11.2f" % (
                label, result[1], result[2], result[3], result[4], result[0]))
        emit(u"  （现有常数：滑块 %.5f，增益 %.5f）耗时 %.0f 秒"
             % (S.DEFAULT_DB_PER_LEVEL, S.GAIN_DB_PER_LEVEL, time.time() - started))

    text = u"\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_actuator_constants.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text.encode("ascii", "replace").decode())
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
