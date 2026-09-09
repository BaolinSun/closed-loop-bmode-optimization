# -*- coding: utf-8 -*-
"""滑块与增益的 dB/档常数，在调色板反解后的灰度域上重新测。

    为什么要重测

DEFAULT_DB_PER_LEVEL = 0.06559（滑块每档多少 dB）与 GAIN_DB_PER_LEVEL = 0.1705
（增益每档多少 dB），都是在发现调色板之前、用 PIL 亮度测出来的。而主机截图
不是灰度图：它经过一张一维彩色查找表显示，反解出的灰度比 PIL 亮度高 3–16 个
灰阶，且随亮度段变化（暗端 +3、中段 +16、亮端 +7）。凡是按亮度拟合的常数都要
重拟，这两个也不例外。

    哪些帧真正能约束这两个常数

组内滑块不动时，滑块常数错了会被 pivot_db 整体吸收，残差看不出来；增益同理
（谐波固定 75、通用固定 167/169）。所以只有真正扫过这两个旋钮的场次有约束力，
全部 15 组里只有三个：

    场次           帧数   滑块档取值                    斜坡帧
    20260819       10    0 38 77 127 174 214 254      无（非平帧只抖动 <=5 档）
    20260901_E3     7    6 127 242                    3 帧
    20260904_DR    28    3 69 127 176 254             无（非平帧只抖动 <=8 档）

其中 20260901_E3 的三帧斜坡最有价值 —— [242 210 178 146 114 78 42 6]、它的镜像、
以及一条拱形 [127 172 202 254 249 197 158 127]，顶到底跨 236 档约 15.5 dB。
【滑块在单帧之内就变化】，对 dB/档 的约束力远强于跨帧比较，因为帧间还混着位置、
漂移和其他差异。

    第一版的教训

第一版只取 tgc_levels[0] 当整条曲线，等于把那三帧斜坡当成「平的、值 242/6/127」。
后果是 E3 的逐带误差报成 29.69，而 tools_refit_calibration.py 在同一场次上（它按
is_flat_tgc 把斜坡帧滤掉了）报 2.70。十一倍的差距不是常数不对，是模型不对。

同一版还解出三个互不相容的滑块常数：20260819 得 0.18297（冲出搜索上界）、
E3 得 0.00500（顶到下界）、20260904_DR 得 0.07523，误差只降 15%。那不是三个
场次不通用，是拟合在跟一个错的模型较劲。

现在按每个深度带的实际滑块曲线值建模，斜坡帧也能正确参与。

    看什么

三个场次解出的滑块 dB/档 若彼此接近，即为新的常数值。若仍互不相同，才谈得上
「这个常数不跨场次通用」，那要单独记。20260819 在两版里都是误差最大的一组，
若模型修好后它仍然突出，说明那一组另有问题。

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
    fraction = 0.5 * (edges[:-1] + edges[1:]) / actual.shape[0]
    centres = fraction * capture.geometry.depth_mm

    # The whole slider curve, in levels relative to neutral, sampled where each band sits.
    # Taking only tgc_levels[0] and calling the frame flat is wrong on the ramped frames -
    # 20260901_E3 holds [242 210 178 146 114 78 42 6] and its mirror, 236 levels top to
    # bottom - and those are the most informative frames there are, because the slider varies
    # inside a single frame rather than only between frames.
    sliders = np.asarray(capture.tgc_levels, dtype=np.float64) - S.TGC_CENTER_LEVEL
    slider_fraction = (np.arange(sliders.size) + 0.5) / sliders.size
    tgc_levels_at_band = np.interp(fraction, slider_fraction, sliders)

    return {
        "actual": np.array([np.median(actual[edges[k]:edges[k + 1]])
                            for k in range(FIT_BANDS)]),
        "counts": np.array([np.median(counts[edges[k]:edges[k + 1]])
                            for k in range(FIT_BANDS)]),
        "centres_mm": centres,
        "tgc_levels_at_band": tgc_levels_at_band,
        "tgc_span": float(sliders.max() - sliders.min()),
        "gain_level": float(capture.gain_level),
        "window_db": S.capture_window_db(capture),
        "depth_mm": float(capture.geometry.depth_mm),
        "name": capture.name,
    }


def predict(summary, counts_per_db, pivot_db, tgc_per_level, gain_per_level,
            axis_mm=None, response_db=None):
    offset = (summary["tgc_levels_at_band"] * tgc_per_level
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
        offset = (summary["tgc_levels_at_band"][usable] * tgc_per_level
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
    tgc_span = (0.01, 0.30) if fit_tgc else (S.DEFAULT_DB_PER_LEVEL,) * 2
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
                tgc = (max(0.001, tgc_per_level - step), tgc_per_level + step)
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
        emit(u"=========== %s (%s, %d frames) ===========" % (session, note, len(summaries)))
        ramped = [s for s in summaries if s["tgc_span"] > 20]
        emit(u"  %d frames, of which %d carry a ramped slider curve (span > 20 levels)"
             % (len(summaries), len(ramped)))
        if ramped:
            emit(u"  ramped frames span %s levels"
                 % sorted(int(s["tgc_span"]) for s in ramped))
        emit(u"  gain levels present: %s" % sorted({int(s["gain_level"]) for s in summaries}))
        emit(u"")
        emit(u"%-26s %11s %10s %13s %13s %11s" % (
            u"fit", u"counts/dB", u"pivot_dB", u"slider dB/lvl", u"gain dB/lvl", u"band error"))
        for label, fit_tgc, fit_gain in [(u"current constants", False, False),
                                         (u"refit slider only", True, False),
                                         (u"refit slider and gain", True, True)]:
            result = fit(summaries, fit_tgc, fit_gain)
            emit(u"%-26s %11.1f %10.2f %13.5f %13.5f %11.2f" % (
                label, result[1], result[2], result[3], result[4], result[0]))
        emit(u"  (current constants: slider %.5f, gain %.5f) took %.0f s"
             % (S.DEFAULT_DB_PER_LEVEL, S.GAIN_DB_PER_LEVEL, time.time() - started))

    text = u"\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_actuator_constants.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
