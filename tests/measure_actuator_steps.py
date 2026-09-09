# -*- coding: utf-8 -*-
"""直接测滑块与增益的 dB/档：同场景两帧相减，不经过任何标定常数。

    为什么换这个办法

measure_actuator_constants.py 用全局拟合同时解 counts_per_db、pivot_db、C(z) 与
两个执行器常数。修好斜坡帧的建模之后，两个场次一致了（滑块 0.08307 / 0.07457），
但 20260819 解出 0.32549、pivot_db 跑到 58.9（其他场次 24–31），逐带误差仍有
26.63。那是拟合在发散，不是测量结果。

五组参数互相纠缠，本来就不该指望它们各自被钉住。而这里要的两个常数其实有更
直接的读法：同一静止场景下、只差一个旋钮的两帧，相减即可。

    原理

一帧的显示模型是

    gray = GRAY_PIVOT + (counts/counts_per_db + C(z) + (T-127)*t + (G-75)*g - pivot) / W * 255

同一场景的两帧，counts、C(z)、pivot 完全相同，相减后全部消失：

    (gray_B - gray_A) / 255 * W = (T_B - T_A) * t + (G_B - G_A) * g

左边全部可测（W 由 dr_ui_to_window_db 给出），右边只剩要求的两个常数。不需要
counts_per_db，不需要深度响应，不需要 pivot。前提只有一个：场景没动。20260819
是夹持探头采的，E3 与 20260904_DR 也都在固定位置上扫后端。

逐深度带做，一对帧就给 32 个独立读数；E3 的斜坡帧每带的滑块差还各不相同，读数
更有分量。

    这个脚本回答两个问题

一、常数是多少。把所有配对的读数放进一个最小二乘，同时解 t 和 g。

二、【一个常数够不够】。逐配对报告「实测 ΔdB ÷ Δ档数」——若该比值随 Δ档数 变化，
说明响应非线性，那 20260819 拟不动就不是拟合的毛病，是模型本身不成立，得改成
查找表而不是一个标量。20260819 的滑块跨度最大（0–254），最有可能在这里露馅。

只用两帧都未裁剪的深度带：显示一旦裁到 0 或 255，差值就不再反映 dB 差。

用法：python tests/measure_actuator_steps.py
"""

import io
import itertools
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import hisense_backend_sim as S
import calibration as CAL
import display_palette as DP
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

FIT_BANDS = CAL.FIT_BANDS

# 两帧都要落在这个灰阶区间内，该带才算数。裁剪处灰阶差不再是 dB 差。
UNCLIPPED_GRAY = (15, 240)

# 场景静止、只扫后端的场次。前三个是谐波（增益 59-125），最后一个是基波
# （增益 129-229）。两段增益区间不重叠，这正是 GAIN_DB_PER_LEVEL 必须分段、
# 而分段的成因（档位区间还是成像模式）又分不开的原因。
SESSIONS = ["20260819", "20260901_E3", "20260904_DR", "20260909_GEN"]

# 增益阶梯换斜率的档位。谐波数据止于 125，基波数据始于 129，界在其间。
GAIN_LADDER_BOUNDARY = 127.0


def summarise(capture, palette):
    """一帧的逐带显示灰阶、逐带滑块档、增益档与窗宽。"""
    display = DP.capture_display_gray(capture, palette=palette)[0]
    edges = np.linspace(0, display.shape[0], FIT_BANDS + 1).round().astype(int)
    fraction = 0.5 * (edges[:-1] + edges[1:]) / display.shape[0]

    sliders = np.asarray(capture.tgc_levels, dtype=np.float64)
    slider_fraction = (np.arange(sliders.size) + 0.5) / sliders.size

    return {
        "name": capture.name,
        "gray": np.array([np.median(display[edges[k]:edges[k + 1]])
                          for k in range(FIT_BANDS)]),
        "tgc_at_band": np.interp(fraction, slider_fraction, sliders),
        "tgc_span": float(sliders.max() - sliders.min()),
        "gain_level": float(capture.gain_level),
        "window_db": S.capture_window_db(capture),
        "depth_mm": float(capture.geometry.depth_mm),
        "dr_ui": float(capture.dynamic_range_level),
        "frequency_mhz": CAL.capture_frequency(capture),
    }


def split_gain(a_level, b_level, boundary=GAIN_LADDER_BOUNDARY):
    """把一次增益改变拆成界下与界上两段，各自带符号。"""
    low = min(a_level, b_level)
    high = max(a_level, b_level)
    below = max(0.0, min(high, boundary) - min(low, boundary))
    above = max(0.0, max(high, boundary) - max(low, boundary))
    sign = 1.0 if b_level >= a_level else -1.0
    return sign * below, sign * above


def pair_rows(a, b):
    """一对帧给出的读数：每个未裁剪深度带一行 (ΔdB, Δ滑块档, Δ增益档)。"""
    usable = ((a["gray"] > UNCLIPPED_GRAY[0]) & (a["gray"] < UNCLIPPED_GRAY[1])
              & (b["gray"] > UNCLIPPED_GRAY[0]) & (b["gray"] < UNCLIPPED_GRAY[1]))
    if not usable.any():
        return None
    # 窗宽若不同，各自换算回 dB 再相减。
    delta_db = (b["gray"][usable] / 255.0 * b["window_db"]
                - a["gray"][usable] / 255.0 * a["window_db"])
    delta_tgc = b["tgc_at_band"][usable] - a["tgc_at_band"][usable]
    delta_gain = np.full(delta_db.shape, b["gain_level"] - a["gain_level"])
    below, above = split_gain(a["gain_level"], b["gain_level"])
    return np.column_stack([delta_db, delta_tgc, delta_gain,
                            np.full(delta_db.shape, below),
                            np.full(delta_db.shape, above)])


def is_flat(summary):
    """Whether the slider curve is flat enough to be one number (jitter of a few levels)."""
    return summary["tgc_span"] <= 10.0


def solve(rows, label, emit, split=False):
    """One gain slope, or one either side of the boundary."""
    if not rows:
        return None
    stack = np.vstack(rows)
    columns = [1, 3, 4] if split else [1, 2]
    solution, *_ = np.linalg.lstsq(stack[:, columns], stack[:, 0], rcond=None)
    residual = stack[:, 0] - stack[:, columns] @ solution
    if split:
        emit(u"  %-34s slider %.5f  gain<=%d %.5f  gain>%d %.5f  residual sd %7.3f dB  (%d)"
             % (label, solution[0], GAIN_LADDER_BOUNDARY, solution[1],
                GAIN_LADDER_BOUNDARY, solution[2], residual.std(), stack.shape[0]))
    else:
        emit(u"  %-34s slider %.5f  gain %.5f  residual sd %7.3f dB  (%d readings)"
             % (label, solution[0], solution[1], residual.std(), stack.shape[0]))
    return solution


def main():
    lines = []
    emit = lines.append
    every, clean = [], []

    for session in SESSIONS:
        captures = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        palette = DP.session_palette(captures)[0]
        summaries = [summarise(c, palette) for c in captures]
        emit(u"")
        emit(u"=========== %s (%d frames) ===========" % (session, len(summaries)))
        ramped = [s for s in summaries if not is_flat(s)]
        if ramped:
            emit(u"  %d frame(s) carry a ramped slider curve, span %s levels"
                 % (len(ramped), sorted(int(s["tgc_span"]) for s in ramped)))
        emit(u"%-22s %-22s %7s %7s %7s %9s %11s %6s" % (
            u"frame A", u"frame B", u"d_tgc", u"d_gain", u"bands", u"d_dB",
            u"dB/level", u"flat"))

        session_rows, session_clean = [], []
        for a, b in itertools.combinations(summaries, 2):
            # Same scene means same depth, same window and same transmit frequency. The
            # frequency guard matters: the console applies something frequency dependent
            # between BC0 and the display worth about 3.5 dB between 8.0 and 11.4 MHz in
            # fundamental mode, and pairing across it would charge that to the actuators.
            if (abs(a["depth_mm"] - b["depth_mm"]) > 1e-6 or a["dr_ui"] != b["dr_ui"]
                    or a["frequency_mhz"] != b["frequency_mhz"]):
                continue
            block = pair_rows(a, b)
            if block is None or block.shape[0] < 4:
                continue
            mean_tgc = block[:, 1].mean()
            mean_gain = block[:, 2].mean()
            if abs(mean_tgc) < 1e-6 and abs(mean_gain) < 1e-6:
                continue                       # repeat frame, kept out of the fit
            both_flat = is_flat(a) and is_flat(b)
            single_knob = abs(mean_tgc) < 1e-6 or abs(mean_gain) < 1e-6
            session_rows.append(block)
            every.append(block)
            if both_flat and single_knob:
                session_clean.append(block)
                clean.append(block)
            if abs(mean_gain) < 1e-6:
                ratio = u"%11.5f" % (block[:, 0].mean() / mean_tgc)
            elif abs(mean_tgc) < 1e-6:
                ratio = u"%11.5f" % (block[:, 0].mean() / mean_gain)
            else:
                ratio = u"%11s" % u"both moved"
            emit(u"%-22s %-22s %7.0f %7.0f %7d %9.2f %s %6s" % (
                a["name"][:22], b["name"][:22], mean_tgc, mean_gain,
                block.shape[0], block[:, 0].mean(), ratio,
                u"yes" if both_flat else u"no"))

        solve(session_rows, u"all pairs", emit)
        solve(session_clean, u"flat frames, one knob only", emit)

    emit(u"")
    emit(u"=========== all sessions together ===========")
    emit(u"  one gain slope over the whole ladder:")
    solve(every, u"all pairs", emit)
    solve(clean, u"flat frames, one knob only", emit)
    emit(u"  gain slope split at level %d:" % GAIN_LADDER_BOUNDARY)
    solve(every, u"all pairs", emit, split=True)
    solve(clean, u"flat frames, one knob only", emit, split=True)
    emit(u"  current constants                  slider %.5f  gain<=%d %.5f  gain>%d %.5f"
         % (S.DEFAULT_DB_PER_LEVEL, GAIN_LADDER_BOUNDARY,
            S.GAIN_DB_PER_LEVEL_TABLE[0][2], GAIN_LADDER_BOUNDARY,
            S.GAIN_DB_PER_LEVEL_TABLE[1][2]))

    if clean:
        stack = np.vstack(clean)
        emit(u"")
        emit(u"=========== linearity: does dB per level depend on step size ===========")
        emit(u"  slider-only readings from flat frames, grouped by |d_tgc|")
        emit(u"%14s %10s %14s %12s" % (u"|d_tgc|", u"readings", u"dB/level", u"sd"))
        only_tgc = stack[np.abs(stack[:, 2]) < 1e-6]
        only_tgc = only_tgc[np.abs(only_tgc[:, 1]) > 5]
        for low, high in [(5, 25), (25, 50), (50, 80), (80, 120), (120, 300)]:
            sel = only_tgc[(np.abs(only_tgc[:, 1]) >= low)
                           & (np.abs(only_tgc[:, 1]) < high)]
            if sel.shape[0] < 4:
                continue
            per_level = sel[:, 0] / sel[:, 1]
            emit(u"%14s %10d %14.5f %12.5f" % (
                u"%d-%d" % (low, high), sel.shape[0], per_level.mean(), per_level.std()))

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_actuator_steps.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
