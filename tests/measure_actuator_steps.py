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

# 场景静止、只扫后端的三个场次。
SESSIONS = ["20260819", "20260901_E3", "20260904_DR"]


def summarise(capture):
    """一帧的逐带显示灰阶、逐带滑块档、增益档与窗宽。"""
    display = DP.capture_display_gray(capture)[0]
    edges = np.linspace(0, display.shape[0], FIT_BANDS + 1).round().astype(int)
    fraction = 0.5 * (edges[:-1] + edges[1:]) / display.shape[0]

    sliders = np.asarray(capture.tgc_levels, dtype=np.float64)
    slider_fraction = (np.arange(sliders.size) + 0.5) / sliders.size

    return {
        "name": capture.name,
        "gray": np.array([np.median(display[edges[k]:edges[k + 1]])
                          for k in range(FIT_BANDS)]),
        "tgc_at_band": np.interp(fraction, slider_fraction, sliders),
        "gain_level": float(capture.gain_level),
        "window_db": S.capture_window_db(capture),
        "depth_mm": float(capture.geometry.depth_mm),
        "dr_ui": float(capture.dynamic_range_level),
    }


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
    return np.column_stack([delta_db, delta_tgc, delta_gain])


def main():
    lines = []
    emit = lines.append
    everything = []

    for session in SESSIONS:
        captures = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        summaries = [summarise(c) for c in captures]
        # 只在同一显示深度、同一动态范围内配对；深度不同则场景采样不同，不可直接相减。
        emit(u"")
        emit(u"=========== %s（%d 帧）===========" % (session, len(summaries)))
        emit(u"%-22s %-22s %7s %7s %8s %10s %12s" % (
            u"帧 A", u"帧 B", u"Δ滑块", u"Δ增益", u"可用带", u"实测ΔdB", u"ΔdB/Δ档"))

        rows = []
        for a, b in itertools.combinations(summaries, 2):
            if abs(a["depth_mm"] - b["depth_mm"]) > 1e-6 or a["dr_ui"] != b["dr_ui"]:
                continue
            block = pair_rows(a, b)
            if block is None or block.shape[0] < 4:
                continue
            mean_tgc = block[:, 1].mean()
            mean_gain = block[:, 2].mean()
            if abs(mean_tgc) < 1e-6 and abs(mean_gain) < 1e-6:
                continue                       # 重复帧，用来看噪声，不进拟合
            rows.append(block)
            everything.append(block)
            # 只动了一个旋钮时，比值才有单独的意义
            if abs(mean_gain) < 1e-6:
                ratio = u"%12.5f" % (block[:, 0].mean() / mean_tgc)
            elif abs(mean_tgc) < 1e-6:
                ratio = u"%12.5f" % (block[:, 0].mean() / mean_gain)
            else:
                ratio = u"%12s" % u"两个都动"
            emit(u"%-22s %-22s %7.0f %7.0f %8d %10.2f %s" % (
                a["name"][:22], b["name"][:22], mean_tgc, mean_gain,
                block.shape[0], block[:, 0].mean(), ratio))

        if rows:
            stack = np.vstack(rows)
            solution, *_ = np.linalg.lstsq(stack[:, 1:], stack[:, 0], rcond=None)
            residual = stack[:, 0] - stack[:, 1:] @ solution
            emit(u"  本场次最小二乘：滑块 %.5f dB/档，增益 %.5f dB/档，残差标准差 %.3f dB"
                 % (solution[0], solution[1], residual.std()))

    if everything:
        stack = np.vstack(everything)
        solution, *_ = np.linalg.lstsq(stack[:, 1:], stack[:, 0], rcond=None)
        residual = stack[:, 0] - stack[:, 1:] @ solution
        emit(u"")
        emit(u"=========== 三个场次合并 ===========")
        emit(u"  读数 %d 条" % stack.shape[0])
        emit(u"  滑块 %.5f dB/档（现值 %.5f）" % (solution[0], S.DEFAULT_DB_PER_LEVEL))
        emit(u"  增益 %.5f dB/档（现值 %.5f）" % (solution[1], S.GAIN_DB_PER_LEVEL))
        emit(u"  残差标准差 %.3f dB" % residual.std())

        emit(u"")
        emit(u"=========== 线性检验：比值随步长变不变 ===========")
        emit(u"  只取单独动滑块的读数，按 |Δ档数| 分组")
        emit(u"%14s %10s %14s %12s" % (u"|Δ档数| 区间", u"读数", u"ΔdB/Δ档", u"标准差"))
        only_tgc = stack[np.abs(stack[:, 2]) < 1e-6]
        only_tgc = only_tgc[np.abs(only_tgc[:, 1]) > 5]
        for low, high in [(5, 25), (25, 50), (50, 80), (80, 120), (120, 300)]:
            sel = only_tgc[(np.abs(only_tgc[:, 1]) >= low) & (np.abs(only_tgc[:, 1]) < high)]
            if sel.shape[0] < 4:
                continue
            per_level = sel[:, 0] / sel[:, 1]
            emit(u"%14s %10d %14.5f %12.5f" % (
                u"%d–%d" % (low, high), sel.shape[0], per_level.mean(), per_level.std()))
        emit(u"")
        emit(u"  若各行的 ΔdB/Δ档 明显不同，说明一个标量常数不足以描述滑块，")
        emit(u"  需要改成查找表 —— 那才是 20260819 拟不动的原因。")

    text = u"\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "measure_actuator_steps.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text.encode("ascii", "replace").decode())
    print("\nwrote %s" % out)


if __name__ == "__main__":
    main()
