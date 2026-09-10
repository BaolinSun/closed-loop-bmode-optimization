# -*- coding: utf-8 -*-
"""把实机帧切成「同一场景」的族，前端标签的前提。

    为什么需要族

后端的 θ_optimal 是算出来的：BC0 对增益、TGC、动态范围不变，一帧就能在连续空间
里搜出最优。前端三个参数（显示深度、发射频率、发射聚焦）改变 BC0 本身，8 MHz 的
采集里没有任何信息能算出 5 MHz 会长什么样。所以前端的最优只能靠比较**同一场景下
真实采到的多帧**得到，标签生成从「连续求极小」变成「族内离散选择」。

族 = 同一探头位置下的一组帧。族不等于场次：20260903 的 15 帧锚点里有 14 帧互相
0.98-1.00，最后一帧对全部只有 0.69，一个场次里含多个探头位置。

    为什么只能靠锚点切，不能逐帧切

tests/measure_scene_families.py 量过：探头移位后同设置重复帧的相关掉到 0.69，
而**同一位置**下把焦点从 15 mm 转到 30 mm 掉到 0.15，换成像模式掉到 -0.01。旋钮
变化比探头移位更能改变图像，两个分布重叠，相关值区分不了「换了设置」和「动了探头」。

能区分的只有**设置相同的两帧**。所以取场次里出现次数最多的那个前端设置作锚点，
按时间顺序看相邻锚点是否还相关：掉下去就是换了位置，族在此断开。其余帧按时间戳
归入所在区间。当初重复拍锚点就是为了这个。

    阈值

825 对同设置帧的相关直方图在 0.70-0.80 之间是空的——要么 >=0.80（没动），要么
<=0.70（动了）。阈值取在空档中间。
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

import calibration as CAL
import hisense_backend_sim as S

# 相关只在这段物理深度上算。近场是固定的探头结构，最深处是噪声，两头都不带场景信息。
CORR_DEPTH_MM = (8.0, 30.0)
COMMON_ROWS = 256

# 相邻锚点低于此值即认为探头动过。见模块开头「阈值」一节。
SAME_SCENE_CORR = 0.75

# 锚点少于这么多帧，就没有路标可切，整个场次只能当作一族并标记存疑。
MIN_ANCHOR_FRAMES = 3

FRONT_AXES = ("depth_mm", "frequency_mhz", "focus_mm")


def front_setting(capture):
    """一帧的前端设置：成像模式、显示深度、发射频率、发射聚焦。

    成像模式一并带上，因为基波与谐波的频率阶梯几乎不重叠（5.0-11.4 对 4.4-5.7），
    两者的「同一个设置」不是一回事。
    """
    return (S.capture_image_mode(capture), round(capture.geometry.depth_mm, 1),
            round(CAL.capture_frequency(capture), 2), capture.focus_mm)


def scene_profile(capture, counts_per_db=877.3):
    """重采样到公共物理深度轴上的 BC0（dB）。深度设置不同的两帧也能比。

    深度一变，BC0 的行数与 mm/点都变，直接按行比会把不同的物理深度对上。
    """
    db = S.bc0_to_db(capture.bc0, counts_per_db)
    geometry = capture.geometry
    depth = (geometry.min_depth_mm
             + (np.arange(geometry.num_points) + 0.5) * geometry.mm_per_point)
    if depth[-1] < CORR_DEPTH_MM[1]:
        return None
    want = np.linspace(CORR_DEPTH_MM[0], CORR_DEPTH_MM[1], COMMON_ROWS)
    out = np.empty((COMMON_ROWS, db.shape[1]))
    for line in range(db.shape[1]):
        out[:, line] = np.interp(want, depth, db[:, line])
    return out


def scene_correlation(a, b):
    """两帧在公共深度轴上的相关。线数不同则截到较少的那个。"""
    lines = min(a.shape[1], b.shape[1])
    return float(np.corrcoef(a[:, :lines].ravel(), b[:, :lines].ravel())[0, 1])


@dataclass
class SceneFamily:
    """一个探头位置下的全部帧，以及它在三根前端轴上扫到了什么。"""

    session: str
    family_id: str
    frame_names: List[str] = field(default_factory=list)
    anchor_names: List[str] = field(default_factory=list)
    settings: List[Tuple] = field(default_factory=list)
    # 首尾没有锚点夹住的帧：它们归入本族只是因为时间上最近，探头可能已经动了。
    unbracketed: List[str] = field(default_factory=list)
    # 锚点不足，整个场次当作一族。族内比较未经验证。
    anchor_starved: bool = False

    def axis_values(self, axis):
        """本族在某根前端轴上出现过的取值。成像模式分开数。"""
        index = 1 + FRONT_AXES.index(axis)
        out = {}
        for setting in self.settings:
            out.setdefault(setting[0], set()).add(setting[index])
        return {mode: sorted(values) for mode, values in out.items()}

    def swept_axes(self):
        """本族真正扫过的轴——同一成像模式下取到两个以上不同值才算。

        只扫了一根轴的族，另外两根轴没有可比对象，那两个标签必须留空而不是
        填一个默认值。
        """
        return tuple(axis for axis in FRONT_AXES
                     if any(len(v) > 1 for v in self.axis_values(axis).values()))

    def __len__(self):
        return len(self.frame_names)


def _anchor_setting(captures):
    """场次里出现次数最多的前端设置。"""
    counts = {}
    for capture in captures:
        counts[front_setting(capture)] = counts.get(front_setting(capture), 0) + 1
    return max(counts, key=lambda k: (counts[k], k)), counts


def segment_session(captures, profiles=None, threshold=SAME_SCENE_CORR,
                    session=None):
    """把一个场次的帧按探头位置切成若干族。

    captures 必须已按时间戳（也就是目录名）排序。profiles 可预先算好复用；
    为 None 时现算。
    """
    captures = list(captures)
    if not captures:
        return []
    # 20260828 这类场次目录是嵌套的（20260828/GEN），父目录名不足以命名，
    # 所以允许调用方给出完整的场次标签。
    if session is None:
        session = captures[0].path.parent.name
    if profiles is None:
        profiles = [scene_profile(c) for c in captures]

    anchor, counts = _anchor_setting(captures)
    marks = [i for i, c in enumerate(captures)
             if front_setting(c) == anchor and profiles[i] is not None]

    if len(marks) < MIN_ANCHOR_FRAMES:
        family = SceneFamily(session=session, family_id="%s/0" % session,
                             frame_names=[c.name for c in captures],
                             settings=[front_setting(c) for c in captures],
                             unbracketed=[c.name for c in captures],
                             anchor_starved=True)
        return [family]

    # 相邻锚点掉到阈值以下，就在这里断开。
    breaks = [k for k in range(1, len(marks))
              if scene_correlation(profiles[marks[k - 1]],
                                   profiles[marks[k]]) < threshold]

    blocks, start = [], 0
    for cut in breaks + [len(marks)]:
        blocks.append(marks[start:cut])
        start = cut

    families = []
    for number, block in enumerate(blocks):
        first, last = block[0], block[-1]
        # 该族的时间范围：从上一族与本族的中点，到本族与下一族的中点。
        low = 0 if number == 0 else (blocks[number - 1][-1] + first) // 2 + 1
        high = (len(captures) - 1 if number == len(blocks) - 1
                else (last + blocks[number + 1][0]) // 2)
        members = list(range(low, high + 1))
        families.append(SceneFamily(
            session=session,
            family_id="%s/%d" % (session, number),
            frame_names=[captures[i].name for i in members],
            anchor_names=[captures[i].name for i in block],
            settings=[front_setting(captures[i]) for i in members],
            unbracketed=[captures[i].name for i in members
                         if i < first or i > last],
        ))
    return families
