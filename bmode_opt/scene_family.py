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

    人体数据按采集方案切（segment_protocol）

上面的空档是体模上的：体模不会动。人体颈动脉（2026-10）有呼吸、搏动、手抖，同设置
相邻锚点的相关落满了 0.70-0.80，40 名受试者里 15% 的锚点对低于 0.75，锚点法只切出
105/160 个完整切面。但人体采集是按逐帧列表做的，结构本身就说明了哪些帧属于同一个切面：

    医生帧 -> 深度扫描 25-67 mm -> 回位 -> 频率扫描 4.4-5.7 MHz -> 回位
           -> 聚焦扫描 10-35 mm -> 回位      （后端阶梯受试者再加 12 帧「改一项、存、拨回、存」）

所以人体数据先找扫描（只有一根前端轴在变、其余设置不变的连续帧），按 深度 -> 频率 ->
聚焦 的顺序把扫描归成切面，切面之间按最大的时间间隔断开——换切面要挪探头、重新找图，
那是整段里停得最久的一次（切面内相邻两帧 1-5 s，后端阶梯里最长约 12 s，切面之间
15-90 s）。相关只用来给族打质量标记（motion_flag），不再用来切。
"""

import collections
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
    # anchor_correlation（segment_session）或 protocol（segment_protocol）。
    method: str = "anchor_correlation"
    # 族内相邻锚点相关的最小值；锚点不足两帧时为 None。
    anchor_min_corr: float = None
    # 按方案切出的族里，相邻锚点相关低于 SAME_SCENE_CORR：族内可能动过探头，只作标记。
    motion_flag: bool = False
    # 按方案切出的族里找到的扫描：[(轴, [取值])]。
    sweeps: list = field(default_factory=list)

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
    corr = {k: scene_correlation(profiles[marks[k - 1]], profiles[marks[k]])
            for k in range(1, len(marks))}
    breaks = [k for k in range(1, len(marks)) if corr[k] < threshold]

    blocks, start = [], 0
    for cut in breaks + [len(marks)]:
        blocks.append(marks[start:cut])
        start = cut

    families = []
    position = 0
    for number, block in enumerate(blocks):
        inside = [corr[k] for k in range(position + 1, position + len(block))]
        position += len(block)
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
            anchor_min_corr=min(inside) if inside else None,
        ))
    return families


# ---------------------------------------------------------------- 按采集方案切（人体数据）

# 采集方案里一个切面内三组扫描的先后。
PROTOCOL_ORDER = ("depth_mm", "frequency_mhz", "focus_mm")
# 一组扫描至少取到这么多档才算。方案要求取全（深度 6、频率 5、聚焦 6 档），中途停下的
# 半组仍能给出比较集，所以门槛放低，取到了哪些档记在 sweeps 里。
MIN_SWEEP_VALUES = 3


def capture_seconds(name):
    """17 位采集目录名 YYYYMMDDHHMMSSmmm 换成秒（同一天内可比）；不是这种名字时返回 None。"""
    if len(name) != 17 or not name.isdigit():
        return None
    return (int(name[8:10]) * 3600 + int(name[10:12]) * 60 + int(name[12:14])
            + int(name[14:17]) / 1000.0 + int(name[:8]) * 86400)


def back_setting(capture):
    """后端设置：增益、动态范围、TGC。扫描期间它们不变。"""
    return (capture.gain_level, capture.dynamic_range_level,
            tuple(int(v) for v in capture.tgc_levels))


def find_sweeps(settings, backs, min_values=MIN_SWEEP_VALUES):
    """[(轴, 核心首帧, 核心末帧, 取值)]，按核心首帧排序。

    对每根前端轴，把帧序列切成「成像模式、后端与另外两根前端轴都不变」的最长连续段；
    段内这根轴取到 min_values 档以上就是一组扫描。段里出现最多的取值是回位值（医生帧、
    回位帧都在它上面，并列时取段末那帧的值），核心是第一帧到最后一帧不在回位值上的帧。
    回位帧与医生帧都不进核心，它们属于哪个切面由时间间隔决定（见 protocol_views）。
    """
    out = []
    n = len(settings)
    for axis in PROTOCOL_ORDER:
        index = 1 + FRONT_AXES.index(axis)
        others = [i for i in range(4) if i != index]

        def key(k):
            return tuple(settings[k][i] for i in others), backs[k]

        start = 0
        while start < n:
            end = start
            while end + 1 < n and key(end + 1) == key(start):
                end += 1
            values = [settings[k][index] for k in range(start, end + 1)]
            if len(set(values)) >= min_values:
                counts = collections.Counter(values)
                home = max(counts, key=lambda v: (counts[v], v == values[-1]))
                core = [start + k for k, v in enumerate(values) if v != home]
                out.append((axis, core[0], core[-1], sorted(set(values))))
            start = end + 1
    return sorted(out, key=lambda s: (s[1], s[2]))


def protocol_views(settings, backs, seconds):
    """把帧序列按采集方案切成切面：[(首帧, 末帧, [扫描])]，首尾相接、覆盖全部帧。

    扫描按出现先后归入切面：下一组扫描的轴在方案里排在当前切面最后一组之后
    （深度 -> 频率 -> 聚焦），且这根轴本切面还没有，就并入；否则另起一个切面。
    两个切面之间（前一切面最后一组扫描的核心末帧之后、后一切面第一组的核心首帧及以前）
    在相邻两帧时间间隔最大处断开。没有找到任何扫描时返回 []。
    """
    sweeps = find_sweeps(settings, backs)
    if not sweeps:
        return []
    order = {axis: i for i, axis in enumerate(PROTOCOL_ORDER)}
    groups = []
    for sweep in sweeps:
        current = groups[-1] if groups else None
        if (current is not None and sweep[0] not in {s[0] for s in current}
                and order[sweep[0]] > order[current[-1][0]]):
            current.append(sweep)
        else:
            groups.append([sweep])

    def gap(k):
        if seconds[k] is None or seconds[k - 1] is None:
            return 0.0
        return seconds[k] - seconds[k - 1]

    starts = [0]
    for before, after in zip(groups, groups[1:]):
        last = max(s[2] for s in before)
        first = min(s[1] for s in after)
        # 后端阶梯是「改一项、存、拨回、存」，跟在本切面之后；下一切面从医生帧开始，医生帧
        # 的后端与它的扫描相同。所以断点前一帧须在前一切面的后端上（回位帧），断点处须在
        # 后一切面的后端上。只看时间的话，阶梯里一次 11-12 s 的停顿会把阶梯切走一半。
        cuts = [k for k in range(last + 1, first + 1)
                if backs[k - 1] == backs[last] and backs[k] == backs[first]]
        cuts = cuts or list(range(last + 1, first + 1))
        # 并列取最早的一处：后面多出的同设置帧更可能是下一切面的医生帧。
        cut = max(cuts, key=lambda k: (gap(k), -k)) if cuts else first
        starts.append(max(cut, starts[-1] + 1))
    ends = [k - 1 for k in starts[1:]] + [len(settings) - 1]
    return [(lo, hi, group) for lo, hi, group in zip(starts, ends, groups)]


def segment_protocol(captures, profiles=None, threshold=SAME_SCENE_CORR, session=None):
    """按采集方案把一名受试者的帧切成切面（族）。captures 须已按时间戳排序。

    锚点是族内出现最多的前端设置（即医生帧与各回位帧的设置）。锚点相关只作质量标记：
    相邻锚点低于 threshold 记 motion_flag，最小值记 anchor_min_corr。profiles 为 None
    时只为锚点现算。找不到任何扫描时退回 segment_session。
    """
    captures = list(captures)
    if not captures:
        return []
    if session is None:
        session = captures[0].path.parent.name
    settings = [front_setting(c) for c in captures]
    backs = [back_setting(c) for c in captures]
    seconds = [capture_seconds(c.name) for c in captures]
    views = protocol_views(settings, backs, seconds)
    if not views:
        return segment_session(captures, profiles=profiles, threshold=threshold, session=session)

    families = []
    for number, (lo, hi, sweeps) in enumerate(views):
        members = list(range(lo, hi + 1))
        counts = collections.Counter(settings[i] for i in members)
        anchor = max(counts, key=lambda s: (counts[s], s))
        marks = [i for i in members if settings[i] == anchor]
        corr = []
        for a, b in zip(marks, marks[1:]):
            pa = profiles[a] if profiles is not None else scene_profile(captures[a])
            pb = profiles[b] if profiles is not None else scene_profile(captures[b])
            if pa is not None and pb is not None:
                corr.append(scene_correlation(pa, pb))
        families.append(SceneFamily(
            session=session,
            family_id="%s/%d" % (session, number),
            frame_names=[captures[i].name for i in members],
            anchor_names=[captures[i].name for i in marks],
            settings=[settings[i] for i in members],
            unbracketed=[captures[i].name for i in members if i < marks[0] or i > marks[-1]],
            anchor_starved=len(marks) < 2,
            method="protocol",
            anchor_min_corr=min(corr) if corr else None,
            motion_flag=bool(corr) and min(corr) < threshold,
            sweeps=[(axis, values) for axis, _, _, values in sweeps],
        ))
    return families


def segment(captures, method, session=None, profiles=None):
    """method 为 anchor_correlation（体模）或 protocol（人体）。"""
    if method == "protocol":
        return segment_protocol(captures, profiles=profiles, session=session)
    return segment_session(captures, profiles=profiles, session=session)
