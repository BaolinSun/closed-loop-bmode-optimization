# -*- coding: utf-8 -*-
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, get_leaf

lines = []; w = lines.append
SESSIONS = ["20260903", "20260903_GEN", "20260903_replication_check",
            "20260904", "20260904_DR"]
MODE = {0: u"通用GEN", 1: u"谐波THI"}

groups = {}
for sess in SESSIONS:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        m = int(get_leaf(c.fe_params, "BImageMode"))
        groups.setdefault((sess, m), []).append(c)

w(u"=========== U. 按（场次, 模式）分组测噪声底 ===========")
w(u"  只用显示深度 >= 58 mm 的帧；最深 10% 行的中位数")
w(u"")
w(u"%-30s %10s %7s %10s %12s %9s %10s" % (
    u"场次", u"模式", u"总帧数", u"够深的帧", u"噪声底dB", u"标准差", u"帧间极差"))
floors = {}
for (sess, m), caps in sorted(groups.items()):
    deep = [c for c in caps if c.geometry.depth_mm >= 58.0]
    if not deep:
        w(u"%-30s %10s %7d %10d %12s %9s %10s" % (
            sess, MODE.get(m, m), len(caps), 0, u"—", u"—", u"无够深的帧"))
        continue
    vals = np.array([float(np.median(S.bc0_to_db(c.bc0)[int(c.bc0.shape[0] * 0.9):]))
                     for c in deep])
    floors[(sess, m)] = float(np.median(vals))
    w(u"%-30s %10s %7d %10d %12.2f %9.2f %10.2f" % (
        sess, MODE.get(m, m), len(caps), len(deep), np.median(vals), np.std(vals),
        vals.max() - vals.min()))

w(u"")
w(u"=========== V. 同一模式跨场次是否一致 ===========")
for m in (0, 1):
    same = {k[0]: v for k, v in floors.items() if k[1] == m}
    if len(same) >= 2:
        vals = list(same.values())
        w(u"  %s：%s → 极差 %.2f dB" % (
            MODE[m], u"  ".join(u"%s %.2f" % (k, v) for k, v in same.items()),
            max(vals) - min(vals)))
    elif same:
        w(u"  %s：只有一个场次可测（%s %.2f dB）"
          % (MODE[m], list(same)[0], list(same.values())[0]))
    else:
        w(u"  %s：没有场次可测" % MODE[m])

w(u"")
w(u"=========== W. 没有够深帧的分组怎么办 ===========")
w(u"%-30s %10s %7s %-40s" % (u"场次", u"模式", u"帧数", u"噪声底来源"))
for (sess, m), caps in sorted(groups.items()):
    if (sess, m) in floors:
        src = u"本组自测 %.2f dB" % floors[(sess, m)]
    else:
        same_mode = [(k, v) for k, v in floors.items() if k[1] == m]
        src = (u"借用同模式场次：%s %.2f dB（需核验）" % (same_mode[0][0][0], same_mode[0][1])
               if same_mode else u"无可用来源 → 标为不可判定")
    w(u"%-30s %10s %7d %-40s" % (sess, MODE.get(m, m), len(caps), src))

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_floor2.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
