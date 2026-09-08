# -*- coding: utf-8 -*-
"""第二步验证（实机侧）：按（场次,模式）实测噪声底后，掩膜和解怎么变。"""
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import objective as OBJ
import backend_solver as BS
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

lines = []; w = lines.append
SESSIONS = ["20260903", "20260903_GEN", "20260903_replication_check",
            "20260904", "20260904_DR"]
MODE = {0: u"通用GEN", 1: u"谐波THI"}

groups = {}
for sess in SESSIONS:
    for path in find_captures(DEFAULT_DATA_DIR / sess):
        c = load_capture(path)
        groups.setdefault((sess, T.capture_image_mode(c)), []).append(c)

to_db = lambda cap: S.bc0_to_db(cap.bc0)
floors = T.measure_session_noise_floors(groups, to_db)

w(u"=========== AB. 各组噪声底，以及是自测还是借用 ===========")
w(u"%-30s %9s %7s %11s %9s %-24s" % (
    u"场次", u"模式", u"帧数", u"噪声底dB", u"标准差", u"来源"))
for key in sorted(groups):
    f = floors.get(key)
    if f is None:
        w(u"%-30s %9s %7d %11s %9s %-24s" % (
            key[0], MODE[key[1]], len(groups[key]), u"—", u"—", u"不可判定"))
        continue
    src = (u"自测（%d 帧 >=58mm）" % f["num_frames"] if f["borrowed_from"] is None
           else u"借用 %s" % f["borrowed_from"][0])
    w(u"%-30s %9s %7d %11.2f %9.2f %-24s" % (
        key[0], MODE[key[1]], len(groups[key]), f["floor_db"], f["std_db"], src))

w(u"")
w(u"=========== AC. 实机掩膜：旧的逐帧估计 vs 新的场次实测 ===========")
w(u"%-9s %7s %8s %14s %14s %10s" % (
    u"模式", u"显示深度", u"帧数", u"旧：逐帧估计", u"新：场次实测", u"差"))
rows = {}
for key, caps in groups.items():
    f = floors.get(key)
    if f is None:
        continue
    for c in caps:
        db = S.bc0_to_db(c.bc0)
        old = OBJ.signal_mask(db).mean()
        new = T.console_tissue_mask(db, f["floor_db"]).mean()
        rows.setdefault((MODE[key[1]], round(c.geometry.depth_mm, 1)), []).append((old, new))
for (mode, depth) in sorted(rows):
    vals = np.array(rows[(mode, depth)])
    w(u"%-9s %7.1f %8d %13.1f%% %13.1f%% %9.1f%%" % (
        mode, depth, len(vals), 100 * vals[:, 0].mean(), 100 * vals[:, 1].mean(),
        100 * (vals[:, 1] - vals[:, 0]).mean()))

w(u"")
w(u"=========== AD. 实机最优解：旧掩膜 vs 新掩膜 ===========")
w(u"%-24s %7s %7s %22s %22s" % (
    u"帧", u"深度", u"模式", u"旧（增益/范围/缩放）", u"新（增益/范围/缩放）"))
sample = []
for key in sorted(groups):
    if floors.get(key) is None:
        continue
    for c in groups[key][:2]:
        sample.append((key, c))
for key, c in sample[:8]:
    db = S.bc0_to_db(c.bc0)
    new_mask = T.console_tissue_mask(db, floors[key]["floor_db"])
    pivot_old = float(np.median(db[OBJ.signal_mask(db)]))
    pivot_new = float(np.median(db[new_mask]))
    old = BS.solve_backend(db, reference_db=pivot_old)
    new = BS.solve_backend(db, reference_db=pivot_new, valid_mask=new_mask)
    w(u"%-24s %7.1f %7s %22s %22s" % (
        c.name[:24], c.geometry.depth_mm, MODE[key[1]],
        u"%+.1f / %.0f / %.2f" % (old["gain_db"], old["dr_ui"], old["shape_scale"]),
        u"%+.1f / %.0f / %.2f" % (new["gain_db"], new["dr_ui"], new["shape_scale"])))

w(u"")
w(u"=========== AE. 解有没有顶到搜索网格的边 ===========")
n_edge_gain = n_edge_dr = n = 0
for ptype in ["uniform", "cyst", "point"]:
    for depth, freq in [(30.0, 5.0), (42.0, 5.0), (60.0, 5.0), (42.0, 8.0)]:
        c = load_shard(find_shards(phantom_type=ptype, depth_mm=depth,
                                   frequency_mhz=freq, focus_mm=15.0)[0])
        r = BS.solve_backend(c.db_image, reference_db=c.tissue_median_db,
                             valid_mask=T.fieldii_tissue_mask(c))
        n += 1
        n_edge_gain += r["at_gain_edge"]; n_edge_dr += r["at_dr_edge"]
w(u"  Field II %d 帧：增益顶边界 %d 帧，动态范围顶边界 %d 帧" % (n, n_edge_gain, n_edge_dr))

io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s2_console.txt"),
        "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
