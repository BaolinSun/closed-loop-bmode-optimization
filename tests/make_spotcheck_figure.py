# -*- coding: utf-8 -*-
"""目视抽查：主机截图 / 按采集设置重建 / 目标函数选出的设置。"""
import io, json, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hisense_backend_sim as S
import calibration as CAL
import backend_solver as BS
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
cal_data = json.load(io.open("bmode_opt/console_calibration.json", encoding="utf-8"))
cals = {}
for g in cal_data["groups"]:
    cals[(g["session"], g["image_mode"])] = CAL.GroupCalibration(
        g["counts_per_db"], g["pivot_db"], g["screenshot_gray_error"], 0,
        np.array(g["depth_axis_mm"]), np.array(g["depth_response_db"]))
meta = {(g["session"], g["image_mode"]): (g["noise_floor_db"], g["target_gray"])
        for g in cal_data["groups"]}

PICKS = [("20260903", 1), ("20260903", 1), ("20260903_replication_check", 1),
         ("20260903_GEN", 0), ("20260904", 0), ("20260904", 0)]
seen = {}
rows = []
for sess, mode in PICKS:
    caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / sess)]
    caps = [c for c in caps if T.capture_image_mode(c) == mode]
    idx = seen.get((sess, mode), 0)
    # spread over display depth
    caps = sorted(caps, key=lambda c: (c.geometry.depth_mm, c.name))
    pick = caps[min(idx * (len(caps) // 2 or 1), len(caps) - 1)]
    seen[(sess, mode)] = idx + 1
    rows.append(((sess, mode), pick))

fig, axes = plt.subplots(len(rows), 3, figsize=(11.5, 3.1 * len(rows)))
for r, (key, cap) in enumerate(rows):
    cal = cals[key]
    floor, target = meta[key]
    shot = crop_capture_image(cap)[0]
    db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
    vm = T.console_tissue_mask(db, floor)
    as_acquired = S.render(
        db_image=db, tgc_levels=cap.tgc_levels, gain_db=S.gain_level_to_db(cap.gain_level),
        dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
        reference_db=cal.pivot_db, depth_response_db=None, out_shape=shot.shape)
    sol = BS.solve_backend(
        db, vm, dr_ui=cap.dynamic_range_level, reference_db=cal.pivot_db,
        target_gray=target,
        current=(S.gain_level_to_db(cap.gain_level),
                 np.asarray(cap.tgc_levels, dtype=np.float64),
                 float(cap.dynamic_range_level)))
    proposed = S.render(
        db_image=db, tgc_levels=sol["tgc_levels"], gain_db=sol["gain_db"],
        dynamic_range_db=S.dr_ui_to_window_db(sol["dr_ui"]),
        reference_db=cal.pivot_db, depth_response_db=None, out_shape=shot.shape)

    mode_name = "harmonic" if key[1] else "general"
    titles = [
        "console screenshot\n%s / %s / %.1f mm" % (key[0][:16], mode_name, cap.geometry.depth_mm),
        "rebuilt at acquired settings\ngain %d, sliders %d, DR %d"
        % (cap.gain_level, cap.tgc_levels[0], cap.dynamic_range_level),
        "objective's choice\nd(gain) %+.0f clicks, sliders %d, scale %.1f"
        % (sol["delta_gain_levels"], sol["tgc_levels"][0], sol["shape_scale"]),
    ]
    for col, (img, title) in enumerate(zip([shot, as_acquired, proposed], titles)):
        ax = axes[r, col]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Step 4 spot check: does the objective's setting look right on real frames?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.985])
out = os.path.join(HERE, "step4_spotcheck.png")
fig.savefig(out, dpi=110)
print("saved", out)
