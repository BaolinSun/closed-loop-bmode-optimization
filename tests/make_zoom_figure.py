# -*- coding: utf-8 -*-
import io, json, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hisense_backend_sim as S
import calibration as CAL
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
cal_data = json.load(io.open("bmode_opt/console_calibration.json", encoding="utf-8"))
cals = {(g["session"], g["image_mode"]): CAL.GroupCalibration(
    g["counts_per_db"], g["pivot_db"], g["screenshot_gray_error"], 0,
    np.array(g["depth_axis_mm"]), np.array(g["depth_response_db"]))
    for g in cal_data["groups"]}

PICKS = [("20260903_GEN", 0), ("20260904", 0), ("20260903", 1)]
fig, axes = plt.subplots(len(PICKS), 3, figsize=(11, 3.6 * len(PICKS)))
for r, (sess, mode) in enumerate(PICKS):
    caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / sess)]
    caps = [c for c in caps if T.capture_image_mode(c) == mode]
    cap = sorted(caps, key=lambda c: (c.geometry.depth_mm, c.name))[len(caps) // 2]
    cal = cals[(sess, mode)]
    shot = crop_capture_image(cap)[0]
    db = S.bc0_to_db(cap.bc0, cal.counts_per_db) + CAL.depth_response_for(cap, cal)[:, None]
    reb = S.render(db_image=db, tgc_levels=cap.tgc_levels,
                   gain_db=S.capture_gain_db(cap),
                   dynamic_range_db=S.dr_ui_to_window_db(cap.dynamic_range_level),
                   reference_db=cal.pivot_db, depth_response_db=None,
                   out_shape=shot.shape).astype(np.float64)
    h, w = shot.shape
    box = (slice(h // 3, h // 3 + 260), slice(w // 2 - 130, w // 2 + 130))
    for col, (img, title) in enumerate(zip(
            [shot[box], reb[box], np.abs(shot - reb)[box]],
            ["console screenshot (1:1)", "rebuilt (1:1)", "absolute difference"])):
        ax = axes[r, col]
        if col == 2:
            im = ax.imshow(img, cmap="magma", vmin=0, vmax=40)
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title("%s\n%s / %s" % (title, sess[:18],
                                      "harmonic" if mode else "fundamental"), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
out = os.path.join(HERE, "zoom.png")
fig.savefig(out, dpi=115)
print("saved", out)
