# -*- coding: utf-8 -*-
"""人工核对图：主机截图 / 按采集设置重建 / 目标函数选出的设置。

重建图经同一张调色板上色后再显示，所以三列在同一个视觉域里，可以直接对着看。
第二列与第一列像不像，检验的是标定；第三列比第二列有没有变差，检验的是目标函数。

标注的逐像素误差在主机自己的灰度索引域上算（即截图经调色板反解之后），不是
在 PIL 亮度上——主机不是用亮度显示的，见 display_palette。

用法：python tests/make_spotcheck_figure.py [--out docs/spotcheck.png]
"""

import argparse
import io
import json
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backend_solver as BS
import calibration as CAL
import display_palette as DP
import hisense_backend_sim as S
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

CAL_PATH = "bmode_opt/console_calibration.json"

# 每行一帧，挑成两种模式、几个显示深度、几个场次都覆盖到。
PICKS = [
    ("20260903", 1, 25.1),
    ("20260903", 1, 41.9),
    ("20260903_replication_check", 1, 41.9),
    ("20260904_DR", 1, 41.9),
    ("20260903_GEN", 0, 25.1),
    ("20260903_GEN", 0, 41.9),
    ("20260904", 0, 41.9),
]


def load_groups():
    data = json.load(io.open(CAL_PATH, encoding="utf-8"))
    out = {}
    for g in data["groups"]:
        if not g.get("calibratable", True):
            continue
        out[(g["session"], g["image_mode"])] = {
            "cal": CAL.GroupCalibration(g["counts_per_db"], g["pivot_db"],
                                        g["screenshot_gray_error"], 0,
                                        np.array(g["depth_axis_mm"]),
                                        np.array(g["depth_response_db"])),
            "floor": g["noise_floor_db"],
            "target": g["target_gray"],
            "uncertainty": g["label_uncertainty"],
        }
    return out


def colourise(gray, palette):
    """把灰度索引经调色板上成 RGB，也就是主机显示这幅图时会画出来的样子。"""
    index = np.clip(np.round(np.asarray(gray)), 0, palette.shape[0] - 1).astype(int)
    return np.clip(palette[index], 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/spotcheck.png")
    args = parser.parse_args()

    entries = load_groups()
    rows = []
    for session, mode, depth in PICKS:
        key = (session, mode)
        if key not in entries:
            continue
        captures = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / session)]
        captures = [c for c in captures if T.capture_image_mode(c) == mode
                    and abs(c.geometry.depth_mm - depth) < 1.0]
        if not captures:
            continue
        rows.append((key, sorted(captures, key=lambda c: c.name)[0], entries[key]))

    fig, axes = plt.subplots(len(rows), 3, figsize=(12.5, 3.4 * len(rows)))
    for index, (key, capture, entry) in enumerate(rows):
        calibration, floor = entry["cal"], entry["floor"]
        palette = DP.session_palette(
            [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / key[0])])[0]

        screen_rgb = DP.capture_screen_rgb(capture)
        actual_gray, _ = DP.capture_display_gray(capture, palette=palette)
        from hisense_loader import crop_image_area
        luma = (0.299 * screen_rgb[..., 0] + 0.587 * screen_rgb[..., 1]
                + 0.114 * screen_rgb[..., 2])
        _, (r0, r1, c0, c1) = crop_image_area(luma, capture.geometry.image_width_px)
        shot = screen_rgb[r0:r1, c0:c1].astype(np.uint8)

        db = (S.bc0_to_db(capture.bc0, calibration.counts_per_db)
              + CAL.depth_response_for(capture, calibration)[:, None])
        mask = T.console_tissue_mask(db, floor)

        as_acquired = S.render(
            db_image=db, tgc_levels=capture.tgc_levels,
            gain_db=S.gain_level_to_db(capture.gain_level),
            dynamic_range_db=S.dr_ui_to_window_db(capture.dynamic_range_level),
            reference_db=calibration.pivot_db, depth_response_db=None,
            out_shape=actual_gray.shape)
        error = float(np.median(np.abs(actual_gray - as_acquired.astype(np.float64))))

        solution = BS.solve_backend(
            db, mask, dr_ui=capture.dynamic_range_level,
            reference_db=calibration.pivot_db, target_gray=entry["target"],
            j_uncertainty=entry["uncertainty"],
            current=(S.gain_level_to_db(capture.gain_level),
                     np.asarray(capture.tgc_levels, dtype=np.float64),
                     float(capture.dynamic_range_level)))
        proposed = S.render(
            db_image=db, tgc_levels=solution["tgc_levels"], gain_db=solution["gain_db"],
            dynamic_range_db=S.dr_ui_to_window_db(solution["dr_ui"]),
            reference_db=calibration.pivot_db, depth_response_db=None,
            out_shape=actual_gray.shape)

        mode_name = "harmonic" if key[1] else "general"
        titles = [
            "console screenshot\n%s / %s / %.1f mm"
            % (key[0][:20], mode_name, capture.geometry.depth_mm),
            "rebuilt at acquired settings\ngain %d, sliders %d, DR %d   per-pixel err %.1f"
            % (capture.gain_level, capture.tgc_levels[0],
               capture.dynamic_range_level, error),
            "objective's choice\nd(gain) %+.0f clicks, sliders %d, scale %.1f"
            % (solution["delta_gain_levels"], solution["tgc_levels"][0],
               solution["shape_scale"]),
        ]
        images = [shot, colourise(as_acquired, palette), colourise(proposed, palette)]
        for column, (image, title) in enumerate(zip(images, titles)):
            axis = axes[index, column] if len(rows) > 1 else axes[column]
            axis.imshow(image, aspect="auto")
            axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle("Spot check after the palette fix and the new actuator constants: "
                 "console versus rebuild versus the objective's choice", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print("wrote %s (%d rows)" % (args.out, len(rows)))


if __name__ == "__main__":
    main()
