# -*- coding: utf-8 -*-
"""把 demo 做成一段视频：模型把旋钮拧过去，图像跟着变。

    每一帧都是真的

增益与 TGC 不改变 BC0、只改变渲染，所以从一个设置到另一个设置之间的【任意中间值】都能精确渲染
出来——那正是操作者转旋钮时屏幕上会出现的画面。视频里的平滑过渡因此不是插值出来的假象，是同一帧
原始数据在一串真实设置下的真实渲染。聚焦那一段不同：它改变 BC0 本身，所以用的是同一探头位置下
真实采到的六帧，逐档切换。

    四段

  1. 后端闭环：起点 -> 模型建议 -> 再问一次 -> 停。左边动，中间是教师最优（静止参考），
     右边是增益档位、离最优多少 dB、TGC 曲线。
  2. 聚焦：六档真实采集逐档切换，另给远壁附近的放大图，标出教师与模型各选哪档。
  3. 多位受试者：若干位测试集受试者各走一遍后端闭环，看这件事不是只在一帧上成立。
  4. 数字卡：测试集指标与查表基线。

    用法

    python tools_build_demo_video.py
    python tools_build_demo_video.py --language en --fps 30 --subjects 8
    python tools_build_demo_video.py --out data/demo.mp4 --gif data/demo.gif

写 MP4（优先 ffmpeg 的 libx264，PowerPoint 与浏览器都能放；没有 ffmpeg 时退回 OpenCV 的 mp4v）。
--gif 另写一份动图。终端输出与写在脚本旁边的 tools_build_demo_video.txt 同内容（ASCII）。
"""

import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "bmode_opt")
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import bmode_dl.constants as K
import tools_build_demo_page as DP
import tools_suggest_six_param as SG

REPORT_PATH = "tools_build_demo_video.txt"
WIDTH, HEIGHT = 1280, 720
IMAGE_BOX = (392, 450)            # 单幅 B 超图的最大显示尺寸
FONTS = {"body": "C:/Windows/Fonts/segoeui.ttf", "bold": "C:/Windows/Fonts/segoeuib.ttf",
         "data": "C:/Windows/Fonts/consola.ttf",
         "cjk": "C:/Windows/Fonts/msyh.ttc", "cjk_bold": "C:/Windows/Fonts/msyhbd.ttc"}
INK = (238, 238, 232)
INK_DIM = (150, 150, 142)
BG = (18, 18, 17)
PANEL = (26, 26, 25)
BLUE = (57, 135, 229)
ORANGE = (217, 89, 38)
GREEN = (25, 158, 112)

TEXT = {
    "zh": {
        "title": "超声成像六参数闭环优化",
        "subtitle": "颈动脉在体数据；受试者取自留出测试集，未参与训练与模型选择",
        "scene1": "一、后端参数闭环迭代：增益与时间增益补偿",
        "live": "模型输出参数下的渲染结果",
        "teacher": "求解器最优解（参考基准）",
        "gain": "增益档位",
        "off": "与最优解偏差",
        "tgc": "时间增益补偿曲线（dB）",
        "profile": "逐行亮度剖面（灰阶中位数）",
        "profile_axis": "深度（mm）",
        "tgc_now": "当前",
        "tgc_teacher": "最优解",
        "step": "第 %d 次迭代",
        "start": "初始状态：主机采集设置，时间增益补偿各段均为 127",
        "moving": "执行模型输出：增益 %+d 档，时间增益补偿曲线更新",
        "moving_tgc": "执行模型输出：增益保持，时间增益补偿曲线更新",
        "stopped": "停止判据满足，迭代终止",
        "scene2": "二、发射聚焦：六档实采图像与判据",
        "widths": "侧向散斑宽度（mm）——数值越低分辨率越高",
        "widths_axis": "深度（mm）",
        "probs": "模型对各档的输出概率",
        "far_wall_mark": "远壁 %.1f mm",
        "scene_bias": "四、误差的分布形态：增益残差以受试者为单位呈常数偏置",
        "bias_axis": "验证集各受试者（按偏置排序）",
        "bias_unit": "与最优解的偏置（dB，带符号）",
        "bias_note": "受试者内一致性 %.2f dB；%d / %d 位在 ±1 dB 以内；平方误差中 %.0f%% 来自受试者间偏置",
        "focus_label": "发射聚焦 %g mm",
        "zoom": "远壁区域局部放大",
        "pick_teacher": "求解器最优档",
        "pick_model": "模型输出档",
        "scene3": "三、跨受试者验证：%d 位留出测试集受试者同屏演示",
        "subject": "受试者 %s",
        "from_to": "增益与最优解偏差 %.2f dB -> %.2f dB",
        "grid_note": "六位按收敛结果均匀取秩：最好、中间各档与最差各一位，故最差一例一并在列",
        "scene4": "五、留出测试集评估结果（29 位受试者）",
        "net": "本方法",
        "base": "设置查表基线",
        "row_tgc": "时间增益补偿\n平均绝对误差",
        "row_gain": "增益平均绝对误差",
        "row_focus": "发射聚焦\n首选命中率",
        "lower": "数值越低越优",
        "higher": "数值越高越优",
        "closing": "增益与时间增益补偿已达可用精度；发射聚焦略优于基线；显示深度、发射频率与动态范围尚未训练",
        "note_render": "各帧均由同一组原始回波数据经后端渲染模型精确重建，非示意图",
    },
    "en": {
        "title": "Closed-loop optimisation of six B-mode parameters",
        "subtitle": "In-vivo carotid data; subject drawn from the held-out test set, unseen in training "
                    "and model selection",
        "scene1": "1. Closed-loop iteration of the back-end parameters: gain and time-gain compensation",
        "live": "render under the model's predicted parameters",
        "teacher": "solver optimum (reference)",
        "gain": "gain level",
        "off": "deviation from the optimum",
        "tgc": "time-gain compensation curve (dB)",
        "profile": "depth brightness profile (median grey)",
        "profile_axis": "depth (mm)",
        "tgc_now": "current",
        "tgc_teacher": "optimum",
        "step": "iteration %d",
        "start": "initial state: console acquisition setting, all TGC bands at 127",
        "moving": "applying the prediction: gain %+d levels, TGC curve updated",
        "moving_tgc": "applying the prediction: gain unchanged, TGC curve updated",
        "stopped": "stopping criterion met; iteration terminated",
        "scene2": "2. Transmit focus: six acquisitions and the criterion behind the label",
        "widths": "lateral speckle width (mm) - lower means finer resolution",
        "widths_axis": "depth (mm)",
        "probs": "model probability per step",
        "far_wall_mark": "far wall %.1f mm",
        "scene_bias": "4. Structure of the error: the gain residual is a constant offset per subject",
        "bias_axis": "validation subjects, sorted by bias",
        "bias_unit": "signed deviation from the optimum (dB)",
        "bias_note": "within-subject consistency %.2f dB; %d of %d inside +-1 dB; %.0f%% of the squared "
                     "error is between-subject bias",
        "focus_label": "transmit focus %g mm",
        "zoom": "far-wall region, magnified",
        "pick_teacher": "solver optimum",
        "pick_model": "model prediction",
        "scene3": "3. Cross-subject validation: %d held-out test subjects in parallel",
        "subject": "subject %s",
        "from_to": "deviation from the optimum %.2f dB -> %.2f dB",
        "grid_note": "the six are taken at even ranks of the converged deviation - best, intermediate and "
                     "worst - so the worst case is deliberately included",
        "scene4": "5. Evaluation on the held-out test set (29 subjects)",
        "net": "proposed method",
        "base": "settings-lookup baseline",
        "row_tgc": "TGC mean\nabsolute error",
        "row_gain": "gain mean\nabsolute error",
        "row_focus": "transmit focus\ntop-1 accuracy",
        "lower": "lower is better",
        "higher": "higher is better",
        "closing": "gain and TGC reach usable accuracy; transmit focus marginally exceeds the baseline; "
                   "display depth, transmit frequency and dynamic range are not yet trained",
        "note_render": "every frame is reconstructed exactly from the same recorded envelope through the "
                       "back-end render model; none is an illustration",
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Render the six-parameter demo as a video")
    p.add_argument("--family", default=None, help="scene for part 1 and 2 (default: the same median pick "
                                                 "tools_build_demo_page.py makes)")
    p.add_argument("--subjects", type=int, default=6, help="how many extra subjects in part 3")
    p.add_argument("--language", choices=("zh", "en"), default="zh")
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--out", default="data/demo_invivo_carotid.mp4")
    p.add_argument("--gif", default=None, help="also write an animated GIF here")
    p.add_argument("--frontend", default=os.path.join(DP.DEFAULT_RUN, "frontend.pt"))
    p.add_argument("--backend", default=os.path.join(DP.DEFAULT_RUN, "last.pt"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--labels", default="data/labels_console_invivo_carotid_target.jsonl")
    p.add_argument("--dataset", default="invivo", choices=("auto", "phantom", "invivo"))
    p.add_argument("--calibration", default=None)
    p.add_argument("--calibration-group", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--stop-deadband-levels", type=float, default=0.5)
    p.add_argument("--max-gain-step-clicks", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=3)
    p.add_argument("--candidates", type=int, default=16)
    p.add_argument("--single-step-json", default="runs/invivo_bf_v2/fold_0/evaluation_combined_test_noloop.json")
    p.add_argument("--loop-trajectories",
                   default="runs/invivo_bf_v2/fold_0/closed_loop_trajectories_combined_bloop.jsonl")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------
#   画布

def load_fonts(language):
    body = FONTS["cjk"] if language == "zh" else FONTS["body"]
    bold = FONTS["cjk_bold"] if language == "zh" else FONTS["bold"]
    pick = lambda path, size: ImageFont.truetype(path, size)
    return {"h1": pick(bold, 40), "h2": pick(bold, 24), "body": pick(body, 19), "small": pick(body, 16),
            "data": pick(FONTS["data"], 20), "data_small": pick(FONTS["data"], 15),
            "big": pick(FONTS["data"], 44)}


def gray_to_image(gray, depth_mm, box=IMAGE_BOX):
    """灰阶数组 -> 按物理比例缩放到 box 内的 RGB 图。"""
    image = Image.fromarray(np.clip(gray, 0, 255).astype(np.uint8), mode="L").convert("RGB")
    scale = min(box[0] / (DP.SCAN_WIDTH_MM * DP.PIXELS_PER_MM), box[1] / (depth_mm * DP.PIXELS_PER_MM))
    size = (max(1, int(round(DP.SCAN_WIDTH_MM * DP.PIXELS_PER_MM * scale))),
            max(1, int(round(depth_mm * DP.PIXELS_PER_MM * scale))))
    return image.resize(size, Image.BILINEAR)


def canvas(fonts, strings, header, caption=None):
    """统一的底板：标题带 + 底部字幕带。"""
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, WIDTH, 52], fill=PANEL)
    draw.text((36, 14), header, font=fonts["h2"], fill=INK)
    draw.line([0, 52, WIDTH, 52], fill=(54, 54, 52))
    draw.line([0, HEIGHT - 58, WIDTH, HEIGHT - 58], fill=(54, 54, 52))
    if caption:
        draw.text((36, HEIGHT - 44), caption, font=fonts["body"], fill=INK_DIM)
    return image, draw


def paste_panel(image, draw, fonts, picture, x, y, label, accent=None):
    """一幅图 + 上方小标题，带细边框。"""
    draw.text((x, y - 26), label, font=fonts["small"], fill=INK_DIM)
    image.paste(picture, (x, y))
    box = [x - 1, y - 1, x + picture.size[0], y + picture.size[1]]
    draw.rectangle(box, outline=accent or (70, 70, 66), width=2 if accent else 1)


def draw_tgc_curve(draw, fonts, strings, x, y, w, h, now_levels, teacher_levels, slope):
    """小幅 TGC 曲线：当前与教师两条。"""
    draw.text((x, y - 24), strings["tgc"], font=fonts["small"], fill=INK_DIM)
    draw.rectangle([x, y, x + w, y + h], fill=PANEL, outline=(54, 54, 52))
    now = (np.asarray(now_levels, float) - K.TGC_CENTER_LEVEL) * slope
    teacher = (np.asarray(teacher_levels, float) - K.TGC_CENTER_LEVEL) * slope
    span = max(6.0, float(np.abs(np.concatenate([now, teacher])).max()) * 1.15)
    bands = len(now)
    px = lambda i: x + 14 + i * (w - 28) / (bands - 1.0)
    py = lambda v: y + h / 2.0 - v / span * (h / 2.0 - 12)
    draw.line([x + 10, py(0.0), x + w - 10, py(0.0)], fill=(74, 74, 70))
    for values, colour in ((teacher, ORANGE), (now, BLUE)):
        points = [(px(i), py(float(values[i]))) for i in range(bands)]
        draw.line(points, fill=colour, width=3, joint="curve")
        for point in points:
            draw.ellipse([point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3], fill=colour)
    draw.text((x + 12, y + h - 24), strings["tgc_now"], font=fonts["small"], fill=BLUE)
    draw.text((x + 12 + 60, y + h - 24), strings["tgc_teacher"], font=fonts["small"], fill=ORANGE)


def draw_wrapped(draw, font, text, x, y, width, fill, line_height=27):
    """按像素宽度断行。学术表达比口语长，截断会丢信息。中文逐字断，西文按空格断。"""
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ") if " " in paragraph else list(paragraph)
        joiner = " " if " " in paragraph else ""
        line = ""
        for word in words:
            candidate = (line + joiner + word) if line else word
            if draw.textlength(candidate, font=font) <= width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
    for i, one in enumerate(lines[:3]):
        draw.text((x, y + i * line_height), one, font=font, fill=fill)
    return len(lines[:3])


def draw_series_chart(draw, fonts, x, y, w, h, series, title, axis_label, mark_x=None, mark_label=None):
    """小幅折线图。series = [(标签, [(x, y), ...], 颜色, 是否高亮)]。"""
    draw.text((x, y - 24), title, font=fonts["small"], fill=INK_DIM)
    draw.rectangle([x, y, x + w, y + h], fill=PANEL, outline=(54, 54, 52))
    points = [p for _, pts, _, _ in series for p in pts]
    if not points:
        draw.text((x + 12, y + h / 2 - 10), "-", font=fonts["small"], fill=INK_DIM)
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-6:
        x1 = x0 + 1.0
    pad = (y1 - y0) * 0.25 + 1e-6
    y0, y1 = max(0.0, y0 - pad), y1 + pad
    px = lambda v: x + 44 + (v - x0) / (x1 - x0) * (w - 60)
    py = lambda v: y + h - 30 - (v - y0) / (y1 - y0) * (h - 48)
    draw.text((x + 6, py(y1) - 7), "%.1f" % y1, font=fonts["data_small"], fill=INK_DIM)
    draw.text((x + 6, py(y0) - 7), "%.1f" % y0, font=fonts["data_small"], fill=INK_DIM)
    draw.text((x + 44, y + h - 18), "%.0f" % x0, font=fonts["data_small"], fill=INK_DIM)
    draw.text((x + w - 70, y + h - 18), "%.0f" % x1, font=fonts["data_small"], fill=INK_DIM)
    draw.text((x + w / 2 - 28, y + h - 18), axis_label, font=fonts["small"], fill=INK_DIM)
    if mark_x is not None and x0 <= mark_x <= x1:
        draw.line([px(mark_x), y + 10, px(mark_x), y + h - 24], fill=ORANGE, width=1)
        if mark_label:
            draw.text((px(mark_x) + 4, y + 8), mark_label, font=fonts["small"], fill=ORANGE)
    for label, pts, colour, strong in series:
        if len(pts) >= 2:
            draw.line([(px(a), py(b)) for a, b in pts], fill=colour, width=3 if strong else 1,
                      joint="curve")
        for a, b in pts:
            r = 4 if strong else 2
            draw.ellipse([px(a) - r, py(b) - r, px(a) + r, py(b) + r], fill=colour)
        if strong and label:
            draw.text((px(pts[-1][0]) - 34, py(pts[-1][1]) - 26), label, font=fonts["small"], fill=colour)


def draw_probability_bars(draw, fonts, x, y, w, h, ladder, probabilities, current, teacher, title):
    """模型对各档的概率：当前档填蓝，求解器最优档加橙框。"""
    draw.text((x, y - 24), title, font=fonts["small"], fill=INK_DIM)
    draw.rectangle([x, y, x + w, y + h], fill=PANEL, outline=(54, 54, 52))
    if not ladder:
        return
    high = max(max(probabilities), 1e-6)
    slot = (w - 24) / float(len(ladder))
    for i, (value, probability) in enumerate(zip(ladder, probabilities)):
        bx = x + 12 + i * slot
        bar = (h - 44) * probability / high
        here = abs(value - current) < 1e-6
        best = abs(value - teacher) < 1e-6
        draw.rectangle([bx + 4, y + h - 26 - bar, bx + slot - 8, y + h - 26],
                       fill=BLUE if here else (74, 74, 70))
        if best:
            draw.rectangle([bx + 2, y + h - 28 - bar, bx + slot - 6, y + h - 24], outline=ORANGE, width=2)
        draw.text((bx + 4, y + h - 22), "%g" % value, font=fonts["data_small"], fill=INK if here else INK_DIM)
    draw.text((x + w - 54, y + 8), "%.2f" % high, font=fonts["data_small"], fill=INK_DIM)


def draw_readout(draw, fonts, strings, x, y, gain_level, gain_error, tgc_error):
    """右栏的数字。"""
    draw.text((x, y), strings["gain"], font=fonts["small"], fill=INK_DIM)
    draw.text((x, y + 20), "%d" % gain_level, font=fonts["big"], fill=INK)
    draw.text((x, y + 78), strings["off"], font=fonts["small"], fill=INK_DIM)
    colour = GREEN if abs(gain_error) <= 1.0 else (ORANGE if abs(gain_error) <= 3.0 else (210, 90, 70))
    draw.text((x, y + 98), "%+.2f dB" % gain_error, font=fonts["data"], fill=colour)
    draw.text((x, y + 124), "TGC %.2f dB" % tgc_error, font=fonts["data"], fill=INK_DIM)


# ---------------------------------------------------------------------------------------
#   场景

def interpolate(a, b, t):
    return a + (b - a) * t


def loop_timeline(steps, fps, hold_start=1.6, move=1.2, hold_mid=0.7, hold_end=2.2):
    """闭环关键帧 -> 逐视频帧的 (gain_db, tgc_levels, 阶段, 第几步) 序列。"""
    out = []
    for i, step in enumerate(steps):
        gain_a, tgc_a = step["gain_db"], np.asarray(step["tgc_levels"], float)
        hold = hold_start if i == 0 else (hold_end if step["stopped"] else hold_mid)
        for _ in range(int(round(hold * fps))):
            out.append((gain_a, tgc_a, "hold", i))
        if step["stopped"] or i + 1 >= len(steps):
            break
        gain_b = steps[i + 1]["gain_db"]
        tgc_b = np.asarray(steps[i + 1]["tgc_levels"], float)
        frames = int(round(move * fps))
        for k in range(frames):
            t = (k + 1) / float(frames)
            smooth = t * t * (3 - 2 * t)                 # 缓入缓出，像手在拧旋钮
            out.append((interpolate(gain_a, gain_b, smooth),
                        np.round(interpolate(tgc_a, tgc_b, smooth)), "move", i))
    return out


def row_profile(gray, depth_mm, points=56):
    """逐行灰阶中位数 vs 深度（降采样）。"""
    median = np.median(np.asarray(gray, float), axis=1)
    rows = len(median)
    edges = np.linspace(0, rows, points + 1).round().astype(int)
    out = []
    for i in range(points):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out.append((float((lo + hi) / 2.0 / rows * depth_mm), float(median[lo:hi].mean())))
    return out


def scene_backend(frame, row, steps, fonts, strings, fps, write):
    """第 1 段：左边动、中间教师最优、右边数字与曲线。"""
    slope = K.TGC_DB_PER_LEVEL[int(frame["mode"])]
    optimal_gain = float(row["optimal_gain_db"])
    optimal_levels = np.asarray(row["optimal_tgc_levels"], float)
    teacher_gray = DP.render_gray(frame, optimal_gain, optimal_levels)
    teacher_picture = gray_to_image(teacher_gray, frame["max_depth_mm"], box=(336, 400))
    teacher_profile = row_profile(teacher_gray, frame["max_depth_mm"])
    for gain_db, tgc_levels, phase, index in loop_timeline(steps, fps):
        image, draw = canvas(fonts, strings, strings["scene1"], strings["note_render"])
        gray = DP.render_gray(frame, gain_db, tgc_levels)
        live = gray_to_image(gray, frame["max_depth_mm"], box=(336, 400))
        paste_panel(image, draw, fonts, live, 48, 104, strings["live"], BLUE if phase == "move" else None)
        paste_panel(image, draw, fonts, teacher_picture, 48 + 336 + 40, 104, strings["teacher"])
        right = 48 + 2 * (336 + 40)
        gain_error = optimal_gain - gain_db
        tgc_error = float(np.abs(optimal_levels - np.asarray(tgc_levels, float)).mean() * slope)
        draw_readout(draw, fonts, strings, right, 104, int(round(frame["gain_level"]
                                                                 + (gain_db - frame["gain_db"])
                                                                 / K.GAIN_DB_PER_LEVEL[int(frame["mode"])])),
                     gain_error, tgc_error)
        draw_tgc_curve(draw, fonts, strings, right, 290, WIDTH - right - 48, 150, tgc_levels,
                       optimal_levels, slope)
        draw_series_chart(draw, fonts, right, 486, WIDTH - right - 48, 150,
                          [(strings["tgc_teacher"], teacher_profile, ORANGE, False),
                           (strings["tgc_now"], row_profile(gray, frame["max_depth_mm"]), BLUE, True)],
                          strings["profile"], strings["profile_axis"])
        step = steps[index]
        if index == 0 and phase == "hold":
            note = strings["start"]
        elif step["stopped"]:
            note = strings["stopped"]
        elif phase == "move":
            note = (strings["moving"] % step["suggestion"]["clicks"] if step["suggestion"]["clicks"]
                    else strings["moving_tgc"])
        else:
            note = strings["step"] % step["step"]
        draw_wrapped(draw, fonts["body"], note, 48, 104 + 400 + 36, WIDTH - 96, INK)
        write(image)


def focus_sharpness(frames, args):
    """每档聚焦的逐深度带侧向散斑宽度——聚焦标签正是按这个量定的。"""
    import console_sessions as CS
    import tools_generate_console_labels as G
    import tools_generate_labels as TG
    from hisense_loader import load_capture

    cal_by_group = TG.load_calibration(args.calibration or CS.output_path(args.dataset, "calibration"))
    G.floors_in_counts(cal_by_group)
    out = []
    for frame in frames:
        try:
            session, mode = frame["calibration"]["group"].rsplit("/", 1)
            entry = cal_by_group[(session, int(mode))]
            widths = G.measure_frame(load_capture(frame["path"]), entry["cal"], entry["floor"])["widths"]
            out.append(sorted((float(d), float(v)) for d, v in widths.items()))
        except Exception:
            out.append([])
    return out


def scene_focus(frames, results, order, rows, fonts, strings, fps, args, hold=1.5):
    """第 2 段：图像 + 远壁放大 + 侧向散斑宽度曲线 + 模型概率。"""
    widths = focus_sharpness([frames[i] for i in order], args)
    out = []
    for position, i in enumerate(order):
        f = frames[i]
        row = rows[f["frame_id"]]
        gray = DP.render_gray(f, row["optimal_gain_db"], row["optimal_tgc_levels"])
        picture = gray_to_image(gray, f["max_depth_mm"], box=(300, 450))
        far_wall = row.get("far_wall_mm")
        rows_total = gray.shape[0]
        centre = int(rows_total * (far_wall or f["max_depth_mm"] * 0.45) / f["max_depth_mm"])
        half = max(24, int(rows_total * 6.0 / f["max_depth_mm"]))
        crop = gray[max(0, centre - half):min(rows_total, centre + half), :]
        zoom = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8), mode="L").convert("RGB")
        zoom = zoom.resize((300, max(60, int(300 * crop.shape[0] / float(crop.shape[1]) * 0.55))),
                           Image.LANCZOS)
        teacher_focus = round(float(row["optimal_focus_mm"]), 1)
        focus_axis = results[i]["frontend"]["focus"]
        ladder = [v for v, p in zip(focus_axis.get("ladder", []), focus_axis.get("probabilities", []))
                  if any(abs(v - s) < 1e-6 for s in focus_axis.get("selectable", []))]
        probabilities = [p for v, p in zip(focus_axis.get("ladder", []), focus_axis.get("probabilities", []))
                         if any(abs(v - s) < 1e-6 for s in focus_axis.get("selectable", []))]
        series = []
        for k, other in enumerate(order):
            strong = other == i
            series.append(("%g mm" % frames[other]["focus_mm"], widths[k], BLUE if strong else (70, 70, 66),
                           strong))
        series.sort(key=lambda item: item[3])
        for _ in range(int(round(hold * fps))):
            image, draw = canvas(fonts, strings, strings["scene2"], strings["note_render"])
            accent = ORANGE if abs(teacher_focus - f["focus_mm"]) < 1e-6 else (
                BLUE if focus_axis["steps"] == 0 else None)
            paste_panel(image, draw, fonts, picture, 48, 104, strings["focus_label"] % f["focus_mm"], accent)
            if far_wall:
                yline = 104 + int(picture.size[1] * far_wall / f["max_depth_mm"])
                draw.line([48, yline, 48 + picture.size[0], yline], fill=ORANGE, width=1)
                label = strings["far_wall_mark"] % far_wall
                tw = draw.textlength(label, font=fonts["small"])
                draw.rectangle([50, yline + 2, 50 + tw + 10, yline + 26], fill=(14, 14, 13))
                draw.text((55, yline + 4), label, font=fonts["small"], fill=ORANGE)
            middle = 48 + 300 + 36
            paste_panel(image, draw, fonts, zoom, middle, 104, strings["zoom"])
            draw_series_chart(draw, fonts, middle, 104 + zoom.size[1] + 58, 300, 182, series,
                              strings["widths"], strings["widths_axis"],
                              mark_x=far_wall, mark_label=strings["far_wall_mark"] % far_wall
                              if far_wall else None)
            far = 48 + 2 * (300 + 36)
            width_right = WIDTH - far - 48
            draw_probability_bars(draw, fonts, far, 104, width_right, 182, ladder, probabilities,
                                  f["focus_mm"], teacher_focus, strings["probs"])
            y = 104 + 182 + 28
            if abs(teacher_focus - f["focus_mm"]) < 1e-6:
                draw.text((far, y), strings["pick_teacher"], font=fonts["body"], fill=ORANGE)
                y += 32
            if focus_axis["steps"] == 0:
                draw.text((far, y), strings["pick_model"], font=fonts["body"], fill=BLUE)
            out.append(image)
    return out


def scene_subjects(stories, fonts, strings, fps, per=5.2):
    """\u7b2c 3 \u6bb5\uff1a2x3 \u5bab\u683c\uff0c\u516d\u4f4d\u53d7\u8bd5\u8005\u540c\u5c4f\u4ece\u91c7\u96c6\u8bbe\u7f6e\u8d70\u5230\u6a21\u578b\u8f93\u51fa\u3002"""
    out = []
    columns, rows_count = 3, 2
    cells = stories[:columns * rows_count]
    if not cells:
        return out
    gap_x, gap_y = 24, 18
    cell_w = (WIDTH - 96 - (columns - 1) * gap_x) // columns
    cell_h = (HEIGHT - 52 - 58 - 40 - (rows_count - 1) * gap_y) // rows_count
    box = (cell_w, cell_h - 46)
    prepared = []
    for index, story in enumerate(cells):
        frame = story["frames"][story["anchor"]]
        row = story["row"]
        first, last = story["steps"][0], story["steps"][-1]
        prepared.append({
            "frame": frame, "row": row,
            "gain_a": first["gain_db"], "gain_b": last["gain_db"],
            "tgc_a": np.asarray(first["tgc_levels"], float), "tgc_b": np.asarray(last["tgc_levels"], float),
            "optimal_gain": float(row["optimal_gain_db"]),
            "optimal_levels": np.asarray(row["optimal_tgc_levels"], float),
            "subject": "S%02d" % (index + 1),
            "start": story["start_gain"], "final": story["final_gain"]})
    total = int(round(per * fps))
    settle = int(0.8 * fps)
    for k in range(total):
        t = min(1.0, max(0.0, (k - settle) / float(max(1, total - 2 * settle))))
        smooth = t * t * (3 - 2 * t)
        image, draw = canvas(fonts, strings, strings["scene3"] % len(cells), strings["grid_note"])
        for index, item in enumerate(prepared):
            cx = 48 + (index % columns) * (cell_w + gap_x)
            cy = 92 + (index // columns) * (cell_h + gap_y)
            gain_db = interpolate(item["gain_a"], item["gain_b"], smooth)
            tgc = np.round(interpolate(item["tgc_a"], item["tgc_b"], smooth))
            picture = gray_to_image(DP.render_gray(item["frame"], gain_db, tgc),
                                    item["frame"]["max_depth_mm"], box=box)
            image.paste(picture, (cx, cy))
            draw.rectangle([cx - 1, cy - 1, cx + picture.size[0], cy + picture.size[1]],
                           outline=BLUE if 0.0 < t < 1.0 else (70, 70, 66), width=1)
            error = item["optimal_gain"] - gain_db
            colour = GREEN if abs(error) <= 1.0 else (ORANGE if abs(error) <= 3.0 else (210, 90, 70))
            draw.text((cx, cy + picture.size[1] + 6), strings["subject"] % item["subject"],
                      font=fonts["small"], fill=INK_DIM)
            draw.text((cx, cy + picture.size[1] + 26), "%+.2f dB" % error, font=fonts["data"], fill=colour)
        out.append(image)
    return out


def scene_bias(bias, fonts, strings, fps, hold=4.5):
    """第 4 段：每位受试者一个点，点逐个出现。"""
    out = []
    subjects = bias["subjects"]
    total = int(round(hold * fps))
    appear = int(total * 0.6)
    span = max(5.0, max(abs(s["bias"]) for s in subjects) * 1.12)
    left, right, top, bottom = 90, 80, 120, 150
    width, height = WIDTH - left - right, HEIGHT - top - bottom
    for k in range(total):
        shown = len(subjects) if k >= appear else max(1, int(len(subjects) * (k + 1) / float(appear)))
        image, draw = canvas(fonts, strings, strings["scene_bias"], strings["note_render"])
        px = lambda i: left + i * width / max(1.0, len(subjects) - 1.0)
        py = lambda v: top + (span - v) * height / (2 * span)
        draw.rectangle([left, py(1.0), left + width, py(-1.0)], fill=(24, 42, 34))
        draw.line([left, py(0.0), left + width, py(0.0)], fill=(74, 74, 70))
        for value in (-4, -2, 0, 2, 4):
            if abs(value) <= span:
                draw.text((left - 42, py(value) - 10), "%+d" % value, font=fonts["data_small"], fill=INK_DIM)
        for i, subject in enumerate(subjects[:shown]):
            colour = ORANGE if abs(subject["bias"]) > 1.0 else BLUE
            draw.line([px(i), py(0.0), px(i), py(subject["bias"])], fill=colour, width=2)
            draw.ellipse([px(i) - 5, py(subject["bias"]) - 5, px(i) + 5, py(subject["bias"]) + 5],
                         fill=colour)
        draw.text((left, HEIGHT - 148), strings["bias_axis"], font=fonts["small"], fill=INK_DIM)
        draw.text((left, HEIGHT - 124), strings["bias_unit"], font=fonts["small"], fill=INK_DIM)
        draw.text((left, HEIGHT - 94),
                  strings["bias_note"] % (bias["sd_median"], bias["within_1db"], len(subjects),
                                          100 * bias["bias_share"]),
                  font=fonts["body"], fill=INK)
        out.append(image)
    return out


def scene_numbers(net, base, fonts, strings, fps, hold=4.0):
    """第 4 段：数字卡。"""
    pairs = [(strings["row_tgc"], net.get("tgc_mae_db"), base.get("tgc_mae_db"), "%.2f dB", "lower"),
             (strings["row_gain"], net.get("gain_mae_db"), base.get("gain_mae_db"), "%.2f dB", "lower"),
             (strings["row_focus"], net.get("focus_top1"), base.get("focus_top1"), "%.3f", "higher")]
    image, draw = canvas(fonts, strings, strings["scene4"], strings["closing"])
    draw.text((64, 112), strings["net"], font=fonts["small"], fill=BLUE)
    draw.text((64 + 180, 112), strings["base"], font=fonts["small"], fill=INK_DIM)
    y = 160
    for name, a, b, unit, direction in pairs:
        if a is None or b is None:
            continue
        used = draw_wrapped(draw, fonts["body"], name, 64, y, 170, INK, line_height=24)
        draw.text((64, y + 6 + used * 24), strings[direction], font=fonts["small"], fill=(110, 110, 104))
        draw.text((64 + 180, y - 6), unit % a, font=fonts["big"], fill=BLUE)
        draw.text((64 + 420, y + 4), unit % b, font=fonts["data"], fill=INK_DIM)
        high = max(a, b) * 1.25
        for j, (value, colour) in enumerate(((a, BLUE), (b, (92, 92, 86)))):
            w = (WIDTH - 700) * value / high
            draw.rectangle([640, y + 2 + j * 20, 640 + max(3, w), y + 16 + j * 20], fill=colour)
        y += 140
    return [image] * int(round(hold * fps))


def scene_title(fonts, strings, fps, hold=2.4):
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((64, 280), strings["title"], font=fonts["h1"], fill=INK)
    draw.text((64, 346), strings["subtitle"], font=fonts["body"], fill=INK_DIM)
    draw.line([64, 262, 300, 262], fill=BLUE, width=3)
    return [image] * int(round(hold * fps))


# ---------------------------------------------------------------------------------------
#   写文件

def write_video(frames_dir, count, out_path, fps, emit):
    """优先 ffmpeg 的 libx264（到处都能放），没有就退回 OpenCV 的 mp4v。"""
    pattern = os.path.join(frames_dir, "%06d.png")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
                   out_path]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode == 0:
            emit("  encoder    ffmpeg libx264")
            return True
        emit("  ffmpeg failed: %s" % result.stdout.decode("utf-8", "replace").strip()[:200])
    import cv2

    first = cv2.imread(pattern % 0)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (first.shape[1], first.shape[0]))
    for i in range(count):
        writer.write(cv2.imread(pattern % i))
    writer.release()
    emit("  encoder    OpenCV mp4v (ffmpeg not available; H.264 is more portable)")
    return True


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    lines = []

    def emit(text=""):
        lines.append(text)

    strings = TEXT[args.language]
    fonts = load_fonts(args.language)
    quiet = lambda text="": None

    import tools_track_console_loop as TR
    rows = TR.teacher_rows(args.labels)
    by_family = {}
    for row in rows.values():
        by_family.setdefault(row.get("family_id"), []).append(row)
    candidates = []
    for name, group in sorted(by_family.items()):
        if not name or any(r.get("split") != "test" for r in group):
            continue
        if len(set(r["focus_mm"] for r in group if r.get("focus_determined"))) >= 5:
            candidates.append((name, len(group)))

    emit("=========== demo video ===========")
    emit("  language   %s, %d fps, %dx%d" % (args.language, args.fps, WIDTH, HEIGHT))
    model, payload, front_payload, model_text = SG.load_model(args)
    trained = SG.trained_axes(payload)
    emit("  model      %s" % model_text)

    stories = []
    for name, _ in candidates[:max(args.candidates, args.subjects + 1)]:
        group = [r for r in by_family.get(name, []) if r.get("focus_determined")]
        try:
            story = DP.scene_story(name, group, model, payload, trained, args, quiet, with_images=False)
        except Exception as error:
            emit("    scene %s skipped (%s)" % (name, error))
            continue
        if story:
            stories.append(story)
    if not stories:
        raise SystemExit("no test-set scene could be prepared")
    starts = sorted(s["start_gain"] for s in stories)
    median_start = starts[len(starts) // 2]
    main_story = (next(s for s in stories if s["family"] == args.family) if args.family
                  else min(stories, key=lambda s: abs(s["start_gain"] - median_start)))
    emit("  scene      %s (start %.2f dB, median of %d candidates %.2f dB)"
         % (main_story["family"], main_story["start_gain"], len(stories), median_start))

    # 第 1、2 段要图像，重跑一次拿到完整的 steps
    group = [r for r in by_family.get(main_story["family"], []) if r.get("focus_determined")]
    main_story = DP.scene_story(main_story["family"], group, model, payload, trained, args, quiet,
                               with_images=False)
    frame = main_story["frames"][main_story["anchor"]]
    emit("  anchor     %s, loop %d step(s), gain %.2f -> %.2f dB, TGC %.2f -> %.2f dB"
         % (frame["frame_id"], len(main_story["steps"]) - 1, main_story["start_gain"],
            main_story["final_gain"], main_story["start_tgc"], main_story["final_tgc"]))

    main_subject = main_story["family"].rsplit("/", 1)[0]
    pool, seen = [], {main_subject}
    for story in sorted(stories, key=lambda x: x["family"]):
        subject = story["family"].rsplit("/", 1)[0]
        if subject in seen:
            continue
        seen.add(subject)
        pool.append(story)
    # 宫格里的六位按收敛结果均匀取秩（最好、最差与中间各档），而不是取名字靠前的六位——
    # 任意一段都会偶然落进好的一批或差的一批，均匀取秩才是这批受试者的真实跨度
    pool.sort(key=lambda x: x["final_gain"])
    if len(pool) > args.subjects and args.subjects > 1:
        ranks = [int(round(i * (len(pool) - 1) / float(args.subjects - 1))) for i in range(args.subjects)]
        others = [pool[r] for r in sorted(set(ranks))]
    else:
        others = pool[:args.subjects]
    emit("  part 3     %d further subjects (one scene each): %s"
         % (len(others), ", ".join(x["family"] for x in others)))

    single = DP.read_metrics(args.single_step_json)
    net = single.get("network_recorded_start", {})
    base = single.get("settings_lookup_baseline", {})
    bias = DP.subject_bias(args.loop_trajectories, rows)
    if bias:
        emit("  bias       %d subjects, within-subject sd %.2f dB, %d within +-1 dB"
             % (len(bias["subjects"]), bias["sd_median"], bias["within_1db"]))

    temp = tempfile.mkdtemp(prefix="demo_video_")
    count = [0]

    def write(image):
        image.save(os.path.join(temp, "%06d.png" % count[0]))
        count[0] += 1

    try:
        for image in scene_title(fonts, strings, args.fps):
            write(image)
        scene_backend(frame, main_story["row"], main_story["steps"], fonts, strings, args.fps, write)
        for image in scene_focus(main_story["frames"], main_story["results"], main_story["order"],
                                 main_story["rows"], fonts, strings, args.fps, args):
            write(image)
        if others:
            for image in scene_subjects(others, fonts, strings, args.fps):
                write(image)
        if bias:
            for image in scene_bias(bias, fonts, strings, args.fps):
                write(image)
        for image in scene_numbers(net, base, fonts, strings, args.fps):
            write(image)
        emit("  frames     %d (%.1f s)" % (count[0], count[0] / float(args.fps)))
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory and not os.path.isdir(directory):
            os.makedirs(directory)
        write_video(temp, count[0], args.out, args.fps, emit)
        emit("  wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))
        if args.gif:
            step = max(1, args.fps // 10)
            pictures = [Image.open(os.path.join(temp, "%06d.png" % i)).convert("P", palette=Image.ADAPTIVE,
                                                                               colors=128)
                        for i in range(0, count[0], step)]
            pictures[0].save(args.gif, save_all=True, append_images=pictures[1:],
                             duration=int(1000 * step / float(args.fps)), loop=0, optimize=True)
            emit("  wrote %s (%.1f MB)" % (args.gif, os.path.getsize(args.gif) / 1e6))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    emit("  done in %.0f s" % (time.time() - started))

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return args.out


if __name__ == "__main__":
    main()
