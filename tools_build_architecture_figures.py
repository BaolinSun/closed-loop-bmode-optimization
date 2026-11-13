# -*- coding: utf-8 -*-
"""投稿用的网络架构图：英文、白底、矢量输出，三张图各自成文件。

    与 tools_build_architecture_deck.py 的区别

那一份是汇报用的幻灯片：深色、中文、13.3 x 7.5 英寸。期刊图不一样——

  画布按成图尺寸做  双栏宽 7.2 英寸，字号就是印出来的字号（5-7 pt），不靠缩放。
                    幻灯片缩到 7 英寸宽时 9 pt 会变成 4.7 pt，小于期刊下限。
  白底             正文是白的，深色图在印刷上既不常见也费墨。
  矢量优先          PDF 给 LaTeX，EMF 给 Word，另出 600 dpi PNG 备用。
  少字多图          能画出来的就不写成句子：下采样画成金字塔，CORAL 画成累积概率
                    阶梯加差分条，TGC 的拼接画成按长度成比例的条，参数预算画成条形图。
                    成段的说明属于 figure caption，不属于图里——见 captions.txt。

    三张图

fig1  架构总览：(a) 主干通路，(b) 八个输出头。
fig2  四处设计选择：(a) FiLM 条件调制，(b) 后端输出最优值而非修正量（含反馈斜率示意），
      (c) 聚焦头的 CORAL 有序回归，(d) TGC 分段头的输入拼接。
fig3  参数预算：占比堆叠条 + 按模块的条形图。

数字与形状来自实例化后的 SixParamNet，不是手抄。

用法：python tools_build_architecture_figures.py
终端输出与写在脚本旁边的 tools_build_architecture_figures.txt 同内容（ASCII）。
"""

import argparse
import io
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

REPORT_PATH = "tools_build_architecture_figures.txt"

FIG_W = 7.2                       # 双栏宽，英寸
MARGIN = 0.10
FONT = "Arial"

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
BODY = RGBColor(0x3F, 0x3F, 0x46)
MUTED = RGBColor(0x6B, 0x72, 0x80)
RULE = RGBColor(0xD4, 0xD4, 0xD8)
PANEL = RGBColor(0xFA, 0xFA, 0xFB)

GREY = RGBColor(0x4A, 0x55, 0x68)
BLUE = RGBColor(0x2B, 0x6C, 0xB0)
ORANGE = RGBColor(0xC0, 0x56, 0x21)
GREEN = RGBColor(0x2F, 0x7A, 0x5A)
VIOLET = RGBColor(0x6B, 0x4F, 0xA8)

LADDERS = {"depth_mm": [25.1, 33.5, 41.9, 50.2, 58.6, 67.0, 75.4, 83.7, 92.1],
           "frequency_mhz": [4.4, 4.7, 5.0, 5.3, 5.7],
           "focus_mm": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]}

LINES = []


def emit(text_line=""):
    LINES.append(text_line)
    print(text_line)


# ---------------------------------------------------------------------------------------
#   画图零件

def tint(colour, t):
    """往白色方向淡化：t=1 为原色，t=0 为白。"""
    return RGBColor(*[int(round(255 + (colour[i] - 255) * t)) for i in range(3)])


def new_figure(width, height):
    deck = Presentation()
    deck.slide_width = Inches(width)
    deck.slide_height = Inches(height)
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = WHITE
    return deck, slide


def set_font(run, size, colour, bold=False, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = colour
    run.font.name = name


def text(slide, x, y, w, h, lines, size=6.5, colour=BODY, bold=False,
         align=PP_ALIGN.LEFT, name=FONT, spacing=1.2, space_after=0):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.TOP
    if isinstance(lines, str):
        lines = [lines]
    for i, item in enumerate(lines):
        value, item_size, item_colour, item_bold = (
            item if isinstance(item, tuple) else (item, size, colour, bold))
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.alignment = align
        paragraph.line_spacing = spacing
        paragraph.space_after = Pt(space_after)
        run = paragraph.add_run()
        run.text = value
        set_font(run, item_size, item_colour, item_bold, name)
    return shape


def mono(slide, x, y, w, value, size=5.5, colour=MUTED, align=PP_ALIGN.CENTER):
    return text(slide, x, y, w, 0.12, value, size, colour, False, align, "Consolas")


def panel_letter(slide, x, y, value):
    text(slide, x, y, 0.20, 0.14, value, 8, INK, True)


def box(slide, x, y, w, h, fill=None, outline=RULE, width=0.6, radius=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    if radius is not None:
        shape.adjustments[0] = radius
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if outline is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = outline
        shape.line.width = Pt(width)
    shape.shadow.inherit = False
    shape.text_frame.text = ""
    return shape


def node(slide, x, y, w, h, caption, accent=BLUE, size=6.5, bold=False, fill_t=0.12):
    """一个带框的小节点，字垂直居中。"""
    shape = box(slide, x, y, w, h, tint(accent, fill_t), accent, 0.6, 0.12)
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = Inches(0.03)
    frame.margin_top = frame.margin_bottom = Inches(0.01)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    paragraph.line_spacing = 1.1
    run = paragraph.add_run()
    run.text = caption
    set_font(run, size, INK, bold)
    return shape


def _plain_shape(slide, shape_type, x, y, w, h, colour):
    shape = slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False
    shape.text_frame.text = ""
    return shape


def arrow(slide, x, y, w=0.14, h=0.07, colour=MUTED):
    return _plain_shape(slide, MSO_SHAPE.RIGHT_ARROW, x, y, w, h, colour)


def up_arrow(slide, x, y, w=0.07, h=0.10, colour=MUTED):
    return _plain_shape(slide, MSO_SHAPE.UP_ARROW, x, y, w, h, colour)


def down_arrow(slide, x, y, w=0.07, h=0.10, colour=MUTED):
    return _plain_shape(slide, MSO_SHAPE.DOWN_ARROW, x, y, w, h, colour)


def line(slide, x1, y1, x2, y2, colour=RULE, width=0.6):
    shape = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1),
                                       Inches(x2), Inches(y2))
    shape.line.color.rgb = colour
    shape.line.width = Pt(width)
    shape.shadow.inherit = False
    return shape


def bar(slide, x, y, w, h, colour, outline=None):
    return box(slide, x, y, w, h, colour, outline, 0.5, None if h < 0.08 else 0.18)


# ---------------------------------------------------------------------------------------
#   模型事实

def probe_model():
    try:
        import torch
        from bmode_dl.model import SixParamNet, count_parameters
    except Exception as error:                                        # noqa: BLE001
        emit("  could not import the model (%s); using the recorded figures" % error)
        return None
    model = SixParamNet(LADDERS, focus_head="ordinal", far_wall=True)
    groups = {}
    for name, parameter in model.named_parameters():
        groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0) + parameter.numel()
    plain = SixParamNet(LADDERS, focus_head="ordinal", far_wall=False)
    return {"groups": groups, "total": count_parameters(model),
            "total_plain": count_parameters(plain)}


FALLBACK = {"groups": {"image_encoder": 2951456, "sequence_model": 1054208,
                       "profile_encoder": 278336, "trunk": 197120, "seq_proj": 98560,
                       "param_encoder": 70400, "tgc_band_head": 66945, "depth_head": 34444,
                       "slider_dir_head": 34057, "frequency_head": 33928, "focus_head": 33540,
                       "gain_head": 33412, "dr_head": 33412, "aux_head": 33154,
                       "far_wall_head": 33025, "pos_embed": 8448, "seq_norm": 512,
                       "focus_cut": 6},
            "total": 4994963, "total_plain": 4961810}

MODULE_EN = {"image_encoder": "2-D image encoder", "sequence_model": "Transformer encoder",
             "profile_encoder": "1-D profile encoder", "trunk": "Shared trunk",
             "seq_proj": "Sequence projection", "param_encoder": "Parameter embedding",
             "tgc_band_head": "TGC band head", "depth_head": "Imaging-depth head",
             "slider_dir_head": "Slider-direction head", "frequency_head": "Frequency head",
             "focus_head": "Focus head (CORAL)", "gain_head": "Gain head",
             "dr_head": "Dynamic-range head", "aux_head": "Auxiliary head",
             "far_wall_head": "Far-wall head", "pos_embed": "Positional embedding",
             "seq_norm": "Sequence LayerNorm", "focus_cut": "CORAL cut-points"}

# 四个组：按架构图里的配色归并，堆叠条用
GROUPING = [("2-D image encoder", BLUE, ["image_encoder"]),
            ("Transformer", VIOLET, ["sequence_model", "seq_proj", "pos_embed", "seq_norm"]),
            ("1-D profile encoder", GREEN, ["profile_encoder"]),
            ("Embedding + trunk", GREY, ["param_encoder", "trunk"]),
            ("Output heads", ORANGE, ["tgc_band_head", "depth_head", "slider_dir_head",
                                      "frequency_head", "focus_head", "gain_head",
                                      "dr_head", "aux_head", "far_wall_head", "focus_cut"])]


# ---------------------------------------------------------------------------------------
#   图 1：架构总览

def figure_architecture(facts):
    height = 3.98
    deck, slide = new_figure(FIG_W, height)
    inner = FIG_W - 2 * MARGIN

    # ---------- (a) 主干通路
    panel_letter(slide, MARGIN, 0.08, "a")
    top, tall = 0.26, 1.90
    widths = [1.26, 1.90, 1.20, 1.12, 1.00]
    gap = (inner - sum(widths)) / (len(widths) - 1)
    xs, run_x = [], MARGIN
    for w in widths:
        xs.append(run_x)
        run_x += w + gap
    for x, w in zip(xs, widths):
        box(slide, x, top, w, tall, PANEL, RULE, 0.6, 0.04)
    for i in range(4):
        arrow(slide, xs[i] + widths[i] + gap / 2 - 0.07, top + tall / 2 - 0.035)

    # 输入三路
    x, w = xs[0], widths[0]
    text(slide, x + 0.07, top + 0.06, w - 0.14, 0.12, "Inputs", 7, INK, True)
    inputs = [("B-mode image", "3 x 512 x 128",
               ["rendered gray / 255", "absolute dB (standardized)", "depth / 60 mm"], GREY),
              ("Depth profile", "5 x 512",
               ["dB percentiles 10/50/90", "gray median, median - noise"], GREY),
              ("Current settings", "17 scalars",
               ["6 parameters, 8 TGC sliders,", "reference, mode, noise floor"], GREY)]
    iy = top + 0.23
    for name, shape_text, detail, accent in inputs:
        h = 0.25 + len(detail) * 0.095
        box(slide, x + 0.07, iy, w - 0.14, h, tint(accent, 0.07), accent, 0.6, 0.10)
        text(slide, x + 0.12, iy + 0.035, w - 0.24, 0.10, name, 6.5, INK, True)
        mono(slide, x + 0.12, iy + 0.135, w - 0.24, shape_text, 5.5, accent, PP_ALIGN.LEFT)
        for j, item in enumerate(detail):
            text(slide, x + 0.12, iy + 0.235 + j * 0.095, w - 0.24, 0.09, item, 5.5, MUTED)
        iy += h + 0.075

    # 编码器
    x, w = xs[1], widths[1]
    text(slide, x + 0.07, top + 0.06, w - 0.14, 0.12,
         "Residual encoders, FiLM-conditioned", 7, INK, True)
    text(slide, x + 0.07, top + 0.22, w - 0.14, 0.10,
         "2-D: stem 7x3 + 4 x ResBlock2d", 5.5, MUTED)
    pyramid = [(0.30, 0.50), (0.28, 0.42), (0.26, 0.35), (0.24, 0.29), (0.22, 0.24)]
    base = top + 0.82
    px = x + 0.12
    for i, (bw, bh) in enumerate(pyramid):
        bar(slide, px, base - bh, bw, bh, tint(BLUE, 0.85 - 0.10 * i), BLUE)
        px += 0.33
    mono(slide, x + 0.07, base + 0.03, w - 0.14,
         "3 x 512 x 128  ->  256 x 32 x 16", 5.5, BLUE)
    text(slide, x + 0.07, base + 0.20, w - 0.14, 0.10,
         "1-D: pool to 128 + 2 x ResBlock1d", 5.5, MUTED)
    px = x + 0.12
    for i in range(3):
        bar(slide, px, base + 0.34, 0.30, 0.13, tint(GREEN, 0.80 - 0.12 * i), GREEN)
        px += 0.33
    mono(slide, x + 0.07, base + 0.50, w - 0.14, "5 x 512  ->  128 x 32", 5.5, GREEN)

    film_y = top + tall - 0.28
    for i in range(5):
        up_arrow(slide, x + 0.12 + i * 0.33 + 0.11, film_y - 0.12, 0.07, 0.10,
                 tint(ORANGE, 0.8))
    node(slide, x + 0.07, film_y, w - 0.14, 0.22,
         "FiLM  x <- x (1 + gamma) + beta", ORANGE, 6, True)

    # 融合
    x, w = xs[2], widths[2]
    text(slide, x + 0.07, top + 0.06, w - 0.14, 0.12, "Depth-token fusion", 7, INK, True)
    node(slide, x + 0.07, top + 0.26, w - 0.14, 0.22, "lateral average  256 x 32", BLUE, 6)
    node(slide, x + 0.07, top + 0.54, w - 0.14, 0.22, "profile features  128 x 32", GREEN, 6)
    node(slide, x + 0.07, top + 0.82, w - 0.14, 0.24, "concat 384 -> linear 256", VIOLET, 6)
    tok_y = top + 1.18
    cw, cg = 0.095, 0.022
    tx = x + 0.07
    bar(slide, tx, tok_y, cw, 0.15, tint(ORANGE, 0.85), ORANGE)
    for i in range(8):
        bar(slide, tx + (i + 1) * (cw + cg), tok_y, cw, 0.15, tint(VIOLET, 0.75), VIOLET)
    text(slide, tx + 9 * (cw + cg), tok_y + 0.01, 0.12, 0.12, "...", 7, MUTED)
    text(slide, x + 0.07, tok_y + 0.21, w - 0.14, 0.26,
         "1 parameter token (orange) + 32 depth tokens + positional embedding",
         5.5, MUTED, False, PP_ALIGN.LEFT, FONT, 1.25)

    # Transformer
    x, w = xs[3], widths[3]
    text(slide, x + 0.07, top + 0.06, w - 0.14, 0.12, "Transformer encoder", 7, INK, True)
    node(slide, x + 0.07, top + 0.26, w - 0.14, 0.28,
         "LayerNorm -> MHSA, 8 heads -> add", VIOLET, 6)
    node(slide, x + 0.07, top + 0.60, w - 0.14, 0.28,
         "LayerNorm -> FFN 256-512-256 -> add", VIOLET, 6)
    box(slide, x + 0.04, top + 0.22, w - 0.08, 0.72, None, VIOLET, 0.6, 0.06)
    text(slide, x + 0.07, top + 0.97, w - 0.14, 0.12, "x 2 layers, pre-norm",
         6, VIOLET, True, PP_ALIGN.CENTER)
    node(slide, x + 0.07, top + 1.15, w - 0.14, 0.20, "LayerNorm", VIOLET, 6)
    text(slide, x + 0.07, top + 1.42, w - 0.14, 0.34,
         "attention runs along depth: the parameter token and all 32 depth tokens "
         "are mutually visible",
         5.5, MUTED, False, PP_ALIGN.LEFT, FONT, 1.25)

    # 主干
    x, w = xs[4], widths[4]
    text(slide, x + 0.07, top + 0.06, w - 0.14, 0.12, "Trunk", 7, INK, True)
    node(slide, x + 0.07, top + 0.26, w - 0.14, 0.34,
         "parameter token + mean of depth tokens", GREY, 6)
    node(slide, x + 0.07, top + 0.66, w - 0.14, 0.24, "concat 512 -> MLP 256", GREY, 6)
    node(slide, x + 0.07, top + 0.98, w - 0.14, 0.34,
         "depth tokens -> 8 bands (TGC head only)", GREEN, 6)
    text(slide, x + 0.07, top + 1.40, w - 0.14, 0.44,
         "each band sees its own depth, not the whole-image average",
         5.5, MUTED, False, PP_ALIGN.LEFT, FONT, 1.25)

    # ---------- (b) 输出头
    panel_letter(slide, MARGIN, 2.26, "b")
    text(slide, MARGIN + 0.18, 2.27, 2.0, 0.12, "Output heads", 7, INK, True)
    legend_x = MARGIN + 1.00
    for name, accent in (("back end", BLUE), ("front end", ORANGE), ("auxiliary", GREEN)):
        bar(slide, legend_x, 2.305, 0.10, 0.065, tint(accent, 0.55), accent)
        text(slide, legend_x + 0.13, 2.285, 0.62, 0.11, name, 6, MUTED)
        legend_x += 0.80

    heads = [("Gain", "MLP 256-128-(1+3)",
              ["optimal gain in dB", "delta computed outside"], BLUE),
             ("TGC, 8 bands", "MLP 521-128-1 per band",
              ["trunk + band feature", "+ slider + band index"], BLUE),
             ("Slider direction", "MLP 256-128-3x3",
              ["near / mid / far groups", "low / correct / high"], BLUE),
             ("Dynamic range, aux", "MLP 256-128-(1+3), -2",
              ["unsupervised here", "aux: attenuation, noise"], MUTED),
             ("Imaging depth", "MLP 256-128-(9+3)",
              ["9 rungs + direction"], ORANGE),
             ("Transmit frequency", "MLP 256-128-(5+3)",
              ["5 rungs + direction"], ORANGE),
             ("Focus depth (CORAL)", "MLP 257-128-(1+3)",
              ["1 score + 6 cut-points", "far-wall estimate appended"], ORANGE),
             ("Far-wall depth", "MLP 256-128-1",
              ["auxiliary (Huber)", "detached into focus head"], GREEN)]
    card_w = (inner - 3 * 0.09) / 4
    card_h = 0.52
    for i, (name, shape_text, detail, accent) in enumerate(heads):
        cx = MARGIN + (i % 4) * (card_w + 0.09)
        cy = 2.46 + (i // 4) * (card_h + 0.09)
        box(slide, cx, cy, card_w, card_h, tint(accent, 0.06), accent, 0.6, 0.10)
        text(slide, cx + 0.06, cy + 0.045, card_w - 0.12, 0.11, name, 6.5, INK, True)
        mono(slide, cx + 0.06, cy + 0.165, card_w - 0.12, shape_text, 5.5, accent,
             PP_ALIGN.LEFT)
        for j, item in enumerate(detail):
            text(slide, cx + 0.06, cy + 0.285 + j * 0.095, card_w - 0.12, 0.09,
                 item, 5.5, MUTED)

    text(slide, MARGIN, 3.72, inner, 0.14,
         "A single forward pass yields a recommended value and a direction class for every "
         "parameter; %.2f M parameters (%.2f M without the far-wall head)."
         % (facts["total"] / 1e6, facts["total_plain"] / 1e6),
         5.5, MUTED)
    return deck


# ---------------------------------------------------------------------------------------
#   图 2：四处设计选择

def figure_choices(facts):
    height = 3.05
    deck, slide = new_figure(FIG_W, height)
    inner = FIG_W - 2 * MARGIN
    col_w = (inner - 0.14) / 2
    col_x = (MARGIN, MARGIN + col_w + 0.14)
    row_y = (0.24, 1.72)
    row_h = (1.32, 1.18)

    # ---------- (a) FiLM 条件调制
    x, w, y, h = col_x[0], col_w, row_y[0], row_h[0]
    panel_letter(slide, x, y - 0.17, "a")
    text(slide, x + 0.18, y - 0.16, w - 0.18, 0.12,
         "Conditioning every encoder stage on the current settings", 7, INK, True)
    box(slide, x, y, w, h, PANEL, RULE, 0.6, 0.04)
    node(slide, x + 0.08, y + 0.09, 0.62, 0.22, "17 scalars", GREY, 6)
    arrow(slide, x + 0.74, y + 0.165, 0.12, 0.06)
    node(slide, x + 0.90, y + 0.09, 0.86, 0.22, "MLP 17-256-256", ORANGE, 6)
    arrow(slide, x + 1.80, y + 0.165, 0.12, 0.06)
    node(slide, x + 1.96, y + 0.09, 0.70, 0.22, "gamma, beta", ORANGE, 6)
    text(slide, x + 0.08, y + 0.40, w - 0.16, 0.18,
         "x  <-  x (1 + gamma) + beta", 9, INK, True, PP_ALIGN.CENTER, "Consolas")

    # 被调制的六级要画出来，上面的箭头才有落点
    stage_w, stage_gap = 0.40, 0.06
    sx = x + (w - (6 * stage_w + 5 * stage_gap)) / 2
    for i in range(6):
        accent = BLUE if i < 4 else GREEN
        caption = ("2-D %d" % (i + 1)) if i < 4 else ("1-D %d" % (i - 3))
        down_arrow(slide, sx + i * (stage_w + stage_gap) + stage_w / 2 - 0.035,
                   y + 0.64, 0.07, 0.09, tint(ORANGE, 0.8))
        node(slide, sx + i * (stage_w + stage_gap), y + 0.75, stage_w, 0.20,
             caption, accent, 5.5)
    text(slide, x + 0.08, y + 1.03, w - 0.16, 0.12,
         "the projection is zero-initialized, so the layer is the identity at step 0",
         5.5, MUTED, False, PP_ALIGN.CENTER)

    # ---------- (b) 预测最优值而非修正量
    x, w, y, h = col_x[1], col_w, row_y[0], row_h[0]
    panel_letter(slide, x, y - 0.17, "b")
    text(slide, x + 0.18, y - 0.16, w - 0.18, 0.12,
         "Predicting the optimum, not the correction", 7, INK, True)
    box(slide, x, y, w, h, PANEL, RULE, 0.6, 0.04)
    node(slide, x + 0.08, y + 0.09, 1.06, 0.26, "optimum (network)", BLUE, 6)
    text(slide, x + 1.16, y + 0.16, 0.10, 0.14, "-", 9, INK, True, PP_ALIGN.CENTER)
    node(slide, x + 1.28, y + 0.09, 0.94, 0.26, "current (console)", GREY, 6)
    text(slide, x + 2.24, y + 0.16, 0.10, 0.14, "=", 9, INK, True, PP_ALIGN.CENTER)
    node(slide, x + 2.36, y + 0.09, 0.80, 0.26, "correction", BLUE, 6, True)
    text(slide, x + 0.08, y + 0.39, w - 0.16, 0.12,
         "the subtraction is exact and happens outside the network",
         5.5, MUTED, False, PP_ALIGN.CENTER)

    gx, gy, gw, gh = x + 0.14, y + 0.62, 1.10, 0.44
    box(slide, gx, gy, gw, gh, WHITE, RULE, 0.6)
    line(slide, gx, gy + gh * 0.55, gx + gw, gy + gh * 0.55, tint(BLUE, 0.95), 1.2)
    line(slide, gx, gy + gh * 0.94, gx + gw, gy + gh * 0.06, tint(ORANGE, 0.95), 1.2)
    text(slide, gx - 0.06, gy - 0.12, 1.10, 0.10, "predicted optimum", 5, MUTED)
    text(slide, gx, gy + gh + 0.03, gw, 0.10, "current gain", 5, MUTED, False,
         PP_ALIGN.CENTER)
    ly = gy + 0.01
    for colour, caption in ((ORANGE, "slope 1.45: loop diverges"),
                            (BLUE, "slope 0.01-0.03: loop stable")):
        line(slide, gx + gw + 0.08, ly + 0.045, gx + gw + 0.20, ly + 0.045, colour, 1.2)
        text(slide, gx + gw + 0.24, ly, 1.34, 0.10, caption, 5.5, colour, True)
        ly += 0.15
    text(slide, gx + gw + 0.24, ly + 0.02, 1.34, 0.24,
         "no frame above 1; 99.4 % of closed-loop runs converge",
         5, MUTED, False, PP_ALIGN.LEFT, FONT, 1.25)

    # ---------- (c) 聚焦头的有序回归
    x, w, y, h = col_x[0], col_w, row_y[1], row_h[1]
    panel_letter(slide, x, y - 0.17, "c")
    text(slide, x + 0.18, y - 0.16, w - 0.18, 0.12,
         "Ordinal focus head (CORAL)", 7, INK, True)
    box(slide, x, y, w, h, PANEL, RULE, 0.6, 0.04)
    text(slide, x + 0.08, y + 0.06, w - 0.16, 0.14,
         "P(rung > k) = sigmoid(s + b_k),  k = 0 ... 5", 7.5, INK, True,
         PP_ALIGN.CENTER, "Consolas")

    sx, sy, sw, sh = x + 0.20, y + 0.32, 1.14, 0.42
    cumulative = [0.97, 0.88, 0.70, 0.41, 0.18, 0.06]
    step = sw / 7.0
    line(slide, sx, sy + sh, sx + sw, sy + sh, RULE, 0.6)
    line(slide, sx, sy, sx, sy + sh, RULE, 0.6)
    previous = 1.0
    for i, value in enumerate(cumulative + [0.0]):
        bx = sx + i * step
        line(slide, bx, sy + sh * (1 - previous), bx + step, sy + sh * (1 - previous),
             BLUE, 1.2)
        line(slide, bx + step, sy + sh * (1 - previous), bx + step,
             sy + sh * (1 - value), BLUE, 1.2)
        previous = value
    text(slide, sx - 0.14, sy - 0.03, 0.12, 0.10, "1", 5, MUTED, False, PP_ALIGN.RIGHT)
    text(slide, sx - 0.14, sy + sh - 0.07, 0.12, 0.10, "0", 5, MUTED, False, PP_ALIGN.RIGHT)
    text(slide, sx, sy + sh + 0.03, sw, 0.10,
         "monotone by construction", 5, BLUE, False, PP_ALIGN.CENTER)
    text(slide, sx, sy - 0.12, sw, 0.10, "cumulative", 5, BLUE, False, PP_ALIGN.CENTER)

    dx, dy, dw, dh = x + 1.52, y + 0.32, 1.14, 0.42
    rungs = [0.03, 0.09, 0.18, 0.29, 0.23, 0.12, 0.06]
    bw = dw / 7.0
    line(slide, dx, dy + dh, dx + dw, dy + dh, RULE, 0.6)
    for i, value in enumerate(rungs):
        bh = dh * value / 0.32
        bar(slide, dx + i * bw + 0.012, dy + dh - bh, bw - 0.024, bh,
            tint(ORANGE, 0.45 + 0.08 * i), ORANGE)
    text(slide, dx, dy + dh + 0.03, dw, 0.10,
         "10   15   20   25   30   35   40  mm", 5, MUTED, False, PP_ALIGN.CENTER)
    text(slide, dx, dy - 0.12, dw, 0.10, "per-rung probability", 5, ORANGE,
         False, PP_ALIGN.CENTER)
    arrow(slide, x + 1.38, dy + dh / 2 - 0.03, 0.11, 0.06)
    text(slide, x + 0.08, y + 0.94, w - 0.16, 0.12,
         "one score and six cut-points replace seven independent classes",
         5.5, MUTED, False, PP_ALIGN.CENTER)

    # ---------- (d) TGC 分段头
    x, w, y, h = col_x[1], col_w, row_y[1], row_h[1]
    panel_letter(slide, x, y - 0.17, "d")
    text(slide, x + 0.18, y - 0.16, w - 0.18, 0.12,
         "TGC band head: one head applied to each of eight bands", 7, INK, True)
    box(slide, x, y, w, h, PANEL, RULE, 0.6, 0.04)

    parts = [("trunk h", 256, BLUE), ("band feature", 256, VIOLET),
             ("slider", 1, ORANGE), ("band #", 8, GREEN)]
    # 小的两段按比例算出来窄到看不见，给它们一个下限，剩下的宽度再在两个大段之间按比例分
    cat_w = w - 0.16
    floor = 0.34
    large = float(sum(p[1] for p in parts if p[1] >= 32))
    spare = cat_w - floor * sum(1 for p in parts if p[1] < 32)
    widths = dict((p[0], floor if p[1] < 32 else spare * p[1] / large) for p in parts)
    cx = x + 0.08
    for name, size, accent in parts:
        pw = widths[name]
        bar(slide, cx, y + 0.08, pw, 0.20, tint(accent, 0.45), accent)
        text(slide, cx, y + 0.305, pw, 0.10, name, 5, INK, False, PP_ALIGN.CENTER)
        mono(slide, cx, y + 0.405, pw, str(size), 5, MUTED)
        cx += pw
    text(slide, x + 0.08, y + 0.54, cat_w, 0.12, "concat 521  ->  MLP 521-128-1",
         6.5, INK, True, PP_ALIGN.CENTER, "Consolas")

    strip_x, strip_y, strip_w = x + 0.08, y + 0.84, cat_w
    seg = strip_w / 8.0
    text(slide, strip_x, strip_y - 0.12, strip_w, 0.10,
         "depth tokens pooled into 8 bands, near to far", 5, MUTED, False, PP_ALIGN.CENTER)
    for i in range(8):
        bar(slide, strip_x + i * seg, strip_y, seg - 0.012, 0.14,
            tint(BLUE, 0.25 + 0.08 * i), BLUE)
        text(slide, strip_x + i * seg, strip_y + 0.155, seg - 0.012, 0.10,
             str(i + 1), 5, MUTED, False, PP_ALIGN.CENTER)
    return deck


# ---------------------------------------------------------------------------------------
#   图 3：参数预算

def figure_budget(facts):
    height = 2.70
    deck, slide = new_figure(FIG_W, height)
    inner = FIG_W - 2 * MARGIN
    groups = facts["groups"]
    total = float(sum(groups.values()))

    # ---------- (a) 占比堆叠条
    panel_letter(slide, MARGIN, 0.08, "a")
    text(slide, MARGIN + 0.18, 0.09, 3.0, 0.12, "Share of parameters", 7, INK, True)
    sx, sy, sw, sh = MARGIN, 0.30, inner, 0.26
    run_x = sx
    for name, accent, keys in GROUPING:
        count = sum(groups.get(k, 0) for k in keys)
        pw = sw * count / total
        bar(slide, run_x, sy, pw, sh, tint(accent, 0.75), accent)
        if pw > 0.42:
            text(slide, run_x, sy + 0.075, pw, 0.12, "%.0f%%" % (100 * count / total),
                 6.5, WHITE, True, PP_ALIGN.CENTER)
        run_x += pw
    legend_x = MARGIN
    for name, accent, keys in GROUPING:
        count = sum(groups.get(k, 0) for k in keys)
        bar(slide, legend_x, 0.645, 0.09, 0.065, tint(accent, 0.75), accent)
        text(slide, legend_x + 0.12, 0.625, 1.30, 0.11,
             "%s  %.0f%%" % (name, 100 * count / total), 5.5, BODY)
        legend_x += 1.40

    # ---------- (b) 按模块的条形图
    panel_letter(slide, MARGIN, 0.84, "b")
    text(slide, MARGIN + 0.18, 0.85, 3.0, 0.12, "Parameters by module", 7, INK, True)
    order = sorted(groups.items(), key=lambda item: -item[1])
    shown = [item for item in order if item[1] >= 30000]
    biggest = float(shown[0][1])
    label_w = 1.30
    span = inner / 2.0 - label_w * 0.86 - 0.72
    half = (len(shown) + 1) // 2
    for column in range(2):
        cx = MARGIN + column * (inner / 2.0 + 0.06)
        for row, (name, count) in enumerate(shown[column * half:(column + 1) * half]):
            ry = 1.06 + row * 0.175
            text(slide, cx, ry, label_w * 0.86, 0.11, MODULE_EN.get(name, name), 5.5, BODY)
            accent = BLUE
            for _, group_accent, keys in GROUPING:
                if name in keys:
                    accent = group_accent
            bw = max(span * count / biggest, 0.008)
            bar(slide, cx + label_w * 0.86 + 0.04, ry + 0.012, bw, 0.075,
                tint(accent, 0.8), None)
            text(slide, cx + label_w * 0.86 + 0.08 + bw, ry, 0.44, 0.11,
                 "%.3f M" % (count / 1e6), 5.5, MUTED, False, PP_ALIGN.LEFT, "Consolas")

    text(slide, MARGIN, 2.42, inner, 0.14,
         "Modules below 0.03 M (positional embedding, LayerNorm, CORAL cut-points) are "
         "omitted. Total %.2f M; %.2f M with the far-wall head disabled."
         % (facts["total"] / 1e6, facts["total_plain"] / 1e6),
         5.5, MUTED)
    return deck


# ---------------------------------------------------------------------------------------
#   图注：成段的说明写在这里，不写进图里

CAPTIONS = u"""Figure 1. Architecture of SixParamNet.
(a) A single forward pass maps a B-mode frame, its depth profile and the 17 scalars that
describe the settings currently in force onto a shared 256-dimensional representation.
A 2-D residual encoder reduces the frame from 3 x 512 x 128 to 256 x 32 x 16 and a 1-D
encoder reduces the profile to 128 x 32; every stage of both encoders is modulated by FiLM
parameters derived from the current settings. Lateral averaging and concatenation give 32
depth tokens, a parameter token is prepended, and a two-layer pre-norm Transformer encoder
(d = 256, 8 heads) lets the parameter token and the depth tokens attend to one another. The
parameter token and the mean of the depth tokens form the input to the shared trunk; the
depth tokens are separately pooled into 8 bands for the TGC head. (b) Eight heads read the
trunk. Back-end heads (blue) emit the optimal gain and the optimal TGC curve; front-end
heads (orange) emit a distribution over the discrete imaging-depth, transmit-frequency and
focus-depth rungs; auxiliary heads (green) are supervised during training but not used at
inference. Every rung head additionally emits a three-way direction class (too low /
correct / too high).

Figure 2. Four design choices and what motivates them.
(a) FiLM conditioning. The same raw dB frame is consistent with any current gain and TGC,
so image brightness is only interpretable relative to the settings in force; the parameter
embedding therefore modulates all four 2-D and both 1-D encoder stages rather than being
concatenated once at the end. The modulating projection is zero-initialized, so the layer
is the identity at initialization. (b) Back-end parameterization. The network predicts the
optimal gain and TGC, and the correction is obtained by subtracting the current setting
outside the network. Predicting the correction directly left a residual of about 0.3 dB
and, when the current gain was standardized into a strong input, degenerated into
optimum = current + correction: the feedback slope d(predicted optimum)/d(current gain)
rose to a median of 1.45 and the closed loop diverged. With the present parameterization
the slope is 0.01-0.03, no frame exceeds 1, and 99.4 % of closed-loop runs converge.
(c) Ordinal focus head. The seven focus rungs are ordered, so a softmax over seven
independent classes penalizes a one-rung error as heavily as a five-rung error. CORAL
replaces it with a single score s and six learnable cut-points b_k, with
P(rung > k) = sigmoid(s + b_k). The cumulative values are made monotone and then
differenced, so the head still exposes seven log-probabilities and the decoder is
unchanged. (d) TGC band head. One head is applied to each of eight depth bands; its input
concatenates the trunk representation, the band's own pooled sequence feature, the band's
current slider value and a one-hot band index. Sharing the head keeps the number of bands
a property of the data rather than of the architecture.

Figure 3. Parameter budget.
(a) Share of the 4.99 M parameters by functional group. (b) Parameters per module, coloured
by the same grouping as in (a). The 2-D image encoder accounts for most of the model and
the Transformer for most of the remainder; the eight output heads together account for
under 7 %. Modules below 0.03 M are omitted. Disabling the far-wall head gives 4.96 M.
"""


# ---------------------------------------------------------------------------------------
#   检查与导出

def check_bounds(deck, name):
    width = deck.slide_width / 914400.0
    height = deck.slide_height / 914400.0
    bad = []
    for slide in deck.slides:
        for shape in slide.shapes:
            left = shape.left / 914400.0
            top = shape.top / 914400.0
            right = left + shape.width / 914400.0
            bottom = top + shape.height / 914400.0
            if (left < -0.005 or top < -0.005
                    or right > width + 0.005 or bottom > height + 0.005):
                bad.append((name, shape.shape_type, round(left, 3), round(top, 3),
                            round(right, 3), round(bottom, 3)))
    return bad


def export(paths, dpi=600):
    """每张图导出 PDF（矢量，给 LaTeX）、EMF（矢量，给 Word）、PNG（600 dpi 备用）。"""
    try:
        import win32com.client
    except Exception as error:                                        # noqa: BLE001
        emit("  no pywin32 (%s); only the pptx files were written" % error)
        return []
    written = []
    app = None
    try:
        app = win32com.client.Dispatch("PowerPoint.Application")
        for pptx_path, width_in, height_in in paths:
            full = os.path.abspath(pptx_path)
            stem = os.path.splitext(full)[0]
            presentation = app.Presentations.Open(full, WithWindow=False)
            presentation.SaveAs(stem + ".pdf", 32)                    # ppSaveAsPDF
            written.append(stem + ".pdf")
            presentation.Slides(1).Export(stem + ".emf", "EMF")
            written.append(stem + ".emf")
            presentation.Slides(1).Export(stem + ".png", "PNG",
                                          int(round(width_in * dpi)),
                                          int(round(height_in * dpi)))
            written.append(stem + ".png")
            presentation.Close()
    except Exception as error:                                        # noqa: BLE001
        emit("  PowerPoint refused the export (%s)" % error)
    finally:
        if app is not None:
            try:
                app.Quit()
            except Exception:                                         # noqa: BLE001
                pass
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the publication architecture figures")
    parser.add_argument("--out-dir", default="docs/figures")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args(argv)

    emit("=========== publication figures: SixParamNet architecture ===========")
    emit("  canvas width %.1f in (two-column), type set at the printed size" % FIG_W)
    facts = probe_model()
    if facts is None:
        facts = FALLBACK
    else:
        emit("  probed SixParamNet: %d parameters, %d without the far-wall head"
             % (facts["total"], facts["total_plain"]))

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    builders = [("fig1_architecture", figure_architecture),
                ("fig2_design_choices", figure_choices),
                ("fig3_parameter_budget", figure_budget)]
    paths, bad = [], []
    emit("")
    emit("--- figures ---")
    for name, builder in builders:
        deck = builder(facts)
        bad.extend(check_bounds(deck, name))
        path = os.path.join(args.out_dir, name + ".pptx")
        deck.save(path)
        width = deck.slide_width / 914400.0
        height = deck.slide_height / 914400.0
        shapes = sum(len(s.shapes) for s in deck.slides)
        emit("  %-24s %.2f x %.2f in  %3d shapes  -> %d x %d px at %d dpi"
             % (name, width, height, shapes, int(width * args.dpi),
                int(height * args.dpi), args.dpi))
        paths.append((path, width, height))

    emit("")
    emit("--- bounds check ---")
    if bad:
        for name, kind, left, top, right, bottom in bad:
            emit("  %s %s outside: left %.3f top %.3f right %.3f bottom %.3f"
                 % (name, kind, left, top, right, bottom))
    else:
        emit("  every shape is inside its canvas")

    caption_path = os.path.join(args.out_dir, "captions.txt")
    io.open(caption_path, "w", encoding="utf-8", newline="\n").write(CAPTIONS)

    emit("")
    emit("--- written ---")
    for path, _, _ in paths:
        emit("  %s  (%.1f KB)" % (path, os.path.getsize(path) / 1024.0))
    emit("  %s  (%.1f KB)" % (caption_path, os.path.getsize(caption_path) / 1024.0))
    if not args.no_export:
        for path in export(paths, args.dpi):
            if os.path.exists(path):
                emit("  %s  (%.0f KB)" % (path, os.path.getsize(path) / 1024.0))

    io.open(args.report, "w", encoding="utf-8", newline="\n").write("\n".join(LINES) + "\n")
    print("")
    print("report written to %s" % args.report)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
