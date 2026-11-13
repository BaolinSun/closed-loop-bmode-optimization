# -*- coding: utf-8 -*-
"""六参数网络的详细架构图：PPT 三页 + 同内容的 PNG。

    为什么自己画而不是用框图工具

页面上的每个方块都是 PowerPoint 原生形状，不是贴图，所以汇报时能直接改字、改配色、拆页。
张量形状与参数量不是手抄的：--probe 会真的把 SixParamNet 实例化一遍，按模块统计参数量、
跑一次前向拿到各输出的形状，再写进页面；模型改了重新生成就对得上。

    三页

1  网络架构总览：输入三路 -> 二维/一维残差编码器（逐级 FiLM）-> 融合为 33 个 token ->
   沿深度的 Transformer -> 分流与主干 -> 八个输出头。
2  关键机制放大：FiLM 条件调制、后端输出「最优值」而非「修正量」、聚焦头的 CORAL 有序回归、
   TGC 分段头的输入拼接。四处都是有理由的选择，理由写在图旁边。
3  参数预算与张量尺寸：按模块的参数量占比、从输入到输出的形状全表。

    图片

--png 用 PowerPoint 的 COM 接口导出（本机装了 Office 才行），所以 PNG 与 PPT 一定一致；
没有 Office 时只写 pptx，并在报告里说明。

用法：python tools_build_architecture_deck.py
终端输出与写在脚本旁边的 tools_build_architecture_deck.txt 同内容（ASCII）。
"""

import argparse
import io
import os

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

import tools_build_demo_deck as D

REPORT_PATH = "tools_build_architecture_deck.txt"

WIDE, HIGH, MARGIN = D.WIDE, D.HIGH, D.MARGIN
BG, PANEL, LINE = D.BG, D.PANEL, D.LINE
INK, DIM, MUTED = D.INK, D.DIM, D.MUTED
BLUE, ORANGE, GREEN = D.BLUE, D.ORANGE, D.GREEN
VIOLET = RGBColor(0x8B, 0x6F, 0xD6)

LADDERS = {"depth_mm": [25.1, 33.5, 41.9, 50.2, 58.6, 67.0, 75.4, 83.7, 92.1],
           "frequency_mhz": [4.4, 4.7, 5.0, 5.3, 5.7],
           "focus_mm": [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]}

LINES = []


def emit(text=""):
    LINES.append(text)
    print(text)


# ---------------------------------------------------------------------------------------
#   额外的画图零件（其余直接用 tools_build_demo_deck 里的）

def up_arrow(slide, x, y, w=0.16, h=0.18, colour=MUTED):
    shape = slide.shapes.add_shape(MSO_SHAPE.UP_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = colour
    shape.line.fill.background()
    shape.shadow.inherit = False
    shape.text_frame.text = ""
    return shape


def label(slide, x, y, w, text, size=11.5, colour=DIM, bold=False,
          align=PP_ALIGN.LEFT, latin=D.LATIN, h=0.26, spacing=1.25):
    D.write(D.text_box(slide, x, y, w, h, align), [(text, size, colour, bold)],
            space_after=0, line_spacing=spacing, latin=latin)


def mono(slide, x, y, w, text, size=9.5, colour=MUTED, align=PP_ALIGN.CENTER):
    label(slide, x, y, w, text, size, colour, False, align, "Consolas", 0.22)


def head_card(slide, x, y, w, h, head, accent=BLUE, size=11.5):
    """只有标题一行的卡片：自选图形的文本默认垂直居中，
    卡片剩下的地方还要写别的，所以要改成顶对齐。"""
    shape = D.block(slide, x, y, w, h, head, [], accent, size, 9)
    shape.text_frame.vertical_anchor = MSO_ANCHOR.TOP
    return shape


def flow_cell(slide, x, y, w, h, text, accent=BLUE, size=10.5, fill_t=0.18):
    """流程里的一格：淡色底 + 一行字，字号自己排不下时靠自动换行。"""
    shape = D.panel(slide, x, y, w, h, D.tint(accent, fill_t), D.tint(accent, 0.55))
    frame = shape.text_frame
    frame.margin_left = frame.margin_right = Inches(0.10)
    frame.margin_top = frame.margin_bottom = Inches(0.04)
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    frame.paragraphs[0].line_spacing = 1.2
    run = frame.paragraphs[0].add_run()
    run.text = text
    D.set_font(run, size, INK, False)
    return shape


def table_rows(slide, x, y, w, head, rows, widths, accent=BLUE,
               row_h=0.24, head_size=10, body_size=10):
    """手画的表：表头一行 + 每行一条细线。widths 是各列占比。"""
    total = float(sum(widths))
    xs, run_x = [], x
    for fraction in widths:
        xs.append((run_x, w * fraction / total))
        run_x += w * fraction / total
    for i, text in enumerate(head):
        cx, cw = xs[i]
        align = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT
        label(slide, cx, y, cw, text, head_size, accent, True, align, D.LATIN, 0.22)
    D.hairline(slide, x, y + 0.26, x + w, y + 0.26, D.tint(accent, 0.5))
    for r, row in enumerate(rows):
        ry = y + 0.34 + r * row_h
        for i, text in enumerate(row):
            cx, cw = xs[i]
            align = PP_ALIGN.LEFT if i == 0 else PP_ALIGN.RIGHT
            latin = D.LATIN if i == 0 else "Consolas"
            colour = DIM if i == 0 else INK
            label(slide, cx, ry, cw, text, body_size, colour, False, align, latin, 0.22)
        if r + 1 < len(rows):
            D.hairline(slide, x, ry + row_h - 0.04, x + w, ry + row_h - 0.04, LINE, 0.5)
    return y + 0.34 + len(rows) * row_h


# ---------------------------------------------------------------------------------------
#   模型事实：实例化一遍，别手抄

def probe_model():
    """返回 (参数量按模块, 总参数量, 输出形状)。拿不到 torch 时返回 None。"""
    try:
        import torch
        from bmode_dl.model import SixParamNet, count_parameters
    except Exception as error:                                        # noqa: BLE001
        emit("  could not import the model (%s); the deck falls back to the" % error)
        emit("  figures recorded in this script")
        return None
    model = SixParamNet(LADDERS, focus_head="ordinal", far_wall=True)
    groups = {}
    for name, parameter in model.named_parameters():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + parameter.numel()
    with torch.no_grad():
        out = model({"image": torch.randn(2, 3, 512, 128),
                     "profile": torch.randn(2, 5, 512),
                     "scalars": torch.randn(2, 17)})
    shapes = dict((k, tuple(v.shape[1:]) or (1,)) for k, v in out.items())
    plain = SixParamNet(LADDERS, focus_head="ordinal", far_wall=False)
    return {"groups": groups, "total": count_parameters(model),
            "total_plain": count_parameters(plain), "shapes": shapes}


FALLBACK = {"groups": {"image_encoder": 2951456, "sequence_model": 1054208,
                       "profile_encoder": 278336, "trunk": 197120, "seq_proj": 98560,
                       "param_encoder": 70400, "tgc_band_head": 66945, "depth_head": 34444,
                       "slider_dir_head": 34057, "frequency_head": 33928, "focus_head": 33540,
                       "gain_head": 33412, "dr_head": 33412, "aux_head": 33154,
                       "far_wall_head": 33025, "pos_embed": 8448, "seq_norm": 512,
                       "focus_cut": 6},
            "total": 4994963, "total_plain": 4961944, "shapes": {}}

MODULE_CN = {"image_encoder": u"二维残差编码器", "sequence_model": u"Transformer 编码器",
             "profile_encoder": u"一维残差编码器", "trunk": u"共享主干",
             "seq_proj": u"序列投影", "param_encoder": u"参数嵌入",
             "tgc_band_head": u"TGC 分段头", "depth_head": u"深度头",
             "slider_dir_head": u"滑块方向头", "frequency_head": u"频率头",
             "focus_head": u"聚焦头（CORAL）", "gain_head": u"增益头",
             "dr_head": u"动态范围头", "aux_head": u"辅助头",
             "far_wall_head": u"远壁深度头", "pos_embed": u"位置编码",
             "seq_norm": u"序列 LayerNorm", "focus_cut": u"CORAL 阈值"}


# ---------------------------------------------------------------------------------------
#   第 1 页：网络架构总览

def page_overview(deck, facts):
    slide = D.add_slide(deck)
    D.title(slide, u"六参数网络的架构：一次前向，八个输出头",
            u"SixParamNet · 图像 + 深度剖面 + 当前参数 → 六参数建议")

    row_y, row_h = 1.38, 2.78
    x1, w1 = MARGIN, 2.45
    x2, w2 = 3.39, 3.62
    x3, w3 = 7.33, 2.53
    x4, w4 = 10.18, 2.53
    for x, w in ((x1, w1), (x2, w2), (x3, w3), (x4, w4)):
        D.panel(slide, x, row_y, w, row_h)
    mid = row_y + row_h / 2 - 0.10
    for gx in (x1 + w1 + 0.03, x2 + w2 + 0.03, x3 + w3 + 0.03):
        D.arrow(slide, gx, mid, 0.26, 0.20)

    # ---- 输入三路
    D.badge(slide, x1 + 0.16, row_y + 0.12, u"1", BLUE)
    label(slide, x1 + 0.50, row_y + 0.14, w1 - 0.66, u"输入", 13, INK, True)
    D.block(slide, x1 + 0.16, 1.80, w1 - 0.32, 0.80, u"image  3 x 512 x 128",
            [u"当前设置下的渲染灰阶 / 255",
             u"绝对 dB（标准化）",
             u"物理深度 / 60 mm"], BLUE, 11.5, 9)
    D.block(slide, x1 + 0.16, 2.66, w1 - 0.32, 0.66, u"profile  5 x 512",
            [u"逐行 dB 的 10/50/90 分位",
             u"灰阶中位数、中位减底噪"], BLUE, 11.5, 9)
    D.block(slide, x1 + 0.16, 3.38, w1 - 0.32, 0.74, u"scalars  17",
            [u"六参数当前值 + TGC 八段滑块",
             u"曝光参考 · 成像模式 · 底噪上下"], BLUE, 11.5, 9)

    # ---- 编码器：逐级 FiLM
    D.badge(slide, x2 + 0.16, row_y + 0.12, u"2", BLUE)
    label(slide, x2 + 0.50, row_y + 0.14, w2 - 0.66,
          u"残差编码器（每级由当前参数调制）", 13, INK, True)

    label(slide, x2 + 0.17, 1.82, w2 - 0.34,
          u"二维：stem 7x3 + 四级 ResBlock2d", 9.5, DIM,
          False, PP_ALIGN.LEFT, D.LATIN, 0.22)
    stages = [(u"3→32", u"256x128", 0.56), (u"32→32", u"128x128", 0.48),
              (u"32→64", u"64x64", 0.41), (u"64→128", u"64x32", 0.35),
              (u"128→256", u"32x16", 0.29)]
    base, step, sw = 2.60, 0.655, 0.50
    xs = [x2 + 0.17 + i * step for i in range(5)]
    for i, (channels, resolution, height) in enumerate(stages):
        D.stack(slide, xs[i], base - height, sw, height, D.tint(BLUE, 0.80 - 0.07 * i))
        mono(slide, xs[i] - 0.07, base + 0.03, sw + 0.14, channels, 8.5, INK)
        mono(slide, xs[i] - 0.07, base + 0.18, sw + 0.14, resolution, 8, MUTED)

    label(slide, x2 + 0.17, 2.96, w2 - 0.34,
          u"一维：平均池化到 128 + 两级 ResBlock1d", 9.5, DIM,
          False, PP_ALIGN.LEFT, D.LATIN, 0.22)
    for i, channels in enumerate((u"5→64", u"64→64", u"64→128")):
        cx = x2 + 0.17 + i * step
        D.cell(slide, cx, 3.18, sw, 0.24, D.tint(GREEN, 0.75 - 0.10 * i))
        mono(slide, cx - 0.07, 3.44, sw + 0.14, channels, 8.5, INK)
    mono(slide, x2 + 0.17 + 3 * step, 3.22, 1.10, u"→ 128 x 32", 8.5, MUTED, PP_ALIGN.LEFT)

    for i in range(5):
        up_arrow(slide, xs[i] + sw / 2 - 0.08, 3.60, 0.16, 0.14, D.tint(ORANGE, 0.75))
    flow_cell(slide, x2 + 0.17, 3.78, w2 - 0.34, 0.36,
              u"FiLM 调制：参数嵌入 MLP 17→256→256 → γ, β",
              ORANGE, 10.5)

    # ---- 融合为深度序列
    D.badge(slide, x3 + 0.16, row_y + 0.12, u"3", VIOLET)
    label(slide, x3 + 0.50, row_y + 0.14, w3 - 0.66, u"融合为深度序列", 13, INK, True)
    flow_cell(slide, x3 + 0.17, 1.84, w3 - 0.34, 0.40,
              u"图像沿横向平均 → 256 x 32", BLUE)
    flow_cell(slide, x3 + 0.17, 2.32, w3 - 0.34, 0.40, u"剖面特征 128 x 32", GREEN)
    flow_cell(slide, x3 + 0.17, 2.80, w3 - 0.34, 0.44,
              u"拼接 384 → 线性投影 256", VIOLET)
    token_y = 3.38
    cw, gap = 0.21, 0.045
    tx = x3 + 0.17
    D.cell(slide, tx, token_y, cw, 0.28, ORANGE, 0.3)
    for i in range(8):
        D.cell(slide, tx + (i + 1) * (cw + gap), token_y, cw, 0.28, D.tint(VIOLET, 0.85), 0.3)
    label(slide, tx + 9 * (cw + gap) + 0.02, token_y - 0.02, 0.4, u"…", 13, MUTED)
    label(slide, x3 + 0.17, token_y + 0.34, w3 - 0.34,
          u"1 个参数 token（橙）+ 32 个深度 token + 位置编码",
          8.5, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.34, 1.15)

    # ---- Transformer
    D.badge(slide, x4 + 0.16, row_y + 0.12, u"4", VIOLET)
    label(slide, x4 + 0.50, row_y + 0.14, w4 - 0.66, u"沿深度的自注意力", 13, INK, True)
    flow_cell(slide, x4 + 0.17, 1.84, w4 - 0.34, 0.46,
              u"LayerNorm → 多头自注意力 8 头 → 残差", VIOLET, 10)
    flow_cell(slide, x4 + 0.17, 2.38, w4 - 0.34, 0.46,
              u"LayerNorm → 前馈 256→512→256 → 残差", VIOLET, 10)
    label(slide, x4 + 0.17, 2.92, w4 - 0.34,
          u"以上两步 × 2 层（pre-norm, GELU）",
          9.5, ORANGE, True, PP_ALIGN.CENTER, D.LATIN, 0.22)
    flow_cell(slide, x4 + 0.17, 3.20, w4 - 0.34, 0.32, u"输出 LayerNorm", VIOLET, 10)
    label(slide, x4 + 0.17, 3.60, w4 - 0.34,
          u"参数 token 与 32 个深度 token 彼此可见",
          8.5, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.40, 1.2)

    # ---- 第二行：主干与输出头
    row2_y, row2_h = 4.38, 2.72
    D.panel(slide, MARGIN, row2_y, w1, row2_h)
    heads_x, heads_w = 3.39, 9.32
    D.panel(slide, heads_x, row2_y, heads_w, row2_h)
    D.arrow(slide, MARGIN + w1 + 0.03, row2_y + row2_h / 2 - 0.10, 0.26, 0.20)

    D.badge(slide, MARGIN + 0.16, row2_y + 0.12, u"5", GREEN)
    label(slide, MARGIN + 0.50, row2_y + 0.14, w1 - 0.66, u"分流与主干", 13, INK, True)
    flow_cell(slide, MARGIN + 0.16, 4.86, w1 - 0.32, 0.46,
              u"参数 token + 深度 token 均值", GREEN, 10)
    flow_cell(slide, MARGIN + 0.16, 5.40, w1 - 0.32, 0.42,
              u"拼接 512 → MLP 512→256→256", GREEN, 10)
    D.block(slide, MARGIN + 0.16, 5.94, w1 - 0.32, 0.76,
            u"另一路：池化为 8 段",
            [u"只给 TGC 头：每段看本段回波，",
             u"不是整幅图的平均"], GREEN, 11, 9)

    D.badge(slide, heads_x + 0.16, row2_y + 0.12, u"6", ORANGE)
    label(slide, heads_x + 0.50, row2_y + 0.14, 1.8, u"八个输出头", 13, INK, True)
    legend = heads_x + 2.40
    for text, colour in ((u"后端 增益·TGC", BLUE),
                         (u"前端 深度·频率·聚焦", ORANGE),
                         (u"辅助监督", GREEN)):
        legend += D.chip(slide, legend, row2_y + 0.14, text, colour, 9) + 0.14

    heads = [
        (u"增益", u"MLP 256→128→(1+3)",
         [u"输出最优增益 dB", u"修正量在网络外相减"], BLUE),
        (u"TGC 八段", u"逐段 MLP 521→128→1",
         [u"主干 + 本段序列特征", u"+ 当前滑块 + 段编号"], BLUE),
        (u"滑块方向", u"MLP 256→128→3x3",
         [u"近 / 中 / 远三组", u"每组偏低/合适/偏高"], BLUE),
        (u"动态范围 · 辅助", u"MLP 256→128→(1+3)",
         [u"动态范围本集无监督", u"辅助头出衰减与噪声"], MUTED),
        (u"深度", u"MLP 256→128→(9+3)",
         [u"9 档 logits", u"+ 方向三类"], ORANGE),
        (u"频率", u"MLP 256→128→(5+3)",
         [u"5 档 logits", u"+ 方向三类"], ORANGE),
        (u"聚焦（CORAL）", u"MLP 257→128→(1+3)",
         [u"1 个分数 + 6 个阈值", u"输入拼入远壁预测"], ORANGE),
        (u"远壁深度", u"MLP 256→128→1",
         [u"辅助监督（Huber）", u"detach 后喂聚焦头"], GREEN),
    ]
    card_w, card_h, card_gap = 2.11, 1.03, 0.16
    for i, (name, shape_text, lines, accent) in enumerate(heads):
        cx = heads_x + 0.20 + (i % 4) * (card_w + card_gap)
        cy = 4.84 + (i // 4) * (card_h + 0.10)
        head_card(slide, cx, cy, card_w, card_h, name, accent)
        mono(slide, cx + 0.12, cy + 0.34, card_w - 0.24, shape_text, 8.5, INK, PP_ALIGN.LEFT)
        for j, line in enumerate(lines):
            label(slide, cx + 0.12, cy + 0.55 + j * 0.19, card_w - 0.24, line,
                  8.5, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.19, 1.1)

    total = facts["total"] / 1e6
    plain = facts["total_plain"] / 1e6
    label(slide, MARGIN, 7.18, 12.09,
          u"参数量 %.2f M（不含远壁头 %.2f M）；"
          u"一次前向同时给出六个参数的建议值与方向，"
          u"闭环只用后端两路，前端三档一次给定。" % (total, plain),
          9.5, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.26)
    return slide


# ---------------------------------------------------------------------------------------
#   第 2 页：关键机制

def page_mechanisms(deck, facts):
    slide = D.add_slide(deck)
    D.title(slide, u"四处不是默认选择的地方，以及它们的理由",
            u"条件调制 · 后端输出方式 · 聚焦头的序关系 · TGC 的分段输入")

    cols = ((MARGIN, 5.88), (6.82, 5.89))
    rows = ((1.38, 2.78), (4.38, 2.72))

    # ---- FiLM
    x, w = cols[0]
    y, h = rows[0]
    D.panel(slide, x, y, w, h)
    D.badge(slide, x + 0.18, y + 0.14, u"1", ORANGE)
    label(slide, x + 0.54, y + 0.16, w - 0.72, u"FiLM：让编码器看见「当前设置」", 14, INK, True)
    label(slide, x + 0.20, y + 0.56, w - 0.40,
          u"同一幅原始 dB 图可以对应任意当前增益与 TGC，"
          u"「偏暗还是偏亮」只有和当前设置放在一起才有定义，"
          u"所以调制要进编码器。",
          11, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.60, 1.3)
    fy = y + 1.36
    flow_cell(slide, x + 0.20, fy, 1.52, 0.42, u"scalars 17", BLUE, 10.5)
    D.arrow(slide, x + 1.80, fy + 0.11, 0.26, 0.20)
    flow_cell(slide, x + 2.14, fy, 1.72, 0.42, u"参数嵌入 256", ORANGE, 10.5)
    D.arrow(slide, x + 3.94, fy + 0.11, 0.26, 0.20)
    flow_cell(slide, x + 4.28, fy, 1.40, 0.42, u"γ, β 各 C 维", ORANGE, 10.5)
    label(slide, x + 0.20, fy + 0.60, w - 0.40, u"x ← x · (1 + γ) + β",
          16, INK, True, PP_ALIGN.CENTER, "Consolas", 0.34)
    label(slide, x + 0.20, fy + 1.02, w - 0.40,
          u"投影层权重与偏置都零初始化，训练开始时 γ = β = 0，这一层是恒等映射，"
          u"不会在头几个 epoch 把编码器的特征打乱；二维编码器四级、一维编码器两级，各有一组。",
          10, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.60, 1.3)

    # ---- optimum vs delta
    x, w = cols[1]
    y, h = rows[0]
    D.panel(slide, x, y, w, h)
    D.badge(slide, x + 0.18, y + 0.14, u"2", BLUE)
    label(slide, x + 0.54, y + 0.16, w - 0.72, u"后端输出「最优值」，不输出「修正量」", 14, INK, True)
    fy = y + 0.46
    flow_cell(slide, x + 0.20, fy, 2.50, 0.42, u"网络 → 最优增益 / 最优 TGC（dB）", BLUE, 10)
    D.arrow(slide, x + 2.78, fy + 0.11, 0.26, 0.20)
    flow_cell(slide, x + 3.12, fy, 2.57, 0.42, u"修正量 = 最优 − 当前（网络外相减）", BLUE, 10)
    label(slide, x + 0.20, fy + 0.50, w - 0.40,
          u"最优值 = 训练集均值 + 标准差 × 网络输出，末层零初始化，"
          u"训练起点就是「按训练集均值给建议」，不是随机值。",
          10, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.44, 1.3)
    ry = fy + 1.00
    label(slide, x + 0.20, ry, w - 0.40, u"换成让网络直接输出修正量，踩过两个坑：",
          10.5, INK, True, PP_ALIGN.LEFT, D.LATIN, 0.26)
    label(slide, x + 0.20, ry + 0.30, w - 0.40,
          u"· 自己做减法留下约 0.3 dB 残差，输给只看设置的查表基线；",
          10, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.26, 1.3)
    label(slide, x + 0.20, ry + 0.60, w - 0.40,
          u"· 当前增益被标准化成强输入时，网络学成「最优 ≈ 当前 + 修正」，"
          u"斜率中位 1.45，闭环正反馈发散（最终误差 12–36 dB）。",
          10, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.44, 1.3)
    label(slide, x + 0.20, ry + 1.08, w - 0.40,
          u"现在的斜率 0.01–0.03，超过 1 的样本占 0%，闭环收敛率 99.4%。",
          10, GREEN, True, PP_ALIGN.LEFT, D.LATIN, 0.26)

    # ---- CORAL
    x, w = cols[0]
    y, h = rows[1]
    D.panel(slide, x, y, w, h)
    D.badge(slide, x + 0.18, y + 0.14, u"3", ORANGE)
    label(slide, x + 0.54, y + 0.16, w - 0.72, u"聚焦头：有序回归（CORAL）+ 远壁先验", 14, INK, True)
    label(slide, x + 0.20, y + 0.54, w - 0.40,
          u"聚焦七档 10–40 mm 本来有序，softmax 把它当成七个无关类别，"
          u"猜错一档和猜错五档的代价一样。CORAL 把序关系写进结构：",
          10.5, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.50, 1.3)
    label(slide, x + 0.20, y + 1.10, w - 0.40,
          u"P(档 > k) = σ(s + b_k)，  k = 0 … 5",
          14, INK, True, PP_ALIGN.CENTER, "Consolas", 0.32)
    label(slide, x + 0.20, y + 1.48, w - 0.40,
          u"一个分数 s 配 6 个可学习阈值 b_k，所有阈值共享同一个分数，"
          u"所以「越深越可能超过」天然成立；累积概率先按 cummin 压成单调，再差分得逐档概率，"
          u"保证非负且和为 1，对外仍是 7 维 logits，解码与闭环代码不用改。",
          10, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.74, 1.3)
    cut_y = y + 2.26
    for i in range(7):
        cx = x + 0.26 + i * 0.52
        D.cell(slide, cx, cut_y, 0.42, 0.22, D.tint(ORANGE, 0.45 + 0.08 * i), 0.3)
        mono(slide, cx - 0.05, cut_y + 0.25, 0.52, u"%d" % (10 + 5 * i), 8, MUTED)
    label(slide, x + 3.96, cut_y - 0.04, w - 4.16,
          u"远壁深度头的预测值 detach 后拼进聚焦头输入——"
          u"教师定档位就是先找远壁再比散斑。",
          9, GREEN, False, PP_ALIGN.LEFT, D.LATIN, 0.50, 1.25)

    # ---- TGC band head
    x, w = cols[1]
    y, h = rows[1]
    D.panel(slide, x, y, w, h)
    D.badge(slide, x + 0.18, y + 0.14, u"4", BLUE)
    label(slide, x + 0.54, y + 0.16, w - 0.72, u"TGC 分段头：八段共享一个头", 14, INK, True)
    label(slide, x + 0.20, y + 0.54, w - 0.40,
          u"八段不是八个独立输出，而是同一个头跑八遍，每遍换一份输入。"
          u"这样段数变了不用改网络，段与段之间的规律也能共享。",
          10.5, DIM, False, PP_ALIGN.LEFT, D.LATIN, 0.50, 1.3)
    parts = [(u"主干 h", u"256", BLUE), (u"本段序列特征", u"256", VIOLET),
             (u"本段当前滑块", u"1", ORANGE), (u"段编号 one-hot", u"8", GREEN)]
    px = x + 0.20
    for name, width, accent in parts:
        pw = 1.36 if len(name) > 5 else 1.00
        flow_cell(slide, px, y + 1.08, pw, 0.42, name, accent, 9.5)
        mono(slide, px, y + 1.54, pw, width, 9, MUTED)
        px += pw + 0.10
    label(slide, px - 0.02, y + 1.16, 0.5, u"→", 14, MUTED)
    flow_cell(slide, x + 0.20, y + 1.86, w - 0.40, 0.42,
              u"拼接 521 → MLP 521→128→1 → 该段最优 dB", BLUE, 10.5)
    label(slide, x + 0.20, y + 2.36, w - 0.40,
          u"段特征来自 Transformer 输出的深度序列按 8 段平均池化，"
          u"每段看的是本段深度的回波，不是整幅图的平均。",
          9.5, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.34, 1.25)
    return slide


# ---------------------------------------------------------------------------------------
#   第 3 页：参数预算与张量尺寸

def page_budget(deck, facts):
    slide = D.add_slide(deck)
    D.title(slide, u"参数预算与张量尺寸",
            u"实例化 SixParamNet 后统计，不是手抄")

    x, w = MARGIN, 5.88
    y, h = 1.38, 5.72
    D.panel(slide, x, y, w, h)
    label(slide, x + 0.22, y + 0.16, w - 0.44, u"参数量按模块", 14, INK, True)
    groups = facts["groups"]
    total = float(sum(groups.values()))
    order = sorted(groups.items(), key=lambda item: -item[1])
    rows = []
    for name, count in order:
        if count < 1000:
            continue
        rows.append([MODULE_CN.get(name, name), "%.3f M" % (count / 1e6),
                     "%.1f%%" % (100 * count / total)])
    rows.append([u"其余（LayerNorm、CORAL 阈值）",
                 "%.3f M" % (sum(c for _, c in order if c < 1000) / 1e6), u"< 0.1%"])
    end = table_rows(slide, x + 0.22, y + 0.56, w - 0.44,
                     [u"模块", u"参数量", u"占比"], rows, (3.1, 1.0, 0.8), BLUE)
    label(slide, x + 0.22, end + 0.14, w - 0.44,
          u"六成参数在二维图像编码器，两成在 Transformer；八个输出头加起来不到 6%%。"
          u"合计 %.2f M（关掉远壁头 %.2f M）。"
          % (facts["total"] / 1e6, facts["total_plain"] / 1e6),
          10, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.50, 1.3)

    x, w = 6.82, 5.89
    D.panel(slide, x, y, w, h)
    label(slide, x + 0.22, y + 0.16, w - 0.44, u"从输入到输出的形状", 14, INK, True)
    flow = [
        [u"image", u"3 x 512 x 128", u"输入"],
        [u"stem 7x3 + ResBlock2d x 4", u"256 x 32 x 16", u"FiLM x 4"],
        [u"沿横向平均 + 池化", u"256 x 32", u""],
        [u"profile", u"5 x 512", u"输入"],
        [u"池化 + ResBlock1d x 2", u"128 x 32", u"FiLM x 2"],
        [u"拼接 384 + 线性投影", u"32 x 256", u""],
        [u"前置参数 token + 位置编码", u"33 x 256", u""],
        [u"Transformer x 2 层", u"33 x 256", u"8 头"],
        [u"参数 token + 深度均值 → 主干 MLP", u"512 → 256", u""],
        [u"深度序列 → 8 段池化", u"8 x 256", u"只给 TGC"],
    ]
    end = table_rows(slide, x + 0.22, y + 0.56, w - 0.44,
                     [u"阶段", u"形状", u"备注"], flow, (3.0, 1.5, 1.0), VIOLET)

    label(slide, x + 0.22, end + 0.16, w - 0.44, u"输出", 12, INK, True)
    shapes = facts.get("shapes") or {}

    def shape_of(key, fallback):
        if key in shapes:
            dims = shapes[key]
            return " x ".join(str(d) for d in dims) if dims != (1,) else "1"
        return fallback

    outs = [
        [u"最优增益 / 方向", "%s  +  %s" % (shape_of("gain_optimal_db", "1"),
                                            shape_of("gain_dir", "3")), u"后端"],
        [u"最优 TGC 八段 / 滑块方向", "%s  +  %s" % (shape_of("tgc_optimal_db", "8"),
                                                     shape_of("slider_dir", "3 x 3")), u"后端"],
        [u"深度 / 频率 / 聚焦 logits",
         "%s · %s · %s" % (shape_of("depth_logits", "9"), shape_of("frequency_logits", "5"),
                           shape_of("focus_logits", "7")), u"前端"],
        [u"聚焦累积 logits（CORAL）", shape_of("focus_cum_logits", "6"), u"前端"],
        [u"远壁深度 / 辅助", "%s  +  %s" % (shape_of("far_wall_mm", "1"),
                                            shape_of("aux", "2")), u"辅助"],
    ]
    end = table_rows(slide, x + 0.22, end + 0.46, w - 0.44,
                     [u"输出", u"形状", u"归属"], outs, (3.0, 1.5, 1.0), ORANGE)
    label(slide, x + 0.22, end + 0.12, w - 0.44,
          u"每个档位头另出三类方向（偏低 / 合适 / 偏高），评估里与档位 logits 分开统计。",
          9.5, MUTED, False, PP_ALIGN.LEFT, D.LATIN, 0.26, 1.25)
    return slide


# ---------------------------------------------------------------------------------------
#   检查与导出

def check_bounds(deck):
    """所有形状都要落在画布里，溢出就报出来——PowerPoint 不会提醒，导出的图会少东西。"""
    bad = []
    for index, slide in enumerate(deck.slides, 1):
        for shape in slide.shapes:
            left = shape.left / 914400.0
            top = shape.top / 914400.0
            right = left + shape.width / 914400.0
            bottom = top + shape.height / 914400.0
            if left < -0.01 or top < -0.01 or right > WIDE + 0.01 or bottom > HIGH + 0.01:
                bad.append((index, shape.shape_type, round(left, 2), round(top, 2),
                            round(right, 2), round(bottom, 2)))
    return bad


def export_png(pptx_path, out_dir, width=2400):
    """用 PowerPoint 的 COM 接口把每页导成 PNG；没装 Office 就返回空列表。"""
    try:
        import win32com.client
    except Exception as error:                                        # noqa: BLE001
        emit("  no pywin32 (%s); the PNG export is skipped" % error)
        return []
    full = os.path.abspath(pptx_path)
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    app = None
    written = []
    try:
        app = win32com.client.Dispatch("PowerPoint.Application")
        deck = app.Presentations.Open(full, WithWindow=False)
        height = int(round(width * HIGH / WIDE))
        for index in range(1, deck.Slides.Count + 1):
            target = os.path.join(out_dir, "%s_p%d.png"
                                  % (os.path.splitext(os.path.basename(full))[0], index))
            deck.Slides(index).Export(target, "PNG", width, height)
            written.append(target)
        deck.Close()
    except Exception as error:                                        # noqa: BLE001
        emit("  PowerPoint refused the export (%s); only the pptx was written" % error)
        return []
    finally:
        if app is not None:
            try:
                app.Quit()
            except Exception:                                         # noqa: BLE001
                pass
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the architecture deck and its images")
    parser.add_argument("--out", default="docs/six_param_architecture.pptx")
    parser.add_argument("--png-dir", default="docs")
    parser.add_argument("--png-width", type=int, default=2400)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args(argv)

    emit("=========== six-parameter network architecture deck ===========")
    facts = probe_model()
    if facts is None:
        facts = FALLBACK
        emit("  using the recorded figures")
    else:
        emit("  probed SixParamNet: %d parameters with the far-wall head, %d without"
             % (facts["total"], facts["total_plain"]))
        emit("  %d output tensors, %d parameter groups"
             % (len(facts["shapes"]), len(facts["groups"])))

    deck = D.new_deck()
    page_overview(deck, facts)
    page_mechanisms(deck, facts)
    page_budget(deck, facts)

    emit("")
    emit("--- pages ---")
    for index, slide in enumerate(deck.slides, 1):
        emit("  page %d: %d shapes" % (index, len(slide.shapes)))

    bad = check_bounds(deck)
    emit("")
    emit("--- bounds check (canvas %.3f x %.3f in) ---" % (WIDE, HIGH))
    if bad:
        for index, kind, left, top, right, bottom in bad:
            emit("  page %d %s outside: left %.2f top %.2f right %.2f bottom %.2f"
                 % (index, kind, left, top, right, bottom))
    else:
        emit("  every shape is inside the canvas")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    deck.save(args.out)
    emit("")
    emit("--- written ---")
    emit("  %s  (%.1f KB)" % (args.out, os.path.getsize(args.out) / 1024.0))

    if not args.no_png:
        for path in export_png(args.out, args.png_dir, args.png_width):
            emit("  %s  (%.0f KB)" % (path, os.path.getsize(path) / 1024.0))

    io.open(args.report, "w", encoding="utf-8", newline="\n").write("\n".join(LINES) + "\n")
    print("")
    print("report written to %s" % args.report)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
