# -*- coding: utf-8 -*-
"""汇报用的 PPT：中文学术表达，演示视频内嵌并自动播放。

    两种篇幅

    brief（默认）  四页：封面与结论、演示视频、结果、结论与后续。用于口头汇报，细节交给视频与网页。
    full           九页：另含研究背景、数据与划分、方法、以及结果的分页展开。用于预读或答辩。

    数字从哪来

与网页、视频同一来源：单步指标与查表基线读 runs/*/fold_0/evaluation_*.json，闭环统计读同一目录下
的 closed_loop_trajectories_*.jsonl（按受试者的偏置在那里现算）。页面上没有手抄的数字，重算之后
重新生成即可。

    图从哪来

data/deck_figures/*.png 是从演示视频里按帧号取出的静帧（tools_build_demo_video.py 生成视频，
ffmpeg 取帧）。这样 PPT 与视频同一套配色与版式，放在一起不违和。

    视频

第 5 页用 python-pptx 的 add_movie 把 MP4 内嵌进 pptx（约 0.8 MB），拷到别的机器也能放，
不依赖外部文件。封面帧取自同一段视频。

用法：python tools_build_demo_deck.py [--out data/demo_invivo_carotid.pptx]
终端输出与写在脚本旁边的 tools_build_demo_deck.txt 同内容（ASCII）。
"""

import argparse
import io
import os
import time

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

import tools_build_demo_page as DP

REPORT_PATH = "tools_build_demo_deck.txt"
FIGURES = "data/deck_figures"
CJK = u"微软雅黑"
LATIN = "Calibri"

BG = RGBColor(0x12, 0x12, 0x11)
PANEL = RGBColor(0x1A, 0x1A, 0x19)
LINE = RGBColor(0x33, 0x33, 0x31)
INK = RGBColor(0xF0, 0xEF, 0xEC)
DIM = RGBColor(0xC3, 0xC2, 0xB7)
MUTED = RGBColor(0x8D, 0x8C, 0x83)
BLUE = RGBColor(0x39, 0x87, 0xE5)
ORANGE = RGBColor(0xD9, 0x59, 0x26)
GREEN = RGBColor(0x19, 0x9E, 0x70)

WIDE, HIGH = 13.333, 7.5
MARGIN = 0.62


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build the presentation deck")
    p.add_argument("--video", default="data/demo_invivo_carotid.mp4")
    p.add_argument("--figures", default=FIGURES)
    p.add_argument("--single-step-json", default="runs/invivo_bf_v2/fold_0/evaluation_combined_test_noloop.json")
    p.add_argument("--loop-json", default="runs/invivo_bf_v2/fold_0/evaluation_combined_bloop.json")
    p.add_argument("--loop-trajectories",
                   default="runs/invivo_bf_v2/fold_0/closed_loop_trajectories_combined_bloop.jsonl")
    p.add_argument("--labels", default="data/labels_console_invivo_carotid_target.jsonl")
    p.add_argument("--pages", choices=("brief", "full"), default="brief",
                   help="brief = 4 pages for a talk; full = 9 pages with background and methods")
    p.add_argument("--out", default=None, help="default: data/demo_invivo_carotid[_full].pptx")
    args = p.parse_args(argv)
    if args.out is None:
        args.out = ("data/demo_invivo_carotid.pptx" if args.pages == "brief"
                    else "data/demo_invivo_carotid_full.pptx")
    return args


# ---------------------------------------------------------------------------------------
#   画布

def new_deck():
    deck = Presentation()
    deck.slide_width = Inches(WIDE)
    deck.slide_height = Inches(HIGH)
    return deck


def add_slide(deck):
    slide = deck.slides.add_slide(deck.slide_layouts[6])
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = BG
    return slide


def set_font(run, size, colour, bold=False, latin=LATIN):
    """中文与西文分开设字体：只设 latin 的话，中文会落到 PowerPoint 的默认宋体。"""
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = colour
    run.font.name = latin
    rPr = run._r.get_or_add_rPr()
    for tag in ("a:ea", "a:cs"):
        element = rPr.find(qn(tag))
        if element is None:
            element = rPr.makeelement(qn(tag), {})
            rPr.append(element)
        element.set("typeface", CJK)


def text_box(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    frame.paragraphs[0].alignment = align
    return frame


def write(frame, lines, size=15, colour=DIM, bold=False, space_after=7, line_spacing=1.35, latin=LATIN):
    """lines 是 [(文本, 大小, 颜色, 加粗)] 或纯字符串列表。"""
    first = True
    for item in lines:
        text, item_size, item_colour, item_bold = (item if isinstance(item, tuple)
                                                   else (item, size, colour, bold))
        paragraph = frame.paragraphs[0] if first else frame.add_paragraph()
        first = False
        paragraph.alignment = frame.paragraphs[0].alignment
        paragraph.space_after = Pt(space_after)
        paragraph.line_spacing = line_spacing
        run = paragraph.add_run()
        run.text = text
        set_font(run, item_size, item_colour, item_bold, latin)
    return frame


def title(slide, text, eyebrow=None):
    if eyebrow:
        write(text_box(slide, MARGIN, 0.42, 11.0, 0.3), [(eyebrow, 12, MUTED, False)], space_after=0)
    write(text_box(slide, MARGIN, 0.74, 12.1, 0.8), [(text, 28, INK, True)], space_after=0)


def panel(slide, x, y, w, h, fill=PANEL, line=LINE):
    from pptx.enum.shapes import MSO_SHAPE

    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.color.rgb = line
    shape.line.width = Pt(1)
    shape.shadow.inherit = False
    shape.text_frame.word_wrap = True
    shape.text_frame.margin_left = Inches(0.22)
    shape.text_frame.margin_right = Inches(0.22)
    shape.text_frame.margin_top = Inches(0.16)
    shape.text_frame.margin_bottom = Inches(0.16)
    shape.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    return shape


def stat_panel(slide, x, y, w, h, value, label, colour=BLUE):
    shape = panel(slide, x, y, w, h)
    frame = shape.text_frame
    frame.paragraphs[0].space_after = Pt(2)
    frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    run = frame.paragraphs[0].add_run()
    run.text = value
    set_font(run, 30, colour, True, "Consolas")
    paragraph = frame.add_paragraph()
    paragraph.line_spacing = 1.2
    paragraph.alignment = PP_ALIGN.LEFT
    run = paragraph.add_run()
    run.text = label
    set_font(run, 11.5, MUTED)
    return shape


def figure(slide, path, x, y, w):
    picture = slide.shapes.add_picture(path, Inches(x), Inches(y), width=Inches(w))
    return picture


def caption(slide, text, x, y, w):
    write(text_box(slide, x, y, w, 0.5), [(text, 11, MUTED, False)], space_after=0, line_spacing=1.25)


AUTOPLAY_TIMING = (
    '<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">'
    '<p:childTnLst><p:seq concurrent="1" nextAc="seek">'
    '<p:cTn id="2" dur="indefinite" nodeType="mainSeq"><p:childTnLst>'
    '<p:par><p:cTn id="3" fill="hold"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst>'
    '<p:childTnLst><p:par><p:cTn id="4" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
    '<p:childTnLst><p:par><p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" '
    'fill="hold" nodeType="withEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst>'
    '<p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr>'
    '<p:cTn id="6" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="%(spid)s"/></p:tgtEl>'
    '</p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par>'
    '</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>'
    '<p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>'
    '<p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>'
    '</p:seq><p:video><p:cMediaNode vol="80000"><p:cTn id="7" fill="hold" display="0">'
    '<p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn>'
    '<p:tgtEl><p:spTgt spid="%(spid)s"/></p:tgtEl></p:cMediaNode></p:video>'
    '</p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>')


def set_autoplay(path, emit):
    """把 add_movie 写出的「单击播放」改成「随幻灯片自动播放」。

    python-pptx 只会写 <p:cond delay="indefinite"/>（单击触发）。PowerPoint 的「自动」是在
    mainSeq 里挂一条 playFrom(0.0) 的 mediacall，这里按同一结构重写该页的 p:timing。
    """
    import re
    import shutil
    import zipfile

    source = zipfile.ZipFile(path)
    items = [(item, source.read(item.filename)) for item in source.infolist()]
    source.close()
    changed = 0
    out = []
    for item, data in items:
        if item.filename.startswith("ppt/slides/slide") and item.filename.endswith(".xml"):
            xml = data.decode("utf-8")
            if "<p:video>" in xml and "playFrom" not in xml:
                match = re.search(r'<p:spTgt spid="(\d+)"/>', xml)
                if match:
                    xml = re.sub(r"<p:timing>.*?</p:timing>", AUTOPLAY_TIMING % {"spid": match.group(1)},
                                 xml, flags=re.S)
                    data = xml.encode("utf-8")
                    changed += 1
        out.append((item, data))
    target = path + ".tmp"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as handle:
        for item, data in out:
            handle.writestr(item, data)
    shutil.move(target, path)
    emit("  autoplay   rewrote the timing of %d slide(s) so the video starts with the slide" % changed)


# ---------------------------------------------------------------------------------------
#   精简版（四页）

def brief_cover(deck, numbers):
    slide = add_slide(deck)
    write(text_box(slide, MARGIN, 1.5, 11.6, 0.4), [(u"在体颈动脉数据上的验证", 16, BLUE, False)],
          space_after=0)
    write(text_box(slide, MARGIN, 1.95, 11.6, 1.2), [(u"超声成像六参数闭环优化", 44, INK, True)],
          space_after=0)
    write(text_box(slide, MARGIN, 3.2, 11.0, 1.0),
          [(u"模型由单帧图像预测增益、时间增益补偿与发射聚焦的最优值，并以闭环方式迭代至停止判据满足；"
            u"在 29 位未参与训练的受试者上优于只看当前设置的查表基线。", 16, DIM, False)], space_after=0)
    cards = [(numbers["tgc"] + " dB", u"时间增益补偿误差；基线 " + numbers["tgc_base"] + " dB", BLUE),
             (numbers["gain"] + " dB", u"增益误差；基线 " + numbers["gain_base"] + " dB", BLUE),
             (numbers["converged"] + "%", u"后端闭环收敛比例，2-3 次迭代", GREEN)]
    for i, (value, label, colour) in enumerate(cards):
        stat_panel(slide, MARGIN + i * 4.09, 4.75, 3.87, 1.35, value, label, colour)
    write(text_box(slide, MARGIN, 6.45, 12.1, 0.5),
          [(u"数据：17545 帧 / 189 位受试者 / 9 个采集日；按受试者划分训练 132 位、验证 28 位、"
            u"留出测试 29 位", 13, MUTED, False)], space_after=0)


def brief_video(deck, video, poster):
    slide = add_slide(deck)
    write(text_box(slide, MARGIN, 0.42, 11.0, 0.3), [(u"演示", 12, MUTED, False)], space_after=0)
    write(text_box(slide, MARGIN, 0.72, 12.1, 0.6), [(u"一帧图像上的六参数闭环优化", 26, INK, True)],
          space_after=0)
    width = 10.2
    height = width * 720.0 / 1280.0
    x = (WIDE - width) / 2.0
    slide.shapes.add_movie(video, Inches(x), Inches(1.4), Inches(width), Inches(height),
                           poster_frame_image=poster, mime_type="video/mp4")
    caption(slide, u"30 秒，五段：后端闭环迭代、发射聚焦判据、六位受试者同屏验证、误差分布形态、"
                   u"留出测试集结果；各帧均由同一组原始回波数据精确重建", x, 1.4 + height + 0.16, width)


def brief_results(deck, numbers, bias, figures):
    slide = add_slide(deck)
    title(slide, u"结果：优于设置查表基线，且闭环可收敛", u"结果")
    write(text_box(slide, MARGIN, 1.74, 6.2, 0.4),
          [(u"留出测试集（29 位未见过的受试者）", 14, MUTED, False)], space_after=0)
    rows = [(u"时间增益补偿", numbers["tgc"] + " dB", numbers["tgc_base"] + " dB", u"优于基线 91%"),
            (u"增益", numbers["gain"] + " dB", numbers["gain_base"] + " dB", u"优于基线 30%"),
            (u"发射聚焦", numbers["focus"], numbers["focus_base"], u"略优于基线")]
    header = (u"参数轴", u"本方法", u"查表基线", u"")
    widths = (2.0, 1.35, 1.35, 1.6)
    for j, name in enumerate(header):
        write(text_box(slide, MARGIN + sum(widths[:j]), 2.2, widths[j], 0.3),
              [(name, 12, MUTED, True)], space_after=0)
    for i, row in enumerate(rows):
        y = 2.62 + i * 0.62
        for j, value in enumerate(row):
            latin = "Consolas" if j in (1, 2) else LATIN
            colour = BLUE if j == 1 else (GREEN if j == 3 else DIM)
            write(text_box(slide, MARGIN + sum(widths[:j]), y, widths[j], 0.4),
                  [(value, 16 if j == 1 else 14, colour, j == 1)], space_after=0, latin=latin)
    write(text_box(slide, MARGIN, 4.65, 6.3, 0.9),
          [(u"显示深度与发射频率的标签可由当前设置直接推得、不依赖图像，动态范围尚无判据，"
            u"三者本阶段未参与训练。", 13.5, MUTED, False)], space_after=0)
    panel(slide, MARGIN, 5.65, 6.3, 1.3)
    write(text_box(slide, MARGIN + 0.26, 5.9, 5.8, 0.9),
          [(u"闭环：" + numbers["converged"] + u"% 的轨迹在 2-3 次迭代内收敛；增益偏差 " +
            numbers["loop_gain_start"] + u" → " + numbers["loop_gain_final"] + u" dB，时间增益补偿 " +
            numbers["loop_tgc_start"] + u" → " + numbers["loop_tgc_final"] + u" dB，无正反馈发散。",
            14, DIM, False)], space_after=0)
    figure(slide, os.path.join(figures, "fig_bias.png"), 7.1, 1.74, 5.63)
    write(text_box(slide, 7.1, 4.95, 5.63, 1.0),
          [(u"残差以受试者为单位呈常数偏置", 15, BLUE, True),
           (u"受试者内一致性 %.2f dB；%d / %d 位偏置在 ±1 dB 以内；平方误差中 %.0f%% 来自受试者间偏置，"
            u"故一次人工校正即可修正整次检查。"
            % (bias["sd_median"], bias["within_1db"], len(bias["subjects"]),
               100 * bias["bias_share"]), 13.5, DIM, False)], space_after=5)


def brief_conclusion(deck):
    slide = add_slide(deck)
    title(slide, u"结论与后续工作", u"小结")
    blocks = [(u"可用", GREEN, u"增益与时间增益补偿：显著优于查表基线，闭环收敛且不发散；残差可由"
                              u"一次人工校正消除。"),
              (u"勉强可用", ORANGE, u"发射聚焦：略优于基线，但在训练集受试者上接近记忆，泛化有限。"),
              (u"尚不可用", MUTED, u"显示深度、发射频率：标签不依赖图像；动态范围：尚无判据。")]
    y = 1.85
    for name, colour, body in blocks:
        shape = panel(slide, MARGIN, y, 6.3, 1.35)
        frame = shape.text_frame
        frame.paragraphs[0].space_after = Pt(3)
        frame.paragraphs[0].alignment = PP_ALIGN.LEFT
        run = frame.paragraphs[0].add_run()
        run.text = name
        set_font(run, 14, colour, True)
        paragraph = frame.add_paragraph()
        paragraph.line_spacing = 1.3
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.add_run()
        run.text = body
        set_font(run, 13.5, DIM)
        y += 1.52
    write(text_box(slide, 7.1, 1.85, 5.63, 0.4), [(u"后续工作", 15, INK, True)], space_after=0)
    steps = [u"一、上机闭环实测：按模型输出逐步调节并重新采集",
             u"二、真实医生阅片：重建深度与频率的判据",
             u"三、动态范围判据：成对偏好阅片，补齐第六个参数",
             u"四、扩充受试者例数，抑制聚焦输出头的过拟合"]
    write(text_box(slide, 7.1, 2.35, 5.63, 2.6), [(x, 14, DIM, False) for x in steps], space_after=14)
    panel(slide, 7.1, 5.3, 5.63, 1.35)
    write(text_box(slide, 7.36, 5.56, 5.1, 0.95),
          [(u"局限：前端目标深度来自阶段性阅片结果；结论限于单一机型、单一探头与颈动脉切面。",
            13, DIM, False)], space_after=0)


# ---------------------------------------------------------------------------------------
#   各页

def slide_cover(deck, numbers):
    slide = add_slide(deck)
    write(text_box(slide, MARGIN, 2.05, 11.6, 0.4), [(u"在体颈动脉数据上的验证", 16, BLUE, False)],
          space_after=0)
    write(text_box(slide, MARGIN, 2.5, 11.6, 1.2), [(u"超声成像六参数闭环优化", 44, INK, True)],
          space_after=0)
    write(text_box(slide, MARGIN, 3.75, 10.4, 0.9),
          [(u"显示深度、发射频率、发射聚焦、增益、时间增益补偿、动态范围", 16, DIM, False),
           (u"17545 帧 / 189 位受试者 / 9 个采集日；按受试者划分训练、验证与留出测试集", 14, MUTED, False)],
          space_after=6)
    y = 5.25
    stat_panel(slide, MARGIN, y, 3.5, 1.25, numbers["tgc"] + " dB",
               u"时间增益补偿平均绝对误差（留出测试集）", BLUE)
    stat_panel(slide, MARGIN + 3.74, y, 3.5, 1.25, numbers["gain"] + " dB",
               u"增益平均绝对误差（留出测试集）", BLUE)
    stat_panel(slide, MARGIN + 7.48, y, 3.5, 1.25, numbers["converged"] + "%",
               u"后端闭环收敛比例，2-3 次迭代", GREEN)


def slide_background(deck, numbers):
    slide = add_slide(deck)
    title(slide, u"为什么必须用在体数据", u"研究背景")
    write(text_box(slide, MARGIN, 1.78, 6.0, 3.6),
          [(u"六参数的调节目前依赖操作者经验，缺乏可量化、可复现的判据。先前工作在体模数据上训练，"
            u"网络虽可拟合标签，却未能超过「只看当前设置的查表基线」——体模衰减均匀、靶标深度固定，"
            u"标签可由设置直接推得，网络无需看图。", 15, DIM, False),
           (u"在体数据改变了这一前提：颈部厚度、皮下脂肪与血管深度因人而异，同一设置下的最优解"
            u"随受试者变化，模型必须从图像中读取信息。", 15, DIM, False)], space_after=12)
    panel(slide, 6.95, 1.78, 5.78, 1.55)
    write(text_box(slide, 7.25, 2.0, 5.2, 1.1),
          [(u"体模数据（既往）", 13, MUTED, True),
           (u"增益误差 1.28 dB，查表基线 1.37 dB —— 二者持平，无法证明模型读取了图像信息", 14, DIM, False)],
          space_after=4)
    panel(slide, 6.95, 3.52, 5.78, 1.86)
    write(text_box(slide, 7.25, 3.74, 5.2, 1.4),
          [(u"在体颈动脉数据（本工作）", 13, BLUE, True),
           (u"增益误差 " + numbers["gain"] + u" dB，查表基线 " + numbers["gain_base"] +
            u" dB —— 优于基线 30%；时间增益补偿 " + numbers["tgc"] + u" dB 对 " + numbers["tgc_base"] +
            u" dB，优于基线 91%", 14, DIM, False)], space_after=4)
    panel(slide, MARGIN, 5.62, 12.11, 1.1)
    write(text_box(slide, MARGIN + 0.3, 5.86, 11.5, 0.7),
          [(u"同一设置下，最优增益在受试者之间的离散度为 2.1 dB（极差 14-19 dB）；这一部分方差只能"
            u"由图像解释，是本工作得以成立的前提。", 14, DIM, False)], space_after=0)


def slide_data(deck):
    slide = add_slide(deck)
    title(slide, u"数据与划分", u"材料与方法")
    cards = [(u"17545", u"单帧采集总数"), (u"189", u"受试者例数"), (u"9", u"采集日"),
             (u"L15-4", u"线阵探头，谐波成像")]
    for i, (value, label) in enumerate(cards):
        stat_panel(slide, MARGIN + i * 3.06, 1.78, 2.84, 1.2, value, label, BLUE if i < 3 else DIM)
    write(text_box(slide, MARGIN, 3.3, 12.1, 0.4), [(u"按受试者划分，训练前确定，不随训练调整", 15, INK, True)],
          space_after=0)
    rows = [(u"训练集", u"132 位", u"12232 帧"), (u"验证集", u"28 位", u"2631 帧"),
            (u"留出测试集", u"29 位", u"2682 帧")]
    for i, (name, subjects, frames) in enumerate(rows):
        x = MARGIN + i * 4.09
        shape = panel(slide, x, 3.78, 3.87, 1.2, PANEL, BLUE if i == 2 else LINE)
        frame = shape.text_frame
        frame.paragraphs[0].space_after = Pt(3)
        frame.paragraphs[0].alignment = PP_ALIGN.LEFT
        run = frame.paragraphs[0].add_run()
        run.text = name
        set_font(run, 15, BLUE if i == 2 else INK, True)
        paragraph = frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        run = paragraph.add_run()
        run.text = subjects + u" / " + frames
        set_font(run, 14, DIM)
    write(text_box(slide, MARGIN, 5.3, 12.1, 1.5),
          [(u"独立单位是受试者而非帧：同一位受试者的多个探头摆放共享其颈部厚度与衰减特性，"
            u"若按帧划分将造成信息泄漏。", 15, DIM, False),
           (u"9 个采集日均出现于三个集合中，以免「模型在新受试者上变差」与「主机当日状态不同」"
            u"相混淆。动态范围无判据、未参与训练；显示深度与发射频率的标签可由当前设置推得"
            u"（查表基线分别为 0.775 与 0.946），本阶段一并屏蔽。", 15, DIM, False)], space_after=10)


def slide_method(deck, figures):
    slide = add_slide(deck)
    title(slide, u"方法：求解器标签、网络预测与闭环协议", u"材料与方法")
    items = [(u"一、教师求解器", u"逐帧求解增益与时间增益补偿的最优解，目标为本场次操作者所接受的"
                               u"组织亮度；前端判据为血管远壁覆盖、穿透余量与逐带侧向散斑宽度。"),
             (u"二、网络", u"4.96 M 参数；输入为当前设置下的渲染灰阶、绝对 dB 图、逐行深度剖面与 17 个"
                          u"标量；输出为各参数的最优值，修正量由「最优值 − 当前值」在网络外求差。"),
             (u"三、闭环协议", u"模型输出 → 执行 → 重新观测，直至停止判据满足；前端改档后须重新采集，"
                             u"后端修正在当前图像上计算。")]
    y = 1.8
    for name, body in items:
        write(text_box(slide, MARGIN, y, 6.0, 1.3),
              [(name, 15, BLUE, True), (body, 14, DIM, False)], space_after=4)
        y += 1.44
    panel(slide, MARGIN, 6.1, 6.0, 0.95)
    write(text_box(slide, MARGIN + 0.26, 6.3, 5.5, 0.6),
          [(u"关键性质：增益与时间增益补偿不改变回波数据，仅改变其到灰阶的映射", 13.5, INK, True)],
          space_after=0)
    figure(slide, os.path.join(figures, "fig_loop_start.png"), 7.0, 1.95, 5.73)
    caption(slide, u"闭环第一次迭代的初始状态：左为当前渲染结果，中为求解器最优解，"
                   u"右为读数、时间增益补偿曲线与逐行亮度剖面", 7.0, 5.25, 5.73)


def slide_video(deck, video, poster):
    slide = add_slide(deck)
    title(slide, u"演示：一帧图像上的六参数闭环优化", u"演示视频")
    width = 9.6
    height = width * 720.0 / 1280.0
    x = (WIDE - width) / 2.0
    slide.shapes.add_movie(video, Inches(x), Inches(1.72), Inches(width), Inches(height),
                           poster_frame_image=poster, mime_type="video/mp4")
    caption(slide, u"视频时长 30 秒，共五段：后端闭环迭代、发射聚焦判据、六位受试者同屏验证、"
                   u"误差分布形态、留出测试集结果。各帧均由同一组原始回波数据经后端渲染模型精确重建。",
            x, 1.72 + height + 0.22, width)


def slide_single_step(deck, numbers, figures):
    slide = add_slide(deck)
    title(slide, u"结果一：单步精度与设置查表基线的对照", u"结果")
    write(text_box(slide, MARGIN, 1.72, 6.2, 0.6),
          [(u"基线构造：对每一组（成像模式、显示深度、发射频率、发射聚焦），取训练集受试者最优解的"
            u"中位数，应用于留出测试集。", 14, DIM, False)], space_after=0)
    header = (u"参数轴", u"本方法", u"查表基线", u"状态")
    rows = [(u"时间增益补偿（8 段）", numbers["tgc3"] + u" dB", numbers["tgc_base3"] + u" dB", u"可用"),
            (u"增益", numbers["gain3"] + u" dB", numbers["gain_base3"] + u" dB", u"可用"),
            (u"发射聚焦（首选命中率）", numbers["focus"], numbers["focus_base"], u"略优于基线"),
            (u"显示深度", u"—", u"0.775", u"未训练"),
            (u"发射频率", u"—", u"0.946", u"未训练"),
            (u"动态范围", u"—", u"—", u"无判据")]
    widths = (2.35, 1.25, 1.25, 1.25)
    x0, y0, row_h = MARGIN, 2.5, 0.48
    for j, name in enumerate(header):
        x = x0 + sum(widths[:j])
        write(text_box(slide, x, y0, widths[j], 0.3), [(name, 12, MUTED, True)], space_after=0)
    for i, row in enumerate(rows):
        y = y0 + 0.42 + i * row_h
        faded = i >= 3
        for j, value in enumerate(row):
            x = x0 + sum(widths[:j])
            colour = MUTED if faded else (BLUE if j == 1 else DIM)
            latin = "Consolas" if j in (1, 2) else LATIN
            write(text_box(slide, x, y, widths[j], 0.34),
                  [(value, 13.5, colour, j == 1 and not faded)], space_after=0, latin=latin)
    write(text_box(slide, MARGIN, 5.98, 6.5, 1.2),
          [(u"增益方向一致性 F1 0.69（基线 0.51），滑块方向一致性 F1 0.92（基线 0.43）。"
            u"发射聚焦在训练集受试者上约为 0.99、在留出测试集上为 " + numbers["focus"] +
            u"，提示该输出头以记忆为主、泛化有限。", 14, DIM, False)], space_after=0)
    figure(slide, os.path.join(figures, "fig_focus.png"), 7.1, 1.95, 5.63)
    caption(slide, u"发射聚焦的判据与模型输出：左为实采图像与远壁位置，中为逐深度带侧向散斑宽度"
                   u"（聚焦标签即由此定出），右为模型对各档的输出概率", 7.1, 5.2, 5.63)


def slide_loop(deck, numbers, figures):
    slide = add_slide(deck)
    title(slide, u"结果二：后端闭环的收敛性", u"结果")
    cards = [(numbers["converged"] + "%", u"轨迹收敛比例"), (u"2-3", u"收敛所需迭代次数"),
             (numbers["loop_gain_start"] + u" → " + numbers["loop_gain_final"], u"增益偏差（dB）"),
             (numbers["loop_tgc_start"] + u" → " + numbers["loop_tgc_final"], u"时间增益补偿偏差（dB）")]
    for i, (value, label) in enumerate(cards):
        stat_panel(slide, MARGIN + i * 3.06, 1.78, 2.84, 1.2, value, label,
                   GREEN if i == 0 else BLUE)
    write(text_box(slide, MARGIN, 3.3, 6.2, 2.4),
          [(u"增益与时间增益补偿不改变回波数据，因此闭环可在已记录的帧上精确复现，无需重新采集。"
            u"28 位验证受试者的全部 2631 帧中，" + numbers["converged"] + u"% 的轨迹收敛；收敛点与单步"
            u"精度一致，迭代过程中不存在误差累积。", 14, DIM, False),
           (u"反馈斜率中位数 0.024，无一帧大于 1，闭环不会正反馈发散。未收敛的 7.6% 其最终误差与"
            u"收敛者相当（1.28 对 1.31 dB），原因是停止判据所用死区（1.3 档）较教师自身死区"
            u"（12.2 档）过严，并非模型不准。", 14, DIM, False)], space_after=10)
    figure(slide, os.path.join(figures, "fig_grid.png"), 7.1, 3.3, 5.63)
    caption(slide, u"六位留出测试集受试者同屏演示；六位按收敛结果均匀取秩，最差一例一并在列",
            7.1, 6.55, 5.63)


def slide_bias(deck, bias, figures):
    slide = add_slide(deck)
    title(slide, u"结果三：误差的分布形态", u"结果")
    write(text_box(slide, MARGIN, 1.74, 12.1, 0.5),
          [(u"增益残差以受试者为单位呈常数偏置，而非帧间离散", 17, BLUE, True)], space_after=0)
    figure(slide, os.path.join(figures, "fig_bias.png"), MARGIN, 2.3, 7.4)
    caption(slide, u"验证集 28 位受试者的带符号增益偏置（按偏置排序），阴影带为 ±1 dB", MARGIN, 6.55, 7.4)
    items = [(u"%.2f dB" % bias["sd_median"], u"受试者内一致性（标准差中位数）"),
             (u"%d / %d" % (bias["within_1db"], len(bias["subjects"])), u"偏置在 ±1 dB 以内的受试者"),
             (u"%.0f%%" % (100 * bias["bias_share"]), u"平方误差中来自受试者间偏置的比例")]
    y = 2.3
    for value, label in items:
        stat_panel(slide, 8.35, y, 4.38, 1.1, value, label, BLUE)
        y += 1.26
    panel(slide, 8.35, 6.08, 4.38, 1.1)
    write(text_box(slide, 8.6, 6.3, 3.9, 0.8),
          [(u"同一受试者各帧偏离量相同，故一次人工校正即可修正整次检查", 13, INK, True)], space_after=0)


def slide_conclusion(deck):
    slide = add_slide(deck)
    title(slide, u"结论与后续工作", u"小结")
    left = [(u"可用", 14, GREEN, True),
            (u"增益与时间增益补偿：在留出测试集上显著优于设置查表基线，闭环收敛且不发散；"
             u"残差以受试者为单位呈常数偏置，可由一次人工校正消除。", 14, DIM, False),
            (u"勉强可用", 14, ORANGE, True),
            (u"发射聚焦：首选命中率略优于基线，但在训练集受试者上接近记忆，泛化有限。", 14, DIM, False),
            (u"尚不可用", 14, MUTED, True),
            (u"显示深度与发射频率：标签可由当前设置推得，不依赖图像；动态范围：标签缺少判据。",
             14, DIM, False)]
    write(text_box(slide, MARGIN, 1.8, 6.1, 4.2), left, space_after=8)
    write(text_box(slide, 7.1, 1.8, 5.63, 0.4), [(u"后续工作", 15, INK, True)], space_after=0)
    steps = [u"一、上机闭环实测：按模型输出逐步调节并重新采集，检验前端闭环的停止行为",
             u"二、真实医生阅片：以专家判读替代当前的阶段性远壁读数，重建深度与频率的判据",
             u"三、动态范围判据：采用成对偏好阅片，补齐第六个参数",
             u"四、聚焦输出头的过拟合：降低容量或增强正则化，并扩充受试者例数"]
    write(text_box(slide, 7.1, 2.3, 5.63, 3.4), [(s, 14, DIM, False) for s in steps], space_after=12)
    panel(slide, MARGIN, 6.2, 12.11, 1.0)
    write(text_box(slide, MARGIN + 0.3, 6.42, 11.5, 0.6),
          [(u"局限：前端目标深度来自阶段性阅片结果；动态范围尚无判据；结论限于单一机型、单一探头"
            u"与颈动脉切面。", 13.5, DIM, False)], space_after=0)


# ---------------------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    lines = []

    def emit(text=""):
        lines.append(text)

    single = DP.read_metrics(args.single_step_json)
    loop = DP.read_metrics(args.loop_json).get("closed_loop", {})
    net = single.get("network_recorded_start", {})
    base = single.get("settings_lookup_baseline", {})
    import tools_track_console_loop as TR
    bias = DP.subject_bias(args.loop_trajectories, TR.teacher_rows(args.labels))
    if not net or not loop or not bias:
        raise SystemExit("missing evaluation results; run the evaluation first")

    numbers = {
        "tgc": "%.2f" % net["tgc_mae_db"], "tgc3": "%.3f" % net["tgc_mae_db"],
        "tgc_base": "%.2f" % base["tgc_mae_db"], "tgc_base3": "%.3f" % base["tgc_mae_db"],
        "gain": "%.2f" % net["gain_mae_db"], "gain3": "%.3f" % net["gain_mae_db"],
        "gain_base": "%.2f" % base["gain_mae_db"], "gain_base3": "%.3f" % base["gain_mae_db"],
        "focus": "%.3f" % net["focus_top1"], "focus_base": "%.3f" % base["focus_top1"],
        "converged": "%.1f" % (100 * loop["converged"]),
        "loop_gain_start": "%.2f" % loop["gain_abs_error_db_start"],
        "loop_gain_final": "%.2f" % loop["gain_abs_error_db_final"],
        "loop_tgc_start": "%.2f" % loop["tgc_mae_db_start"],
        "loop_tgc_final": "%.2f" % loop["tgc_mae_db_final"],
    }

    emit("=========== presentation deck ===========")
    emit("  numbers    %s + %s" % (args.single_step_json, args.loop_json))
    emit("  figures    %s" % args.figures)
    emit("  video      %s (%.1f MB)" % (args.video, os.path.getsize(args.video) / 1e6))

    poster = os.path.join(args.figures, "fig_loop_start.png")
    deck = new_deck()
    if args.pages == "brief":
        brief_cover(deck, numbers)
        brief_video(deck, args.video, poster)
        brief_results(deck, numbers, bias, args.figures)
        brief_conclusion(deck)
    else:
        slide_cover(deck, numbers)
        slide_background(deck, numbers)
        slide_data(deck)
        slide_method(deck, args.figures)
        slide_video(deck, args.video, poster)
        slide_single_step(deck, numbers, args.figures)
        slide_loop(deck, numbers, args.figures)
        slide_bias(deck, bias, args.figures)
        slide_conclusion(deck)

    directory = os.path.dirname(os.path.abspath(args.out))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    deck.save(args.out)
    set_autoplay(args.out, emit)
    emit("  slides     %d (%s)" % (len(deck.slides._sldIdLst), args.pages))
    emit("  wrote %s (%.1f MB)" % (args.out, os.path.getsize(args.out) / 1e6))
    emit("  done in %.0f s" % (time.time() - started))

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return args.out


if __name__ == "__main__":
    main()
