# -*- coding: utf-8 -*-
"""这个彩色映射是不是「单一灰度索引 -> RGB」的一维查找表？"""
import io, os, sys
sys.path.insert(0, "bmode_opt")
import numpy as np
from PIL import Image
import tissue as T
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture, crop_capture_image

HERE = os.path.dirname(os.path.abspath(__file__))
lines = []; w = lines.append
PICKS = [("20260903", 1), ("20260903_GEN", 0), ("20260904", 0)]

w(u"=========== DC. B 图区里有多少种不同的 RGB 三元组 ===========")
w(u"  若是一维彩色查找表，最多 256 种；否则就不是简单的着色")
w(u"")
w(u"%-28s %7s %14s %16s %16s" % (
    u"场次", u"模式", u"不同RGB数", u"像素数", u"是否 <=256"))
shots = []
for sess, mode in PICKS:
    caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / sess)]
    caps = [c for c in caps if T.capture_image_mode(c) == mode]
    cap = sorted(caps, key=lambda c: (c.geometry.depth_mm, c.name))[len(caps) // 2]
    rgb = np.asarray(Image.open(cap.path / "Screenthum.bmp").convert("RGB"))
    _, (r0, r1, c0, c1) = crop_capture_image(cap)
    sub = rgb[r0:r1, c0:c1].reshape(-1, 3)
    codes = (sub[:, 0].astype(np.int64) << 16) | (sub[:, 1].astype(np.int64) << 8) | sub[:, 2]
    uniq = np.unique(codes)
    shots.append(((sess, mode), cap, rgb, (r0, r1, c0, c1)))
    w(u"%-28s %7s %14d %16d %16s" % (
        sess[:26], u"谐波" if mode else u"通用", uniq.size, codes.size,
        u"是" if uniq.size <= 256 else u"否"))

w(u"")
w(u"=========== DD. 把这些 RGB 按亮度排序，看它们是不是一条曲线 ===========")
key, cap, rgb, (r0, r1, c0, c1) = shots[1]
sub = rgb[r0:r1, c0:c1].reshape(-1, 3)
codes = (sub[:, 0].astype(np.int64) << 16) | (sub[:, 1].astype(np.int64) << 8) | sub[:, 2]
uniq, counts = np.unique(codes, return_counts=True)
tri = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], axis=1)
order = np.argsort(tri.sum(axis=1))
tri, counts = tri[order], counts[order]
w(u"  %s / %s，共 %d 种颜色，按 R+G+B 排序后每隔若干取一个：" % (
    key[0], u"谐波" if key[1] else u"通用", tri.shape[0]))
w(u"%8s %6s %6s %6s %12s" % (u"序号", u"R", u"G", u"B", u"像素数"))
step = max(1, tri.shape[0] // 24)
for i in range(0, tri.shape[0], step):
    w(u"%8d %6d %6d %6d %12d" % (i, tri[i, 0], tri[i, 1], tri[i, 2], counts[i]))

w(u"")
w(u"=========== DE. 屏幕上找灰阶条（一列内颜色单调变化的窄条）===========")
full = rgb
h, wd = full.shape[:2]
best = []
for c in range(0, wd - 8, 4):
    strip = full[:, c:c + 8].reshape(h, -1, 3).mean(axis=1)
    lum = strip.sum(axis=1)
    nz = np.flatnonzero(lum > 8)
    if nz.size < 100:
        continue
    seg = lum[nz[0]:nz[-1] + 1]
    if seg.size < 100:
        continue
    d = np.diff(seg)
    mono = max((d >= -1).mean(), (d <= 1).mean())
    span = seg.max() - seg.min()
    if mono > 0.97 and span > 300:
        best.append((c, nz[0], nz[-1], mono, span))
w(u"  候选竖条（列, 起行, 止行, 单调比例, 亮度跨度）：")
for b in best[:12]:
    w(u"    %5d %5d %5d %8.3f %8.0f" % b)
if not best:
    w(u"    没找到；灰阶条可能是横的或不在这张图里")

io.open(os.path.join(HERE, "s7_colour.txt"), "w", encoding="utf-8").write(u"\n".join(lines))
print("ok")
