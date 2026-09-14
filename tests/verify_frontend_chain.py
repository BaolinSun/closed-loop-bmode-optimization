# -*- coding: utf-8 -*-
"""前端三根轴的标签连起来走，会不会停下来。

闭环调节就是反复「看图 -> 按标签拧旋钮 -> 再看图」。每根轴的标签单独看都说得通，
连起来却可能互相踢皮球：频率标签说「降频」，降完深度标签说「加深」，加深后频率标签
又说「再降」。本脚本在每个场景族内，从每个采到的设置出发，按标签把三根轴同时拧到
最优，落到的设置若也采到过，就读它的标签接着走，直到：

  停下      下一步就是当前设置（三根轴都说 correct，频率落在可接受集合内也算）
  打转      回到走过的设置
  出网格    下一步的设置族里没采到，无从知道它的标签

同时统计一步一致性：从 A 走到 B 且 B 采到过时，B 的标签是否说「不用再动」。

用法：python tests/verify_frontend_chain.py [labels.jsonl ...]
      默认检查 data/labels_console.jsonl；有改动前的备份时一并检查，便于对照。
"""

import collections
import io
import json
import os
import sys

DEFAULT = ["data/labels_console_before_frequency_fix.jsonl", "data/labels_console.jsonl"]
MAX_STEPS = 10


def next_setting(row):
    depth = row["optimal_depth_mm"] if row.get("depth_determined") else row["depth_mm"]
    focus = row["optimal_focus_mm"] if row.get("focus_determined") else row["focus_mm"]
    frequency = row["frequency_mhz"]
    if row.get("frequency_determined"):
        acceptable = row.get("frequency_acceptable_mhz")
        if not (acceptable and row["frequency_mhz"] in acceptable):
            frequency = row["optimal_frequency_mhz"]
    return (round(depth, 1), round(frequency, 2), round(focus, 1))


def check(path, emit):
    rows = [json.loads(l) for l in io.open(path, encoding="utf-8")]
    families = collections.defaultdict(dict)
    conflicts = 0
    for r in rows:
        if "family_id" not in r:
            continue
        key = (r["family_id"], r["imaging_mode"])
        setting = (round(r["depth_mm"], 1), round(r["frequency_mhz"], 2), round(r["focus_mm"], 1))
        if setting in families[key] and next_setting(families[key][setting]) != next_setting(r):
            conflicts += 1
        families[key].setdefault(setting, r)

    outcomes = collections.Counter()
    one_step = collections.Counter()
    moves = collections.Counter()
    examples = []
    for key, grid in sorted(families.items()):
        for start in sorted(grid):
            path_, current = [start], start
            while True:
                row = grid[current]
                nxt = next_setting(row)
                if nxt == current:
                    outcomes["stops"] += 1
                    break
                if nxt in path_:
                    outcomes["cycles"] += 1
                    examples.append((key, path_ + [nxt], "cycle"))
                    break
                if nxt not in grid:
                    outcomes["leaves captured grid"] += 1
                    break
                path_.append(nxt)
                current = nxt
                if len(path_) > MAX_STEPS:
                    outcomes["too long"] += 1
                    break
            # 一步一致性与方向统计只看第一步。
            row = grid[start]
            nxt = next_setting(row)
            if nxt != start:
                moves[(u"depth %s" % (u"deeper" if nxt[0] > start[0] else u"shallower" if nxt[0] < start[0] else u"same"),
                       u"freq %s" % (u"higher" if nxt[1] > start[1] else u"lower" if nxt[1] < start[1] else u"same"))] += 1
                if nxt in grid:
                    one_step["target also says stop" if next_setting(grid[nxt]) == nxt
                             else "target wants to move again"] += 1
                    if next_setting(grid[nxt]) != nxt and len(examples) < 40:
                        examples.append((key, [start, nxt, next_setting(grid[nxt])], "second move"))
                else:
                    one_step["target not captured"] += 1
    # 往回走：某根轴在一条链上先往一个方向拧、后来又往反方向拧。不是打转（最终会停），
    # 但说明第一步拧过了头——通常是频率在当前深度下只能取最低档，深度收浅后又能升回去。
    reversals = collections.Counter()
    for key, grid in sorted(families.items()):
        for start in sorted(grid):
            chain, current = [start], start
            while current in grid and len(chain) <= MAX_STEPS:
                nxt = next_setting(grid[current])
                if nxt == current or nxt in chain:
                    break
                chain.append(nxt)
                current = nxt
            for axis, label in enumerate(("depth", "frequency", "focus")):
                signs = [1 if b[axis] > a[axis] else -1 for a, b in zip(chain, chain[1:])
                         if b[axis] != a[axis]]
                if any(x != y for x, y in zip(signs, signs[1:])):
                    reversals[label] += 1
    total = sum(len(g) for g in families.values())
    emit(u"--- %s ---" % path)
    emit(u"  %d distinct settings in %d families (same setting, different labels: %d)"
         % (total, len(families), conflicts))
    emit(u"  following the labels from every setting: %s" % dict(outcomes))
    emit(u"  first move, where the target was captured: %s" % dict(one_step))
    emit(u"  chains that turn an axis back the way it came: %s" % dict(reversals))
    emit(u"  first move directions (depth, frequency): %s"
         % u", ".join(u"%s/%s %d" % (d, f, n) for (d, f), n in moves.most_common()))
    shown = 0
    for key, chain, kind in examples:
        if kind == "second move" and shown < 12:
            emit(u"    %-30s %-11s %s" % (key[0], key[1], u" -> ".join(
                u"(%g mm, %g MHz, focus %g)" % s for s in chain)))
            shown += 1
    for key, chain, kind in examples:
        if kind == "cycle":
            emit(u"    CYCLE %-24s %-11s %s" % (key[0], key[1], u" -> ".join(
                u"(%g mm, %g MHz, focus %g)" % s for s in chain)))
    return outcomes


def main():
    paths = sys.argv[1:] or [p for p in DEFAULT if os.path.exists(p)]
    lines = []
    emit = lines.append
    results = {p: check(p, emit) for p in paths}
    emit(u"")
    text = u"\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_frontend_chain.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    return 1 if results[paths[-1]].get("cycles") else 0


if __name__ == "__main__":
    sys.exit(main())
