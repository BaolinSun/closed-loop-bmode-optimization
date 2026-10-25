# -*- coding: utf-8 -*-
"""按受试者划分人体数据的训练 / 验证 / 测试集，并把 split 字段写进标签文件。

    为什么要单独一步

人体数据的独立单位是人：同一名受试者的不同切面、不同摆放都共享他的颈部厚度与衰减，落到两边
就是泄漏。采集方案 §3.4 要求划分事先定好、不随训练变动，所以划分写在一个单独的映射文件里
（默认 data/invivo_carotid_splits.json），标签重新生成之后照样套用同一套划分；两个口径的标签
文件（穿透口径与目标口径）也因此拿到完全一致的划分。

    怎么分

受试者 = 标签 group_id 去掉成像模式后缀（与 bmode_dl.dataset.group_key(row, "subject") 同一规则）。
按采集日分层：同一天的受试者排在一起分配，让三个集合都覆盖全部采集日——主机与探头在采集期间
可能漂移（采集方案 §16.6 把这件事交给每日体模质控监测），全部测试受试者挤在最后两天就分不清
「模型在新受试者上变差」与「机器那两天不一样」。日内按帧数从多到少贪心分配，每次给到当前
离目标帧数份额最远的那个集合，所以受试者数与帧数两头都接近设定比例。

同一名受试者若有多个检查（方案 §5.3 是一人一检查，但重采会多出一个），用 --subject-alias
把它们并成一个受试者，否则同一个人会被当成两个。

    用法

    python tools_assign_invivo_splits.py                          # 建映射并写入两个标签文件
    python tools_assign_invivo_splits.py --ratios 0.7 0.15 0.15
    python tools_assign_invivo_splits.py --dry-run                # 只看会怎么分，不写文件

已有映射文件时默认直接套用（不重新分），--reassign 才会重算。写入是原地改写标签文件的 split
字段，其余字段逐字段比对确认未变。终端输出与写在脚本旁边的 tools_assign_invivo_splits.txt
同内容（ASCII）。
"""

import argparse
import collections
import io
import json
import os
import time

import numpy as np

from bmode_dl.dataset import group_key

REPORT_PATH = "tools_assign_invivo_splits.txt"
DEFAULT_LABELS = ["data/labels_console_invivo_carotid.jsonl",
                  "data/labels_console_invivo_carotid_target.jsonl"]
DEFAULT_MAP = "data/invivo_carotid_splits.json"
SPLITS = ("train", "val", "test")
AXES = ("depth", "frequency", "focus")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Assign per-subject train / val / test splits to in-vivo labels")
    p.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    p.add_argument("--map", default=DEFAULT_MAP, help="where the subject -> split assignment is kept")
    p.add_argument("--ratios", nargs=3, type=float, default=[0.70, 0.15, 0.15],
                   metavar=("TRAIN", "VAL", "TEST"), help="target share of frames per split")
    p.add_argument("--seed", type=int, default=20261025, help="tie-break order within a day")
    p.add_argument("--reassign", action="store_true", help="recompute the assignment even if the map exists")
    p.add_argument("--subject-alias", nargs="*", default=[], metavar="SESSION=SUBJECT",
                   help="merge several exams of the same person, e.g. 20261011_161332_00001=S042")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    return p.parse_args(argv)


def read_jsonl(path):
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def subject_of(row, alias):
    subject = group_key(row, "subject")
    return alias.get(subject, subject)


def day_of(subject):
    """受试者所在的采集日：场次号以 YYYYMMDD 起头。"""
    head = subject.split("_", 1)[0]
    return head if len(head) == 8 and head.isdigit() else "unknown"


def assign(frames_per_subject, ratios, seed):
    """按采集日分层 + 日内按帧数贪心，返回 {受试者: 划分}。"""
    ratios = np.asarray(ratios, float)
    ratios = ratios / ratios.sum()
    total = float(sum(frames_per_subject.values()))
    target = {name: ratios[i] * total for i, name in enumerate(SPLITS)}
    taken = dict((name, 0.0) for name in SPLITS)
    by_day = collections.defaultdict(list)
    for subject, frames in frames_per_subject.items():
        by_day[day_of(subject)].append(subject)
    rng = np.random.RandomState(seed)
    out = {}
    for day in sorted(by_day):
        subjects = sorted(by_day[day])
        rng.shuffle(subjects)
        # 帧数多的先分，大块先落位，剩下的小块才好把份额补平
        subjects.sort(key=lambda s: -frames_per_subject[s])
        for subject in subjects:
            # 离目标份额最远（缺口最大）的集合先拿
            name = max(SPLITS, key=lambda k: (target[k] - taken[k]) / max(target[k], 1.0))
            out[subject] = name
            taken[name] += frames_per_subject[subject]
    return out


def write_split(path, assignment, alias, emit, dry_run):
    """把 split 字段写进标签文件，并确认其余字段没动。"""
    rows = read_jsonl(path)
    before = [dict(r) for r in rows]
    counts, missing = collections.Counter(), set()
    for row in rows:
        subject = subject_of(row, alias)
        name = assignment.get(subject)
        if name is None:
            missing.add(subject)
            continue
        row["split"] = name
        counts[name] += 1
    if missing:
        raise SystemExit("%s: %d subject(s) are not in the map (%s); run with --reassign"
                         % (path, len(missing), ", ".join(sorted(missing)[:3])))
    changed = set()
    for old, new in zip(before, rows):
        for key in set(old) | set(new):
            if key != "split" and old.get(key) != new.get(key):
                changed.add(key)
    if changed:
        raise SystemExit("%s: fields other than split would change: %s" % (path, sorted(changed)))
    emit("  %s: %d rows -> %s" % (path, len(rows),
                                  "  ".join("%s %d" % (k, counts[k]) for k in SPLITS)))
    if dry_run:
        return rows
    with io.open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def main(argv=None):
    args = parse_args(argv)
    started = time.time()
    lines = []

    def emit(text=""):
        lines.append(text)

    alias = {}
    for item in args.subject_alias:
        if "=" not in item:
            raise SystemExit("--subject-alias wants SESSION=SUBJECT, got %r" % item)
        session, subject = item.split("=", 1)
        alias[session] = subject

    emit("=========== per-subject splits for the in-vivo labels ===========")
    rows = read_jsonl(args.labels[0])
    frames_per_subject = collections.Counter(subject_of(r, alias) for r in rows)
    emit("  %s: %d rows, %d subject(s), %d acquisition day(s)"
         % (args.labels[0], len(rows), len(frames_per_subject),
            len(set(day_of(s) for s in frames_per_subject))))

    if os.path.exists(args.map) and not args.reassign:
        payload = json.load(io.open(args.map, encoding="utf-8"))
        assignment = payload["subjects"]
        emit("  reusing %s (created %s, seed %s, ratios %s)"
             % (args.map, payload.get("created"), payload.get("seed"), payload.get("ratios")))
    else:
        assignment = assign(frames_per_subject, args.ratios, args.seed)
        payload = {"note": "per-subject train/val/test split for the in-vivo carotid labels; "
                           "the acquisition protocol requires it to be fixed before training",
                   "created": time.strftime("%Y-%m-%d %H:%M"), "seed": args.seed,
                   "ratios": {name: float(r) for name, r in zip(SPLITS, args.ratios)},
                   "alias": alias, "subjects": assignment}
        if not args.dry_run:
            io.open(args.map, "w", encoding="utf-8").write(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        emit("  %s %s" % ("would write" if args.dry_run else "wrote", args.map))

    # 划分本身的样子
    emit("")
    emit("  %-6s %9s %9s %9s %9s %9s" % ("split", "subjects", "frames", "share", "depth", "frequency"))
    per_split_rows = collections.defaultdict(list)
    for row in rows:
        per_split_rows[assignment[subject_of(row, alias)]].append(row)
    for name in SPLITS:
        subjects = [s for s, v in assignment.items() if v == name]
        subset = per_split_rows[name]
        determined = dict((axis, sum(1 for r in subset if r.get("%s_determined" % axis))) for axis in AXES)
        emit("  %-6s %9d %9d %8.1f%% %9d %9d"
             % (name, len(subjects), len(subset), 100.0 * len(subset) / max(1, len(rows)),
                determined["depth"], determined["frequency"]))
        emit("         focus %d, back end %d" % (determined["focus"], len(subset)))
    emit("")
    emit("  acquisition days per split (every day should appear in all three)")
    for name in SPLITS:
        days = collections.Counter(day_of(s) for s, v in assignment.items() if v == name)
        emit("    %-6s %s" % (name, "  ".join("%s:%d" % (d, days[d]) for d in sorted(days))))
    overlap = [s for s in frames_per_subject if s not in assignment]
    emit("  subjects without an assignment: %d" % len(overlap))

    emit("")
    emit("  writing the split field")
    for path in args.labels:
        write_split(path, assignment, alias, emit, args.dry_run)
    if args.dry_run:
        emit("  dry run: nothing was written")
    emit("  done in %.1f s" % (time.time() - started))

    text = "\n".join(lines)
    io.open(REPORT_PATH, "w", encoding="utf-8").write(text + "\n")
    print(text)
    return assignment


if __name__ == "__main__":
    main()
