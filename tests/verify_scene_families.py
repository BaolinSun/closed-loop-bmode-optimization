# -*- coding: utf-8 -*-
"""检查 data/scene_families.json 的切分是否成立。

族的用途是：族内的帧可以互相比较来选前端设置，族间的不能。所以要查的不变量就是
这句话的两半——

一、【族内一致】同一族里前端设置相同的两帧，相关必须 >= 阈值。若低于，说明这一族
   里混进了别的探头位置，族内比较会把探头移动当成旋钮的功劳。

二、【族间确有区别】同一场次、不同族里前端设置相同的两帧，相关必须 < 阈值。若高于，
   说明切多了，本可以合并成一族的帧被拆开，白白损失可比对象。

三、【帧不重不漏】291 帧每帧属于且只属于一族。

用法：python tests/verify_scene_families.py
"""

import io
import itertools
import json
import os
import sys

sys.path.insert(0, "bmode_opt")
import numpy as np

import scene_family as SF
from hisense_loader import DEFAULT_DATA_DIR, find_captures, load_capture

FAMILIES_JSON = "data/scene_families.json"


def main():
    lines = []
    emit = lines.append
    failures = []

    data = json.load(io.open(FAMILIES_JSON, encoding="utf-8"))
    threshold = data["same_scene_corr"]
    families = data["families"]

    profiles, settings, owner = {}, {}, {}
    for record in families:
        for path in find_captures(DEFAULT_DATA_DIR / record["session"]):
            capture = load_capture(path)
            if capture.name not in record["frame_names"]:
                continue
            profiles[capture.name] = SF.scene_profile(capture)
            settings[capture.name] = SF.front_setting(capture)
            owner.setdefault(capture.name, []).append(record["family_id"])

    emit(u"=========== 3. every frame in exactly one family ===========")
    on_disk = sum(1 for _ in DEFAULT_DATA_DIR.rglob("Algo_BC0.bin"))
    assigned = sum(len(r["frame_names"]) for r in families)
    duplicated = [name for name, ids in owner.items() if len(ids) > 1]
    emit(u"  captures on disk %d, assigned to a family %d, in more than one %d"
         % (on_disk, assigned, len(duplicated)))
    if on_disk != assigned:
        failures.append("frame count: %d on disk, %d assigned" % (on_disk, assigned))
    if duplicated:
        failures.append("%d frames in more than one family" % len(duplicated))

    emit(u"")
    emit(u"=========== 1. inside a family, same setting must still correlate ===========")
    emit(u"%-26s %-11s %6s %6s %6s %4s %9s %s"
         % (u"family", u"mode", u"depth", u"freq", u"focus", u"n", u"min corr", u""))
    worst_inside = 1.0
    for record in families:
        groups = {}
        for name in record["frame_names"]:
            if profiles.get(name) is not None:
                groups.setdefault(settings[name], []).append(name)
        for key, members in sorted(groups.items()):
            if len(members) < 2:
                continue
            values = [SF.scene_correlation(profiles[a], profiles[b])
                      for a, b in itertools.combinations(members, 2)]
            low = min(values)
            worst_inside = min(worst_inside, low)
            flag = u""
            if low < threshold:
                flag = u"BELOW THRESHOLD"
                failures.append("%s holds a %.4f pair at one setting"
                                % (record["family_id"], low))
            emit(u"%-26s %-11s %6.1f %6.2f %6.1f %4d %9.4f %s"
                 % (record["family_id"],
                    "harmonic" if key[0] else "fundamental",
                    key[1], key[2], key[3], len(members), low, flag))
    emit(u"  lowest within-family, same-setting correlation: %.4f (threshold %.2f)"
         % (worst_inside, threshold))

    emit(u"")
    emit(u"=========== 2. across families of one session, the split must be real ===========")
    emit(u"  Same setting, different family. These are the pairs the split separated;")
    emit(u"  each must sit below the threshold or the split was gratuitous.")
    emit(u"")
    emit(u"%-20s %-24s %-24s %9s %s"
         % (u"session", u"family A", u"family B", u"corr", u""))
    by_session = {}
    for record in families:
        by_session.setdefault(record["session"], []).append(record)
    best_across, checked = 0.0, 0
    for session, records in sorted(by_session.items()):
        for a, b in itertools.combinations(records, 2):
            pairs = []
            for name_a in a["frame_names"]:
                for name_b in b["frame_names"]:
                    if (profiles.get(name_a) is None or profiles.get(name_b) is None
                            or settings[name_a] != settings[name_b]):
                        continue
                    pairs.append(SF.scene_correlation(profiles[name_a],
                                                      profiles[name_b]))
            if not pairs:
                continue
            checked += 1
            high = max(pairs)
            best_across = max(best_across, high)
            flag = u""
            if high >= threshold:
                flag = u"ABOVE THRESHOLD, split may be gratuitous"
                failures.append("%s and %s reach %.4f at one setting"
                                % (a["family_id"], b["family_id"], high))
            emit(u"%-20s %-24s %-24s %9.4f %s"
                 % (session, a["family_id"], b["family_id"], high, flag))
    if checked:
        emit(u"  highest across-family, same-setting correlation: %.4f (threshold %.2f)"
             % (best_across, threshold))
        emit(u"  margin between the two distributions: %.4f" % (worst_inside - best_across))
    else:
        emit(u"  no session has two families sharing a setting, nothing to check")

    emit(u"")
    emit(u"=========== result ===========")
    if failures:
        for line in failures:
            emit(u"  FAIL  " + line)
    else:
        emit(u"  all invariants hold")

    text = "\n".join(lines)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "verify_scene_families.txt")
    io.open(out, "w", encoding="utf-8").write(text)
    print(text)
    print("\nwrote %s" % out)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
