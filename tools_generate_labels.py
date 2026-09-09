# -*- coding: utf-8 -*-
"""Generate back-end labels for both data sources into one file with one schema."""
import argparse, io, json, os, sys, time, zlib
sys.path.insert(0, "bmode_opt")
import numpy as np
import hisense_backend_sim as S
import calibration as CAL
import labels as LB
import tissue as T
from fieldii_loader import find_shards, load_shard
from hisense_loader import DEFAULT_DATA_DIR, find_captures, get_leaf, load_capture

CAL_PATH = "bmode_opt/console_calibration.json"
CONSOLE_SESSIONS = ["20260814", "20260819", "20260831", "20260901", "20260901_E2",
                    "20260901_E3", "20260903", "20260903_GEN",
                    "20260903_replication_check", "20260904", "20260904_DR", "20260909_GEN",
                    "20260828/GEN", "20260828/THI"]


def load_calibration():
    data = json.load(io.open(CAL_PATH, encoding="utf-8"))
    out, skipped = {}, []
    for g in data["groups"]:
        # A group whose frames are mostly black cannot pin the mapping, and the fit runs off
        # along the counts/pivot ridge instead of failing visibly. Labels built on it would be
        # confident nonsense, so it is left out and named.
        if not g.get("calibratable", True):
            skipped.append("%s/%s (usable %.0f%%)"
                           % (g["session"], g["image_mode_name"],
                              100 * g.get("usable_fraction", float("nan"))))
            continue
        out[(g["session"], g["image_mode"])] = {
            "cal": CAL.GroupCalibration(g["counts_per_db"], g["pivot_db"],
                                        g["screenshot_gray_error"], 0,
                                        np.array(g["depth_axis_mm"]),
                                        np.array(g["depth_response_db"]),
                                        {float(k): np.array(v) for k, v in
                                         g.get("depth_response_by_frequency", {}).items()}),
            "floor": g["noise_floor_db"], "floor_measured": g["noise_floor_measured"],
            "uncertainty": g["label_uncertainty"],
            "uncertainty_measured": g["label_uncertainty_measured"],
        }
    # Groups with no deep frame borrow a floor from the same imaging mode.
    for key, entry in out.items():
        if entry["floor"] is not None:
            continue
        donors = [v["floor"] for k, v in out.items()
                  if k[1] == key[1] and v["floor"] is not None]
        entry["floor"] = float(np.median(donors)) if donors else None
    if skipped:
        print("skipping %d group(s) as not calibratable: %s"
              % (len(skipped), ", ".join(skipped)))
    return out


def console_targets(cal_by_group):
    """Accepted tissue brightness per group, under that group's own calibration."""
    targets = {}
    for key, entry in cal_by_group.items():
        caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / key[0])]
        caps = [c for c in caps if T.capture_image_mode(c) == key[1]]
        cal, floor = entry["cal"], entry["floor"]
        if floor is None or not caps:
            continue
        r = T.measure_accepted_brightness(
            caps,
            lambda c: S.render(c.bc0, tgc_levels=c.tgc_levels,
                               gain_db=S.gain_level_to_db(c.gain_level),
                               dynamic_range_db=S.dr_ui_to_window_db(c.dynamic_range_level),
                               depth_response_db=CAL.depth_response_for(c, cal),
                               reference_db=cal.pivot_db, counts_per_db=cal.counts_per_db),
            lambda c: T.console_tissue_mask(S.bc0_to_db(c.bc0, cal.counts_per_db), floor))
        if r:
            targets[key] = r["target_gray"]
    return targets


def label_console(cal_by_group, targets, limit=None):
    rows = []
    for key in sorted(cal_by_group):
        entry = cal_by_group[key]
        if key not in targets or entry["floor"] is None:
            continue
        cal, floor = entry["cal"], entry["floor"]
        caps = [load_capture(p) for p in find_captures(DEFAULT_DATA_DIR / key[0])]
        caps = [c for c in caps if T.capture_image_mode(c) == key[1]]
        if limit:
            caps = caps[:limit]
        for cap in caps:
            db = (S.bc0_to_db(cap.bc0, cal.counts_per_db)
                  + CAL.depth_response_for(cap, cal)[:, None])
            vm = T.console_tissue_mask(db, floor)
            if vm.sum() < 1000:
                continue
            notes = []
            if not entry["floor_measured"]:
                notes.append("noise floor borrowed from another session in this mode")
            if not entry["uncertainty_measured"]:
                notes.append("label uncertainty borrowed from another session in this mode")
            def leaf(name):
                try:
                    return float(get_leaf(cap.fe_params, name))
                except Exception:
                    return None
            freq = leaf("BFreqValue")
            # BFocusArrayPos is a 16-slot list with the active foci first; a single transmit
            # focus is the first entry, the rest are zero padding.
            try:
                focus = float(str(get_leaf(cap.fe_params, "BFocusArrayPos")).split(",")[0])
            except Exception:
                focus = None
            rows.append(LB.label_frame(
                db, vm, dr_ui=cap.dynamic_range_level, reference_db=cal.pivot_db,
                current=(S.gain_level_to_db(cap.gain_level),
                         np.asarray(cap.tgc_levels, dtype=np.float64),
                         float(cap.dynamic_range_level)),
                target_gray=targets[key], source="console", frame_id=cap.name,
                group_id="%s/%d" % key, imaging_mode=T.IMAGE_MODE_NAMES[key[1]],
                depth_mm=cap.geometry.depth_mm, frequency_mhz=freq,
                focus_mm=focus, split=None,
                gain_level=cap.gain_level,
                label_uncertainty=entry["uncertainty"],
                calibration_borrowed=not entry["floor_measured"],
                notes=notes).as_dict())
    return rows


def label_fieldii(target_gray, limit=None, seed=20260909):
    """Field II frames, each with a drawn starting point.

    Console frames arrive with a starting point already - the operator's. These do not, so one
    is drawn per frame, seeded on the frame's own name so the label set is reproducible and a
    frame keeps its start no matter what order the run happens to visit it in.
    """
    rows = []
    paths = find_shards()
    if limit:
        paths = paths[:limit]
    for path in paths:
        shard = load_shard(path)
        vm = T.fieldii_tissue_mask(shard)
        if vm.sum() < 1000:
            continue
        rng = np.random.RandomState(
            (seed + zlib.crc32(shard.name.encode("utf-8"))) % (2 ** 32))
        rows.append(LB.label_frame(
            shard.db_image, vm, dr_ui=shard.dynamic_range_level,
            reference_db=shard.tissue_median_db,
            current=None, rng=rng,
            target_gray=target_gray, source="fieldii", frame_id=shard.name,
            group_id="seed%s/%s" % (shard.seed, shard.phantom_type),
            imaging_mode="fundamental", depth_mm=shard.geometry.depth_mm,
            frequency_mhz=shard.frequency_mhz, focus_mm=shard.focus_mm,
            split=shard.split, label_uncertainty=0.0,
            notes=["render is ground truth; no screenshot to match",
                   "starting point drawn, not an operator's"]).as_dict())
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/labels_backend.jsonl")
    ap.add_argument("--console-limit", type=int, default=None)
    ap.add_argument("--fieldii-limit", type=int, default=None)
    args = ap.parse_args()

    started = time.time()
    cal_by_group = load_calibration()
    targets = console_targets(cal_by_group)
    print("console groups with a brightness target: %d" % len(targets))

    rows = label_console(cal_by_group, targets, args.console_limit)
    print("console labels: %d  (%.0f s)" % (len(rows), time.time() - started))

    fundamental = [v for k, v in targets.items() if k[1] == 0]
    fieldii_target = float(np.median(fundamental)) if fundamental else 64.0
    rows += label_fieldii(fieldii_target, args.fieldii_limit)
    print("total labels: %d  (%.0f s)" % (len(rows), time.time() - started))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
