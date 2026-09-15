# -*- coding: utf-8 -*-
"""闭环优化仿真：网络建议 -> 执行 -> 重新观测，在缓存的设置网格上跑到停下。

每个体模缓存了全部 深度 x 频率 x 聚焦 设置（168 个），所以前端改动可以真实地"换一幅图"，
不是在同一幅图上假装。每条轨迹从一个分片及其标签起点出发，每一步：

  1. 网络看当前图像与当前设置，给出三轴最优档与增益 / TGC 修正量。
  2. 前端建议与当前不同：跳到该设置的分片（聚焦若在新深度下不存在，取存在的最接近档，优先较浅），
     后端保持同一绝对曝光（gain_db - reference_db 不变）和同一组滑块值；不做后端修正，下一步重看。
  3. 前端不动：执行后端修正——增益按每级 dB 取整成点击数，滑块取整裁剪到 0..255。
     增益点击为 0 且三组滑块修正都在死区内，则停下。
  4. 达到 max_steps 仍未停下记为未收敛；前端回到走过的设置记为打转。

停下后用该分片的标签评判：前端三轴的标签方向是否都为"正确"、增益与 TGC 离教师最优还有多少 dB。
停止用的死区是固定值（stop_deadband_levels，默认 0.5 级，约为训练集死区中位数），因为实机上
没有逐帧死区可用；评判用的死区是该分片标签自己的。
"""

from collections import Counter

import numpy as np
import torch

from . import constants as K
from .dataset import make_batch
from .metrics import decode


def build_grid(data):
    """(体模, 深度下标, 频率下标, 聚焦下标) -> 行下标。"""
    grid = {}
    enc = data.encoded
    for i in range(data.n):
        grid[(data.group_ids[i], int(enc["depth_idx"][i]), int(enc["frequency_idx"][i]),
              int(enc["focus_idx"][i]))] = i
    return grid


def _resolve_setting(grid, group, depth_idx, frequency_idx, focus_idx):
    """建议的设置在网格里不存在时（聚焦超过新深度），取存在的最接近聚焦，优先较浅。"""
    key = (group, depth_idx, frequency_idx, focus_idx)
    if key in grid:
        return key
    candidates = [k for k in grid if k[0] == group and k[1] == depth_idx and k[2] == frequency_idx]
    if not candidates:
        return None
    return min(candidates, key=lambda k: (abs(k[3] - focus_idx), k[3] > focus_idx))


@torch.no_grad()
def run_closed_loop(model, data, builder, idx, max_steps=8, batch_size=64, amp=False,
                    stop_deadband_levels=0.5):
    model.eval()
    enc = data.encoded
    grid = build_grid(data)
    idx = np.asarray(idx, np.int64)
    n = len(idx)
    cur = idx.copy()
    gain_rel = (enc["gain_db"][idx] - enc["reference_db"][idx]).astype(np.float64)
    levels = enc["tgc_levels"][idx].astype(np.float64)
    done = np.zeros(n, bool)
    oscillated = np.zeros(n, bool)
    steps = np.zeros(n, np.int64)
    frontend_moves = np.zeros(n, np.int64)
    backend_moves = np.zeros(n, np.int64)
    visited = [{(int(enc["depth_idx"][i]), int(enc["frequency_idx"][i]), int(enc["focus_idx"][i]))} for i in idx]

    for _ in range(int(max_steps)):
        active = np.flatnonzero(~done)
        if active.size == 0:
            break
        for s in range(0, active.size, batch_size):
            chunk = active[s:s + batch_size]
            rows = cur[chunk]
            state = {"gain_db": torch.as_tensor(gain_rel[chunk] + enc["reference_db"][rows], dtype=torch.float32,
                                                device=data.device),
                     "tgc_levels": torch.as_tensor(levels[chunk], dtype=torch.float32, device=data.device)}
            inputs, tg = make_batch(data, builder, rows, state=state)
            with torch.autocast(device_type=data.device.type, dtype=torch.float16, enabled=bool(amp)):
                out = model(inputs)
            dec = {k: v.float().cpu().numpy() if v.is_floating_point() else v.cpu().numpy()
                   for k, v in decode(out, tg).items()}

            for j, traj in enumerate(chunk):
                row = int(rows[j])
                steps[traj] += 1
                mode = int(enc["mode"][row])
                gain_slope = K.GAIN_DB_PER_LEVEL[mode]
                tgc_slope = K.TGC_DB_PER_LEVEL[mode]
                here = (int(enc["depth_idx"][row]), int(enc["frequency_idx"][row]), int(enc["focus_idx"][row]))
                wanted = (int(dec["depth_idx"][j]), int(dec["frequency_idx"][j]), int(dec["focus_idx"][j]))
                key = _resolve_setting(grid, data.group_ids[row], *wanted) if wanted != here else None
                if key is not None and key[1:] != here:
                    cur[traj] = grid[key]
                    frontend_moves[traj] += 1
                    if key[1:] in visited[traj]:
                        oscillated[traj] = True
                    visited[traj].add(key[1:])
                    continue

                clicks = np.round(dec["gain_delta_db"][j] / gain_slope)
                if abs(dec["gain_delta_db"][j] / gain_slope) <= stop_deadband_levels:
                    clicks = 0.0
                delta_levels = dec["tgc_delta_db"][j] / tgc_slope
                slider_deadband = stop_deadband_levels * gain_slope / tgc_slope
                sliders_ok = all(abs(delta_levels[lo:hi].mean()) <= slider_deadband for _, lo, hi in K.SLIDER_GROUPS)
                if clicks == 0 and sliders_ok:
                    done[traj] = True
                    continue
                gain_rel[traj] += clicks * gain_slope
                new_levels = levels[traj] if sliders_ok else np.clip(np.round(levels[traj] + delta_levels),
                                                                    K.TGC_MIN_LEVEL, K.TGC_MAX_LEVEL)
                levels[traj] = new_levels
                backend_moves[traj] += 1

    # 评判
    final = cur
    records = []
    for t in range(n):
        start_row, row = int(idx[t]), int(final[t])
        mode = int(enc["mode"][row])
        gain_slope, tgc_slope = K.GAIN_DB_PER_LEVEL[mode], K.TGC_DB_PER_LEVEL[mode]
        gain_db = gain_rel[t] + enc["reference_db"][row]
        gain_err = float(enc["optimal_gain_db"][row] - gain_db)
        start_gain_err = float(enc["optimal_gain_db"][start_row] - enc["gain_db"][start_row])
        tgc_err = float(np.abs(enc["optimal_tgc_levels"][row] - levels[t]).mean() * tgc_slope)
        start_tgc_err = float(np.abs(enc["optimal_tgc_levels"][start_row] - enc["tgc_levels"][start_row]).mean()
                              * tgc_slope)
        axes_ok = {}
        for axis in ("depth", "frequency", "focus"):
            if enc["%s_mask" % axis][row] > 0:
                axes_ok[axis] = bool(enc["%s_dir" % axis][row] == 1)
        records.append({
            "start_frame": data.frame_ids[start_row], "final_frame": data.frame_ids[row],
            "converged": bool(done[t]), "oscillated": bool(oscillated[t]), "steps": int(steps[t]),
            "frontend_moves": int(frontend_moves[t]), "backend_moves": int(backend_moves[t]),
            "frontend_correct": axes_ok, "gain_error_db": gain_err, "start_gain_error_db": start_gain_err,
            "gain_within_deadband": bool(abs(gain_err) <= enc["deadband_gain_levels"][row] * gain_slope),
            "tgc_mae_db": tgc_err, "start_tgc_mae_db": start_tgc_err,
        })
    return summarise(records), records


def summarise(records):
    n = len(records)
    if n == 0:
        return {"trajectories": 0}
    out = {"trajectories": n,
           "converged": float(np.mean([r["converged"] for r in records])),
           "oscillated": float(np.mean([r["oscillated"] for r in records])),
           "steps_hist": dict(sorted(Counter(r["steps"] for r in records).items())),
           "frontend_moves_hist": dict(sorted(Counter(r["frontend_moves"] for r in records).items())),
           "gain_abs_error_db_start": float(np.mean([abs(r["start_gain_error_db"]) for r in records])),
           "gain_abs_error_db_final": float(np.mean([abs(r["gain_error_db"]) for r in records])),
           "gain_within_deadband_final": float(np.mean([r["gain_within_deadband"] for r in records])),
           "tgc_mae_db_start": float(np.mean([r["start_tgc_mae_db"] for r in records])),
           "tgc_mae_db_final": float(np.mean([r["tgc_mae_db"] for r in records]))}
    for axis in ("depth", "frequency", "focus"):
        vals = [r["frontend_correct"][axis] for r in records if axis in r["frontend_correct"]]
        out["final_%s_correct" % axis] = float(np.mean(vals)) if vals else float("nan")
    all_ok = [all(r["frontend_correct"].values()) for r in records if r["frontend_correct"]]
    out["final_frontend_all_correct"] = float(np.mean(all_ok)) if all_ok else float("nan")
    return out
