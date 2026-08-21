#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""keyframe.py: 轨迹关键点化(L1 一体化演示核心后处理)。

背景: 执行层对每帧间位移做 rest-to-rest S 曲线重定时, 30fps 自然动作的帧间
微抖动(加速度 335°/s²、jerk 14732°/s³)远超安全限速(60°/s²、300°/s³),
导致整条轨迹被放慢 ~6 倍, 动作与语音脱同步。

思路(用户确认): 只保留轨迹的"大致趋势"(波峰波谷), 不逐帧跟踪。
  1. 高斯平滑: 去掉 30fps 高频抖动, 保留低频走向;
  2. 等间隔关键点采样(时间锚定): 说话段总时长 = 语音时长 + 收尾预算,
     关键点在段内均匀分布 —— 动作节奏由语音时长决定, 不由模型决定;
  3. 幅度削峰: 用与执行层同款 S 曲线公式(见 s_curve_time)计算每段位移的
     最短可行时间, 超预算则整段等比缩放(保形) —— 保证执行层重定时比例 ≈ 1;
  4. Silent 段直线细分回中, 终点锚定 neutral;
  5. 输出低 fps 关键点 V1 JSON(执行层逐段 S 曲线自动平滑插值)。

用法(独立):
    python tools/live/keyframe.py --traj neck_l1/unified_relative_rpy.npy \
        --states neck_l1/states.npy --out neck_l1/traj_kp.json

被 l1_demo.py 内联调用(常驻模型生成后直接关键点化)。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

RAD2DEG = 180.0 / np.pi
DEG2RAD = np.pi / 180.0
STATE_ID = {"silent": 0, "speaking": 1, "listening": 2}
STATE_NAME = {0: "silent", 1: "speaking", 2: "listening"}

# 执行层安全限速(与 Project-Motor/neck_control/neck_trajectory_config.txt 保持一致;
# keyframe.py 会尝试从该配置文件读取, 读不到用此默认值)
DEFAULT_LIMITS = {
    "vmax_deg_s": [25.0, 25.0, 30.0],
    "amax_deg_s2": [60.0, 60.0, 90.0],
    "jmax_deg_s3": [300.0, 300.0, 450.0],
}


# --------------------------------------------------------------------------- #
# S 曲线最短时间(与执行层 trajectory_retimer.cpp planSegment 同款公式)
# --------------------------------------------------------------------------- #
def ramp_dist(J: float, Tj: float, Ta: float) -> float:
    """加速 ramp 总位移: s1+s2+s3 = J*Tj^3 + 1.5*J*Tj^2*Ta + 0.5*J*Tj*Ta^2。"""
    return J * Tj ** 3 + 1.5 * J * Tj * Tj * Ta + 0.5 * J * Tj * Ta * Ta


def s_curve_time(D: float, vmax: float, amax: float, jmax: float) -> float:
    """单轴 rest-to-rest 7 段 S 曲线最短时间(秒)。D 为位移(度), 与 retimer 同款。"""
    if D <= 1e-9:
        return 0.0
    # 满速剖面
    if amax * amax / jmax >= vmax:
        Tj, Ta = vmax / amax, 0.0
    else:
        Tj, Ta = amax / jmax, vmax / amax - amax / jmax
    v1 = jmax * Tj * Tj + amax * Ta
    s_ramp = ramp_dist(jmax, Tj, Ta)
    if 2.0 * s_ramp <= D:
        Tv = (D - 2.0 * s_ramp) / v1
        return 4.0 * Tj + 2.0 * Ta + Tv
    # 位移不足: 二分峰值速度
    lo, hi = 0.0, v1
    for _ in range(80):
        vp = 0.5 * (lo + hi)
        if vp <= amax * amax / jmax:
            Tj2, Ta2 = vp / amax, 0.0
        else:
            Tj2, Ta2 = amax / jmax, vp / amax - amax / jmax
        d = 2.0 * ramp_dist(jmax, Tj2, Ta2)
        if d < D:
            lo = vp
        else:
            hi = vp
    vp = 0.5 * (lo + hi)
    if vp <= amax * amax / jmax:
        Tj2, Ta2 = vp / amax, 0.0
    else:
        Tj2, Ta2 = amax / jmax, vp / amax - amax / jmax
    return 4.0 * Tj2 + 2.0 * Ta2


def seg_time(d3_deg: np.ndarray, limits: dict) -> float:
    """三轴位移向量(度)的段最短时间 = 三轴最大。"""
    return max(s_curve_time(abs(float(d3_deg[i])),
                            limits["vmax_deg_s"][i],
                            limits["amax_deg_s2"][i],
                            limits["jmax_deg_s3"][i]) for i in range(3))


def cap_for_interval(interval: float, limits: dict, margin: float) -> float:
    """给定段间隔, 求"三轴同位移"下 S 曲线时间 = interval*margin 的位移上限(度)。"""
    target = interval * margin
    lo, hi = 0.0, 200.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if seg_time(np.full(3, mid), limits) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_limits(config_path: str | Path | None) -> dict:
    """从执行层配置文件读取安全限速(失败则用默认值)。"""
    limits = {k: list(v) for k, v in DEFAULT_LIMITS.items()}
    if config_path and Path(config_path).exists():
        try:
            for line in Path(config_path).read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                k, v = [s.strip() for s in line.split("=", 1)]
                if k in ("safety.max_velocity_deg_s", "safety.max_acceleration_deg_s2",
                         "safety.max_jerk_deg_s3"):
                    vals = [float(x) for x in v.split(",") if x.strip()]
                    if k == "safety.max_velocity_deg_s" and len(vals) == 3:
                        limits["vmax_deg_s"] = vals
                    elif k == "safety.max_acceleration_deg_s2" and len(vals) == 3:
                        limits["amax_deg_s2"] = vals
                    elif k == "safety.max_jerk_deg_s3" and len(vals) == 3:
                        limits["jmax_deg_s3"] = vals
        except Exception as e:  # pragma: no cover
            print(f"[keyframe] 警告: 读取限速配置失败({e}), 用默认值")
    return limits


# --------------------------------------------------------------------------- #
# 关键点化主流程
# --------------------------------------------------------------------------- #
def keyframe(traj_30: np.ndarray, states_30: np.ndarray,
             neutral_unified: np.ndarray, limits: dict,
             interval_target: float = 0.9, min_interval: float = 0.5,
             margin: float = 0.85, smooth_sigma: float = 3.0,
             fps_in: float = 30.0,
             max_scale: dict | None = None,
             motor_gain: float = 6.0, motor_over: float = 2.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """30fps 轨迹 → 关键点轨迹(统一坐标, 弧度)。

    max_scale: 每状态幅度缩放上限 {speaking: 2.0, listening: 1.0, silent: 1.0}。
    缩放语义: 把段内最大位移映射到“S 曲线时间 = 间隔*margin”对应的容量——
    模型幅度不足时放大(方向不变), 超过时削峰(安全)。
    motor_gain: 电机角/RPY 位移放大系数(执行层 S 曲线约束在电机角空间,
    IK 放大 ~6×; 容量按 RPY/motor_gain 折算, 保证重定时比例 ≈ 1)。
    返回 (rpy_kp [N,3] 弧度, states_kp [N] 整数, meta)。
    """
    traj = np.asarray(traj_30, dtype=np.float64)
    states = np.asarray(states_30, dtype=np.int64)
    neutral = np.asarray(neutral_unified, dtype=np.float64)
    assert traj.ndim == 2 and traj.shape[1] == 3 and len(traj) == len(states)

    # 1. 高斯平滑(零相位, 线性算子: silent 直线段保持直线)
    smooth = gaussian_filter1d(traj, sigma=smooth_sigma, axis=0, mode="nearest")

    # 2. 按 states 分段
    bounds = [0]
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            bounds.append(i)
    bounds.append(len(states))

    kp_list: list[np.ndarray] = []       # 各段关键点(统一坐标, 度)
    st_list: list[int] = []
    seg_meta: list[dict] = []
    interval = interval_target

    for si in range(len(bounds) - 1):
        a, b = bounds[si], bounds[si + 1]
        st = int(states[a])
        name = STATE_NAME[st]
        if st == 0:  # silent: 直线回中(2 点; 执行层 S 曲线自然决定时长, 不细分)
            start = smooth[a] * RAD2DEG if si > 0 else neutral * RAD2DEG
            end = neutral * RAD2DEG
            d_home = float(np.abs(end - start).max())
            pts = np.array([start, end])                   # [2, 3] 度
            pts[-1] = end                                  # 锚定 neutral
            kp_list.append(pts)
            st_list += [0] * 2
            seg_meta.append({"state": "silent", "points": 2,
                             "duration_sec": round(d_home / 3.0, 3),
                             "home_displacement_deg": round(d_home, 3)})
        else:  # speaking / listening: 时间锚定等间隔采样(统一网格 interval)
            n_frames = b - a
            budget = n_frames / fps_in                     # 段时长预算(名义)
            n_pts = max(2, int(budget // interval) + 1)   # floor 段数: 动作在预算内
            # 等间隔时间戳(段首=段首时刻; 末点 = 首点 + (n-1)*interval ≤ 预算末)
            t_pts = a + np.arange(n_pts) * (interval * fps_in)
            t_pts = np.clip(t_pts, a, b - 1)
            # 平滑轨迹线性插值(度)
            pts = np.stack([np.interp(t_pts, np.arange(len(smooth)), smooth[:, k])
                            for k in range(3)], axis=1) * RAD2DEG
            # 首点锚定段边界(连续)
            if si > 0:
                pts[0] = kp_list[-1][-1]
            # 3. 幅度标准化(放大或削峰): 段内(除首点)等比缩放。
            #    目标: 最大段位移 ≈ 容量 × motor_over(容量按电机角折算, 双保险;
            #    轻微超容量由“floor 段数”预算余量吸收, dry-run 实测兜底)。
            cap = cap_for_interval(interval, limits, margin) / motor_gain
            d_pts = np.abs(np.diff(pts, axis=0)).max(axis=1)     # 每段位移(度, 3轴max)
            d_max = float(np.max(d_pts)) if len(d_pts) else 0.0
            ms = 1.0 if max_scale is None else max_scale.get(name, 1.0)
            scale = 1.0
            if d_max > 1e-6:
                scale = cap / d_max * motor_over              # 目标: 最大段 → 容量×over
                scale = min(scale, ms)                        # 放大上限(演示档)
                if scale < 0.1:
                    scale = 0.1
                rel = pts - pts[0]
                pts = pts[0] + scale * rel
            t_max = max((seg_time(np.full(3, scale * d_pts[j]), limits)
                         for j in range(len(d_pts))), default=0.0) if len(d_pts) else 0.0
            kp_list.append(pts)
            st_list += [st] * n_pts
            seg_meta.append({"state": name, "points": n_pts,
                             "budget_sec": round(budget, 3),
                             "interval_sec": interval,
                             "scale": round(scale, 3),
                             "max_seg_time_sec": round(t_max, 3),
                             "max_displacement_deg": round(float(np.abs(np.diff(pts, axis=0)).max()), 3)})

    kp_deg = np.concatenate(kp_list, axis=0)
    kp_states = np.asarray(st_list, dtype=np.int64)

    # 4. 全局自检: 每段 T_s 必须 ≤ 名义帧间隔(执行层重定时比例 = 1 的条件)
    fps_out = 1.0 / interval
    nominal_dt = interval
    worst_ratio = 0.0
    for j in range(len(kp_deg) - 1):
        t_s = seg_time(kp_deg[j + 1] - kp_deg[j], limits)
        worst_ratio = max(worst_ratio, t_s / nominal_dt)
    ok = worst_ratio <= margin + 1e-9
    # 若 silent 细分段的位移仍超(理论上不会: 细分保证), 检查后警告

    # 5. Silent 终点锚定(统一坐标下精确 = neutral)
    kp_deg[-1] = neutral * RAD2DEG

    meta = {
        "interval_sec": interval,
        "fps": fps_out,
        "smooth_sigma_frames": smooth_sigma,
        "margin": margin,
        "worst_ts_interval_ratio": round(worst_ratio, 3),
        "within_budget": bool(ok),
        "limits": limits,
        "segments": seg_meta,
    }
    return kp_deg * DEG2RAD, kp_states, meta


def export_v1(kp_rpy: np.ndarray, kp_states: np.ndarray, meta: dict,
              robot_actual_initial, robot_neutral_pose) -> dict:
    """关键点 → 自描述 V1 JSON 文档。"""
    return {
        "format_version": "1.0",
        "trajectory": {
            "fps": meta["fps"],
            "units": "radian",
            "rpy_order": "roll,pitch,yaw",
            "rotation_convention": "R = Ry(yaw) @ Rx(pitch) @ Rz(roll)",
            "reference": "keyframes (smoothed+anchored) relative to robot_actual_initial; "
                         "executor fills S-curve motion between keyframes",
            "n_frames": int(len(kp_rpy)),
            "robot_actual_initial": [float(x) for x in robot_actual_initial],
            "robot_neutral_pose": [float(x) for x in robot_neutral_pose],
            "states": [int(x) for x in kp_states],
            "states_legend": {"0": "silent", "1": "speaking", "2": "listening"},
            "rpy": [[float(x) for x in row] for row in kp_rpy],
        },
        "meta": {
            "generated_by": "tools/live/keyframe.py",
            "max_frame_rate_deg_per_s": round(
                float(np.abs(np.diff(kp_rpy, axis=0)).max(axis=1).max()) * RAD2DEG * meta["fps"], 3)
            if len(kp_rpy) > 1 else 0.0,
            **{k: v for k, v in meta.items()},
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="轨迹关键点化(L1 演示后处理)")
    ap.add_argument("--traj", required=True, help="30fps unified_relative_rpy.npy")
    ap.add_argument("--states", required=True, help="states.npy")
    ap.add_argument("--neutral", type=str, default="0,0,0",
                    help="neutral 统一坐标(度, roll,pitch,yaw)")
    ap.add_argument("--out", required=True, help="输出 V1 JSON 路径")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="关键点网格间隔(秒), 默认 0.5(收尾误差 ≤ interval)")
    ap.add_argument("--min-interval", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.85)
    ap.add_argument("--smooth-sigma", type=float, default=3.0)
    ap.add_argument("--max-scale-speaking", type=float, default=2.0,
                    help="speaking 段幅度放大上限(1.0=只削峰不放大)")
    ap.add_argument("--max-scale-listening", type=float, default=1.0,
                    help="listening 段幅度放大上限(1.0=只削峰不放大)")
    ap.add_argument("--config", default=None,
                    help="执行层 neck_trajectory_config.txt 路径(读取限速)")
    args = ap.parse_args()

    traj = np.load(args.traj)
    states = np.load(args.states)
    neutral = np.asarray([float(x) for x in args.neutral.split(",")])
    limits = load_limits(args.config)
    print(f"[keyframe] 限速: v={limits['vmax_deg_s']} a={limits['amax_deg_s2']} "
          f"j={limits['jmax_deg_s3']}")
    kp, st, meta = keyframe(traj, states, neutral, limits,
                            interval_target=args.interval,
                            min_interval=args.min_interval,
                            margin=args.margin, smooth_sigma=args.smooth_sigma,
                            max_scale={"speaking": args.max_scale_speaking,
                                       "listening": args.max_scale_listening})
    doc = export_v1(kp, st, meta, [0.0, 0.0, 0.0], neutral * DEG2RAD)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[keyframe] 关键点 {len(kp)} 帧, fps={meta['fps']:.3f}, "
          f"间隔 {meta['interval_sec']}s, 最差 T_s/间隔 = {meta['worst_ts_interval_ratio']} "
          f"({'OK' if meta['within_budget'] else '超预算!'})")
    for s in meta["segments"]:
        print(f"  {s}")
    print(f"[keyframe] 已写: {args.out}")


if __name__ == "__main__":
    main()
