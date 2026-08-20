#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neck 线性标定工具（离线，手动测量）。

原理（单电机激励法，全部用「度」）：
    头部度 = C · 电机度
        p = p0 + c11·Δm1 + c12·Δm2
        r = r0 + c21·Δm1 + c22·Δm2
        y = y0 + k3·Δm3
    - 只动电机1：c11 = Δp/Δm1, c21 = Δr/Δm1
    - 只动电机2：c12 = Δp/Δm2, c22 = Δr/Δm2
    - 只动电机3：k3  = Δy/Δm3
    正、反方向各测一次取平均以减小回差。

采集方法：
    用现有 CLI：MotorPositionSet 转单个电机到某角度，MotorAngleGet 读回电机
    输出轴角度(度)，头部角用量角器/手机 APP 测量(度)。

用法：
    python3 tools/neck_calibrate.py            # 交互式采集并写出 neck_config.txt
    python3 tools/neck_calibrate.py --out xxx  # 指定输出文件

脚本只重写标定相关键(c*/k3/m*0)，其余键(限位/下发参数)若已有文件则保留。
"""

import argparse
import os
import sys

TEMPLATE_KEYS_ORDER = [
    "c11", "c12", "c21", "c22", "k3",
    "m10", "m20", "m30", "p0", "r0", "y0",
    "motor1_min", "motor1_max", "motor2_min", "motor2_max",
    "motor3_min", "motor3_max",
    "pitch_min", "pitch_max", "roll_min", "roll_max", "yaw_min", "yaw_max",
    "det_eps",
    "passage1", "passage2", "passage3", "id1", "id2", "id3",
    "speed", "current", "ack_status",
]

DEFAULTS = {
    "c11": 0.5, "c12": -0.5, "c21": 0.5, "c22": 0.5, "k3": 1.0,
    "m10": -84.0, "m20": 113.0, "m30": 177.0, "p0": 0.0, "r0": 0.0, "y0": 0.0,
    "motor1_min": -84.0, "motor1_max": 64.0,
    "motor2_min": -34.0, "motor2_max": 113.0,
    "motor3_min": 60.0, "motor3_max": 235.0,
    "pitch_min": -40.0, "pitch_max": 25.0,
    "roll_min": -30.0, "roll_max": 30.0,
    "yaw_min": -117.0, "yaw_max": 58.0,
    "det_eps": 1e-6,
    "passage1": 1, "passage2": 2, "passage3": 3,
    "id1": 1, "id2": 2, "id3": 3,
    "speed": 50, "current": 500, "ack_status": 2,
}


def load_existing(path):
    cfg = dict(DEFAULTS)
    if not os.path.exists(path):
        return cfg
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            v = v.split("#", 1)[0].strip()
            if k in cfg:
                try:
                    cfg[k] = float(v) if "." in v or "e" in v.lower() else int(v)
                except ValueError:
                    pass
    return cfg


def ask(prompt):
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  请输入数字。")
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。")
            sys.exit(1)


def calibrate_axis(name):
    """采集单电机正反两次激励，返回 (avg_dhead_a, avg_dhead_b, avg_dmotor)。

    对某电机：记录中位电机角 m0；正向转到 m+ 记录头部 (a+, b+)；
    反向转到 m- 记录头部 (a-, b-)。用差分求比值并取平均。
    a/b 是该电机主要影响的两个头部分量。
    """
    print(f"\n=== 标定{name} ===")
    m0 = ask(f"  中位时{name}的电机输出轴角(度): ")
    print("  正向：用 MotorPositionSet 把该电机转到一个正向角度后：")
    mp = ask(f"    转后{name}电机角(度): ")
    ap = ask("    此时头部分量A(度): ")
    bp = ask("    此时头部分量B(度): ")
    print("  反向：把该电机转到中位另一侧后：")
    mn = ask(f"    转后{name}电机角(度): ")
    an = ask("    此时头部分量A(度): ")
    bn = ask("    此时头部分量B(度): ")
    dmp, dmn = mp - m0, mn - m0
    if abs(dmp) < 1e-9 or abs(dmn) < 1e-9:
        print("  警告：电机位移过小，结果可能不可靠。")
    ka = 0.5 * (safe_div(ap, dmp) + safe_div(an, dmn))
    kb = 0.5 * (safe_div(bp, dmp) + safe_div(bn, dmn))
    return m0, ka, kb


def safe_div(num, den):
    return num / den if abs(den) > 1e-9 else 0.0


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=os.path.join(here, "neck_config.txt"))
    args = ap.parse_args()

    cfg = load_existing(args.out)
    print("Neck 线性标定（单位全部为度）。头部分量 A=pitch, B=roll。")
    print("先把机构摆到头部正中(pitch=roll=yaw=0)，再逐轴采集。\n")

    # 电机1：A=pitch, B=roll
    m10, c11, c21 = calibrate_axis("电机1")
    # 电机2：A=pitch, B=roll
    m20, c12, c22 = calibrate_axis("电机2")
    # 电机3：A=yaw（B 忽略）
    print("\n=== 标定电机3（yaw）===")
    m30 = ask("  中位时电机3的电机输出轴角(度): ")
    m3p = ask("  正向转后电机3角(度): ")
    yp = ask("  此时头部 yaw(度): ")
    m3n = ask("  反向转后电机3角(度): ")
    yn = ask("  此时头部 yaw(度): ")
    k3 = 0.5 * (safe_div(yp, m3p - m30) + safe_div(yn, m3n - m30))

    cfg.update({
        "c11": c11, "c12": c12, "c21": c21, "c22": c22, "k3": k3,
        "m10": m10, "m20": m20, "m30": m30,
    })

    det = c11 * c22 - c12 * c21
    print("\n标定结果：")
    print(f"  C = [[{c11:.5f}, {c12:.5f}], [{c21:.5f}, {c22:.5f}]]  det={det:.5f}")
    print(f"  k3 = {k3:.5f}")
    print(f"  m0 = ({m10:.3f}, {m20:.3f}, {m30:.3f})")
    if abs(det) < 1e-4:
        print("  警告：det(C) 接近 0，逆解会奇异，请检查激励是否过于共线。")
    if abs(k3) < 1e-4:
        print("  警告：k3 接近 0，请检查电机3 激励。")

    write_config(args.out, cfg)
    print(f"\n已写出 {args.out}")


def write_config(path, cfg):
    def fmt(k):
        v = cfg[k]
        if k in ("passage1", "passage2", "passage3", "id1", "id2", "id3",
                 "speed", "current", "ack_status"):
            return str(int(v))
        return repr(float(v))
    lines = ["# Neck 建模参数配置（由 neck_calibrate.py 生成）。单位：度。", ""]
    for k in TEMPLATE_KEYS_ORDER:
        lines.append(f"{k} = {fmt(k)}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
