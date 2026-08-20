#pragma once

// Neck 头部姿态逆解 —— 纯数学，度制，无 EtherCAT 依赖，便于单元测试。
// 数学模型与单位约定见 neck_model.md 与 neck_config.h。

#include "neck_config.h"

struct NeckAngles { double pitch, roll, yaw; };   // 头部姿态，单位：度
struct MotorAngles { double motor1, motor2, motor3; }; // 电机输出轴角，单位：度

// 本地绝对值，避免引入 <cmath>（其与 math_ops.h 里对 fmaxf/fminf 的重声明冲突）。
inline double neckAbs(double x) { return x < 0.0 ? -x : x; }

enum NeckStatus {
    NECK_OK = 0,            // 可达，out 有效
    NECK_SINGULAR,          // 耦合矩阵接近奇异 (|det(C)| < eps) 或 k3≈0
    NECK_POSE_OUT_OF_RANGE, // 输入头部姿态超出安全范围
    NECK_MOTOR_OUT_OF_RANGE // 逆解出的电机角超出安全范围
};

// 逆解：头部姿态 → 三个电机目标输出轴角（度，绝对角）。
// 不做任何范围检查，调用方应先用 solve() 校验。det(C) 为零时结果为 inf/nan。
inline MotorAngles neckInverse(NeckAngles pose, const NeckConfig& c) {
    double dp = pose.pitch - c.p0;
    double dr = pose.roll  - c.r0;
    double dy = pose.yaw   - c.y0;
    double det = c.c11 * c.c22 - c.c12 * c.c21;
    return {
        c.m10 + ( c.c22 * dp - c.c12 * dr) / det,
        c.m20 + (-c.c21 * dp + c.c11 * dr) / det,
        c.m30 + dy / c.k3
    };
}

// 正解：三个电机输出轴角 → 头部姿态（度）。逆解的逆运算，供标定校核。
inline NeckAngles neckForward(MotorAngles m, const NeckConfig& c) {
    double d1 = m.motor1 - c.m10;
    double d2 = m.motor2 - c.m20;
    double d3 = m.motor3 - c.m30;
    return {
        c.p0 + c.c11 * d1 + c.c12 * d2,
        c.r0 + c.c21 * d1 + c.c22 * d2,
        c.y0 + c.k3 * d3
    };
}

// 带安全校验的逆解。全部通过才返回 NECK_OK 并写入 out。
// 校验顺序：矩阵奇异 → 输入姿态限位 → 逆解 → 电机限位。
inline NeckStatus neckSolve(NeckAngles pose, const NeckConfig& c, MotorAngles* out) {
    double det = c.c11 * c.c22 - c.c12 * c.c21;
    if (neckAbs(det) < c.det_eps || neckAbs(c.k3) < c.det_eps)
        return NECK_SINGULAR;

    if (pose.pitch < c.pitch_min || pose.pitch > c.pitch_max ||
        pose.roll  < c.roll_min  || pose.roll  > c.roll_max  ||
        pose.yaw   < c.yaw_min   || pose.yaw   > c.yaw_max)
        return NECK_POSE_OUT_OF_RANGE;

    MotorAngles m = neckInverse(pose, c);
    if (m.motor1 < c.motor1_min || m.motor1 > c.motor1_max ||
        m.motor2 < c.motor2_min || m.motor2 > c.motor2_max ||
        m.motor3 < c.motor3_min || m.motor3 > c.motor3_max)
        return NECK_MOTOR_OUT_OF_RANGE;

    if (out) *out = m;
    return NECK_OK;
}

inline const char* neckStatusString(NeckStatus s) {
    switch (s) {
        case NECK_OK:                return "OK";
        case NECK_SINGULAR:          return "耦合矩阵奇异 (det(C)≈0 或 k3≈0)";
        case NECK_POSE_OUT_OF_RANGE: return "头部姿态超出安全范围";
        case NECK_MOTOR_OUT_OF_RANGE:return "逆解电机角超出安全范围";
        default:                     return "未知状态";
    }
}
