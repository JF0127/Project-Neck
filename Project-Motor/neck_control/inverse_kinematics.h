// RPY → 关节空间逆运动学适配层。
// 复用现有线性标定模型（neck_kinematics.h / neck_config.txt）：
//   电机1+2 → pitch/roll 耦合（2x2 矩阵 C），电机3 → yaw（k3）。
// 本层只做单位与错误码适配，不重复实现数学。
#pragma once

#include <string>

#include "command_types.h"
#include "config.h"
#include "neck_config.h"

namespace neck_control {

// 头部姿态（度，硬件坐标）→ 三个电机输出轴角（度）。
// 返回 NeckError，out 仅在 OK 时有效。
NeckError neckInverseSolve(const RpyDeg& rpyDeg, const NeckConfig& ik,
                           MotorAngles& out, std::string& message);

// 正解：电机角（度）→ 头部姿态（度）。用于处理后验证与跟踪误差的 RPY 域检查。
RpyDeg neckForwardSolve(const MotorAngles& m, const NeckConfig& ik);

// 硬件坐标 RPY 是否在安全范围内（度）。
bool rpyWithinLimits(const RpyDeg& rpyDeg, const double rpyLimits[6]);

// 电机角是否在安全范围内（度）。
bool jointsWithinLimits(const MotorAngles& m, const double jointLimits[6]);

// 头部 RPY 超限明细（用于日志）。
std::string rpyLimitReport(const RpyDeg& rpyDeg, const double rpyLimits[6]);

} // namespace neck_control
