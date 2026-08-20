// 上游 JSON ↔ NeckTrajectoryCommand 转换 + 审计文件输出。
#pragma once

#include <string>

#include "command_types.h"

namespace neck_control {

// 解析上游 JSON 为 NeckTrajectoryCommand。
// 失败时返回错误码并填写 message（含具体字段信息）。
// 自动识别模型输出 V1 格式（顶层 format_version=1.0 / trajectory 对象包装），转 V1 映射。
NeckError loadNeckTrajectoryJson(const std::string& jsonText,
                                 NeckTrajectoryCommand& out,
                                 std::string& message);

// 模型输出 V1 格式（format_version 1.0）显式映射。
// 映射规则（全部来自文件自身字段，不做猜测）：
//   1) 顶层 trajectory 对象承载 fps/units/rpy_order/rotation_convention/n_frames/
//      robot_actual_initial/robot_neutral_pose/states(+states_legend)/rpy；
//   2) states 为整数编码（states_legend 定义 0=silent/1=speaking/2=listening 等）；
//   3) reference 声明 rpy 为“统一参考系下相对 robot_actual_initial 的旋转”（帧 0=[0,0,0]），
//      绝对姿态 R_abs[i] = R(robot_actual_initial) @ R(rpy[i])，
//      与交付校验 R(ai)@R(rpy[-1])==R(neutral_pose) 一致（前乘=固定基座组合）；
//   4) 映射后仍走标准校验（Silent 终点=robot_neutral_pose、起点偏差等）。
NeckError loadNeckTrajectoryJsonV1(const std::string& jsonText,
                                   NeckTrajectoryCommand& out,
                                   std::string& message);

// 读取文件内容（UTF-8/ASCII）。
bool readTextFile(const std::string& path, std::string& out);

// 审计输出：把原始命令与处理结果写为 JSON 文件（供对比审计）。
// dir 必须存在；写入 dir/original_trajectory.json、dir/processed_trajectory.json。
struct AuditSamples {
    std::vector<std::array<double, 3>> rpy_deg;     // 每帧硬件坐标 RPY（度，[roll,pitch,yaw]）
    std::vector<std::array<double, 3>> joints_deg;  // 每帧电机角（度）
    // 重定时后的采样（按控制周期）
    std::vector<double> sample_t;                   // 秒
    std::vector<std::array<double, 3>> sample_joints_deg;
    std::vector<std::array<double, 3>> sample_vel_deg_s;
    std::vector<std::string> sample_states;
    double fps = 30.0;
    double loop_rate_hz = 100.0;
};

bool dumpAuditJson(const std::string& dir, const NeckTrajectoryCommand& cmd,
                   const AuditSamples& audit, std::string& message);

} // namespace neck_control
