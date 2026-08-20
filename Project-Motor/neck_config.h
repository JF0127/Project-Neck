#pragma once

#include <cstdint>
#include <string>

// Neck 建模参数集合。
// 单位约定（全链路统一）：
//   - 头部姿态 pitch/roll/yaw：度
//   - 电机 m1/m2/m3：减速器输出轴角度，度（直接匹配 set_motor_position）
//   - C(c11..c22)、k3：无量纲比值（头部度 / 电机度）
//
// 线性标定模型：
//   p = p0 + c11·Δm1 + c12·Δm2
//   r = r0 + c21·Δm1 + c22·Δm2
//   y = y0 + k3·Δm3
//   其中 Δmi = mi - mi0
struct NeckConfig {
    // 一、二号电机到 pitch/roll 的 2x2 耦合传动比
    double c11, c12, c21, c22;
    // 三号电机到 yaw 的传动比
    double k3;

    // 头部正中(pitch=roll=yaw=0)时的电机输出轴角度（度），标定记录
    double m10, m20, m30;
    // 头部姿态中位残余偏置（度），一般为 0
    double p0, r0, y0;

    // 电机输出轴安全角度范围（度）
    double motor1_min, motor1_max;
    double motor2_min, motor2_max;
    double motor3_min, motor3_max;

    // 头部姿态输入安全范围（度）
    double pitch_min, pitch_max;
    double roll_min, roll_max;
    double yaw_min, yaw_max;

    // 耦合矩阵奇异判定阈值 |det(C)|
    double det_eps;

    // 下发参数
    uint8_t passage1, passage2, passage3;
    uint16_t id1, id2, id3;
    uint16_t speed, current;
    uint8_t ack_status;
};

// 返回带内置默认值的配置（限位取自原 neck_kinematics.h，C/k3/m0 为占位待标定值）。
NeckConfig defaultNeckConfig();

// 从 key=value 文本文件加载配置，缺项保留默认值。
// 文件语法：每行 `key = value`，`#` 起始为注释，空行忽略。
// 返回 true 表示文件成功打开并解析（未识别的键会打印告警但不算失败）。
bool loadNeckConfig(const std::string& path, NeckConfig& out);

// 将配置写回 key=value 文本文件（供标定脚本/命令使用）。
bool saveNeckConfig(const std::string& path, const NeckConfig& cfg);
