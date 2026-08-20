// 硬件适配器抽象接口（纯 C++，无 EtherCAT 依赖）。
// dry-run 不创建适配器；mock 使用 MockAdapter；hardware 使用 EthercatNeckAdapter。
#pragma once

#include <cstdint>
#include <string>

#include "command_types.h"

namespace neck_control {

// 反馈快照（关节角为电机输出轴角，度）。
struct NeckFeedback {
    bool valid = false;                  // 是否拿到本轮有效反馈
    double joint_deg[3] = {0, 0, 0};     // m1/m2/m3 实际角（度）
    uint8_t error[3] = {0, 0, 0};        // 电机故障码字节（0=无）
    double temperature[3] = {0, 0, 0};   // 温度（度）
    uint64_t timestamp_ms = 0;           // 单调时钟时间戳
};

class NeckHardwareAdapter {
public:
    virtual ~NeckHardwareAdapter() = default;

    virtual std::string name() const = 0;

    // 写一帧位置指令：target 为关节角（度），spdParam 为电机速度参数
    // （0~18000 ↔ 0~1800 rpm，调用方已含安全余量）。
    // 返回 false 表示发送失败（触发指令超时监控）。
    virtual bool write(const MotorAngles& target, const double spdParam[3]) = 0;

    // 读取最新反馈。
    virtual NeckFeedback read() = 0;

    // 急停：立即进入保持/制动，直到 release() 前不得恢复。
    virtual bool emergencyStop() = 0;

    // 解除急停/停止发布（仅限显式确认后调用）。
    virtual void release() = 0;

    // 健康检查（硬件模式：EtherCAT 运行中且有从站）。
    virtual bool healthy() const = 0;
};

} // namespace neck_control
