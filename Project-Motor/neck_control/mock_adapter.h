// Mock 执行器：模拟电机位置伺服 + 反馈。
// 动力学简单（速度受限的一阶跟踪），带噪声；支持测试注入：
//   冻结反馈 / 写入失败 / 动力学滞后。
#pragma once

#include <array>
#include <atomic>
#include <mutex>
#include <string>

#include "adapter_interface.h"
#include "command_types.h"

namespace neck_control {

class MockAdapter : public NeckHardwareAdapter {
public:
    // initialJointDeg: 初始电机角（度）
    explicit MockAdapter(const MotorAngles& initialJointDeg);

    std::string name() const override { return "mock"; }

    bool write(const MotorAngles& target, const double spdDegS[3]) override;
    NeckFeedback read() override;
    bool emergencyStop() override;
    void release() override;
    bool healthy() const override { return true; }

    // ---- 测试注入 ----
    void setVelocityLimit(double vmaxDegS[3]);
    void setNoiseDeg(double noise);
    void freezeFeedback(bool freeze);       // read() 返回 valid=false
    void setWriteFail(bool fail);           // write() 恒失败
    void setDynamicsLag(double lagDegPerS); // 附加固定滞后误差（度），模拟跟踪能力下降

    // 当前模拟位置（度）
    MotorAngles actual() const;
    // 急停（制动）是否激活
    bool estopActive() const;

private:
    mutable std::mutex mtx_;
    MotorAngles pos_;
    double vmax_[3] = {30, 30, 40};
    double noise_ = 0.02;
    double lag_ = 0.0; // 固定滞后（度）
    bool frozen_ = false;
    bool writeFail_ = false;
    bool estop_ = false;
    std::atomic<uint64_t> clockMs_{1000};
};

} // namespace neck_control
