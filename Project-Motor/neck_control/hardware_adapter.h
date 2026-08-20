// 实机 EtherCAT 适配器（声明）。实现位于 hardware_adapter.cpp，
// 仅该文件依赖 SOEM/EtherCAT，因此 dry-run 与 mock 构建无需 root/网卡。
#pragma once

#include <string>

#include "adapter_interface.h"
#include "command_types.h"
#include "neck_config.h"

namespace neck_control {

class EthercatNeckAdapter : public NeckHardwareAdapter {
public:
    // slave: 从站索引（0 起）；ik: 电机通道/ID 等下发参数
    EthercatNeckAdapter(int slave, const NeckConfig& ik);

    std::string name() const override { return "ethercat"; }

    bool write(const MotorAngles& target, const double spdParam[3]) override;
    NeckFeedback read() override;
    bool emergencyStop() override;
    void release() override;
    bool healthy() const override;

private:
    int slave_;
    NeckConfig ik_;
};

} // namespace neck_control
