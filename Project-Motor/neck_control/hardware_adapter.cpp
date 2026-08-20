// 实机 EtherCAT 适配器实现。仅本文件依赖 SOEM/电机协议。
// 位置指令通过 NeckFramePublish 持续发布（通信线程 ~1kHz 周期重发）；
// 反馈通过 transmit.cpp 的 NeckFeedbackGet 取互斥保护快照。
#include "hardware_adapter.h"

#include "config.h"
#include "queue.h"

extern "C" {
#include "motor_control.h"
#include "transmit.h"
}

namespace neck_control {

EthercatNeckAdapter::EthercatNeckAdapter(int slave, const NeckConfig& ik)
    : slave_(slave), ik_(ik) {}

bool EthercatNeckAdapter::write(const MotorAngles& target, const double spdParam[3]) {
    EtherCAT_Msg frame{};
    auto clampSpd = [](double v) -> uint16_t {
        if (v < 1.0) v = 1.0;
        if (v > 18000.0) v = 18000.0;
        return (uint16_t)(v + 0.5);
    };
    set_motor_position(&frame, ik_.passage1, ik_.id1, (float)target.motor1,
                       clampSpd(spdParam[0]), ik_.current, ik_.ack_status);
    set_motor_position(&frame, ik_.passage2, ik_.id2, (float)target.motor2,
                       clampSpd(spdParam[1]), ik_.current, ik_.ack_status);
    set_motor_position(&frame, ik_.passage3, ik_.id3, (float)target.motor3,
                       clampSpd(spdParam[2]), ik_.current, ik_.ack_status);
    return NeckFramePublish(slave_, &frame);
}

NeckFeedback EthercatNeckAdapter::read() {
    NeckFeedback fb;
    uint64_t ts = 0;
    double deg[3] = {0, 0, 0};
    uint8_t err[3] = {0, 0, 0};
    double temp[3] = {0, 0, 0};
    if (!NeckFeedbackGet(slave_, deg, err, temp, &ts)) return fb;
    fb.valid = true;
    fb.joint_deg[0] = deg[0];
    fb.joint_deg[1] = deg[1];
    fb.joint_deg[2] = deg[2];
    fb.error[0] = err[0];
    fb.error[1] = err[1];
    fb.error[2] = err[2];
    fb.temperature[0] = temp[0];
    fb.temperature[1] = temp[1];
    fb.temperature[2] = temp[2];
    fb.timestamp_ms = ts;
    return fb;
}

bool EthercatNeckAdapter::emergencyStop() {
    // 全制动并持续发布，直到 release()
    EtherCAT_Msg frame{};
    set_motor_cur_tor(&frame, ik_.passage1, ik_.id1, 10, 2, 0); // 变量阻尼制动
    set_motor_cur_tor(&frame, ik_.passage2, ik_.id2, 10, 2, 0);
    set_motor_cur_tor(&frame, ik_.passage3, ik_.id3, 10, 2, 0);
    return NeckFramePublish(slave_, &frame);
}

void EthercatNeckAdapter::release() { NeckFrameStop(slave_); }

bool EthercatNeckAdapter::healthy() const {
    return running && ec_slavecount > slave_ && slave_ >= 0;
}

} // namespace neck_control
