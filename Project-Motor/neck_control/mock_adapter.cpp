#include "mock_adapter.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace neck_control {

namespace {
inline double clampAbs(double x, double lim) {
    return x > lim ? lim : (x < -lim ? -lim : x);
}
} // namespace

MockAdapter::MockAdapter(const MotorAngles& initialJointDeg) : pos_(initialJointDeg) {}

bool MockAdapter::write(const MotorAngles& target, const double spdParam[3]) {
    (void)spdParam; // mock 用自身动力学，不直接采用命令速度
    std::lock_guard<std::mutex> lock(mtx_);
    if (writeFail_) return false;
    if (estop_) return true; // 急停期间忽略写入（模拟制动保持）

    static const double dt = 0.01;
    double step = clampAbs(target.motor1 - pos_.motor1, vmax_[0] * dt);
    pos_.motor1 += step;
    step = clampAbs(target.motor2 - pos_.motor2, vmax_[1] * dt);
    pos_.motor2 += step;
    step = clampAbs(target.motor3 - pos_.motor3, vmax_[2] * dt);
    pos_.motor3 += step;
    return true;
}

NeckFeedback MockAdapter::read() {
    std::lock_guard<std::mutex> lock(mtx_);
    clockMs_ += 10;
    NeckFeedback fb;
    if (frozen_) return fb; // valid=false

    fb.valid = true;
    // 轻微周期噪声，模拟编码器量化/电气噪声
    fb.joint_deg[0] = pos_.motor1 + noise_ * std::sin(clockMs_ * 0.013);
    fb.joint_deg[1] = pos_.motor2 + noise_ * std::cos(clockMs_ * 0.011);
    fb.joint_deg[2] = pos_.motor3 + noise_ * std::sin(clockMs_ * 0.007);
    fb.error[0] = fb.error[1] = fb.error[2] = 0;
    fb.temperature[0] = fb.temperature[1] = fb.temperature[2] = 35.0;
    fb.timestamp_ms = clockMs_;
    return fb;
}

bool MockAdapter::emergencyStop() {
    std::lock_guard<std::mutex> lock(mtx_);
    estop_ = true;
    return true;
}

void MockAdapter::release() {
    std::lock_guard<std::mutex> lock(mtx_);
    estop_ = false;
}

void MockAdapter::setVelocityLimit(double vmaxDegS[3]) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int i = 0; i < 3; ++i) vmax_[i] = vmaxDegS[i];
}

void MockAdapter::setNoiseDeg(double noise) {
    std::lock_guard<std::mutex> lock(mtx_);
    noise_ = noise;
}

void MockAdapter::freezeFeedback(bool freeze) {
    std::lock_guard<std::mutex> lock(mtx_);
    frozen_ = freeze;
}

void MockAdapter::setWriteFail(bool fail) {
    std::lock_guard<std::mutex> lock(mtx_);
    writeFail_ = fail;
}

// 设置 mock 执行器最大跟随速度（度/秒）；数值越小模拟"跟不上命令"越严重。
void MockAdapter::setDynamicsLag(double maxFollowDegPerS) {
    std::lock_guard<std::mutex> lock(mtx_);
    for (int i = 0; i < 3; ++i) vmax_[i] = maxFollowDegPerS;
}

MotorAngles MockAdapter::actual() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pos_;
}

bool MockAdapter::estopActive() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return estop_;
}

} // namespace neck_control
