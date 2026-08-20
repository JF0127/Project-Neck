// 安全轨迹重定时器：jerk-limited S 曲线（7 段式）时间参数化。
// 输入 30FPS 关节空间路径，输出满足 速度/加速度/jerk 上限的连续采样轨迹。
// 每段从静止到静止，段间速度连续；时长取三关节最大值，其余关节驻留。
#pragma once

#include <array>
#include <string>
#include <vector>

#include "command_types.h"

namespace neck_control {

struct RetimerLimits {
    double vmax[3] = {0, 0, 0};   // 度/秒
    double amax[3] = {0, 0, 0};   // 度/秒²
    double jmax[3] = {0, 0, 0};   // 度/秒³
};

// 单关节 S 曲线规划结果（采样序列）。
struct SCurveSegment {
    double duration = 0.0;
    std::vector<double> t;  // 采样时刻（秒，0..duration）
    std::vector<double> q;  // 位置（度）
    std::vector<double> v;  // 速度（度/秒）
};

class TrajectoryRetimer {
public:
    TrajectoryRetimer(const RetimerLimits& limits, double dt);

    // 帧序列（等间隔 1/fps 秒）→ 全局重定时采样。
    // nominal_dt > 0 时，每段时长取下限 max(S曲线时长, nominal_dt)，保留模型节奏（不过快）。
    // 成功返回 true；samples 与 vel 与 times 等长；segment_durations 为每段时长（含首段）。
    bool retime(const std::vector<MotorAngles>& frames, double fps,
                double nominal_dt,
                std::vector<double>& times,
                std::vector<MotorAngles>& samples,
                std::vector<std::array<double, 3>>& vel,
                std::vector<double>& segment_durations,
                std::string& message) const;

    // 单段 rest-to-rest S 曲线（供单元测试直接调用）。
    static bool planSegment(double q0, double q1,
                            double vmax, double amax, double jmax,
                            double dt, SCurveSegment& out, std::string& message);

    double dt() const { return dt_; }

private:
    RetimerLimits limits_;
    double dt_;
};

} // namespace neck_control
