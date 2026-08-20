// 遥测与审计：控制循环时序、故障记录、原始/处理后轨迹保存。
#pragma once

#include <cstdint>
#include <string>

#include "command_types.h"

namespace neck_control {

struct LoopTelemetry {
    uint64_t ticks = 0;
    uint64_t writes_ok = 0;
    uint64_t writes_fail = 0;
    uint64_t feedback_ok = 0;
    uint64_t feedback_miss = 0;
    double last_jitter_ms = 0.0;
    double max_jitter_ms = 0.0;
    double jitter_sum_ms = 0.0;
    double planned_duration_s = 0.0;   // 计划时长
    double actual_duration_s = 0.0;    // 实际时长
    double max_tracking_error_deg = 0.0; // 执行期间最大跟踪误差（三轴最大）
    uint64_t estop_count = 0;
    uint64_t fault_count = 0;
    NeckError last_error = NeckError::OK;
    std::string last_error_message;
};

// 带时间戳的日志（项目风格：printf）。
void neckLog(const char* fmt, ...);

} // namespace neck_control
