// 输入校验：拒绝 NaN/Inf、形状错误、fps 越界、约定不匹配、过期/重复/乱序命令、
// Silent 终点不在中位等。所有拒绝原因记录日志并返回明确错误码。
#pragma once

#include <cstdint>
#include <string>

#include "command_types.h"
#include "config.h"

namespace neck_control {

struct ValidationContext {
    const NeckControlConfig& cfg;
    // 防重放：最近一次接受的命令
    uint64_t last_command_id = 0;
    double last_timestamp = 0.0;
    bool require_fresh_timestamp = false; // 实机模式强制
    double now_s = 0.0;
};

NeckError validateTrajectoryCommand(const NeckTrajectoryCommand& cmd,
                                    ValidationContext& ctx,
                                    std::string& message);

} // namespace neck_control
