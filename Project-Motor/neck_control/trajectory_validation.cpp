#include "trajectory_validation.h"

#include <cmath>

#include "telemetry.h"

namespace neck_control {

namespace {

// 去除空白后比较旋转约定字符串（允许大小写与空格差异，但语义必须一致）。
std::string compact(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == ' ' || c == '\t') continue;
        out += (char)std::tolower((unsigned char)c);
    }
    return out;
}

} // namespace

NeckError validateTrajectoryCommand(const NeckTrajectoryCommand& cmd,
                                    ValidationContext& ctx,
                                    std::string& message) {
    const NeckControlConfig& cfg = ctx.cfg;

    // 1. N > 0 且不超过上限
    int n = cmd.frameCount();
    if (n <= 0) {
        message = "轨迹帧数必须大于 0";
        neckLog("拒绝: %s", message.c_str());
        return NeckError::EMPTY_TRAJECTORY;
    }
    if (n > cfg.max_trajectory_frames) {
        message = "轨迹帧数 " + std::to_string(n) + " 超过上限 " +
                  std::to_string(cfg.max_trajectory_frames);
        neckLog("拒绝: %s", message.c_str());
        return NeckError::TRAJECTORY_TOO_LONG;
    }

    // 2. 数值有限性 + 形状 [N,3]（解析层已保证行内 3 个，这里再防御性检查）
    for (int i = 0; i < n; ++i) {
        const auto& p = cmd.trajectory_rpy[i];
        for (int k = 0; k < 3; ++k) {
            if (!isFinite(p[k])) {
                message = "第 " + std::to_string(i) + " 帧包含 NaN/Inf";
                neckLog("拒绝: %s", message.c_str());
                return NeckError::NON_FINITE;
            }
        }
    }

    // 3. fps 范围
    if (!(cmd.fps >= cfg.fps_min && cmd.fps <= cfg.fps_max)) {
        message = "fps=" + std::to_string(cmd.fps) + " 不在允许范围 [" +
                  std::to_string(cfg.fps_min) + ", " + std::to_string(cfg.fps_max) + "]";
        neckLog("拒绝: %s", message.c_str());
        return NeckError::FPS_OUT_OF_RANGE;
    }

    // 4. 单位/轴序/旋转约定必须匹配
    const std::string unit = compact(cmd.coordinate_convention.unit);
    if (unit != "radian" && unit != "rad") {
        message = "单位 '" + cmd.coordinate_convention.unit + "' 不匹配 (需要 radian)";
        neckLog("拒绝: %s", message.c_str());
        return NeckError::CONVENTION_MISMATCH;
    }
    {
        const auto& order = cmd.coordinate_convention.order;
        const char* expect[3] = {"roll", "pitch", "yaw"};
        bool ok = order.size() == 3;
        if (ok) {
            for (int i = 0; i < 3; ++i)
                if (compact(order[i]) != expect[i]) { ok = false; break; }
        }
        if (!ok) {
            message = "轴序不匹配 (需要 [roll, pitch, yaw])";
            neckLog("拒绝: %s", message.c_str());
            return NeckError::CONVENTION_MISMATCH;
        }
    }
    {
        const char* expect = "r=ry(yaw)@rx(pitch)@rz(roll)";
        if (compact(cmd.coordinate_convention.rotation) != expect) {
            message = "旋转约定不匹配 (需要 R = Ry(yaw) @ Rx(pitch) @ Rz(roll))";
            neckLog("拒绝: %s", message.c_str());
            return NeckError::CONVENTION_MISMATCH;
        }
    }

    // 5. 状态序列长度
    if (!cmd.states.empty() && (int)cmd.states.size() != n) {
        message = "states 长度 " + std::to_string(cmd.states.size()) + " 与轨迹帧数 " +
                  std::to_string(n) + " 不一致";
        neckLog("拒绝: %s", message.c_str());
        return NeckError::STATE_LENGTH_MISMATCH;
    }

    // 6. Silent 状态终点应为 robot_neutral_pose
    if (!cmd.states.empty()) {
        const std::string& last = cmd.states.back();
        if (last == "silent") {
            const auto& end = cmd.trajectory_rpy.back();
            const double tol = 1e-3; // rad
            for (int k = 0; k < 3; ++k) {
                if (std::fabs(end[k] - cmd.robot_neutral_pose[k]) > tol) {
                    message = "Silent 终点偏离 robot_neutral_pose: 分量 " + std::to_string(k) +
                              " 偏差 " + std::to_string(end[k] - cmd.robot_neutral_pose[k]) + " rad";
                    neckLog("拒绝: %s", message.c_str());
                    return NeckError::SILENT_NOT_AT_NEUTRAL;
                }
            }
        }
    }

    // 7. 命令防重放：重复/乱序
    if (cmd.command_id != 0) {
        if (cmd.command_id <= ctx.last_command_id) {
            message = "command_id=" + std::to_string(cmd.command_id) + " 重复或乱序 (上次 " +
                      std::to_string(ctx.last_command_id) + ")";
            neckLog("拒绝: %s", message.c_str());
            return cmd.command_id == ctx.last_command_id ? NeckError::DUPLICATE_COMMAND
                                                         : NeckError::OUT_OF_ORDER_COMMAND;
        }
    }
    if (cmd.timestamp > 0.0) {
        if (cmd.timestamp < ctx.last_timestamp) {
            message = "时间戳乱序";
            neckLog("拒绝: %s", message.c_str());
            return NeckError::OUT_OF_ORDER_COMMAND;
        }
        if (ctx.require_fresh_timestamp) {
            double skew = std::fabs(cmd.timestamp - ctx.now_s);
            if (skew > cfg.max_timestamp_skew_s) {
                message = "时间戳过期或超前 " + std::to_string(skew) + " s";
                neckLog("拒绝: %s", message.c_str());
                return NeckError::STALE_TIMESTAMP;
            }
        }
    }

    // 更新防重放状态（调用方确认接受后应调用 commit）
    return NeckError::OK;
}

} // namespace neck_control
