// 硬件执行层状态机。
// 状态：DISABLED / READY / EXECUTING / STOPPING / FAULT / ESTOP / CALIBRATION。
// 规则：
//   - FAULT 与 ESTOP 不能由普通轨迹命令解除，只能显式 acknowledge；
//   - 急停可从任何状态触发；
//   - 所有迁移记录日志。
#pragma once

#include <string>

namespace neck_control {

enum class ExecutorState {
    DISABLED,
    READY,
    EXECUTING,
    STOPPING,
    FAULT,
    ESTOP,
    CALIBRATION
};

inline const char* executorStateString(ExecutorState s) {
    switch (s) {
        case ExecutorState::DISABLED:    return "DISABLED";
        case ExecutorState::READY:       return "READY";
        case ExecutorState::EXECUTING:   return "EXECUTING";
        case ExecutorState::STOPPING:    return "STOPPING";
        case ExecutorState::FAULT:       return "FAULT";
        case ExecutorState::ESTOP:       return "ESTOP";
        case ExecutorState::CALIBRATION: return "CALIBRATION";
    }
    return "?";
}

class StateMachine {
public:
    ExecutorState state() const { return state_; }
    bool isActive() const { return state_ == ExecutorState::EXECUTING || state_ == ExecutorState::STOPPING; }
    const std::string& lastReason() const { return lastReason_; }

    // 迁移。返回 false 表示非法迁移（拒绝并记录日志）。
    // allowFrom 为 nullptr 表示任意来源。
    bool transition(ExecutorState to, const char* reason);

    // 急停：任何状态（含 DISABLED）→ ESTOP。
    bool estop(const char* reason);

    // 恢复确认：ESTOP/FAULT → READY（调用方需先确认安全）。
    bool acknowledge(const char* reason);

private:
    ExecutorState state_ = ExecutorState::DISABLED;
    std::string lastReason_;
    void logTransition(ExecutorState from, ExecutorState to, const char* reason);
};

} // namespace neck_control
