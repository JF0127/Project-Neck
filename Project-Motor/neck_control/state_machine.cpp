#include "state_machine.h"

#include <cstdio>

namespace neck_control {

void StateMachine::logTransition(ExecutorState from, ExecutorState to, const char* reason) {
    printf("[NeckState] %s -> %s (%s)\n",
           executorStateString(from), executorStateString(to),
           reason ? reason : "无原因");
}

bool StateMachine::transition(ExecutorState to, const char* reason) {
    ExecutorState from = state_;
    bool ok = false;
    switch (to) {
        case ExecutorState::READY:
            ok = (from == ExecutorState::DISABLED || from == ExecutorState::STOPPING ||
                  from == ExecutorState::EXECUTING || // EXECUTING→READY: 轨迹正常完成
                  from == ExecutorState::CALIBRATION); // 标定运动完成/退出标定模式
            break;
        case ExecutorState::EXECUTING:
            ok = (from == ExecutorState::READY);
            break;
        case ExecutorState::STOPPING:
            ok = (from == ExecutorState::EXECUTING);
            break;
        case ExecutorState::CALIBRATION:
            ok = (from == ExecutorState::READY);
            break;
        case ExecutorState::FAULT:
            ok = (from == ExecutorState::EXECUTING || from == ExecutorState::STOPPING ||
                  from == ExecutorState::READY || from == ExecutorState::CALIBRATION);
            break;
        case ExecutorState::ESTOP:
            ok = true; // 任意状态
            break;
        case ExecutorState::DISABLED:
            ok = (from == ExecutorState::READY || from == ExecutorState::CALIBRATION);
            break;
    }
    if (!ok) {
        printf("[NeckState] 拒绝迁移 %s -> %s (%s)\n",
               executorStateString(from), executorStateString(to),
               reason ? reason : "无原因");
        return false;
    }
    state_ = to;
    lastReason_ = reason ? reason : "";
    logTransition(from, to, reason);
    return true;
}

bool StateMachine::estop(const char* reason) {
    if (state_ == ExecutorState::ESTOP) {
        printf("[NeckState] 已处于 ESTOP，忽略重复急停\n");
        return true;
    }
    return transition(ExecutorState::ESTOP, reason);
}

bool StateMachine::acknowledge(const char* reason) {
    if (state_ != ExecutorState::ESTOP && state_ != ExecutorState::FAULT) {
        printf("[NeckState] 非 ESTOP/FAULT 状态无需确认\n");
        return false;
    }
    // 恢复确认走专用路径：普通 transition() 永远不允许离开 ESTOP/FAULT
    ExecutorState from = state_;
    state_ = ExecutorState::READY;
    lastReason_ = reason ? reason : "";
    logTransition(from, ExecutorState::READY, reason);
    return true;
}

} // namespace neck_control
