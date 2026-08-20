// Neck Trajectory Executor：独立、安全的颈部轨迹执行层。
//
// 管线：
//   上游 RPY 轨迹 → 输入校验 → 坐标系转换 → RPY→关节 IK → 角度限制
//   → 速度/加速度/jerk 限制（S 曲线重定时）→ 重新验证 → 时间驱动控制循环
//   → 硬件指令 → 姿态反馈与故障监控
//
// 三种模式：
//   DRY_RUN  : 无适配器，只做解析/转换/IK/限幅/重定时/日志（不连接硬件）
//   MOCK     : MockAdapter 模拟执行器与反馈（验证控制循环/超时/急停）
//   HARDWARE : 默认关闭；需 hardware_enabled && calib_confirmed && safety_confirmed
//              且显式传入才允许
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

#include "adapter_interface.h"
#include "coordinate_calibration.h"
#include "command_types.h"
#include "config.h"
#include "neck_config.h"
#include "state_machine.h"
#include "telemetry.h"
#include "trajectory_retimer.h"

namespace neck_control {

enum class ExecutorMode { DRY_RUN, MOCK, HARDWARE };

// 一次轨迹的完整处理结果（审计用）。
struct ProcessResult {
    NeckError error = NeckError::OK;
    std::string message;

    int frames = 0;
    std::vector<RpyDeg> frame_rpy_deg;          // 每帧硬件坐标 RPY（度）
    std::vector<MotorAngles> frame_joints_deg;  // 每帧电机角（度）

    MotorAngles actual_joints_deg;   // 实际基准（衔接起点）
    MotorAngles neutral_joints_deg;  // 中位电机角
    double start_pose_error_deg = 0.0;

    std::vector<double> times;                  // 重定时采样时刻（秒）
    std::vector<MotorAngles> samples;           // 重定时采样关节角
    std::vector<std::array<double, 3>> sample_vel;  // 采样速度（度/秒）
    std::vector<std::string> sample_states;     // 采样对应行为状态
    std::vector<double> segment_durations;      // 每段时长（含过渡段）
    double duration_s = 0.0;
    double nominal_duration_s = 0.0;  // 名义时长（帧数/fps），用于重定时比例

    double max_vel[3] = {0, 0, 0};    // 处理后实际峰值
    double max_acc[3] = {0, 0, 0};
    double max_jerk[3] = {0, 0, 0};
};

class NeckTrajectoryExecutor {
public:
    // adapter 可为空（DRY_RUN）。
    NeckTrajectoryExecutor(NeckControlConfig cfg, NeckConfig ik, ExecutorMode mode,
                           NeckHardwareAdapter* adapter);
    ~NeckTrajectoryExecutor();
    NeckTrajectoryExecutor(const NeckTrajectoryExecutor&) = delete;
    NeckTrajectoryExecutor& operator=(const NeckTrajectoryExecutor&) = delete;

    // ---- 离线全流程（dry-run 入口；也供 start 内部调用）----
    ProcessResult process(const NeckTrajectoryCommand& cmd);

    // ---- 执行控制 ----
    NeckError enable();                        // DISABLED → READY（校验模式闸门）
    NeckError start(const NeckTrajectoryCommand& cmd);  // → EXECUTING（新轨迹安全打断旧轨迹）
    void pause();                              // EXECUTING 内暂停推进（保持位置）
    void resume();
    NeckError stop();                          // → STOPPING → READY
    NeckError cancel();                        // 同 stop（立即安全停止当前轨迹）
    void estop();                              // 任何状态 → ESTOP
    NeckError acknowledge();                   // 显式确认，解除 ESTOP/FAULT → READY
    NeckError enterCalibration();
    NeckError exitCalibration();
    // 标定运动：单轴小幅度低速运动（受标定限速/幅度上限/限位/反馈监控/急停保护）。
    // axis=1..3；deltaDeg 为电机输出轴增量（度），|delta| <= calib_max_step_deg。
    // 仅在 READY 或 CALIBRATION 状态可用；运动完成后回到 READY。
    NeckError runCalibrationMove(int axis, double deltaDeg);
    // 标定姿态：三轴联动到绝对头部姿态（度，硬件坐标），各分量相对当前姿态
    // 变化 <= calib_max_step_deg，经 IK 与限位校验，其余保护同 runCalibrationMove。
    NeckError runCalibrationPose(double pitchDeg, double rollDeg, double yawDeg);

    ExecutorState state() const;
    // 执行进度（秒）：未在运动时为 0。
    void progress(double& elapsed, double& planned) const;
    // 进程级 watchdog 是否曾触发（控制循环心跳超时）。
    bool watchdogFired() const;
    // 测试钩子：让控制循环在下次 tick 前挂起 ms 毫秒（模拟卡死，仅供回归测试）。
    void testStallLoop(int ms);
    // 最近一次实际下发的目标电机角（度）；从未下发过返回 false。只读诊断用。
    bool lastCommanded(MotorAngles& out) const;
    const LoopTelemetry& telemetry() const { return telemetry_; }
    const ProcessResult& lastResult() const { return lastResult_; }
    ExecutorMode mode() const { return mode_; }

    // 审计输出：dir/original_trajectory.json、dir/processed_trajectory.json
    bool saveAudit(const std::string& dir, std::string& message) const;

    // 测试辅助：等待状态迁移（毫秒超时）
    bool waitForState(ExecutorState s, int timeoutMs);

private:
    NeckError enableLocked();
    void beginStopLocked();
    NeckError calibGateLocked();
    NeckError startCalibMoveToLocked(const MotorAngles& target, const char* what);
    void loopMain();
    void executeTick(const std::chrono::steady_clock::time_point& now);
    void stoppingTick(const std::chrono::steady_clock::time_point& now);
    void calibTick(const std::chrono::steady_clock::time_point& now);
    void sampleAt(double elapsed, MotorAngles& q, std::array<double, 3>& v, std::string& st) const;
    void monitorFeedback(const std::chrono::steady_clock::time_point& now,
                         const MotorAngles& desired, bool checkTracking = true);
    void enterFault(NeckError e, const std::string& msg);
    void enterSafeStop(NeckError reason, const std::string& msg);
    uint16_t computeSpd(double velDegS) const;
    double nowSec() const;
    void startLoopIfNeeded();

    NeckControlConfig cfg_;
    NeckConfig ik_;
    ExecutorMode mode_;
    NeckHardwareAdapter* adapter_;
    CoordinateCalibration calib_;
    TrajectoryRetimer retimer_;

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::thread loopThread_;
    std::atomic<bool> loopRunning_{false};

    // 进程级 watchdog：独立线程监视控制循环心跳，心跳超时直接制动
    std::thread watchdogThread_;
    std::atomic<bool> watchdogRunning_{false};
    std::atomic<uint64_t> heartbeatMs_{0};   // 控制循环心跳（单调毫秒）
    std::atomic<int> activeMotion_{0};       // 1=运动中（EXECUTING/STOPPING/标定运动）
    std::atomic<bool> watchdogFired_{false};
    std::atomic<int> testStallMs_{0};        // 测试钩子：挂起毫秒数（0=正常）
    std::atomic<uint64_t> lastWatchdogFireMs_{0}; // 上次 watchdog 触发时刻（防重复触发）
    void watchdogMain();

    StateMachine sm_;
    LoopTelemetry telemetry_;
    ProcessResult lastResult_;

    bool hasPlan_ = false;
    ProcessResult plan_;

    // 防重放
    uint64_t lastCommandId_ = 0;
    double lastTimestamp_ = 0.0;
    std::chrono::steady_clock::time_point sessionT0_;

    // 循环时序
    std::chrono::steady_clock::time_point loopT0_;
    std::chrono::steady_clock::time_point t0_;  // 轨迹起始
    bool paused_ = false;
    std::chrono::steady_clock::time_point pauseStart_;
    uint64_t tickIndex_ = 0;
    int overrunCount_ = 0;
    mutable size_t sampleCursor_ = 0;

    // 停止/故障监控
    std::chrono::steady_clock::time_point stopT0_;
    double stopDurationS_ = 0.0;
    MotorAngles lastCommand_;
    std::array<double, 3> lastSpd_{1, 1, 1};
    std::chrono::steady_clock::time_point firstWriteT0_;
    bool everWrote_ = false;
    int consecutiveWriteFail_ = 0;
    std::chrono::steady_clock::time_point lastFbOk_;
    std::chrono::steady_clock::time_point trackErrT0_;
    bool trackErrActive_ = false;
    bool neutralReturn_ = false;
    std::chrono::steady_clock::time_point neutralT0_;
    std::vector<MotorAngles> neutralSamples_;
    std::vector<double> neutralTimes_;
    std::vector<std::array<double, 3>> neutralVel_;
    double neutralDurationS_ = 0.0;

    // 审计：最近一次成功处理的原始命令
    NeckTrajectoryCommand lastAuditCmd_;

    // 标定运动
    bool calibActive_ = false;
    std::chrono::steady_clock::time_point calibT0_;
    std::vector<double> calibTimes_;
    std::vector<MotorAngles> calibSamples_;
    std::vector<std::array<double, 3>> calibVel_;
    double calibDurationS_ = 0.0;
    mutable size_t calibCursor_ = 0;
    bool calibEverWrote_ = false;
};

} // namespace neck_control
