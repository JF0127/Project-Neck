#include "executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "inverse_kinematics.h"
#include "trajectory_io.h"
#include "trajectory_validation.h"

namespace neck_control {

namespace {
const double kNumTol = 1e-6; // 数值微分容差（绝对，度）

// 反馈可信度：数值有限且三个电机角都在机械限位内（防空快照/乱码被当作合法读数）
bool plausibleFeedback(const NeckFeedback& fb, const double jointLimits[6]) {
    if (!fb.valid) return false;
    MotorAngles m{fb.joint_deg[0], fb.joint_deg[1], fb.joint_deg[2]};
    return isFinite(m.motor1) && isFinite(m.motor2) && isFinite(m.motor3) &&
           jointsWithinLimits(m, jointLimits);
}
} // namespace

NeckTrajectoryExecutor::NeckTrajectoryExecutor(NeckControlConfig cfg, NeckConfig ik,
                                               ExecutorMode mode, NeckHardwareAdapter* adapter)
    : cfg_(cfg), ik_(ik), mode_(mode), adapter_(adapter),
      calib_(cfg),
      retimer_(RetimerLimits{{cfg.limits.max_velocity_deg_s[0], cfg.limits.max_velocity_deg_s[1],
                              cfg.limits.max_velocity_deg_s[2]},
                             {cfg.limits.max_acceleration_deg_s2[0],
                              cfg.limits.max_acceleration_deg_s2[1],
                              cfg.limits.max_acceleration_deg_s2[2]},
                             {cfg.limits.max_jerk_deg_s3[0], cfg.limits.max_jerk_deg_s3[1],
                              cfg.limits.max_jerk_deg_s3[2]}},
               1.0 / cfg.loop_rate_hz) {
    sessionT0_ = std::chrono::steady_clock::now();
    startLoopIfNeeded();
    // 进程级 watchdog（始终运行，仅运动中起作用）
    watchdogRunning_ = true;
    watchdogThread_ = std::thread(&NeckTrajectoryExecutor::watchdogMain, this);
}

NeckTrajectoryExecutor::~NeckTrajectoryExecutor() {
    loopRunning_ = false;
    watchdogRunning_ = false;
    if (watchdogThread_.joinable()) watchdogThread_.join();
    if (loopThread_.joinable()) loopThread_.join();
}

double NeckTrajectoryExecutor::nowSec() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - sessionT0_).count();
}

void NeckTrajectoryExecutor::startLoopIfNeeded() {
    if (mode_ == ExecutorMode::DRY_RUN || !adapter_) return;
    if (loopThread_.joinable()) return;
    loopRunning_ = true;
    loopThread_ = std::thread(&NeckTrajectoryExecutor::loopMain, this);
}

uint16_t NeckTrajectoryExecutor::computeSpd(double velDegS) const {
    // 电机 spd 参数: 0~18000 ↔ 0~1800 rpm; 输出轴 deg/s = spd*0.6 → spd = deg_s*5/3
    double spd = std::fabs(velDegS) * (10.0 / 6.0) * cfg_.spd_margin;
    if (spd < 1.0) spd = 1.0;
    if (spd > 18000.0) spd = 18000.0;
    return (uint16_t)(spd + 0.5);
}

// ============================ 离线处理 ============================

ProcessResult NeckTrajectoryExecutor::process(const NeckTrajectoryCommand& cmd) {
    std::lock_guard<std::mutex> lock(mtx_);
    ProcessResult res;
    res.frames = cmd.frameCount();

    // 1. 输入校验（含防重放）
    ValidationContext vctx{cfg_, lastCommandId_, lastTimestamp_,
                           mode_ == ExecutorMode::HARDWARE, nowSec()};
    std::string msg;
    NeckError e = validateTrajectoryCommand(cmd, vctx, msg);
    if (e != NeckError::OK) {
        res.error = e;
        res.message = msg;
        lastResult_ = res;
        return res;
    }

    // 2~4. 坐标转换 → IK → 角度限制（neckSolve 内含姿态/电机限位）
    for (int i = 0; i < cmd.frameCount(); ++i) {
        RpyDeg r = calib_.modelToHardware(cmd.trajectory_rpy[i]);
        if (cfg_.safety_enabled && !rpyWithinLimits(r, cfg_.rpy_limits_deg)) {
            res.error = NeckError::POSE_OUT_OF_RANGE;
            res.message = "帧 " + std::to_string(i) + ": " +
                          rpyLimitReport(r, cfg_.rpy_limits_deg);
            neckLog("拒绝: %s", res.message.c_str());
            lastResult_ = res;
            return res;
        }
        MotorAngles m;
        e = neckInverseSolve(r, ik_, m, msg);
        if (e != NeckError::OK) {
            res.error = e;
            res.message = "帧 " + std::to_string(i) + " 逆解失败: " + msg;
            neckLog("拒绝: %s", res.message.c_str());
            lastResult_ = res;
            return res;
        }
        res.frame_rpy_deg.push_back(r);
        res.frame_joints_deg.push_back(m);
    }

    // 中位
    RpyDeg neutralHw = calib_.modelToHardware(cmd.robot_neutral_pose);
    e = neckInverseSolve(neutralHw, ik_, res.neutral_joints_deg, msg);
    if (e != NeckError::OK) {
        res.error = e;
        res.message = "中位逆解失败: " + msg;
        neckLog("拒绝: %s", res.message.c_str());
        lastResult_ = res;
        return res;
    }

    // 5. 起点基准：实机必须以当前反馈为准；离线用 robot_actual_initial
    bool haveActual = false;
    MotorAngles actualJ;
    if (adapter_) {
        NeckFeedback fb = adapter_->read();
        if (plausibleFeedback(fb, cfg_.joint_limits_deg)) {
            haveActual = true;
            actualJ = MotorAngles{fb.joint_deg[0], fb.joint_deg[1], fb.joint_deg[2]};
        }
    }
    if (!haveActual) {
        if (mode_ == ExecutorMode::HARDWARE) {
            res.error = NeckError::FEEDBACK_TIMEOUT;
            res.message = "实机启动必须有有效反馈（每段开始姿态必须以硬件当前反馈为准）";
            neckLog("拒绝: %s", res.message.c_str());
            lastResult_ = res;
            return res;
        }
        RpyDeg ai = calib_.modelToHardware(cmd.robot_actual_initial);
        if (neckInverseSolve(ai, ik_, actualJ, msg) != NeckError::OK) {
            actualJ = res.neutral_joints_deg; // 兜底：从中位开始
            neckLog("警告: robot_actual_initial 逆解失败，回退到中位");
        }
    }
    res.actual_joints_deg = actualJ;

    // 6. 起点偏差检查（禁止直接跳转）
    const MotorAngles& first = res.frame_joints_deg.front();
    double err = std::fabs(actualJ.motor1 - first.motor1);
    err = std::max(err, std::fabs(actualJ.motor2 - first.motor2));
    err = std::max(err, std::fabs(actualJ.motor3 - first.motor3));
    res.start_pose_error_deg = err;
    if (cfg_.safety_enabled && err > cfg_.max_start_pose_error_deg) {
        res.error = NeckError::START_POSE_ERROR_TOO_LARGE;
        res.message = "起点偏差 " + std::to_string(err) + "° 超过允许值 " +
                      std::to_string(cfg_.max_start_pose_error_deg) + "°，禁止跳转";
        neckLog("拒绝: %s", res.message.c_str());
        lastResult_ = res;
        return res;
    }

    // 7. 安全重定时（首段为 实际→首帧 过渡段；保留名义节奏）
    std::vector<MotorAngles> execFrames;
    execFrames.reserve(res.frame_joints_deg.size() + 1);
    execFrames.push_back(actualJ);
    execFrames.insert(execFrames.end(), res.frame_joints_deg.begin(), res.frame_joints_deg.end());
    const double nominalDt = 1.0 / cmd.fps;
    if (!retimer_.retime(execFrames, cmd.fps, nominalDt, res.times, res.samples,
                         res.sample_vel, res.segment_durations, msg)) {
        res.error = NeckError::RETIME_FAILED;
        res.message = msg;
        neckLog("拒绝: %s", msg.c_str());
        lastResult_ = res;
        return res;
    }
    res.duration_s = res.times.empty() ? 0.0 : res.times.back();
    res.nominal_duration_s = cmd.frameCount() > 1 ? (cmd.frameCount() - 1) / cmd.fps : 0.0;

    // 行为状态标签（按“目标帧”归属）：段 k 结束于原轨迹帧 f_k，样本状态 = states[k]；
    // 段 0 为 实际→f0 过渡段（f0 的状态即 states[0]）。
    res.sample_states.assign(res.samples.size(), "transition");
    double acc = 0.0;
    size_t seg = 0;
    for (size_t si = 0; si < res.samples.size(); ++si) {
        double t = res.times[si];
        while (seg + 1 < res.segment_durations.size() &&
               t > acc + res.segment_durations[seg] + 1e-9) {
            acc += res.segment_durations[seg];
            ++seg;
        }
        if (seg < cmd.states.size())
            res.sample_states[si] = cmd.states[seg];
    }

    // 8. 处理后重新验证全部约束
    if (cfg_.safety_enabled) {
        for (size_t i = 0; i < res.samples.size(); ++i) {
            if (!jointsWithinLimits(res.samples[i], cfg_.joint_limits_deg)) {
                res.error = NeckError::REVALIDATION_FAILED;
                res.message = "处理后采样 " + std::to_string(i) + " 电机角超限";
                neckLog("拒绝: %s", res.message.c_str());
                lastResult_ = res;
                return res;
            }
            RpyDeg ar = neckForwardSolve(res.samples[i], ik_);
            if (!rpyWithinLimits(ar, cfg_.rpy_limits_deg)) {
                res.error = NeckError::REVALIDATION_FAILED;
                res.message = "处理后采样 " + std::to_string(i) + " 头部姿态超限";
                neckLog("拒绝: %s", res.message.c_str());
                lastResult_ = res;
                return res;
            }
        }
        // 数值微分检查速度/加速度/jerk（非均匀时间步长公式）
        const size_t n = res.samples.size();
        auto qj = [&](size_t k, int j) {
            return j == 0 ? res.samples[k].motor1
                          : (j == 1 ? res.samples[k].motor2 : res.samples[k].motor3);
        };
        for (int j = 0; j < 3; ++j) {
            std::vector<double> v(n, 0.0), a(n, 0.0), jk(n, 0.0);
            for (size_t i = 1; i < n; ++i) {
                double h = res.times[i] - res.times[i - 1];
                if (h > 0) v[i] = (qj(i, j) - qj(i - 1, j)) / h;
            }
            for (size_t i = 2; i < n; ++i) {
                double h0 = res.times[i - 1] - res.times[i - 2];
                double h1 = res.times[i] - res.times[i - 1];
                if (h0 > 0 && h1 > 0)
                    a[i] = (v[i] - v[i - 1]) / (0.5 * (h0 + h1));
            }
            for (size_t i = 3; i < n; ++i) {
                double h0 = res.times[i - 1] - res.times[i - 2];
                double h1 = res.times[i] - res.times[i - 1];
                if (h0 > 0 && h1 > 0)
                    jk[i] = (a[i] - a[i - 1]) / (0.5 * (h0 + h1));
            }
            for (size_t i = 1; i < n; ++i) {
                if (std::fabs(v[i]) > cfg_.vmax(j) * (1.0 + 1e-6) + kNumTol) {
                    res.error = NeckError::REVALIDATION_FAILED;
                    res.message = "处理后采样速度超限 (j" + std::to_string(j + 1) + ")";
                    neckLog("拒绝: %s", res.message.c_str());
                    lastResult_ = res;
                    return res;
                }
                if (std::fabs(a[i]) > cfg_.amax(j) * (1.0 + 1e-6) + kNumTol) {
                    res.error = NeckError::REVALIDATION_FAILED;
                    res.message = "处理后采样加速度超限 (j" + std::to_string(j + 1) + ")";
                    neckLog("拒绝: %s", res.message.c_str());
                    lastResult_ = res;
                    return res;
                }
                if (std::fabs(jk[i]) > cfg_.jmax(j) * (1.0 + 1e-6) + kNumTol) {
                    res.error = NeckError::REVALIDATION_FAILED;
                    res.message = "处理后采样 jerk 超限 (j" + std::to_string(j + 1) + ")";
                    neckLog("拒绝: %s", res.message.c_str());
                    lastResult_ = res;
                    return res;
                }
            }
        }
    }

    // 9. 统计峰值（审计）
    res.max_vel[0] = res.max_vel[1] = res.max_vel[2] = 0.0;
    res.max_acc[0] = res.max_acc[1] = res.max_acc[2] = 0.0;
    res.max_jerk[0] = res.max_jerk[1] = res.max_jerk[2] = 0.0;
    {
        const size_t n = res.samples.size();
        auto qj = [&](size_t k, int j) {
            return j == 0 ? res.samples[k].motor1
                          : (j == 1 ? res.samples[k].motor2 : res.samples[k].motor3);
        };
        for (int j = 0; j < 3; ++j) {
            std::vector<double> v(n, 0.0), a(n, 0.0), jk(n, 0.0);
            for (size_t i = 1; i < n; ++i) {
                double h = res.times[i] - res.times[i - 1];
                if (h > 0) {
                    v[i] = (qj(i, j) - qj(i - 1, j)) / h;
                    res.max_vel[j] = std::max(res.max_vel[j], std::fabs(v[i]));
                }
            }
            for (size_t i = 2; i < n; ++i) {
                double h0 = res.times[i - 1] - res.times[i - 2];
                double h1 = res.times[i] - res.times[i - 1];
                if (h0 > 0 && h1 > 0) {
                    a[i] = (v[i] - v[i - 1]) / (0.5 * (h0 + h1));
                    res.max_acc[j] = std::max(res.max_acc[j], std::fabs(a[i]));
                }
            }
            for (size_t i = 3; i < n; ++i) {
                double h0 = res.times[i - 1] - res.times[i - 2];
                double h1 = res.times[i] - res.times[i - 1];
                if (h0 > 0 && h1 > 0) {
                    jk[i] = (a[i] - a[i - 1]) / (0.5 * (h0 + h1));
                    res.max_jerk[j] = std::max(res.max_jerk[j], std::fabs(jk[i]));
                }
            }
        }
    }

    // 提交防重放状态
    if (cmd.command_id != 0) lastCommandId_ = cmd.command_id;
    if (cmd.timestamp > 0.0) lastTimestamp_ = cmd.timestamp;
    lastAuditCmd_ = cmd;
    lastResult_ = res;
    return res;
}

// ============================ 生命周期 ============================

NeckError NeckTrajectoryExecutor::enable() {
    std::lock_guard<std::mutex> lock(mtx_);
    return enableLocked();
}

NeckError NeckTrajectoryExecutor::enableLocked() {
    if (sm_.state() != ExecutorState::DISABLED) return NeckError::OK; // 幂等
    switch (mode_) {
        case ExecutorMode::HARDWARE:
            if (!cfg_.hardware_enabled) {
                neckLog("拒绝使能: 实机模式未显式使能");
                return NeckError::HARDWARE_DISABLED;
            }
            // 注: EXECUTING 的三把锁闸门在 start() 内强制；READY 仅要求显式使能与适配器健康，
            // 以便进行标定运动（标定是 calib_confirmed 的前提，不能在使能阶段要求）。
            if (!adapter_ || !adapter_->healthy()) {
                neckLog("拒绝使能: 硬件适配器不健康");
                return NeckError::ADAPTER_FAILURE;
            }
            break;
        case ExecutorMode::MOCK:
            if (!adapter_) return NeckError::ADAPTER_FAILURE;
            break;
        case ExecutorMode::DRY_RUN:
            break;
    }
    sm_.transition(ExecutorState::READY, "enable");
    return NeckError::OK;
}

NeckError NeckTrajectoryExecutor::start(const NeckTrajectoryCommand& cmd) {
    std::unique_lock<std::mutex> lock(mtx_);
    if (sm_.state() == ExecutorState::ESTOP || sm_.state() == ExecutorState::FAULT) {
        neckLog("拒绝启动: 需先显式确认解除 %s",
                executorStateString(sm_.state()));
        return NeckError::STATE_NOT_READY;
    }
    if (mode_ == ExecutorMode::DRY_RUN) {
        neckLog("拒绝启动: dry-run 不执行");
        return NeckError::MODE_NOT_SUPPORTED;
    }
    auto waitReady = [&]() {
        return cv_.wait_for(lock, std::chrono::milliseconds((int)cfg_.stop_wait_ms),
                            [&] { return sm_.state() == ExecutorState::READY; });
    };
    // 安全打断策略：新轨迹先安全停止当前运动
    if (sm_.state() == ExecutorState::STOPPING) {
        if (!waitReady()) {
            neckLog("拒绝启动: 等待安全停止超时");
            return NeckError::STATE_NOT_READY;
        }
    }
    if (sm_.state() == ExecutorState::EXECUTING) {
        beginStopLocked();
        if (!waitReady()) {
            neckLog("拒绝启动: 打断旧轨迹超时");
            return NeckError::STATE_NOT_READY;
        }
    }
    if (sm_.state() == ExecutorState::DISABLED) {
        NeckError e = enableLocked();
        if (e != NeckError::OK) return e;
    }
    if (sm_.state() != ExecutorState::READY) {
        neckLog("拒绝启动: 状态 %s 不允许", executorStateString(sm_.state()));
        return NeckError::STATE_NOT_READY;
    }
    if (mode_ == ExecutorMode::HARDWARE &&
        !(cfg_.hardware_enabled && cfg_.calib_confirmed && cfg_.safety_confirmed))
        return NeckError::HARDWARE_DISABLED;

    lock.unlock();
    ProcessResult res = process(cmd);
    lock.lock();
    if (res.error != NeckError::OK) {
        neckLog("拒绝启动: %s", res.message.c_str());
        return res.error;
    }

    plan_ = res;
    hasPlan_ = true;
    auto now = std::chrono::steady_clock::now();
    t0_ = now;
    paused_ = false;
    firstWriteT0_ = now;
    everWrote_ = false;
    consecutiveWriteFail_ = 0;
    lastFbOk_ = now;
    trackErrActive_ = false;
    sampleCursor_ = 0;
    lastCommand_ = res.actual_joints_deg;
    lastSpd_ = {1, 1, 1};
    sm_.transition(ExecutorState::EXECUTING, "start trajectory");
    cv_.notify_all();
    neckLog("开始执行: %d 帧, 计划时长 %.3fs, 起点偏差 %.3f°",
            res.frames, res.duration_s, res.start_pose_error_deg);
    return NeckError::OK;
}

void NeckTrajectoryExecutor::pause() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (sm_.state() == ExecutorState::EXECUTING && !paused_) {
        paused_ = true;
        pauseStart_ = std::chrono::steady_clock::now();
        neckLog("暂停轨迹（保持当前位置）");
    }
}

void NeckTrajectoryExecutor::resume() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (paused_) {
        t0_ += std::chrono::steady_clock::now() - pauseStart_;
        paused_ = false;
        neckLog("恢复轨迹");
    }
}

NeckError NeckTrajectoryExecutor::stop() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (sm_.state() == ExecutorState::EXECUTING) {
        beginStopLocked();
        return NeckError::OK;
    }
    if (sm_.state() == ExecutorState::STOPPING) return NeckError::OK;
    neckLog("拒绝停止: 状态 %s", executorStateString(sm_.state()));
    return NeckError::STATE_NOT_READY;
}

NeckError NeckTrajectoryExecutor::cancel() { return stop(); }

void NeckTrajectoryExecutor::estop() {
    std::lock_guard<std::mutex> lock(mtx_);
    sm_.estop("用户急停");
    if (adapter_) adapter_->emergencyStop();
    telemetry_.estop_count++;
    hasPlan_ = false;
    calibActive_ = false;
    cv_.notify_all();
    neckLog("急停触发（状态 -> ESTOP）");
}

NeckError NeckTrajectoryExecutor::acknowledge() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (sm_.state() != ExecutorState::ESTOP && sm_.state() != ExecutorState::FAULT)
        return NeckError::STATE_NOT_READY;
    if (mode_ == ExecutorMode::HARDWARE && adapter_ && !adapter_->healthy())
        return NeckError::ADAPTER_FAILURE;
    if (adapter_) adapter_->release();
    sm_.acknowledge("显式确认");
    cv_.notify_all();
    return NeckError::OK;
}

NeckError NeckTrajectoryExecutor::enterCalibration() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (sm_.state() != ExecutorState::READY) return NeckError::STATE_NOT_READY;
    if (mode_ == ExecutorMode::HARDWARE && !cfg_.hardware_enabled) {
        neckLog("拒绝进入标定模式: 实机未显式使能");
        return NeckError::HARDWARE_DISABLED;
    }
    sm_.transition(ExecutorState::CALIBRATION, "进入标定模式");
    return NeckError::OK;
}

NeckError NeckTrajectoryExecutor::exitCalibration() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (sm_.state() != ExecutorState::CALIBRATION) return NeckError::STATE_NOT_READY;
    sm_.transition(ExecutorState::READY, "退出标定模式");
    return NeckError::OK;
}

bool NeckTrajectoryExecutor::waitForState(ExecutorState s, int timeoutMs) {
    std::unique_lock<std::mutex> lock(mtx_);
    return cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs),
                        [&] { return sm_.state() == s; });
}

NeckError NeckTrajectoryExecutor::runCalibrationMove(int axis, double deltaDeg) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (axis < 1 || axis > 3) return NeckError::INVALID_INPUT;
    if (!(deltaDeg == deltaDeg) || std::fabs(deltaDeg) < 1e-9) {
        neckLog("拒绝标定运动: 幅度必须非零");
        return NeckError::INVALID_INPUT;
    }
    if (std::fabs(deltaDeg) > cfg_.calib_max_step_deg) {
        neckLog("拒绝标定运动: |delta|=%.2f° 超过标定单步上限 %.1f°", std::fabs(deltaDeg),
                cfg_.calib_max_step_deg);
        return NeckError::POSE_OUT_OF_RANGE;
    }
    NeckError gate = calibGateLocked();
    if (gate != NeckError::OK) return gate;

    // 起点 = 当前实际反馈
    NeckFeedback fb = adapter_->read();
    if (!plausibleFeedback(fb, cfg_.joint_limits_deg)) {
        neckLog("拒绝标定运动: 无有效反馈（若刚重启，请先运行 MotorAngleGet 建立反馈）");
        return NeckError::FEEDBACK_TIMEOUT;
    }
    MotorAngles start{fb.joint_deg[0], fb.joint_deg[1], fb.joint_deg[2]};
    MotorAngles target = start;
    if (axis == 1) target.motor1 += deltaDeg;
    else if (axis == 2) target.motor2 += deltaDeg;
    else target.motor3 += deltaDeg;
    return startCalibMoveToLocked(target, "标定运动");
}

NeckError NeckTrajectoryExecutor::runCalibrationPose(double pitchDeg, double rollDeg,
                                                     double yawDeg) {
    std::lock_guard<std::mutex> lock(mtx_);
    // 入参顺序 (pitch, roll, yaw)；RpyDeg 内部字段为 {roll, pitch, yaw}，须显式赋值
    RpyDeg pose{};
    pose.pitch = pitchDeg;
    pose.roll = rollDeg;
    pose.yaw = yawDeg;
    if (!isFinite(pitchDeg) || !isFinite(rollDeg) || !isFinite(yawDeg)) {
        neckLog("拒绝标定姿态: 数值非法");
        return NeckError::INVALID_INPUT;
    }
    if (cfg_.safety_enabled && !rpyWithinLimits(pose, cfg_.rpy_limits_deg)) {
        neckLog("拒绝标定姿态: 超出头部姿态限位");
        return NeckError::POSE_OUT_OF_RANGE;
    }
    NeckError gate = calibGateLocked();
    if (gate != NeckError::OK) return gate;

    // 相对当前姿态的各分量变化须在标定幅度上限内
    NeckFeedback fb = adapter_->read();
    if (!plausibleFeedback(fb, cfg_.joint_limits_deg)) {
        neckLog("拒绝标定姿态: 无有效反馈（若刚重启，请先运行 MotorAngleGet 建立反馈）");
        return NeckError::FEEDBACK_TIMEOUT;
    }
    RpyDeg cur = neckForwardSolve(MotorAngles{fb.joint_deg[0], fb.joint_deg[1], fb.joint_deg[2]}, ik_);
    double dP = std::fabs(pose.pitch - cur.pitch);
    double dR = std::fabs(pose.roll - cur.roll);
    double dY = std::fabs(pose.yaw - cur.yaw);
    const double tol = 0.05; // 幅度上限容差（反馈噪声/收敛残差量级，度）
    if (dP > cfg_.calib_max_step_deg + tol || dR > cfg_.calib_max_step_deg + tol ||
        dY > cfg_.calib_max_step_deg + tol) {
        neckLog("拒绝标定姿态: 相对当前姿态变化 %.1f/%.1f/%.1f° 超过单步上限 %.1f°", dP, dR,
                dY, cfg_.calib_max_step_deg);
        return NeckError::POSE_OUT_OF_RANGE;
    }

    MotorAngles target;
    std::string msg;
    NeckError e = neckInverseSolve(pose, ik_, target, msg);
    if (e != NeckError::OK) {
        neckLog("拒绝标定姿态: 逆解失败 %s", msg.c_str());
        return e;
    }
    return startCalibMoveToLocked(target, "标定姿态");
}


NeckError NeckTrajectoryExecutor::calibGateLocked() {
    if (mode_ == ExecutorMode::DRY_RUN) return NeckError::MODE_NOT_SUPPORTED;
    if (mode_ == ExecutorMode::HARDWARE && !cfg_.hardware_enabled)
        return NeckError::HARDWARE_DISABLED;
    if (sm_.state() == ExecutorState::ESTOP || sm_.state() == ExecutorState::FAULT)
        return NeckError::STATE_NOT_READY;
    if (sm_.state() == ExecutorState::STOPPING || sm_.state() == ExecutorState::EXECUTING)
        return NeckError::STATE_NOT_READY;
    if (calibActive_) return NeckError::STATE_NOT_READY;
    if (sm_.state() == ExecutorState::DISABLED) {
        NeckError e = enableLocked();
        if (e != NeckError::OK) return e;
    }
    if (sm_.state() == ExecutorState::READY) {
        sm_.transition(ExecutorState::CALIBRATION, "标定运动");
    }
    if (sm_.state() != ExecutorState::CALIBRATION) return NeckError::STATE_NOT_READY;
    if (!adapter_) return NeckError::ADAPTER_FAILURE;
    return NeckError::OK;
}

NeckError NeckTrajectoryExecutor::startCalibMoveToLocked(const MotorAngles& target,
                                                         const char* what) {
    if (!jointsWithinLimits(target, cfg_.joint_limits_deg)) {
        neckLog("拒绝%s: 目标电机角超限", what);
        return NeckError::JOINT_OUT_OF_RANGE;
    }
    // 起点 = 当前实际反馈
    NeckFeedback fb = adapter_->read();
    if (!plausibleFeedback(fb, cfg_.joint_limits_deg)) {
        neckLog("拒绝%s: 无有效反馈（若刚重启，请先运行 MotorAngleGet 建立反馈）", what);
        return NeckError::FEEDBACK_TIMEOUT;
    }
    MotorAngles start{fb.joint_deg[0], fb.joint_deg[1], fb.joint_deg[2]};

    // 标定限速 S 曲线（两帧: 起点→目标）
    RetimerLimits lim{{cfg_.calib_limits.max_velocity_deg_s[0],
                       cfg_.calib_limits.max_velocity_deg_s[1],
                       cfg_.calib_limits.max_velocity_deg_s[2]},
                      {cfg_.calib_limits.max_acceleration_deg_s2[0],
                       cfg_.calib_limits.max_acceleration_deg_s2[1],
                       cfg_.calib_limits.max_acceleration_deg_s2[2]},
                      {cfg_.calib_limits.max_jerk_deg_s3[0],
                       cfg_.calib_limits.max_jerk_deg_s3[1],
                       cfg_.calib_limits.max_jerk_deg_s3[2]}};
    TrajectoryRetimer rt(lim, 1.0 / cfg_.loop_rate_hz);
    std::vector<MotorAngles> frames{start, target};
    std::vector<double> times, segs;
    std::string msg;
    if (!rt.retime(frames, 30.0, 0.0, times, calibSamples_, calibVel_, segs, msg)) {
        neckLog("拒绝%s: 规划失败 %s", what, msg.c_str());
        return NeckError::RETIME_FAILED;
    }
    calibTimes_ = times;
    calibDurationS_ = times.empty() ? 0.0 : times.back();
    calibT0_ = std::chrono::steady_clock::now();
    calibCursor_ = 0;
    calibEverWrote_ = false;
    calibActive_ = true;
    lastCommand_ = start;
    lastSpd_ = {1, 1, 1};
    cv_.notify_all();
    neckLog("%s开始: (%.2f, %.2f, %.2f) → (%.2f, %.2f, %.2f), 时长 %.2fs, 限速 %.0f/%.0f/%.0f °/s",
            what, start.motor1, start.motor2, start.motor3, target.motor1, target.motor2,
            target.motor3, calibDurationS_, cfg_.calib_limits.max_velocity_deg_s[0],
            cfg_.calib_limits.max_velocity_deg_s[1], cfg_.calib_limits.max_velocity_deg_s[2]);
    return NeckError::OK;
}

ExecutorState NeckTrajectoryExecutor::state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return sm_.state();
}

bool NeckTrajectoryExecutor::lastCommanded(MotorAngles& out) const {
    std::lock_guard<std::mutex> lock(mtx_);
    if (!everWrote_) return false;
    out = lastCommand_;
    return true;
}

// ============================ 内部：停止/故障 ============================

void NeckTrajectoryExecutor::beginStopLocked() {
    if (sm_.state() != ExecutorState::EXECUTING) return;
    sm_.transition(ExecutorState::STOPPING, "安全停止");
    stopT0_ = std::chrono::steady_clock::now();
    double maxT = 0.0;
    for (int j = 0; j < 3; ++j)
        maxT = std::max(maxT, cfg_.vmax(j) / std::max(cfg_.amax(j), 1e-9));
    stopDurationS_ = maxT + 0.2;
    neutralReturn_ = false;
    neckLog("安全停止开始（保持 %d ms）", (int)(stopDurationS_ * 1000));
}

void NeckTrajectoryExecutor::enterSafeStop(NeckError reason, const std::string& msg) {
    if (sm_.state() != ExecutorState::EXECUTING) return;
    telemetry_.last_error = reason;
    telemetry_.last_error_message = msg;
    beginStopLocked();
    neckLog("超时/指令异常触发安全停止: %s (%s)", neckErrorString(reason), msg.c_str());
}

void NeckTrajectoryExecutor::enterFault(NeckError e, const std::string& msg) {
    if (sm_.state() == ExecutorState::FAULT || sm_.state() == ExecutorState::ESTOP) return;
    telemetry_.last_error = e;
    telemetry_.last_error_message = msg;
    telemetry_.fault_count++;
    sm_.transition(ExecutorState::FAULT, msg.c_str());
    hasPlan_ = false;
    calibActive_ = false;
    cv_.notify_all();
    neckLog("故障: %s (%s)", neckErrorString(e), msg.c_str());
}

// ============================ 控制循环 ============================

void NeckTrajectoryExecutor::watchdogMain() {
    while (watchdogRunning_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!activeMotion_.load()) continue;
        auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now().time_since_epoch())
                         .count();
        uint64_t hb = heartbeatMs_.load();
        if (hb == 0 || (uint64_t)nowMs - hb <= (uint64_t)cfg_.watchdog_timeout_ms)
            continue;
        // 防重复触发：距上次触发不足 2s 不再重复（等待循环恢复心跳；恢复后自动解除）
        uint64_t lastFire = lastWatchdogFireMs_.load();
        if (lastFire != 0 && (uint64_t)nowMs - lastFire < 2000)
            continue;
        lastWatchdogFireMs_.store((uint64_t)nowMs);
        // 控制循环心跳超时：直接制动（不依赖互斥锁，保证卡死时仍能执行）
        if (adapter_) adapter_->emergencyStop();
        watchdogFired_ = true;
        neckLog("WATCHDOG: 控制循环心跳超时 %.0f ms，已触发制动", cfg_.watchdog_timeout_ms);
        if (mtx_.try_lock()) {
            if (sm_.state() != ExecutorState::FAULT && sm_.state() != ExecutorState::ESTOP)
                enterFault(NeckError::LOOP_OVERRUN, "watchdog: 控制循环心跳超时");
            mtx_.unlock();
        }
    }
}

void NeckTrajectoryExecutor::testStallLoop(int ms) { testStallMs_.store(ms); }

bool NeckTrajectoryExecutor::watchdogFired() const { return watchdogFired_.load(); }

void NeckTrajectoryExecutor::progress(double& elapsed, double& planned) const {
    std::lock_guard<std::mutex> lock(mtx_);
    elapsed = 0.0;
    planned = 0.0;
    if (!hasPlan_) return;
    planned = plan_.duration_s;
    if (sm_.state() == ExecutorState::EXECUTING) {
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        if (paused_) {
            elapsed = std::chrono::duration<double>(pauseStart_ - t0_).count();
        }
    }
}

void NeckTrajectoryExecutor::loopMain() {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        loopT0_ = std::chrono::steady_clock::now();
        tickIndex_ = 0;
    }
    const double dt = 1.0 / cfg_.loop_rate_hz;
    while (loopRunning_) {
        // 测试钩子：模拟控制循环挂起（在取锁之前，保证 watchdog 可用 try_lock 判故障）
        int stall = testStallMs_.load();
        if (stall > 0) {
            testStallMs_.store(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(stall));
        }
        auto tickStart = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lock(mtx_);
            // 心跳 + 运动标志（供进程级 watchdog）
            heartbeatMs_ = (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
            activeMotion_.store((sm_.state() == ExecutorState::EXECUTING ||
                                 sm_.state() == ExecutorState::STOPPING ||
                                 (sm_.state() == ExecutorState::CALIBRATION && calibActive_))
                                    ? 1
                                    : 0);
            double ideal = std::chrono::duration<double>(
                               tickStart - (loopT0_ + std::chrono::duration<double>(tickIndex_ * dt)))
                               .count() *
                           1000.0;
            telemetry_.ticks++;
            telemetry_.last_jitter_ms = ideal;
            telemetry_.max_jitter_ms = std::max(telemetry_.max_jitter_ms, ideal);
            telemetry_.jitter_sum_ms += ideal;

            switch (sm_.state()) {
                case ExecutorState::EXECUTING: executeTick(tickStart); break;
                case ExecutorState::STOPPING: stoppingTick(tickStart); break;
                case ExecutorState::CALIBRATION: calibTick(tickStart); break;
                default: break;
            }
            tickIndex_++;
            cv_.notify_all();
        }
        auto next = loopT0_ + std::chrono::duration<double>(tickIndex_ * dt);
        auto now = std::chrono::steady_clock::now();
        if (next < now) {
            if (++overrunCount_ > 20) {
                std::lock_guard<std::mutex> lock(mtx_);
                if (sm_.state() == ExecutorState::EXECUTING ||
                    sm_.state() == ExecutorState::STOPPING)
                    enterFault(NeckError::LOOP_OVERRUN, "控制循环持续超时");
            }
            next = now; // 不累积落后
        } else {
            overrunCount_ = 0;
        }
        std::this_thread::sleep_until(next);
    }
}

void NeckTrajectoryExecutor::sampleAt(double elapsed, MotorAngles& q,
                                      std::array<double, 3>& v, std::string& st) const {
    const std::vector<double>& times = plan_.times;
    if (times.empty()) {
        q = plan_.actual_joints_deg;
        v = {0.0, 0.0, 0.0};
        st = "transition";
        return;
    }
    size_t k = sampleCursor_;
    while (k + 1 < times.size() && times[k + 1] <= elapsed) ++k;
    sampleCursor_ = k;
    if (k + 1 < times.size()) {
        double f = (elapsed - times[k]) / (times[k + 1] - times[k]);
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        const MotorAngles& a = plan_.samples[k];
        const MotorAngles& b = plan_.samples[k + 1];
        q.motor1 = a.motor1 + f * (b.motor1 - a.motor1);
        q.motor2 = a.motor2 + f * (b.motor2 - a.motor2);
        q.motor3 = a.motor3 + f * (b.motor3 - a.motor3);
        for (int j = 0; j < 3; ++j)
            v[j] = plan_.sample_vel[k][j] + f * (plan_.sample_vel[k + 1][j] - plan_.sample_vel[k][j]);
        st = plan_.sample_states[k];
    } else {
        q = plan_.samples.back();
        v = plan_.sample_vel.back();
        st = plan_.sample_states.back();
    }
}

void NeckTrajectoryExecutor::executeTick(const std::chrono::steady_clock::time_point& now) {
    if (sm_.state() != ExecutorState::EXECUTING) return;

    if (paused_) {
        adapter_->write(lastCommand_, lastSpd_.data());
        monitorFeedback(now, lastCommand_, true);
        return;
    }

    double elapsed = std::chrono::duration<double>(now - t0_).count();
    if (hasPlan_ && elapsed >= plan_.duration_s) {
        // 正常完成：保持末帧（Silent 终点=中位）
        std::array<double, 3> holdSpd{1, 1, 1};
        adapter_->write(plan_.samples.back(), holdSpd.data());
        telemetry_.actual_duration_s = elapsed;
        telemetry_.planned_duration_s = plan_.duration_s;
        neckLog("轨迹完成: 计划 %.3fs, 实际 %.3fs", plan_.duration_s, elapsed);
        hasPlan_ = false;
        sm_.transition(ExecutorState::READY, "轨迹完成");
        adapter_->release();
        return;
    }

    MotorAngles q;
    std::array<double, 3> v{};
    std::string st;
    sampleAt(elapsed, q, v, st);

    double spd[3] = {(double)computeSpd(v[0]), (double)computeSpd(v[1]),
                     (double)computeSpd(v[2])};
    bool ok = adapter_->write(q, spd);
    if (ok) {
        everWrote_ = true;
        consecutiveWriteFail_ = 0;
        telemetry_.writes_ok++;
        lastCommand_ = q;
        lastSpd_ = {spd[0], spd[1], spd[2]};
        monitorFeedback(now, q, true);
    } else {
        telemetry_.writes_fail++;
        consecutiveWriteFail_++;
        if (!everWrote_ &&
            std::chrono::duration<double>(now - firstWriteT0_).count() * 1000.0 >
                cfg_.command_timeout_ms) {
            enterSafeStop(NeckError::COMMAND_TIMEOUT, "指令发送失败且超时");
            return;
        }
        if (consecutiveWriteFail_ > 100) {
            enterFault(NeckError::ADAPTER_FAILURE, "连续写入失败");
            return;
        }
        monitorFeedback(now, q, false); // 未成功发送，不做跟踪误差判断
    }
}

void NeckTrajectoryExecutor::calibTick(const std::chrono::steady_clock::time_point& now) {
    if (!calibActive_) return;
    double elapsed = std::chrono::duration<double>(now - calibT0_).count();
    if (elapsed >= calibDurationS_) {
        // 完成：保持目标
        adapter_->write(calibSamples_.back(), lastSpd_.data());
        neckLog("标定运动完成: 实际 %.3fs", elapsed);
        calibActive_ = false;
        sm_.transition(ExecutorState::READY, "标定运动完成");
        adapter_->release();
        cv_.notify_all();
        return;
    }
    size_t k = calibCursor_;
    while (k + 1 < calibTimes_.size() && calibTimes_[k + 1] <= elapsed) ++k;
    calibCursor_ = k;
    MotorAngles q = calibSamples_[k];
    double spd[3] = {1, 1, 1};
    if (k + 1 < calibTimes_.size()) {
        double f = (elapsed - calibTimes_[k]) / (calibTimes_[k + 1] - calibTimes_[k]);
        if (f < 0.0) f = 0.0;
        if (f > 1.0) f = 1.0;
        q.motor1 += f * (calibSamples_[k + 1].motor1 - calibSamples_[k].motor1);
        q.motor2 += f * (calibSamples_[k + 1].motor2 - calibSamples_[k].motor2);
        q.motor3 += f * (calibSamples_[k + 1].motor3 - calibSamples_[k].motor3);
        for (int j = 0; j < 3; ++j)
            spd[j] = (double)computeSpd(calibVel_[k][j] +
                                        f * (calibVel_[k + 1][j] - calibVel_[k][j]));
    }
    bool ok = adapter_->write(q, spd);
    if (ok) {
        calibEverWrote_ = true;
        telemetry_.writes_ok++;
        lastCommand_ = q;
        lastSpd_ = {spd[0], spd[1], spd[2]};
        monitorFeedback(now, q, true);
    } else {
        telemetry_.writes_fail++;
        if (!calibEverWrote_ &&
            std::chrono::duration<double>(now - calibT0_).count() * 1000.0 >
                cfg_.command_timeout_ms) {
            enterFault(NeckError::COMMAND_TIMEOUT, "标定指令发送失败超时");
            return;
        }
        monitorFeedback(now, q, false);
    }
}

void NeckTrajectoryExecutor::stoppingTick(const std::chrono::steady_clock::time_point& now) {
    if (sm_.state() != ExecutorState::STOPPING) return;

    // 阶段 1：保持当前位置（电机自然减速）
    if (!neutralReturn_ &&
        std::chrono::duration<double>(now - stopT0_).count() < stopDurationS_) {
        adapter_->write(lastCommand_, lastSpd_.data());
        monitorFeedback(now, lastCommand_, true);
        return;
    }

    // 阶段 2：超时回中（默认关闭；开启时仅超时原因允许，且低速）
    if (!neutralReturn_ && cfg_.return_to_neutral_on_timeout &&
        (telemetry_.last_error == NeckError::COMMAND_TIMEOUT ||
         telemetry_.last_error == NeckError::FEEDBACK_TIMEOUT)) {
        const MotorAngles target = plan_.neutral_joints_deg;
        RetimerLimits slow{
            {cfg_.vmax(0) * 0.25, cfg_.vmax(1) * 0.25, cfg_.vmax(2) * 0.25},
            {cfg_.amax(0) * 0.25, cfg_.amax(1) * 0.25, cfg_.amax(2) * 0.25},
            {cfg_.jmax(0) * 0.25, cfg_.jmax(1) * 0.25, cfg_.jmax(2) * 0.25}};
        TrajectoryRetimer slowRetimer(slow, 1.0 / cfg_.loop_rate_hz);
        std::vector<MotorAngles> frames{lastCommand_, target};
        std::vector<double> times, segs;
        std::vector<std::array<double, 3>> vel;
        std::string msg;
        if (slowRetimer.retime(frames, 30.0, 0.0, times, neutralSamples_, neutralVel_,
                               segs, msg)) {
            neutralTimes_ = times;
            neutralDurationS_ = times.empty() ? 0.0 : times.back();
            neutralT0_ = now;
            neutralReturn_ = true;
            neckLog("超时回中: %.2fs 低速回中位", neutralDurationS_);
        } else {
            neckLog("回中规划失败: %s", msg.c_str());
        }
    }

    if (neutralReturn_) {
        double el = std::chrono::duration<double>(now - neutralT0_).count();
        if (el >= neutralDurationS_) {
            adapter_->write(plan_.neutral_joints_deg, lastSpd_.data());
            sm_.transition(ExecutorState::READY, "超时回中完成");
            hasPlan_ = false;
            adapter_->release();
            return;
        }
        // 采样中性轨迹
        size_t k = 0;
        while (k + 1 < neutralTimes_.size() && neutralTimes_[k + 1] <= el) ++k;
        MotorAngles q = neutralSamples_[k];
        double spd[3] = {1, 1, 1};
        if (k + 1 < neutralTimes_.size()) {
            double f = (el - neutralTimes_[k]) / (neutralTimes_[k + 1] - neutralTimes_[k]);
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
            q.motor1 += f * (neutralSamples_[k + 1].motor1 - neutralSamples_[k].motor1);
            q.motor2 += f * (neutralSamples_[k + 1].motor2 - neutralSamples_[k].motor2);
            q.motor3 += f * (neutralSamples_[k + 1].motor3 - neutralSamples_[k].motor3);
            for (int j = 0; j < 3; ++j) {
                double vv = neutralVel_[k][j] + f * (neutralVel_[k + 1][j] - neutralVel_[k][j]);
                spd[j] = (double)computeSpd(vv);
            }
        }
        adapter_->write(q, spd);
        lastCommand_ = q;
        monitorFeedback(now, q, true);
        return;
    }

    // 阶段 3：停止完成
    adapter_->write(lastCommand_, lastSpd_.data());
    sm_.transition(ExecutorState::READY, "安全停止完成");
    hasPlan_ = false;
    adapter_->release();
    neckLog("安全停止完成 -> READY");
}

void NeckTrajectoryExecutor::monitorFeedback(const std::chrono::steady_clock::time_point& now,
                                             const MotorAngles& desired, bool checkTracking) {
    const bool inMotion = sm_.state() == ExecutorState::EXECUTING ||
                          (sm_.state() == ExecutorState::CALIBRATION && calibActive_);
    NeckFeedback fb = adapter_ ? adapter_->read() : NeckFeedback{};
    if (!fb.valid) {
        telemetry_.feedback_miss++;
        if (inMotion && !paused_ &&
            std::chrono::duration<double>(now - lastFbOk_).count() * 1000.0 >
                cfg_.feedback_timeout_ms) {
            if (sm_.state() == ExecutorState::CALIBRATION) {
                enterFault(NeckError::FEEDBACK_TIMEOUT, "标定中反馈超时");
            } else if (cfg_.stop_on_timeout) {
                enterSafeStop(NeckError::FEEDBACK_TIMEOUT, "反馈超时");
            } else {
                paused_ = true;
                pauseStart_ = now;
                neckLog("反馈超时且 stop_on_timeout=false：保持位置");
            }
        }
        return;
    }
    telemetry_.feedback_ok++;
    lastFbOk_ = now;

    for (int j = 0; j < 3; ++j) {
        if (!isFinite(fb.joint_deg[j])) {
            enterFault(NeckError::NAN_FEEDBACK, "反馈含 NaN/Inf");
            return;
        }
        if (fb.error[j] != 0) {
            enterFault(NeckError::MOTOR_FAULT,
                       "电机 " + std::to_string(j + 1) + " 故障码 " +
                           std::to_string(fb.error[j]));
            return;
        }
    }
    if (cfg_.safety_enabled) {
        MotorAngles actual{fb.joint_deg[0], fb.joint_deg[1], fb.joint_deg[2]};
        if (!jointsWithinLimits(actual, cfg_.joint_limits_deg)) {
            enterFault(NeckError::JOINT_OUT_OF_RANGE, "实际电机角超限");
            return;
        }
        RpyDeg ar = neckForwardSolve(actual, ik_);
        if (!rpyWithinLimits(ar, cfg_.rpy_limits_deg)) {
            enterFault(NeckError::POSE_OUT_OF_RANGE, "实际头部姿态超限");
            return;
        }
    }
    if (checkTracking && inMotion && !paused_) {
        double err = std::fabs(desired.motor1 - fb.joint_deg[0]);
        err = std::max(err, std::fabs(desired.motor2 - fb.joint_deg[1]));
        err = std::max(err, std::fabs(desired.motor3 - fb.joint_deg[2]));
        if (err > telemetry_.max_tracking_error_deg)
            telemetry_.max_tracking_error_deg = err;
        if (err > cfg_.max_tracking_error_deg) {
            if (!trackErrActive_) {
                trackErrActive_ = true;
                trackErrT0_ = now;
            } else if (std::chrono::duration<double>(now - trackErrT0_).count() * 1000.0 >
                       cfg_.tracking_error_timeout_ms) {
                enterFault(NeckError::TRACKING_ERROR,
                           "跟踪误差持续超限 " + std::to_string(err) + "°");
            }
        } else {
            trackErrActive_ = false;
        }
    }
}

bool NeckTrajectoryExecutor::saveAudit(const std::string& dir, std::string& message) const {
    std::lock_guard<std::mutex> lock(mtx_);
    const ProcessResult& r = lastResult_;
    if (r.error != NeckError::OK || r.frames == 0) {
        message = "无成功处理结果可审计";
        return false;
    }
    AuditSamples audit;
    audit.fps = lastAuditCmd_.fps;
    audit.loop_rate_hz = cfg_.loop_rate_hz;
    for (const RpyDeg& p : r.frame_rpy_deg)
        audit.rpy_deg.push_back({p.roll, p.pitch, p.yaw});
    for (const MotorAngles& m : r.frame_joints_deg)
        audit.joints_deg.push_back({m.motor1, m.motor2, m.motor3});
    audit.sample_t = r.times;
    for (const MotorAngles& m : r.samples)
        audit.sample_joints_deg.push_back({m.motor1, m.motor2, m.motor3});
    audit.sample_vel_deg_s = r.sample_vel;
    audit.sample_states = r.sample_states;
    return dumpAuditJson(dir, lastAuditCmd_, audit, message);
}

} // namespace neck_control
