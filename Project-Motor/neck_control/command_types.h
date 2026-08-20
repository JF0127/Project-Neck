// Neck Trajectory Executor —— 统一输入数据类型与错误码。
// 纯类型定义，无外部依赖，便于单元测试。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace neck_control {

// 上游三态调度：Speaking / Listening / Silent。
// 注意：这是"行为状态"，不是硬件安全状态；硬件安全状态始终拥有最高优先级。
enum class TrajectoryState { Speaking, Listening, Silent, Unknown };

inline const char* stateName(TrajectoryState s) {
    switch (s) {
        case TrajectoryState::Speaking:  return "speaking";
        case TrajectoryState::Listening: return "listening";
        case TrajectoryState::Silent:    return "silent";
        default:                         return "unknown";
    }
}

// 上游模型 RPY 约定：
//   单位 radian；顺序 [roll, pitch, yaw]；
//   旋转组合 R = Ry(yaw) @ Rx(pitch) @ Rz(roll)
struct CoordinateConvention {
    std::string unit = "radian";
    std::vector<std::string> order = {"roll", "pitch", "yaw"};
    std::string rotation = "R = Ry(yaw) @ Rx(pitch) @ Rz(roll)";
};

// 统一输入命令（对应上游 JSON）。
struct NeckTrajectoryCommand {
    double fps = 30.0;
    CoordinateConvention coordinate_convention;

    // 本次轨迹开始时的实际姿态基准（模型坐标，弧度，[roll,pitch,yaw]）
    std::array<double, 3> robot_actual_initial = {0.0, 0.0, 0.0};
    // 机器人标定中位（模型坐标，弧度）
    std::array<double, 3> robot_neutral_pose = {0.0, 0.0, 0.0};

    // 统一坐标系中的目标绝对 RPY，shape [N,3]，弧度，顺序 [roll,pitch,yaw]
    std::vector<std::array<double, 3>> trajectory_rpy;

    // 与轨迹等长的状态序列（可空）
    std::vector<std::string> states;

    // 可选防重放字段：0 表示未提供
    double timestamp = 0.0;   // 模型时间（秒，任意纪元，仅用于相对序/新鲜度判断）
    uint64_t command_id = 0;  // 单调递增的命令序号，0 表示未提供

    int frameCount() const { return (int)trajectory_rpy.size(); }
};

// 硬件坐标下的头部姿态（度）
struct RpyDeg {
    double roll = 0.0, pitch = 0.0, yaw = 0.0;
};

// 电机输出轴角（度）—— 与 neck_kinematics.h 的 MotorAngles 同义，避免强耦合
struct MotorAngles {
    double motor1 = 0.0, motor2 = 0.0, motor3 = 0.0;
};

// 错误码：所有拒绝原因必须有明确错误码并记录日志。
enum class NeckError {
    OK = 0,
    INVALID_JSON,             // JSON 无法解析
    INVALID_INPUT,            // 输入字段缺失/类型错误
    EMPTY_TRAJECTORY,         // N <= 0
    BAD_SHAPE,                // RPY 行数/列数错误（须 [N,3]）
    TRAJECTORY_TOO_LONG,      // N 超上限
    NON_FINITE,               // NaN/Inf
    FPS_OUT_OF_RANGE,         // fps 超出配置允许范围
    STATE_LENGTH_MISMATCH,    // states 长度 != N
    CONVENTION_MISMATCH,      // 单位/轴序/旋转约定不匹配
    STALE_TIMESTAMP,          // 时间戳过期
    DUPLICATE_COMMAND,        // 重复命令 id
    OUT_OF_ORDER_COMMAND,     // 乱序命令
    START_POSE_ERROR_TOO_LARGE, // 轨迹起点与实际姿态偏差过大
    SILENT_NOT_AT_NEUTRAL,    // Silent 终点不在 robot_neutral_pose
    POSE_OUT_OF_RANGE,        // 硬件坐标 RPY 超限
    JOINT_OUT_OF_RANGE,       // 逆解电机角超限
    IK_SINGULAR,              // 耦合矩阵奇异
    RETIME_FAILED,            // 重定时失败
    REVALIDATION_FAILED,      // 处理后重新验证失败
    NOT_CALIBRATED,           // 未标定/未确认
    HARDWARE_DISABLED,        // 实机模式未显式使能
    SAFETY_NOT_CONFIRMED,     // 安全参数未确认
    MODE_NOT_SUPPORTED,       // 当前模式不支持该操作
    STATE_NOT_READY,          // 状态机不允许
    ADAPTER_FAILURE,          // 硬件写入失败
    COMMAND_TIMEOUT,          // 指令超时
    FEEDBACK_TIMEOUT,         // 反馈超时
    TRACKING_ERROR,           // 跟踪误差持续超限
    MOTOR_FAULT,              // 电机故障码
    NAN_FEEDBACK,             // 反馈 NaN/Inf
    LOOP_OVERRUN,             // 控制循环超时
    ESTOP,                    // 急停
    INTERNAL                  // 内部错误
};

inline const char* neckErrorString(NeckError e) {
    switch (e) {
        case NeckError::OK: return "OK";
        case NeckError::INVALID_JSON: return "JSON 无法解析";
        case NeckError::INVALID_INPUT: return "输入字段缺失或类型错误";
        case NeckError::EMPTY_TRAJECTORY: return "轨迹为空 (N<=0)";
        case NeckError::BAD_SHAPE: return "RPY 形状错误（须为 [N,3]）";
        case NeckError::TRAJECTORY_TOO_LONG: return "轨迹过长";
        case NeckError::NON_FINITE: return "含 NaN/Inf 数值";
        case NeckError::FPS_OUT_OF_RANGE: return "fps 超出配置允许范围";
        case NeckError::STATE_LENGTH_MISMATCH: return "states 长度与轨迹不一致";
        case NeckError::CONVENTION_MISMATCH: return "单位/轴序/旋转约定不匹配";
        case NeckError::STALE_TIMESTAMP: return "时间戳过期";
        case NeckError::DUPLICATE_COMMAND: return "重复命令";
        case NeckError::OUT_OF_ORDER_COMMAND: return "命令乱序";
        case NeckError::START_POSE_ERROR_TOO_LARGE: return "起点与实际姿态偏差过大";
        case NeckError::SILENT_NOT_AT_NEUTRAL: return "Silent 终点不在 robot_neutral_pose";
        case NeckError::POSE_OUT_OF_RANGE: return "头部姿态超出安全范围";
        case NeckError::JOINT_OUT_OF_RANGE: return "逆解电机角超出安全范围";
        case NeckError::IK_SINGULAR: return "耦合矩阵奇异";
        case NeckError::RETIME_FAILED: return "安全重定时失败";
        case NeckError::REVALIDATION_FAILED: return "处理后重新验证失败";
        case NeckError::NOT_CALIBRATED: return "未完成标定";
        case NeckError::HARDWARE_DISABLED: return "实机模式未使能";
        case NeckError::SAFETY_NOT_CONFIRMED: return "安全参数未确认";
        case NeckError::MODE_NOT_SUPPORTED: return "当前模式不支持该操作";
        case NeckError::STATE_NOT_READY: return "状态机不允许该操作";
        case NeckError::ADAPTER_FAILURE: return "硬件写入失败";
        case NeckError::COMMAND_TIMEOUT: return "指令超时";
        case NeckError::FEEDBACK_TIMEOUT: return "反馈超时";
        case NeckError::TRACKING_ERROR: return "跟踪误差持续超限";
        case NeckError::MOTOR_FAULT: return "电机故障";
        case NeckError::NAN_FEEDBACK: return "反馈含 NaN/Inf";
        case NeckError::LOOP_OVERRUN: return "控制循环超时";
        case NeckError::ESTOP: return "急停";
        case NeckError::INTERNAL: return "内部错误";
        default: return "未知错误";
    }
}

// 角度工具（避免与 math_ops.h 的 fmaxf 重声明冲突，不使用 <cmath> 的 fmaxf 名字）
inline constexpr double kDeg2Rad = 3.14159265358979323846 / 180.0;
inline constexpr double kRad2Deg = 180.0 / 3.14159265358979323846;

inline double degToRad(double d) { return d * kDeg2Rad; }
inline double radToDeg(double r) { return r * kRad2Deg; }

inline bool isFinite(double x) {
    return x == x && x > -1.7976931348623157e308 && x < 1.7976931348623157e308;
}

} // namespace neck_control
