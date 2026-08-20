// Neck Trajectory Executor 配置。
// 沿用项目 key=value 文本配置风格（与 neck_config.txt 一致），所有安全参数必须进配置，
// 不硬编码。硬件使能/标定确认/安全确认三把锁默认全部关闭。
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace neck_control {

struct JointTriple {
    double m1 = 0.0, m2 = 0.0, m3 = 0.0;
};

// 安全限制（电机输出轴，度制）
struct SafetyLimits {
    double max_velocity_deg_s[3] = {0, 0, 0};       // 速度（度/秒）
    double max_acceleration_deg_s2[3] = {0, 0, 0};  // 加速度（度/秒²）
    double max_jerk_deg_s3[3] = {0, 0, 0};          // jerk（度/秒³）
};

struct NeckControlConfig {
    // ============ 模式与使能（默认全部关闭） ============
    bool hardware_enabled = false;  // 实机模式必须显式使能
    bool calib_confirmed = false;   // 标定完成并确认
    bool safety_confirmed = false;  // 安全参数经实机确认（未确认则禁止实机）

    // ============ 坐标系标定 ============
    // 模型分量命名（对应上游 order 数组元素顺序）
    std::vector<std::string> model_axis_order = {"roll", "pitch", "yaw"};
    // 硬件欧拉分量命名（R = Rz(第3分量) @ Rx(第2分量) @ Ry(第1分量)）
    std::vector<std::string> hardware_axis_order = {"roll", "pitch", "yaw"};
    // 各硬件分量符号（1 或 -1）
    std::array<double, 3> axis_sign = {1.0, 1.0, 1.0};
    // 模型坐标系 → 机器人坐标系的固定旋转（行主序 3x3），默认单位阵
    std::array<double, 9> rotation_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    // 硬件坐标分量残余偏置（度，按 hardware_axis_order 顺序）
    std::array<double, 3> rpy_offset_deg = {0, 0, 0};
    // 硬件零位对应的电机角（度）；留空则使用 neck_config.txt 的 m10/m20/m30
    bool use_neck_config_neutral = true;

    // 标定模式限速（比安全限速更保守）
    SafetyLimits calib_limits{{10, 10, 15}, {30, 30, 45}, {150, 150, 225}};
    // 标定单步最大幅度（度），防止标定指令误输入造成大动作
    double calib_max_step_deg = 5.0;

    // ============ 控制循环 ============
    double loop_rate_hz = 100.0;    // 硬件控制频率（>= 30 Hz）
    double watchdog_timeout_ms = 1000.0; // 控制循环心跳超时（进程级 watchdog 触发制动）
    double fps_min = 5.0;           // 上游 fps 允许范围
    double fps_max = 120.0;
    int max_trajectory_frames = 200000;  // 防内存滥用上限
    double max_timestamp_skew_s = 5.0;   // 时间戳新鲜度（仅实机模式强制）
    double stop_wait_ms = 1000.0;        // 安全打断等待上限

    // ============ 安全 ============
    bool safety_enabled = true;
    // 头部姿态限位（度）：pitch_min,pitch_max,roll_min,roll_max,yaw_min,yaw_max
    double rpy_limits_deg[6] = {-40, 25, -35, 35, -117, 58};
    // 电机输出轴限位（度）：m1_min,m1_max,m2_min,m2_max,m3_min,m3_max
    double joint_limits_deg[6] = {-85, 64, -34, 113, 60, 235};
    // 速度/加速度/jerk 上限（度制，电机输出轴；保守占位，待实机标定）
    SafetyLimits limits{{25, 25, 30}, {60, 60, 90}, {300, 300, 450}};

    double max_start_pose_error_deg = 5.0;
    double max_tracking_error_deg = 3.0;
    double tracking_error_timeout_ms = 300.0;
    double command_timeout_ms = 200.0;
    double feedback_timeout_ms = 200.0;
    bool return_to_neutral_on_timeout = false;  // 故障/超时默认不自动回中
    bool stop_on_timeout = true;
    bool emergency_stop_enabled = true;
    double spd_margin = 1.25;  // 电机 spd 参数安全余量

    // ============ mock ============
    JointTriple mock_max_velocity_deg_s{30, 30, 40};  // mock 执行器动力学

    // 便捷访问
    double vmax(int j) const { return limits.max_velocity_deg_s[j]; }
    double amax(int j) const { return limits.max_acceleration_deg_s2[j]; }
    double jmax(int j) const { return limits.max_jerk_deg_s3[j]; }
};

// 从 key=value 文件加载配置；缺项保留默认值；文件不存在返回 false（保留默认）。
// 语法与 neck_config.cpp 一致：# 注释、key = value、行内 # 注释、逗号分隔列表。
bool loadNeckControlConfig(const std::string& path, NeckControlConfig& out);

// 将配置写回文件（供确认/审计）。
bool saveNeckControlConfig(const std::string& path, const NeckControlConfig& cfg);

} // namespace neck_control
