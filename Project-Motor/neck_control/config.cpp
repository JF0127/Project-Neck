#include "config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace neck_control {

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// 解析逗号分隔 double 列表，要求数量正好为 n。
bool parseDoubles(const std::string& val, int n, double* out) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : val) {
        if (c == ',') { parts.push_back(trim(cur)); cur.clear(); }
        else cur += c;
    }
    parts.push_back(trim(cur));
    if ((int)parts.size() != n) return false;
    for (int i = 0; i < n; ++i) {
        if (parts[i].empty()) return false;
        char* end = nullptr;
        double d = std::strtod(parts[i].c_str(), &end);
        if (end == nullptr || *end != '\0') return false;
        out[i] = d;
    }
    return true;
}

bool parseBool(const std::string& val, bool* out) {
    std::string v = trim(val);
    if (v == "true" || v == "1" || v == "yes") { *out = true; return true; }
    if (v == "false" || v == "0" || v == "no") { *out = false; return true; }
    return false;
}

bool assign(NeckControlConfig& c, const std::string& key, const std::string& val) {
    double d = std::atof(val.c_str());
    long l = std::atol(val.c_str());
    double arr3[3], arr6[6], arr9[9];
    bool b = false;

    if (key == "mode.hardware_enabled") { if (!parseBool(val, &b)) return false; c.hardware_enabled = b; }
    else if (key == "calib.confirmed") { if (!parseBool(val, &b)) return false; c.calib_confirmed = b; }
    else if (key == "safety.confirmed") { if (!parseBool(val, &b)) return false; c.safety_confirmed = b; }

    else if (key == "calib.model_axis_order") {
        std::vector<std::string> parts; std::string cur;
        for (char ch : val) { if (ch == ',') { parts.push_back(trim(cur)); cur.clear(); } else cur += ch; }
        parts.push_back(trim(cur));
        if (parts.size() != 3) return false;
        c.model_axis_order = parts;
    }
    else if (key == "calib.hardware_axis_order") {
        std::vector<std::string> parts; std::string cur;
        for (char ch : val) { if (ch == ',') { parts.push_back(trim(cur)); cur.clear(); } else cur += ch; }
        parts.push_back(trim(cur));
        if (parts.size() != 3) return false;
        c.hardware_axis_order = parts;
    }
    else if (key == "calib.axis_sign") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.axis_sign[i] = arr3[i]; }
    else if (key == "calib.rotation_matrix") { if (!parseDoubles(val, 9, arr9)) return false; for (int i = 0; i < 9; ++i) c.rotation_matrix[i] = arr9[i]; }
    else if (key == "calib.rpy_offset_deg") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.rpy_offset_deg[i] = arr3[i]; }
    else if (key == "calib.use_neck_config_neutral") { if (!parseBool(val, &b)) return false; c.use_neck_config_neutral = b; }

    else if (key == "loop.rate_hz") c.loop_rate_hz = d;
    else if (key == "loop.watchdog_timeout_ms") c.watchdog_timeout_ms = d;
    else if (key == "input.fps_min") c.fps_min = d;
    else if (key == "input.fps_max") c.fps_max = d;
    else if (key == "input.max_trajectory_frames") c.max_trajectory_frames = (int)l;
    else if (key == "input.max_timestamp_skew_s") c.max_timestamp_skew_s = d;
    else if (key == "input.stop_wait_ms") c.stop_wait_ms = d;

    else if (key == "safety.enabled") { if (!parseBool(val, &b)) return false; c.safety_enabled = b; }
    else if (key == "safety.rpy_limits_deg") { if (!parseDoubles(val, 6, arr6)) return false; for (int i = 0; i < 6; ++i) c.rpy_limits_deg[i] = arr6[i]; }
    else if (key == "safety.joint_limits_deg") { if (!parseDoubles(val, 6, arr6)) return false; for (int i = 0; i < 6; ++i) c.joint_limits_deg[i] = arr6[i]; }
    else if (key == "safety.max_velocity_deg_s") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.limits.max_velocity_deg_s[i] = arr3[i]; }
    else if (key == "safety.max_acceleration_deg_s2") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.limits.max_acceleration_deg_s2[i] = arr3[i]; }
    else if (key == "safety.max_jerk_deg_s3") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.limits.max_jerk_deg_s3[i] = arr3[i]; }
    else if (key == "safety.max_start_pose_error_deg") c.max_start_pose_error_deg = d;
    else if (key == "safety.max_tracking_error_deg") c.max_tracking_error_deg = d;
    else if (key == "safety.tracking_error_timeout_ms") c.tracking_error_timeout_ms = d;
    else if (key == "safety.command_timeout_ms") c.command_timeout_ms = d;
    else if (key == "safety.feedback_timeout_ms") c.feedback_timeout_ms = d;
    else if (key == "safety.return_to_neutral_on_timeout") { if (!parseBool(val, &b)) return false; c.return_to_neutral_on_timeout = b; }
    else if (key == "safety.stop_on_timeout") { if (!parseBool(val, &b)) return false; c.stop_on_timeout = b; }
    else if (key == "safety.emergency_stop_enabled") { if (!parseBool(val, &b)) return false; c.emergency_stop_enabled = b; }
    else if (key == "safety.spd_margin") c.spd_margin = d;

    else if (key == "calib.max_velocity_deg_s") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.calib_limits.max_velocity_deg_s[i] = arr3[i]; }
    else if (key == "calib.max_acceleration_deg_s2") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.calib_limits.max_acceleration_deg_s2[i] = arr3[i]; }
    else if (key == "calib.max_jerk_deg_s3") { if (!parseDoubles(val, 3, arr3)) return false; for (int i = 0; i < 3; ++i) c.calib_limits.max_jerk_deg_s3[i] = arr3[i]; }
    else if (key == "calib.max_step_deg") c.calib_max_step_deg = d;

    else if (key == "mock.max_velocity_deg_s") { if (!parseDoubles(val, 3, arr3)) return false; c.mock_max_velocity_deg_s.m1 = arr3[0]; c.mock_max_velocity_deg_s.m2 = arr3[1]; c.mock_max_velocity_deg_s.m3 = arr3[2]; }
    else return false;
    return true;
}

} // namespace

bool loadNeckControlConfig(const std::string& path, NeckControlConfig& out) {
    NeckControlConfig c{}; // 默认即保守占位
    std::ifstream f(path);
    if (!f) {
        printf("[NeckControl] 无法打开配置文件 %s，使用内置保守默认值\n", path.c_str());
        out = c;
        return false;
    }
    std::string line;
    int lineno = 0;
    while (std::getline(f, line)) {
        ++lineno;
        std::string s = trim(line);
        if (s.empty() || s[0] == '#') continue;
        size_t eq = s.find('=');
        if (eq == std::string::npos) {
            printf("[NeckControl] 第 %d 行缺少 '='，已忽略: %s\n", lineno, s.c_str());
            continue;
        }
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        size_t hash = val.find('#');
        if (hash != std::string::npos) val = trim(val.substr(0, hash));
        if (!assign(c, key, val))
            printf("[NeckControl] 第 %d 行未知键或非法值 '%s'，已忽略\n", lineno, key.c_str());
    }
    out = c;
    return true;
}

bool saveNeckControlConfig(const std::string& path, const NeckControlConfig& c) {
    std::ofstream f(path);
    if (!f) { printf("[NeckControl] 无法写入配置文件 %s\n", path.c_str()); return false; }
    f << "# Neck Trajectory Executor 配置。单位：度。\n";
    f << "# 安全参数为保守占位值，实机前必须经标定/测试确认并置 confirmed=true。\n\n";
    f << "mode.hardware_enabled = " << (c.hardware_enabled ? "true" : "false") << "\n";
    f << "calib.confirmed = " << (c.calib_confirmed ? "true" : "false") << "\n";
    f << "safety.confirmed = " << (c.safety_confirmed ? "true" : "false") << "\n\n";

    f << "calib.model_axis_order = " << c.model_axis_order[0] << "," << c.model_axis_order[1] << "," << c.model_axis_order[2] << "\n";
    f << "calib.hardware_axis_order = " << c.hardware_axis_order[0] << "," << c.hardware_axis_order[1] << "," << c.hardware_axis_order[2] << "\n";
    f << "calib.axis_sign = " << c.axis_sign[0] << "," << c.axis_sign[1] << "," << c.axis_sign[2] << "\n";
    f << "calib.rotation_matrix = ";
    for (int i = 0; i < 9; ++i) f << c.rotation_matrix[i] << (i < 8 ? "," : "");
    f << "\ncalib.rpy_offset_deg = " << c.rpy_offset_deg[0] << "," << c.rpy_offset_deg[1] << "," << c.rpy_offset_deg[2] << "\n";
    f << "calib.use_neck_config_neutral = " << (c.use_neck_config_neutral ? "true" : "false") << "\n\n";

    f << "loop.rate_hz = " << c.loop_rate_hz << "\n";
    f << "loop.watchdog_timeout_ms = " << c.watchdog_timeout_ms << "\n";
    f << "input.fps_min = " << c.fps_min << "\ninput.fps_max = " << c.fps_max << "\n";
    f << "input.max_trajectory_frames = " << c.max_trajectory_frames << "\n";
    f << "input.max_timestamp_skew_s = " << c.max_timestamp_skew_s << "\n";
    f << "input.stop_wait_ms = " << c.stop_wait_ms << "\n\n";

    f << "safety.enabled = " << (c.safety_enabled ? "true" : "false") << "\n";
    f << "safety.rpy_limits_deg = ";
    for (int i = 0; i < 6; ++i) f << c.rpy_limits_deg[i] << (i < 5 ? "," : "");
    f << "\nsafety.joint_limits_deg = ";
    for (int i = 0; i < 6; ++i) f << c.joint_limits_deg[i] << (i < 5 ? "," : "");
    f << "\nsafety.max_velocity_deg_s = " << c.limits.max_velocity_deg_s[0] << "," << c.limits.max_velocity_deg_s[1] << "," << c.limits.max_velocity_deg_s[2] << "\n";
    f << "safety.max_acceleration_deg_s2 = " << c.limits.max_acceleration_deg_s2[0] << "," << c.limits.max_acceleration_deg_s2[1] << "," << c.limits.max_acceleration_deg_s2[2] << "\n";
    f << "safety.max_jerk_deg_s3 = " << c.limits.max_jerk_deg_s3[0] << "," << c.limits.max_jerk_deg_s3[1] << "," << c.limits.max_jerk_deg_s3[2] << "\n";
    f << "safety.max_start_pose_error_deg = " << c.max_start_pose_error_deg << "\n";
    f << "safety.max_tracking_error_deg = " << c.max_tracking_error_deg << "\n";
    f << "safety.tracking_error_timeout_ms = " << c.tracking_error_timeout_ms << "\n";
    f << "safety.command_timeout_ms = " << c.command_timeout_ms << "\n";
    f << "safety.feedback_timeout_ms = " << c.feedback_timeout_ms << "\n";
    f << "safety.return_to_neutral_on_timeout = " << (c.return_to_neutral_on_timeout ? "true" : "false") << "\n";
    f << "safety.stop_on_timeout = " << (c.stop_on_timeout ? "true" : "false") << "\n";
    f << "safety.emergency_stop_enabled = " << (c.emergency_stop_enabled ? "true" : "false") << "\n";
    f << "safety.spd_margin = " << c.spd_margin << "\n\n";

    f << "calib.max_velocity_deg_s = " << c.calib_limits.max_velocity_deg_s[0] << "," << c.calib_limits.max_velocity_deg_s[1] << "," << c.calib_limits.max_velocity_deg_s[2] << "\n";
    f << "calib.max_acceleration_deg_s2 = " << c.calib_limits.max_acceleration_deg_s2[0] << "," << c.calib_limits.max_acceleration_deg_s2[1] << "," << c.calib_limits.max_acceleration_deg_s2[2] << "\n";
    f << "calib.max_jerk_deg_s3 = " << c.calib_limits.max_jerk_deg_s3[0] << "," << c.calib_limits.max_jerk_deg_s3[1] << "," << c.calib_limits.max_jerk_deg_s3[2] << "\n";
    f << "calib.max_step_deg = " << c.calib_max_step_deg << "\n\n";
    f << "mock.max_velocity_deg_s = " << c.mock_max_velocity_deg_s.m1 << "," << c.mock_max_velocity_deg_s.m2 << "," << c.mock_max_velocity_deg_s.m3 << "\n";
    return true;
}

} // namespace neck_control
