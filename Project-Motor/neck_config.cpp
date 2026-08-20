#include "neck_config.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

NeckConfig defaultNeckConfig() {
    NeckConfig c{};
    // --- 耦合参数：占位值，必须实机标定后替换（见 neck_model.md §8 / tools/neck_calibrate.py）---
    c.c11 = 0.5;  c.c12 = -0.5;
    c.c21 = 0.5;  c.c22 = 0.5;   // det(C) = 0.5，非奇异占位
    c.k3  = 1.0;

    // --- 头部正中时电机角：占位用 START(最低/静止位)，实为待标定的中位角 ---
    c.m10 = -84.0;  c.m20 = 113.0;  c.m30 = 177.0;
    c.p0 = 0.0;  c.r0 = 0.0;  c.y0 = 0.0;

    // --- 电机输出轴安全范围（度），取自原 neck_kinematics.h ---
    c.motor1_min = -84.0;  c.motor1_max = 64.0;
    c.motor2_min = -34.0;  c.motor2_max = 113.0;
    c.motor3_min = 60.0; c.motor3_max = 235.0;

    // --- 头部姿态安全范围（度）---
    c.pitch_min = -40.0;  c.pitch_max = 25.0;
    c.roll_min  = -30.0;  c.roll_max  = 30.0;
    c.yaw_min   = -117.0; c.yaw_max   = 58.0;

    c.det_eps = 1e-6;

    // --- 下发参数 ---
    c.passage1 = 1;  c.passage2 = 2;  c.passage3 = 3;
    c.id1 = 1;  c.id2 = 2;  c.id3 = 3;
    c.speed = 50;  c.current = 500;  c.ack_status = 2;
    return c;
}

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// 将 key 分派到对应字段。识别返回 true。
bool assign(NeckConfig& c, const std::string& key, const std::string& val) {
    double d = std::atof(val.c_str());
    long   l = std::atol(val.c_str());
    if      (key == "c11") c.c11 = d;
    else if (key == "c12") c.c12 = d;
    else if (key == "c21") c.c21 = d;
    else if (key == "c22") c.c22 = d;
    else if (key == "k3")  c.k3 = d;
    else if (key == "m10") c.m10 = d;
    else if (key == "m20") c.m20 = d;
    else if (key == "m30") c.m30 = d;
    else if (key == "p0")  c.p0 = d;
    else if (key == "r0")  c.r0 = d;
    else if (key == "y0")  c.y0 = d;
    else if (key == "motor1_min") c.motor1_min = d;
    else if (key == "motor1_max") c.motor1_max = d;
    else if (key == "motor2_min") c.motor2_min = d;
    else if (key == "motor2_max") c.motor2_max = d;
    else if (key == "motor3_min") c.motor3_min = d;
    else if (key == "motor3_max") c.motor3_max = d;
    else if (key == "pitch_min") c.pitch_min = d;
    else if (key == "pitch_max") c.pitch_max = d;
    else if (key == "roll_min")  c.roll_min = d;
    else if (key == "roll_max")  c.roll_max = d;
    else if (key == "yaw_min")   c.yaw_min = d;
    else if (key == "yaw_max")   c.yaw_max = d;
    else if (key == "det_eps")   c.det_eps = d;
    else if (key == "passage1")  c.passage1 = (uint8_t)l;
    else if (key == "passage2")  c.passage2 = (uint8_t)l;
    else if (key == "passage3")  c.passage3 = (uint8_t)l;
    else if (key == "id1")       c.id1 = (uint16_t)l;
    else if (key == "id2")       c.id2 = (uint16_t)l;
    else if (key == "id3")       c.id3 = (uint16_t)l;
    else if (key == "speed")     c.speed = (uint16_t)l;
    else if (key == "current")   c.current = (uint16_t)l;
    else if (key == "ack_status") c.ack_status = (uint8_t)l;
    else return false;
    return true;
}

} // namespace

bool loadNeckConfig(const std::string& path, NeckConfig& out) {
    out = defaultNeckConfig();
    std::ifstream f(path);
    if (!f) {
        printf("[NeckConfig] 无法打开配置文件 %s，使用内置默认值\n", path.c_str());
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
            printf("[NeckConfig] 第 %d 行缺少 '='，已忽略: %s\n", lineno, s.c_str());
            continue;
        }
        std::string key = trim(s.substr(0, eq));
        std::string val = trim(s.substr(eq + 1));
        // 去掉行内 '#' 注释
        size_t hash = val.find('#');
        if (hash != std::string::npos) val = trim(val.substr(0, hash));
        if (!assign(out, key, val))
            printf("[NeckConfig] 第 %d 行未知键 '%s'，已忽略\n", lineno, key.c_str());
    }
    return true;
}

bool saveNeckConfig(const std::string& path, const NeckConfig& c) {
    std::ofstream f(path);
    if (!f) {
        printf("[NeckConfig] 无法写入配置文件 %s\n", path.c_str());
        return false;
    }
    f << "# Neck 建模参数配置。单位：角度一律为度。\n";
    f << "# 头部度 = C·电机度，C/k3 为无量纲比值。详见 neck_model.md。\n\n";
    f << "# 一、二号电机 → pitch/roll 的 2x2 耦合矩阵\n";
    f << "c11 = " << c.c11 << "\nc12 = " << c.c12 << "\n";
    f << "c21 = " << c.c21 << "\nc22 = " << c.c22 << "\n";
    f << "# 三号电机 → yaw 传动比\nk3 = " << c.k3 << "\n\n";
    f << "# 头部正中(pitch=roll=yaw=0)时的电机输出轴角(度)\n";
    f << "m10 = " << c.m10 << "\nm20 = " << c.m20 << "\nm30 = " << c.m30 << "\n";
    f << "# 头部中位残余偏置(度)，一般为 0\n";
    f << "p0 = " << c.p0 << "\nr0 = " << c.r0 << "\ny0 = " << c.y0 << "\n\n";
    f << "# 电机输出轴安全范围(度)\n";
    f << "motor1_min = " << c.motor1_min << "\nmotor1_max = " << c.motor1_max << "\n";
    f << "motor2_min = " << c.motor2_min << "\nmotor2_max = " << c.motor2_max << "\n";
    f << "motor3_min = " << c.motor3_min << "\nmotor3_max = " << c.motor3_max << "\n\n";
    f << "# 头部姿态安全范围(度)\n";
    f << "pitch_min = " << c.pitch_min << "\npitch_max = " << c.pitch_max << "\n";
    f << "roll_min = " << c.roll_min << "\nroll_max = " << c.roll_max << "\n";
    f << "yaw_min = " << c.yaw_min << "\nyaw_max = " << c.yaw_max << "\n\n";
    f << "# 耦合矩阵奇异判定阈值\ndet_eps = " << c.det_eps << "\n\n";
    f << "# 下发参数\n";
    f << "passage1 = " << (int)c.passage1 << "\npassage2 = " << (int)c.passage2
      << "\npassage3 = " << (int)c.passage3 << "\n";
    f << "id1 = " << c.id1 << "\nid2 = " << c.id2 << "\nid3 = " << c.id3 << "\n";
    f << "speed = " << c.speed << "\ncurrent = " << c.current
      << "\nack_status = " << (int)c.ack_status << "\n";
    return true;
}
