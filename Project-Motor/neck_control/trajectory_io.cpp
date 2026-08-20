#include "trajectory_io.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#include "coordinate_calibration.h"
#include "json_parser.h"

namespace neck_control {

bool readTextFile(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

namespace {

template <size_t N>
bool readDouble3(const json::ValuePtr& v, std::array<double, N>& out, const char* what) {
    if (!v || !v->isArray() || v->arr.size() != N) return false;
    for (size_t i = 0; i < N; ++i) {
        if (!v->arr[i]->isNumber()) return false;
        double d = v->arr[i]->numVal;
        if (!isFinite(d)) return false;
        out[i] = d;
    }
    (void)what;
    return true;
}

std::string trimStr(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

} // namespace

NeckError loadNeckTrajectoryJsonV1(const std::string& jsonText,
                                   NeckTrajectoryCommand& out,
                                   std::string& message) {
    std::string jerr;
    json::ValuePtr root = json::parse(jsonText, &jerr);
    if (!root || !root->isObject()) {
        message = "JSON 解析失败: " + jerr;
        return NeckError::INVALID_JSON;
    }
    json::ValuePtr t = root->find("trajectory");
    if (!t || !t->isObject()) {
        message = "V1 格式缺少 trajectory 对象";
        return NeckError::INVALID_INPUT;
    }
    auto num = [&](const char* k, double* v) -> bool {
        json::ValuePtr f = t->find(k);
        if (!f || !f->isNumber()) return false;
        *v = f->numVal;
        return true;
    };

    // 1. fps / units / rpy_order / rotation_convention
    double fps = 30.0;
    if (t->find("fps") && !num("fps", &fps)) {
        message = "V1: fps 字段非法";
        return NeckError::INVALID_INPUT;
    }
    out.fps = fps;
    out.coordinate_convention.unit = "radian";
    out.coordinate_convention.order = {"roll", "pitch", "yaw"};
    out.coordinate_convention.rotation = "R = Ry(yaw) @ Rx(pitch) @ Rz(roll)";
    if (t->find("units")) {
        json::ValuePtr u = t->find("units");
        if (!u->isString()) {
            message = "V1: units 必须为字符串";
            return NeckError::INVALID_INPUT;
        }
        out.coordinate_convention.unit = u->strVal;
    }
    if (t->find("rpy_order")) {
        json::ValuePtr o = t->find("rpy_order");
        if (!o->isString()) {
            message = "V1: rpy_order 必须为字符串";
            return NeckError::INVALID_INPUT;
        }
        std::vector<std::string> parts;
        std::string cur;
        for (char c : o->strVal) {
            if (c == ',') { parts.push_back(trimStr(cur)); cur.clear(); }
            else cur += c;
        }
        parts.push_back(trimStr(cur));
        if (parts.size() != 3) {
            message = "V1: rpy_order 必须为 3 个分量";
            return NeckError::INVALID_INPUT;
        }
        out.coordinate_convention.order = parts;
    }
    if (t->find("rotation_convention")) {
        json::ValuePtr rc = t->find("rotation_convention");
        if (!rc->isString()) {
            message = "V1: rotation_convention 必须为字符串";
            return NeckError::INVALID_INPUT;
        }
        out.coordinate_convention.rotation = rc->strVal;
    }

    // 2. robot_actual_initial / robot_neutral_pose
    json::ValuePtr ai = t->find("robot_actual_initial");
    if (!ai || !readDouble3(ai, out.robot_actual_initial, "robot_actual_initial")) {
        message = "V1: robot_actual_initial 必须为 [N,3] 有限数值";
        return NeckError::INVALID_INPUT;
    }
    json::ValuePtr np = t->find("robot_neutral_pose");
    if (!np || !readDouble3(np, out.robot_neutral_pose, "robot_neutral_pose")) {
        message = "V1: robot_neutral_pose 必须为 [N,3] 有限数值";
        return NeckError::INVALID_INPUT;
    }

    // 3. states（整数编码 + states_legend）→ 字符串
    out.states.clear();
    json::ValuePtr st = t->find("states");
    if (st && st->isArray()) {
        json::ValuePtr legend = t->find("states_legend");
        if (!legend || !legend->isObject()) {
            message = "V1: 整数编码 states 需要 states_legend";
            return NeckError::INVALID_INPUT;
        }
        for (const auto& s : st->arr) {
            if (!s->isNumber()) {
                message = "V1: states 元素必须为整数编码";
                return NeckError::INVALID_INPUT;
            }
            long code = (long)s->numVal;
            std::string key = std::to_string(code);
            const json::ValuePtr v = legend->find(key);
            if (!v || !v->isString()) {
                message = "V1: states_legend 缺少编码 " + key;
                return NeckError::INVALID_INPUT;
            }
            out.states.push_back(v->strVal);
        }
    }

    // 4. rpy：相对 robot_actual_initial 的统一参考系 → 绝对 RPY（旋转矩阵组合）
    json::ValuePtr rpy = t->find("rpy");
    if (!rpy || !rpy->isArray()) {
        message = "V1: 缺少 rpy 数组";
        return NeckError::INVALID_INPUT;
    }
    out.trajectory_rpy.clear();
    out.trajectory_rpy.reserve(rpy->arr.size());
    double Rai[9];
    CoordinateCalibration::modelRotationMatrix(out.robot_actual_initial, Rai);
    for (size_t i = 0; i < rpy->arr.size(); ++i) {
        std::array<double, 3> rel{};
        if (!readDouble3(rpy->arr[i], rel, "rpy 行")) {
            message = "V1: rpy 第 " + std::to_string(i) + " 行必须为 3 个有限数值";
            return NeckError::INVALID_INPUT;
        }
        double Rrel[9], Rabs[9];
        CoordinateCalibration::modelRotationMatrix(rel, Rrel);
        CoordinateCalibration::matMul(Rai, Rrel, Rabs); // R_abs = R(ai) @ R(rel)
        double e[3];
        CoordinateCalibration::extractModelEuler(Rabs, e);
        out.trajectory_rpy.push_back({e[0], e[1], e[2]});
    }
    return NeckError::OK;
}

NeckError loadNeckTrajectoryJson(const std::string& jsonText,
                                 NeckTrajectoryCommand& out,
                                 std::string& message) {
    // 自动识别模型输出 V1 格式（顶层 format_version=1.0 或 trajectory 为对象包装）
    {
        std::string jerr;
        json::ValuePtr probe = json::parse(jsonText, &jerr);
        if (probe && probe->isObject()) {
            json::ValuePtr fv = probe->find("format_version");
            if (fv && fv->isString() && fv->strVal == "1.0") {
                return loadNeckTrajectoryJsonV1(jsonText, out, message);
            }
            json::ValuePtr t = probe->find("trajectory");
            if (t && t->isObject() && !probe->find("fps"))
                return loadNeckTrajectoryJsonV1(jsonText, out, message);
        }
    }
    std::string jerr;
    json::ValuePtr root = json::parse(jsonText, &jerr);
    if (!root || !root->isObject()) {
        message = "JSON 解析失败: " + jerr;
        return NeckError::INVALID_JSON;
    }

    auto numField = [&](const char* key, double* v) -> bool {
        json::ValuePtr f = root->find(key);
        if (!f || !f->isNumber()) return false;
        *v = f->numVal;
        return true;
    };

    // fps
    double fps = 30.0;
    if (root->find("fps")) {
        if (!numField("fps", &fps) || !isFinite(fps)) {
            message = "fps 字段非法";
            return NeckError::INVALID_INPUT;
        }
    }
    out.fps = fps;

    // coordinate_convention
    json::ValuePtr cc = root->find("coordinate_convention");
    if (cc && cc->isObject()) {
        json::ValuePtr unit = cc->find("unit");
        if (unit && unit->isString()) out.coordinate_convention.unit = unit->strVal;
        json::ValuePtr order = cc->find("order");
        if (order && order->isArray()) {
            out.coordinate_convention.order.clear();
            for (const auto& o : order->arr)
                if (o->isString()) out.coordinate_convention.order.push_back(o->strVal);
        }
        json::ValuePtr rot = cc->find("rotation");
        if (rot && rot->isString()) out.coordinate_convention.rotation = rot->strVal;
    }

    // robot_actual_initial / robot_neutral_pose
    json::ValuePtr ai = root->find("robot_actual_initial");
    if (ai && !readDouble3(ai, out.robot_actual_initial, "robot_actual_initial")) {
        message = "robot_actual_initial 必须为 [N,3] 的有限数值数组";
        return NeckError::INVALID_INPUT;
    }
    json::ValuePtr np = root->find("robot_neutral_pose");
    if (np && !readDouble3(np, out.robot_neutral_pose, "robot_neutral_pose")) {
        message = "robot_neutral_pose 必须为 [N,3] 的有限数值数组";
        return NeckError::INVALID_INPUT;
    }

    // trajectory
    json::ValuePtr traj = root->find("trajectory");
    if (!traj || !traj->isArray()) {
        message = "缺少 trajectory 数组";
        return NeckError::INVALID_INPUT;
    }
    out.trajectory_rpy.clear();
    out.trajectory_rpy.reserve(traj->arr.size());
    for (size_t i = 0; i < traj->arr.size(); ++i) {
        std::array<double, 3> row{};
        if (!readDouble3(traj->arr[i], row, "trajectory 行")) {
            message = "trajectory 第 " + std::to_string(i) + " 行必须为 3 个有限数值";
            return NeckError::INVALID_INPUT;
        }
        out.trajectory_rpy.push_back(row);
    }

    // states
    out.states.clear();
    json::ValuePtr states = root->find("states");
    if (states && states->isArray()) {
        for (const auto& st : states->arr) {
            if (!st->isString()) {
                message = "states 必须全为字符串";
                return NeckError::INVALID_INPUT;
            }
            out.states.push_back(st->strVal);
        }
    }

    // 可选防重放字段
    numField("timestamp", &out.timestamp);
    json::ValuePtr cid = root->find("command_id");
    if (cid && cid->isNumber()) out.command_id = (uint64_t)cid->numVal;

    return NeckError::OK;
}

bool dumpAuditJson(const std::string& dir, const NeckTrajectoryCommand& cmd,
                   const AuditSamples& audit, std::string& message) {
    // 自动创建审计目录
    std::string mkdirCmd = "mkdir -p '" + dir + "'";
    if (::system(mkdirCmd.c_str()) != 0) {
        message = "无法创建审计目录 " + dir;
        return false;
    }
    char buf[4096];

    // 原始命令（尽量保真回写）
    {
        std::string path = dir + "/original_trajectory.json";
        std::ofstream f(path);
        if (!f) { message = "无法写入 " + path; return false; }
        f << "{\n";
        f << "  \"fps\": " << cmd.fps << ",\n";
        f << "  \"coordinate_convention\": {\n";
        f << "    \"unit\": \"" << cmd.coordinate_convention.unit << "\",\n";
        f << "    \"order\": [";
        for (size_t i = 0; i < cmd.coordinate_convention.order.size(); ++i) {
            if (i) f << ", ";
            f << "\"" << cmd.coordinate_convention.order[i] << "\"";
        }
        f << "],\n";
        f << "    \"rotation\": \"" << cmd.coordinate_convention.rotation << "\"\n";
        f << "  },\n";
        f << "  \"robot_actual_initial\": [" << cmd.robot_actual_initial[0] << ", "
          << cmd.robot_actual_initial[1] << ", " << cmd.robot_actual_initial[2] << "],\n";
        f << "  \"robot_neutral_pose\": [" << cmd.robot_neutral_pose[0] << ", "
          << cmd.robot_neutral_pose[1] << ", " << cmd.robot_neutral_pose[2] << "],\n";
        f << "  \"command_id\": " << cmd.command_id << ",\n";
        f << "  \"timestamp\": " << cmd.timestamp << ",\n";
        f << "  \"trajectory\": [\n";
        for (size_t i = 0; i < cmd.trajectory_rpy.size(); ++i) {
            const auto& p = cmd.trajectory_rpy[i];
            snprintf(buf, sizeof(buf), "    [%.9f, %.9f, %.9f]%s\n",
                     p[0], p[1], p[2], i + 1 < cmd.trajectory_rpy.size() ? "," : "");
            f << buf;
        }
        f << "  ],\n  \"states\": [";
        for (size_t i = 0; i < cmd.states.size(); ++i) {
            if (i) f << ", ";
            f << "\"" << cmd.states[i] << "\"";
        }
        f << "]\n}\n";
    }

    // 处理后轨迹
    {
        std::string path = dir + "/processed_trajectory.json";
        std::ofstream f(path);
        if (!f) { message = "无法写入 " + path; return false; }
        f << "{\n";
        f << "  \"note\": \"安全处理后的硬件坐标系轨迹\",\n";
        f << "  \"units_deg\": \"rpy 与电机角均为度\",\n";
        f << "  \"fps\": " << audit.fps << ",\n";
        f << "  \"loop_rate_hz\": " << audit.loop_rate_hz << ",\n";
        f << "  \"frames\": " << audit.rpy_deg.size() << ",\n";
        f << "  \"hardware_rpy_deg\": [\n";
        for (size_t i = 0; i < audit.rpy_deg.size(); ++i) {
            const auto& p = audit.rpy_deg[i];
            snprintf(buf, sizeof(buf), "    [%.6f, %.6f, %.6f]%s\n",
                     p[0], p[1], p[2], i + 1 < audit.rpy_deg.size() ? "," : "");
            f << buf;
        }
        f << "  ],\n  \"joints_deg\": [\n";
        for (size_t i = 0; i < audit.joints_deg.size(); ++i) {
            const auto& p = audit.joints_deg[i];
            snprintf(buf, sizeof(buf), "    [%.6f, %.6f, %.6f]%s\n",
                     p[0], p[1], p[2], i + 1 < audit.joints_deg.size() ? "," : "");
            f << buf;
        }
        f << "  ],\n  \"retimed_samples\": [\n";
        for (size_t i = 0; i < audit.sample_t.size(); ++i) {
            const auto& j = audit.sample_joints_deg[i];
            const auto& v = audit.sample_vel_deg_s[i];
            const char* st = i < audit.sample_states.size() ? audit.sample_states[i].c_str() : "unknown";
            snprintf(buf, sizeof(buf),
                     "    {\"t\": %.6f, \"joints_deg\": [%.6f, %.6f, %.6f], \"vel_deg_s\": [%.6f, %.6f, %.6f], \"state\": \"%s\"}%s\n",
                     audit.sample_t[i], j[0], j[1], j[2], v[0], v[1], v[2], st,
                     i + 1 < audit.sample_t.size() ? "," : "");
            f << buf;
        }
        f << "  ]\n}\n";
    }
    return true;
}

} // namespace neck_control
