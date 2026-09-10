#include "neck/neck_config.h"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

using Values = std::unordered_map<std::string, std::string>;

std::string trim(const std::string& text) {
    const std::string whitespace = " \t\r\n";
    const std::size_t first = text.find_first_not_of(whitespace);
    if (first == std::string::npos) {
        return {};
    }
    const std::size_t last = text.find_last_not_of(whitespace);
    return text.substr(first, last - first + 1);
}

bool readValues(const std::string& path, Values& values, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "cannot open configuration file: " + path;
        return false;
    }

    std::string line;
    std::size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        const std::size_t separator = line.find('=');
        if (separator == std::string::npos ||
            line.find('=', separator + 1) != std::string::npos) {
            error = "line " + std::to_string(line_number) +
                    ": expected exactly one '='";
            return false;
        }

        const std::string key = trim(line.substr(0, separator));
        std::string value = trim(line.substr(separator + 1));
        if (key.empty() || value.empty()) {
            error = "line " + std::to_string(line_number) +
                    ": key and value must not be empty";
            return false;
        }
        // Optional matching quotes keep path values valid Python syntax, since
        // this file is also parsed by generic source checks.
        if (value.size() >= 2 &&
            (value.front() == '"' || value.front() == '\'') &&
            value.back() == value.front()) {
            value = value.substr(1, value.size() - 2);
        }
        if (!values.emplace(key, value).second) {
            error = "line " + std::to_string(line_number) +
                    ": duplicate key '" + key + "'";
            return false;
        }
    }

    if (!input.eof()) {
        error = "failed while reading configuration file: " + path;
        return false;
    }
    return true;
}

bool requireString(const Values& values, const std::string& key,
                   std::string& output, std::string& error) {
    const auto it = values.find(key);
    if (it == values.end()) {
        error = "missing required key '" + key + "'";
        return false;
    }
    output = it->second;
    return true;
}

bool requireInt(const Values& values, const std::string& key,
                int& output, std::string& error) {
    std::string text;
    if (!requireString(values, key, text, error)) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        output = parsed;
        return true;
    } catch (const std::exception&) {
        error = "key '" + key + "' must be a valid integer, got '" + text + "'";
        return false;
    }
}

bool requireDouble(const Values& values, const std::string& key,
                   double& output, std::string& error) {
    std::string text;
    if (!requireString(values, key, text, error)) {
        return false;
    }
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(parsed)) {
            throw std::invalid_argument("invalid floating point value");
        }
        output = parsed;
        return true;
    } catch (const std::exception&) {
        error = "key '" + key + "' must be a finite number, got '" + text + "'";
        return false;
    }
}

bool requireBool(const Values& values, const std::string& key,
                 bool& output, std::string& error) {
    std::string text;
    if (!requireString(values, key, text, error)) {
        return false;
    }
    for (char& character : text) {
        character = static_cast<char>(
            std::tolower(static_cast<unsigned char>(character)));
    }
    if (text == "true" || text == "1") {
        output = true;
        return true;
    }
    if (text == "false" || text == "0") {
        output = false;
        return true;
    }
    error = "key '" + key + "' must be true/false or 1/0, got '" + text + "'";
    return false;
}

bool loadMotor(const Values& values, int number, MotorConfig& motor,
               std::string& error) {
    const std::string prefix = "motor" + std::to_string(number) + ".";
    return requireInt(values, prefix + "passage", motor.passage, error) &&
           requireInt(values, prefix + "id", motor.id, error) &&
           requireDouble(values, prefix + "min_position_deg", motor.min_position_deg, error) &&
           requireDouble(values, prefix + "max_position_deg", motor.max_position_deg, error) &&
           requireDouble(values, prefix + "center_position_deg", motor.center_position_deg, error) &&
           requireDouble(values, prefix + "max_velocity_deg_s", motor.max_velocity_deg_s, error) &&
           requireInt(values, prefix + "speed_param", motor.speed_param, error) &&
           requireInt(values, prefix + "current_param", motor.current_param, error);
}

std::unordered_set<std::string> knownKeys() {
    std::unordered_set<std::string> keys = {
        "network_interface", "slave_id", "ack_status",
        "pitch_min_deg", "pitch_max_deg", "roll_min_deg", "roll_max_deg",
        "yaw_min_deg", "yaw_max_deg", "c11", "c12", "c21", "c22", "k3",
        "pitch_center_deg", "roll_center_deg", "yaw_center_deg", "det_eps",
        "feedback.enabled", "feedback.socket_path", "feedback.rate_hz"
    };
    for (int number = 1; number <= 3; ++number) {
        const std::string prefix = "motor" + std::to_string(number) + ".";
        keys.insert(prefix + "passage");
        keys.insert(prefix + "id");
        keys.insert(prefix + "min_position_deg");
        keys.insert(prefix + "max_position_deg");
        keys.insert(prefix + "center_position_deg");
        keys.insert(prefix + "max_velocity_deg_s");
        keys.insert(prefix + "speed_param");
        keys.insert(prefix + "current_param");
    }
    return keys;
}

bool validateConfig(const NeckConfig& config, std::string& error) {
    if (config.network_interface.empty()) {
        error = "network_interface must not be empty";
        return false;
    }
    if (config.slave_id < 0) {
        error = "slave_id must be non-negative";
        return false;
    }
    if (config.ack_status < 0 || config.ack_status > 3) {
        error = "ack_status must be in [0, 3]";
        return false;
    }

    std::unordered_set<int> passages;
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        const MotorConfig& motor = config.motors[index];
        const std::string name = "motor" + std::to_string(index + 1);
        if (motor.passage < 1 || motor.passage > 6) {
            error = name + ".passage must be in [1, 6]";
            return false;
        }
        if (!passages.insert(motor.passage).second) {
            error = "motor passages must be unique";
            return false;
        }
        if (motor.id < 1 || motor.id > 0x7FE) {
            error = name + ".id must be in [1, 0x7FE]";
            return false;
        }
        if (!(motor.min_position_deg < motor.center_position_deg &&
              motor.center_position_deg < motor.max_position_deg)) {
            error = name + " positions must satisfy min < center < max";
            return false;
        }
        if (motor.max_velocity_deg_s <= 0.0) {
            error = name + ".max_velocity_deg_s must be greater than 0";
            return false;
        }
        if (motor.speed_param <= 0 || motor.speed_param > 18000) {
            error = name + ".speed_param must be in [1, 18000]";
            return false;
        }
        if (motor.current_param < 0 || motor.current_param > 3000) {
            error = name + ".current_param must be in [0, 3000]";
            return false;
        }
    }

    if (!(config.pitch_min_deg < config.pitch_max_deg)) {
        error = "pitch range must satisfy min < max";
        return false;
    }
    if (!(config.roll_min_deg < config.roll_max_deg)) {
        error = "roll range must satisfy min < max";
        return false;
    }
    if (!(config.yaw_min_deg < config.yaw_max_deg)) {
        error = "yaw range must satisfy min < max";
        return false;
    }
    if (config.det_eps <= 0.0) {
        error = "det_eps must be greater than 0";
        return false;
    }

    const double determinant = config.c11 * config.c22 - config.c12 * config.c21;
    if (std::abs(determinant) <= config.det_eps) {
        error = "kinematics matrix determinant is too close to zero";
        return false;
    }
    if (std::abs(config.k3) <= config.det_eps) {
        error = "k3 is too close to zero";
        return false;
    }
    if (config.feedback.socket_path.empty()) {
        error = "feedback.socket_path must not be empty";
        return false;
    }
    if (!std::isfinite(config.feedback.rate_hz) ||
        config.feedback.rate_hz <= 0.0 || config.feedback.rate_hz > 1000.0) {
        error = "feedback.rate_hz must be finite and in (0, 1000]";
        return false;
    }
    return true;
}

}  // namespace

bool loadNeckConfig(const std::string& path,
                    NeckConfig& config,
                    std::string& error_message) {
    error_message.clear();
    Values values;
    if (!readValues(path, values, error_message)) {
        return false;
    }

    const auto known = knownKeys();
    for (const auto& entry : values) {
        if (known.find(entry.first) == known.end()) {
            error_message = "unknown configuration key '" + entry.first + "'";
            return false;
        }
    }

    NeckConfig loaded;
    if (!requireString(values, "network_interface", loaded.network_interface, error_message) ||
        !requireInt(values, "slave_id", loaded.slave_id, error_message) ||
        !requireInt(values, "ack_status", loaded.ack_status, error_message) ||
        !loadMotor(values, 1, loaded.motors[0], error_message) ||
        !loadMotor(values, 2, loaded.motors[1], error_message) ||
        !loadMotor(values, 3, loaded.motors[2], error_message) ||
        !requireDouble(values, "pitch_min_deg", loaded.pitch_min_deg, error_message) ||
        !requireDouble(values, "pitch_max_deg", loaded.pitch_max_deg, error_message) ||
        !requireDouble(values, "roll_min_deg", loaded.roll_min_deg, error_message) ||
        !requireDouble(values, "roll_max_deg", loaded.roll_max_deg, error_message) ||
        !requireDouble(values, "yaw_min_deg", loaded.yaw_min_deg, error_message) ||
        !requireDouble(values, "yaw_max_deg", loaded.yaw_max_deg, error_message) ||
        !requireDouble(values, "c11", loaded.c11, error_message) ||
        !requireDouble(values, "c12", loaded.c12, error_message) ||
        !requireDouble(values, "c21", loaded.c21, error_message) ||
        !requireDouble(values, "c22", loaded.c22, error_message) ||
        !requireDouble(values, "k3", loaded.k3, error_message) ||
        !requireDouble(values, "pitch_center_deg", loaded.pitch_center_deg, error_message) ||
        !requireDouble(values, "roll_center_deg", loaded.roll_center_deg, error_message) ||
        !requireDouble(values, "yaw_center_deg", loaded.yaw_center_deg, error_message) ||
        !requireDouble(values, "det_eps", loaded.det_eps, error_message) ||
        !requireBool(values, "feedback.enabled", loaded.feedback.enabled, error_message) ||
        !requireString(values, "feedback.socket_path", loaded.feedback.socket_path, error_message) ||
        !requireDouble(values, "feedback.rate_hz", loaded.feedback.rate_hz, error_message)) {
        return false;
    }

    if (!validateConfig(loaded, error_message)) {
        return false;
    }

    config = loaded;
    return true;
}

bool loadDefaultNeckConfig(NeckConfig& config, std::string& error_message) {
    const std::array<const char*, 2> paths = {
        "neck/neck_config.py",
        "../neck/neck_config.py"
    };
    for (const char* path : paths) {
        std::ifstream probe(path);
        if (probe.good()) {
            return loadNeckConfig(path, config, error_message);
        }
    }
    error_message =
        "cannot find neck/neck_config.py (tried current and parent directories)";
    return false;
}
