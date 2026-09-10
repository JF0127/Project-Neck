#ifndef PROJECT_MOTOR_NECK_CONFIG_H
#define PROJECT_MOTOR_NECK_CONFIG_H

#include <array>
#include <string>

struct MotorConfig {
    int passage = 0;
    int id = 0;

    double min_position_deg = 0.0;
    double max_position_deg = 0.0;
    double center_position_deg = 0.0;
    double max_velocity_deg_s = 0.0;

    int speed_param = 0;
    int current_param = 0;
};

struct FeedbackConfig {
    bool enabled = true;
    std::string socket_path = "/tmp/neck_feedback.sock";
    double rate_hz = 30.0;
};

struct NeckConfig {
    std::string network_interface;
    int slave_id = 0;
    int ack_status = 0;

    std::array<MotorConfig, 3> motors{};

    double pitch_min_deg = 0.0;
    double pitch_max_deg = 0.0;
    double roll_min_deg = 0.0;
    double roll_max_deg = 0.0;
    double yaw_min_deg = 0.0;
    double yaw_max_deg = 0.0;

    double c11 = 0.0;
    double c12 = 0.0;
    double c21 = 0.0;
    double c22 = 0.0;
    double k3 = 0.0;

    double pitch_center_deg = 0.0;
    double roll_center_deg = 0.0;
    double yaw_center_deg = 0.0;
    double det_eps = 0.0;

    FeedbackConfig feedback;
};

// Loads and validates the complete configuration. On failure, config is not
// changed and error_message contains the reason. No mechanical defaults are
// applied for missing or invalid values.
bool loadNeckConfig(const std::string& path,
                    NeckConfig& config,
                    std::string& error_message);

// Resolves and loads the single neck configuration, trying
// "neck/neck_config.py" and then "../neck/neck_config.py" so the motor can be
// launched from either the repository root or its build directory.
bool loadDefaultNeckConfig(NeckConfig& config, std::string& error_message);

#endif  // PROJECT_MOTOR_NECK_CONFIG_H
