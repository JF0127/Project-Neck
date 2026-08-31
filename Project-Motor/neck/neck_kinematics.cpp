#include "neck/neck_kinematics.h"

#include <cmath>

namespace {

bool isFinite(const NeckPose& pose) {
    return std::isfinite(pose.pitch) &&
           std::isfinite(pose.roll) &&
           std::isfinite(pose.yaw);
}

bool isFinite(const MotorAngles& motors) {
    return std::isfinite(motors.motor1) &&
           std::isfinite(motors.motor2) &&
           std::isfinite(motors.motor3);
}

bool poseWithinLimits(const NeckPose& pose, const NeckConfig& config) {
    return pose.pitch >= config.pitch_min_deg &&
           pose.pitch <= config.pitch_max_deg &&
           pose.roll >= config.roll_min_deg &&
           pose.roll <= config.roll_max_deg &&
           pose.yaw >= config.yaw_min_deg &&
           pose.yaw <= config.yaw_max_deg;
}

bool motorsWithinLimits(const MotorAngles& motors, const NeckConfig& config) {
    return motors.motor1 >= config.motors[0].min_position_deg &&
           motors.motor1 <= config.motors[0].max_position_deg &&
           motors.motor2 >= config.motors[1].min_position_deg &&
           motors.motor2 <= config.motors[1].max_position_deg &&
           motors.motor3 >= config.motors[2].min_position_deg &&
           motors.motor3 <= config.motors[2].max_position_deg;
}

}  // namespace

NeckKinematicsStatus forwardKinematics(const MotorAngles& motors,
                                       const NeckConfig& config,
                                       NeckPose& pose) {
    if (!isFinite(motors)) {
        return NeckKinematicsStatus::NonFiniteInput;
    }

    const double delta_motor1 =
        motors.motor1 - config.motors[0].center_position_deg;
    const double delta_motor2 =
        motors.motor2 - config.motors[1].center_position_deg;
    const double delta_motor3 =
        motors.motor3 - config.motors[2].center_position_deg;

    const NeckPose result{
        config.pitch_center_deg +
            config.c11 * delta_motor1 + config.c12 * delta_motor2,
        config.roll_center_deg +
            config.c21 * delta_motor1 + config.c22 * delta_motor2,
        config.yaw_center_deg + config.k3 * delta_motor3
    };

    if (!isFinite(result)) {
        return NeckKinematicsStatus::NonFiniteResult;
    }

    pose = result;
    return NeckKinematicsStatus::Ok;
}

NeckKinematicsStatus inverseKinematics(const NeckPose& pose,
                                       const NeckConfig& config,
                                       MotorAngles& motors) {
    if (!isFinite(pose)) {
        return NeckKinematicsStatus::NonFiniteInput;
    }

    const double determinant =
        config.c11 * config.c22 - config.c12 * config.c21;
    if (!std::isfinite(determinant) || !std::isfinite(config.k3) ||
        !std::isfinite(config.det_eps) || config.det_eps <= 0.0 ||
        std::abs(determinant) <= config.det_eps ||
        std::abs(config.k3) <= config.det_eps) {
        return NeckKinematicsStatus::Singular;
    }

    if (!poseWithinLimits(pose, config)) {
        return NeckKinematicsStatus::PoseOutOfRange;
    }

    const double delta_pitch = pose.pitch - config.pitch_center_deg;
    const double delta_roll = pose.roll - config.roll_center_deg;
    const double delta_yaw = pose.yaw - config.yaw_center_deg;

    const MotorAngles result{
        config.motors[0].center_position_deg +
            (config.c22 * delta_pitch - config.c12 * delta_roll) /
                determinant,
        config.motors[1].center_position_deg +
            (-config.c21 * delta_pitch + config.c11 * delta_roll) /
                determinant,
        config.motors[2].center_position_deg + delta_yaw / config.k3
    };

    if (!isFinite(result)) {
        return NeckKinematicsStatus::NonFiniteResult;
    }
    if (!motorsWithinLimits(result, config)) {
        return NeckKinematicsStatus::MotorOutOfRange;
    }

    motors = result;
    return NeckKinematicsStatus::Ok;
}

const char* neckKinematicsStatusString(NeckKinematicsStatus status) {
    switch (status) {
        case NeckKinematicsStatus::Ok:
            return "OK";
        case NeckKinematicsStatus::NonFiniteInput:
            return "input contains NaN or Inf";
        case NeckKinematicsStatus::Singular:
            return "kinematics is singular (det(C) or k3 is near zero)";
        case NeckKinematicsStatus::PoseOutOfRange:
            return "neck pose is outside configured limits";
        case NeckKinematicsStatus::MotorOutOfRange:
            return "motor angle is outside configured limits";
        case NeckKinematicsStatus::NonFiniteResult:
            return "kinematics produced NaN or Inf";
    }
    return "unknown kinematics status";
}
