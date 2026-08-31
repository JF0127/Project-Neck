#include "neck/neck_sequence.h"

#include <cmath>
#include <sstream>
#include <utility>

bool precheckNeckTrajectory(const Trajectory& trajectory,
                            const NeckConfig& config,
                            std::vector<MotorAngles>& motor_frames,
                            std::string& error_message) {
    error_message.clear();
    if (!std::isfinite(trajectory.fps) || trajectory.fps <= 0.0) {
        error_message = "trajectory fps must be finite and greater than 0";
        return false;
    }
    if (trajectory.points.empty()) {
        error_message = "trajectory must contain at least one frame";
        return false;
    }

    std::vector<MotorAngles> checked_frames;
    checked_frames.reserve(trajectory.points.size());
    for (std::size_t index = 0; index < trajectory.points.size(); ++index) {
        const TrajectoryPoint& point = trajectory.points[index];
        const NeckPose pose{point.pitch_deg, point.roll_deg, point.yaw_deg};
        MotorAngles motors{};
        const NeckKinematicsStatus status =
            inverseKinematics(pose, config, motors);
        if (status != NeckKinematicsStatus::Ok) {
            error_message = "frame " + std::to_string(index) +
                            " failed inverse kinematics: " +
                            neckKinematicsStatusString(status);
            return false;
        }

        if (!checked_frames.empty()) {
            const MotorAngles& previous = checked_frames.back();
            const double velocities[3] = {
                std::abs(motors.motor1 - previous.motor1) * trajectory.fps,
                std::abs(motors.motor2 - previous.motor2) * trajectory.fps,
                std::abs(motors.motor3 - previous.motor3) * trajectory.fps
            };
            for (std::size_t motor_index = 0;
                 motor_index < config.motors.size(); ++motor_index) {
                const double limit =
                    config.motors[motor_index].max_velocity_deg_s;
                if (!std::isfinite(velocities[motor_index]) ||
                    velocities[motor_index] > limit + 1e-9) {
                    std::ostringstream message;
                    message << "frame " << index << " motor "
                            << (motor_index + 1) << " velocity "
                            << velocities[motor_index] << " deg/s exceeds "
                            << limit << " deg/s";
                    error_message = message.str();
                    return false;
                }
            }
        }
        checked_frames.push_back(motors);
    }

    motor_frames = std::move(checked_frames);
    return true;
}
