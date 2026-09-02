#include "neck/trajectory_postprocessor.h"

#include "neck/neck_kinematics.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

namespace {

constexpr std::size_t kMaxStepsPerInputSegment = 32;
constexpr std::size_t kMaxOutputFrames = 100000;
constexpr double kVelocityTolerance = 1e-9;

bool validState(BehaviorState state) {
    switch (state) {
        case BehaviorState::Speaking:
        case BehaviorState::Listening:
        case BehaviorState::Silent:
            return true;
    }
    return false;
}

std::array<double, 3> motorValues(const MotorAngles& motors) {
    return {motors.motor1, motors.motor2, motors.motor3};
}

bool pointToMotors(const TrajectoryPoint& point,
                   const NeckConfig& config,
                   std::size_t frame_index,
                   MotorAngles& motors,
                   std::string& error_message) {
    const NeckPose pose{point.pitch_deg, point.roll_deg, point.yaw_deg};
    const NeckKinematicsStatus status = inverseKinematics(pose, config, motors);
    if (status == NeckKinematicsStatus::Ok) {
        return true;
    }
    error_message = "frame " + std::to_string(frame_index) +
                    " failed inverse kinematics during postprocess: " +
                    neckKinematicsStatusString(status);
    return false;
}

TrajectoryPoint interpolate(const TrajectoryPoint& start,
                            const TrajectoryPoint& end,
                            double weight) {
    return {
        start.pitch_deg + (end.pitch_deg - start.pitch_deg) * weight,
        start.roll_deg + (end.roll_deg - start.roll_deg) * weight,
        start.yaw_deg + (end.yaw_deg - start.yaw_deg) * weight,
    };
}

void updateMaximumVelocity(const MotorAngles& previous,
                           const MotorAngles& current,
                           double fps,
                           std::array<double, 3>& maximum) {
    const auto before = motorValues(previous);
    const auto after = motorValues(current);
    for (std::size_t index = 0; index < maximum.size(); ++index) {
        maximum[index] = std::max(
            maximum[index], std::abs(after[index] - before[index]) * fps);
    }
}

bool velocitiesWithinLimits(const MotorAngles& previous,
                            const MotorAngles& current,
                            double fps,
                            const NeckConfig& config) {
    const auto before = motorValues(previous);
    const auto after = motorValues(current);
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        const double velocity = std::abs(after[index] - before[index]) * fps;
        if (!std::isfinite(velocity) ||
            velocity > config.motors[index].max_velocity_deg_s +
                           kVelocityTolerance) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool postprocessNeckTrajectory(
    const Trajectory& raw,
    const NeckConfig& config,
    Trajectory& processed,
    TrajectoryPostprocessStats& stats,
    std::string& error_message) {
    error_message.clear();
    if (!std::isfinite(raw.fps) || raw.fps <= 0.0) {
        error_message = "trajectory fps must be finite and greater than 0";
        return false;
    }
    if (raw.points.empty()) {
        error_message = "trajectory must contain at least one frame";
        return false;
    }
    if (!raw.states.empty() && raw.states.size() != raw.points.size()) {
        error_message = "trajectory states must be empty or match trajectory length";
        return false;
    }
    for (BehaviorState state : raw.states) {
        if (!validState(state)) {
            error_message = "trajectory contains an invalid behavior state";
            return false;
        }
    }
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        if (!std::isfinite(config.motors[index].max_velocity_deg_s) ||
            config.motors[index].max_velocity_deg_s <= 0.0) {
            error_message = "motor velocity limits must be finite and greater than 0";
            return false;
        }
    }
    if (raw.points.size() >
        std::numeric_limits<std::size_t>::max() / kMaxStepsPerInputSegment) {
        error_message = "trajectory is too large to postprocess";
        return false;
    }
    const std::size_t expansion_limit =
        std::min(kMaxOutputFrames,
                 raw.points.size() * kMaxStepsPerInputSegment);

    std::vector<MotorAngles> raw_motors;
    raw_motors.reserve(raw.points.size());
    for (std::size_t index = 0; index < raw.points.size(); ++index) {
        MotorAngles motors;
        if (!pointToMotors(raw.points[index], config, index, motors,
                           error_message)) {
            return false;
        }
        raw_motors.push_back(motors);
    }

    TrajectoryPostprocessStats calculated;
    calculated.input_frames = raw.points.size();
    calculated.original_duration_sec = raw.points.size() / raw.fps;
    for (std::size_t index = 1; index < raw_motors.size(); ++index) {
        updateMaximumVelocity(raw_motors[index - 1], raw_motors[index],
                              raw.fps,
                              calculated.max_motor_velocity_before);
    }

    Trajectory result;
    result.name = raw.name;
    result.fps = raw.fps;
    result.points.reserve(std::min(expansion_limit, kMaxOutputFrames));
    result.points.push_back(raw.points.front());
    if (!raw.states.empty()) {
        result.states.reserve(std::min(expansion_limit, kMaxOutputFrames));
        result.states.push_back(raw.states.front());
    }
    MotorAngles previous_output_motors = raw_motors.front();

    for (std::size_t segment = 1; segment < raw.points.size(); ++segment) {
        const auto start_motor = motorValues(raw_motors[segment - 1]);
        const auto end_motor = motorValues(raw_motors[segment]);
        std::size_t steps = 1;
        for (std::size_t motor = 0; motor < config.motors.size(); ++motor) {
            const double required =
                std::abs(end_motor[motor] - start_motor[motor]) * raw.fps /
                config.motors[motor].max_velocity_deg_s;
            if (!std::isfinite(required) ||
                required > static_cast<double>(kMaxStepsPerInputSegment)) {
                error_message = "segment " + std::to_string(segment - 1) +
                                "->" + std::to_string(segment) +
                                " exceeds maximum expansion of " +
                                std::to_string(kMaxStepsPerInputSegment) +
                                " steps";
                return false;
            }
            steps = std::max(steps, static_cast<std::size_t>(std::ceil(required)));
        }

        bool segment_safe = false;
        std::vector<TrajectoryPoint> segment_points;
        std::vector<MotorAngles> segment_motors;
        while (steps <= kMaxStepsPerInputSegment) {
            segment_points.clear();
            segment_motors.clear();
            segment_points.reserve(steps);
            segment_motors.reserve(steps);
            MotorAngles previous = previous_output_motors;
            segment_safe = true;
            for (std::size_t step = 1; step <= steps; ++step) {
                const TrajectoryPoint point =
                    step == steps
                        ? raw.points[segment]
                        : interpolate(raw.points[segment - 1],
                                      raw.points[segment],
                                      static_cast<double>(step) / steps);
                MotorAngles motors;
                if (!pointToMotors(point, config, segment, motors,
                                   error_message)) {
                    return false;
                }
                if (!velocitiesWithinLimits(previous, motors, raw.fps,
                                             config)) {
                    segment_safe = false;
                    break;
                }
                segment_points.push_back(point);
                segment_motors.push_back(motors);
                previous = motors;
            }
            if (segment_safe) {
                break;
            }
            ++steps;
        }
        if (!segment_safe) {
            error_message = "segment " + std::to_string(segment - 1) +
                            "->" + std::to_string(segment) +
                            " remains over velocity limits after maximum expansion";
            return false;
        }
        if (result.points.size() > expansion_limit - segment_points.size()) {
            error_message = "postprocessed trajectory exceeds output frame limit";
            return false;
        }

        for (std::size_t index = 0; index < segment_points.size(); ++index) {
            updateMaximumVelocity(previous_output_motors,
                                  segment_motors[index], raw.fps,
                                  calculated.max_motor_velocity_after);
            result.points.push_back(segment_points[index]);
            if (!raw.states.empty()) {
                result.states.push_back(raw.states[segment]);
            }
            previous_output_motors = segment_motors[index];
        }
    }

    calculated.output_frames = result.points.size();
    calculated.inserted_frames =
        calculated.output_frames - calculated.input_frames;
    calculated.processed_duration_sec = result.points.size() / result.fps;

    processed = std::move(result);
    stats = calculated;
    return true;
}

void printTrajectoryPostprocessStats(const TrajectoryPostprocessStats& stats) {
    std::cout << "[TrajectoryPostprocessor]\n"
              << "input_frames=" << stats.input_frames << "\n"
              << "output_frames=" << stats.output_frames << "\n"
              << "inserted_frames=" << stats.inserted_frames << "\n"
              << "duration: " << stats.original_duration_sec << "s -> "
              << stats.processed_duration_sec << "s\n"
              << "max motor velocity before: M1="
              << stats.max_motor_velocity_before[0] << " M2="
              << stats.max_motor_velocity_before[1] << " M3="
              << stats.max_motor_velocity_before[2] << " deg/s\n"
              << "max motor velocity after: M1="
              << stats.max_motor_velocity_after[0] << " M2="
              << stats.max_motor_velocity_after[1] << " M3="
              << stats.max_motor_velocity_after[2] << " deg/s\n";
}
