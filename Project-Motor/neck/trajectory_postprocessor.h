#ifndef PROJECT_MOTOR_TRAJECTORY_POSTPROCESSOR_H
#define PROJECT_MOTOR_TRAJECTORY_POSTPROCESSOR_H

#include "neck/neck_config.h"
#include "neck/trajectory.h"

#include <array>
#include <cstddef>
#include <string>

struct TrajectoryPostprocessStats {
    std::size_t input_frames = 0;
    std::size_t output_frames = 0;
    std::size_t inserted_frames = 0;
    double original_duration_sec = 0.0;
    double processed_duration_sec = 0.0;
    std::array<double, 3> max_motor_velocity_before{};
    std::array<double, 3> max_motor_velocity_after{};
};

// Produces a separate 30 fps trajectory by inserting linear RPY samples until
// every adjacent IK motor target satisfies the configured velocity limit.
// Inserted samples inherit the destination frame's behavior state. The input
// trajectory is never modified. No hardware access is performed.
bool postprocessNeckTrajectory(
    const Trajectory& raw,
    const NeckConfig& config,
    Trajectory& processed,
    TrajectoryPostprocessStats& stats,
    std::string& error_message);

void printTrajectoryPostprocessStats(const TrajectoryPostprocessStats& stats);

#endif  // PROJECT_MOTOR_TRAJECTORY_POSTPROCESSOR_H
