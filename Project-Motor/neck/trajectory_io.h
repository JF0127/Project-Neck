#ifndef PROJECT_MOTOR_TRAJECTORY_IO_H
#define PROJECT_MOTOR_TRAJECTORY_IO_H

#include "neck/trajectory.h"

#include <string>

// Parses the single supported trajectory JSON protocol. Input samples must be
// radians in [roll, pitch, yaw] order. Output points are degrees with explicit
// pitch/roll/yaw fields. On failure, trajectory is not changed.
bool parseTrajectoryJson(const std::string& json_text,
                         Trajectory& trajectory,
                         std::string& error_message);

// Reads a file and then applies parseTrajectoryJson().
bool loadTrajectoryJson(const std::string& path,
                         Trajectory& trajectory,
                         std::string& error_message);

#endif  // PROJECT_MOTOR_TRAJECTORY_IO_H
