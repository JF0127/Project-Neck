#ifndef PROJECT_MOTOR_NECK_MOTION_H
#define PROJECT_MOTOR_NECK_MOTION_H

#include "neck/trajectory.h"

#include <string>

// Prechecks and asynchronously executes one trajectory using the configured
// neck hardware. Returns true only when the complete trajectory has passed
// validation and the execution worker has been started.
bool executeTrajectory(const Trajectory& trajectory,
                       std::string& error_message);

// Stops the currently executing trajectory and publishes one complete
// three-motor brake frame. Returns false when no trajectory is active or when
// the stop frame cannot be published.
bool stopTrajectory(std::string& error_message);

bool isTrajectoryExecuting();

#endif  // PROJECT_MOTOR_NECK_MOTION_H
