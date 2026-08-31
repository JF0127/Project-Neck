#ifndef PROJECT_MOTOR_NECK_SEQUENCE_H
#define PROJECT_MOTOR_NECK_SEQUENCE_H

#include "neck/neck_config.h"
#include "neck/neck_kinematics.h"
#include "neck/trajectory.h"

#include <string>
#include <vector>

// Converts every trajectory point to motor angles and validates all adjacent
// frame velocities before execution. On failure, motor_frames is not changed.
// This function performs no hardware access and sends no motor commands.
bool precheckNeckTrajectory(const Trajectory& trajectory,
                            const NeckConfig& config,
                            std::vector<MotorAngles>& motor_frames,
                            std::string& error_message);

#endif  // PROJECT_MOTOR_NECK_SEQUENCE_H
