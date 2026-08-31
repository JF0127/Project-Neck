#ifndef PROJECT_MOTOR_NECK_KINEMATICS_H
#define PROJECT_MOTOR_NECK_KINEMATICS_H

#include "neck/neck_config.h"

struct NeckPose {
    double pitch;
    double roll;
    double yaw;
};

struct MotorAngles {
    double motor1;
    double motor2;
    double motor3;
};

enum class NeckKinematicsStatus {
    Ok = 0,
    NonFiniteInput,
    Singular,
    PoseOutOfRange,
    MotorOutOfRange,
    NonFiniteResult
};

// Forward kinematics: absolute motor output angles (degree) to neck RPY
// (degree). The output is changed only when Ok is returned.
NeckKinematicsStatus forwardKinematics(const MotorAngles& motors,
                                       const NeckConfig& config,
                                       NeckPose& pose);

// Inverse kinematics: neck RPY (degree) to absolute motor output angles
// (degree). Pose limits, singularities, and motor limits are checked. The
// output is changed only when Ok is returned.
NeckKinematicsStatus inverseKinematics(const NeckPose& pose,
                                       const NeckConfig& config,
                                       MotorAngles& motors);

const char* neckKinematicsStatusString(NeckKinematicsStatus status);

#endif  // PROJECT_MOTOR_NECK_KINEMATICS_H
