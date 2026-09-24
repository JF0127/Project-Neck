#ifndef PROJECT_MOTOR_TRAJECTORY_IO_H
#define PROJECT_MOTOR_TRAJECTORY_IO_H

#include "neck/neck_kinematics.h"
#include "neck/trajectory.h"

#include <string>

// Optional model-socket message that directly requests one neck pose set.
struct NeckPoseSetRequest {
    int slave_id = 0;
    NeckPose pose{};
};

// Parses the optional neck_pose_set model-socket message. When the JSON root is
// not an object, is malformed, or has another "type", is_pose_set stays false
// and true is returned so the caller can fall back to trajectory parsing.
// Returns false only when the message is a neck_pose_set with invalid fields.
bool parseNeckPoseSetJson(const std::string& json_text,
                          NeckPoseSetRequest& request,
                          bool& is_pose_set,
                          std::string& error_message);

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
