#ifndef PROJECT_MOTOR_TRAJECTORY_H
#define PROJECT_MOTOR_TRAJECTORY_H

#include <string>
#include <vector>

enum class BehaviorState {
    Speaking,
    Listening,
    Silent
};

struct TrajectoryPoint {
    double pitch_deg;
    double roll_deg;
    double yaw_deg;
};

struct Trajectory {
    std::string name;
    double fps = 0.0;
    std::vector<TrajectoryPoint> points;

    // Optional metadata. An empty vector means that the JSON omitted states.
    // When present, this vector has exactly the same size as points.
    std::vector<BehaviorState> states;
};

const char* behaviorStateString(BehaviorState state);

#endif  // PROJECT_MOTOR_TRAJECTORY_H
