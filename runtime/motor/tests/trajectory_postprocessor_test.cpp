#include "neck/neck_config.h"
#include "neck/neck_sequence.h"
#include "neck/trajectory_io.h"
#include "neck/trajectory_postprocessor.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[FAIL] " << message << "\n";
        return false;
    }
    return true;
}

Trajectory makeTrajectory(std::vector<TrajectoryPoint> points,
                          std::vector<BehaviorState> states = {}) {
    Trajectory trajectory;
    trajectory.name = "postprocessor_test";
    trajectory.fps = 30.0;
    trajectory.points = std::move(points);
    trajectory.states = std::move(states);
    return trajectory;
}

bool samePoints(const std::vector<TrajectoryPoint>& left,
                const std::vector<TrajectoryPoint>& right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].pitch_deg != right[index].pitch_deg ||
            left[index].roll_deg != right[index].roll_deg ||
            left[index].yaw_deg != right[index].yaw_deg) {
            return false;
        }
    }
    return true;
}

bool precheckPasses(const Trajectory& trajectory, const NeckConfig& config) {
    std::vector<MotorAngles> motors;
    std::string error;
    return precheckNeckTrajectory(trajectory, config, motors, error);
}

bool withinLimits(const TrajectoryPostprocessStats& stats,
                  const NeckConfig& config) {
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        if (stats.max_motor_velocity_after[index] >
            config.motors[index].max_velocity_deg_s + 1e-9) {
            return false;
        }
    }
    return true;
}

void writeTrajectory(const std::filesystem::path& path,
                     const Trajectory& trajectory) {
    std::ofstream output(path);
    output << std::setprecision(12)
           << "{\n  \"name\": \"" << trajectory.name << "\",\n"
           << "  \"fps\": " << trajectory.fps << ",\n"
           << "  \"unit\": \"radian\",\n"
           << "  \"order\": [\"roll\", \"pitch\", \"yaw\"],\n"
           << "  \"trajectory\": [\n";
    for (std::size_t index = 0; index < trajectory.points.size(); ++index) {
        const TrajectoryPoint& point = trajectory.points[index];
        output << "    [" << point.roll_deg * kPi / 180.0 << ", "
               << point.pitch_deg * kPi / 180.0 << ", "
               << point.yaw_deg * kPi / 180.0 << "]"
               << (index + 1 == trajectory.points.size() ? "\n" : ",\n");
    }
    output << "  ]";
    if (!trajectory.states.empty()) {
        output << ",\n  \"states\": [\n";
        for (std::size_t index = 0; index < trajectory.states.size(); ++index) {
            output << "    \"" << behaviorStateString(trajectory.states[index])
                   << "\""
                   << (index + 1 == trajectory.states.size() ? "\n" : ",\n");
        }
        output << "  ]";
    }
    output << "\n}\n";
}

void writeStats(const std::filesystem::path& path,
                const TrajectoryPostprocessStats& stats) {
    std::ofstream output(path);
    output << std::setprecision(12)
           << "{\n"
           << "  \"input_frames\": " << stats.input_frames << ",\n"
           << "  \"output_frames\": " << stats.output_frames << ",\n"
           << "  \"inserted_frames\": " << stats.inserted_frames << ",\n"
           << "  \"original_duration_sec\": " << stats.original_duration_sec << ",\n"
           << "  \"processed_duration_sec\": " << stats.processed_duration_sec << ",\n"
           << "  \"max_motor_velocity_before_deg_s\": ["
           << stats.max_motor_velocity_before[0] << ", "
           << stats.max_motor_velocity_before[1] << ", "
           << stats.max_motor_velocity_before[2] << "],\n"
           << "  \"max_motor_velocity_after_deg_s\": ["
           << stats.max_motor_velocity_after[0] << ", "
           << stats.max_motor_velocity_after[1] << ", "
           << stats.max_motor_velocity_after[2] << "]\n"
           << "}\n";
}

bool testAlreadySafe(const NeckConfig& config) {
    const Trajectory raw = makeTrajectory(
        {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.5}},
        {BehaviorState::Silent, BehaviorState::Speaking});
    Trajectory processed;
    TrajectoryPostprocessStats stats;
    std::string error;
    return require(postprocessNeckTrajectory(raw, config, processed, stats, error),
                   "safe trajectory postprocess: " + error) &&
           require(samePoints(processed.points, raw.points),
                   "safe trajectory must not gain or change points") &&
           require(processed.states == raw.states,
                   "safe trajectory states must remain unchanged") &&
           require(stats.inserted_frames == 0,
                   "safe trajectory must insert zero frames") &&
           require(precheckPasses(processed, config),
                   "safe processed trajectory must pass precheck");
}

bool testKnownOverspeed(const NeckConfig& config) {
    const Trajectory raw = makeTrajectory(
        {{0.0, 0.0, 0.0}, {0.5508787156, 2.7091106510, 0.0312534515}},
        {BehaviorState::Silent, BehaviorState::Speaking});
    Trajectory processed;
    TrajectoryPostprocessStats stats;
    std::string error;
    if (!require(!precheckPasses(raw, config),
                 "known overspeed raw trajectory must fail precheck") ||
        !require(postprocessNeckTrajectory(raw, config, processed, stats, error),
                 "known overspeed postprocess: " + error) ||
        !require(processed.points.size() > raw.points.size(),
                 "known overspeed trajectory must gain frames") ||
        !require(precheckPasses(processed, config),
                 "known overspeed processed trajectory must pass precheck") ||
        !require(withinLimits(stats, config),
                 "known overspeed processed velocities must satisfy limits") ||
        !require(processed.states.size() == processed.points.size(),
                 "processed states must match processed points")) {
        return false;
    }
    for (std::size_t index = 1; index < processed.states.size(); ++index) {
        if (!require(processed.states[index] == BehaviorState::Speaking,
                     "inserted states must inherit destination state")) {
            return false;
        }
    }
    return true;
}

bool testInvalidInputs(const NeckConfig& config) {
    std::string error;
    Trajectory processed;
    TrajectoryPostprocessStats stats;

    const Trajectory empty = makeTrajectory({});
    if (!require(!postprocessNeckTrajectory(empty, config, processed, stats, error),
                 "empty trajectory must be rejected")) {
        return false;
    }
    const Trajectory nan = makeTrajectory(
        {{0.0, 0.0, 0.0}, {0.0, std::numeric_limits<double>::quiet_NaN(), 0.0}});
    if (!require(!postprocessNeckTrajectory(nan, config, processed, stats, error),
                 "NaN trajectory must be rejected")) {
        return false;
    }
    const Trajectory infinity = makeTrajectory(
        {{0.0, 0.0, 0.0}, {0.0, 0.0, std::numeric_limits<double>::infinity()}});
    if (!require(!postprocessNeckTrajectory(infinity, config, processed, stats, error),
                 "Inf trajectory must be rejected")) {
        return false;
    }
    const Trajectory ik_fail = makeTrajectory({{100.0, 0.0, 0.0}});
    if (!require(!postprocessNeckTrajectory(ik_fail, config, processed, stats, error),
                 "IK-invalid trajectory must be rejected")) {
        return false;
    }
    const Trajectory invalid_state = makeTrajectory(
        {{0.0, 0.0, 0.0}}, {static_cast<BehaviorState>(99)});
    if (!require(!postprocessNeckTrajectory(invalid_state, config, processed,
                                            stats, error),
                 "invalid state must be rejected")) {
        return false;
    }
    const Trajectory extreme = makeTrajectory(
        {{0.0, 0.0, -80.0}, {0.0, 0.0, 50.0}});
    return require(!postprocessNeckTrajectory(extreme, config, processed, stats, error),
                   "extreme expansion trajectory must be rejected");
}

bool testLatestFixture(const NeckConfig& config,
                       const std::string& fixture_path,
                       const std::string& output_dir) {
    Trajectory raw;
    std::string error;
    if (!require(loadTrajectoryJson(fixture_path, raw, error),
                 "load latest fixture: " + error)) {
        return false;
    }
    std::vector<MotorAngles> raw_motors;
    std::string precheck_error;
    if (!require(!precheckNeckTrajectory(raw, config, raw_motors,
                                         precheck_error),
                 "latest raw fixture must fail precheck") ||
        !require(precheck_error.find("velocity") != std::string::npos,
                 "latest raw fixture must fail due to velocity")) {
        return false;
    }

    Trajectory processed;
    TrajectoryPostprocessStats stats;
    if (!require(postprocessNeckTrajectory(raw, config, processed, stats, error),
                 "latest fixture postprocess: " + error) ||
        !require(stats.max_motor_velocity_before[0] > 100.0,
                 "latest fixture must reproduce >100 deg/s motor velocity") ||
        !require(processed.points.size() > raw.points.size(),
                 "latest fixture must gain frames") ||
        !require(withinLimits(stats, config),
                 "latest fixture processed velocities must satisfy limits") ||
        !require(precheckPasses(processed, config),
                 "latest fixture processed trajectory must pass precheck") ||
        !require(processed.states.size() == processed.points.size(),
                 "latest fixture states must remain aligned")) {
        return false;
    }

    printTrajectoryPostprocessStats(stats);
    if (!output_dir.empty()) {
        const std::filesystem::path directory(output_dir);
        std::filesystem::create_directories(directory);
        writeTrajectory(directory / "raw_rpy.json", raw);
        processed.name = "latest_turn_003_processed";
        writeTrajectory(directory / "processed_rpy.json", processed);
        writeStats(directory / "stats.json", stats);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: trajectory_postprocessor_test <neck_config> "
                  << "<latest_fixture> [verification_output_dir]\n";
        return 2;
    }

    NeckConfig config;
    std::string error;
    if (!loadNeckConfig(argv[1], config, error)) {
        std::cerr << "cannot load neck config: " << error << "\n";
        return 2;
    }

    const std::string output_dir = argc == 4 ? argv[3] : "";
    if (!testAlreadySafe(config) ||
        !testKnownOverspeed(config) ||
        !testInvalidInputs(config) ||
        !testLatestFixture(config, argv[2], output_dir)) {
        return 1;
    }
    std::cout << "[PASS] TrajectoryPostprocessor V0 software tests\n";
    return 0;
}
