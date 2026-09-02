#include "neck/neck_motion.h"

#include "neck/measured_rpy.h"
#include "neck/neck_config.h"
#include "neck/neck_sequence.h"
#include "neck/trajectory_postprocessor.h"
#include "queue.h"

extern "C" {
#include "ethercat.h"
#include "motor_control.h"
}

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

std::mutex motion_mutex;
std::condition_variable motion_condition;
bool motion_active = false;
bool motion_stop_requested = false;
int motion_slave = -1;
EtherCAT_Msg motion_stop_frame{};

bool loadMotionConfig(NeckConfig& config, std::string& error) {
    const std::array<const char*, 2> paths = {
        "neck/neck_config.py",
        "../neck/neck_config.py"
    };
    for (const char* path : paths) {
        std::ifstream probe(path);
        if (probe.good()) {
            return loadNeckConfig(path, config, error);
        }
    }
    error = "cannot find neck/neck_config.py (tried current and parent directories)";
    return false;
}

bool validMotionSlave(int slave_id, std::string& error) {
    if (!running) {
        error = "EtherCAT runtime is not running";
        return false;
    }
    if (ec_slavecount <= 0) {
        error = "no EtherCAT slave is available";
        return false;
    }
    if (slave_id < 0 || slave_id >= ec_slavecount || slave_id >= SLAVE_NUMBER) {
        error = "configured slave_id must be in [0, " +
                std::to_string(std::min(ec_slavecount, SLAVE_NUMBER) - 1) + "]";
        return false;
    }
    return true;
}

EtherCAT_Msg makePositionFrame(const MotorAngles& targets,
                               const NeckConfig& config) {
    EtherCAT_Msg frame{};
    const double positions[] = {
        targets.motor1, targets.motor2, targets.motor3
    };
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        const MotorConfig& motor = config.motors[index];
        set_motor_position(&frame, motor.passage, motor.id,
                           static_cast<float>(positions[index]),
                           motor.speed_param, motor.current_param,
                           config.ack_status);
    }
    return frame;
}

EtherCAT_Msg makeStopFrame(const NeckConfig& config) {
    EtherCAT_Msg frame{};
    for (const MotorConfig& motor : config.motors) {
        set_motor_cur_tor(&frame, motor.passage, motor.id, 10, 2, 0);
    }
    return frame;
}

void runTrajectory(std::vector<EtherCAT_Msg> frames,
                   std::string trajectory_name, double fps, int slave_id,
                   NeckConfig config, bool measurement_enabled,
                   MeasuredRpyRequest measurement_request) {
    const auto start = std::chrono::steady_clock::now();
    const double trajectory_start_sec =
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count() -
        measurement_request.turn_origin_unix_sec;
    bool publish_failed = false;
    std::vector<MeasuredRpySample> measured_samples;
    std::uint64_t last_sequences[3]{};

    const auto capture_feedback = [&] {
        if (!measurement_enabled) {
            return;
        }
        const double timestamp_sec =
            std::chrono::duration<double>(
                std::chrono::system_clock::now().time_since_epoch()).count() -
            measurement_request.turn_origin_unix_sec;
        std::uint64_t sequences[3]{};
        MeasuredRpySample sample;
        if (!sampleMeasuredRpy(config, timestamp_sec, sequences, sample)) {
            return;
        }
        bool all_new = measured_samples.empty();
        if (!all_new) {
            all_new = true;
            for (int index = 0; index < 3; ++index) {
                all_new = all_new && sequences[index] != last_sequences[index];
            }
        }
        if (!all_new) {
            return;
        }
        measured_samples.push_back(sample);
        for (int index = 0; index < 3; ++index) {
            last_sequences[index] = sequences[index];
        }
    };

    for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto deadline = start +
            std::chrono::duration<double>(index / fps);
        std::unique_lock<std::mutex> lock(motion_mutex);
        if (motion_condition.wait_until(lock, deadline, [] {
                return motion_stop_requested;
            })) {
            break;
        }
        capture_feedback();
        if (!NeckFramePublish(slave_id, &frames[index])) {
            std::cerr << "[NeckMotion] publish failed at frame " << index
                      << "\n";
            publish_failed = true;
            motion_stop_requested = true;
            break;
        }
    }

    {
        std::unique_lock<std::mutex> lock(motion_mutex);
        if (!motion_stop_requested && !publish_failed) {
            const auto end_time = start +
                std::chrono::duration<double>(frames.size() / fps);
            motion_condition.wait_until(lock, end_time, [] {
                return motion_stop_requested;
            });
            capture_feedback();
        }

        const bool stopped = motion_stop_requested;
        NeckFrameStop(slave_id);
        if (measurement_enabled) {
            std::string measurement_error;
            if (writeMeasuredRpy(measurement_request, fps,
                                 trajectory_start_sec, measured_samples,
                                 measurement_error)) {
                std::cout << "[MeasuredRPY] saved "
                          << measured_samples.size() << " samples to "
                          << measurement_request.output_path << "\n";
            } else {
                std::cerr << "[MeasuredRPY] not saved: "
                          << measurement_error << "\n";
            }
        }
        motion_slave = -1;
        motion_active = false;
        lock.unlock();
        motion_condition.notify_all();

        if (publish_failed) {
            std::cerr << "[NeckMotion] trajectory aborted: "
                      << trajectory_name << "\n";
        } else if (stopped) {
            std::cout << "[NeckMotion] trajectory stopped: "
                      << trajectory_name << "\n";
        } else {
            std::cout << "[NeckMotion] trajectory completed: "
                      << trajectory_name << "\n";
        }
    }
}

}  // namespace

bool executeTrajectory(const Trajectory& trajectory,
                       std::string& error_message) {
    error_message.clear();
    MeasuredRpyRequest measurement_request;
    const bool measurement_enabled =
        takeMeasuredRpyRequest(trajectory.name, measurement_request);
    if (isTrajectoryExecuting()) {
        error_message = "another neck trajectory is already executing";
        return false;
    }

    NeckConfig config;
    if (!loadMotionConfig(config, error_message)) {
        return false;
    }

    Trajectory processed_trajectory;
    TrajectoryPostprocessStats postprocess_stats;
    if (!postprocessNeckTrajectory(trajectory, config, processed_trajectory,
                                   postprocess_stats, error_message)) {
        error_message = "trajectory postprocess failed: " + error_message;
        return false;
    }
    printTrajectoryPostprocessStats(postprocess_stats);

    std::vector<MotorAngles> motor_targets;
    if (!precheckNeckTrajectory(processed_trajectory, config, motor_targets,
                                error_message)) {
        error_message = "postprocessed trajectory precheck failed: " +
                        error_message;
        return false;
    }

    const int slave_id = config.slave_id;
    if (!validMotionSlave(slave_id, error_message)) {
        return false;
    }

    std::vector<EtherCAT_Msg> frames;
    frames.reserve(motor_targets.size());
    for (const MotorAngles& targets : motor_targets) {
        frames.push_back(makePositionFrame(targets, config));
    }

    std::unique_lock<std::mutex> lock(motion_mutex);
    if (motion_active) {
        error_message = "another neck trajectory is already executing";
        return false;
    }

    motion_active = true;
    motion_stop_requested = false;
    motion_slave = slave_id;
    motion_stop_frame = makeStopFrame(config);
    try {
        std::thread(runTrajectory, std::move(frames), processed_trajectory.name,
                    processed_trajectory.fps, slave_id, config, measurement_enabled,
                    std::move(measurement_request)).detach();
    } catch (const std::exception& exception) {
        motion_active = false;
        motion_slave = -1;
        error_message = std::string("cannot start trajectory worker: ") +
                        exception.what();
        return false;
    }
    lock.unlock();

    std::cout << "[NeckMotion] trajectory accepted: name=" << processed_trajectory.name
              << " fps=" << processed_trajectory.fps
              << " frames=" << motor_targets.size() << "\n";
    return true;
}

bool stopTrajectory(std::string& error_message) {
    error_message.clear();
    std::unique_lock<std::mutex> lock(motion_mutex);
    if (!motion_active) {
        error_message = "no neck trajectory is executing";
        return false;
    }

    motion_stop_requested = true;
    motion_condition.notify_all();

    const int slave_id = motion_slave;
    const bool brake_published =
        NeckFramePublish(slave_id, &motion_stop_frame);
    if (brake_published) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    NeckFrameStop(slave_id);

    motion_condition.wait(lock, [] { return !motion_active; });
    if (!brake_published) {
        error_message = "failed to publish three-motor stop frame";
        return false;
    }
    return true;
}

bool isTrajectoryExecuting() {
    std::lock_guard<std::mutex> lock(motion_mutex);
    return motion_active;
}
