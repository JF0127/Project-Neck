#include "neck/neck_motion.h"

#include "neck/neck_config.h"
#include "neck/neck_sequence.h"
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
        "neck/neck_config.txt",
        "../neck/neck_config.txt"
    };
    for (const char* path : paths) {
        std::ifstream probe(path);
        if (probe.good()) {
            return loadNeckConfig(path, config, error);
        }
    }
    error = "cannot find neck/neck_config.txt (tried current and parent directories)";
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
                   std::string trajectory_name, double fps, int slave_id) {
    const auto start = std::chrono::steady_clock::now();
    bool publish_failed = false;

    for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto deadline = start +
            std::chrono::duration<double>(index / fps);
        std::unique_lock<std::mutex> lock(motion_mutex);
        if (motion_condition.wait_until(lock, deadline, [] {
                return motion_stop_requested;
            })) {
            break;
        }
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
        }

        const bool stopped = motion_stop_requested;
        NeckFrameStop(slave_id);
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
    if (isTrajectoryExecuting()) {
        error_message = "another neck trajectory is already executing";
        return false;
    }

    NeckConfig config;
    if (!loadMotionConfig(config, error_message)) {
        return false;
    }

    std::vector<MotorAngles> motor_targets;
    if (!precheckNeckTrajectory(trajectory, config, motor_targets,
                                error_message)) {
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
        std::thread(runTrajectory, std::move(frames), trajectory.name,
                    trajectory.fps, slave_id).detach();
    } catch (const std::exception& exception) {
        motion_active = false;
        motion_slave = -1;
        error_message = std::string("cannot start trajectory worker: ") +
                        exception.what();
        return false;
    }
    lock.unlock();

    std::cout << "[NeckMotion] trajectory accepted: name=" << trajectory.name
              << " fps=" << trajectory.fps
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
