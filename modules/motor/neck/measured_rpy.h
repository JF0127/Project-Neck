#ifndef PROJECT_MOTOR_MEASURED_RPY_H
#define PROJECT_MOTOR_MEASURED_RPY_H

#include "neck/neck_config.h"
#include "neck/neck_kinematics.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <vector>

struct MeasuredRpyRequest {
    std::string trajectory_name;
    std::string output_path;
    double turn_origin_unix_sec = 0.0;
};

struct MeasuredRpySample {
    double timestamp_sec = 0.0;
    NeckPose pose_deg{};
};

class MeasurementSocketServer {
public:
    explicit MeasurementSocketServer(
        std::string socket_path = "/tmp/neck_measurement.sock");
    ~MeasurementSocketServer();

    MeasurementSocketServer(const MeasurementSocketServer&) = delete;
    MeasurementSocketServer& operator=(const MeasurementSocketServer&) = delete;

    bool start(std::string& error_message);
    void stop();

private:
    void acceptLoop();
    void handleClient(int client_fd);
    bool removeStaleSocket(std::string& error_message);
    void removeOwnedSocket();

    std::string socket_path_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    std::thread server_thread_;
};

bool takeMeasuredRpyRequest(const std::string& trajectory_name,
                            MeasuredRpyRequest& request);

bool sampleMeasuredRpy(const NeckConfig& config,
                       double timestamp_sec,
                       std::uint64_t sequences[3],
                       MeasuredRpySample& sample);

bool writeMeasuredRpy(const MeasuredRpyRequest& request,
                      double fps,
                      double trajectory_start_sec,
                      const std::vector<MeasuredRpySample>& samples,
                      std::string& error_message);

#endif  // PROJECT_MOTOR_MEASURED_RPY_H
