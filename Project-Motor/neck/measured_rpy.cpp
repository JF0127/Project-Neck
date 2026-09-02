#include "neck/measured_rpy.h"

extern "C" {
#include "transmit.h"
}

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

constexpr std::size_t kMaximumRequestSize = 8192;
constexpr double kPi = 3.14159265358979323846;
std::mutex request_mutex;
std::map<std::string, MeasuredRpyRequest> pending_requests;

std::string systemError(const std::string& operation) {
    return operation + ": " + std::strerror(errno);
}

bool parseRequest(const std::string& text,
                  MeasuredRpyRequest& request,
                  std::string& error) {
    std::istringstream stream(text);
    std::string magic;
    std::string trajectory_name;
    std::string origin_text;
    std::string output_path;
    std::string extra;
    if (!std::getline(stream, magic) || magic != "NECK_MEASUREMENT_V1" ||
        !std::getline(stream, trajectory_name) || trajectory_name.empty() ||
        !std::getline(stream, origin_text) ||
        !std::getline(stream, output_path) || output_path.empty()) {
        error = "invalid measurement request";
        return false;
    }
    while (std::getline(stream, extra)) {
        if (!extra.empty()) {
            error = "measurement request contains unexpected data";
            return false;
        }
    }

    std::size_t consumed = 0;
    double turn_origin = 0.0;
    try {
        turn_origin = std::stod(origin_text, &consumed);
    } catch (const std::exception&) {
        error = "turn origin must be a finite Unix timestamp";
        return false;
    }
    if (consumed != origin_text.size() || !std::isfinite(turn_origin) ||
        turn_origin <= 0.0) {
        error = "turn origin must be a finite Unix timestamp";
        return false;
    }
    if (output_path.front() != '/') {
        error = "measurement output path must be absolute";
        return false;
    }
    const std::string filename = "measured_rpy.json";
    if (output_path.size() < filename.size() ||
        output_path.compare(output_path.size() - filename.size(),
                            filename.size(), filename) != 0) {
        error = "measurement output path must end with measured_rpy.json";
        return false;
    }

    request = {trajectory_name, output_path, turn_origin};
    return true;
}

void sendResponse(int client_fd, const std::string& response) {
    const char* data = response.data();
    std::size_t remaining = response.size();
    while (remaining > 0) {
        const ssize_t sent = send(client_fd, data, remaining, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        data += sent;
        remaining -= static_cast<std::size_t>(sent);
    }
}

}  // namespace

MeasurementSocketServer::MeasurementSocketServer(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

MeasurementSocketServer::~MeasurementSocketServer() {
    stop();
}

bool MeasurementSocketServer::removeStaleSocket(std::string& error_message) {
    struct stat status {};
    if (lstat(socket_path_.c_str(), &status) != 0) {
        if (errno == ENOENT) {
            return true;
        }
        error_message = systemError("cannot inspect measurement socket path");
        return false;
    }
    if (!S_ISSOCK(status.st_mode)) {
        error_message = "refusing to remove non-socket path: " + socket_path_;
        return false;
    }
    if (unlink(socket_path_.c_str()) != 0) {
        error_message = systemError("cannot remove stale measurement socket");
        return false;
    }
    return true;
}

void MeasurementSocketServer::removeOwnedSocket() {
    struct stat status {};
    if (lstat(socket_path_.c_str(), &status) == 0 && S_ISSOCK(status.st_mode)) {
        unlink(socket_path_.c_str());
    }
}

bool MeasurementSocketServer::start(std::string& error_message) {
    error_message.clear();
    if (running_.load()) {
        error_message = "measurement socket server is already running";
        return false;
    }
    if (socket_path_.empty()) {
        error_message = "measurement socket path must not be empty";
        return false;
    }

    sockaddr_un address {};
    address.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(address.sun_path)) {
        error_message = "measurement socket path is too long";
        return false;
    }
    if (!removeStaleSocket(error_message)) {
        return false;
    }

    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        error_message = systemError("measurement socket() failed");
        return false;
    }
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0 || listen(listen_fd_, 4) != 0) {
        error_message = systemError("cannot bind/listen measurement socket");
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        return false;
    }
    if (chmod(socket_path_.c_str(), 0666) != 0) {
        error_message = systemError("chmod() failed for measurement socket");
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        return false;
    }

    running_ = true;
    try {
        server_thread_ = std::thread(&MeasurementSocketServer::acceptLoop, this);
    } catch (const std::exception& exception) {
        running_ = false;
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        error_message = std::string("cannot start measurement socket thread: ") +
                        exception.what();
        return false;
    }
    std::cout << "[MeasuredRPY] listening on " << socket_path_ << "\n";
    return true;
}

void MeasurementSocketServer::stop() {
    if (!running_.exchange(false)) {
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        removeOwnedSocket();
        return;
    }

    const int wake_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (wake_fd >= 0) {
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        if (socket_path_.size() < sizeof(address.sun_path)) {
            std::memcpy(address.sun_path, socket_path_.c_str(),
                        socket_path_.size() + 1);
            connect(wake_fd, reinterpret_cast<sockaddr*>(&address),
                    sizeof(address));
        }
        close(wake_fd);
    }
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    removeOwnedSocket();
}

void MeasurementSocketServer::acceptLoop() {
    while (running_.load()) {
        const int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (running_.load()) {
                std::cerr << "[MeasuredRPY] " << systemError("accept() failed")
                          << "\n";
            }
            break;
        }
        if (running_.load()) {
            handleClient(client_fd);
        }
        close(client_fd);
    }
    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
    running_ = false;
    removeOwnedSocket();
}

void MeasurementSocketServer::handleClient(int client_fd) {
    std::string text;
    char buffer[1024];
    while (true) {
        const ssize_t received = recv(client_fd, buffer, sizeof(buffer), 0);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        if (text.size() + static_cast<std::size_t>(received) >
            kMaximumRequestSize) {
            sendResponse(client_fd, "ERROR request too large\n");
            return;
        }
        text.append(buffer, static_cast<std::size_t>(received));
    }

    MeasuredRpyRequest request;
    std::string error;
    if (!parseRequest(text, request, error)) {
        sendResponse(client_fd, "ERROR " + error + "\n");
        return;
    }
    {
        std::lock_guard<std::mutex> lock(request_mutex);
        pending_requests[request.trajectory_name] = request;
    }
    sendResponse(client_fd, "OK\n");
}

bool takeMeasuredRpyRequest(const std::string& trajectory_name,
                            MeasuredRpyRequest& request) {
    std::lock_guard<std::mutex> lock(request_mutex);
    const auto found = pending_requests.find(trajectory_name);
    if (found == pending_requests.end()) {
        return false;
    }
    request = std::move(found->second);
    pending_requests.erase(found);
    return true;
}

bool sampleMeasuredRpy(const NeckConfig& config,
                       double timestamp_sec,
                       std::uint64_t sequences[3],
                       MeasuredRpySample& sample) {
    int passages[3];
    int motor_ids[3];
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        passages[index] = config.motors[index].passage;
        motor_ids[index] = config.motors[index].id;
    }
    double angles_deg[3];
    if (!NeckFeedbackRead(config.slave_id, passages, motor_ids,
                          angles_deg, sequences)) {
        return false;
    }

    MotorAngles motors{angles_deg[0], angles_deg[1], angles_deg[2]};
    NeckPose pose;
    if (forwardKinematics(motors, config, pose) != NeckKinematicsStatus::Ok) {
        return false;
    }
    sample = {timestamp_sec, pose};
    return true;
}

bool writeMeasuredRpy(const MeasuredRpyRequest& request,
                      double fps,
                      double trajectory_start_sec,
                      const std::vector<MeasuredRpySample>& samples,
                      std::string& error_message) {
    error_message.clear();
    if (samples.empty()) {
        error_message = "no valid encoder feedback samples were collected";
        return false;
    }

    const std::string temporary = request.output_path + ".tmp." +
                                  std::to_string(getpid());
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output) {
        error_message = "cannot open temporary measured RPY file";
        return false;
    }

    output << std::setprecision(12)
           << "{\n"
           << "  \"name\": \"measured_rpy\",\n"
           << "  \"fps\": " << fps << ",\n"
           << "  \"unit\": \"radian\",\n"
           << "  \"order\": [\"roll\", \"pitch\", \"yaw\"],\n"
           << "  \"trajectory_start_sec\": " << trajectory_start_sec << ",\n"
           << "  \"timestamps_sec\": [\n";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        output << "    " << samples[index].timestamp_sec
               << (index + 1 == samples.size() ? "\n" : ",\n");
    }
    output << "  ],\n  \"trajectory\": [\n";
    for (std::size_t index = 0; index < samples.size(); ++index) {
        const NeckPose& pose = samples[index].pose_deg;
        output << "    [" << pose.roll * kPi / 180.0 << ", "
               << pose.pitch * kPi / 180.0 << ", "
               << pose.yaw * kPi / 180.0 << "]"
               << (index + 1 == samples.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    output.flush();
    if (!output) {
        output.close();
        std::remove(temporary.c_str());
        error_message = "failed while writing measured RPY file";
        return false;
    }
    output.close();
    if (std::rename(temporary.c_str(), request.output_path.c_str()) != 0) {
        error_message = systemError("cannot commit measured RPY file");
        std::remove(temporary.c_str());
        return false;
    }
    return true;
}
