#include "neck/model_socket.h"

#include "neck/neck_motion.h"
#include "neck/trajectory_io.h"

#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

namespace {

constexpr std::size_t kReceiveBufferSize = 4096;
constexpr std::size_t kMaximumJsonSize = 16 * 1024 * 1024;

std::string systemError(const std::string& operation) {
    return operation + ": " + std::strerror(errno);
}

}  // namespace

ModelSocketServer::ModelSocketServer(std::string socket_path)
    : socket_path_(std::move(socket_path)) {}

ModelSocketServer::~ModelSocketServer() {
    stop();
}

bool ModelSocketServer::removeStaleSocket(std::string& error_message) {
    struct stat file_status {};
    if (lstat(socket_path_.c_str(), &file_status) != 0) {
        if (errno == ENOENT) {
            return true;
        }
        error_message = systemError("cannot inspect existing socket path");
        return false;
    }
    if (!S_ISSOCK(file_status.st_mode)) {
        error_message = "refusing to remove non-socket path: " + socket_path_;
        return false;
    }
    if (unlink(socket_path_.c_str()) != 0) {
        error_message = systemError("cannot remove stale socket");
        return false;
    }
    return true;
}

void ModelSocketServer::removeOwnedSocket() {
    struct stat file_status {};
    if (lstat(socket_path_.c_str(), &file_status) == 0 &&
        S_ISSOCK(file_status.st_mode)) {
        unlink(socket_path_.c_str());
    }
}

bool ModelSocketServer::start(std::string& error_message) {
    error_message.clear();
    if (running_.load()) {
        error_message = "model socket server is already running";
        return false;
    }
    if (socket_path_.empty()) {
        error_message = "socket path must not be empty";
        return false;
    }

    sockaddr_un address {};
    address.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(address.sun_path)) {
        error_message = "socket path is too long: " + socket_path_;
        return false;
    }
    if (!removeStaleSocket(error_message)) {
        return false;
    }

    listen_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd_ < 0) {
        error_message = systemError("socket() failed");
        return false;
    }
    std::memcpy(address.sun_path, socket_path_.c_str(), socket_path_.size() + 1);
    if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&address),
             sizeof(address)) != 0) {
        error_message = systemError("bind() failed");
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        return false;
    }
    if (listen(listen_fd_, 8) != 0) {
        error_message = systemError("listen() failed");
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        return false;
    }
    if (chmod(socket_path_.c_str(), 0666) != 0) {
        error_message = systemError("chmod() failed for model socket");
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        return false;
    }

    running_ = true;
    try {
        server_thread_ = std::thread(&ModelSocketServer::acceptLoop, this);
    } catch (const std::exception& exception) {
        running_ = false;
        close(listen_fd_);
        listen_fd_ = -1;
        removeOwnedSocket();
        error_message = std::string("cannot start model socket thread: ") +
                        exception.what();
        return false;
    }

    std::cout << "[ModelSocket] listening on " << socket_path_ << "\n";
    return true;
}

void ModelSocketServer::stop() {
    if (!running_.exchange(false)) {
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        removeOwnedSocket();
        return;
    }

    // Wake a blocking accept(). The accept loop observes running_ == false and
    // closes this connection without parsing it.
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

bool ModelSocketServer::isRunning() const {
    return running_.load();
}

const std::string& ModelSocketServer::socketPath() const {
    return socket_path_;
}

void ModelSocketServer::acceptLoop() {
    while (running_.load()) {
        const int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (running_.load()) {
                std::cerr << "[ModelSocket] " << systemError("accept() failed")
                          << "\n";
            }
            break;
        }
        if (!running_.load()) {
            close(client_fd);
            break;
        }
        handleClient(client_fd);
        close(client_fd);
    }

    if (listen_fd_ >= 0) {
        close(listen_fd_);
        listen_fd_ = -1;
    }
    running_ = false;
    removeOwnedSocket();
}

void ModelSocketServer::handleClient(int client_fd) {
    std::string json_text;
    char buffer[kReceiveBufferSize];
    while (true) {
        const ssize_t received = recv(client_fd, buffer, sizeof(buffer), 0);
        if (received == 0) {
            break;
        }
        if (received < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "[ModelSocket] " << systemError("recv() failed")
                      << "\n";
            return;
        }
        if (json_text.size() + static_cast<std::size_t>(received) >
            kMaximumJsonSize) {
            std::cerr << "[ModelSocket] rejected message: JSON exceeds "
                      << kMaximumJsonSize << " bytes\n";
            return;
        }
        json_text.append(buffer, static_cast<std::size_t>(received));
    }

    Trajectory trajectory;
    std::string error;
    if (!parseTrajectoryJson(json_text, trajectory, error)) {
        std::cerr << "[ModelSocket] rejected trajectory: " << error << "\n";
        return;
    }

    const TrajectoryPoint& first = trajectory.points.front();
    const TrajectoryPoint& last = trajectory.points.back();
    std::ostringstream summary;
    summary << std::setprecision(12)
            << "[ModelSocket] valid trajectory\n"
            << "name=" << trajectory.name << "\n"
            << "fps=" << trajectory.fps << "\n"
            << "frame count=" << trajectory.points.size() << "\n"
            << "first degree RPY: pitch=" << first.pitch_deg
            << " roll=" << first.roll_deg
            << " yaw=" << first.yaw_deg << "\n"
            << "last degree RPY: pitch=" << last.pitch_deg
            << " roll=" << last.roll_deg
            << " yaw=" << last.yaw_deg << "\n";
    std::cout << summary.str();

    if (!executeTrajectory(trajectory, error)) {
        std::cerr << "[ModelSocket] trajectory execution rejected: "
                  << error << "\n";
        return;
    }
    std::cout << "[ModelSocket] trajectory submitted to NeckMotion\n";
}
