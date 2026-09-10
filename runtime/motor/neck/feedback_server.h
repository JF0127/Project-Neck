#ifndef PROJECT_MOTOR_FEEDBACK_SERVER_H
#define PROJECT_MOTOR_FEEDBACK_SERVER_H

#include "neck/neck_config.h"
#include "neck/neck_kinematics.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// One read-only head pose snapshot sampled from the EtherCAT feedback cache.
// ``valid`` is true only when fresh three-motor feedback was available and
// forward kinematics succeeded.
struct FeedbackSample {
    bool valid = false;
    std::string reason;                    // invalid reason, safe ASCII
    double timestamp_sec = 0.0;            // Unix time seconds
    NeckPose pose_deg{};                   // pitch/roll/yaw degrees, motor order
    double motor_angles_deg[3] = {0.0, 0.0, 0.0};
    std::uint64_t sequence[3] = {0, 0, 0};
    bool motion_executing = false;
};

// Produces one sample for the given Unix timestamp. Must be cheap and must
// never touch the motor command path.
using FeedbackProvider = std::function<FeedbackSample(double timestamp_sec)>;

// Builds the production provider from the loaded neck configuration. It reads
// NeckFeedbackRead + forwardKinematics and never sends commands.
FeedbackProvider makeNeckFeedbackProvider(const NeckConfig& config);

// Pushes one NDJSON ``motor_state`` line per period to every connected Unix
// domain socket client at the configured rate. Sampling continues even when no
// client is connected; messages are discarded in that case. Read-only: this
// server never accepts commands and never publishes motor frames.
class FeedbackServer {
public:
    FeedbackServer(std::string socket_path = "/tmp/neck_feedback.sock",
                   double rate_hz = 30.0,
                   FeedbackProvider provider = {});
    ~FeedbackServer();

    FeedbackServer(const FeedbackServer&) = delete;
    FeedbackServer& operator=(const FeedbackServer&) = delete;

    bool start(std::string& error_message);
    void stop();

    bool isRunning() const;
    const std::string& socketPath() const;

private:
    void acceptLoop();
    void publishLoop();
    void broadcast(const std::string& line);
    bool removeStaleSocket(std::string& error_message);
    void removeOwnedSocket();

    std::string socket_path_;
    double rate_hz_ = 30.0;
    FeedbackProvider provider_;
    std::atomic<bool> running_{false};
    int listen_fd_ = -1;
    std::thread accept_thread_;
    std::thread publish_thread_;
    std::mutex clients_mutex_;
    std::vector<int> clients_;
};

#endif  // PROJECT_MOTOR_FEEDBACK_SERVER_H
