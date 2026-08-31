#ifndef PROJECT_MOTOR_MODEL_SOCKET_H
#define PROJECT_MOTOR_MODEL_SOCKET_H

#include <atomic>
#include <string>
#include <thread>

class ModelSocketServer {
public:
    explicit ModelSocketServer(
        std::string socket_path = "/tmp/neck_model.sock");
    ~ModelSocketServer();

    ModelSocketServer(const ModelSocketServer&) = delete;
    ModelSocketServer& operator=(const ModelSocketServer&) = delete;

    // Starts the background accept loop. Returns only after bind/listen has
    // succeeded. No motor or trajectory execution function is called.
    bool start(std::string& error_message);

    // Stops accepting clients, joins the background thread, and removes the
    // Unix Domain Socket file. Safe to call more than once.
    void stop();

    bool isRunning() const;
    const std::string& socketPath() const;

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

#endif  // PROJECT_MOTOR_MODEL_SOCKET_H
