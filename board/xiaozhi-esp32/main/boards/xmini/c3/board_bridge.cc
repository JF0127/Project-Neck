#include "board_bridge.h"

#include "protocol.h"

#include <cJSON.h>
#include <esp_log.h>

#include <lwip/inet.h>
#include <lwip/sockets.h>

#include <errno.h>
#include <cstring>
#include <memory>
#include <new>
#include <unistd.h>

#include <wifi_manager.h>

namespace {

constexpr char kUbuntuIp[] = "192.168.110.234";
constexpr uint16_t kUbuntuPort = 8766;
constexpr UBaseType_t kTextQueueCapacity = 12;
constexpr UBaseType_t kUserAudioQueueCapacity = 4;
constexpr UBaseType_t kRobotAudioQueueCapacity = 4;
constexpr size_t kMaxAudioPayload = 8192;
constexpr uint8_t kRobotAudioChannels = 1;
constexpr TickType_t kWifiCheckDelay = pdMS_TO_TICKS(500);
constexpr TickType_t kReconnectDelay = pdMS_TO_TICKS(2000);
constexpr char kTag[] = "BoardBridge";

bool SendAll(int sock, const char* data, size_t length) {
    size_t sent_total = 0;
    while (sent_total < length) {
        int sent = send(sock, data + sent_total, length - sent_total, 0);
        if (sent <= 0) {
            return false;
        }
        sent_total += static_cast<size_t>(sent);
    }
    return true;
}

int ConnectToUbuntu() {
    int sock = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    if (sock < 0) {
        ESP_LOGW(kTag, "socket() failed, errno=%d", errno);
        return -1;
    }

    struct timeval timeout = {};
    timeout.tv_sec = 2;
    if (setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        ESP_LOGW(kTag, "Failed to set send timeout, errno=%d", errno);
    }

    struct sockaddr_in dest_addr = {};
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(kUbuntuPort);
    if (inet_pton(AF_INET, kUbuntuIp, &dest_addr.sin_addr) != 1) {
        ESP_LOGE(kTag, "Invalid Ubuntu IP");
        close(sock);
        return -1;
    }

    if (connect(sock, reinterpret_cast<struct sockaddr*>(&dest_addr), sizeof(dest_addr)) != 0) {
        ESP_LOGW(kTag, "Connect failed, errno=%d", errno);
        close(sock);
        return -1;
    }

    ESP_LOGI(kTag, "Connected to %s:%u", kUbuntuIp, kUbuntuPort);
    return sock;
}

bool SendUserAudioFrame(int sock, uint32_t sequence, const std::vector<uint8_t>& payload) {
    uint8_t header[9] = {0x01};
    uint32_t sequence_be = htonl(sequence);
    uint32_t length_be = htonl(static_cast<uint32_t>(payload.size()));
    memcpy(header + 1, &sequence_be, sizeof(sequence_be));
    memcpy(header + 5, &length_be, sizeof(length_be));
    return SendAll(sock, reinterpret_cast<const char*>(header), sizeof(header)) &&
           SendAll(sock, reinterpret_cast<const char*>(payload.data()), payload.size());
}

bool SendRobotAudioFrame(int sock, uint32_t sequence, uint32_t sample_rate,
                         uint16_t frame_duration_ms, uint8_t channels,
                         const std::vector<uint8_t>& payload) {
    uint8_t header[16] = {0x02};
    uint32_t sequence_be = htonl(sequence);
    uint32_t length_be = htonl(static_cast<uint32_t>(payload.size()));
    uint32_t sample_rate_be = htonl(sample_rate);
    uint16_t frame_duration_be = htons(frame_duration_ms);
    memcpy(header + 1, &sequence_be, sizeof(sequence_be));
    memcpy(header + 5, &length_be, sizeof(length_be));
    memcpy(header + 9, &sample_rate_be, sizeof(sample_rate_be));
    memcpy(header + 13, &frame_duration_be, sizeof(frame_duration_be));
    header[15] = channels;
    return SendAll(sock, reinterpret_cast<const char*>(header), sizeof(header)) &&
           SendAll(sock, reinterpret_cast<const char*>(payload.data()), payload.size());
}

std::string SerializeText(const char* type, const std::string& text) {
    cJSON* root = cJSON_CreateObject();
    if (root == nullptr || cJSON_AddStringToObject(root, "type", type) == nullptr ||
        cJSON_AddStringToObject(root, "text", text.c_str()) == nullptr) {
        cJSON_Delete(root);
        return {};
    }

    char* json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    if (json == nullptr) {
        return {};
    }

    std::string message(json);
    cJSON_free(json);
    message.push_back('\n');
    return message;
}

}  // namespace

BoardBridge& BoardBridge::GetInstance() {
    static BoardBridge instance;
    return instance;
}

void BoardBridge::Start() {
    if (task_ != nullptr) {
        return;
    }

    text_queue_ = xQueueCreate(kTextQueueCapacity, sizeof(Message*));
    user_audio_queue_ = xQueueCreate(kUserAudioQueueCapacity, sizeof(AudioMessage*));
    robot_audio_queue_ = xQueueCreate(kRobotAudioQueueCapacity, sizeof(RobotAudioMessage*));
    if (text_queue_ == nullptr || user_audio_queue_ == nullptr || robot_audio_queue_ == nullptr) {
        ESP_LOGE(kTag, "Failed to create bridge queues");
        if (text_queue_ != nullptr) {
            vQueueDelete(text_queue_);
            text_queue_ = nullptr;
        }
        if (user_audio_queue_ != nullptr) {
            vQueueDelete(user_audio_queue_);
            user_audio_queue_ = nullptr;
        }
        if (robot_audio_queue_ != nullptr) {
            vQueueDelete(robot_audio_queue_);
            robot_audio_queue_ = nullptr;
        }
        return;
    }

    if (xTaskCreate(SenderTaskEntry, "board_bridge", 4096, this, 1, &task_) != pdPASS) {
        ESP_LOGE(kTag, "Failed to create sender task");
        vQueueDelete(text_queue_);
        vQueueDelete(user_audio_queue_);
        vQueueDelete(robot_audio_queue_);
        text_queue_ = nullptr;
        user_audio_queue_ = nullptr;
        robot_audio_queue_ = nullptr;
    }
}

bool BoardBridge::EnqueueUserText(const std::string& text) {
    return EnqueueText(MessageType::UserText, text);
}

bool BoardBridge::EnqueueRobotText(const std::string& text) {
    return EnqueueText(MessageType::RobotText, text);
}

bool BoardBridge::EnqueueText(MessageType type, const std::string& text) {
    if (text_queue_ == nullptr) {
        return false;
    }

    auto* message = new (std::nothrow) Message{type, text};
    if (message != nullptr && xQueueSend(text_queue_, &message, 0) == pdPASS) {
        if (task_ != nullptr) {
            xTaskNotifyGive(task_);
        }
        return true;
    }

    delete message;
    uint32_t dropped = dropped_count_.fetch_add(1) + 1;
    if (dropped == 1 || dropped % 32 == 0) {
        const char* type_name = "robot_audio_end";
        if (type == MessageType::UserText) {
            type_name = "user_text";
        } else if (type == MessageType::RobotText) {
            type_name = "robot_text";
        }
        ESP_LOGW(kTag, "Dropped %s, total=%lu", type_name, static_cast<unsigned long>(dropped));
    }
    return false;
}

bool BoardBridge::EnqueueUserAudio(const AudioStreamPacket& packet) {
    uint32_t sequence = user_audio_sequence_.fetch_add(1);
    if (!connected_.load() || user_audio_queue_ == nullptr ||
        uxQueueSpacesAvailable(user_audio_queue_) == 0 || packet.payload.empty() ||
        packet.payload.size() > kMaxAudioPayload) {
        dropped_count_.fetch_add(1);
        return false;
    }

    auto* message = new (std::nothrow) AudioMessage{sequence, packet.payload};
    if (message != nullptr && xQueueSend(user_audio_queue_, &message, 0) == pdPASS) {
        if (task_ != nullptr) {
            xTaskNotifyGive(task_);
        }
        return true;
    }

    delete message;
    dropped_count_.fetch_add(1);
    return false;
}

bool BoardBridge::EnqueueRobotAudio(const AudioStreamPacket& packet) {
    uint32_t sequence = robot_audio_sequence_.fetch_add(1);
    if (!connected_.load() || robot_audio_queue_ == nullptr ||
        uxQueueSpacesAvailable(robot_audio_queue_) == 0 || packet.payload.empty() ||
        packet.payload.size() > kMaxAudioPayload || packet.sample_rate <= 0 ||
        packet.frame_duration <= 0 || packet.frame_duration > UINT16_MAX) {
        dropped_count_.fetch_add(1);
        return false;
    }

    auto* message = new (std::nothrow) RobotAudioMessage{
        sequence,
        static_cast<uint32_t>(packet.sample_rate),
        static_cast<uint16_t>(packet.frame_duration),
        kRobotAudioChannels,
        packet.payload,
    };
    if (message != nullptr && xQueueSend(robot_audio_queue_, &message, 0) == pdPASS) {
        if (task_ != nullptr) {
            xTaskNotifyGive(task_);
        }
        return true;
    }

    delete message;
    dropped_count_.fetch_add(1);
    return false;
}

bool BoardBridge::EnqueueRobotAudioEnd() {
    return EnqueueText(MessageType::RobotAudioEnd, {});
}

void BoardBridge::SenderTaskEntry(void* arg) {
    static_cast<BoardBridge*>(arg)->SenderTask();
}

void BoardBridge::ClearUserAudioQueue() {
    AudioMessage* message = nullptr;
    while (xQueueReceive(user_audio_queue_, &message, 0) == pdPASS) {
        delete message;
    }
}

void BoardBridge::ClearRobotAudioQueue() {
    RobotAudioMessage* message = nullptr;
    while (xQueueReceive(robot_audio_queue_, &message, 0) == pdPASS) {
        delete message;
    }
}

void BoardBridge::SenderTask() {
    int sock = -1;
    bool prefer_user_audio = true;

    while (true) {
        if (!WifiManager::GetInstance().IsConnected()) {
            connected_.store(false);
            if (sock >= 0) {
                shutdown(sock, SHUT_RDWR);
                close(sock);
                sock = -1;
            }
            ClearUserAudioQueue();
            ClearRobotAudioQueue();
            vTaskDelay(kWifiCheckDelay);
            continue;
        }

        if (sock < 0) {
            connected_.store(false);
            sock = ConnectToUbuntu();
            if (sock < 0) {
                ClearUserAudioQueue();
                ClearRobotAudioQueue();
                vTaskDelay(kReconnectDelay);
                continue;
            }
            connected_.store(true);
        }

        Message* text_message = nullptr;
        AudioMessage* user_audio_message = nullptr;
        RobotAudioMessage* robot_audio_message = nullptr;
        bool send_ok = true;

        if (xQueueReceive(text_queue_, &text_message, 0) == pdPASS && text_message != nullptr) {
            std::unique_ptr<Message> queued(text_message);
            const char* type = "robot_audio_end";
            std::string message;
            if (queued->type == MessageType::UserText) {
                type = "user_text";
                message = SerializeText(type, queued->text);
            } else if (queued->type == MessageType::RobotText) {
                type = "robot_text";
                message = SerializeText(type, queued->text);
            } else {
                message = "{\"type\":\"robot_audio_end\"}\n";
            }
            if (message.empty()) {
                ESP_LOGW(kTag, "Failed to serialize %s", type);
                continue;
            }
            send_ok = SendAll(sock, message.data(), message.size());
        } else {
            if (prefer_user_audio) {
                xQueueReceive(user_audio_queue_, &user_audio_message, 0);
                if (user_audio_message == nullptr) {
                    xQueueReceive(robot_audio_queue_, &robot_audio_message, 0);
                }
            } else {
                xQueueReceive(robot_audio_queue_, &robot_audio_message, 0);
                if (robot_audio_message == nullptr) {
                    xQueueReceive(user_audio_queue_, &user_audio_message, 0);
                }
            }

            if (user_audio_message != nullptr) {
                std::unique_ptr<AudioMessage> queued(user_audio_message);
                send_ok = SendUserAudioFrame(sock, queued->sequence, queued->payload);
                prefer_user_audio = false;
            } else if (robot_audio_message != nullptr) {
                std::unique_ptr<RobotAudioMessage> queued(robot_audio_message);
                send_ok = SendRobotAudioFrame(
                    sock, queued->sequence, queued->sample_rate, queued->frame_duration_ms,
                    queued->channels, queued->payload);
                prefer_user_audio = true;
            } else {
                ulTaskNotifyTake(pdTRUE, kWifiCheckDelay);
                continue;
            }
        }

        if (!send_ok) {
            ESP_LOGW(kTag, "Send failed, errno=%d", errno);
            connected_.store(false);
            shutdown(sock, SHUT_RDWR);
            close(sock);
            sock = -1;
            ClearUserAudioQueue();
            ClearRobotAudioQueue();
            vTaskDelay(kReconnectDelay);
        }
    }
}
