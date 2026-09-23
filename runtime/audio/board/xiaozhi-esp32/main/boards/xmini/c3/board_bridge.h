#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <freertos/FreeRTOS.h>
#include <freertos/queue.h>
#include <freertos/task.h>

struct AudioStreamPacket;

class BoardBridge {
public:
    static BoardBridge& GetInstance();

    void Start();
    bool EnqueueUserText(const std::string& text);
    bool EnqueueRobotText(uint32_t turn_id, uint64_t timestamp_ms, const std::string& text);
    bool EnqueueUserAudio(const AudioStreamPacket& packet);
    bool EnqueueRobotAudio(const AudioStreamPacket& packet);
    bool EnqueueRobotFirstAudio(uint32_t turn_id, uint64_t timestamp_ms);
    bool EnqueueRobotAudioEnd(uint32_t turn_id, uint64_t timestamp_ms);
    bool EnqueueRobotPlaybackStart(uint32_t turn_id, uint64_t timestamp_ms);
    bool EnqueueRobotPlaybackEnd(uint32_t turn_id, uint64_t timestamp_ms);
    bool EnqueueRobotPlaybackAbort(uint32_t turn_id, uint64_t timestamp_ms,
                                   const std::string& reason);

private:
    enum class MessageType {
        UserText,
        RobotText,
        RobotFirstAudio,
        RobotAudioEnd,
        RobotPlaybackStart,
        RobotPlaybackEnd,
        RobotPlaybackAbort,
    };

    struct Message {
        MessageType type;
        uint32_t turn_id;
        uint64_t timestamp_ms;
        std::string text;
    };

    struct AudioMessage {
        uint32_t sequence;
        std::vector<uint8_t> payload;
    };

    struct RobotAudioMessage {
        uint32_t sequence;
        uint32_t turn_id;
        uint32_t sample_rate;
        uint16_t frame_duration_ms;
        uint8_t channels;
        std::vector<uint8_t> payload;
    };

    BoardBridge() = default;
    BoardBridge(const BoardBridge&) = delete;
    BoardBridge& operator=(const BoardBridge&) = delete;

    static void SenderTaskEntry(void* arg);
    void SenderTask();
    bool EnqueueText(MessageType type, uint32_t turn_id, uint64_t timestamp_ms,
                     const std::string& text);
    void ClearUserAudioQueue();
    void ClearRobotAudioQueue();

    QueueHandle_t text_queue_ = nullptr;
    QueueHandle_t user_audio_queue_ = nullptr;
    QueueHandle_t robot_audio_queue_ = nullptr;
    TaskHandle_t task_ = nullptr;
    std::atomic<uint32_t> dropped_count_{0};
    std::atomic<uint32_t> user_audio_sequence_{0};
    std::atomic<uint32_t> robot_audio_sequence_{0};
    std::atomic<bool> connected_{false};
};
