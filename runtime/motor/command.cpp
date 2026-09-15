//
// Created by bismarck on 11/19/22.
//

#include "command.h"

#include "neck/neck_config.h"
#include "neck/neck_kinematics.h"
#include "neck/neck_motion.h"
#include "neck/trajectory_io.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>

void sendToQueue(int slaveId, const Queue_Msg_ptr& msg) {

    if (messages[slaveId].write_available()) {
        messages[slaveId].push(msg);
    } else {
        std::cout << "Queue Fulled, Waiting For Command Executing\n";
        while(!messages[slaveId].push(msg)) sleep(1);
    }

}

Queue_Msg_ptr createQueueMsg(EtherCAT_Msg_ptr& msg, uint8_t passage) {
    Queue_Msg_ptr queue_msg = std::make_shared<Queue_Msg>();
    queue_msg->passage = passage;
    queue_msg->motor = msg->motor[queue_msg->passage - 1];
    return queue_msg;
}

namespace {

bool parseInt(const std::string& text, int& value) {
    try {
        std::size_t consumed = 0;
        const int parsed = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parseDouble(const std::string& text, double& value) {
    try {
        std::size_t consumed = 0;
        const double parsed = std::stod(text, &consumed);
        if (consumed != text.size()) {
            return false;
        }
        value = parsed;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool loadNeckConfiguration(NeckConfig& config, std::string& error) {
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

bool validNeckSlave(int slave_id, std::string& error) {
    if (!running) {
        error = "EtherCAT runtime is not running";
        return false;
    }
    if (ec_slavecount <= 0) {
        error = "no EtherCAT slave is available";
        return false;
    }
    if (slave_id < 0 || slave_id >= ec_slavecount || slave_id >= SLAVE_NUMBER) {
        error = "SlaveId must be in [0, " +
                std::to_string(std::min(ec_slavecount, SLAVE_NUMBER) - 1) + "]";
        return false;
    }
    return true;
}

bool validTrajectoryName(const std::string& name) {
    if (name.empty()) {
        return false;
    }
    for (unsigned char character : name) {
        if (!std::isalnum(character) && character != '_' && character != '-') {
            return false;
        }
    }
    return true;
}

bool loadNamedTrajectory(const std::string& name, Trajectory& trajectory,
                         std::string& error) {
    const std::array<std::string, 2> paths = {
        "trajectories/" + name + ".json",
        "../trajectories/" + name + ".json"
    };
    for (const std::string& path : paths) {
        std::ifstream probe(path);
        if (probe.good()) {
            if (!loadTrajectoryJson(path, trajectory, error)) {
                return false;
            }
            if (trajectory.name != name) {
                error = "trajectory name '" + trajectory.name +
                        "' does not match requested name '" + name + "'";
                return false;
            }
            return true;
        }
    }
    error = "cannot find trajectories/" + name + ".json";
    return false;
}

}  // namespace

unsigned help(const std::vector<std::string> &) {
    std::cout << "Available Commands:\n"
        << "\tMotorIdGet <SlaveId>\n"
        << "\tMotorIdSet <SlaveId> <MotorId> <NewMotorId>\n"
        << "\tMotorIdReset <SlaveId>\n"
        << "\tMotorZeroSet <SlaveId> <PassAge> <MotorId>\n"
        << "\tMotorStop <SlaveId> <PassAge> <MotorId>\n"
        << "\tMotorSpeedSet <SlaveId> <PassAge> <MotorId> <Speed>(0) <Current>(500) <AckStatus>(2)\n"
        << "\tMotorPositionSet <SlaveId> <PassAge> <MotorId> <Position>(0) <Speed>(50) <Current>(500) <AckStatus>(2)\n"
        << "\tMotorAngleGet <SlaveId> <PassAge> <MotorId>\n"
        << "\tNeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>  (degree)\n"
        << "\tNeckSequence <TrajectoryName>\n"
        << "\tNeckSequenceStop\n";
    return 0;
}

unsigned motorIdGet(const std::vector<std::string> & input) {
    int slaveId;
    switch (input.size()-1) {
        case 1:
            slaveId = std::stoi(input[1]);
            break;
        default:
            std::cout << "Command format error\n" <<
                      "\tShould be \"MotorIdGet <SlaveId>\"\n";
            return 1;
    }


    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    MotorIDReading(msg.get());
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, 1);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

unsigned motorIdSet(const std::vector<std::string> & input) {
    int slaveId;
    int motor_id,  motor_id_new;
    try {
        switch (input.size()-1) {
            case 3:
                slaveId = std::stoi(input[1]);
                motor_id = std::stoi(input[2]);
                motor_id_new = std::stoi(input[3]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"MotorIdSet <SlaveId> <OldMotorId> <NewMotorId>\"\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << '\n';
        return 1;
    }

    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    MotorIDSetting(msg.get(), motor_id, motor_id_new);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, 1);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

unsigned motorIdReset(const std::vector<std::string> & input) {
    int slaveId;
    try {
        switch (input.size() - 1) {
            case 1:
                slaveId = std::stoi(input[1]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"MotorIdReset <SlaveId>\"\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    MotorIDReset(msg.get());
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, 1);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

unsigned motorZeroSet(const std::vector<std::string> & input) {
    int slaveId;
    int passage;
    int motorId;
    try {
        switch (input.size() - 1) {
            case 3:
                slaveId = std::stoi(input[1]);
                passage = std::stoi(input[2]);
                motorId = std::stoi(input[3]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"MotorZeroSet <SlaveId> <PassAge> <MotorId>\"\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();

    Motor_Setzero(msg.get(), passage, motorId);

    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

unsigned motorSpeedSet(const std::vector<std::string> & input) {
    int slaveId;
    int motor_id ;
    float spd = 0;
    uint8_t passage;
    uint16_t cur = 500;
    uint8_t ack_status = 2;
    try {
        switch (input.size()-1) {
            case 6:
                ack_status = std::stoi(input[6]);
            case 5:
                cur = std::stoi(input[5]);
            case 4:
                spd = std::stof(input[4]);
            case 3:
                slaveId = std::stoi(input[1]);
                passage = std::stoi(input[2]);
                motor_id = std::stoi(input[3]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"MotorSpeedSet <SlaveId> <PassAge> <OldMotorId> <Speed>(0) <Current>(500) <AckStatus>(2)\"\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }



    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    set_motor_speed(msg.get(), passage ,motor_id, spd, cur, ack_status);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

unsigned motorPositionSet(const std::vector<std::string> & input) {
    int slaveId;
    int motor_id;
    float pos = 0;
    uint8_t passage;
    uint16_t spd = 50;
    uint16_t cur = 500;
    uint8_t ack_status = 2;
    try {
        switch (input.size()-1) {
            case 7:
                ack_status = std::stoi(input[7]);
            case 6:
                cur = std::stoi(input[6]);
            case 5:
                spd = std::stoi(input[5]);
            case 4:
                pos = std::stof(input[4]);
            case 3:
                slaveId = std::stoi(input[1]);
                passage = std::stoi(input[2]);
                motor_id = std::stoi(input[3]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"MotorPositionSet <SlaveId> <PassAge> <MotorId> <Position>(0) <Speed>(50) <Current>(500) <AckStatus>(2)\"\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    set_motor_position(msg.get(), passage, motor_id, pos, spd, cur, ack_status);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

unsigned motorAngleGet(const std::vector<std::string>& input) {
    if (input.size() != 4) {
        std::cout << "Command format error\n"
                  << "\tShould be \"MotorAngleGet <SlaveId> <PassAge> <MotorId>\"\n";
        return 1;
    }

    int slave_id;
    int passage;
    int motor_id;
    if (!parseInt(input[1], slave_id) ||
        !parseInt(input[2], passage) ||
        !parseInt(input[3], motor_id)) {
        std::cout << "Parameter error: all parameters must be integers\n";
        return 1;
    }

    std::string error;
    if (!validNeckSlave(slave_id, error) || passage < 1 || passage > 6 ||
        motor_id < 1 || motor_id > 0x7FE) {
        std::cout << "MotorAngleGet parameter error: "
                  << (error.empty() ? "PassAge must be in [1, 6] and MotorId in [1, 0x7FE]"
                                    : error)
                  << "\n";
        return 1;
    }

    EtherCAT_Msg_ptr message = std::make_shared<EtherCAT_Msg>();
    get_motor_parameter(message.get(), passage, motor_id, param_get_pos);
    sendToQueue(slave_id, createQueueMsg(message, passage));
    return 0;
}

unsigned neckPoseSet(const std::vector<std::string>& input) {
    if (input.size() != 5) {
        std::cout << "Command format error\n"
                  << "\tShould be \"NeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>\" (degree)\n";
        return 1;
    }

    int slave_id = 0;
    NeckPose pose{};
    if (!parseInt(input[1], slave_id) ||
        !parseDouble(input[2], pose.pitch) ||
        !parseDouble(input[3], pose.roll) ||
        !parseDouble(input[4], pose.yaw)) {
        std::cout << "Parameter error: SlaveId must be an integer and RPY must be numbers\n";
        return 1;
    }

    std::string error;
    if (!validNeckSlave(slave_id, error)) {
        std::cout << "NeckPoseSet error: " << error << "\n";
        return 1;
    }

    NeckConfig config;
    if (!loadNeckConfiguration(config, error)) {
        std::cout << "NeckPoseSet configuration error: " << error << "\n";
        return 1;
    }

    MotorAngles targets{};
    const NeckKinematicsStatus status = inverseKinematics(pose, config, targets);
    if (status != NeckKinematicsStatus::Ok) {
        std::cout << "NeckPoseSet kinematics error: "
                  << neckKinematicsStatusString(status) << "\n";
        return 1;
    }

    std::array<Queue_Msg_ptr, 3> commands;
    for (std::size_t index = 0; index < config.motors.size(); ++index) {
        const MotorConfig& motor = config.motors[index];
        const double positions[] = {targets.motor1, targets.motor2, targets.motor3};
        EtherCAT_Msg_ptr message = std::make_shared<EtherCAT_Msg>();
        set_motor_position(message.get(), motor.passage, motor.id,
                           static_cast<float>(positions[index]),
                           motor.speed_param, motor.current_param,
                           config.ack_status);
        commands[index] = createQueueMsg(message, motor.passage);
    }

    std::cout << "Target neck pose:\n"
              << "pitch=" << pose.pitch << "\n"
              << "roll=" << pose.roll << "\n"
              << "yaw=" << pose.yaw << "\n"
              << "Motor targets:\n"
              << "M1=" << targets.motor1 << "\n"
              << "M2=" << targets.motor2 << "\n"
              << "M3=" << targets.motor3 << "\n";

    for (const Queue_Msg_ptr& command : commands) {
        sendToQueue(slave_id, command);
    }
    return 0;
}

unsigned neckSequence(const std::vector<std::string>& input) {
    if (input.size() != 2 || !validTrajectoryName(input[1])) {
        std::cout << "Command format error\n"
                  << "\tShould be \"NeckSequence <TrajectoryName>\"\n"
                  << "\tTrajectoryName may contain letters, digits, '_' and '-' only\n";
        return 1;
    }

    Trajectory trajectory;
    std::string error;
    if (!loadNamedTrajectory(input[1], trajectory, error)) {
        std::cout << "NeckSequence trajectory error: " << error << "\n";
        return 1;
    }
    if (!executeTrajectory(trajectory, error)) {
        std::cout << "NeckSequence rejected: " << error << "\n";
        return 1;
    }
    return 0;
}

unsigned neckSequenceStop(const std::vector<std::string>& input) {
    if (input.size() != 1) {
        std::cout << "Command format error\n\tShould be \"NeckSequenceStop\"\n";
        return 1;
    }

    std::string error;
    if (!stopTrajectory(error)) {
        std::cout << "NeckSequenceStop error: " << error << "\n";
        return 1;
    }
    std::cout << "NeckSequenceStop completed for all three neck motors\n";
    return 0;
}

unsigned motorStop(const std::vector<std::string> & input) {
    int slaveId;
    int motor_id;
    int passage;
    try {
        switch (input.size() - 1) {
            case 3:
                slaveId = std::stoi(input[1]);
                passage = std::stoi(input[2]);
                motor_id = std::stoi(input[3]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"MotorStop <SlaveId> <PassAge> <MotorId>\"\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    set_motor_cur_tor(msg.get(), passage, motor_id, 10, 2, 0);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
    sendToQueue(slaveId, queue_msg);
    return 0;
}

