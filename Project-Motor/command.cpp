//
// Created by bismarck on 11/19/22.
//

#include "command.h"
#include "neck_kinematics.h"
#include <cstring>
#include <limits>
#include <fstream>
#include <thread>
#include <chrono>

static bool validMotorCommand(int slaveId, int passage, int motorId) {
    if (!running || ec_slavecount <= 0 || slaveId < 0 || slaveId >= ec_slavecount ||
        slaveId >= SLAVE_NUMBER || passage < 1 || passage > 6 || motorId < 1 || motorId > 0x7FE) {
        std::cout << "Device or parameter error\n";
        return false;
    }
    return true;
}

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


unsigned help(const std::vector<std::string> &) {
    std::cout << "Available Commands:\n"
        << "\tMotorIdGet <SlaveId>\n"
        << "\tMotorIdSet <SlaveId> <MotorId> <NewMotorId>\n"
        << "\tMotorIdReset <SlaveId>\n"
        << "\tMotorAngleGet <SlaveId> <PassAge> <MotorId>\n"
        << "\tMotorZeroSet <SlaveId> <PassAge> <MotorId>\n"
        << "\tMotorStop <SlaveId> <PassAge> <MotorId>\n"
        << "\tMotorSpeedSet <SlaveId> <PassAge> <MotorId> <Speed>(0) <Current>(500) <AckStatus>(2)\n"
        << "\tMotorPositionSet <SlaveId> <PassAge> <MotorId> <Position>(0) <Speed>(50) <Current>(500) <AckStatus>(2)\n"
        << "\tNeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>  (单位:度, 参数取自 neck_config.txt)\n"
        << "\tNeckSquence <SequenceId> [SlaveId]  (0:标准测试 1:平静说话 2:活跃说话 3:强调说话)\n"
        << "\tNeckNpy <SlaveId> [Path]  (30fps机械动作序列)\n"
        << "\tNeckTrajDryRun <JsonPath> [AuditDir]  (离线解析/转换/IK/限幅/重定时, 不连接硬件)\n"
        << "\tNeckTrajMock <JsonPath> [AuditDir]  (mock 执行, 模拟反馈/超时/急停验证)\n"
        << "\tNeckTrajRun <JsonPath> [AuditDir]  (实机执行, 需三把锁全开, 默认拒绝)\n"
        << "\tNeckTrajStop   (安全停止当前轨迹)\n"
        << "\tNeckTrajEStop  (急停, 任何状态均可触发)\n"
        << "\tNeckTrajAck    (显式确认, 解除 ESTOP/FAULT)\n"
        << "\tNeckTrajStatus (状态/遥测)\n"
        << "\tNeckStaticCheck <秒数> [SlaveId]  (只读静止检查: 10Hz参数查询+反馈日志, 不下发任何运动指令)\n"
        << "\tNeckCalibMove <SlaveId> <Axis1..3> <DeltaDeg>  (标定运动: 单轴低速小幅度, 受限速/幅度/限位/急停保护)\n"
        << "\tNeckCalibPose <SlaveId> <Pitch> <Roll> <Yaw>  (标定姿态: 三轴联动低速, 各分量相对当前变化<=5°, 同保护)\n"
        << "\tNeckDisable [SlaveId]  (收工: 停止轨迹+制动保持; 离开前仍需物理断电)\n";
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

unsigned motorAngleGet(const std::vector<std::string> & input) {
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
                          "\tShould be \"MotorAngleGet <SlaveId> <PassAge> <MotorId>\"\n";
                return 1;
        }
    } catch (const std::exception &e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    if (!validMotorCommand(slaveId, passage, motorId))
        return 1;

    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    get_motor_parameter(msg.get(), passage, motorId, param_get_pos);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
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
    if (!validMotorCommand(slaveId, passage, motorId))
        return 1;
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
    if (!validMotorCommand(slaveId, passage, motor_id))
        return 1;
    if (pos != pos || pos > std::numeric_limits<float>::max() ||
        pos < -std::numeric_limits<float>::max())
    {
        std::cout << "Parameter error\n";
        return 1;
    }

    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    set_motor_position(msg.get(), passage, motor_id, pos, spd, cur, ack_status);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
    sendToQueue(slaveId, queue_msg);
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

// 首次调用时加载 neck 配置并缓存。依次尝试常见路径（从 build/ 运行时需回上级）。
static const NeckConfig& neckConfig() {
    static NeckConfig cfg;
    static bool loaded = false;
    if (!loaded) {
        const char* candidates[] = {"neck_config.txt", "../neck_config.txt", "../../neck_config.txt"};
        bool ok = false;
        for (const char* p : candidates) {
            std::ifstream probe(p);
            if (probe.good()) { probe.close(); ok = loadNeckConfig(p, cfg); break; }
        }
        if (!ok) {
            std::cout << "[Neck] 未找到 neck_config.txt，使用内置默认(占位)参数\n";
            cfg = defaultNeckConfig();
        }
        loaded = true;
    }
    return cfg;
}

// 向同一从站顺序下发一个电机的位置指令（度）。
static void sendOneMotor(int slaveId, uint8_t passage, uint16_t motorId, double posDeg,
                         uint16_t speed, const NeckConfig& cfg) {
    EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
    set_motor_position(msg.get(), passage, motorId, (float)posDeg, speed, cfg.current, cfg.ack_status);
    Queue_Msg_ptr queue_msg = createQueueMsg(msg, passage);
    sendToQueue(slaveId, queue_msg);
}

unsigned neckPoseSet(const std::vector<std::string> & input) {
    int slaveId;
    NeckAngles pose{};
    try {
        switch (input.size() - 1) {
            case 4:
                slaveId    = std::stoi(input[1]);
                pose.pitch = std::stod(input[2]);
                pose.roll  = std::stod(input[3]);
                pose.yaw   = std::stod(input[4]);
                break;
            default:
                std::cout << "Command format error\n" <<
                          "\tShould be \"NeckPoseSet <SlaveId> <Pitch> <Roll> <Yaw>\" (单位:度)\n";
                return 1;
        }
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    const NeckConfig& cfg = neckConfig();

    // 先做逆解与安全校验（不依赖从站，离线也能看到结果）。
    MotorAngles m{};
    NeckStatus st = neckSolve(pose, cfg, &m);
    if (st != NECK_OK) {
        std::cout << "姿态不可达: " << neckStatusString(st) << "，不下发指令\n";
        return 1;
    }
    std::cout << "逆解(度): m1=" << m.motor1 << " m2=" << m.motor2 << " m3=" << m.motor3 << "\n";

    // 设备与参数校验（三个电机各自校验；离线无从站会在此拦下）。
    if (!validMotorCommand(slaveId, cfg.passage1, cfg.id1) ||
        !validMotorCommand(slaveId, cfg.passage2, cfg.id2) ||
        !validMotorCommand(slaveId, cfg.passage3, cfg.id3))
        return 1;

    // 三个电机分别入队（不同 passage，状态机会累加进 Tx_Message）。
    sendOneMotor(slaveId, cfg.passage1, cfg.id1, m.motor1, cfg.speed, cfg);
    sendOneMotor(slaveId, cfg.passage2, cfg.id2, m.motor2, cfg.speed, cfg);
    sendOneMotor(slaveId, cfg.passage3, cfg.id3, m.motor3, cfg.speed, cfg);
    return 0;
}

unsigned neckSquence(const std::vector<std::string> & input) {
    int sequenceId;
    int slaveId = 0;
    try {
        if (input.size() < 2 || input.size() > 3) {
            std::cout << "Command format error\n\tShould be \"NeckSquence <SequenceId> [SlaveId]\"\n";
            return 1;
        }
        sequenceId = std::stoi(input[1]);
        if (input.size() == 3)
            slaveId = std::stoi(input[2]);
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    struct Frame { const char* name; double pitch, roll, yaw; uint16_t speed; unsigned waitMs; };
    const Frame standardFrames[] = {
        {"中位",       0,   0,   0,  0, 1500},
        {"抬头",      20,   0,   0,  0, 2000}, {"回中", 0, 0, 0, 0, 1500},
        {"低头",     -30,   0,   0,  0, 2200}, {"回中", 0, 0, 0, 0, 1700},
        {"向右侧倾",   0,  25,   0,  0, 2000}, {"回中", 0, 0, 0, 0, 1500},
        {"向左侧倾",   0, -25,   0,  0, 2000}, {"回中", 0, 0, 0, 0, 1700},
        {"向左转头",   0,   0,  55,  0, 2200}, {"回中", 0, 0, 0, 0, 1700},
        {"向右转头",   0,   0, -65,  0, 2200}, {"回中", 0, 0, 0, 0, 2000}
    };
    const Frame calmSpeechFrames[] = {
        {"准备",       0,  0,  0, 25, 600}, {"自然左看", -1, -2,  7, 35, 600},
        {"轻点头",    -5, -1,  8, 55, 320}, {"自然回弹",  2,  0,  7, 35, 520},
        {"短暂停顿",   0,  1,  3, 25, 800}, {"自然右看", -2,  3, -9, 40, 650},
        {"轻微强调",  -6,  2, -7, 60, 340}, {"自然回弹",  2,  1, -4, 35, 500},
        {"思考停顿",   0, -2,  2, 20, 900}, {"收尾点头", -4,  0,  0, 50, 360},
        {"回正",       1,  0,  0, 30, 500}, {"中位",      0,  0,  0, 20, 800}
    };
    const Frame livelySpeechFrames[] = {
        {"准备",       0,  0,   0, 35, 450}, {"快速起句", -3, -4,  12, 80, 300},
        {"转向表达",   5,  3, -10, 95, 280}, {"重点说明", -7, -3,  14,100, 300},
        {"换侧表达",   3,  5, -14, 90, 320}, {"快速强调", -9,  0,  -5,110, 260},
        {"自然回弹",   4, -2,   4, 85, 300}, {"继续表达", -6,  4,  10, 90, 280},
        {"换侧收束",   2, -4,  -8, 75, 350}, {"短暂停顿",  0,  2,   5, 35, 650},
        {"结束点头",  -5,  0,   0, 70, 300}, {"回弹",      2,  0,   0, 45, 400},
        {"中位",       0,  0,   0, 20, 800}
    };
    const Frame emphaticSpeechFrames[] = {
        {"准备",       0,  0,   0, 25, 500}, {"转向听众",  0, -4,   8, 40, 600},
        {"深度强调", -10, -3,   6, 90, 400}, {"抬头续讲",  4, -2,   5, 55, 550},
        {"再次强调",  -8,  2,  -6, 80, 380}, {"停顿",      3,  3,  -7, 50, 700},
        {"侧头说明",   0,  8,  15, 30, 900}, {"确认点头", -6,  6,  12, 70, 400},
        {"自然回弹",   2,  4,   8, 45, 600}, {"换侧说明",  0, -6, -12, 30, 900},
        {"收尾强调",  -7, -4, -10, 75, 400}, {"回弹",      2, -2,  -5, 40, 500},
        {"中位",       0,  0,   0, 20, 900}
    };

    const Frame* frames = nullptr;
    size_t frameCount = 0;
    const char* sequenceName = nullptr;
    switch (sequenceId) {
        case 0:
            frames = standardFrames;
            frameCount = sizeof(standardFrames) / sizeof(standardFrames[0]);
            sequenceName = "标准六方向测试";
            break;
        case 1:
            frames = calmSpeechFrames;
            frameCount = sizeof(calmSpeechFrames) / sizeof(calmSpeechFrames[0]);
            sequenceName = "平静说话动作";
            break;
        case 2:
            frames = livelySpeechFrames;
            frameCount = sizeof(livelySpeechFrames) / sizeof(livelySpeechFrames[0]);
            sequenceName = "活跃说话动作";
            break;
        case 3:
            frames = emphaticSpeechFrames;
            frameCount = sizeof(emphaticSpeechFrames) / sizeof(emphaticSpeechFrames[0]);
            sequenceName = "强调说话动作";
            break;
        default:
            std::cout << "[Neck] SequenceId 仅支持 0、1、2、3\n";
            return 1;
    }

    const NeckConfig& cfg = neckConfig();
    if (!validMotorCommand(slaveId, cfg.passage1, cfg.id1) ||
        !validMotorCommand(slaveId, cfg.passage2, cfg.id2) ||
        !validMotorCommand(slaveId, cfg.passage3, cfg.id3))
        return 1;

    for (size_t i = 0; i < frameCount; ++i) {
        const Frame& frame = frames[i];
        if (neckSolve({frame.pitch, frame.roll, frame.yaw}, cfg, nullptr) != NECK_OK) {
            std::cout << "[Neck] 序列包含不可达姿态，不执行\n";
            return 1;
        }
    }

    std::cout << "[Neck] 开始" << sequenceName << "\n";
    for (size_t i = 0; i < frameCount; ++i) {
        const Frame& frame = frames[i];
        std::cout << "[Neck] " << frame.name << "\n";
        MotorAngles motor = neckInverse({frame.pitch, frame.roll, frame.yaw}, cfg);
        uint16_t speed = frame.speed == 0 ? cfg.speed : frame.speed;
        sendOneMotor(slaveId, cfg.passage1, cfg.id1, motor.motor1, speed, cfg);
        sendOneMotor(slaveId, cfg.passage2, cfg.id2, motor.motor2, speed, cfg);
        sendOneMotor(slaveId, cfg.passage3, cfg.id3, motor.motor3, speed, cfg);
        std::this_thread::sleep_for(std::chrono::milliseconds(frame.waitMs));
    }
    std::cout << "[Neck] " << sequenceName << "完成\n";
    return 0;
}

static bool loadNpyPoses(const std::string& path, std::vector<NeckAngles>& poses) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return false;

    char magic[6];
    unsigned char version[2];
    uint16_t headerSize;
    if (!file.read(magic, sizeof(magic)) || std::memcmp(magic, "\x93NUMPY", 6) != 0 ||
        !file.read(reinterpret_cast<char*>(version), sizeof(version)) ||
        (version[0] != 1 || version[1] != 0) ||
        !file.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize)))
        return false;

    std::string header(headerSize, '\0');
    if (!file.read(header.data(), header.size()) ||
        header.find("'descr': '<f8'") == std::string::npos ||
        header.find("'fortran_order': False") == std::string::npos)
        return false;

    size_t shapeKey = header.find("'shape':");
    size_t open = header.find('(', shapeKey);
    size_t comma = header.find(',', open);
    size_t close = header.find(')', comma);
    if (shapeKey == std::string::npos || open == std::string::npos ||
        comma == std::string::npos || close == std::string::npos)
        return false;

    size_t rows;
    size_t columns;
    try {
        rows = std::stoull(header.substr(open + 1, comma - open - 1));
        columns = std::stoull(header.substr(comma + 1, close - comma - 1));
    } catch (const std::exception&) {
        return false;
    }
    if (rows == 0 || rows > 1000000 || columns != 3)
        return false;

    poses.resize(rows);
    for (NeckAngles& pose : poses) {
        double* values[] = {&pose.pitch, &pose.roll, &pose.yaw};
        for (double* value : values) {
            if (!file.read(reinterpret_cast<char*>(value), sizeof(double)) ||
                *value != *value ||
                *value > std::numeric_limits<double>::max() ||
                *value < -std::numeric_limits<double>::max()) {
                poses.clear();
                return false;
            }
        }
    }
    return true;
}

static bool loadNpyPosesFromCandidates(const std::string& path, std::vector<NeckAngles>& poses) {
    if (loadNpyPoses(path, poses))
        return true;
    if (!path.empty() && path[0] != '/') {
        if (loadNpyPoses("../" + path, poses))
            return true;
        if (loadNpyPoses("../../" + path, poses))
            return true;
    }
    return false;
}

static uint16_t neckFrameSpeed(const MotorAngles& previous, const MotorAngles& current,
                               const NeckConfig& cfg) {
    double d1 = previous.motor1 - current.motor1;
    double d2 = previous.motor2 - current.motor2;
    double d3 = previous.motor3 - current.motor3;
    if (d1 < 0.0) d1 = -d1;
    if (d2 < 0.0) d2 = -d2;
    if (d3 < 0.0) d3 = -d3;
    double maxStep = d1 > d2 ? d1 : d2;
    if (d3 > maxStep) maxStep = d3;
    double requested = maxStep * 50.0 * 1.25;
    uint16_t speed = requested >= 18000.0 ? 18000 : (uint16_t)(requested + 1.0);
    return speed > cfg.speed ? speed : cfg.speed;
}

static EtherCAT_Msg makeNeckMotorFrame(const MotorAngles& motor, uint16_t speed,
                                       const NeckConfig& cfg) {
    EtherCAT_Msg frame{};
    set_motor_position(&frame, cfg.passage1, cfg.id1, (float)motor.motor1,
                       speed, cfg.current, cfg.ack_status);
    set_motor_position(&frame, cfg.passage2, cfg.id2, (float)motor.motor2,
                       speed, cfg.current, cfg.ack_status);
    set_motor_position(&frame, cfg.passage3, cfg.id3, (float)motor.motor3,
                       speed, cfg.current, cfg.ack_status);
    return frame;
}

unsigned neckNpy(const std::vector<std::string>& input) {
    int slaveId;
    std::string path = "test/mech_cmd.npy";
    try {
        if (input.size() < 2 || input.size() > 3) {
            std::cout << "Command format error\n"
                      << "\tShould be \"NeckNpy <SlaveId> [Path]\"\n";
            return 1;
        }
        slaveId = std::stoi(input[1]);
        if (input.size() == 3)
            path = input[2];
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    const NeckConfig& cfg = neckConfig();
    if (!validMotorCommand(slaveId, cfg.passage1, cfg.id1) ||
        !validMotorCommand(slaveId, cfg.passage2, cfg.id2) ||
        !validMotorCommand(slaveId, cfg.passage3, cfg.id3))
        return 1;

    std::vector<NeckAngles> poses;
    if (!loadNpyPosesFromCandidates(path, poses)) {
        std::cout << "[NeckNpy] 无法读取 NPY 文件: " << path << "\n";
        return 1;
    }

    std::vector<MotorAngles> motors;
    motors.reserve(poses.size());
    for (size_t i = 0; i < poses.size(); ++i) {
        MotorAngles motor{};
        NeckStatus status = neckSolve(poses[i], cfg, &motor);
        if (status != NECK_OK) {
            std::cout << "[NeckNpy] 第 " << i << " 帧不可达: "
                      << neckStatusString(status) << "\n";
            return 1;
        }
        motors.push_back(motor);
    }

    MotorAngles neutral{};
    if (neckSolve({0.0, 0.0, 0.0}, cfg, &neutral) != NECK_OK)
        return 1;
    std::vector<EtherCAT_Msg> frames;
    frames.reserve(motors.size());
    MotorAngles previous = neutral;
    for (const MotorAngles& motor : motors) {
        frames.push_back(makeNeckMotorFrame(motor, neckFrameSpeed(previous, motor, cfg), cfg));
        previous = motor;
    }
    EtherCAT_Msg neutralFrame = makeNeckMotorFrame(neutral, cfg.speed, cfg);

    std::cout << "[NeckNpy] 开始播放 " << poses.size() << " 帧，30fps\n";
    const auto start = std::chrono::steady_clock::now();
    bool published = false;
    for (size_t i = 0; i < frames.size(); ++i) {
        auto offset = std::chrono::nanoseconds((i * 1000000000ULL) / 30);
        std::this_thread::sleep_until(start + offset);
        if (!NeckFramePublish(slaveId, &frames[i])) {
            std::cout << "[NeckNpy] 发布失败，动作中止\n";
            if (published)
                NeckFrameStop(slaveId);
            return 1;
        }
        published = true;
    }

    NeckFramePublish(slaveId, &neutralFrame);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    NeckFrameStop(slaveId);
    std::cout << "[NeckNpy] 播放完成，已回到中位\n";
    return 0;
}

// ==================== Neck Trajectory Executor 命令 ====================

#include "neck_control/executor.h"
#include "neck_control/hardware_adapter.h"
#include "neck_control/inverse_kinematics.h"
#include "neck_control/mock_adapter.h"
#include "neck_control/trajectory_io.h"

namespace {

using neck_control::NeckControlConfig;
using neck_control::NeckTrajectoryCommand;
using neck_control::NeckTrajectoryExecutor;
using neck_control::ExecutorMode;
using neck_control::ExecutorState;
using neck_control::NeckError;
using neck_control::CoordinateCalibration;
using neck_control::neckInverseSolve;
using neck_control::MockAdapter;
using neck_control::RpyDeg;
using neck_control::NeckFeedback;

// 全局执行器（mock/hardware 共享，跨命令保持状态）
NeckTrajectoryExecutor* gTrajExecutor = nullptr;
MockAdapter* gTrajMock = nullptr;
std::string gPendingAuditDir; // 异步执行完成后待写的审计目录

NeckControlConfig loadTrajControlConfig() {
    NeckControlConfig cfg;
    const char* candidates[] = {
        "neck_control/neck_trajectory_config.txt",
        "../neck_control/neck_trajectory_config.txt",
        "../../neck_control/neck_trajectory_config.txt",
    };
    for (const char* p : candidates) {
        std::ifstream probe(p);
        if (probe.good()) {
            probe.close();
            neck_control::loadNeckControlConfig(p, cfg);
            return cfg;
        }
    }
    std::cout << "[NeckTraj] 未找到 neck_trajectory_config.txt，使用内置保守默认值"
              << "（实机默认禁用）\n";
    neck_control::loadNeckControlConfig("", cfg);
    return cfg;
}

// 读取 JSON 文件 → NeckTrajectoryCommand
NeckError loadTrajJson(const std::string& path, NeckTrajectoryCommand& cmd, std::string& msg) {
    std::string text;
    if (!neck_control::readTextFile(path, text)) {
        msg = "无法读取文件 " + path;
        return NeckError::INVALID_JSON;
    }
    return neck_control::loadNeckTrajectoryJson(text, cmd, msg);
}

// 打印处理报告
void printTrajReport(const NeckTrajectoryExecutor& ex) {
    const neck_control::ProcessResult& r = ex.lastResult();
    std::cout << "[NeckTraj] 帧数=" << r.frames
              << " 名义时长=" << r.nominal_duration_s << "s"
              << " 计划时长=" << r.duration_s << "s"
              << " 重定时比例=" << (r.nominal_duration_s > 0 ? r.duration_s / r.nominal_duration_s : 1.0)
              << " 起点偏差=" << r.start_pose_error_deg << "°\n";
    std::cout << "[NeckTraj] 处理后峰值 (m1/m2/m3):\n"
              << "  速度: " << r.max_vel[0] << " / " << r.max_vel[1] << " / " << r.max_vel[2]
              << " °/s\n"
              << "  加速度: " << r.max_acc[0] << " / " << r.max_acc[1] << " / " << r.max_acc[2]
              << " °/s²\n"
              << "  jerk: " << r.max_jerk[0] << " / " << r.max_jerk[1] << " / " << r.max_jerk[2]
              << " °/s³\n";
}

// 建立反馈：若反馈快照为空（如刚重启），发送 3 个只读参数查询并等待应答。
// 只读操作，不发送任何位置/速度/力矩指令。
static bool primeFeedback(int slaveId, int attempts = 5) {
    double deg[3]; uint8_t err[3]; double temp[3]; uint64_t ts = 0;
    if (NeckFeedbackGet(slaveId, deg, err, temp, &ts))
        return true;
    const NeckConfig& cfg = neckConfig();
    const uint8_t passages[3] = {cfg.passage1, cfg.passage2, cfg.passage3};
    const uint16_t ids[3] = {cfg.id1, cfg.id2, cfg.id3};
    for (int attempt = 0; attempt < attempts; ++attempt) {
        for (int k = 0; k < 3; ++k) {
            EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
            get_motor_parameter(msg.get(), passages[k], ids[k], param_get_pos);
            Queue_Msg_ptr qmsg = createQueueMsg(msg, passages[k]);
            sendToQueue(slaveId, qmsg);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        if (NeckFeedbackGet(slaveId, deg, err, temp, &ts))
            return true;
    }
    return false;
}

} // namespace

unsigned neckTrajDryRun(const std::vector<std::string>& input) {
    std::string path, auditDir;
    try {
        if (input.size() < 2 || input.size() > 3) {
            std::cout << "Command format error\n\tShould be \"NeckTrajDryRun <JsonPath> [AuditDir]\"\n";
            return 1;
        }
        path = input[1];
        if (input.size() == 3) auditDir = input[2];
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    NeckTrajectoryCommand cmd;
    std::string msg;
    NeckError e = loadTrajJson(path, cmd, msg);
    if (e != NeckError::OK) {
        std::cout << "[NeckTraj] 加载失败: " << msg << "\n";
        return 1;
    }

    NeckControlConfig cfg = loadTrajControlConfig();
    NeckConfig ik = neckConfig();
    NeckTrajectoryExecutor ex(cfg, ik, ExecutorMode::DRY_RUN, nullptr);
    neck_control::ProcessResult r = ex.process(cmd);
    if (r.error != NeckError::OK) {
        std::cout << "[NeckTraj] 拒绝: " << r.message << "\n";
        return 1;
    }
    std::cout << "[NeckTraj] dry-run 通过: " << path << "\n";
    printTrajReport(ex);
    if (!auditDir.empty()) {
        if (ex.saveAudit(auditDir, msg))
            std::cout << "[NeckTraj] 审计已写入 " << auditDir << "\n";
        else
            std::cout << "[NeckTraj] 审计写入失败: " << msg << "\n";
    }
    return 0;
}

// 等待执行结束（READY/FAULT/ESTOP），超时 60s（仅 mock 同步演示用）
static bool waitTrajDone(NeckTrajectoryExecutor& ex) {
    for (int i = 0; i < 1200; ++i) {
        ExecutorState s = ex.state();
        if (s == ExecutorState::READY || s == ExecutorState::FAULT || s == ExecutorState::ESTOP)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    return false;
}

unsigned neckTrajMock(const std::vector<std::string>& input) {
    std::string path, auditDir;
    try {
        if (input.size() < 2 || input.size() > 3) {
            std::cout << "Command format error\n\tShould be \"NeckTrajMock <JsonPath> [AuditDir]\"\n";
            return 1;
        }
        path = input[1];
        if (input.size() == 3) auditDir = input[2];
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    NeckTrajectoryCommand cmd;
    std::string msg;
    NeckError e = loadTrajJson(path, cmd, msg);
    if (e != NeckError::OK) {
        std::cout << "[NeckTraj] 加载失败: " << msg << "\n";
        return 1;
    }

    NeckControlConfig cfg = loadTrajControlConfig();
    NeckConfig ik = neckConfig();

    // mock 初始姿态 = robot_actual_initial（经标定+IK）
    CoordinateCalibration cal(cfg);
    RpyDeg ai = cal.modelToHardware(cmd.robot_actual_initial);
    neck_control::MotorAngles init{};
    std::string imsg;
    if (neckInverseSolve(ai, ik, init, imsg) != NeckError::OK) {
        // 回退到中位
        RpyDeg n = cal.modelToHardware(cmd.robot_neutral_pose);
        neckInverseSolve(n, ik, init, imsg);
    }

    if (gTrajExecutor) {
        gTrajExecutor->stop(); // 安全打断旧轨迹（start 内部也会处理）
        delete gTrajExecutor;
        gTrajExecutor = nullptr;
        delete gTrajMock;
        gTrajMock = nullptr;
    }
    gTrajMock = new MockAdapter(init);
    gTrajExecutor = new NeckTrajectoryExecutor(cfg, ik, ExecutorMode::MOCK, gTrajMock);

    e = gTrajExecutor->start(cmd);
    if (e != NeckError::OK) {
        std::cout << "[NeckTraj] mock 启动失败: " << neck_control::neckErrorString(e) << "\n";
        delete gTrajExecutor; gTrajExecutor = nullptr;
        delete gTrajMock; gTrajMock = nullptr;
        return 1;
    }
    std::cout << "[NeckTraj] mock 执行中...\n";
    bool done = waitTrajDone(*gTrajExecutor);
    if (!done) std::cout << "[NeckTraj] 等待超时\n";
    printTrajReport(*gTrajExecutor);
    const neck_control::LoopTelemetry& t = gTrajExecutor->telemetry();
    std::cout << "[NeckTraj] 状态=" << neck_control::executorStateString(gTrajExecutor->state())
              << " 错误=" << neck_control::neckErrorString(t.last_error)
              << " (" << t.last_error_message << ")\n"
              << "[NeckTraj] 发送=" << t.writes_ok << " 失败=" << t.writes_fail
              << " 反馈=" << t.feedback_ok << " 抖动max=" << t.max_jitter_ms << "ms\n";
    if (!auditDir.empty()) {
        if (gTrajExecutor->saveAudit(auditDir, msg))
            std::cout << "[NeckTraj] 审计已写入 " << auditDir << "\n";
    }
    return gTrajExecutor->state() == ExecutorState::READY ? 0 : 1;
}

unsigned neckTrajRun(const std::vector<std::string>& input) {
    std::string path, auditDir;
    int slaveId = 0;
    try {
        if (input.size() < 2 || input.size() > 4) {
            std::cout << "Command format error\n"
                      << "\tShould be \"NeckTrajRun <JsonPath> [SlaveId] [AuditDir]\"\n";
            return 1;
        }
        path = input[1];
        if (input.size() >= 3) slaveId = std::stoi(input[2]);
        if (input.size() >= 4) auditDir = input[3];
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }

    NeckTrajectoryCommand cmd;
    std::string msg;
    NeckError e = loadTrajJson(path, cmd, msg);
    if (e != NeckError::OK) {
        std::cout << "[NeckTraj] 加载失败: " << msg << "\n";
        return 1;
    }

    if (!primeFeedback(slaveId)) {
        std::cout << "[NeckTraj] 无法建立电机反馈（电机不应答？），拒绝执行\n";
        return 1;
    }

    NeckControlConfig cfg = loadTrajControlConfig();
    NeckConfig ik = neckConfig();

    if (gTrajExecutor) {
        gTrajExecutor->stop();
        delete gTrajExecutor;
        gTrajExecutor = nullptr;
        delete gTrajMock;
        gTrajMock = nullptr;
    }
    // 实机适配器（三把锁在 enable/start 内强制校验）
    auto* adapter = new neck_control::EthercatNeckAdapter(slaveId, ik);
    gTrajExecutor = new NeckTrajectoryExecutor(cfg, ik, ExecutorMode::HARDWARE, adapter);

    e = gTrajExecutor->start(cmd);
    if (e != NeckError::OK) {
        std::cout << "[NeckTraj] 实机启动被拒绝: " << neck_control::neckErrorString(e)
                  << " (" << gTrajExecutor->lastResult().message << ")\n";
        std::cout << "[NeckTraj] 注意: 实机需要 mode.hardware_enabled / calib.confirmed /"
                     " safety.confirmed 全部为 true 且标定完成\n";
        delete gTrajExecutor; gTrajExecutor = nullptr;
        return 1;
    }
    std::cout << "[NeckTraj] 实机执行中（后台线程，CLI 保持响应）...\n";
    printTrajReport(*gTrajExecutor);
    const neck_control::LoopTelemetry& t = gTrajExecutor->telemetry();
    std::cout << "[NeckTraj] 状态=" << neck_control::executorStateString(gTrajExecutor->state())
              << " 错误=" << neck_control::neckErrorString(t.last_error) << "\n"
              << "[NeckTraj] 已转入后台：随时可用 NeckTrajStatus 看进度 / "
                 "NeckTrajStop 安全停止 / NeckTrajEStop 急停（物理急停始终优先）\n";
    if (!auditDir.empty()) {
        gPendingAuditDir = auditDir;
        std::cout << "[NeckTraj] 审计目录: " << auditDir << "（完成后由 NeckTrajStatus 落盘）\n";
    }
    return 0;
}

unsigned neckTrajStop(const std::vector<std::string>&) {
    if (!gTrajExecutor) {
        std::cout << "[NeckTraj] 无活动执行器\n";
        return 1;
    }
    NeckError e = gTrajExecutor->stop();
    std::cout << "[NeckTraj] stop: " << neck_control::neckErrorString(e) << "\n";
    return 0;
}

unsigned neckTrajEStop(const std::vector<std::string>&) {
    if (!gTrajExecutor) {
        std::cout << "[NeckTraj] 无活动执行器\n";
        return 1;
    }
    gTrajExecutor->estop();
    std::cout << "[NeckTraj] 急停已触发\n";
    return 0;
}

unsigned neckTrajAck(const std::vector<std::string>&) {
    if (!gTrajExecutor) {
        std::cout << "[NeckTraj] 无活动执行器\n";
        return 1;
    }
    NeckError e = gTrajExecutor->acknowledge();
    std::cout << "[NeckTraj] acknowledge: " << neck_control::neckErrorString(e) << "\n";
    return 0;
}

unsigned neckTrajStatus(const std::vector<std::string>& input) {
    if (!gTrajExecutor) {
        std::cout << "[NeckTraj] 无活动执行器（尚未执行过 mock/run）\n";
        return 1;
    }
    // 异步轨迹完成后自动落盘审计
    if (!gPendingAuditDir.empty() && gTrajExecutor->state() == ExecutorState::READY) {
        std::string amsg;
        if (gTrajExecutor->saveAudit(gPendingAuditDir, amsg))
            std::cout << "[NeckTraj] 审计已写入 " << gPendingAuditDir << "\n";
        else
            std::cout << "[NeckTraj] 审计写入失败: " << amsg << "\n";
        gPendingAuditDir.clear();
    }
    const neck_control::LoopTelemetry& t = gTrajExecutor->telemetry();
    double elapsed = 0.0, planned = 0.0;
    gTrajExecutor->progress(elapsed, planned);
    std::cout << "[NeckTraj] 状态=" << neck_control::executorStateString(gTrajExecutor->state())
              << " 模式=" << (gTrajExecutor->mode() == ExecutorMode::MOCK ? "mock"
                            : gTrajExecutor->mode() == ExecutorMode::HARDWARE ? "hardware"
                            : "dry-run")
              << (gTrajExecutor->watchdogFired() ? " WATCHDOG_FIRED" : "") << "\n"
              << "[NeckTraj] 进度=" << elapsed << "/" << planned << "s"
              << (planned > 0 ? " (" + std::to_string((int)(100.0 * elapsed / planned)) + "%)" : "")
              << "\n"
              << "[NeckTraj] 错误=" << neck_control::neckErrorString(t.last_error)
              << " 消息=" << t.last_error_message << "\n"
              << "[NeckTraj] 最大跟踪误差=" << t.max_tracking_error_deg << "°"
              << " ticks=" << t.ticks << " 发送=" << t.writes_ok
              << " 失败=" << t.writes_fail << " 反馈=" << t.feedback_ok
              << " 丢失=" << t.feedback_miss << " 抖动max=" << t.max_jitter_ms << "ms\n";
    return 0;
}

// ==================== NeckStaticCheck：只读静止检查 ====================
// 安全声明：本命令【只发送参数查询】(get_motor_parameter, 0xE0)，绝不发送
// 位置/速度/力矩/制动指令，绝不发布 NeckFrame，绝不启动执行器。
// 用途：确认设备处于可安全停止的静止状态；采集 5~10s 静止日志。
unsigned neckStaticCheck(const std::vector<std::string>& input) {
    double seconds = 6.0;
    int slaveId = 0;
    try {
        if (input.size() < 2 || input.size() > 3) {
            std::cout << "Command format error\n"
                      << "\tShould be \"NeckStaticCheck <Seconds> [SlaveId]\"\n";
            return 1;
        }
        seconds = std::stod(input[1]);
        if (seconds < 1.0 || seconds > 30.0) {
            std::cout << "秒数须在 [1, 30] 内\n";
            return 1;
        }
        if (input.size() == 3) slaveId = std::stoi(input[2]);
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    if (!running || ec_slavecount <= 0 || slaveId < 0 || slaveId >= ec_slavecount) {
        std::cout << "[StaticCheck] 无 EtherCAT 会话/从站，无法采集\n";
        return 1;
    }
    const NeckConfig& cfg = neckConfig();

    std::cout << "[StaticCheck] 只读静止检查开始: " << seconds << "s @10Hz, slave=" << slaveId
              << "。仅参数查询，不下发运动指令。\n";
    std::cout << "timestamp_ms, m1_target, m1_actual, m1_status, "
                 "m2_target, m2_actual, m2_status, "
                 "m3_target, m3_actual, m3_status, "
                 "feedback_age_ms, communication_state, fault_code, estop_state, executor_state\n";

    auto monoNow = []() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    const uint64_t t0 = monoNow();
    const uint64_t endMs = t0 + (uint64_t)(seconds * 1000.0);
    uint64_t sample = 0;
    double minAct[3] = {1e9, 1e9, 1e9}, maxAct[3] = {-1e9, -1e9, -1e9};
    double lastAct[3] = {0, 0, 0};
    bool hasSample = false;
    bool anyFault = false;
    double maxJump = 0.0;

    while (monoNow() < endMs) {
        uint64_t tickT = monoNow();
        // 只读：三个电机的位置参数查询（问答模式返回角度/温度/电流/错误字节）
        const uint8_t passages[3] = {cfg.passage1, cfg.passage2, cfg.passage3};
        const uint16_t ids[3] = {cfg.id1, cfg.id2, cfg.id3};
        for (int k = 0; k < 3; ++k) {
            EtherCAT_Msg_ptr msg = std::make_shared<EtherCAT_Msg>();
            get_motor_parameter(msg.get(), passages[k], ids[k], param_get_pos);
            Queue_Msg_ptr qmsg = createQueueMsg(msg, passages[k]);
            sendToQueue(slaveId, qmsg);
        }
        // 反馈快照
        NeckFeedback fb;
        {
            double deg[3] = {0, 0, 0};
            uint8_t err[3] = {0, 0, 0};
            double temp[3] = {0, 0, 0};
            uint64_t fbTs = 0;
            if (NeckFeedbackGet(slaveId, deg, err, temp, &fbTs)) {
                fb.valid = true;
                for (int k = 0; k < 3; ++k) {
                    fb.joint_deg[k] = deg[k];
                    fb.error[k] = err[k];
                    fb.temperature[k] = temp[k];
                }
                fb.timestamp_ms = fbTs;
            }
        }
        // 通信状态
        NeckCommSnapshot comm{};
        NeckCommSnapshotGet(&comm);
        // 执行器状态（若存在）
        const char* estopState = "RELEASED";
        const char* execState = "NO_EXECUTOR";
        neck_control::MotorAngles target;
        bool haveTarget = false;
        if (gTrajExecutor) {
            execState = neck_control::executorStateString(gTrajExecutor->state());
            if (gTrajExecutor->state() == neck_control::ExecutorState::ESTOP)
                estopState = "TRIGGERED";
            haveTarget = gTrajExecutor->lastCommanded(target);
        }
        // 通信/故障汇总
        int faultCode = 0;
        if (comm.wkc < comm.expected_wkc) faultCode |= 1;
        std::string commState = "OP";
        if (!comm.running) commState = "DOWN";
        else if (!comm.in_op) commState = "NOT_OP";
        else if (comm.wkc < comm.expected_wkc) commState = "WKC_LOW";

        char line[1024];
        auto fmtStatus = [](uint8_t err, double temp) {
            char buf[48];
            if (err != 0) snprintf(buf, sizeof(buf), "ERR=0x%02X", err);
            else snprintf(buf, sizeof(buf), "OK T=%.0fC", temp);
            return std::string(buf);
        };
        std::string s1 = fb.valid ? fmtStatus(fb.error[0], fb.temperature[0]) : "NO_FB";
        std::string s2 = fb.valid ? fmtStatus(fb.error[1], fb.temperature[1]) : "NO_FB";
        std::string s3 = fb.valid ? fmtStatus(fb.error[2], fb.temperature[2]) : "NO_FB";
        if (fb.valid) {
            for (int k = 0; k < 3; ++k) {
                if (fb.error[k] != 0) { anyFault = true; faultCode |= (2 << k); }
            }
        } else {
            anyFault = true;
            faultCode |= 8; // 无反馈
        }
        uint64_t fbAge = fb.valid ? (monoNow() > fb.timestamp_ms ? monoNow() - fb.timestamp_ms : 0)
                                  : 9999;
        auto tgtStr = [&](int k) {
            if (!haveTarget) return std::string("n/a");
            char buf[32];
            snprintf(buf, sizeof(buf), "%.3f", k == 0 ? target.motor1
                                                      : (k == 1 ? target.motor2 : target.motor3));
            return std::string(buf);
        };
        snprintf(line, sizeof(line),
                 "%llu, %s, %.3f, %s, %s, %.3f, %s, %s, %.3f, %s, %llu, %s, %d, %s, %s",
                 (unsigned long long)(tickT - t0),
                 tgtStr(0).c_str(), fb.valid ? fb.joint_deg[0] : 0.0, s1.c_str(),
                 tgtStr(1).c_str(), fb.valid ? fb.joint_deg[1] : 0.0, s2.c_str(),
                 tgtStr(2).c_str(), fb.valid ? fb.joint_deg[2] : 0.0, s3.c_str(),
                 (unsigned long long)fbAge, commState.c_str(), faultCode, estopState, execState);
        std::cout << line << "\n";

        if (fb.valid) {
            for (int k = 0; k < 3; ++k) {
                if (fb.joint_deg[k] < minAct[k]) minAct[k] = fb.joint_deg[k];
                if (fb.joint_deg[k] > maxAct[k]) maxAct[k] = fb.joint_deg[k];
                if (hasSample) maxJump = std::max(maxJump, neckAbs(fb.joint_deg[k] - lastAct[k]));
                lastAct[k] = fb.joint_deg[k];
            }
            hasSample = true;
        }
        sample++;
        // 10 Hz 节拍
        std::this_thread::sleep_until(
            std::chrono::steady_clock::now() + std::chrono::milliseconds(100));
    }

    std::cout << "[StaticCheck] 采样 " << sample << " 行\n";
    if (hasSample) {
        std::cout << "[StaticCheck] 各电机实际角范围(度):\n"
                  << "  m1: [" << minAct[0] << ", " << maxAct[0] << "]  最大帧间跳变 "
                  << (sample > 1 ? maxJump : 0.0) << "\n"
                  << "  m2: [" << minAct[1] << ", " << maxAct[1] << "]\n"
                  << "  m3: [" << minAct[2] << ", " << maxAct[2] << "]\n";
    }
    std::cout << "[StaticCheck] fault_code=" << (anyFault ? "非零" : "0 (无)")
              << (anyFault ? "，请勿进行任何运动" : "，未发现故障码/反馈中断")
              << "\n[StaticCheck] 结束。若反馈中断/目标与实际持续偏离/抖动/异响：立即物理急停。\n";
    return anyFault ? 2 : 0;
}

// ==================== NeckCalibMove：标定运动（低速单轴小幅度） ====================
// 安全声明：只允许 单轴 ±calib.max_step_deg 以内、标定档限速（默认 10°/s）的运动；
// 全程在 CALIBRATION 状态（非 EXECUTING），受反馈/跟踪/限位监控，急停随时可触发。
unsigned neckCalibMove(const std::vector<std::string>& input) {
    int slaveId = 0, axis = 0;
    double delta = 0.0;
    try {
        if (input.size() != 4) {
            std::cout << "Command format error\n"
                      << "\tShould be \"NeckCalibMove <SlaveId> <Axis1..3> <DeltaDeg>\"\n";
            return 1;
        }
        slaveId = std::stoi(input[1]);
        axis = std::stoi(input[2]);
        delta = std::stod(input[3]);
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    if (!running || ec_slavecount <= 0 || slaveId < 0 || slaveId >= ec_slavecount) {
        std::cout << "[NeckCalibMove] 无 EtherCAT 会话/从站\n";
        return 1;
    }
    if (!primeFeedback(slaveId)) {
        std::cout << "[NeckCalibMove] 无法建立电机反馈（电机不应答？），拒绝执行\n";
        return 1;
    }

    if (!gTrajExecutor) {
        NeckControlConfig cfg = loadTrajControlConfig();
        NeckConfig ik = neckConfig();
        auto* adapter = new neck_control::EthercatNeckAdapter(slaveId, ik);
        gTrajExecutor = new NeckTrajectoryExecutor(cfg, ik, ExecutorMode::HARDWARE, adapter);
    }
    NeckError e = gTrajExecutor->runCalibrationMove(axis, delta);
    if (e != NeckError::OK) {
        std::cout << "[NeckCalibMove] 拒绝: " << neck_control::neckErrorString(e) << "\n";
        return 1;
    }
    std::cout << "[NeckCalibMove] 请目视确认；运动结束后用 MotorAngleGet 验证实际角（无异常声音/抖动）。\n";
    return 0;
}

// ==================== NeckCalibPose：标定姿态（三轴联动，低速小幅度） ====================
// 安全声明：目标头部姿态相对当前姿态各分量变化 <= calib.max_step_deg（5°），
// 标定档限速（10°/s），经 IK + 限位校验，CALIBRATION 状态 + 反馈/跟踪/急停保护。
unsigned neckCalibPose(const std::vector<std::string>& input) {
    int slaveId = 0;
    double pitch = 0.0, roll = 0.0, yaw = 0.0;
    try {
        if (input.size() != 5) {
            std::cout << "Command format error\n"
                      << "\tShould be \"NeckCalibPose <SlaveId> <PitchDeg> <RollDeg> <YawDeg>\"\n";
            return 1;
        }
        slaveId = std::stoi(input[1]);
        pitch = std::stod(input[2]);
        roll = std::stod(input[3]);
        yaw = std::stod(input[4]);
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    if (!running || ec_slavecount <= 0 || slaveId < 0 || slaveId >= ec_slavecount) {
        std::cout << "[NeckCalibPose] 无 EtherCAT 会话/从站\n";
        return 1;
    }
    if (!primeFeedback(slaveId)) {
        std::cout << "[NeckCalibPose] 无法建立电机反馈（电机不应答？），拒绝执行\n";
        return 1;
    }

    if (!gTrajExecutor) {
        NeckControlConfig cfg = loadTrajControlConfig();
        NeckConfig ik = neckConfig();
        auto* adapter = new neck_control::EthercatNeckAdapter(slaveId, ik);
        gTrajExecutor = new NeckTrajectoryExecutor(cfg, ik, ExecutorMode::HARDWARE, adapter);
    }
    NeckError e = gTrajExecutor->runCalibrationPose(pitch, roll, yaw);
    if (e != NeckError::OK) {
        std::cout << "[NeckCalibPose] 拒绝: " << neck_control::neckErrorString(e) << "\n";
        return 1;
    }
    std::cout << "[NeckCalibPose] 请目视确认；运动结束后用 MotorAngleGet 验证实际角（无异常声音/抖动）。\n";
    return 0;
}

// ==================== NeckDisable：收工/禁用（制动保持） ====================
// 停止一切轨迹并发布制动帧保持电机（等同软件急停的保持动作），
// 直到 NeckTrajAck 或重新下发命令。物理断电仍是最彻底的禁用方式。
unsigned neckDisable(const std::vector<std::string>& input) {
    int slaveId = 0;
    try {
        if (input.size() > 2) {
            std::cout << "Command format error\n\tShould be \"NeckDisable [SlaveId]\"\n";
            return 1;
        }
        if (input.size() == 2) slaveId = std::stoi(input[1]);
    } catch (const std::exception& e) {
        std::cout << "Parameter error" << e.what() << "\n";
        return 1;
    }
    if (!running || ec_slavecount <= 0 || slaveId < 0 || slaveId >= ec_slavecount) {
        std::cout << "[NeckDisable] 无 EtherCAT 会话/从站\n";
        return 1;
    }
    if (gTrajExecutor) {
        gTrajExecutor->stop();
        gTrajExecutor->estop(); // 触发制动帧保持（软件级禁用）
        std::cout << "[NeckDisable] 已停止轨迹并进入 ESTOP 制动保持\n";
    } else {
        // 无执行器：直接发布制动帧
        NeckConfig ik = neckConfig();
        neck_control::EthercatNeckAdapter adapter(slaveId, ik);
        adapter.emergencyStop();
        std::cout << "[NeckDisable] 已发布制动保持（无执行器）\n";
    }
    std::cout << "[NeckDisable] 注意：软件制动 ≠ 断电。离开前请按流程物理断电/急停。\n";
    return 0;
}
