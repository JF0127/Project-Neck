#include "motor_control.h"

#define KP_MIN 0.0f
#define KP_MAX 500.0f
#define KD_MIN 0.0f
#define KD_MAX 5.0f
#define POS_MIN -12.5f
#define POS_MAX 12.5f
#define SPD_MIN -18.0f
#define SPD_MAX 18.0f
#define T_MIN -30.0f
#define T_MAX 30.0f
#define I_MIN -30.0f
#define I_MAX 30.0f

union RV_TypeConvert
{
    float to_float;
    int to_int;
    unsigned int to_uint;
    uint8_t buf[4];
} rv_type_convert;

union RV_TypeConvert2
{
    int16_t to_int16;
    uint16_t to_uint16;
    uint8_t buf[2];
} rv_type_convert2;

MotorCommFbd motor_comm_fbd;

/**
 * @brief 设置电机零点。
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机的通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机唯一标识符（ID）。
 *
 * @return None
 */
void Motor_Setzero(EtherCAT_Msg* tx_msg, uint16_t passage, uint16_t motor_id)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].id = 0x7FF;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].dlc = 4;

    tx_msg->motor[passage - 1].data[0] = motor_id >> 8;
    tx_msg->motor[passage - 1].data[1] = motor_id & 0xff;
    tx_msg->motor[passage - 1].data[2] = 0x00;
    tx_msg->motor[passage - 1].data[3] = 0x03;
}


/**
 * @brief 设置电机的通信模式或零点设置。
 *
 * 需要使用CAN1设置。
 *
 * @param tx_msg 指向存储构建的CAN消息的指针。
 * @param motor_id  电机唯一标识符（ID）。
 * @param cmd       控制命令，0x00：无操作，0x01：自动模式，0x02：问答模式，0x03：设置电机零点。
 *
 * @return None
 */
void MotorSetting(EtherCAT_Msg* tx_msg, uint16_t motor_id, uint8_t cmd)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[0].id = 0x7FF;
    tx_msg->motor[0].rtr = 0;
    tx_msg->motor[0].dlc = 4;

    if (cmd == 0)
        return;

    tx_msg->motor[0].data[0] = motor_id >> 8;
    tx_msg->motor[0].data[1] = motor_id & 0xff;
    tx_msg->motor[0].data[2] = 0x00;
    tx_msg->motor[0].data[3] = cmd;
}

/**
 * @brief 重置电机ID。
 *
 * 使用CAN1重置，重置后电机ID为1。
 *
 * @param tx_msg 指向存储构建的CAN消息的指针。
 *
 * @return None
 */
void MotorIDReset(EtherCAT_Msg* tx_msg)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[0].id = 0x7FF;
    tx_msg->motor[0].dlc = 6;
    tx_msg->motor[0].rtr = 0;

    tx_msg->motor[0].data[0] = 0x7F;
    tx_msg->motor[0].data[1] = 0x7F;
    tx_msg->motor[0].data[2] = 0x00;
    tx_msg->motor[0].data[3] = 0x05;
    tx_msg->motor[0].data[4] = 0x7F;
    tx_msg->motor[0].data[5] = 0x7F;
}


/**
 * @brief 设置电机ID。
 *
 * 需要使用CAN1设置。
 *
 * @param tx_msg     指向存储构建的CAN消息的指针。
 * @param motor_id      电机ID。
 * @param motor_id_new  电机新ID。
 *
 * @return None
 */
void MotorIDSetting(EtherCAT_Msg* tx_msg, uint16_t motor_id, uint16_t motor_id_new)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[0].id = 0x7FF;
    tx_msg->motor[0].dlc = 6;
    tx_msg->motor[0].rtr = 0;

    tx_msg->motor[0].data[0] = motor_id >> 8;
    tx_msg->motor[0].data[1] = motor_id & 0xff;
    tx_msg->motor[0].data[2] = 0x00;
    tx_msg->motor[0].data[3] = 0x04;
    tx_msg->motor[0].data[4] = motor_id_new >> 8;
    tx_msg->motor[0].data[5] = motor_id_new & 0xff;
}

/**
 * @brief 读取电机的通信模式。
 *
 * 需要使用CAN1读取。
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param motor_id  电机的唯一标识符（ID）。
 *
 * @return None
 */
void MotorCommModeReading(EtherCAT_Msg* tx_msg, uint16_t motor_id)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[0].rtr = 0;
    tx_msg->motor[0].id = 0x7FF;
    tx_msg->motor[0].dlc = 4;

    tx_msg->motor[0].data[0] = motor_id >> 8;
    tx_msg->motor[0].data[1] = motor_id & 0xff;
    tx_msg->motor[0].data[2] = 0x00;
    tx_msg->motor[0].data[3] = 0x81;
}


/**
 * @brief 读取电机ID。
 *
 * 需要使用CAN1读取。
 *
 * @param tx_msg 存储构建CAN消息的指针。
 *
 * @return None
 */
void MotorIDReading(EtherCAT_Msg* tx_msg)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[0].rtr = 0;
    tx_msg->motor[0].id = 0x7FF;
    tx_msg->motor[0].dlc = 4;

    tx_msg->motor[0].data[0] = 0xFF;
    tx_msg->motor[0].data[1] = 0xFF;
    tx_msg->motor[0].data[2] = 0x00;
    tx_msg->motor[0].data[3] = 0x82;
}


/**
 * @brief 力位混合模式
 *
 * 使用问答模式
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机唯一标识符（ID）。
 * @param kp        参数kp范围：0.0 ~ 500.0f。
 * @param kd        参数kd范围：0.0 ~ 5.0f。
 * @param pos       期望位置，范围：-12.5 ~ 12.5rad。
 * @param spd       期望速度，范围：-18.0 ~ 18.0rad/s。
 * @param tor       电机前馈扭矩，范围：-30 ~ 30Nm。
 *
 * @return None
 */
void send_motor_ctrl_cmd(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, float kp, float kd, float pos,
                         float spd, float tor)
{
    int kp_int;
    int kd_int;
    int pos_int;
    int spd_int;
    int tor_int;

    tx_msg->can_ide = 1;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 8;

    if (kp > KP_MAX)
        kp = KP_MAX;
    else if (kp < KP_MIN)
        kp = KP_MIN;
    if (kd > KD_MAX)
        kd = KD_MAX;
    else if (kd < KD_MIN)
        kd = KD_MIN;
    if (pos > POS_MAX)
        pos = POS_MAX;
    else if (pos < POS_MIN)
        pos = POS_MIN;
    if (spd > SPD_MAX)
        spd = SPD_MAX;
    else if (spd < SPD_MIN)
        spd = SPD_MIN;
    if (tor > T_MAX)
        tor = T_MAX;
    else if (tor < T_MIN)
        tor = T_MIN;

    kp_int = float_to_uint(kp, KP_MIN, KP_MAX, 12);
    kd_int = float_to_uint(kd, KD_MIN, KD_MAX, 9);
    pos_int = float_to_uint(pos, POS_MIN, POS_MAX, 16);
    spd_int = float_to_uint(spd, SPD_MIN, SPD_MAX, 12);
    tor_int = float_to_uint(tor, T_MIN, T_MAX, 12);

    tx_msg->motor[passage - 1].data[0] = 0x00 | (kp_int >> 7); // kp5
    tx_msg->motor[passage - 1].data[1] = ((kp_int & 0x7F) << 1) | ((kd_int & 0x100) >> 8); // kp7+kd1
    tx_msg->motor[passage - 1].data[2] = kd_int & 0xFF;
    tx_msg->motor[passage - 1].data[3] = pos_int >> 8;
    tx_msg->motor[passage - 1].data[4] = pos_int & 0xFF;
    tx_msg->motor[passage - 1].data[5] = spd_int >> 4;
    tx_msg->motor[passage - 1].data[6] = (spd_int & 0x0F) << 4 | (tor_int >> 8);
    tx_msg->motor[passage - 1].data[7] = tor_int & 0xff;
}


/**
 * @brief 伺服位置控制
 *
 * 使用问答模式
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机的唯一标识符（ID）。
 * @param pos       期望位置，单位为度。
 * @param spd       期望速度，范围 0 ~ 18000，对应 0 ~ 1800.0rpm，比例为10：1
 * @param cur       电流阈值，范围 0 ~ 3000，对应 0 ~ 300.0A，比例为10：1
 * @param ack_status 报文返回状态，0：不返回，1：返回报文类型1，2：返回报文类型2，3：返回报文类型3。详见技术文档-问答模式反馈报文章节。
 */
void set_motor_position(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, float pos, uint16_t spd,
                        uint16_t cur, uint8_t ack_status)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 8;

    if (ack_status > 3)
        return;

    rv_type_convert.to_float = pos;
    tx_msg->motor[passage - 1].data[0] = 0x20 | (rv_type_convert.buf[3] >> 3);
    tx_msg->motor[passage - 1].data[1] = (rv_type_convert.buf[3] << 5) | (rv_type_convert.buf[2] >> 3);
    tx_msg->motor[passage - 1].data[2] = (rv_type_convert.buf[2] << 5) | (rv_type_convert.buf[1] >> 3);
    tx_msg->motor[passage - 1].data[3] = (rv_type_convert.buf[1] << 5) | (rv_type_convert.buf[0] >> 3);
    tx_msg->motor[passage - 1].data[4] = (rv_type_convert.buf[0] << 5) | (spd >> 10);
    tx_msg->motor[passage - 1].data[5] = (spd & 0x3FC) >> 2;
    tx_msg->motor[passage - 1].data[6] = (spd & 0x03) << 6 | (cur >> 6);
    tx_msg->motor[passage - 1].data[7] = (cur & 0x3F) << 2 | ack_status;
}


/**
 * @brief 伺服速度控制
 *
 * 使用问答模式
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机的通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机的唯一标识符（ID）。
 * @param spd       期望速度，范围 -18000 ~ 18000rpm。
 * @param cur       电流阈值，范围 0 ~ 3000, 对应 0 ~ 300.0A，比例10：1。
 * @param ack_status 报文返回状态，0：不返回，1~3：返回报文类型1~3。详见技术文档-问答模式反馈报文章节。
 */
void set_motor_speed(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, float spd, uint16_t cur,
                     uint8_t ack_status)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 7;

    rv_type_convert.to_float = spd;
    tx_msg->motor[passage - 1].data[0] = 0x40 | ack_status;
    tx_msg->motor[passage - 1].data[1] = rv_type_convert.buf[3];
    tx_msg->motor[passage - 1].data[2] = rv_type_convert.buf[2];
    tx_msg->motor[passage - 1].data[3] = rv_type_convert.buf[1];
    tx_msg->motor[passage - 1].data[4] = rv_type_convert.buf[0];
    tx_msg->motor[passage - 1].data[5] = cur >> 8;
    tx_msg->motor[passage - 1].data[6] = cur & 0xff;
}


/**
 * @brief 电流、力矩控制和刹车控制。
 *
 * 需要使用问答模式
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机的通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机的唯一标识符（ID）。
 * @param cur_tor   期望电流/力矩，范围 -3000 ~ 3000, 对应 -30.00A(Nm) ~ 30.00A(Nm)，比例100：1。
 * @param ctrl_status 控制状态，0：电流控制，1：力矩控制，2：变量阻尼制动控制（也叫全制动），3：能耗制动控制，4：再生制动控制。
 * @param ack_status 报文返回状态，0：不返回，1~3：返回报文类型1~3。详见技术文档-问答模式反馈报文章节。
 */
void set_motor_cur_tor(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, int16_t cur_tor,
                       uint8_t ctrl_status, uint8_t ack_status)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 3;

    if (ack_status > 3)
        return;
    if (ctrl_status > 7)
        return;
    if (ctrl_status) // enter torque control mode or brake mode
    {
        if (cur_tor > 3000)
            cur_tor = 3000;
        else if (cur_tor < -3000)
            cur_tor = -3000;
    }
    else
    {
        if (cur_tor > 2000)
            cur_tor = 2000;
        else if (cur_tor < -2000)
            cur_tor = -2000;
    }

    tx_msg->motor[passage - 1].data[0] = 0x60 | ctrl_status << 2 | ack_status;
    tx_msg->motor[passage - 1].data[1] = cur_tor >> 8;
    tx_msg->motor[passage - 1].data[2] = cur_tor & 0xff;
}


/**
 * @brief 电机加速度设置
 *
 * 电机加速度内部已限幅。加速度减小，运行更柔和，但是过小易超调，系统默认值为2000，即20rad/s
 * 使用问答模式
 *
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机的通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机的唯一标识符（ID）。
 * @param acc       加速度，范围 0 ~ 2000，对应 0 ~ 20.00rad/s，比例100：1。
 * @param ack_status 报文返回状态，0：不返回，1~3：返回报文类型1~3。详见技术文档-问答模式反馈报文章节。
 */
void set_motor_acceleration(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, uint16_t acc,
                            uint8_t ack_status)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 4;

    if (ack_status > 2)
        return;
    if (acc > 2000)
        acc = 2000;

    tx_msg->motor[passage - 1].data[0] = 0xC0 | ack_status;
    tx_msg->motor[passage - 1].data[1] = 0x01;
    tx_msg->motor[passage - 1].data[2] = acc >> 8;
    tx_msg->motor[passage - 1].data[3] = acc & 0xff;
}

// This function use in ask communication mode.
/*
motor_id:1~0x7FE
linkage:0~10000
speedKI:0~10000
ack_status:0/1
*/
void set_motor_linkage_speedKI(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, uint16_t linkage,
                               uint16_t speedKI, uint8_t ack_status)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 6;

    if (ack_status > 2)
        return;
    if (linkage > 10000)
        linkage = 10000;
    if (speedKI > 10000)
        speedKI = 10000;

    tx_msg->motor[passage - 1].data[0] = 0xC0 | ack_status;
    tx_msg->motor[passage - 1].data[1] = 0x02;
    tx_msg->motor[passage - 1].data[2] = linkage >> 8;
    tx_msg->motor[passage - 1].data[3] = linkage & 0xff;
    tx_msg->motor[passage - 1].data[4] = speedKI >> 8;
    tx_msg->motor[passage - 1].data[5] = speedKI & 0xff;
}

// This function use in ask communication mode.
/*
motor_id:1~0x7FE
fdbKP:0~10000
fbdKD:0~10000
ack_status:0/1
*/
void set_motor_feedbackKP_KD(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, uint16_t fdbKP,
                             uint16_t fdbKD, uint8_t ack_status)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 6;

    if (ack_status > 2)
        return;
    if (fdbKP > 10000)
        fdbKP = 10000;
    if (fdbKD > 10000)
        fdbKD = 10000;

    tx_msg->motor[passage - 1].data[0] = 0xC0 | ack_status;
    tx_msg->motor[passage - 1].data[1] = 0x03;
    tx_msg->motor[passage - 1].data[2] = fdbKP >> 8;
    tx_msg->motor[passage - 1].data[3] = fdbKP & 0xff;
    tx_msg->motor[passage - 1].data[4] = fdbKD >> 8;
    tx_msg->motor[passage - 1].data[5] = fdbKD & 0xff;
}


/**
 * @brief 电机控制参数查询
 * @param tx_msg 存储构建CAN消息的指针。
 * @param passage   电机的通道编号，CAN1（1，2，3），CAN2（4，5，6）。
 * @param motor_id  电机的唯一标识符（ID）。
 * @param param_cmd 参数查询命令，1：位置，2：速度，3：电流，4：功率，5：加速度，6：磁链观测增益，7：扰动补偿系数，8：反馈补偿系数，9阻尼系数。其他数据请查看技术文档。
 */
void get_motor_parameter(EtherCAT_Msg* tx_msg, uint8_t passage, uint16_t motor_id, uint8_t param_cmd)
{
    tx_msg->can_ide = 0;
    tx_msg->motor[passage - 1].rtr = 0;
    tx_msg->motor[passage - 1].id = motor_id;
    tx_msg->motor[passage - 1].dlc = 2;

    tx_msg->motor[passage - 1].data[0] = 0xE0;
    tx_msg->motor[passage - 1].data[1] = param_cmd;
}


void Rv_Message_Print(uint8_t ack_status, OD_Motor_Msg* motor_msg)
{
    if (ack_status == 0)
    {
        if (motor_comm_fbd.motor_fbd == 0x01)
        {
            printf("自动模式.\n");
        }
        else if (motor_comm_fbd.motor_fbd == 0x02)
        {
            printf("问答模式.\n");
        }
        else if (motor_comm_fbd.motor_fbd == 0x03)
        {
            printf("零点设置成功.\n");
        }
        else if (motor_comm_fbd.motor_fbd == 0x04)
        {
            printf("新设置的Id为: %d\n", motor_comm_fbd.motor_id);
        }
        else if (motor_comm_fbd.motor_fbd == 0x05)
        {
            printf("重置Id成功.\n");
        }
        else if (motor_comm_fbd.motor_fbd == 0x06)
        {
            printf("当前电机Id: %d\n", motor_comm_fbd.motor_id);
        }
        else if (motor_comm_fbd.motor_fbd == 0x80)
        {
            printf("查询失败.\n");
        }
    }
    else
    {
        if (motor_msg->motor_id == 0)
        {
            return;
        }
        switch (ack_status)
        {
        case 1:
            printf("motor id: %d\n", motor_msg->motor_id);
            printf("angle_actual_rad: %f\n", motor_msg->angle_actual_rad);
            printf("speed_actual_rad: %f\n", motor_msg->speed_actual_rad);
            printf("current_actual_float: %f\n", motor_msg->current_actual_float);
            printf("temperature: %d\n", motor_msg->temperature);
            break;
        case 2:
            printf("motor id: %d\n", motor_msg->motor_id);
            printf("angle_actual_float: %f\n", motor_msg->angle_actual_float);
            printf("current_actual_int: %d\n", motor_msg->current_actual_int);
            printf("temperature: %d\n", motor_msg->temperature);
            printf("current_actual_float: %f\n", motor_msg->current_actual_float);
            break;
        case 3:
            printf("motor id: %d\n", motor_msg->motor_id);
            printf("speed_actual_float: %f\n", motor_msg->speed_actual_float);
            printf("current_actual_int: %d\n", motor_msg->current_actual_int);
            printf("temperature: %d\n", motor_msg->temperature);
            printf("current_actual_float: %f\n", motor_msg->current_actual_float);
            break;
        case 4:
            if (motor_comm_fbd.motor_fbd == 0)
            {
                printf("配置成功.\n");
            }
            else
            {
                printf("配置失败.\n");
            }
            break;
        case 5:
            printf("motor id: %d\n", motor_msg->motor_id);
            if (motor_comm_fbd.INS_code == 1)
            {
                printf("angle_actual_float: %f\n", motor_msg->angle_actual_float);
            }
            else if (motor_comm_fbd.INS_code == 2)
            {
                printf("speed_actual_float: %f\n", motor_msg->speed_actual_float);
            }
            else if (motor_comm_fbd.INS_code == 3)
            {
                printf("current_actual_float: %f\n", motor_msg->current_actual_float);
            }
            else if (motor_comm_fbd.INS_code == 4)
            {
                printf("power: %f\n", motor_msg->power);
            }
            else if (motor_comm_fbd.INS_code == 5)
            {
                printf("acceleration: %d\n", motor_msg->acceleration);
            }
            else if (motor_comm_fbd.INS_code == 6)
            {
                printf("linkage_KP: %d\n", motor_msg->linkage_KP);
            }
            else if (motor_comm_fbd.INS_code == 7)
            {
                printf("speed_KI: %d\n", motor_msg->speed_KI);
            }
            else if (motor_comm_fbd.INS_code == 8)
            {
                printf("feedback_KP: %d\n", motor_msg->feedback_KP);
            }
            else if (motor_comm_fbd.INS_code == 9)
            {
                printf("feedback_KD: %d\n", motor_msg->feedback_KD);
            }
            break;
        case 6:
            printf("motor id: %d\n", motor_msg->motor_id);
            printf("angle_actual_int: %d\n", motor_msg->angle_actual_int);
            printf("speed_actual_int: %d\n", motor_msg->speed_actual_int);
            printf("current_actual_int: %d\n", motor_msg->current_actual_int);
            printf("temperature: %d\n", motor_msg->temperature);
            break;
        default:
            break;
        }
    }
}

static void handle_motor_config_message(Motor_Msg* motor_data, OD_Motor_Msg* motor_msg, int print)
{
    if (motor_data->data[2] != 0x01) // determine whether it is a motor feedback instruction
        return; // it is not a motor feedback instruction
    if ((motor_data->data[0] == 0xff) && (motor_data->data[1] == 0xFF))
    {
        motor_comm_fbd.motor_id = motor_data->data[3] << 8 | motor_data->data[4];
        motor_comm_fbd.motor_fbd = 0x06;
    }
    else if ((motor_data->data[0] == 0x80) && (motor_data->data[1] == 0x80)) // inquire failed
    {
        motor_comm_fbd.motor_id = 0;
        motor_comm_fbd.motor_fbd = 0x80;
    }
    else if ((motor_data->data[0] == 0x7F) && (motor_data->data[1] == 0x7F)) // reset ID succeed
    {
        motor_comm_fbd.motor_id = 1;
        motor_comm_fbd.motor_fbd = 0x05;
    }
    else
    {
        motor_comm_fbd.motor_id = motor_data->data[0] << 8 | motor_data->data[1];
        motor_comm_fbd.motor_fbd = motor_data->data[3];
    }
    if (print) Rv_Message_Print(0, motor_msg);
}

static void handle_response_mode(Motor_Msg* motor_data, OD_Motor_Msg* motor_msg, int ack_status, int print)
{
    // int motor_id_t = motor_data->id - 1;
    motor_id_check = motor_data->id;
    motor_msg->motor_id = motor_id_check;
    motor_msg->error = motor_data->data[0] & 0x1F;
    // printf("%d\n", motor_id_t);
    if (ack_status == 1) // response frame 1
    {
        int pos_int = motor_data->data[1] << 8 | motor_data->data[2];
        int spd_int = motor_data->data[3] << 4 | (motor_data->data[4] & 0xF0) >> 4;
        int cur_int = (motor_data->data[4] & 0x0F) << 8 | motor_data->data[5];

        motor_msg->angle_actual_rad = uint_to_float(pos_int, POS_MIN, POS_MAX, 16);
        motor_msg->speed_actual_rad = uint_to_float(spd_int, SPD_MIN, SPD_MAX, 12);
        motor_msg->current_actual_float = uint_to_float(cur_int, I_MIN, I_MAX, 12);
        motor_msg->temperature = (motor_data->data[6] - 50) / 2;
    }
    else if (ack_status == 2) // response frame 2
    {
        rv_type_convert.buf[0] = motor_data->data[4];
        rv_type_convert.buf[1] = motor_data->data[3];
        rv_type_convert.buf[2] = motor_data->data[2];
        rv_type_convert.buf[3] = motor_data->data[1];
        motor_msg->angle_actual_float = rv_type_convert.to_float;
        motor_msg->current_actual_int = motor_data->data[5] << 8 | motor_data->data[6];
        motor_msg->temperature = (motor_data->data[7] - 50) / 2;
        motor_msg->current_actual_float = motor_msg->current_actual_int / 100.0f;
    }
    else if (ack_status == 3) // response frame 3
    {
        rv_type_convert.buf[0] = motor_data->data[4];
        rv_type_convert.buf[1] = motor_data->data[3];
        rv_type_convert.buf[2] = motor_data->data[2];
        rv_type_convert.buf[3] = motor_data->data[1];
        motor_msg->speed_actual_float = rv_type_convert.to_float;
        motor_msg->current_actual_int = motor_data->data[5] << 8 | motor_data->data[6];
        motor_msg->temperature = (motor_data->data[7] - 50) / 2;
        motor_msg->current_actual_float = motor_msg->current_actual_int / 100.0f;
    }
    else if (ack_status == 4) // response frame 4
    {
        if (motor_data->dlc != 3)
            return;
        motor_comm_fbd.INS_code = motor_data->data[1];
        motor_comm_fbd.motor_fbd = motor_data->data[2];
    }
    else if (ack_status == 5) // response frame 5
    {
        motor_comm_fbd.INS_code = motor_data->data[1];

        if (motor_data->dlc == 6)
        {
            if (motor_comm_fbd.INS_code < 1 || motor_comm_fbd.INS_code > 4)
                return;

            rv_type_convert.buf[0] = motor_data->data[5];
            rv_type_convert.buf[1] = motor_data->data[4];
            rv_type_convert.buf[2] = motor_data->data[3];
            rv_type_convert.buf[3] = motor_data->data[2];

            float* table[4] = {
                &motor_msg->angle_actual_float,
                &motor_msg->speed_actual_float,
                &motor_msg->current_actual_float,
                &motor_msg->power,
            };
            *table[motor_comm_fbd.INS_code - 1] = rv_type_convert.to_float;
        }
        else if (motor_data->dlc == 4)
        {
            if (motor_comm_fbd.INS_code < 5 || motor_comm_fbd.INS_code > 9)
                return;

            uint16_t data = motor_data->data[2] << 8 | motor_data->data[3];

            uint16_t* table[5] = {
                &motor_msg->acceleration,
                &motor_msg->linkage_KP,
                &motor_msg->speed_KI,
                &motor_msg->feedback_KP,
                &motor_msg->feedback_KD,
            };
            *table[motor_comm_fbd.INS_code - 5] = data;
        }
    }
    if (print) Rv_Message_Print(ack_status, motor_msg);
}

static void handle_automatic_mode(Motor_Msg* motor_data, OD_Motor_Msg motor_msg[6], int print)
{
    int motor_id_t = motor_data->id - 0x205;
    motor_msg[motor_id_t].motor_id = motor_data->id;
    motor_msg[motor_id_t].angle_actual_int = (uint16_t)(motor_data->data[0] << 8 | motor_data->data[1]);
    motor_msg[motor_id_t].speed_actual_int = (int16_t)(motor_data->data[2] << 8 | motor_data->data[3]);
    motor_msg[motor_id_t].current_actual_int = (motor_data->data[4] << 8 | motor_data->data[5]);
    motor_msg[motor_id_t].temperature = motor_data->data[6];
    motor_msg[motor_id_t].error = motor_data->data[7];
    if (print) Rv_Message_Print(6, &motor_msg[motor_id_t]);
}

uint16_t motor_id_check = 0;

void RV_can_data_repack(EtherCAT_Msg* rx_msg, uint8_t comm_mode, OD_Motor_Msg motor_msg[6], uint8_t slave, int print)
{
    if (print) printf("slave %d msg:\n", slave);
    uint8_t ack_status = 0;
    for (int i = 0; i < 6; ++i)
    {
        if (rx_msg->motor[i].dlc == 0)
            continue;
        if (print) printf("motor%d message:\n", i);
        if (rx_msg->motor[i].id == 0x7FF)
        {
            handle_motor_config_message(&rx_msg->motor[i], &motor_msg[i], print);
        }
        else if (comm_mode == 0x00 && rx_msg->motor[i].dlc != 0) // Response mode
        {
            ack_status = rx_msg->motor[i].data[0] >> 5;
            handle_response_mode(&rx_msg->motor[i], &motor_msg[i], ack_status, print);
        }
        else if (comm_mode == 0x01 && rx_msg->motor[i].dlc != 0) // automatic feedback mode
        {
            handle_automatic_mode(&rx_msg->motor[i], motor_msg, print);
        }
    }
}

void RV_can_data_repack_all(EtherCAT_Msg *rx_msg, uint8_t comm_mode, OD_Motor_Msg (*motor_msg)[6], uint8_t slave_num, int print)
{
    for (int slave = 0; slave < slave_num; slave++)
    {
        RV_can_data_repack(&rx_msg[slave], comm_mode, motor_msg[slave], slave, print);
    }
}
