/*
 * @Description:
 * @Author: kx zhang
 * @Date: 2022-09-20 11:17:58
 * @LastEditTime: 2022-11-13 17:16:18
 */
#ifndef TRANSMIT_H
#define TRANSMIT_H

#define SLAVE_NUMBER 4 //可该最大从机数

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include "config.h"
#include "ethercat.h"
#include "sys/time.h"

#ifdef __cplusplus
extern "C" {
#endif



void EtherCAT_Transmit();
void EtherCAT_Init(char *ifname);
void EtherCAT_Run();
void EtherCAT_Command_Set();
void startRun();
bool NeckFramePublish(int slaveId, const EtherCAT_Msg* frame);
void NeckFrameStop(int slaveId);
// 取从站电机反馈快照（通道 0..2 = 电机 1..3，角度为减速器输出轴角，度）。
bool NeckFeedbackGet(int slave, double joint_deg[3], uint8_t error[3],
                     double temperature[3], uint64_t* ts_ms);

// EtherCAT 通信状态快照（诊断用）。
typedef struct {
    int running;        // 通信线程运行中
    int slavecount;     // 识别到的从站数
    int wkc;            // 最近一次 workcounter
    int expected_wkc;   // 期望 workcounter
    int in_op;          // 从站处于 OPERATIONAL
} NeckCommSnapshot;
bool NeckCommSnapshotGet(NeckCommSnapshot* out);

#ifdef __cplusplus
};
#endif

#endif // PROJECT_RT_ETHERCAT_H
