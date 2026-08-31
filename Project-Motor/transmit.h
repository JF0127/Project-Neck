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

// Publishes one complete three-motor frame as a single control-time target.
// The real-time EtherCAT loop copies the latest active frame every cycle.
bool NeckFramePublish(int slave_id, const EtherCAT_Msg* frame);
void NeckFrameStop(int slave_id);

#ifdef __cplusplus
};
#endif

#endif // PROJECT_RT_ETHERCAT_H