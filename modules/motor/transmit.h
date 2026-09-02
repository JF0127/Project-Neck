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
#include <stdint.h>
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

// Reads the latest real position feedback already decoded by the EtherCAT loop.
// Returns false unless all three requested motors have fresh angle feedback.
bool NeckFeedbackRead(int slave_id, const int passages[3],
                      const int motor_ids[3], double angles_deg[3],
                      uint64_t sequences[3]);

#ifdef __cplusplus
};
#endif

#endif // PROJECT_RT_ETHERCAT_H