//
// Created by bismarck on 11/19/22.
//

#ifndef MASTERSTACK_COMMAND_H
#define MASTERSTACK_COMMAND_H

#include <iostream>
#include <vector>
#include <unistd.h>
#include <chrono>
#include "queue.h"

extern "C" {
#include "config.h"
#include "motor_control.h"
#include "transmit.h"
}

unsigned help(const std::vector<std::string> &);
unsigned motorIdGet(const std::vector<std::string> & input);
unsigned motorIdSet(const std::vector<std::string> & input);
unsigned motorIdReset(const std::vector<std::string> & input);
unsigned motorAngleGet(const std::vector<std::string> & input);
unsigned motorZeroSet(const std::vector<std::string> & input);
unsigned motorStop(const std::vector<std::string> & input);
unsigned motorSpeedSet(const std::vector<std::string> & input);
unsigned motorPositionSet(const std::vector<std::string> & input);
unsigned neckPoseSet(const std::vector<std::string> & input);
unsigned neckSquence(const std::vector<std::string> & input);
unsigned neckNpy(const std::vector<std::string> & input);

// Neck Trajectory Executor 命令
unsigned neckTrajDryRun(const std::vector<std::string> & input);
unsigned neckTrajMock(const std::vector<std::string> & input);
unsigned neckTrajRun(const std::vector<std::string> & input);
unsigned neckTrajStop(const std::vector<std::string> & input);
unsigned neckTrajEStop(const std::vector<std::string> & input);
unsigned neckTrajAck(const std::vector<std::string> & input);
unsigned neckTrajStatus(const std::vector<std::string> & input);
unsigned neckStaticCheck(const std::vector<std::string> & input);
unsigned neckCalibMove(const std::vector<std::string> & input);
unsigned neckCalibPose(const std::vector<std::string> & input);
unsigned neckDisable(const std::vector<std::string> & input);

#endif //MASTERSTACK_COMMAND_H
