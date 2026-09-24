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
#include "neck/neck_kinematics.h"

extern "C" {
#include "config.h"
#include "motor_control.h"
#include "transmit.h"
}

unsigned help(const std::vector<std::string> &);
unsigned motorIdGet(const std::vector<std::string> & input);
unsigned motorIdSet(const std::vector<std::string> & input);
unsigned motorIdReset(const std::vector<std::string> & input);
unsigned motorZeroSet(const std::vector<std::string> & input);
unsigned motorStop(const std::vector<std::string> & input);
unsigned motorSpeedSet(const std::vector<std::string> & input);
unsigned motorPositionSet(const std::vector<std::string> & input);
unsigned motorAngleGet(const std::vector<std::string> & input);
// Shared NeckPoseSet core used by the console command and the model socket.
// Validates the slave and configuration, runs inverse kinematics, prints the
// same target report as the command, and queues the three motor position
// commands. Returns 0 on success; error contains the reason on failure.
unsigned applyNeckPoseSet(int slave_id, const NeckPose& pose, std::string& error);
unsigned neckPoseSet(const std::vector<std::string> & input);
unsigned neckSequence(const std::vector<std::string> & input);
unsigned neckSequenceStop(const std::vector<std::string> & input);

#endif //MASTERSTACK_COMMAND_H
