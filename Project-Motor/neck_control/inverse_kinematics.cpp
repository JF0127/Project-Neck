#include "inverse_kinematics.h"

#include <cstdio>

#include "neck_kinematics.h"

namespace neck_control {

NeckError neckInverseSolve(const RpyDeg& rpyDeg, const NeckConfig& ik,
                           MotorAngles& out, std::string& message) {
    NeckAngles pose{rpyDeg.pitch, rpyDeg.roll, rpyDeg.yaw};
    ::MotorAngles m{}; // neck_kinematics.h 的全局 MotorAngles
    NeckStatus st = neckSolve(pose, ik, &m);
    switch (st) {
        case NECK_OK:
            out.motor1 = m.motor1;
            out.motor2 = m.motor2;
            out.motor3 = m.motor3;
            return NeckError::OK;
        case NECK_SINGULAR:
            message = neckStatusString(st);
            return NeckError::IK_SINGULAR;
        case NECK_POSE_OUT_OF_RANGE:
            message = neckStatusString(st);
            return NeckError::POSE_OUT_OF_RANGE;
        case NECK_MOTOR_OUT_OF_RANGE:
            message = neckStatusString(st);
            return NeckError::JOINT_OUT_OF_RANGE;
    }
    message = "未知逆解错误";
    return NeckError::INTERNAL;
}

RpyDeg neckForwardSolve(const MotorAngles& m, const NeckConfig& ik) {
    ::MotorAngles mm{m.motor1, m.motor2, m.motor3};
    NeckAngles pose = neckForward(mm, ik);
    RpyDeg out;
    out.pitch = pose.pitch;
    out.roll = pose.roll;
    out.yaw = pose.yaw;
    return out;
}

bool rpyWithinLimits(const RpyDeg& rpyDeg, const double rpyLimits[6]) {
    return rpyDeg.pitch >= rpyLimits[0] && rpyDeg.pitch <= rpyLimits[1] &&
           rpyDeg.roll >= rpyLimits[2] && rpyDeg.roll <= rpyLimits[3] &&
           rpyDeg.yaw >= rpyLimits[4] && rpyDeg.yaw <= rpyLimits[5];
}

bool jointsWithinLimits(const MotorAngles& m, const double jointLimits[6]) {
    return m.motor1 >= jointLimits[0] && m.motor1 <= jointLimits[1] &&
           m.motor2 >= jointLimits[2] && m.motor2 <= jointLimits[3] &&
           m.motor3 >= jointLimits[4] && m.motor3 <= jointLimits[5];
}

std::string rpyLimitReport(const RpyDeg& r, const double L[6]) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "pitch=%.2f[%.1f,%.1f] roll=%.2f[%.1f,%.1f] yaw=%.2f[%.1f,%.1f]",
             r.pitch, L[0], L[1], r.roll, L[2], L[3], r.yaw, L[4], L[5]);
    return buf;
}

} // namespace neck_control
