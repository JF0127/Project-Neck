#include "coordinate_calibration.h"

#include <cmath>

namespace neck_control {

namespace {
inline double clamp1(double x) { return x > 1.0 ? 1.0 : (x < -1.0 ? -1.0 : x); }
} // namespace

CoordinateCalibration::CoordinateCalibration(const NeckControlConfig& cfg) : cfg_(cfg) {}

void CoordinateCalibration::rotX(double t, double R[9]) {
    double c = std::cos(t), s = std::sin(t);
    R[0] = 1; R[1] = 0; R[2] = 0;
    R[3] = 0; R[4] = c; R[5] = -s;
    R[6] = 0; R[7] = s; R[8] = c;
}

void CoordinateCalibration::rotY(double t, double R[9]) {
    double c = std::cos(t), s = std::sin(t);
    R[0] = c; R[1] = 0; R[2] = s;
    R[3] = 0; R[4] = 1; R[5] = 0;
    R[6] = -s; R[7] = 0; R[8] = c;
}

void CoordinateCalibration::rotZ(double t, double R[9]) {
    double c = std::cos(t), s = std::sin(t);
    R[0] = c; R[1] = -s; R[2] = 0;
    R[3] = s; R[4] = c; R[5] = 0;
    R[6] = 0; R[7] = 0; R[8] = 1;
}

void CoordinateCalibration::matMul(const double A[9], const double B[9], double C[9]) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += A[i * 3 + k] * B[k * 3 + j];
            C[i * 3 + j] = sum;
        }
}

void CoordinateCalibration::modelRotationMatrix(const std::array<double, 3>& aRad, double R[9]) {
    // R_model = Ry(a2) @ Rx(a1) @ Rz(a0)
    double Rz0[9], Rx1[9], Ry2[9], T[9];
    rotZ(aRad[0], Rz0);
    rotX(aRad[1], Rx1);
    rotY(aRad[2], Ry2);
    matMul(Rx1, Rz0, T);
    matMul(Ry2, T, R);
}

void CoordinateCalibration::extractHardwareEuler(const double R[9], double e[3]) {
    // R = Rz(e2) @ Rx(e1) @ Ry(e0)
    // 元素关系:
    //   r21 = sin(e1);  r20 = -cos(e1) sin(e0);  r22 = cos(e1) cos(e0)
    //   r01 = -sin(e2) cos(e1);  r11 = cos(e2) cos(e1)
    e[1] = std::asin(clamp1(R[7]));
    e[0] = std::atan2(-R[6], R[8]);
    e[2] = std::atan2(-R[1], R[4]);
}

void CoordinateCalibration::extractModelEuler(const double R[9], double e[3]) {
    // R = Ry(a2) @ Rx(a1) @ Rz(a0)
    // 元素关系:
    //   r12 = -sin(a1);  r10 = cos(a1) sin(a0);  r11 = cos(a1) cos(a0)
    //   r02 = sin(a2) cos(a1);  r22 = cos(a2) cos(a1)
    e[1] = -std::asin(clamp1(R[5])); // 注意索引: R[5] = r12
    e[0] = std::atan2(R[3], R[4]);   // r10, r11
    e[2] = std::atan2(R[2], R[8]);   // r02, r22
}

void CoordinateCalibration::hardwareRpyToMatrix(const RpyDeg& rpyDeg, double R[9]) {
    double Rr[9], Rp[9], Ry[9], T[9];
    rotY(degToRad(rpyDeg.roll), Rr);   // roll 绕 Y(前向)
    rotX(degToRad(rpyDeg.pitch), Rp);  // pitch 绕 X(横向)
    rotZ(degToRad(rpyDeg.yaw), Ry);    // yaw 绕 Z(竖直)
    matMul(Rp, Rr, T);
    matMul(Ry, T, R);
}

RpyDeg CoordinateCalibration::modelToHardware(const std::array<double, 3>& modelRpyRad) const {
    // 1. 模型旋转矩阵
    double Rm[9], Rhw[9];
    modelRotationMatrix(modelRpyRad, Rm);

    // 2. 固定旋转变换 R_hw = R_cal @ R_model
    matMul(cfg_.rotation_matrix.data(), Rm, Rhw);

    // 3. 提取硬件欧拉角（弧度，顺序 e0,e1,e2）
    double e[3];
    extractHardwareEuler(Rhw, e);

    // 4. 符号映射
    for (int i = 0; i < 3; ++i)
        e[i] *= cfg_.axis_sign[i];

    // 5. 按命名装配 RPY（度）+ 残余偏置
    const std::vector<std::string>& names = cfg_.hardware_axis_order;
    auto named = [&](const char* want, int fallbackIdx) -> double {
        for (int i = 0; i < 3 && (int)names.size() == 3; ++i)
            if (names[i] == want) return radToDeg(e[i]);
        return radToDeg(e[fallbackIdx]);
    };

    RpyDeg out;
    out.roll = named("roll", 0) + cfg_.rpy_offset_deg[0];
    out.pitch = named("pitch", 1) + cfg_.rpy_offset_deg[1];
    out.yaw = named("yaw", 2) + cfg_.rpy_offset_deg[2];
    return out;
}

} // namespace neck_control
