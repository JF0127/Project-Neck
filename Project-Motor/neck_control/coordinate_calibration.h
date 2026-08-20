// 坐标系与标定映射层。
// 所有轴交换、正负号、固定旋转变换集中在配置中，业务代码不做零散的正负号/轴交换。
//
// 模型约定（上游）：R_model = Ry(a2) @ Rx(a1) @ Rz(a0)，分量命名见 model_axis_order。
// 硬件约定：        R_hw    = Rz(e2) @ Rx(e1) @ Ry(e0)，分量命名见 hardware_axis_order。
// 映射：            R_hw = R_calibration @ R_model，再按命名与符号提取硬件欧拉角。
// 大角度旋转使用旋转矩阵组合，不做逐轴简单相加。
#pragma once

#include <array>

#include "command_types.h"
#include "config.h"

namespace neck_control {

class CoordinateCalibration {
public:
    explicit CoordinateCalibration(const NeckControlConfig& cfg);

    // 模型坐标 RPY（弧度，[roll,pitch,yaw]）→ 硬件坐标 RPY（度）。
    RpyDeg modelToHardware(const std::array<double, 3>& modelRpyRad) const;

    // ---- 纯数学辅助（供单元测试与审计） ----

    // 按模型约定构建旋转矩阵 R_model = Ry(a2) @ Rx(a1) @ Rz(a0)（行主序 9 元素）。
    static void modelRotationMatrix(const std::array<double, 3>& aRad, double R[9]);

    // 从旋转矩阵按硬件约定提取欧拉角（弧度）：R = Rz(e2) @ Rx(e1) @ Ry(e0)。
    static void extractHardwareEuler(const double R[9], double e[3]);

    // 从旋转矩阵按模型约定提取欧拉角（弧度）：R = Ry(a2) @ Rx(a1) @ Rz(a0)。
    // 用于把绝对旋转矩阵转回模型坐标 RPY（V1 轨迹映射用）。
    static void extractModelEuler(const double R[9], double e[3]);

    // 硬件 RPY（度）→ 旋转矩阵 R = Rz(yaw) @ Rx(pitch) @ Ry(roll)（行主序）。
    static void hardwareRpyToMatrix(const RpyDeg& rpyDeg, double R[9]);

    // 矩阵乘法 C = A @ B（行主序）。
    static void matMul(const double A[9], const double B[9], double C[9]);

    // 绕 X/Y/Z 轴旋转矩阵（弧度）。
    static void rotX(double t, double R[9]);
    static void rotY(double t, double R[9]);
    static void rotZ(double t, double R[9]);

private:
    NeckControlConfig cfg_;
};

} // namespace neck_control
