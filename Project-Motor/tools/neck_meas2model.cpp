// 实测电机角 → 头部姿态（硬件坐标）→ 模型坐标（robot_actual_initial 用）。
// 用法: neck_meas2model <m1> <m2> <m3>   (电机输出轴角, 度)
// 输出: 硬件坐标 (pitch, roll, yaw) 度；模型坐标 (roll, pitch, yaw) 弧度
// 模型坐标可直接作为上游模型的 robot_actual_initial 字段。
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "neck_config.h"
#include "neck_control/command_types.h"
#include "neck_control/coordinate_calibration.h"
#include "neck_control/inverse_kinematics.h"

static bool fileExists(const char* p) {
    std::ifstream f(p);
    return f.good();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("用法: neck_meas2model <m1> <m2> <m3>   (度)\n");
        return 1;
    }
    double m1 = std::atof(argv[1]);
    double m2 = std::atof(argv[2]);
    double m3 = std::atof(argv[3]);

    NeckConfig ik;
    {
        bool ok = false;
        const char* cands[] = {"neck_config.txt", "../neck_config.txt", "../../neck_config.txt"};
        for (const char* p : cands) {
            if (!fileExists(p)) continue;
            if (loadNeckConfig(p, ik)) { ok = true; break; }
        }
        if (!ok) {
            printf("错误: 找不到 neck_config.txt（请在项目根目录或 build 目录运行）\n");
            return 2;
        }
    }
    neck_control::NeckControlConfig cfg;
    {
        const char* cands[] = {"neck_control/neck_trajectory_config.txt",
                               "../neck_control/neck_trajectory_config.txt",
                               "../../neck_control/neck_trajectory_config.txt"};
        for (const char* p : cands) {
            if (!fileExists(p)) continue;
            if (loadNeckControlConfig(p, cfg)) break;
        }
    }

    neck_control::MotorAngles m{m1, m2, m3};
    if (!neck_control::jointsWithinLimits(m, cfg.joint_limits_deg)) {
        printf("警告: 电机角超出机械限位，结果不可信\n");
    }
    // 1) 硬件坐标姿态（正解，度）
    neck_control::RpyDeg hw = neck_control::neckForwardSolve(m, ik);
    printf("硬件坐标姿态(度): pitch=%.4f  roll=%.4f  yaw=%.4f\n", hw.pitch, hw.roll, hw.yaw);

    // 2) 模型坐标：同一物理旋转的模型欧拉表示 R=Ry(y)Rx(p)Rz(r)
    double Rhw[9], e[3];
    neck_control::CoordinateCalibration::hardwareRpyToMatrix(hw, Rhw);
    neck_control::CoordinateCalibration::extractModelEuler(Rhw, e);
    printf("模型坐标(rad)   : roll=%.9f  pitch=%.9f  yaw=%.9f\n", e[0], e[1], e[2]);
    printf("模型坐标(度)   : roll=%.6f  pitch=%.6f  yaw=%.6f\n",
           neck_control::radToDeg(e[0]), neck_control::radToDeg(e[1]), neck_control::radToDeg(e[2]));
    printf("\nrobot_actual_initial = [%.9f, %.9f, %.9f]\n", e[0], e[1], e[2]);
    return 0;
}
