// Neck Trajectory Executor 自动测试。
// 运行级别：静态/单元 + dry-run + mock。不连接硬件、不使能电机。
// 编译：无需 SOEM/网卡/root。 运行：neck_control_tests
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "neck_control/adapter_interface.h"
#include "neck_control/command_types.h"
#include "neck_control/config.h"
#include "neck_control/coordinate_calibration.h"
#include "neck_control/executor.h"
#include "neck_control/inverse_kinematics.h"
#include "neck_control/json_parser.h"
#include "neck_control/mock_adapter.h"
#include "neck_control/state_machine.h"
#include "neck_control/trajectory_io.h"
#include "neck_control/trajectory_retimer.h"
#include "neck_control/trajectory_validation.h"

using namespace neck_control;

// ---------------- 极简测试框架 ----------------
static int g_pass = 0, g_fail = 0;
static int g_current = 0;

#define TEST_BEGIN(name)                       \
    do {                                       \
        g_current++;                           \
        printf("\n=== 测试 %d: %s ===\n", g_current, name); \
    } while (0)

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (cond) {                                                          \
            g_pass++;                                                        \
        } else {                                                             \
            g_fail++;                                                        \
            printf("  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                    \
    } while (0)

#define CHECK_MSG(cond, ...)                                                 \
    do {                                                                     \
        if (cond) {                                                          \
            g_pass++;                                                        \
        } else {                                                             \
            g_fail++;                                                        \
            printf("  FAIL %s:%d  %s | ", __FILE__, __LINE__, #cond);         \
            printf(__VA_ARGS__);                                             \
            printf("\n");                                                    \
        }                                                                    \
    } while (0)

// ---------------- 公共配置 ----------------
// 测试用安全限位（比交付默认值宽松，缩短重定时时长；语义等价）
static NeckControlConfig testConfig() {
    NeckControlConfig c;
    c.loop_rate_hz = 100.0;
    c.command_timeout_ms = 100.0;
    c.feedback_timeout_ms = 200.0;
    c.tracking_error_timeout_ms = 300.0;
    c.max_start_pose_error_deg = 5.0;
    c.max_tracking_error_deg = 3.0;
    c.limits.max_velocity_deg_s[0] = 100.0;
    c.limits.max_velocity_deg_s[1] = 100.0;
    c.limits.max_velocity_deg_s[2] = 120.0;
    c.limits.max_acceleration_deg_s2[0] = 400.0;
    c.limits.max_acceleration_deg_s2[1] = 400.0;
    c.limits.max_acceleration_deg_s2[2] = 500.0;
    c.limits.max_jerk_deg_s3[0] = 2000.0;
    c.limits.max_jerk_deg_s3[1] = 2000.0;
    c.limits.max_jerk_deg_s3[2] = 2500.0;
    return c;
}

// 已标定 IK 参数（与 neck_calibration.md 一致）
static NeckConfig testIk() {
    NeckConfig ik = defaultNeckConfig();
    ik.c11 = 0.2170; ik.c12 = -0.2413; ik.c21 = 0.3218; ik.c22 = 0.2963; ik.k3 = 1.0;
    ik.m10 = 10.0; ik.m20 = 26.0; ik.m30 = 177.0;
    ik.p0 = 0.0; ik.r0 = 0.0; ik.y0 = 0.0;
    ik.pitch_min = -40.0; ik.pitch_max = 25.0;
    ik.roll_min = -35.0; ik.roll_max = 35.0;
    ik.yaw_min = -117.0; ik.yaw_max = 58.0;
    ik.motor1_min = -85.0; ik.motor1_max = 64.0;
    ik.motor2_min = -34.0; ik.motor2_max = 113.0;
    ik.motor3_min = 60.0; ik.motor3_max = 235.0;
    return ik;
}

// 生成命令：帧间线性插值从 start 到 end（模型坐标，弧度），fps 默认 30
static NeckTrajectoryCommand rampCommand(double pitch0, double pitch1, int frames,
                                         const char* lastState = "silent",
                                         double yaw0 = 0.0, double yaw1 = 0.0) {
    NeckTrajectoryCommand cmd;
    cmd.fps = 30.0;
    cmd.robot_actual_initial = {0, 0, 0};
    cmd.robot_neutral_pose = {0, 0, 0};
    for (int i = 0; i < frames; ++i) {
        double f = frames <= 1 ? 0.0 : (double)i / (frames - 1);
        cmd.trajectory_rpy.push_back(
            {degToRad(0.0), degToRad(pitch0 + (pitch1 - pitch0) * f),
             degToRad(yaw0 + (yaw1 - yaw0) * f)});
        cmd.states.push_back("speaking");
    }
    if (lastState) cmd.states.back() = lastState;
    if (lastState && std::string(lastState) == "silent") {
        cmd.trajectory_rpy.back() = cmd.robot_neutral_pose; // 终点=中位
    }
    return cmd;
}

static std::string cmdToJson(const NeckTrajectoryCommand& cmd) {
    std::string s = "{\"fps\": " + std::to_string(cmd.fps) + ",";
    s += "\"coordinate_convention\": {\"unit\": \"radian\", \"order\": [\"roll\",\"pitch\",\"yaw\"], "
         "\"rotation\": \"R = Ry(yaw) @ Rx(pitch) @ Rz(roll)\"},";
    s += "\"robot_actual_initial\": [" + std::to_string(cmd.robot_actual_initial[0]) + "," +
         std::to_string(cmd.robot_actual_initial[1]) + "," +
         std::to_string(cmd.robot_actual_initial[2]) + "],";
    s += "\"robot_neutral_pose\": [" + std::to_string(cmd.robot_neutral_pose[0]) + "," +
         std::to_string(cmd.robot_neutral_pose[1]) + "," +
         std::to_string(cmd.robot_neutral_pose[2]) + "],";
    s += "\"trajectory\": [";
    for (size_t i = 0; i < cmd.trajectory_rpy.size(); ++i) {
        if (i) s += ",";
        s += "[" + std::to_string(cmd.trajectory_rpy[i][0]) + "," +
             std::to_string(cmd.trajectory_rpy[i][1]) + "," +
             std::to_string(cmd.trajectory_rpy[i][2]) + "]";
    }
    s += "],\"states\": [";
    for (size_t i = 0; i < cmd.states.size(); ++i) {
        if (i) s += ",";
        s += "\"" + cmd.states[i] + "\"";
    }
    s += "]}";
    return s;
}

static double maxAbs3(const MotorAngles& a, const MotorAngles& b) {
    double e = std::fabs(a.motor1 - b.motor1);
    e = std::max(e, std::fabs(a.motor2 - b.motor2));
    e = std::max(e, std::fabs(a.motor3 - b.motor3));
    return e;
}

// ============================ 测试实现 ============================

static void test01_smallTrajectoryPasses() {
    TEST_BEGIN("正常小幅轨迹通过");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    NeckTrajectoryCommand cmd = rampCommand(0.0, 3.0, 10, "silent");
    ProcessResult r = ex.process(cmd);
    CHECK_MSG(r.error == NeckError::OK, "error=%d %s", (int)r.error, r.message.c_str());
    CHECK(r.frames == 10);
    CHECK(!r.samples.empty());
    CHECK(r.duration_s > 0.0);
    CHECK(r.sample_states.size() == r.samples.size());
    CHECK(r.sample_states.back() == "silent");
}

static void test02_nanInfRejected() {
    TEST_BEGIN("NaN/Inf 被拒绝");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    NeckTrajectoryCommand cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.trajectory_rpy[2][1] = std::nan("");
    CHECK(ex.process(cmd).error == NeckError::NON_FINITE);

    cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.trajectory_rpy[2][0] = std::numeric_limits<double>::infinity();
    CHECK(ex.process(cmd).error == NeckError::NON_FINITE);
}

static void test03_shapeFpsConventionRejected() {
    TEST_BEGIN("shape/fps/轴序/单位错误被拒绝");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    // 空轨迹
    NeckTrajectoryCommand empty;
    empty.fps = 30.0;
    CHECK(ex.process(empty).error == NeckError::EMPTY_TRAJECTORY);
    // fps
    NeckTrajectoryCommand cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.fps = 0.0;
    CHECK(ex.process(cmd).error == NeckError::FPS_OUT_OF_RANGE);
    cmd.fps = 1000.0;
    CHECK(ex.process(cmd).error == NeckError::FPS_OUT_OF_RANGE);
    // 单位
    cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.coordinate_convention.unit = "degree";
    CHECK(ex.process(cmd).error == NeckError::CONVENTION_MISMATCH);
    // 轴序
    cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.coordinate_convention.order = {"pitch", "roll", "yaw"};
    CHECK(ex.process(cmd).error == NeckError::CONVENTION_MISMATCH);
    // 旋转约定
    cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.coordinate_convention.rotation = "R = Rz(yaw) @ Rx(pitch) @ Ry(roll)";
    CHECK(ex.process(cmd).error == NeckError::CONVENTION_MISMATCH);
    // states 长度
    cmd = rampCommand(0.0, 3.0, 5, "silent");
    cmd.states.pop_back();
    CHECK(ex.process(cmd).error == NeckError::STATE_LENGTH_MISMATCH);
    // JSON 形状错误（行内元素数 != 3 / 非法 JSON）
    std::string badJson = "{\"fps\":30,\"trajectory\":[[0,0]]}";
    NeckTrajectoryCommand out;
    std::string msg;
    CHECK(loadNeckTrajectoryJson(badJson, out, msg) == NeckError::INVALID_INPUT);
    CHECK(loadNeckTrajectoryJson("{bad", out, msg) == NeckError::INVALID_JSON);
    // 防重放
    NeckTrajectoryCommand c1 = rampCommand(0.0, 3.0, 5, "silent");
    c1.command_id = 7;
    c1.timestamp = 1.0;
    CHECK(ex.process(c1).error == NeckError::OK);
    NeckTrajectoryCommand c2 = c1; // 重复 id
    CHECK(ex.process(c2).error == NeckError::DUPLICATE_COMMAND);
    NeckTrajectoryCommand c3 = c1;
    c3.command_id = 3; // 乱序
    CHECK(ex.process(c3).error == NeckError::OUT_OF_ORDER_COMMAND);
    NeckTrajectoryCommand c4 = c1;
    c4.command_id = 8;
    c4.timestamp = 0.5; // 时间戳乱序
    CHECK(ex.process(c4).error == NeckError::OUT_OF_ORDER_COMMAND);
}

static void test04_outOfAngleRejected() {
    TEST_BEGIN("超角度轨迹被拒绝");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    NeckTrajectoryCommand cmd = rampCommand(0.0, 100.0, 5, "silent"); // pitch 100°
    ProcessResult r = ex.process(cmd);
    CHECK(r.error == NeckError::POSE_OUT_OF_RANGE);
}

static void test05_06_07_retimedVelocityAccelJerk() {
    TEST_BEGIN("超速度/加速度/jerk 轨迹被安全重定时");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    // 名义速度: 5 帧从 0 到 15° pitch (电机 ~69°) @30fps → 远超 100°/s
    ProcessResult r = ex.process(rampCommand(0.0, 15.0, 5, "silent"));
    CHECK_MSG(r.error == NeckError::OK, "error=%d %s", (int)r.error, r.message.c_str());
    for (int j = 0; j < 3; ++j) {
        CHECK_MSG(r.max_vel[j] <= 100.0 * 1.001 + 1e-3, "vmax[%d]=%.3f", j, r.max_vel[j]);
        CHECK_MSG(r.max_acc[j] <= 400.0 * 1.01 + 1e-3, "amax[%d]=%.3f", j, r.max_acc[j]);
        CHECK_MSG(r.max_jerk[j] <= 2000.0 * 1.05 + 1e-3, "jmax[%d]=%.3f", j, r.max_jerk[j]);
    }
    // 单帧阶跃（jerk/加速度严重超标）
    NeckTrajectoryCommand step;
    step.fps = 30.0;
    step.robot_neutral_pose = {0, 0, 0};
    step.trajectory_rpy = {{0, 0, 0}, {0, 0, 0}, {0, degToRad(8.0), 0}, {0, 0, 0}, {0, 0, 0}};
    step.states = {"speaking", "speaking", "speaking", "speaking", "silent"};
    r = ex.process(step);
    CHECK_MSG(r.error == NeckError::OK, "error=%d %s", (int)r.error, r.message.c_str());
    for (int j = 0; j < 3; ++j) {
        CHECK_MSG(r.max_vel[j] <= 100.0 * 1.001 + 1e-3, "vmax[%d]=%.3f", j, r.max_vel[j]);
        CHECK_MSG(r.max_acc[j] <= 400.0 * 1.01 + 1e-3, "amax[%d]=%.3f", j, r.max_acc[j]);
        CHECK_MSG(r.max_jerk[j] <= 2000.0 * 1.1 + 1e-3, "jmax[%d]=%.3f", j, r.max_jerk[j]);
    }
}

static void test08_revalidateProcessed() {
    TEST_BEGIN("处理后轨迹重新验证全部约束");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    ProcessResult r = ex.process(rampCommand(0.0, 15.0, 5, "silent"));
    CHECK(r.error == NeckError::OK);
    const NeckControlConfig& cfg = testConfig();
    bool posOk = true, velOk = true, accOk = true, jrkOk = true;
    for (size_t i = 0; i < r.samples.size(); ++i) {
        if (!jointsWithinLimits(r.samples[i], cfg.joint_limits_deg)) posOk = false;
        RpyDeg fw = neckForwardSolve(r.samples[i], testIk());
        if (!rpyWithinLimits(fw, cfg.rpy_limits_deg)) posOk = false;
    }
    for (size_t i = 1; i < r.samples.size(); ++i) {
        double h = r.times[i] - r.times[i - 1];
        for (int j = 0; j < 3; ++j) {
            auto qj = [&](size_t k) {
                return j == 0 ? r.samples[k].motor1
                              : (j == 1 ? r.samples[k].motor2 : r.samples[k].motor3);
            };
            double v = (qj(i) - qj(i - 1)) / h;
            if (std::fabs(v) > cfg.vmax(j) * 1.001 + 1e-3) velOk = false;
        }
    }
    for (size_t i = 1; i + 1 < r.samples.size(); ++i) {
        double h1 = r.times[i] - r.times[i - 1];
        double h2 = r.times[i + 1] - r.times[i];
        if (h1 <= 0 || h2 <= 0) continue;
        for (int j = 0; j < 3; ++j) {
            auto qj = [&](size_t k) {
                return j == 0 ? r.samples[k].motor1
                              : (j == 1 ? r.samples[k].motor2 : r.samples[k].motor3);
            };
            double v0 = (qj(i) - qj(i - 1)) / h1;
            double v1 = (qj(i + 1) - qj(i)) / h2;
            double a = (v1 - v0) / (0.5 * (h1 + h2));
            if (std::fabs(a) > cfg.amax(j) * 1.01 + 1e-3) accOk = false;
            if (i + 2 < r.samples.size()) {
                double h3 = r.times[i + 2] - r.times[i + 1];
                if (h3 > 0) {
                    double v2 = (qj(i + 2) - qj(i + 1)) / h3;
                    double a1 = (v2 - v1) / (0.5 * (h2 + h3));
                    double jrk = (a1 - a) / (0.5 * (h1 + h2));
                    if (std::fabs(jrk) > cfg.jmax(j) * 1.05 + 1e-3) jrkOk = false;
                }
            }
        }
    }
    CHECK(posOk);
    CHECK(velOk);
    CHECK(accOk);
    CHECK(jrkOk);
}

static void test09_startPoseErrorNoJump() {
    TEST_BEGIN("起始姿态偏差过大禁止跳转");
    // mock 初始位置远离轨迹起点
    MotorAngles far{40.0, 60.0, 200.0};
    MockAdapter mock(far);
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    CHECK(ex.start(rampCommand(0.0, 3.0, 5, "silent")) == NeckError::START_POSE_ERROR_TOO_LARGE);
    CHECK(ex.state() != ExecutorState::EXECUTING);
}

static void test10_silentEndsAtNeutral() {
    TEST_BEGIN("Silent 最终到达 robot_neutral_pose");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    NeckTrajectoryCommand cmd = rampCommand(0.0, 4.0, 8, "silent");
    ProcessResult r = ex.process(cmd);
    CHECK(r.error == NeckError::OK);
    // 中位电机角 = IK(neutral)
    MotorAngles neutral{};
    std::string msg;
    CHECK(neckInverseSolve(RpyDeg{0, 0, 0}, testIk(), neutral, msg) == NeckError::OK);
    CHECK_MSG(maxAbs3(r.samples.back(), neutral) < 1e-6, "final 偏差 %.6f",
              maxAbs3(r.samples.back(), neutral));
    // silent 终点偏离中位 → 拒绝
    cmd = rampCommand(0.0, 4.0, 8, "silent");
    cmd.trajectory_rpy.back()[1] = degToRad(5.0);
    CHECK(ex.process(cmd).error == NeckError::SILENT_NOT_AT_NEUTRAL);
}

static void test11_axisSwapAndSign() {
    TEST_BEGIN("坐标轴交换与正负号映射正确");
    // 默认标定：模型 pitch(绕X) → 硬件 pitch(绕X)
    {
        NeckControlConfig cfg = testConfig();
        CoordinateCalibration cal(cfg);
        RpyDeg r = cal.modelToHardware({0.0, degToRad(10.0), 0.0});
        CHECK_MSG(std::fabs(r.pitch - 10.0) < 1e-9, "pitch=%.6f", r.pitch);
        CHECK(std::fabs(r.roll) < 1e-9 && std::fabs(r.yaw) < 1e-9);
    }
    // 约定交叉：模型 roll(绕Z) → 硬件 yaw(绕Z)；模型 yaw(绕Y) → 硬件 roll(绕Y)
    {
        CoordinateCalibration cal(testConfig());
        RpyDeg r = cal.modelToHardware({degToRad(12.0), 0.0, 0.0});
        CHECK_MSG(std::fabs(r.yaw - 12.0) < 1e-9, "yaw=%.6f", r.yaw);
        RpyDeg r2 = cal.modelToHardware({0.0, 0.0, degToRad(-8.0)});
        CHECK_MSG(std::fabs(r2.roll + 8.0) < 1e-9, "roll=%.6f", r2.roll);
    }
    // 符号映射
    {
        NeckControlConfig cfg = testConfig();
        cfg.axis_sign = {1.0, -1.0, 1.0};
        CoordinateCalibration cal(cfg);
        RpyDeg r = cal.modelToHardware({0.0, degToRad(10.0), 0.0});
        CHECK_MSG(std::fabs(r.pitch + 10.0) < 1e-9, "pitch=%.6f", r.pitch);
    }
    // 固定旋转 R_cal = Rz(-90°)：模型 yaw(绕Y, θ) → 硬件 roll=θ, yaw=-90°
    {
        NeckControlConfig cfg = testConfig();
        double R[9];
        CoordinateCalibration::rotZ(degToRad(-90.0), R);
        for (int i = 0; i < 9; ++i) cfg.rotation_matrix[i] = R[i];
        CoordinateCalibration cal(cfg);
        RpyDeg r = cal.modelToHardware({0.0, 0.0, degToRad(5.0)});
        CHECK_MSG(std::fabs(r.roll - 5.0) < 1e-9, "roll=%.6f", r.roll);
        CHECK_MSG(std::fabs(r.pitch) < 1e-9, "pitch=%.6f", r.pitch);
        CHECK_MSG(std::fabs(r.yaw + 90.0) < 1e-9, "yaw=%.6f", r.yaw);
    }
}

static void test12_rpyRotationMatrixConvention() {
    TEST_BEGIN("RPY 与旋转矩阵转换约定正确");
    CoordinateCalibration cal(testConfig());
    const double angles[][3] = {
        {0, 0, 0}, {0.1, -0.2, 0.3}, {-1.0, 0.5, 0.2}, {0.3, 1.2, -0.4}, {1.4, -1.1, 0.7}};
    for (const auto& a : angles) {
        double R1[9], R2[9];
        std::array<double, 3> model = {a[0], a[1], a[2]};
        CoordinateCalibration::modelRotationMatrix(model, R1);
        double e[3];
        CoordinateCalibration::extractHardwareEuler(R1, e);
        RpyDeg hw{radToDeg(e[0]), radToDeg(e[1]), radToDeg(e[2])};
        CoordinateCalibration::hardwareRpyToMatrix(hw, R2);
        double maxDiff = 0.0;
        for (int i = 0; i < 9; ++i)
            maxDiff = std::max(maxDiff, std::fabs(R1[i] - R2[i]));
        CHECK_MSG(maxDiff < 1e-9, "角度 (%.2f,%.2f,%.2f) 往返误差 %.2e", a[0], a[1], a[2],
                  maxDiff);
    }
    // 零角 = 单位阵
    double I[9];
    CoordinateCalibration::modelRotationMatrix({0, 0, 0}, I);
    bool identity = true;
    for (int i = 0; i < 9; ++i)
        if (std::fabs(I[i] - (i % 4 == 0 ? 1.0 : 0.0)) > 1e-12) identity = false;
    CHECK(identity);
}

static void test13_noClockDrift() {
    TEST_BEGIN("控制循环无累计 sleep 漂移");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    NeckError e = ex.start(rampCommand(0.0, 6.0, 6, "silent"));
    CHECK(e == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::READY, 8000));
    const LoopTelemetry& t = ex.telemetry();
    CHECK_MSG(t.ticks > 0, "ticks=%llu", (unsigned long long)t.ticks);
    CHECK_MSG(t.max_jitter_ms < 50.0, "max_jitter=%.2fms", t.max_jitter_ms);
    CHECK_MSG(std::fabs(t.actual_duration_s - t.planned_duration_s) < 0.15,
              "|%.3f-%.3f|", t.actual_duration_s, t.planned_duration_s);
    CHECK(t.last_error == NeckError::OK);
}

static void test14_commandTimeoutSafeStop() {
    TEST_BEGIN("指令超时进入安全停止");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    mock.setWriteFail(true); // 写入恒失败
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    NeckError e = ex.start(rampCommand(0.0, 3.0, 5, "silent"));
    CHECK(e == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK_MSG(ex.telemetry().last_error == NeckError::COMMAND_TIMEOUT, "error=%d",
              (int)ex.telemetry().last_error);
    CHECK(ex.state() != ExecutorState::FAULT);
}

static void test15_feedbackTimeoutSafeStop() {
    TEST_BEGIN("反馈超时进入安全停止");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    NeckError e = ex.start(rampCommand(0.0, 8.0, 8, "silent")); // 时长 > 2s
    CHECK(e == NeckError::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    mock.freezeFeedback(true);
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK_MSG(ex.telemetry().last_error == NeckError::FEEDBACK_TIMEOUT, "error=%d",
              (int)ex.telemetry().last_error);
    CHECK(ex.state() != ExecutorState::FAULT);
}

static void test16_trackingErrorFault() {
    TEST_BEGIN("跟踪误差持续超限进入 FAULT");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    mock.setDynamicsLag(0.5); // 执行器最大跟随 0.5°/s，远低于命令
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    NeckError e = ex.start(rampCommand(0.0, 15.0, 10, "silent"));
    CHECK(e == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::FAULT, 5000));
    CHECK_MSG(ex.telemetry().last_error == NeckError::TRACKING_ERROR, "error=%d",
              (int)ex.telemetry().last_error);
    // 确认可恢复（显式确认 → READY）
    CHECK(ex.acknowledge() == NeckError::OK);
    CHECK(ex.state() == ExecutorState::READY);
}

static void test17_estopFromAnyActiveState() {
    TEST_BEGIN("急停可从任何活动状态触发");
    MotorAngles neutral{10.0, 26.0, 177.0};

    // 从 READY
    {
        MockAdapter mock(neutral);
        NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
        ex.estop();
        CHECK(ex.state() == ExecutorState::ESTOP);
    }
    // 从 EXECUTING
    {
        MockAdapter mock(neutral);
        NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
        CHECK(ex.start(rampCommand(0.0, 6.0, 6, "silent")) == NeckError::OK);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        ex.estop();
        CHECK(ex.state() == ExecutorState::ESTOP);
    }
    // 从 STOPPING
    {
        MockAdapter mock(neutral);
        NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
        CHECK(ex.start(rampCommand(0.0, 6.0, 6, "silent")) == NeckError::OK);
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        ex.stop();
        CHECK(ex.waitForState(ExecutorState::STOPPING, 500));
        ex.estop();
        CHECK(ex.state() == ExecutorState::ESTOP);
    }
    // 从 CALIBRATION
    {
        MockAdapter mock(neutral);
        NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
        CHECK(ex.enable() == NeckError::OK);
        CHECK(ex.enterCalibration() == NeckError::OK);
        CHECK(ex.state() == ExecutorState::CALIBRATION);
        ex.estop();
        CHECK(ex.state() == ExecutorState::ESTOP);
    }
}

static void test18_estopNotClearedByNormalCommands() {
    TEST_BEGIN("ESTOP 不能被普通命令解除");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    CHECK(ex.start(rampCommand(0.0, 6.0, 6, "silent")) == NeckError::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    ex.estop();
    CHECK(ex.state() == ExecutorState::ESTOP);
    // 普通命令全部被拒
    CHECK(ex.start(rampCommand(0.0, 3.0, 4, "silent")) == NeckError::STATE_NOT_READY);
    CHECK(ex.stop() == NeckError::STATE_NOT_READY);
    CHECK(ex.enterCalibration() == NeckError::STATE_NOT_READY);
    // 显式确认后才恢复
    CHECK(ex.acknowledge() == NeckError::OK);
    CHECK(ex.state() == ExecutorState::READY);
}

static void test19_mockSpeakingListeningSilent() {
    TEST_BEGIN("mock 完成 Speaking→Listening→Silent 轨迹");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);

    NeckTrajectoryCommand cmd;
    cmd.fps = 30.0;
    cmd.robot_actual_initial = {0, 0, 0};
    cmd.robot_neutral_pose = {0, 0, 0};
    // speaking: 小幅摆动 → listening: 轻微 → silent: 中位（首帧=中位，避免起点偏差）
    const double frames[][3] = {{0, 0, 0}, {0, 2, 1}, {0, 3, 2}, {0, 2, -1},
                                {0, 1, 2}, {0, -1, 3}, {0, -2, 1}, {0, 0, 0}};
    for (int i = 0; i < 8; ++i) {
        cmd.trajectory_rpy.push_back(
            {degToRad(frames[i][0]), degToRad(frames[i][1]), degToRad(frames[i][2])});
        cmd.states.push_back(i < 4 ? "speaking" : (i < 7 ? "listening" : "silent"));
    }
    cmd.trajectory_rpy.back() = cmd.robot_neutral_pose;

    NeckError e = ex.start(cmd);
    CHECK_MSG(e == NeckError::OK, "start error=%d", (int)e);
    CHECK(ex.waitForState(ExecutorState::READY, 15000));
    CHECK(ex.state() == ExecutorState::READY);
    CHECK(ex.telemetry().last_error == NeckError::OK);
    CHECK_MSG(maxAbs3(mock.actual(), neutral) < 0.5, "最终位姿偏差 %.3f",
              maxAbs3(mock.actual(), neutral));
    // 状态序列覆盖 speaking/listening/silent
    const ProcessResult& r = ex.lastResult();
    bool hasSpeaking = false, hasListening = false, hasSilent = false;
    for (const std::string& s : r.sample_states) {
        if (s == "speaking") hasSpeaking = true;
        if (s == "listening") hasListening = true;
        if (s == "silent") hasSilent = true;
    }
    CHECK(hasSpeaking && hasListening && hasSilent);
    // 审计输出
    std::string msg;
    std::string dir = "/tmp/neck_audit_test";
    std::system(("mkdir -p " + dir).c_str());
    CHECK(ex.saveAudit(dir, msg));
    std::ifstream f1(dir + "/original_trajectory.json");
    std::ifstream f2(dir + "/processed_trajectory.json");
    CHECK(f1.good() && f2.good());
}

static void test20_dryRunNoHardware() {
    TEST_BEGIN("dry-run 不连接/不写入真实硬件");
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    ProcessResult r = ex.process(rampCommand(0.0, 3.0, 6, "silent"));
    CHECK(r.error == NeckError::OK);
    CHECK(ex.state() == ExecutorState::DISABLED);
    CHECK(ex.start(rampCommand(0.0, 3.0, 4, "silent")) == NeckError::MODE_NOT_SUPPORTED);
    CHECK(ex.telemetry().writes_ok == 0);
    CHECK(ex.telemetry().writes_fail == 0);
    // 完整 JSON 往返（loadNeckTrajectoryJson）
    NeckTrajectoryCommand cmd = rampCommand(0.0, 2.0, 4, "silent");
    std::string json = cmdToJson(cmd);
    NeckTrajectoryCommand parsed;
    std::string msg;
    CHECK(loadNeckTrajectoryJson(json, parsed, msg) == NeckError::OK);
    CHECK(parsed.frameCount() == 4);
    CHECK(parsed.states.size() == 4);
    CHECK(parsed.states.back() == "silent");
    CHECK(std::fabs(parsed.fps - 30.0) < 1e-9);
}

static void test_extra_stateMachine() {
    TEST_BEGIN("状态机迁移规则");
    StateMachine sm;
    CHECK(sm.transition(ExecutorState::READY, "enable"));
    CHECK(sm.transition(ExecutorState::EXECUTING, "start"));
    CHECK(!sm.transition(ExecutorState::EXECUTING, "重复启动")); // 非法
    CHECK(sm.transition(ExecutorState::STOPPING, "stop"));
    CHECK(sm.transition(ExecutorState::READY, "done"));
    CHECK(sm.estop("急停"));
    CHECK(sm.state() == ExecutorState::ESTOP);
    CHECK(!sm.transition(ExecutorState::EXECUTING, "普通命令解除急停")); // 非法
    CHECK(!sm.transition(ExecutorState::READY, "普通命令解除急停"));    // 非法
    CHECK(sm.acknowledge("显式确认"));                                  // 唯一恢复途径
    CHECK(sm.state() == ExecutorState::READY);
}

static void test_extra_retimerUnit() {
    TEST_BEGIN("S 曲线规划器单元测试");
    SCurveSegment seg;
    std::string msg;
    CHECK(TrajectoryRetimer::planSegment(0.0, 20.0, 100.0, 400.0, 2000.0, 0.01, seg, msg));
    CHECK(seg.duration > 0.0);
    CHECK_MSG(std::fabs(seg.q.back() - 20.0) < 1e-9, "终点 %.9f", seg.q.back());
    double maxV = 0, maxA = 0, maxJ = 0;
    for (size_t i = 1; i < seg.t.size(); ++i) {
        double h = seg.t[i] - seg.t[i - 1];
        if (h <= 0) continue;
        double v = (seg.q[i] - seg.q[i - 1]) / h;
        maxV = std::max(maxV, std::fabs(v));
        if (i > 1) {
            double h0 = seg.t[i - 1] - seg.t[i - 2];
            if (h0 > 0) {
                double v0 = (seg.q[i - 1] - seg.q[i - 2]) / h0;
                double a = (v - v0) / (0.5 * (h + h0));
                maxA = std::max(maxA, std::fabs(a));
                if (i > 2) {
                    double h2 = seg.t[i] - seg.t[i - 1];
                    double h3 = seg.t[i - 1] - seg.t[i - 2];
                    double v1 = (seg.q[i] - seg.q[i - 1]) / h2;
                    double v2 = (seg.q[i - 1] - seg.q[i - 2]) / h3;
                    double a1 = (v1 - v0) / (0.5 * (h2 + h3));
                    double jrk = (a1 - a) / (0.5 * (h + h0));
                    maxJ = std::max(maxJ, std::fabs(jrk));
                }
            }
        }
    }
    CHECK_MSG(maxV <= 100.0 * 1.001 + 1e-3, "maxV=%.3f", maxV);
    CHECK_MSG(maxA <= 400.0 * 1.01 + 1e-3, "maxA=%.3f", maxA);
    CHECK_MSG(maxJ <= 2000.0 * 1.05 + 1e-3, "maxJ=%.3f", maxJ);
    // 静止段
    CHECK(TrajectoryRetimer::planSegment(5.0, 5.0, 100.0, 400.0, 2000.0, 0.01, seg, msg));
    CHECK(seg.duration == 0.0);
}

static void test_extra_configLoader() {
    TEST_BEGIN("配置文件加载");
    std::string path = "/tmp/neck_test_config.txt";
    {
        std::ofstream f(path);
        f << "# 测试配置\n";
        f << "mode.hardware_enabled = true\n";
        f << "calib.confirmed = true\n";
        f << "safety.confirmed = true\n";
        f << "safety.max_velocity_deg_s = 10, 20, 30\n";
        f << "calib.axis_sign = -1, 1, 1\n";
        f << "calib.rotation_matrix = 1,0,0,0,1,0,0,0,1\n";
        f << "loop.rate_hz = 50\n";
    }
    NeckControlConfig c;
    CHECK(loadNeckControlConfig(path, c));
    CHECK(c.hardware_enabled && c.calib_confirmed && c.safety_confirmed);
    CHECK(std::fabs(c.vmax(0) - 10.0) < 1e-9 && std::fabs(c.vmax(2) - 30.0) < 1e-9);
    CHECK(std::fabs(c.axis_sign[0] + 1.0) < 1e-9);
    CHECK(std::fabs(c.loop_rate_hz - 50.0) < 1e-9);
    // 缺失文件 → 保守默认
    NeckControlConfig d;
    CHECK(!loadNeckControlConfig("/tmp/不存在_xxx.txt", d));
    CHECK(!d.hardware_enabled && !d.calib_confirmed && !d.safety_confirmed);
}

static void test_extra_integrationShippedConfig() {
    TEST_BEGIN("集成：交付配置文件 + 上游示例 JSON");
    // 交付配置：三把锁默认关闭
    NeckControlConfig cfg;
    std::string cfgText;
    bool loadedFile = readTextFile("../neck_control/neck_trajectory_config.txt", cfgText);
    CHECK(loadedFile);
    if (loadedFile) {
        std::string tmp = "/tmp/neck_shipped_cfg.txt";
        {
            std::ofstream f(tmp);
            f << cfgText;
        }
        CHECK(loadNeckControlConfig(tmp, cfg));
        // 交付配置文件为"实机运营态"文件（锁状态随会话开关变化，不断言）；
        // 无配置文件时的内置默认三把锁全关（见 test_extra_configLoader）。
        CHECK(std::fabs(cfg.vmax(0) - 25.0) < 1e-9); // 保守默认
        CHECK(std::fabs(cfg.loop_rate_hz - 100.0) < 1e-9);
    }
    // 上游示例 JSON
    std::string json;
    CHECK(readTextFile("../test/upstream_sample.json", json));
    NeckTrajectoryCommand cmd;
    std::string msg;
    CHECK(loadNeckTrajectoryJson(json, cmd, msg) == NeckError::OK);
    CHECK(cmd.frameCount() == 30);
    CHECK(cmd.states.size() == 30);
    // dry-run 全流程（保守限速也可即时计算）
    NeckControlConfig dryCfg = cfg;
    NeckTrajectoryExecutor ex(dryCfg, testIk(), ExecutorMode::DRY_RUN, nullptr);
    ProcessResult r = ex.process(cmd);
    CHECK_MSG(r.error == NeckError::OK, "error=%d %s", (int)r.error, r.message.c_str());
    CHECK(r.sample_states.back() == "silent");
    // mock 执行（用测试限速避免拉长演示时长）
    MockAdapter mock(MotorAngles{10.0, 26.0, 177.0});
    NeckControlConfig fast = testConfig();
    NeckTrajectoryExecutor ex2(fast, testIk(), ExecutorMode::MOCK, &mock);
    CHECK(ex2.start(cmd) == NeckError::OK);
    CHECK(ex2.waitForState(ExecutorState::READY, 30000));
    CHECK(ex2.telemetry().last_error == NeckError::OK);
    CHECK_MSG(maxAbs3(mock.actual(), MotorAngles{10.0, 26.0, 177.0}) < 0.5, "末位 %.3f",
              maxAbs3(mock.actual(), MotorAngles{10.0, 26.0, 177.0}));
}

static void test_extra_v1ModelOutput() {
    TEST_BEGIN("真实模型输出 V1 映射 + 完整 dry-run（mvp_stage1 89 帧）");
    std::string json;
    CHECK(readTextFile("../test/fixtures/mvp_stage1_trajectory.json", json));
    NeckTrajectoryCommand cmd;
    std::string msg;
    // 自动识别 V1 并显式映射
    CHECK_MSG(loadNeckTrajectoryJson(json, cmd, msg) == NeckError::OK, "%s", msg.c_str());
    CHECK(cmd.frameCount() == 89);
    CHECK(cmd.states.size() == 89);
    CHECK(cmd.states.back() == "silent");
    int nSpk = 0, nLis = 0, nSil = 0;
    for (const auto& s : cmd.states) {
        if (s == "speaking") nSpk++;
        else if (s == "listening") nLis++;
        else nSil++;
    }
    CHECK(nSpk == 22 && nLis == 22 && nSil == 45);
    // 组合语义：R(ai) @ R(rel[-1]) == R(neutral)，与模型侧交付校验一致
    {
        // 取原始 rel 末帧与 ai/neutral（直接读原始 JSON 字段）
        json::ValuePtr root = json::parse(json, &msg);
        json::ValuePtr t = root->find("trajectory");
        std::array<double, 3> ai, np, rel;
        auto rd3 = [&](const json::ValuePtr& v, std::array<double, 3>& o) {
            for (int i = 0; i < 3; ++i) o[i] = v->arr[i]->numVal;
        };
        rd3(t->find("robot_actual_initial"), ai);
        rd3(t->find("robot_neutral_pose"), np);
        const auto& rpyArr = t->find("rpy")->arr;
        rd3(rpyArr.back(), rel);
        double Rai[9], Rrel[9], Rabs[9], Rnp[9];
        CoordinateCalibration::modelRotationMatrix(ai, Rai);
        CoordinateCalibration::modelRotationMatrix(rel, Rrel);
        CoordinateCalibration::modelRotationMatrix(np, Rnp);
        CoordinateCalibration::matMul(Rai, Rrel, Rabs);
        double maxDiff = 0.0;
        for (int i = 0; i < 9; ++i)
            maxDiff = std::max(maxDiff, std::fabs(Rabs[i] - Rnp[i]));
        CHECK_MSG(maxDiff < 1e-5, "组合偏差 %.2e", maxDiff);
        // 映射后绝对末帧 ≈ neutral（弧度，容差 1e-3）
        for (int i = 0; i < 3; ++i)
            CHECK(std::fabs(cmd.trajectory_rpy.back()[i] - np[i]) < 1e-3);
        // 映射后绝对首帧 ≈ actual_initial
        for (int i = 0; i < 3; ++i)
            CHECK(std::fabs(cmd.trajectory_rpy.front()[i] - ai[i]) < 1e-3);
    }
    // 完整 dry-run：重定时比例显著（38°/s 模型峰值被安全拉长），峰值全部低于保守限速
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::DRY_RUN, nullptr);
    ProcessResult r = ex.process(cmd);
    CHECK_MSG(r.error == NeckError::OK, "error=%d %s", (int)r.error, r.message.c_str());
    CHECK(r.nominal_duration_s > 0.0);
    double ratio = r.duration_s / r.nominal_duration_s;
    CHECK_MSG(ratio > 5.0, "重定时比例 %.2f", ratio);
    for (int j = 0; j < 3; ++j) {
        CHECK(r.max_vel[j] <= testConfig().vmax(j) * 1.001 + 1e-3);
        CHECK(r.max_acc[j] <= testConfig().amax(j) * 1.01 + 1e-3);
    }
    CHECK(r.start_pose_error_deg < 1.0);
    CHECK(r.sample_states.back() == "silent");
}

static void test_extra_watchdog() {
    TEST_BEGIN("后台执行 + 进程级 watchdog（控制循环心跳超时 → 制动 + FAULT）");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    mock.setNoiseDeg(0.0);
    NeckControlConfig cfg = testConfig();
    cfg.watchdog_timeout_ms = 300.0;
    NeckTrajectoryExecutor ex(cfg, testIk(), ExecutorMode::MOCK, &mock);

    // 异步执行：start() 返回时即已处于 EXECUTING（后台线程驱动）
    CHECK(ex.start(rampCommand(0.0, 6.0, 6, "silent")) == NeckError::OK);
    CHECK(ex.state() == ExecutorState::EXECUTING);
    // 控制通道立即可用：进度可查、stop() 立即可响应
    double el = 0.0, pl = 0.0;
    ex.progress(el, pl);
    CHECK(pl > 0.0 && el >= 0.0);

    // 让轨迹跑一会儿，然后模拟控制循环挂起
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    ex.testStallLoop(1500); // 控制循环在下次 tick 前挂起 1.5s
    // watchdog（300ms 无心跳）应触发制动 + FAULT
    std::this_thread::sleep_for(std::chrono::milliseconds(700));
    CHECK(ex.watchdogFired());
    CHECK_MSG(mock.estopActive(), "mock 制动未激活");
    CHECK(ex.state() == ExecutorState::FAULT);
    CHECK(ex.telemetry().last_error == NeckError::LOOP_OVERRUN);

    // 等控制循环从挂起中恢复（心跳恢复，运动标志清零）
    std::this_thread::sleep_for(std::chrono::milliseconds(1500));
    // 恢复确认：释放制动
    CHECK(ex.acknowledge() == NeckError::OK);
    CHECK(ex.state() == ExecutorState::READY);
    CHECK(!mock.estopActive());

    // 再次启动正常轨迹，验证 STOP 在运行中始终可达（异步控制通道）
    CHECK(ex.start(rampCommand(0.0, 6.0, 6, "silent")) == NeckError::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(ex.state() == ExecutorState::EXECUTING);
    CHECK(ex.stop() == NeckError::OK); // 运行中 STOP 立即可达
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK(ex.state() == ExecutorState::READY);
}

static void test_extra_hardwareGates() {
    TEST_BEGIN("实机模式默认关闭；EXECUTING 需三把锁");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    // 未使能
    {
        NeckControlConfig cfg = testConfig();
        NeckTrajectoryExecutor ex(cfg, testIk(), ExecutorMode::HARDWARE, &mock);
        CHECK(ex.enable() == NeckError::HARDWARE_DISABLED);
        CHECK(ex.state() == ExecutorState::DISABLED);
    }
    // 仅 hardware_enabled → READY 允许（供标定），但 EXECUTING 被拒
    {
        NeckControlConfig cfg = testConfig();
        cfg.hardware_enabled = true;
        NeckTrajectoryExecutor ex(cfg, testIk(), ExecutorMode::HARDWARE, &mock);
        CHECK(ex.enable() == NeckError::OK);
        CHECK(ex.state() == ExecutorState::READY);
        CHECK(ex.start(rampCommand(0.0, 3.0, 4, "silent")) == NeckError::HARDWARE_DISABLED);
    }
    // 标定确认但安全参数未确认 → EXECUTING 仍拒
    {
        NeckControlConfig cfg = testConfig();
        cfg.hardware_enabled = true;
        cfg.calib_confirmed = true;
        NeckTrajectoryExecutor ex(cfg, testIk(), ExecutorMode::HARDWARE, &mock);
        CHECK(ex.enable() == NeckError::OK);
        CHECK(ex.start(rampCommand(0.0, 3.0, 4, "silent")) == NeckError::HARDWARE_DISABLED);
    }
    // 三把锁全开 → enable + start 允许（随后急停清理）
    {
        NeckControlConfig cfg = testConfig();
        cfg.hardware_enabled = true;
        cfg.calib_confirmed = true;
        cfg.safety_confirmed = true;
        NeckTrajectoryExecutor ex(cfg, testIk(), ExecutorMode::HARDWARE, &mock);
        CHECK(ex.enable() == NeckError::OK);
        CHECK(ex.start(rampCommand(0.0, 3.0, 4, "silent")) == NeckError::OK);
        CHECK(ex.waitForState(ExecutorState::EXECUTING, 1000));
        ex.estop();
        CHECK(ex.state() == ExecutorState::ESTOP);
        CHECK(ex.acknowledge() == NeckError::OK);
        CHECK(ex.state() == ExecutorState::READY);
    }
}

static void test_extra_calibMove() {
    TEST_BEGIN("标定运动（mock）：低速单轴小幅度 + 保护");
    MotorAngles neutral{10.0, 26.0, 177.0};
    MockAdapter mock(neutral);
    mock.setNoiseDeg(0.0); // 确定性测试：关闭反馈噪声
    NeckTrajectoryExecutor ex(testConfig(), testIk(), ExecutorMode::MOCK, &mock);
    // 幅度超限拒绝
    CHECK(ex.runCalibrationMove(3, 10.0) == NeckError::POSE_OUT_OF_RANGE);
    // 非法轴
    CHECK(ex.runCalibrationMove(0, 2.0) == NeckError::INVALID_INPUT);
    // m3 +2°
    CHECK(ex.runCalibrationMove(3, 2.0) == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK_MSG(std::fabs(mock.actual().motor3 - 179.0) < 0.3, "m3=%.3f", mock.actual().motor3);
    // 运动中状态必须是 CALIBRATION（非 EXECUTING）
    CHECK(ex.runCalibrationMove(3, 2.0) == NeckError::OK);
    std::this_thread::sleep_for(std::chrono::milliseconds(120));
    CHECK(ex.state() == ExecutorState::CALIBRATION);
    // 运动中急停
    ex.estop();
    CHECK(ex.state() == ExecutorState::ESTOP);
    CHECK(ex.acknowledge() == NeckError::OK);
    // 回程
    CHECK(ex.runCalibrationMove(3, -2.0) == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK(std::fabs(mock.actual().motor3 - 177.0) < 0.3);
    // 标定姿态：三轴联动 (pitch 5°)
    // 预期: Δm1 = c22*5/det = 10.44, Δm2 = -c21*5/det = -11.34
    CHECK(ex.runCalibrationPose(5.0, 0.0, 0.0) == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK_MSG(std::fabs(mock.actual().motor1 - 20.44) < 0.5, "m1=%.3f", mock.actual().motor1);
    CHECK_MSG(std::fabs(mock.actual().motor2 - 14.66) < 0.5, "m2=%.3f", mock.actual().motor2);
    CHECK(std::fabs(mock.actual().motor3 - 177.0) < 0.3);
    // 相对当前姿态变化超过 5° → 拒绝
    CHECK(ex.runCalibrationPose(12.0, 0.0, 0.0) == NeckError::POSE_OUT_OF_RANGE);
    // 超姿态限位 → 拒绝 (yaw 100° > 58°)
    CHECK(ex.runCalibrationPose(0.0, 0.0, 100.0) == NeckError::POSE_OUT_OF_RANGE);
    // 回中位
    CHECK(ex.runCalibrationPose(0.0, 0.0, 0.0) == NeckError::OK);
    CHECK(ex.waitForState(ExecutorState::READY, 5000));
    CHECK(std::fabs(mock.actual().motor1 - 10.0) < 0.5);
    CHECK(std::fabs(mock.actual().motor2 - 26.0) < 0.5);
}

int main() {
    printf("Neck Trajectory Executor 自动测试（dry-run + mock，不连接硬件）\n");
    test01_smallTrajectoryPasses();
    test02_nanInfRejected();
    test03_shapeFpsConventionRejected();
    test04_outOfAngleRejected();
    test05_06_07_retimedVelocityAccelJerk();
    test08_revalidateProcessed();
    test09_startPoseErrorNoJump();
    test10_silentEndsAtNeutral();
    test11_axisSwapAndSign();
    test12_rpyRotationMatrixConvention();
    test13_noClockDrift();
    test14_commandTimeoutSafeStop();
    test15_feedbackTimeoutSafeStop();
    test16_trackingErrorFault();
    test17_estopFromAnyActiveState();
    test18_estopNotClearedByNormalCommands();
    test19_mockSpeakingListeningSilent();
    test20_dryRunNoHardware();
    test_extra_stateMachine();
    test_extra_retimerUnit();
    test_extra_configLoader();
    test_extra_integrationShippedConfig();
    test_extra_hardwareGates();
    test_extra_calibMove();
    test_extra_watchdog();
    test_extra_v1ModelOutput();

    printf("\n========================================\n");
    printf("通过 %d 项, 失败 %d 项\n", g_pass, g_fail);
    if (g_fail > 0) {
        printf("存在失败项！\n");
        return 1;
    }
    printf("全部通过。\n");
    return 0;
}
