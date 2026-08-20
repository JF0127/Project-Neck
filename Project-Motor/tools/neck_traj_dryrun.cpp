// 独立离线预检工具: 用执行层真实代码处理 L1 轨迹 JSON(V1 格式)。
// 管线: V1 解析 → 输入校验 → 坐标变换 → IK → 安全重定时 → 复验。
// 无需 SOEM / root / 硬件。也可用于实机前快速预检。
//
// 用法: build/neck_traj_dryrun <trajectory.json> [audit_dir]
//   (在 build/ 目录下运行, 自动找 ../neck_config.txt 与 ../neck_control/neck_trajectory_config.txt)
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>

#include "neck_config.h"
#include "neck_control/config.h"
#include "neck_control/coordinate_calibration.h"
#include "neck_control/executor.h"
#include "neck_control/inverse_kinematics.h"
#include "neck_control/mock_adapter.h"
#include "neck_control/trajectory_io.h"

using namespace neck_control;

static bool fileExists(const char* p) {
    std::ifstream f(p);
    return f.good();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("用法: neck_traj_dryrun <trajectory.json> [audit_dir] [--vmax a,b,c] [--amax a,b,c] [--jmax a,b,c]\n");
        printf("用执行层真实代码离线处理轨迹(V1 解析→校验→坐标变换→IK→重定时→复验)。\n");
        printf("--vmax/--amax/--jmax 覆盖安全限速(度/秒, 度/秒², 度/秒³), 用于同步实验。\n");
        return 1;
    }
    const char* traj = argv[1];

    std::string text;
    if (!readTextFile(traj, text)) {
        printf("错误: 无法读取 %s\n", traj);
        return 2;
    }
    NeckTrajectoryCommand cmd;
    std::string msg;
    NeckError e = loadNeckTrajectoryJson(text, cmd, msg);
    if (e != NeckError::OK) {
        printf("加载失败: %s (%s)\n", neckErrorString(e), msg.c_str());
        return 3;
    }

    // 真实配置(相对 build/ 运行目录)
    NeckControlConfig cfg;
    bool cfgLoaded = false;
    for (const char* p : {"neck_control/neck_trajectory_config.txt",
                          "../neck_control/neck_trajectory_config.txt"}) {
        if (fileExists(p)) { loadNeckControlConfig(p, cfg); cfgLoaded = true; break; }
    }
    if (!cfgLoaded) {
        printf("警告: 未找到 neck_trajectory_config.txt, 用内置默认\n");
        loadNeckControlConfig("", cfg);
    }
    NeckConfig ik;
    bool ikLoaded = false;
    for (const char* p : {"neck_config.txt", "../neck_config.txt"}) {
        if (fileExists(p)) { loadNeckConfig(p, ik); ikLoaded = true; break; }
    }
    if (!ikLoaded) {
        printf("警告: 未找到 neck_config.txt, 用内置默认\n");
        ik = defaultNeckConfig();
    }

    // 限速覆盖(实验用)
    auto parse3 = [](const char* s, double out[3]) -> bool {
        if (!s) return false;
        return sscanf(s, "%lf,%lf,%lf", &out[0], &out[1], &out[2]) == 3;
    };
    for (int i = 2; i < argc; ++i) {
        double v[3];
        if (strcmp(argv[i], "--vmax") == 0 && i + 1 < argc && parse3(argv[++i], v))
            for (int j = 0; j < 3; ++j) cfg.limits.max_velocity_deg_s[j] = v[j];
        else if (strcmp(argv[i], "--amax") == 0 && i + 1 < argc && parse3(argv[++i], v))
            for (int j = 0; j < 3; ++j) cfg.limits.max_acceleration_deg_s2[j] = v[j];
        else if (strcmp(argv[i], "--jmax") == 0 && i + 1 < argc && parse3(argv[++i], v))
            for (int j = 0; j < 3; ++j) cfg.limits.max_jerk_deg_s3[j] = v[j];
    }
    printf("限速: v=%g/%g/%g °/s  a=%g/%g/%g °/s²  j=%g/%g/%g °/s³\n",
           cfg.limits.max_velocity_deg_s[0], cfg.limits.max_velocity_deg_s[1],
           cfg.limits.max_velocity_deg_s[2], cfg.limits.max_acceleration_deg_s2[0],
           cfg.limits.max_acceleration_deg_s2[1], cfg.limits.max_acceleration_deg_s2[2],
           cfg.limits.max_jerk_deg_s3[0], cfg.limits.max_jerk_deg_s3[1],
           cfg.limits.max_jerk_deg_s3[2]);

    // mock 模式: 用模拟电机跑完整控制循环(需真实时间等待)
    bool mockMode = false;
    for (int i = 2; i < argc; ++i)
        if (strcmp(argv[i], "--mock") == 0) mockMode = true;
    if (mockMode) {
        CoordinateCalibration cal(cfg);
        RpyDeg ai = cal.modelToHardware(cmd.robot_actual_initial);
        MotorAngles init{};
        std::string imsg;
        if (neckInverseSolve(ai, ik, init, imsg) != NeckError::OK)
            neckInverseSolve(cal.modelToHardware(cmd.robot_neutral_pose), ik, init, imsg);
        MockAdapter* mock = new MockAdapter(init);
        NeckTrajectoryExecutor ex(cfg, ik, ExecutorMode::MOCK, mock);
        NeckError e = ex.start(cmd);
        if (e != NeckError::OK) {
            printf("❌ mock 启动失败: %s\n", neckErrorString(e));
            return 5;
        }
        double planned = ex.lastResult().duration_s;
        printf("mock 执行中(计划 %.1fs, 等待完成)...\n", planned);
        bool done = ex.waitForState(ExecutorState::READY,
                                    (int)(planned * 1000.0) + 60000);
        const LoopTelemetry& t = ex.telemetry();
        printf("%s 状态=%s 错误=%s(%s)\n", done ? "✅ mock 完成" : "❌ mock 超时",
               executorStateString(ex.state()), neckErrorString(t.last_error),
               t.last_error_message.c_str());
        printf("  发送=%d 失败=%d 反馈=%d 丢失=%d 抖动max=%.1fms 跟踪max=%.2f° watchdog=%s\n",
               t.writes_ok, t.writes_fail, t.feedback_ok, t.feedback_miss,
               t.max_jitter_ms, t.max_tracking_error_deg,
               ex.watchdogFired() ? "触发" : "未触发");
        return (done && t.last_error == NeckError::OK) ? 0 : 6;
    }


    NeckTrajectoryExecutor ex(cfg, ik, ExecutorMode::DRY_RUN, nullptr);
    ProcessResult r = ex.process(cmd);
    if (r.error != NeckError::OK) {
        printf("❌ 拒绝: %s (%s)\n", neckErrorString(r.error), r.message.c_str());
        return 4;
    }
    printf("✅ dry-run 通过\n");
    printf("  帧数: %d, 名义时长: %.2fs, 计划时长: %.2fs\n",
           r.frames, r.nominal_duration_s, r.duration_s);
    if (r.nominal_duration_s > 0) {
        double ratio = r.duration_s / r.nominal_duration_s;
        printf("  重定时比例(计划/名义): %.3f  %s\n", ratio,
               ratio > 1.15 ? "(偏高! 动作会被拉长, 语音可能脱同步)"
                            : "(≈1, 动作/语音同步良好)");
    }
    printf("  峰值速度(°/s)  : %.1f / %.1f / %.1f\n", r.max_vel[0], r.max_vel[1], r.max_vel[2]);
    printf("  峰值加速度(°/s²): %.1f / %.1f / %.1f\n",
           r.max_acc[0], r.max_acc[1], r.max_acc[2]);
    printf("  峰值 jerk(°/s³): %.1f / %.1f / %.1f\n",
           r.max_jerk[0], r.max_jerk[1], r.max_jerk[2]);
    printf("  起点偏差: %.3f°\n", r.start_pose_error_deg);
    if (argc >= 3) {
        if (ex.saveAudit(argv[2], msg))
            printf("  审计已写: %s\n", argv[2]);
        else
            printf("  审计写入失败: %s\n", msg.c_str());
    }
    return 0;
}
