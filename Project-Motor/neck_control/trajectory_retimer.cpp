#include "trajectory_retimer.h"

#include <algorithm>
#include <cmath>

namespace neck_control {

namespace {

// 加速段（jerk 上升 + 恒加速 + jerk 下降）总位移。
// 参数：J=jerk, Tj=jerk 段时长, Ta=恒加速段时长。
inline double rampDist(double J, double Tj, double Ta) {
    // s1 = J*Tj^3/6
    // s2 = J*Tj^2*Ta/2 + J*Tj*Ta^2/2
    // s3 = 5*J*Tj^3/6 + J*Tj^2*Ta
    double Tj3 = Tj * Tj * Tj;
    return J * Tj3 + 1.5 * J * Tj * Tj * Ta + 0.5 * J * Tj * Ta * Ta;
}

// 给定峰值速度 vp，返回满足 a/j 限制的 (Tj, Ta) 剖面参数。
inline void profileForPeak(double vp, double amax, double jmax, double& Tj, double& Ta) {
    if (vp <= amax * amax / jmax) {
        Tj = vp / amax;   // 未达到恒加速段，加速峰值 < amax
        Ta = 0.0;
    } else {
        Tj = amax / jmax;
        Ta = vp / amax - Tj;
    }
}

// 按剖面参数采样生成 S 曲线（相对位移 D>=0，方向 sgn）。
void sampleProfile(double q0, double sgn, double D, double J, double Tj, double Ta, double Tv,
                   double dt, SCurveSegment& out) {
    const double A = J * Tj;
    const double t1 = Tj;
    const double t2 = t1 + Ta;
    const double t3 = t2 + Tj;
    const double t4 = t3 + Tv;
    const double t5 = t4 + Tj;
    const double t6 = t5 + Ta;
    const double T = t6 + Tj; // == 4*Tj + 2*Ta + Tv

    // 各阶段起点状态（相对量）
    const double vA1 = 0.5 * J * Tj * Tj;             // P1 末速度
    const double qA1 = J * Tj * Tj * Tj / 6.0;        // P1 末位移
    const double vA2 = vA1 + A * Ta;                  // P2 末速度
    const double qA2 = qA1 + vA1 * Ta + 0.5 * A * Ta * Ta;
    const double vA3 = vA2 + A * Tj - 0.5 * J * Tj * Tj; // = 峰值 v1
    const double qA3 = qA2 + vA2 * Tj + 0.5 * A * Tj * Tj - J * Tj * Tj * Tj / 6.0;
    const double vA4 = vA3;                           // 巡航
    const double qA4 = qA3 + vA3 * Tv;
    const double vA5 = vA4 - 0.5 * J * Tj * Tj;       // P5 末
    const double qA5 = qA4 + vA4 * Tj - J * Tj * Tj * Tj / 6.0;
    const double vA6 = vA5 - A * Ta;                  // P6 末
    const double qA6 = qA5 + vA5 * Ta - 0.5 * A * Ta * Ta;
    // P7 末应恰好 (D, 0)

    auto eval = [&](double t, double& qr, double& vr) {
        if (t <= t1) {            // P1
            vr = 0.5 * J * t * t;
            qr = J * t * t * t / 6.0;
        } else if (t <= t2) {     // P2
            double lt = t - t1;
            vr = vA1 + A * lt;
            qr = qA1 + vA1 * lt + 0.5 * A * lt * lt;
        } else if (t <= t3) {     // P3
            double lt = t - t2;
            vr = vA2 + A * lt - 0.5 * J * lt * lt;
            qr = qA2 + vA2 * lt + 0.5 * A * lt * lt - J * lt * lt * lt / 6.0;
        } else if (t <= t4) {     // P4 巡航
            double lt = t - t3;
            vr = vA3;
            qr = qA3 + vA3 * lt;
        } else if (t <= t5) {     // P5
            double lt = t - t4;
            vr = vA4 - 0.5 * J * lt * lt;
            qr = qA4 + vA4 * lt - J * lt * lt * lt / 6.0;
        } else if (t <= t6) {     // P6
            double lt = t - t5;
            vr = vA5 - A * lt;
            qr = qA5 + vA5 * lt - 0.5 * A * lt * lt;
        } else {                  // P7
            double lt = t - t6;
            vr = vA6 - A * lt + 0.5 * J * lt * lt;
            qr = qA6 + vA6 * lt - 0.5 * A * lt * lt + J * lt * lt * lt / 6.0;
        }
    };

    out.t.clear(); out.q.clear(); out.v.clear();
    double lastT = -1e9;
    for (double t = 0.0; t <= T + 1e-12; t += dt) {
        if (t - lastT < 1e-12) continue;
        double qr, vr;
        eval(t, qr, vr);
        out.t.push_back(t);
        out.q.push_back(q0 + sgn * qr);
        out.v.push_back(sgn * vr);
        lastT = t;
    }
    // 精确终点
    if (out.t.empty() || out.t.back() < T - 1e-12) {
        out.t.push_back(T);
        out.q.push_back(q0 + sgn * D);
        out.v.push_back(0.0);
    } else {
        out.t.back() = T;
        out.q.back() = q0 + sgn * D;
        out.v.back() = 0.0;
    }
    out.duration = T;
}

} // namespace

TrajectoryRetimer::TrajectoryRetimer(const RetimerLimits& limits, double dt)
    : limits_(limits), dt_(dt) {}

bool TrajectoryRetimer::planSegment(double q0, double q1,
                                    double vmax, double amax, double jmax,
                                    double dt, SCurveSegment& out, std::string& message) {
    out = SCurveSegment{};
    if (!(vmax > 0.0 && amax > 0.0 && jmax > 0.0 && dt > 0.0)) {
        message = "规划参数必须为正";
        return false;
    }
    double D = std::fabs(q1 - q0);
    double sgn = q1 >= q0 ? 1.0 : -1.0;
    if (D < 1e-9) {
        out.duration = 0.0;
        out.t.push_back(0.0);
        out.q.push_back(q0);
        out.v.push_back(0.0);
        return true;
    }

    double Tj, Ta, Tv = 0.0, v1;
    // 满速剖面
    if (amax * amax / jmax >= vmax) {
        Tj = vmax / amax;
        Ta = 0.0;
    } else {
        Tj = amax / jmax;
        Ta = vmax / amax - Tj;
    }
    v1 = jmax * Tj * Tj + amax * Ta;
    double sRamp = rampDist(jmax, Tj, Ta);

    if (2.0 * sRamp <= D) {
        Tv = (D - 2.0 * sRamp) / v1;
    } else {
        // 距离不足：二分峰值速度
        double lo = 0.0, hi = v1;
        for (int it = 0; it < 80; ++it) {
            double vp = 0.5 * (lo + hi);
            double Tj2, Ta2;
            profileForPeak(vp, amax, jmax, Tj2, Ta2);
            double d = 2.0 * rampDist(jmax, Tj2, Ta2);
            if (d < D) lo = vp; else hi = vp;
        }
        double vp = 0.5 * (lo + hi);
        profileForPeak(vp, amax, jmax, Tj, Ta);
        v1 = vp;
        Tv = 0.0;
    }
    sampleProfile(q0, sgn, D, jmax, Tj, Ta, Tv, dt, out);
    return true;
}

bool TrajectoryRetimer::retime(const std::vector<MotorAngles>& frames, double fps,
                               double nominal_dt,
                               std::vector<double>& times,
                               std::vector<MotorAngles>& samples,
                               std::vector<std::array<double, 3>>& vel,
                               std::vector<double>& segment_durations,
                               std::string& message) const {
    times.clear(); samples.clear(); vel.clear(); segment_durations.clear();
    if (frames.size() < 2) {
        if (frames.empty()) {
            message = "帧序列为空";
            return false;
        }
        // 单帧：零时长，单个采样
        times.push_back(0.0);
        samples.push_back(frames[0]);
        vel.push_back({0.0, 0.0, 0.0});
        segment_durations.push_back(0.0);
        return true;
    }
    if (!(fps > 0.0 && dt_ > 0.0)) {
        message = "fps/dt 必须为正";
        return false;
    }

    std::array<SCurveSegment, 3> segs;
    double tGlobal = 0.0;
    for (size_t i = 0; i + 1 < frames.size(); ++i) {
        const MotorAngles& a = frames[i];
        const MotorAngles& b = frames[i + 1];
        double jd[3] = {0, 0, 0};
        for (int j = 0; j < 3; ++j) {
            const double q0 = (j == 0 ? a.motor1 : (j == 1 ? a.motor2 : a.motor3));
            const double q1 = (j == 0 ? b.motor1 : (j == 1 ? b.motor2 : b.motor3));
            if (!planSegment(q0, q1, limits_.vmax[j], limits_.amax[j], limits_.jmax[j],
                             dt_, segs[j], message))
                return false;
            jd[j] = segs[j].duration;
        }
        double Tseg = jd[0] > jd[1] ? jd[0] : jd[1];
        if (jd[2] > Tseg) Tseg = jd[2];
        // 保留名义节奏：每段不短于名义帧间隔（静止驻留补齐）
        if (nominal_dt > Tseg) Tseg = nominal_dt;
        // 段时长对齐到控制周期网格：保证数值微分良态（避免剖面终点非均匀采样）
        Tseg = std::ceil(Tseg / dt_ - 1e-12) * dt_;
        if (Tseg < 1e-12) Tseg = dt_;
        segment_durations.push_back(Tseg);

        // 段内采样网格：0, dt, 2dt, ..., Tseg（统一网格）
        std::vector<double> grid;
        {
            int nGrid = (int)std::llround(Tseg / dt_);
            for (int k = 0; k <= nGrid; ++k) grid.push_back((double)k * dt_);
            grid.back() = Tseg;
        }

        for (double t : grid) {
            // 段边界去重：与上一段最后一个采样同时间同位置时跳过
            if (!times.empty() && std::fabs(tGlobal + t - times.back()) < 1e-12)
                continue;
            MotorAngles q{};
            std::array<double, 3> v{};
            for (int j = 0; j < 3; ++j) {
                // 线性插值当前关节剖面（短剖面在 Tseg 前自然驻留）
                const SCurveSegment& s = segs[j];
                double qv, vv;
                if (s.t.size() > 1) {
                    size_t k = 0;
                    while (k + 1 < s.t.size() && s.t[k + 1] <= t + 1e-12) ++k;
                    if (k + 1 < s.t.size() && s.t[k + 1] > t) {
                        double f = (t - s.t[k]) / (s.t[k + 1] - s.t[k]);
                        qv = s.q[k] + f * (s.q[k + 1] - s.q[k]);
                        vv = s.v[k] + f * (s.v[k + 1] - s.v[k]);
                    } else {
                        qv = s.q.back();
                        vv = 0.0;
                    }
                } else {
                    qv = s.q.back();
                    vv = 0.0;
                }
                if (j == 0) { q.motor1 = qv; v[0] = vv; }
                else if (j == 1) { q.motor2 = qv; v[1] = vv; }
                else { q.motor3 = qv; v[2] = vv; }
            }
            times.push_back(tGlobal + t);
            samples.push_back(q);
            vel.push_back(v);
        }
        tGlobal += Tseg;
    }
    return true;
}

} // namespace neck_control
