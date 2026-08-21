#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""l1_demo.py: L1 一体化演示主控(单终端运行)。

一条命令启动整个 L1 轮转式对话:
    sudo python tools/live/l1_demo.py          # 实机(自动拉起 Motor 服务)
    python tools/live/l1_demo.py --mock        # 开发验证(服务端 mock, 无硬件)

主循环:
    键盘输入(select 3s 超时检测)
      ├─ 并行 TTS: 用户语音(不播放, 仅模型输入) + 机器人回复语音
      ├─ 倾听轨迹(模型 → 平滑 → 关键点化) → 下发执行 → 脖子倾听
      ├─ 说话轨迹(后台生成) → 倾听完成后下发
      │    → 轮询执行层段边界, 进入 speaking 段起点 → 播放机器人语音
      │    → 终端打印回复文字 → 说话动作随语音展开, 说完 ~0.5s 收尾 + 回中
      └─ 3s 无输入 → 执行器忙则下发回中轨迹打断, 否则保持

收工: Ctrl+C → 自动 NeckDisable → 提示关闭三把锁 + 物理断电。

关键设计(与 docs/INTERACTION_BLUEPRINT.md、L1_STATUS.md 一致):
  - 时间锚定: 说话段时长 = 语音时长 + 0.5s 收尾预算(关键点化保证执行层
    重定时比例 ≈ 1, 动作与语音同步);
  - 幅度标准化: keyframe 把段内最大位移映射到 S 曲线容量(放大或削峰,
    方向不变) —— 演示档 speaking 放大到 ~4-5°;
  - 语音触发: 依据执行层实际段边界(含过渡段), 不依赖名义时间。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

# 项目根(与 mvp.py 相同约定)
NET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NET_ROOT))

from tools.live.keyframe import (DEG2RAD, RAD2DEG, keyframe, load_limits,  # noqa: E402
                                 export_v1)
from tools.live.tts_align import tts_align, DEFAULT_VOICE, BOT_VOICE  # noqa: E402

DEFAULT_WORKDIR = NET_ROOT.parent / "neck_l1"
DEFAULT_CHECKPOINT = NET_ROOT / "outputs/neck_motion_v3/checkpoints/best.pt"
DEFAULT_WHISPER = NET_ROOT.parent / "model"
MOTOR_BIN = NET_ROOT.parent / "Project-Motor" / "build" / "master_stack_test"
MOTOR_CONFIG = NET_ROOT.parent / "Project-Motor" / "neck_control" / "neck_trajectory_config.txt"
SOCK_PATH = "/tmp/neck_ctl.sock"
RESP_END = b"###END###\n"
AUDIT_DIR = "/tmp/l1_demo_audit"

# 预置回复表(英文, 与训练数据语言一致; 后续可换规则/LLM)
REPLIES = [
    "Hello! I am very happy to see you.",
    "Nice to meet you. How are you doing today?",
    "That sounds interesting. Please tell me more.",
    "I see. Thank you for sharing that with me.",
    "It is a beautiful day today, do you agree?",
    "I am always glad to have a chat with you.",
    "Really? I would love to hear more about that.",
    "You make a good point. I think so too.",
]

# 关键点化参数(B 档: jerk=1000; 动作风格: 大而从容)
KP_INTERVAL = 1.0          # 关键点网格间隔(秒); 收尾误差 ≤ KP_INTERVAL
KP_MARGIN = 0.85           # S 曲线时间预算余量
KP_SMOOTH_SIGMA = 3.0      # 高斯平滑 σ(帧)
KP_MAX_SCALE = {"speaking": 4.0, "listening": 1.0}   # 说话放大到容量/倾听保持
KP_MOTOR_GAIN = 6.0        # 电机角/RPY 放大系数(执行层 S 曲线约束在电机角空间)

# 模型候选能量(°/s): 演示取高能量候选(幅度大); 倾听轻一档
ENERGY_SPEAKING = 6.0
ENERGY_LISTENING = 3.5

IDLE_TIMEOUT = 10.0        # 无按键超时(秒) → 自动回中(打字停顿不会误触发)
SPEAK_TAIL = 0.5           # 说话段收尾预算(秒, 语音结束后)
MIN_SILENT_TAIL = 0.8      # 说话轨迹回中段预算(秒)


# --------------------------------------------------------------------------- #
# Neck 服务客户端(Unix socket 行协议)
# --------------------------------------------------------------------------- #
class NeckClient:
    def __init__(self, sock_path: str = SOCK_PATH):
        self.sock_path = sock_path

    def send(self, line: str, timeout: float = 120.0) -> str:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect(self.sock_path)
        s.sendall((line + "\n").encode())
        buf = b""
        while RESP_END not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        s.close()
        return buf.decode(errors="replace")

    def alive(self) -> bool:
        try:
            self.send("help", timeout=3.0)
            return True
        except Exception:
            return False

    def run(self, path: Path, audit_dir: str, mock: bool) -> tuple[bool, str]:
        cmd = ("NeckTrajMock" if mock else "NeckTrajRun")
        resp = self.send(f"{cmd} {path} 0 {audit_dir}", timeout=600.0)
        body = resp.split("###END###")[0]
        # 成功标志(避免响应内其他“拒绝/失败”字样误报):
        #   Run 后台成功 → “实机执行中”; Mock 阻塞完成 → “状态=READY 错误=OK”
        ok = ("实机执行中" in body) or ("状态=READY" in body and "错误=OK" in body)
        return ok, resp

    def dryrun(self, path: Path) -> tuple[float | None, str]:
        """下发前预检, 返回 (重定时比例, 响应)。"""
        resp = self.send(f"NeckTrajDryRun {path} /tmp/l1_demo_dryrun", timeout=60.0)
        m = re.search(r"重定时比例.*?([0-9.]+)", resp)
        return (float(m.group(1)) if m else None), resp

    def status(self) -> dict | None:
        try:
            resp = self.send("NeckTrajStatus", timeout=10.0)
        except Exception:
            return None
        d: dict = {}
        m = re.search(r"状态=(\w+)", resp)
        d["state"] = m.group(1) if m else None
        m = re.search(r"进度=([0-9.]+)/([0-9.]+)s", resp)
        d["elapsed"] = float(m.group(1)) if m else 0.0
        d["planned"] = float(m.group(2)) if m else 0.0
        m = re.search(r"段边界\(s\):([ 0-9.]+)", resp)
        d["seg_boundaries"] = ([float(x) for x in m.group(1).split()]
                               if m else [])
        return d

    def estop(self) -> str:
        return self.send("NeckTrajEStop", timeout=10.0)

    def ack(self) -> str:
        return self.send("NeckTrajAck", timeout=10.0)

    def disable(self) -> str:
        return self.send("NeckDisable", timeout=10.0)


# --------------------------------------------------------------------------- #
# 轨迹生成(模型常驻, 内联调用)
# --------------------------------------------------------------------------- #
class TrajGen:
    def __init__(self, checkpoint: Path, workdir: Path, limits: dict,
                 mock: bool):
        self.workdir = workdir
        self.limits = limits
        self.mock = mock
        import torch
        from models.neck_motion.mvp import generate
        self._generate = generate
        self._torch = torch
        self.ckpt = str(checkpoint)
        self.device = "cpu"   # 推理量小, CPU 足够(避免 CUDA 初始化延迟)
        # 预加载模型(常驻)
        print("[demo] 加载 v3 模型(常驻)...")
        t0 = time.time()
        self._warm = self._warmup()
        print(f"[demo] 模型就绪 ({time.time()-t0:.1f}s)")

    def _warmup(self):
        # 空 events 跑一次, 触发模型加载
        spec = {"robot_actual_initial": [0, 0, 0], "robot_neutral_pose": [0, 0, 0],
                "events": [{"state": "silent", "duration": 0.2}]}
        return self._generate(spec, self.ckpt, str(self.workdir / "warm"),
                              data_root=str(self.workdir),
                              device=self.device, strategy="energy_match",
                              expected_energy=3.0, export_json=None)

    def _make_events(self, frag: dict, silent_tail: float) -> dict:
        state = "listening" if frag["role"] == "listener" else "speaking"
        return {
            "robot_actual_initial": [0.0, 0.0, 0.0],
            "robot_neutral_pose": [0.0, 0.0, 0.0],
            "events": [
                {"state": state, "fragment": frag},
                {"state": "silent", "duration": silent_tail},
            ],
        }

    def _gen_kp(self, role: str, text: str, audio_rel: str, words: list,
                energy: float, silent_tail: float, out_path: Path,
                max_scale: float) -> dict:
        """fragment → 模型轨迹 → 关键点轨迹 → V1 JSON, 返回 doc。"""
        last_end = max((w["end_time"] for w in words), default=0.0)
        n_frames = max(2, round((last_end + SPEAK_TAIL) * 30.0))
        frag = {
            "role": role,
            "current_utterance": {
                "text": text,
                "audio_path": audio_rel,
                "word_timestamps": words,
            },
            "fps": 30.0,
            "num_frames": n_frames,
        }
        spec = self._make_events(frag, silent_tail)
        traj, states, summaries, _ = self._generate(
            spec, self.ckpt, str(self.workdir), data_root=str(self.workdir),
            device=self.device, strategy="energy_match",
            expected_energy=energy, export_json=None)
        neutral = np_zero3()
        kp, kp_states, meta = keyframe(
            traj, states, neutral, self.limits,
            interval_target=KP_INTERVAL,
            margin=KP_MARGIN, smooth_sigma=KP_SMOOTH_SIGMA,
            max_scale={"speaking": max_scale, "listening": max_scale},
            motor_gain=KP_MOTOR_GAIN)
        doc = export_v1(kp, kp_states, meta, [0.0, 0.0, 0.0], neutral)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        return doc

    def listen(self, text: str, audio_rel: str, words: list,
               out_path: Path) -> dict:
        return self._gen_kp("listener", text, audio_rel, words,
                            ENERGY_LISTENING, 0.6, out_path,
                            max_scale=KP_MAX_SCALE["listening"])

    def speak(self, text: str, audio_rel: str, words: list,
              out_path: Path) -> dict:
        return self._gen_kp("speaker", text, audio_rel, words,
                            ENERGY_SPEAKING, 1.0, out_path,
                            max_scale=KP_MAX_SCALE["speaking"])

    def home(self, out_path: Path) -> dict:
        """回中轨迹: 2 点 silent 静止于 neutral(执行层自动从实际位置平滑过渡)。"""
        n = np_zero3()
        doc = {
            "format_version": "1.0",
            "trajectory": {
                "fps": 1.0 / KP_INTERVAL,
                "units": "radian",
                "rpy_order": "roll,pitch,yaw",
                "rotation_convention": "R = Ry(yaw) @ Rx(pitch) @ Rz(roll)",
                "reference": "home: hold at neutral (executor transitions from actual pose)",
                "n_frames": 2,
                "robot_actual_initial": [0.0, 0.0, 0.0],
                "robot_neutral_pose": [0.0, 0.0, 0.0],
                "states": [0, 0],
                "states_legend": {"0": "silent", "1": "speaking", "2": "listening"},
                "rpy": [n.tolist(), n.tolist()],
            },
            "meta": {"generated_by": "l1_demo.home", "segments": []},
        }
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        return doc


def np_zero3():
    import numpy as np
    return np.zeros(3)


# --------------------------------------------------------------------------- #
# 主控
# --------------------------------------------------------------------------- #
class L1Demo:
    def __init__(self, mock: bool, workdir: Path, checkpoint: Path,
                 whisper_model: str, no_play: bool, skip_dryrun: bool,
                 no_server: bool = False):
        self.mock = mock
        self.workdir = workdir
        self.workdir.mkdir(parents=True, exist_ok=True)
        # sudo(root) 运行时会生成 root 属主文件; 放宽权限避免后续非 root 运行冲突
        for p in self.workdir.rglob("*"):
            if p.is_file():
                try:
                    p.chmod(0o666)
                except OSError:
                    pass
        self.no_play = no_play
        self.skip_dryrun = skip_dryrun
        self.no_server = no_server      # 双终端模式: 只连接不管理服务
        self.client = NeckClient()
        self.proc: subprocess.Popen | None = None
        self.gen = TrajGen(checkpoint, workdir, load_limits(MOTOR_CONFIG), mock)
        self.whisper_model = str(whisper_model)

    # ---------------- 服务进程管理 ----------------
    def start_server(self) -> bool:
        if self.client.alive():
            print("[demo] 已连接 Motor 服务")
            return True
        if self.no_server:
            print("[demo] 未检测到 Motor 服务; 请先在另一终端启动:\n"
                  f"       sudo {MOTOR_BIN} --server")
            return False
        # 残留 socket(无响应服务): 给出清理指引, 避免 bind 失败后困惑
        if Path(SOCK_PATH).exists():
            print("[demo] 检测到残留 socket(旧服务无响应), 请先清理后重跑:\n"
                  "       sudo pkill -f master_stack_test; sudo rm -f /tmp/neck_ctl.sock")
        # 清理残留 socket(无监听进程的陈旧文件; 删除失败则由 sudo 服务端处理)
        try:
            Path(SOCK_PATH).unlink()
        except OSError:
            pass
        cmd = ["sudo", str(MOTOR_BIN), "--server"]
        if self.mock:
            cmd.append("--no-ethercat")
        print("[demo] 启动 Motor 服务: " + " ".join(cmd))
        log = open(self.workdir / "motor_server.log", "a", encoding="utf-8")
        try:
            # cwd 必须是 Project-Motor: 执行层按相对路径找 neck_trajectory_config.txt
            self.proc = subprocess.Popen(cmd, cwd=str(MOTOR_BIN.parent.parent),
                                         stdout=log, stderr=log)
        except Exception as e:
            print(f"[demo] 启动失败: {e}(需要 sudo 权限)")
            return False
        for _ in range(50):
            if self.client.alive():
                print("[demo] Motor 服务就绪")
                return True
            if self.proc.poll() is not None:
                print(f"[demo] Motor 服务退出码 {self.proc.returncode}, "
                      f"见 {self.workdir}/motor_server.log")
                return False
            time.sleep(0.2)
        print("[demo] Motor 服务启动超时")
        return False

    def stop_server(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    # ---------------- 实机启动姿态检查与自动回中 ----------------
    @staticmethod
    def _parse_motor_actual(resp: str) -> list[float] | None:
        """解析 NeckStaticCheck 输出 CSV 的最后一行, 返回 [m1, m2, m3] actual。"""
        for line in reversed(resp.splitlines()):
            parts = line.split(",")
            if len(parts) >= 9 and parts[0].strip().isdigit():
                try:
                    return [float(parts[2].strip()), float(parts[5].strip()),
                            float(parts[8].strip())]   # m1/m2/m3 actual
                except ValueError:
                    continue
        return None

    @staticmethod
    def _read_neutral_motor_deg() -> list[float] | None:
        """读 Project-Motor/neck_config.txt 的 m10/m20/m30(中位电机角)。"""
        cfg = MOTOR_BIN.parent.parent / "neck_config.txt"
        vals: dict[str, float] = {}
        try:
            for line in cfg.read_text(encoding="utf-8").splitlines():
                line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                k, v = [s.strip() for s in line.split("=", 1)]
                if k in ("m10", "m20", "m30"):
                    vals[k] = float(v)
        except Exception:
            return None
        if len(vals) == 3:
            return [vals["m10"], vals["m20"], vals["m30"]]
        return None

    def ensure_home(self) -> None:
        """实机模式: 检查当前电机角是否在中位, 偏离则用标定档运动自动回中。
        任何异常不阻塞启动(打印指引后继续, 由执行层起点检查兜底)。"""
        if self.mock:
            return
        try:
            self._ensure_home_impl()
        except Exception as e:
            print(f"[demo] 回中检查失败: {e}")
            print("[demo] 提示: 若 Motor 服务无响应, 请先手动清理后重跑:\n"
                  "       sudo pkill -f master_stack_test; sudo rm -f /tmp/neck_ctl.sock")

    def _home_by_calib(self, desc: str) -> bool:
        """电机角步进回中(标定档): 读当前角 → 逐轴 NeckCalibMove 到中位。
        返回是否到位。任何异常返回 False(不抛)。"""
        home = self._read_neutral_motor_deg()
        if home is None:
            print("[demo] 警告: 无法读取中位电机角, 跳过回中")
            return False
        try:
            resp = self.client.send("NeckStaticCheck 2 0", timeout=30.0)
            act = self._parse_motor_actual(resp)
            if act is None:
                print("[demo] 警告: 无法读取当前电机角, 跳过回中")
                return False
            dev = [abs(act[i] - home[i]) for i in range(3)]
            if max(dev) <= 2.0:
                print(f"[demo] {desc}: 已在中位(偏差 {max(dev):.1f}°)")
                return True
            print(f"[demo] {desc}: m1={act[0]:.1f} m2={act[1]:.1f} m3={act[2]:.1f}° "
                  f"(中位 {home[0]:.0f}/{home[1]:.0f}/{home[2]:.0f}), 自动回中...")
            for axis in range(3):
                for _ in range(30):
                    if abs(act[axis] - home[axis]) <= 2.0:
                        break
                    delta = home[axis] - act[axis]
                    delta = max(-4.0, min(4.0, delta))   # 单步 ≤4°(标定保护 ≤5°)
                    r = self.client.send(f"NeckCalibMove 0 {axis + 1} {delta:.3f}", timeout=30.0)
                    if "拒绝" in r:
                        print(f"[demo] 回中轴{axis+1}被拒: {r[-200:]}")
                        return False
                    # 等待标定运动完成(CALIBRATION → READY)
                    for _ in range(50):
                        st = self.client.status()
                        if st and st["state"] == "READY":
                            break
                        time.sleep(0.2)
                    resp = self.client.send("NeckStaticCheck 1 0", timeout=15.0)
                    act = self._parse_motor_actual(resp)
                    if act is None:
                        return False
            print(f"[demo] 回中完成: m1={act[0]:.1f} m2={act[1]:.1f} m3={act[2]:.1f}°")
            return True
        except Exception as e:
            print(f"[demo] 回中失败: {e}")
            return False

    def _ensure_home_impl(self) -> None:
        self._home_by_calib("姿态检查")

    # ---------------- 语音 ----------------
    def play_bot_audio(self, wav: Path):
        if self.no_play:
            print(f"[demo] (no-play) 应播放: {wav}")
            return
        try:
            subprocess.run(["aplay", str(wav)], check=True)
        except Exception as e:
            print(f"[demo] 播放失败({e}); 手动: aplay {wav}")

    # ---------------- 执行与同步 ----------------
    def _dryrun_check(self, path: Path):
        if self.skip_dryrun:
            return
        try:
            ratio, resp = self.client.dryrun(path)
            if ratio is None:
                print("[demo] 警告: dry-run 未返回比例, 输出:\n" + resp[-400:])
            elif ratio > 1.3:
                print(f"[demo] 警告: 重定时比例 {ratio:.2f} > 1.3, 动作可能放慢; "
                      "建议调小 KP_INTERVAL 或增大 KP_MARGIN")
        except Exception as e:
            print(f"[demo] dry-run 失败: {e}")

    def _exec(self, path: Path, audit: str) -> bool:
        """下发轨迹并等待完成(mock: 同步阻塞; 实机: 轮询)。返回成功。"""
        self._dryrun_check(path)
        ok, resp = self.client.run(path, audit, self.mock)
        if not ok:
            print("[demo] 执行被拒绝:\n" + resp[-500:])
            return False
        if self.mock:
            # mock 命令阻塞到完成
            print("[demo] mock 执行完成")
            return True
        # 实机: 轮询到 READY
        while True:
            st = self.client.status()
            if st is None:
                time.sleep(0.1)
                continue
            if st["state"] == "READY":
                return True
            if st["state"] in ("FAULT", "ESTOP"):
                print(f"[demo] 执行器异常: {st['state']}")
                return False
            time.sleep(0.1)

    def _exec_speak(self, doc: dict, wav: Path, path: Path, audit: str) -> bool:
        """下发说话轨迹, 语音触发线程与执行并行。返回成功。"""
        self._dryrun_check(path)
        trigger = threading.Thread(target=self._play_at_speaking,
                                   args=(doc, wav), daemon=True)
        trigger.start()
        ok, resp = self.client.run(path, audit, self.mock)
        if not ok:
            print("[demo] 执行被拒绝:\n" + resp[-500:])
            return False
        if self.mock:
            print("[demo] mock 执行完成")
            return True
        while True:
            st = self.client.status()
            if st is None:
                time.sleep(0.1)
                continue
            if st["state"] == "READY":
                return True
            if st["state"] in ("FAULT", "ESTOP"):
                print(f"[demo] 执行器异常: {st['state']}")
                return False
            time.sleep(0.1)

    def _play_at_speaking(self, doc: dict, wav: Path):
        """在 speaking 段实际起点播放语音。mock: 定时线程; 实机: 轮询段边界。"""
        if self.no_play:
            print(f"[demo] (no-play) 应在 speaking 段起点播放: {wav}")
            return
        states = doc["trajectory"]["states"]
        # 首个 speaking 段在轨迹中的段索引
        first_speak = next((k for k, s in enumerate(states) if s == 1), None)
        if first_speak is None:
            print("[demo] 轨迹无 speaking 段, 直接播放")
            self.play_bot_audio(wav)
            return

        if self.mock:
            # mock 命令阻塞, 用定时线程(名义时间 + 过渡段)
            seg_start = (first_speak + 1) * KP_INTERVAL   # 含过渡段(1 段)

            def _t():
                time.sleep(max(0.0, seg_start - 0.2))
                self.play_bot_audio(wav)
            threading.Thread(target=_t, daemon=True).start()
            return

        # 实机: 轮询 Status 的段边界(含过渡段)
        played = False
        t0 = time.time()
        while True:
            st = self.client.status()
            if st is not None:
                bounds = st["seg_boundaries"]
                if bounds and first_speak <= len(bounds):
                    seg_start = bounds[first_speak - 1] if first_speak >= 1 else 0.0
                    if st["elapsed"] >= seg_start - 0.05 and not played:
                        print(f"[demo] speaking 段开始({seg_start:.1f}s), 播放语音")
                        self.play_bot_audio(wav)
                        played = True
                        return
                if st["state"] == "READY":
                    if not played:
                        print("[demo] 执行已完成但未触发(异常), 补播语音")
                        self.play_bot_audio(wav)
                    return
            if time.time() - t0 > 600:
                print("[demo] 等待 speaking 段超时")
                return
            time.sleep(0.1)

    # ---------------- 一轮交互 ----------------
    def handle_input(self, text: str):
        try:
            self._handle_input_impl(text)
        except Exception as e:
            import traceback
            print(f"[demo] 本轮失败: {e}")
            traceback.print_exc()
            print("[demo] 已回到输入循环, 可重试")

    def _handle_input_impl(self, text: str):
        round_dir = self.workdir
        bot_text = random.choice(REPLIES)
        print(f"[bot] 回复: {bot_text}")

        # 1. 并行 TTS(用户不播放, 仅模型输入; 机器人回复语音稍后播放)
        res: dict = {}
        errs: list[str] = []

        def _tts(name: str, txt: str, voice: str):
            try:
                res[name] = tts_align(txt, name, round_dir, voice=voice,
                                      whisper_model=self.whisper_model,
                                      device="cpu")   # 本机缺 libcublas, CUDA 加载可能挂起; CPU int8 对齐 ~1s
            except Exception as e:
                errs.append(f"{name}: {e}")
        t1 = threading.Thread(target=_tts, args=("user", text, DEFAULT_VOICE))
        t2 = threading.Thread(target=_tts, args=("bot", bot_text, BOT_VOICE))
        t1.start(); t2.start(); t1.join(); t2.join()
        if errs:
            for e in errs:
                print(f"[demo] TTS 失败({e})")
            print("[demo] 本轮取消(模型需要语音输入); 请检查网络后重试")
            return
        u, b = res["user"], res["bot"]

        # 2. 倾听轨迹: 生成 + 下发执行(尽快开始动)
        listen_doc = self.gen.listen(text, "audio/user.wav", u["words"],
                                     round_dir / "trajectory_listen.json")
        print("[demo] 下发倾听轨迹")
        if not self._exec(round_dir / "trajectory_listen.json",
                          AUDIT_DIR + "/listen"):
            return

        # 3. 说话轨迹(模型常驻, 生成很快; 语音在 speaking 段起点播放)
        speak_doc = self.gen.speak(bot_text, "audio/bot.wav", b["words"],
                                   round_dir / "trajectory_speak.json")
        print("[demo] 下发说话轨迹(语音随 speaking 段触发)")
        if not self._exec_speak(speak_doc, round_dir / "audio" / "bot.wav",
                                round_dir / "trajectory_speak.json",
                                AUDIT_DIR + "/speak"):
            return

        # 4. 展示回复(文字)
        print(f"[bot] {bot_text}")
        print("[demo] 本轮完成, 等待下一轮输入...")

    # ---------------- 空闲超时回中 ----------------
    def idle_timeout(self):
        st = self.client.status()
        if st is None:
            return
        if st["state"] in ("EXECUTING",):
            print(f"[demo] {IDLE_TIMEOUT:.0f}s 无按键: 安全停止当前动作并回中")
            try:
                self.client.send("NeckTrajStop", timeout=10.0)
            except Exception:
                pass
            for _ in range(50):          # 等安全停止完成 → READY
                st = self.client.status()
                if st and st["state"] == "READY":
                    break
                time.sleep(0.2)
        self._home_by_calib("空闲回中")

    # ---------------- 收工 ----------------
    def cleanup(self):
        print("\n[demo] 收工: NeckDisable(安全制动)")
        try:
            r = self.client.disable()
            lines = [l for l in r.strip().splitlines() if l.strip() and "###END###" not in l]
            if lines:
                print(lines[-1])
        except Exception:
            pass
        if self.no_server:
            print("[demo] Motor 服务保持运行, 可在服务终端 Ctrl+C 关闭")
        else:
            self.stop_server()
        print("[demo] 请按流程: 关闭三把锁 + 物理断电")

    # ---------------- 主循环 ----------------
    def run(self):
        print("=" * 62)
        print("L1 轮转式对话演示 (英文; 输入 q/quit 退出; Ctrl+C 收工)")
        print(f"> 直接输入文字, 按回车提交; {IDLE_TIMEOUT:.0f}s 无按键自动回中")
        print(f"mock={'是' if self.mock else '否(实机)'}")
        print("=" * 62)
        if sys.stdin.isatty():
            self._run_tty()
        else:
            self._run_line()

    def _run_tty(self):
        """终端逐字符模式: 每个按键立即响应, IDLE_TIMEOUT 无按键才判定空闲。"""
        import os as _os
        import termios
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] &= ~termios.ICANON   # 关闭行缓冲(逐字符可读)
        new[3] |= termios.ECHO      # 保留回显
        termios.tcsetattr(fd, termios.TCSANOW, new)
        idle_notified = False
        buf: list[bytes] = []
        try:
            while True:
                try:
                    r, _, _ = select.select([fd], [], [], IDLE_TIMEOUT)
                except KeyboardInterrupt:
                    self.cleanup()
                    return
                if r:
                    idle_notified = False
                    try:
                        ch = _os.read(fd, 1)
                    except OSError:
                        self.cleanup()
                        return
                    if not ch:
                        self.cleanup()
                        return
                    if ch in (b"\x03",):              # Ctrl+C
                        raise KeyboardInterrupt
                    if ch in (b"\r", b"\n"):          # 回车提交
                        line = b"".join(buf).decode("utf-8", errors="ignore").strip()
                        buf = []
                        print()
                        if not line:
                            print("[demo] 空输入忽略; 请输入文字后回车")
                            continue
                        if line.lower() in ("q", "quit", "exit"):
                            self.cleanup()
                            return
                        self.handle_input(line)
                    elif ch in (b"\x7f", b"\x08"):    # 退格
                        if buf:
                            buf.pop()
                            _os.write(fd, b"\b \b")
                    elif ch in (b"\x04",):              # Ctrl+D
                        self.cleanup()
                        return
                    elif ch >= b" ":                     # 可见字符
                        buf.append(ch)
                else:
                    # IDLE_TIMEOUT 无按键: 空闲回中(打字期间按键重置计时)
                    if not idle_notified:
                        print(f"[demo] {IDLE_TIMEOUT:.0f}s 无按键, 若在打字请继续(回车提交); 执行器忙则自动回中")
                        idle_notified = True
                    try:
                        self.idle_timeout()
                    except KeyboardInterrupt:
                        self.cleanup()
                        return
        finally:
            termios.tcsetattr(fd, termios.TCSANOW, old)

    def _run_line(self):
        """管道/非 tty 模式: 按行读取(原逻辑)。"""
        idle_notified = False
        while True:
            try:
                r, _, _ = select.select([sys.stdin], [], [], IDLE_TIMEOUT)
            except KeyboardInterrupt:
                self.cleanup()
                return
            if r:
                idle_notified = False
                line = sys.stdin.readline()
                if not line:
                    self.cleanup()
                    return
                text = line.strip()
                if not text:
                    print("[demo] 空输入忽略; 请输入文字后回车")
                    continue
                if text.lower() in ("q", "quit", "exit"):
                    self.cleanup()
                    return
                self.handle_input(text)
            else:
                if not idle_notified:
                    print("[demo] 等待输入中(输入文字后回车; 3s 无输入自动回中)")
                    idle_notified = True
                try:
                    self.idle_timeout()
                except KeyboardInterrupt:
                    self.cleanup()
                    return


def main() -> None:
    ap = argparse.ArgumentParser(description="L1 一体化演示主控")
    ap.add_argument("--mock", action="store_true", help="服务端 mock 模式(无硬件)")
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--whisper-model", default=str(DEFAULT_WHISPER))
    ap.add_argument("--no-play", action="store_true", help="不播放语音(离线调试)")
    ap.add_argument("--skip-dryrun", action="store_true", help="下发前跳过 dry-run 预检")
    ap.add_argument("--no-server", action="store_true",
                    help="双终端模式: 只连接已运行的 Motor 服务, 不管理其生命周期")
    args = ap.parse_args()

    demo = L1Demo(args.mock, Path(args.workdir), Path(args.checkpoint),
                  args.whisper_model, args.no_play, args.skip_dryrun,
                  no_server=args.no_server)
    if not demo.start_server():
        print("[demo] 服务未就绪, 退出")
        sys.exit(1)
    demo.ensure_home()      # 实机: 姿态检查 + 自动回中
    try:
        demo.run()
    except KeyboardInterrupt:
        demo.cleanup()
    finally:
        if not demo.no_server and demo.proc and demo.proc.poll() is None:
            demo.stop_server()


if __name__ == "__main__":
    main()
