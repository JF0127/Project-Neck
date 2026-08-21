#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""live_round.py: L1 轮转式人机对话主控(一轮 = 听 + 说 + 回中)。

流程:
    键盘输入用户文本
    → 预置回复表选机器人回复
    → tts_align: 合成用户语音(女声, 不播放, 仅模型输入) + 机器人语音(男声)
    → 组装 events.json(预备→倾听→思考→说话→回中, 5 段)
    → 复用 mvp.py 生成 V1 轨迹(energy_match 低能量, 保证动作/语音同步)
    → 同步预检(meta.max_frame_rate_deg_per_s, 超标自动换更安静候选)
    → 提示在 Motor 终端执行 NeckTrajRun, 回车后播放机器人语音

用法:
    python tools/live/live_round.py
    python tools/live/live_round.py --reply "自定义回复" --no-play

手动执行流程(实机):
    1. 脚本提示后, 在 Motor 终端(需要 root):
         cd Project-Motor && sudo ./build/master_stack_test
         > NeckTrajDryRun <workdir>/trajectory_1.json     # 预检
         > NeckTrajRun    <workdir>/trajectory_1.json 0 <audit_dir>
    2. 回到本脚本按回车 → 自动播放机器人语音
       (说话段前有 ~2.5s + 倾听段长的缓冲, 手动操作延迟落在缓冲内)
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

# 项目根(与 mvp.py 相同约定)
NET_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(NET_ROOT))

from tools.live.tts_align import tts_align, DEFAULT_VOICE, BOT_VOICE  # noqa: E402

DEFAULT_WORKDIR = NET_ROOT.parent / "neck_l1"  # 生成物目录: Project-Neck/neck_l1 (已被 .gitignore 忽略)
DEFAULT_CHECKPOINT = NET_ROOT / "outputs/neck_motion_v3/checkpoints/best.pt"
DEFAULT_WHISPER = NET_ROOT.parent / "model"  # 本地 faster-whisper base (Project-Neck/model)

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

# 同步预检阈值: 峰值帧速(°/s)超过该值会触发执行层重定时拉长动作(脱同步)
MAX_FRAME_RATE_OK = 15.0


def load_words(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["words"]


def make_fragment(role: str, text: str, audio_path: str,
                  words: list[dict]) -> dict:
    """与训练样本 / mvp_stage1 events 同构的 fragment。"""
    last_end = max((w["end_time"] for w in words), default=0.0)
    n_frames = max(2, round((last_end + 0.5) * 30.0))
    return {
        "role": role,
        "current_utterance": {
            "text": text,
            "audio_path": audio_path,
            "word_timestamps": words,
        },
        "fps": 30.0,
        "num_frames": n_frames,
    }


def build_events(user_text: str, bot_text: str,
                 words_user: list[dict], words_bot: list[dict]) -> dict:
    """5 段一轮: 预备 → 倾听 → 思考 → 说话 → 回中(缓冲见蓝图 §4)。"""
    return {
        "robot_actual_initial": [0.0, 0.0, 0.0],
        "robot_neutral_pose": [0.0, 0.0, 0.0],
        "events": [
            {"state": "silent", "duration": 1.5},
            {"state": "listening", "fragment": make_fragment(
                "listener", user_text, "audio/user.wav", words_user)},
            {"state": "silent", "duration": 1.0},
            {"state": "speaking", "fragment": make_fragment(
                "speaker", bot_text, "audio/bot.wav", words_bot)},
            {"state": "silent", "duration": 2.0},
        ],
    }


def run_mvp(workdir: Path, events_path: Path, traj_path: Path,
            checkpoint: Path, expected_energy: float) -> tuple[bool, float]:
    """子进程跑 mvp.py 生成 V1 轨迹。返回 (成功, max_frame_rate_deg_per_s)。"""
    cmd = [
        sys.executable, "models/neck_motion/mvp.py",
        "--checkpoint", str(checkpoint),
        "--events", str(events_path),
        "--data-root", str(workdir),
        "--strategy", "energy_match",
        "--expected-energy", str(expected_energy),
        "--output-dir", str(workdir),
        "--export-json", str(traj_path),
    ]
    print(f"[live] 运行 mvp.py (期望能量 {expected_energy}°/s) ...")
    try:
        r = subprocess.run(cmd, cwd=str(NET_ROOT), timeout=180,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print("[live] 错误: mvp.py 超时")
        return False, 0.0
    if r.returncode != 0:
        print("[live] mvp.py 失败:\n" + r.stderr[-3000:])
        return False, 0.0
    try:
        doc = json.loads(traj_path.read_text(encoding="utf-8"))
        mfr = float(doc["meta"]["max_frame_rate_deg_per_s"])
        n = doc["trajectory"]["n_frames"]
    except Exception:
        print("[live] 警告: 无法解析轨迹 JSON, 请检查 mvp.py 输出")
        return False, 0.0
    print(f"[live] 轨迹 {n} 帧, 峰值帧速 {mfr:.1f}°/s"
          f"({'OK' if mfr <= MAX_FRAME_RATE_OK else '偏高'})")
    return True, mfr


def generate_round(workdir: Path, user_text: str, bot_text: str,
                   checkpoint: Path, whisper_model: str, round_no: int,
                   no_play: bool, expected_energy: float) -> Path | None:
    workdir.mkdir(parents=True, exist_ok=True)

    # 1) 双方语音 + 词戳
    r_user = tts_align(user_text, "user", workdir, voice=DEFAULT_VOICE,
                       whisper_model=whisper_model)
    r_bot = tts_align(bot_text, "bot", workdir, voice=BOT_VOICE,
                      whisper_model=whisper_model)

    # 2) events.json
    events = build_events(user_text, bot_text,
                          r_user["words"], r_bot["words"])
    events_path = workdir / f"events_{round_no}.json"
    events_path.write_text(json.dumps(events, ensure_ascii=False, indent=1),
                           encoding="utf-8")
    print(f"[live] events 已写: {events_path}")

    # 3) 生成轨迹(低能量优先; 超标自动降一档重试)
    traj_path = workdir / f"trajectory_{round_no}.json"
    ok, mfr = run_mvp(workdir, events_path, traj_path, checkpoint, expected_energy)
    if ok and mfr > MAX_FRAME_RATE_OK and expected_energy > 0.5:
        print("[live] 峰值帧速偏高, 用更低期望能量(0.5°/s)重试 ...")
        ok, mfr = run_mvp(workdir, events_path, traj_path, checkpoint, 0.5)
    if not ok:
        return None
    if mfr > MAX_FRAME_RATE_OK:
        print("[live] 警告: 峰值帧速仍偏高, 动作可能被执行层拉长(脱同步); "
              "建议换更短的回复或调低能量")

    # 4) 手动执行指引 + 播放
    print("\n" + "=" * 62)
    print("下一步(在 Motor 终端执行, 需要 root):")
    print(f"  NeckTrajDryRun {traj_path}")
    print(f"  NeckTrajRun    {traj_path} 0 <audit_dir>")
    print("=" * 62)
    if not no_play:
        input("动作启动后回到这里按回车 → 播放机器人语音 ...")
        bot_wav = workdir / "audio/bot.wav"
        try:
            subprocess.run(["aplay", str(bot_wav)], check=True)
            print(f"[live] 已播放: {bot_wav}")
        except Exception as e:
            print(f"[live] 播放失败({e}); 可手动: aplay {bot_wav}")
    return traj_path


def main() -> None:
    ap = argparse.ArgumentParser(description="L1 轮转式对话一轮主控")
    ap.add_argument("--workdir", default=str(DEFAULT_WORKDIR),
                    help="工作目录(默认 Project-Neck/neck_l1)")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--whisper-model", default=str(DEFAULT_WHISPER),
                    help="faster-whisper 模型路径或尺寸")
    ap.add_argument("--reply", default=None, help="指定机器人回复(默认随机)")
    ap.add_argument("--expected-energy", type=float, default=2.0)
    ap.add_argument("--no-play", action="store_true", help="不播放语音(离线测试用)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    print("[live] L1 轮转式对话 (输入 q/quit 退出)")
    round_no = 0
    while True:
        user_text = input("\n你: ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("q", "quit", "exit"):
            print("[live] 退出")
            return
        round_no += 1
        bot_text = args.reply or random.choice(REPLIES)
        print(f"[live] 机器人回复: {bot_text}")
        traj = generate_round(workdir, user_text, bot_text,
                              Path(args.checkpoint), args.whisper_model,
                              round_no, args.no_play, args.expected_energy)
        if traj is None:
            print("[live] 本轮失败, 请检查上方错误")
        else:
            print(f"[live] 第 {round_no} 轮轨迹就绪: {traj}")


if __name__ == "__main__":
    main()
