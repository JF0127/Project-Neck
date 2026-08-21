#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tts_align.py: 文本 → 16kHz wav + 词级时间戳(L1 对话演示用)。

管线:
    edge-tts 合成语音(mp3) → soundfile 解码 + torchaudio 重采样到 16k mono wav
    → faster-whisper 对齐(词级时间戳) → 输出 words json

用法:
    python tools/live/tts_align.py --text "Hello, nice to meet you." --name user
    python tools/live/tts_align.py --voice en-US-GuyNeural --text "..." --name bot

输出(均在 --out-dir 下, 默认 Project-Neck/neck_l1):
    audio/<name>.wav         16kHz 单声道 wav(模型输入)
    audio/<name>.mp3         原始 TTS 合成(调试/试听用)
    words_<name>.json        [{"text","start_time","end_time"}, ...](与数据集 word_timestamps 同构)

依赖: edge-tts, faster-whisper, soundfile, torchaudio(项目 venv 已有)。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# 复用项目内音频加载(mono + 重采样 16k), 与 dataset.py 完全一致
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.neck_motion.audio_features import load_audio_waveform  # noqa: E402

DEFAULT_VOICE = "en-US-JennyNeural"  # 女声(模拟"用户")
BOT_VOICE = "en-US-GuyNeural"         # 男声(机器人)

_whisper_cache: dict = {}  # (model_size, device) -> model, 进程内只加载一次


def _get_whisper(model_size: str = "base", device: str | None = None):
    key = (model_size, device or "auto")
    if key in _whisper_cache:
        return _whisper_cache[key]
    from faster_whisper import WhisperModel
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    compute = "float16" if device == "cuda" else "int8"
    print(f"[tts_align] 加载 whisper ({device}, {compute}) ...")
    t0 = time.time()
    _whisper_cache[key] = WhisperModel(model_size, device=device, compute_type=compute)
    print(f"[tts_align] whisper 就绪 ({time.time()-t0:.1f}s)")
    return _whisper_cache[key]


def _transcribe_with_fallback(wav_path: str, text: str, language: str,
                              model_size: str, device: str | None):
    """转写。CUDA 库缺失(如 libcublas)时自动降级 CPU int8。"""
    for attempt, dev in enumerate([device, "cpu"]):
        if attempt == 1 and dev == device:
            break  # 已试过
        model = _get_whisper(model_size, dev)
        t0 = time.time()
        try:
            segments, info = model.transcribe(
                wav_path, language=language, word_timestamps=True, vad_filter=False)
            words = []
            for seg in segments:
                for w in (seg.words or []):
                    t = (w.word or "").strip()
                    if t:
                        words.append({"text": t, "start_time": w.start, "end_time": w.end})
            print(f"[tts_align] whisper 对齐 {len(words)} 词 ({time.time()-t0:.1f}s, "
                  f"device={dev})")
            return words
        except RuntimeError as e:
            if dev == "cpu":
                raise
            print(f"[tts_align] 警告: GPU 转写失败({e}), 降级 CPU int8 重试")
    return []


def _fallback_words(text: str, duration: float) -> list[dict]:
    """whisper 无词级输出时的兜底: 按字符均匀分配时间。"""
    chars = [c for c in text.strip() if not c.isspace()]
    n = max(len(chars), 1)
    step = duration / n
    return [{"text": c, "start_time": i * step, "end_time": (i + 1) * step}
            for i, c in enumerate(chars)]


def tts_align(text: str, name: str, out_dir: str | Path,
              voice: str = DEFAULT_VOICE, language: str = "en",
              whisper_model: str = "base", device: str | None = None,
              keep_mp3: bool = True) -> dict:
    """合成并对齐。返回 {"audio_path"(相对 out_dir), "words", "duration_sec"}。"""
    import edge_tts

    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    mp3_path = out_dir / "audio" / f"{name}.mp3"
    wav_path = out_dir / "audio" / f"{name}.wav"
    words_path = out_dir / f"words_{name}.json"

    # 1) TTS 合成(带硬超时: 网络不通时避免永久挂起)
    t0 = time.time()
    try:
        asyncio.run(asyncio.wait_for(
            edge_tts.Communicate(text, voice).save(str(mp3_path)), timeout=20.0))
    except (asyncio.TimeoutError, Exception) as e:
        raise RuntimeError(
            f"TTS 合成超时/失败({e}); 请检查网络(需要访问微软 edge-tts 服务)") from e
    print(f"[tts_align] TTS 合成 {name}.mp3 ({time.time()-t0:.1f}s, voice={voice})")

    # 2) 解码 → 16k mono wav(复用项目加载逻辑, 保证与训练一致)
    wav = load_audio_waveform(mp3_path, target_sr=16000)  # [1, T] float32
    wav = wav.squeeze(0)
    sf.write(wav_path, wav.numpy(), 16000, format="WAV", subtype="FLOAT")
    duration = wav.shape[0] / 16000.0

    # 3) whisper 词级对齐(GPU 失败自动降级 CPU)
    words = _transcribe_with_fallback(str(wav_path), text, language,
                                      whisper_model, device)

    if not words:
        print("[tts_align] 警告: whisper 未返回词级时间戳, 使用字符均匀兜底")
        words = _fallback_words(text, duration)

    # 清洗: 中文词间空格/标点粘连
    for w in words:
        w["text"] = w["text"].replace(" ", "")

    with open(words_path, "w", encoding="utf-8") as f:
        json.dump({"text": text, "voice": voice, "duration_sec": round(duration, 3),
                   "words": words}, f, ensure_ascii=False, indent=1)
    if not keep_mp3:
        mp3_path.unlink(missing_ok=True)

    rel = f"audio/{name}.wav"
    print(f"[tts_align] 完成: {wav_path} | 词戳 {words_path}")
    return {"audio_path": rel, "words": words, "duration_sec": duration}


def main() -> None:
    ap = argparse.ArgumentParser(description="文本 → 16k wav + 词级时间戳")
    ap.add_argument("--text", required=True, help="要合成的文本")
    ap.add_argument("--name", default="utterance", help="输出文件名前缀")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录(默认 Project-Neck/neck_l1)")
    ap.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts 音色")
    ap.add_argument("--language", default="en", help="whisper 语言(数据集为英文, 默认 en)")
    ap.add_argument("--whisper-model", default=str(Path(__file__).resolve().parents[3] / "model"),
                    help="whisper 模型路径(默认本地 Project-Neck/model; 传尺寸名会从 HF 下载)")
    ap.add_argument("--device", default=None, help="cuda/cpu, 默认自动")
    args = ap.parse_args()

    out_dir = args.out_dir or str(Path(__file__).resolve().parents[3] / "neck_l1")
    res = tts_align(args.text, args.name, out_dir, args.voice,
                    args.language, args.whisper_model, args.device)
    for w in res["words"]:
        print(f"  {w['start_time']:6.2f} ~ {w['end_time']:6.2f}  {w['text']}")


if __name__ == "__main__":
    main()
