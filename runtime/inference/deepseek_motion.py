"""DeepSeek sparse-plan Motion backend."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

from ..contracts import MotionOutput, MotionRequest, RobotSpeech
from ..logging_utils import log
from .base import MotionBackend
from .motion_compiler import (
    MotionPlanError,
    RejectedMotionAction,
    SparseMotionAction,
    compile_motion_plan,
    parse_motion_plan_with_rejections,
)
from .speech_alignment import SpeechAlignment, align_speech

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT_SEC = 15.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024
PLAN_FILENAME = "deepseek_motion_plan.json"
RELATIVE_TRAJECTORY_FILENAME = "deepseek_relative_trajectory.json"

MOTION_SYSTEM_PROMPT = """你是仿生机器人颈部动作规划器。根据机器人即将说出的文本和语音总时长，生成少量、自然、克制的颈部 gesture。

输入中的 robot_text 决定动作语义，segments 给出该文本在真实 TTS 音频中的时间位置。动作时间应优先与相关语义 segment 对齐，不要只根据总 duration 猜测语句发生时间。segments 缺失时才按 robot_text 和 duration_sec 规划。

你必须只输出一个合法 JSON 对象，不得输出 Markdown、代码围栏、解释或其他文字。JSON schema 严格为：
{"actions":[{"start":0.35,"end":1.25,"roll":0.0,"pitch":2.5,"yaw":0.0}]}

规则：
1. start/end 单位为秒，必须在 [0, duration_sec] 内，且 start < end；每个动作至少持续 0.7 秒，通常使用 0.8 到 1.2 秒。
2. roll/pitch/yaw 单位为 degree，表示相对该动作开始时 neutral baseline 的临时 gesture amplitude。每个动作都必须在 end 时自然回到 baseline，动作之间不累加姿态。
3. 真机动作保持保守：roll 通常不超过 1.5 度，pitch 通常不超过 2.5 度，yaw 通常不超过 2.5 度；任何轴绝对值不得超过 5 度。
4. 动作不得重叠；两个动作之间至少保留约 0.2 秒 neutral HOLD。时间不足时减少动作数量，短句通常只生成一个动作。
5. 大部分时间保持 neutral，动作必须稀疏，不追求持续运动，禁止高频震颤，避免三个轴同时明显运动。
6. 可根据强调、确认、疑问、转折安排动作：确认可轻微点头，疑问可轻微侧倾或转头，强调可安排一次克制 gesture。
7. 必须至少给出一个合法动作；不要用永久偏移表达姿态。
"""


class DeepSeekMotionError(RuntimeError):
    """DeepSeek did not produce a usable sparse motion plan."""


class DeepSeekMotionBackend(MotionBackend):
    """Request a sparse degree plan and compile it to fixed 30 fps offsets."""

    failure_mode = "audio_only"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        output_dir: str | Path = "generated",
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        aligner: Callable[..., SpeechAlignment] = align_speech,
    ) -> None:
        if not model.strip():
            raise ValueError("DeepSeek Motion model must not be empty")
        if not base_url.strip():
            raise ValueError("DeepSeek Motion base_url must not be empty")
        if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
            raise ValueError("DeepSeek Motion timeout_sec must be positive")
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("DeepSeek Motion temperature must be between 0 and 2")
        if max_output_tokens < 1:
            raise ValueError("DeepSeek Motion max_output_tokens must be positive")
        if client is not None and client_factory is not None:
            raise ValueError("provide either client or client_factory, not both")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.output_dir = Path(output_dir)
        self.plan_path = self.output_dir / PLAN_FILENAME
        self.relative_trajectory_path = (
            self.output_dir / RELATIVE_TRAJECTORY_FILENAME
        )
        if not callable(aligner):
            raise TypeError("aligner must be callable")
        self._aligner = aligner

        if client is not None:
            self._client = client
        else:
            api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
            if not api_key:
                raise DeepSeekMotionError(
                    "DEEPSEEK_API_KEY is required for motion.backend=deepseek"
                )
            if client_factory is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise DeepSeekMotionError(
                        "openai package is required for motion.backend=deepseek"
                    ) from exc
                client_factory = OpenAI
            self._client = client_factory(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout_sec,
                max_retries=0,
            )

    @staticmethod
    def _response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(response, dict):
            output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output = response.get("output", []) if isinstance(response, dict) else getattr(response, "output", [])
        parts: list[str] = []
        for item in output or []:
            content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
            for part in content or []:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts).strip()

    def _request_plan(
        self,
        robot_text: str,
        duration_sec: float,
        segments: list[dict[str, str | float]] | None,
    ) -> str:
        payload: dict[str, Any] = {
            "robot_text": robot_text,
            "duration_sec": duration_sec,
        }
        if segments is not None:
            payload["segments"] = segments
        user_input = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        try:
            response = self._client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": MOTION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ],
                max_output_tokens=self.max_output_tokens,
                temperature=self.temperature,
                reasoning={"effort": "none"},
                timeout=self.timeout_sec,
            )
        except Exception as exc:
            raise DeepSeekMotionError(
                f"DeepSeek Motion API request failed ({type(exc).__name__})"
            ) from exc
        text = self._response_text(response)
        if not text:
            raise DeepSeekMotionError("DeepSeek Motion returned empty output")
        return text

    def _align_speech(
        self, robot_text: str, speech: RobotSpeech
    ) -> tuple[list[dict[str, str | float]] | None, dict[str, Any]]:
        started = time.perf_counter()
        try:
            result = self._aligner(
                robot_text,
                speech.pcm_s16le,
                speech.duration_sec,
                sample_rate=16_000,
            )
            if not isinstance(result, SpeechAlignment):
                raise TypeError("aligner must return SpeechAlignment")
            segments = result.segments_as_dicts()
            if result.fallback_reason:
                log(f"[MOTION][ALIGN] fallback: {result.fallback_reason}")
            log(f"[MOTION][ALIGN] method={result.method}")
            log(
                "[MOTION][ALIGN] segments="
                + json.dumps(
                    segments,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            log(f"[MOTION][ALIGN] latency={result.latency_sec:.3f}s")
            return segments, result.metadata()
        except Exception as exc:
            latency = time.perf_counter() - started
            reason = f"{type(exc).__name__}: {exc}"
            log(f"[MOTION][ALIGN] fallback: {reason}")
            log(f"[MOTION][ALIGN] latency={latency:.3f}s")
            return None, {
                "method": "text_duration_fallback",
                "latency_sec": latency,
                "fallback_reason": reason,
            }

    def _write_plan_artifact(
        self,
        robot_text: str,
        duration_sec: float,
        raw_response: str | None,
        actions: tuple[SparseMotionAction, ...] | None,
        rejected_actions: tuple[RejectedMotionAction, ...] | None,
        segments: list[dict[str, str | float]] | None,
        alignment: dict[str, Any],
        error: str | None,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        document = {
            "robot_text": robot_text,
            "duration_sec": duration_sec,
            "raw_response": raw_response,
            "actions": (
                [action.as_dict() for action in actions]
                if actions is not None
                else None
            ),
            "rejected_actions": (
                [action.as_dict() for action in rejected_actions]
                if rejected_actions is not None
                else None
            ),
            "segments": segments,
            "alignment": alignment,
            "error": error,
            "created_unix_sec": time.time(),
        }
        temporary = self.plan_path.with_name(self.plan_path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.plan_path)

    def _write_relative_trajectory(
        self,
        robot_text: str,
        duration_sec: float,
        output: MotionOutput,
    ) -> None:
        document = {
            "robot_text": robot_text,
            "duration_sec": duration_sec,
            "fps": float(output.fps),
            "unit": output.unit,
            "order": list(output.order),
            "representation": output.representation,
            "trajectory": [list(frame) for frame in output.rpy_offset],
            "created_unix_sec": time.time(),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.relative_trajectory_path.with_name(
            self.relative_trajectory_path.name + ".tmp"
        )
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.relative_trajectory_path)

    @staticmethod
    def _log_range(output: MotionOutput) -> None:
        frames = tuple(output.rpy_offset)
        ranges: list[str] = []
        for axis, name in enumerate(("roll", "pitch", "yaw")):
            values = [math.degrees(float(frame[axis])) for frame in frames]
            ranges.append(f"{name}=[{min(values):.3f},{max(values):.3f}]deg")
        log(f"[MOTION] frames={len(frames)}")
        log("[MOTION] range " + " ".join(ranges))

    def _infer(self, request: MotionRequest) -> MotionOutput:
        if not isinstance(request, MotionRequest):
            raise TypeError("DeepSeekMotionBackend requires MotionRequest")
        robot_text = request.current_turn.robot_text
        speech = request.current_turn.robot_speech
        if not isinstance(robot_text, str) or not robot_text.strip():
            raise DeepSeekMotionError("motion request has no robot_text")
        if not isinstance(speech, RobotSpeech):
            raise DeepSeekMotionError("motion request has no RobotSpeech")
        duration_sec = float(speech.duration_sec)
        if not math.isfinite(duration_sec) or duration_sec <= 0.0:
            raise DeepSeekMotionError("robot speech duration must be positive")

        started = time.perf_counter()
        raw_response: str | None = None
        parsed_actions: tuple[SparseMotionAction, ...] | None = None
        rejected_actions: tuple[RejectedMotionAction, ...] | None = None
        segments, alignment = self._align_speech(robot_text.strip(), speech)
        log("[MOTION] request_start")
        try:
            raw_response = self._request_plan(
                robot_text.strip(), duration_sec, segments
            )
            try:
                document = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise DeepSeekMotionError(
                    "DeepSeek Motion output is not valid JSON"
                ) from exc
            parsed_actions, rejected_actions = parse_motion_plan_with_rejections(
                document, duration_sec
            )
            for rejected in rejected_actions:
                log(
                    f"[MOTION] dropped action {rejected.index}: "
                    f"{rejected.reason}"
                )
            if not parsed_actions:
                raise DeepSeekMotionError(
                    "DeepSeek Motion plan contains no valid actions"
                )
            output = compile_motion_plan(parsed_actions, duration_sec)
            self._write_plan_artifact(
                robot_text.strip(),
                duration_sec,
                raw_response,
                parsed_actions,
                rejected_actions,
                segments,
                alignment,
                None,
            )
            self._write_relative_trajectory(
                robot_text.strip(), duration_sec, output
            )
            log(
                "[MOTION] plan: "
                + json.dumps(
                    {"actions": [action.as_dict() for action in parsed_actions]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            self._log_range(output)
            return output
        except (DeepSeekMotionError, MotionPlanError) as exc:
            self._write_plan_artifact(
                robot_text.strip(),
                duration_sec,
                raw_response,
                parsed_actions,
                rejected_actions,
                segments,
                alignment,
                str(exc),
            )
            raise DeepSeekMotionError(str(exc)) from exc
        finally:
            log(f"[MOTION] latency={time.perf_counter() - started:.3f}s")
