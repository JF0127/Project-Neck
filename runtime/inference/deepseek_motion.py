"""DeepSeek high-level MotionPlan V2 backend."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

from ..contracts import FinalTrajectory, MotionOutput, MotionRequest, RobotSpeech
from ..logging_utils import log
from .base import MotionBackend
from .motion_plan import MotionPlan
from .motion_plan_validator import (
    MotionPlanValidationError,
    validate_motion_plan,
)
from .prosody import ProsodyAnalysis, extract_prosody
from .speech_alignment import SpeechAlignment, align_speech
from .trajectory_generator import TrajectoryGenerator

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_TIMEOUT_SEC = 15.0
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_OUTPUT_TOKENS = 1024
PROSODY_FILENAME = "prosody.json"
RAW_PLAN_FILENAME = "deepseek_motion_plan_raw.json"
MOTION_PLAN_FILENAME = "motion_plan.json"
RAW_RELATIVE_TRAJECTORY_FILENAME = "raw_relative_trajectory.json"
FINAL_TRAJECTORY_FILENAME = "final_trajectory.json"
_LEGACY_ARTIFACT_FILENAMES = (
    "deepseek_motion_plan.json",
    "deepseek_relative_trajectory.json",
)

MOTION_SYSTEM_PROMPT = """你是仿生机器人颈部 Speaking Motion Planner V2。你只规划高层动作，不生成逐帧轨迹，也不输出具体三轴曲线。

输入说明：
- robot_text 决定为什么动、是否需要动以及选择什么动作。
- duration_sec 是完整 TTS 语音时长。
- segments 给出文本片段的真实语音时间和可解释 prosody。prosody 中的 segment duration、前后停顿、相对能量、能量峰值时间、语速用于判断动作时机和强弱。
- 文本决定动作语义；prosody 优先帮助决定什么时候动以及动作幅度。不要把动作机械地放在句子中间。

只允许以下临时 engineering action vocabulary：
- nod：主要由 pitch 表达的单次点头或俯仰 gesture，primary_axis 必须是 pitch。
- turn：主要由 yaw 表达的单次偏转，primary_axis 必须是 yaw。
- tilt：主要由 roll 表达的侧倾，primary_axis 必须是 roll。
- shake：主要由 yaw 表达的一次左右摆动或否定型 gesture，primary_axis 必须是 yaw。

规划原则：
1. 不需要为了“看起来有动作”而动。普通、弱语义或没有明显韵律事件时，segments 可以是空数组，表示整句 HOLD。
2. 不要把 pitch 或 nod 当作默认动作。只有当前语义和 prosody 确实支持时才选择 nod。
3. agreement 或 emphasis 可能适合 nod；question 或 attention shift 可能适合 tilt 或 turn；negation 可能适合 shake；hesitation 或 reflection 可能适合较慢的小幅 turn 或 tilt；普通陈述可能无需显式动作。这些只是语言指导，不是硬编码映射，必须结合当前语义和 prosody 判断。
4. 动作 timing 优先参考 segment start/end、energy_peak_time_sec、pause_before_sec、pause_after_sec、relative_energy 和 speech_rate_chars_sec，不要固定放在 normalized middle。
5. amplitude_deg 是高层动作强度，单位 degree，不是最终绝对角度。通常保持约 1 到 4 度，绝对值不得超过 5 度；允许正负方向。
6. 动作数量由语义事件决定，0 个、1 个、多个都合法。保持稀疏，不为了填满语音而增加动作。
7. segment 必须在 [0, duration_sec] 内，start_sec < end_sec，按时间顺序排列且不得重叠。
8. reason 用简短字符串记录选择动作的语义原因。Trajectory Generator 将根据 action 决定具体曲线形状和三轴实现。

输出要求：
- 只能输出一个合法 JSON object，不得输出 Markdown、代码围栏、解释或其他文字。
- 顶层必须且只能包含 mode、duration_sec、segments。
- mode 必须是字符串 speaking。
- duration_sec 必须原样复制输入的 duration_sec。
- segments 必须是 JSON array，可以为空。
- 每个 segment 必须且只能包含 start_sec、end_sec、action、primary_axis、amplitude_deg、reason。
- action 只能是 nod、turn、tilt、shake，并严格遵守 action 与 primary_axis 的对应关系。
"""


class DeepSeekMotionError(RuntimeError):
    """DeepSeek did not produce a usable MotionPlan V2."""


class DeepSeekMotionBackend(MotionBackend):
    """Request MotionPlan V2 and generate fixed 30 fps relative RPY."""

    failure_mode = "audio_only"
    writes_motion_v2_artifacts = True

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
        prosody_extractor: Callable[..., ProsodyAnalysis] = extract_prosody,
        trajectory_generator: TrajectoryGenerator | None = None,
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
        if not callable(aligner) or not callable(prosody_extractor):
            raise TypeError("aligner and prosody_extractor must be callable")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.output_dir = Path(output_dir)
        self.prosody_path = self.output_dir / PROSODY_FILENAME
        self.raw_plan_path = self.output_dir / RAW_PLAN_FILENAME
        self.motion_plan_path = self.output_dir / MOTION_PLAN_FILENAME
        self.raw_relative_trajectory_path = (
            self.output_dir / RAW_RELATIVE_TRAJECTORY_FILENAME
        )
        self.final_trajectory_path = self.output_dir / FINAL_TRAJECTORY_FILENAME
        self._aligner = aligner
        self._prosody_extractor = prosody_extractor
        self._trajectory_generator = trajectory_generator or TrajectoryGenerator()

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
        output = (
            response.get("output", [])
            if isinstance(response, dict)
            else getattr(response, "output", [])
        )
        parts: list[str] = []
        for item in output or []:
            content = (
                item.get("content", [])
                if isinstance(item, dict)
                else getattr(item, "content", [])
            )
            for part in content or []:
                text = (
                    part.get("text")
                    if isinstance(part, dict)
                    else getattr(part, "text", None)
                )
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts).strip()

    @staticmethod
    def _payload(
        robot_text: str,
        duration_sec: float,
        segments: list[dict[str, Any]] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "robot_text": robot_text,
            "duration_sec": duration_sec,
        }
        if segments is not None:
            payload["segments"] = segments
        return payload

    def _request_plan(
        self,
        robot_text: str,
        duration_sec: float,
        segments: list[dict[str, Any]] | None,
    ) -> str:
        payload = self._payload(robot_text, duration_sec, segments)
        user_input = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
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

    def _write_json(self, path: Path, document: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _clear_turn_artifacts(self) -> None:
        for path in (
            self.motion_plan_path,
            self.raw_relative_trajectory_path,
            self.final_trajectory_path,
            *(self.output_dir / name for name in _LEGACY_ARTIFACT_FILENAMES),
        ):
            path.unlink(missing_ok=True)

    def _align_speech(
        self,
        robot_text: str,
        speech: RobotSpeech,
    ) -> tuple[SpeechAlignment | None, dict[str, Any]]:
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
            if result.fallback_reason:
                log(f"[MOTION][ALIGN] fallback: {result.fallback_reason}")
            log(f"[MOTION][ALIGN] method={result.method}")
            log(
                "[MOTION][ALIGN] segments="
                + json.dumps(
                    result.segments_as_dicts(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            log(f"[MOTION][ALIGN] latency={result.latency_sec:.3f}s")
            return result, result.metadata()
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

    def _write_raw_plan(
        self,
        payload: dict[str, Any],
        raw_response: str | None,
        error: str | None,
    ) -> None:
        self._write_json(
            self.raw_plan_path,
            {
                "request_payload": payload,
                "raw_response": raw_response,
                "error": error,
                "created_unix_sec": time.time(),
            },
        )

    def _write_raw_relative_trajectory(
        self,
        robot_text: str,
        duration_sec: float,
        output: MotionOutput,
    ) -> None:
        self._write_json(
            self.raw_relative_trajectory_path,
            {
                "robot_text": robot_text,
                "duration_sec": duration_sec,
                "fps": float(output.fps),
                "unit": output.unit,
                "order": list(output.order),
                "representation": output.representation,
                "trajectory": [list(frame) for frame in output.rpy_offset],
                "created_unix_sec": time.time(),
            },
        )

    def write_final_trajectory_artifact(self, final: FinalTrajectory) -> None:
        self._write_json(
            self.final_trajectory_path,
            {
                "fps": float(final.fps),
                "unit": final.unit,
                "order": list(final.order),
                "representation": "absolute_rpy",
                "duration_sec": final.duration_sec,
                "states": list(final.states),
                "trajectory": [list(frame) for frame in final.rpy],
                "created_unix_sec": time.time(),
            },
        )

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

        self._clear_turn_artifacts()
        started = time.perf_counter()
        raw_response: str | None = None
        alignment, alignment_metadata = self._align_speech(robot_text.strip(), speech)
        payload_segments: list[dict[str, Any]] | None = None
        if alignment is not None:
            try:
                prosody = self._prosody_extractor(
                    speech.pcm_s16le,
                    16_000,
                    alignment,
                )
                payload_segments = prosody.payload_segments()
                self._write_json(
                    self.prosody_path,
                    {
                        "robot_text": robot_text.strip(),
                        "alignment": alignment_metadata,
                        **prosody.to_dict(),
                    },
                )
                log(f"[MOTION][PROSODY] segments={len(prosody.segments)}")
                log(f"[MOTION][PROSODY] latency={prosody.latency_sec:.3f}s")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                self._write_json(
                    self.prosody_path,
                    {
                        "robot_text": robot_text.strip(),
                        "duration_sec": duration_sec,
                        "alignment": alignment_metadata,
                        "segments": None,
                        "error": reason,
                    },
                )
                log(f"[MOTION][PROSODY] fallback: {reason}")
                payload_segments = None
        else:
            self._write_json(
                self.prosody_path,
                {
                    "robot_text": robot_text.strip(),
                    "duration_sec": duration_sec,
                    "alignment": alignment_metadata,
                    "segments": None,
                    "error": "speech alignment unavailable",
                },
            )

        payload = self._payload(robot_text.strip(), duration_sec, payload_segments)
        log("[MOTION] request_start")
        try:
            raw_response = self._request_plan(
                robot_text.strip(), duration_sec, payload_segments
            )
            self._write_raw_plan(payload, raw_response, None)
            try:
                document = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise DeepSeekMotionError(
                    "DeepSeek Motion output is not valid JSON"
                ) from exc
            plan = MotionPlan.from_dict(document)
            validate_motion_plan(plan, expected_duration_sec=duration_sec)
            output = self._trajectory_generator.generate(plan)
            self._write_json(self.motion_plan_path, plan.to_dict())
            self._write_raw_relative_trajectory(
                robot_text.strip(), duration_sec, output
            )
            log(
                "[MOTION] plan_v2: "
                + json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
            self._log_range(output)
            return output
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._write_raw_plan(payload, raw_response, error)
            if isinstance(exc, DeepSeekMotionError):
                raise
            if isinstance(exc, (MotionPlanValidationError, TypeError, ValueError)):
                raise DeepSeekMotionError(error) from exc
            raise DeepSeekMotionError(
                f"MotionPlan V2 generation failed ({type(exc).__name__})"
            ) from exc
        finally:
            log(f"[MOTION] latency={time.perf_counter() - started:.3f}s")
