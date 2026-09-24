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
from .text_motion_plan import TextMotionPlan
from .motion_plan import MotionSegment
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

MOTION_SYSTEM_PROMPT = """你是机器人的语义动作规划器。输入只有 JSON {"reply_text":"完整机器人回复"}。
只依据文本挑选少量关键语义动作；不推测音频、语速、时长、韵律或播放时间。
只允许 nod（明确肯定/认同/强调）和 shake（明确否定/拒绝）。普通陈述、礼貌语和没有强语义事件的句子必须返回空 actions；不要每句话强行安排动作。最多 3 个，宁缺毋滥。
输出且只输出合法 JSON：{"reply_text":"原样复制输入文本","actions":[{"type":"nod 或 shake","anchor":"原文中的连续片段或关键词","position":0,"intensity":"low 或 medium 或 high"}]}。
position 是 anchor 在 reply_text 中的零基 Unicode 字符起始下标（不是字节下标或时间），必须与原文完全匹配；按位置升序排列，不重叠。actions 可以为空。
不输出动作时间、角度、逐帧 RPY、Global Motion、解释或 Markdown。
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
    def _payload(reply_text: str) -> dict[str, str]:
        return {"reply_text": reply_text}

    def _request_plan(self, reply_text: str) -> str:
        payload = self._payload(reply_text)
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

    def plan_text(self, reply_text: str) -> tuple[str, TextMotionPlan]:
        """Only DeepSeek call: text in, sparse semantic plan out."""
        if not isinstance(reply_text, str) or not reply_text.strip():
            raise DeepSeekMotionError("reply_text is empty")
        raw = self._request_plan(reply_text)
        try:
            plan = TextMotionPlan.from_dict(json.loads(raw), reply_text)
        except (ValueError, TypeError) as exc:
            raise DeepSeekMotionError(f"invalid text motion plan: {exc}") from exc
        return raw, plan

    @staticmethod
    def _for_generator(plan: TextMotionPlan, duration_sec: float) -> MotionPlan:
        """Coarse positional adapter; NOT a TTS alignment or planner input."""
        segments = []
        previous_end = 0.0
        for action in plan.actions:
            start = duration_sec * action.position / len(plan.reply_text)
            start = max(start, previous_end)
            end = min(duration_sec, start + min(0.5, duration_sec / max(1, len(plan.actions))))
            if end - start < 1 / 30:
                continue
            segments.append(MotionSegment(
                start, end, action.type,
                "pitch" if action.type == "nod" else "yaw",
                {"low": 1.0, "medium": 2.0, "high": 3.5}[action.intensity],
                f"text anchor: {action.anchor}",
            ))
            previous_end = end
        return MotionPlan("speaking", duration_sec, tuple(segments))

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
        # Speech duration belongs to the downstream trajectory adapter only.
        # The DeepSeek request and TextMotionPlan never see speech or its duration.
        self._clear_turn_artifacts()
        self.prosody_path.unlink(missing_ok=True)
        started = time.perf_counter()
        raw_response: str | None = None
        payload = self._payload(robot_text)
        log("[MOTION] request_start")
        try:
            raw_response, text_plan = self.plan_text(robot_text)
            self._write_raw_plan(payload, raw_response, None)
            self._write_json(self.motion_plan_path, text_plan.to_dict())
            if not isinstance(speech, RobotSpeech) or not math.isfinite(speech.duration_sec) or speech.duration_sec <= 0:
                raise DeepSeekMotionError("downstream trajectory requires robot speech duration")
            duration_sec = float(speech.duration_sec)
            timed_plan = self._for_generator(text_plan, duration_sec)
            validate_motion_plan(timed_plan, expected_duration_sec=duration_sec)
            output = self._trajectory_generator.generate(timed_plan)
            self._write_raw_relative_trajectory(robot_text, duration_sec, output)
            log("[MOTION] text_plan: " + json.dumps(text_plan.to_dict(), ensure_ascii=False))
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
