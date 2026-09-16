"""Stateless DeepSeek dialogue backend for the new Runtime."""
from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Callable, Iterator

from .contracts import DialogueRequest
from .datetime_tool import CurrentDateTime
from .logging_utils import log
from .web_search import WebSearch, WebSearchError

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_TEMPERATURE = 0.7
MAX_WEB_SEARCH_CALLS = 3
WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": (
        "搜索互联网。普通静态知识和纯当前日期时间问题不需要调用；当回答依赖"
        "天气、新闻、价格、赛事、人物现状、近期事件、最新版本等实时互联网信息"
        "时必须调用。"
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
}
CURRENT_DATETIME_TOOL = {
    "type": "function",
    "name": "get_current_datetime",
    "description": (
        "获取 Asia/Shanghai 当前日期、时间和星期。今天几号、当前日期、星期几、"
        "现在几点以及当前年/月/日等纯日期时间问题必须调用此工具。"
    ),
    "strict": True,
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
}
REALTIME_SEARCH_POLICY = (
    "涉及今天几号、当前日期、星期几、现在几点、当前年份/月/日等纯日期时间问题，"
    "必须调用 get_current_datetime；中国当前时间直接使用 Asia/Shanghai。回答时必须"
    "严格使用工具返回的数据，禁止自行推算、转换或修改时间，也禁止调用 web_search "
    "获取当前日期。涉及天气、"
    "新闻、价格、赛事、人物现状、近期事件、最新版本、实时数据等实时互联网信息，"
    "必须调用 web_search，禁止仅凭模型记忆回答。普通静态知识问题不应调用工具。"
)
SPOKEN_RESPONSE_POLICY = (
    "使用自然、简洁、适合语音播放的口语回答，回复尽量控制在30个汉字以内，"
    "包括标点符号；仅在无法完整回答时才允许略微超过。"
)

DEFAULT_SYSTEM_PROMPT = """你是一个自然的人形机器人对话助手。默认使用中文回答；如果用户明确使用其他语言并希望以该语言交流，可以自然响应。请使用自然、简洁、适合口语表达的回答，通常只回答一到三句话。除非用户明确要求详细解释，否则不要长篇回答。不要使用 Markdown、列表、标题、代码块或复杂排版。回答将直接交给语音合成系统播放，因此应像真实口语交流。"""


class DialogueError(RuntimeError):
    """A dialogue backend could not produce valid robot text."""


class FixedDialogue:
    """Small deterministic backend for pure-software tests."""

    def __init__(self, reply_text: str = "I heard you.") -> None:
        if not reply_text.strip():
            raise ValueError("fixed dialogue reply must not be empty")
        self.reply_text = reply_text.strip()

    def reply(self, request: DialogueRequest) -> str:
        if not isinstance(request, DialogueRequest):
            raise TypeError("FixedDialogue.reply requires DialogueRequest")
        return self.reply_text


class DeepSeekDialogue:
    """Synchronous OpenAI-compatible client with request-owned history."""

    backend = "deepseek"
    thinking_mode = "disabled"

    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        web_search: WebSearch | None = None,
        current_datetime: CurrentDateTime | None = None,
        debug: bool = False,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise DialogueError("DEEPSEEK_API_KEY is required for dialogue=deepseek")
        if not model.strip():
            raise ValueError("DeepSeek model must not be empty")
        if not base_url.strip():
            raise ValueError("DeepSeek base_url must not be empty")
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be finite and between 0 and 2")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if client is not None and client_factory is not None:
            raise ValueError("provide either client or client_factory, not both")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.system_prompt = (
            f"{system_prompt.strip()}\n\n{REALTIME_SEARCH_POLICY}\n"
            f"{SPOKEN_RESPONSE_POLICY}"
        )
        self.current_datetime = current_datetime or CurrentDateTime()
        self.web_search = web_search or WebSearch(
            current_datetime=self.current_datetime
        )
        self.debug = bool(debug)
        self._last_metadata: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
        }
        self._last_response_text = ""

        if client is not None:
            self._client = client
        else:
            if client_factory is None:
                try:
                    from openai import OpenAI
                except ImportError as exc:
                    raise DialogueError(
                        "openai package is required for dialogue=deepseek"
                    ) from exc
                client_factory = OpenAI
            self._client = client_factory(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout_sec,
                max_retries=0,
            )

    def __repr__(self) -> str:
        return (
            f"DeepSeekDialogue(model={self.model!r}, base_url={self.base_url!r}, "
            f"timeout_sec={self.timeout_sec}, temperature={self.temperature})"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        result = dict(self._last_metadata)
        if "usage" in result:
            result["usage"] = dict(result["usage"])
        return result

    @property
    def response_text(self) -> str:
        """Final no-pending-tool answer from the most recent response."""
        return self._last_response_text

    def _messages(self, request: DialogueRequest) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for turn in request.history:
            messages.append({"role": "user", "content": turn.user_text})
            messages.append({"role": "assistant", "content": turn.robot_text})
        messages.append({"role": "user", "content": request.user_text.strip()})
        return messages

    @staticmethod
    def _request_error(error: Exception) -> DialogueError:
        name = type(error).__name__
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            message = "DeepSeek authentication failed"
        elif name == "RateLimitError":
            message = "DeepSeek rate limit exceeded"
        elif name in {"APITimeoutError", "TimeoutError"}:
            message = "DeepSeek API request timed out"
        elif name in {"APIConnectionError", "ConnectError"}:
            message = "DeepSeek API network connection failed"
        elif name == "APIStatusError":
            status_code = getattr(error, "status_code", None)
            message = (
                f"DeepSeek API returned HTTP {status_code}"
                if status_code is not None
                else "DeepSeek API returned an error status"
            )
        else:
            message = f"DeepSeek API request failed ({name})"
        return DialogueError(message)

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        result: dict[str, int] = {}
        for field in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "prompt_tokens",
            "completion_tokens",
        ):
            value = getattr(usage, field, None)
            if isinstance(value, int) and value >= 0:
                result[field] = value
        return result

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @classmethod
    def _completed_text(cls, response: Any) -> str:
        output_text = cls._field(response, "output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        parts: list[str] = []
        for item in cls._field(response, "output", []) or []:
            text = cls._field(item, "text")
            if isinstance(text, str) and text:
                parts.append(text)
            for content in cls._field(item, "content", []) or []:
                text = cls._field(content, "text")
                if isinstance(text, str) and text:
                    parts.append(text)
        return "".join(parts).strip()

    @classmethod
    def _terminal_reason(cls, event: Any, response: Any) -> str:
        error = cls._field(response, "error") or cls._field(event, "error")
        message = cls._field(error, "message")
        if isinstance(message, str) and message.strip():
            return message.strip()
        details = cls._field(response, "incomplete_details") or cls._field(
            event, "incomplete_details"
        )
        reason = cls._field(details, "reason")
        if isinstance(reason, str) and reason.strip():
            return reason.strip()
        return "reason unavailable"

    @classmethod
    def _function_calls(cls, response: Any) -> list[Any]:
        return [
            item
            for item in cls._field(response, "output", []) or []
            if cls._field(item, "type") == "function_call"
        ]

    @classmethod
    def _function_arguments(cls, call: Any) -> tuple[str, str, dict[str, Any]]:
        call_id = cls._field(call, "call_id")
        name = cls._field(call, "name")
        arguments = cls._field(call, "arguments")
        try:
            value = (
                json.loads(arguments)
                if isinstance(arguments, str) and arguments.strip()
                else arguments
            )
        except json.JSONDecodeError as exc:
            raise DialogueError("DeepSeek returned invalid function arguments") from exc
        if value is None:
            value = {}
        if not isinstance(call_id, str) or not call_id:
            raise DialogueError("DeepSeek function call has no call_id")
        if not isinstance(name, str) or not name:
            raise DialogueError("DeepSeek function call has no name")
        if not isinstance(value, dict):
            raise DialogueError("DeepSeek function arguments must be an object")
        return call_id, name, value

    def reply_stream(self, request: DialogueRequest) -> Iterator[str]:
        """Run Responses API function calls and yield final text deltas."""
        if not isinstance(request, DialogueRequest):
            raise TypeError("DeepSeekDialogue.reply_stream requires DialogueRequest")
        if not request.user_text.strip():
            raise DialogueError("DeepSeek dialogue requires non-empty user text")

        started = time.monotonic()
        first_token_at: float | None = None
        log("DeepSeek request_start")

        def note_first_token() -> None:
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = time.monotonic()
                log("DeepSeek first_token")
                log(f"DeepSeek TTFT: {first_token_at - started:.3f}s")

        self._last_response_text = ""
        response_input: list[Any] = list(self._messages(request))
        assistant_parts: list[str] = []
        searches_used = 0
        final_response: Any | None = None
        incomplete_at_token_limit = False

        while True:
            terminal_type: str | None = None
            terminal_event: Any | None = None
            terminal_response: Any | None = None
            round_parts: list[str] = []
            try:
                events = self._client.responses.create(
                    model=self.model,
                    input=response_input,
                    max_output_tokens=4096,
                    temperature=self.temperature,
                    reasoning={"effort": "none"},
                    tools=[CURRENT_DATETIME_TOOL, WEB_SEARCH_TOOL],
                    tool_choice="auto",
                    timeout=self.timeout_sec,
                    stream=True,
                )
                for event in events:
                    event_type = str(self._field(event, "type", "unknown"))
                    if self.debug:
                        log(f"[deepseek][event] {event_type}")
                    if event_type == "response.output_text.delta":
                        delta = self._field(event, "delta")
                        if isinstance(delta, str) and delta:
                            note_first_token()
                            round_parts.append(delta)
                            assistant_parts.append(delta)
                            yield delta
                    elif event_type in {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    }:
                        terminal_type = event_type
                        terminal_event = event
                        terminal_response = self._field(event, "response")
            except DialogueError:
                raise
            except Exception as exc:
                self._last_metadata = {
                    "backend": self.backend,
                    "model": self.model,
                    "thinking_mode": self.thinking_mode,
                    "latency_sec": time.monotonic() - started,
                    "status": "error",
                }
                raise self._request_error(exc) from exc

            if terminal_type == "response.incomplete":
                reason = self._terminal_reason(terminal_event, terminal_response)
                if reason == "max_output_tokens":
                    if not assistant_parts:
                        fallback_text = self._completed_text(terminal_response)
                        if fallback_text:
                            note_first_token()
                            assistant_parts.append(fallback_text)
                            yield fallback_text
                    if assistant_parts:
                        log(
                            "[deepseek] response incomplete: max_output_tokens; "
                            "keeping streamed assistant text"
                        )
                        final_response = terminal_response
                        incomplete_at_token_limit = True
                        break
                raise DialogueError(f"DeepSeek response incomplete: {reason}")
            if terminal_type == "response.failed":
                reason = self._terminal_reason(terminal_event, terminal_response)
                raise DialogueError(f"DeepSeek response failed: {reason}")
            if terminal_type != "response.completed":
                raise DialogueError(
                    "DeepSeek stream ended without a terminal response event"
                )

            function_calls = self._function_calls(terminal_response)
            if not function_calls:
                final_response = terminal_response
                if not round_parts:
                    fallback_text = self._completed_text(terminal_response)
                    if fallback_text:
                        note_first_token()
                        round_parts.append(fallback_text)
                        assistant_parts.append(fallback_text)
                        yield fallback_text
                self._last_response_text = "".join(round_parts).strip()
                break

            response_input.extend(
                list(self._field(terminal_response, "output", []) or [])
            )
            for call in function_calls:
                call_id, name, arguments = self._function_arguments(call)
                if name == "get_current_datetime":
                    result = json.dumps(
                        self.current_datetime.get_current_datetime(),
                        ensure_ascii=False,
                    )
                elif name == "web_search":
                    if searches_used >= MAX_WEB_SEARCH_CALLS:
                        raise DialogueError(
                            "DeepSeek exceeded the limit of 3 web_search calls per turn"
                        )
                    query = arguments.get("query")
                    if not isinstance(query, str) or not query.strip():
                        raise DialogueError(
                            "DeepSeek web_search call has no query"
                        )
                    try:
                        result = self.web_search.search(query.strip())
                    except WebSearchError as exc:
                        raise DialogueError(f"web_search failed: {exc}") from exc
                    searches_used += 1
                else:
                    raise DialogueError(
                        f"DeepSeek requested unsupported function: {name}"
                    )
                response_input.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": result,
                    }
                )

        assistant_text = "".join(assistant_parts).strip()
        if incomplete_at_token_limit:
            self._last_response_text = assistant_text
        if not assistant_text or not self._last_response_text:
            raise DialogueError(
                "DeepSeek response.completed contained no assistant text"
            )

        metadata: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
            "latency_sec": time.monotonic() - started,
            "status": (
                "incomplete_max_output_tokens"
                if incomplete_at_token_limit
                else "complete"
            ),
            "web_search_calls": searches_used,
        }
        usage = self._usage(final_response)
        if usage:
            metadata["usage"] = usage
        self._last_metadata = metadata
        log(f"DEEPSEEK FINAL: {self._last_response_text}")
        log(f"DeepSeek total: {time.monotonic() - started:.3f}s")

    def reply(self, request: DialogueRequest) -> str:
        if not isinstance(request, DialogueRequest):
            raise TypeError("DeepSeekDialogue.reply requires DialogueRequest")
        for _ in self.reply_stream(request):
            pass
        if not self._last_response_text:
            raise DialogueError("DeepSeek API returned empty robot text")
        return self._last_response_text
