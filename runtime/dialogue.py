"""Small replaceable dialogue backends for Runtime V1."""
from __future__ import annotations

import os
import time
from typing import Any, Callable, Protocol

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_HISTORY_TURNS = 5
DEFAULT_MAX_TOKENS = 128
DEFAULT_TIMEOUT_SEC = 30.0

DEFAULT_SYSTEM_PROMPT = """你是一个自然的人形机器人对话助手。默认使用中文回答；如果用户明确使用其他语言并希望以该语言交流，可以自然响应。请使用自然、简洁、适合口语表达的回答，通常只回答一到三句话。除非用户明确要求详细解释，否则不要长篇回答。不要使用 Markdown、列表、标题、代码块或复杂排版。回答将直接交给语音合成系统播放，因此应像真实口语交流。"""


class DialogueError(RuntimeError):
    """A dialogue backend could not produce valid robot text."""


class DialoguePolicy(Protocol):
    def reply(self, user_text: str) -> str: ...


class FixedDialogue:
    backend = "fixed"
    model = None

    def __init__(self, reply_text: str = "I heard you. Thank you for talking with me."):
        self.reply_text = reply_text

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model}

    def reply(self, user_text: str) -> str:
        return self.reply_text


class EchoDialogue:
    backend = "echo"
    model = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": self.backend, "model": self.model}

    def reply(self, user_text: str) -> str:
        clean = user_text.strip()
        return f"You said: {clean}" if clean else "I could not hear any speech."


class DeepSeekDialogue:
    """Synchronous OpenAI-compatible DeepSeek client with short in-process history."""

    backend = "deepseek"
    thinking_mode = "disabled"

    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        base_url: str = DEEPSEEK_BASE_URL,
        history_turns: int = DEFAULT_HISTORY_TURNS,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise DialogueError("DEEPSEEK_API_KEY is required for dialogue=deepseek")
        if not model.strip():
            raise ValueError("DeepSeek model must not be empty")
        if not base_url.strip():
            raise ValueError("DeepSeek base_url must not be empty")
        if history_turns < 1:
            raise ValueError("history_turns must be at least 1")
        if timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be greater than zero")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not system_prompt.strip():
            raise ValueError("system_prompt must not be empty")
        if client is not None and client_factory is not None:
            raise ValueError("provide either client or client_factory, not both")

        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.history_turns = int(history_turns)
        self.timeout_sec = float(timeout_sec)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt.strip()
        self._history: list[tuple[str, str]] = []
        self._last_metadata: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
        }

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
            # Disable SDK retries so a real-time turn has one bounded API attempt.
            self._client = client_factory(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout_sec,
                max_retries=0,
            )

    def __repr__(self) -> str:
        return (
            f"DeepSeekDialogue(model={self.model!r}, base_url={self.base_url!r}, "
            f"history_turns={self.history_turns}, timeout_sec={self.timeout_sec})"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        result = dict(self._last_metadata)
        if "usage" in result:
            result["usage"] = dict(result["usage"])
        return result

    @property
    def history_messages(self) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": self.system_prompt}]
        for user_text, assistant_text in self._history:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
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
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(usage, field, None)
            if isinstance(value, int) and value >= 0:
                result[field] = value
        return result

    def reply(self, user_text: str) -> str:
        clean_user_text = user_text.strip()
        if not clean_user_text:
            raise DialogueError("DeepSeek dialogue requires non-empty user text")

        messages = self.history_messages
        messages.append({"role": "user", "content": clean_user_text})
        started = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                # DeepSeek's OpenAI-compatible API requires custom parameters in
                # extra_body. This explicitly selects low-latency non-thinking mode.
                extra_body={"thinking": {"type": "disabled"}},
                timeout=self.timeout_sec,
            )
        except Exception as exc:
            latency = time.monotonic() - started
            self._last_metadata = {
                "backend": self.backend,
                "model": self.model,
                "thinking_mode": self.thinking_mode,
                "latency_sec": latency,
                "status": "error",
            }
            raise self._request_error(exc) from exc

        latency = time.monotonic() - started
        try:
            choices = response.choices
            content = choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            self._last_metadata = {
                "backend": self.backend,
                "model": self.model,
                "thinking_mode": self.thinking_mode,
                "latency_sec": latency,
                "status": "invalid_response",
            }
            raise DialogueError("DeepSeek API returned an invalid response") from exc
        if not isinstance(content, str) or not content.strip():
            self._last_metadata = {
                "backend": self.backend,
                "model": self.model,
                "thinking_mode": self.thinking_mode,
                "latency_sec": latency,
                "status": "empty_response",
            }
            raise DialogueError("DeepSeek API returned empty robot text")

        robot_text = content.strip()
        self._history.append((clean_user_text, robot_text))
        if len(self._history) > self.history_turns:
            del self._history[: len(self._history) - self.history_turns]

        metadata: dict[str, Any] = {
            "backend": self.backend,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
            "latency_sec": latency,
            "status": "complete",
        }
        usage = self._usage(response)
        if usage:
            metadata["usage"] = usage
        self._last_metadata = metadata
        return robot_text
