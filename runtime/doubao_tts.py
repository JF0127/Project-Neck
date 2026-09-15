"""Doubao/Volcengine Seed TTS V3 complete-PCM backend."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import time
import uuid
from typing import Any

import requests

from .contracts import RobotSpeech
from .logging_utils import log
from .tts import SAMPLE_RATE

DOUBAO_TTS_URL = "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
DOUBAO_RESOURCE_ID = "seed-tts-2.0"
# Official Seed TTS 2.0 Chinese speaker.
DEFAULT_DOUBAO_SPEAKER = "zh_female_vv_uranus_bigtts"
COMPLETED_CODE = 20_000_000
MAX_ATTEMPTS = 3


class DoubaoTTSError(RuntimeError):
    """Doubao TTS could not produce a complete PCM response."""


class _NonRetryableDoubaoError(DoubaoTTSError):
    pass


class DoubaoTTS:
    """Reuse one HTTP session and collect all V3 chunks before returning."""

    provider = "doubao"

    def __init__(
        self,
        api_key: str | None = None,
        speaker: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = (
            api_key or os.environ.get("VOLCENGINE_TTS_API_KEY", "")
        ).strip()
        self.speaker = (
            speaker
            or os.environ.get("VOLCENGINE_TTS_SPEAKER", "")
            or DEFAULT_DOUBAO_SPEAKER
        ).strip()
        if not self.api_key:
            raise DoubaoTTSError("VOLCENGINE_TTS_API_KEY is required")
        if not self.speaker:
            raise DoubaoTTSError("VOLCENGINE_TTS_SPEAKER must not be empty")
        self._session = session or requests.Session()
        self.last_ttfa_sec: float | None = None
        self._synthesis_started_at: float | None = None

    async def synthesize(self, robot_text: str) -> RobotSpeech:
        clean_text = robot_text.strip()
        if not clean_text:
            raise ValueError("DoubaoTTS requires non-empty robot_text")

        started = time.perf_counter()
        self.last_ttfa_sec = None
        self._synthesis_started_at = started
        log("TTS start provider=doubao")
        speech = await asyncio.to_thread(
            self._synthesize_with_retries,
            clean_text,
        )
        synthesis_sec = time.perf_counter() - started
        log("TTS done")
        if self.last_ttfa_sec is not None:
            log(f"TTS TTFA={self.last_ttfa_sec:.3f}s")
        log(f"TTS synthesis={synthesis_sec:.3f}s")
        log(f"audio_duration={speech.duration_sec:.3f}s")
        log(f"audio_bytes={len(speech.pcm_s16le)}")
        return speech

    def _synthesize_with_retries(self, text: str) -> RobotSpeech:
        last_error: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                return self._synthesize_once(text)
            except _NonRetryableDoubaoError:
                raise
            except (requests.RequestException, DoubaoTTSError) as exc:
                last_error = exc
                if attempt == MAX_ATTEMPTS:
                    break
                log(f"[tts] doubao retry={attempt + 1}/{MAX_ATTEMPTS}")
                time.sleep(0.25 * attempt)
        raise DoubaoTTSError(
            f"Doubao TTS failed after {MAX_ATTEMPTS} attempts: {last_error}"
        ) from last_error

    def _log_http_error(self, response: requests.Response) -> None:
        try:
            response_text = response.text
        except requests.RequestException as exc:
            response_text = f"<failed to read response body: {type(exc).__name__}>"
        log(f"[tts][doubao] status={response.status_code}")
        log(f"[tts][doubao] response={response_text}")
        log(
            "[tts][doubao] "
            f"logid={response.headers.get('X-Tt-Logid', '')}"
        )
        log(f"[tts][doubao] resource_id={DOUBAO_RESOURCE_ID}")
        log(f"[tts][doubao] speaker={self.speaker}")
        log("[tts][doubao] format=pcm")
        log(f"[tts][doubao] sample_rate={SAMPLE_RATE}")

    def _synthesize_once(self, text: str) -> RobotSpeech:
        headers = {
            "Content-Type": "application/json",
            "X-Api-Key": self.api_key,
            "X-Api-Resource-Id": DOUBAO_RESOURCE_ID,
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        payload = {
            "user": {"uid": "project-neck"},
            "req_params": {
                "text": text,
                "speaker": self.speaker,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": SAMPLE_RATE,
                    "speech_rate": 0,
                    "loudness_rate": 0,
                },
                "additions": json.dumps(
                    {"disable_markdown_filter": True},
                    ensure_ascii=False,
                ),
            },
        }

        try:
            response = self._session.post(
                DOUBAO_TTS_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(5.0, 60.0),
            )
        except requests.RequestException:
            raise
        if not 200 <= response.status_code < 300:
            self._log_http_error(response)
        if response.status_code in {400, 401, 403, 404, 422}:
            response.close()
            raise _NonRetryableDoubaoError(
                f"Doubao TTS rejected the request with HTTP {response.status_code}"
            )
        if response.status_code in {408, 429} or response.status_code >= 500:
            response.close()
            raise DoubaoTTSError(
                f"Doubao TTS temporary HTTP error {response.status_code}"
            )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            response.close()
            raise _NonRetryableDoubaoError(
                f"Doubao TTS returned HTTP {response.status_code}"
            ) from exc

        audio_chunks: list[bytes] = []
        completed = False
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    event: Any = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DoubaoTTSError("Doubao TTS returned invalid JSON") from exc
                if not isinstance(event, dict):
                    raise DoubaoTTSError("Doubao TTS returned a non-object event")

                data = event.get("data")
                if isinstance(data, str) and data:
                    try:
                        audio = base64.b64decode(data, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise DoubaoTTSError(
                            "Doubao TTS returned invalid base64 audio"
                        ) from exc
                    if audio:
                        if self.last_ttfa_sec is None:
                            synthesis_started = (
                                self._synthesis_started_at or time.perf_counter()
                            )
                            self.last_ttfa_sec = (
                                time.perf_counter() - synthesis_started
                            )
                            log("TTS first_audio")
                        audio_chunks.append(audio)

                raw_code = event.get("code")
                try:
                    code = int(raw_code) if raw_code is not None else None
                except (TypeError, ValueError):
                    code = raw_code
                if code == COMPLETED_CODE:
                    completed = True
                    break
                if code not in (None, 0):
                    message = str(event.get("message", "unknown error"))
                    raise _NonRetryableDoubaoError(
                        f"Doubao TTS error code={code}: {message}"
                    )
        finally:
            response.close()

        if not completed:
            raise DoubaoTTSError("Doubao TTS stream ended without code=20000000")
        pcm_s16le = b"".join(audio_chunks)
        if not pcm_s16le or len(pcm_s16le) % 2:
            raise DoubaoTTSError("Doubao TTS returned invalid pcm_s16le audio")

        return RobotSpeech(
            text=text,
            pcm_s16le=pcm_s16le,
            words=(),
            duration_sec=len(pcm_s16le) / 2 / SAMPLE_RATE,
        )
