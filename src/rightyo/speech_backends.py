"""Configuration-selected speech backends, including opt-in hosted ones.

Hosted backends send audio only after explicit `allow_hosted` consent, use stdlib
HTTP with no environment proxies, refuse redirects, bound response bytes and raise
sanitized errors without audio, transcript, URL, or credential content. Endpoint
schemas were read from the official sources on 2026-10-02 and are cited on each
class. Nothing here is an accuracy claim for any hosted model.
"""

from __future__ import annotations

import http.client
import io
import json
import math
import re
import secrets
import socket
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any, Callable

from rightyo.credentials import CredentialError, load_diarizer_api_key, load_transcriber_api_key
from rightyo.live_audio import BYTES_PER_MS, SAMPLE_RATE, LiveAudioError, LiveConfig
from rightyo.providers import Diarizer, Transcriber

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_AUDIO_MS = 16000  # the largest utterance window plus the diarizer's margin
DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEEPGRAM_DIARIZE_MODELS = ("latest", "v1", "v2")
TRANSCRIBER_KINDS = ("whisper.cpp", "hosted-openai-compatible")
DIARIZER_KINDS = ("nemotron.cpp", "hosted-deepgram")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?")


class HostedSpeechError(LiveAudioError):
    """Sanitized hosted-backend failure: no audio, transcript, URL or credential content."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any):
        # Never forward a credential to a redirected endpoint.
        fp.close()
        raise HostedSpeechError("Hosted speech redirect refused")


def _https_endpoint(value: Any, label: str) -> str:
    parts = None
    if isinstance(value, str) and len(value) <= 2048 and value.isascii():
        try:
            parts = urllib.parse.urlsplit(value)
        except ValueError:
            pass
    if (
        parts is None
        or parts.scheme != "https"
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise LiveAudioError(f"{label} requires an https endpoint")
    return value


def _timeout(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 120:
        raise LiveAudioError("Invalid hosted processing timeout")
    return value


def _name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise LiveAudioError(f"Invalid {label}")
    return value


def _wav(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm)
    return buffer.getvalue()


def _interval(start: Any, end: Any, duration_ms: int, label: str) -> tuple[int, int]:
    """Seconds to bounded utterance-relative milliseconds, as the local path bounds them."""
    if (
        type(start) not in (int, float)
        or type(end) not in (int, float)
        or not math.isfinite(start)
        or not math.isfinite(end)
        or not 0 <= start <= end
    ):
        raise HostedSpeechError(f"{label} timestamp is invalid")
    start_ms, end_ms = round(start * 1000), round(end * 1000)
    if start_ms > duration_ms or end_ms > duration_ms + 1000:
        raise HostedSpeechError(f"{label} timestamp is invalid")
    return start_ms, min(end_ms, duration_ms)


class _HostedClient:
    """One consented endpoint: bounded, sanitized, redirect-free, credential at use only."""

    def __init__(
        self,
        label: str,
        *,
        allow_hosted: bool,
        timeout_seconds: float,
        cancelled: Callable[[], bool] | None,
        load_key: Callable[[], str],
    ) -> None:
        if allow_hosted is not True:
            raise HostedSpeechError(f"{label} requires explicit hosted consent to send audio")
        self.label = label
        self.timeout = _timeout(timeout_seconds)
        self.cancelled = cancelled if cancelled is not None else (lambda: False)
        self.load_key = load_key
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def post(self, url: str, body: bytes, headers: dict[str, str], scheme: str) -> dict[str, Any]:
        if self.cancelled():
            raise HostedSpeechError(f"{self.label} request was cancelled")
        key = None
        try:
            key = self.load_key()
        except CredentialError:
            pass
        if key is None:
            raise HostedSpeechError(f"{self.label} credential is unavailable")
        if self.cancelled():
            del key
            raise HostedSpeechError(f"{self.label} request was cancelled")
        request = urllib.request.Request(
            url, data=body, method="POST", headers={**headers, "Authorization": scheme + key}
        )
        del key
        failure = None
        content = b""
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                content = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            try:
                error.close()
            except OSError:
                pass
            if error.code in (429, 503):
                failure = f"{self.label} temporarily unavailable; no automatic retry"
            elif type(error.code) is int and 100 <= error.code <= 599:
                failure = f"{self.label} request failed (HTTP {error.code})"
            else:
                failure = f"{self.label} request failed"
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            OSError,
            http.client.HTTPException,
        ):
            failure = f"{self.label} connection failed or timed out"
        except HostedSpeechError:
            failure = f"{self.label} redirect refused"
        finally:
            request.remove_header("Authorization")
        # Raise outside the handlers so reflected bodies never survive as __context__.
        if failure is not None:
            raise HostedSpeechError(failure)
        if len(content) > MAX_RESPONSE_BYTES:
            raise HostedSpeechError(f"{self.label} response exceeds size limit")
        document = None
        try:
            document = json.loads(content)
        except (ValueError, UnicodeError, RecursionError):
            pass
        if not isinstance(document, dict):
            raise HostedSpeechError(f"{self.label} returned an invalid response")
        return document


def _multipart(fields: list[tuple[str, str]], filename: str, audio: bytes) -> tuple[bytes, str]:
    boundary = "rightyo-" + secrets.token_hex(16)
    parts = []
    for name, value in fields:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        )
    body = "".join(parts).encode("utf-8")
    body += (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: audio/wav\r\n\r\n'
    ).encode("ascii")
    body += audio + f"\r\n--{boundary}--\r\n".encode("ascii")
    return body, f"multipart/form-data; boundary={boundary}"


def _punctuate(words: list[str], text: str) -> list[str] | None:
    """Carry the response's punctuation onto bare words when both align exactly."""
    lowered = text.lower()
    cursor = 0
    result = []
    for word in words:
        index = lowered.find(word.lower(), cursor)
        if index < 0:
            return None
        # Only whitespace and leading punctuation (quotes, brackets) may precede a word.
        prefix = text[cursor:index].strip()
        if any(char.isalnum() or char.isspace() for char in prefix):
            return None
        end = index + len(word)
        while end < len(text) and not text[end].isspace():
            end += 1
        result.append(prefix + text[index:end])
        cursor = end
    if text[cursor:].strip():
        return None
    return result


def openai_units(document: dict[str, Any], duration_ms: int) -> list[dict[str, Any]]:
    """Utterance-relative units from a verbose_json transcription object."""
    label = "Hosted transcriber"
    text = document.get("text")
    if not isinstance(text, str) or len(text) > 4000:
        raise HostedSpeechError(f"{label} returned an invalid response")
    if not text.strip():
        return []
    words = document.get("words")
    if isinstance(words, list) and words:
        if len(words) > 4000:
            raise HostedSpeechError(f"{label} returned an invalid response")
        units = []
        for word in words:
            value = word.get("word") if isinstance(word, dict) else None
            if not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise HostedSpeechError(f"{label} returned an invalid response")
            start, end = _interval(word.get("start"), word.get("end"), duration_ms, label)
            units.append({"text": " " + value.strip(), "start_ms": start, "end_ms": end})
        punctuated = _punctuate([unit["text"].strip() for unit in units], text)
        if punctuated is not None:
            for unit, value in zip(units, punctuated):
                unit["text"] = " " + value
        return units
    segments = document.get("segments")
    if isinstance(segments, list) and segments:
        if len(segments) > 1000:
            raise HostedSpeechError(f"{label} returned an invalid response")
        units = []
        for segment in segments:
            value = segment.get("text") if isinstance(segment, dict) else None
            if not isinstance(value, str) or len(value) > 4000:
                raise HostedSpeechError(f"{label} returned an invalid response")
            if not value.strip():
                continue
            start, end = _interval(segment.get("start"), segment.get("end"), duration_ms, label)
            if not value[:1].isspace():
                value = " " + value
            units.append({"text": value, "start_ms": start, "end_ms": end})
        return units
    raise HostedSpeechError(f"{label} returned no timed units")


class OpenAICompatibleTranscriber:
    """`Transcriber` over `POST <endpoint>` in the OpenAI audio transcription schema.

    Schema read 2026-10-02 from the official OpenAPI document
    (github.com/openai/openai-openapi, main bb9b870, `info.version` 2.3.0), operation
    `POST /audio/transcriptions` on `https://api.openai.com/v1`: multipart/form-data
    fields `file`, `model`, `response_format`, optional `language`, and
    `timestamp_granularities[]` (`word`, `segment`); "`response_format` must be set
    `verbose_json` to use timestamp granularities", which "is not available for
    `gpt-4o-transcribe-diarize`", and `gpt-4o-transcribe`/`gpt-4o-mini-transcribe`
    support only `json`. The verbose_json response has `text`, `words[]` (`word`,
    `start`, `end` in seconds) and `segments[]` (`start`, `end`, `text`).

    Word units are preferred, with the response's punctuation restored when words and
    `text` align exactly and bare words otherwise; without `words` the segments are
    used; without either the utterance fails closed. Any other server must follow the
    same schema. `register` is accepted and ignored: cancellation is the `cancelled`
    guard before each request plus the request timeout.
    """

    recognizer_id = "hosted-openai-compatible"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        allow_hosted: bool = False,
        language: str | None = None,
        timeout_seconds: float = 30,
        cancelled: Callable[[], bool] | None = None,
        load_key: Callable[[], str] = load_transcriber_api_key,
    ) -> None:
        self.endpoint = _https_endpoint(endpoint, "Hosted transcriber")
        self.model = _name(model, "hosted transcriber model")
        if language is not None and (
            not isinstance(language, str) or not _LANGUAGE.fullmatch(language)
        ):
            raise LiveAudioError("Invalid hosted transcriber language")
        self.language = language
        self._client = _HostedClient(
            "Hosted transcriber",
            allow_hosted=allow_hosted,
            timeout_seconds=timeout_seconds,
            cancelled=cancelled,
            load_key=load_key,
        )

    def transcribe(
        self, pcm: bytes, register: Callable[[Any], None] | None = None
    ) -> list[dict[str, Any]]:
        if (
            not isinstance(pcm, bytes)
            or not pcm
            or len(pcm) % 2
            or len(pcm) > MAX_AUDIO_MS * BYTES_PER_MS
        ):
            raise HostedSpeechError("Invalid utterance audio")
        fields = [
            ("model", self.model),
            ("response_format", "verbose_json"),
            ("timestamp_granularities[]", "word"),
            ("timestamp_granularities[]", "segment"),
        ]
        if self.language is not None:
            fields.append(("language", self.language))
        body, content_type = _multipart(fields, "utterance.wav", _wav(pcm))
        document = self._client.post(self.endpoint, body, {"Content-Type": content_type}, "Bearer ")
        return openai_units(document, len(pcm) // BYTES_PER_MS)


def deepgram_timeline(
    document: dict[str, Any], window_ms: int, offset_ms: int
) -> list[dict[str, Any]]:
    """Stream-relative speaker runs from one pre-recorded response's labelled words."""
    label = "Hosted diarizer"
    words = None
    try:
        words = document["results"]["channels"][0]["alternatives"][0].get("words")
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    if not isinstance(words, list) or len(words) > 4000:
        raise HostedSpeechError(f"{label} returned an invalid response")
    timeline: list[dict[str, Any]] = []
    previous = None
    for word in words:
        if not isinstance(word, dict):
            raise HostedSpeechError(f"{label} returned an invalid response")
        start, end = _interval(word.get("start"), word.get("end"), window_ms, label)
        speaker = word.get("speaker")
        if speaker is None:
            # An unlabelled word breaks the run; the gap stays unknown downstream.
            previous = None
            continue
        if type(speaker) is not int or not 0 <= speaker <= 25:
            raise HostedSpeechError(f"{label} returned an invalid response")
        start, end = start + offset_ms, end + offset_ms
        if (
            previous is not None
            and previous["speaker"] == speaker + 1
            and start >= previous["start_ms"]
        ):
            previous["end_ms"] = max(previous["end_ms"], end)
        else:
            previous = {"speaker": speaker + 1, "start_ms": start, "end_ms": end}
            timeline.append(previous)
    return [segment for segment in timeline if segment["start_ms"] < segment["end_ms"]]


class DeepgramDiarizer:
    """`Diarizer` over Deepgram pre-recorded `POST https://api.deepgram.com/v1/listen`.

    Read 2026-10-02 from developers.deepgram.com/docs/pre-recorded-audio and
    developers.deepgram.com/docs/diarization: the raw audio is the request body with
    `Content-Type: audio/wav` and `Authorization: Token <key>`; query `model` selects
    the model and `diarize_model` (`latest`, `v1`, `v2`) enables diarization ("The
    `diarize` parameter is deprecated. Use `diarize_model` instead"). The response's
    `results.channels[0].alternatives[0].words[]` carries `word`, `start`, `end`
    (seconds), `speaker` (zero-based integer) and `speaker_confidence`.

    Labels are assigned per request. Each `segments`/`finish` call sends only the
    trailing `window_ms` of pushed audio, sized to cover one utterance, and returns
    that window's timeline in stream milliseconds with consecutive words of one
    speaker merged into a segment. "Speaker A" in one utterance is therefore not
    known to be the same person as "Speaker A" in the next: the service documents no
    cross-request label stability and none is invented here. Audio outside the window
    is not retained, and the whole-session timeline cap of the native backend does not
    apply.
    """

    def __init__(
        self,
        *,
        allow_hosted: bool = False,
        endpoint: str = DEEPGRAM_ENDPOINT,
        model: str = "nova-3",
        diarize_model: str = "latest",
        window_ms: int = MAX_AUDIO_MS,
        timeout_seconds: float = 30,
        cancelled: Callable[[], bool] | None = None,
        load_key: Callable[[], str] = load_diarizer_api_key,
    ) -> None:
        self.endpoint = _https_endpoint(endpoint, "Hosted diarizer")
        self.model = _name(model, "hosted diarizer model")
        if diarize_model not in DEEPGRAM_DIARIZE_MODELS:
            raise LiveAudioError("Invalid hosted diarizer version")
        self.diarize_model = diarize_model
        if type(window_ms) is not int or not 1000 <= window_ms <= MAX_AUDIO_MS:
            raise LiveAudioError("Invalid hosted diarizer window")
        self.window_bytes = window_ms * BYTES_PER_MS
        self._client = _HostedClient(
            "Hosted diarizer",
            allow_hosted=allow_hosted,
            timeout_seconds=timeout_seconds,
            cancelled=cancelled,
            load_key=load_key,
        )
        self.pushed_bytes = 0
        self.closed = False
        self._buffer = bytearray()

    def push(self, pcm: bytes) -> None:
        if self.closed:
            raise HostedSpeechError("Hosted diarizer is closed")
        if not isinstance(pcm, bytes) or len(pcm) % 2 or len(pcm) > self.window_bytes:
            raise HostedSpeechError("Invalid audio frame")
        self._buffer.extend(pcm)
        self.pushed_bytes += len(pcm)
        if len(self._buffer) > self.window_bytes:
            del self._buffer[: len(self._buffer) - self.window_bytes]

    def segments(self) -> list[dict[str, Any]]:
        if self.closed:
            raise HostedSpeechError("Hosted diarizer is closed")
        if not self._buffer:
            return []
        window = bytes(self._buffer)
        query = urllib.parse.urlencode({"model": self.model, "diarize_model": self.diarize_model})
        document = self._client.post(
            self.endpoint + "?" + query, _wav(window), {"Content-Type": "audio/wav"}, "Token "
        )
        offset_ms = (self.pushed_bytes - len(window)) // BYTES_PER_MS
        return deepgram_timeline(document, len(window) // BYTES_PER_MS, offset_ms)

    def finish(self) -> list[dict[str, Any]]:
        return self.segments()

    def close(self) -> None:
        self.closed = True
        self._buffer.clear()


def _spec(value: Any, kinds: tuple[str, ...], allowed: dict[str, set[str]]) -> dict[str, Any]:
    if value is None:
        return {"kind": kinds[0]}
    if not isinstance(value, dict) or value.get("kind") not in kinds:
        raise LiveAudioError("Unknown speech backend kind")
    if set(value) - allowed[value["kind"]] - {"kind"}:
        raise LiveAudioError("Unknown speech backend setting")
    return dict(value)


def transcriber_spec(value: Any) -> dict[str, Any]:
    """Validate a `transcriber` configuration section; None selects whisper.cpp."""
    spec = _spec(
        value,
        TRANSCRIBER_KINDS,
        {
            "whisper.cpp": set(),
            "hosted-openai-compatible": {"endpoint", "model", "language", "timeout_seconds"},
        },
    )
    if spec["kind"] == "hosted-openai-compatible":
        spec["endpoint"] = _https_endpoint(spec.get("endpoint"), "Hosted transcriber")
        spec["model"] = _name(spec.get("model"), "hosted transcriber model")
        spec["timeout_seconds"] = _timeout(spec.get("timeout_seconds", 30))
        language = spec.get("language")
        if language is not None and (
            not isinstance(language, str) or not _LANGUAGE.fullmatch(language)
        ):
            raise LiveAudioError("Invalid hosted transcriber language")
        spec["language"] = language
    return spec


def diarizer_spec(value: Any) -> dict[str, Any]:
    """Validate a `diarizer` configuration section; None selects the native Nemotron stream."""
    spec = _spec(
        value,
        DIARIZER_KINDS,
        {
            "nemotron.cpp": set(),
            "hosted-deepgram": {"endpoint", "model", "diarize_model", "timeout_seconds"},
        },
    )
    if spec["kind"] == "hosted-deepgram":
        spec["endpoint"] = _https_endpoint(
            spec.get("endpoint", DEEPGRAM_ENDPOINT), "Hosted diarizer"
        )
        spec["model"] = _name(spec.get("model", "nova-3"), "hosted diarizer model")
        if spec.get("diarize_model", "latest") not in DEEPGRAM_DIARIZE_MODELS:
            raise LiveAudioError("Invalid hosted diarizer version")
        spec["diarize_model"] = spec.get("diarize_model", "latest")
        spec["timeout_seconds"] = _timeout(spec.get("timeout_seconds", 30))
    return spec


def is_hosted(spec: dict[str, Any]) -> bool:
    return spec["kind"].startswith("hosted-")


def transcriber_factory(
    spec: dict[str, Any], *, allow_hosted: bool = False
) -> Callable[[LiveConfig], Transcriber] | None:
    """A `LiveConfig.transcriber` value for a validated spec; None keeps the local default."""
    spec = transcriber_spec(spec)
    if spec["kind"] == "whisper.cpp":
        return None

    def build(config: LiveConfig) -> Transcriber:
        return OpenAICompatibleTranscriber(
            endpoint=spec["endpoint"],
            model=spec["model"],
            allow_hosted=allow_hosted,
            language=spec["language"],
            timeout_seconds=spec["timeout_seconds"],
            cancelled=config.cancelled,
        )

    return build


def diarizer_factory(
    spec: dict[str, Any], *, allow_hosted: bool = False
) -> Callable[[LiveConfig], Diarizer] | None:
    """A `LiveConfig.diarizer` value for a validated spec; None keeps the local default."""
    spec = diarizer_spec(spec)
    if spec["kind"] == "nemotron.cpp":
        return None

    def build(config: LiveConfig) -> Diarizer:
        return DeepgramDiarizer(
            allow_hosted=allow_hosted,
            endpoint=spec["endpoint"],
            model=spec["model"],
            diarize_model=spec["diarize_model"],
            window_ms=config.max_utterance_ms + 1000,
            timeout_seconds=spec["timeout_seconds"],
            cancelled=config.cancelled,
        )

    return build
