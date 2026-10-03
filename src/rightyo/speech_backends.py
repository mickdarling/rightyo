"""Configuration-selected speech backends, including opt-in hosted ones.

Hosted backends send audio only after explicit `allow_hosted` consent, use stdlib
HTTP with no environment proxies, refuse redirects, bound response bytes and raise
sanitized errors without audio, transcript, URL, or credential content. Endpoint
schemas were read from the official sources on 2026-10-02 and are cited on each
class. Nothing here is an accuracy claim for any hosted model.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import math
import re
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any, Callable

from rightyo.credentials import CredentialError, load_diarizer_api_key, load_transcriber_api_key
from rightyo.live_audio import (
    BYTES_PER_MS,
    SAMPLE_RATE,
    LiveAudioError,
    LiveConfig,
    NemotronCppDiarizer,
    WhisperCppTranscriber,
)
from rightyo.providers import Diarizer, Transcriber

MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_AUDIO_MS = 16000  # the largest utterance window plus the diarizer's margin
DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"
DEEPGRAM_DIARIZE_MODELS = ("latest", "v1", "v2")
TRANSCRIBER_KINDS = ("whisper.cpp", "hosted-openai-compatible")
DIARIZER_KINDS = ("nemotron.cpp", "hosted-deepgram")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}")
_LANGUAGE = re.compile(r"[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})?")
_UNSAFE = re.compile(r"[^A-Za-z0-9_. -]")
MERGE_GAP_MS = 300  # same-speaker words further apart than this stay separate segments
READ_POLL_SECONDS = 0.05  # how often the waiting loop re-checks cancel/deadline
READ_ABORT_JOIN_SECONDS = 2  # how long an aborted reader is given to notice the shutdown
MIN_TRANSPORT_SECONDS = 0.1  # the least a nearly exhausted budget gives the transport


class HostedSpeechError(LiveAudioError):
    """Sanitized hosted-backend failure: no audio, transcript, URL or credential content."""


class _ReadDeadline(Exception):
    """The wall-clock deadline passed while reading a response body."""


class _ReadCancelled(Exception):
    """The owner cancelled while a response body was being read."""


class _ReadFailed(Exception):
    """The body read ended without content; details stay on the reader thread."""


def provenance_id(prefix: str, *parts: str) -> str:
    """A `contracts.identifier`-safe label naming the backend family and its model.

    Characters outside the identifier charset become `-`, and a 12-hex SHA-256 prefix of
    the original names is appended so sanitized or truncated names cannot collide; the
    whole stays within 96 characters. Only the configured model/version names are
    included, never an endpoint or key.
    """
    digest = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:12]
    safe = " ".join(_UNSAFE.sub("-", part) for part in parts)
    budget = 96 - len(prefix) - len(digest) - 2
    return f"{prefix} {safe[:budget].strip()} {digest}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any):
        # Never forward a credential to a redirected endpoint.
        fp.close()
        raise HostedSpeechError("Hosted speech redirect refused")


def _https_endpoint(value: Any, label: str) -> str:
    parts = None
    # A bare trailing `?` or `#` parses as an empty query/fragment yet would still
    # corrupt the appended query, so the delimiters themselves are refused.
    if (
        isinstance(value, str)
        and len(value) <= 2048
        and value.isascii()
        and not set(value) & set("?#")
    ):
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
        or parts.query
        or parts.fragment
    ):
        # Query and fragment are refused: the backend appends its own query string.
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


def _abort_response(response: Any) -> None:
    """End a body read in progress by shutting the connection, then closing the response.

    urllib's response wraps `http.client.HTTPResponse` -> `BufferedReader` (`fp`) ->
    `SocketIO` (`raw`) -> socket (`_sock`); shutting the first layer with `shutdown`
    makes a blocked `recv` return, so the reader thread ends without a step timeout.
    """
    layer = response
    for _ in range(6):
        if callable(getattr(layer, "shutdown", None)):
            try:
                layer.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            break
        following = None
        for name in ("raw", "fp", "_sock"):
            candidate = getattr(layer, name, None)
            if candidate is not None:
                following = candidate
                break
        if following is None:
            break
        layer = following
    try:
        response.close()
    except OSError:
        pass


class _HostedClient:
    """One consented endpoint: bounded, sanitized, redirect-free, credential at use only."""

    def __init__(
        self,
        label: str,
        *,
        allow_hosted: bool,
        timeout_seconds: float,
        cancelled: Callable[[], bool] | None,
        load_key: Callable[..., str],
    ) -> None:
        if allow_hosted is not True:
            raise HostedSpeechError(f"{label} requires explicit hosted consent to send audio")
        self.label = label
        self.timeout = _timeout(timeout_seconds)
        self.cancelled = cancelled if cancelled is not None else (lambda: False)
        self.load_key = load_key
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _read_body(self, response: Any, deadline: float) -> bytes:
        """Read at most the byte cap, ending early on cancellation or the deadline.

        A helper thread performs the blocking read with the transport's own inactivity
        timeout, so a pause shorter than the remaining budget is tolerated and chunked
        framing is never interrupted; the waiting loop polls the cancellation guard and
        the wall-clock deadline and, on either, shuts the connection so the read returns.
        """
        outcome: dict[str, Any] = {}

        def read() -> None:
            try:
                outcome["content"] = response.read(MAX_RESPONSE_BYTES + 1)
            except BaseException:
                # Any transport detail stays on this thread; the caller reports a
                # sanitized failure.
                outcome["failed"] = True

        thread = threading.Thread(target=read, name="rightyo-hosted-read", daemon=True)
        thread.start()
        while thread.is_alive():
            if self.cancelled():
                _abort_response(response)
                thread.join(READ_ABORT_JOIN_SECONDS)
                raise _ReadCancelled
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _abort_response(response)
                thread.join(READ_ABORT_JOIN_SECONDS)
                raise _ReadDeadline
            thread.join(min(READ_POLL_SECONDS, remaining))
        if "content" not in outcome:
            # The transport's own inactivity timeout equals the remaining budget, so a
            # read that dies at the deadline is reported as the deadline, not a failure.
            if time.monotonic() >= deadline:
                raise _ReadDeadline
            raise _ReadFailed
        return outcome["content"]

    def post(self, url: str, body: bytes, headers: dict[str, str], scheme: str) -> dict[str, Any]:
        if self.cancelled():
            raise HostedSpeechError(f"{self.label} request was cancelled")
        # One wall-clock deadline covers the credential lookup and the whole exchange.
        deadline = time.monotonic() + self.timeout
        key = None
        try:
            key = self.load_key(cancelled=self.cancelled, timeout_seconds=self.timeout)
        except CredentialError:
            pass
        if key is None:
            if self.cancelled():
                raise HostedSpeechError(f"{self.label} request was cancelled")
            raise HostedSpeechError(f"{self.label} credential is unavailable")
        if self.cancelled():
            del key
            raise HostedSpeechError(f"{self.label} request was cancelled")
        # Only the budget the credential lookup left is given to the transport, so the
        # whole operation stays within one `timeout_seconds` of its start.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            del key
            raise HostedSpeechError(f"{self.label} request exceeded its deadline")
        request = urllib.request.Request(
            url, data=body, method="POST", headers={**headers, "Authorization": scheme + key}
        )
        del key
        failure = None
        content = b""
        # urllib's timeout is per operation, so a slowly dripping body could outlive it;
        # the body is read on a helper thread that the deadline or cancellation aborts.
        try:
            with self._opener.open(
                request, timeout=max(remaining, MIN_TRANSPORT_SECONDS)
            ) as response:
                content = self._read_body(response, deadline)
        except _ReadDeadline:
            failure = f"{self.label} response exceeded the request deadline"
        except _ReadCancelled:
            failure = f"{self.label} request was cancelled"
        except _ReadFailed:
            failure = f"{self.label} connection failed or timed out"
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


def _monotonic(units: list[dict[str, Any]]) -> bool:
    """Units must not run backwards: starts and ends both non-decreasing in order.

    Reordered units would otherwise make a turn's start exclude earlier words or emit
    turns in decreasing time order; slight overlaps between neighbours are tolerated.
    """
    return all(
        later["start_ms"] >= earlier["start_ms"] and later["end_ms"] >= earlier["end_ms"]
        for earlier, later in zip(units, units[1:])
    )


def openai_units(document: dict[str, Any], duration_ms: int) -> list[dict[str, Any]]:
    """Utterance-relative units from a verbose_json transcription object."""
    label = "Hosted transcriber"
    text = document.get("text")
    if not isinstance(text, str) or len(text) > 4000:
        raise HostedSpeechError(f"{label} returned an invalid response")
    words, segments = document.get("words"), document.get("segments")
    if not text.strip():
        # Silence is only silence when nothing was timed: every timed field that is
        # present must be a list whose units, if any, carry blank text; anything else
        # contradicts the empty transcript and must not be dropped as if unsaid.
        for name, field in (("words", "word"), ("segments", "text")):
            if name not in document:
                continue
            value = document[name]
            if not isinstance(value, list) or not all(
                isinstance(unit, dict)
                and isinstance(unit.get(field), str)
                and not unit[field].strip()
                for unit in value
            ):
                raise HostedSpeechError(f"{label} returned inconsistent word timing")
        return []
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
        if not _monotonic(units):
            raise HostedSpeechError(f"{label} returned inconsistent word timing")
        punctuated = _punctuate([unit["text"].strip() for unit in units], text)
        if punctuated is not None:
            for unit, value in zip(units, punctuated):
                unit["text"] = " " + value
            return units
        # Words that do not cover `text` in order would truncate or reorder the
        # transcript; use the complete segments instead, or fail closed.
        if not isinstance(document.get("segments"), list) or not document.get("segments"):
            raise HostedSpeechError(f"{label} returned inconsistent word timing")
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
        # Segments must reproduce the transcript in time order; otherwise the response
        # is inconsistent.
        if " ".join("".join(unit["text"] for unit in units).split()) != " ".join(
            text.split()
        ) or not _monotonic(units):
            raise HostedSpeechError(f"{label} returned inconsistent segments")
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

    recognizer_id = "hosted-openai-compatible"  # instances append the configured model

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        allow_hosted: bool = False,
        language: str | None = None,
        timeout_seconds: float = 30,
        cancelled: Callable[[], bool] | None = None,
        load_key: Callable[..., str] = load_transcriber_api_key,
    ) -> None:
        self.endpoint = _https_endpoint(endpoint, "Hosted transcriber")
        self.model = _name(model, "hosted transcriber model")
        self.recognizer_id = transcriber_id({"kind": "hosted-openai-compatible", "model": model})
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
    # Evidence that the diarizer ran: per developers.deepgram.com/docs/diarization (read
    # 2026-10-03) `metadata.diarize_info` "is either present with both fields
    # [`model_uuid`, `arch`] or absent entirely", and "An absent block on a request that
    # asked for diarization means the diarizer did not run." A transcription without it
    # must not pass as valid unknown-speaker output.
    metadata = document.get("metadata")
    info = metadata.get("diarize_info") if isinstance(metadata, dict) else None
    if (
        not isinstance(info, dict)
        or not isinstance(info.get("model_uuid"), str)
        or not isinstance(info.get("arch"), str)
    ):
        raise HostedSpeechError(f"{label} diarization unavailable")
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
        # Only words that touch or nearly touch are joined; a longer gap between two
        # words of one speaker stays uncovered so a unit inside it remains unknown.
        if (
            previous is not None
            and previous["speaker"] == speaker + 1
            and start >= previous["start_ms"]
            and start - previous["end_ms"] <= MERGE_GAP_MS
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
    `diarize` parameter is deprecated. Use `diarize_model` instead"). Participation in
    Deepgram's Model Improvement Program "is the default"; per
    developers.deepgram.com/docs/the-deepgram-model-improvement-partnership-program and
    developers.deepgram.com/trust-security/your-data (read 2026-10-02), "Add
    `mip_opt_out=true` as a query parameter of all API requests that you want to be
    excluded from the Model Improvement Program", and "Opting out gives you zero data
    retention". Every request here carries `mip_opt_out=true`. The response's
    `results.channels[0].alternatives[0].words[]` carries `word`, `start`, `end`
    (seconds), `speaker` (zero-based integer) and `speaker_confidence`.

    Labels are assigned per request. Each `segments`/`finish` call sends only the
    trailing `window_ms` of pushed audio, sized to cover one utterance, and returns
    that window's timeline in stream milliseconds with consecutive words of one
    speaker at most `MERGE_GAP_MS` apart merged into a segment (a wider gap stays
    uncovered, so a unit inside it is unknown). "Speaker A" in one utterance is therefore not
    known to be the same person as "Speaker A" in the next: the service documents no
    cross-request label stability and none is invented here, so turns carry
    `speaker_provenance="diarization-utterance"`. Audio outside the window is not
    retained, and the whole-session timeline cap of the native backend does not apply.
    """

    speaker_provenance = "diarization-utterance"

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
        load_key: Callable[..., str] = load_diarizer_api_key,
    ) -> None:
        self.endpoint = _https_endpoint(endpoint, "Hosted diarizer")
        self.model = _name(model, "hosted diarizer model")
        if diarize_model not in DEEPGRAM_DIARIZE_MODELS:
            raise LiveAudioError("Invalid hosted diarizer version")
        self.diarize_model = diarize_model
        # `speaker_provenance` stays the allowlisted contract value; this names the backend.
        self.diarizer_id = diarizer_id(
            {"kind": "hosted-deepgram", "model": model, "diarize_model": diarize_model}
        )
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
        # Every request opts out of Deepgram's Model Improvement Program (see the class
        # docstring): consented audio is for this session's labels only.
        query = urllib.parse.urlencode(
            {"model": self.model, "diarize_model": self.diarize_model, "mip_opt_out": "true"}
        )
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


def utterance_local_labels(spec: dict[str, Any]) -> bool:
    """Whether the selected diarizer's labels hold only within one utterance.

    Such labels are namespaced per utterance (`u7 Speaker A`), so a configured role for
    `Speaker A` could never match one; only a session-stable diarizer can carry roles.
    """
    implementation = _DIARIZERS.get(spec["kind"])
    provenance = getattr(implementation, "speaker_provenance", "diarization-timeline")
    return provenance == "diarization-utterance"


def transcriber_id(spec: dict[str, Any]) -> str:
    """The `recognizer_id` the selected transcriber stamps on turns (display-safe)."""
    if spec["kind"] == "hosted-openai-compatible":
        return provenance_id("hosted-openai-compatible", _name(spec["model"], "model"))
    return WhisperCppTranscriber.recognizer_id


def diarizer_id(spec: dict[str, Any]) -> str:
    """A display-safe id naming the selected diarizer and its version.

    The hosted id names the configured model and diarizer version; Deepgram's response
    metadata may name a more specific resolved model, which is not surfaced here because
    the id is advertised when the session starts, before any request.
    """
    if spec["kind"] == "hosted-deepgram":
        model = _name(spec["model"], "model")
        version = spec["diarize_model"]
        if version not in DEEPGRAM_DIARIZE_MODELS:
            raise LiveAudioError("Invalid hosted diarizer version")
        return provenance_id("hosted-deepgram", model, version)
    return NemotronCppDiarizer.diarizer_id


def speech_summary(transcriber: dict[str, Any], diarizer: dict[str, Any]) -> dict[str, Any]:
    """The `speech` object advertised on a live session's `started` event.

    Kinds and ids only: never an endpoint, local path, model file or credential.
    """
    transcriber, diarizer = transcriber_spec(transcriber), diarizer_spec(diarizer)
    return {
        "transcriber": {"kind": transcriber["kind"], "id": transcriber_id(transcriber)},
        "diarizer": {"kind": diarizer["kind"], "id": diarizer_id(diarizer)},
    }


_DIARIZERS = {"nemotron.cpp": NemotronCppDiarizer, "hosted-deepgram": DeepgramDiarizer}
SERVICE_NAMES = {
    "whisper.cpp": "Whisper (configured model)",
    "nemotron.cpp": "Nemotron 3 (configured GGUF)",
    "hosted-openai-compatible": "OpenAI-compatible hosted",
    "hosted-deepgram": "Deepgram hosted",
}


def describe(spec: dict[str, Any]) -> dict[str, Any]:
    """A display-safe summary of a validated spec: kind, hosted flag and service name.

    Never includes the endpoint, model name, language or any credential, so it can be
    shown to a page or written to a snapshot.
    """
    return {
        "kind": spec["kind"],
        "hosted": is_hosted(spec),
        "service": SERVICE_NAMES[spec["kind"]],
        # True only for a diarizer whose labels hold within one utterance; the page
        # words its legend from this.
        "utterance_local": utterance_local_labels(spec),
    }


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
