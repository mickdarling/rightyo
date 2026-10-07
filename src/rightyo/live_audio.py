"""Opt-in causal mono 16 kHz PCM16 processing with bounded utterance audio.

Nemotron's C ABI is pinned to NeMo-Speech.cpp 0f706e43. A private subprocess
owns one model/stream for the entire session, preserving speaker channel IDs.
Whisper CLI token offsets are from whisper.cpp v1.9.4. No models, microphone,
hosted processing, recording archive, or subprocess logs are provisioned here.
Energy endpointing and token timestamps are experimental, not calibrated VAD
or forced alignment. Ambiguous speaker coverage remains unknown.
"""

from __future__ import annotations

import array
import base64
import contextlib
import ctypes
import inspect
import json
import math
import os
import selectors
import subprocess
import sys
import tempfile
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .contracts import PROVENANCE, Turn, identifier, utterance_scoped_speaker
from .providers import Diarizer, Transcriber

SAMPLE_RATE = 16000
FRAME_BYTES = 640  # 20 ms of mono signed little-endian PCM16
BYTES_PER_MS = 32  # mono PCM16 at 16 kHz
MAX_CHUNK_BYTES = 32000  # one second of audio per push
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_SPEAKER = 702  # "Speaker A" .. "Speaker ZZ"; native streams use 1..8
# The native stream returns only segments ending within this trailing window of pushed
# audio. An utterance spans at most `max_utterance_ms` (<= 15 s, pre-roll included), so
# every segment that can overlap one is inside it with a wide margin, and per-utterance
# transfer and attribution stay constant however long the session runs (#54).
TIMELINE_WINDOW_MS = 60000
# Per-response bound for any backend: a 60 s window holds 6,000 native 10 ms frames.
MAX_TIMELINE_SEGMENTS = 18000


class LiveAudioError(ValueError):
    """Sanitized failure: never embeds paths, PCM, transcripts, or native logs."""


def _is_backend_instance(value: Any, method: str) -> bool:
    # A class exposes the method too, but as an unbound function: treat it as a factory.
    return hasattr(value, method) and not inspect.isclass(value)


def _check_backend(value: Any, method: str, label: str) -> None:
    """Accept None, an instance exposing `method`, or a factory (class or callable)."""
    if not (value is None or _is_backend_instance(value, method) or callable(value)):
        raise LiveAudioError(f"Invalid {label} backend")


@dataclass(frozen=True)
class LiveConfig:
    session_id: str
    # Local runtime assets; required only by the default local backends below.
    whisper_executable: str | Path | None = None
    whisper_model: str | Path | None = None
    diarization_library: str | Path | None = None
    diarization_model: str | Path | None = None
    provenance: str = "live-microphone"
    energy_threshold: float = 0.008
    hangover_ms: int = 1440
    pre_roll_ms: int = 240
    max_utterance_ms: int = 12000
    timeout_seconds: float = 30
    cancelled: Callable[[], bool] | None = None
    # Optional content-free stderr diagnostics (counts only): skipped segments and
    # suppressed utterances.
    report: Callable[[str], None] | None = None
    # Total audio accepted per session, in stream milliseconds. None means no ceiling:
    # memory stays bounded by the utterance window and downstream retention limits.
    session_budget_ms: int | None = None
    # A `Transcriber`/`Diarizer` instance, or a factory called with this config. None
    # selects the local whisper.cpp recognizer and Nemotron native stream.
    transcriber: Transcriber | Callable[[LiveConfig], Transcriber] | None = None
    diarizer: Diarizer | Callable[[LiveConfig], Diarizer] | None = None

    def __post_init__(self) -> None:
        identifier(self.session_id, "session_id")
        _check_backend(self.transcriber, "transcribe", "transcriber")
        _check_backend(self.diarizer, "push", "diarizer")
        if self.session_budget_ms is not None and (
            type(self.session_budget_ms) is not int or self.session_budget_ms < 1
        ):
            raise LiveAudioError("Invalid session budget")
        if self.cancelled is not None and not callable(self.cancelled):
            raise LiveAudioError("Invalid cancellation guard")
        if self.report is not None and not callable(self.report):
            raise LiveAudioError("Invalid diagnostic reporter")
        if self.provenance not in PROVENANCE:
            raise LiveAudioError("Invalid audio provenance")
        if (
            type(self.energy_threshold) not in (int, float)
            or not math.isfinite(self.energy_threshold)
            or not 0 < self.energy_threshold < 1
        ):
            raise LiveAudioError("Invalid energy threshold")
        for value, minimum, maximum in (
            (self.hangover_ms, 1440, 3000),
            (self.pre_roll_ms, 20, 1000),
            (self.max_utterance_ms, 4000, 15000),
        ):
            if type(value) is not int or not minimum <= value <= maximum or value % 20:
                raise LiveAudioError("Invalid utterance window")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 120
        ):
            raise LiveAudioError("Invalid local processing timeout")
        assets = (
            self.whisper_executable,
            self.whisper_model,
            self.diarization_library,
            self.diarization_model,
        )
        required = assets[:2] if self.transcriber is None else ()
        required += assets[2:] if self.diarizer is None else ()
        if any(value is None for value in required) or any(
            value is not None and not Path(value).is_file() for value in assets
        ):
            raise LiveAudioError("Explicit existing runtimes and models are required")


def _pcm_samples(pcm: bytes) -> array.array:
    values = array.array("h")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    return values


class _ModelConfig(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("gpu", ctypes.c_int32),
        ("preset", ctypes.c_char_p),
        ("chunk_frames", ctypes.c_int32),
        ("right_context_frames", ctypes.c_int32),
        ("left_context_frames", ctypes.c_int32),
        ("fifo_frames", ctypes.c_int32),
        ("spkcache_frames", ctypes.c_int32),
        ("update_period_frames", ctypes.c_int32),
    ]


class _Segment(ctypes.Structure):
    _fields_ = [
        ("start_time", ctypes.c_double),
        ("end_time", ctypes.c_double),
        ("speaker", ctypes.c_int32),
    ]


class _NativeStream:
    """Used in the isolated child only; never call it concurrently."""

    def __init__(self, library: str, model_path: str):
        self.lib = ctypes.CDLL(library)
        self.model = ctypes.c_void_p()
        self.stream = ctypes.c_void_p()
        self.pushed_bytes = 0
        signatures = {
            "create": (
                [ctypes.POINTER(_ModelConfig), ctypes.POINTER(ctypes.c_void_p)],
                ctypes.c_int32,
            ),
            "destroy": ([ctypes.c_void_p], None),
            "stream_open": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_int32),
            "stream_close": ([ctypes.c_void_p], None),
            "stream_push_f32": (
                [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32],
                ctypes.c_int32,
            ),
            "stream_finish": ([ctypes.c_void_p], ctypes.c_int32),
            "segments": (
                [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.POINTER(_Segment),
                    ctypes.c_size_t,
                    ctypes.POINTER(ctypes.c_size_t),
                ],
                ctypes.c_int32,
            ),
            "frame_count": ([ctypes.c_void_p], ctypes.c_int64),
            "seconds_per_frame": ([ctypes.c_void_p], ctypes.c_double),
            "num_speakers": ([ctypes.c_void_p], ctypes.c_int32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, "nemo_speech_diar_" + name)
            function.argtypes, function.restype = arguments, result
        config = _ModelConfig(
            ctypes.sizeof(_ModelConfig),
            os.fsencode(model_path),
            0,
            b"v3-streaming",
            0,
            0,
            -1,
            0,
            0,
            0,
        )
        try:
            self.check(
                self.lib.nemo_speech_diar_create(ctypes.byref(config), ctypes.byref(self.model))
            )
            self.check(self.lib.nemo_speech_diar_stream_open(self.model, ctypes.byref(self.stream)))
            if self.lib.nemo_speech_diar_num_speakers(self.model) != 8:
                raise LiveAudioError("Unexpected local diarizer capacity")
            cadence = self.lib.nemo_speech_diar_seconds_per_frame(self.model)
            if not math.isfinite(cadence) or not 0.009 <= cadence <= 0.011:
                raise LiveAudioError("Unexpected local diarizer cadence")
        except BaseException:
            self.close()
            raise

    @staticmethod
    def check(status: int) -> None:
        if status != 0:
            raise LiveAudioError("Local diarizer failed")

    def push(self, pcm: bytes) -> None:
        values = _pcm_samples(pcm)
        floats = (ctypes.c_float * len(values))(*(value / 32768 for value in values))
        self.check(
            self.lib.nemo_speech_diar_stream_push_f32(self.stream, floats, len(values), SAMPLE_RATE)
        )
        self.pushed_bytes += len(pcm)

    def finish(self) -> None:
        self.check(self.lib.nemo_speech_diar_stream_finish(self.stream))

    def segments(self) -> list[dict[str, Any]]:
        count = ctypes.c_size_t()
        fn = self.lib.nemo_speech_diar_segments
        self.check(fn(self.stream, None, None, 0, ctypes.byref(count)))
        output = (_Segment * count.value)()
        self.check(fn(self.stream, None, output, count.value, ctypes.byref(count)))
        # Segment times must lie within audio actually pushed (plus one second of
        # native lookahead/rounding); the bound follows the stream, not a fixed ceiling.
        horizon = self.pushed_bytes / (BYTES_PER_MS * 1000) + 1
        cutoff_ms = self.pushed_bytes // BYTES_PER_MS - TIMELINE_WINDOW_MS
        result, latest = [], None
        for segment in output[: count.value]:
            start, end = segment.start_time, segment.end_time
            if (
                not math.isfinite(start)
                or not math.isfinite(end)
                or not 0 <= start < end <= horizon
                or not 1 <= segment.speaker <= 8
            ):
                raise LiveAudioError("Invalid local diarizer result")
            item = {
                "start_ms": round(start * 1000),
                "end_ms": round(end * 1000),
                "speaker": segment.speaker,
            }
            if item["end_ms"] > cutoff_ms:
                result.append(item)
            elif latest is None or item["end_ms"] > latest["end_ms"]:
                latest = item
        # Older segments cannot overlap any pending utterance and are not returned. When
        # none is recent, the latest one is kept so that a non-empty session timeline
        # stays non-empty (it decides `speaker_provenance`); it overlaps no utterance.
        if not result and latest is not None:
            result.append(latest)
        if len(result) > MAX_TIMELINE_SEGMENTS:
            raise LiveAudioError("Invalid local diarizer result")
        return result

    def close(self) -> None:
        if self.stream:
            self.lib.nemo_speech_diar_stream_close(self.stream)
            self.stream = ctypes.c_void_p()
        if self.model:
            self.lib.nemo_speech_diar_destroy(self.model)
            self.model = ctypes.c_void_p()


def _native_worker(library: str, model: str) -> int:
    # Preserve a protocol-only descriptor before redirecting every native log.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    native = None
    try:
        native = _NativeStream(library, model)
        protocol.write('{"ok":true}\n')
        while True:
            line = sys.stdin.buffer.readline(64000)
            if not line or not line.endswith(b"\n"):
                break
            request = json.loads(line)
            command = request.get("command")
            if command == "push":
                pcm = base64.b64decode(request["pcm"], validate=True)
                if len(pcm) > MAX_CHUNK_BYTES or len(pcm) % 2:
                    raise LiveAudioError("Invalid audio frame")
                native.push(pcm)
                response = {"ok": True}
            elif command == "segments":
                response = {"ok": True, "segments": native.segments()}
            elif command == "finish":
                native.finish()
                response = {"ok": True, "segments": native.segments()}
            elif command == "close":
                break
            else:
                raise LiveAudioError("Invalid native command")
            protocol.write(json.dumps(response, separators=(",", ":")) + "\n")
        return 0
    except Exception:
        protocol.write('{"ok":false}\n')
        return 1
    finally:
        if native is not None:
            native.close()
        protocol.close()


class NemotronCppDiarizer:
    """The default `Diarizer`: one persistent native stream in a private subprocess.

    Speaker channels are arrival-ordered and stable for the whole session. Each
    `segments`/`finish` call returns only the trailing `TIMELINE_WINDOW_MS` of the
    timeline, so there is no session-length segment limit.
    """

    diarizer_id = "nemotron.cpp v3-streaming"

    def __init__(self, config: LiveConfig):
        if config.diarization_library is None or config.diarization_model is None:
            raise LiveAudioError("Explicit existing runtimes and models are required")
        self.timeout = config.timeout_seconds
        self.cancelled = config.cancelled if config.cancelled is not None else (lambda: False)
        self._check_cancelled()
        source = str(Path(__file__).resolve().parent.parent)
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "rightyo.live_audio",
                "--native-worker",
                str(Path(config.diarization_library).resolve()),
                str(Path(config.diarization_model).resolve()),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
            env={"PATH": os.defpath, "PYTHONPATH": source},
        )
        self.buffer = bytearray()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _check_cancelled(self) -> None:
        if self.cancelled():
            raise LiveAudioError("Audio session was stopped")

    def _receive(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        assert self.process.stdout is not None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.buffer:
                self._check_cancelled()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LiveAudioError("Local diarizer timed out")
                if not selector.select(min(0.05, remaining)):
                    continue
                self._check_cancelled()
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    raise LiveAudioError("Local diarizer stopped unexpectedly")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_RESPONSE_BYTES:
                    raise LiveAudioError("Local diarizer exceeded output limit")
        self._check_cancelled()
        line, _, rest = self.buffer.partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            result = json.loads(line)
            if not isinstance(result, dict):
                raise ValueError
            if result.get("ok") is not True:
                raise ValueError
        except (ValueError, RecursionError):
            raise LiveAudioError("Local diarizer failed") from None
        return result

    def request(self, command: str, **fields: Any) -> dict[str, Any]:
        try:
            self._check_cancelled()
            assert self.process.stdin is not None
            self.process.stdin.write((json.dumps({"command": command, **fields}) + "\n").encode())
            return self._receive()
        except LiveAudioError:
            self.close()
            raise
        except (BrokenPipeError, OSError, ValueError):
            self.close()
            raise LiveAudioError("Local diarizer stopped unexpectedly") from None

    def push(self, pcm: bytes) -> None:
        self.request("push", pcm=base64.b64encode(pcm).decode("ascii"))

    def segments(self) -> list[dict[str, Any]]:
        return self.request("segments")["segments"]

    def finish(self) -> list[dict[str, Any]]:
        return self.request("finish")["segments"]

    def close(self) -> None:
        # No potentially blocking graceful RPC on cancellation; no transcript is finalized.
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream is not None:
                stream.close()
        self.buffer.clear()


# Compatibility name; the default selection below resolves it at call time.
_Diarizer = NemotronCppDiarizer


def _attribute(start: int, end: int, timeline: list[dict[str, Any]]) -> tuple[str | None, bool]:
    relevant = [s for s in timeline if s["start_ms"] < end and s["end_ms"] > start]
    edges = sorted(
        {start, end}
        | {max(start, min(end, s[k])) for s in relevant for k in ("start_ms", "end_ms")}
    )
    speakers = set()
    ambiguous = overlap = False
    for left, right in zip(edges, edges[1:]):
        active = {s["speaker"] for s in relevant if s["start_ms"] < right and s["end_ms"] > left}
        overlap |= len(active) > 1
        ambiguous |= len(active) != 1
        speakers.update(active)
    if ambiguous or len(speakers) != 1:
        return None, overlap
    # Native channels are arrival-ordered and persist for the whole session.
    return "Speaker " + _speaker_label(next(iter(speakers))), overlap


def _speaker_label(number: int) -> str:
    """Spreadsheet-column letters: 1..26 -> A..Z (unchanged), 27 -> AA, 52 -> AZ, 53 -> BA."""
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _check_timeline(timeline: Any, received_ms: int) -> None:
    """Enforce the `Diarizer` contract at the common boundary, whatever the backend.

    A list of at most `MAX_TIMELINE_SEGMENTS` dicts (one call's window, not the
    session), each with integer `0 <= start_ms <= end_ms` within the audio received so
    far plus one second of lookahead (the native stream's own bound) and a positive
    integer `speaker` no greater than `MAX_SPEAKER`. Order is not required: attribution
    intersects intervals and the native ABI promises none.
    """
    if not isinstance(timeline, list) or len(timeline) > MAX_TIMELINE_SEGMENTS:
        raise LiveAudioError("Invalid diarizer timeline")
    horizon = received_ms + 1000
    for segment in timeline:
        if not isinstance(segment, dict):
            raise LiveAudioError("Invalid diarizer timeline")
        start, end, speaker = (segment.get(k) for k in ("start_ms", "end_ms", "speaker"))
        if (
            type(start) is not int
            or type(end) is not int
            or type(speaker) is not int
            or not 0 <= start <= end <= horizon
            or not 1 <= speaker <= MAX_SPEAKER
        ):
            raise LiveAudioError("Invalid diarizer timeline")


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)
    except ProcessLookupError:
        pass


def _transcribe(
    config: LiveConfig,
    pcm: bytes,
    register: Callable[[subprocess.Popen], None] | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rightyo-live-asr-") as directory:
        audio = Path(directory) / "utterance.wav"
        with wave.open(str(audio), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(SAMPLE_RATE)
            target.writeframes(pcm)
        output = Path(directory) / "transcript"
        try:
            process = subprocess.Popen(
                [
                    str(Path(config.whisper_executable).resolve()),
                    "--model",
                    str(Path(config.whisper_model).resolve()),
                    "--file",
                    str(audio),
                    "--output-json-full",
                    "--output-file",
                    str(output),
                    "--no-prints",
                    "--max-context",
                    "0",
                    "--suppress-nst",
                ],
                cwd=directory,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env={"PATH": os.defpath, "HOME": directory, "TMPDIR": directory},
            )
            try:
                if register is not None:
                    register(process)
                if process.wait(timeout=config.timeout_seconds) != 0:
                    raise LiveAudioError("Local recognizer failed")
            finally:
                _terminate(process)
            with output.with_suffix(".json").open("rb") as source:
                raw = source.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise LiveAudioError("Local recognizer exceeded output limit")
            document = json.loads(raw)
            if not isinstance(document, dict):
                raise ValueError
            return document
        except subprocess.TimeoutExpired:
            raise LiveAudioError("Local recognizer timed out") from None
        except (OSError, subprocess.CalledProcessError, ValueError, RecursionError):
            raise LiveAudioError("Local recognizer failed") from None


def _units(document: dict[str, Any], duration_ms: int) -> list[dict[str, Any]] | None:
    """The usable units of one whisper.cpp document, or None; see `_counted_units`."""
    return _counted_units(document, duration_ms)[0]


def _counted_units(
    document: dict[str, Any], duration_ms: int
) -> tuple[list[dict[str, Any]] | None, int]:
    """Use complete token-derived word intervals; fall back to whole segments.

    No majority speaker vote, guessed timestamps, or splitting a word between
    speakers. Zero-duration punctuation attaches to its word; words with only
    zero-duration timestamps remain unassigned.

    whisper.cpp emits odd offsets on short, noisy or near-silent windows (#78). A
    zero-length segment with text inside the received audio (`start == end <=
    duration`) is kept as a zero-length unit, so its words stay in place. Any other
    segment whose own offsets are unusable (not integers, negative, reversed,
    starting after the received audio, or ending more than the CLI's one second of
    padding past it) makes the whole utterance unusable: publishing the remaining
    segments could invert meaning ("do not stop" becoming "do stop"). Returns
    `(None, n)` then, with `n` the number of unusable segments, so the caller
    suppresses the utterance and keeps listening; otherwise `(units, 0)`. A
    malformed document structure still fails closed.
    """
    segments = document.get("transcription")
    if not isinstance(segments, list) or len(segments) > 1000:
        raise LiveAudioError("Invalid local recognizer result")
    result = []
    skipped = 0
    for segment in segments:
        if not isinstance(segment, dict):
            raise LiveAudioError("Invalid local recognizer result")
        text, offsets = segment.get("text"), segment.get("offsets")
        if not isinstance(text, str) or len(text) > 4000 or not isinstance(offsets, dict):
            raise LiveAudioError("Invalid local recognizer result")
        if not text.strip():
            continue
        start, end = offsets.get("from"), offsets.get("to")
        if type(start) is int and type(end) is int and 0 <= start == end <= duration_ms:
            # In range but zero-length: keep the text in order as a zero-length unit.
            result.append({"text": text, "start_ms": start, "end_ms": end})
            continue
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= duration_ms + 1000
            or start >= duration_ms
        ):
            skipped += 1
            continue
        # The pinned CLI rounds final segment offsets into its padded audio.
        # Bound evidence to actually received PCM; never invent future frames.
        end = min(end, duration_ms)
        fallback = {"text": text, "start_ms": start, "end_ms": end}
        tokens = segment.get("tokens")
        if not isinstance(tokens, list) or len(tokens) > 4000:
            result.append(fallback)
            continue
        words: list[dict[str, Any]] = []
        reconstructed = ""
        usable = True
        for token in tokens:
            if not isinstance(token, dict) or not isinstance(token.get("text"), str):
                usable = False
                break
            value = token["text"]
            if (value.startswith("[_") and value.endswith("]")) or (
                value.startswith("<|") and value.endswith("|>")
            ):
                continue
            reconstructed += value
            interval = token.get("offsets", {})
            left, right = interval.get("from"), interval.get("to")
            if (
                type(left) is not int
                or type(right) is not int
                or not start <= left <= right <= end + 1000
            ):
                usable = False
                break
            left, right = min(left, end), min(right, end)
            if not words or (value[:1].isspace() and value.strip()):
                words.append({"text": value, "start_ms": left, "end_ms": right})
            else:
                words[-1]["text"] += value
                words[-1]["start_ms"] = min(words[-1]["start_ms"], left)
                words[-1]["end_ms"] = max(words[-1]["end_ms"], right)
        usable &= reconstructed.strip() == text.strip()
        usable &= bool(words) and all(w["start_ms"] <= w["end_ms"] for w in words)
        # Word intervals must also be in order; otherwise the common unit check would
        # suppress the whole utterance, so keep the segment's text as one unit (#78).
        usable &= all(
            a["start_ms"] <= b["start_ms"] and a["end_ms"] <= b["end_ms"]
            for a, b in zip(words, words[1:])
        )
        result.extend(words if usable else [fallback])
    return (None if skipped else result), skipped


class WhisperCppTranscriber:
    """The default `Transcriber`: the pinned whisper.cpp CLI on a temporary WAV file."""

    recognizer_id = "whisper.cpp-live-window"

    def __init__(self, config: LiveConfig):
        if config.whisper_executable is None or config.whisper_model is None:
            raise LiveAudioError("Explicit existing runtimes and models are required")
        self.config = config
        # Unusable segments, and the utterances they suppressed, this session (see
        # `_counted_units`); read by `LiveProcessor`.
        self.skipped_segments = 0
        self.suppressed_utterances = 0

    def transcribe(
        self, pcm: bytes, register: Callable[[subprocess.Popen], None] | None = None
    ) -> list[dict[str, Any]]:
        document = _transcribe(self.config, pcm, register)
        units, skipped = _counted_units(document, len(pcm) // BYTES_PER_MS)
        self.skipped_segments += skipped
        if units is None:
            # The whole utterance is unusable: publish nothing from it.
            self.suppressed_utterances += 1
            return []
        return units


def _checked_units(units: Any, duration_ms: int) -> list[dict[str, Any]] | None:
    """Enforce the `Transcriber` contract at the common boundary, whatever the backend.

    Units are dicts with `text` (str) and integer `start_ms`/`end_ms` with
    `0 <= start_ms <= end_ms <= duration_ms` (zero-length units are valid), and
    non-decreasing in order. A malformed result structure (not a list, too many units,
    a unit that is not a dict with string text) fails closed and ends the session.
    If any unit breaks the timestamp contract (not an integer, NaN, negative, reversed,
    past the utterance, or earlier than the unit before it), returns None: the caller
    suppresses the whole utterance and keeps listening (#78). Dropping only that unit
    could invert meaning ("do not stop" becoming "do stop"), and merging or sorting
    would attribute text to the wrong time span and so possibly the wrong speaker.
    Nothing is clamped, merged, reordered or partially kept.
    """
    if not isinstance(units, list) or len(units) > 4000:
        raise LiveAudioError("Invalid recognizer result")
    for unit in units:
        if not isinstance(unit, dict) or not isinstance(unit.get("text"), str):
            raise LiveAudioError("Invalid recognizer result")
    previous_start = previous_end = 0
    for unit in units:
        start, end = unit.get("start_ms"), unit.get("end_ms")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= duration_ms
            or start < previous_start
            or end < previous_end
        ):
            return None
        previous_start, previous_end = start, end
    return units


def _select(value: Any, default: Callable[[LiveConfig], Any], config: LiveConfig, method: str):
    if value is None:
        return default(config)
    return value if _is_backend_instance(value, method) else value(config)


class LiveProcessor:
    """Single-owner synchronous processor; feed only explicitly consented audio.

    `close` cancels and discards an unfinished utterance. `finish` flushes the
    finite causal input and emits its tail. Failures leave `failed=True` and
    cancel resources. The caller must stop capture immediately on failure.
    """

    def __init__(self, config: LiveConfig, on_turn: Callable[[Turn], None]):
        self.config = config
        self.on_turn = on_turn
        self.failed = False
        self.closed = False
        self._transcriber: Transcriber = _select(
            config.transcriber, WhisperCppTranscriber, config, "transcribe"
        )
        # Resolved through the module at call time so the default stays patchable.
        self._diarizer: Diarizer = _select(
            config.diarizer, lambda value: _Diarizer(value), config, "push"
        )
        self._partial = bytearray()
        self._pre_roll: deque[bytes] = deque(maxlen=config.pre_roll_ms // 20)
        self._utterance = bytearray()
        self._utterance_start = 0
        self._last_voice_ms = 0
        self._received_ms = 0
        self._counter = 0
        self._utterances = 0
        self._skipped_utterances = 0
        self._asr_process: subprocess.Popen | None = None

    @property
    def received_ms(self) -> int:
        return self._received_ms

    @property
    def skipped_segments(self) -> int:
        """whisper.cpp segments skipped this session for unusable offsets (see `_units`)."""
        inner = getattr(self._transcriber, "skipped_segments", 0)
        return inner if type(inner) is int and inner > 0 else 0

    @property
    def skipped_utterances(self) -> int:
        """Utterances suppressed whole this session for invalid or out-of-order timing.

        Counts the common unit check's suppressions plus a transcriber's own (whisper.cpp
        suppresses an utterance with an unusable segment before the unit check).
        """
        inner = getattr(self._transcriber, "suppressed_utterances", 0)
        inner = inner if type(inner) is int and inner > 0 else 0
        return self._skipped_utterances + inner

    @property
    def buffered_audio_bytes(self) -> int:
        return len(self._partial) + len(self._utterance) + sum(map(len, self._pre_roll))

    def push_pcm16(self, pcm: bytes) -> None:
        if self.closed or self.failed:
            raise LiveAudioError("Audio session is closed")
        try:
            if not isinstance(pcm, bytes) or len(pcm) > MAX_CHUNK_BYTES or len(pcm) % 2:
                raise LiveAudioError("Invalid mono PCM16 frame")
            budget = self.config.session_budget_ms
            if (
                budget is not None
                and self._received_ms + (len(self._partial) + len(pcm)) // BYTES_PER_MS > budget
            ):
                raise LiveAudioError("Audio session exceeded duration limit")
            self._partial.extend(pcm)
            while len(self._partial) >= FRAME_BYTES:
                frame = bytes(self._partial[:FRAME_BYTES])
                del self._partial[:FRAME_BYTES]
                self._frame(frame)
        except Exception:
            self.failed = True
            self.close()
            raise

    def _frame(self, frame: bytes) -> None:
        self._diarizer.push(frame)
        frame_start = self._received_ms
        self._received_ms += 20
        values = _pcm_samples(frame)
        rms = math.sqrt(sum(value * value for value in values) / len(values)) / 32768
        voiced = rms >= self.config.energy_threshold
        if voiced and not self._utterance:
            self._utterance_start = frame_start - len(self._pre_roll) * 20
            self._utterance.extend(b"".join(self._pre_roll))
            self._pre_roll.clear()
        if self._utterance or voiced:
            self._utterance.extend(frame)
            if voiced:
                self._last_voice_ms = self._received_ms
            if (
                self._received_ms - self._last_voice_ms >= self.config.hangover_ms
                or self._received_ms - self._utterance_start >= self.config.max_utterance_ms
            ):
                self._finalize(self._diarizer.segments())
        else:
            self._pre_roll.append(frame)

    def _finalize(self, timeline: list[dict[str, Any]]) -> None:
        if not self._utterance:
            return
        _check_timeline(timeline, self._received_ms)
        pcm = bytes(self._utterance)
        offset = self._utterance_start
        self._utterance.clear()
        self._pre_roll.clear()
        before = (self.skipped_segments, self.skipped_utterances)
        try:
            units = self._transcriber.transcribe(pcm, self._register_asr)
        finally:
            self._asr_process = None
        if self.closed:
            raise LiveAudioError("Audio session was stopped")
        provenance = getattr(self._diarizer, "speaker_provenance", "diarization-timeline")
        self._utterances += 1
        checked = _checked_units(units, len(pcm) // BYTES_PER_MS)
        if checked is None:
            self._skipped_utterances += 1
        skipped = self.skipped_segments - before[0]
        suppressed = self.skipped_utterances > before[1]
        if (skipped or suppressed) and self.config.report is not None:
            # Content-free: counts only, never recognizer text or timestamps. A failing
            # diagnostic channel never ends the session.
            segments = f"{skipped} recognizer segment(s) with unusable timestamps"
            if suppressed:
                message = (
                    "suppressed 1 utterance with invalid or out-of-order recognizer timestamps"
                )
                message += f" ({segments})" if skipped else ""
            else:
                message = f"skipped {segments}"
            with contextlib.suppress(Exception):
                self.config.report(message + "; the session continues")
        if suppressed:
            return
        units = checked
        # Units start at or after `offset`; a segment ending by then overlaps none of them,
        # so attribution scans only this utterance's part of the timeline.
        relevant = [segment for segment in timeline if segment["end_ms"] > offset]
        groups: list[dict[str, Any]] = []
        for unit in units:
            start, end = unit["start_ms"] + offset, unit["end_ms"] + offset
            speaker, overlap = _attribute(start, end, relevant)
            if speaker is not None and provenance == "diarization-utterance":
                # Per-request labels are namespaced by utterance so that equal labels
                # from independent requests can never be merged into one participant.
                speaker = utterance_scoped_speaker(self._utterances, speaker)
            if groups and (groups[-1]["speaker"], groups[-1]["overlap"]) == (speaker, overlap):
                groups[-1]["text"] += unit["text"]
                groups[-1]["end_ms"] = max(groups[-1]["end_ms"], end)
            else:
                groups.append(
                    {
                        "text": unit["text"],
                        "start_ms": start,
                        "end_ms": end,
                        "speaker": speaker,
                        "overlap": overlap,
                    }
                )
        for group in groups:
            text = group["text"].strip()
            if not text:
                continue
            self._counter += 1
            turn = Turn(
                session_id=self.config.session_id,
                utterance_id=f"live-{self._counter}",
                revision=1,
                start_ms=group["start_ms"],
                end_ms=group["end_ms"],
                text=text,
                speaker_id=group["speaker"],
                finalized=True,
                overlap=group["overlap"],
                recognizer_id=self._transcriber.recognizer_id,
                provenance=self.config.provenance,
                speaker_provenance=provenance if timeline else "unknown",
            )
            self.on_turn(turn)

    def finish(self) -> None:
        if self.closed:
            return
        try:
            if self._partial:
                # Process actual tail only. Synthetic zero-padding must not enter native history.
                pcm = bytes(self._partial)
                self._partial.clear()
                self._diarizer.push(pcm)
                if self._utterance:
                    self._utterance.extend(pcm)
            # Flush only for a pending utterance: nothing else is emitted, and a hosted
            # diarizer must not send audio that no turn will use.
            if self._utterance:
                self._finalize(self._diarizer.finish())
        except Exception:
            self.failed = True
            raise
        finally:
            self.close()

    def _register_asr(self, process: subprocess.Popen) -> None:
        self._asr_process = process
        if self.closed:
            _terminate(process)
            raise LiveAudioError("Audio session was stopped")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._asr_process is not None:
            _terminate(self._asr_process)
        self._partial.clear()
        self._pre_roll.clear()
        self._utterance.clear()
        self._diarizer.close()


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--native-worker":
        raise SystemExit(_native_worker(sys.argv[2], sys.argv[3]))
    raise SystemExit(2)
