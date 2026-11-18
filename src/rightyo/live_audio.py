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
import errno
import inspect
import io
import json
import math
import os
import secrets
import selectors
import stat
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from .addressedness import reply_wait
from .contracts import PROVENANCE, Turn, identifier, utterance_scoped_speaker
from .providers import Diarizer, Transcriber
from .turn_merge import TurnMerger, complete_hold, merge_gap, owner_pause, tail_join

SAMPLE_RATE = 16000
FRAME_BYTES = 640  # 20 ms of mono signed little-endian PCM16
BYTES_PER_MS = 32  # mono PCM16 at 16 kHz
MAX_CHUNK_BYTES = 32000  # one second of audio per push
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
# Preserve the pinned CLI's existing allowance for its one second of end padding.
# Clamp only the end, never extend the received audio or accept an onset beyond it.
ASR_END_PADDING_MS = 1000
# Two 10 ms whisper timestamp ticks (one PCM frame): tolerate rounding jitter only.
ASR_TIMING_JITTER_MS = 20
# Recovery alone is asynchronous and bounded. The first decode retains its configured
# timeout, including the base's pre-existing synchronous input-path blocking.
ASR_REDECODE_SECONDS = 5.0
ASR_CLEANUP_SECONDS = 6.0  # terminate 2 s + kill/reap 2 s + feeder join 2 s
ASR_RECOVERY_WINDOWS = 4  # includes the active/cancelling worker's window
# Four serial jobs can take 4 * (5 + 6) = 44 s. The fastest endpoint is a 20 ms
# voice frame plus 20 ms silence, so 44,000 / 40 = 1,100 completed windows.
ASR_HELD_WINDOWS = 1100
# Independent payload guard: PCM plus conservative charges for text/timeline entries.
ASR_HELD_BYTES = 16 * 1024 * 1024
ASR_PRIVACY_SPANS = 64  # bounded boundary history for late result admission
MAX_SPEAKER = 702  # "Speaker A" .. "Speaker ZZ"; native streams use 1..8
# A diarizer renewal deferred past an open utterance or held turn is forced this long after the
# privacy end is reached (#242 round 21), so steady speech cannot keep the old one in use.
SPEAKER_RENEWAL_LIMIT_MS = 10_000
# The native stream returns only segments ending within this trailing window of pushed
# audio. An utterance spans at most `max_utterance_ms` (<= 15 s, pre-roll included), so
# every segment that can overlap one is inside it with a wide margin, and per-utterance
# transfer and attribution stay constant however long the session runs (#54). This is the
# default; `LiveConfig.timeline_window_ms` sets a session's window (#226).
TIMELINE_WINDOW_MS = 60000
# The window's bounds, in whole seconds (#226). The lower bound is twice the longest
# utterance: a shorter window could drop a segment that still overlaps the open utterance,
# leaving its words unlabelled or, with edge attribution, labelled as a neighbour. The upper
# bound keeps the worst-case native response under `MAX_RESPONSE_BYTES`, since the worker
# returns the whole window as one JSON line: each segment `{"start_ms":S,"end_ms":E,
# "speaker":N},` is 36 bytes plus its two timestamps, so even 16-digit timestamps (a stream
# of over 30,000 years) give 68 bytes, and 100 s allows 30,000 segments: 30,000 x 68 + 25 =
# 2,040,025 bytes, under 2,097,152. A 10-hour session's 8-digit timestamps give 52 bytes and
# 1,560,025 bytes (74%).
MIN_TIMELINE_WINDOW_MS = 30000
MAX_TIMELINE_WINDOW_MS = 100000
# Per-response bound for any backend, derived from the window: an enforced cap of 300
# segments per second (three for each native 10 ms frame; a 60 s window holds 6,000 frames,
# so 18,000 segments). It is RightyO's limit, not a native guarantee: a response over it
# fails closed with a sanitized error rather than being truncated.
TIMELINE_SEGMENTS_PER_SECOND = 300
# The bound for the default window, kept for callers that name it.
MAX_TIMELINE_SEGMENTS = TIMELINE_WINDOW_MS // 1000 * TIMELINE_SEGMENTS_PER_SECOND


class LiveAudioError(ValueError):
    """Sanitized failure: never embeds paths, PCM, transcripts, or native logs."""


def timeline_window(value: Any) -> int:
    """`value` when it is a valid timeline window in milliseconds (#226), else an error.

    A whole number of seconds from `MIN_TIMELINE_WINDOW_MS` to `MAX_TIMELINE_WINDOW_MS`
    inclusive; nothing is clamped.
    """
    if (
        type(value) is not int
        or not MIN_TIMELINE_WINDOW_MS <= value <= MAX_TIMELINE_WINDOW_MS
        or value % 1000
    ):
        raise LiveAudioError("Invalid timeline window")
    return value


def timeline_segment_limit(window_ms: int) -> int:
    """The most segments one diarizer response may hold for a window (#226)."""
    return timeline_window(window_ms) // 1000 * TIMELINE_SEGMENTS_PER_SECOND


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
    # Whether audio received between two stream times was heard in privacy mode
    # (hailing-station #735 S-1). Per-utterance diagnostics (end-of-turn scores and
    # timing, skipped or suppressed recognizer output) are withheld for it, so logs keep
    # nothing about private speech. None means never private; a failure counts as private.
    private: Callable[[int, int], bool] | None = None
    # Total audio accepted per session, in stream milliseconds. None means no ceiling:
    # memory stays bounded by the utterance window and downstream retention limits.
    session_budget_ms: int | None = None
    # A `Transcriber`/`Diarizer` instance, or a factory called with this config. None
    # selects the local whisper.cpp recognizer and Nemotron native stream.
    transcriber: Transcriber | Callable[[LiveConfig], Transcriber] | None = None
    diarizer: Diarizer | Callable[[LiveConfig], Diarizer] | None = None
    # Join a speaker's next finalized turn into the previous one when it starts within
    # this many stream milliseconds (#73). 0, the library default, emits every turn at
    # once as before; `listen` and the lab pass their configured gap.
    turn_merge_gap_ms: int = 0
    # Text predicate for a fragment that must stay a turn of its own and is never joined
    # or held, such as an owner stop phrase matched against a whole turn downstream.
    turn_break: Callable[[str], bool] | None = None
    # Text predicate for a turn that looks cut off mid-request (#199): when the end-of-turn
    # model calls it complete, it is held for the merge gap anyway. None trusts the model.
    turn_unfinished: Callable[[str], bool] | None = None
    # Hold a request-shaped turn at least this long after its last word to observe the
    # post-turn gap (#96); 0, the library default, observes nothing. Joining still uses
    # only `turn_merge_gap_ms`.
    reply_wait_ms: int = 0
    # Called with (utterance_id, post_turn_gap) just before `on_turn` for each turn whose
    # gap was observed, so a decision can use it without changing the `Turn` contract.
    on_post_turn_gap: Callable[[str, dict[str, Any]], None] | None = None
    # Called with the utterance id just before `on_turn` for each turn whose speaker label
    # was partly inferred (edge attribution or tail join), so roles can withhold owner
    # authority from it (#137) without changing the `Turn` contract.
    on_inferred: Callable[[str], None] | None = None
    # Internal #292 fact: leading diarized characters in published text, or None when
    # its first word is inferred/unknown. Called before on_turn; never a wire key.
    on_diarized_head: Callable[[str, int | None], None] | None = None
    # Called with (utterance_id, spans) just before `on_turn` for each `diarization-timeline`
    # turn, where `spans` are the `(start_ms, end_ms)` stream-time pieces of the turn's span
    # that the diarizer timeline gave the turn's own label and no other speaker (#148), so
    # live speaker identification embeds only that audio; empty when there are none. None,
    # the default, computes nothing. Internal plumbing: the `Turn` contract is unchanged.
    on_labelled_spans: Callable[[str, tuple[tuple[int, int], ...]], None] | None = None
    # Optional end-of-turn model (#117), such as `SmartTurn.score`: called with the open
    # utterance's PCM once `end_of_turn_silence_ms` of silence follows speech, returning
    # P(turn complete). At or above `end_of_turn_threshold` the utterance is finalized as
    # soon as a streaming diarizer's timeline covers its last voiced audio (at once for an
    # utterance-level diarizer), and its turn is not held for a continuation (its
    # post-turn gap is unobserved); below it, the hangover and merge hold apply as without
    # a model. It is asked again at each later pause. None, the default, never asks.
    end_of_turn: Callable[[bytes], float] | None = None
    # Edge attribution (opt-in, 0 = off). An inference, not an observation: a word the
    # streaming diarizer's timeline left unlabelled (the recognizer's word timestamps run
    # slightly past the segment, or the word has zero length) takes the utterance's only
    # labelled speaker when the word overlaps no one and either touches only that
    # speaker's segments or, touching none, lies within this many milliseconds of one.
    # A second speaker's first word in that slack would be mislabelled, so session-label
    # roles refuse it (see `PrototypeController.start`); live identification roles allow
    # it but withhold owner authority from such a turn (`on_inferred`, #137). Only for
    # `diarization-timeline`.
    edge_attribution_ms: int = 0
    # Coverage labelling (0 = off; #220). When no word of an utterance has a speaker under
    # the strict rule (each word covered throughout by exactly one speaker), as with a
    # one-word turn whose timestamps run past the timeline segment, a word that only one
    # speaker's segments starting inside the utterance touch takes that speaker when they
    # cover at least this fraction of it. A segment carried over from before the utterance
    # never counts, since a late speaker switch can stretch it over the next person's word.
    # Two speakers in a word still leave it unlabelled. Only for `diarization-timeline`;
    # utterances with any strictly labelled word are unchanged. The labels are not marked
    # inferred; edge attribution from them is, as usual.
    word_label_coverage: float = 0.0
    # Tail join (opt-in, 0 = off; #129). Also an inference: an unlabelled, non-overlapping
    # turn starting within this many milliseconds of a held labelled turn's end joins it
    # and takes its speaker (see `TurnMerger`). Edge attribution only reaches words near
    # the speaker's timeline segment; this catches the rest of a trailing piece. Refused
    # with session-label roles, and reported through `on_inferred`, like edge attribution.
    tail_join_ms: int = 0
    # One thought split into several turns (#216; see `TurnMerger`). Hold a turn the
    # end-of-turn model called complete this long after its last word for the same
    # labelled speaker's continuation (0, the default, emits it at once); keep a restart's
    # repeated words once when a continuation joins such a turn; and keep a held turn held
    # across another or an unattributed speaker's short backchannel. All off by default.
    turn_complete_hold_ms: int = 0
    turn_collapse_restarts: bool = False
    turn_bridge_backchannels: bool = False
    # A lone wake word split from the owner's sentence (#230; see `TurnMerger`). With both
    # set and merging on, an unattributed turn whose text is only a configured name is held
    # for at most the merge gap, and joined by the owner's labelled continuation, which
    # keeps its label; the joined turn is reported through `on_inferred`. `turn_wake_word`
    # takes a turn's text; `turn_owner` a fragment (`speaker`, `speaker_provenance`,
    # `overlap`) and says whether its label is the owner's now. None, the default, is off.
    turn_wake_word: Callable[[str], bool] | None = None
    turn_owner: Callable[[dict[str, Any]], bool] | None = None
    # The owner-only pause (hailing-station #735; see `TurnMerger`): with `turn_owner`
    # set, a turn whose label is the owner's is held at least this long, as the complete
    # hold and as the merge gap for the same speaker's continuation. 0, the default, is off.
    turn_owner_pause_ms: int = 0
    end_of_turn_threshold: float = 0.5
    end_of_turn_silence_ms: int = 200
    # The trailing window of the diarizer timeline returned for attribution (#226), and
    # with it the per-response segment bound (`timeline_segment_limit`). Memory only; set at
    # session start and never changed during it. The native worker receives it at launch.
    timeline_window_ms: int = TIMELINE_WINDOW_MS
    # Local privacy-exit plumbing only. Borrowed view of a finalized turn's exact PCM
    # from the existing ASR assembly; never a ring or speaker-evidence input. Private
    # turns bypass continuation holds so only the current sentence needs retaining.
    on_private_turn_audio: Callable[[Turn, memoryview], None] | None = None

    def __post_init__(self) -> None:
        identifier(self.session_id, "session_id")
        timeline_window(self.timeline_window_ms)
        try:
            merge_gap(self.turn_merge_gap_ms)
        except ValueError:
            raise LiveAudioError("Invalid turn merge gap") from None
        try:
            reply_wait(self.reply_wait_ms)
        except ValueError:
            raise LiveAudioError("Invalid reply wait") from None
        try:
            tail_join(self.tail_join_ms)
        except ValueError:
            raise LiveAudioError("Invalid tail join window") from None
        try:
            complete_hold(self.turn_complete_hold_ms)
        except ValueError:
            raise LiveAudioError("Invalid complete hold") from None
        try:
            owner_pause(self.turn_owner_pause_ms)
        except ValueError:
            raise LiveAudioError("Invalid owner pause") from None
        if type(self.turn_collapse_restarts) is not bool or (
            type(self.turn_bridge_backchannels) is not bool
        ):
            raise LiveAudioError("Invalid turn merge option")
        if self.on_post_turn_gap is not None and not callable(self.on_post_turn_gap):
            raise LiveAudioError("Invalid post-turn gap observer")
        if self.on_inferred is not None and not callable(self.on_inferred):
            raise LiveAudioError("Invalid inferred-label observer")
        if self.on_diarized_head is not None and not callable(self.on_diarized_head):
            raise LiveAudioError("Invalid diarized-head observer")
        if self.on_labelled_spans is not None and not callable(self.on_labelled_spans):
            raise LiveAudioError("Invalid labelled-span observer")
        if self.turn_break is not None and not callable(self.turn_break):
            raise LiveAudioError("Invalid turn break predicate")
        if self.turn_unfinished is not None and not callable(self.turn_unfinished):
            raise LiveAudioError("Invalid unfinished-turn predicate")
        if self.turn_wake_word is not None and not callable(self.turn_wake_word):
            raise LiveAudioError("Invalid wake word predicate")
        if self.turn_owner is not None and not callable(self.turn_owner):
            raise LiveAudioError("Invalid owner predicate")
        if (
            type(self.word_label_coverage) not in (int, float)
            or not math.isfinite(self.word_label_coverage)
            or not 0 <= self.word_label_coverage <= 1
        ):
            raise LiveAudioError("Invalid word label coverage")
        if self.end_of_turn is not None and not callable(self.end_of_turn):
            raise LiveAudioError("Invalid end-of-turn model")
        if (
            type(self.end_of_turn_threshold) not in (int, float)
            or not math.isfinite(self.end_of_turn_threshold)
            or not 0 < self.end_of_turn_threshold <= 1
        ):
            raise LiveAudioError("Invalid end-of-turn threshold")
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
            (self.end_of_turn_silence_ms, 20, 1000),
            (self.edge_attribution_ms, 0, 2000),
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

    # The stream's window and its segment bound (#226): set per instance from the window
    # the worker was launched with; these class values are the defaults.
    window_ms = TIMELINE_WINDOW_MS
    max_segments = MAX_TIMELINE_SEGMENTS

    def __init__(self, library: str, model_path: str, window_ms: int = TIMELINE_WINDOW_MS):
        # Validated before anything is loaded: the worker never trusts its argument.
        self.window_ms = timeline_window(window_ms)
        self.max_segments = timeline_segment_limit(self.window_ms)
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
        cutoff_ms = self.pushed_bytes // BYTES_PER_MS - self.window_ms
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
        if len(result) > self.max_segments:
            raise LiveAudioError("Invalid local diarizer result")
        return result

    def close(self) -> None:
        if self.stream:
            self.lib.nemo_speech_diar_stream_close(self.stream)
            self.stream = ctypes.c_void_p()
        if self.model:
            self.lib.nemo_speech_diar_destroy(self.model)
            self.model = ctypes.c_void_p()


def _response_line(response: dict[str, Any]) -> str:
    """One worker response exactly as it crosses the pipe: compact JSON and a newline."""
    return json.dumps(response, separators=(",", ":")) + "\n"


def _worker_window(text: Any) -> int | None:
    """The `--native-worker` window argument as milliseconds, or None when invalid (#226).

    Only the canonical decimal form of a valid window (`timeline_window`) is accepted: no
    sign, spaces, leading zeros or fraction, and nothing outside the bounds.
    """
    if not isinstance(text, str) or not (text.isascii() and text.isdigit()) or len(text) > 7:
        return None
    value = int(text)
    if str(value) != text:
        return None
    try:
        return timeline_window(value)
    except LiveAudioError:
        return None


def _native_worker(library: str, model: str, window_ms: int = TIMELINE_WINDOW_MS) -> int:
    # Preserve a protocol-only descriptor before redirecting every native log.
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
    native = None
    try:
        native = _NativeStream(library, model, window_ms)
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
            protocol.write(_response_line(response))
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
    `segments`/`finish` call returns only the trailing `LiveConfig.timeline_window_ms`
    of the timeline (`TIMELINE_WINDOW_MS` by default), so there is no session-length
    segment limit. The worker receives the window as its last argument and validates it
    again.
    """

    diarizer_id = "nemotron.cpp v3-streaming"

    def __init__(self, config: LiveConfig):
        if config.diarization_library is None or config.diarization_model is None:
            raise LiveAudioError("Explicit existing runtimes and models are required")
        self.timeout = config.timeout_seconds
        self.cancelled = config.cancelled if config.cancelled is not None else (lambda: False)
        self.window_ms = timeline_window(config.timeline_window_ms)
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
                str(self.window_ms),
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


def _attribute(
    start: int, end: int, timeline: list[dict[str, Any]], min_coverage: float = 0.0
) -> tuple[str | None, bool]:
    """The speaker of the word spanning `start`..`end`, and whether speakers overlap in it.

    By default the word must be covered throughout by exactly one speaker. With
    `min_coverage` above 0 (#220), uncovered slack at its edges is allowed: a word that
    only one speaker's segments touch is theirs when they cover at least that fraction of
    it. Two speakers in the word still leave it unlabelled.
    """
    relevant = [s for s in timeline if s["start_ms"] < end and s["end_ms"] > start]
    edges = sorted(
        {start, end}
        | {max(start, min(end, s[k])) for s in relevant for k in ("start_ms", "end_ms")}
    )
    speakers = set()
    ambiguous = overlap = False
    covered = 0
    for left, right in zip(edges, edges[1:]):
        active = {s["speaker"] for s in relevant if s["start_ms"] < right and s["end_ms"] > left}
        overlap |= len(active) > 1
        ambiguous |= len(active) != 1
        covered += right - left if len(active) == 1 else 0
        speakers.update(active)
    if ambiguous and min_coverage > 0 and not overlap and len(speakers) == 1 and end > start:
        ambiguous = covered < min_coverage * (end - start)
    if ambiguous or len(speakers) != 1:
        return None, overlap
    # Native channels are arrival-ordered and persist for the whole session.
    return "Speaker " + _speaker_label(next(iter(speakers))), overlap


def _attribute_edges(
    attributed: list[list[Any]],
    timeline: list[dict[str, Any]],
    gap_ms: int,
    *,
    blockers: list[dict[str, Any]] | None = None,
) -> set[int]:
    """Give unlabelled edge words the utterance's only labelled speaker, in place.

    Returns the indices of the entries it labelled: their labels are inferred (#137).

    Each entry is `[unit, start, end, speaker, overlap]`. Only when the utterance has
    exactly one labelled speaker; only for a word that overlaps no one and either touches
    segments of that speaker alone, or touches no segment and lies within `gap_ms` of one
    of that speaker's segments (timestamps running just past the segment's edge).
    `blockers` can retain touching evidence across a run's loss boundary solely to
    refuse a label; it cannot supply positive evidence or extend the inference slack.
    """
    inferred: set[int] = set()
    labels = {entry[3] for entry in attributed if entry[3] is not None}
    if len(labels) != 1:
        return inferred
    label = next(iter(labels))
    own = [
        (segment["start_ms"], segment["end_ms"])
        for segment in timeline
        if "Speaker " + _speaker_label(segment["speaker"]) == label
    ]
    for index, entry in enumerate(attributed):
        _, start, end, speaker, overlap = entry
        if speaker is not None or overlap:
            continue
        touching = {
            "Speaker " + _speaker_label(segment["speaker"])
            for segment in (timeline if blockers is None else blockers)
            if segment["start_ms"] <= end and segment["end_ms"] >= start
        }
        supported = any(seg_start <= end and seg_end >= start for seg_start, seg_end in own)
        near = any(max(start - seg_end, seg_start - end, 0) <= gap_ms for seg_start, seg_end in own)
        if (touching == {label} and supported) or (not touching and near):
            entry[3] = label
            inferred.add(index)
    return inferred


def _labelled_spans(
    label: str | None, start: int, end: int, timeline: list[dict[str, Any]]
) -> list[tuple[int, int]]:
    """The pieces of `[start, end)` the timeline gives `label` and no other speaker (#148).

    Clipped to the span, merged where they touch, in time order. Audio the timeline leaves
    unlabelled (such as an edge-attributed or tail-joined word) or gives to another speaker,
    including any overlap with this label, is never included.
    """
    if label is None or end <= start:
        return []
    own: list[tuple[int, int]] = []
    others: list[tuple[int, int]] = []
    for segment in timeline:
        left, right = max(start, segment["start_ms"]), min(end, segment["end_ms"])
        if right > left:
            mine = "Speaker " + _speaker_label(segment["speaker"]) == label
            (own if mine else others).append((left, right))
    if not own:
        return []
    edges = sorted({edge for piece in own + others for edge in piece})
    spans: list[tuple[int, int]] = []
    for left, right in zip(edges, edges[1:]):
        if not any(a <= left and right <= b for a, b in own):
            continue
        if any(a < right and b > left for a, b in others):
            continue
        if spans and spans[-1][1] == left:
            spans[-1] = (spans[-1][0], right)
        else:
            spans.append((left, right))
    return spans


def _speaker_label(number: int) -> str:
    """Spreadsheet-column letters: 1..26 -> A..Z (unchanged), 27 -> AA, 52 -> AZ, 53 -> BA."""
    letters = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def _check_timeline(
    timeline: Any, received_ms: int, max_segments: int = MAX_TIMELINE_SEGMENTS
) -> None:
    """Enforce the `Diarizer` contract at the common boundary, whatever the backend.

    A list of at most `max_segments` dicts (one call's window, not the session; the
    processor passes the bound derived from its configured window), each with integer
    `0 <= start_ms <= end_ms` within the audio received so far plus one second of
    lookahead (the native stream's own bound) and a positive integer `speaker` no greater
    than `MAX_SPEAKER`. Order is not required: attribution intersects intervals and the
    native ABI promises none.
    """
    if not isinstance(timeline, list) or len(timeline) > max_segments:
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


# Where the local recognizer runs (hailing-station #735 S-1): its working directory, HOME
# and TMPDIR. The audio goes in on stdin and the JSON comes back on stdout, so nothing is
# ever written to disk, even briefly or after a kill; the directory must be one this
# process cannot write, so a stray write by the binary fails rather than lands anywhere.
# There is no fallback: without one, transcription is refused (fail closed). macOS has
# /var/empty; on Linux, /proc is a real directory in which no file can be created.
NO_FILES_CANDIDATES = ("/var/empty", "/proc")
# How much of the recognizer's stdout is read at a time.
READ_CHUNK_BYTES = 65536


class RecognizerSandboxError(LiveAudioError):
    """No directory this process cannot write is available for the local recognizer."""


# What a refused exclusive create in an unwritable directory fails with.
UNWRITABLE_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS, errno.ENOENT})
# The directory chosen by the first successful probe in this process (see `recognizer_dir`).
_recognizer_dir_cache: str | None = None


def _real_directory(path: str) -> bool:
    """Whether `path` itself is a directory, by `lstat`: a symlink never is."""
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _refuses_creation(directory: str) -> bool:
    """Whether creating a file in `directory` really fails, by trying it.

    `os.access` is not enough: root's override reports procfs (which refuses new files)
    as writable. So this exclusively creates an unpredictably named file, never following
    a link, and writes nothing. A refusal with EACCES, EPERM, EROFS or ENOENT means
    unwritable. Any success means writable: the file is closed and unlinked at once. Any
    other error counts as writable (fail closed).
    """
    probe = os.path.join(directory, secrets.token_hex(16))
    try:
        descriptor = os.open(
            probe, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600
        )
    except OSError as error:
        return error.errno in UNWRITABLE_ERRNOS
    os.close(descriptor)
    with contextlib.suppress(OSError):
        os.unlink(probe)
    return False


def recognizer_dir() -> str:
    """The first candidate (`NO_FILES_CANDIDATES`) that is a real directory (by `lstat`,
    so never a symlink, which is never probed) in which this process cannot create a file
    (`_refuses_creation`).

    Probed once per process and cached; probed again only if the cached directory stops
    being a real directory. Decided by behaviour rather than permission bits, so it is right
    for root too. With no candidate left this raises `RecognizerSandboxError`.
    """
    global _recognizer_dir_cache
    cached = _recognizer_dir_cache
    if cached is not None and _real_directory(cached):
        return cached
    _recognizer_dir_cache = None
    for candidate in NO_FILES_CANDIDATES:
        if _real_directory(candidate) and _refuses_creation(candidate):
            _recognizer_dir_cache = candidate
            return candidate
    raise RecognizerSandboxError(
        "The local recognizer needs a directory this process cannot write, such as /var/empty"
    )


def _wav_bytes(pcm: bytes) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm)
    return buffer.getvalue()


def _feed(stream, data: bytes) -> None:
    """Write `data` to the recognizer's stdin and close it; a closed pipe ends quietly."""
    try:
        stream.write(data)
    except (OSError, ValueError):
        pass
    finally:
        with contextlib.suppress(OSError, ValueError):
            stream.close()


def _bounded_stdout(process: subprocess.Popen, deadline: float) -> bytes:
    """The recognizer's whole stdout, read in chunks; never more than `MAX_RESPONSE_BYTES`.

    Raises `LiveAudioError` as soon as the output passes the cap, before any more is read,
    and `subprocess.TimeoutExpired` at the deadline.
    """
    chunks, total = [], 0
    descriptor = process.stdout.fileno()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            left = deadline - time.monotonic()
            if left <= 0 or not selector.select(left):
                raise subprocess.TimeoutExpired(process.args, 0)
            chunk = os.read(descriptor, READ_CHUNK_BYTES)
            if not chunk:
                return b"".join(chunks)
            total += len(chunk)
            if total > MAX_RESPONSE_BYTES:
                raise LiveAudioError("Local recognizer failed")
            chunks.append(chunk)


def _transcribe(
    config: LiveConfig,
    pcm: bytes,
    register: Callable[[subprocess.Popen], None] | None = None,
) -> dict[str, Any]:
    """One whisper.cpp run over `pcm`, in memory only: WAV on stdin, JSON on stdout.

    The WAV is fed from a thread while stdout is read in bounded chunks, so neither pipe
    can deadlock the other and a runaway recognizer is stopped at `MAX_RESPONSE_BYTES`.
    stderr goes to the null device, so it is never buffered and never blocks.
    """
    sandbox = recognizer_dir()
    deadline = time.monotonic() + config.timeout_seconds
    try:
        process = subprocess.Popen(
            [
                str(Path(config.whisper_executable).resolve()),
                "--model",
                str(Path(config.whisper_model).resolve()),
                "--file",
                "-",
                "--output-json-full",
                "--output-file",
                "-",
                "--no-prints",
                "--max-context",
                "0",
                "--suppress-nst",
            ],
            cwd=sandbox,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={"PATH": os.defpath, "HOME": sandbox, "TMPDIR": sandbox},
        )
        feeder = threading.Thread(
            target=_feed,
            args=(process.stdin, _wav_bytes(pcm)),
            name="rightyo-asr-feed",
            daemon=True,
        )
        try:
            if register is not None:
                register(process)
            feeder.start()
            raw = _bounded_stdout(process, deadline)
            if process.wait(timeout=max(0.0, deadline - time.monotonic())) != 0:
                raise LiveAudioError("Local recognizer failed")
        finally:
            _terminate(process)
            # Every path closes both pipes; the feeder ends once its pipe is closed.
            for stream in (process.stdin, process.stdout):
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
            if feeder.ident is not None:
                feeder.join(timeout=2)
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError
        return document
    except subprocess.TimeoutExpired:
        raise LiveAudioError("Local recognizer timed out") from None
    except (OSError, subprocess.CalledProcessError, ValueError, RecursionError):
        raise LiveAudioError("Local recognizer failed") from None


def _units(document: dict[str, Any], duration_ms: int) -> list[dict[str, Any]]:
    """The salvaged units of one whisper.cpp document; see `_recognizer_window`."""
    return _counted_units(document, duration_ms)[0]


@dataclass
class _RecognizerWindow:
    """Internal validation evidence only; never serialized into host events."""

    units: list[dict[str, Any]]
    gaps: set[int]  # unit indices before/after unresolved text, including 0 and len(units)
    segments: int = 0
    repaired: int = 0
    # 1=type, 2=bounds, 3=reversal, 4=order, 5=retry failure/deadline, 6=queue/hold full.
    reasons: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    retry_segments: int = 0
    retried: bool = False
    elapsed_ms: int = 0

    @property
    def rejected(self) -> int:
        return sum(self.reasons[:4])


def _counted_units(document: dict[str, Any], duration_ms: int) -> tuple[list[dict[str, Any]], int]:
    """Compatibility view of the validated units and rejected segment count."""
    window = _recognizer_window(document, duration_ms)
    return window.units, window.rejected


def _recognizer_window(document: dict[str, Any], duration_ms: int) -> _RecognizerWindow:
    """Keep every usable segment in source order, without moving its valid times.

    Clamp CLI end padding, retain zero-duration text (no direct speaker evidence), and
    collapse <=20 ms reversed intervals to a point. Small inter-segment jitter is
    clamped to the preceding retained interval; larger reversals are rejected, never
    sorted. Dropped segments leave barriers, including at the window's two edges.
    Token timing still falls back to the complete segment on any inconsistency.
    Malformed document structures fail closed as before.
    """
    segments = document.get("transcription")
    if not isinstance(segments, list) or len(segments) > 1000:
        raise LiveAudioError("Invalid local recognizer result")
    result: list[dict[str, Any]] = []
    fallbacks: list[dict[str, Any]] = []
    segment_gaps: set[int] = set()
    window = _RecognizerWindow(result, set(), segments=len(segments))
    reasons = [0] * 6
    for segment in segments:
        if not isinstance(segment, dict):
            raise LiveAudioError("Invalid local recognizer result")
        text, offsets = segment.get("text"), segment.get("offsets")
        if not isinstance(text, str) or len(text) > 4000 or not isinstance(offsets, dict):
            raise LiveAudioError("Invalid local recognizer result")
        if not text.strip():
            continue
        start, end = offsets.get("from"), offsets.get("to")
        previous_start = fallbacks[-1]["start_ms"] if fallbacks else 0
        previous_end = fallbacks[-1]["end_ms"] if fallbacks else 0
        reason = 0
        original = (start, end)
        if type(start) is not int or type(end) is not int:
            reason = 1
        elif not (0 <= start <= duration_ms and 0 <= end <= duration_ms + ASR_END_PADDING_MS):
            reason = 2
        elif start - end > ASR_TIMING_JITTER_MS:
            reason = 3
        else:
            end = min(max(start, end), duration_ms)
            if max(previous_start - start, previous_end - end) > ASR_TIMING_JITTER_MS:
                reason = 4
            else:
                start, end = max(start, previous_start), max(end, previous_end)
        if reason:
            reasons[reason - 1] += 1
            window.gaps.add(len(result))
            segment_gaps.add(len(fallbacks))
            continue
        repaired = (start, end) != original or start == end
        window.repaired += int(repaired)
        fallback = {"text": text, "start_ms": start, "end_ms": end}
        fallbacks.append(fallback)
        tokens = segment.get("tokens")
        if (
            start != original[0]
            or end > original[1]
            or start == end
            or not isinstance(tokens, list)
            or len(tokens) > 4000
        ):
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
            if not isinstance(interval, dict):
                usable = False
                break
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
        usable &= bool(words) and (
            words[0]["start_ms"] >= previous_start and words[0]["end_ms"] >= previous_end
        )
        result.extend(words if usable else [fallback])
    if any(
        a["start_ms"] > b["start_ms"] or a["end_ms"] > b["end_ms"]
        for a, b in zip(result, result[1:])
    ):
        # Ordered overlapping segments can have token detail that backtracks across
        # their boundary. Keep all segment text/times; sacrifice detail, never words.
        result[:] = fallbacks
        window.gaps = segment_gaps
    window.reasons = tuple(reasons)
    return window


class WhisperCppTranscriber:
    """The default `Transcriber`: the pinned whisper.cpp CLI, audio and JSON over pipes."""

    recognizer_id = "whisper.cpp-live-window"

    def __init__(self, config: LiveConfig):
        if config.whisper_executable is None or config.whisper_model is None:
            raise LiveAudioError("Explicit existing runtimes and models are required")
        # Refused before any audio when no unwritable directory is available (fail closed).
        recognizer_dir()
        self.config = config
        # First-pass evidence is transferred to the input owner for ordered accounting.
        self._window: _RecognizerWindow | None = None

    def transcribe(
        self, pcm: bytes, register: Callable[[subprocess.Popen], None] | None = None
    ) -> list[dict[str, Any]]:
        started = time.monotonic()
        duration = len(pcm) // BYTES_PER_MS
        self._window = None
        document = _transcribe(self.config, pcm, register)
        window = _recognizer_window(document, duration)
        window.elapsed_ms = max(0, round((time.monotonic() - started) * 1000))
        self._window = window
        # The processor owns ordered accounting and optional background recovery.
        return window.units


@dataclass
class _FinalizedWindow:
    pcm: bytearray
    offset: int
    through_ms: int
    voice_start: int
    number: int
    timeline: list[dict[str, Any]]
    provenance: str
    complete: bool
    units: Any
    window: _RecognizerWindow | None
    before: tuple[int, int]
    duration_ms: int
    timeline_present: bool
    retry_pending: bool = False
    overflow: bool = False
    loss_count: int = 1
    last_number: int = 0
    private_loss: bool = False
    # Ordered after this result, even if recovery has not resolved it yet. Adjacent
    # barriers have no intervening text, so the stronger privacy settlement subsumes
    # a plain flush. One token per bounded window; no separate growing event queue.
    barrier_after: str | None = None
    discarded: bool = False
    private_cleared_through: int = 0

    def release(self) -> None:
        """Discard owned PCM/text evidence; safe to repeat during shutdown."""
        if isinstance(self.pcm, bytearray):
            self.pcm[:] = bytes(len(self.pcm))
        self.pcm = bytearray()
        if isinstance(self.units, list):
            self.units.clear()
        if self.window is not None:
            self.window.units.clear()
        self.units = []
        self.window = None
        self.timeline = []
        # A released barrier has no recognizer metadata left to coalesce into.
        self.overflow = False

    def held_bytes(self) -> int:
        # A logical admission budget, not a claim about a native model's RSS.
        return (
            1024
            + len(self.pcm)
            + 512 * len(self.timeline)
            + sum(512 + 4 * len(unit["text"]) for unit in self.units)
            + (512 * len(self.window.gaps) if self.window is not None else 0)
        )


class _WindowRecovery:
    """One local retry, constructed without starting; the owner publishes before start.

    Lifecycle state uses the processor's one lock. Process cleanup and shutdown joins
    run outside it. Normal ingestion never waits for the worker or invokes its callbacks.
    """

    def __init__(
        self,
        config: LiveConfig,
        packet: _FinalizedWindow,
        lock,
        shutdown: threading.Event,
    ):
        self.packet = packet
        self._lock = lock
        self._shutdown = shutdown
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.process: subprocess.Popen | None = None
        self.result: _RecognizerWindow | None = None
        self.config = replace(
            config, timeout_seconds=min(config.timeout_seconds, ASR_REDECODE_SECONDS)
        )
        self.started = self.expires = 0.0
        assert packet.window is not None
        self.pcm = packet.pcm
        self._initial = replace(packet.window)
        # No PCM/text in Thread.args: ownership stays explicit and accountable.
        self.thread = threading.Thread(target=self._run, name="rightyo-asr-recovery", daemon=True)

    def start(self) -> None:
        """Called under the lifecycle lock, after publication, while still open."""
        if self._shutdown.is_set():
            self.pcm = b""
            self._initial = None
            self.result = None
            self.packet.release()
            self.done.set()
            return
        self.started = time.monotonic()
        self.expires = self.started + self.config.timeout_seconds + ASR_CLEANUP_SECONDS
        self.thread.start()

    def _register(self, process: subprocess.Popen) -> None:
        with self._lock:
            self.process = process
            stopped = self.cancelled.is_set() or self._shutdown.is_set()
            if stopped:
                self.cancelled.set()
        if stopped:
            with contextlib.suppress(OSError):
                process.kill()
            raise LiveAudioError("Audio recovery was cancelled")

    def _run(self) -> None:
        # Shared mutable PCM lets a privacy boundary wipe even a running input.
        pcm, initial = self.pcm, self._initial
        retry = None
        try:
            if self.cancelled.is_set() or self._shutdown.is_set():
                return
            try:
                retry = _recognizer_window(
                    _transcribe(self.config, pcm, self._register), len(pcm) // BYTES_PER_MS
                )
            except Exception:  # noqa: BLE001 - no private exception escapes the worker
                retry = _RecognizerWindow([], {0}, reasons=(0, 0, 0, 0, 1, 0))
            retry.retry_segments = retry.segments
            retry.segments = initial.segments
            retry.reasons = tuple(a + b for a, b in zip(initial.reasons, retry.reasons))
            retry.retried = True
            retry.elapsed_ms = initial.elapsed_ms + max(
                0, round((time.monotonic() - self.started) * 1000)
            )
            with self._lock:
                if not self.cancelled.is_set() and not self._shutdown.is_set():
                    self.result = retry
        finally:
            # _transcribe normally reaps its child. Also cover cancellation during
            # registration/startup, before a decoder can enter its usual cleanup.
            with self._lock:
                process = self.process
            if process is not None:
                with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                    _terminate(process)
            pcm, initial, retry = b"", None, None
            with self._lock:
                self.process = None
                self.pcm = b""
                self._initial = None
                if self.cancelled.is_set() or self._shutdown.is_set():
                    self.result = None
                if self._shutdown.is_set():
                    self.packet.release()
                self.done.set()

    def cancel(self) -> None:
        with self._lock:
            already_cancelled = self.cancelled.is_set()
            self.cancelled.set()
            self.result = None
            process = None if already_cancelled else self.process
            if self.thread.ident is None:
                self.pcm = b""
                self._initial = None
                if self._shutdown.is_set():
                    self.packet.release()
                self.done.set()
        if process is not None:
            with contextlib.suppress(OSError):
                process.kill()

    def join(self) -> None:
        if self.thread.ident is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=self.config.timeout_seconds + ASR_CLEANUP_SECONDS)


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


class _OffsetDiarizer:
    """A diarizer started mid-stream (#735 S-1): its own clock starts at zero, so its timeline
    is shifted onto the stream clock the processor's utterances use."""

    def __init__(self, inner: Diarizer, offset_ms: int):
        self._inner, self._offset = inner, offset_ms

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _shifted(self, timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **segment,
                **{
                    key: segment[key] + self._offset
                    for key in ("start_ms", "end_ms")
                    if type(segment.get(key)) is int
                },
            }
            if isinstance(segment, dict)
            else segment
            for segment in timeline
        ]

    def push(self, pcm: bytes) -> None:
        self._inner.push(pcm)

    def segments(self) -> list[dict[str, Any]]:
        return self._shifted(self._inner.segments())

    def finish(self) -> list[dict[str, Any]]:
        return self._shifted(self._inner.finish())

    def close(self) -> None:
        self._inner.close()


class LiveProcessor:
    """Single-owner processor with one bounded recovery worker.

    Feed only explicitly consented audio.

    `close` cancels and discards an unfinished utterance. `finish` flushes the
    finite causal input and emits its tail. Failures leave `failed=True` and
    cancel resources. The caller must stop capture immediately on failure.
    """

    def __init__(self, config: LiveConfig, on_turn: Callable[[Turn], None]):
        self.config = config
        self.on_turn = on_turn
        self.failed = False
        self._recovery_lock = threading.Lock()
        self._recovery_shutdown = threading.Event()
        self.closed = False
        self._transcriber: Transcriber = _select(
            config.transcriber, WhisperCppTranscriber, config, "transcribe"
        )
        # Resolved through the module at call time so the default stays patchable.
        self._diarizer: Diarizer = _select(
            config.diarizer, lambda value: _Diarizer(value), config, "push"
        )
        # A diarizer the processor made (the default, or a factory) can be made again when
        # privacy ends (`reset_speakers`, hailing-station #735 S-1); a supplied instance cannot.
        self._diarizer_renewable = config.diarizer is None or not _is_backend_instance(
            config.diarizer, "push"
        )
        self._speaker_reset = False
        # The stream byte offset privacy last ended at, until the frame holding it is processed;
        # then what is still buffered from before it is silenced (`_private_cutoff`). Written on
        # the host thread and taken on the audio thread, both under `_private_lock`.
        self._private_until: int | None = None
        self._privacy_start_bytes = 0
        self._private_ranges: list[tuple[int, int]] = []
        self._private_cleared_through = 0  # audio-owner watermark, snapshotted per window
        self._private_lock = threading.Lock()
        self._frame_private = False
        self._diarizer_private_frames = 0
        self._privacy_uncertain = False
        self._diarizer_pending = 0
        self._diarizer_audited_bytes = 0
        # Where the last privacy end was reached (audio thread only), bounding a deferred renewal;
        # and, after a forced one, the renewal's offset: a turn starting before it is unlabelled.
        self._end_reached_ms: int | None = None
        self._unlabelled_before_ms = 0
        # Told when the fresh diarizer takes over, before its first frame: labels restart there,
        # so whatever was bound to the old ones in the meantime must be forgotten again.
        self.on_speakers_renewed: Callable[[], None] | None = None
        # The per-response segment bound for this session's timeline window (#226).
        self._timeline_limit = timeline_segment_limit(config.timeline_window_ms)
        self._partial = bytearray()
        self._pre_roll: deque[bytes] = deque(maxlen=config.pre_roll_ms // 20)
        self._utterance = bytearray()
        self._utterance_start = 0
        self._last_voice_ms = 0
        self._received_ms = 0
        self._counter = 0
        self._utterances = 0
        self._skipped_utterances = 0
        self._skipped_segments = 0
        self._waiting: deque[_FinalizedWindow] = deque()
        self._recoveries: deque[_FinalizedWindow] = deque()
        self._recovery: _WindowRecovery | None = None
        self._finalizing: _FinalizedWindow | None = None
        self._publishing: _FinalizedWindow | None = None
        self._ready: list[_FinalizedWindow] = []
        self._recovery_privacy_spans: list[list[int | None]] = []
        self._voice_start = 0
        # Of those, the ones from utterances heard in privacy (hailing-station #735 S-1): kept
        # apart so the session's reported totals never count private speech.
        self._private_skipped_segments = 0
        self._private_skipped_utterances = 0
        self._end_of_turn = config.end_of_turn
        # Whether the end-of-turn model was already asked during the current pause, and
        # whether it judged the turn complete while the diarizer timeline lagged behind.
        self._end_of_turn_asked = False
        self._complete = False
        self._asr_process: subprocess.Popen | None = None
        # A joined turn spans at most two full utterance windows, so a long monologue
        # with short pauses is still decided in bounded time.
        self._merger = TurnMerger(
            config.turn_merge_gap_ms,
            2 * config.max_utterance_ms,
            self._emit,
            config.turn_break,
            reply_wait_ms=config.reply_wait_ms,
            tail_join_ms=config.tail_join_ms,
            unfinished=config.turn_unfinished,
            complete_hold_ms=config.turn_complete_hold_ms,
            collapse_restarts=config.turn_collapse_restarts,
            bridge_backchannels=config.turn_bridge_backchannels,
            wake_word=config.turn_wake_word,
            owner=config.turn_owner,
            owner_pause_ms=config.turn_owner_pause_ms,
        )

    @property
    def received_ms(self) -> int:
        return self._received_ms

    @property
    def skipped_segments(self) -> int:
        """whisper.cpp segments skipped this session for unusable offsets (see `_units`), not
        counting those in utterances heard in privacy."""
        inner = getattr(self._transcriber, "skipped_segments", 0)
        inner = inner if type(inner) is int and inner > 0 else 0
        return max(0, self._skipped_segments + inner - self._private_skipped_segments)

    @property
    def skipped_utterances(self) -> int:
        """Utterances suppressed whole this session for invalid or out-of-order timing.

        Counts the common unit check's suppressions plus windows with no salvageable
        whisper.cpp text after the single recovery attempt.
        """
        inner = getattr(self._transcriber, "suppressed_utterances", 0)
        inner = inner if type(inner) is int and inner > 0 else 0
        return max(0, self._skipped_utterances + inner - self._private_skipped_utterances)

    @property
    def buffered_audio_bytes(self) -> int:
        with self._recovery_lock:
            packets = (
                *self._waiting,
                *self._recoveries,
                *self._ready,
                self._finalizing,
                self._publishing,
            )
            owned = {id(packet.pcm): packet.pcm for packet in packets if packet is not None}
            recovery = self._recovery
            if recovery is not None:
                owned[id(recovery.packet.pcm)] = recovery.packet.pcm
                owned[id(recovery.pcm)] = recovery.pcm
            held = sum(len(pcm) for pcm in owned.values())
            return len(self._partial) + len(self._utterance) + sum(map(len, self._pre_roll)) + held

    def push_pcm16(self, pcm: bytes) -> None:
        if self.closed or self.failed:
            raise LiveAudioError("Audio session is closed")
        try:
            self._drain_results()
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

    def reset_speakers(self, private_until_bytes: int | None = None, *, renew: bool = True) -> None:
        """Start a fresh diarizer once nothing is open (hailing-station #735 S-1): called when
        privacy ends, so no speaker channel or state a private voice made outlives it. Deferred
        to the first frame with no open utterance or held turn, so a turn under way keeps its
        labels; its timeline continues on the stream clock. The deferral is bounded (round 21):
        `SPEAKER_RENEWAL_LIMIT_MS` (10 s of stream audio) after the first end reached since the
        last renewal (round 22: a later end never restarts it), the renewal is forced, once any
        later end is reached too, and a turn then under way is published with no speaker label.

        `private_until_bytes` is the stream byte offset privacy ended at (by default, the end
        of the audio pushed so far), exact to the sample (round 19). Only samples
        since the on edge are silenced; public audio before that edge survives."""
        if private_until_bytes is None:
            private_until_bytes = self._received_ms * BYTES_PER_MS + len(self._partial)
        private_until_bytes += private_until_bytes % 2  # a whole sample
        with self._private_lock:
            self._private_until = max(private_until_bytes, self._private_until or 0)
            span = (self._privacy_start_bytes, private_until_bytes)
            if span[0] < span[1] and span not in self._private_ranges:
                self._private_ranges.append(span)
            # Under the same lock as the end (round 21): a renewal taken meanwhile never clears
            # the request this end makes.
            if renew and self._diarizer_renewable:
                self._speaker_reset = True

    def begin_privacy(self, private_from_bytes: int) -> None:
        """Stamp the exact on edge; discard recovery windows spanning it."""
        with self._private_lock:
            self._privacy_start_bytes = private_from_bytes
        uncertain = self._discard_private_recovery(private_from_bytes, None)
        with self._private_lock:
            self._privacy_uncertain |= uncertain

    def _discard_private_recovery(self, start: int, end: int | None) -> bool:
        """Leave ordered empty barriers, never a private retry or held result.

        No host callbacks or waits under the lifecycle lock. A running decoder may
        own native copies we cannot wipe, so its cancellation also requires full forget.
        """
        active = None
        uncertain = False
        with self._recovery_lock:
            spans = self._recovery_privacy_spans
            if end is not None and spans and spans[-1] == [start, None]:
                if start < end:
                    spans[-1][1] = end
                else:
                    spans.pop()  # empty privacy covers no samples
            elif end is None or start < end:
                spans.append([start, end])
            if len(spans) > ASR_PRIVACY_SPANS:
                # Like the publisher's bounded history, conservatively cover the gap.
                spans[0][1] = spans.pop(1)[1]
            packets = (
                *self._waiting,
                *self._recoveries,
                *self._ready,
                self._finalizing,
                self._publishing,
            )
            for packet in {id(p): p for p in packets if p is not None}.values():
                if (end is None or packet.offset * BYTES_PER_MS < end) and (
                    (packet.offset + packet.duration_ms) * BYTES_PER_MS > start
                ):
                    packet.discarded = True
                    packet.retry_pending = False
                    packet.barrier_after = "privacy"
                    packet.release()
                    uncertain |= packet is self._finalizing or packet is self._publishing
                    if self._recovery is not None and self._recovery.packet is packet:
                        active = self._recovery
                        uncertain |= not active.done.is_set()
                        active.cancelled.set()
                        if active.result is not None:
                            active.result.units.clear()
                        active.result = None
                        if active._initial is not None:
                            active._initial.units.clear()
            # Queued retries need no worker; keep the active slot until it is reaped.
            self._recoveries = deque(
                p
                for p in self._recoveries
                if not p.discarded or (self._recovery and p is self._recovery.packet)
            )
        if active is not None:
            # cancel() sees the flag set above; explicitly take this owned child.
            with self._recovery_lock:
                process = active.process
            if process is not None:
                with contextlib.suppress(OSError):
                    process.kill()
        return uncertain

    def _discard_late_private_window_locked(self, packet: _FinalizedWindow) -> None:
        """Cover a result created/admitted after the boundary's queue snapshot.

        Active privacy still permits local exit recognition. Only a crossing on edge
        or an already closed private interval invalidates a newly admitted window.
        """
        left = packet.offset * BYTES_PER_MS
        right = (packet.offset + packet.duration_ms) * BYTES_PER_MS
        if any(
            right > start
            and (
                left < start if end is None else left < end and end > packet.private_cleared_through
            )
            for start, end in self._recovery_privacy_spans
        ):
            packet.discarded = True
            packet.barrier_after = "privacy"
            packet.release()

    def end_privacy(self, private_until_bytes: int) -> tuple[int, str | None]:
        """Silence buffered private audio; audit what actually entered the diarizer.

        Pending pushes are audited without waiting for native code. Unknown predicates
        and unaccountable in-flight decoder copies also require renewal.
        """
        self.reset_speakers(private_until_bytes, renew=False)
        recovery_uncertain = self._discard_private_recovery(
            self._privacy_start_bytes, private_until_bytes
        )
        with self._private_lock:
            count = self._diarizer_private_frames
            # The controller accepts a chunk before the audio thread processes it.
            # A sub-frame tail is safe: finish's cutoff silences it before any push.
            # Complete accepted frames still awaiting their verdict are not certified.
            audited_bytes = self._diarizer_audited_bytes + min(len(self._partial), FRAME_BYTES - 2)
            reason = (
                "private_frames"
                if count
                else "privacy_unknown"
                if (
                    self._privacy_uncertain
                    or recovery_uncertain
                    or self._diarizer_pending
                    or private_until_bytes > audited_bytes
                    or self.config.private is None
                )
                else None
            )
            self._diarizer_private_frames = 0
            self._privacy_uncertain = False
        return count, reason

    def _diarizer_frame(self, frame: bytes, private: bool) -> bytes:
        # Same PCM16 format and byte length: native time never skips a private frame.
        return bytes(len(frame)) if private else frame

    def _private_cutoff(self, pcm: bytes, final: bool = False) -> bytes:
        """Silence only the private intervals in the next audio and buffered prefixes
        (#735 S-1, rounds 18-19). Once the end falls within or before it (or at
        `finish`, `final`), silence those intervals in the
        pre-roll kept between turns and the start of an utterance open across the end, all
        before `pcm` reaches the diarizer, the end-of-turn model or ASR, so no audio heard
        while private is processed with public speech. Silence, not removal, keeps the stream
        offsets. The end is taken under the lock, so a later one set meanwhile is kept."""
        start = self._received_ms * BYTES_PER_MS
        with self._private_lock:
            until = self._private_until
            if until is None or (until >= start + len(pcm) and not final):
                return pcm
            self._private_until = None
            ranges, self._private_ranges = self._private_ranges, []
        if self._speaker_reset and self._end_reached_ms is None:
            # The first end not yet renewed for (round 22): a later one never restarts the bound.
            self._end_reached_ms = self._received_ms
        self._settle_held_at_privacy_end()
        first = start - FRAME_BYTES * len(self._pre_roll)
        frames = list(self._pre_roll)
        self._pre_roll.clear()
        for index, frame in enumerate(frames):
            self._pre_roll.append(self._silence_ranges(frame, first + FRAME_BYTES * index, ranges))
        self._utterance[:] = self._silence_ranges(
            self._utterance, self._utterance_start * BYTES_PER_MS, ranges
        )
        self._private_cleared_through = max(
            self._private_cleared_through, max((end for _, end in ranges), default=0)
        )
        return self._silence_ranges(pcm, start, ranges)

    @staticmethod
    def _silence_ranges(pcm, offset: int, ranges: list[tuple[int, int]]) -> bytes:
        result = bytearray(pcm)
        for start, end in ranges:
            left, right = max(0, start - offset), min(len(result), end - offset)
            if left < right:
                result[left:right] = bytes(right - left)
        return bytes(result)

    def _settle_held_at_privacy_end(self) -> None:
        """#242 round 27 (Codex P1): what the merger holds when the privacy end is reached was
        all heard before it. If any of it, the held turn or a bridged backchannel, was heard in
        privacy (`LiveConfig.private`; unknown counts), all of it is discarded, words and
        all, so no speech after the end can be joined to it: a joined turn would overlap the
        privacy span and be dropped whole, losing the first public request. Otherwise it is
        released now, on its own, for the same reason. The end-of-turn judgement of an
        utterance open across the end was made on private audio, so it is made again."""
        self._end_of_turn_asked = self._complete = False
        self._queue_merger_barrier("privacy")

    def _queue_merger_barrier(self, kind: str) -> None:
        """Audio-owner event, ordered after all finalized results heard before it.

        A held result can be absent from the merger for the whole privacy interval
        or input stall. Carry its boundary with it instead of settling an empty merger.
        Host callbacks stay outside the lifecycle lock.
        """
        with self._recovery_lock:
            if self.closed:
                return
            if self._waiting:
                tail = self._waiting[-1]
                if tail.barrier_after != "privacy":
                    tail.barrier_after = kind
                return
        self._apply_merger_barrier(kind)

    def _apply_merger_barrier(self, kind: str | None) -> None:
        if kind is None or self.closed:
            return
        if kind == "flush":
            self._merger.flush()
            return
        spans = self._merger.held_spans()
        if not spans:
            return
        if self.config.private is None or any(self._heard_private(*span) for span in spans):
            self._merger.discard()
        else:
            self._merger.flush()

    def _take_renewal(self) -> bool:
        """Claims a requested renewal once its privacy end is reached (round 21): the request
        and the end are read and the request cleared under one lock, so an end stamped while a
        frame is taken is never lost."""
        with self._private_lock:
            if not self._speaker_reset or self._private_until is not None:
                return False
            self._speaker_reset = False
            return True

    def _renew_diarizer(self) -> None:
        offset = self._received_ms
        self._end_reached_ms = None
        with self._private_lock:
            # Audit from frames handled after the off edge belongs to the old stream too.
            self._diarizer_private_frames = 0
            self._privacy_uncertain = False
        if self._utterance or self._merger.holding or self._waiting:
            # Forced mid-turn (round 21): the old diarizer's labels for it must not cross over.
            self._unlabelled_before_ms = offset
        old, self._diarizer = (
            self._diarizer,
            _OffsetDiarizer(
                _select(self.config.diarizer, lambda value: _Diarizer(value), self.config, "push"),
                offset,
            ),
        )
        with contextlib.suppress(Exception):
            old.close()
        if self.on_speakers_renewed is not None:
            self.on_speakers_renewed()

    def _diarizer_privacy(self, duration_ms: int) -> tuple[bool, bool]:
        # Register before asking a predicate that may wait for the controller lock.
        # An exit racing an unfinished verdict/substitution cannot certify preservation.
        with self._private_lock:
            self._diarizer_pending += 1
        if self.config.private is None:
            return False, False
        try:
            state = self.config.private(self._received_ms, self._received_ms + duration_ms)
            return state is not False, type(state) is not bool
        except Exception:  # noqa: BLE001 - no predicate content escapes
            return True, True

    def _push_diarizer(self, pcm: bytes, private: bool, uncertain: bool) -> None:
        diarizer_pcm = self._diarizer_frame(pcm, private)
        with self._private_lock:
            self._privacy_uncertain |= uncertain
            if private:
                self._diarizer_private_frames += any(diarizer_pcm)
        # Native push can stall. Exit must see it as pending without waiting on it
        # under the controller lock, which Stop needs to set its cancellation flag.
        self._diarizer.push(diarizer_pcm)
        with self._private_lock:
            self._diarizer_pending -= 1
            self._diarizer_audited_bytes += len(pcm)

    def _frame(self, frame: bytes) -> None:
        private, uncertain = self._diarizer_privacy(20)
        if self.config.private is not None:
            if private != self._frame_private:
                # Observe privacy starts as well as ends on the received-audio clock.
                # No frame between toggles means no private audio and no new text.
                self._queue_merger_barrier("privacy")
                self._frame_private = private
        if self._private_until is not None:
            frame = self._private_cutoff(frame)
        # Renewed only once the privacy end is reached (#242 round 20): the controller counts a
        # chunk before pushing it, so the end can lie ahead, and the fresh diarizer must never
        # be fed a frame heard while private.
        # Forced once the end was reached SPEAKER_RENEWAL_LIMIT_MS ago (round 21).
        if (
            self._speaker_reset
            and (
                (not self._utterance and not self._merger.holding and not self._waiting)
                or (
                    self._end_reached_ms is not None
                    and self._received_ms - self._end_reached_ms >= SPEAKER_RENEWAL_LIMIT_MS
                )
            )
            and self._take_renewal()
        ):
            self._renew_diarizer()
        self._push_diarizer(frame, private, uncertain)
        frame_start = self._received_ms
        self._received_ms += 20
        values = _pcm_samples(frame)
        rms = math.sqrt(sum(value * value for value in values) / len(values)) / 32768
        voiced = rms >= self.config.energy_threshold
        if voiced and not self._utterance:
            self._voice_start = frame_start
            self._utterance_start = frame_start - len(self._pre_roll) * 20
            self._utterance.extend(b"".join(self._pre_roll))
            self._pre_roll.clear()
            if self._merger.holding and not self._waiting:
                # Followed by detected speech, whether or not it yields text (#96).
                self._merger.heard(frame_start)
        if self._utterance or voiced:
            self._utterance.extend(frame)
            if voiced:
                self._last_voice_ms = self._received_ms
                self._end_of_turn_asked = self._complete = False
            if (
                self._received_ms - self._last_voice_ms >= self.config.hangover_ms
                or self._received_ms - self._utterance_start >= self.config.max_utterance_ms
            ):
                self._finalize(self._diarizer.segments(), complete=self._complete)
            elif self._complete or self._ask_end_of_turn():
                self._finalize_when_attributed()
        else:
            self._pre_roll.append(frame)
        if self._merger.holding and not self._waiting:
            self._merger.due(self._received_ms, self._utterance_start if self._utterance else None)

    def release_pending(self) -> None:
        """Emit a turn held for a possible continuation now, for example when input stalls."""
        if self.closed or self.failed:
            return
        try:
            self._drain_results()
            self._queue_merger_barrier("flush")
        except Exception:
            self.failed = True
            self.close()
            raise

    def _ask_end_of_turn(self) -> bool:
        """Whether the end-of-turn model judges the current pause the end of the turn."""
        silence = self._received_ms - self._last_voice_ms
        if (
            self._end_of_turn is None
            or self._end_of_turn_asked
            or silence < self.config.end_of_turn_silence_ms
        ):
            return False
        self._end_of_turn_asked = True
        try:
            probability = self._end_of_turn(bytes(self._utterance))
            if type(probability) not in (int, float) or not 0 <= probability <= 1:
                raise ValueError
        except Exception:
            if self.closed:
                raise LiveAudioError("Audio session was stopped") from None
            # A failing model never ends the session: the silence rules take over.
            self._end_of_turn = None
            # Withheld for an utterance heard in privacy too: even its failure is about that speech.
            self._diagnostic(
                "end-of-turn model unavailable; using silence end-of-turn",
                span=(self._utterance_start, self._received_ms),
            )
            return False
        complete = probability >= self.config.end_of_turn_threshold
        # Content-free: a probability and stream timing only, never audio or text.
        self._diagnostic(
            f"end_of_turn p={probability:.3f} silence_ms={silence}"
            f" outcome={'complete' if complete else 'wait'}",
            span=(self._utterance_start, self._received_ms),
        )
        return complete

    def _finalize_when_attributed(self) -> None:
        """Finalize a turn judged complete once its last words can be attributed.

        A streaming diarizer's timeline trails the audio (measured 0.4-1.2 s after speech
        stops), and words past its end would lose their speaker and split the turn. So the
        utterance stays open, unchanged, until the timeline reaches its last voiced audio;
        the hangover still bounds the wait. An utterance-level diarizer labels the whole
        utterance when it is finalized, so it is finalized at once.
        """
        self._complete = True
        timeline = self._diarizer.segments()
        if getattr(self._diarizer, "speaker_provenance", "diarization-timeline") == (
            "diarization-timeline"
        ) and not any(
            # A segment spanning the last voiced audio, not merely one ending after it: a
            # later, disjoint segment says nothing about the words before it.
            isinstance(segment, dict)
            and type(segment.get("start_ms")) is int
            and type(segment.get("end_ms")) is int
            and segment["start_ms"] < self._last_voice_ms <= segment["end_ms"]
            for segment in timeline
        ):
            return
        self._diagnostic(
            f"end_of_turn finalized silence_ms={self._received_ms - self._last_voice_ms}",
            span=(self._utterance_start, self._received_ms),
        )
        self._finalize(timeline, complete=True)

    def _diagnostic(self, message: str, *, span: tuple[int, int] | None = None) -> None:
        """Report `message`; one about the utterance heard over `span` is withheld if that
        audio was heard in privacy mode (`LiveConfig.private`)."""
        if self.config.report is None or (span is not None and self._heard_private(*span)):
            return
        with contextlib.suppress(Exception):
            self.config.report(message)

    def _heard_private(self, start_ms: int, end_ms: int) -> bool:
        private = self.config.private
        if private is None:
            return False
        try:
            return private(start_ms, end_ms) is not False
        except Exception:  # noqa: BLE001 - fail closed: withhold the diagnostic
            return True

    def _finalize(self, timeline: list[dict[str, Any]], complete: bool = False) -> None:
        self._complete = False
        if not self._utterance:
            return
        _check_timeline(timeline, self._received_ms, self._timeline_limit)
        offset = self._utterance_start
        before = (self.skipped_segments, self.skipped_utterances)
        self._utterances += 1
        with self._recovery_lock:
            pcm = bytearray(self._utterance)
            packet = _FinalizedWindow(
                pcm,
                offset,
                self._received_ms,
                self._voice_start,
                self._utterances,
                [
                    dict(segment)
                    for segment in timeline
                    if segment["start_ms"] <= offset + len(pcm) // BYTES_PER_MS
                    and segment["end_ms"] >= offset
                ],
                getattr(self._diarizer, "speaker_provenance", "diarization-timeline"),
                complete,
                [],
                None,
                before,
                len(pcm) // BYTES_PER_MS,
                bool(timeline),
                private_cleared_through=self._private_cleared_through,
            )
            self._finalizing = packet
            self._discard_late_private_window_locked(packet)
        try:
            # Preserve the replaceable transcriber's bytes contract. If a boundary
            # races this synchronous/native copy, the audit requires full forget.
            units = (
                []
                if packet.discarded
                else self._transcriber.transcribe(bytes(pcm), self._register_asr)
            )
            window = getattr(self._transcriber, "_window", None)
            if isinstance(window, _RecognizerWindow):
                self._transcriber._window = None
            else:
                window = None
            with self._recovery_lock:
                if packet.discarded or self.closed:
                    if isinstance(units, list):
                        units.clear()
                    if window is not None:
                        window.units.clear()
                else:
                    packet.units, packet.window = units, window
            if self.closed:
                raise LiveAudioError("Audio session was stopped")
        finally:
            self._asr_process = None
        self._utterance.clear()
        self._pre_roll.clear()
        self._drain_results()
        try:
            self._queue_window(packet)
        finally:
            with self._recovery_lock:
                self._finalizing = None

    def _queue_window(self, packet: _FinalizedWindow) -> None:
        private_loss = None
        while True:
            queued = False
            with self._recovery_lock:
                if self.closed:
                    packet.release()
                    return
                self._discard_late_private_window_locked(packet)
                self._recovery_privacy_spans = [
                    span
                    for span in self._recovery_privacy_spans
                    if span[1] is None or span[1] > packet.offset * BYTES_PER_MS
                ]
                # Recompute admission after every host callback. Privacy may have
                # discarded either this packet or the former overflow tail meanwhile.
                overflow = not packet.discarded and bool(
                    self._waiting and self._waiting[-1].overflow
                )
                invalid = (
                    packet.window is not None and not packet.window.units and packet.window.rejected
                )
                retry = not overflow and invalid and len(self._recoveries) < ASR_RECOVERY_WINDOWS
                if not overflow and not retry and not packet.discarded:
                    if invalid:
                        self._lose_window(packet, reason=5, compact=True)
                    held = [entry for entry in self._waiting if not entry.retry_pending]
                    overflow = bool(self._waiting) and (
                        len(held) >= ASR_HELD_WINDOWS
                        or sum(entry.held_bytes() for entry in held) + packet.held_bytes()
                        > ASR_HELD_BYTES
                    )
                if overflow and private_loss is not None:
                    packet.private_loss = private_loss
                    self._append_overflow(packet)
                    return
                if not overflow:
                    if retry:
                        packet.retry_pending = True
                        self._recoveries.append(packet)
                    if self._waiting or packet.retry_pending:
                        self._waiting.append(packet)
                        queued = True
                    break
            # The privacy callback may take a host lock: never invoke it while holding
            # the lifecycle lock that Stop needs. No queue/window decision survives it.
            private_loss = self._heard_private(packet.offset, packet.offset + packet.duration_ms)
        if queued:
            self._drain_results()
        else:
            self._commit_window(packet)

    def _lose_window(self, packet: _FinalizedWindow, *, reason: int, compact: bool) -> None:
        window = packet.window
        if window is None:
            window = _RecognizerWindow([], {0})
        reasons = list(window.reasons)
        if reason != 5 or not reasons[5]:
            reasons[reason] += 1
        packet.window = replace(window, units=[], gaps={0}, reasons=tuple(reasons))
        packet.units = []
        packet.retry_pending = False
        if compact:
            if isinstance(packet.pcm, bytearray):
                packet.pcm[:] = bytes(len(packet.pcm))
            packet.pcm = bytearray()
            packet.timeline = []

    def _append_overflow(self, packet: _FinalizedWindow) -> None:
        self._lose_window(packet, reason=5, compact=True)
        if (
            self._waiting
            and self._waiting[-1].overflow
            and not self._waiting[-1].discarded
            and self._waiting[-1].window is not None
        ):
            gap = self._waiting[-1]
            gap.duration_ms = packet.offset + packet.duration_ms - gap.offset
            gap.through_ms = packet.through_ms
            gap.last_number = packet.number
            gap.loss_count += 1
            gap.private_loss |= packet.private_loss
            gap.window.reasons = tuple(
                a + b for a, b in zip(gap.window.reasons, packet.window.reasons)
            )
            gap.window.segments += packet.window.segments
            gap.window.elapsed_ms += packet.window.elapsed_ms
        else:
            packet.overflow = True
            self._waiting.append(packet)

    def _expire_recovery_locked(self, recovery: _WindowRecovery) -> None:
        packet = recovery.packet
        if packet.retry_pending:
            self._lose_window(packet, reason=4, compact=False)
            packet.window.retried = True
            packet.window.elapsed_ms += max(0, round((time.monotonic() - recovery.started) * 1000))

    def _start_recovery_locked(self) -> None:
        if self.closed or self._recovery is not None or not self._recoveries:
            return
        recovery = _WindowRecovery(
            self.config, self._recoveries[0], self._recovery_lock, self._recovery_shutdown
        )
        self._recovery = recovery
        recovery.start()

    def _drain_results(self, *, wait: bool = False) -> None:
        while True:
            with self._recovery_lock:
                if self.closed:
                    return
                recovery = self._recovery
            if wait and recovery is not None:
                # Only EOF waits. No joins, waits or host callbacks under the lock.
                recovery.done.wait(max(0.0, recovery.expires - time.monotonic()))
            expired = None
            with self._recovery_lock:
                if self.closed:
                    return
                if recovery is not None:
                    if recovery.done.is_set():
                        packet = recovery.packet
                        if packet.retry_pending:
                            if recovery.result is not None:
                                packet.window = recovery.result
                                packet.units = recovery.result.units
                                packet.retry_pending = False
                            else:
                                self._lose_window(packet, reason=4, compact=False)
                        assert self._recoveries[0] is packet
                        self._recoveries.popleft()
                        self._recovery = None
                    elif time.monotonic() >= recovery.expires:
                        self._expire_recovery_locked(recovery)
                        expired = recovery
                self._start_recovery_locked()
                ready = []
                while self._waiting and not self._waiting[0].retry_pending:
                    ready.append(self._waiting.popleft())
                self._ready = ready
                pending = any(packet.retry_pending for packet in self._recoveries)
                active = self._recovery
            if expired is not None:
                expired.cancel()
            for packet in ready:
                self._commit_window(packet)
            with self._recovery_lock:
                self._ready = []
            if not wait or not pending:
                return
            if active.cancelled.is_set() and not active.done.is_set():
                if not active.done.wait(ASR_CLEANUP_SECONDS):
                    raise LiveAudioError("Local recovery worker could not stop")

    def _commit_window(self, packet: _FinalizedWindow) -> None:
        with self._recovery_lock:
            self._publishing = packet
            self._discard_late_private_window_locked(packet)
        try:
            self._publish_window(packet)
            self._apply_merger_barrier(packet.barrier_after)
        finally:
            with self._recovery_lock:
                self._publishing = None

    def _publish_window(self, packet: _FinalizedWindow) -> None:
        # Privacy-off/Stop release the packet under this same lock. Never iterate or
        # slice its shared units outside it; cuts remain valid only while not discarded.
        with self._recovery_lock:
            if self.closed or packet.discarded:
                packet.release()
                return
            pcm, offset, timeline = packet.pcm, packet.offset, packet.timeline
            window, complete, provenance = packet.window, packet.complete, packet.provenance
            checked = _checked_units(packet.units, packet.duration_ms)
            unit_count = len(checked) if checked is not None else 0
        if window is not None:
            skipped = window.rejected
            self._skipped_segments += skipped
            suppressed = checked is None or (
                not unit_count and bool(window.rejected or window.retried or window.reasons[5])
            )
            self._skipped_utterances += packet.loss_count if suppressed else 0
        else:
            if checked is None:
                self._skipped_utterances += 1
            skipped = self.skipped_segments - packet.before[0]
            suppressed = self.skipped_utterances > packet.before[1]
        private = (skipped or suppressed) and (
            packet.private_loss or self._heard_private(offset, offset + packet.duration_ms)
        )
        if private:
            self._private_skipped_segments += skipped
            self._private_skipped_utterances += packet.loss_count if suppressed else 0
        if self._merger.holding:
            self._merger.heard(packet.voice_start)
        if (
            window is not None
            and (window.repaired or window.rejected or window.retried or window.reasons[5])
            and not private
        ):
            outcome = "lost" if suppressed else "redecoded" if window.retried else "salvaged"
            # Fixed tokens and numbers only. The tool's report sink adds `rightyo:` on stderr.
            extent = (
                f" last_window_id={packet.last_number} windows_lost={packet.loss_count}"
                if packet.loss_count > 1
                else ""
            )
            self._diagnostic(
                f"asr_window outcome={outcome} window_id={packet.number}{extent}"
                f" start_ms={offset} end_ms={offset + packet.duration_ms}"
                f" segments={window.segments} retry_segments={window.retry_segments}"
                f" units={unit_count} repaired={window.repaired} rejected={window.rejected}"
                f" reasons={','.join(map(str, window.reasons))}"
                f" asr_elapsed_ms={window.elapsed_ms}",
                span=(offset, offset + packet.duration_ms),
            )
        elif (skipped or suppressed) and self.config.report is not None and not private:
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
            # A held turn must not be joined across the suppressed utterance to a later
            # one ("Do not stop." published as "Do stop."): release it on its own now.
            self._merger.flush()
            return
        gaps = window.gaps if window is not None else set()
        cuts = sorted({0, unit_count} | gaps)
        for left, right in zip(cuts, cuts[1:]):
            with self._recovery_lock:
                if self.closed or packet.discarded:
                    return
                units = checked[left:right]
            if left in gaps:
                self._merger.flush()
            self._publish_units(
                units,
                timeline,
                offset,
                pcm,
                provenance,
                number=packet.number,
                timeline_present=packet.timeline_present,
                complete=complete and right == unit_count and right not in gaps,
                packet=packet,
                loss_before=left in gaps,
                loss_after=right in gaps,
            )
        if self.closed or packet.discarded:
            return
        if unit_count in gaps:
            self._merger.flush()
        self._merger.due(packet.through_ms, None)

    def _publish_units(
        self,
        units: list[dict[str, Any]],
        timeline: list[dict[str, Any]],
        offset: int,
        pcm: bytes,
        provenance: str,
        *,
        number: int,
        timeline_present: bool,
        complete: bool,
        packet: _FinalizedWindow,
        loss_before: bool = False,
        loss_after: bool = False,
    ) -> None:
        """Attribute one uninterrupted run; neither edge inference nor merging crosses loss."""
        if self.closed or packet.discarded or not units:
            return
        # Touching segments overlap no duration, but still block edge inference.
        relevant = [segment for segment in timeline if segment["end_ms"] >= offset]
        attributed = []
        for unit in units:
            start, end = unit["start_ms"] + offset, unit["end_ms"] + offset
            attributed.append([unit, start, end, *_attribute(start, end, relevant)])
        coverage = self.config.word_label_coverage
        if (
            coverage
            and provenance == "diarization-timeline"
            and not any(entry[3] is not None for entry in attributed)
        ):
            # No word has a strict label (#220): let one speaker's coverage label them.
            # Only segments that start inside this utterance (pre-roll included) count: a
            # streaming diarizer switches speaker late, so an earlier speaker's segment can
            # run on over the next person's first word before their own segment arrives.
            # A carried-over segment may still block a label: any other speaker touching
            # the word, carried over or not, leaves it unlabelled.
            own = [segment for segment in relevant if segment["start_ms"] >= offset]
            for entry in attributed:
                # The overlap flag stays the strict pass's, over every segment.
                start, end = entry[1], entry[2]
                label = _attribute(start, end, own, coverage)[0]
                touching = {
                    "Speaker " + _speaker_label(segment["speaker"])
                    for segment in relevant
                    if segment["start_ms"] < end and segment["end_ms"] > start
                }
                if label is not None and touching == {label}:
                    entry[3] = label
        inferred: set[int] = set()
        if attributed and self.config.edge_attribution_ms and provenance == "diarization-timeline":
            # Edge inference may use evidence only inside this uninterrupted run.
            # Boundary-touching segments still block a competing label. At a loss,
            # evidence entirely on its other side cannot supply that label itself.
            # Keep strict attribution/overlap checks on the original timeline.
            run_start, run_end = attributed[0][1], attributed[-1][2]
            edges = [
                dict(
                    segment,
                    start_ms=max(run_start, segment["start_ms"]),
                    end_ms=min(run_end, segment["end_ms"]),
                )
                for segment in relevant
                if segment["start_ms"] <= run_end and segment["end_ms"] >= run_start
            ]
            evidence = [
                segment
                for segment in edges
                if (not loss_before or segment["end_ms"] > run_start)
                and (not loss_after or segment["start_ms"] < run_end)
            ]
            inferred = _attribute_edges(
                attributed, evidence, self.config.edge_attribution_ms, blockers=edges
            )
        groups: list[dict[str, Any]] = []
        for index, (unit, start, end, speaker, overlap) in enumerate(attributed):
            if speaker is not None and provenance == "diarization-utterance":
                # Per-request labels are namespaced by utterance so that equal labels
                # from independent requests can never be merged into one participant.
                speaker = utterance_scoped_speaker(number, speaker)
            diarized = (
                index not in inferred
                and speaker is not None
                and not overlap
                and provenance == "diarization-timeline"
            )
            if groups and (groups[-1]["speaker"], groups[-1]["overlap"]) == (speaker, overlap):
                group = groups[-1]
                if diarized and group["diarized_head"] == len(group["text"]):
                    group["diarized_head"] += len(unit["text"])
                groups[-1]["text"] += unit["text"]
                groups[-1]["end_ms"] = max(groups[-1]["end_ms"], end)
            else:
                groups.append(
                    {
                        "text": unit["text"],
                        "diarized_head": len(unit["text"]) if diarized else None,
                        "start_ms": start,
                        "end_ms": end,
                        "speaker": speaker,
                        "overlap": overlap,
                    }
                )
            if index in inferred:
                # Edge-attributed words make the group's label partly inferred (#137).
                groups[-1]["inferred"] = True
        kept = [group for group in groups if group["text"].strip()]
        for index, group in enumerate(kept):
            if self.closed or packet.discarded:
                return
            head = group["diarized_head"]
            leading = len(group["text"]) - len(group["text"].lstrip())
            group["text"] = group["text"].strip()
            group["diarized_head"] = (
                min(len(group["text"]), head - leading)
                if head is not None and head > leading
                else None
            )
            group["speaker_provenance"] = provenance if timeline_present else "unknown"
            if self.config.on_labelled_spans is not None and provenance == "diarization-timeline":
                # Only the label's own, unoverlapped timeline audio (#148); joined turns
                # carry the union of their fragments' pieces (see `TurnMerger`).
                group["labelled_spans"] = (
                    []
                    if group["overlap"]
                    else _labelled_spans(
                        group["speaker"], group["start_ms"], group["end_ms"], relevant
                    )
                )
            # Only the speaker who just paused was judged finished; earlier groups in the
            # utterance were followed by other speech already.
            private_turn = self.config.on_private_turn_audio is not None and self._heard_private(
                group["start_ms"], group["end_ms"]
            )
            # The host predicate can yield to privacy-off, releasing this packet's
            # PCM/units. Drop our attributed snapshot before either publication path.
            with self._recovery_lock:
                if self.closed or packet.discarded:
                    return
            if private_turn:
                # Privacy wins a continuation tie: discard the held public prefix without
                # publishing it, while keeping only this private sentence's PCM for exit.
                self._merger.discard()
                start = (group["start_ms"] - offset) * BYTES_PER_MS
                end = (group["end_ms"] - offset) * BYTES_PER_MS
                self._emit(group, memoryview(pcm)[start:end])
            else:
                self._merger.offer(group, complete=complete and index == len(kept) - 1)

    def _emit(self, group: dict[str, Any], private_pcm: memoryview | None = None) -> None:
        """Number and emit one finalized (possibly joined) turn; ids follow emission order."""
        if self.closed:
            return
        self._counter += 1
        gap = group.pop("post_turn_gap", None)
        inferred = group.pop("inferred", False) is True
        spans = group.pop("labelled_spans", None)
        head = group.pop("diarized_head", None)
        if group["start_ms"] < self._unlabelled_before_ms or self._heard_private(
            group["start_ms"], group["end_ms"]
        ):
            # Under way at a forced renewal (round 21): labelled, wholly or partly, by the old
            # diarizer, or joined to its label, so published unlabelled with no label evidence.
            # Private PCM was replaced by silence: even a stale timeline's owner label
            # cannot authenticate that voice. Full exits use the isolated sentence check.
            group["speaker"], inferred, spans, head = None, False, None, None
        turn = Turn(
            session_id=self.config.session_id,
            utterance_id=f"live-{self._counter}",
            revision=1,
            start_ms=group["start_ms"],
            end_ms=group["end_ms"],
            text=group["text"],
            speaker_id=group["speaker"],
            finalized=True,
            overlap=group["overlap"],
            recognizer_id=self._transcriber.recognizer_id,
            provenance=self.config.provenance,
            speaker_provenance=group["speaker_provenance"],
        )
        if gap is not None and self.config.on_post_turn_gap is not None:
            self.config.on_post_turn_gap(turn.utterance_id, gap)
        if inferred and self.config.on_inferred is not None:
            self.config.on_inferred(turn.utterance_id)
        if self.config.on_diarized_head is not None:
            self.config.on_diarized_head(turn.utterance_id, head)
        if spans is not None and self.config.on_labelled_spans is not None:
            self.config.on_labelled_spans(turn.utterance_id, tuple(spans))
        if private_pcm is not None:
            self.config.on_private_turn_audio(turn, private_pcm)
        self.on_turn(turn)

    def finish(self) -> None:
        if self.closed:
            return
        try:
            # Any privacy end not reached yet is applied here, to the tail and what is buffered.
            pcm = self._private_cutoff(bytes(self._partial), final=True)
            self._partial.clear()
            if pcm:
                # Process actual tail only. Synthetic zero-padding must not enter native history.
                private, uncertain = self._diarizer_privacy(math.ceil(len(pcm) / BYTES_PER_MS))
                self._push_diarizer(pcm, private, uncertain)
                if self._utterance:
                    self._utterance.extend(pcm)
            # Flush only for a pending utterance: nothing else is emitted, and a hosted
            # diarizer must not send audio that no turn will use.
            if self._utterance:
                self._finalize(self._diarizer.finish())
            self._drain_results(wait=True)
            self._merger.flush()
        except Exception:
            self.failed = True
            raise
        finally:
            self.close()

    def _register_asr(self, process: subprocess.Popen) -> None:
        with self._recovery_lock:
            stopped = self.closed
            if not stopped:
                self._asr_process = process
        if stopped:
            _terminate(process)
            raise LiveAudioError("Audio session was stopped")

    def close(self) -> None:
        with self._recovery_lock:
            first = not self.closed
            self.closed = True
            self._recovery_shutdown.set()
            recovery = self._recovery
            process = self._asr_process if first else None
            self._asr_process = None
            for packet in (
                *self._waiting,
                *self._recoveries,
                *self._ready,
                self._finalizing,
                self._publishing,
            ):
                if packet is None:
                    continue
                packet.release()
            self._waiting.clear()
            self._recoveries.clear()
        # Startup/publication cannot race this snapshot. Native cleanup and worker
        # joins must stay outside the lock needed by registration and the finalizer.
        try:
            if recovery is not None:
                recovery.cancel()
                recovery.join()
            if process is not None:
                _terminate(process)
        finally:
            with self._recovery_lock:
                if recovery is not None and recovery.done.is_set() and self._recovery is recovery:
                    self._recovery = None
            if first:
                self._merger.discard()
                self._partial.clear()
                self._pre_roll.clear()
                self._utterance.clear()
                self._diarizer.close()


if __name__ == "__main__":
    # `--native-worker LIBRARY MODEL [WINDOW_MS]`: without the window the default applies.
    if len(sys.argv) in (4, 5) and sys.argv[1] == "--native-worker":
        window = _worker_window(sys.argv[4]) if len(sys.argv) == 5 else TIMELINE_WINDOW_MS
        if window is None:
            raise SystemExit(2)
        raise SystemExit(_native_worker(sys.argv[2], sys.argv[3], window))
    raise SystemExit(2)
