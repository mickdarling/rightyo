"""Explicit loopback speech lab; startup never enables microphone or hosted inference."""

from __future__ import annotations

import contextlib
import hmac
import inspect
import json
import math
import queue
import secrets
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from rightyo.addressedness import (
    DEFAULT_REPLY_WAIT_MS,
    DEFAULT_SCENE,
    dismissal_shaped,
    reply_wait,
    scene_text,
    unfinished_shaped,
)
from rightyo.capture import CaptureError, MacMicrophoneCapture, StdinPcmCapture
from rightyo.contracts import (
    PROVENANCE,
    Acknowledgement,
    Addressing,
    ContractError,
    Conversation,
    DecisionEvent,
    Dismissal,
    RequestForming,
    SpeakerPriority,
    Turn,
    identifier,
)
from rightyo.credentials import CredentialError
from rightyo.live_audio import (
    MAX_TIMELINE_WINDOW_MS,
    MIN_TIMELINE_WINDOW_MS,
    TIMELINE_WINDOW_MS,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    RecognizerSandboxError,
    recognizer_dir,
    timeline_window,
)
from rightyo.live_speaker_id import (
    MAX_RING_MS,
    MIN_RING_MS,
    RING_MS,
    EnrolledRoles,
    PrivateTurnAudio,
    ShadowSpeakerId,
    speaker_ring_ms,
)
from rightyo.memory import MemorySessionLimitError, TranscriptMemory
from rightyo.pipeline import ReplayRunner
from rightyo.providers import (
    ConfiguredPriorityProvider,
    JevProvider,
    MockProvider,
    ProviderError,
    ProviderUnavailable,
    request_former_for,
    unavailable_decision,
)
from rightyo.smart_turn import EndOfTurn, SmartTurn, SmartTurnError
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.speech_backends import (
    HostedSpeechError,
    describe,
    diarizer_factory,
    diarizer_spec,
    is_hosted,
    is_synthetic,
    speech_summary,
    transcriber_factory,
    transcriber_spec,
    utterance_local_labels,
)
from rightyo.state_log import DecisionStateLog
from rightyo.tool_events import privacy_phrase
from rightyo.turn_merge import (
    DEFAULT_TURN_MERGE_GAP_MS,
    complete_hold,
    merge_gap,
    owner_pause,
    tail_join,
)

BROWSER_LEASE_SECONDS = 15
# Jev requests per demo session unless the session or configuration sets a cap. Live
# microphone and stdin sessions call Jev once per finalized turn and are uncapped by
# default (#75): the decision worker sends one request at a time, so the speech itself
# paces them, and the bounded decision queue fails closed if Jev falls behind.
DEFAULT_REQUEST_LIMIT = 20
LIVE_MODES = frozenset({"microphone", "stdin"})
# A transiently unavailable hosted decision (timeout, connection, HTTP 429/529/5xx, or a
# malformed answer, #77) degrades only its own turn (#71); this many in a row end the
# session as before.
MAX_CONSECUTIVE_DECISION_FAILURES = 5
# Observed post-turn gaps (#96) waiting for their turn's decision; a bound, not a queue.
MAX_PENDING_GAPS = 64
PCM_BYTES_PER_MS = 32
# Advertised on `session` started for stdin input: a separate top-level object, so the
# strictly validated capability set is unchanged. Turn provenance is host-declared.
STDIN_AUDIO_INPUT = {
    "source": "stdin",
    "encoding": "s16le",
    "sample_rate": 16000,
    "channels": 1,
}
LOCAL_ASSETS = ("whisper_executable", "whisper_model", "diarization_library", "diarization_model")


class PrototypeError(ValueError):
    """A safe configuration or control error."""


class DecisionConfigError(PrototypeError):
    """An invalid `decision` section; its message names the rule rather than a value."""


class RetentionConfigError(PrototypeError):
    """An invalid `audio_retention` section; its message names the rule, never the value."""


DECISION_PROVIDERS = ("mock", "jev")
AUDIO_RETENTION_KEYS = frozenset({"speaker_ring_seconds", "timeline_window_seconds"})


def audio_retention_spec(value: Any) -> tuple[int, int]:
    """The `audio_retention` section (#226) as (speaker ring ms, timeline window ms).

    Both keys are optional and independent; an absent key keeps its 60 s default. Each is
    a whole number of seconds within its bounds, and anything else is refused, never
    clamped. Memory only: neither buffer is ever written, and a change applies when
    RightyO next starts.
    """
    if not isinstance(value, dict) or not set(value) <= AUDIO_RETENTION_KEYS:
        raise RetentionConfigError(
            "The audio_retention section must be an object with only the optional keys "
            "speaker_ring_seconds and timeline_window_seconds"
        )
    ring = value.get("speaker_ring_seconds", RING_MS // 1000)
    window = value.get("timeline_window_seconds", TIMELINE_WINDOW_MS // 1000)
    try:
        if type(ring) is not int:
            raise ValueError
        ring_ms = speaker_ring_ms(ring * 1000)
    except ValueError:
        raise RetentionConfigError(
            "The audio_retention speaker_ring_seconds value must be a whole number of "
            f"seconds from {MIN_RING_MS // 1000} to {MAX_RING_MS // 1000}"
        ) from None
    try:
        if type(window) is not int:
            raise LiveAudioError("Invalid timeline window")
        window_ms = timeline_window(window * 1000)
    except LiveAudioError:
        raise RetentionConfigError(
            "The audio_retention timeline_window_seconds value must be a whole number of "
            f"seconds from {MIN_TIMELINE_WINDOW_MS // 1000} to {MAX_TIMELINE_WINDOW_MS // 1000}"
        ) from None
    return ring_ms, window_ms


def decision_spec(value: Any) -> bool:
    """Whether a configuration `decision` section selects hosted Jev decisions.

    The section is the file form of `listen --use-jev --allow-hosted`: both keys are
    required and exactly typed, and consent must match the provider, as on the CLI.
    """
    if not isinstance(value, dict):
        raise DecisionConfigError("The decision section must be an object")
    if (
        not {"provider", "allow_hosted"}
        <= set(value)
        <= {
            "provider",
            "allow_hosted",
            "max_requests",
            "scene",
            "attend_min_probability",
            "other_answered_blocks",
            "state_log",
            "request_context",
        }
    ):
        raise DecisionConfigError(
            "The decision section requires exactly the keys provider and allow_hosted, "
            "with an optional max_requests, scene, attend_min_probability, "
            "other_answered_blocks, state_log and request_context"
        )
    provider, allow_hosted = value["provider"], value["allow_hosted"]
    if type(provider) is not str or provider not in DECISION_PROVIDERS:
        raise DecisionConfigError("Unknown decision provider; expected mock or jev")
    if type(allow_hosted) is not bool:
        raise DecisionConfigError("The decision allow_hosted value must be true or false")
    if provider == "jev" and not allow_hosted:
        raise DecisionConfigError('The jev decision provider requires "allow_hosted": true')
    if provider == "mock" and allow_hosted:
        raise DecisionConfigError('"allow_hosted": true applies only to the jev decision provider')
    if "max_requests" in value:
        limit = value["max_requests"]
        if type(limit) is not int or limit < 1:
            raise DecisionConfigError(
                "The decision max_requests value must be a positive whole number"
            )
        if provider != "jev":
            raise DecisionConfigError("max_requests applies only to the jev decision provider")
    if "attend_min_probability" in value:
        bar = value["attend_min_probability"]
        if type(bar) not in {int, float} or not 0 <= bar <= 1:
            raise DecisionConfigError(
                "The decision attend_min_probability value must be a number from 0 to 1"
            )
        if provider != "jev":
            raise DecisionConfigError(
                "attend_min_probability applies only to the jev decision provider"
            )
    if "request_context" in value and type(value["request_context"]) is not bool:
        raise DecisionConfigError("The decision request_context value must be true or false")
    if "other_answered_blocks" in value and type(value["other_answered_blocks"]) is not bool:
        raise DecisionConfigError("The decision other_answered_blocks value must be true or false")
    if "state_log" in value:
        log = value["state_log"]
        if type(log) is not str or not log.startswith("/") or "\x00" in log:
            raise DecisionConfigError("The decision state_log value must be an absolute path")
        if any((parent / ".git").exists() for parent in Path(log).resolve().parents):
            raise DecisionConfigError(
                "The decision state_log must be stored outside the repository"
            )
        if provider != "jev":
            raise DecisionConfigError("state_log applies only to the jev decision provider")
    if "scene" in value:
        try:
            scene_text(value["scene"])
        except ContractError:
            raise DecisionConfigError(
                "The decision scene must be null or 1 to 1000 printable characters"
            ) from None
    return provider == "jev"


def _reject_constant(_value):
    raise ValueError("Invalid JSON constant")


def validate_session_budget(value: Any) -> int | None:
    """None means no session ceiling; otherwise a positive whole number of seconds."""
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise PrototypeError("Session budget must be a positive number of seconds")
    return value


# One speaker's share of a word that labels it when no word of the utterance is
# otherwise labelled (#220). Off until validated live; 0.5 is the suggested value.
DEFAULT_WORD_LABEL_COVERAGE = 0


@dataclass(frozen=True)
class PrototypeConfig:
    # Native assets are required only by native backends; explicit mocks need none.
    whisper_executable: Path | None
    whisper_model: Path | None
    diarization_library: Path | None
    diarization_model: Path | None
    microphone_helper: Path | None
    demo_audio: Path | None = None
    addressing: Addressing | None = None
    other_agents: Addressing | None = None
    speakers: SpeakerPriority | None = None
    session_budget_seconds: int | None = None
    request_former: RequestForming | None = None
    # Validated `transcriber`/`diarizer` sections; the defaults are the local backends.
    transcriber: dict[str, Any] = field(default_factory=lambda: transcriber_spec(None))
    diarizer: dict[str, Any] = field(default_factory=lambda: diarizer_spec(None))
    # The optional `decision` section: hosted Jev decisions with their consent, for `listen`.
    hosted_decisions: bool = False
    # The section's optional `max_requests`: a per-session Jev request cap for sessions
    # that set none themselves (`listen`). None leaves live sessions uncapped (#75).
    decision_max_requests: int | None = None
    # The section's optional `attend_min_probability`: Jev's attend probability an attend
    # needs, above `min_confidence` (evaluation favoured 0.9). None leaves attend as is.
    decision_attend_min_probability: float | None = None
    # The section's optional `other_answered_blocks` (#199): with the post-turn gap on, an
    # unnamed turn another speaker answers within the reply window is not attended.
    decision_other_answered_blocks: bool = False
    # The section's optional `state_log` (#199): an absolute path where each decided live
    # turn's exact state and answer are appended (transcript text included; `0600`).
    decision_state_log: Path | None = None
    # The section's optional `request_context` (#199): decision states carry the latest
    # request formed and how long ago, so a follow-up that refers back to it is recognized.
    decision_request_context: bool = False
    # The optional `turns` section's `merge_gap_ms` (#73): a same-speaker turn starting
    # within this gap of the previous one is joined before it is emitted or decided.
    turn_merge_gap_ms: int = DEFAULT_TURN_MERGE_GAP_MS
    # The `turns` section's `hold_unfinished` (#199): a turn that looks cut off mid-request
    # ("Jarvis, add the") is held for the merge gap even when the end-of-turn model calls
    # it complete, so its continuation can join it. On by default.
    turn_hold_unfinished: bool = True
    # The `turns` section's `complete_hold_ms`, `collapse_restarts` and
    # `bridge_backchannels` (#216): see `LiveConfig`. All off by default: the hold delays
    # every complete request, and the other two only act with it or change a turn's text.
    turn_complete_hold_ms: int = 0
    turn_collapse_restarts: bool = False
    turn_bridge_backchannels: bool = False
    # The `turns` section's `wake_join` (#230): a lone unattributed wake word joins the
    # owner's following sentence. On by default; acts only with names and owners.
    turn_wake_join: bool = True
    # The `turns` section's `owner_pause_ms` (hailing-station #735 R-1): the owner's own
    # turns wait this long for a continuation, as the complete hold and the merge gap; see
    # `LiveConfig.turn_owner_pause_ms`. 0, the default, is off; acts only with owners.
    turn_owner_pause_ms: int = 0
    # The `decision` section's optional `scene` (#96): the setting text given to the
    # decision model. Absent means the single-user pilot default; null turns it off.
    decision_scene: str | None = DEFAULT_SCENE
    # The `turns` section's `reply_wait_ms` (#96): how long a request-shaped turn is held
    # to observe the gap after it; 0 turns the post-turn gap signal off.
    reply_wait_ms: int = DEFAULT_REPLY_WAIT_MS
    # The `turns` section's `edge_attribution_ms`: labelling a speaker's unlabelled edge
    # words (see `LiveConfig.edge_attribution_ms`); 0, the default, is off.
    edge_attribution_ms: int = 0
    # The `turns` section's `tail_join_ms` (#129): joining an unlabelled tail into the
    # held labelled turn (see `LiveConfig.tail_join_ms`); 0, the default, is off.
    tail_join_ms: int = 0
    # The `turns` section's `word_label_coverage` (#220): see
    # `LiveConfig.word_label_coverage`; 0 turns it off.
    word_label_coverage: float = DEFAULT_WORD_LABEL_COVERAGE
    # The optional `dismissal` section (#98): natural dismissal and the `dismiss` event.
    # Absent means off; `{}` turns it on with the defaults.
    dismissal: Dismissal | None = None
    # The optional `end_of_turn` section (#117): Smart Turn in front of the silence
    # end-of-turn. Absent or `"enabled": false` means off.
    end_of_turn: EndOfTurn | None = None
    # The optional `conversation` section (#82): engaged follow-ups after a request and
    # the `conversation` event. Absent means off; `{}` turns it on with the defaults.
    conversation: Conversation | None = None
    # The optional `acknowledgement` section (#132): gate the host's instant
    # acknowledgement on confidence or a name. Absent means off (every request is
    # acknowledged, as before); `{}` turns it on with the default threshold.
    acknowledgement: Acknowledgement | None = None
    # The optional `speaker_id` section (#137 step 3): the local speaker-embedding model
    # and enrollment store used by `rightyo enroll`. Absent or `"enabled": false` means
    # off. With `"live": true`, live sessions also run shadow identification (step 4a):
    # per-label scores on stderr, with no change to turns, roles or events. With
    # `"roles": true` as well (step 4b), bound labels carry their enrolled role.
    speaker_id: SpeakerIdConfig | None = None
    # The optional `audio_retention` section (#226), memory only and fixed for the process:
    # how much session PCM live speaker identification keeps to cut turns from, and the
    # trailing diarizer timeline window used for attribution (see `LiveConfig`). Both
    # default to 60 s and are independent.
    speaker_ring_ms: int = RING_MS
    timeline_window_ms: int = TIMELINE_WINDOW_MS

    @property
    def voiceprint_roles(self) -> bool:
        """Whether live identification applies roles (#137 step 4b)."""
        return self.speaker_id is not None and self.speaker_id.roles

    @property
    def hosted_audio_consumers(self) -> tuple[str, ...]:
        """#242 round 27: every consumer of session audio that is not local, by name. Each one
        declares it here: the transcriber and diarizer by their selected kind; the end-of-turn
        model (`end_of_turn`) and the speaker-ID embedder (`speaker_id`) are local subprocesses
        by construction (no hosted kind exists); decisions and roles never take audio (a private
        turn's words never reach them, `_accept`, and model-sourced roles are refused live)."""
        return tuple(
            name
            for name, spec in (("transcriber", self.transcriber), ("diarizer", self.diarizer))
            if is_hosted(spec)
        )

    @property
    def hosted_speech(self) -> bool:
        """Whether a selected speech backend would send audio to a hosted service."""
        return bool(self.hosted_audio_consumers)

    @property
    def synthetic_speech(self) -> bool:
        return is_synthetic(self.transcriber) or is_synthetic(self.diarizer)

    @classmethod
    def load(cls, path: Path) -> PrototypeConfig:
        try:
            with path.open("rb") as source:
                content = source.read(65537)
            if len(content) > 65536:
                raise ValueError
            raw = json.loads(content, parse_constant=_reject_constant)
            if not isinstance(raw, dict):
                raise ValueError
            transcriber = transcriber_spec(raw.pop("transcriber", None))
            diarizer = diarizer_spec(raw.pop("diarizer", None))
            synthetic = is_synthetic(transcriber) or is_synthetic(diarizer)
            if synthetic and (
                not (is_synthetic(transcriber) and is_synthetic(diarizer))
                or "end_of_turn" in raw
                or "speaker_id" in raw
            ):
                raise ValueError
            has_decision = "decision" in raw
            decision = raw.pop("decision", None)
            hosted_decisions = has_decision and decision_spec(decision)
            if synthetic and hosted_decisions:
                raise ValueError
            # Optional per-session cap on live Jev requests; absent means none (#75).
            decision_requests = decision.get("max_requests") if hosted_decisions else None
            attend_bar = decision.get("attend_min_probability") if hosted_decisions else None
            answered = bool(has_decision and decision.get("other_answered_blocks", False))
            state_log = decision.get("state_log") if hosted_decisions else None
            acted = bool(has_decision and decision.get("request_context", False))
            scene = (
                scene_text(decision.get("scene", DEFAULT_SCENE)) if has_decision else DEFAULT_SCENE
            )
            required = set() if synthetic else {"microphone_helper"}
            if transcriber["kind"] == "whisper.cpp":
                required |= {"whisper_executable", "whisper_model"}
            if diarizer["kind"] == "nemotron.cpp":
                required |= {"diarization_library", "diarization_model"}
            if not required <= raw.keys():
                raise ValueError
            optional = {
                "demo_audio",
                "addressing",
                "speakers",
                "session_budget_seconds",
                "request_former",
                "turns",
                "dismissal",
                "end_of_turn",
                "conversation",
                "acknowledgement",
                "speaker_id",
                "other_agents",
                "audio_retention",
                "microphone_helper",
                *LOCAL_ASSETS,
            }
            if raw.keys() - required - optional:
                raise ValueError
            addressing = raw.pop("addressing", None)
            if addressing is not None:
                addressing = Addressing.from_dict(addressing)
            # Other agents' names and variants (#161), in the `addressing` shape.
            other_agents = raw.pop("other_agents", None)
            if other_agents is not None:
                other_agents = Addressing.from_dict(other_agents)
            speakers = raw.pop("speakers", None)
            if speakers is not None:
                speakers = SpeakerPriority.from_dict(speakers)
            budget = validate_session_budget(raw.pop("session_budget_seconds", None))
            forming = raw.pop("request_former", None)
            if forming is not None:
                forming = RequestForming.from_dict(forming)
            dismissal = raw.pop("dismissal", None)
            if dismissal is not None:
                dismissal = Dismissal.from_dict(dismissal)
            conversation = raw.pop("conversation", None)
            if conversation is not None:
                conversation = Conversation.from_dict(conversation)
            acknowledgement = raw.pop("acknowledgement", None)
            if acknowledgement is not None:
                acknowledgement = Acknowledgement.from_dict(acknowledgement)
            end_of_turn = raw.pop("end_of_turn", None)
            if end_of_turn is not None:
                end_of_turn = EndOfTurn.from_dict(end_of_turn)
            speaker_id = raw.pop("speaker_id", None)
            if speaker_id is not None:
                speaker_id = SpeakerIdConfig.from_dict(speaker_id)
            ring_ms, window_ms = audio_retention_spec(raw.pop("audio_retention", {}))
            turns = raw.pop("turns", {})
            if not isinstance(turns, dict) or set(turns) - {
                "merge_gap_ms",
                "reply_wait_ms",
                "edge_attribution_ms",
                "tail_join_ms",
                "hold_unfinished",
                "word_label_coverage",
                "complete_hold_ms",
                "collapse_restarts",
                "bridge_backchannels",
                "wake_join",
                "owner_pause_ms",
            }:
                raise ValueError
            hold_unfinished = turns.get("hold_unfinished", True)
            restarts = turns.get("collapse_restarts", False)
            bridge = turns.get("bridge_backchannels", False)
            wake_join = turns.get("wake_join", True)
            if type(hold_unfinished) is not bool or {
                type(restarts),
                type(bridge),
                type(wake_join),
            } != {bool}:
                raise ValueError
            hold_complete = complete_hold(turns.get("complete_hold_ms", 0))
            pause = owner_pause(turns.get("owner_pause_ms", 0))
            tail = tail_join(turns.get("tail_join_ms", 0))
            coverage = turns.get("word_label_coverage", DEFAULT_WORD_LABEL_COVERAGE)
            if type(coverage) not in (int, float) or not 0 <= coverage <= 1:
                raise ValueError
            edges = turns.get("edge_attribution_ms", 0)
            if type(edges) is not int or not 0 <= edges <= 2000 or edges % 20:
                raise ValueError
            gap = merge_gap(turns.get("merge_gap_ms", DEFAULT_TURN_MERGE_GAP_MS))
            wait = reply_wait(turns.get("reply_wait_ms", DEFAULT_REPLY_WAIT_MS))
            if answered and not wait:
                # The gate reads the post-turn gap (#199); without it, it would do nothing.
                raise DecisionConfigError("other_answered_blocks needs turns.reply_wait_ms above 0")
            values = {}
            for name, value in raw.items():
                if not isinstance(value, str) or not value or not Path(value).is_absolute():
                    raise ValueError
                values[name] = Path(value)
            config = cls(
                **{name: values.get(name) for name in LOCAL_ASSETS},
                microphone_helper=values.get("microphone_helper"),
                demo_audio=values.get("demo_audio"),
                addressing=addressing,
                other_agents=other_agents,
                speakers=speakers,
                session_budget_seconds=budget,
                request_former=forming,
                transcriber=transcriber,
                diarizer=diarizer,
                hosted_decisions=hosted_decisions,
                decision_max_requests=decision_requests,
                decision_attend_min_probability=attend_bar,
                decision_other_answered_blocks=answered,
                decision_state_log=None if state_log is None else Path(state_log),
                decision_request_context=acted,
                turn_merge_gap_ms=gap,
                turn_hold_unfinished=hold_unfinished,
                turn_complete_hold_ms=hold_complete,
                turn_collapse_restarts=restarts,
                turn_bridge_backchannels=bridge,
                turn_wake_join=wake_join,
                turn_owner_pause_ms=pause,
                decision_scene=scene,
                reply_wait_ms=wait,
                edge_attribution_ms=edges,
                tail_join_ms=tail,
                word_label_coverage=coverage,
                dismissal=dismissal,
                end_of_turn=end_of_turn,
                conversation=conversation,
                acknowledgement=acknowledgement,
                speaker_id=speaker_id,
                speaker_ring_ms=ring_ms,
                timeline_window_ms=window_ms,
            )
            if not all(value.is_file() for value in values.values()):
                raise ValueError
            if end_of_turn is not None and not (
                end_of_turn.python.is_file() and end_of_turn.model.is_file()
            ):
                raise ValueError
            if speaker_id is not None and not (
                speaker_id.python.is_file() and speaker_id.model.is_file()
            ):
                raise ValueError
            if config.voiceprint_roles and (
                speakers is None
                or not (speakers.owners or speakers.trusted)
                or speakers.source != "configured"
            ):
                # Roles from voiceprints name enrolled identifiers in `speakers`; with
                # none named every voice would be a participant, which is a mistake.
                raise ValueError
            return config
        except (DecisionConfigError, RetentionConfigError):
            raise
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            raise PrototypeError(
                "Prototype requires an existing local asset configuration"
            ) from None


def _takes_keyword(factory: Any, name: str) -> bool:
    """Whether `factory` accepts the keyword argument `name` (#226).

    True when its signature names it as a keyword parameter or takes `**kwargs`, and also
    when the signature cannot be read, so such a factory is called exactly as before.
    `inspect.signature` follows `functools.wraps`, so a wrapper that accepts `ring_ms`
    around a function that does not counts as ring-less: the ring is not applied, and the
    note says so.
    """
    try:
        parameters = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):
        return True
    keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (parameter.name == name and parameter.kind in keyword)
        for parameter in parameters
    )


def _no_report(_message: str) -> None:
    """The diagnostic sink when a session has no channel (the lab)."""


def _release_pending(processor) -> None:
    # Alternative processors need not hold turns back; only call the hook when present.
    release = getattr(processor, "release_pending", None)
    if release is not None:
        release()


class PrototypeController:
    def __init__(
        self,
        config: PrototypeConfig,
        *,
        processor_factory=LiveProcessor,
        capture_factory=MacMicrophoneCapture,
        provider_factory=JevProvider,
        event_publisher=None,
        allow_hosted_speech=False,
        audio_input=None,
        audio_provenance=None,
        report=None,
        end_of_turn_factory=SmartTurn,
        speaker_id_factory=ShadowSpeakerId,
        owner_voice=None,
        input_route=None,
        retain_transcripts=True,
    ):
        if type(retain_transcripts) is not bool:
            raise PrototypeError("Invalid transcript retention setting")
        self.retain_transcripts = retain_transcripts
        self.config = config
        # Owner-voice status events (rightyo#164): an `OwnerVoice` from the host's opt-in,
        # else None and nothing new is advertised or emitted.
        self.owner_voice = owner_voice
        # The host-declared input route kind for speaker_id notes (rightyo#164), already
        # checked against the allowlist by `listen`, and checked again by the identifier.
        self.input_route = input_route
        self.processor_factory = processor_factory
        self.end_of_turn_factory = end_of_turn_factory
        self.speaker_id_factory = speaker_id_factory
        self.capture_factory = capture_factory
        self.provider_factory = provider_factory
        # Explicit consent for configured hosted speech backends to receive audio.
        self.allow_hosted_speech = allow_hosted_speech is True
        # A host-supplied raw PCM stream (`listen --mode stdin`); the web lab never has one.
        self.audio_input = audio_input
        # Stdin bytes say nothing about their origin, so the host must declare it.
        self.audio_provenance = audio_provenance
        self.report = report
        self._stdin_capture: StdinPcmCapture | None = None
        # The current session's processor, kept until the next start (like the stdin
        # capture) so the terminal event can report its skipped segments and utterances.
        self._session_processor = None
        # The stream is consumed by one session only. A stopped session's reader may
        # still be blocked in a read it cannot be interrupted from; it discards whatever
        # it reads, so no later session may share the stream with it.
        self._stdin_used = False
        # Optional host-facing stream. The lab retains its browser lease and UI controls.
        self._events = event_publisher
        # #242 round 27: a publisher already private (as `--start-private` makes it) is refused
        # here with hosted speech, as the CLI and `duck(True)` refuse it: the audio heard while
        # private would still be sent to the hosted backend. Checked again at every start.
        self._refuse_private_hosted()
        # #242 round 28: a publisher that can be private must say what it heard in privacy
        # (`heard_ducked`); one that cannot would leave every privacy judgement unknown.
        if callable(getattr(event_publisher, "duck", None)) and not callable(
            getattr(event_publisher, "heard_ducked", None)
        ):
            raise PrototypeError("A private-capable publisher must report what it heard privately")
        self._event_terminal = True
        self._lock = threading.RLock()
        self._generation = 0
        self._phase = "idle"
        self._mode = "microphone"
        self._error = None
        self._decision_status = "off"
        # Speaker role source for the tool stream: off, or configured. Model-sourced
        # roles are refused here and available to tool-replay only (tracked in #55).
        self._role_status = "off"
        self._memory = TranscriptMemory()
        self._runner: ReplayRunner | None = None
        self._capture = None
        self._processor = None
        # The session's speaker identifier, while one runs (forgotten when privacy ends).
        self._shadow = None
        self._audio_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._decision_cancel = threading.Event()
        self._decision_queue: queue.Queue[Turn] = queue.Queue(maxsize=32)
        self._decisions: dict[str, dict[str, Any]] = {}
        # Post-turn gaps observed by the processor, by utterance id, until decided (#96).
        self._turn_gaps: dict[str, dict[str, Any]] = {}
        # Turns whose speaker label was partly inferred, until published (#137).
        self._inferred_turns: set[str] = set()
        # Each turn's own-label timeline pieces, until the identifier takes them (#148).
        self._turn_spans: dict[str, tuple[tuple[int, int], ...]] = {}
        # The session's live identification role provider, or None (#137 step 4b).
        self._enrolled_roles: EnrolledRoles | None = None
        self._private_sentence = PrivateTurnAudio()
        self._pending = 0
        self._requests = 0
        self._request_limit: int | None = DEFAULT_REQUEST_LIMIT
        self._received_ms = 0
        self._received_bytes = 0
        # #242 round 27: the stream byte offset the current privacy interval began at, or None
        # when unknown or public; an interval that received no audio forgets no speaker (`duck`).
        self._private_from_bytes: int | None = 0 if self._private_audio() else None
        self._budget_ms: int | None = None
        self._budget_reached = False
        self._audio_started: float | None = None
        self._completed_at: float | None = None
        self._started = 0.0
        self._last_browser = time.monotonic()
        self._closed = threading.Event()
        self._timer = threading.Thread(target=self._tick, daemon=True)
        self._timer.start()

    def _live(self) -> bool:
        """Wall-clock timing for live speech only; declared replays use media time."""
        return self._mode == "microphone" or (
            self._mode == "stdin" and self.audio_provenance == "live-microphone"
        )

    def _now_ms(self) -> int:
        if self._completed_at is not None:
            return self._received_ms + int((time.monotonic() - self._completed_at) * 1000)
        wall_ms = (
            int((time.monotonic() - self._audio_started) * 1000)
            if self._live() and self._audio_started is not None
            else 0
        )
        return max(self._received_ms, wall_ms)

    def _tick(self) -> None:
        while not self._closed.wait(0.5):
            with self._lock:
                active = self._phase in {"starting", "listening", "replaying", "finishing"}
                # Microphone sessions are wall-clock bounded from Start; replay feeders
                # end only at the exact audio boundary, however slowly they process.
                expired = active and (
                    time.monotonic() - self._last_browser > BROWSER_LEASE_SECONDS
                    or self._budget_reached
                    or (
                        self._budget_ms is not None
                        and self._live()
                        and (time.monotonic() - self._started) * 1000 > self._budget_ms
                    )
                )
                if self._runner is not None:
                    self._runner.expire(self._now_ms())
                    self._prune_pending()
                if self._events is not None:
                    self._events.expire(self._now_ms())
            if expired:
                self.stop()

    def start(self, options: dict[str, Any]) -> None:
        if set(options) - {
            "mode",
            "use_jev",
            "retention_seconds",
            "confidence",
            "max_requests",
            "session_id",
        }:
            raise PrototypeError("Unrecognized prototype setting")
        mode = options.get("mode", "microphone")
        hosted = options.get("use_jev", False)
        retention = options.get("retention_seconds", 300)
        confidence = options.get("confidence", 0.7)
        explicit_budget = "max_requests" in options
        budget = options.get("max_requests")
        session = options.get("session_id", "prototype-" + uuid.uuid4().hex)
        try:
            identifier(session, "session_id")
        except ContractError:
            raise PrototypeError("Invalid session identifier") from None
        if (
            not isinstance(mode, str)
            or mode not in {"microphone", "demo", "stdin"}
            or type(hosted) is not bool
            or type(retention) is not int
            or not 60 <= retention <= 600
            or (explicit_budget and (type(budget) is not int or not 1 <= budget <= 100))
            or type(confidence) not in {int, float}
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
        ):
            raise PrototypeError("Invalid prototype setting")
        if mode == "demo" and self.config.demo_audio is None:
            raise PrototypeError("No generated audio demo is configured")
        if not explicit_budget:
            # The page always sends its own cap; `listen` takes the configuration's.
            budget = self.config.decision_max_requests
            if budget is None and mode not in LIVE_MODES:
                budget = DEFAULT_REQUEST_LIMIT
        if mode == "stdin" and self.audio_input is None:
            raise PrototypeError("Stdin audio is available only from the listen command")
        if mode == "stdin" and self.audio_provenance not in PROVENANCE:
            raise PrototypeError("Stdin audio requires an explicit source provenance")
        if mode == "stdin" and self._stdin_used:
            raise PrototypeError("Stdin audio feeds one session only; start a new process")
        if not self.retain_transcripts and hosted:
            raise PrototypeError("Ordinary decisions require transcript retention")
        if self.config.synthetic_speech and (
            not (is_synthetic(self.config.transcriber) and is_synthetic(self.config.diarizer))
            or mode != "stdin"
            or self.audio_provenance != "synthetic"
            or hosted
            or self.config.hosted_decisions
            or self.config.end_of_turn is not None
            or self.config.speaker_id is not None
        ):
            # Reject before starting workers or selecting any backend/helper. A host
            # must explicitly identify authored PCM; mocks never label live speech.
            raise PrototypeError("Synthetic speech requires model-free synthetic stdin mode")
        if self.config.speakers is not None and self.config.speakers.source == "model":
            # A hosted role question would run on the audio path under the controller
            # lock, where Stop and lease expiry cannot reach it; tool-replay has no such
            # path and keeps model-sourced roles.
            raise PrototypeError(
                "Model-sourced speaker roles are not available in live microphone or demo "
                "mode yet; use configured roles, or tool-replay (tracked in #55)"
            )
        if self.config.hosted_speech and not self.allow_hosted_speech:
            raise PrototypeError("Hosted speech backends require explicit hosted consent")
        # #242 round 27: a restart, a replay or a demo, a configuration swapped since
        # construction, or a publisher made private outside the controller since: none starts
        # a hosted-speech session in privacy.
        self._refuse_private_hosted()
        if self.processor_factory is LiveProcessor and self.config.transcriber["kind"] == (
            "whisper.cpp"
        ):
            # The local recognizer runs only in a directory this process cannot write
            # (hailing-station #735 S-1); without one, refuse before any audio.
            try:
                recognizer_dir()
            except RecognizerSandboxError:
                raise PrototypeError(
                    "Transcription refused: the local recognizer needs a directory this "
                    "process cannot write, such as /var/empty"
                ) from None
        roles = self.config.speakers
        if (
            roles is not None
            and (roles.owners or roles.trusted or roles.owner_only)
            and utterance_local_labels(self.config.diarizer)
        ):
            # Utterance-local labels never equal a configured `Speaker A`, so roles
            # could never apply and `owner_only` would silence every request.
            raise PrototypeError(
                "Configured speaker roles require a session-stable diarizer; the selected "
                "diarizer labels speakers per utterance"
            )
        # With live identification roles (#137 step 4b), inferred labels are allowed: a
        # turn with inferred words keeps its role for attention and precedence, but never
        # exercises owner authority (`SpeechEvents.transcript(inferred=True)`).
        voiceprint = self.config.voiceprint_roles
        if voiceprint and utterance_local_labels(self.config.diarizer):
            raise PrototypeError(
                "Speaker identification roles require a session-stable diarizer; the "
                "selected diarizer labels speakers per utterance"
            )
        if (
            roles is not None
            and (roles.owners or roles.trusted or roles.owner_only)
            and self.config.edge_attribution_ms
            and not voiceprint
        ):
            # An inferred label must never carry a role's authority: a guest's first word
            # in the slack after the owner's segment would become the owner's.
            raise PrototypeError(
                "Edge attribution infers speaker labels; it cannot be combined with "
                "configured speaker roles"
            )
        if (
            roles is not None
            and (roles.owners or roles.trusted or roles.owner_only)
            and self.config.tail_join_ms
            and not voiceprint
        ):
            # Likewise for tail join (#129): a guest's short reply right after the owner
            # would be joined into the owner's turn.
            raise PrototypeError(
                "Tail join infers speaker labels; it cannot be combined with "
                "configured speaker roles"
            )
        budget_seconds = validate_session_budget(self.config.session_budget_seconds)
        with self._lock:
            if self._phase in {"starting", "listening", "replaying", "finishing", "stopping"}:
                raise PrototypeError("Stop the active session before starting another")
            if self._audio_thread is not None and self._audio_thread.is_alive():
                raise PrototypeError("The previous audio runtime is still stopping")
            stop = threading.Event()
            decision_cancel = threading.Event()

            def cancelled():
                return stop.is_set() or decision_cancel.is_set()

            memory = TranscriptMemory(retention_ms=retention * 1000)
            try:
                bar = self.config.decision_attend_min_probability
                provider = (
                    self.provider_factory(
                        allow_hosted=True,
                        max_requests=budget,
                        timeout_seconds=10,
                        min_confidence=confidence,
                        cancelled=cancelled,
                        **({} if bar is None else {"attend_min_probability": bar}),
                    )
                    if hosted
                    else MockProvider()
                )
            except (ProviderError, CredentialError):
                raise PrototypeError("Hosted decisions could not be initialized") from None
            runner = ReplayRunner(
                provider,
                memory=memory if self.retain_transcripts else None,
                cancelled=cancelled,
                addressing=self.config.addressing,
                scene=self.config.decision_scene,
                post_turn_gaps=self.config.reply_wait_ms > 0,
                other_answered_blocks=self.config.decision_other_answered_blocks,
                state_log=(
                    None
                    if self.config.decision_state_log is None
                    else DecisionStateLog(self.config.decision_state_log, report=self.report)
                ),
                request_context=self.config.decision_request_context,
                dismissal_phrases=(
                    None
                    if self.config.dismissal is None
                    else (self.config.speakers or SpeakerPriority()).stop_phrases
                ),
            )
            # Only configured roles run here: no hosted role question ever executes
            # under the controller lock (model-sourced roles are refused above).
            priority = (
                None
                if self.config.speakers is None
                else EnrolledRoles(
                    self.config.speakers,
                    self.config.speaker_id.enrolled_follow_up_min_probability,
                    emit_names=self.config.speaker_id.emit_names,
                )
                if voiceprint
                else ConfiguredPriorityProvider(self.config.speakers)
            )
            self._enrolled_roles = priority if voiceprint else None
            self._private_sentence.clear()
            if self._enrolled_roles is not None:
                self._enrolled_roles.exit_audio = self._private_sentence
            runner.restart(session)
            self._generation += 1
            generation = self._generation
            self._stop = stop
            self._decision_cancel = decision_cancel
            self._memory = memory
            self._runner = runner
            self._decision_queue = work = queue.Queue(maxsize=32)
            self._decisions = {}
            self._turn_gaps = {}
            self._inferred_turns = set()
            self._turn_spans = {}
            self._pending = self._requests = self._received_ms = 0
            self._received_bytes = 0
            # #242 round 27: a session that starts private (`--start-private`, or the mode held
            # from before) has heard nothing in it yet; one that starts public, nothing private.
            self._private_from_bytes = 0 if self._private_audio() else None
            self._budget_ms = None if budget_seconds is None else budget_seconds * 1000
            self._budget_reached = False
            self._audio_started = None
            self._completed_at = None
            self._request_limit = budget
            self._mode = mode
            self._stdin_capture = None
            self._session_processor = None
            self._stdin_used = self._stdin_used or mode == "stdin"
            self._error = None
            self._decision_status = "ready" if hosted else "off"
            self._role_status = "off" if priority is None else "configured"
            self._phase = "starting"
            self._started = self._last_browser = time.monotonic()
            if self._events is not None:
                self._events.start(
                    session,
                    now_ms=0,
                    attention_enabled=hosted,
                    addressing=self.config.addressing,
                    priority=priority,
                    former=request_former_for(self.config.request_former),
                    speech=speech_summary(self.config.transcriber, self.config.diarizer),
                    audio_input=STDIN_AUDIO_INPUT if mode == "stdin" else None,
                    dismissal=self.config.dismissal,
                    conversation=self.config.conversation,
                    acknowledgement=self.config.acknowledgement,
                    other_agents=self.config.other_agents,
                    **({} if self.owner_voice is None else {"owner_voice": self.owner_voice}),
                    # Privacy mode (internally duck, hailing-station #735): the live
                    # controller takes the control, so a publisher that holds it advertises
                    # it, unless its host control channel does not carry the line (a
                    # coaching listener), or a hosted speech backend would still be sent the
                    # audio heard while private (S-1, fail closed: the host then refuses).
                    **(
                        {"duck": True}
                        if getattr(self._events, "duck_control", False) is True
                        and callable(getattr(self._events, "duck", None))
                        and not self.config.hosted_speech
                        else {}
                    ),
                )
                self._event_terminal = False
                if self.owner_voice is not None and not voiceprint and self.report is not None:
                    with contextlib.suppress(Exception):
                        self.report("owner_voice off: speaker identification roles are not on")
            if self.report is not None:
                # The in-memory audio retention in force (#226): seconds only, once per
                # session (one per `listen` process); best-effort, like every note.
                with contextlib.suppress(Exception):
                    self.report(
                        f"audio_retention speaker_ring_s={self.config.speaker_ring_ms // 1000}"
                        f" timeline_window_s={self.config.timeline_window_ms // 1000}"
                    )
            threading.Thread(
                target=self._decide,
                args=(generation, stop, work, runner, hosted),
                daemon=True,
            ).start()
            self._audio_thread = threading.Thread(
                target=self._audio,
                args=(generation, stop, session, work, memory, mode),
                daemon=True,
            )
            self._audio_thread.start()

    def _accept(
        self, generation: int, work: queue.Queue, memory: TranscriptMemory, turn: Turn
    ) -> bool | None:
        """Accept a final turn; whether it was heard in privacy, or None once stopped."""
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return None
            # Privacy mode (internally duck, hailing-station #735 S-1), judged by when the
            # turn was heard (see `SpeechEvents.heard_ducked`): no hosted decision is ever
            # asked for it and it is kept out of the memory every later decision's context
            # is read from. It is still passed to the publisher, which only matches it for
            # the local exit and status phrases and emits no transcript (`_private`).
            ducked = self._heard_private(turn)
            if ducked:
                # Observed just before this accept (#96); no decision will ever take it.
                self._turn_gaps.pop(turn.utterance_id, None)
            if self.retain_transcripts and not ducked:
                memory.append(turn)
            inferred = turn.utterance_id in self._inferred_turns
            self._inferred_turns.discard(turn.utterance_id)
            if self._events is not None:
                # Passed only when set, so publishers without live roles see no change.
                self._publish("transcript", turn, **({"inferred": True} if inferred else {}))
            if self._decision_status == "off" or self._decision_cancel.is_set() or ducked:
                return ducked
            try:
                work.put_nowait(turn)
            except queue.Full:
                self._decision_cancel.set()
                if self._decision_status != "off":
                    self._decision_status = "unavailable"
                self._discard_pending()
                self._tool_attention_error("attention-backlog")
                return ducked
            self._pending += 1
            return ducked

    def _heard_private_span(self, start_ms: int, end_ms: int) -> bool | None:
        """Whether audio over this stream span was heard in privacy mode (#735 S-1), by the
        same rule as a turn (`SpeechEvents.heard_ducked`); for the processor's diagnostics and
        its settle at a privacy end. A publisher that can be private but cannot say gives None,
        which the processor counts as private (#242 round 28: unknown counts); one with no
        privacy at all never heard anything privately."""
        heard_ducked = getattr(self._events, "heard_ducked", None)
        if not callable(heard_ducked):
            return None if callable(getattr(self._events, "duck", None)) else False
        return heard_ducked(SimpleNamespace(start_ms=start_ms, end_ms=end_ms)) is True

    def _heard_private(self, turn: Turn) -> bool:
        """Whether the publisher heard this turn in privacy mode (hailing-station #735)."""
        heard_ducked = getattr(self._events, "heard_ducked", None)
        return callable(heard_ducked) and heard_ducked(turn) is True

    def _turn_break(self):
        """The predicate for turns that are never joined or held (#89, #98).

        Privacy mode (internally duck, hailing-station #735 S-1): while it is on, a piece
        holding a named exit or a status question, so the owner pause never delays the way
        back in; while it is off, a piece that is an intentional entry ("stop recording",
        "privacy mode", "go private" and the like as the whole piece), so the host's safety
        matcher sees it at once. Matched as `privacy_phrase` does; nothing is kept.
        """
        stop = (self.config.speakers or SpeakerPriority()).is_stop_phrase
        addressing = self.config.addressing
        dismissal = self.config.dismissal is not None

        def ducking() -> bool:
            # Read as the merger releases the piece; the publisher holds the mode.
            return getattr(self._events, "ducked", False) is True

        def breaks(text: str) -> bool:
            if stop(text) or (dismissal and dismissal_shaped(text)):
                return True
            try:
                kinds = privacy_phrase(text, addressing)
            except Exception:  # noqa: BLE001 - the words must never reach an error or a log
                return False
            return bool(kinds & ({"exit", "status"} if ducking() else {"entry"}))

        return breaks

    def _wake_join(
        self,
    ) -> tuple[Callable[[str], bool] | None, Callable[[dict[str, Any]], bool] | None]:
        """The wake-word and owner predicates of the lone wake word join (#230), or Nones.

        Only with configured names and owners, and `turns.wake_join` on: the wake word is
        a turn whose text is only a configured name or variant; the owner is a fragment
        whose label resolves to the owner role now, by the same rules the turn's published
        role uses (configured session labels, or live identification bindings, looked up
        without side effects). A guest, a participant and an unknown speaker are never the
        owner, so the name never raises anyone's role.
        """
        addressing = self.config.addressing
        if not self.config.turn_wake_join or addressing is None:
            return None, None
        owner = self._turn_owner()
        if owner is None:
            return None, None

        def wake_word(text: str) -> bool:
            return addressing.name_for(text) is not None

        return wake_word, owner

    def _turn_owner(self) -> Callable[[dict[str, Any]], bool] | None:
        """The turn merger's owner predicate (#230, #735), or None without owners.

        A fragment is the owner's when its label resolves to the owner role now, by the
        same rules the turn's published role uses: configured session labels, or live
        identification bindings looked up without side effects.
        """
        speakers = self.config.speakers
        if speakers is None or not speakers.owners:
            return None
        enrolled = self._enrolled_roles

        def owner(fragment: dict[str, Any]) -> bool:
            if enrolled is None:
                return speakers.configured_role(fragment["speaker"]) == "owner"
            # The identification role, as `SpeechEvents` would resolve it at emission,
            # looked up without touching the binding its `name_for` reads.
            probe = SimpleNamespace(
                utterance_id=None,
                speaker_id=fragment["speaker"],
                overlap=fragment["overlap"],
                speaker_provenance=fragment["speaker_provenance"],
            )
            return enrolled.role_of(probe) == "owner"

        return owner

    def _turn_unfinished(self):
        """The predicate for turns held despite end-of-turn's "complete" (#199), or None.

        A stop phrase or a dismissal-shaped turn ("stop it", "forget it") is never held,
        whether or not natural dismissal is on, so a stop is never delayed or joined.
        """
        if not self.config.turn_hold_unfinished:
            return None
        stop = (self.config.speakers or SpeakerPriority()).is_stop_phrase
        addressing = self.config.addressing
        return lambda text: (
            not stop(text) and not dismissal_shaped(text) and unfinished_shaped(text, addressing)
        )

    def _observe_gap(self, generation: int, utterance_id: str, gap: dict[str, Any]) -> None:
        """Keep a turn's observed post-turn gap until its decision takes it (#96)."""
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            self._turn_gaps[utterance_id] = gap
            while len(self._turn_gaps) > MAX_PENDING_GAPS:
                # Turns that were never queued for a decision leave stale entries.
                del self._turn_gaps[next(iter(self._turn_gaps))]

    def _observe_inferred(self, generation: int, utterance_id: str) -> None:
        """Remember that a turn's label was partly inferred until it is published (#137)."""
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            # Called just before the same turn is accepted, so this holds at most one id.
            self._inferred_turns.add(utterance_id)

    def _observe_spans(
        self, generation: int, utterance_id: str, spans: tuple[tuple[int, int], ...]
    ) -> None:
        """Keep a turn's own-label timeline pieces until the identifier takes them (#148)."""
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            # Called just before the same turn is accepted, so this holds at most one id.
            self._turn_spans[utterance_id] = spans

    def _end_of_turn_model(self, stop):
        """Start the configured end-of-turn model, or None; never fails the session (#117)."""
        settings = self.config.end_of_turn
        if settings is None:
            return None
        try:
            return self.end_of_turn_factory(
                settings.python, settings.model, threads=settings.threads, cancelled=stop.is_set
            )
        except SmartTurnError:
            if self.report is not None:
                with contextlib.suppress(Exception):
                    self.report("end-of-turn model unavailable; using silence end-of-turn")
            return None

    def _shadow_speaker_id(self):
        """Start shadow speaker identification (#137 step 4a), or None.

        Only with a `speaker_id` section that sets `"live": true`, and only with a
        diagnostic channel (`listen`'s stderr), since its scores are its only output. Its
        worker thread loads the model and voiceprints, so nothing here waits on them, and a
        failure only ever switches shadow identification off: the session never depends
        on it.
        """
        settings = self.config.speaker_id
        roles = self._enrolled_roles
        if settings is None or not settings.live or (self.report is None and roles is None):
            return None
        try:
            # With roles on (step 4b) it runs without a diagnostic channel too (the lab).
            report = self.report if self.report is not None else _no_report
            options = {} if self.input_route is None else {"route": self.input_route}
            # A configured ring length (#226); the default passes nothing, as before. A
            # provider that takes no `ring_ms` runs with its own ring, and a note says so.
            ring_not_applied = False
            if self.config.speaker_ring_ms != RING_MS:
                if _takes_keyword(self.speaker_id_factory, "ring_ms"):
                    options["ring_ms"] = self.config.speaker_ring_ms
                else:
                    ring_not_applied = True
            shadow = self.speaker_id_factory(settings, report=report, **options)
            shadow.start()
            if roles is not None:
                # Turns read the bindings from here on; until now every label is unknown.
                roles.source = shadow
            if ring_not_applied:
                # Content-free and best-effort: speaker ID stays on either way.
                with contextlib.suppress(Exception):
                    report("audio_retention speaker_ring not applied: provider takes no ring_ms")
            return shadow
        except Exception:
            if self.report is not None:
                with contextlib.suppress(Exception):
                    self.report(
                        "speaker_id unavailable; shadow identification off for this session"
                    )
            return None

    def _turn(self, generation, work, memory, shadow, turn: Turn) -> None:
        with self._lock:
            # Read before `_accept` consumes it. The identifier embeds only the turn's
            # own-label timeline pieces (#148), so an inferred-label word's audio is never
            # voiceprint evidence (#137), in shadow mode or with roles.
            inferred = turn.utterance_id in self._inferred_turns
            spans = self._turn_spans.pop(turn.utterance_id, None)
            roles = self._enrolled_roles
            if roles is not None:
                # The privacy-entry voice check (hailing-station #735) scores these pieces and
                # the rest of the span apart, when the transcript is published below.
                roles.note_spans(turn.utterance_id, spans)
        # A private turn is never speaker evidence (#735 S-1): no binding or count is
        # derived from speech heard in privacy, judged by the same decision `_accept` made
        # (so a mode change between the two cannot split them). Only a matched local exit
        # may score its isolated sentence, without ever offering it to shadow.turn.
        with self._lock:
            private = self._accept(generation, work, memory, turn)
            # #242 round 23: the speaker epoch is read in the same critical section as the
            # privacy state, so a privacy cycle completing before `shadow.turn` queues the
            # turn (`forget` advances the epoch) discards it instead of binding its old label.
            epoch = (
                getattr(shadow, "epoch", None) if shadow is not None and private is False else None
            )
        if shadow is not None and private is False:
            # After the turn is accepted and published; it never raises or blocks.
            shadow.turn(
                turn,
                inferred=inferred,
                spans=spans,
                **({"epoch": epoch} if type(epoch) is int else {}),
            )

    def _private_turn_audio(self, generation, turn, pcm):
        with self._lock:
            if generation != self._generation:
                return
            self._private_sentence.clear()
            if self._stop.is_set() or not self._private_audio() or self._enrolled_roles is None:
                return
            start = self._private_from_bytes
            # Unknown on-edge or a public prefix: refuse instead of copying public audio.
            if start is None or turn.start_ms * PCM_BYTES_PER_MS < start:
                return
            self._private_sentence.replace(turn, pcm)

    def _audio(self, generation, stop, session, work, memory, mode):
        capture = processor = end_of_turn = shadow = None
        try:
            end_of_turn = self._end_of_turn_model(stop)
            shadow = self._shadow_speaker_id()
            with self._lock:
                if generation == self._generation:
                    self._shadow = shadow
            settings = self.config.end_of_turn
            wake_word, owner = self._wake_join()
            if owner is None and self.config.turn_owner_pause_ms:
                # The owner-only pause (#735) uses the same owner predicate.
                owner = self._turn_owner()
            processor = self.processor_factory(
                LiveConfig(
                    session_id=session,
                    whisper_executable=self.config.whisper_executable,
                    whisper_model=self.config.whisper_model,
                    diarization_library=self.config.diarization_library,
                    diarization_model=self.config.diarization_model,
                    provenance=(
                        self.audio_provenance
                        if mode == "stdin"
                        else "live-microphone"
                        if mode == "microphone"
                        else "causal-replay"
                    ),
                    cancelled=stop.is_set,
                    report=self.report,
                    private=self._heard_private_span,
                    on_private_turn_audio=lambda turn, pcm: self._private_turn_audio(
                        generation, turn, pcm
                    ),
                    session_budget_ms=self._budget_ms,
                    transcriber=transcriber_factory(
                        self.config.transcriber, allow_hosted=self.allow_hosted_speech
                    ),
                    diarizer=diarizer_factory(
                        self.config.diarizer, allow_hosted=self.allow_hosted_speech
                    ),
                    turn_merge_gap_ms=self.config.turn_merge_gap_ms,
                    # Stop phrases are matched against a whole turn, so joining must
                    # never absorb one; the defaults apply when roles are off. With
                    # dismissal on (#98), a dismissal-shaped turn is released at once too.
                    turn_break=self._turn_break(),
                    turn_unfinished=self._turn_unfinished(),
                    reply_wait_ms=self.config.reply_wait_ms,
                    on_post_turn_gap=lambda utterance_id, gap: self._observe_gap(
                        generation, utterance_id, gap
                    ),
                    on_inferred=lambda utterance_id: self._observe_inferred(
                        generation, utterance_id
                    ),
                    # Only with an identifier to use them (#148); None computes nothing.
                    on_labelled_spans=(
                        None
                        if shadow is None
                        else lambda utterance_id, spans: self._observe_spans(
                            generation, utterance_id, spans
                        )
                    ),
                    end_of_turn=end_of_turn.score if end_of_turn is not None else None,
                    edge_attribution_ms=self.config.edge_attribution_ms,
                    tail_join_ms=self.config.tail_join_ms,
                    word_label_coverage=self.config.word_label_coverage,
                    turn_complete_hold_ms=self.config.turn_complete_hold_ms,
                    turn_collapse_restarts=self.config.turn_collapse_restarts,
                    turn_bridge_backchannels=self.config.turn_bridge_backchannels,
                    # A lone wake word joins the owner's following sentence (#230).
                    turn_wake_word=wake_word,
                    turn_owner=owner,
                    # The owner's own turns wait longer for a continuation (#735).
                    turn_owner_pause_ms=self.config.turn_owner_pause_ms,
                    end_of_turn_threshold=settings.threshold if settings else 0.5,
                    end_of_turn_silence_ms=settings.silence_ms if settings else 200,
                    timeline_window_ms=self.config.timeline_window_ms,
                ),
                lambda turn: self._turn(generation, work, memory, shadow, turn),
            )
            with self._lock:
                if generation != self._generation or stop.is_set():
                    return
                self._processor = self._session_processor = processor
                if hasattr(processor, "on_speakers_renewed"):
                    processor.on_speakers_renewed = self._speakers_renewed
            if mode == "microphone":
                capture = self.capture_factory(self.config.microphone_helper)
                with self._lock:
                    if generation != self._generation or stop.is_set():
                        return
                    self._capture = capture
                    capture.start()
                while not stop.is_set():
                    pcm = capture.read(timeout=0.25)
                    if pcm:
                        self._feed(generation, processor, pcm, mode, shadow)
                    else:
                        _release_pending(processor)
            elif mode == "stdin":
                capture = StdinPcmCapture(self.audio_input, report=self.report)
                with self._lock:
                    if generation != self._generation or stop.is_set():
                        return
                    self._capture = self._stdin_capture = capture
                    capture.start()
                accepted = True
                while accepted and not stop.is_set():
                    pcm = capture.read(timeout=0.25)
                    if pcm == b"":
                        # EOF finalizes the open utterance. After an overrun it is clipped
                        # mid-speech, so close() discards it instead; turns finalized
                        # before the drop stand, and the session ends as an error.
                        break
                    if pcm:
                        accepted = self._feed(generation, processor, pcm, mode, shadow)
                    else:
                        # No audio arrived for a read timeout: stream time cannot reach
                        # a held turn's merge deadline, so emit it rather than wait.
                        _release_pending(processor)
                if accepted and not stop.is_set():
                    if not capture.overrun:
                        processor.finish()
                    else:
                        # Turns finalized before the drop stand, including a held one.
                        _release_pending(processor)
                    with self._lock:
                        # The flush can itself end the session (an attention backlog
                        # records an error terminal and sets stop); never overwrite that.
                        if generation == self._generation and not stop.is_set():
                            self._phase = "finishing" if self._pending else "complete"
                            self._completed_at = time.monotonic()
            else:
                with wave.open(str(self.config.demo_audio), "rb") as audio:
                    if (
                        audio.getnchannels() != 1
                        or audio.getframerate() != 16000
                        or audio.getsampwidth() != 2
                        or audio.getcomptype() != "NONE"
                        or not 0 < audio.getnframes() <= 180 * 16000
                    ):
                        raise PrototypeError("Demo requires bounded mono PCM16 16 kHz audio")
                    accepted = True
                    while not stop.is_set():
                        pcm = audio.readframes(3200)
                        if not pcm:
                            break
                        accepted = self._feed(generation, processor, pcm, mode, shadow)
                        if not accepted:
                            break
                # A session budget ends like any other cancellation: the timer stops it
                # and the unfinished utterance is discarded rather than flushed.
                if accepted and not stop.is_set():
                    processor.finish()
                    with self._lock:
                        if generation == self._generation:
                            self._phase = "finishing" if self._pending else "complete"
                            self._completed_at = time.monotonic()
        except (
            CaptureError,
            LiveAudioError,
            PrototypeError,
            ContractError,
            OSError,
            wave.Error,
        ) as error:
            with self._lock:
                if generation == self._generation and not stop.is_set():
                    self._phase = "error"
                    if isinstance(error, MemorySessionLimitError):
                        self._error = "Session reached its 1,000-turn limit; start a new session."
                    elif isinstance(error, HostedSpeechError):
                        self._error = "Hosted speech backend failed; the session is incomplete."
                    elif mode == "stdin":
                        self._error = "Stdin audio input failed; the session is incomplete."
                    else:
                        self._error = (
                            "Audio stopped. Check microphone permission and local model setup."
                        )
                    stop.set()
                    if self._decision_status != "off":
                        self._decision_status = "unavailable"
                    self._completed_at = time.monotonic()
                    self._discard_pending()
                    if self._runner is not None:
                        self._runner.clear(clear_memory=False)
                    self._event_end("error", "audio-unavailable")
        except Exception:
            with self._lock:
                if generation == self._generation and not stop.is_set():
                    self._phase = "error"
                    self._error = "Audio processing failed; the session is incomplete."
                    stop.set()
                    if self._decision_status != "off":
                        self._decision_status = "unavailable"
                    self._completed_at = time.monotonic()
                    self._discard_pending()
                    if self._runner is not None:
                        self._runner.clear(clear_memory=False)
                    self._event_end("error", "audio-unavailable")
        finally:
            with self._lock:
                if generation == self._generation:
                    self._private_sentence.clear()
            if capture is not None:
                capture.stop()
            if processor is not None:
                processor.close()
            if end_of_turn is not None:
                end_of_turn.close()
            if shadow is not None:
                # A replay or stdin stream that ended normally scores its last turns; a
                # stopped or failed session discards them.
                with contextlib.suppress(Exception):
                    shadow.close(drain=not stop.is_set())
            with self._lock:
                if self._shadow is shadow:
                    self._shadow = None
                if generation == self._generation:
                    self._capture = self._processor = None
                elif generation + 1 == self._generation and self._phase == "stopping":
                    self._phase = "idle"

    def _feed(self, generation, processor, pcm, mode, shadow=None) -> bool:
        """Push audio up to the session budget; False once the session accepts no more."""
        boundary = False
        with self._lock:
            if generation != self._generation or self._stop.is_set() or self._budget_reached:
                return False
            if self._budget_ms is not None:
                remaining = self._budget_ms * PCM_BYTES_PER_MS - self._received_bytes
                if len(pcm) >= remaining:
                    # Accept exactly up to the boundary.
                    pcm = pcm[:remaining]
                    boundary = True
            if not pcm:
                return False
            self._received_bytes += len(pcm)
            self._received_ms = self._received_bytes // PCM_BYTES_PER_MS
            if self._audio_started is None:
                self._audio_started = time.monotonic()
            self._phase = "listening" if self._live() else "replaying"
            # Read with the clock under the lock, so a change stamped after this chunk
            # never leaves a chunk heard in privacy in the speaker ring.
            private = self._private_audio()
            if private and self.config.hosted_speech:
                # #242 round 27, the one choke point every audio consumer is fed through: in
                # privacy, a hosted consumer (`hosted_audio_consumers`) receives nothing, by
                # whatever path privacy became active. Unreachable while the refusals hold;
                # if one is bypassed, the session ends as an error before this chunk goes on.
                self._end_private_hosted(generation)
                return False
        if shadow is not None:
            # Before the processor, so a turn it finalizes on this chunk finds its audio.
            # In privacy mode (hailing-station #735 S-1) the speaker ring keeps silence of
            # the same length instead: its stream offsets stay aligned, and none of the
            # private audio enters the speaker ring. A separate single-sentence buffer
            # may authenticate a matched exit; private audio is never speaker evidence.
            shadow.audio(bytes(len(pcm)) if private else pcm)
        processor.push_pcm16(pcm)
        if boundary:
            # The budget discards the open utterance, but a turn already finalized and
            # held for a possible continuation stands.
            _release_pending(processor)
            # Publish only after the boundary audio is processed, so the timer cannot
            # stop the session before that final push completes.
            with self._lock:
                if generation == self._generation:
                    self._budget_reached = True
        return not boundary

    def _private_audio(self) -> bool:
        """Whether privacy mode is on now, as the publisher holds it."""
        return getattr(self._events, "ducked", False) is True

    def _refuse_private_hosted(self) -> None:
        """#242 round 27: privacy is never active with a hosted audio consumer configured; the
        same refusal as `duck(True)` and `--start-private`, fail closed and announced."""
        if self.config.hosted_speech and self._private_audio():
            raise PrototypeError("duck mode is refused with a hosted speech backend")

    def _end_private_hosted(self, generation: int) -> None:
        """End the session as an error: privacy became active with a hosted audio consumer by
        a path that bypassed every refusal (#242 round 27). Under the lock."""
        if generation != self._generation or self._stop.is_set():
            return
        self._phase = "error"
        self._error = "Privacy mode is refused with a hosted speech backend; the session ended."
        self._stop.set()
        if self._decision_status != "off":
            self._decision_status = "unavailable"
        self._completed_at = time.monotonic()
        self._discard_pending()
        if self._runner is not None:
            self._runner.clear(clear_memory=False)
        self._event_end("error", "privacy-hosted-speech")

    def _decide(self, generation, stop, work, runner, hosted):
        enabled = hosted
        failures = 0  # consecutive transiently unavailable hosted decisions
        while not stop.is_set():
            try:
                with self._lock:
                    turn = work.get_nowait()
            except queue.Empty:
                with self._lock:
                    if generation != self._generation or self._phase == "complete":
                        return
                stop.wait(0.05)
                continue
            try:
                event = None
                with self._lock:
                    gap = self._turn_gaps.pop(turn.utterance_id, None)
                    if generation != self._generation or stop.is_set():
                        continue
                    if self._decision_cancel.is_set():
                        enabled = False
                    runner.expire(self._now_ms())
                    if turn.utterance_id not in self._memory.retained_ids:
                        continue
                if not enabled:
                    if not hosted:
                        runner.process(turn, gap)
                    continue
                if (
                    self._request_limit is not None
                    and runner.provider.requests >= self._request_limit
                ):
                    with self._lock:
                        if generation == self._generation:
                            self._decision_status = "budget-exhausted"
                            self._tool_attention_error("attention-budget-exhausted")
                    enabled = False
                    continue
                unavailable = None
                try:
                    event = runner.process(turn, gap)
                    failures = 0
                except ProviderUnavailable as failure:
                    # One transient hosted failure degrades this turn only: an uncertain
                    # placeholder, never a request, and the session keeps listening.
                    # Anything else, or too many in a row, ends it as before.
                    failures += 1
                    if failures >= MAX_CONSECUTIVE_DECISION_FAILURES:
                        raise
                    unavailable = failure.reason
                    event = DecisionEvent(turn, unavailable_decision(), turn.revision, 0.0, 0.0)
                with self._lock:
                    if (
                        generation == self._generation
                        and not stop.is_set()
                        and not self._decision_cancel.is_set()
                        and event is not None
                    ):
                        self._requests = runner.provider.requests
                        self._decisions[turn.utterance_id] = {
                            **event.public_dict(),
                            "recipient_speaker_id": event.decision.recipient_speaker_id,
                            **(
                                {}
                                if unavailable is None
                                else {"decision_status": "unavailable", "reason": unavailable}
                            ),
                        }
                        if self._events is not None:
                            self._publish(
                                "decision",
                                event,
                                **({} if unavailable is None else {"unavailable": unavailable}),
                            )

            except Exception:
                enabled = False
                with self._lock:
                    if generation == self._generation and not stop.is_set():
                        self._decision_status = "unavailable"
                        self._requests = getattr(runner.provider, "requests", 0)
                        self._tool_attention_error("attention-unavailable")
            finally:
                event = None
                work.task_done()
                with self._lock:
                    if generation == self._generation:
                        self._pending = max(0, self._pending - 1)
                        if self._phase == "finishing" and self._pending == 0:
                            self._phase = "complete"
                del turn
        runner.clear(clear_memory=False)

    def _discard_pending(self):
        while True:
            try:
                self._decision_queue.get_nowait()
            except queue.Empty:
                break
            self._decision_queue.task_done()
            self._pending = max(0, self._pending - 1)

    def _prune_pending(self):
        """Release expired queued plaintext, including while hosted inference is slow."""
        retained = self._memory.retained_ids
        keep = []
        while True:
            try:
                turn = self._decision_queue.get_nowait()
            except queue.Empty:
                break
            self._decision_queue.task_done()
            if turn.utterance_id in retained:
                keep.append(turn)
            else:
                self._pending = max(0, self._pending - 1)
        for turn in keep:
            self._decision_queue.put_nowait(turn)

    def snapshot(self, *, heartbeat=True) -> dict[str, Any]:
        with self._lock:
            if heartbeat:
                self._last_browser = time.monotonic()
            now = self._now_ms()
            if self._events is not None:
                self._events.expire(now)
            if self._runner is not None:
                self._runner.expire(now)
                self._prune_pending()
                self._requests = getattr(self._runner.provider, "requests", 0)
            snapshot = self._memory.snapshot(now)
            retained = {t["utterance_id"] for t in snapshot["turns"]}
            self._decisions = {k: v for k, v in self._decisions.items() if k in retained}
            return {
                **snapshot,
                "phase": self._phase,
                "mode": self._mode,
                "error": self._error,
                "decision_status": self._decision_status,
                "role_status": self._role_status,
                "decisions": dict(self._decisions),
                "pending_decisions": self._pending,
                "jev_requests": self._requests,
                "jev_request_limit": self._request_limit,
                "demo_available": self.config.demo_audio is not None,
                # Stage identities come from the selected backends, never from a fixed
                # local label; hosted selections are reported as hosted, without endpoints.
                "transcriber": describe(self.config.transcriber),
                "diarizer": describe(self.config.diarizer),
                "models": {
                    "diarization": describe(self.config.diarizer)["service"],
                    "asr": describe(self.config.transcriber)["service"],
                    "decision": "Jev 1.13.0 (opt-in)",
                },
                "session_limit_seconds": self.config.session_budget_seconds,
            }

    def stop(self) -> None:
        with self._lock:
            if self._phase == "stopping":
                return
            self._event_end("stopped" if self._phase == "complete" else "cancelled")
            self._generation += 1
            generation = self._generation
            self._stop.set()
            self._decision_cancel.set()
            capture, processor, runner = self._capture, self._processor, self._runner
            audio_thread = self._audio_thread
            self._capture = self._processor = self._runner = None
            self._phase = "stopping"
            self._error = None
            self._memory.clear()
            self._decisions.clear()
            # Per-turn notes observed before an accept that the stop cut off (#735 S-1).
            self._turn_gaps.clear()
            self._inferred_turns.clear()
            self._turn_spans.clear()
            self._pending = self._received_ms = 0
            self._received_bytes = 0
            # #242 round 27: the stream clock restarts; nothing is heard in privacy until audio.
            self._private_from_bytes = 0 if self._private_audio() else None
            self._budget_ms = None
            self._budget_reached = False
            self._audio_started = None
            self._completed_at = None
            self._decision_status = "off"
            self._role_status = "off"
            while not self._decision_queue.empty():
                try:
                    self._decision_queue.get_nowait()
                    self._decision_queue.task_done()
                except queue.Empty:
                    break
        try:
            if runner is not None:
                runner.clear()
            if capture is not None:
                capture.stop()
            if processor is not None:
                processor.close()
            if audio_thread is not None and audio_thread is not threading.current_thread():
                audio_thread.join(timeout=5)
        finally:
            with self._lock:
                if generation == self._generation and (
                    audio_thread is None or not audio_thread.is_alive()
                ):
                    self._phase = "idle"

    def close(self):
        self._closed.set()
        self.stop()

    def reply(self, phase: str, *, asked: bool = False) -> None:
        """Forward the host's report that a spoken reply `started` or `ended` (#124).

        `asked`: an ended reply ended with a question to the owner (#199). Timed by this
        session's own stream clock when it arrives; ignored when no session is running. A
        malformed phase raises before anything is forwarded.
        """
        if phase not in {"started", "ended"} or type(asked) is not bool:
            raise PrototypeError("invalid reply phase")
        if asked and phase != "ended":
            raise PrototypeError("invalid reply phase")
        with self._lock:
            if (
                self._events is None
                or self._event_terminal
                or self._stop.is_set()
                or self._phase not in {"listening", "replaying", "finishing"}
            ):
                return
            self._publish("reply", phase, **({"asked": True} if asked else {}))
            if self._runner is not None:
                # The decision context sees the reply too (rightyo#153), on the same clock.
                self._runner.note_reply(phase, self._now_ms())

    def media(self, on: bool) -> None:
        """Forward the host's media mode (hailing-station #551) to the event publisher.

        Held by the publisher across sessions, so it applies whether or not one is running;
        ignored without one (the web lab). A malformed value raises before anything is
        forwarded.
        """
        if type(on) is not bool:
            raise PrototypeError("invalid media mode")
        with self._lock:
            if self._events is not None:
                self._events.media(on)

    def duck(self, on: bool) -> None:
        """Forward the host's duck mode (hailing-station #735) to the event publisher.

        Held by the publisher across sessions, like media mode; ignored without one (the
        web lab). Stamped with the received-audio clock turns are timed in (never the wall
        clock, which runs ahead of it for good after lost audio), so a turn heard while it
        was on is never decided, even when it is released after it was turned off (see
        `_accept`), and speech after it is. A malformed value raises before anything is
        forwarded. Turning it on is refused with a hosted speech backend, which would
        still be sent the audio heard while private (S-1; `started` never advertises it
        there, and `--start-private` is refused): fail closed, never apparently private.
        """
        if type(on) is not bool:
            raise PrototypeError("invalid duck mode")
        if on and self.config.hosted_speech:
            raise PrototypeError("duck mode is refused with a hosted speech backend")
        with self._lock:
            if self._events is not None:
                self._private_sentence.clear()
                ending = not on and self._events.ducked is True
                if on and self._events.ducked is not True:
                    # #242 round 27: where this privacy interval starts, in stream bytes.
                    self._private_from_bytes = self._received_bytes
                # At least the end of every piece received so far, held ones included. Every
                # applied on, a redundant one included, is acked (#735 S-8): audio received
                # after the ack's `at_ms` falls inside the span.
                try:
                    self._events.duck(on, now_ms=self._received_ms, ack=on)
                except ContractError:
                    # The ack overran the event backlog after the duck was applied: the
                    # session fails as any lost event fails it (Codex on #248), so `listen`
                    # never sits waiting for audio the host withholds, nor ends as a clean
                    # stop. The caller still applies later control lines (an off included).
                    self._event_delivery_failed()
                    raise
                if ending:
                    # #242 round 27: an interval in which no audio was received, judged on the
                    # stream clock in bytes (one sample counts), heard no one, so it forgets no
                    # speaker: every effect of `_forget_speakers` (the publisher's renewed and
                    # stale labels, the runner's, the diarizer renewal, the identifier's
                    # bindings) assumes private audio was heard. Unknown (privacy began outside
                    # the controller mid-session) forgets, fail closed.
                    start = self._private_from_bytes
                    self._private_from_bytes = None
                    if start is None or self._received_bytes > start:
                        self._forget_speakers()

    def _speakers_renewed(self) -> None:
        """The fresh diarizer takes over (#735 S-1, round 15): its labels restart from the first
        voice heard, so every binding or role made for the old labels since privacy ended (a
        turn finalized while the renewal waited) is forgotten again before any new label is
        scored. Called on the audio thread."""
        with self._lock:
            self._private_sentence.clear()
            if self._publisher_forgets_speakers():
                self._runner_forgets_speakers()
            if self._shadow is not None:
                with contextlib.suppress(Exception):
                    self._shadow.forget()

    def _publisher_forgets_speakers(self) -> bool:
        """Under the lock: the publisher forgets the old labels, stamped on the received-audio
        clock turns are timed in. Whether it has `forget_speakers`. An engaged conversation is
        ended aloud there (rightyo#244), after everything is dropped; if that end overruns the
        event backlog the session fails as any lost event fails it (as the ack does on #248),
        and every other effect of the forget still applies."""
        forget = getattr(self._events, "forget_speakers", None)
        if not callable(forget):
            return False
        try:
            forget(now_ms=self._received_ms)
        except ContractError:
            self._event_delivery_failed()
        return True

    def _runner_forgets_speakers(self) -> None:
        """#242 round 25: the decision runner forgets the old labels with the publisher, so no
        later decision state names one: not its past turns or participants, a turn decided
        after the forget, nor the latest request, whose copy loses its label too. Stamped on
        the received-audio clock turns are timed in. Under the lock, like `note_request`."""
        runner = self._runner
        if runner is None:
            return
        runner.forget_speakers(self._received_ms)
        if runner.request_context:
            runner.note_request(getattr(self._events, "last_request", None))

    def _forget_speakers(self) -> None:
        """Privacy ended (#735 S-1, Codex P1 on #242): no speaker state from it may outlive it.
        Under the lock. The diarizer starts afresh (once nothing is open), every speaker-ID
        label binding is dropped and the publisher's per-label roles with it, so the owner is
        identified again from public speech. The owner's exit already used its labels. Audio
        heard before the end and still buffered is silenced, stamped on the same clock as the
        mode (round 18) but in bytes, exact where the millisecond stamp rounds down (round 19),
        so none of it, not one sample, is sent to ASR with the next public utterance."""
        self._private_sentence.clear()
        if self._publisher_forgets_speakers():
            self._runner_forgets_speakers()
        if self._processor is not None and callable(
            getattr(self._processor, "reset_speakers", None)
        ):
            self._processor.reset_speakers(self._received_bytes)
        if self._shadow is not None:
            with contextlib.suppress(Exception):
                self._shadow.forget()

    def _publish(self, method, value, **fields):
        """Run under the controller lock; a broken consumer cancels observation."""
        try:
            options = (
                {"expect_decision": self._decision_status == "ready", **fields}
                if method == "transcript"
                else fields
            )
            getattr(self._events, method)(value, now_ms=self._now_ms(), **options)
            runner = self._runner
            if runner is not None and runner.request_context:
                # What the assistant last acted on, or None once dismissed (#199).
                runner.note_request(getattr(self._events, "last_request", None))
        except Exception:
            self._event_delivery_failed()
            raise PrototypeError("Tool event delivery failed") from None

    def _event_delivery_failed(self):
        """Run under the controller lock: a lost event fails the session, terminally."""
        self._stop.set()
        self._decision_cancel.set()
        self._phase = "error"
        self._error = "Tool event delivery failed; the session is incomplete."
        self._memory.clear()
        self._decisions.clear()
        self._discard_pending()
        self._event_end("error", "event-delivery-failed")

    def _tool_attention_error(self, reason):
        """A tool cannot silently continue after losing its promised attention service."""
        if self._events is None or self._event_terminal:
            return
        self._stop.set()
        self._decision_cancel.set()
        self._phase = "error"
        self._error = "Attention unavailable; the tool session is incomplete."
        self._completed_at = time.monotonic()
        self._memory.clear()
        self._decisions.clear()
        self._discard_pending()
        self._event_end("error", reason)

    def _event_end(self, phase, reason=None):
        self._private_sentence.clear()
        if self._events is not None and not self._event_terminal:
            capture = self._stdin_capture
            gaps = None
            if capture is not None:
                if capture.overrun and phase in {"stopped", "cancelled"}:
                    # Dropped audio would splice speech across a gap and corrupt the
                    # session-persistent diarizer state: fail closed, even on a later Stop.
                    phase, reason = "error", "input-overrun"
                gaps = {
                    "gaps": int(capture.overrun),
                    "dropped_bytes": capture.dropped_bytes,
                    "discarded_tail_bytes": capture.discarded_tail_bytes,
                }
            counts = {
                name: getattr(self._session_processor, name, 0)
                for name in ("skipped_segments", "skipped_utterances")
            }
            self._events.end(
                phase=phase,
                now_ms=self._now_ms(),
                reason=reason,
                input_gaps=gaps,
                **{
                    name: value if type(value) is int and value > 0 else None
                    for name, value in counts.items()
                },
            )
            self._event_terminal = True

    @property
    def input_overrun(self) -> bool:
        """Whether the stdin input overran its queue; the session then ends as an error."""
        capture = self._stdin_capture
        return capture is not None and capture.overrun

    def drain_events(self):
        """Drain only this explicit session's events, independently of the lab UI."""
        with self._lock:
            return [] if self._events is None else self._events.drain()


class PrototypeServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, controller: PrototypeController, port: int = 0):
        super().__init__(("127.0.0.1", port), PrototypeHandler)
        self.controller = controller
        self.token = secrets.token_urlsafe(32)
        self.origin = f"http://127.0.0.1:{self.server_port}"

    @property
    def launch_url(self):
        return self.origin + "/#" + self.token


class PrototypeHandler(BaseHTTPRequestHandler):
    server: PrototypeServer

    def setup(self):
        self.request.settimeout(5)
        super().setup()

    def log_message(self, *_args):
        pass

    def _response(self, status: int, payload: bytes, kind="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(payload)
        except OSError:
            pass

    def _json(self, status, value):
        self._response(status, json.dumps(value, allow_nan=False).encode())

    def _allowed(self, *, authenticated: bool) -> bool:
        host = self.server.origin.removeprefix("http://")
        if self.headers.get("Host") != host:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin != self.server.origin:
            return False
        return not authenticated or hmac.compare_digest(
            self.headers.get("Authorization", "").encode("utf-8"),
            ("Bearer " + self.server.token).encode("utf-8"),
        )

    def do_GET(self):
        if not self._allowed(authenticated=self.path.startswith("/api/")):
            self._json(403, {"error": "Local session authorization required"})
            return
        if self.path == "/api/state":
            self._json(200, self.server.controller.snapshot())
            return
        assets = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
            "/style.css": ("style.css", "text/css; charset=utf-8"),
        }
        if self.path not in assets:
            self._json(404, {"error": "Not found"})
            return
        name, kind = assets[self.path]
        self._response(200, (Path(__file__).parent / "web" / name).read_bytes(), kind)

    def do_POST(self):
        if not self._allowed(authenticated=True):
            self._json(403, {"error": "Local session authorization required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= 4096 or self.headers.get("Transfer-Encoding") is not None:
                raise ValueError
            raw = json.loads(self.rfile.read(length), parse_constant=_reject_constant)
            if not isinstance(raw, dict):
                raise ValueError
        except (ValueError, UnicodeError, RecursionError, OSError):
            self._json(400, {"error": "Invalid control request"})
            return
        try:
            if self.path == "/api/start":
                self.server.controller.start(raw)
            elif self.path == "/api/stop" and not raw:
                self.server.controller.stop()
            else:
                self._json(404, {"error": "Not found"})
                return
        except PrototypeError as error:
            self._json(400, {"error": str(error)})
            return
        self._json(200, {"ok": True})


def serve(
    config_path: Path,
    port: int = 8765,
    *,
    addressing: Addressing | None = None,
    session_budget_seconds: int | None = None,
    allow_hosted: bool = False,
) -> None:
    if type(port) is not int or not 0 <= port <= 65535:
        raise PrototypeError("Invalid local port")
    config = PrototypeConfig.load(config_path)
    if config.hosted_speech and not allow_hosted:
        raise PrototypeError("Hosted speech backends require --allow-hosted")
    if addressing is not None:
        # Command-line names take precedence over the configuration file's names.
        config = replace(config, addressing=addressing)
    if session_budget_seconds is not None:
        config = replace(
            config, session_budget_seconds=validate_session_budget(session_budget_seconds)
        )
    controller = PrototypeController(config, allow_hosted_speech=allow_hosted)
    server = None
    try:
        server = PrototypeServer(controller, port)
        print("Open the local speech lab (idle): " + server.launch_url, flush=True)
        server.serve_forever(poll_interval=0.25)
    finally:
        controller.close()
        if server is not None:
            server.server_close()
