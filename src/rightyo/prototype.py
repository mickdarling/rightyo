"""Explicit loopback speech lab; startup never enables microphone or hosted inference."""

from __future__ import annotations

import contextlib
import hmac
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
from typing import Any

from rightyo.addressedness import (
    DEFAULT_REPLY_WAIT_MS,
    DEFAULT_SCENE,
    dismissal_shaped,
    reply_wait,
    scene_text,
)
from rightyo.capture import CaptureError, MacMicrophoneCapture, StdinPcmCapture
from rightyo.contracts import (
    PROVENANCE,
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
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
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
from rightyo.speech_backends import (
    HostedSpeechError,
    describe,
    diarizer_factory,
    diarizer_spec,
    is_hosted,
    speech_summary,
    transcriber_factory,
    transcriber_spec,
    utterance_local_labels,
)
from rightyo.turn_merge import DEFAULT_TURN_MERGE_GAP_MS, merge_gap, tail_join

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


DECISION_PROVIDERS = ("mock", "jev")


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
        }
    ):
        raise DecisionConfigError(
            "The decision section requires exactly the keys provider and allow_hosted, "
            "with an optional max_requests and scene"
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


@dataclass(frozen=True)
class PrototypeConfig:
    # Local assets are required by the local backends only; a hosted selection may omit them.
    whisper_executable: Path | None
    whisper_model: Path | None
    diarization_library: Path | None
    diarization_model: Path | None
    microphone_helper: Path
    demo_audio: Path | None = None
    addressing: Addressing | None = None
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
    # The optional `turns` section's `merge_gap_ms` (#73): a same-speaker turn starting
    # within this gap of the previous one is joined before it is emitted or decided.
    turn_merge_gap_ms: int = DEFAULT_TURN_MERGE_GAP_MS
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
    # The optional `dismissal` section (#98): natural dismissal and the `dismiss` event.
    # Absent means off; `{}` turns it on with the defaults.
    dismissal: Dismissal | None = None
    # The optional `end_of_turn` section (#117): Smart Turn in front of the silence
    # end-of-turn. Absent or `"enabled": false` means off.
    end_of_turn: EndOfTurn | None = None
    # The optional `conversation` section (#82): engaged follow-ups after a request and
    # the `conversation` event. Absent means off; `{}` turns it on with the defaults.
    conversation: Conversation | None = None

    @property
    def hosted_speech(self) -> bool:
        """Whether a selected speech backend would send audio to a hosted service."""
        return is_hosted(self.transcriber) or is_hosted(self.diarizer)

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
            has_decision = "decision" in raw
            decision = raw.pop("decision", None)
            hosted_decisions = has_decision and decision_spec(decision)
            # Optional per-session cap on live Jev requests; absent means none (#75).
            decision_requests = decision.get("max_requests") if hosted_decisions else None
            scene = (
                scene_text(decision.get("scene", DEFAULT_SCENE)) if has_decision else DEFAULT_SCENE
            )
            required = {"microphone_helper"}
            if not is_hosted(transcriber):
                required |= {"whisper_executable", "whisper_model"}
            if not is_hosted(diarizer):
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
                *LOCAL_ASSETS,
            }
            if raw.keys() - required - optional:
                raise ValueError
            addressing = raw.pop("addressing", None)
            if addressing is not None:
                addressing = Addressing.from_dict(addressing)
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
            end_of_turn = raw.pop("end_of_turn", None)
            if end_of_turn is not None:
                end_of_turn = EndOfTurn.from_dict(end_of_turn)
            turns = raw.pop("turns", {})
            if not isinstance(turns, dict) or set(turns) - {
                "merge_gap_ms",
                "reply_wait_ms",
                "edge_attribution_ms",
                "tail_join_ms",
            }:
                raise ValueError
            tail = tail_join(turns.get("tail_join_ms", 0))
            edges = turns.get("edge_attribution_ms", 0)
            if type(edges) is not int or not 0 <= edges <= 2000 or edges % 20:
                raise ValueError
            gap = merge_gap(turns.get("merge_gap_ms", DEFAULT_TURN_MERGE_GAP_MS))
            wait = reply_wait(turns.get("reply_wait_ms", DEFAULT_REPLY_WAIT_MS))
            values = {}
            for name, value in raw.items():
                if not isinstance(value, str) or not value or not Path(value).is_absolute():
                    raise ValueError
                values[name] = Path(value)
            config = cls(
                **{name: values.get(name) for name in LOCAL_ASSETS},
                microphone_helper=values["microphone_helper"],
                demo_audio=values.get("demo_audio"),
                addressing=addressing,
                speakers=speakers,
                session_budget_seconds=budget,
                request_former=forming,
                transcriber=transcriber,
                diarizer=diarizer,
                hosted_decisions=hosted_decisions,
                decision_max_requests=decision_requests,
                turn_merge_gap_ms=gap,
                decision_scene=scene,
                reply_wait_ms=wait,
                edge_attribution_ms=edges,
                tail_join_ms=tail,
                dismissal=dismissal,
                end_of_turn=end_of_turn,
                conversation=conversation,
            )
            if not all(value.is_file() for value in values.values()):
                raise ValueError
            if end_of_turn is not None and not (
                end_of_turn.python.is_file() and end_of_turn.model.is_file()
            ):
                raise ValueError
            return config
        except DecisionConfigError:
            raise
        except (OSError, ValueError, TypeError, UnicodeError, RecursionError):
            raise PrototypeError(
                "Prototype requires an existing local asset configuration"
            ) from None


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
    ):
        self.config = config
        self.processor_factory = processor_factory
        self.end_of_turn_factory = end_of_turn_factory
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
        self._audio_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._decision_cancel = threading.Event()
        self._decision_queue: queue.Queue[Turn] = queue.Queue(maxsize=32)
        self._decisions: dict[str, dict[str, Any]] = {}
        # Post-turn gaps observed by the processor, by utterance id, until decided (#96).
        self._turn_gaps: dict[str, dict[str, Any]] = {}
        self._pending = 0
        self._requests = 0
        self._request_limit: int | None = DEFAULT_REQUEST_LIMIT
        self._received_ms = 0
        self._received_bytes = 0
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
        if (
            roles is not None
            and (roles.owners or roles.trusted or roles.owner_only)
            and self.config.edge_attribution_ms
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
                provider = (
                    self.provider_factory(
                        allow_hosted=True,
                        max_requests=budget,
                        timeout_seconds=10,
                        min_confidence=confidence,
                        cancelled=cancelled,
                    )
                    if hosted
                    else MockProvider()
                )
            except (ProviderError, CredentialError):
                raise PrototypeError("Hosted decisions could not be initialized") from None
            runner = ReplayRunner(
                provider,
                memory=memory,
                cancelled=cancelled,
                addressing=self.config.addressing,
                scene=self.config.decision_scene,
                post_turn_gaps=self.config.reply_wait_ms > 0,
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
                else ConfiguredPriorityProvider(self.config.speakers)
            )
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
            self._pending = self._requests = self._received_ms = 0
            self._received_bytes = 0
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
                )
                self._event_terminal = False
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

    def _accept(self, generation: int, work: queue.Queue, memory: TranscriptMemory, turn: Turn):
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            memory.append(turn)
            if self._events is not None:
                self._publish("transcript", turn)
            if self._decision_status == "off" or self._decision_cancel.is_set():
                return
            try:
                work.put_nowait(turn)
            except queue.Full:
                self._decision_cancel.set()
                if self._decision_status != "off":
                    self._decision_status = "unavailable"
                self._discard_pending()
                self._tool_attention_error("attention-backlog")
                return
            self._pending += 1

    def _turn_break(self):
        """The predicate for turns that are never joined or held (#89, #98)."""
        stop = (self.config.speakers or SpeakerPriority()).is_stop_phrase
        if self.config.dismissal is None:
            return stop
        return lambda text: stop(text) or dismissal_shaped(text)

    def _observe_gap(self, generation: int, utterance_id: str, gap: dict[str, Any]) -> None:
        """Keep a turn's observed post-turn gap until its decision takes it (#96)."""
        with self._lock:
            if generation != self._generation or self._stop.is_set():
                return
            self._turn_gaps[utterance_id] = gap
            while len(self._turn_gaps) > MAX_PENDING_GAPS:
                # Turns that were never queued for a decision leave stale entries.
                del self._turn_gaps[next(iter(self._turn_gaps))]

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

    def _audio(self, generation, stop, session, work, memory, mode):
        capture = processor = end_of_turn = None
        try:
            end_of_turn = self._end_of_turn_model(stop)
            settings = self.config.end_of_turn
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
                    reply_wait_ms=self.config.reply_wait_ms,
                    on_post_turn_gap=lambda utterance_id, gap: self._observe_gap(
                        generation, utterance_id, gap
                    ),
                    end_of_turn=end_of_turn.score if end_of_turn is not None else None,
                    edge_attribution_ms=self.config.edge_attribution_ms,
                    tail_join_ms=self.config.tail_join_ms,
                    end_of_turn_threshold=settings.threshold if settings else 0.5,
                    end_of_turn_silence_ms=settings.silence_ms if settings else 200,
                ),
                lambda turn: self._accept(generation, work, memory, turn),
            )
            with self._lock:
                if generation != self._generation or stop.is_set():
                    return
                self._processor = self._session_processor = processor
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
                        self._feed(generation, processor, pcm, mode)
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
                        accepted = self._feed(generation, processor, pcm, mode)
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
                        accepted = self._feed(generation, processor, pcm, mode)
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
            if capture is not None:
                capture.stop()
            if processor is not None:
                processor.close()
            if end_of_turn is not None:
                end_of_turn.close()
            with self._lock:
                if generation == self._generation:
                    self._capture = self._processor = None
                elif generation + 1 == self._generation and self._phase == "stopping":
                    self._phase = "idle"

    def _feed(self, generation, processor, pcm, mode) -> bool:
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
            self._pending = self._received_ms = 0
            self._received_bytes = 0
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

    def reply(self, phase: str) -> None:
        """Forward the host's report that a spoken reply `started` or `ended` (#124).

        Timed by this session's own stream clock when it arrives; ignored when no session
        is running. A malformed phase raises before anything is forwarded.
        """
        if phase not in {"started", "ended"}:
            raise PrototypeError("invalid reply phase")
        with self._lock:
            if (
                self._events is None
                or self._event_terminal
                or self._stop.is_set()
                or self._phase not in {"listening", "replaying", "finishing"}
            ):
                return
            self._publish("reply", phase)

    def _publish(self, method, value, **fields):
        """Run under the controller lock; a broken consumer cancels observation."""
        try:
            options = (
                {"expect_decision": self._decision_status == "ready"}
                if method == "transcript"
                else fields
            )
            getattr(self._events, method)(value, now_ms=self._now_ms(), **options)
        except Exception:
            self._stop.set()
            self._decision_cancel.set()
            self._phase = "error"
            self._error = "Tool event delivery failed; the session is incomplete."
            self._memory.clear()
            self._decisions.clear()
            self._discard_pending()
            self._event_end("error", "event-delivery-failed")
            raise PrototypeError("Tool event delivery failed") from None

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
