"""Explicit owner-private coaching with bounded context and host-qualified delivery.

Separate from normal attention/request dispatch. No capture, playback, file retention,
network backend, command parsing or automatic enablement is performed here.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from uuid import UUID, uuid4

from rightyo.contracts import ContractError, ListenerProfile, Turn, identifier, integer, probability
from rightyo.listener_profiles import ListenerProfiles

MAX_QUALIFICATION_MS = 2000
MAX_SESSION_SECONDS = 900
MAX_CONTEXT_SECONDS = 60
MAX_CONTEXT_TURNS = 32
MAX_CONTEXT_BYTES = 12288
MAX_ADVICE_BYTES = 8192
MAX_ADVICE_CHARS = 280
MAX_ADVICE_WORDS = 45
MIN_CADENCE_SECONDS = 20
MIN_PAUSE_MS = 400
MAX_TIMING_MS = 60000
MAX_FINALS = 1000
MAX_CONTROL_IDS = 4096
INVALIDATION_REASONS = frozenset(
    {
        "route_changed",
        "route_lost",
        "participant_awareness_lost",
        "owner_cancelled",
        "capture_stopped",
        "host_invalidated",
    }
)
STATE_REASONS = INVALIDATION_REASONS | {
    "initial",
    "enabled",
    "qualified",
    "disabled",
    "closed",
    "lease_expired",
    "session_expired",
    "profile_changed",
    "provider_failed",
    "control_budget",
    "turn_budget",
    "clock_invalid",
}


def wire_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode(
        "ascii"
    )


def canonical_uuid(value: Any) -> str:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value or UUID(value).int == 0:
            raise ValueError
    except (ValueError, AttributeError):
        raise ContractError("invalid coaching UUID") from None
    return value


def validate_coaching_control(raw: Any) -> dict[str, Any]:
    """Validate data shape only; the caller separately supplies trusted host provenance."""
    if not isinstance(raw, dict):
        raise ContractError("invalid coaching control")
    common = {"schema_version", "type", "action", "session_id"}
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 1:
        raise ContractError("unsupported coaching control version")
    if raw.get("type") != "coaching_control":
        raise ContractError("invalid coaching control type")
    action = raw.get("action")
    if action == "enable":
        fields = {"destination_id", "listener_id", "participant_awareness"}
    elif action == "qualify":
        fields = {"destination_id", "generation_id", "qualification_id", "valid_for_ms"}
    elif action == "invalidate":
        fields = {"generation_id", "reason"}
    elif action == "disable":
        fields = set()
    else:
        raise ContractError("invalid coaching control action")
    if set(raw) != common | fields:
        raise ContractError("invalid coaching control fields")
    canonical_uuid(raw["session_id"])
    for field in ("destination_id", "generation_id", "qualification_id"):
        if field in fields:
            canonical_uuid(raw[field])
    if action == "enable":
        identifier(raw["listener_id"], "listener id")
        if raw["participant_awareness"] is not True:
            raise ContractError("coaching requires explicit participant awareness")
    if action == "qualify":
        integer(raw["valid_for_ms"], "coaching qualification duration", 1)
        if raw["valid_for_ms"] > MAX_QUALIFICATION_MS:
            raise ContractError("coaching qualification duration exceeds budget")
    if action == "invalidate":
        if not isinstance(raw["reason"], str) or raw["reason"] not in INVALIDATION_REASONS:
            raise ContractError("invalid coaching invalidation reason")
    return dict(raw)


@dataclass(frozen=True)
class CoachSuggestion:
    text: str
    provider: str = "mock"
    model: str = "authored-fixture"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text) > MAX_ADVICE_CHARS
            or len(self.text.split()) > MAX_ADVICE_WORDS
            or not self.text.isprintable()
        ):
            raise ContractError("invalid coaching advice text")
        try:
            self.text.encode("utf-8")
        except UnicodeError:
            raise ContractError("invalid coaching advice encoding") from None
        identifier(self.provider, "coach provider")
        identifier(self.model, "coach model")
        probability(self.confidence)


@dataclass(frozen=True)
class CoachContext:
    turn: Turn
    playback: bool

    def to_dict(self) -> dict[str, Any]:
        return {"turn": self.turn.to_dict(), "playback": self.playback}


@dataclass(frozen=True)
class CoachWork:
    """Immutable untrusted conversation evidence, never instructions or permissions."""

    profile: ListenerProfile
    context: tuple[CoachContext, ...]
    current: Turn
    pause_ms: int
    request_id: str


class CoachProvider(Protocol):
    def advise(self, work: CoachWork, cancelled: threading.Event) -> CoachSuggestion | None:
        """Return a bounded suggestion or hold. Cooperate with cancellation and timeout."""


class MockCoachProvider:
    """Authored utterance-ID lookup, not a model or a judgment about real conversations."""

    def __init__(self, suggestions: dict[str, str] | None = None) -> None:
        suggestions = {} if suggestions is None else suggestions
        if not isinstance(suggestions, dict) or len(suggestions) > 512:
            raise ContractError("invalid authored coaching suggestions")
        self._suggestions = {
            canonical_uuid(key): CoachSuggestion(value) for key, value in suggestions.items()
        }

    def advise(self, work: CoachWork, cancelled: threading.Event) -> CoachSuggestion | None:
        if cancelled.is_set():
            return None
        return self._suggestions.get(work.current.utterance_id)


@dataclass(frozen=True)
class _Qualification:
    generation: str
    qualification: str
    deadline: float


@dataclass(frozen=True)
class _Ticket:
    session: str
    destination: str
    lease: _Qualification
    work: CoachWork
    cancelled: threading.Event


class CoachingSession:
    """One explicitly selected private listener; no multi-agent or room arbitration.

    The host must tick while idle and independently enforce actual private route and
    original qualification deadlines before playback. Calls here grant no authority.
    A blocked provider holds at most one bounded snapshot and prevents new calls.
    """

    def __init__(
        self,
        *,
        profiles: ListenerProfiles,
        provider: CoachProvider,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(profiles, ListenerProfiles) or not callable(
            getattr(provider, "advise", None)
        ):
            raise ContractError("invalid coaching components")
        self.profiles = profiles
        self._provider = provider
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._session: str | None = None
        self._destination: str | None = None
        self._profile: ListenerProfile | None = None
        self._profile_revision = 0
        self._enabled = False
        self._closed = False
        self._lease: _Qualification | None = None
        self._context: deque[tuple[float, CoachContext]] = deque()
        self._seen: dict[str, bytes] = {}
        self._last_end = 0
        self._last_call: float | None = None
        self._started = 0.0
        self._last_clock: float | None = None
        self._retired: set[str] = set()
        self._qualifications: set[str] = set()
        self._sessions: set[str] = set()
        self._pending: _Ticket | None = None
        self._busy = False
        self._advice: tuple[_Ticket, CoachSuggestion] | None = None
        self._state_event: dict[str, Any] | None = None
        self._state("disabled", "initial")

    def _state(self, state: str, reason: str) -> None:
        # Coalesce state to latest; one advice + one state is the entire output queue.
        assert reason in STATE_REASONS
        self._state_event = {
            "schema_version": 1,
            "type": "coaching_state",
            "session_id": self._session,
            "state": state,
            "capture_permitted": state == "qualified",
            "reason": reason,
        }

    def _stop(self, reason: str, *, disable: bool = False) -> None:
        if self._lease is not None:
            self._retired.add(self._lease.generation)
        self._lease = None
        self._context.clear()
        self._advice = None
        if self._pending is not None:
            self._pending.cancelled.set()
            self._pending = None
        if disable:
            self._enabled = False
            self._profile = None
            self._seen.clear()
        self._state("suspended" if self._enabled else "disabled", reason)

    def _now(self) -> float:
        try:
            now = self._clock()
            if (
                type(now) not in (float, int)
                or not math.isfinite(now)
                or now < 0
                or (self._last_clock is not None and now < self._last_clock)
            ):
                raise ValueError
        except Exception:
            self._stop("clock_invalid", disable=True)
            raise ContractError("invalid coaching clock") from None
        self._last_clock = now
        return now

    def _tick(self, now: float) -> None:
        if not self._enabled:
            return
        if now - self._started >= MAX_SESSION_SECONDS:
            self._stop("session_expired", disable=True)
            return
        if self._profile_revision != self.profiles.revisioned_snapshot()[0]:
            self._stop("profile_changed", disable=True)
            return
        if self._lease is not None and now >= self._lease.deadline:
            self._stop("lease_expired")
            return
        while self._context and now - self._context[0][0] >= MAX_CONTEXT_SECONDS:
            self._context.popleft()
        if self._pending is not None and now >= self._pending.lease.deadline:
            self._pending.cancelled.set()
            self._pending = None
        if self._advice is not None and now >= self._advice[0].lease.deadline:
            self._advice = None

    @property
    def capture_permitted(self) -> bool:
        with self._condition:
            self._tick(self._now())
            return self._enabled and self._lease is not None

    def tick(self) -> None:
        with self._condition:
            self._tick(self._now())

    def apply_control(self, raw: Any) -> None:
        """Trusted host channel only; never call this from transcripts/model output."""
        control = validate_coaching_control(raw)
        with self._condition:
            now = self._now()
            self._tick(now)
            if self._closed:
                raise ContractError("coaching coordinator is closed")
            action = control["action"]
            if action == "enable":
                if self._enabled or control["session_id"] in self._sessions:
                    raise ContractError("coaching enable requires a fresh session")
                if len(self._sessions) >= MAX_CONTROL_IDS:
                    self._stop("control_budget", disable=True)
                    raise ContractError("coaching control budget exceeded")
                revision, profiles = self.profiles.revisioned_snapshot()
                selected = next(
                    (p for p in profiles if p.listener_id == control["listener_id"]),
                    None,
                )
                if selected is None or selected.delivery != "owner_private":
                    raise ContractError("coaching requires an existing private listener")
                self._session = control["session_id"]
                self._destination = control["destination_id"]
                self._sessions.add(self._session)
                self._profile = selected
                self._profile_revision = revision
                self._started = now
                self._last_call = None
                self._last_end = 0
                self._seen.clear()
                self._enabled = True
                self._state("enabled", "enabled")
                return
            if control["session_id"] != self._session:
                raise ContractError("coaching control session mismatch")
            if action == "disable":
                self._stop("disabled", disable=True)
            elif action == "invalidate":
                # Old generation notifications cannot cancel a newly qualified one.
                if self._lease is not None and control["generation_id"] == self._lease.generation:
                    self._stop(control["reason"])
            else:
                if not self._enabled or control["destination_id"] != self._destination:
                    raise ContractError("coaching qualification scope mismatch")
                generation = control["generation_id"]
                qualification = control["qualification_id"]
                if generation in self._retired or qualification in self._qualifications:
                    raise ContractError("coaching qualification is stale or reused")
                if self._lease is not None and generation != self._lease.generation:
                    raise ContractError("invalidate before changing coaching generation")
                if (
                    len(self._qualifications) >= MAX_CONTROL_IDS
                    or len(self._retired) >= MAX_CONTROL_IDS
                ):
                    self._stop("control_budget", disable=True)
                    raise ContractError("coaching control budget exceeded")
                self._qualifications.add(qualification)
                self._lease = _Qualification(
                    generation, qualification, now + control["valid_for_ms"] / 1000
                )
                self._state("qualified", "qualified")

    def _snapshot(self) -> tuple[CoachContext, ...]:
        return tuple(item for _, item in self._context)

    def observe(
        self,
        turn: Turn,
        *,
        generation_id: str,
        pause_ms: int,
        owner_mid_answer: bool,
        playback: bool = False,
    ) -> None:
        """Accept finalized evidence tagged with its capture generation by the host.

        Measured timing and playback/owner activity are host evidence, not parsed text.
        Old capture callbacks must retain their original generation tag after reconnect.
        """
        if type(turn) is not Turn:
            raise ContractError("invalid coaching transcript")
        canonical_uuid(turn.session_id)
        canonical_uuid(turn.utterance_id)
        canonical_uuid(generation_id)
        integer(pause_ms, "coaching pause")
        if pause_ms > MAX_TIMING_MS:
            raise ContractError("coaching pause exceeds budget")
        if type(owner_mid_answer) is not bool or type(playback) is not bool:
            raise ContractError("invalid coaching timing flags")
        with self._condition:
            now = self._now()
            self._tick(now)
            lease = self._lease
            if (
                not self._enabled
                or lease is None
                or generation_id != lease.generation
                or turn.session_id != self._session
                or not turn.finalized
            ):
                return
            fingerprint = hashlib.sha256(wire_json(turn.to_dict())).digest()
            if turn.utterance_id in self._seen:
                if self._seen[turn.utterance_id] != fingerprint:
                    raise ContractError("conflicting finalized coaching transcript")
                return
            if turn.end_ms < self._last_end:
                raise ContractError("reversed coaching transcript timeline")
            if len(self._seen) >= MAX_FINALS:
                self._stop("turn_budget", disable=True)
                raise ContractError("coaching finalized turn budget exceeded")
            self._seen[turn.utterance_id] = fingerprint
            self._last_end = turn.end_ms
            # A new floor observation makes earlier suggestions stale even when the
            # provider has not finished. Host playback still has its own final gate.
            if self._pending is not None:
                self._pending.cancelled.set()
                self._pending = None
            self._advice = None
            current = CoachContext(turn, playback)
            # Evict whole turns; never invent text at a retention cut or partial attribution.
            if len(wire_json([current.to_dict()])) <= MAX_CONTEXT_BYTES:
                self._context.append((now, current))
            while self._context and (
                len(self._context) > MAX_CONTEXT_TURNS
                or turn.end_ms - self._context[0][1].turn.end_ms >= MAX_CONTEXT_SECONDS * 1000
                or len(wire_json([item.to_dict() for item in self._snapshot()])) > MAX_CONTEXT_BYTES
            ):
                self._context.popleft()
            cadence = max(MIN_CADENCE_SECONDS, self._profile.cooldown_ms / 1000)
            if (
                playback
                or owner_mid_answer
                or turn.overlap
                or turn.speaker_id is None
                or turn.speaker_provenance not in {"authored-fixture", "diarization-timeline"}
                or not turn.text.strip()
                or pause_ms < MIN_PAUSE_MS
                or self._busy
                or self._advice is not None
                or (self._last_call is not None and now - self._last_call < cadence)
                or not self._context
                or self._context[-1][1].turn is not turn
            ):
                return
            work = CoachWork(self._profile, self._snapshot(), turn, pause_ms, str(uuid4()))
            ticket = _Ticket(self._session, self._destination, lease, work, threading.Event())
            self._pending = ticket
            self._busy = True
            self._last_call = now
            worker = threading.Thread(target=self._compute, args=(ticket,), daemon=True)
            try:
                worker.start()
            except Exception:
                self._busy = False
                self._stop("provider_failed", disable=True)
                raise ContractError("coaching provider unavailable") from None

    def _valid_ticket(self, ticket: _Ticket) -> bool:
        return (
            not ticket.cancelled.is_set()
            and self._enabled
            and self._lease is not None
            and ticket.session == self._session
            and ticket.destination == self._destination
            and ticket.lease.generation == self._lease.generation
            and ticket.work.profile == self._profile
        )

    def _compute(self, ticket: _Ticket) -> None:
        result = None
        failed = False
        try:
            raw = self._provider.advise(ticket.work, ticket.cancelled)
            if raw is not None:
                if type(raw) is not CoachSuggestion:
                    raise ContractError("invalid coaching provider result")
                result = CoachSuggestion(raw.text, raw.provider, raw.model, raw.confidence)
        except Exception:
            # No exception text, transcript, speaker ID or provider payload enters state.
            failed = True
        with self._condition:
            try:
                now = self._now()
                self._tick(now)
                if self._pending is ticket and self._valid_ticket(ticket):
                    if failed:
                        self._stop("provider_failed", disable=True)
                    elif result is not None:
                        if (
                            len(wire_json(self._advice_event(ticket, result, now)))
                            <= MAX_ADVICE_BYTES
                        ):
                            self._advice = (ticket, result)
                        else:
                            self._stop("provider_failed", disable=True)
            except ContractError:
                pass  # _now already clears content and reports a sanitized clock failure.
            finally:
                if self._pending is ticket:
                    self._pending = None
                self._busy = False
                self._condition.notify_all()

    def _advice_event(
        self, ticket: _Ticket, suggestion: CoachSuggestion, now: float
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "type": "coaching_advice",
            "session_id": ticket.session,
            "generation_id": ticket.lease.generation,
            "destination_id": ticket.destination,
            "qualification_id": ticket.lease.qualification,
            "request_id": ticket.work.request_id,
            "listener_id": ticket.work.profile.listener_id,
            "delivery": "owner_private",
            "valid_for_ms": math.floor((ticket.lease.deadline - now) * 1000),
            "text": suggestion.text,
            "decision": {
                "label": "speak_now",
                "confidence": suggestion.confidence,
                "provider": suggestion.provider,
                "model": suggestion.model,
            },
        }

    def drain(self) -> list[dict[str, Any]]:
        with self._condition:
            now = self._now()
            self._tick(now)
            events = []
            if self._state_event is not None:
                events.append(dict(self._state_event))
                self._state_event = None
            if self._advice is not None:
                ticket, suggestion = self._advice
                self._advice = None
                event = self._advice_event(ticket, suggestion, now)
                if (
                    self._valid_ticket(ticket)
                    and 1 <= event["valid_for_ms"] <= MAX_QUALIFICATION_MS
                ):
                    events.append(event)
            return events

    def wait_idle(self, timeout: float = 0.5) -> bool:
        """Bounded receipt for local authored replay; does not advance the injected clock."""
        if type(timeout) not in (float, int) or not math.isfinite(timeout) or not 0 <= timeout <= 1:
            raise ContractError("invalid coaching receipt timeout")
        with self._condition:
            return self._condition.wait_for(lambda: not self._busy, timeout=timeout)

    def close(self) -> None:
        with self._condition:
            if not self._closed:
                self._stop("closed", disable=True)
                self._closed = True
