"""Pair final speech events with measured activity from a separate trusted host lane.

Only the latest actual final can receive activity. Missing measurements preserve a
conservative context turn, never permission to speak. No text or speaker role is
interpreted as owner identity, playback state, qualification or a control.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from typing import Any, Callable

from rightyo.coaching import MAX_FINALS, MAX_TIMING_MS, CoachingSession, canonical_uuid, wire_json
from rightyo.coaching_input import CoachingInputAdapter, validate_coaching_transcript
from rightyo.contracts import ContractError, identifier, integer
from rightyo.tool_events import speaker_name

MAX_ACTIVITY_DELAY_MS = 2000


def validate_coaching_activity(raw: Any) -> dict[str, Any]:
    fields = {
        "schema_version",
        "type",
        "session_id",
        "generation_id",
        "utterance_id",
        "pause_ms",
        "owner_mid_answer",
        "playback",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ContractError("invalid coaching activity fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ContractError("unsupported coaching activity version")
    if raw["type"] != "coaching_activity":
        raise ContractError("invalid coaching activity type")
    canonical_uuid(raw["session_id"])
    canonical_uuid(raw["generation_id"])
    identifier(raw["utterance_id"], "coaching activity utterance id")
    integer(raw["pause_ms"], "coaching activity pause")
    if raw["pause_ms"] > MAX_TIMING_MS:
        raise ContractError("coaching activity pause exceeds budget")
    if type(raw["owner_mid_answer"]) is not bool or type(raw["playback"]) is not bool:
        raise ContractError("invalid coaching activity flags")
    return dict(raw)


class CoachingActivityAdapter:
    """One immutable child/capture binding and one final awaiting trusted activity.

    The runtime serializes transcript and control delivery through this adapter and
    calls tick while idle. Boundary failures cancel advice and discard the pending
    plaintext; the owning runtime then retires the child. This cannot retract advice
    already handed to a downstream consumer.
    """

    def __init__(
        self,
        session: CoachingSession,
        *,
        session_id: str,
        generation_id: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._session_id = canonical_uuid(session_id)
        self._generation_id = canonical_uuid(generation_id)
        self._input = CoachingInputAdapter(session, generation_id=generation_id)
        self._session = session
        if not callable(clock):
            raise ContractError("invalid coaching activity clock")
        self._clock = clock
        self._last_clock: float | None = None
        self._lock = threading.RLock()
        self._pending: tuple[dict[str, Any], float] | None = None
        self._seen: dict[str, bytes] = {}
        self._last_sequence = 0
        self._last_end = 0
        self._closed = False

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
            raise ContractError("invalid coaching activity clock") from None
        self._last_clock = now
        return now

    def _permitted(self) -> bool:
        return self._session.capture_permitted_for(
            session_id=self._session_id, generation_id=self._generation_id
        )

    def _require_permitted(self) -> None:
        if self._closed or not self._permitted():
            raise ContractError("coaching activity scope is not qualified")

    def _discard(self) -> None:
        self._pending = None
        self._session.cancel_pending()

    def _hold(self) -> None:
        if self._pending is None:
            return
        event, _ = self._pending
        self._pending = None
        if self._permitted():
            self._input.accept(event, pause_ms=0, owner_mid_answer=True, playback=True)

    @property
    def pending(self) -> bool:
        """Whether a final still awaits activity; tick determines idle expiry."""
        with self._lock:
            return self._pending is not None

    def accept_transcript(self, event: Any) -> dict[str, Any]:
        """Validate before host emission, cancel older advice, then await activity."""
        with self._lock:
            try:
                turn = validate_coaching_transcript(event)
                if not turn.finalized or turn.session_id != self._session_id:
                    raise ContractError("invalid coaching activity transcript scope")
                if "speaker_name" in event and speaker_name(event["speaker_name"]) is None:
                    raise ContractError("invalid coaching transcript display name")
                identifier(turn.utterance_id, "coaching utterance id")
                fingerprint = hashlib.sha256(wire_json(turn.to_dict())).digest()
                if turn.utterance_id in self._seen:
                    reason = (
                        "duplicate coaching activity transcript"
                        if self._seen[turn.utterance_id] == fingerprint
                        else "conflicting coaching activity transcript"
                    )
                    raise ContractError(reason)
                if event["sequence"] <= self._last_sequence or turn.end_ms < self._last_end:
                    raise ContractError("reversed coaching activity transcript")
                if len(self._seen) >= MAX_FINALS:
                    raise ContractError("coaching activity final budget exceeded")
                now = self._now()
                self._require_permitted()
                # Keep only validated bounded content; neither caller nor returned
                # event can mutate the retained final after this boundary.
                saved = {key: value for key, value in event.items() if key != "turn"}
                saved["turn"] = turn.to_dict()
                if "role" in event["turn"]:
                    saved["turn"]["role"] = event["turn"]["role"]
                self._session.cancel_pending()
                self._hold()
                self._seen[turn.utterance_id] = fingerprint
                self._last_sequence = event["sequence"]
                self._last_end = turn.end_ms
                self._pending = (saved, now)
                return {**saved, "turn": dict(saved["turn"])}
            except ContractError:
                self._discard()
                raise

    def accept_activity(self, raw: Any) -> None:
        """The caller supplies trusted host provenance; shape alone grants none."""
        with self._lock:
            try:
                activity = validate_coaching_activity(raw)
                now = self._now()
                self._require_permitted()
                if (
                    activity["session_id"] != self._session_id
                    or activity["generation_id"] != self._generation_id
                    or self._pending is None
                    or activity["utterance_id"] != self._pending[0]["turn"]["utterance_id"]
                ):
                    raise ContractError("coaching activity has no matching final")
                if now - self._pending[1] >= MAX_ACTIVITY_DELAY_MS / 1000:
                    self._hold()
                    raise ContractError("coaching activity arrived too late")
                event, _ = self._pending
                self._pending = None
                self._input.accept(
                    event,
                    pause_ms=activity["pause_ms"],
                    owner_mid_answer=activity["owner_mid_answer"],
                    playback=activity["playback"],
                )
            except ContractError:
                self._discard()
                raise

    def tick(self) -> None:
        with self._lock:
            try:
                now = self._now()
                self._session.tick()
                if self._closed or not self._permitted():
                    self._discard()
                elif (
                    self._pending is not None
                    and now - self._pending[1] >= MAX_ACTIVITY_DELAY_MS / 1000
                ):
                    self._hold()
            except ContractError:
                self._discard()
                raise

    def finish_pending(self) -> None:
        """Finite PCM EOF fallback after the host's bounded activity receipt window."""
        with self._lock:
            try:
                self._now()
                self._hold()
            except ContractError:
                self._discard()
                raise

    def close(self) -> None:
        """Fatal/retired input discards pending text; it is never appended as context."""
        with self._lock:
            self._discard()
            self._seen.clear()
            self._closed = True
