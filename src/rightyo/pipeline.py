"""Bounded, causal transcript context and once-only committed decisions."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from functools import wraps
from threading import RLock
from typing import Any

from rightyo.contracts import ContractError, DecisionEvent, Turn, identifier
from rightyo.memory import TranscriptMemory
from rightyo.providers import DecisionProvider


def _locked(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded


@dataclass(frozen=True)
class _Revision:
    revision: int
    digest: bytes


def _fingerprint(turn: Turn) -> _Revision:
    encoded = json.dumps(turn.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return _Revision(turn.revision, hashlib.sha256(encoded).digest())


def _utterance_key(turn: Turn) -> str:
    return hashlib.sha256(turn.utterance_id.encode("utf-8")).hexdigest()


class ReplayRunner:
    """Single-threaded explicit-session runner; partial turns never call the provider.

    For file replay this sees only past completed turns, not future file content. This does
    not simulate incremental acoustic processing or establish live latency.
    """

    def __init__(
        self,
        provider: DecisionProvider,
        *,
        max_context_turns: int = 8,
        max_context_chars: int = 12000,
        max_utterances: int = 1000,
        no_speakers: bool = False,
        playback_active: bool = False,
        expected_reply: str | None = None,
        memory: TranscriptMemory | None = None,
    ) -> None:
        if not 1 <= max_context_turns <= 32 or not 4000 <= max_context_chars <= 16000:
            raise ContractError("invalid context budget")
        if not 1 <= max_utterances <= 1000:
            raise ContractError("invalid utterance budget")
        if expected_reply is not None and (
            not isinstance(expected_reply, str) or len(expected_reply) > 1000
        ):
            raise ContractError("invalid expected reply context")
        self._lock = RLock()
        self.provider = provider
        self.memory = memory
        self.failed_decisions = 0
        self.max_context_turns = max_context_turns
        self.max_context_chars = max_context_chars
        self.max_utterances = max_utterances
        self.no_speakers = no_speakers
        self.playback_active = playback_active
        self.expected_reply = expected_reply
        self.session_id: str | None = None
        self._epoch = 0
        # Retain revision hashes after expiry, never old partial/final plaintext.
        self._latest: dict[str, _Revision] = {}
        self._sealed: set[str] = set()
        self._emitted: set[str] = set()
        self._history: deque[Turn] = deque(maxlen=max_context_turns)
        self._last_end_ms = 0
        self.skipped = 0
        self.partial_turns = 0

    @_locked
    def restart(self, session_id: str) -> None:
        identifier(session_id, "session_id")
        self._epoch += 1
        self.session_id = session_id
        self._latest.clear()
        self._emitted.clear()
        self._sealed.clear()
        self.failed_decisions = 0
        if self.memory is not None:
            self.memory.clear()
        self._history.clear()
        self._last_end_ms = 0

    @_locked
    def clear(self) -> None:
        """Invalidate outstanding decisions and release runner-owned conversation text."""
        self._epoch += 1
        self.session_id = None
        self._latest.clear()
        self._emitted.clear()
        self._sealed.clear()
        self._history.clear()
        self.expected_reply = None
        self._last_end_ms = 0
        self.failed_decisions = 0
        if self.memory is not None:
            self.memory.clear()

    @_locked
    def expire(self, now_ms: int) -> None:
        if self.memory is not None:
            self.memory.expire(now_ms)
            retained = self.memory.retained_ids
            self._history = deque(
                (turn for turn in self._history if turn.utterance_id in retained),
                maxlen=self.max_context_turns,
            )

    @_locked
    def snapshot(self, now_ms: int) -> dict[str, Any]:
        self.expire(now_ms)
        result = (
            self.memory.snapshot(now_ms)
            if self.memory is not None
            else {"turns": [], "retention": None}
        )
        result["decisions"] = {
            "completed": len(self._emitted),
            "failed": self.failed_decisions,
            "incomplete": self.failed_decisions > 0,
        }
        return result

    def _retain_final(self, turn: Turn, key: str) -> None:
        self._sealed.add(key)
        self._history.append(turn)
        self._last_end_ms = turn.end_ms
        if self.memory is not None:
            self.memory.append(turn)
            self.expire(turn.end_ms)

    def _turn_state(self, turn: Turn) -> dict[str, Any]:
        # Session/source identifiers are not needed for addressee inference.
        return {
            "text": turn.text,
            "speaker_id": None if self.no_speakers else turn.speaker_id,
            "start_ms": turn.start_ms,
            "end_ms": turn.end_ms,
            "overlap": turn.overlap,
        }

    def _state(self, turn: Turn) -> dict[str, Any]:
        past: list[Turn] = []
        chars = len(turn.text)
        for previous in reversed(self._history):
            if chars + len(previous.text) > self.max_context_chars:
                break
            chars += len(previous.text)
            past.insert(0, previous)
        participants = sorted(
            {
                item.speaker_id
                for item in [*past, turn]
                if item.speaker_id is not None and not self.no_speakers
            }
        )
        return {
            "past_turns": [self._turn_state(item) for item in past],
            "current_turn": self._turn_state(turn),
            "known_participants": participants,
            "expected_reply": self.expected_reply,
            "playback_active": self.playback_active,
        }

    def process(self, turn: Turn) -> DecisionEvent | None:
        started = time.perf_counter()
        with self._lock:
            if self.session_id is None:
                self.restart(turn.session_id)
            if turn.session_id != self.session_id:
                raise ContractError("session changed without explicit restart")
            key = _utterance_key(turn)
            current = _fingerprint(turn)
            previous = self._latest.get(key)
            if previous is not None:
                if turn.revision < previous.revision:
                    self.skipped += 1
                    return None
                if turn.revision == previous.revision:
                    if current != previous:
                        raise ContractError("same turn revision has conflicting content")
                    self.skipped += 1
                    return None
                if key in self._sealed:
                    raise ContractError("a committed utterance cannot be revised")
            elif len(self._latest) >= self.max_utterances:
                # Do not evict old IDs and accidentally emit them twice.
                raise ContractError("utterance budget exhausted; explicitly start a new session")
            if turn.finalized and turn.end_ms < self._last_end_ms:
                raise ContractError("final turns must arrive in timestamp order")
            self.expire(turn.end_ms)
            self._latest[key] = current
            if not turn.finalized:
                self.partial_turns += 1
                return None
            epoch = self._epoch
            state = self._state(turn)
        provider_started = time.perf_counter()
        try:
            decision = self.provider.decide(state)
        except Exception:
            with self._lock:
                if self._epoch == epoch and self._latest.get(key) == current:
                    # ASR completed even when the hosted decision did not. Preserve
                    # local recall, without fabricating a completed attention event.
                    self._retain_final(turn, key)
                    self.failed_decisions += 1
            raise
        provider_ms = (time.perf_counter() - provider_started) * 1000
        with self._lock:
            if self._epoch != epoch or self._latest.get(key) != current:
                self.skipped += 1
                return None
            self._emitted.add(key)
            self._retain_final(turn, key)
            return DecisionEvent(
                turn, decision, turn.revision, provider_ms, (time.perf_counter() - started) * 1000
            )
