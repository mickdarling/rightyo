"""Bounded local transcript retention, independent of attention decisions."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from functools import wraps
from threading import RLock
from typing import Any

from rightyo.contracts import ContractError, Turn, integer


def _locked(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return guarded


class MemorySessionLimitError(ContractError):
    """Unique-turn bookkeeping reached the explicit per-session bound."""


class TranscriptMemory:
    """Retain complete final turns by end time, with explicit boundary coverage.

    A turn straddling the cutoff is retained whole: trimming text to a timestamp
    would fabricate word timing. Snapshots disclose that extra coverage. This is
    process memory, not an archive or a promise of secure physical erasure.
    """

    def __init__(
        self,
        *,
        retention_ms: int = 300000,
        max_turns: int = 1000,
        max_bytes: int = 1048576,
    ) -> None:
        integer(retention_ms, "retention_ms", 1)
        integer(max_turns, "max_turns", 1)
        integer(max_bytes, "max_bytes", 1)
        if retention_ms > 900000 or max_turns > 1000 or max_bytes > 4194304:
            raise ContractError("invalid transcript memory budget")
        self._lock = RLock()
        self.retention_ms = retention_ms
        self.max_turns = max_turns
        self.max_bytes = max_bytes
        self.clear()

    @_locked
    def clear(self) -> None:
        self._turns: deque[tuple[Turn, int]] = deque()
        self._bytes = 0
        self._now_ms = 0
        self._last_end_ms = 0
        self._session_id: str | None = None
        self._seen: dict[bytes, bytes] = {}
        self._expired = 0
        self._capacity_evicted = 0

    @_locked
    def expire(self, now_ms: int) -> None:
        integer(now_ms, "now_ms")
        # Late processing cannot move the retention clock backwards.
        self._now_ms = max(self._now_ms, now_ms)
        cutoff = self._now_ms - self.retention_ms
        while self._turns and self._turns[0][0].end_ms <= cutoff:
            _, size = self._turns.popleft()
            self._bytes -= size
            self._expired += 1

    @_locked
    def append(self, turn: Turn) -> None:
        if not isinstance(turn, Turn) or not turn.finalized:
            raise ContractError("transcript memory accepts only final turns")
        if self._session_id is not None and turn.session_id != self._session_id:
            raise ContractError("memory session changed without explicit clear")
        key = hashlib.sha256(turn.utterance_id.encode("utf-8")).digest()
        encoded = json.dumps(turn.to_dict(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        digest = hashlib.sha256(encoded).digest()
        previous = self._seen.get(key)
        if previous is not None:
            if previous != digest:
                raise ContractError("final transcript memory turn cannot be revised")
            return
        if len(self._seen) >= 1000:
            raise MemorySessionLimitError(
                "Session reached its 1,000-turn limit; start a new session."
            )
        if turn.end_ms < self._last_end_ms:
            raise ContractError("memory final turns must arrive in timestamp order")
        self._seen[key] = digest
        self._session_id = turn.session_id
        self._last_end_ms = turn.end_ms
        size = len(encoded)
        self.expire(turn.end_ms)
        if turn.end_ms <= self._now_ms - self.retention_ms:
            self._expired += 1
            return
        self._turns.append((turn, size))
        self._bytes += size
        while len(self._turns) > self.max_turns or self._bytes > self.max_bytes:
            _, removed_size = self._turns.popleft()
            self._bytes -= removed_size
            self._capacity_evicted += 1

    @property
    @_locked
    def retained_ids(self) -> frozenset[str]:
        return frozenset(turn.utterance_id for turn, _ in self._turns)

    @_locked
    def snapshot(self, now_ms: int) -> dict[str, Any]:
        """Return detached dictionaries; callers cannot mutate retained state."""
        self.expire(now_ms)
        cutoff = max(0, self._now_ms - self.retention_ms)
        turns = [turn.to_dict() for turn, _ in self._turns]
        return {
            "turns": turns,
            "retention": {
                "retention_ms": self.retention_ms,
                "max_turns": self.max_turns,
                "max_session_turns": 1000,
                "session_turn_count": len(self._seen),
                "max_bytes": self.max_bytes,
                "now_ms": self._now_ms,
                "cutoff_ms": cutoff,
                "turn_count": len(turns),
                "retained_bytes": self._bytes,
                "oldest_start_ms": min((turn["start_ms"] for turn in turns), default=None),
                "newest_end_ms": turns[-1]["end_ms"] if turns else None,
                "boundary_overlap_ms": max(
                    (max(0, cutoff - turn["start_ms"]) for turn in turns), default=0
                ),
                "expired_turns": self._expired,
                "capacity_evicted_turns": self._capacity_evicted,
                "boundary_policy": "retain whole turns whose end exceeds cutoff",
            },
        }
