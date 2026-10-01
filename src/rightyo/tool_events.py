"""Versioned bounded speech events; no routing, replies, or application actions."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from threading import RLock

from rightyo.contracts import ContractError, DecisionEvent, Turn, identifier, integer
from rightyo.memory import TranscriptMemory

MAX_EVENT_BYTES = 1200000
MAX_QUEUE_BYTES = 4194304
MAX_PENDING_BYTES = 1048576


class SpeechEvents:
    """Local v1 producer. Frozen request context never observes later turns.

    Consumers still need their own stale-session and once-only dispatch policy.
    Overflow fails closed rather than silently losing a request. Cancellation
    releases pending plaintext; a normal end preserves already queued delivery.
    """

    def __init__(self, *, retention_ms=300000, max_pending=128):
        integer(max_pending, "max_pending", 2)
        if max_pending > 128:
            raise ContractError("invalid event queue budget")
        self.retention_ms = retention_ms
        self.max_pending = max_pending
        self._lock = RLock()
        self._memory = TranscriptMemory(retention_ms=retention_ms)
        self._session = None
        self._sequence = self._now = 0
        self._active = False
        self._terminal = False
        self._queue = deque()
        self._queue_bytes = 0
        self._pending = {}
        self._pending_bytes = 0
        self._seen = {}
        self._decided = set()

    def _emit(self, kind, **fields):
        self._sequence += 1
        payload = {
            "schema_version": 1,
            "type": kind,
            "session_id": self._session,
            "sequence": self._sequence,
            "emitted_at_ms": self._now,
            **fields,
        }
        size = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        if (
            size > MAX_EVENT_BYTES
            or len(self._queue) >= self.max_pending
            or self._queue_bytes + size > MAX_QUEUE_BYTES
        ):
            self._clear_content()
            self._active = False
            raise ContractError("speech event consumer backlog exceeded")
        self._queue.append((payload, size))
        self._queue_bytes += size

    def _clear_content(self):
        self._memory.clear()
        self._pending.clear()
        self._pending_bytes = 0
        self._queue.clear()
        self._queue_bytes = 0

    def start(self, session_id, now_ms=0, *, attention_enabled=True):
        with self._lock:
            identifier(session_id, "session_id")
            integer(now_ms, "now_ms")
            if type(attention_enabled) is not bool:
                raise ContractError("invalid attention capability")
            if self._active or self._queue:
                raise ContractError("finish and drain the previous event session")
            if session_id == self._session:
                raise ContractError("restart requires a new event session identity")
            self._clear_content()
            self._seen.clear()
            self._decided.clear()
            self._session = session_id
            self._sequence = 0
            self._now = now_ms
            self._active = True
            self._terminal = False
            self._emit(
                "session",
                phase="started",
                capabilities={
                    "activation": "finalized-turn" if attention_enabled else "disabled",
                    "partials": False,
                    "speakers": "anonymous",
                    "context": True,
                },
            )

    def expire(self, now_ms):
        with self._lock:
            integer(now_ms, "now_ms")
            self._now = max(self._now, now_ms)
            self._memory.expire(self._now)
            cutoff = self._now - self.retention_ms
            for key, (turn, _context, size) in list(self._pending.items()):
                if turn.end_ms <= cutoff:
                    del self._pending[key]
                    self._pending_bytes -= size
            # Pending frozen contexts must also obey elapsed retention at emission.
            for _key, (_turn, context, _size) in self._pending.items():
                context["turns"] = [t for t in context["turns"] if t["end_ms"] > cutoff]
            kept = deque()
            for payload, size in self._queue:
                if payload["type"] in {"transcript", "request", "attention"}:
                    end = payload.get("turn", {}).get("end_ms", payload.get("speech_end_ms"))
                    if end is not None and end <= cutoff:
                        self._queue_bytes -= size
                        continue
                    if payload["type"] == "request":
                        payload["context"]["turns"] = [
                            t for t in payload["context"]["turns"] if t["end_ms"] > cutoff
                        ]
                kept.append((payload, size))
            self._queue = kept

    def transcript(self, turn, now_ms, *, expect_decision=True):
        with self._lock:
            if not self._active:
                raise ContractError("speech event session is not active")
            if not isinstance(turn, Turn) or not turn.finalized or turn.session_id != self._session:
                raise ContractError("event producer requires a final turn in its active session")
            if type(expect_decision) is not bool:
                raise ContractError("invalid decision expectation")
            integer(now_ms, "now_ms")
            self.expire(max(now_ms, turn.end_ms))
            digest = hashlib.sha256(
                json.dumps(turn.to_dict(), sort_keys=True).encode("utf-8")
            ).digest()
            previous = self._seen.get(turn.utterance_id)
            if previous is not None:
                if previous != digest:
                    raise ContractError("final event turn cannot be revised")
                return
            if len(self._seen) >= 1000:
                raise ContractError("event session turn budget exceeded")
            context = self._memory.snapshot(self._now)
            context["turns"] = [t for t in context["turns"] if t["end_ms"] <= turn.start_ms]
            self._memory.append(turn)
            self._seen[turn.utterance_id] = digest
            if turn.end_ms <= self._now - self.retention_ms:
                return
            if expect_decision:
                size = len(json.dumps(context, ensure_ascii=False).encode("utf-8"))
                if len(self._pending) >= 32 or self._pending_bytes + size > MAX_PENDING_BYTES:
                    self._clear_content()
                    self._active = False
                    raise ContractError("pending attention context budget exceeded")
                self._pending[turn.utterance_id] = (turn, context, size)
                self._pending_bytes += size
            self._emit("transcript", turn=turn.to_dict())

    def decision(self, event, now_ms):
        with self._lock:
            if not self._active:
                return
            if not isinstance(event, DecisionEvent) or event.turn.session_id != self._session:
                raise ContractError("decision event belongs to another session")
            integer(now_ms, "now_ms")
            self.expire(max(now_ms, event.turn.end_ms))
            key = event.turn.utterance_id
            if key in self._decided:
                return
            pending = self._pending.pop(key, None)
            if pending is None:
                return
            turn, context, size = pending
            self._pending_bytes -= size
            if turn != event.turn:
                raise ContractError("decision does not match the committed transcript")
            self._decided.add(key)
            # Keep public evidence separate from recipient-based host target selection.
            evidence = {
                name: value
                for name, value in event.public_dict().items()
                if name in {"label", "recipient_kind", "confidence", "provider", "model"}
            }
            request_id = self._session + ":" + key
            attended = evidence["label"] == "attend" and evidence["recipient_kind"] == "system"
            self._emit(
                "attention",
                utterance_id=key,
                speech_end_ms=turn.end_ms,
                decision=evidence,
                **({"request_id": request_id} if attended else {}),
            )
            if attended:
                self._emit(
                    "request",
                    request_id=request_id,
                    turn=turn.to_dict(),
                    decision=evidence,
                    context=context,
                    decision_at_ms=self._now,
                )

    def end(self, phase="cancelled", now_ms=0, reason=None):
        with self._lock:
            if phase not in {"stopped", "cancelled", "error"}:
                raise ContractError("invalid event terminal phase")
            integer(now_ms, "now_ms")
            if reason is not None:
                identifier(reason, "reason")
            if self._session is None or self._terminal:
                return
            self.expire(now_ms)
            if phase != "stopped":
                self._queue.clear()
                self._queue_bytes = 0
            self._memory.clear()
            self._pending.clear()
            self._pending_bytes = 0
            self._active = False
            self._terminal = True
            self._emit("session", phase=phase, **({"reason": reason} if reason else {}))

    def drain(self):
        with self._lock:
            result = [payload for payload, _size in self._queue]
            self._queue.clear()
            self._queue_bytes = 0
            return copy.deepcopy(result)
