"""Versioned bounded speech events; no routing, replies, or application actions."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from threading import RLock

from rightyo.contracts import (
    Addressing,
    ContractError,
    DecisionEvent,
    SpeakerPriority,
    Turn,
    formed_request_text,
    identifier,
    integer,
    speaker_role,
)
from rightyo.memory import TranscriptMemory
from rightyo.providers import ConfiguredPriorityProvider, ProviderError

MAX_EVENT_BYTES = 1200000
MAX_QUEUE_BYTES = 4194304
MAX_PENDING_BYTES = 1048576
# The context a role provider sees when a speaker first appears.
ROLE_CONTEXT_TURNS = 8


def _role_state(record):
    """The bounded fields a role provider needs; session/source identifiers are omitted."""
    state = {name: record[name] for name in ("text", "speaker_id", "start_ms", "end_ms", "overlap")}
    if "role" in record:
        state["role"] = record["role"]
    return state


def _with_role(turn, role):
    record = turn.to_dict()
    if role is not None:
        record["role"] = role
    return record


def encode_json(payload):
    """The ASCII-safe JSON representation shared by wire output and byte budgets."""
    return json.dumps(payload, ensure_ascii=True, allow_nan=False)


class SpeechEvents:
    """Local v1 producer. Frozen request context never observes later turns.

    Consumers still need their own stale-session and once-only dispatch policy.
    Overflow fails closed rather than silently losing a request. Cancellation
    releases pending plaintext; a normal end preserves already queued delivery.
    """

    def __init__(self, *, retention_ms=300000, max_pending=128):
        # At least five: one open request plus the owner's transcript, attention and
        # request, with a terminal slot, so the static and dynamic override bounds agree.
        integer(max_pending, "max_pending", 5)
        if max_pending > 128:
            raise ContractError("invalid event queue budget")
        self.retention_ms = retention_ms
        self.max_pending = max_pending
        # Open non-owner requests are bounded below the queue capacity, with headroom for
        # the owner's transcript, attention and request, so one owner decision's burst of
        # overrides can never overflow the queue; exceeding the bound fails closed.
        self.max_open = max_pending - 4
        self._lock = RLock()
        self._memory = TranscriptMemory(retention_ms=retention_ms)
        self._session = None
        self._sequence = self._now = 0
        self._active = False
        self._terminal = False
        self._attention_enabled = True
        self._sessions = set()
        self._queue = deque()
        self._queue_bytes = 0
        self._pending = {}
        self._pending_bytes = 0
        self._seen = {}
        self._decided = set()
        # Speaker roles are fixed per speaker when first emitted; open requests are the
        # emitted non-owner requests an owner's attended turn or stop phrase supersedes.
        self._priority = None
        self._roles = {}
        self._open = {}
        # Pending non-owner turns an owner superseded before their decisions arrived,
        # mapped to the owner's utterance: their late decisions never emit a request.
        self._superseded = {}
        # "off" without a provider, "ready" with one, "unavailable" after a hosted role
        # question failed, or "rejected" after a provider contract violation; either
        # degrades roles to configured/unknown for the rest of the session.
        self.role_status = "off"
        self._degraded_turn = None
        # Optional request forming renders a convenience string beside the raw turns;
        # off by default, and the raw turn/context are unchanged either way.
        self._former = None
        self._addressing = None

    def _payload(self, kind, sequence, fields):
        return {
            "schema_version": 1,
            "type": kind,
            "session_id": self._session,
            "sequence": sequence,
            "emitted_at_ms": self._now,
            **fields,
        }

    @staticmethod
    def _size(payload):
        return len(encode_json(payload)) + 1  # Include the JSONL newline.

    def _emit(self, kind, **fields):
        self._sequence += 1
        payload = self._payload(kind, self._sequence, fields)
        size = self._size(payload)
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
        self._open.clear()
        self._superseded.clear()

    def start(
        self,
        session_id,
        now_ms=0,
        *,
        attention_enabled=True,
        addressing=None,
        priority=None,
        former=None,
    ):
        with self._lock:
            identifier(session_id, "session_id")
            integer(now_ms, "now_ms")
            if type(attention_enabled) is not bool:
                raise ContractError("invalid attention capability")
            if addressing is not None and not isinstance(addressing, Addressing):
                raise ContractError("invalid addressing")
            if priority is not None and (
                not callable(getattr(priority, "assign", None))
                or not isinstance(getattr(priority, "priority", None), SpeakerPriority)
            ):
                raise ContractError("invalid speaker priority provider")
            if former is not None and not callable(getattr(former, "form", None)):
                raise ContractError("invalid request former")
            if former is not None:
                identifier(getattr(former, "kind", None), "request former kind")
            if self._active or self._queue:
                raise ContractError("finish and drain the previous event session")
            session_key = hashlib.sha256(session_id.encode("utf-8")).digest()
            if session_key in self._sessions:
                raise ContractError("restart requires a new event session identity")
            if len(self._sessions) >= 1000:
                raise ContractError("event publisher session budget exceeded")
            self._sessions.add(session_key)
            self._attention_enabled = attention_enabled
            self._clear_content()
            self._seen.clear()
            self._decided.clear()
            self._roles.clear()
            self._priority = priority
            self._former = former
            self._addressing = addressing
            self.role_status = "off" if priority is None else "ready"
            self._degraded_turn = None
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
                    "speakers": "anonymous" if priority is None else "enrolled",
                    "context": True,
                },
                # Configured forms of address are advertised beside, not inside, the
                # existing capability set so strict consumers of that set are unchanged.
                **({} if addressing is None else {"addressing": addressing.to_dict()}),
                # Likewise for request forming: a separate top-level object, never a key
                # inside the strictly validated capability set.
                **({} if former is None else {"request_forming": {"kind": former.kind}}),
            )

    def expire(self, now_ms):
        with self._lock:
            integer(now_ms, "now_ms")
            self._now = max(self._now, now_ms)
            self._memory.expire(self._now)
            cutoff = self._now - self.retention_ms
            for key, (turn, _context, size, _role) in list(self._pending.items()):
                if turn.end_ms <= cutoff:
                    del self._pending[key]
                    self._pending_bytes -= size
            # Reclaim accounting as well as plaintext when a frozen context shrinks.
            self._pending_bytes = 0
            for key, (turn, context, _size, role) in list(self._pending.items()):
                context["turns"] = [t for t in context["turns"] if t["end_ms"] > cutoff]
                size = len(encode_json(context))
                self._pending[key] = (turn, context, size, role)
                self._pending_bytes += size
            for request_id, end_ms in list(self._open.items()):
                if end_ms <= cutoff:
                    del self._open[request_id]
            for key in list(self._superseded):
                if key not in self._pending:
                    del self._superseded[key]
            kept = deque()
            self._queue_bytes = 0
            for payload, _size in self._queue:
                if payload["type"] in {"transcript", "request", "attention"}:
                    end = payload.get("turn", {}).get("end_ms", payload.get("speech_end_ms"))
                    if end is not None and end <= cutoff:
                        continue
                    if payload["type"] == "request":
                        payload["context"]["turns"] = [
                            t for t in payload["context"]["turns"] if t["end_ms"] > cutoff
                        ]
                size = len(encode_json(payload)) + 1
                kept.append((payload, size))
                self._queue_bytes += size
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
            # The role is fixed before the first emission and repeated unchanged on the
            # decision, the request turn and every later context snapshot.
            role = None if self._priority is None else self._assign_role(turn, context["turns"])
            self._memory.append(turn, role=role)
            self._seen[turn.utterance_id] = digest
            if turn.end_ms <= self._now - self.retention_ms:
                return
            if expect_decision and self._attention_enabled:
                size = len(encode_json(context))
                if len(self._pending) >= 32 or self._pending_bytes + size > MAX_PENDING_BYTES:
                    self._clear_content()
                    self._active = False
                    raise ContractError("pending attention context budget exceeded")
                self._pending[turn.utterance_id] = (turn, context, size, role)
                self._pending_bytes += size
            self._emit("transcript", turn=_with_role(turn, role))

    def _assign_role(self, turn, past):
        """Fix a speaker's role the first time that speaker is emitted in this session."""
        if turn.speaker_id is None:
            return "unknown"
        role = self._roles.get(turn.speaker_id)
        if role is not None:
            return role
        recent = [_role_state(record) for record in past[-ROLE_CONTEXT_TURNS:]]
        current = _role_state(turn.to_dict())
        participants = sorted(
            {r["speaker_id"] for r in [*recent, current] if r["speaker_id"] is not None}
        )
        state = {
            "past_turns": recent,
            "current_turn": current,
            "known_participants": participants,
            # Only the retained participants' roles travel with the request: the whole
            # session's role map could exceed the payload budget, which prunes turns only.
            "roles": {s: self._roles[s] for s in participants if s in self._roles},
        }
        rules = self._priority.priority
        degraded = None
        try:
            assigned = self._priority.assign(state)
            if not isinstance(assigned, dict):
                raise ContractError("invalid speaker role assignment")
            validated = {
                identifier(speaker, "speaker_id"): speaker_role(value)
                for speaker, value in assigned.items()
            }
            if any(v == "owner" and s not in rules.owners for s, v in validated.items()):
                # Owners come only from configuration; a provider naming one is rejected.
                degraded = "rejected"
            if not set(validated) <= set(state["known_participants"]):
                # A role for a speaker not yet observed would let that speaker's first
                # real turn skip the provider; treat it as a contract violation.
                degraded = "rejected"
            # The configured overlay always wins: a configured owner or trusted speaker
            # keeps that role whatever the provider answered or omitted.
            validated = {s: rules.configured_role(s) or v for s, v in validated.items()}
            configured = rules.configured_role(turn.speaker_id)
            if configured is not None:
                validated[turn.speaker_id] = configured
        except ProviderError:
            # A hosted role question failed (timeout, unavailability, budget, cancellation).
            degraded = "unavailable"
        if degraded is not None:
            # Degrade rather than abort: this speaker takes the configured role or unknown,
            # later speakers use configured roles only, and the session keeps listening.
            self._priority = ConfiguredPriorityProvider(rules)
            self.role_status = degraded
            # The turn whose decision evidence will carry the degradation on the stream.
            self._degraded_turn = turn.utterance_id
            validated = {turn.speaker_id: rules.configured_role(turn.speaker_id) or "unknown"}
        for speaker, value in validated.items():
            # An earlier fixed role is never revised by a later answer.
            self._roles.setdefault(speaker, value)
        # A provider that omits the current speaker still fixes that speaker's role.
        return self._roles.setdefault(turn.speaker_id, "unknown")

    def decision(self, event, now_ms):
        with self._lock:
            if not self._active or not self._attention_enabled:
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
            turn, context, size, role = pending
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
            stop = False
            if role is not None:
                evidence["role"] = role
                if key == self._degraded_turn:
                    # The stream shows where model lookups degraded, not only the object.
                    evidence["role_status"] = self.role_status
                rules = self._priority.priority
                if rules.owner_only and role != "owner" and evidence["label"] == "attend":
                    # Owner-only mode: other speakers remain context, never a request.
                    evidence["label"] = "ignore"
                stop = role == "owner" and rules.is_stop_phrase(turn.text)
            request_id = self._session + ":" + key
            superseded_by = self._superseded.pop(key, None)
            would_attend = evidence["label"] == "attend" and evidence["recipient_kind"] == "system"
            attended = would_attend and not stop and superseded_by is None
            overriding = role == "owner" and (attended or stop)
            to_supersede = []
            if overriding:
                # Earlier is decided by turn time (end_ms at or before the owner's), not by
                # emission order: decisions arrive out of order, and a request spoken after
                # the owner's turn is not what the owner was superseding.
                to_supersede = [r for r, end in self._open.items() if end <= turn.end_ms]
                # The whole burst plus the owner's own attention/request/terminal must fit
                # the undrained queue; otherwise fail closed before emitting any override,
                # never a partial batch.
                if len(to_supersede) + 3 > self.max_pending - len(self._queue):
                    self._clear_content()
                    self._active = False
                    raise ContractError("speech event consumer backlog exceeded")
                # Byte accounting with the exact payloads about to be emitted, in the
                # sequence order they will receive, plus a terminal event upper bound
                # (longest identifier reason, sequence digit growth), so the burst can
                # never trip the queue byte bound part way through.
                first = self._sequence + 1
                burst = sum(
                    self._size(
                        self._payload(
                            "override",
                            first + 1 + offset,
                            {"superseded_request_id": r, "by_utterance_id": key, "role": "owner"},
                        )
                    )
                    for offset, r in enumerate(to_supersede)
                )
                reserve = self._size(
                    self._payload(
                        "attention",
                        first,
                        {
                            "utterance_id": key,
                            "speech_end_ms": turn.end_ms,
                            "decision": evidence,
                            **({"request_id": request_id} if attended else {}),
                        },
                    )
                )
                after = first + 1 + len(to_supersede)
                if attended:
                    reserve += self._size(
                        self._payload(
                            "request",
                            after,
                            {
                                "request_id": request_id,
                                "turn": _with_role(turn, role),
                                "decision": evidence,
                                "context": context,
                                "decision_at_ms": self._now,
                            },
                        )
                    )
                reserve += 32 + self._size(
                    self._payload(
                        "session",
                        after + 1,
                        {"phase": "cancelled", "reason": "x" * 96, "role_status": "unavailable"},
                    )
                )
                if self._queue_bytes + burst + reserve > MAX_QUEUE_BYTES:
                    self._clear_content()
                    self._active = False
                    raise ContractError("speech event consumer backlog exceeded")
            self._emit(
                "attention",
                utterance_id=key,
                speech_end_ms=turn.end_ms,
                decision=evidence,
                **({"request_id": request_id} if attended else {}),
            )
            if superseded_by is not None and would_attend:
                # The owner superseded this turn before its decision arrived: keep the
                # evidence trail, name the request it would have carried, deliver nothing.
                self._emit(
                    "override",
                    superseded_request_id=request_id,
                    by_utterance_id=superseded_by,
                    role="owner",
                )
            if overriding:
                # The owner's own attended turn or a stop phrase supersedes every earlier
                # open non-owner request before any new request of the owner's is
                # delivered, and every earlier non-owner turn still awaiting its decision.
                for superseded in to_supersede:
                    self._emit(
                        "override",
                        superseded_request_id=superseded,
                        by_utterance_id=key,
                        role="owner",
                    )
                    del self._open[superseded]
                for other, (earlier, _context, _size, other_role) in self._pending.items():
                    if other_role != "owner" and earlier.end_ms <= turn.end_ms:
                        self._superseded.setdefault(other, key)
            if attended:
                formed = self._form_request(turn, role, context)
                self._emit(
                    "request",
                    request_id=request_id,
                    turn=_with_role(turn, role),
                    decision=evidence,
                    context=context,
                    decision_at_ms=self._now,
                    **formed,
                )
                if role is not None and role != "owner":
                    if len(self._open) >= self.max_open:
                        # Fail closed before an owner's override burst could overflow the
                        # queue; nothing is silently dropped.
                        self._clear_content()
                        self._active = False
                        raise ContractError("open request budget exceeded")
                    self._open[request_id] = turn.end_ms

    def _form_request(self, turn, role, context):
        """The optional ``formed_request`` field, present only when a former is configured.

        A former that raises or returns an invalid value fails closed: the session's
        content is released and the error ends the session, never a silently dropped field.
        """
        if self._former is None:
            return {}
        state = {
            "current_turn": _with_role(turn, role),
            "context_turns": copy.deepcopy(context["turns"]),
            "addressing": None if self._addressing is None else self._addressing.to_dict(),
            "speakers": "anonymous" if self._priority is None else "enrolled",
        }
        formed = None
        try:
            formed = formed_request_text(self._former.form(state))
        except Exception:
            pass
        if formed is None:
            self._clear_content()
            self._active = False
            raise ContractError("request forming failed")
        return {"formed_request": formed}

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
            self._open.clear()
            self._active = False
            self._terminal = True
            self._emit(
                "session",
                phase=phase,
                **({"reason": reason} if reason else {}),
                # Only enrolled sessions report role health; anonymous output is unchanged.
                **({"role_status": self.role_status} if self._priority is not None else {}),
            )

    def drain(self):
        with self._lock:
            result = [payload for payload, _size in self._queue]
            self._queue.clear()
            self._queue_bytes = 0
            return copy.deepcopy(result)
