"""Versioned bounded speech events; no routing, replies, or application actions."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from threading import RLock

from rightyo.addressedness import mentions_name
from rightyo.contracts import (
    DISMISSING,
    Acknowledgement,
    Addressing,
    ContractError,
    Conversation,
    DecisionEvent,
    Dismissal,
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
# How long a reply reported as playing can keep a conversation engaged when its end is
# never reported (#124).
MAX_REPLY_HOLD_MS = 180000
MAX_QUEUE_BYTES = 4194304
MAX_PENDING_BYTES = 1048576
# The context a role provider sees when a speaker first appears.
ROLE_CONTEXT_TURNS = 8
# What a host should do on a `dismiss` (#98), by kind. A late withdrawal of a request
# whose decision was still pending names only `pending_request`.
DISMISS_SCOPES = {
    "stop": ["playback", "pending_request"],
    "disengage": ["playback", "pending_request", "engagement"],
}
# Delivered requests a later dismissal can still withdraw; the oldest is dropped beyond.
MAX_WITHDRAWABLE = 32


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


def _speech_summary(value):
    """A validated, detached copy of the advertised speech backends, or None."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"transcriber", "diarizer"}:
        raise ContractError("invalid speech backend summary")
    summary = {}
    for stage, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {"kind", "id"}:
            raise ContractError("invalid speech backend summary")
        identifier(entry["kind"], f"{stage} kind")
        identifier(entry["id"], f"{stage} id")
        summary[stage] = {"kind": entry["kind"], "id": entry["id"]}
    return summary


def _audio_input(value):
    """A detached copy of the advertised host-supplied audio input format."""
    expected = {"source": "stdin", "encoding": "s16le", "sample_rate": 16000, "channels": 1}
    if value != expected:
        raise ContractError("invalid audio input summary")
    return dict(expected)


class SpeechEvents:
    """Local v1 producer. Frozen request context never observes later turns.

    Consumers still need their own stale-session and once-only dispatch policy.
    Overflow fails closed rather than silently losing a request. Cancellation
    releases pending plaintext; a normal end preserves already queued delivery.
    """

    def __init__(self, *, retention_ms=300000, max_pending=128, report=None):
        # At least five: one open request plus the owner's transcript, attention and
        # request, with a terminal slot, so the static and dynamic override bounds agree.
        integer(max_pending, "max_pending", 5)
        if max_pending > 128:
            raise ContractError("invalid event queue budget")
        self.retention_ms = retention_ms
        self.max_pending = max_pending
        # Optional content-free diagnostics (labels and numbers only, never text or ids).
        self._report = report
        self._acknowledgement = None
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
        # Live identification roles (#137 step 4b): resolved per turn from the provider's
        # current bindings instead of fixed per speaker, with enrolled precedence.
        self._per_turn = False
        # Turns whose speaker label was partly inferred (edge attribution or tail join):
        # they keep their role for attention and precedence, never for owner authority.
        # Utterance id -> turn end, pruned by retention like the other per-turn state.
        self._inferred = {}
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
        # Natural dismissal (#98), off unless configured at start.
        self._dismissal = None
        self._stop_rules = SpeakerPriority()
        # Delivered requests a dismissal can withdraw: request id -> request turn facts.
        self._delivered = {}
        # Pending turns withdrawn before their decisions arrived -> the dismissal's fields.
        self._withdrawn = {}
        # Turns already dismissed, mapped to whether the turn's own decision may still
        # dismiss it fully: only the stop-phrase fast path limited by incomparable
        # attribution, not by the speaker's role, which would limit the decision too.
        self._dismissed = {}
        # Cool-downs after `disengage`: speaker label -> (from_ms, until_ms); the key None
        # applies to every speaker (a dismissal whose speaker cannot be compared).
        self._cooldowns = {}
        # Conversation mode (#82), off unless configured at start, and the one engaged
        # speaker: {"key", "speaker_id", "from_ms", "until_ms"}, plus "replying_since" while
        # the host reports a reply playing (#124), or None when ambient.
        self._conversation = None
        self._engaged = None
        # One acknowledgement per spoken request (#122), with acknowledgement gating on:
        # the last acknowledged request's un-answered acknowledgement,
        # {"until_ms", "heard_end"}, or None. See `_hold_ack` and `reply`.
        self._pending_ack = None

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
        self._delivered.clear()
        self._withdrawn.clear()

    def start(
        self,
        session_id,
        now_ms=0,
        *,
        attention_enabled=True,
        addressing=None,
        priority=None,
        former=None,
        speech=None,
        audio_input=None,
        dismissal=None,
        conversation=None,
        acknowledgement=None,
    ):
        with self._lock:
            identifier(session_id, "session_id")
            if acknowledgement is not None and not isinstance(acknowledgement, Acknowledgement):
                raise ContractError("invalid acknowledgement")
            if dismissal is not None and not isinstance(dismissal, Dismissal):
                raise ContractError("invalid dismissal")
            if conversation is not None and not isinstance(conversation, Conversation):
                raise ContractError("invalid conversation")
            speech = _speech_summary(speech)
            if audio_input is not None:
                audio_input = _audio_input(audio_input)
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
            self._inferred.clear()
            self._priority = priority
            self._per_turn = (
                priority is not None
                and getattr(priority, "per_turn", False) is True
                and callable(getattr(priority, "role_for", None))
            )
            self._former = former
            self._addressing = addressing
            self._dismissal = dismissal
            # The configured stop phrases, or the defaults when roles are off.
            self._stop_rules = SpeakerPriority() if priority is None else priority.priority
            self._dismissed.clear()
            self._cooldowns.clear()
            self._conversation = conversation
            self._acknowledgement = acknowledgement
            self._pending_ack = None
            self._engaged = None
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
                # And for the selected speech backends: kinds and display-safe ids only.
                **({} if speech is None else {"speech": speech}),
                # And for a host-supplied PCM stream: where the live audio came from.
                **({} if audio_input is None else {"audio_input": audio_input}),
                # And for natural dismissal (#98): a host that accepts this object must
                # accept `dismiss` events, which are never emitted without it.
                **({} if dismissal is None else {"dismissal": dismissal.to_dict()}),
                # And for conversation mode (#82): a host that accepts this object must
                # accept `conversation` events, which are never emitted without it.
                **({} if conversation is None else {"conversation": conversation.to_dict()}),
                # And for acknowledgement gating (#132): a host that accepts this object
                # plays its instant acknowledgement only for a request whose
                # `acknowledge` is true; the field is never emitted without it.
                **(
                    {}
                    if acknowledgement is None
                    else {"acknowledgement": acknowledgement.to_dict()}
                ),
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
            for request_id, facts in list(self._delivered.items()):
                if facts["end_ms"] <= cutoff:
                    del self._delivered[request_id]
            for key in list(self._withdrawn):
                if key not in self._pending:
                    del self._withdrawn[key]
            for key, end_ms in list(self._inferred.items()):
                if end_ms <= cutoff:
                    del self._inferred[key]
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
                        turns = [t for t in payload["context"]["turns"] if t["end_ms"] > cutoff]
                        if len(turns) != len(payload["context"]["turns"]):
                            payload["context"]["turns"] = turns
                            if "formed_request" in payload:
                                # The convenience string must not outlive the retention
                                # window either: re-render it from the pruned context,
                                # failing closed exactly as at emission. An unpruned
                                # request keeps its string byte-identical.
                                payload["formed_request"] = self._render_request(
                                    payload["turn"], turns
                                )
                size = len(encode_json(payload)) + 1
                # A re-rendered string may grow, so the rebuilt queue is held to the
                # same per-event, count and aggregate bounds as emission, failing closed
                # the same way rather than letting the backlog exceed its budget.
                if (
                    size > MAX_EVENT_BYTES
                    or len(kept) >= self.max_pending
                    or self._queue_bytes + size > MAX_QUEUE_BYTES
                ):
                    self._clear_content()
                    self._active = False
                    raise ContractError("speech event consumer backlog exceeded")
                kept.append((payload, size))
                self._queue_bytes += size
            self._queue = kept

    def transcript(self, turn, now_ms, *, expect_decision=True, inferred=False):
        """Emit one final turn; `inferred` marks a speaker label partly inferred (#137).

        An inferred turn (edge-attributed words or a tail join) keeps its role for
        attention and precedence, but never exercises owner or trusted authority:
        no override, no owner stop phrase, and a playback-only dismissal that withdraws
        nothing and leaves the speaker's engagement in place.
        """
        with self._lock:
            if not self._active:
                raise ContractError("speech event session is not active")
            if not isinstance(turn, Turn) or not turn.finalized or turn.session_id != self._session:
                raise ContractError("event producer requires a final turn in its active session")
            if type(expect_decision) is not bool:
                raise ContractError("invalid decision expectation")
            if type(inferred) is not bool:
                raise ContractError("invalid inferred-label flag")
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
            # With live identification the role is resolved per turn instead, from the
            # bindings at this moment, and then repeated unchanged the same way.
            role = (
                None
                if self._priority is None
                else speaker_role(self._priority.role_for(turn))
                if self._per_turn
                else self._assign_role(turn, context["turns"])
            )
            if inferred and role is not None:
                self._inferred[turn.utterance_id] = turn.end_ms
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
            if self._dismissal is not None and self._stop_rules.is_stop_phrase(turn.text):
                # The deterministic fast path (#98): an exact stop phrase dismisses at
                # once, without waiting for (or depending on) the decision model.
                plan = self._plan(
                    turn,
                    role,
                    "stop",
                    "stop-phrase",
                    None,
                    loose=False,
                    authority=self._authority_role(turn.utterance_id, role),
                )
                if plan is not None:
                    self._dismissed[turn.utterance_id] = plan["upgradable"]
                    self._emit("dismiss", **self._apply(turn, plan))
                    # Ending one's own engagement affects no one else, so even a dismissal
                    # limited to playback ends it; an inferred label's dismissal may not
                    # be the engaged speaker's own (#137), so it leaves it.
                    if turn.utterance_id not in self._inferred:
                        self._disengage(turn, "dismissed")

    @staticmethod
    def _comparable(first, second):
        """Whether two turns' speaker labels can be compared (session-stable, labelled)."""
        return (
            first["speaker_id"] is not None
            and second["speaker_id"] is not None
            and not first["overlap"]
            and not second["overlap"]
            and first["speaker_provenance"] == second["speaker_provenance"]
            and (
                first["speaker_provenance"] != "diarization-utterance"
                or first["speaker_id"].split(" ", 1)[0] == second["speaker_id"].split(" ", 1)[0]
            )
        )

    @staticmethod
    def _cooldown_key(facts):
        """The per-speaker cool-down key, or None when the label holds for one utterance.

        The key carries the speaker provenance: two sources may reuse a label, and labels
        from different provenances are never the same speaker (`_comparable`).
        """
        if (
            facts["speaker_id"] is None
            or facts["overlap"]
            or facts["speaker_provenance"] == "diarization-utterance"
        ):
            return None
        return (facts["speaker_provenance"], facts["speaker_id"])

    def _withdrawable(self, dismissing, facts, loose):
        """Whether a dismissal may withdraw a request turn: the speaker's own, recent one.

        The request must have ended at or before the dismissal started (a request that
        overlaps the dismissal is concurrent work, not withdrawn) and at most `window_ms`
        before it, from the same speaker label. When the labels cannot be compared
        (no label, overlap, or utterance-local labels from different utterances), it is
        withdrawn only when `loose`: a model-judged dismissal addressed to the system on an
        anonymous session. A different known speaker's request is never withdrawn.
        """
        dismisser = self._facts(dismissing)
        if facts["end_ms"] > dismissing.start_ms:
            return False
        if dismissing.start_ms - facts["end_ms"] > self._dismissal.window_ms:
            return False
        if self._comparable(dismisser, facts):
            return dismisser["speaker_id"] == facts["speaker_id"]
        return loose

    @staticmethod
    def _facts(turn):
        return {
            "end_ms": turn.end_ms,
            "speaker_id": turn.speaker_id,
            "speaker_provenance": turn.speaker_provenance,
            "overlap": turn.overlap,
        }

    def _authority(self, role):
        """What a speaker's dismissal may do: "full", "playback" only, or None (nothing).

        Anonymous sessions: full. Enrolled: owners and trusted speakers have full effect;
        other speakers only stop playback (and withdraw their own requests); with
        `owner_only`, only owners dismiss at all.
        """
        if self._priority is None:
            return "full"
        rules = self._priority.priority
        if role == "owner":
            return "full"
        if rules.owner_only:
            return None
        return "full" if role == "trusted" else "playback"

    def _authority_role(self, key, role):
        """The role a turn exercises authority with: an inferred label never grants one.

        A turn with inferred-label words (#137) has no override, owner stop phrase,
        dismissal authority or owner supersession, whatever role it carries, and its
        dismissal withdraws nothing (see `_plan`).
        """
        if key in self._inferred and role in {"owner", "trusted"}:
            return "unknown"
        return role

    def _plan(self, turn, role, kind, reason, confidence, *, loose, authority=None):
        """What a dismissal does, without changing state; None when it does nothing.

        `role` is the turn's published role; `authority`, when given, is the role it acts
        with (see `_authority_role`).
        """
        authority = self._authority(role if authority is None else authority)
        if authority is None:
            return None
        dismisser = self._facts(turn)
        # Only a model-judged dismissal by a fully trusted speaker withdraws on
        # incomparable attribution; the fast path never does.
        loose = loose and authority == "full" and self._priority is None
        # A turn with inferred-label words (#137) may be another voice under the
        # speaker's label: it withdraws nothing, delivered or pending.
        inferred = turn.utterance_id in self._inferred
        delivered = [
            request_id
            for request_id, facts in self._delivered.items()
            if not inferred and self._withdrawable(turn, facts, loose)
        ]
        pending = [
            key
            for key, (earlier, _context, _size, _role) in self._pending.items()
            if not inferred
            and key != turn.utterance_id
            and key not in self._withdrawn
            and self._withdrawable(turn, self._facts(earlier), loose)
        ]
        # The stop-phrase fast path from a speaker whose label cannot be compared only
        # stops playback; so does any dismissal from a speaker without full authority.
        limited = authority != "full" or (
            reason == "stop-phrase" and self._cooldown_key(dismisser) is None
        )
        scope = ["playback"] if limited else list(DISMISS_SCOPES[kind])
        fields = {
            "utterance_id": turn.utterance_id,
            "speech_end_ms": turn.end_ms,
            "speaker_id": turn.speaker_id,
            **({} if role is None else {"role": role}),
            "scope": scope,
            "withdrawn_request_ids": delivered,
            "reason": reason,
            "confidence": confidence,
        }
        if kind == "disengage" and not limited and self._dismissal.cooldown_ms:
            fields["cooldown_until_ms"] = turn.end_ms + self._dismissal.cooldown_ms
        # Only a limit from incomparable attribution can be lifted by the turn's decision.
        upgradable = limited and authority == "full"
        return {"fields": fields, "pending": pending, "limited": limited, "upgradable": upgradable}

    def _apply(self, turn, plan):
        """Withdraw what a planned dismissal covers and return its `dismiss` fields."""
        fields = plan["fields"]
        for request_id in fields["withdrawn_request_ids"]:
            del self._delivered[request_id]
            self._open.pop(request_id, None)
        for key in plan["pending"]:
            # Its decision is still to come: if it would form a request, it is withdrawn.
            self._withdrawn[key] = {
                **{k: fields[k] for k in ("utterance_id", "speech_end_ms", "speaker_id")},
                **({"role": fields["role"]} if "role" in fields else {}),
                "scope": ["pending_request"],
                "reason": fields["reason"],
                "confidence": fields["confidence"],
            }
        if "cooldown_until_ms" in fields:
            key = self._cooldown_key(self._facts(turn))
            until = fields["cooldown_until_ms"]
            current = self._cooldowns.get(key)
            if current is None or until > current[1]:
                self._cooldowns[key] = (turn.end_ms, until)
        return fields

    def _cooled(self, turn, decision):
        """Whether a cool-down after `disengage` holds this attended turn back (#98).

        A cool-down started by a speaker with a comparable label applies to that speaker
        only; one whose speaker cannot be compared applies to everyone.
        """
        if not self._cooldowns or self._dismissal is None:
            return False
        keys = [None]
        speaker = self._cooldown_key(self._facts(turn))
        if speaker is not None:
            keys.append(speaker)
        active = any(
            start <= turn.start_ms < until
            for start, until in (self._cooldowns[k] for k in keys if k in self._cooldowns)
        )
        return (
            active
            and decision.confidence < self._dismissal.cooldown_min_confidence
            and not mentions_name(self._addressing, turn.text)
        )

    def _engaged_with(self, turn):
        """Whether `turn` is the engaged speaker's, within the engagement window (#82).

        Only a session-stable, unoverlapped speaker label is ever engaged, so an
        unattributed turn or another speaker's never matches (rightyo#113).
        """
        engaged = self._engaged
        return (
            engaged is not None
            and self._cooldown_key(self._facts(turn)) == engaged["key"]
            # Decisions may arrive out of turn order: speech from before the request
            # that engaged is never a follow-up.
            and engaged["from_ms"] <= turn.start_ms < self._window_end(engaged)
        )

    @staticmethod
    def _window_end(engaged):
        """The engagement window's end: held open while a reply plays, up to a bound."""
        replying = engaged.get("replying_since")
        if replying is None:
            return engaged["until_ms"]
        return max(engaged["until_ms"], replying + MAX_REPLY_HOLD_MS)

    def reply(self, phase, now_ms):
        """The host's report that the assistant's spoken reply `started` or `ended` (#124).

        While engaged, a playing reply holds the window open, and its end restarts the
        window from that moment, so follow-ups are timed from the end of what was spoken,
        not from the request. Ignored when conversation mode is off or nothing is engaged.

        With acknowledgement gating on, a reply that starts after an acknowledgement's own
        clip has ended answers it (#122), so the next request may be acknowledged again.
        Emits nothing.
        """
        with self._lock:
            if phase not in {"started", "ended"}:
                raise ContractError("invalid reply phase")
            integer(now_ms, "now_ms")
            if self._active and self._pending_ack is not None:
                self._answer_ack(phase)
            engaged = self._engaged
            if not self._active or self._conversation is None or engaged is None:
                return
            if now_ms >= self._window_end(engaged):
                # Lapsed already, though no decision has said so yet: a late report never
                # revives a conversation.
                return
            if phase == "started":
                engaged.setdefault("replying_since", now_ms)
            else:
                engaged.pop("replying_since", None)
                engaged["until_ms"] = max(
                    engaged["until_ms"], now_ms + self._conversation.window_ms
                )

    def _answer_ack(self, phase):
        """Track the pending acknowledgement through the host's playback reports (#122).

        The host reports every playback, the acknowledgement clip included, so the first
        `started` after an acknowledgement is usually that clip. A `started` clears the
        pending acknowledgement only after an `ended` has been heard: clip, then answer.
        When the reports run together (a host collapsing a burst to its last phase, or one
        continuous playback), the answer is not told apart and the window decides instead.
        """
        if phase == "ended":
            self._pending_ack["heard_end"] = True
        elif self._pending_ack["heard_end"]:
            self._pending_ack = None

    def _hold_ack(self, turn):
        """Hold an emitted acknowledgement as pending until answered or the window ends."""
        window = self._acknowledgement.dedup_window_ms
        if window > 0:
            # Only reached with nothing pending or a lapsed hold (an acknowledged turn
            # never starts inside a pending window), so this never shortens a hold.
            self._pending_ack = {"until_ms": turn.end_ms + window, "heard_end": False}

    def _lapse(self, now_ms):
        """Return to ambient once stream time passes the engagement window (#82)."""
        engaged = self._engaged
        if engaged is not None and now_ms >= self._window_end(engaged):
            self._engaged = None
            self._emit(
                "conversation",
                state="ambient",
                reason="timeout",
                speaker_id=engaged["speaker_id"],
                at_ms=self._window_end(engaged),
            )

    def _disengage(self, turn, reason):
        """Return the engaged speaker to ambient because of their own `turn`."""
        if self._engaged is None or not self._engaged_with(turn):
            return
        self._engaged = None
        self._emit(
            "conversation",
            state="ambient",
            reason=reason,
            speaker_id=turn.speaker_id,
            utterance_id=turn.utterance_id,
            at_ms=turn.end_ms,
        )

    def _engage(self, turn, request_id, role=None):
        """Engage a request's speaker, or extend their window; one speaker at a time.

        With live identification roles (#113), an engaged owner or trusted speaker keeps
        the engagement for their window: another speaker's request is still delivered, but
        does not move the engagement to them.
        """
        key = self._cooldown_key(self._facts(turn))
        if self._conversation is None or key is None:
            return
        until = turn.end_ms + self._conversation.window_ms
        if self._engaged is not None and turn.end_ms < self._engaged["from_ms"]:
            # Decided out of order: an older request never takes the engagement from a
            # newer one (or shortens it).
            return
        if self._engaged is not None and self._engaged["key"] == key:
            # Each exchange extends the window; the state itself does not change.
            self._engaged["until_ms"] = max(self._engaged["until_ms"], until)
            if self._per_turn:
                self._engaged["role"] = role
            return
        if (
            self._per_turn
            and self._engaged is not None
            and self._engaged.get("role") in {"owner", "trusted"}
            and role not in {"owner", "trusted"}
        ):
            # Enrolled precedence: the window has not lapsed (`_lapse` ran first).
            return
        self._engaged = {
            "key": key,
            "speaker_id": turn.speaker_id,
            "from_ms": turn.end_ms,
            "until_ms": until,
        }
        if self._per_turn:
            self._engaged["role"] = role
        self._emit(
            "conversation",
            state="engaged",
            reason="request",
            speaker_id=turn.speaker_id,
            utterance_id=turn.utterance_id,
            request_id=request_id,
            at_ms=turn.end_ms,
            until_ms=until,
        )

    def _follow_up(self, turn, role, decision, evidence, unavailable):
        """Whether an engaged speaker's undecided turn forms a request as a follow-up."""
        conversation = self._conversation
        if (
            conversation is None
            or unavailable is not None
            or evidence["label"] != "uncertain"
            or not self._engaged_with(turn)
            or decision.recipient not in {"system", "unknown"}
            or decision.probabilities["attend"] < self._follow_up_bar(role)
            # A closing phrase ends the conversation; it is never a request of its own.
            or conversation.is_closing(turn.text)
        ):
            return False
        # Owner-only mode never lets another speaker's turn become a request.
        return not (role is not None and self._priority.priority.owner_only and role != "owner")

    def _follow_up_bar(self, role):
        """The follow-up attend bar: optionally lower for enrolled owners and trusted (#113)."""
        bar = self._conversation.follow_up_min_probability
        enrolled = getattr(self._priority, "follow_up_min_probability", None)
        if self._per_turn and enrolled is not None and role in {"owner", "trusted"}:
            return min(bar, enrolled)
        return bar

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

    def decision(self, event, now_ms, *, unavailable=None):
        """Emit one turn's attention evidence and, when attended, its request.

        ``unavailable`` names why a live turn's hosted decision was transiently
        unavailable (#71). The placeholder decision must be `uncertain`, so it never
        forms a request, and its evidence carries `decision_status: "unavailable"` and
        that `reason` beside the unchanged keys.
        """
        with self._lock:
            if not self._active or not self._attention_enabled:
                return
            if not isinstance(event, DecisionEvent) or event.turn.session_id != self._session:
                raise ContractError("decision event belongs to another session")
            if unavailable is not None:
                identifier(unavailable, "unavailable reason")
                if event.decision.label != "uncertain":
                    raise ContractError("an unavailable decision must be uncertain")
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
            if unavailable is not None:
                # Optional keys, like role_status: strict hosts ignore unknown keys.
                evidence["decision_status"] = "unavailable"
                evidence["reason"] = unavailable
            if event.decision.dismissal is not None:
                # Optional keys too, present only when the dismissal question was asked.
                evidence["dismissal"] = event.decision.dismissal
                evidence["dismissal_confidence"] = event.decision.dismissal_confidence
                if event.decision.dismissal_malformed:
                    # A degraded answer, as distinct from a genuine `uncertain`.
                    evidence["dismissal_status"] = "malformed"
            stop = False
            authority = self._authority_role(key, role)
            if self._per_turn and authority in {"owner", "trusted"}:
                # Re-checked at decision time, without waiting (#137): a label whose
                # binding no longer gives it this role exercises no authority now. The
                # published role is unchanged, so the host's fingerprints still match.
                try:
                    current = self._priority.role_for(turn)
                except Exception:  # noqa: BLE001 - identification never fails the session
                    current = "unknown"
                if current != authority:
                    authority = "unknown"
            if role is not None:
                evidence["role"] = role
                if key == self._degraded_turn:
                    # The stream shows where model lookups degraded, not only the object.
                    evidence["role_status"] = self.role_status
                rules = self._priority.priority
                if rules.owner_only and role != "owner" and evidence["label"] == "attend":
                    # Owner-only mode: other speakers remain context, never a request.
                    evidence["label"] = "ignore"
                stop = authority == "owner" and rules.is_stop_phrase(turn.text)
            request_id = self._session + ":" + key
            superseded_by = self._superseded.pop(key, None)
            withdrawn_by = self._withdrawn.pop(key, None)
            # A model-judged dismissal (#98), unless the stop-phrase fast path already
            # dismissed this turn at its transcript.
            kind = None
            if (
                self._dismissal is not None
                and event.decision.dismissal in DISMISSING
                and self._dismissed.get(key, True)
            ):
                kind = event.decision.dismissal
            dismissed = kind is not None or key in self._dismissed
            would_attend = evidence["label"] == "attend" and evidence["recipient_kind"] == "system"
            if self._conversation is not None:
                # The window is judged in stream time at each decided turn.
                self._lapse(turn.start_ms)
                if not would_attend and self._follow_up(
                    turn, role, event.decision, evidence, unavailable
                ):
                    # An engaged speaker's undecided follow-up (#82): the request evidence
                    # says so, beside Jev's unchanged recipient and confidence.
                    evidence["label"] = "attend"
                    evidence["follow_up"] = True
                    would_attend = True
            if (
                would_attend
                and not dismissed
                and withdrawn_by is None
                and self._cooled(turn, event.decision)
            ):
                # After "go away", an unnamed attend needs more confidence for a while.
                evidence["label"] = "uncertain"
                evidence["cooldown"] = True
                # A follow-up held back is not reported as one.
                evidence.pop("follow_up", None)
                would_attend = False
            attended = (
                would_attend
                and not stop
                and not dismissed
                and superseded_by is None
                and withdrawn_by is None
            )
            overriding = authority == "owner" and (attended or stop or dismissed)
            plan = None
            if kind is not None:
                plan = self._plan(
                    turn,
                    role,
                    kind,
                    "decision",
                    event.decision.dismissal_confidence,
                    loose=event.decision.recipient == "system",
                    authority=authority,
                )
            # A pending turn withdrawn earlier only produces an event when it would have
            # formed a request.
            late = (
                withdrawn_by is not None
                and would_attend
                and not dismissed
                and not stop
                and superseded_by is None
            )
            # Dismiss events this decision emits, in order: a late withdrawal of this
            # turn's own request, then this turn's own dismissal.
            extra = []
            if late:
                extra.append({**withdrawn_by, "withdrawn_request_ids": [request_id]})
            if plan is not None:
                extra.append(plan["fields"])
            # Form before any reservation or emission so the burst reserve below counts
            # the exact request bytes, formed string included, and a failing former
            # fails closed before anything of this decision is queued.
            formed = self._form_request(turn, role, context) if attended else {}
            ack, ack_note = (
                self._acknowledge(turn, event.decision, evidence) if attended else ({}, None)
            )
            to_supersede = []
            if overriding:
                # Earlier is decided by turn time (end_ms at or before the owner's), not by
                # emission order: decisions arrive out of order, and a request spoken after
                # the owner's turn is not what the owner was superseding.
                to_supersede = [r for r, end in self._open.items() if end <= turn.end_ms]
                # The whole burst plus the owner's own attention/request/terminal must fit
                # the undrained queue; otherwise fail closed before emitting any override,
                # never a partial batch.
                # Plus one `conversation` event when conversation mode is on (#82).
                spare = 1 if self._conversation is not None else 0
                if len(to_supersede) + 3 + len(extra) + spare > self.max_pending - len(self._queue):
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
                for offset, fields in enumerate(extra):
                    reserve += self._size(self._payload("dismiss", after + offset, fields))
                after += len(extra)
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
                                **formed,
                                **ack,
                            },
                        )
                    )
                if spare:
                    after += 1
                    reserve += self._size(
                        self._payload(
                            "conversation",
                            after,
                            {
                                "state": "engaged",
                                "reason": "other_human",
                                "speaker_id": turn.speaker_id,
                                "utterance_id": key,
                                "request_id": request_id,
                                "at_ms": 2**53 - 1,
                                "until_ms": 2**53 - 1,
                            },
                        )
                    )
                reserve += 32 + self._size(
                    self._payload(
                        "session",
                        after + 1,
                        {
                            "phase": "cancelled",
                            "reason": "x" * 96,
                            "role_status": "unavailable",
                            "input_gaps": {
                                "gaps": 1,
                                "dropped_bytes": 2**53 - 1,
                                "discarded_tail_bytes": 1,
                            },
                            "skipped_segments": 2**53 - 1,
                            "skipped_utterances": 2**53 - 1,
                        },
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
            if late:
                # A dismissal withdrew this turn while its decision was pending (#98).
                self._emit("dismiss", **extra[0])
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
                    self._delivered.pop(superseded, None)
                for other, (earlier, _context, _size, other_role) in self._pending.items():
                    if (
                        self._authority_role(other, other_role) != "owner"
                        and earlier.end_ms <= turn.end_ms
                    ):
                        self._superseded.setdefault(other, key)
            if plan is not None:
                self._dismissed[key] = plan["upgradable"]
                self._emit("dismiss", **self._apply(turn, plan))
            if self._conversation is not None and self._engaged_with(turn):
                # A close ends engagement even if the closing turn itself was attended,
                # and never re-engages. With the timeout above (emitted before any
                # reservation), this decision emits at most one more `conversation`
                # event: a disengaging turn never also engages.
                if plan is not None:
                    if key not in self._inferred:
                        self._disengage(turn, "dismissed")
                elif self._conversation.is_closing(turn.text):
                    self._disengage(turn, "closed")
                elif evidence["label"] == "ignore" and (
                    event.decision.recipient == "other_human"
                    or event.decision.recipient.startswith("speaker_")
                ):
                    self._disengage(turn, "other_human")
            if attended:
                self._emit(
                    "request",
                    request_id=request_id,
                    turn=_with_role(turn, role),
                    decision=evidence,
                    context=context,
                    decision_at_ms=self._now,
                    **formed,
                    **ack,
                )
                if self._dismissal is not None:
                    if len(self._delivered) >= MAX_WITHDRAWABLE:
                        # Bounded: the oldest delivered request can no longer be withdrawn.
                        del self._delivered[next(iter(self._delivered))]
                    self._delivered[request_id] = self._facts(turn)
                if role is not None and authority != "owner":
                    if len(self._open) >= self.max_open:
                        # Fail closed before an owner's override burst could overflow the
                        # queue; nothing is silently dropped.
                        self._clear_content()
                        self._active = False
                        raise ContractError("open request budget exceeded")
                    self._open[request_id] = turn.end_ms
                if self._conversation is not None and not self._conversation.is_closing(turn.text):
                    self._engage(turn, request_id, role)
                if ack.get("acknowledge") is True:
                    self._hold_ack(turn)
                if ack_note is not None and self._report is not None:
                    # Best-effort and last: a failed diagnostic write (a closed stderr)
                    # must never undo or skip the request's state transitions above.
                    try:
                        self._report(ack_note)
                    except Exception:  # noqa: BLE001
                        pass

    def _acknowledge(self, turn, decision, evidence):
        """The optional ``acknowledge`` field and its diagnostic note (#132).

        Both are absent unless acknowledgement gating is configured.

        A request is acknowledged when its turn uses a configured name or its attend
        confidence reaches the threshold. That confidence is the decision's own for a
        direct `attend`, and Jev's attend probability for a follow-up, whose
        `confidence` belongs to the `uncertain` choice it was promoted from. Either way it
        is not acknowledged while an earlier acknowledgement is still pending (#122).
        """
        gate = self._acknowledgement
        if gate is None:
            return {}, None
        follow_up = evidence.get("follow_up") is True
        score = decision.probabilities["attend"] if follow_up else decision.confidence
        named = mentions_name(self._addressing, turn.text)
        acknowledge = named or score >= gate.min_confidence
        reason = "named" if named else "confident" if acknowledge else "low_confidence"
        pending = self._pending_ack
        if acknowledge and pending is not None and turn.start_ms < pending["until_ms"]:
            # One acknowledgement per spoken request (#122): a fragment or quick follow-up
            # of a request still awaiting its answer is delivered without another one.
            acknowledge = False
            reason = "pending_ack"
        # The diagnostic note is content-free: labels and numbers, never text or ids.
        note = (
            f"ack outcome={'ack' if acknowledge else 'skip'} reason={reason}"
            f" attend_confidence={score:.2f} min={gate.min_confidence:.2f}"
            f" follow_up={'true' if follow_up else 'false'}"
        )
        return {"acknowledge": acknowledge}, note

    def _form_request(self, turn, role, context):
        """The optional ``formed_request`` field, present only when a former is configured.

        A former that raises or returns an invalid value fails closed: the session's
        content is released and the error ends the session, never a silently dropped field.
        """
        if self._former is None:
            return {}
        return {"formed_request": self._render_request(_with_role(turn, role), context["turns"])}

    def _render_request(self, record, context_turns):
        """Run the former on the bounded state for one request record and its context."""
        state = {
            "current_turn": copy.deepcopy(record),
            "context_turns": copy.deepcopy(context_turns),
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
        return formed

    def end(
        self,
        phase="cancelled",
        now_ms=0,
        reason=None,
        input_gaps=None,
        skipped_segments=None,
        skipped_utterances=None,
    ):
        with self._lock:
            if phase not in {"stopped", "cancelled", "error"}:
                raise ContractError("invalid event terminal phase")
            integer(now_ms, "now_ms")
            if skipped_segments is not None:
                integer(skipped_segments, "skipped_segments", 1)
            if skipped_utterances is not None:
                integer(skipped_utterances, "skipped_utterances", 1)
            if reason is not None:
                identifier(reason, "reason")
            if input_gaps is not None:
                if not isinstance(input_gaps, dict) or set(input_gaps) != {
                    "gaps",
                    "dropped_bytes",
                    "discarded_tail_bytes",
                }:
                    raise ContractError("invalid input gap report")
                for name, value in input_gaps.items():
                    integer(value, name)
                input_gaps = dict(input_gaps)
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
            self._delivered.clear()
            self._withdrawn.clear()
            self._active = False
            self._terminal = True
            self._emit(
                "session",
                phase=phase,
                **({"reason": reason} if reason else {}),
                # Only enrolled sessions report role health; anonymous output is unchanged.
                **({"role_status": self.role_status} if self._priority is not None else {}),
                # Stdin input only: audio dropped under back-pressure, never buffered unbounded.
                **({} if input_gaps is None else {"input_gaps": input_gaps}),
                # Live input only, and only when nonzero (#78): whisper.cpp segments skipped
                # and whole utterances suppressed for unusable timestamps while the session
                # kept listening.
                **({} if skipped_segments is None else {"skipped_segments": skipped_segments}),
                **(
                    {} if skipped_utterances is None else {"skipped_utterances": skipped_utterances}
                ),
            )

    def drain(self):
        with self._lock:
            result = [payload for payload, _size in self._queue]
            self._queue.clear()
            self._queue_bytes = 0
            return copy.deepcopy(result)
