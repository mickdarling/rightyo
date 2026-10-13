"""Opt-in continuous coaching over the existing PCM listener, never action dispatch.

The inherited control descriptor is a trusted capability supplied by the embedding
host. PCM and finalized transcripts cannot configure profiles or enable this mode.
One child owns one immutable capture generation. Host revocation retires the child.
"""

from __future__ import annotations

import json
import math
import os
import select
import signal
import threading
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

from rightyo.coaching import CoachingSession, MockCoachProvider, canonical_uuid, wire_json
from rightyo.contracts import ContractError
from rightyo.listener_profiles import ListenerProfiles
from rightyo.prototype import PrototypeError
from rightyo.tool_events import encode_json

MAX_CONTROL_BYTES = 8192
MAX_CONTROL_RECORDS = 4096
MAX_SUGGESTIONS_BYTES = 65536
MAX_OUTPUT_BYTES = 262144
STARTUP_SECONDS = 10
OUTPUT_STALL_SECONDS = 0.25
POLL_SECONDS = 0.05
EOF_RECEIPT_SECONDS = 2


def _note(message):
    """Best-effort content-free diagnostic on the already nonblocking stderr fd."""
    try:
        os.write(2, ("rightyo: " + message[:384] + "\n").encode("ascii", "replace"))
    except OSError:
        pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate coaching control field")
        result[key] = value
    return result


def _nonfinite(_value):
    raise ContractError("nonfinite coaching control value")


def decode_object(line: bytes):
    try:
        value = json.loads(line, object_pairs_hook=_unique, parse_constant=_nonfinite)
    except (ValueError, UnicodeError, RecursionError):
        raise ContractError("invalid coaching control JSON") from None
    if not isinstance(value, dict):
        raise ContractError("coaching control requires an object")
    return value


def mock_provider(path: Path | None):
    if path is None:
        return MockCoachProvider()
    try:
        with Path(path).open("rb") as source:
            content = source.read(MAX_SUGGESTIONS_BYTES + 1)
    except OSError:
        raise ContractError("authored coaching suggestions unavailable") from None
    if len(content) > MAX_SUGGESTIONS_BYTES:
        raise ContractError("authored coaching suggestions exceed byte budget")
    return MockCoachProvider(decode_object(content))


class ControlReader:
    """Only the foreground loop reads; no blocked reader outlives the child session."""

    def __init__(self, fd):
        self.fd = fd
        self.pending = bytearray()
        self.count = 0

    def read(self):
        try:
            chunk = os.read(self.fd, MAX_CONTROL_BYTES)
        except BlockingIOError:
            return []
        if not chunk:
            raise ContractError("coaching control channel ended")
        self.pending.extend(chunk)
        records = []
        while b"\n" in self.pending:
            line, _, rest = self.pending.partition(b"\n")
            self.pending = bytearray(rest)
            if not line or len(line) > MAX_CONTROL_BYTES:
                raise ContractError("invalid coaching control line size")
            self.count += 1
            if self.count > MAX_CONTROL_RECORDS:
                raise ContractError("coaching control record budget exceeded")
            records.append(decode_object(line))
        if len(self.pending) > MAX_CONTROL_BYTES:
            raise ContractError("coaching control line exceeds byte budget")
        return records


class OutputQueue:
    """Bounded nonblocking output; tick/revocation never waits for a stdout consumer.

    Advice may leave in partial writes, so the host independently checks its original
    qualification at receipt/playback. Unsent advice is discarded on newer evidence.
    """

    def __init__(self, fd, *, clock=time.monotonic):
        self.fd = fd
        self.clock = clock
        self.items = deque()
        self.bytes = 0
        self.blocked_at = None

    def add(self, events, *, observed_at=None):
        for event in events:
            observed = self.clock() if observed_at is None else observed_at
            data = (encode_json(event) + "\n").encode("ascii")
            if self.bytes + len(data) > MAX_OUTPUT_BYTES:
                raise ContractError("coaching output backlog exceeded")
            deadline = (
                observed + event["valid_for_ms"] / 1000
                if event.get("type") == "coaching_advice"
                else None
            )
            self.items.append([dict(event), data, 0, deadline])
            self.bytes += len(data)

    def add_coaching(self, session):
        # Drain can wait on a profile guard. Timestamp before that wait so transport
        # bookkeeping never extends the producer's original qualification deadline.
        observed = self.clock()
        self.add(session.drain(), observed_at=observed)

    def discard_advice(self):
        kept = deque()
        for item in self.items:
            if item[0].get("type") == "coaching_advice":
                if item[2]:
                    # Part of a revoked message already entered the pipe; retire
                    # rather than splice a different JSON line onto its tail.
                    raise ContractError("coaching output revoked during handoff")
                self.bytes -= len(item[1])
            else:
                kept.append(item)
        self.items = kept

    def retire(self):
        partial = any(item[2] for item in self.items)
        self.items.clear()
        self.bytes = 0
        self.blocked_at = None
        if partial:
            raise ContractError("coaching output retired during handoff")

    def write(self):
        if not self.items:
            self.blocked_at = None
            return
        event, data, offset, deadline = item = self.items[0]
        now = self.clock()
        if deadline is not None:
            remaining = math.floor((deadline - now) * 1000)
            if remaining < 1:
                if offset:
                    raise ContractError("coaching output expired during handoff")
                self.bytes -= len(data)
                self.items.popleft()
                return
            if not offset:
                event["valid_for_ms"] = remaining
                updated = wire_json(event) + b"\n"
                self.bytes += len(updated) - len(data)
                item[1] = data = updated
        try:
            written = os.write(self.fd, data[offset:])
        except BlockingIOError:
            written = 0
        if written:
            item[2] += written
            self.blocked_at = None
            if item[2] == len(data):
                self.bytes -= len(data)
                self.items.popleft()
        else:
            if self.blocked_at is None:
                self.blocked_at = now
            if now - self.blocked_at >= OUTPUT_STALL_SECONDS:
                raise ContractError("coaching output consumer stalled")


def run(args, *, config, events, consent, controller_factory, output, use_jev=False):
    """Own one foreground listener; no controller/audio construction before qualification."""
    from rightyo.coaching_activity import CoachingActivityAdapter

    fd = args.control_fd
    if args.mode != "stdin" or type(fd) is not int or fd < 3:
        raise PrototypeError("--coaching requires --mode stdin and --control-fd above 2")
    if use_jev:
        raise PrototypeError(
            "coaching uses a separate provider; ordinary Jev decisions must be off"
        )
    # Ordinary request/dismissal/conversation settings do not control coaching.
    # Speech content remains evidence; only the trusted fd can retire this mode.
    config = replace(
        config, dismissal=None, conversation=None, request_former=None, acknowledgement=None
    )
    try:
        session_id = canonical_uuid(args.session_id)
        provider = mock_provider(getattr(args, "coaching_mock_suggestions", None))
        output_fd = output.fileno()
        if output_fd == fd:
            raise ValueError
        was_control_blocking = os.get_blocking(fd)
        was_output_blocking = os.get_blocking(output_fd)
        was_stderr_blocking = os.get_blocking(2)
    except (ContractError, ValueError, OSError, AttributeError):
        raise PrototypeError(
            "coaching requires valid session, provider and pipe descriptors"
        ) from None
    profiles = ListenerProfiles()
    session = CoachingSession(profiles=profiles, provider=provider)
    reader = ControlReader(fd)
    sink = OutputQueue(output_fd)
    controller = adapter = None
    enabled = False
    generation = None
    eof_deadline = None
    startup_deadline = time.monotonic() + STARTUP_SECONDS
    result = 0
    retirement_events = []
    terminal_event = None
    failure_note = None
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
    try:
        os.set_blocking(fd, False)
        os.set_blocking(output_fd, False)
        os.set_blocking(2, False)
        sink.add_coaching(session)
        retire = False
        while not retire:
            complete_receipt = False
            ready, _, _ = select.select([fd], [], [], POLL_SECONDS)
            if ready:
                for raw in reader.read():
                    kind = raw.get("type")
                    if kind == "listener_profile":
                        profiles.apply_control(raw)
                        if enabled:
                            session.tick()
                            retire = True
                    elif kind == "coaching_control":
                        # Existing validator and session own authority/deadline checks.
                        if raw.get("session_id") != session_id:
                            raise ContractError("coaching child session mismatch")
                        if raw.get("action") == "qualify":
                            candidate = canonical_uuid(raw.get("generation_id"))
                            if generation is not None and generation != candidate:
                                raise ContractError("coaching child generation is immutable")
                        session.apply_control(raw)
                        if raw["action"] == "enable":
                            enabled = True
                        elif raw["action"] == "qualify":
                            generation = raw["generation_id"]
                        elif raw["action"] == "disable" or (
                            raw["action"] == "invalidate" and raw["generation_id"] == generation
                        ):
                            retire = True
                    elif kind == "coaching_activity" and adapter is not None:
                        adapter.accept_activity(raw)
                    else:
                        raise ContractError("unsupported coaching control message")
                    if retire:
                        break
            session.tick()
            if retire:
                sink.discard_advice()
                retirement_events = session.drain()
                break
            permitted = session.capture_permitted
            if controller is not None and not permitted:
                sink.discard_advice()
                retirement_events = session.drain()
                result = 2
                break
            if controller is None:
                if permitted:
                    adapter = CoachingActivityAdapter(
                        session, session_id=session_id, generation_id=generation
                    )
                    controller = controller_factory(
                        config,
                        event_publisher=events,
                        **{**consent, "retain_transcripts": False, "report": _note},
                    )
                    controller.start({"mode": "stdin", "use_jev": False, "session_id": session_id})
                elif time.monotonic() >= startup_deadline:
                    raise ContractError("coaching qualification startup timed out")
            if controller is not None:
                state = controller.snapshot()
                batch = controller.drain_events()
                deliver = []
                for index, event in enumerate(batch):
                    if event.get("type") == "transcript":
                        sink.discard_advice()
                        batch[index] = adapter.accept_transcript(event)
                    elif event.get("type") not in {"session", "owner_voice"}:
                        # Attention/requests/action-shaped output cannot enter coaching.
                        raise ContractError("unexpected coaching speech output")
                    if event.get("type") == "session" and event.get("phase") in {
                        "stopped",
                        "cancelled",
                        "error",
                    }:
                        # Preserve a terminal produced by an audio error as well as
                        # the one close() produces. Hand it off only after cancellation.
                        terminal_event = event
                    else:
                        deliver.append(batch[index])
                sink.add(deliver)
                adapter.tick()
                if state["phase"] == "error" or getattr(controller, "input_overrun", False):
                    raise ContractError("coaching speech input failed")
                if state["phase"] in {"complete", "idle"}:
                    if eof_deadline is None:
                        eof_deadline = time.monotonic() + EOF_RECEIPT_SECONDS
                    if not adapter.pending and session.wait_idle(timeout=0):
                        complete_receipt = True
                    elif time.monotonic() >= eof_deadline:
                        raise ContractError("coaching input receipt timed out")
            sink.add_coaching(session)
            sink.write()
            if complete_receipt and not sink.items:
                break
    except KeyboardInterrupt:
        result = 0
    except (ContractError, PrototypeError, OSError, ValueError):
        failure_note = "coaching listener stopped: invalid or unavailable input/output"
        result = 2
    finally:
        # Revoke before potentially slower speech cleanup. Neither PCM EOF on a
        # cancelled source nor output pressure may flush or resurrect open speech.
        if adapter is not None:
            adapter.close()
        session.close()
        if controller is not None:
            # The coordinator does not own the audio worker's cancellation token.
            # Signal/stop that worker before any final output grace period; blocked
            # stdout must never keep PCM processing alive after host revocation.
            try:
                controller.close()
            except Exception:
                failure_note = "coaching speech cleanup failed"
                result = 2
            try:
                for event in controller.drain_events():
                    if event.get("type") == "session" and event.get("phase") in {
                        "stopped",
                        "cancelled",
                        "error",
                    }:
                        terminal_event = event
            except Exception:
                failure_note = "coaching speech terminal unavailable"
                result = 2
        if failure_note is not None:
            _note(failure_note)
        try:
            sink.retire()
            if terminal_event is not None:
                sink.add([terminal_event])
            sink.add(retirement_events)
            sink.add_coaching(session)
            deadline = time.monotonic() + OUTPUT_STALL_SECONDS
            while sink.items and time.monotonic() < deadline:
                sink.write()
                if sink.items:
                    select.select([], [output_fd], [], POLL_SECONDS)
        except (ContractError, OSError, ValueError):
            result = 2
        with_context = (
            (fd, was_control_blocking),
            (output_fd, was_output_blocking),
            (2, was_stderr_blocking),
        )
        for descriptor, blocking in with_context:
            try:
                os.set_blocking(descriptor, blocking)
            except OSError:
                pass
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    return result
