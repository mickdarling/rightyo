"""Explicit foreground speech tools: versioned local JSONL, no application actions."""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from rightyo.cli import (
    addressing_from_args,
    dismissal_from_args,
    forming_from_args,
    load_turns,
    priority_from_args,
)
from rightyo.contracts import ContractError, SpeakerPriority
from rightyo.pipeline import ReplayRunner
from rightyo.prototype import (
    PrototypeConfig,
    PrototypeController,
    PrototypeError,
    validate_session_budget,
)
from rightyo.providers import (
    ConfiguredPriorityProvider,
    JevProvider,
    MockProvider,
    ModelPriorityProvider,
    ProviderError,
    replay_request_budget,
    request_former_for,
)
from rightyo.tool_events import SpeechEvents, encode_json


def _emit(events, output: TextIO):
    for event in events:
        # ASCII JSON escapes preserve every Unicode string even when a host's
        # text pipe is configured with a narrower encoding than UTF-8.
        output.write(encode_json(event) + "\n")
        output.flush()


def replay(args, *, output=None) -> int:
    """Replay explicit supplied text; mock is visibly labelled fixture behavior."""
    output = sys.stdout if output is None else output
    addressing = addressing_from_args(args)
    speakers = priority_from_args(args)
    former = request_former_for(forming_from_args(args))
    dismissal = dismissal_from_args(args)
    model_roles = speakers is not None and speakers.source == "model"
    if model_roles and args.provider != "jev":
        raise ProviderError("model-assigned speaker roles require the Jev provider")
    turns = load_turns(args.input)
    preflight = ReplayRunner(MockProvider(), addressing=addressing)
    committed = [turn for turn in turns if preflight.process(turn) is not None]
    required_requests = len(committed)
    if model_roles:
        # One role question per newly observed unconfigured speaker shares the budget.
        required_requests += len(
            {
                turn.speaker_id
                for turn in committed
                if turn.speaker_id is not None and speakers.configured_role(turn.speaker_id) is None
            }
        )
    if args.provider == "jev" and required_requests > args.max_requests:
        raise ProviderError("session exceeds Jev request budget; no requests were sent")
    provider = (
        MockProvider()
        if args.provider == "mock"
        else JevProvider(
            allow_hosted=args.allow_hosted,
            max_requests=replay_request_budget(args.max_requests),
            timeout_seconds=args.timeout,
            min_confidence=args.min_confidence,
        )
    )
    priority = None
    if speakers is not None:
        priority = (
            ModelPriorityProvider(provider, speakers)
            if model_roles
            else ConfiguredPriorityProvider(speakers)
        )
    runner = ReplayRunner(
        provider,
        addressing=addressing,
        dismissal_phrases=(
            None if dismissal is None else (speakers or SpeakerPriority()).stop_phrases
        ),
    )
    events = SpeechEvents()
    now = 0
    events.start(
        turns[0].session_id,
        now_ms=now,
        addressing=addressing,
        priority=priority,
        former=former,
        dismissal=dismissal,
    )
    _emit(events.drain(), output)
    try:
        for turn in committed:
            now = max(now, turn.end_ms)
            events.transcript(turn, now_ms=now)
            decision = runner.process(turn)
            if decision is not None:
                events.decision(decision, now_ms=now)
            _emit(events.drain(), output)
        events.end(phase="stopped", now_ms=now)
        _emit(events.drain(), output)
        return 0
    except (ContractError, ProviderError):
        events.end(phase="error", now_ms=now, reason="replay-failed")
        _emit(events.drain(), output)
        raise
    finally:
        runner.clear()


def _stderr(message: str) -> None:
    print(f"rightyo: {message}", file=sys.stderr, flush=True)


MAX_CONTROL_LINE = 256


def _read_control(fd: int, controller) -> None:
    """Forward the host's reply reports to the session until EOF (#124).

    Bounded lines of `{"reply": "started"}` or `{"reply": "ended"}`; anything else is
    reported on stderr (content-free) and skipped. The reader never ends the session.
    """
    try:
        with os.fdopen(fd, "rb", buffering=0) as source:
            pending, discarding = b"", False
            while True:
                chunk = source.read(MAX_CONTROL_LINE)
                if not chunk:
                    return
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    if discarding:
                        # The rest of an overlong line is never read as a line of its own.
                        discarding = False
                    elif line.strip():
                        _apply_control(line, controller)
                if len(pending) > MAX_CONTROL_LINE:
                    if not discarding:
                        _stderr("control line too long; skipped")
                    pending, discarding = b"", True
    except OSError:
        _stderr("control input unavailable; reply timing off")


def _apply_control(line: bytes, controller) -> None:
    try:
        message = json.loads(line)
        if not isinstance(message, dict) or set(message) != {"reply"}:
            raise ValueError
        controller.reply(message["reply"])
    except (ValueError, TypeError, AttributeError, PrototypeError):
        _stderr("invalid control line; skipped")


def listen(args, *, output=None, controller_factory=PrototypeController, audio_input=None) -> int:
    """The command itself authorizes foreground capture; startup/import never does.

    ``--mode stdin`` reads headerless mono 16 kHz s16le PCM from stdin (or
    ``audio_input``) instead of the Mac microphone; EOF is a clean stop.
    """
    if args.use_jev and not args.allow_hosted:
        raise PrototypeError("Jev requires both --use-jev and --allow-hosted")
    # Stdin bytes carry no origin: the host declares it, and only for stdin.
    provenance = getattr(args, "provenance", None)
    if (args.mode == "stdin") != (provenance is not None):
        raise PrototypeError("--provenance is required with, and only with, --mode stdin")
    control_fd = getattr(args, "control_fd", None)
    if control_fd is not None and (args.mode != "stdin" or control_fd < 3):
        raise PrototypeError("--control-fd needs --mode stdin and a descriptor above 2")
    output = sys.stdout if output is None else output
    addressing = addressing_from_args(args)
    config = PrototypeConfig.load(Path(args.config))
    # The configuration's `decision` section is the file form of --use-jev --allow-hosted
    # (both validated at load). Flags still apply on their own; neither source can turn
    # the other's Jev selection off, and the section never consents to hosted speech.
    use_jev = args.use_jev or config.hosted_decisions
    # --allow-hosted is the one consent for sending text (Jev) or audio (configured
    # hosted speech backends); it is meaningless, and refused, without either.
    if config.hosted_speech and not args.allow_hosted:
        raise PrototypeError("Hosted speech backends require --allow-hosted")
    if args.allow_hosted and not (use_jev or config.hosted_speech):
        raise PrototypeError("--allow-hosted requires --use-jev or a hosted speech backend")
    if addressing is not None:
        # Command-line names take precedence over the configuration file's names.
        config = replace(config, addressing=addressing)
    budget = getattr(args, "session_budget", None)
    if budget is not None:
        # The command line's budget likewise replaces the configuration file's.
        config = replace(config, session_budget_seconds=validate_session_budget(budget))
    forming = forming_from_args(args)
    if forming is not None:
        # The command-line former replaces the configuration file's, like --name.
        config = replace(config, request_former=forming)
    if dismissal_from_args(args) is not None and config.dismissal is None:
        # --dismissal turns natural dismissal on with the defaults; a configuration
        # file's `dismissal` section keeps its own values.
        config = replace(config, dismissal=dismissal_from_args(args))
    # Stdout carries the events; acknowledgement-gating notes (#132) go to stderr.
    events = SpeechEvents(report=_stderr)
    consent = {"allow_hosted_speech": True} if config.hosted_speech else {}
    # Content-free session notes (end-of-turn scores, capture, shadow speaker
    # identification) go to stderr in every mode.
    consent["report"] = _stderr
    if args.mode == "stdin":
        consent["audio_input"] = sys.stdin.buffer if audio_input is None else audio_input
        consent["audio_provenance"] = provenance
    controller = controller_factory(config, event_publisher=events, **consent)
    # Signal handlers are installed only by this explicit foreground operation.
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
    try:
        options = {"mode": args.mode, "use_jev": use_jev}
        if args.session_id is not None:
            options["session_id"] = args.session_id
        controller.start(options)
        if control_fd is not None:
            threading.Thread(
                target=_read_control, args=(control_fd, controller), daemon=True
            ).start()
        while True:
            # Same controller lease, owned by this foreground consumer rather than a page.
            state = controller.snapshot()
            _emit(controller.drain_events(), output)
            if state["phase"] in {"complete", "idle", "error"}:
                overrun = getattr(controller, "input_overrun", False)
                result = 2 if state["phase"] == "error" or overrun else 0
                controller.stop()
                _emit(controller.drain_events(), output)
                return result
            threading.Event().wait(0.05)
    except KeyboardInterrupt:
        controller.stop()
        _emit(controller.drain_events(), output)
        return 2 if getattr(controller, "input_overrun", False) else 0
    finally:
        controller.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
