"""Explicit foreground speech tools: versioned local JSONL, no application actions."""

from __future__ import annotations

import signal
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from rightyo.cli import addressing_from_args, forming_from_args, load_turns, priority_from_args
from rightyo.contracts import ContractError
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
            max_requests=args.max_requests,
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
    runner = ReplayRunner(provider, addressing=addressing)
    events = SpeechEvents()
    now = 0
    events.start(
        turns[0].session_id,
        now_ms=now,
        addressing=addressing,
        priority=priority,
        former=former,
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
    output = sys.stdout if output is None else output
    addressing = addressing_from_args(args)
    config = PrototypeConfig.load(Path(args.config))
    # --allow-hosted is the one consent for sending text (Jev) or audio (configured
    # hosted speech backends); it is meaningless, and refused, without either.
    if config.hosted_speech and not args.allow_hosted:
        raise PrototypeError("Hosted speech backends require --allow-hosted")
    if args.allow_hosted and not (args.use_jev or config.hosted_speech):
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
    events = SpeechEvents()
    consent = {"allow_hosted_speech": True} if config.hosted_speech else {}
    if args.mode == "stdin":
        consent["audio_input"] = sys.stdin.buffer if audio_input is None else audio_input
        consent["audio_provenance"] = provenance
        consent["report"] = _stderr
    controller = controller_factory(config, event_publisher=events, **consent)
    # Signal handlers are installed only by this explicit foreground operation.
    previous = None
    if threading.current_thread() is threading.main_thread():
        previous = signal.getsignal(signal.SIGTERM)

        def interrupted(_signum, _frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, interrupted)
    try:
        options = {"mode": args.mode, "use_jev": args.use_jev}
        if args.session_id is not None:
            options["session_id"] = args.session_id
        controller.start(options)
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
