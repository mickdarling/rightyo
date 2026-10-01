"""Explicit foreground speech tools: versioned local JSONL, no application actions."""

from __future__ import annotations

import json
import signal
import sys
import threading
from pathlib import Path
from typing import TextIO

from rightyo.cli import load_turns
from rightyo.contracts import ContractError
from rightyo.pipeline import ReplayRunner
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import JevProvider, MockProvider, ProviderError
from rightyo.tool_events import SpeechEvents


def _emit(events, output: TextIO):
    for event in events:
        # ASCII JSON escapes preserve every Unicode string even when a host's
        # text pipe is configured with a narrower encoding than UTF-8.
        output.write(json.dumps(event, allow_nan=False, ensure_ascii=True) + "\n")
        output.flush()


def replay(args, *, output=None) -> int:
    """Replay explicit supplied text; mock is visibly labelled fixture behavior."""
    output = sys.stdout if output is None else output
    turns = load_turns(args.input)
    preflight = ReplayRunner(MockProvider())
    committed = [turn for turn in turns if preflight.process(turn) is not None]
    required_requests = len(committed)
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
    runner = ReplayRunner(provider)
    events = SpeechEvents()
    now = 0
    events.start(turns[0].session_id, now_ms=now)
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


def listen(args, *, output=None, controller_factory=PrototypeController) -> int:
    """The command itself authorizes foreground capture; startup/import never does."""
    if args.use_jev and not args.allow_hosted:
        raise PrototypeError("Jev requires both --use-jev and --allow-hosted")
    if args.allow_hosted and not args.use_jev:
        raise PrototypeError("--allow-hosted requires --use-jev")
    output = sys.stdout if output is None else output
    config = PrototypeConfig.load(Path(args.config))
    events = SpeechEvents()
    controller = controller_factory(config, event_publisher=events)
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
                result = 2 if state["phase"] == "error" else 0
                controller.stop()
                _emit(controller.drain_events(), output)
                return result
            threading.Event().wait(0.05)
    except KeyboardInterrupt:
        controller.stop()
        _emit(controller.drain_events(), output)
        return 0
    finally:
        controller.close()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
