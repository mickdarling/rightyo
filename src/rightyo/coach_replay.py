"""Bounded authored coaching replay; control records are never derived from speech."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, TextIO
from uuid import UUID

from rightyo.contracts import ContractError, Turn, integer
from rightyo.listener_profiles import ListenerProfiles

MAX_REPLAY_BYTES = 1048576
MAX_REPLAY_RECORDS = 512
MAX_REPLAY_MS = 120000


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError("duplicate coaching replay field")
        result[key] = value
    return result


def _uuid(value: Any) -> None:
    try:
        if not isinstance(value, str) or str(UUID(value)) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ContractError("invalid coaching replay UUID") from None


class ReplayClock:
    """Controlled monotonic seconds; advancing never sleeps through fixture time."""

    def __init__(self) -> None:
        self.at_ms = 0

    def __call__(self) -> float:
        return self.at_ms / 1000

    def advance(self, at_ms: int) -> None:
        integer(at_ms, "coaching replay time")
        if not self.at_ms <= at_ms <= MAX_REPLAY_MS:
            raise ContractError("coaching replay time is reversed or exceeds budget")
        self.at_ms = at_ms


def load_document(path: Path) -> dict[str, Any]:
    """Read and preflight all records before starting an authored replay session."""
    from rightyo.coaching import CoachSuggestion, validate_coaching_control

    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_REPLAY_BYTES + 1)
    except OSError:
        raise ContractError("coaching replay input unavailable") from None
    if len(content) > MAX_REPLAY_BYTES:
        raise ContractError("coaching replay byte budget exceeded")
    try:
        raw = json.loads(content, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError):
        raise ContractError("invalid coaching replay document") from None
    if not isinstance(raw, dict) or set(raw) != {
        "version",
        "provenance",
        "profiles",
        "suggestions",
        "records",
    }:
        raise ContractError("invalid coaching replay fields")
    if type(raw["version"]) is not int or raw["version"] != 1:
        raise ContractError("unsupported coaching replay version")
    if raw["provenance"] != "synthetic":
        raise ContractError("coaching replay requires synthetic provenance")
    profiles = ListenerProfiles(raw["profiles"])
    if not isinstance(raw["suggestions"], dict) or len(raw["suggestions"]) > MAX_REPLAY_RECORDS:
        raise ContractError("invalid coaching replay suggestions")
    for utterance_id, suggestion in raw["suggestions"].items():
        _uuid(utterance_id)
        if not isinstance(suggestion, str):
            raise ContractError("invalid authored coaching suggestion")
        CoachSuggestion(suggestion)
    records = raw["records"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAX_REPLAY_RECORDS:
        raise ContractError("invalid coaching replay record count")
    clock = ReplayClock()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"channel", "at_ms", "event"}:
            raise ContractError("invalid coaching replay record")
        clock.advance(record["at_ms"])
        if not isinstance(record["event"], dict):
            raise ContractError("invalid coaching replay event")
        if record["channel"] == "control":
            if record["event"].get("type") not in {"coaching_control", "listener_profile"}:
                raise ContractError("invalid coaching replay control channel")
            if record["event"]["type"] == "listener_profile":
                profiles.apply_control(record["event"])
            else:
                record["event"] = validate_coaching_control(record["event"])
        elif record["channel"] == "transcript":
            event = record["event"]
            if set(event) != {"turn", "generation_id", "pause_ms", "owner_mid_answer", "playback"}:
                raise ContractError("invalid coaching replay transcript fields")
            turn = Turn.from_dict(event["turn"])
            _uuid(turn.session_id)
            _uuid(turn.utterance_id)
            _uuid(event["generation_id"])
            if turn.provenance != "synthetic" or turn.end_ms > record["at_ms"]:
                raise ContractError("coaching replay turn requires causal synthetic provenance")
            integer(event["pause_ms"], "coaching replay pause")
            if type(event["owner_mid_answer"]) is not bool or type(event["playback"]) is not bool:
                raise ContractError("invalid coaching replay transcript flags")
        else:
            raise ContractError("invalid coaching replay channel")
    return raw


def replay_document(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a preflighted authored document with no wall-clock pacing or I/O backend."""
    from rightyo.coaching import CoachingSession, MockCoachProvider

    profiles = ListenerProfiles(raw["profiles"])
    clock = ReplayClock()
    session = CoachingSession(
        profiles=profiles, provider=MockCoachProvider(raw["suggestions"]), clock=clock
    )
    events = session.drain()
    try:
        for record in raw["records"]:
            clock.advance(record["at_ms"])
            session.tick()
            event = record["event"]
            if record["channel"] == "control":
                if event["type"] == "listener_profile":
                    profiles.apply_control(event)
                else:
                    session.apply_control(event)
            else:
                session.observe(
                    Turn.from_dict(event["turn"]),
                    generation_id=event["generation_id"],
                    pause_ms=event["pause_ms"],
                    owner_mid_answer=event["owner_mid_answer"],
                    playback=event["playback"],
                )
            # Keep logical time fixed until even a hold's fixture work has receipted.
            if not session.wait_idle(timeout=1.0):
                raise ContractError("authored coaching worker did not finish within budget")
            session.tick()
            events.extend(session.drain())
    finally:
        session.close()
    events.extend(session.drain())
    return events


def run(args: Any, *, output: TextIO | None = None) -> int:
    """Explicit CLI entry: authored mock-only JSONL, emitted after successful replay."""
    events = replay_document(load_document(Path(args.input)))
    stream = sys.stdout if output is None else output
    try:
        for event in events:
            stream.write(
                json.dumps(event, ensure_ascii=True, allow_nan=False, sort_keys=True) + "\n"
            )
        stream.flush()
    except (OSError, BrokenPipeError):
        raise ContractError("coaching replay output unavailable") from None
    return 0
