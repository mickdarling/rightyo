"""Explicit replay experiments; default output contains no transcript text or source IDs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rightyo.contracts import ContractError, Turn
from rightyo.credentials import CredentialError
from rightyo.pipeline import ReplayRunner
from rightyo.providers import JevProvider, MockProvider, ProviderError

MAX_INPUT_BYTES = 1048576
MAX_INPUT_TURNS = 1000


def load_turns(path: Path) -> list[Turn]:
    try:
        with path.open("rb") as stream:
            content = stream.read(MAX_INPUT_BYTES + 1)
    except OSError:
        raise ContractError("input file could not be read") from None
    if len(content) > MAX_INPUT_BYTES:
        raise ContractError("input file exceeds size limit")
    try:
        if path.suffix.lower() == ".jsonl":
            raw = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            document = json.loads(content)
            if not isinstance(document, dict) or document.get("schema_version") != 1:
                raise ContractError("input requires schema_version 1 and a turns array")
            raw = document.get("turns")
    except (ValueError, UnicodeError):
        raise ContractError("input is not valid replay JSON") from None
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_INPUT_TURNS:
        raise ContractError("input requires between 1 and 1000 turns")
    turns = [Turn.from_dict(item) for item in raw]
    if len({turn.session_id for turn in turns}) != 1:
        raise ContractError("evaluate one explicit session at a time")
    return turns


def save_turns(path: Path, turns: list[Turn]) -> None:
    if not 1 <= len(turns) <= MAX_INPUT_TURNS:
        raise ContractError("output requires between 1 and 1000 turns")
    # Private transcript exports must stay outside the development checkout.
    resolved = path.resolve()
    if any((parent / ".git").exists() for parent in resolved.parents):
        raise ContractError("transcript exports must be stored outside the repository")
    payload = (
        json.dumps(
            {"schema_version": 1, "turns": [turn.to_dict() for turn in turns]},
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n"
    )
    if len(payload.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ContractError("output file exceeds replay input size limit")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
    except OSError:
        raise ContractError("output could not be created; choose a new external file") from None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    turns = load_turns(args.input)
    # Validate the entire session before sending any of it to a hosted provider.
    # Count committed turns rather than raw revisions/partials/duplicates.
    preflight = ReplayRunner(
        MockProvider(), no_speakers=args.no_speakers, playback_active=args.playback
    )
    required_requests = sum(preflight.process(turn) is not None for turn in turns)
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
    runner = ReplayRunner(provider, no_speakers=args.no_speakers, playback_active=args.playback)
    events = []
    for turn in turns:
        event = runner.process(turn)
        if event is not None:
            events.append(event)
    speaker_provenance = sorted({turn.speaker_provenance for turn in turns})
    return {
        "schema_version": 1,
        "mode": "transcript-file-replay",
        "hosted_text_processing": args.provider == "jev",
        "speaker_ablation": args.no_speakers,
        "speaker_capability": (
            "supplied-speaker-labels"
            if any(turn.speaker_id is not None for turn in turns)
            else "unknown-speakers"
        ),
        "speaker_provenance": speaker_provenance,
        "results": [event.public_dict(include_text=args.include_text) for event in events],
        "metrics": {
            "input_events": len(turns),
            "committed_decisions": len(events),
            "partial_events": runner.partial_turns,
            "skipped_events": runner.skipped,
            "label_counts": dict(Counter(event.decision.label for event in events)),
            "provider_ms_total": round(sum(event.provider_ms for event in events), 3),
            "pipeline_ms_total": round(sum(event.total_ms for event in events), 3),
            "asr_ms": None,
            "speaker_finalization_ms": None,
            "utterance_completion_ms": None,
        },
        "limitations": [
            "Transcript replay does not measure acoustic streaming or live response latency.",
            "Mock rules do not establish model accuracy."
            if args.provider == "mock"
            else "Jev confidence is distribution-derived; it does not authenticate speakers.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RightyO explicit transcript-first experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    replay = commands.add_parser(
        "evaluate", help="evaluate a bounded JSON/JSONL transcript session"
    )
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--provider", choices=("mock", "jev"), default="mock")
    replay.add_argument(
        "--allow-hosted", action="store_true", help="send transcript/context to Jev"
    )
    replay.add_argument("--max-requests", type=int, default=20)
    replay.add_argument("--timeout", type=float, default=10)
    replay.add_argument("--min-confidence", type=float, default=0.7)
    replay.add_argument(
        "--no-speakers", action="store_true", help="hide labels from decision provider"
    )
    replay.add_argument(
        "--playback", action="store_true", help="explicit assistant playback context"
    )
    replay.add_argument(
        "--include-text", action="store_true", help="include private text and IDs in output"
    )
    local = commands.add_parser(
        "audio-import", help="transcribe a supplied local file with whisper.cpp"
    )
    local.add_argument("--audio", type=Path, required=True)
    local.add_argument("--whisper-executable", type=Path, required=True)
    local.add_argument("--model", type=Path, required=True)
    local.add_argument("--session-id", required=True)
    local.add_argument("--diarization-input", type=Path)
    local.add_argument(
        "--output", type=Path, required=True, help="new private JSON file outside repo"
    )
    local.add_argument("--timeout", type=float, default=120)
    timeline = commands.add_parser(
        "timeline-import", help="join supplied local transcript/speaker data"
    )
    timeline.add_argument("--input", type=Path, required=True)
    timeline.add_argument(
        "--output", type=Path, required=True, help="new private JSON file outside repo"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            result = evaluate(args)
        else:
            from rightyo.audio import AudioError, load_timeline, transcribe_whisper_cpp

            try:
                raw = (
                    load_timeline(args.input)
                    if args.command == "timeline-import"
                    else transcribe_whisper_cpp(
                        args.audio,
                        executable=args.whisper_executable,
                        model_path=args.model,
                        session_id=args.session_id,
                        diarization_path=args.diarization_input,
                        timeout_seconds=args.timeout,
                    )
                )
            except AudioError:
                raise ContractError(
                    "local audio/timeline import failed; check configured inputs"
                ) from None
            turns = [Turn.from_dict(item) for item in raw]
            save_turns(args.output, turns)
            result = {"imported_turns": len(turns), "hosted_text_processing": False}
        print(json.dumps(result, allow_nan=False, sort_keys=True, indent=2))
        return 0
    except (ContractError, ProviderError, CredentialError) as error:
        print(f"rightyo: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
