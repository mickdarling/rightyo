"""Explicit file ASR and conservative anonymous speaker timeline imports.

No capture, downloads, or hosted calls. JSON offsets were checked against
whisper.cpp v1.9.4 (927cfce34f31707e17f2bff35c349632fb9e2c3a),
examples/cli/cli.cpp. RTTM was checked against Argmax OSS v1.1.0
(1e2a163736dfa5a198e637ae44c114e1c6d5cc2d), SpeakerKit/RTTMLine.swift.
This records the inspected schema, not the user's installed binary version.
Whole-file output is an offline baseline, never evidence of causal latency.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

MAX_BYTES = 10 * 1024 * 1024
MAX_RECORDS = 10_000
MAX_MS = 86_400_000
PROVENANCE = {"synthetic", "recorded-file", "causal-replay"}


class AudioError(ValueError):
    """Safe boundary error; messages contain no transcript, path, or vendor log."""


def _id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or not re.fullmatch(r"[A-Za-z0-9_.: -]{1,128}", value)
    ):
        raise AudioError("Invalid identifier")
    return value


def _interval(value: dict[str, Any]) -> tuple[int, int]:
    start, end = value.get("start_ms"), value.get("end_ms")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= MAX_MS:
        raise AudioError("Invalid timestamp interval")
    return start, end


def _boolean(value: Any) -> bool:
    if type(value) is not bool:
        raise AudioError("Invalid finality")
    return value


def _records(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) > MAX_RECORDS:
        raise AudioError("Invalid or oversized record list")
    if any(not isinstance(item, dict) for item in value):
        raise AudioError("Invalid record")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AudioError("Duplicate JSON field")
        result[key] = value
    return result


def _read(path: str | Path) -> str:
    try:
        with Path(path).open("rb") as source:
            raw = source.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise AudioError("Input exceeds size limit")
        return raw.decode("utf-8")
    except (OSError, UnicodeError):
        raise AudioError("Cannot read input") from None


def _json(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        raise AudioError("Invalid JSON input") from None
    if not isinstance(value, dict):
        raise AudioError("JSON input must be an object")
    return value


def parse_rttm(text: str, *, expected_file_id: str | None = None) -> list[dict[str, Any]]:
    """Read single-file, single-channel SPEAKER RTTM, including Argmax output.

    UNKNOWN/<NA> labels stay unknown. A second recording/channel is rejected,
    rather than accidentally fusing unrelated speakers. Source labels are kept
    internally until the join anonymizes them; they are never emitted as names.
    """
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise AudioError("Input exceeds size limit")
    result: list[dict[str, Any]] = []
    source_key: tuple[str, str] | None = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 10 or fields[0] != "SPEAKER":
            raise AudioError("Invalid SPEAKER RTTM record")
        key = (fields[1], fields[2])
        if expected_file_id is not None and key[0] != expected_file_id:
            raise AudioError("RTTM recording ID does not match the audio file")
        if source_key is not None and key != source_key:
            raise AudioError("RTTM must describe one recording and channel")
        source_key = key
        try:
            start, duration = float(fields[3]), float(fields[4])
            if not math.isfinite(start) or not math.isfinite(duration):
                raise ValueError
            if start < 0 or duration <= 0 or start + duration > MAX_MS / 1000:
                raise ValueError
            segment = {
                "start_ms": round(start * 1000),
                "end_ms": round((start + duration) * 1000),
                "speaker_id": None if fields[7] in {"UNKNOWN", "<NA>"} else _id(fields[7]),
                "finalized": True,
            }
            _interval(segment)
        except (ValueError, OverflowError):
            raise AudioError("Invalid RTTM time or speaker") from None
        result.append(segment)
        if len(result) > MAX_RECORDS:
            raise AudioError("Too many RTTM records")
    return result


def _anonymous(index: int) -> str:
    label = ""
    index += 1
    while index:
        index, digit = divmod(index - 1, 26)
        label = chr(65 + digit) + label
    return f"Speaker {label}"


def _speaker_segments(timeline: Any) -> list[dict[str, Any]]:
    result = []
    for record in _records(timeline):
        start, end = _interval(record)
        speaker = record.get("speaker_id")
        result.append(
            {
                "start_ms": start,
                "end_ms": end,
                "speaker_id": None if speaker is None else _id(speaker),
                "finalized": _boolean(record.get("finalized")),
            }
        )
    return sorted(result, key=lambda item: (item["start_ms"], item["end_ms"]))


def join_transcript_timeline(
    transcript: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    *,
    session_id: str,
    recognizer_id: str,
    provenance: str = "recorded-file",
) -> list[dict[str, Any]]:
    """Assign A/B/A only from full unambiguous timeline coverage.

    Text crossing a speaker change, unknown interval, or gap stays unknown.
    Simultaneous different speakers set overlap=True and stay unknown. We do
    not split whole-segment text into invented word/speaker alignments.
    Tentative diarization prevents a joined final even if ASR text is final.
    """
    session_id, recognizer_id = _id(session_id), _id(recognizer_id)
    if not isinstance(provenance, str) or provenance not in PROVENANCE:
        raise AudioError("Invalid provenance")
    segments = _speaker_segments(timeline)
    aliases: dict[str, str] = {}
    for segment in segments:
        speaker = segment["speaker_id"]
        if speaker is not None and speaker not in aliases:
            aliases[speaker] = _anonymous(len(aliases))
    result = []
    for index, record in enumerate(_records(transcript)):
        start, end = _interval(record)
        text = record.get("text")
        revision = record.get("revision", 0)
        if not isinstance(text, str) or not text.strip() or len(text) > 16_000:
            raise AudioError("Invalid transcript text")
        if type(revision) is not int or not 0 <= revision <= 1_000_000:
            raise AudioError("Invalid revision")
        finalized = _boolean(record.get("finalized"))
        relevant = [s for s in segments if s["start_ms"] < end and s["end_ms"] > start]
        boundaries = sorted(
            {start, end}
            | {max(start, min(end, s[edge])) for s in relevant for edge in ("start_ms", "end_ms")}
        )
        candidate = None
        ambiguous, overlap = False, False
        for left, right in zip(boundaries, boundaries[1:]):
            active = {
                s["speaker_id"] for s in relevant if s["start_ms"] < right and s["end_ms"] > left
            }
            overlap |= len(active) > 1
            if len(active) != 1 or None in active:
                ambiguous = True
                continue
            speaker = next(iter(active))
            if candidate is not None and candidate != speaker:
                ambiguous = True
            candidate = speaker
        assigned = aliases[candidate] if candidate is not None and not ambiguous else None
        result.append(
            {
                "session_id": session_id,
                "utterance_id": _id(record.get("utterance_id", f"u{index + 1}")),
                "revision": revision,
                "start_ms": start,
                "end_ms": end,
                "text": text,
                "speaker_id": assigned,
                "finalized": finalized and all(s["finalized"] for s in relevant),
                "overlap": overlap,
                "recognizer_id": recognizer_id,
                "provenance": provenance,
                "speaker_provenance": "diarization-timeline" if relevant else "unknown",
            }
        )
    return result


def load_timeline(path: str | Path, *, session_id: str | None = None) -> list[dict[str, Any]]:
    """Import RightyO neutral schema v1: transcript + speakers (integer ms).

    This is an explicit interchange format, not a claimed vendor JSON schema.
    `session_id`, `recognizer_id`, and `provenance` are required at the root.
    """
    document = _json(_read(path))
    if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
        raise AudioError("Unsupported timeline schema")
    return join_transcript_timeline(
        document.get("transcript"),
        document.get("speakers"),
        session_id=session_id if session_id is not None else document.get("session_id"),
        recognizer_id=document.get("recognizer_id"),
        provenance=document.get("provenance"),
    )


def parse_whisper_cpp(document: dict[str, Any], *, session_id: str) -> list[dict[str, Any]]:
    """Normalize upstream --output-json millisecond offsets; no diarization."""
    if not isinstance(document, dict):
        raise AudioError("Invalid Whisper JSON output")
    transcript = []
    for index, segment in enumerate(_records(document.get("transcription"))):
        offsets = segment.get("offsets")
        if not isinstance(offsets, dict):
            raise AudioError("Missing Whisper timestamp offsets")
        transcript.append(
            {
                "utterance_id": f"u{index + 1}",
                "revision": 0,
                "start_ms": offsets.get("from"),
                "end_ms": offsets.get("to"),
                "text": segment.get("text"),
                "finalized": True,
            }
        )
    return join_transcript_timeline(
        transcript,
        [],
        session_id=session_id,
        recognizer_id="whisper.cpp:external-cli",
    )


def transcribe_whisper_cpp(
    audio_path: str | Path,
    *,
    executable: str | Path,
    model_path: str | Path,
    session_id: str,
    timeout_seconds: float = 120,
    diarization_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run an explicitly supplied existing CLI/model/file; never download.

    Optional diarization is existing single-file RTTM or neutral speakers JSON
    {"schema_version": 1, "session_id": ..., "speakers": [...]}; it must
    share the audio timebase. RTTM recording ID must match the audio file stem.
    No vendor stdout/stderr or paths are copied into errors/public provenance.
    Temporary transcripts are in a private directory removed on every exit.
    """
    _id(session_id)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 3600
    ):
        raise AudioError("Invalid timeout")
    executable, model_path, audio_path = map(Path, (executable, model_path, audio_path))
    if not all(p.is_file() for p in (executable, model_path, audio_path)):
        raise AudioError("Explicit existing executable, model, and audio file are required")
    timeline = []
    if diarization_path is not None:
        payload = _read(diarization_path)
        if Path(diarization_path).suffix.lower() == ".rttm":
            timeline = parse_rttm(payload, expected_file_id=audio_path.stem)
        else:
            document = _json(payload)
            if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
                raise AudioError("Unsupported timeline schema")
            if document.get("session_id") != session_id:
                raise AudioError("Timeline session does not match the requested session")
            timeline = _speaker_segments(document.get("speakers"))
    with tempfile.TemporaryDirectory(prefix="rightyo-asr-") as directory:
        output = Path(directory) / "transcript"
        try:
            subprocess.run(
                [
                    str(executable.resolve()),
                    "--model",
                    str(model_path.resolve()),
                    "--file",
                    str(audio_path.resolve()),
                    "--output-json",
                    "--output-file",
                    str(output),
                    "--no-prints",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout_seconds,
                check=True,
            )
        except subprocess.TimeoutExpired:
            raise AudioError("Local recognizer timed out") from None
        except (OSError, subprocess.CalledProcessError):
            raise AudioError("Local recognizer failed") from None
        turns = parse_whisper_cpp(_json(_read(output.with_suffix(".json"))), session_id=session_id)
    if diarization_path is not None:
        return join_transcript_timeline(
            turns,
            timeline,
            session_id=session_id,
            recognizer_id="whisper.cpp:external-cli",
        )
    return turns
