"""Explicit, offline whole-file Nemotron/Whisper benchmark; no downloads or hosted calls.

Run with the checkout's src directory on PYTHONPATH. Each repetition launches a
new runtime process; later invocations may benefit from the operating-system
file cache but do not keep model state resident between invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable

from rightyo.audio import (
    MAX_BYTES,
    AudioError,
    diarize_nemotron_cpp,
    join_transcript_timeline,
    parse_rttm,
    transcribe_whisper_cpp,
)

MAX_AUDIO_SECONDS = 3600
MAX_AUDIO_BYTES = 128 * 1024 * 1024


def artifact(path: Path) -> dict[str, Any]:
    """Record a supplied artifact without exposing its identifying filename."""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError:
        raise AudioError("Cannot read supplied artifact") from None
    return {"sha256": digest.hexdigest(), "size_bytes": size}


def wav_duration(path: Path) -> float:
    """Bounded, explicit PCM16 mono 16 kHz WAV only; reject truncated payloads."""
    try:
        if path.stat().st_size > MAX_AUDIO_BYTES:
            raise AudioError("Supplied audio exceeds benchmark size limit")
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            if (
                audio.getframerate() != 16000
                or audio.getnchannels() != 1
                or audio.getsampwidth() != 2
                or audio.getcomptype() != "NONE"
                or not 0 < frames <= MAX_AUDIO_SECONDS * 16000
            ):
                raise AudioError("Benchmark requires bounded 16 kHz mono PCM16 WAV")
            remaining = frames
            while remaining:
                requested = min(remaining, 16000)
                if len(audio.readframes(requested)) != requested * 2:
                    raise AudioError("Supplied WAV is truncated")
                remaining -= requested
            return frames / 16000
    except (OSError, EOFError, wave.Error):
        raise AudioError("Cannot read supplied WAV") from None


def hardware() -> dict[str, Any]:
    result = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        for name, key in (("machdep.cpu.brand_string", "cpu"), ("hw.memsize", "memory_bytes")):
            try:
                value = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", name],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                ).stdout.strip()
                result[key] = int(value) if key == "memory_bytes" else value
            except (OSError, ValueError, subprocess.SubprocessError):
                result[key] = None
    return result


def counts(turns: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "turns": len(turns),
        "assigned": sum(turn["speaker_id"] is not None for turn in turns),
        "unknown": sum(turn["speaker_id"] is None for turn in turns),
        "overlap": sum(turn["overlap"] for turn in turns),
        "unique_assigned_speakers": len(
            {turn["speaker_id"] for turn in turns if turn["speaker_id"] is not None}
        ),
    }


def timed_runs(
    operation: Callable[[], list[dict[str, Any]]], *, reruns: int, duration: float
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    elapsed, results = [], []
    for _ in range(reruns + 1):
        start = time.perf_counter()
        result = operation()
        seconds = time.perf_counter() - start
        if not math.isfinite(seconds) or seconds <= 0:
            raise AudioError("Invalid benchmark clock measurement")
        elapsed.append(seconds)
        results.append(result)
    warmed = elapsed[1:]
    summary = {
        "initial_invocation_seconds": elapsed[0],
        "initial_fullfile_real_time_factor": elapsed[0] / duration,
        "cache_warmed_separate_process_seconds": warmed,
        "cache_warmed_fullfile_real_time_factors": [seconds / duration for seconds in warmed],
        "cache_warmed_seconds": (
            {"median": statistics.median(warmed), "min": min(warmed), "max": max(warmed)}
            if warmed
            else None
        ),
    }
    return summary, results


def benchmark(
    *,
    audio: Path,
    whisper_executable: Path,
    whisper_model: Path,
    nemotron_executable: Path,
    nemotron_model: Path,
    reruns: int = 3,
    timeout_seconds: float = 120,
    backend: str = "metal",
    speakerkit_rttm: Path | None = None,
) -> dict[str, Any]:
    if type(reruns) is not int or not 0 <= reruns <= 9:
        raise AudioError("Reruns must be between zero and nine")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= 3600
    ):
        raise AudioError("Invalid benchmark timeout")
    if backend not in {"cpu", "metal"}:
        raise AudioError("Unsupported benchmark backend")
    if not all(
        path.is_file()
        for path in (audio, whisper_executable, whisper_model, nemotron_executable, nemotron_model)
    ):
        raise AudioError("Explicit existing audio, runtimes and models are required")
    duration = wav_duration(audio)
    baseline = None
    if speakerkit_rttm is not None:
        try:
            with speakerkit_rttm.open("rb") as source:
                payload = source.read(MAX_BYTES + 1)
            if len(payload) > MAX_BYTES:
                raise AudioError("Supplied RTTM exceeds size limit")
            baseline = parse_rttm(payload.decode("utf-8"), expected_file_id=audio.stem)
        except (OSError, UnicodeError):
            raise AudioError("Cannot read supplied RTTM") from None
    # Hash before inference so a complete result records the exact supplied artifacts.
    provenance = {
        "audio": artifact(audio),
        "whisper_executable": artifact(whisper_executable),
        "whisper_model": artifact(whisper_model),
        "nemotron_executable": artifact(nemotron_executable),
        "nemotron_model": artifact(nemotron_model),
    }
    if speakerkit_rttm is not None:
        provenance["speakerkit_supplied_rttm"] = artifact(speakerkit_rttm)
    diarization_timing, timelines = timed_runs(
        lambda: diarize_nemotron_cpp(
            audio,
            executable=nemotron_executable,
            model_path=nemotron_model,
            timeout_seconds=timeout_seconds,
            backend=backend,
        ),
        reruns=reruns,
        duration=duration,
    )
    asr_timing, transcriptions = timed_runs(
        lambda: transcribe_whisper_cpp(
            audio,
            executable=whisper_executable,
            model_path=whisper_model,
            session_id="benchmark",
            timeout_seconds=timeout_seconds,
        ),
        reruns=reruns,
        duration=duration,
    )

    def joined(timeline):
        return join_transcript_timeline(
            transcriptions[0],
            timeline,
            session_id="benchmark",
            recognizer_id="whisper.cpp-external-cli",
        )

    report = {
        "schema_version": 1,
        "mode": "offline-file-benchmark",
        "completed": True,
        "hardware": hardware(),
        "artifacts": provenance,
        "audio": {"duration_seconds": duration, "sample_rate_hz": 16000, "channels": 1},
        "configuration": {
            "reruns": reruns,
            "total_invocations_per_stage": reruns + 1,
            "timeout_seconds_per_invocation": timeout_seconds,
            "nemotron_backend": backend,
            "nemotron_preset": "v3-streaming",
        },
        "nemotron": {
            "timing": diarization_timing,
            "timeline_counts_by_invocation": [
                {
                    "segments": len(timeline),
                    "unique_known_speakers": len(
                        {item["speaker_id"] for item in timeline if item["speaker_id"] is not None}
                    ),
                }
                for timeline in timelines
            ],
            "strict_join_counts_by_invocation": [counts(joined(t)) for t in timelines],
        },
        "whisper": {
            "timing": asr_timing,
            "turn_counts_by_invocation": [len(turns) for turns in transcriptions],
        },
        "limitations": [
            "One supplied recording; no held-out accuracy, DER or WER measurement.",
            "Serialized whole-file stages are not live end-to-end or streaming latency.",
            "Each invocation starts a new process; model state is not retained between runs.",
            "Initial invocation is not guaranteed cold: existing file caches are not flushed.",
            "Strict joins use the first ASR result; gaps and ambiguous segments stay unknown.",
            "No Jev call, audio capture, model download or hosted request is performed.",
        ],
    }
    if baseline is not None:
        report["speakerkit_supplied_rttm"] = {
            "mode": "import-only-no-runtime-measurement",
            "segments": len(baseline),
            "unique_known_speakers": len(
                {item["speaker_id"] for item in baseline if item["speaker_id"] is not None}
            ),
            "strict_join_counts": counts(joined(baseline)),
        }
    return report


def save_report(path: Path, report: dict[str, Any]) -> None:
    """Create a complete new metrics file atomically; never overwrite any input."""
    try:
        with tempfile.TemporaryDirectory(prefix="rightyo-metrics-", dir=path.parent) as directory:
            temporary = Path(directory) / "report.json"
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            os.link(temporary, path)
    except (OSError, ValueError):
        raise AudioError("Cannot create new benchmark report") from None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "audio",
        "whisper-executable",
        "whisper-model",
        "nemotron-executable",
        "nemotron-model",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--speakerkit-rttm", type=Path)
    parser.add_argument("--backend", choices=("cpu", "metal"), default="metal")
    parser.add_argument("--reruns", type=int, default=3, help="0..9 cache-warmed process reruns")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, help="create a new metrics-only JSON file")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and (args.output.exists() or args.output.is_symlink()):
            raise AudioError("Benchmark output must be a new file")
        report = benchmark(
            audio=args.audio,
            whisper_executable=args.whisper_executable,
            whisper_model=args.whisper_model,
            nemotron_executable=args.nemotron_executable,
            nemotron_model=args.nemotron_model,
            reruns=args.reruns,
            timeout_seconds=args.timeout_seconds,
            backend=args.backend,
            speakerkit_rttm=args.speakerkit_rttm,
        )
        if args.output is None:
            print(json.dumps(report, indent=2, allow_nan=False))
        else:
            save_report(args.output, report)
    except (AudioError, OSError, ValueError):
        print("Benchmark failed; no complete result was produced", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
