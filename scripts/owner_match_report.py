"""Content-free owner voice match measurement for #243: misses first, false accepts beside them.

Two explicit steps; neither changes a threshold, an enrolment or the live pipeline:

    turns       read the shadow `speaker_id` notes RightyO writes on stderr (one per scored
                turn, already content-free) and report the owner's turn scores by speech
                length, input route and position among the label's scored turns. A session
                label that ever bound to the owner counts as the owner's own voice for all of
                its turns, so any of its turns not bound to the owner at the time is a miss.
                Only scores against the owner's print enter the quantiles. Labels that never
                bound are reported apart: they mix guests, media and owner turns that never
                got there.
    negatives   score 16 kHz mono WAV clips of other voices (built-in `say` voices from
                `voiceprint_eval.py generate`, or voices kept outside Git) against the
                enrolled owner, cut into speech segments of fixed lengths and scored exactly
                as a live turn is (speech trim, 30 s cap, mean of window embeddings, cosine),
                and report false accepts at the tentative and bind thresholds.

Output is aggregate JSON: counts, rates and score quantiles. It never holds audio, embeddings,
text, session labels, file names or local paths; `negatives` groups clips by the name of
their top-level folder under `--data`, which you choose. Run with the checkout's src directory
on PYTHONPATH:

    PYTHONPATH=src python scripts/owner_match_report.py turns --log <stderr log>
    PYTHONPATH=src python scripts/owner_match_report.py negatives --config <prototype.json> \\
        --data <folder of group folders of WAVs>

The `turns` evidence is only as good as diarization: a label that merged two people counts
both as the owner. Synthetic voices are far easier to reject than real ones, so negatives
bound the false-accept rate from below, not above.
"""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

from rightyo.enroll import MAX_VERIFY_SECONDS, SAMPLE_RATE, cosine, read_wav, speech, voiceprint
from rightyo.live_speaker_id import input_route

BIND, TENTATIVE = 0.60, 0.45
LENGTHS_MS = ((1000, 2000), (2000, 3000), (3000, 6000), (6000, 12000), (12000, None))
POSITIONS = ((0, 1), (1, 3), (3, 10), (10, None))
DURATIONS = (1.0, 2.0, 3.0, 6.0, 12.0)
SEGMENTS_PER_CLIP = 8


@dataclass(frozen=True)
class Turn:
    session: int
    label: str
    speech_ms: int
    score: float
    state: str
    identity: str
    route: str


SUMMARY_COUNTS = (
    "offered",
    "scored",
    "short",
    "overlap",
    "dropped",
    "clipped",
    "inferred",
    "unsegmented",
)


def parse_notes(lines: Iterable[str]) -> tuple[list[Turn], dict[str, int]]:
    """Scored turns from shadow notes, sessions split at each `start`; plus the summed session
    summary counts, which say how many turns were never scored and why."""
    turns: list[Turn] = []
    session = 0
    counts = dict.fromkeys(SUMMARY_COUNTS, 0)
    for line in lines:
        at = line.find("speaker_id ")
        if at < 0:
            continue
        try:
            words = shlex.split(line[at + len("speaker_id ") :])
        except ValueError:
            continue
        if not words:
            continue
        if words[0] == "start":
            session += 1
            continue
        fields = dict(word.split("=", 1) for word in words if "=" in word)
        if words[0] == "summary":
            for name in SUMMARY_COUNTS:
                counts[name] += _int(fields.get(name))
            continue
        if not words[0].startswith("label="):
            continue
        try:
            turns.append(
                Turn(
                    session=session,
                    label=fields["label"],
                    speech_ms=int(fields["speech_ms"]),
                    score=float(fields["turn_score"]),
                    state=fields["state"],
                    identity=fields.get("id", "-"),
                    # Only allowlisted route tokens, so no device name can reach the report.
                    route=_route(fields.get("route")),
                )
            )
        except (KeyError, ValueError):
            continue
    return turns, counts


def turn_report(
    turns: list[Turn], *, owner: str = "owner", bind: float = BIND, tentative: float = TENTATIVE
) -> dict:
    """The owner's turns by length, route and position.

    A miss is a turn not bound to the owner when it arrived. Each note's score is against that
    note's closest identifier, so only scores against the owner's print enter the quantiles;
    the rest are counted. With one enrolled print, every score is against the owner's.
    """
    confirmed = {
        (turn.session, turn.label)
        for turn in turns
        if turn.state == "bound" and turn.identity == owner
    }
    positions: dict[tuple[int, str], int] = {}
    own: list[tuple[Turn, int]] = []
    other: list[Turn] = []
    for turn in turns:
        key = (turn.session, turn.label)
        if key in confirmed:
            own.append((turn, positions.get(key, 0)))
            positions[key] = positions.get(key, 0) + 1
        else:
            other.append(turn)

    def summary(group: list[Turn]) -> dict:
        missed = sum(not (turn.state == "bound" and turn.identity == owner) for turn in group)
        scored = [turn.score for turn in group if turn.identity == owner]
        return _scores(scored, bind, tentative) | {
            "turns": len(group),
            "missed": missed,
            "miss_rate": _rate(missed, len(group)),
            "scored_against_another_print": len(group) - len(scored),
        }

    def by(key: Callable[[Turn, int], str]) -> dict:
        groups: dict[str, list[Turn]] = {}
        for turn, position in own:
            groups.setdefault(key(turn, position), []).append(turn)
        return {name: summary(group) for name, group in sorted(groups.items())}

    return {
        "thresholds": {"bind": bind, "tentative": tentative},
        "sessions": len({turn.session for turn in turns}),
        "owner_labels": len(confirmed),
        "owner_turns": summary([turn for turn, _ in own]),
        "by_speech_length": by(lambda turn, _: _bucket(turn.speech_ms, LENGTHS_MS, "ms")),
        "by_route": by(lambda turn, _: turn.route),
        "by_position_in_label": by(lambda _, position: _bucket(position, POSITIONS, "turn")),
        "unconfirmed_labels": len({(turn.session, turn.label) for turn in other}),
        "unconfirmed_turns": _scores([turn.score for turn in other], bind, tentative),
    }


def segments(samples: array, seconds: float, limit: int = SEGMENTS_PER_CLIP) -> Iterator[array]:
    """Consecutive non-overlapping speech segments of exactly `seconds`, at most `limit`."""
    size = int(seconds * SAMPLE_RATE)
    for index, start in enumerate(range(0, len(samples) - size + 1, size)):
        if index >= limit:
            return
        yield samples[start : start + size]


def negative_report(
    clips: Iterable[tuple[str, array]],
    embed: Callable[[bytes], list[float]],
    owner_print: list[float],
    *,
    durations: tuple[float, ...] = DURATIONS,
    bind: float = BIND,
    tentative: float = TENTATIVE,
) -> dict:
    """False accepts per group and segment length, each segment scored as a live turn."""
    scores: dict[str, dict[str, list[float]]] = {}
    clip_counts: dict[str, int] = {}
    for group, samples in clips:
        clip_counts[group] = clip_counts.get(group, 0) + 1
        trimmed = speech(samples)[: int(MAX_VERIFY_SECONDS * SAMPLE_RATE)]
        per = scores.setdefault(group, {})
        for seconds in durations:
            for segment in segments(trimmed, seconds):
                vector, _ = voiceprint(embed, segment)
                per.setdefault(f"{seconds:g}s", []).append(cosine(vector, owner_print))
        if len(trimmed) >= SAMPLE_RATE:
            vector, _ = voiceprint(embed, trimmed)
            per.setdefault("whole", []).append(cosine(vector, owner_print))
    return {
        "thresholds": {"bind": bind, "tentative": tentative},
        "groups": {
            group: {"clips": clip_counts[group]}
            | {length: _scores(values, bind, tentative) for length, values in per.items()}
            for group, per in sorted(scores.items())
        },
    }


def _scores(values: list[float], bind: float, tentative: float) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    cuts = statistics.quantiles(ordered, n=10, method="inclusive") if len(ordered) > 1 else []
    return {
        "count": len(ordered),
        "at_or_above_bind": _rate(sum(value >= bind for value in ordered), len(ordered)),
        "at_or_above_tentative": _rate(sum(value >= tentative for value in ordered), len(ordered)),
        "min": round(ordered[0], 3),
        "p10": round(cuts[0], 3) if cuts else round(ordered[0], 3),
        "median": round(statistics.median(ordered), 3),
        "p90": round(cuts[-1], 3) if cuts else round(ordered[-1], 3),
        "max": round(ordered[-1], 3),
    }


def _bucket(value: int, edges: tuple[tuple[int, int | None], ...], unit: str) -> str:
    for low, high in edges:
        if value >= low and (high is None or value < high):
            return f"{low}{unit}+" if high is None else f"{low}-{high}{unit}"
    return f"under-{edges[0][0]}{unit}"


def _route(value: str | None) -> str:
    if value is None:
        return "unrecorded"
    return input_route(value) or "other"


def _rate(part: int, whole: int) -> float | None:
    return round(part / whole, 3) if whole else None


def _int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _clips(data: Path) -> Iterator[tuple[str, array]]:
    for path in sorted(data.rglob("*.wav")):
        relative = path.relative_to(data)
        group = relative.parts[0] if len(relative.parts) > 1 else "ungrouped"
        yield group, read_wav(path)


def _negatives(args) -> dict:
    from rightyo.enroll import Store
    from rightyo.prototype import PrototypeConfig
    from rightyo.speaker_id import SpeakerEmbedder

    settings = PrototypeConfig.load(args.config).speaker_id
    if settings is None:
        raise SystemExit("The configuration has no enabled speaker_id section")
    entries, _ = Store(args.store or settings.store).entries()
    embedder = SpeakerEmbedder(settings.python, settings.model, threads=settings.threads)
    try:
        model = dict(embedder.model)
        owner = next(
            (
                entry
                for entry in entries
                if entry["id"] == args.owner and entry["model"]["sha256"] == model["sha256"]
            ),
            None,
        )
        if owner is None:
            raise SystemExit("No voiceprint for that identifier was made with this model")
        durations = tuple(float(value) for value in args.durations.split(","))
        return negative_report(
            _clips(args.data),
            embedder.embed,
            owner["embedding"],
            durations=durations,
            bind=settings.bind_threshold,
            tentative=settings.tentative_threshold,
        )
    finally:
        embedder.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    turns = commands.add_parser("turns", help="owner turn scores from shadow speaker_id notes")
    turns.add_argument("--log", type=Path, required=True)
    turns.add_argument("--owner", default="owner")
    turns.add_argument("--bind", type=float, default=BIND)
    turns.add_argument("--tentative", type=float, default=TENTATIVE)
    negatives = commands.add_parser("negatives", help="false accepts of other voices")
    negatives.add_argument("--config", type=Path, required=True)
    negatives.add_argument("--data", type=Path, required=True)
    negatives.add_argument("--store", type=Path)
    negatives.add_argument("--owner", default="owner")
    negatives.add_argument("--durations", default=",".join(f"{value:g}" for value in DURATIONS))
    parser.add_argument("--output", type=Path, help="write the JSON here instead of stdout")
    args = parser.parse_args(argv)
    if args.command == "turns":
        with args.log.open(encoding="utf-8", errors="replace") as handle:
            parsed, counts = parse_notes(handle)
        report = turn_report(parsed, owner=args.owner, bind=args.bind, tentative=args.tentative)
        report["session_summaries"] = counts
    else:
        report = _negatives(args)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
