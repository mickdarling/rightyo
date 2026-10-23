"""Shadow live speaker identification (#137 step 4a): scores only, no behaviour change.

When the `speaker_id` section sets `"live": true`, a live session runs this alongside the
processor. For each finalized turn with a session speaker label from the diarization
timeline, it takes only the pieces of the turn's span that the timeline gave that label
and no other speaker (#148) from a bounded in-memory ring of the session's PCM, joins
them, embeds them the way `rightyo enroll verify` does (silence trimmed, 3 s windows,
renormalised mean), and accumulates a duration-weighted mean per session label. A turn
without such pieces is counted, never scored. So the audio of inferred-label words (edge
attribution, tail join), of overlap, and of another voice the timeline marks inside the
span is never voiceprint evidence, while a turn with inferred words can still contribute
its own labelled audio. The accumulated vector is scored against the enrolled voiceprints
made with the same model, and each label gets a shadow binding state: `bound` to an
enrolled identifier, `tentative` or `unknown`.

In shadow mode nothing here changes turns, roles, requests, attention or events: the
result is a content-free stderr note per scored turn (labels, enrolled identifiers,
durations and scores only; never audio, embeddings, transcript text or paths), which is
calibration data for the accumulated-score thresholds (#141). With `"roles": true` (#137
step 4b, #113), `EnrolledRoles` reads the current bindings when each turn is emitted and
gives a bound label its enrolled identifier's configured role; it never waits for one.

Embedding runs on a dedicated worker thread behind a small bounded queue, never on the
audio thread: when the worker is busy, a turn is dropped and counted. A worker that fails
is reported once and switches shadow identification off for the rest of the session; the
live session itself never stalls or fails because of it. Audio is never written: the ring
holds at most `RING_MS` of PCM in memory by default (`audio_retention.speaker_ring_seconds`
sets it, #226) and is cleared when the session ends.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from array import array
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from rightyo.contracts import SpeakerPriority
from rightyo.enroll import MAX_VERIFY_SECONDS, Store, cosine, normalize, speech, voiceprint
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.speaker_id import SAMPLE_RATE, SpeakerEmbedder, SpeakerIdConfig

BYTES_PER_MS = 32  # mono PCM16 at 16 kHz
# A finalized turn spans at most two utterance windows (2 x 15 s) and may be emitted a few
# seconds after its end (merge gap, reply wait), so a minute of audio covers any turn. This
# is the default; the configuration may set 10 to 600 s (#226), memory only: PCM16 at
# 16 kHz is 32 KB/s, about 19 MB at 600 s. A shorter ring clips long turns (counted).
RING_MS = 60000
MIN_RING_MS = 10000
MAX_RING_MS = 600000
QUEUE_SIZE = 4
# A privacy-entry voice check (hailing-station #735) waits at most this long for the worker;
# a check not done by then refuses, the safe failure. At most this many wait at once.
ENTRY_CHECK_TIMEOUT_S = 2.0
ENTRY_CHECKS = 4


class _EntryCheck:
    """One privacy-entry voice check: a turn's own-label audio (`labelled`) and each gap of
    the rest of its span (`gaps`, the inferred words), each scored on its own by the worker
    against the owners' voiceprints. `result` is an `EntryScore`, set once before `done`."""

    def __init__(self, labelled: bytes, gaps: tuple[bytes, ...], owners: tuple[str, ...]):
        self.labelled = labelled
        self.gaps = gaps
        self.owners = owners
        self.done = threading.Event()
        self.result = EntryScore(None, None, 0, "unavailable")


@dataclass(frozen=True)
class EntryScore:
    """A privacy-entry voice check's numbers (hailing-station #735): the own-label audio's
    best owner score, the remainder's (the lowest of its voiced gaps' scores, each scored on
    its own; None when no gap held speech, or a voiced gap could not be scored), the
    remainder's speech in ms over all gaps, the outcome: `scored`, or why not (`short`,
    `clipped`, `timeout`, `busy`, `unavailable`), and the longest voiced gap's speech in ms
    (logged only, for tuning)."""

    labelled: float | None
    remainder: float | None
    remainder_ms: int
    outcome: str
    longest_gap_ms: int = 0


CLOSE_DRAIN_SECONDS = 3.0
# The input route kinds a host may name for diagnostics (rightyo#164): exactly
# HailProtocol's diagnostic `route` tokens, never a port or device name.
INPUT_ROUTES = frozenset(
    {
        "none",
        "built_in_mic",
        "built_in_speaker",
        "built_in_receiver",
        "headphones",
        "headset_mic",
        "line_in",
        "line_out",
        "bluetooth_a2dp",
        "bluetooth_hfp",
        "bluetooth_le",
        "airplay",
        "hdmi",
        "car_audio",
        "usb_audio",
        "other",
    }
)


def input_route(value: Any) -> str | None:
    """`value` when it is exactly one of `INPUT_ROUTES`, else None (never raises)."""
    return value if isinstance(value, str) and value in INPUT_ROUTES else None


def speaker_ring_ms(value: Any) -> int:
    """`value` when it is a valid configured ring length in milliseconds (#226).

    A whole number of seconds from `MIN_RING_MS` to `MAX_RING_MS` inclusive; anything else
    raises ValueError, and nothing is clamped.
    """
    if type(value) is not int or not MIN_RING_MS <= value <= MAX_RING_MS or value % 1000:
        raise ValueError("Invalid speaker ring length")
    return value


def _ring_bytes(max_ms: Any) -> int:
    if type(max_ms) is not int or max_ms < 1:
        raise ValueError("Invalid ring length")
    return max_ms * BYTES_PER_MS


class PcmRing:
    """The last `max_ms` of session PCM, addressed by stream milliseconds; memory only.

    `max_bytes` is the one place its size is held; `resize` changes it (#226).
    """

    def __init__(self, max_ms: int = RING_MS):
        self.max_bytes = _ring_bytes(max_ms)
        self._buffer = bytearray()
        self._start = 0  # stream byte offset of `_buffer[0]`
        self._lock = threading.Lock()

    def append(self, pcm: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm)
            self._trim()

    def resize(self, max_ms: int) -> None:
        """Hold the last `max_ms` from now on; a shrink drops the oldest audio at once."""
        max_bytes = _ring_bytes(max_ms)
        with self._lock:
            self.max_bytes = max_bytes
            self._trim()

    def _trim(self) -> None:
        # Called with the lock held.
        excess = len(self._buffer) - self.max_bytes
        if excess > 0:
            excess += excess % 2  # keep sample alignment
            del self._buffer[:excess]
            self._start += excess

    def span(self, start_ms: int, end_ms: int) -> tuple[bytes, bool]:
        """The PCM for `[start_ms, end_ms)` that is still held, and whether it was clipped."""
        with self._lock:
            want_start, want_end = start_ms * BYTES_PER_MS, end_ms * BYTES_PER_MS
            start = max(want_start, self._start)
            end = min(want_end, self._start + len(self._buffer))
            if end <= start:
                return b"", want_end > want_start
            pcm = bytes(self._buffer[start - self._start : end - self._start])
            return pcm, start > want_start or end < want_end

    def clear(self) -> None:
        """Drop all held audio; the ring is then exactly a fresh ring of its size (#226)."""
        with self._lock:
            self._buffer.clear()
            self._start = 0


@dataclass
class LabelState:
    """One session label's accumulated evidence and shadow binding."""

    total: list[float] | None = None  # sum of speech-seconds x unit turn embedding
    seconds: float = 0.0
    turns: int = 0
    state: str = "unknown"
    identity: str | None = None  # bound identifier, else the closest enrolled one
    score: float | None = None  # accumulated score against `identity`
    scores: dict[str, float] = field(default_factory=dict)

    def add(self, vector: list[float], seconds: float) -> list[float]:
        """Accumulate one unit turn embedding weighted by its speech seconds; the new mean."""
        weighted = [value * seconds for value in vector]
        self.total = (
            weighted if self.total is None else [a + b for a, b in zip(self.total, weighted)]
        )
        self.seconds += seconds
        self.turns += 1
        return normalize(self.total)


def bind(state: LabelState, scores: dict[str, float], settings: SpeakerIdConfig) -> None:
    """Update a label's shadow binding from its accumulated scores, with hysteresis.

    A bound label stays bound to its identifier while that score stays at or above the
    tentative threshold. Otherwise the label binds to the best-scoring identifier at or
    above the bind threshold once `bind_min_seconds` of speech has accumulated; is
    `tentative` at or above the tentative threshold (or above the bind threshold with too
    little speech); and is `unknown` below it.
    """
    state.scores = dict(scores)
    if state.state == "bound" and scores.get(state.identity, -1.0) >= settings.tentative_threshold:
        state.score = scores[state.identity]
        return
    best = min(scores, key=lambda identity: (-scores[identity], identity))
    score = scores[best]
    if score >= settings.bind_threshold and state.seconds >= settings.bind_min_seconds:
        state.state = "bound"
    elif score >= settings.tentative_threshold:
        state.state = "tentative"
    else:
        state.state = "unknown"
    state.identity, state.score = best, score


def _quoted(label: str) -> str:
    """A session label in double quotes: labels may hold spaces but never quotes.

    Session labels are contract identifiers (`[A-Za-z0-9_. -]`), so every note stays
    `key=value` parseable with `shlex.split`; all other values never hold spaces.
    """
    return '"' + label.replace('"', "") + '"'


def _number(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def _samples(pcm: bytes) -> array:
    values = array("h")
    values.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _pieces(spans: Any, start_ms: int, end_ms: int) -> list[tuple[int, int]]:
    """Valid `(start_ms, end_ms)` pieces clipped to `[start_ms, end_ms)`, sorted and merged.

    Anything malformed is dropped, so a bad value can only make a turn unsegmented.
    """
    if not isinstance(spans, (list, tuple)):
        return []
    clipped = []
    for piece in spans:
        if (
            isinstance(piece, (list, tuple))
            and len(piece) == 2
            and all(type(value) is int for value in piece)
        ):
            start, end = max(piece[0], start_ms), min(piece[1], end_ms)
            if end > start:
                clipped.append((start, end))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _default_embedder(settings: SpeakerIdConfig):
    return SpeakerEmbedder(settings.python, settings.model, threads=settings.threads)


def _default_entries(settings: SpeakerIdConfig) -> list[dict[str, Any]]:
    return Store(settings.store).entries()[0]


class ShadowSpeakerId:
    """Per-session shadow identification; `audio` and `turn` never block or raise."""

    def __init__(
        self,
        settings: SpeakerIdConfig,
        *,
        report: Callable[[str], None] | None = None,
        embedder_factory: Callable[[SpeakerIdConfig], Any] = _default_embedder,
        entries: Callable[[SpeakerIdConfig], list[dict[str, Any]]] = _default_entries,
        queue_size: int = QUEUE_SIZE,
        ring_ms: int = RING_MS,
        route: str | None = None,
    ):
        self.settings = settings
        # The host's input route kind (rightyo#164), only from the allowlist, appended to
        # the `key=value` notes so a mismatch can be attributed to an input chain.
        route = input_route(route)
        self._route = "" if route is None else f" route={route}"
        self._report_to = report
        self._embedder_factory = embedder_factory
        self._entries = entries
        self._queue: queue.Queue[tuple[str, int, int, bytes]] = queue.Queue(maxsize=queue_size)
        # Privacy-entry voice checks (#735), taken before queued turns: a caller waits on one.
        self._checks: queue.Queue[_EntryCheck] = queue.Queue(maxsize=ENTRY_CHECKS)
        self._ring = PcmRing(ring_ms)
        self._labels: dict[str, LabelState] = {}
        # Enrolled display names by identifier (hailing-station #434), for the voiceprints
        # compared this session; replaced whole, never changed, and never reported.
        self._names: dict[str, str] = {}
        # Owner-voice evidence (rightyo#164), never reported: the identifiers compared this
        # session (replaced whole), every identifier a label has bound to (only added to),
        # and the seconds of speech scored; the last two restart at `forget`, and are written
        # and read together under `_epoch_lock`.
        self._compared: frozenset[str] = frozenset()
        self._bound_ids: set[str] = set()
        # Bumped by `forget` (hailing-station #735 S-1): a queued or in-flight turn scored for
        # an earlier epoch never commits, so nothing from before a reset survives it.
        self._epoch = 0
        self._epoch_lock = threading.Lock()
        self.scored_seconds = 0.0
        self._stop = threading.Event()  # discard pending work and exit
        self._closing = threading.Event()  # finish pending work, then exit
        self._off = threading.Event()  # failed or nothing to compare against
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="rightyo-speaker-id", daemon=True)
        self.offered = self.scored = self.short = self.overlap = 0
        self.dropped = self.clipped = self.inferred = 0
        self.unsegmented = 0  # turns without own-label timeline pieces (#148)
        self.labelled_ms = 0  # own-label timeline audio taken from the ring (#148)

    @property
    def active(self) -> bool:
        return not (self._off.is_set() or self._stop.is_set() or self._closing.is_set())

    def start(self) -> None:
        self._thread.start()

    def binding(self, label: str) -> tuple[str, str | None] | None:
        """A label's current state and identifier, or None before it was first scored.

        Never blocks: committed `LabelState` objects are replaced, never changed, so this
        reads one consistent snapshot without waiting for the worker. Once identification
        has switched itself off mid-session (a failed model, no voiceprint), no label is
        reported bound any more.
        """
        if self._off.is_set() and not (self._closing.is_set() or self._stop.is_set()):
            return None
        state = self._labels.get(label)
        if state is None:
            return None
        if state.state == "unknown" and state.seconds < self.settings.bind_min_seconds:
            # Too little speech to say this is *not* an enrolled voice: one short, noisy turn
            # of the owner scores low. Only a label with as much evidence as binding needs
            # may be judged a non-enrolled participant; until then it stays undetermined.
            return "insufficient", state.identity
        return state.state, state.identity

    def display_name(self, identity: str) -> str | None:
        """The enrolled display name of a compared identifier, or None; never blocks."""
        name = self._names.get(identity)
        return name if isinstance(name, str) else None

    def owner_evidence(self, owners) -> tuple[bool, bool, float]:
        """Owner-voice evidence for `owners` (rightyo#164); never waits on the worker.

        Whether an owner identifier's voiceprint is compared this session, whether any
        label has bound to one, and the seconds of speech scored so far. Scored speech is
        always a single label's own, unoverlapped timeline audio (#148). The seconds and the
        bindings are read together under the epoch lock (held briefly by `_score` and
        `forget`), so a `forget` between them never pairs old seconds with no bindings.
        """
        owners = tuple(owners)
        compared = self._compared
        with self._epoch_lock:
            seconds = self.scored_seconds
            bound = any(owner in self._bound_ids for owner in owners)
        return any(owner in compared for owner in owners), bound, seconds

    def _report(self, message: str) -> None:
        if self._report_to is not None:
            with contextlib.suppress(Exception):
                self._report_to(message)

    # ------------------------------------------------------------------ audio thread

    def audio(self, pcm: bytes) -> None:
        """Keep session PCM for later turn spans; called before the processor sees it."""
        if self.active:
            with contextlib.suppress(Exception):
                self._ring.append(pcm)

    @property
    def epoch(self) -> int:
        """The current speaker epoch, bumped by each `forget` (#735 S-1). A caller that decides
        a turn is public reads it in the same step (#242 round 23) and passes it to `turn`, so
        a `forget` in between discards the turn instead of scoring it for the new labels."""
        with self._epoch_lock:
            return self._epoch

    def turn(self, turn, *, inferred: bool = False, spans=None, epoch: int | None = None) -> None:
        """Queue one finalized turn for scoring; never blocks, never raises.

        `spans` are the `(start_ms, end_ms)` stream-time pieces of the turn's span that the
        diarizer timeline gave the turn's own label and no other speaker (#148). Only that
        audio is embedded. A turn without them (None, or none inside its span) is counted
        as unsegmented and never scored: its whole span may hold another voice.

        A turn with inferred-label words (edge attribution or a tail join, #137) is
        counted as inferred and scored on its labelled pieces only. The inferred words lie
        outside the label's timeline segments, so their audio, which may be another
        speaker's, is never accumulated into the label.

        `epoch` is the speaker epoch read when the turn was accepted (see `epoch`): a turn
        from an earlier epoch is never queued, and one queued just before a `forget` is
        dropped by the worker. None takes the epoch now.
        """
        try:
            if epoch is None:
                epoch = self.epoch
            elif epoch != self.epoch:
                # #242 round 23: accepted before a `forget`; its label may now be someone else.
                return
            if (
                not self.active
                or turn.speaker_id is None
                or turn.speaker_provenance != "diarization-timeline"
            ):
                return
            self.offered += 1
            if inferred:
                self.inferred += 1
            if turn.overlap:
                # Two voices at once would blur the label's evidence.
                self.overlap += 1
                return
            pieces = _pieces(spans, turn.start_ms, turn.end_ms)
            if not pieces:
                self.unsegmented += 1
                return
            labelled_ms = sum(end - start for start, end in pieces)
            self.labelled_ms += labelled_ms
            parts, clipped = [], False
            for start, end in pieces:
                part, cut = self._ring.span(start, end)
                parts.append(part)
                clipped |= cut
            self.clipped += clipped
            pcm = b"".join(parts)
            if not pcm:
                return
            try:
                self._queue.put_nowait(
                    (epoch, turn.speaker_id, turn.end_ms - turn.start_ms, labelled_ms, pcm)
                )
            except queue.Full:
                self.dropped += 1
                if self.dropped == 1:
                    self._report("speaker_id worker busy; dropping turns (counted)")
        except Exception:  # noqa: BLE001 - shadow work never affects the session
            pass

    # ------------------------------------------------------------------ worker thread

    def _run(self) -> None:
        embedder = None
        try:
            embedder = self._embedder_factory(self.settings)
            with self._embedder_lock:
                self._embedder = embedder
            if self._stop.is_set():
                return
            model = dict(embedder.model)
            matched = [
                entry
                for entry in self._entries(self.settings)
                if all(entry["model"].get(key) == model.get(key) for key in ("id", "revision"))
                and entry["model"].get("sha256") == model.get("sha256")
            ]
            prints = {entry["id"]: normalize(entry["embedding"]) for entry in matched}
            self._names = {
                entry["id"]: entry["display_name"]
                for entry in matched
                if isinstance(entry.get("display_name"), str)
            }
            if not prints:
                self._off.set()
                self._report("speaker_id live: no voiceprint enrolled with this model; shadow off")
                return
            self._compared = frozenset(prints)
            self._report(f"speaker_id start enrolled={len(prints)}{self._route}")
            while not self._stop.is_set():
                try:
                    check = self._checks.get_nowait()
                except queue.Empty:
                    pass
                else:
                    self._check(embedder, prints, check)
                    continue
                try:
                    item = self._queue.get(timeout=0.05)
                except queue.Empty:
                    if self._closing.is_set():
                        break
                    continue
                if self._stop.is_set():
                    break
                self._score(embedder, prints, *item)
        except Exception:  # noqa: BLE001 - any failure switches shadow ID off, once
            self._off.set()
            if not self._stop.is_set():
                self._report("speaker_id unavailable; shadow identification off for this session")
        finally:
            self._off.set()
            self._discard_queue()
            self._release_checks()
            if embedder is not None:
                with contextlib.suppress(Exception):
                    embedder.close()

    def forget(self) -> None:
        """Drop every label's evidence and binding (hailing-station #735 S-1): called when
        privacy ends, with the diarizer reset, so no speaker state from before it survives.
        The owner binds again from public speech. Never blocks on the worker.

        The owner-voice evidence restarts with them (round 18): seconds scored before, kept
        without the bindings they produced, would soon report a recognised owner as not."""
        with self._epoch_lock:
            self._epoch += 1
            self._labels = {}
            self._bound_ids = set()
            self.scored_seconds = 0.0
        self._discard_queue()

    def _score(
        self,
        embedder,
        prints: dict[str, list[float]],
        epoch: int,
        label: str,
        turn_ms: int,
        labelled_ms: int,
        pcm: bytes,
    ):
        samples = speech(_samples(pcm))
        seconds = len(samples) / SAMPLE_RATE
        if seconds < self.settings.min_turn_seconds:
            self.short += 1
            return
        samples = samples[: int(MAX_VERIFY_SECONDS * SAMPLE_RATE)]
        seconds = len(samples) / SAMPLE_RATE
        vector, _ = voiceprint(embedder.embed, samples)
        comparable = {key: value for key, value in prints.items() if len(value) == len(vector)}
        if not comparable:
            raise ValueError("no comparable voiceprint")
        turn_scores = {key: cosine(vector, value) for key, value in comparable.items()}
        # Update a copy and commit it only once complete, so a failure part-way never
        # leaves a label half-updated (or created without a score).
        with self._epoch_lock:
            if epoch != self._epoch:
                return  # Queued before a `forget`: its label may now be someone else.
            previous = self._labels.get(label)
            state = LabelState() if previous is None else replace(previous)
            mean = state.add(vector, seconds)
            bind(
                state,
                {key: cosine(mean, value) for key, value in comparable.items()},
                self.settings,
            )
            self._labels[label] = state
            self.scored += 1
            if state.state == "bound" and state.identity is not None:
                self._bound_ids.add(state.identity)
            # In the same epoch as its binding: never counted after a `forget`.
            self.scored_seconds += seconds
        # Report after the bookkeeping, best-effort: a failing channel changes nothing.
        self._report(
            f"speaker_id label={_quoted(label)} turn_ms={turn_ms}"
            f" labelled_ms={labelled_ms} speech_ms={round(seconds * 1000)}"
            f" turn_score={turn_scores[state.identity]:.3f} acc_score={state.score:.3f}"
            f" acc_s={state.seconds:.1f} state={state.state} id={state.identity}{self._route}"
        )

    def entry_score(self, start_ms: int, end_ms: int, spans, owners) -> EntryScore:
        """A privacy-entry voice check (hailing-station #735, rightyo#250): the turn's own
        label timeline pieces (`spans`, #148) and the rest of `[start_ms, end_ms)`, the
        inferred words, each taken from the ring and scored on its own against `owners`'
        voiceprints by the worker, so a guest's words joined under the owner's label are
        judged as themselves, never averaged into his. Any part outside the ring is
        `clipped`. Blocks the caller at most `ENTRY_CHECK_TIMEOUT_S`; never raises. Nothing
        is kept or reported here."""
        try:
            if not self.active:
                return EntryScore(None, None, 0, "unavailable")
            pieces = _pieces(spans, start_ms, end_ms)
            gaps, at = [], start_ms
            for start, end in pieces:
                if start > at:
                    gaps.append((at, start))
                at = end
            if end_ms > at:
                gaps.append((at, end_ms))
            parts: list[list[bytes]] = [[], []]
            for index, ranges in enumerate((pieces, gaps)):
                for start, end in ranges:
                    pcm, clipped = self._ring.span(start, end)
                    if clipped or not pcm:
                        return EntryScore(None, None, 0, "clipped")
                    parts[index].append(pcm)
            check = _EntryCheck(b"".join(parts[0]), tuple(parts[1]), tuple(owners))
            try:
                self._checks.put_nowait(check)
            except queue.Full:
                return EntryScore(None, None, 0, "busy")
            if not check.done.wait(ENTRY_CHECK_TIMEOUT_S):
                return EntryScore(None, None, 0, "timeout")
            return check.result
        except Exception:  # noqa: BLE001 - a check that fails refuses, never raises
            return EntryScore(None, None, 0, "unavailable")

    def _check(self, embedder, prints: dict[str, list[float]], check: _EntryCheck) -> None:
        """Scores one entry check on the worker; any failure is its outcome, never the
        worker's (the turn scoring and the session carry on)."""

        def best(samples) -> float | None:
            samples = samples[: int(MAX_VERIFY_SECONDS * SAMPLE_RATE)]
            vector, _ = voiceprint(embedder.embed, samples)
            scores = [
                cosine(vector, prints[owner])
                for owner in check.owners
                if owner in prints and len(prints[owner]) == len(vector)
            ]
            return max(scores) if scores else None

        result = EntryScore(None, None, 0, "unavailable")
        try:
            own = speech(_samples(check.labelled)) if check.labelled else array("h")
            # Codex P1 on #250 (0f40847): each gap is its own inferred piece, possibly another
            # voice, so each is scored alone, never averaged with the others.
            voiced = [rest for rest in (speech(_samples(gap)) for gap in check.gaps) if len(rest)]
            rest_ms = round(sum(len(rest) for rest in voiced) * 1000 / SAMPLE_RATE)
            longest = round(max((len(rest) for rest in voiced), default=0) * 1000 / SAMPLE_RATE)
            if len(own) / SAMPLE_RATE < self.settings.entry_min_seconds:
                result = EntryScore(None, None, rest_ms, "short", longest)
            else:
                labelled = best(own)
                # Codex P1 on #250 (2b90c8c): any voiced gap is scored, however short, since a
                # short word ("go") can complete an entry; one that cannot be scored refuses
                # (`entry_voice`: no score with speech), never passes unjudged.
                scores = [best(rest) for rest in voiced]
                remainder = (
                    min(scores) if scores and all(score is not None for score in scores) else None
                )
                if labelled is not None:
                    result = EntryScore(labelled, remainder, rest_ms, "scored", longest)
        except Exception:  # noqa: BLE001 - one check's failure refuses that entry only
            result = EntryScore(None, None, 0, "unavailable")
        finally:
            check.result = result
            check.done.set()

    def _release_checks(self) -> None:
        while True:
            try:
                check = self._checks.get_nowait()
            except queue.Empty:
                return
            check.done.set()

    def _discard_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # ------------------------------------------------------------------ teardown

    def close(self, *, drain: bool = False) -> None:
        """End the worker; with `drain`, score already queued turns first (bounded wait)."""
        if drain:
            self._closing.set()
            if self._thread.is_alive():
                self._thread.join(CLOSE_DRAIN_SECONDS)
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(0.5)
        if self._thread.is_alive():
            # A pending embed is unblocked by ending its worker process.
            with self._embedder_lock:
                embedder = self._embedder
            if embedder is not None:
                with contextlib.suppress(Exception):
                    embedder.close()
            self._thread.join(2)
        self._ring.clear()
        if not self.offered:
            return
        # Each line is formatted on its own and best-effort, so one bad value never costs
        # the others (and the summary always follows).
        for label in sorted(dict(self._labels)):
            with contextlib.suppress(Exception):
                state = self._labels[label]
                self._report(
                    f"speaker_id final label={_quoted(label)} turns={state.turns}"
                    f" acc_score={_number(state.score, '.3f')}"
                    f" acc_s={_number(state.seconds, '.1f')}"
                    f" state={state.state} id={state.identity or '-'}{self._route}"
                )
        self._report(
            f"speaker_id summary offered={self.offered} scored={self.scored}"
            f" short={self.short} overlap={self.overlap} dropped={self.dropped}"
            f" clipped={self.clipped} inferred={self.inferred}"
            f" unsegmented={self.unsegmented} labelled_ms={self.labelled_ms}{self._route}"
        )


class EnrolledRoles(ConfiguredPriorityProvider):
    """Per-turn speaker roles from live identification (#137 step 4b); never blocks.

    `role_for` is asked once per turn when the turn is first emitted, and reads whatever
    the identifier has bound by then:

    - no speaker label: `unknown`;
    - a label configured by session label in `speakers`: that role, as before;
    - a label bound to an enrolled identifier: the role `speakers` gives that identifier
      (`owner` or `trusted`), else `participant`;
    - a label with at least `bind_min_seconds` of speech matching no enrolled voice: `participant`;
    - a label not yet scored, with too little speech to judge, `tentative`, or with
      identification off: `unknown`.

    Only session-stable, unoverlapped `diarization-timeline` labels are ever looked up,
    since those are the only ones the identifier scores.

    With `emit_names` (hailing-station #434), `name_for` gives the enrolled display name
    of a turn's bound, non-owner identity; the event producer validates it before use.
    """

    # Read by `SpeechEvents`: roles are resolved per turn, and enrolled precedence applies.
    per_turn = True

    def __init__(
        self, priority: SpeakerPriority, follow_up_min_probability=None, *, emit_names=False
    ) -> None:
        super().__init__(priority)
        self.follow_up_min_probability = follow_up_min_probability
        self.emit_names = emit_names is True
        self.source = None  # the session's `ShadowSpeakerId`, once started
        # The binding the latest `role_for` resolved, as (utterance id, label, binding), so
        # `name_for` names the identity that role came from, never a later rebinding.
        self._resolved = None
        # The latest turn's own-label pieces, as (utterance id, spans), for `entry_voice`.
        self._spans = None

    def role_for(self, turn) -> str:
        self._resolved = None
        role, binding = self._lookup(turn)
        if binding is not None:
            self._resolved = (turn.utterance_id, turn.speaker_id, binding)
        return role

    def role_of(self, turn) -> str:
        """The role `role_for` would give `turn` now, without recording the binding.

        For a look-ahead from another thread, such as the turn merger's owner check
        (#230): it leaves the binding `name_for` reads untouched. Never blocks or raises.
        """
        return self._lookup(turn)[0]

    def resolved_by_voiceprint(self, utterance_id: str) -> bool:
        """Whether the latest `role_for`, for this utterance, came from a live
        identification binding rather than a label configured in `speakers`."""
        resolved = self._resolved
        return resolved is not None and resolved[0] == utterance_id

    def note_spans(self, utterance_id: str, spans) -> None:
        """The label's own timeline pieces of the turn about to be published (#148), for its
        entry voice check; only the latest is kept, as `role_for` keeps its binding."""
        self._spans = (utterance_id, spans)

    def entry_voice(self, turn) -> bool:
        """Whether `turn` is the owner's voice for privacy entry (hailing-station #735,
        rightyo#250): its label's own pieces hold at least `entry_min_seconds` of speech
        scoring at least `entry_threshold` against an owner's voiceprint, and the rest of
        its span (the inferred words), gap by gap, each that holds any speech holding no
        more than `entry_gap_max_seconds` of it and scoring at least
        `entry_remainder_threshold` on its own however short, so a guest's words joined
        under the owner's label are judged as themselves and never pass unjudged. Reports
        one line of numbers per check, pass or refuse; blocks at most
        `ENTRY_CHECK_TIMEOUT_S`; never raises (a failure refuses)."""
        source = self.source
        check = getattr(source, "entry_score", None)
        if not callable(check):
            return False
        try:
            noted = self._spans
            spans = noted[1] if noted is not None and noted[0] == turn.utterance_id else None
            if spans is None:
                return False
            result = check(turn.start_ms, turn.end_ms, spans, self.priority.owners)
            settings = source.settings
            # Codex P1 on #250 (c9b3907): one gap is one embedding and cannot be split
            # without word boundaries, so a voiced gap longer than about a word, which could
            # hold his word and a guest's together, refuses.
            passed = (
                result.outcome == "scored"
                and result.longest_gap_ms <= settings.entry_gap_max_seconds * 1000
                and type(result.labelled) is float
                and result.labelled >= settings.entry_threshold
                and (
                    result.remainder >= settings.entry_remainder_threshold
                    if result.remainder is not None
                    else result.remainder_ms == 0
                )
            )
            report = getattr(source, "_report", None)
            if callable(report):
                report(
                    f"speaker_id entry_check outcome={'pass' if passed else 'refuse'}"
                    f" reason={result.outcome} score={_number(result.labelled, '.3f')}"
                    f" threshold={_number(settings.entry_threshold, '.2f')}"
                    f" remainder_ms={result.remainder_ms}"
                    f" remainder_gap_ms={result.longest_gap_ms}"
                    f" remainder_score={_number(result.remainder, '.3f')}"
                    f" remainder_threshold={_number(settings.entry_remainder_threshold, '.2f')}"
                    f" gap_max_ms={round(settings.entry_gap_max_seconds * 1000)}"
                )
            return passed
        except Exception:  # noqa: BLE001 - a failed check refuses entry, never the turn
            return False

    def enrolled_owner(self, turn) -> bool:
        """Whether `turn` is the owner's by live voiceprint identification now (hailing-station
        #735 S-1): its label is bound to an enrolled owner identity. A label configured as the
        owner by session label is not, so a TV diarized into that slot never qualifies.
        Looked up without side effects; never blocks or raises."""
        try:
            role, binding = self._lookup(turn)
        except Exception:  # noqa: BLE001 - identification never fails the session
            return False
        return role == "owner" and binding is not None and binding[0] == "bound"

    def _lookup(self, turn) -> tuple[str, tuple | None]:
        """A turn's role and the binding it came from (None when no binding was used)."""
        label = turn.speaker_id
        if label is None:
            return "unknown", None
        configured = self.priority.configured_role(label)
        if configured is not None:
            return configured, None
        source = self.source
        if source is None or turn.overlap or turn.speaker_provenance != "diarization-timeline":
            return "unknown", None
        try:
            binding = source.binding(label)
        except Exception:  # noqa: BLE001 - identification never fails the session
            binding = None
        if binding is None:
            return "unknown", None
        state, identity = binding
        if state == "bound" and identity is not None:
            return self.priority.configured_role(identity) or "participant", binding
        return ("participant" if state == "unknown" else "unknown"), binding

    def owner_voice(self, threshold_s: float) -> str | None:
        """This session's owner-voice status (rightyo#164), or None; never blocks or raises.

        `recognized` once a label has bound to an owner identifier; `unrecognized` while
        none has, once an owner's voiceprint is compared and at least `threshold_s` seconds
        of speech have been scored; otherwise None (no owner enrolled for this model, too
        little speech, or identification not started).
        """
        try:
            evidence = getattr(self.source, "owner_evidence", None)
            if not callable(evidence):
                return None
            enrolled, bound, seconds = evidence(self.priority.owners)
            if not enrolled:
                return None
            if bound:
                return "recognized"
            return "unrecognized" if seconds >= threshold_s else None
        except Exception:  # noqa: BLE001 - identification never fails the session
            return None

    def name_for(self, turn, role: str | None) -> str | None:
        """The bound guest's enrolled display name for a turn emitted with `role`, or None.

        Only with `emit_names`, for an unoverlapped `diarization-timeline` turn whose label
        is not configured by session label and is `bound` to an enrolled identifier whose
        role is exactly `role` and is not `owner`. It uses the binding the immediately
        preceding `role_for` call resolved for this same turn (one snapshot), so a label
        rebound in between never names another voice; without that call, no name. Never
        blocks or raises.
        """
        try:
            label = turn.speaker_id
            source = self.source
            resolved = self._resolved
            if (
                not self.emit_names
                or resolved is None
                or resolved[:2] != (turn.utterance_id, label)
                or role in (None, "owner")
                or label is None
                or source is None
                or turn.overlap
                or turn.speaker_provenance != "diarization-timeline"
                or self.priority.configured_role(label) is not None
            ):
                return None
            state, identity = resolved[2]
            if state != "bound" or identity is None:
                return None
            bound_role = self.priority.configured_role(identity) or "participant"
            if bound_role == "owner" or bound_role != role:
                return None
            name = getattr(source, "display_name", None)
            return name(identity) if callable(name) else None
        except Exception:  # noqa: BLE001 - identification never fails the session
            return None
