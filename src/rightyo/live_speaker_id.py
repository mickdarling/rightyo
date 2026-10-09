"""Shadow live speaker identification (#137 step 4a): scores only, no behaviour change.

When the `speaker_id` section sets `"live": true`, a live session runs this alongside the
processor. For each finalized turn with a session speaker label from the diarization
timeline, it takes that turn's audio span from a bounded in-memory ring of the session's
PCM, embeds it the way `rightyo enroll verify` does (silence trimmed, 3 s windows,
renormalised mean), and accumulates a duration-weighted mean per session label. The
accumulated vector is scored against the enrolled voiceprints made with the same model,
and each label gets a shadow binding state: `bound` to an enrolled identifier, `tentative`
or `unknown`.

Nothing here changes turns, roles, requests, attention or events: the result is a
content-free stderr note per scored turn (labels, enrolled identifiers, durations and
scores only; never audio, embeddings, transcript text or paths), which is calibration data
for the accumulated-score thresholds (#141). A later step applies roles (#113).

Embedding runs on a dedicated worker thread behind a small bounded queue, never on the
audio thread: when the worker is busy, a turn is dropped and counted. A worker that fails
is reported once and switches shadow identification off for the rest of the session; the
live session itself never stalls or fails because of it. Audio is never written: the ring
holds at most `RING_MS` of PCM in memory and is cleared when the session ends.
"""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from array import array
from dataclasses import dataclass, field
from typing import Any, Callable

from rightyo.enroll import MAX_VERIFY_SECONDS, Store, cosine, normalize, speech, voiceprint
from rightyo.speaker_id import SAMPLE_RATE, SpeakerEmbedder, SpeakerIdConfig

BYTES_PER_MS = 32  # mono PCM16 at 16 kHz
# A finalized turn spans at most two utterance windows (2 x 15 s) and may be emitted a few
# seconds after its end (merge gap, reply wait), so a minute of audio covers any turn.
RING_MS = 60000
QUEUE_SIZE = 4
CLOSE_DRAIN_SECONDS = 3.0


class PcmRing:
    """The last `max_ms` of session PCM, addressed by stream milliseconds; memory only."""

    def __init__(self, max_ms: int = RING_MS):
        self.max_bytes = max_ms * BYTES_PER_MS
        self._buffer = bytearray()
        self._start = 0  # stream byte offset of `_buffer[0]`
        self._lock = threading.Lock()

    def append(self, pcm: bytes) -> None:
        with self._lock:
            self._buffer.extend(pcm)
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
        with self._lock:
            self._buffer.clear()


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


def _samples(pcm: bytes) -> array:
    values = array("h")
    values.frombytes(pcm[: len(pcm) // 2 * 2])
    if sys.byteorder != "little":
        values.byteswap()
    return values


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
    ):
        self.settings = settings
        self._report_to = report
        self._embedder_factory = embedder_factory
        self._entries = entries
        self._queue: queue.Queue[tuple[str, int, bytes]] = queue.Queue(maxsize=queue_size)
        self._ring = PcmRing(ring_ms)
        self._labels: dict[str, LabelState] = {}
        self._stop = threading.Event()  # discard pending work and exit
        self._closing = threading.Event()  # finish pending work, then exit
        self._off = threading.Event()  # failed or nothing to compare against
        self._embedder = None
        self._embedder_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="rightyo-speaker-id", daemon=True)
        self.offered = self.scored = self.short = self.overlap = 0
        self.dropped = self.clipped = 0

    @property
    def active(self) -> bool:
        return not (self._off.is_set() or self._stop.is_set() or self._closing.is_set())

    def start(self) -> None:
        self._thread.start()

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

    def turn(self, turn) -> None:
        """Queue one finalized turn for scoring; never blocks, never raises."""
        try:
            if (
                not self.active
                or turn.speaker_id is None
                or turn.speaker_provenance != "diarization-timeline"
            ):
                return
            self.offered += 1
            if turn.overlap:
                # Two voices at once would blur the label's evidence.
                self.overlap += 1
                return
            pcm, clipped = self._ring.span(turn.start_ms, turn.end_ms)
            self.clipped += clipped
            if not pcm:
                return
            try:
                self._queue.put_nowait((turn.speaker_id, turn.end_ms - turn.start_ms, pcm))
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
            prints = {
                entry["id"]: normalize(entry["embedding"])
                for entry in self._entries(self.settings)
                if all(entry["model"].get(key) == model.get(key) for key in ("id", "revision"))
                and entry["model"].get("sha256") == model.get("sha256")
            }
            if not prints:
                self._off.set()
                self._report("speaker_id live: no voiceprint enrolled with this model; shadow off")
                return
            self._report(f"speaker_id live shadow on: enrolled={len(prints)}")
            while not self._stop.is_set():
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
            if embedder is not None:
                with contextlib.suppress(Exception):
                    embedder.close()

    def _score(self, embedder, prints: dict[str, list[float]], label: str, turn_ms: int, pcm):
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
        state = self._labels.setdefault(label, LabelState())
        mean = state.add(vector, seconds)
        bind(state, {key: cosine(mean, value) for key, value in comparable.items()}, self.settings)
        self.scored += 1
        # Report after the bookkeeping, best-effort: a failing channel changes nothing.
        self._report(
            f"speaker_id label={label} turn_ms={turn_ms} speech_ms={round(seconds * 1000)}"
            f" turn_score={turn_scores[state.identity]:.3f} acc_score={state.score:.3f}"
            f" acc_s={state.seconds:.1f} state={state.state} id={state.identity}"
        )

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
        if self.offered:
            for label in sorted(self._labels):
                state = self._labels[label]
                self._report(
                    f"speaker_id final label={label} turns={state.turns}"
                    f" acc_score={state.score:.3f} acc_s={state.seconds:.1f}"
                    f" state={state.state} id={state.identity}"
                )
            self._report(
                f"speaker_id summary offered={self.offered} scored={self.scored}"
                f" short={self.short} overlap={self.overlap} dropped={self.dropped}"
                f" clipped={self.clipped}"
            )
