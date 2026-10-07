"""Merge a speaker's turn split by a short pause before it is emitted or decided (#73).

The live window finalizes an utterance after a fixed silence, so one spoken sentence can
arrive as several finalized fragments ("<name>, what time is" + "it?"). This bounded
stage holds the newest fragment for a short gap of stream time. A continuation by the
same speaker within the gap is joined into it; anything else releases it unchanged.

Only fragments whose speaker label is known, equal, and not overlapping are joined, with
the same speaker provenance. An unattributed or overlapping fragment is never joined, so
no speaker attribution is invented. Fragments are plain dictionaries; the caller turns a
released fragment into a `Turn` and assigns its utterance id at that moment, so ids stay
unique and in emission order, and a joined turn reports the first fragment's start and
the last fragment's end.

This is a fixed-gap first step towards the wait/yield end-of-turn decision of #84, not
an adaptive model.
"""

from __future__ import annotations

from typing import Any, Callable

from .contracts import MAX_TEXT_CHARS

# A continuation that starts within this many milliseconds of the held fragment's last
# word is joined into it. The live window only finalizes after at least `hangover_ms`
# (1,440 ms minimum) of silence, so word-to-word gaps between separately finalized
# utterances are rarely shorter than about 1.4 s; 2,000 ms joins pauses up to that,
# and costs about `2000 - hangover_ms` (560 ms at the default) of extra wait when no
# continuation follows, since most of the gap has already passed at finalization.
DEFAULT_TURN_MERGE_GAP_MS = 2000
MAX_TURN_MERGE_GAP_MS = 5000


def merge_gap(value: Any) -> int:
    """A validated gap in milliseconds: 0 turns merging off."""
    if type(value) is not int or not 0 <= value <= MAX_TURN_MERGE_GAP_MS:
        raise ValueError("invalid turn merge gap")
    return value


class TurnMerger:
    """Single-owner, synchronous; the caller supplies the stream clock.

    `offer` takes each finalized fragment in time order. `due` is called as stream time
    advances and releases the held fragment once no continuation can still arrive.
    `flush` releases it at once (end of input, a stalled source); `discard` drops it
    (cancellation). A joined turn never spans more than `max_span_ms`.
    """

    def __init__(self, gap_ms: int, max_span_ms: int, emit: Callable[[dict[str, Any]], None]):
        self.gap_ms = merge_gap(gap_ms)
        if type(max_span_ms) is not int or max_span_ms < 1:
            raise ValueError("invalid turn merge span")
        self.max_span_ms = max_span_ms
        self.emit = emit
        self._held: dict[str, Any] | None = None

    @property
    def holding(self) -> bool:
        return self._held is not None

    def _joinable(self, fragment: dict[str, Any]) -> bool:
        return fragment["speaker"] is not None and not fragment["overlap"]

    def _continues(self, fragment: dict[str, Any]) -> bool:
        held = self._held
        return (
            held is not None
            and self._joinable(fragment)
            and (fragment["speaker"], fragment["speaker_provenance"])
            == (held["speaker"], held["speaker_provenance"])
            and fragment["start_ms"] - held["end_ms"] <= self.gap_ms
            and max(held["end_ms"], fragment["end_ms"]) - held["start_ms"] <= self.max_span_ms
            and len(held["text"]) + 1 + len(fragment["text"]) <= MAX_TEXT_CHARS
        )

    def offer(self, fragment: dict[str, Any]) -> None:
        if self._continues(fragment):
            held = self._held
            held["text"] = held["text"] + " " + fragment["text"]
            held["end_ms"] = max(held["end_ms"], fragment["end_ms"])
            return
        self.flush()
        if self.gap_ms and self._joinable(fragment):
            self._held = dict(fragment)
        else:
            self.emit(dict(fragment))

    def due(self, now_ms: int, open_since_ms: int | None) -> None:
        """Release the held fragment once its gap has passed with no continuation open.

        `open_since_ms` is the earliest audio of an utterance still being collected, or
        None. Its words cannot start before that, so if it opened within the gap the
        fragment stays held until that utterance is finalized and offered.
        """
        held = self._held
        if held is None:
            return
        deadline = held["end_ms"] + self.gap_ms
        if now_ms >= deadline and (open_since_ms is None or open_since_ms > deadline):
            self.flush()

    def flush(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            self.emit(held)

    def discard(self) -> None:
        self._held = None
