"""Merge a speaker's turn split by a short pause before it is emitted or decided (#73).

The live window finalizes an utterance after a fixed silence, so one spoken sentence can
arrive as several finalized fragments ("<name>, what time is" + "it?"). This bounded
stage holds the newest fragment for a short gap of stream time. A continuation by the
same speaker within the gap is joined into it; anything else releases it unchanged.

A fragment for which `breaks_turn` is true (the configured owner stop phrases, which
are matched against a whole turn) is never joined or held: the held fragment is released
first and the stop fragment is emitted on its own, so joining never hides a stop.

Only fragments whose speaker label is known, equal, and not overlapping are joined, with
the same speaker provenance. An unattributed or overlapping fragment is never joined, so
no speaker attribution is invented. Fragments are plain dictionaries; the caller turns a
released fragment into a `Turn` and assigns its utterance id at that moment, so ids stay
unique and in emission order, and a joined turn reports the first fragment's start and
the last fragment's end.

This is a fixed-gap first step towards the wait/yield end-of-turn decision of #84, not
an adaptive model.

The same hold observes the post-turn gap of #96. With a nonzero `reply_wait_ms`, every
released held fragment carries a `post_turn_gap`: the quiet time after its last word and
who spoke next, within the time it was held. A request-shaped fragment is held for at
least `reply_wait_ms`, even when it could not be joined (no speaker label, overlap) or
merging is off; joining still only happens within `gap_ms`. With the defaults the merge
hold (2,000 ms) is already longer than the reply wait (1,200 ms), so observing the gap
adds no latency. A fragment released early (end of input, a stalled source, a
suppressed utterance) has no observed gap; one released because a stop phrase followed
it is observed, with the stop phrase as the following speech. Cancellation discards the
held fragment rather than releasing it. Speech the caller detected after the held
fragment (`heard`) counts even when recognition produced no text from it: the gap then
ends there and the next speaker is `unattributed`, never a quiet gap.
"""

from __future__ import annotations

from typing import Any, Callable

from .addressedness import observe_gap, reply_wait, request_shaped
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

    def __init__(
        self,
        gap_ms: int,
        max_span_ms: int,
        emit: Callable[[dict[str, Any]], None],
        breaks_turn: Callable[[str], bool] | None = None,
        reply_wait_ms: int = 0,
        shaped: Callable[[str], bool] = request_shaped,
    ):
        self.gap_ms = merge_gap(gap_ms)
        self.reply_wait_ms = reply_wait(reply_wait_ms)
        if not callable(shaped):
            raise ValueError("invalid request shape predicate")
        self.shaped = shaped
        if breaks_turn is not None and not callable(breaks_turn):
            raise ValueError("invalid turn break predicate")
        self.breaks_turn = breaks_turn
        if type(max_span_ms) is not int or max_span_ms < 1:
            raise ValueError("invalid turn merge span")
        self.max_span_ms = max_span_ms
        self.emit = emit
        self._held: dict[str, Any] | None = None
        # How long past its last word the held fragment is kept: the observation window.
        self._hold_ms = 0
        # The earliest speech the caller detected after the held fragment, in stream ms.
        self._heard_ms: int | None = None

    @property
    def holding(self) -> bool:
        return self._held is not None

    def _joinable(self, fragment: dict[str, Any]) -> bool:
        return fragment["speaker"] is not None and not fragment["overlap"]

    def _hold_for(self, fragment: dict[str, Any]) -> int:
        hold = self.gap_ms if self._joinable(fragment) else 0
        if self.reply_wait_ms and self.shaped(fragment["text"]):
            hold = max(hold, self.reply_wait_ms)
        return hold

    def _continues(self, fragment: dict[str, Any]) -> bool:
        held = self._held
        return (
            held is not None
            and self.gap_ms > 0
            and self._joinable(held)
            and self._joinable(fragment)
            and (fragment["speaker"], fragment["speaker_provenance"])
            == (held["speaker"], held["speaker_provenance"])
            and fragment["start_ms"] - held["end_ms"] <= self.gap_ms
            and max(held["end_ms"], fragment["end_ms"]) - held["start_ms"] <= self.max_span_ms
            and len(held["text"]) + 1 + len(fragment["text"]) <= MAX_TEXT_CHARS
        )

    def offer(self, fragment: dict[str, Any]) -> None:
        if self.breaks_turn is not None and self.breaks_turn(fragment["text"]):
            # A stop phrase must stay a whole turn of its own to be recognized downstream.
            self._release(fragment)
            self.emit(dict(fragment))
            return
        if self._continues(fragment):
            held = self._held
            held["text"] = held["text"] + " " + fragment["text"]
            held["end_ms"] = max(held["end_ms"], fragment["end_ms"])
            self._hold_ms = self._hold_for(held)
            # Detected speech up to now belonged to the continuation just joined.
            self._heard_ms = None
            if self.breaks_turn is not None and self.breaks_turn(held["text"]):
                # The recognizer split the stop phrase itself ("never" + "mind"): emit
                # the joined phrase now so no later fragment can join and hide it.
                self.flush()
            return
        self._release(fragment)
        hold = self._hold_for(fragment)
        if hold:
            self._held, self._hold_ms, self._heard_ms = dict(fragment), hold, None
        else:
            self.emit(dict(fragment))

    def heard(self, start_ms: int) -> None:
        """Record speech detected at `start_ms` (voice activity), with or without text.

        A detected utterance that recognition turns into no fragment would otherwise leave
        the window looking quiet; this keeps it as evidence that something followed.
        """
        if self._held is None or type(start_ms) is not int:
            return
        if self._heard_ms is None or start_ms < self._heard_ms:
            self._heard_ms = start_ms

    def due(self, now_ms: int, open_since_ms: int | None) -> None:
        """Release the held fragment once its gap has passed with no continuation open.

        `open_since_ms` is the earliest audio of an utterance still being collected, or
        None. Its words cannot start before that, so if it opened within the gap the
        fragment stays held until that utterance is finalized and offered.
        """
        held = self._held
        if held is None:
            return
        deadline = held["end_ms"] + self._hold_ms
        if now_ms >= deadline and (open_since_ms is None or open_since_ms > deadline):
            # No fragment arrived through the whole window: quiet, unless speech was
            # detected that recognition produced no text from.
            self._release(None, quiet=True)

    def flush(self) -> None:
        """Release the held fragment now; its gap is unobserved."""
        self._release(None)

    def _release(self, following: dict[str, Any] | None, *, quiet: bool = False) -> None:
        """Emit the held fragment, with the gap observed up to `following` when enabled."""
        held, self._held = self._held, None
        heard, self._heard_ms = self._heard_ms, None
        if held is None:
            return
        if following is None and quiet and heard is not None:
            # Detected speech without a fragment: something followed, speaker unknown.
            following = {
                "speaker": None,
                "overlap": False,
                "speaker_provenance": held["speaker_provenance"],
                "start_ms": heard,
                "end_ms": heard,
            }
        if self.reply_wait_ms and (following is not None or quiet):
            held["post_turn_gap"] = observe_gap(held, following, self._hold_ms)
        self.emit(held)

    def discard(self) -> None:
        self._held = None
        self._heard_ms = None
