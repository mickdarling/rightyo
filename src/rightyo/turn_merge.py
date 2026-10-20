"""Merge a speaker's turn split by a short pause before it is emitted or decided (#73).

The live window finalizes an utterance after a fixed silence, so one spoken sentence can
arrive as several finalized fragments ("<name>, what time is" + "it?"). This bounded
stage holds the newest fragment for a short gap of stream time. A continuation by the
same speaker within the gap is joined into it; anything else releases it unchanged.

A fragment for which `breaks_turn` is true (the configured owner stop phrases, which
are matched against a whole turn, and with natural dismissal on (#98) any short
dismissal-shaped turn) is never joined or held: the held fragment is released first and
the stop fragment is emitted on its own, so joining never hides a stop and a dismissal
never waits for the hold.

Only fragments whose speaker label is known, equal, and not overlapping are joined, with
the same speaker provenance. By default an unattributed or overlapping fragment is never
joined, so the merger invents no speaker attribution itself (opt-in edge attribution in
the live window may already have inferred a label; see `LiveConfig.edge_attribution_ms`).
The one exception is opt-in tail join (`tail_join_ms`, #129), an inference: an
unattributed, non-overlapping fragment that starts within that many milliseconds of a
held labelled, non-overlapping fragment's end, with the same session-stable
`diarization-timeline` provenance, is joined into it and takes its speaker. Overlap is
still never joined, nothing is joined into an unattributed held fragment, and a labelled
fragment of another speaker is never joined. A turn formed by a tail join, or joining a
fragment marked `inferred` (edge-attributed words), is marked `"inferred": True`; the key
is absent otherwise. Live identification roles (#137) use it to withhold owner authority.
A fragment's optional `labelled_spans` (its own-label timeline audio, #148) are joined as
their concatenation. Fragments are plain dictionaries; the caller turns a released
fragment into a `Turn` and assigns its utterance id at that moment, so ids stay
unique and in emission order, and a joined turn reports the first fragment's start and
the last fragment's end.

Three opt-in refinements address one thought split into several requests (#216):

- `complete_hold_ms` holds a turn the end-of-turn model called complete for that long
  after its last word, so a pause the model misjudged can still be joined by the same
  labelled speaker's continuation. Only a labelled continuation of the same speaker joins
  it; tail join never reaches it, so an unattributed fragment is never merged into it. It
  costs the hold minus the silence already waited at finalization, and nothing at 0.
- `collapse_restarts` drops a restart when joining: when the continuation opens by
  repeating the held turn's last three or more words ("... that the tool already" +
  "that the tool already has hooks."), the repeated words are kept once.
- `bridge_backchannels` keeps the held turn held across another or an unattributed
  speaker's short backchannel ("mm", "yeah"; see `backchannel_shaped`), after Pipecat's
  user backchannel frames: the speaker's next fragment can still join, and the
  backchannel is emitted as its own turn right after the held turn. Any other speech by
  another speaker releases the held turn as before, so turns never join across a speaker
  change.

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

from .addressedness import backchannel_shaped, observe_gap, reply_wait, request_shaped
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


# Tail join (#129) is an inference; keep its window short.
MAX_TAIL_JOIN_MS = 2000


def tail_join(value: Any) -> int:
    """A validated tail join window in milliseconds: 0 turns tail join off."""
    if type(value) is not int or not 0 <= value <= MAX_TAIL_JOIN_MS:
        raise ValueError("invalid tail join window")
    return value


# A complete-judged turn's hold (#216) delays every complete request; keep it short.
MAX_COMPLETE_HOLD_MS = 3000
# A restart repeats at least this many of the held turn's last words (#216).
MIN_RESTART_WORDS = 3


def complete_hold(value: Any) -> int:
    """A validated hold in milliseconds for a turn judged complete: 0 turns it off."""
    if type(value) is not int or not 0 <= value <= MAX_COMPLETE_HOLD_MS:
        raise ValueError("invalid complete hold")
    return value


def _word_key(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def join_restart(held: str, continuation: str) -> str:
    """`held` and `continuation` joined, with a restart's repeated words kept once (#216).

    When the continuation opens with the held text's last `MIN_RESTART_WORDS` or more
    words (ignoring case and punctuation), those words are dropped from the held text;
    the longest such repeat wins. Otherwise the two are joined by a space.
    """
    tokens = held.split()
    keys = [_word_key(token) for token in tokens]
    following = [_word_key(token) for token in continuation.split()]
    for size in range(min(len(keys), len(following)), MIN_RESTART_WORDS - 1, -1):
        tail = keys[-size:]
        if all(tail) and tail == following[:size]:
            kept = " ".join(tokens[:-size])
            return f"{kept} {continuation}" if kept else continuation
    return held + " " + continuation


class TurnMerger:
    """Single-owner, synchronous; the caller supplies the stream clock.

    `offer` takes each finalized fragment in time order. `due` is called as stream time
    advances and releases the held fragment once no continuation can still arrive.
    `flush` releases it at once (end of input, a stalled source); `discard` drops it
    (cancellation). A joined turn never spans more than `max_span_ms`. With `tail_join_ms`
    above 0, an unattributed tail joins the held labelled fragment (see the module notes),
    and a labelled fragment is held at least that long so its tail can arrive.
    """

    def __init__(
        self,
        gap_ms: int,
        max_span_ms: int,
        emit: Callable[[dict[str, Any]], None],
        breaks_turn: Callable[[str], bool] | None = None,
        reply_wait_ms: int = 0,
        shaped: Callable[[str], bool] = request_shaped,
        tail_join_ms: int = 0,
        unfinished: Callable[[str], bool] | None = None,
        complete_hold_ms: int = 0,
        collapse_restarts: bool = False,
        bridge_backchannels: bool = False,
    ):
        self.gap_ms = merge_gap(gap_ms)
        self.tail_join_ms = tail_join(tail_join_ms)
        self.complete_hold_ms = complete_hold(complete_hold_ms)
        if type(collapse_restarts) is not bool or type(bridge_backchannels) is not bool:
            raise ValueError("invalid turn merge option")
        self.collapse_restarts = collapse_restarts
        self.bridge_backchannels = bridge_backchannels
        self.reply_wait_ms = reply_wait(reply_wait_ms)
        if not callable(shaped):
            raise ValueError("invalid request shape predicate")
        self.shaped = shaped
        if breaks_turn is not None and not callable(breaks_turn):
            raise ValueError("invalid turn break predicate")
        self.breaks_turn = breaks_turn
        if unfinished is not None and not callable(unfinished):
            raise ValueError("invalid unfinished-turn predicate")
        self.unfinished = unfinished
        if type(max_span_ms) is not int or max_span_ms < 1:
            raise ValueError("invalid turn merge span")
        self.max_span_ms = max_span_ms
        self.emit = emit
        self._held: dict[str, Any] | None = None
        # How long past its last word the held fragment is kept: the observation window.
        self._hold_ms = 0
        # The earliest speech the caller detected after the held fragment, in stream ms.
        self._heard_ms: int | None = None
        # Whether the held turn was judged complete and is held only by `complete_hold_ms`.
        self._held_complete = False
        # Backchannels bridged while the turn is held, emitted right after it (#216).
        self._bridged: list[dict[str, Any]] = []

    @property
    def holding(self) -> bool:
        return self._held is not None

    def _joinable(self, fragment: dict[str, Any]) -> bool:
        return fragment["speaker"] is not None and not fragment["overlap"]

    def _hold_for(self, fragment: dict[str, Any]) -> int:
        hold = max(self.gap_ms, self.tail_join_ms) if self._joinable(fragment) else 0
        if self.reply_wait_ms and self.shaped(fragment["text"]):
            hold = max(hold, self.reply_wait_ms)
        return hold

    def _continues(self, fragment: dict[str, Any]) -> bool:
        held = self._held
        if held is None or not self._joinable(held):
            return False
        if self._joinable(fragment):
            # The same speaker continuing after a pause.
            gap = max(self.gap_ms, self.complete_hold_ms) if self._held_complete else self.gap_ms
            same = (fragment["speaker"], fragment["speaker_provenance"]) == (
                held["speaker"],
                held["speaker_provenance"],
            )
        elif self._held_complete:
            # A turn judged complete only takes the same labelled speaker's continuation.
            return False
        else:
            # Tail join (#129): an unattributed, non-overlapping tail takes the held
            # speaker. Overlap is never joined. Only session-stable timeline labels are
            # inferred: an utterance-local label must not reach another utterance's words.
            gap = self.tail_join_ms
            same = (
                fragment["speaker"] is None
                and not fragment["overlap"]
                and fragment["speaker_provenance"] == held["speaker_provenance"]
                and held["speaker_provenance"] == "diarization-timeline"
            )
        return (
            gap > 0
            and same
            and fragment["start_ms"] - held["end_ms"] <= gap
            and max(held["end_ms"], fragment["end_ms"]) - held["start_ms"] <= self.max_span_ms
            and len(held["text"]) + 1 + len(fragment["text"]) <= MAX_TEXT_CHARS
        )

    def offer(self, fragment: dict[str, Any], *, complete: bool = False) -> None:
        """Take the next finalized fragment.

        `complete` means an end-of-turn model judged the speaker finished (#117): the
        fragment, or the turn it joins, is emitted at once instead of being held, and its
        post-turn gap is unobserved. A fragment held before it is still released first,
        with this one as its following speech. When the `unfinished` predicate says the
        text looks cut off mid-request (#199), `complete` is set aside and the turn is held
        as usual, so its continuation can still join it.
        """
        if self.breaks_turn is not None and self.breaks_turn(fragment["text"]):
            # A stop phrase must stay a whole turn of its own to be recognized downstream.
            self._release(fragment)
            self.emit(dict(fragment))
            return
        if self._bridges(fragment):
            self._bridged.append(dict(fragment))
            return
        if self._continues(fragment):
            held = self._held
            joined = self._join_text(held["text"], fragment["text"])
            complete = complete and not self._looks_unfinished(joined)
            if fragment["speaker"] is None or fragment.get("inferred"):
                # A tail join, or a piece with edge-attributed words: the joined turn's
                # label is partly inferred, which roles must know (#137). Absent otherwise.
                held["inferred"] = True
            if "labelled_spans" in held or "labelled_spans" in fragment:
                # Each piece's own-label timeline audio (#148); a tail has none.
                held["labelled_spans"] = [
                    *held.get("labelled_spans", ()),
                    *fragment.get("labelled_spans", ()),
                ]
            held["text"] = joined
            held["end_ms"] = max(held["end_ms"], fragment["end_ms"])
            self._hold_ms = self._hold_for(held)
            self._held_complete = False
            # Detected speech up to now belonged to the continuation just joined.
            self._heard_ms = None
            if complete and self.complete_hold_ms and not self._breaks(held["text"]):
                # Judged complete again: wait the short hold for a further piece.
                self._hold_ms, self._held_complete = self.complete_hold_ms, True
                return
            if complete or self._breaks(held["text"]):
                # The recognizer split the stop phrase itself ("never" + "mind"): emit
                # the joined phrase now so no later fragment can join and hide it. A
                # turn judged complete is not held for a continuation either.
                self.flush()
            return
        self._release(fragment)
        complete = complete and not self._looks_unfinished(fragment["text"])
        held_complete = complete and bool(self.complete_hold_ms) and self._joinable(fragment)
        if held_complete:
            hold = self.complete_hold_ms
        else:
            hold = 0 if complete else self._hold_for(fragment)
        if hold:
            self._held, self._hold_ms, self._heard_ms = dict(fragment), hold, None
            self._held_complete = held_complete
        else:
            self.emit(dict(fragment))

    def _breaks(self, text: str) -> bool:
        return self.breaks_turn is not None and self.breaks_turn(text)

    def _join_text(self, held: str, continuation: str) -> str:
        if self.collapse_restarts:
            return join_restart(held, continuation)
        return held + " " + continuation

    def _bridges(self, fragment: dict[str, Any]) -> bool:
        """Whether a fragment is another or an unattributed speaker's backchannel that
        leaves the held turn held (#216). A stop phrase never is (`offer` checks it first),
        nor is overlap; the held turn must be labelled, so it can still be continued."""
        held = self._held
        return (
            self.bridge_backchannels
            and held is not None
            and self._joinable(held)
            and not fragment["overlap"]
            and fragment["speaker"] != held["speaker"]
            and fragment["start_ms"] - held["end_ms"] <= max(self.gap_ms, self._hold_ms)
            and backchannel_shaped(fragment["text"])
        )

    def _looks_unfinished(self, text: str) -> bool:
        return self.unfinished is not None and self.unfinished(text)

    def heard(self, start_ms: int) -> None:
        """Record speech detected at `start_ms` (voice activity), with or without text.

        A detected utterance that recognition turns into no fragment would otherwise leave
        the window looking quiet; this keeps it as evidence that something followed.
        """
        if type(start_ms) is not int:
            raise ValueError("invalid detected speech start")
        if self._held is None:
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
        held_complete, self._held_complete = self._held_complete, False
        bridged, self._bridged = self._bridged, []
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
        elif following is not None and heard is not None and heard < following["start_ms"]:
            # The following speech began at its detected onset, before its first retained
            # token: measure the gap from there, and classify the speaker from the fragment.
            following = {**following, "start_ms": heard}
        if bridged and (following is None or bridged[0]["start_ms"] < following["start_ms"]):
            # The backchannel was the speech that followed the held turn.
            following = bridged[0]
        if self.reply_wait_ms and not held_complete and (following is not None or quiet):
            # A turn judged complete keeps an unobserved gap, as without the hold.
            held["post_turn_gap"] = observe_gap(held, following, self._hold_ms)
        self.emit(held)
        for backchannel in bridged:
            self.emit(backchannel)

    def discard(self) -> None:
        self._held = None
        self._heard_ms = None
        self._held_complete = False
        self._bridged = []
