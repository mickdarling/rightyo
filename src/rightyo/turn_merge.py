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

- `complete_hold_ms` holds a `diarization-timeline` turn the end-of-turn model called
  complete for that long after its last word, so a pause the model misjudged can still be
  joined by the same labelled speaker's continuation. Only a labelled continuation of the
  same speaker joins it (one with edge-attributed words counts, marked `inferred` as on
  the gap path); tail join never reaches it, so an unattributed fragment is never merged
  into it. When no one speaks within the hold it costs the hold minus the silence already
  waited at finalization, and nothing at 0; like any hold, it lasts until an utterance
  opened within it is finalized (at most `max_utterance_ms`), whoever is speaking.
- `collapse_restarts` drops a restart when a continuation joins a complete-held turn:
  when it opens by repeating the held turn's last three or more words, at least two of
  them different ("... put the milk on" + "put the milk on the list."), the repeated
  words are kept once. A repeated single word ("no no no") is never collapsed.
- `bridge_backchannels` keeps the held turn held across another or an unattributed
  speaker's short backchannel ("mm", "yeah"; see `backchannel_shaped`), after Pipecat's
  user backchannel frames: the speaker's next fragment can still join, and the
  backchannel is emitted as its own turn right after the held turn, so out of time
  order: its `start_ms` may fall inside the joined turn. Only a backchannel after the
  held turn's last word counts as the speech that followed it. Any other speech by
  another speaker releases the held turn as before, so turns never join across a speaker
  change.

One more opt-in inference joins a lone wake word to the owner's sentence (#230). The
live window can finalize the name on its own at a comma pause ("Jarvis," + "are you
there?"); the name alone is too short for speaker identification, so it arrives
unattributed and the rest arrives as the owner's. With both `wake_word` (the text is only
a configured name, plus punctuation or whitespace) and `owner` (the fragment's label
resolves to the owner role now) set, such an unattributed, non-overlapping fragment is
held until `gap_ms` has passed after it, even when the end-of-turn model called it
complete (longer only while an utterance that opened within that window is still being
collected, as for any held turn, or when a request-shaped name's reply wait is longer). A
continuation that starts within `gap_ms` of it, is labelled, does not overlap, has the
same speaker provenance and is attributed to the owner by `owner` joins it: the joined
turn takes the continuation's label and keeps the name, and is marked `inferred`, so it
is dispatched as one addressed owner request but never exercises owner authority (#137).
The wake word never raises a speaker's role: only the continuation's own owner
attribution counts, so a guest's or an unknown speaker's continuation is never joined,
and nothing else is joined into the held name. Without a joining continuation the name
is released unchanged, only later: its post-turn gap is what it would have been without
the hold (unobserved when it was judged complete or is not request-shaped; otherwise
observed over the reply wait and frozen once that window has passed, so speech after it
never counts as following the name). A stop phrase is never held or joined.

One more opt-in hold is for the owner only (hailing-station #735 R-1). With
`owner_pause_ms` above 0 and `owner` set, a labelled, non-overlapping fragment whose label
resolves to the owner role when it is held gets a longer pause: it is complete-held for
at least `owner_pause_ms` when the end-of-turn model called it complete (on the
`complete_hold_ms` path, so only a `diarization-timeline` turn and only the same labelled
speaker's continuation), and otherwise held, and joined by the same speaker's
continuation, for at least that long instead of the merge gap. Every other speaker keeps
the defaults, and the owner predicate is not asked while it is 0. A stop phrase or a
dismissal is still never held; `max_span_ms` and the text cap still bound a joined turn;
tail join keeps its own window. The owner check is made when the fragment starts being
held, again before each piece that could join it or be bridged across it (at most once
per offer), and once when the hold it would have had without the pause runs out, so a
label that stops resolving to the owner (a live identification rebind) behaves as at 0
from then on. Its post-turn gap is what it would have been without the longer hold: observed
over the window it would have had (a bridged backchannel counting as the speech that
followed, as on release) and frozen once that has passed, or unobserved when judged
complete. The pause also joins the owner's pieces with merging off (`gap_ms` 0).

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


# The owner-only pause (#735) delays only the owner's own turns; still keep it bounded.
MAX_OWNER_PAUSE_MS = 10000


def owner_pause(value: Any) -> int:
    """A validated owner-only pause in milliseconds: 0 turns it off."""
    if type(value) is not int or not 0 <= value <= MAX_OWNER_PAUSE_MS:
        raise ValueError("invalid owner pause")
    return value


def _word_key(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def join_restart(held: str, continuation: str) -> str:
    """`held` and `continuation` joined, with a restart's repeated words kept once (#216).

    When the continuation opens with the held text's last `MIN_RESTART_WORDS` or more
    words (ignoring case and punctuation), at least two of them different, those words
    are dropped from the held text; the longest such repeat wins. Otherwise the two are
    joined by a space, so a repeated word ("no no no") is never collapsed.
    """
    tokens = held.split()
    keys = [_word_key(token) for token in tokens]
    following = [_word_key(token) for token in continuation.split()]
    for size in range(min(len(keys), len(following)), MIN_RESTART_WORDS - 1, -1):
        tail = keys[-size:]
        if all(tail) and len(set(tail)) > 1 and tail == following[:size]:
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
        wake_word: Callable[[str], bool] | None = None,
        owner: Callable[[dict[str, Any]], bool] | None = None,
        owner_pause_ms: int = 0,
    ):
        self.gap_ms = merge_gap(gap_ms)
        self.owner_pause_ms = owner_pause(owner_pause_ms)
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
        if wake_word is not None and not callable(wake_word):
            raise ValueError("invalid wake word predicate")
        if owner is not None and not callable(owner):
            raise ValueError("invalid owner predicate")
        self.wake_word = wake_word
        self.owner = owner
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
        # Whether the held fragment is a lone unattributed wake word held for the
        # owner's continuation (#230).
        self._held_wake = False
        # The owner-only pause of the held fragment (#735): `owner_pause_ms` when it was
        # the owner's when it started being held, else 0.
        self._held_pause = 0
        # With the owner pause: the hold the held turn would have had without it (0: it
        # would have been emitted at once), and whether that hold would have released it
        # by now, so a label that stops being the owner's behaves exactly as at 0.
        self._base_hold = 0
        self._base_passed = False
        # Whether the held fragment is held longer than it would have been without the
        # wake word hold or the owner pause, and the post-turn gap window it would then
        # have had (None: it would have been emitted with its gap unobserved).
        self._extended = False
        self._base_window: int | None = None

    @property
    def holding(self) -> bool:
        return self._held is not None

    def _joinable(self, fragment: dict[str, Any]) -> bool:
        return fragment["speaker"] is not None and not fragment["overlap"]

    def _hold_for(self, fragment: dict[str, Any]) -> int:
        hold = max(self.gap_ms, self.tail_join_ms) if self._joinable(fragment) else 0
        if self._observed(fragment):
            hold = max(hold, self.reply_wait_ms)
        return hold

    def _observed(self, fragment: dict[str, Any]) -> bool:
        """Whether a fragment is held to observe its post-turn gap (#96)."""
        return bool(self.reply_wait_ms) and self.shaped(fragment["text"])

    def _wake_only(self, fragment: dict[str, Any]) -> bool:
        """Whether a fragment is a lone wake word, unattributed, to hold for the owner (#230).

        Its text is only a configured name (the predicate's judgement); it has no speaker
        label and does not overlap. Off unless merging, `wake_word` and `owner` are all on.
        """
        return (
            self.gap_ms > 0
            and self.wake_word is not None
            and self.owner is not None
            and fragment["speaker"] is None
            and not fragment["overlap"]
            and self._predicate(self.wake_word, fragment["text"])
        )

    @staticmethod
    def _predicate(predicate: Callable[[Any], bool], value: Any) -> bool:
        """Only a literal True counts; any other value, or an exception, is False."""
        try:
            return predicate(value) is True
        except Exception:  # noqa: BLE001 - an inference that fails never joins
            return False

    def _owner_pause_for(self, fragment: dict[str, Any], asked: dict[tuple, int]) -> int:
        """The owner-only pause for a fragment (#735), or 0.

        Only a labelled, non-overlapping fragment whose label the owner predicate resolves
        to the owner gets it. The predicate is not asked while the pause is off, and
        `asked` keeps its answers within one offer by label, overlap and provenance (all
        it reads), so it is asked at most once per offered fragment.
        """
        if not self.owner_pause_ms or self.owner is None or not self._joinable(fragment):
            return 0
        key = (fragment["speaker"], fragment["overlap"], fragment["speaker_provenance"])
        if key not in asked:
            owner = self._predicate(self.owner, dict(fragment))
            asked[key] = self.owner_pause_ms if owner else 0
        return asked[key]

    @staticmethod
    def _same_label(fragment: dict[str, Any], held: dict[str, Any]) -> bool:
        """Whether two fragments have the same label, overlap and provenance (#735)."""
        return all(
            fragment[key] == held[key] for key in ("speaker", "overlap", "speaker_provenance")
        )

    def _owner_continues(self, fragment: dict[str, Any]) -> bool:
        """Whether the owner's own labelled fragment continues the held wake word (#230).

        Only the continuation's attribution counts: a guest's or an unknown speaker's
        fragment, an unattributed or overlapping one, or one with another provenance is
        never joined. The bounds of the gap path apply, with the merge gap only.
        """
        held = self._held
        return (
            held is not None
            and self._joinable(fragment)
            and fragment["speaker_provenance"] == held["speaker_provenance"]
            and fragment["start_ms"] - held["end_ms"] <= self.gap_ms
            and max(held["end_ms"], fragment["end_ms"]) - held["start_ms"] <= self.max_span_ms
            and len(held["text"]) + 1 + len(fragment["text"]) <= MAX_TEXT_CHARS
            and self._predicate(self.owner, dict(fragment))
        )

    def _continues(self, fragment: dict[str, Any]) -> bool:
        held = self._held
        if held is None or not self._joinable(held):
            return False
        if self._joinable(fragment):
            # The same speaker continuing after a pause (the owner's pause, #735, if longer).
            gap = max(self.gap_ms, self._held_pause)
            if self._held_complete:
                gap = max(gap, self.complete_hold_ms)
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
        asked: dict[tuple, int] = {}
        if self._held_pause and (
            fragment["speaker"] is None
            or self._same_label(fragment, self._held)
            or self._bridges(fragment)
        ):
            # Before a piece that could join it or be bridged across it: whether the held
            # turn's label is still the owner's (#735). A live identification rebind ends
            # its pause, and it keeps the defaults from here. Any other piece releases it.
            self._held_pause = self._owner_pause_for(self._held, asked)
            if not self._held_pause:
                self._end_owner_pause()
        if self._bridges(fragment):
            self._bridged.append(dict(fragment))
            return
        if self._held_wake:
            # Only the owner's own sentence continues a lone wake word (#230); the owner
            # predicate is asked once per offered fragment.
            if self._owner_continues(fragment):
                # It keeps its own label and provenance and takes the name, starting
                # where the name started; the name's own gap observation is dropped.
                wake = self._held
                self._held, self._held_wake = None, False
                self._extended, self._base_window = False, None
                self._heard_ms = None
                joined = dict(fragment)
                joined["text"] = wake["text"] + " " + fragment["text"]
                joined["start_ms"] = wake["start_ms"]
                # The name's speaker is inferred from the continuation: no owner authority.
                joined["inferred"] = True
                self.offer(joined, complete=complete)
                return
        elif self._continues(fragment):
            held = self._held
            joined = self._join_text(held["text"], fragment["text"], self._held_complete)
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
            if self._extended:
                # Only an owner-paused turn is extended here (#735): the joined turn's
                # gap is observed afresh, as for any joined turn.
                held.pop("post_turn_gap", None)
                self._extended, self._base_window = False, None
            if self._held_pause:
                # The owner's joined turn waits the owner pause again (#735), with its gap
                # observed over the window it would have had without it.
                self._base_hold, self._base_passed = self._hold_ms, False
                self._extend(self._hold_ms, self._held_pause)
            # Detected speech up to now belonged to the continuation just joined.
            self._heard_ms = None
            # Backchannels bridged so far fall inside the joined turn: none followed it.
            for backchannel in self._bridged:
                backchannel["inside"] = True
            hold_complete = max(self.complete_hold_ms, self._held_pause)
            if complete and hold_complete and not self._breaks(held["text"]):
                # Judged complete again: wait the short hold for a further piece.
                self._hold_ms, self._held_complete = hold_complete, True
                self._extended, self._base_window = False, None
                self._base_hold, self._base_passed = self.complete_hold_ms, False
                return
            if complete or self._breaks(held["text"]):
                # The recognizer split the stop phrase itself ("never" + "mind"): emit
                # the joined phrase now so no later fragment can join and hide it. A
                # turn judged complete is not held for a continuation either.
                self.flush()
            return
        self._release(fragment)
        complete = complete and not self._looks_unfinished(fragment["text"])
        # The owner's own longer pause (#735); 0 for everyone else and while it is off.
        pause = self._owner_pause_for(fragment, asked)
        held_complete = (
            complete
            and bool(self.complete_hold_ms or pause)
            and self._joinable(fragment)
            # Utterance-local labels never match the next utterance's: no join is possible.
            and fragment["speaker_provenance"] == "diarization-timeline"
        )
        wake = self._wake_only(fragment)
        extended, base_window = False, None
        base_hold = self.complete_hold_ms if held_complete else 0
        if held_complete:
            hold = max(self.complete_hold_ms, pause)
        elif wake:
            # A lone wake word waits the merge gap for the owner's sentence, even when
            # judged complete. Its gap is observed over the window it would have had
            # without this hold, if any, so its evidence is unchanged (#96).
            window = 0 if complete else self._hold_for(fragment)
            hold = max(self.gap_ms, window)
            extended, base_window = True, window or None
        else:
            hold = 0 if complete else self._hold_for(fragment)
            if pause and not complete:
                # The owner's turn waits the owner pause for its continuation (#735), with
                # its gap observed over the window it would have had without it.
                base_hold = hold
                hold = max(hold, pause)
                extended, base_window = self._extension(base_hold, hold)
        if hold:
            self._held, self._hold_ms, self._heard_ms = dict(fragment), hold, None
            self._held_complete = held_complete
            self._held_wake = wake
            self._held_pause = pause
            self._base_hold, self._base_passed = base_hold, False
            self._extended, self._base_window = extended, base_window
        else:
            self.emit(dict(fragment))

    def _extension(self, base: int, hold: int) -> tuple[bool, int | None]:
        """Whether a hold is longer than `base`, the hold it would have had, and the
        post-turn gap window it would then have had (None: unobserved)."""
        if hold <= base:
            return False, None
        return True, (base or None) if self.reply_wait_ms else None

    def _end_owner_pause(self) -> None:
        """The held turn's label is no longer the owner's (#735): it behaves as without the
        pause. If the hold it would have had would already have released it (or it would
        have been emitted at once, as a turn judged complete with no `complete_hold_ms`),
        it is released now, before the piece being offered. Otherwise it is held as long
        as it would have been, which also bounds which backchannels it bridges."""
        if self._base_passed or not self._base_hold:
            self._release(None, quiet=True)
        else:
            self._hold_ms = self._base_hold

    def _extend(self, base: int, pause: int) -> None:
        """Hold the held turn for the owner pause, or `base` if that is longer (#735)."""
        self._hold_ms = max(base, pause)
        self._extended, self._base_window = self._extension(base, self._hold_ms)

    def _breaks(self, text: str) -> bool:
        return self.breaks_turn is not None and self.breaks_turn(text)

    def _join_text(self, held: str, continuation: str, onto_complete: bool) -> str:
        if self.collapse_restarts and onto_complete:
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
        window = self._base_window
        if self._extended and window is not None and "post_turn_gap" not in held:
            # Without the wake hold (#230) or the owner pause (#735) the turn would have
            # been released here: freeze its observation as it would have been, while it
            # waits on for a continuation.
            ended = held["end_ms"] + window
            if now_ms >= ended and (open_since_ms is None or open_since_ms > ended):
                # As `_release` would have: detected speech, or a bridged backchannel
                # (#216) after the turn's last word if it came first.
                following = self._heard_following(held, self._heard_ms)
                after = [b for b in self._bridged if not b.get("inside")]
                if after and (following is None or after[0]["start_ms"] < following["start_ms"]):
                    following = after[0]
                held["post_turn_gap"] = observe_gap(held, following, window)
        if self._held_pause and not self._base_passed:
            # Whether the hold without the owner pause would have released it by now.
            ended = held["end_ms"] + self._base_hold
            if now_ms >= ended and (open_since_ms is None or open_since_ms > ended):
                self._base_passed = True
                # Once, here: if its label is no longer the owner's, release it now, as
                # that hold would have (its gap evidence is frozen above).
                self._held_pause = self._owner_pause_for(held, {})
                if not self._held_pause:
                    self._release(None, quiet=True)
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
        self._held_wake, self._held_pause = False, 0
        self._base_hold, self._base_passed = 0, False
        extended, self._extended = self._extended, False
        base_window, self._base_window = self._base_window, None
        bridged, self._bridged = self._bridged, []
        if held is None:
            return
        if following is None and quiet and heard is not None:
            # Detected speech without a fragment: something followed, speaker unknown.
            following = self._heard_following(held, heard)
        elif following is not None and heard is not None and heard < following["start_ms"]:
            # The following speech began at its detected onset, before its first retained
            # token: measure the gap from there, and classify the speaker from the fragment.
            following = {**following, "start_ms": heard}
        after = [b for b in bridged if not b.pop("inside", False)]
        if after and (following is None or after[0]["start_ms"] < following["start_ms"]):
            # A backchannel after the turn's last word was the speech that followed it.
            following = after[0]
        if extended:
            # A turn held longer only for a continuation (a wake word, #230, or the owner
            # pause, #735) carries the gap it would have had without that hold: frozen in
            # `due`, or observed over its own window (speech after it counts as quiet), or
            # unobserved.
            if (
                base_window is not None
                and "post_turn_gap" not in held
                and (following is not None or quiet)
            ):
                held["post_turn_gap"] = observe_gap(held, following, base_window)
        elif self.reply_wait_ms and not held_complete and (following is not None or quiet):
            # A turn judged complete keeps an unobserved gap, as without the hold.
            held["post_turn_gap"] = observe_gap(held, following, self._hold_ms)
        self.emit(held)
        for backchannel in bridged:
            self.emit(backchannel)

    @staticmethod
    def _heard_following(held: dict[str, Any], heard: int | None) -> dict[str, Any] | None:
        """Detected speech with no fragment, as the following speech: speaker unknown."""
        if heard is None:
            return None
        return {
            "speaker": None,
            "overlap": False,
            "speaker_provenance": held["speaker_provenance"],
            "start_ms": heard,
            "end_ms": heard,
        }

    def discard(self) -> None:
        self._held = None
        self._heard_ms = None
        self._held_complete = False
        self._held_wake = False
        self._held_pause = 0
        self._base_hold, self._base_passed = 0, False
        self._extended = False
        self._base_window = None
        self._bridged = []
