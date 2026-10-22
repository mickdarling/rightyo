"""Joining a speaker's turn split by a short pause before the decision (#73).

Synthetic PCM, a scripted transcriber and a scripted diarizer only: no capture, native
inference, hosted calls or recorded speech.
"""

from __future__ import annotations

import array
import contextlib
import io
import json
import tempfile
import unittest
import wave
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

from rightyo.addressedness import observe_gap, request_shaped
from rightyo.contracts import MAX_TEXT_CHARS, Addressing, SpeakerPriority
from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import MockProvider
from rightyo.tool import listen
from rightyo.turn_merge import DEFAULT_TURN_MERGE_GAP_MS, TurnMerger

SILENCE = bytes(FRAME_BYTES)
VOICE = array.array("h", [5000] * (FRAME_BYTES // 2)).tobytes()


def fragment(text, start, end, speaker="Speaker A", overlap=False):
    return {
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "speaker": speaker,
        "overlap": overlap,
        "speaker_provenance": "diarization-timeline",
    }


class TurnMergerTests(unittest.TestCase):
    def setUp(self):
        self.emitted = []
        self.merger = TurnMerger(1000, 24000, self.emitted.append)

    def texts(self):
        return [item["text"] for item in self.emitted]

    def test_split_question_is_joined_with_first_start_and_last_end(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1600, 1800))
        self.assertEqual(self.emitted, [])
        self.merger.due(2799, None)
        self.assertEqual(self.emitted, [])
        self.merger.due(2800, None)
        self.assertEqual(self.texts(), ["Haili, what time is it?"])
        self.assertEqual((self.emitted[0]["start_ms"], self.emitted[0]["end_ms"]), (0, 1800))
        self.assertEqual(self.emitted[0]["speaker"], "Speaker A")

    def test_three_fragments_chain_into_one(self):
        for text, start, end in (
            ("Haili,", 0, 400),
            ("what time", 1200, 1700),
            ("is it?", 2400, 2900),
        ):
            self.merger.offer(fragment(text, start, end))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is it?"])

    def test_different_speakers_are_not_joined(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1200, 1400, speaker="Speaker B"))
        self.assertEqual(self.texts(), ["Haili, what time is"])
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is", "it?"])
        self.assertEqual([item["speaker"] for item in self.emitted], ["Speaker A", "Speaker B"])

    def test_gap_above_threshold_is_not_joined(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1901, 2100))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is", "it?"])
        # Exactly at the threshold still joins.
        self.emitted.clear()
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1900, 2100))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is it?"])

    def test_unattributed_or_overlapping_fragments_are_never_joined(self):
        # Joining would attribute words to a speaker the diarizer did not name.
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        self.merger.offer(fragment("it?", 1000, 1200, speaker=None))
        self.merger.offer(fragment("now", 1300, 1400, speaker=None, overlap=True))
        self.assertEqual(self.texts(), ["Haili, what time is", "it?", "now"])
        self.assertFalse(self.merger.holding)

    def test_different_speaker_provenance_is_not_joined(self):
        self.merger.offer(fragment("one", 0, 100))
        other = fragment("two", 200, 300)
        other["speaker_provenance"] = "unknown"
        self.merger.offer(other)
        self.merger.flush()
        self.assertEqual(self.texts(), ["one", "two"])

    def test_an_open_utterance_inside_the_gap_keeps_the_fragment_held(self):
        self.merger.offer(fragment("Haili, what time is", 0, 900))
        # Audio of a possible continuation began at 1500 ms, inside the 1,900 ms deadline.
        self.merger.due(5000, 1500)
        self.assertEqual(self.emitted, [])
        # An utterance that opened after the deadline cannot continue the fragment.
        self.merger.due(5000, 1901)
        self.assertEqual(self.texts(), ["Haili, what time is"])

    def test_span_and_text_bounds_stop_joining(self):
        merger = TurnMerger(1000, 2000, self.emitted.append)
        merger.offer(fragment("one", 0, 900))
        merger.offer(fragment("two", 1500, 2100))
        merger.flush()
        self.assertEqual(self.texts(), ["one", "two"])
        self.emitted.clear()
        merger = TurnMerger(1000, 24000, self.emitted.append)
        merger.offer(fragment("x" * 3000, 0, 900))
        merger.offer(fragment("y" * 1000, 1000, 1100))
        merger.flush()
        self.assertEqual([len(text) for text in self.texts()], [3000, 1000])

    def test_zero_gap_emits_immediately_and_discard_drops(self):
        merger = TurnMerger(0, 24000, self.emitted.append)
        merger.offer(fragment("one", 0, 100))
        merger.offer(fragment("two", 100, 200))
        self.assertEqual(self.texts(), ["one", "two"])
        self.emitted.clear()
        self.merger.offer(fragment("held", 0, 100))
        self.merger.discard()
        self.merger.flush()
        self.assertEqual(self.emitted, [])

    def test_a_stop_phrase_is_never_joined_or_held(self):
        merger = TurnMerger(2000, 24000, self.emitted.append, SpeakerPriority().is_stop_phrase)
        merger.offer(fragment("Haili, order a pizza", 0, 900))
        merger.offer(fragment("never mind", 2400, 2900))
        # The request is released first and the stop phrase follows as a turn of its own.
        self.assertEqual(self.texts(), ["Haili, order a pizza", "never mind"])
        self.assertFalse(merger.holding)
        self.emitted.clear()
        merger.offer(fragment("Stop.", 0, 300))
        merger.offer(fragment("stop", 1000, 1300))
        self.assertEqual(self.texts(), ["Stop.", "stop"])
        self.assertFalse(merger.holding)
        # A stop phrase the recognizer split is emitted as soon as its pieces join, so a
        # third fragment cannot join it and hide it.
        self.emitted.clear()
        merger.offer(fragment("never", 0, 300))
        merger.offer(fragment("mind", 1000, 1300))
        self.assertEqual(self.texts(), ["never mind"])
        self.assertFalse(merger.holding)
        merger.offer(fragment("okay", 2000, 2300))
        merger.flush()
        self.assertEqual(self.texts(), ["never mind", "okay"])
        # A stop phrase inside a longer turn is not a stop phrase and joins as usual.
        self.emitted.clear()
        merger.offer(fragment("please do not", 0, 300))
        merger.offer(fragment("stop the music", 1000, 1300))
        merger.flush()
        self.assertEqual(self.texts(), ["please do not stop the music"])
        with self.assertRaises(ValueError):
            TurnMerger(2000, 24000, self.emitted.append, "stop")

    def test_utterance_local_labels_never_join(self):
        first = fragment("Haili, what time is", 0, 900, speaker="u1 Speaker A")
        second = fragment("it?", 1200, 1400, speaker="u2 Speaker A")
        for item in (first, second):
            item["speaker_provenance"] = "diarization-utterance"
            self.merger.offer(item)
        self.merger.flush()
        self.assertEqual(self.texts(), ["Haili, what time is", "it?"])

    def test_gap_is_validated(self):
        for invalid in (-1, 5001, 1.5, True, "700", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TurnMerger(invalid, 24000, self.emitted.append)


# A lone wake word split from the owner's sentence (#230). Synthetic text only.
JARVIS = Addressing(("Jarvis",), (("Jarvis", ("Jarvus",)),))
OWNER = "Speaker A"
GUEST = "Speaker B"


def wake_merger(emitted, gap=2000, **options):
    return TurnMerger(
        gap,
        24000,
        emitted.append,
        SpeakerPriority().is_stop_phrase,
        wake_word=lambda text: JARVIS.name_for(text) is not None,
        owner=lambda item: item["speaker"] == OWNER,
        **options,
    )


class WakeWordJoinTests(unittest.TestCase):
    """A lone unattributed wake word joins the owner's continuation, and nothing else."""

    def setUp(self):
        self.emitted = []
        self.merger = wake_merger(self.emitted)

    def pairs(self):
        return [(item["text"], item["speaker"]) for item in self.emitted]

    def test_owner_continuation_joins_at_0_300_and_1000_ms(self):
        for gap in (0, 300, 1000):
            for complete in (False, True):
                with self.subTest(gap=gap, complete=complete):
                    self.emitted.clear()
                    # The end-of-turn model may call the name alone complete.
                    self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None), complete=complete)
                    self.assertEqual(self.emitted, [])
                    self.merger.offer(fragment("are you there?", 1000 + gap, 2200 + gap))
                    self.merger.flush()
                    self.assertEqual(self.pairs(), [("Jarvis, are you there?", OWNER)])
                    (turn,) = self.emitted
                    self.assertEqual((turn["start_ms"], turn["end_ms"]), (0, 2200 + gap))
                    self.assertEqual(turn["speaker_provenance"], "diarization-timeline")
                    self.assertIs(turn["inferred"], True)
                    self.assertFalse(turn["overlap"])

    def test_a_variant_spelling_counts_and_a_complete_continuation_is_emitted_at_once(self):
        self.merger.offer(fragment("jarvus.", 0, 900, speaker=None))
        self.merger.offer(fragment("are you there?", 900, 1800), complete=True)
        self.assertEqual(self.pairs(), [("jarvus. are you there?", OWNER)])
        self.assertFalse(self.merger.holding)

    def test_the_joined_turn_still_joins_the_owners_next_piece(self):
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        self.merger.offer(fragment("are you", 1000, 1500))
        self.merger.offer(fragment("there?", 2500, 2900))
        self.merger.flush()
        self.assertEqual(self.pairs(), [("Jarvis, are you there?", OWNER)])
        self.assertIs(self.emitted[0]["inferred"], True)

    def test_a_guest_continuation_is_never_joined(self):
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        self.merger.offer(fragment("are you there?", 1000, 2200, speaker=GUEST))
        self.merger.flush()
        self.assertEqual(self.pairs(), [("Jarvis,", None), ("are you there?", GUEST)])
        self.assertTrue(all("inferred" not in item for item in self.emitted))

    def test_an_unknown_continuation_is_never_joined(self):
        for speaker, overlap in (("Speaker C", False), (None, False), (OWNER, True)):
            with self.subTest(speaker=speaker, overlap=overlap):
                self.emitted.clear()
                self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
                self.merger.offer(
                    fragment("are you there?", 1000, 2200, speaker=speaker, overlap=overlap)
                )
                self.merger.flush()
                self.assertEqual(self.pairs(), [("Jarvis,", None), ("are you there?", speaker)])
                self.assertTrue(all("inferred" not in item for item in self.emitted))

    def test_another_provenance_or_a_failing_owner_predicate_never_joins(self):
        other = fragment("are you there?", 1000, 2200)
        other["speaker_provenance"] = "unknown"
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        self.merger.offer(other)
        self.merger.flush()
        self.assertEqual(self.pairs(), [("Jarvis,", None), ("are you there?", OWNER)])

        def broken(_item):
            raise RuntimeError("identification failed")

        emitted = []
        merger = TurnMerger(2000, 24000, emitted.append, wake_word=bool, owner=broken)
        merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        merger.offer(fragment("are you there?", 1000, 2200))
        merger.flush()
        self.assertEqual([item["text"] for item in emitted], ["Jarvis,", "are you there?"])

    def test_without_a_continuation_in_the_gap_the_wake_word_is_emitted_as_today(self):
        baseline = []
        TurnMerger(2000, 24000, baseline.append).offer(fragment("Jarvis,", 0, 1000, speaker=None))
        # A continuation just past the merge gap does not join.
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        self.merger.offer(fragment("are you there?", 3001, 4000))
        self.merger.flush()
        self.assertEqual(self.emitted[0], baseline[0])
        self.assertEqual(self.pairs(), [("Jarvis,", None), ("are you there?", OWNER)])
        # Silence: released, unchanged, once the merge gap has passed and no later.
        self.emitted.clear()
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None), complete=True)
        self.merger.due(2999, None)
        self.assertEqual(self.emitted, [])
        self.merger.due(3000, None)
        self.assertEqual(self.emitted, baseline)
        # With post-turn gaps observed (#96), a name that would not have been held for
        # its gap is still released without one.
        emitted = []
        merger = wake_merger(emitted, reply_wait_ms=1200, shaped=lambda text: False)
        merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        merger.due(3000, None)
        self.assertEqual(emitted, baseline)

    def test_the_hold_is_bounded_by_the_merge_gap(self):
        merger = wake_merger(self.emitted, gap=800)
        merger.offer(fragment("Jarvis,", 0, 1000, speaker=None), complete=True)
        merger.due(1800, None)
        self.assertEqual(self.pairs(), [("Jarvis,", None)])
        # Merging off: no hold, no join.
        self.emitted.clear()
        merger = wake_merger(self.emitted, gap=0)
        merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        self.assertEqual(self.pairs(), [("Jarvis,", None)])
        self.assertFalse(merger.holding)

    def test_a_stop_phrase_is_never_held_or_joined(self):
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        self.merger.offer(fragment("stop", 1000, 1300))
        self.assertEqual(self.pairs(), [("Jarvis,", None), ("stop", OWNER)])
        self.assertFalse(self.merger.holding)
        self.assertTrue(all("inferred" not in item for item in self.emitted))
        # An unattributed stop phrase is not held for the owner either.
        self.emitted.clear()
        self.merger.offer(fragment("never mind", 0, 400, speaker=None))
        self.assertEqual(self.pairs(), [("never mind", None)])
        self.assertFalse(self.merger.holding)

    def test_other_unattributed_fragments_behave_as_today(self):
        for text, overlap in (
            ("are you there?", False),
            ("hey Jarvis", False),
            ("Jarvis, what time is it", False),
            ("Jarvis,", True),
        ):
            with self.subTest(text=text, overlap=overlap):
                baseline = []
                TurnMerger(2000, 24000, baseline.append).offer(
                    fragment(text, 0, 1000, speaker=None, overlap=overlap)
                )
                self.emitted.clear()
                self.merger.offer(fragment(text, 0, 1000, speaker=None, overlap=overlap))
                # Emitted at once, unchanged, and never joined to the owner's next turn.
                self.assertEqual(self.emitted, baseline)
                self.assertFalse(self.merger.holding)
                self.merger.offer(fragment("are you there?", 1000, 2200))
                self.merger.flush()
                self.assertEqual(self.emitted[1]["text"], "are you there?")

    def test_predicates_are_validated(self):
        for options in ({"wake_word": "Jarvis"}, {"owner": OWNER}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                TurnMerger(2000, 24000, self.emitted.append, **options)
        backends = {
            "provenance": "causal-replay",
            "transcriber": ScriptedTranscriber(TEXTS),
            "diarizer": ScriptedDiarizer(),
        }
        # A valid configuration first, so the errors below come from the new checks.
        LiveConfig("wake-test", **backends, turn_wake_word=bool, turn_owner=bool)
        for options, message in (
            ({"turn_wake_word": "Jarvis"}, "Invalid wake word predicate"),
            ({"turn_owner": OWNER}, "Invalid owner predicate"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(LiveAudioError, message):
                LiveConfig("wake-test", **backends, **options)

    def test_only_a_literal_true_from_a_predicate_counts(self):
        for wake_word, owner in ((lambda text: 1, lambda item: True), (bool, lambda item: "yes")):
            with self.subTest(wake_word=wake_word, owner=owner):
                emitted = []
                merger = TurnMerger(2000, 24000, emitted.append, wake_word=wake_word, owner=owner)
                merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
                merger.offer(fragment("are you there?", 1000, 2200))
                merger.flush()
                self.assertEqual([item["text"] for item in emitted], ["Jarvis,", "are you there?"])

    def test_the_mergers_own_guards_reject_overlap_and_unattributed_continuations(self):
        # Even with an owner predicate that accepts everything.
        for speaker, overlap in ((OWNER, True), (None, False)):
            with self.subTest(speaker=speaker, overlap=overlap):
                emitted = []
                merger = TurnMerger(
                    2000, 24000, emitted.append, wake_word=bool, owner=lambda item: True
                )
                merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
                merger.offer(
                    fragment("are you there?", 1000, 2200, speaker=speaker, overlap=overlap)
                )
                merger.flush()
                self.assertEqual([item["text"] for item in emitted], ["Jarvis,", "are you there?"])
                self.assertTrue(all("inferred" not in item for item in emitted))

    def test_span_and_text_limits_apply_to_the_wake_join(self):
        merger = TurnMerger(
            2000, 2000, self.emitted.append, wake_word=bool, owner=lambda item: True
        )
        merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        merger.offer(fragment("are you there?", 1500, 2001))
        merger.flush()
        self.assertEqual(self.pairs(), [("Jarvis,", None), ("are you there?", OWNER)])
        # Exactly at the span limit still joins.
        self.emitted.clear()
        merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
        merger.offer(fragment("are you there?", 1500, 2000))
        merger.flush()
        self.assertEqual(self.pairs(), [("Jarvis, are you there?", OWNER)])
        # The joined text may not exceed the turn text limit.
        fits = "a" * (MAX_TEXT_CHARS - len("Jarvis,") - 1)
        for text, joined in ((fits, True), (fits + "a", False)):
            with self.subTest(length=len(text)):
                self.emitted.clear()
                self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=None))
                self.merger.offer(fragment(text, 1000, 2200))
                self.merger.flush()
                self.assertEqual(len(self.emitted), 1 if joined else 2)

    def test_a_labelled_name_is_never_held_as_a_wake_word(self):
        # A guest's labelled name keeps its own speaker: it joins the guest's own
        # continuation as before, never the owner's sentence.
        self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=GUEST))
        self.merger.offer(fragment("are you there?", 1000, 2200))
        self.merger.flush()
        self.assertEqual(self.pairs(), [("Jarvis,", GUEST), ("are you there?", OWNER)])
        for speaker in (GUEST, OWNER):
            with self.subTest(speaker=speaker):
                self.emitted.clear()
                self.merger.offer(fragment("Jarvis,", 0, 1000, speaker=speaker))
                self.merger.offer(fragment("are you there?", 1000, 2200, speaker=speaker))
                self.merger.flush()
                self.assertEqual(self.pairs(), [("Jarvis, are you there?", speaker)])
                # The ordinary same-speaker join: nothing inferred.
                self.assertNotIn("inferred", self.emitted[0])


def drive(merger, steps):
    """Run a scripted stream against a merger: ("due", now, open_since), ("offer",
    fragment, complete), ("heard", ms) or ("flush",)."""
    for step in steps:
        if step[0] == "due":
            merger.due(step[1], step[2])
        elif step[0] == "offer":
            merger.offer(dict(step[1]), complete=step[2])
        elif step[0] == "heard":
            merger.heard(step[1])
        else:
            merger.flush()


def ticks(start, stop, open_since=None):
    return [("due", now, open_since) for now in range(start, stop + 1, 100)]


class WakeWordEvidenceTests(unittest.TestCase):
    """A request-shaped lone name that joins nothing reaches downstream exactly as on main.

    The baseline is a merger without the wake options; both use the real `request_shaped`
    and the prototype defaults (merge gap 2,000 ms, reply wait 1,200 ms).
    """

    def compare(self, steps):
        baseline, emitted = [], []
        stop = SpeakerPriority().is_stop_phrase
        drive(TurnMerger(2000, 24000, baseline.append, stop, reply_wait_ms=1200), steps)
        drive(wake_merger(emitted, reply_wait_ms=1200), steps)
        self.assertEqual(json.dumps(emitted, sort_keys=True), json.dumps(baseline, sort_keys=True))
        return emitted

    def test_the_name_is_request_shaped(self):
        self.assertTrue(request_shaped("Jarvis?"))

    def scenarios(self, complete):
        name = ("offer", fragment("Jarvis?", 0, 1000, speaker=None), complete)
        guest_late = fragment("hello there", 2500, 3000, speaker=GUEST)
        guest_soon = fragment("hello there", 1500, 2000, speaker=GUEST)
        return {
            "quiet": [name, *ticks(1000, 5000), ("flush",)],
            "another speaker after the reply window": [
                name,
                *ticks(1000, 2400),
                *ticks(2500, 4400, open_since=2500),
                ("offer", guest_late, False),
                *ticks(4500, 7000),
                ("flush",),
            ],
            "another speaker inside the reply window": [
                name,
                *ticks(1000, 1400),
                *ticks(1500, 3400, open_since=1500),
                ("offer", guest_soon, False),
                *ticks(3500, 6000),
                ("flush",),
            ],
            "speech heard without text, then quiet": [
                name,
                ("heard", 1100),
                *ticks(1000, 5000),
                ("flush",),
            ],
            "speech heard after the reply window": [
                name,
                *ticks(1000, 2300),
                ("heard", 2400),
                *ticks(2400, 5000),
                ("flush",),
            ],
        }

    def test_not_judged_complete(self):
        for label, steps in self.scenarios(False).items():
            with self.subTest(label):
                emitted = self.compare(steps)
                self.assertEqual(emitted[0]["post_turn_gap"]["window_ms"], 1200)

    def test_judged_complete(self):
        for label, steps in self.scenarios(True).items():
            with self.subTest(label):
                emitted = self.compare(steps)
                self.assertNotIn("post_turn_gap", emitted[0])

    def test_the_owner_still_joins_and_the_name_alone_is_never_emitted(self):
        emitted = []
        merger = wake_merger(emitted, reply_wait_ms=1200)
        drive(
            merger,
            [
                ("offer", fragment("Jarvis?", 0, 1000, speaker=None), False),
                *ticks(1000, 2900),
                ("offer", fragment("are you there?", 3000, 3500), False),
                ("flush",),
            ],
        )
        self.assertEqual([item["text"] for item in emitted], ["Jarvis? are you there?"])
        self.assertIs(emitted[0]["inferred"], True)


# The owner-only pause (hailing-station #735 R-1). Synthetic text only.
PAUSE = 6000


def is_owner(item):
    return item["speaker"] == OWNER


def pause_merger(emitted, pause=PAUSE, gap=2000, owner=is_owner, **options):
    return TurnMerger(
        gap,
        24000,
        emitted.append,
        SpeakerPriority().is_stop_phrase,
        owner=owner,
        owner_pause_ms=pause,
        **options,
    )


def dumps(items):
    return json.dumps(items, sort_keys=True)


class OwnerPauseTests(unittest.TestCase):
    """The owner's labelled turns wait `owner_pause_ms`; everyone else keeps the defaults."""

    def setUp(self):
        self.emitted = []
        self.merger = pause_merger(self.emitted)

    def texts(self):
        return [item["text"] for item in self.emitted]

    def test_an_owner_turn_judged_complete_is_held_the_owner_pause(self):
        self.merger.offer(fragment("Jarvis, add milk to the list.", 0, 1000), complete=True)
        self.merger.due(1000 + PAUSE - 1, None)
        self.assertEqual(self.emitted, [])
        self.merger.due(1000 + PAUSE, None)
        self.assertEqual(self.texts(), ["Jarvis, add milk to the list."])
        # Judged complete: its post-turn gap stays unobserved, as with `complete_hold_ms`.
        self.assertNotIn("post_turn_gap", self.emitted[0])

    def test_an_owner_continuation_within_the_pause_joins(self):
        for complete in (False, True):
            with self.subTest(complete=complete):
                self.emitted.clear()
                self.merger.offer(fragment("Jarvis, add milk", 0, 1000), complete=complete)
                # Four seconds later: past the 2,000 ms merge gap, within the pause.
                self.merger.due(4900, None)
                self.assertEqual(self.emitted, [])
                self.merger.offer(fragment("and eggs to the list.", 5000, 6000), complete=complete)
                self.merger.due(6000 + PAUSE - 1, None)
                self.assertEqual(self.emitted, [])
                self.merger.due(6000 + PAUSE, None)
                self.assertEqual(self.texts(), ["Jarvis, add milk and eggs to the list."])
                (turn,) = self.emitted
                self.assertEqual((turn["start_ms"], turn["end_ms"]), (0, 6000))
                self.assertEqual(turn["speaker"], OWNER)
                self.assertNotIn("inferred", turn)

    def test_an_owner_continuation_after_the_pause_does_not_join(self):
        self.merger.offer(fragment("Jarvis, add milk.", 0, 1000), complete=True)
        self.merger.due(1000 + PAUSE, None)
        self.merger.offer(fragment("And eggs.", 1000 + PAUSE + 100, 1000 + PAUSE + 600))
        self.merger.flush()
        self.assertEqual(self.texts(), ["Jarvis, add milk.", "And eggs."])

    def run_both(self, steps, **options):
        """The same stream through a merger with the owner pause and one without it."""
        baseline, emitted = [], []
        stop = SpeakerPriority().is_stop_phrase
        drive(TurnMerger(2000, 24000, baseline.append, stop, **options), steps)
        drive(pause_merger(emitted, **options), steps)
        return emitted, baseline

    def guest_and_unknown_streams(self):
        streams = {}
        for label, speaker, overlap in (
            ("guest", GUEST, False),
            ("unattributed", None, False),
            ("overlapping owner", OWNER, True),
        ):
            for complete in (False, True):
                first = fragment("Jarvis, add milk", 0, 1000, speaker=speaker, overlap=overlap)
                later = fragment("and eggs.", 5000, 6000, speaker=speaker, overlap=overlap)
                soon = fragment("and eggs.", 2500, 3000, speaker=speaker, overlap=overlap)
                streams[f"{label}, complete={complete}"] = [
                    ("offer", first, complete),
                    *ticks(1000, 4900),
                    ("offer", later, complete),
                    *ticks(6000, 14000),
                    ("offer", first, complete),
                    *ticks(1000, 2400),
                    ("offer", soon, False),
                    ("flush",),
                ]
        return streams

    def test_guest_and_unknown_fragments_are_unchanged(self):
        for label, steps in self.guest_and_unknown_streams().items():
            for options in ({}, {"reply_wait_ms": 1200}, {"complete_hold_ms": 1500}):
                with self.subTest(label, **options):
                    emitted, baseline = self.run_both(steps, **options)
                    self.assertEqual(dumps(emitted), dumps(baseline))
                    self.assertNotIn("Jarvis, add milk and eggs.", [i["text"] for i in emitted[:2]])

    def test_a_label_the_owner_predicate_does_not_resolve_is_unchanged(self):
        # Speaker C has no role: a participant or an unknown speaker in the prototype.
        steps = [
            ("offer", fragment("Jarvis, add milk", 0, 1000, speaker="Speaker C"), True),
            *ticks(1000, 4900),
            ("offer", fragment("and eggs.", 5000, 6000, speaker="Speaker C"), False),
            ("flush",),
        ]
        emitted, baseline = self.run_both(steps, reply_wait_ms=1200)
        self.assertEqual(dumps(emitted), dumps(baseline))
        self.assertEqual(len(emitted), 2)

    def test_a_failing_or_non_true_owner_predicate_gives_no_pause(self):
        def broken(_item):
            raise RuntimeError("identification failed")

        for owner in (broken, lambda _item: "owner", lambda _item: 1):
            with self.subTest(owner=owner):
                emitted = []
                merger = pause_merger(emitted, owner=owner)
                merger.offer(fragment("Jarvis, add milk.", 0, 1000), complete=True)
                self.assertEqual([item["text"] for item in emitted], ["Jarvis, add milk."])

    def test_a_stop_phrase_is_never_held(self):
        for phrase in ("never mind", "Stop."):
            for complete in (False, True):
                with self.subTest(phrase=phrase, complete=complete):
                    self.emitted.clear()
                    # Alone: emitted at once, judged complete or not.
                    self.merger.offer(fragment(phrase, 0, 300), complete=complete)
                    self.assertEqual(self.texts(), [phrase])
                    self.assertFalse(self.merger.holding)
                    # After a held owner turn: that turn is released first.
                    self.emitted.clear()
                    self.merger.offer(fragment("Jarvis, order a pizza", 0, 900), complete=True)
                    self.merger.offer(fragment(phrase, 4000, 4300), complete=complete)
                    self.assertEqual(self.texts(), ["Jarvis, order a pizza", phrase])
                    self.assertFalse(self.merger.holding)
        # A dismissal predicate counts the same way.
        emitted = []
        merger = TurnMerger(
            2000,
            24000,
            emitted.append,
            lambda text: text == "go away",
            owner=is_owner,
            owner_pause_ms=PAUSE,
        )
        merger.offer(fragment("go away", 0, 300), complete=True)
        self.assertEqual([item["text"] for item in emitted], ["go away"])

    def test_a_recognizer_split_stop_phrase_is_emitted_when_it_joins(self):
        self.merger.offer(fragment("never", 0, 300))
        self.merger.offer(fragment("mind", 3000, 3300))
        self.assertEqual(self.texts(), ["never mind"])
        self.assertFalse(self.merger.holding)

    def test_max_span_still_bounds_a_joined_turn(self):
        emitted = []
        merger = TurnMerger(2000, 5000, emitted.append, owner=is_owner, owner_pause_ms=PAUSE)
        merger.offer(fragment("Jarvis, add milk", 0, 1000), complete=True)
        # Within the pause, but the joined turn would span 5,500 ms > 5,000 ms.
        merger.offer(fragment("and eggs.", 4000, 5500))
        merger.flush()
        self.assertEqual([item["text"] for item in emitted], ["Jarvis, add milk", "and eggs."])
        emitted.clear()
        merger.offer(fragment("Jarvis, add milk", 0, 1000), complete=True)
        merger.offer(fragment("and eggs.", 4000, 5000))
        merger.flush()
        self.assertEqual([item["text"] for item in emitted], ["Jarvis, add milk and eggs."])

    def test_the_text_cap_still_bounds_a_joined_turn(self):
        long = "a" * (MAX_TEXT_CHARS - 5)
        self.merger.offer(fragment(long, 0, 1000), complete=True)
        self.merger.offer(fragment("and eggs", 4000, 4500))
        self.merger.flush()
        self.assertEqual(self.texts(), [long, "and eggs"])

    def test_the_pause_is_validated(self):
        self.assertEqual(pause_merger([], pause=10000).owner_pause_ms, 10000)
        self.assertEqual(TurnMerger(2000, 24000, self.emitted.append).owner_pause_ms, 0)
        for invalid in (-1, 10001, 1.5, True, "4000", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                pause_merger([], pause=invalid)

    def test_a_shorter_pause_never_shortens_the_merge_gap(self):
        emitted = []
        merger = pause_merger(emitted, pause=500)
        merger.offer(fragment("Jarvis, add milk", 0, 1000))
        merger.offer(fragment("and eggs.", 2800, 3000))
        merger.flush()
        self.assertEqual([item["text"] for item in emitted], ["Jarvis, add milk and eggs."])


class OwnerPauseInteractionTests(unittest.TestCase):
    """The owner pause with reply wait, tail join, backchannels, hold_unfinished, wake join."""

    def test_post_turn_gap_evidence_is_what_it_would_have_been(self):
        # The owner's request-shaped turn, released later, carries the same gap evidence.
        name = ("offer", fragment("what time is it?", 0, 1000), False)
        guest_soon = fragment("hello there", 1500, 2000, speaker=GUEST)
        guest_late = fragment("hello there", 4000, 4500, speaker=GUEST)
        scenarios = {
            "quiet": [name, *ticks(1000, 9000), ("flush",)],
            "another speaker inside the window": [
                name,
                *ticks(1000, 1400),
                *ticks(1500, 1900, open_since=1500),
                ("offer", guest_soon, False),
                *ticks(2000, 9000),
                ("flush",),
            ],
            "another speaker after the window": [
                name,
                *ticks(1000, 3900),
                *ticks(4000, 4400, open_since=4000),
                ("offer", guest_late, False),
                *ticks(4500, 12000),
                ("flush",),
            ],
            "speech heard after the window": [
                name,
                *ticks(1000, 3500),
                ("heard", 3600),
                *ticks(3600, 9000),
                ("flush",),
            ],
        }
        for label, steps in scenarios.items():
            with self.subTest(label):
                baseline, emitted = [], []
                stop = SpeakerPriority().is_stop_phrase
                drive(TurnMerger(2000, 24000, baseline.append, stop, reply_wait_ms=1200), steps)
                drive(pause_merger(emitted, reply_wait_ms=1200), steps)
                self.assertEqual(dumps(emitted), dumps(baseline))
                self.assertEqual(emitted[0]["post_turn_gap"]["window_ms"], 2000)

    def test_a_joined_owner_turn_observes_its_gap_afresh(self):
        emitted = []
        merger = pause_merger(emitted, reply_wait_ms=1200)
        drive(
            merger,
            [
                ("offer", fragment("Jarvis, what time", 0, 1000), False),
                *ticks(1000, 4900),
                ("offer", fragment("is it in Tokyo?", 5000, 6000), False),
                *ticks(6000, 12000),
            ],
        )
        (turn,) = emitted
        self.assertEqual(turn["text"], "Jarvis, what time is it in Tokyo?")
        self.assertEqual(
            turn["post_turn_gap"],
            observe_gap(turn, None, 2000),
        )

    def test_tail_join_keeps_its_own_window(self):
        def run(start):
            emitted = []
            merger = pause_merger(emitted, tail_join_ms=1000)
            merger.offer(fragment("Jarvis, add milk", 0, 1000))
            merger.offer(fragment("please", start, start + 300, speaker=None))
            merger.flush()
            return [(item["text"], item["speaker"]) for item in emitted]

        self.assertEqual(run(1800), [("Jarvis, add milk please", OWNER)])
        self.assertEqual(run(3000), [("Jarvis, add milk", OWNER), ("please", None)])

    def test_a_backchannel_is_bridged_across_the_pause(self):
        emitted = []
        merger = pause_merger(emitted, bridge_backchannels=True)
        merger.offer(fragment("Jarvis, add milk", 0, 1000), complete=True)
        merger.offer(fragment("mm", 3000, 3200, speaker=GUEST))
        merger.offer(fragment("and eggs.", 5000, 6000))
        merger.flush()
        self.assertEqual(
            [(item["text"], item["speaker"]) for item in emitted],
            [("Jarvis, add milk and eggs.", OWNER), ("mm", GUEST)],
        )

    def test_another_speakers_turn_still_releases_the_held_owner_turn(self):
        emitted = []
        merger = pause_merger(emitted)
        merger.offer(fragment("Jarvis, add milk", 0, 1000), complete=True)
        merger.offer(fragment("What did you say?", 3000, 3500, speaker=GUEST))
        self.assertEqual([item["text"] for item in emitted], ["Jarvis, add milk"])

    def test_an_unfinished_owner_turn_waits_the_pause(self):
        emitted = []
        merger = pause_merger(emitted, unfinished=lambda text: text.endswith("the"))
        merger.offer(fragment("Jarvis, add the", 0, 1000), complete=True)
        merger.offer(fragment("milk.", 5000, 5500), complete=True)
        merger.flush()
        self.assertEqual([item["text"] for item in emitted], ["Jarvis, add the milk."])

    def test_a_wake_word_joined_turn_then_waits_the_owner_pause(self):
        emitted = []
        merger = pause_merger(emitted, wake_word=lambda text: JARVIS.name_for(text) is not None)
        merger.offer(fragment("Jarvis,", 0, 800, speaker=None), complete=True)
        merger.offer(fragment("add milk", 1500, 2500), complete=True)
        merger.offer(fragment("and eggs.", 6000, 7000), complete=True)
        merger.flush()
        (turn,) = emitted
        self.assertEqual((turn["text"], turn["speaker"]), ("Jarvis, add milk and eggs.", OWNER))
        self.assertIs(turn["inferred"], True)

    def test_a_guest_never_continues_the_wake_word_or_gets_the_pause(self):
        emitted = []
        merger = pause_merger(emitted, wake_word=lambda text: JARVIS.name_for(text) is not None)
        merger.offer(fragment("Jarvis,", 0, 800, speaker=None), complete=True)
        merger.offer(fragment("add milk", 1500, 2500, speaker=GUEST), complete=True)
        self.assertEqual(
            [(item["text"], item["speaker"]) for item in emitted],
            [("Jarvis,", None), ("add milk", GUEST)],
        )


def owner_pause_streams():
    """Scripted streams for the differential at 0: every option the merger has."""
    owner, guest = OWNER, GUEST
    return {
        "split owner question": [
            ("offer", fragment("Jarvis, what time is", 0, 900, speaker=owner), False),
            *ticks(900, 1500),
            ("offer", fragment("it?", 1600, 1800, speaker=owner), False),
            *ticks(1800, 6000),
        ],
        "complete owner turn then continuation": [
            ("offer", fragment("Jarvis, add milk", 0, 1000, speaker=owner), True),
            *ticks(1000, 1900),
            ("offer", fragment("add milk and eggs.", 2000, 3000, speaker=owner), True),
            *ticks(3000, 8000),
        ],
        "owner, backchannel, owner": [
            ("offer", fragment("Jarvis, add milk", 0, 1000, speaker=owner), True),
            ("offer", fragment("mm", 1300, 1500, speaker=guest), False),
            ("offer", fragment("and eggs.", 2200, 3000, speaker=owner), False),
            *ticks(3000, 8000),
        ],
        "owner then tail then guest": [
            ("offer", fragment("Jarvis, add milk", 0, 1000, speaker=owner), False),
            ("offer", fragment("please", 1400, 1600, speaker=None), False),
            *ticks(1600, 2000),
            ("heard", 2100),
            ("offer", fragment("hello there", 2500, 3000, speaker=guest), False),
            *ticks(3000, 8000),
        ],
        "lone wake word then owner": [
            ("offer", fragment("Jarvis?", 0, 800, speaker=None), True),
            ("offer", fragment("are you there?", 1500, 2500, speaker=owner), False),
            *ticks(2500, 8000),
        ],
        "owner stop phrase": [
            ("offer", fragment("Jarvis, order a pizza", 0, 900, speaker=owner), False),
            ("offer", fragment("never mind", 1500, 1800, speaker=owner), False),
            ("offer", fragment("Stop.", 2500, 2800, speaker=owner), True),
            *ticks(2800, 6000),
        ],
        "unfinished owner turn, then flush": [
            ("offer", fragment("Jarvis, add the", 0, 900, speaker=owner), True),
            *ticks(900, 2000),
            ("flush",),
        ],
        "guest and overlap": [
            ("offer", fragment("Jarvis, add milk", 0, 900, speaker=guest), True),
            ("offer", fragment("what time is it?", 1000, 1500, speaker=owner, overlap=True), False),
            *ticks(1500, 6000),
        ],
    }


class OwnerPauseOffDifferentialTests(unittest.TestCase):
    """With `owner_pause_ms` 0 the merger's output is byte-identical to one without the option."""

    OPTIONS = (
        {},
        {"reply_wait_ms": 1200},
        {"complete_hold_ms": 1500, "collapse_restarts": True},
        {"bridge_backchannels": True, "complete_hold_ms": 1500, "reply_wait_ms": 1200},
        {"tail_join_ms": 1000, "reply_wait_ms": 1200},
        {"unfinished": lambda text: text.endswith("the")},
    )

    def test_zero_is_byte_identical_without_an_owner_predicate(self):
        stop = SpeakerPriority().is_stop_phrase
        for options in self.OPTIONS:
            for label, steps in owner_pause_streams().items():
                with self.subTest(label, options=sorted(options)):
                    baseline, emitted, asked = [], [], []

                    def counting(item):
                        asked.append(item)
                        return item["speaker"] == OWNER

                    drive(TurnMerger(2000, 24000, baseline.append, stop, **options), steps)
                    drive(
                        TurnMerger(
                            2000,
                            24000,
                            emitted.append,
                            stop,
                            owner=counting,
                            owner_pause_ms=0,
                            **options,
                        ),
                        steps,
                    )
                    self.assertEqual(dumps(emitted), dumps(baseline))
                    # The owner predicate is never asked while the pause is off.
                    self.assertEqual(asked, [])

    def test_zero_is_byte_identical_with_the_wake_join(self):
        stop = SpeakerPriority().is_stop_phrase
        wake = {"wake_word": lambda text: JARVIS.name_for(text) is not None, "owner": is_owner}
        for options in self.OPTIONS:
            for label, steps in owner_pause_streams().items():
                with self.subTest(label, options=sorted(options)):
                    baseline, emitted = [], []
                    drive(TurnMerger(2000, 24000, baseline.append, stop, **wake, **options), steps)
                    drive(
                        TurnMerger(
                            2000, 24000, emitted.append, stop, owner_pause_ms=0, **wake, **options
                        ),
                        steps,
                    )
                    self.assertEqual(dumps(emitted), dumps(baseline))

    def test_the_streams_exercise_the_owner_pause_when_on(self):
        # Control: the same streams differ with the pause on, so the differential has teeth.
        stop = SpeakerPriority().is_stop_phrase
        differs = []
        for label, steps in owner_pause_streams().items():
            baseline, emitted = [], []
            drive(TurnMerger(2000, 24000, baseline.append, stop), steps)
            drive(pause_merger(emitted), steps)
            if dumps(emitted) != dumps(baseline):
                differs.append(label)
        self.assertIn("complete owner turn then continuation", differs)
        self.assertNotIn("guest and overlap", differs)


class ScriptedTranscriber:
    """One scripted text per utterance, timed to the utterance's voiced frames."""

    recognizer_id = "scripted-fixture"

    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe(self, pcm, register=None):
        voiced = [
            index
            for index in range(len(pcm) // FRAME_BYTES)
            if pcm[index * FRAME_BYTES : (index + 1) * FRAME_BYTES] == VOICE
        ]
        if not voiced or not self.texts:
            return []
        return [
            {
                "text": " " + self.texts.pop(0),
                "start_ms": voiced[0] * 20,
                "end_ms": (voiced[-1] + 1) * 20,
            }
        ]


class ScriptedDiarizer:
    """Speaker 1 throughout, or speaker 2 from `switch_ms` on (a contract-valid timeline)."""

    speaker_provenance = "diarization-timeline"

    def __init__(self, switch_ms=None):
        self.switch_ms, self.pushed_ms = switch_ms, 0

    def push(self, pcm):
        self.pushed_ms += len(pcm) // 32

    def segments(self):
        end = self.pushed_ms + 1000
        if self.switch_ms is None:
            return [{"start_ms": 0, "end_ms": end, "speaker": 1}]
        return [
            {"start_ms": 0, "end_ms": self.switch_ms, "speaker": 1},
            {"start_ms": self.switch_ms, "end_ms": end, "speaker": 2},
        ]

    finish = segments

    def close(self):
        pass


class LateLabelDiarizer(ScriptedDiarizer):
    """No speaker before `labelled_from_ms`, then `speaker` throughout (#230)."""

    def __init__(self, labelled_from_ms, speaker):
        super().__init__()
        self.labelled_from_ms, self.speaker = labelled_from_ms, speaker

    def segments(self):
        end = self.pushed_ms + 1000
        return [{"start_ms": self.labelled_from_ms, "end_ms": end, "speaker": self.speaker}]

    finish = segments


class BoundShadow:
    """A stand-in live identifier with fixed bindings: label -> (state, identity)."""

    def __init__(self, labels):
        self.labels = labels
        self.inferred = []

    def start(self):
        pass

    def audio(self, pcm):
        pass

    def turn(self, current, *, inferred=False, spans=None):
        if inferred:
            self.inferred.append(current.utterance_id)

    def binding(self, label):
        return self.labels.get(label)

    def close(self, drain=False):
        pass


def audio(*parts):
    """Synthetic PCM from (frame, milliseconds) pairs."""
    return b"".join(frame * (ms // 20) for frame, ms in parts)


# "Haili, what time is" (0-200 ms), a 1,500 ms pause, then "it?" (1,700-1,900 ms). The
# pause exceeds the 1,440 ms hangover, so the live window finalizes two utterances.
SPLIT_QUESTION = audio((VOICE, 200), (SILENCE, 1500), (VOICE, 200), (SILENCE, 3000))
TEXTS = ("Hailey, what time is", "it?")


class LiveProcessorMergeTests(unittest.TestCase):
    def run_processor(self, pcm, *, gap, switch_ms=None, finish=True, diarizer=None):
        turns = []
        config = LiveConfig(
            "merge-test",
            provenance="causal-replay",
            transcriber=ScriptedTranscriber(TEXTS),
            diarizer=diarizer or ScriptedDiarizer(switch_ms),
            turn_merge_gap_ms=gap,
        )
        processor = LiveProcessor(config, turns.append)
        self.addCleanup(processor.close)
        for offset in range(0, len(pcm), FRAME_BYTES * 10):
            processor.push_pcm16(pcm[offset : offset + FRAME_BYTES * 10])
        if finish:
            processor.finish()
        return processor, turns

    def test_split_question_becomes_one_turn_with_coherent_ids_and_times(self):
        _, turns = self.run_processor(SPLIT_QUESTION, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        # Released by stream time alone, before end of input: 1,900 + 2,000 ms.
        self.assertEqual(len(turns), 1)
        turn = turns[0]
        self.assertEqual(turn.text, "Hailey, what time is it?")
        self.assertEqual((turn.start_ms, turn.end_ms), (0, 1900))
        self.assertEqual((turn.utterance_id, turn.revision), ("live-1", 1))
        self.assertEqual(turn.speaker_id, "Speaker A")
        self.assertEqual(turn.speaker_provenance, "diarization-timeline")
        self.assertEqual(turn.recognizer_id, "scripted-fixture")

    def test_library_default_keeps_every_finalized_turn_separate(self):
        self.assertEqual(
            LiveConfig(
                "x", transcriber=ScriptedTranscriber(()), diarizer=ScriptedDiarizer()
            ).turn_merge_gap_ms,
            0,
        )
        _, turns = self.run_processor(SPLIT_QUESTION, gap=0)
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([t.utterance_id for t in turns], ["live-1", "live-2"])

    def test_gap_above_threshold_is_not_merged(self):
        _, turns = self.run_processor(SPLIT_QUESTION, gap=1000)
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([(t.start_ms, t.end_ms) for t in turns], [(0, 200), (1700, 1900)])

    def test_different_speakers_are_not_merged(self):
        _, turns = self.run_processor(SPLIT_QUESTION, gap=DEFAULT_TURN_MERGE_GAP_MS, switch_ms=1000)
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([t.speaker_id for t in turns], ["Speaker A", "Speaker B"])

    def test_utterance_local_diarizer_turns_are_never_joined(self):
        diarizer = ScriptedDiarizer()
        diarizer.speaker_provenance = "diarization-utterance"
        _, turns = self.run_processor(
            SPLIT_QUESTION, gap=DEFAULT_TURN_MERGE_GAP_MS, diarizer=diarizer
        )
        self.assertEqual([t.text for t in turns], list(TEXTS))
        self.assertEqual([t.speaker_id for t in turns], ["u1 Speaker A", "u2 Speaker A"])

    def test_a_suppressed_utterance_ends_the_held_turn_instead_of_being_bridged(self):
        # #78: B is suppressed for an invalid timestamp. Joining A and C across it would
        # publish "Do stop." for "Do not stop.", so A must be released on its own.
        replies = [
            lambda start, end: [{"text": " Do", "start_ms": start, "end_ms": end}],
            lambda start, end: [{"text": " not", "start_ms": -5, "end_ms": end}],
            lambda start, end: [{"text": " stop.", "start_ms": start, "end_ms": end}],
        ]

        class Transcriber(ScriptedTranscriber):
            def transcribe(self, pcm, register=None):
                voiced = [
                    index
                    for index in range(len(pcm) // FRAME_BYTES)
                    if pcm[index * FRAME_BYTES : (index + 1) * FRAME_BYTES] == VOICE
                ]
                return replies.pop(0)(voiced[0] * 20, (voiced[-1] + 1) * 20)

        turns = []
        processor = LiveProcessor(
            LiveConfig(
                "merge-suppressed",
                provenance="causal-replay",
                transcriber=Transcriber(()),
                diarizer=ScriptedDiarizer(),
                turn_merge_gap_ms=5000,
            ),
            turns.append,
        )
        self.addCleanup(processor.close)
        pcm = audio(
            (VOICE, 200),
            (SILENCE, 1500),
            (VOICE, 200),
            (SILENCE, 1500),
            (VOICE, 200),
            (SILENCE, 1500),
        )
        for offset in range(0, len(pcm), FRAME_BYTES * 10):
            processor.push_pcm16(pcm[offset : offset + FRAME_BYTES * 10])
        processor.finish()
        self.assertEqual(replies, [])
        self.assertEqual(processor.skipped_utterances, 1)
        self.assertEqual([t.text for t in turns], ["Do", "stop."])
        self.assertEqual([t.utterance_id for t in turns], ["live-1", "live-2"])

    def test_held_turn_waits_only_for_its_gap(self):
        pcm = audio((VOICE, 200), (SILENCE, 1440))
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        # Finalized at 1,640 ms and held until 200 + 2,000 ms of stream time.
        self.assertEqual(turns, [])
        for _ in range(27):
            processor.push_pcm16(SILENCE)
        self.assertEqual(turns, [])
        processor.push_pcm16(SILENCE)
        self.assertEqual([t.text for t in turns], ["Hailey, what time is"])

    def test_release_finish_and_close(self):
        pcm = audio((VOICE, 200), (SILENCE, 1440))
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        processor.release_pending()
        self.assertEqual(len(turns), 1)
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS)
        self.assertEqual(len(turns), 1)
        processor, turns = self.run_processor(pcm, gap=DEFAULT_TURN_MERGE_GAP_MS, finish=False)
        processor.close()
        processor.release_pending()
        self.assertEqual(turns, [])

    def test_gap_is_validated_by_live_config(self):
        for invalid in (-20, 5001, 700.0, None):
            with self.subTest(invalid=invalid), self.assertRaises(LiveAudioError):
                LiveConfig(
                    "x",
                    transcriber=ScriptedTranscriber(()),
                    diarizer=ScriptedDiarizer(),
                    turn_merge_gap_ms=invalid,
                )


class CountingProvider:
    """The fixture rule, counting decisions; stands in for opted-in Jev."""

    def __init__(self, built, **options):
        self.requests = 0
        self.states = []
        built.append(self)

    def decide(self, state):
        self.requests += 1
        self.states.append(state)
        return MockProvider().decide(state)


class DecisionPathTests(unittest.TestCase):
    """`listen` over a demo replay: the merged turn reaches the decision exactly once."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        asset = self.root / "asset"
        asset.touch()
        self.demo = self.root / "authored.wav"
        with wave.open(str(self.demo), "wb") as file:
            file.setnchannels(1)
            file.setsampwidth(2)
            file.setframerate(16000)
            file.writeframes(SPLIT_QUESTION)
        self.base = {
            key: str(asset)
            for key in (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
        }
        self.base["demo_audio"] = str(self.demo)
        self.base["addressing"] = {"names": ["Haili"], "variants": {"Haili": ["Hailey"]}}
        self.config = self.root / "config.json"

    def run_listen(
        self,
        texts=TEXTS,
        pcm=None,
        switch_ms=None,
        diarizer=None,
        shadow=None,
        captured=None,
        **extra,
    ):
        if pcm is not None:
            with wave.open(str(self.demo), "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(16000)
                file.writeframes(pcm)
        self.config.write_text(json.dumps({**self.base, **extra}))
        built = []

        def processor(config, callback):
            if captured is not None:
                captured.append(config)
            backends = {
                "transcriber": ScriptedTranscriber(texts),
                "diarizer": diarizer or ScriptedDiarizer(switch_ms),
            }
            return LiveProcessor(replace(config, **backends), callback)

        def factory(config, *, event_publisher, report=None):
            return PrototypeController(
                config,
                event_publisher=event_publisher,
                processor_factory=processor,
                provider_factory=lambda **options: CountingProvider(built, **options),
                **(
                    {}
                    if shadow is None
                    else {"speaker_id_factory": lambda settings, *, report: shadow}
                ),
            )

        args = Namespace(
            config=self.config,
            mode="demo",
            session_id="merge-decision",
            use_jev=True,
            allow_hosted=True,
            names=None,
        )
        output = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            code = listen(args, output=output, controller_factory=factory)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(code, 0)
        return events, built[0]

    @staticmethod
    def of(events, kind):
        return [event for event in events if event["type"] == kind]

    def test_split_question_is_decided_once_on_the_merged_turn(self):
        events, provider = self.run_listen()
        self.assertEqual(provider.requests, 1)
        self.assertEqual(provider.states[0]["current_turn"]["text"], "Hailey, what time is it?")
        transcripts = self.of(events, "transcript")
        self.assertEqual([e["turn"]["text"] for e in transcripts], ["Hailey, what time is it?"])
        requests = self.of(events, "request")
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            requests[0]["turn"]["utterance_id"], transcripts[0]["turn"]["utterance_id"]
        )
        self.assertEqual(
            events[0]["addressing"], {"names": ["Haili"], "variants": {"Haili": ["Hailey"]}}
        )

    def test_without_merging_the_same_speech_takes_two_decisions(self):
        events, provider = self.run_listen(turns={"merge_gap_ms": 0})
        self.assertEqual(provider.requests, 2)
        self.assertEqual([e["turn"]["text"] for e in self.of(events, "transcript")], list(TEXTS))

    def test_owner_stop_after_a_pause_still_supersedes_the_request(self):
        # A participant (Speaker A) asks; the owner (Speaker B) says "Wait.", pauses for
        # 1.5 s, then "never mind". Joined, "Wait. never mind" would not be a stop phrase.
        pcm = audio(
            (VOICE, 200),
            (SILENCE, 1500),
            (VOICE, 200),
            (SILENCE, 1500),
            (VOICE, 200),
            (SILENCE, 3000),
        )
        events, provider = self.run_listen(
            texts=("Haili, order a pizza", "Wait.", "never mind"),
            pcm=pcm,
            switch_ms=1000,
            speakers={"owner": ["Speaker B"], "trusted": [], "owner_only": False},
        )
        transcripts = [e["turn"] for e in self.of(events, "transcript")]
        self.assertEqual(
            [(t["text"], t["speaker_id"]) for t in transcripts],
            [
                ("Haili, order a pizza", "Speaker A"),
                ("Wait.", "Speaker B"),
                ("never mind", "Speaker B"),
            ],
        )
        self.assertEqual(provider.requests, 3)
        (request,) = self.of(events, "request")
        self.assertEqual(request["turn"]["utterance_id"], transcripts[0]["utterance_id"])
        (override,) = self.of(events, "override")
        self.assertEqual(override["superseded_request_id"], request["request_id"])
        self.assertEqual(override["by_utterance_id"], transcripts[2]["utterance_id"])

    def test_without_the_stop_break_the_pause_would_hide_the_stop(self):
        # Control for the test above: an ordinary second fragment does join.
        pcm = audio((VOICE, 200), (SILENCE, 1500), (VOICE, 200), (SILENCE, 3000))
        events, _ = self.run_listen(texts=("Wait.", "never mind me"), pcm=pcm)
        texts = [e["turn"]["text"] for e in self.of(events, "transcript")]
        self.assertEqual(texts, ["Wait. never mind me"])

    def run_wake_word(self, owner, **extra):
        # "Haili," (0-200 ms) with no speaker label, a 1,500 ms pause, then "are you
        # there?" (1,700-1,900 ms) labelled Speaker A by the timeline from 1,000 ms on.
        return self.run_listen(
            texts=("Haili,", "are you there?"),
            diarizer=LateLabelDiarizer(1000, 1),
            speakers={"owner": [owner], "trusted": [], "owner_only": False},
            **extra,
        )

    def run_identified(self, binding):
        """The same speech with live identification roles (#137): Speaker A's binding is
        `binding`, and the configured owner is the enrolled identity `owner`."""
        shadow = BoundShadow({} if binding is None else {"Speaker A": binding})
        identification = {
            "python": self.base["whisper_executable"],
            "model": self.base["whisper_model"],
            "live": True,
            "roles": True,
        }
        events, provider = self.run_wake_word(
            owner="owner", shadow=shadow, speaker_id=identification
        )
        return [e["turn"] for e in self.of(events, "transcript")], events, shadow

    def test_live_identification_joins_only_an_owner_binding(self):
        transcripts, events, shadow = self.run_identified(("bound", "owner"))
        self.assertEqual(
            [(t["text"], t["speaker_id"], t["role"]) for t in transcripts],
            [("Haili, are you there?", "Speaker A", "owner")],
        )
        (request,) = self.of(events, "request")
        self.assertEqual(request["turn"]["text"], "Haili, are you there?")
        # The joined turn reaches identification as inferred (#137).
        self.assertEqual(shadow.inferred, [transcripts[0]["utterance_id"]])
        for binding, role in (
            (("bound", "guest"), "participant"),
            (("unknown", None), "participant"),
            (("tentative", "owner"), "unknown"),
            (None, "unknown"),
        ):
            with self.subTest(binding=binding):
                transcripts, events, _ = self.run_identified(binding)
                self.assertEqual(
                    [(t["text"], t["speaker_id"], t["role"]) for t in transcripts],
                    [("Haili,", None, "unknown"), ("are you there?", "Speaker A", role)],
                )
                requests = self.of(events, "request")
                self.assertFalse([r for r in requests if r["turn"].get("role") == "owner"])

    def test_the_owner_check_leaves_the_name_binding_alone(self):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        roles.source = BoundShadow({"Speaker A": ("bound", "owner")})
        controller = PrototypeController(
            PrototypeConfig.load(self._write(speakers={"owner": ["owner"]}))
        )
        self.addCleanup(controller.close)
        controller._enrolled_roles = roles
        wake_word, owner = controller._wake_join()
        roles._resolved = sentinel = ("live-1", "Speaker B", ("bound", "guest"))
        self.assertTrue(owner(fragment("are you there?", 0, 100)))
        self.assertIs(roles._resolved, sentinel)
        self.assertFalse(owner(fragment("are you there?", 0, 100, overlap=True)))
        other = fragment("are you there?", 0, 100)
        other["speaker_provenance"] = "diarization-utterance"
        self.assertFalse(owner(other))
        self.assertTrue(wake_word("Hailey."))
        self.assertFalse(wake_word("hey Haili"))

    def _write(self, **extra):
        self.config.write_text(json.dumps({**self.base, **extra}))
        return self.config

    def test_wake_join_can_be_turned_off_alone(self):
        events, _ = self.run_wake_word(owner="Speaker A", turns={"wake_join": False})
        texts = [e["turn"]["text"] for e in self.of(events, "transcript")]
        self.assertEqual(texts, ["Haili,", "are you there?"])
        self.assertTrue(PrototypeConfig.load(self._write()).turn_wake_join)
        self.assertFalse(
            PrototypeConfig.load(self._write(turns={"wake_join": False})).turn_wake_join
        )
        for invalid in ("false", 0, None):
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                PrototypeConfig.load(self._write(turns={"wake_join": invalid}))

    def test_a_lone_wake_word_and_the_owners_sentence_are_one_owner_request(self):
        events, provider = self.run_wake_word(owner="Speaker A")
        transcripts = [e["turn"] for e in self.of(events, "transcript")]
        self.assertEqual(
            [(t["text"], t["speaker_id"], t.get("role")) for t in transcripts],
            [("Haili, are you there?", "Speaker A", "owner")],
        )
        self.assertEqual(provider.requests, 1)
        (request,) = self.of(events, "request")
        self.assertEqual(request["turn"]["utterance_id"], transcripts[0]["utterance_id"])
        self.assertEqual(request["turn"]["text"], "Haili, are you there?")

    def test_a_non_owner_continuation_forms_no_owner_request(self):
        # Speaker A is a participant here: the name is never joined to their sentence.
        events, _ = self.run_wake_word(owner="Speaker B")
        transcripts = [e["turn"] for e in self.of(events, "transcript")]
        self.assertEqual(
            [(t["text"], t["speaker_id"], t.get("role")) for t in transcripts],
            [("Haili,", None, "unknown"), ("are you there?", "Speaker A", "participant")],
        )
        requests = self.of(events, "request")
        self.assertFalse([r for r in requests if r["turn"].get("role") == "owner"])
        self.assertFalse([r for r in requests if r["turn"]["text"] == "are you there?"])

    def run_owner_pause(self, owner="Speaker A", pause=3000, **extra):
        # "Hailey, what time is" then "it?" by Speaker A after a 1,500 ms pause: longer
        # than a 1,000 ms merge gap, shorter than the owner pause (#735).
        events, provider = self.run_listen(
            speakers={"owner": [owner], "trusted": [], "owner_only": False},
            turns={"merge_gap_ms": 1000, "owner_pause_ms": pause},
            **extra,
        )
        return [e["turn"] for e in self.of(events, "transcript")], events, provider

    def test_the_owner_pause_joins_only_the_owners_split_turn(self):
        transcripts, events, provider = self.run_owner_pause()
        self.assertEqual(
            [(t["text"], t["speaker_id"], t["role"]) for t in transcripts],
            [("Hailey, what time is it?", "Speaker A", "owner")],
        )
        self.assertEqual(provider.requests, 1)
        (request,) = self.of(events, "request")
        self.assertEqual(request["turn"]["text"], "Hailey, what time is it?")
        # Off, or a participant speaking: the merge gap splits it as before.
        for owner, pause in (("Speaker A", 0), ("Speaker B", 3000)):
            with self.subTest(owner=owner, pause=pause):
                transcripts, _, _ = self.run_owner_pause(owner=owner, pause=pause)
                self.assertEqual([t["text"] for t in transcripts], list(TEXTS))

    def run_owner_pause_identified(self, binding):
        """The same speech with live identification roles (#137); the owner is the
        enrolled identity `owner`, and Speaker A's binding is `binding`."""
        identification = {
            "python": self.base["whisper_executable"],
            "model": self.base["whisper_model"],
            "live": True,
            "roles": True,
        }
        shadow = BoundShadow({} if binding is None else {"Speaker A": binding})
        transcripts, _, _ = self.run_owner_pause(
            owner="owner", shadow=shadow, speaker_id=identification
        )
        return transcripts

    def test_live_identification_gives_the_pause_only_to_an_owner_binding(self):
        transcripts = self.run_owner_pause_identified(("bound", "owner"))
        self.assertEqual(
            [(t["text"], t["speaker_id"], t["role"]) for t in transcripts],
            [("Hailey, what time is it?", "Speaker A", "owner")],
        )
        # A joined owner turn is not inferred: same label, same speaker, owner authority.
        self.assertNotIn("inferred", transcripts[0])
        for binding, role in (
            (("bound", "guest"), "participant"),
            (("unknown", None), "participant"),
            (("tentative", "owner"), "unknown"),
            (None, "unknown"),
        ):
            with self.subTest(binding=binding):
                transcripts = self.run_owner_pause_identified(binding)
                self.assertEqual(
                    [(t["text"], t["role"]) for t in transcripts],
                    [(TEXTS[0], role), (TEXTS[1], role)],
                )

    def test_the_owner_predicate_reaches_the_merger_only_when_used(self):
        speakers = {"owner": ["Speaker A"], "trusted": [], "owner_only": False}
        for turns, has_owner, pause in (
            # Main's wiring at 0: the predicate only with the wake join.
            ({"owner_pause_ms": 0}, True, 0),
            ({"owner_pause_ms": 0, "wake_join": False}, False, 0),
            ({"owner_pause_ms": 2500, "wake_join": False}, True, 2500),
        ):
            with self.subTest(turns=turns):
                captured = []
                self.run_listen(captured=captured, speakers=speakers, turns=turns)
                (config,) = captured
                self.assertEqual(config.turn_owner is not None, has_owner)
                self.assertEqual(config.turn_owner_pause_ms, pause)

    def test_owner_pause_config_is_validated(self):
        self.assertEqual(PrototypeConfig.load(self._write()).turn_owner_pause_ms, 0)
        for value in (0, 4000, 10000):
            with self.subTest(value=value):
                config = PrototypeConfig.load(self._write(turns={"owner_pause_ms": value}))
                self.assertEqual(config.turn_owner_pause_ms, value)
        for invalid in (-1, 10001, 4000.0, 1.5, True, "4000", None):
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                PrototypeConfig.load(self._write(turns={"owner_pause_ms": invalid}))
        for invalid in (-1, 10001, 1.5, False, "4000"):
            with self.subTest(live=invalid), self.assertRaises(LiveAudioError):
                LiveConfig("s", transcriber=object, diarizer=object, turn_owner_pause_ms=invalid)
        self.assertEqual(
            LiveConfig("s", transcriber=object, diarizer=object).turn_owner_pause_ms, 0
        )

    def test_config_section_is_validated(self):
        self.config.write_text(json.dumps(self.base))
        self.assertEqual(PrototypeConfig.load(self.config).turn_merge_gap_ms, 2000)
        self.config.write_text(json.dumps({**self.base, "turns": {"merge_gap_ms": 800}}))
        self.assertEqual(PrototypeConfig.load(self.config).turn_merge_gap_ms, 800)
        for invalid in (
            {"turns": {"merge_gap_ms": -1}},
            {"turns": {"merge_gap_ms": 5001}},
            {"turns": {"merge_gap_ms": "800"}},
            {"turns": {"merge_gap_ms": True}},
            {"turns": {"gap": 800}},
            {"turns": None},
            {"turns": [800]},
        ):
            self.config.write_text(json.dumps({**self.base, **invalid}))
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                PrototypeConfig.load(self.config)


if __name__ == "__main__":
    unittest.main()
