"""Coverage labelling: a one-word turn whose timestamps run past its speaker's timeline
segment keeps that speaker (rightyo#220).

Synthetic PCM, a scripted transcriber and a scripted diarizer timeline only: no capture,
native inference, hosted calls or recorded speech.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_edge_attribution import SPEECH, WORDS, Timeline, Units
from test_turn_merge import SILENCE, VOICE, audio

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor, _attribute
from rightyo.prototype import PrototypeConfig, PrototypeError

OWNER, OTHER = 1, 2
ROLES = SpeakerPriority.from_dict({"owner": ["Speaker A"]})
WAKE = (" Jarvis.", 0, 1000)
# A second of silence, then speech: the utterance opens one pre-roll (240 ms) early.
LEAD = audio((SILENCE, 1000), (VOICE, 400), (SILENCE, 2000))
START = 760


def run(units, timeline, coverage):
    turns = []
    config = LiveConfig(
        "coverage-test",
        provenance="causal-replay",
        transcriber=units,
        diarizer=timeline,
        word_label_coverage=coverage,
    )
    processor = LiveProcessor(config, turns.append)
    for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
        processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
    processor.finish()
    return [(turn.text, turn.speaker_id, turn.overlap) for turn in turns]


class AttributeTests(unittest.TestCase):
    def test_strict_by_default(self):
        timeline = [{"start_ms": 150, "end_ms": 850, "speaker": OWNER}]
        self.assertEqual(_attribute(0, 1000, timeline), (None, False))
        self.assertEqual(_attribute(0, 1000, timeline, 0.5), ("Speaker A", False))
        self.assertEqual(_attribute(0, 1000, timeline, 0.8), (None, False))

    def test_two_speakers_or_none_never_label(self):
        two = [
            {"start_ms": 0, "end_ms": 400, "speaker": OWNER},
            {"start_ms": 600, "end_ms": 1000, "speaker": OTHER},
        ]
        self.assertEqual(_attribute(0, 1000, two, 0.5), (None, False))
        self.assertEqual(_attribute(0, 1000, [], 0.5), (None, False))
        self.assertEqual(_attribute(500, 500, two, 0.5), (None, False))


class OneWordTurnTests(unittest.TestCase):
    def test_an_owner_wake_word_with_edge_slack_is_labelled_and_owner(self):
        timeline = Timeline((150, 850, OWNER))
        self.assertEqual(run(Units(WAKE), timeline, 0), [("Jarvis.", None, False)])
        turns = run(Units(WAKE), timeline, 0.5)
        self.assertEqual(turns, [("Jarvis.", "Speaker A", False)])
        self.assertEqual(ROLES.configured_role(turns[0][1]), "owner")

    def test_a_wake_word_on_another_track_stays_non_owner(self):
        turns = run(Units(WAKE), Timeline((150, 850, OTHER)), 0.5)
        self.assertEqual(turns, [("Jarvis.", "Speaker B", False)])
        self.assertNotEqual(ROLES.configured_role(turns[0][1]), "owner")

    def test_two_overlapping_speakers_stay_unlabelled(self):
        turns = run(Units(WAKE), Timeline((100, 900, OWNER), (300, 700, OTHER)), 0.5)
        self.assertEqual(turns, [("Jarvis.", None, True)])

    def test_too_little_coverage_stays_unlabelled(self):
        self.assertEqual(
            run(Units(WAKE), Timeline((0, 300, OWNER)), 0.5), [("Jarvis.", None, False)]
        )

    def test_utterance_level_diarization_is_unchanged(self):
        timeline = Timeline((150, 850, OWNER), provenance="diarization-utterance")
        self.assertEqual(run(Units(WAKE), timeline, 0.5)[0][1], None)


class CarriedOverSegmentTests(unittest.TestCase):
    """A late speaker switch must not hand the next person's word to the earlier speaker."""

    def run_lead(self, units, timeline, *, edges=0, coverage=0.5):
        turns, inferred = [], []
        config = LiveConfig(
            "coverage-lead-test",
            provenance="causal-replay",
            transcriber=units,
            diarizer=timeline,
            word_label_coverage=coverage,
            edge_attribution_ms=edges,
            on_inferred=inferred.append,
        )
        processor = LiveProcessor(config, turns.append)
        for offset in range(0, len(LEAD), FRAME_BYTES * 10):
            processor.push_pcm16(LEAD[offset : offset + FRAME_BYTES * 10])
        processor.finish()
        self.assertEqual(turns[0].start_ms - units.units[0]["start_ms"], START)
        return [(turn.text, turn.speaker_id) for turn in turns], [
            turn.utterance_id in inferred for turn in turns
        ]

    def test_an_earlier_speakers_segment_running_on_does_not_label_a_guest_word(self):
        # The owner's segment runs to 1,250 ms; the guest's word is 1,050-1,350 ms and
        # their own segment has not arrived. 2/3 of the word is under the owner's segment.
        word = Units((" Jarvis.", 1050 - START, 1350 - START))
        turns, _ = self.run_lead(word, Timeline((0, 1250, OWNER)))
        self.assertEqual(turns, [("Jarvis.", None)])

    def test_a_carried_over_guest_segment_blocks_a_fresh_owner_label(self):
        word = Units((" Jarvis.", 1050 - START, 1350 - START))
        turns, _ = self.run_lead(word, Timeline((0, 1120, OTHER), (1130, 1400, OWNER)))
        self.assertEqual(turns, [("Jarvis.", None)])

    def test_a_carried_over_owner_segment_blocks_a_fresh_guest_label(self):
        word = Units((" Jarvis.", 1050 - START, 1350 - START))
        turns, _ = self.run_lead(word, Timeline((0, 1120, OWNER), (1130, 1400, OTHER)))
        self.assertEqual(turns, [("Jarvis.", None)])

    def test_a_carried_over_segment_of_the_same_speaker_does_not_block(self):
        word = Units((" Jarvis.", 1050 - START, 1350 - START))
        turns, _ = self.run_lead(word, Timeline((0, 1060, OWNER), (1130, 1400, OWNER)))
        self.assertEqual(turns, [("Jarvis.", "Speaker A")])

    def test_the_owners_own_word_in_the_utterance_is_still_labelled(self):
        word = Units((" Jarvis.", 1050 - START, 1350 - START))
        turns, inferred = self.run_lead(word, Timeline((1100, 1300, OWNER)))
        self.assertEqual(turns, [("Jarvis.", "Speaker A")])
        self.assertEqual(inferred, [False])

    def test_edge_attribution_from_a_coverage_label_stays_inferred(self):
        words = Units(
            (" Jarvis,", 1050 - START, 1250 - START), (" please.", 1300 - START, 1380 - START)
        )
        turns, inferred = self.run_lead(words, Timeline((1080, 1200, OWNER)), edges=200)
        self.assertEqual(turns, [("Jarvis, please.", "Speaker A")])
        self.assertEqual(inferred, [True])

    def test_two_speakers_without_strict_labels_stay_two_turns(self):
        words = Units(
            (" Jarvis.", 1000 - START, 1200 - START), (" Hello.", 1250 - START, 1390 - START)
        )
        timeline = Timeline((1020, 1180, OWNER), (1270, 1380, OTHER))
        turns, _ = self.run_lead(words, timeline)
        self.assertEqual(turns, [("Jarvis.", "Speaker A"), ("Hello.", "Speaker B")])


class MultiWordTurnTests(unittest.TestCase):
    def test_a_turn_with_a_strictly_labelled_word_is_unchanged(self):
        # "down" and "a bit." run past the segment and stay unlabelled, as before.
        timeline = Timeline((0, 650, OWNER))
        before = run(Units(*WORDS), timeline, 0)
        self.assertEqual(run(Units(*WORDS), timeline, 0.5), before)
        self.assertEqual(before[0], ("Turn the volume", "Speaker A", False))
        self.assertTrue(all(speaker is None for _, speaker, _ in before[1:]))

    def test_a_two_speaker_turn_is_unchanged(self):
        timeline = Timeline((0, 620, OWNER), (580, 1000, OTHER))
        self.assertEqual(run(Units(*WORDS), timeline, 0.5), run(Units(*WORDS), timeline, 0))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        keys = (
            "whisper_executable",
            "whisper_model",
            "diarization_library",
            "diarization_model",
            "microphone_helper",
        )
        self.base = {key: str(asset) for key in keys}
        self.path = Path(directory.name) / "config.json"

    def load(self, **turns):
        self.path.write_text(json.dumps({**self.base, "turns": turns}))
        return PrototypeConfig.load(self.path)

    def test_default_and_values(self):
        self.assertEqual(self.load().word_label_coverage, 0)
        self.assertEqual(self.load(word_label_coverage=0).word_label_coverage, 0)
        self.assertEqual(self.load(word_label_coverage=0.75).word_label_coverage, 0.75)
        for value in (-0.1, 1.5, True, "0.5", None):
            with self.subTest(value=value), self.assertRaises(PrototypeError):
                self.load(word_label_coverage=value)

    def test_library_default_is_off_and_validated(self):
        self.assertEqual(
            LiveConfig("x", transcriber=Units(), diarizer=Timeline()).word_label_coverage, 0
        )
        for value in (-1, 2, float("nan"), "0.5"):
            with self.subTest(value=value), self.assertRaises(LiveAudioError):
                LiveConfig("x", transcriber=Units(), diarizer=Timeline(), word_label_coverage=value)


if __name__ == "__main__":
    unittest.main()
