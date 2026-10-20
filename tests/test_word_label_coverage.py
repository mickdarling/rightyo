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

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor, _attribute
from rightyo.prototype import PrototypeConfig, PrototypeError

OWNER, OTHER = 1, 2
ROLES = SpeakerPriority.from_dict({"owner": ["Speaker A"]})
WAKE = (" Jarvis.", 0, 1000)


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
        self.assertEqual(self.load().word_label_coverage, 0.5)
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
