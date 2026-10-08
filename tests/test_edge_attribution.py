"""Edge attribution: a streaming diarizer's unlabelled edge words take the adjacent speaker.

Synthetic PCM, a scripted transcriber and a scripted diarizer timeline only: no capture,
native inference, hosted calls or recorded speech.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_turn_merge import SILENCE, VOICE, audio

from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor
from rightyo.prototype import PrototypeConfig, PrototypeError

# One utterance: a second of speech from stream time 0, then silence past the hangover.
SPEECH = audio((VOICE, 1000), (SILENCE, 2000))


class Units:
    """Fixed recognizer units, in milliseconds from the utterance start."""

    recognizer_id = "scripted-units"

    def __init__(self, *units):
        self.units = [{"text": text, "start_ms": s, "end_ms": e} for text, s, e in units]

    def transcribe(self, pcm, register=None):
        return list(self.units)


class Timeline:
    """A fixed diarizer timeline of (start_ms, end_ms, speaker) segments."""

    def __init__(self, *segments, provenance="diarization-timeline"):
        self.timeline = [{"start_ms": s, "end_ms": e, "speaker": n} for s, e, n in segments]
        self.speaker_provenance = provenance

    def push(self, pcm):
        pass

    def segments(self):
        return list(self.timeline)

    finish = segments

    def close(self):
        pass


WORDS = (
    (" Turn the volume", 0, 600),
    (" down", 600, 800),
    (" a bit.", 850, 1000),
)


def run(units, timeline, edges):
    turns = []
    config = LiveConfig(
        "edge-test",
        provenance="causal-replay",
        transcriber=units,
        diarizer=timeline,
        edge_attribution_ms=edges,
    )
    processor = LiveProcessor(config, turns.append)
    for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
        processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
    processor.finish()
    return [(turn.text, turn.speaker_id) for turn in turns]


class EdgeAttributionTests(unittest.TestCase):
    def test_off_by_default_the_trailing_words_stay_unlabelled(self):
        self.assertEqual(
            LiveConfig("x", transcriber=Units(), diarizer=Timeline()).edge_attribution_ms, 0
        )
        self.assertEqual(
            run(Units(*WORDS), Timeline((0, 700, 1)), 0),
            [("Turn the volume", "Speaker A"), ("down a bit.", None)],
        )

    def test_trailing_words_past_the_timeline_join_the_only_speaker(self):
        self.assertEqual(
            run(Units(*WORDS), Timeline((0, 700, 1)), 1000),
            [("Turn the volume down a bit.", "Speaker A")],
        )

    def test_leading_and_zero_length_words_join_too(self):
        units = Units((" So", 0, 0), (" turn it", 100, 500), (" down.", 900, 900))
        self.assertEqual(
            run(units, Timeline((80, 600, 1)), 1000), [("So turn it down.", "Speaker A")]
        )

    def test_words_too_far_from_the_labelled_span_stay_unlabelled(self):
        units = Units((" Turn it down.", 0, 400), (" Later.", 900, 1000))
        self.assertEqual(
            run(units, Timeline((0, 450, 1)), 300),
            [("Turn it down.", "Speaker A"), ("Later.", None)],
        )

    def test_another_speaker_touching_the_word_blocks_it(self):
        self.assertEqual(
            run(Units(*WORDS), Timeline((0, 700, 1), (950, 1000, 2)), 1000),
            # "down" touches only Speaker A and joins; "a bit." touches Speaker B and stays.
            [("Turn the volume down", "Speaker A"), ("a bit.", None)],
        )

    def test_two_labelled_speakers_in_the_utterance_infer_nothing(self):
        units = Units((" Hi", 0, 300), (" there", 350, 600), (" now.", 800, 1000))
        self.assertEqual(
            run(units, Timeline((0, 320, 1), (340, 620, 2)), 1000),
            [("Hi", "Speaker A"), ("there", "Speaker B"), ("now.", None)],
        )

    def test_overlapping_words_are_never_relabelled(self):
        units = Units((" Turn it", 0, 400), (" down.", 500, 900))
        timeline = Timeline((0, 900, 1), (450, 900, 2))
        # "down." overlaps two speakers: it stays unattributed and marked as overlap.
        result = run(units, timeline, 1000)
        self.assertEqual(result, [("Turn it", "Speaker A"), ("down.", None)])

    def test_utterance_level_diarizers_are_left_alone(self):
        timeline = Timeline((0, 700, 1), provenance="diarization-utterance")
        result = run(Units(*WORDS), timeline, 1000)
        self.assertEqual([speaker for _, speaker in result][1], None)

    def test_the_setting_is_validated(self):
        for value in (-1, 2001, 1.5, "100"):
            with self.subTest(value=value), self.assertRaises(LiveAudioError):
                LiveConfig("x", transcriber=Units(), diarizer=Timeline(), edge_attribution_ms=value)

    def test_the_prototype_turns_section_carries_it(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset"
            asset.touch()
            config = Path(directory) / "config.json"
            base = {
                name: str(asset)
                for name in (
                    "whisper_executable",
                    "whisper_model",
                    "diarization_library",
                    "diarization_model",
                    "microphone_helper",
                )
            }
            config.write_text(json.dumps(base))
            self.assertEqual(PrototypeConfig.load(config).edge_attribution_ms, 0)
            config.write_text(json.dumps(base | {"turns": {"edge_attribution_ms": 1000}}))
            self.assertEqual(PrototypeConfig.load(config).edge_attribution_ms, 1000)
            for invalid in (-1, 2001, True, "1000"):
                config.write_text(json.dumps(base | {"turns": {"edge_attribution_ms": invalid}}))
                with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                    PrototypeConfig.load(config)


if __name__ == "__main__":
    unittest.main()
