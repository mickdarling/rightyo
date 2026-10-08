"""Tail join (#129): an unlabelled tail joins the held labelled turn it trails.

Synthetic fragments, synthetic PCM, a scripted transcriber and a scripted diarizer
timeline only: no capture, native inference, hosted calls or recorded speech.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_edge_attribution import SPEECH, Timeline, Units
from test_turn_merge import fragment

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.turn_merge import TurnMerger


def tail(text, start, end, overlap=False):
    return fragment(text, start, end, speaker=None, overlap=overlap)


class TurnMergerTailJoinTests(unittest.TestCase):
    def setUp(self):
        self.emitted = []

    def merger(self, gap=2000, tail_ms=400, span=24000, **options):
        return TurnMerger(gap, span, self.emitted.append, tail_join_ms=tail_ms, **options)

    def texts(self):
        return [item["text"] for item in self.emitted]

    def test_an_unlabelled_tail_within_the_window_joins_the_held_speaker(self):
        merger = self.merger()
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("down a bit.", 900, 1100))
        self.assertEqual(self.emitted, [])
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume down a bit."])
        joined = self.emitted[0]
        self.assertEqual((joined["start_ms"], joined["end_ms"]), (0, 1100))
        self.assertEqual(joined["speaker"], "Speaker A")
        self.assertEqual(joined["speaker_provenance"], "diarization-timeline")
        self.assertFalse(joined["overlap"])

    def test_the_window_edge_is_inclusive_and_beyond_it_nothing_joins(self):
        merger = self.merger()
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("down.", 1000, 1100))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume down."])
        self.emitted.clear()
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("Later.", 1001, 1100))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "Later."])
        self.assertEqual([item["speaker"] for item in self.emitted], ["Speaker A", None])

    def test_off_by_default_the_tail_stays_its_own_turn(self):
        merger = TurnMerger(2000, 24000, self.emitted.append)
        self.assertEqual(merger.tail_join_ms, 0)
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("down a bit.", 700, 800))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down a bit."])

    def test_nothing_joins_into_an_unlabelled_held_fragment(self):
        # A request-shaped unlabelled fragment is held for the reply wait, but neither an
        # unlabelled nor a labelled fragment is joined into it.
        merger = self.merger(reply_wait_ms=1200)
        merger.offer(tail("Can you turn it", 0, 600))
        self.assertTrue(merger.holding)
        merger.offer(tail("down?", 700, 800))
        merger.flush()
        self.assertEqual(self.texts(), ["Can you turn it", "down?"])
        self.emitted.clear()
        merger.offer(tail("Can you turn it", 0, 600))
        merger.offer(fragment("down?", 700, 800))
        merger.flush()
        self.assertEqual(self.texts(), ["Can you turn it", "down?"])
        self.assertEqual([item["speaker"] for item in self.emitted], [None, "Speaker A"])

    def test_overlap_is_never_joined_in_either_direction(self):
        merger = self.merger()
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("down.", 700, 800, overlap=True))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down."])
        self.emitted.clear()
        merger.offer(fragment("Turn the volume", 0, 600, overlap=True))
        merger.offer(tail("down.", 700, 800))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down."])

    def test_another_speaker_or_provenance_is_never_joined_by_the_tail_window(self):
        merger = self.merger(gap=0)
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(fragment("down.", 700, 800, speaker="Speaker B"))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down."])
        self.emitted.clear()
        # With merging off, the same speaker is not joined through the tail window either.
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(fragment("down.", 700, 800))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down."])
        self.emitted.clear()
        merger.offer(fragment("Turn the volume", 0, 600))
        other = tail("down.", 700, 800)
        other["speaker_provenance"] = "unknown"
        merger.offer(other)
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down."])

    def test_with_merging_off_a_labelled_turn_is_held_for_its_tail(self):
        merger = self.merger(gap=0)
        merger.offer(fragment("Turn the volume", 0, 600))
        self.assertTrue(merger.holding)
        merger.due(999, None)
        self.assertEqual(self.emitted, [])
        merger.offer(tail("down.", 900, 1000))
        merger.due(1399, None)
        self.assertEqual(self.emitted, [])
        merger.due(1400, None)
        self.assertEqual(self.texts(), ["Turn the volume down."])

    def test_a_stop_phrase_tail_is_still_emitted_alone(self):
        merger = self.merger(breaks_turn=SpeakerPriority().is_stop_phrase)
        merger.offer(fragment("Haili, order a pizza", 0, 900))
        merger.offer(tail("never mind", 1000, 1300))
        self.assertEqual(self.texts(), ["Haili, order a pizza", "never mind"])
        self.assertEqual([item["speaker"] for item in self.emitted], ["Speaker A", None])
        self.assertFalse(merger.holding)

    def test_a_tail_that_completes_a_split_stop_phrase_is_emitted_at_once(self):
        merger = self.merger(breaks_turn=SpeakerPriority().is_stop_phrase)
        merger.offer(fragment("never", 0, 300))
        merger.offer(tail("mind", 400, 600))
        self.assertEqual(self.texts(), ["never mind"])
        self.assertFalse(merger.holding)

    def test_a_complete_tail_flushes_the_joined_turn(self):
        merger = self.merger()
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("down a bit.", 700, 900), complete=True)
        self.assertEqual(self.texts(), ["Turn the volume down a bit."])
        self.assertFalse(merger.holding)

    def test_span_and_text_caps_still_apply(self):
        merger = self.merger(span=1000)
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(tail("down a bit.", 900, 1100))
        merger.flush()
        self.assertEqual(self.texts(), ["Turn the volume", "down a bit."])
        self.emitted.clear()
        merger = self.merger()
        merger.offer(fragment("x" * 3000, 0, 600))
        merger.offer(tail("y" * 1000, 700, 800))
        merger.flush()
        self.assertEqual([len(text) for text in self.texts()], [3000, 1000])

    def test_the_window_is_validated(self):
        for invalid in (-1, 2001, 1.5, True, "400", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TurnMerger(2000, 24000, self.emitted.append, tail_join_ms=invalid)


def run(tail_ms, edges=300, gap=2000):
    """One utterance: Speaker A until 500 ms, then a trailing piece the timeline misses."""
    turns = []
    units = Units((" Turn the volume", 0, 500), (" down a bit.", 850, 1000))
    config = LiveConfig(
        "tail-test",
        provenance="causal-replay",
        transcriber=units,
        diarizer=Timeline((0, 500, 1)),
        turn_merge_gap_ms=gap,
        edge_attribution_ms=edges,
        tail_join_ms=tail_ms,
    )
    processor = LiveProcessor(config, turns.append)
    for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
        processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
    processor.finish()
    return [(turn.text, turn.speaker_id, turn.speaker_provenance) for turn in turns]


class LiveTailJoinTests(unittest.TestCase):
    def test_edge_attribution_alone_leaves_the_tail_as_a_second_turn(self):
        # The live failure: the tail is 350 ms past the segment, beyond a 300 ms slack.
        self.assertEqual(
            run(0),
            [
                ("Turn the volume", "Speaker A", "diarization-timeline"),
                ("down a bit.", None, "diarization-timeline"),
            ],
        )

    def test_a_labelled_group_and_its_unlabelled_tail_become_one_turn(self):
        expected = [("Turn the volume down a bit.", "Speaker A", "diarization-timeline")]
        self.assertEqual(run(400), expected)
        # Also without edge attribution and with merging off.
        self.assertEqual(run(400, edges=0, gap=0), expected)

    def test_a_tail_beyond_the_window_stays_its_own_turn(self):
        self.assertEqual(len(run(300)), 2)

    def test_live_config_validates_it(self):
        self.assertEqual(LiveConfig("x", transcriber=Units(), diarizer=Timeline()).tail_join_ms, 0)
        for value in (-1, 2001, 1.5, True, "400"):
            with self.subTest(value=value), self.assertRaises(LiveAudioError):
                LiveConfig("x", transcriber=Units(), diarizer=Timeline(), tail_join_ms=value)


ASSETS = (
    "whisper_executable",
    "whisper_model",
    "diarization_library",
    "diarization_model",
    "microphone_helper",
)


class PrototypeTailJoinTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.asset = Path(directory.name) / "asset"
        self.asset.touch()
        self.config = Path(directory.name) / "config.json"
        self.base = {name: str(self.asset) for name in ASSETS}

    def load(self, extra):
        self.config.write_text(json.dumps(self.base | extra))
        return PrototypeConfig.load(self.config)

    def test_the_turns_section_carries_it(self):
        self.assertEqual(self.load({}).tail_join_ms, 0)
        self.assertEqual(self.load({"turns": {"tail_join_ms": 400}}).tail_join_ms, 400)
        for invalid in (-1, 2001, 1.5, True, "400", None):
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                self.load({"turns": {"tail_join_ms": invalid}})

    def test_configured_roles_refuse_it(self):
        loaded = self.load(
            {
                "demo_audio": str(self.asset),
                "turns": {"tail_join_ms": 400},
                "speakers": {"owner": ["Speaker A"]},
            }
        )
        controller = PrototypeController(loaded)
        self.addCleanup(controller.close)
        with self.assertRaises(PrototypeError) as error:
            controller.start({"mode": "demo"})
        self.assertIn("Tail join", str(error.exception))
        self.assertIn("speaker roles", str(error.exception))


if __name__ == "__main__":
    unittest.main()
