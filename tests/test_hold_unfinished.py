"""A turn that looks cut off mid-request is held even when called complete (rightyo#199)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_turn_merge import fragment

from rightyo.addressedness import unfinished_shaped
from rightyo.contracts import Addressing
from rightyo.live_audio import LiveAudioError, LiveConfig
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.turn_merge import TurnMerger

NAMES = Addressing.from_dict({"names": ["Jarvis"], "variants": {"Jarvis": ["Jervis"]}})


class UnfinishedShapeTests(unittest.TestCase):
    def test_a_dangling_last_word_a_bare_name_or_an_opener_is_unfinished(self):
        for text in (
            "Jarvis, add the",
            "Can you put this on the",
            "Jarvis, remind me to.",
            "and",
            "Jarvis,",
            "Hey Jervis.",
            "can you",
            "so um",
            "Jarvis, add it to my",
        ):
            with self.subTest(text=text):
                self.assertTrue(unfinished_shaped(text, NAMES))

    def test_a_finished_sentence_or_nothing_is_not(self):
        for text in (
            "Jarvis, add milk to the list.",
            "What time is it?",
            "Thanks.",
            "Stop",
            "",
            "Jarvis, stop talking",
            "thank you",
            "do it",
            "mute it",
            "got it",
            "turn the lights on",
            "switch it on",
            "log me in",
            "Jarvis, play this",
            "I think so",
            "okay thanks",
            "Okay.",
            "Thanks Jarvis",
            "Jarvis stop",
        ):
            with self.subTest(text=text):
                self.assertFalse(unfinished_shaped(text, NAMES))

    def test_a_bare_name_needs_the_configured_names(self):
        self.assertFalse(unfinished_shaped("Jarvis.", None))
        self.assertTrue(unfinished_shaped("Jarvis.", NAMES))


class HoldUnfinishedTests(unittest.TestCase):
    def merger(self, unfinished=lambda text: unfinished_shaped(text, NAMES)):
        emitted = []
        return TurnMerger(2000, 24000, emitted.append, unfinished=unfinished), emitted

    def test_an_unfinished_fragment_called_complete_waits_and_joins_its_continuation(self):
        merger, emitted = self.merger()
        merger.offer(fragment("Jarvis, add the", 0, 900), complete=True)
        self.assertEqual(emitted, [])
        merger.offer(fragment("oat milk to the list.", 1900, 2600), complete=True)
        self.assertEqual(
            [turn["text"] for turn in emitted], ["Jarvis, add the oat milk to the list."]
        )

    def test_a_finished_fragment_called_complete_is_still_emitted_at_once(self):
        merger, emitted = self.merger()
        merger.offer(fragment("Jarvis, what time is it?", 0, 900), complete=True)
        self.assertEqual([turn["text"] for turn in emitted], ["Jarvis, what time is it?"])

    def test_an_unfinished_fragment_with_no_continuation_is_released_after_the_gap(self):
        merger, emitted = self.merger()
        merger.offer(fragment("Jarvis,", 0, 500), complete=True)
        merger.due(2400, None)
        self.assertEqual(emitted, [])
        merger.due(2500, None)
        self.assertEqual([turn["text"] for turn in emitted], ["Jarvis,"])

    def test_without_the_predicate_complete_is_trusted_as_before(self):
        merger, emitted = self.merger(unfinished=None)
        merger.offer(fragment("Jarvis, add the", 0, 900), complete=True)
        self.assertEqual([turn["text"] for turn in emitted], ["Jarvis, add the"])

    def test_a_stop_phrase_is_never_held_even_when_it_looks_unfinished(self):
        emitted = []
        merger = TurnMerger(
            2000,
            24000,
            emitted.append,
            breaks_turn=lambda text: text.lower().strip(".") == "stop at",
            unfinished=lambda text: True,
        )
        merger.offer(fragment("Stop at.", 0, 500), complete=True)
        self.assertEqual([turn["text"] for turn in emitted], ["Stop at."])

    def test_an_unjoinable_unfinished_fragment_is_emitted_at_once(self):
        merger, emitted = self.merger()
        merger.offer(fragment("Jarvis, add the", 0, 900, speaker=None), complete=True)
        self.assertEqual([turn["text"] for turn in emitted], ["Jarvis, add the"])

    def test_a_joined_turn_still_unfinished_stays_held_for_a_third_piece(self):
        merger, emitted = self.merger()
        merger.offer(fragment("Jarvis,", 0, 500), complete=True)
        merger.offer(fragment("add the", 1500, 2000), complete=True)
        self.assertEqual(emitted, [])
        merger.offer(fragment("milk to my list.", 3000, 3800), complete=True)
        self.assertEqual([turn["text"] for turn in emitted], ["Jarvis, add the milk to my list."])

    def test_a_non_callable_predicate_is_refused(self):
        with self.assertRaises(ValueError):
            TurnMerger(2000, 24000, print, unfinished="yes")
        with self.assertRaises(LiveAudioError):
            LiveConfig("x", transcriber=object, diarizer=object, turn_unfinished="yes")


class HoldUnfinishedConfigTests(unittest.TestCase):
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

    def load(self, **extra):
        self.path.write_text(json.dumps({**self.base, **extra}))
        return PrototypeConfig.load(self.path)

    def test_on_by_default_and_switchable_off(self):
        self.assertTrue(self.load().turn_hold_unfinished)
        self.assertFalse(self.load(turns={"hold_unfinished": False}).turn_hold_unfinished)

    def test_the_live_predicate_uses_the_names_and_never_holds_a_stop(self):
        from rightyo.prototype import PrototypeController

        config = self.load(addressing={"names": ["Jarvis"]})
        predicate = PrototypeController(config)._turn_unfinished()
        self.assertTrue(predicate("Jarvis,"))
        self.assertTrue(predicate("Jarvis, add the"))
        for stop in ("stop", "never mind", "stop it", "forget it", "Jarvis, ignore that"):
            with self.subTest(stop=stop):
                self.assertFalse(predicate(stop))
        off = self.load(turns={"hold_unfinished": False})
        self.assertIsNone(PrototypeController(off)._turn_unfinished())

    def test_a_non_boolean_is_refused(self):
        for value in (1, "true", None):
            with self.subTest(value=value), self.assertRaises(PrototypeError):
                self.load(turns={"hold_unfinished": value})


if __name__ == "__main__":
    unittest.main()
