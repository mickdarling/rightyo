"""One spoken thought split into several turns becomes one turn (rightyo#216).

Synthetic text and scripted timings only, shaped like Pipecat's scripted turns with
`pause_ms`: each step is a speaker's fragment, the pause before it, and whether the
end-of-turn model called it complete. No recorded speech or transcripts.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_turn_merge import ScriptedDiarizer, ScriptedTranscriber, fragment

from rightyo.addressedness import backchannel_shaped
from rightyo.live_audio import LiveAudioError, LiveConfig, LiveProcessor
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.turn_merge import TurnMerger, join_restart

STEP_MS = 20
# The end-of-turn model is asked after this much silence; a turn it calls complete is
# finalized then (as in the live window's `end_of_turn_silence_ms`).
FINALIZE_MS = 600


class Script:
    """Drive a `TurnMerger` through scripted fragments on a stream clock.

    Each step is (speaker, text, pause_ms, duration_ms, complete): the fragment starts
    `pause_ms` after the previous one ends, lasts `duration_ms`, and is offered
    `FINALIZE_MS` after its end. The clock advances in frames, calling `due` with the open
    utterance's start like the live window does, and each emission's stream time is kept.
    """

    def __init__(self, **options):
        self.now = 0
        self.emitted: list[tuple[int, dict]] = []
        options.setdefault("collapse_restarts", True)
        self.merger = TurnMerger(2000, 24000, self._emit, **options)

    def _emit(self, turn):
        self.emitted.append((self.now, turn))

    def _advance(self, until, open_since=None):
        while self.now < until:
            self.now += STEP_MS
            self.merger.due(self.now, open_since)

    def run(self, steps, start=1000):
        end = start
        for speaker, text, pause, duration, complete in steps:
            begin = end + pause
            # Quiet until the next utterance opens, then open until it is finalized.
            self._advance(begin)
            end = begin + duration
            self._advance(end + FINALIZE_MS, open_since=begin)
            self.merger.offer(fragment(text, begin, end, speaker=speaker), complete=complete)
        self._advance(end + 10000)
        return self

    @property
    def texts(self):
        return [turn["text"] for _, turn in self.emitted]

    @property
    def speakers(self):
        return [turn["speaker"] for _, turn in self.emitted]


A, B = "Speaker A", "Speaker B"
SENTENCE = "For the plan, it matters that we show what the new tool will offer."
PARTIAL = "along with what the old tool already"
RESTART = "along with what the old tool already does with plugins."


class RestartJoinTests(unittest.TestCase):
    def test_a_restart_keeps_its_repeated_words_once(self):
        self.assertEqual(join_restart(f"{SENTENCE} {PARTIAL}", RESTART), f"{SENTENCE} {RESTART}")
        # Case and punctuation do not hide the repeat; a whole-fragment restart replaces it.
        self.assertEqual(
            join_restart("Along with what, the old tool", "along with what the old tool does."),
            "along with what the old tool does.",
        )

    def test_fewer_than_three_repeated_words_are_kept(self):
        self.assertEqual(
            join_restart("Add the milk", "the milk please."), ("Add the milk the milk please.")
        )
        self.assertEqual(join_restart("Turn it on.", "And the fan."), "Turn it on. And the fan.")


class BackchannelShapeTests(unittest.TestCase):
    def test_short_acknowledgements_are_backchannels(self):
        for text in ("mm", "Mm-hmm.", "yeah", "Uh-huh.", "okay sure", "Right.", "yep yep"):
            with self.subTest(text=text):
                self.assertTrue(backchannel_shaped(text))

    def test_questions_and_content_are_not(self):
        for text in ("yeah?", "Yeah, I disagree.", "okay let's go", "no", "", "Stop."):
            with self.subTest(text=text):
                self.assertFalse(backchannel_shaped(text))


class SplitThoughtTests(unittest.TestCase):
    def test_sentence_partial_and_restart_become_one_turn(self):
        # A long sentence cut by the utterance window, a partial clause right after it,
        # then a restart that repeats the partial and completes it.
        script = Script(complete_hold_ms=1500).run(
            [
                (A, SENTENCE, 0, 11000, False),
                (A, PARTIAL, 10, 4000, True),
                (A, RESTART, 1200, 4200, True),
            ]
        )
        self.assertEqual(script.texts, [f"{SENTENCE} {RESTART}"])

    def test_two_complete_halves_with_a_short_pause_become_one_turn(self):
        # The first half ends cleanly, the model calls both complete, 300 ms apart.
        first = "Let's do the careful fix for the parser issue we found."
        second = "a change that joins the pieces."
        script = Script(complete_hold_ms=1500).run(
            [(A, first, 0, 7300, True), (A, second, 300, 2150, True)]
        )
        self.assertEqual(script.texts, [f"{first} {second}"])

    def test_a_long_halved_turn_after_a_misjudged_pause_becomes_one_turn(self):
        first = "For the notes on the shared document, including the words that matter,"
        second = "rather than a filter that just blocks everything."
        script = Script(complete_hold_ms=1500).run(
            [(A, first, 0, 14950, True), (A, second, 90, 3080, True)]
        )
        self.assertEqual(script.texts, [f"{first} {second}"])

    def test_without_the_hold_complete_halves_stay_split_as_before(self):
        script = Script().run(
            [(A, "Book the room.", 0, 2000, True), (A, "For noon.", 300, 900, True)]
        )
        self.assertEqual(script.texts, ["Book the room.", "For noon."])

    def test_a_long_pause_is_not_bridged(self):
        # The hold is bounded: a restart after a ten-second pause is its own turn.
        script = Script(complete_hold_ms=1500).run(
            [(A, PARTIAL, 0, 4000, True), (A, RESTART, 10000, 4200, True)]
        )
        self.assertEqual(script.texts, [PARTIAL, RESTART])


class LatencyTests(unittest.TestCase):
    def added_ms(self, script, end):
        # Delay past the model's own finalization (its silence wait is not added here).
        return script.emitted[0][0] - (end + FINALIZE_MS)

    def test_a_complete_single_fragment_is_not_delayed_without_the_hold(self):
        script = Script().run([(A, "What time is it?", 0, 1500, True)])
        self.assertEqual(self.added_ms(script, 2500), 0)

    def test_the_hold_adds_at_most_its_length_less_the_finalization_silence(self):
        script = Script(complete_hold_ms=1500).run([(A, "What time is it?", 0, 1500, True)])
        self.assertEqual(self.added_ms(script, 2500), 1500 - FINALIZE_MS)

    def test_a_merged_turn_is_emitted_one_hold_after_its_last_piece(self):
        script = Script(complete_hold_ms=1500).run(
            [(A, "Book the room.", 0, 2000, True), (A, "For noon.", 300, 900, True)]
        )
        self.assertEqual(len(script.emitted), 1)
        last_end = 1000 + 2000 + 300 + 900
        self.assertEqual(script.emitted[0][0] - (last_end + FINALIZE_MS), 1500 - FINALIZE_MS)


class MultiSpeakerTests(unittest.TestCase):
    def test_another_speakers_interjection_stays_its_own_turn_and_splits(self):
        script = Script(complete_hold_ms=1500).run(
            [
                (A, "Book the room.", 0, 2000, True),
                (B, "Which room do you mean?", 200, 1500, True),
                (A, "For noon.", 300, 900, True),
            ]
        )
        self.assertEqual(script.texts, ["Book the room.", "Which room do you mean?", "For noon."])
        self.assertEqual(script.speakers, [A, B, A])

    def test_a_backchannel_is_bridged_only_when_enabled(self):
        steps = [
            (A, "Book the room.", 0, 2000, True),
            (B, "mm", 100, 300, True),
            (A, "For noon.", 200, 900, True),
        ]
        bridged = Script(complete_hold_ms=1500, bridge_backchannels=True).run(steps)
        self.assertEqual(bridged.texts, ["Book the room. For noon.", "mm"])
        self.assertEqual(bridged.speakers, [A, B])
        plain = Script(complete_hold_ms=1500).run(steps)
        self.assertEqual(plain.texts, ["Book the room.", "mm", "For noon."])

    def test_an_unattributed_fragment_never_joins_a_complete_held_turn(self):
        script = Script(complete_hold_ms=1500, tail_join_ms=2000).run(
            [(A, "Book the room.", 0, 2000, True), (None, "for noon", 100, 800, True)]
        )
        self.assertEqual(script.texts, ["Book the room.", "for noon"])
        self.assertEqual(script.speakers, [A, None])

    def test_an_unattributed_backchannel_is_bridged_not_joined(self):
        script = Script(complete_hold_ms=1500, tail_join_ms=2000, bridge_backchannels=True).run(
            [(A, "Book the room.", 0, 2000, True), (None, "yeah", 100, 300, True)]
        )
        self.assertEqual(script.texts, ["Book the room.", "yeah"])

    def test_a_stop_phrase_is_never_bridged(self):
        emitted = []
        merger = TurnMerger(
            2000,
            24000,
            emitted.append,
            breaks_turn=lambda text: text.lower().strip(".") == "stop",
            complete_hold_ms=1500,
            bridge_backchannels=True,
        )
        merger.offer(fragment("Book the room.", 0, 2000), complete=True)
        merger.offer(fragment("Stop.", 2100, 2400, speaker=B), complete=True)
        self.assertEqual([turn["text"] for turn in emitted], ["Book the room.", "Stop."])


class LongSessionTests(unittest.TestCase):
    def test_a_long_multi_speaker_session_keeps_turns_ordered_and_bounded(self):
        steps = []
        for index in range(150):
            speaker = (A, B, A, None)[index % 4]
            if index % 10 == 0:
                # A split owner thought: two complete halves 300 ms apart.
                steps.append((A, f"First half of thought {index}.", 2500, 3000, True))
                steps.append((A, f"second half {index}.", 300, 1500, True))
            else:
                steps.append((speaker, f"Remark number {index}.", 2500, 2000, True))
        script = Script(complete_hold_ms=1500, bridge_backchannels=True).run(steps)
        self.assertEqual(len(script.emitted), 150)
        starts = [turn["start_ms"] for _, turn in script.emitted]
        self.assertEqual(starts, sorted(starts))
        joined = [text for text in script.texts if text.startswith("First half")]
        self.assertEqual(len(joined), 15)
        self.assertTrue(all("second half" in text for text in joined))
        self.assertFalse(script.merger.holding)


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

    def test_defaults_and_parsing(self):
        config = self.load()
        self.assertEqual(config.turn_complete_hold_ms, 0)
        self.assertTrue(config.turn_collapse_restarts)
        self.assertFalse(config.turn_bridge_backchannels)
        config = self.load(complete_hold_ms=1500, collapse_restarts=False, bridge_backchannels=True)
        self.assertEqual(config.turn_complete_hold_ms, 1500)
        self.assertFalse(config.turn_collapse_restarts)
        self.assertTrue(config.turn_bridge_backchannels)

    def test_invalid_values_are_refused(self):
        for turns in (
            {"complete_hold_ms": -1},
            {"complete_hold_ms": 3001},
            {"complete_hold_ms": True},
            {"collapse_restarts": 1},
            {"bridge_backchannels": "yes"},
        ):
            with self.subTest(turns=turns), self.assertRaises(PrototypeError):
                self.load(**turns)

    def test_live_config_validates_and_passes_the_options(self):
        with self.assertRaises(LiveAudioError):
            LiveConfig("s", transcriber=object, diarizer=object, turn_complete_hold_ms=5000)
        with self.assertRaises(LiveAudioError):
            LiveConfig("s", transcriber=object, diarizer=object, turn_bridge_backchannels=1)
        with self.assertRaises(ValueError):
            TurnMerger(2000, 24000, print, collapse_restarts=None)
        config = LiveConfig(
            "s",
            provenance="causal-replay",
            transcriber=ScriptedTranscriber(("x",)),
            diarizer=ScriptedDiarizer(None),
            turn_merge_gap_ms=2000,
            turn_complete_hold_ms=1200,
            turn_collapse_restarts=True,
            turn_bridge_backchannels=True,
        )
        processor = LiveProcessor(config, print)
        self.addCleanup(processor.close)
        merger = processor._merger
        self.assertEqual(
            (merger.complete_hold_ms, merger.collapse_restarts, merger.bridge_backchannels),
            (1200, True, True),
        )


if __name__ == "__main__":
    unittest.main()
