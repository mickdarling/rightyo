"""Optional Smart Turn end-of-turn in front of the silence rules (#117).

Synthetic PCM, scripted transcriber, diarizer and end-of-turn model, and a stand-in
worker script only: no model, capture, native inference, hosted calls or recorded speech.
"""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from test_turn_merge import (
    SILENCE,
    SPLIT_QUESTION,
    TEXTS,
    VOICE,
    ScriptedDiarizer,
    ScriptedTranscriber,
    audio,
    fragment,
)

from rightyo.live_audio import FRAME_BYTES, LiveAudioError, LiveConfig, LiveProcessor
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.smart_turn import EndOfTurn, SmartTurn, SmartTurnError
from rightyo.turn_merge import DEFAULT_TURN_MERGE_GAP_MS, TurnMerger

try:
    import numpy
except ImportError:  # the worker's features need numpy; RightyO itself does not
    numpy = None


class ScriptedModel:
    """Returns scripted probabilities in order (the last repeats) and records each call."""

    def __init__(self, *probabilities):
        self.probabilities = list(probabilities)
        self.calls = []

    def __call__(self, pcm):
        self.calls.append(len(pcm))
        value = self.probabilities[0]
        if len(self.probabilities) > 1:
            self.probabilities.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class LaggingDiarizer(ScriptedDiarizer):
    """Speaker 1 throughout, but the timeline ends `lag_ms` behind the pushed audio."""

    def __init__(self, lag_ms):
        super().__init__()
        self.lag_ms = lag_ms

    def segments(self):
        end = self.pushed_ms - self.lag_ms
        return [{"start_ms": 0, "end_ms": end, "speaker": 1}] if end > 0 else []

    finish = segments


class MergerCompleteTests(unittest.TestCase):
    def merger(self, reply_wait_ms=0):
        emitted = []
        return TurnMerger(DEFAULT_TURN_MERGE_GAP_MS, 24000, emitted.append, None, reply_wait_ms), (
            emitted
        )

    def test_a_complete_fragment_is_emitted_at_once_with_its_gap_unobserved(self):
        merger, emitted = self.merger(reply_wait_ms=1200)
        merger.offer(fragment("Haili, what time is it?", 0, 900), complete=True)
        self.assertFalse(merger.holding)
        self.assertEqual([item["text"] for item in emitted], ["Haili, what time is it?"])
        self.assertNotIn("post_turn_gap", emitted[0])

    def test_a_held_fragment_is_released_first_with_the_complete_one_following(self):
        merger, emitted = self.merger(reply_wait_ms=1200)
        merger.offer(fragment("Haili, what time is it?", 0, 900, speaker="Speaker A"))
        self.assertTrue(merger.holding)
        merger.offer(fragment("Nearly five.", 5000, 5600, speaker="Speaker B"), complete=True)
        self.assertEqual(
            [item["text"] for item in emitted], ["Haili, what time is it?", "Nearly five."]
        )
        self.assertTrue(emitted[0]["post_turn_gap"]["observed"])
        self.assertNotIn("post_turn_gap", emitted[1])

    def test_a_complete_continuation_joins_and_is_emitted_at_once(self):
        merger, emitted = self.merger()
        merger.offer(fragment("Haili, what time is", 0, 200))
        merger.offer(fragment("it?", 1700, 1900), complete=True)
        self.assertFalse(merger.holding)
        self.assertEqual([item["text"] for item in emitted], ["Haili, what time is it?"])


class LiveEndOfTurnTests(unittest.TestCase):
    def run_processor(
        self, pcm, model, *, until_ms=None, report=None, gaps=None, diarizer=None, **options
    ):
        turns = []
        config = LiveConfig(
            "end-of-turn-test",
            provenance="causal-replay",
            transcriber=ScriptedTranscriber(TEXTS),
            diarizer=diarizer or ScriptedDiarizer(),
            turn_merge_gap_ms=DEFAULT_TURN_MERGE_GAP_MS,
            end_of_turn=model,
            report=report,
            reply_wait_ms=1200 if gaps is not None else 0,
            on_post_turn_gap=(lambda utterance, gap: gaps.append(utterance))
            if gaps is not None
            else None,
            **options,
        )
        processor = LiveProcessor(config, turns.append)
        self.addCleanup(processor.close)
        pcm = pcm if until_ms is None else pcm[: until_ms * 32]
        for offset in range(0, len(pcm), FRAME_BYTES):
            processor.push_pcm16(pcm[offset : offset + FRAME_BYTES])
        return processor, turns

    def test_complete_finalizes_after_the_short_silence_without_any_hold(self):
        model = ScriptedModel(0.9)
        # 200 ms of speech and 200 ms of silence: well before the 1,440 ms hangover.
        _, turns = self.run_processor(SPLIT_QUESTION, model, until_ms=400)
        self.assertEqual([turn.text for turn in turns], ["Hailey, what time is"])
        self.assertEqual(len(model.calls), 1)
        _, turns = self.run_processor(SPLIT_QUESTION, ScriptedModel(0.9))
        self.assertEqual([turn.text for turn in turns], list(TEXTS))

    def test_incomplete_keeps_the_silence_rules_and_asks_once_per_pause(self):
        _, turns = self.run_processor(SPLIT_QUESTION, ScriptedModel(0.1), until_ms=1000)
        self.assertEqual(turns, [])
        model = ScriptedModel(0.1)
        _, turns = self.run_processor(SPLIT_QUESTION, model)
        self.assertEqual([turn.text for turn in turns], ["Hailey, what time is it?"])
        # One question per pause: after the first fragment and after "it?".
        self.assertEqual(len(model.calls), 2)

    def test_the_threshold_and_silence_are_configurable(self):
        _, turns = self.run_processor(
            SPLIT_QUESTION, ScriptedModel(0.6), until_ms=400, end_of_turn_threshold=0.7
        )
        self.assertEqual(turns, [])
        model = ScriptedModel(0.9)
        _, turns = self.run_processor(
            SPLIT_QUESTION, model, until_ms=400, end_of_turn_silence_ms=400
        )
        self.assertEqual((turns, model.calls), ([], []))

    def test_each_question_is_reported_without_content(self):
        messages = []
        self.run_processor(SPLIT_QUESTION, ScriptedModel(0.25, 0.9), report=messages.append)
        self.assertEqual(
            messages,
            [
                "end_of_turn p=0.250 silence_ms=200 outcome=wait",
                "end_of_turn p=0.900 silence_ms=200 outcome=complete",
                "end_of_turn finalized silence_ms=200",
            ],
        )

    def test_a_complete_turn_waits_for_a_lagging_diarizer_timeline(self):
        messages = []
        model = ScriptedModel(0.9)
        _, turns = self.run_processor(
            SPLIT_QUESTION,
            model,
            until_ms=1100,
            report=messages.append,
            diarizer=LaggingDiarizer(lag_ms=800),
        )
        # Judged complete at 200 ms of silence, finalized once the timeline reached the
        # last voiced audio (200 ms) at 1,000 ms of stream time, still before the hangover.
        self.assertEqual([turn.text for turn in turns], ["Hailey, what time is"])
        self.assertEqual(turns[0].speaker_id, "Speaker A")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(messages[-1], "end_of_turn finalized silence_ms=800")

    def test_a_later_disjoint_segment_does_not_count_as_coverage(self):
        class DisjointDiarizer(LaggingDiarizer):
            def segments(self):
                # Covers nothing near the last word, but a noise segment ends after it.
                return [{"start_ms": 0, "end_ms": 100, "speaker": 1}] + (
                    [{"start_ms": 300, "end_ms": self.pushed_ms, "speaker": 2}]
                    if self.pushed_ms > 300
                    else []
                )

        _, turns = self.run_processor(
            SPLIT_QUESTION, ScriptedModel(0.9), until_ms=1000, diarizer=DisjointDiarizer(0)
        )
        self.assertEqual(turns, [])

    def test_an_utterance_level_diarizer_is_not_waited_for(self):
        diarizer = LaggingDiarizer(lag_ms=5000)
        diarizer.speaker_provenance = "diarization-utterance"
        _, turns = self.run_processor(
            SPLIT_QUESTION, ScriptedModel(0.9), until_ms=400, diarizer=diarizer
        )
        self.assertEqual(len(turns), 1)

    def test_the_hangover_still_bounds_the_wait_for_the_timeline(self):
        _, turns = self.run_processor(
            SPLIT_QUESTION, ScriptedModel(0.9), until_ms=1700, diarizer=LaggingDiarizer(5000)
        )
        # Finalized by the hangover (200 + 1,440 ms), still emitted without a merge hold.
        self.assertEqual([turn.text for turn in turns], ["Hailey, what time is"])
        self.assertIsNone(turns[0].speaker_id)

    def test_a_turn_judged_complete_has_no_observed_post_turn_gap(self):
        gaps = []
        _, turns = self.run_processor(SPLIT_QUESTION, ScriptedModel(0.9), gaps=gaps)
        self.assertEqual(len(turns), 2)
        self.assertEqual(gaps, [])

    def test_a_failing_model_falls_back_to_silence_and_is_not_asked_again(self):
        for failure in (RuntimeError("worker died"), 1.5, float("nan"), "0.9"):
            messages = []
            model = ScriptedModel(failure)
            processor, turns = self.run_processor(SPLIT_QUESTION, model, report=messages.append)
            processor.finish()
            self.assertEqual([turn.text for turn in turns], ["Hailey, what time is it?"])
            self.assertEqual(len(model.calls), 1)
            self.assertEqual(messages, ["end-of-turn model unavailable; using silence end-of-turn"])

    def test_a_model_scores_only_the_open_utterance(self):
        model = ScriptedModel(0.1)
        pcm = audio((VOICE, 400), (SILENCE, 2000), (VOICE, 600), (SILENCE, 400))
        self.run_processor(pcm, model)
        # Speech + 200 ms of silence, in bytes, plus the 240 ms pre-roll once audio preceded
        # it; nothing from the earlier utterance.
        self.assertEqual(model.calls, [(400 + 200) * 32, (240 + 600 + 200) * 32])

    def test_configuration_is_validated(self):
        for options in (
            {"end_of_turn": "model"},
            {"end_of_turn_threshold": 0},
            {"end_of_turn_threshold": 1.5},
            {"end_of_turn_threshold": float("nan")},
            {"end_of_turn_threshold": "0.5"},
            {"end_of_turn_silence_ms": 0},
            {"end_of_turn_silence_ms": 210},
            {"end_of_turn_silence_ms": 1020},
        ):
            with self.assertRaises(LiveAudioError):
                LiveConfig(
                    "x", transcriber=ScriptedTranscriber(()), diarizer=ScriptedDiarizer(), **options
                )


class EndOfTurnSectionTests(unittest.TestCase):
    def test_the_section_is_off_unless_present_and_enabled(self):
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
            self.assertIsNone(PrototypeConfig.load(config).end_of_turn)
            section = {"python": str(asset), "model": str(asset)}
            config.write_text(json.dumps(base | {"end_of_turn": section | {"enabled": False}}))
            self.assertIsNone(PrototypeConfig.load(config).end_of_turn)
            config.write_text(json.dumps(base | {"end_of_turn": section | {"threshold": 0.8}}))
            loaded = PrototypeConfig.load(config).end_of_turn
            self.assertEqual(loaded, EndOfTurn(asset, asset, 0.8, 200, 4))
            for invalid in (
                section | {"model": "relative.onnx"},
                section | {"model": str(asset / "missing")},
                section | {"threshold": 0},
                section | {"silence_ms": 30},
                section | {"threads": 0},
                section | {"enabled": "yes"},
                section | {"unknown": 1},
                {"model": str(asset)},
                [],
            ):
                config.write_text(json.dumps(base | {"end_of_turn": invalid}))
                with self.assertRaises(PrototypeError):
                    PrototypeConfig.load(config)


class SmartTurnClientTests(unittest.TestCase):
    """The client against a stand-in worker that speaks the protocol (no model)."""

    def worker(self, body):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        script = Path(directory.name) / "python"
        script.write_text("#!/bin/sh\n" + body)
        script.chmod(script.stat().st_mode | stat.S_IXUSR)
        model = Path(directory.name) / "model.onnx"
        model.touch()
        return script, model

    def test_scores_through_the_worker_and_sends_at_most_eight_seconds(self):
        python, model = self.worker(
            "echo '{\"ok\":true}'\n"
            'while IFS= read -r line; do printf %s "$line" | wc -c >> "$0.sizes";'
            ' echo "{\\"ok\\":true,\\"p\\":0.75}"; done\n'
        )
        client = SmartTurn(python, model)
        self.addCleanup(client.close)
        self.assertEqual(client.score(VOICE), 0.75)
        self.assertEqual(client.score(VOICE * 1000), 0.75)
        client.close()
        sizes = [int(size) for size in Path(str(python) + ".sizes").read_text().split()]
        # 20 s of audio is cut to the last 8 s (256,000 bytes) before it is sent.
        self.assertLess(sizes[1], 256000 * 4 // 3 + 200)
        self.assertGreater(sizes[1], 256000 * 4 // 3)

    def test_failures_are_sanitized(self):
        for body in (
            "exit 0\n",
            "echo '{\"ok\":false}'\n",
            "echo '{\"ok\":true}'\nread -r line\nexit 0\n",
            'echo \'{"ok":true}\'\nread -r line\necho \'{"ok":true,"p":2.0}\'\n',
            'echo \'{"ok":true}\'\nread -r line\necho \'{"ok":true,"p":"0.5"}\'\n',
        ):
            python, model = self.worker(body)
            with self.assertRaises(SmartTurnError) as error:
                client = SmartTurn(python, model)
                self.addCleanup(client.close)
                client.score(VOICE)
            self.assertNotIn(str(python.parent), str(error.exception))

    def test_explicit_existing_files_are_required(self):
        with self.assertRaises(SmartTurnError):
            SmartTurn("/nonexistent/python", "/nonexistent/model.onnx")
        python, model = self.worker("echo '{\"ok\":true}'\n")
        python.chmod(0o600)
        with self.assertRaises(SmartTurnError) as error:
            SmartTurn(python, model)
        self.assertNotIn(str(python.parent), str(error.exception))

    def test_a_worker_that_cannot_be_executed_is_a_sanitized_error(self):
        # Executable but not a runnable program (no interpreter line): exec fails.
        python, model = self.worker("")
        python.write_bytes(b"\x00\x01not a program")
        with self.assertRaises(SmartTurnError) as error:
            SmartTurn(python, model)
        self.assertEqual(str(error.exception), "Smart Turn could not start")

    def test_a_silent_worker_times_out(self):
        python, model = self.worker("echo '{\"ok\":true}'\nsleep 5\n")
        client = SmartTurn(python, model, timeout_seconds=0.2)
        with self.assertRaises(SmartTurnError):
            client.score(VOICE)


@unittest.skipIf(numpy is None, "numpy is not installed")
class FeatureTests(unittest.TestCase):
    def test_features_have_the_reference_shape_and_range(self):
        from rightyo.smart_turn import WINDOW_SAMPLES, _features, _mel_filters

        filters = _mel_filters(numpy)
        self.assertEqual(filters.shape, (201, 80))
        self.assertTrue((filters >= 0).all())
        samples = numpy.random.default_rng(0).normal(0, 0.1, WINDOW_SAMPLES).astype("float32")
        features = _features(numpy, samples)
        self.assertEqual(features.shape, (1, 80, 800))
        self.assertLessEqual(features.max() - features.min(), 2.0 + 1e-6)


if __name__ == "__main__":
    unittest.main()
