"""Offline tests: no capture, hosted calls, downloaded models or native inference."""

import array
import ctypes
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rightyo.live_audio import (
    BYTES_PER_MS,
    FRAME_BYTES,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    _attribute,
    _Diarizer,
    _NativeStream,
    _units,
)

SILENCE = bytes(FRAME_BYTES)
VOICE = array.array("h", [5000] * 320).tobytes()
THREE_HOURS_MS = 3 * 60 * 60 * 1000


class FakeDiarizer:
    def __init__(self, config):
        self.frames = []
        self.closed = False
        self.flushed = False

    def push(self, pcm):
        self.frames.append(pcm)

    def segments(self):
        return [{"start_ms": 0, "end_ms": 900000, "speaker": 1}]

    def finish(self):
        self.flushed = True
        return self.segments()

    def close(self):
        self.closed = True


def document(start=0, end=200, text=" Hello."):
    return {"transcription": [{"text": text, "offsets": {"from": start, "to": end}}]}


class LiveProcessorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        path = Path(self.directory.name) / "supplied-local-runtime"
        path.touch()
        self.config = LiveConfig("causal-test", path, path, path, path, provenance="causal-replay")
        self.turns = []
        self.native = patch("rightyo.live_audio._Diarizer", FakeDiarizer)
        self.native.start()
        self.addCleanup(self.native.stop)
        self.asr = patch("rightyo.live_audio._transcribe", return_value=document())
        self.transcribe = self.asr.start()
        self.addCleanup(self.asr.stop)
        self.processor = LiveProcessor(self.config, self.turns.append)
        self.addCleanup(self.processor.close)

    def feed(self, frame, count):
        for _ in range(count):
            self.processor.push_pcm16(frame)

    def test_silence_does_not_decode_and_memory_is_bounded(self):
        self.feed(SILENCE, 1000)
        self.assertEqual(self.processor.buffered_audio_bytes, 240 * 32)
        self.processor.finish()
        self.transcribe.assert_not_called()
        self.assertEqual(self.turns, [])
        self.assertEqual(self.processor.buffered_audio_bytes, 0)

    def test_endpoint_waits_for_native_lookahead_preserves_preroll(self):
        self.feed(SILENCE, 20)
        self.feed(VOICE, 10)
        self.feed(SILENCE, 71)
        self.assertEqual(self.turns, [])
        self.feed(SILENCE, 1)
        self.assertEqual(len(self.turns), 1)
        turn = self.turns[0]
        self.assertEqual(turn.speaker_id, "Speaker A")
        self.assertEqual((turn.start_ms, turn.end_ms), (160, 360))
        self.assertEqual(turn.provenance, "causal-replay")
        self.assertTrue(turn.finalized)
        self.assertEqual(len(self.transcribe.call_args.args[1]), (240 + 200 + 1440) * 32)

    def test_second_utterance_has_global_offsets_same_native_stream(self):
        native = self.processor._diarizer
        self.feed(VOICE, 10)
        self.feed(SILENCE, 72)
        self.feed(VOICE, 10)
        self.feed(SILENCE, 72)
        self.assertIs(self.processor._diarizer, native)
        self.assertEqual([t.start_ms for t in self.turns], [0, 1640])
        self.assertEqual([t.speaker_id for t in self.turns], ["Speaker A", "Speaker A"])
        self.assertEqual([t.utterance_id for t in self.turns], ["live-1", "live-2"])

    def test_long_speech_audio_window_is_bounded(self):
        self.feed(VOICE, 620)
        self.assertEqual(len(self.turns), 1)
        self.assertEqual(len(self.transcribe.call_args.args[1]), 12000 * 32)
        self.assertLessEqual(self.processor.buffered_audio_bytes, 12000 * 32)

    def test_arbitrary_pcm_boundaries_reassemble_without_loss(self):
        self.processor.push_pcm16(VOICE[:10])
        self.assertEqual(self.processor.received_ms, 0)
        self.processor.push_pcm16(VOICE[10:])
        self.assertEqual(self.processor._diarizer.frames, [VOICE])
        self.transcribe.return_value = document(0, 20)
        self.processor.finish()
        self.assertTrue(self.processor._diarizer.flushed)
        self.assertEqual(len(self.turns), 1)

    def test_close_discards_tail_no_decision_no_flush(self):
        self.feed(VOICE, 5)
        self.processor.close()
        self.processor.finish()
        self.assertFalse(self.processor._diarizer.flushed)
        self.transcribe.assert_not_called()
        self.assertEqual(self.turns, [])
        with self.assertRaisesRegex(LiveAudioError, "closed"):
            self.processor.push_pcm16(VOICE)

    def test_finish_tail_uses_final_native_probabilities(self):
        self.feed(VOICE, 10)
        self.processor.finish()
        self.assertTrue(self.processor.closed)
        self.assertTrue(self.processor._diarizer.flushed)
        self.assertEqual(len(self.turns), 1)

    def test_failure_is_explicit_cancels_and_clears_pcm(self):
        self.processor._diarizer.push = Mock(side_effect=LiveAudioError("Local diarizer failed"))
        with self.assertRaisesRegex(LiveAudioError, "diarizer failed"):
            self.processor.push_pcm16(VOICE)
        self.assertTrue(self.processor.failed)
        self.assertTrue(self.processor.closed)
        self.assertEqual(self.processor.buffered_audio_bytes, 0)
        self.assertEqual(self.turns, [])

    def test_asr_failure_does_not_fabricate_final(self):
        self.transcribe.side_effect = LiveAudioError("Local recognizer failed")
        self.feed(VOICE, 10)
        with self.assertRaisesRegex(LiveAudioError, "recognizer failed"):
            self.processor.finish()
        self.assertTrue(self.processor.failed)
        self.assertTrue(self.processor.closed)
        self.assertEqual(self.turns, [])

    def test_stop_cancels_active_asr_and_cannot_register_after_close(self):
        process = Mock()
        process.poll.return_value = None
        self.processor._register_asr(process)
        self.processor.close()
        process.terminate.assert_called_once()
        with self.assertRaisesRegex(LiveAudioError, "stopped"):
            self.processor._register_asr(process)

    def test_stop_interrupts_real_blocked_recognizer_process(self):
        self.asr.stop()
        executable = Path(self.directory.name) / "slow-recognizer"
        executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(30)\n")
        executable.chmod(0o700)
        self.processor.config = LiveConfig(
            **(
                vars(self.config)
                | {
                    "whisper_executable": executable,
                }
            )
        )
        self.feed(VOICE, 10)
        failures = []

        def finalize():
            try:
                self.processor.finish()
            except LiveAudioError as error:
                failures.append(str(error))

        thread = threading.Thread(target=finalize)
        thread.start()
        deadline = time.monotonic() + 3
        while self.processor._asr_process is None and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertIsNotNone(self.processor._asr_process)
        started = time.monotonic()
        self.processor.close()
        thread.join(timeout=3)
        self.assertFalse(thread.is_alive())
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(len(failures), 1)
        self.assertEqual(self.turns, [])
        self.assertEqual(self.processor.buffered_audio_bytes, 0)

    def test_invalid_frame_fails_closed(self):
        for bad in (b"x", bytes(32002), bytearray(VOICE)):
            processor = LiveProcessor(self.config, self.turns.append)
            with self.assertRaises(LiveAudioError):
                processor.push_pcm16(bad)
            self.assertTrue(processor.failed)
            self.assertTrue(processor.closed)

    def test_unbounded_default_accepts_an_utterance_three_hours_into_the_stream(self):
        self.assertIsNone(self.config.session_budget_ms)
        self.feed(SILENCE, 100)
        # Advance the stream clock instead of computing 540,000 synthetic RMS frames;
        # every later offset is derived from this clock exactly as in a real session.
        self.processor._received_ms = THREE_HOURS_MS
        diarizer = self.processor._diarizer
        diarizer.segments = lambda: [{"start_ms": 0, "end_ms": THREE_HOURS_MS * 2, "speaker": 1}]
        self.feed(VOICE, 10)
        self.feed(SILENCE, 72)
        self.assertEqual(len(self.turns), 1)
        self.assertEqual(self.turns[0].start_ms, THREE_HOURS_MS - 240)
        self.assertEqual(self.turns[0].end_ms, THREE_HOURS_MS - 240 + 200)
        self.assertEqual(self.turns[0].speaker_id, "Speaker A")
        self.assertEqual(self.processor.received_ms, THREE_HOURS_MS + 82 * 20)
        self.assertFalse(self.processor.failed)
        self.assertLessEqual(self.processor.buffered_audio_bytes, 240 * BYTES_PER_MS)

    def test_configured_budget_ends_at_the_stream_boundary(self):
        config = LiveConfig(**(vars(self.config) | {"session_budget_ms": 2000}))
        processor = LiveProcessor(config, self.turns.append)
        self.addCleanup(processor.close)
        for _ in range(100):
            processor.push_pcm16(SILENCE)
        self.assertEqual(processor.received_ms, 2000)
        self.assertFalse(processor.failed)
        with self.assertRaisesRegex(LiveAudioError, "duration limit"):
            processor.push_pcm16(SILENCE)
        self.assertTrue(processor.failed)
        self.assertTrue(processor.closed)
        # Unframed partial bytes count toward the budget before they form a frame.
        partial = LiveProcessor(config, self.turns.append)
        self.addCleanup(partial.close)
        for _ in range(99):
            partial.push_pcm16(SILENCE)
        partial.push_pcm16(SILENCE[:320])
        with self.assertRaisesRegex(LiveAudioError, "duration limit"):
            partial.push_pcm16(SILENCE)
        self.assertEqual(self.turns, [])

    def test_session_budget_must_be_a_positive_integer_or_absent(self):
        for invalid in (0, -1, True, 1.5, "60", 2000.0):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(LiveAudioError, "budget"):
                LiveConfig(**(vars(self.config) | {"session_budget_ms": invalid}))
        config = LiveConfig(**(vars(self.config) | {"session_budget_ms": THREE_HOURS_MS * 8}))
        self.assertEqual(config.session_budget_ms, THREE_HOURS_MS * 8)

    def test_configuration_requires_explicit_local_assets_and_limits(self):
        with self.assertRaisesRegex(LiveAudioError, "existing"):
            LiveConfig("test", "/missing", "/missing", "/missing", "/missing")
        for name, value in (
            ("energy_threshold", float("nan")),
            ("hangover_ms", 200),
            ("max_utterance_ms", 300000),
            ("timeout_seconds", True),
        ):
            values = vars(self.config) | {name: value}
            with self.assertRaises(LiveAudioError):
                LiveConfig(**values)


class NativeStartupCancellationTests(unittest.TestCase):
    """A real sleeping Python child stands in for model initialization."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        asset = Path(self.directory.name) / "supplied-local-asset"
        asset.touch()
        self.stop = threading.Event()
        self.config = LiveConfig(
            "cancel-startup",
            asset,
            asset,
            asset,
            asset,
            provenance="causal-replay",
            cancelled=self.stop.is_set,
        )
        self.children = []
        self.real_popen = subprocess.Popen
        self.addCleanup(self.reap_children)

    def reap_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
            for stream in (child.stdin, child.stdout):
                if stream is not None:
                    stream.close()

    def spawn_sleeping_child(self, _command, **options):
        child = self.real_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            **options,
        )
        self.children.append(child)
        return child

    def spawn_ready_then_sleeping_child(self, _command, **options):
        child = self.real_popen(
            [
                sys.executable,
                "-c",
                "import time; print('{\"ok\":true}',flush=True); time.sleep(30)",
            ],
            **options,
        )
        self.children.append(child)
        return child

    def cancel_shortly(self):
        timer = threading.Timer(0.1, self.stop.set)
        timer.start()
        self.addCleanup(timer.join)

    def test_cancel_before_start_does_not_spawn_child(self):
        self.stop.set()
        with patch("rightyo.live_audio.subprocess.Popen") as spawn:
            with self.assertRaisesRegex(LiveAudioError, "stopped"):
                _Diarizer(self.config)
        spawn.assert_not_called()

    def test_stop_interrupts_real_child_initialization_and_reaps_it(self):
        self.cancel_shortly()
        started = time.monotonic()
        with patch("rightyo.live_audio.subprocess.Popen", self.spawn_sleeping_child):
            with self.assertRaisesRegex(LiveAudioError, "stopped"):
                _Diarizer(self.config)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(len(self.children), 1)
        self.assertIsNotNone(self.children[0].poll())

    def test_stop_interrupts_waiting_for_native_response_and_reaps_it(self):
        with patch("rightyo.live_audio.subprocess.Popen", self.spawn_ready_then_sleeping_child):
            diarizer = _Diarizer(self.config)
        self.addCleanup(diarizer.close)
        self.cancel_shortly()
        started = time.monotonic()
        with self.assertRaisesRegex(LiveAudioError, "stopped"):
            diarizer.segments()
        self.assertLess(time.monotonic() - started, 1)
        self.assertIsNotNone(self.children[0].poll())

    def test_cancel_guard_must_be_callable(self):
        with self.assertRaisesRegex(LiveAudioError, "cancellation guard"):
            LiveConfig(**(vars(self.config) | {"cancelled": True}))


def _native_stream(segments, pushed_bytes):
    """A ctypes-free stand-in: no library is loaded and no model is provisioned."""
    stream = _NativeStream.__new__(_NativeStream)
    stream.model = ctypes.c_void_p(1)
    stream.stream = ctypes.c_void_p(1)
    stream.pushed_bytes = pushed_bytes
    pushed = []

    def list_segments(_stream, _model, output, _capacity, count):
        count._obj.value = len(segments)
        if output is not None:
            for slot, (start, end, speaker) in zip(output, segments):
                slot.start_time, slot.end_time, slot.speaker = start, end, speaker
        return 0

    def push(_stream, _floats, samples, _rate):
        pushed.append(samples)
        return 0

    stream.lib = Mock(
        nemo_speech_diar_segments=list_segments, nemo_speech_diar_stream_push_f32=push
    )
    return stream, pushed


class NativeSegmentBoundTests(unittest.TestCase):
    def test_segment_times_are_bounded_by_pushed_audio_not_a_fixed_ceiling(self):
        three_hours = THREE_HOURS_MS * BYTES_PER_MS
        stream, _ = _native_stream([(10799.0, 10800.9, 1)], three_hours)
        self.assertEqual(
            stream.segments(), [{"start_ms": 10799000, "end_ms": 10800900, "speaker": 1}]
        )
        for segments, pushed in (
            ([(10799.0, 10801.5, 1)], three_hours),
            ([(0.5, 5.0, 1)], 0),
            ([(0.5, 5.0, 1)], 3 * 1000 * BYTES_PER_MS),
            ([(0.5, 5.0, 9)], 10 * 1000 * BYTES_PER_MS),
        ):
            stream, _ = _native_stream(segments, pushed)
            with self.subTest(segments=segments, pushed=pushed):
                with self.assertRaisesRegex(LiveAudioError, "Invalid local diarizer result"):
                    stream.segments()
        stream, _ = _native_stream([(0.5, 5.0, 1)], 5 * 1000 * BYTES_PER_MS)
        self.assertEqual(stream.segments(), [{"start_ms": 500, "end_ms": 5000, "speaker": 1}])

    def test_push_advances_the_bound_by_the_audio_actually_pushed(self):
        stream, pushed = _native_stream([(0.0, 1.5, 1)], 0)
        with self.assertRaises(LiveAudioError):
            stream.segments()
        stream.push(bytes(1000 * BYTES_PER_MS))
        self.assertEqual(pushed, [16000])
        self.assertEqual(stream.pushed_bytes, 1000 * BYTES_PER_MS)
        self.assertEqual(stream.segments(), [{"start_ms": 0, "end_ms": 1500, "speaker": 1}])


class ConservativeAlignmentTests(unittest.TestCase):
    def test_channel_ids_are_not_reassigned_when_only_returning_speaker_is_present(self):
        timeline = [{"start_ms": 10000, "end_ms": 12000, "speaker": 2}]
        self.assertEqual(_attribute(10100, 11900, timeline), ("Speaker B", False))

    def test_gap_or_speaker_change_never_uses_majority_assignment(self):
        self.assertEqual(
            _attribute(
                0,
                1000,
                [
                    {"start_ms": 0, "end_ms": 900, "speaker": 1},
                    {"start_ms": 900, "end_ms": 1000, "speaker": 2},
                ],
            ),
            (None, False),
        )
        self.assertEqual(
            _attribute(
                0,
                1000,
                [
                    {"start_ms": 0, "end_ms": 990, "speaker": 1},
                ],
            ),
            (None, False),
        )

    def test_overlap_is_unknown_and_flagged(self):
        self.assertEqual(
            _attribute(
                100,
                200,
                [
                    {"start_ms": 0, "end_ms": 300, "speaker": 1},
                    {"start_ms": 100, "end_ms": 250, "speaker": 2},
                ],
            ),
            (None, True),
        )

    def test_tokens_reconstruct_complete_words_including_zero_time_punctuation(self):
        value = document(0, 600, " Hello world.")
        value["transcription"][0]["tokens"] = [
            {"text": "[_BEG_]"},
            {"text": " Hel", "offsets": {"from": 100, "to": 150}},
            {"text": "lo", "offsets": {"from": 150, "to": 200}},
            {"text": " world", "offsets": {"from": 300, "to": 500}},
            {"text": ".", "offsets": {"from": 500, "to": 500}},
            {"text": "[_TT_30]"},
        ]
        self.assertEqual(
            _units(value, 600),
            [
                {"text": " Hello", "start_ms": 100, "end_ms": 200},
                {"text": " world.", "start_ms": 300, "end_ms": 500},
            ],
        )

    def test_padding_is_bounded_to_received_audio_and_zero_words_stay_unknown(self):
        value = document(0, 1000, " Hello and bye.")
        value["transcription"][0]["tokens"] = [
            {"text": " Hello", "offsets": {"from": 100, "to": 300}},
            {"text": " and", "offsets": {"from": 400, "to": 400}},
            {"text": " bye", "offsets": {"from": 450, "to": 600}},
            {"text": ".", "offsets": {"from": 1000, "to": 1000}},
        ]
        units = _units(value, 700)
        self.assertEqual(units[-1]["end_ms"], 700)
        self.assertEqual(units[1]["start_ms"], units[1]["end_ms"])
        self.assertEqual(
            _attribute(
                units[1]["start_ms"],
                units[1]["end_ms"],
                [
                    {"start_ms": 0, "end_ms": 700, "speaker": 1},
                ],
            ),
            (None, False),
        )
        self.assertEqual("".join(u["text"] for u in units), " Hello and bye.")

    def test_missing_or_inconsistent_token_timing_preserves_whole_text(self):
        for tokens in (
            [{"text": " Hello."}],
            [{"text": " Wrong.", "offsets": {"from": 0, "to": 200}}],
            [{"text": " Hello.", "offsets": {"from": 300, "to": 100}}],
        ):
            value = document()
            value["transcription"][0]["tokens"] = tokens
            self.assertEqual(
                _units(value, 200),
                [
                    {"text": " Hello.", "start_ms": 0, "end_ms": 200},
                ],
            )

    def test_malformed_vendor_result_is_sanitized(self):
        for bad in ({}, {"transcription": "private transcript"}, document(-1, 20)):
            with self.assertRaises(LiveAudioError) as caught:
                _units(bad, 200)
            self.assertNotIn("private transcript", str(caught.exception))
            self.assertNotIn(json.dumps(bad), str(caught.exception))


if __name__ == "__main__":
    unittest.main()
