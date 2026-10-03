"""Offline tests: no capture, hosted calls, downloaded models or native inference."""

import array
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

from rightyo.live_audio import (
    BYTES_PER_MS,
    FRAME_BYTES,
    TIMELINE_WINDOW_MS,
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
        # One speaker covering everything pushed so far (a contract-valid timeline).
        pushed_ms = sum(map(len, self.frames)) // BYTES_PER_MS
        return [{"start_ms": 0, "end_ms": pushed_ms + 1000, "speaker": 1}]

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

    def test_dropped_audio_gap_cuts_the_utterance_and_advances_timestamps(self):
        self.feed(VOICE, 10)
        self.processor.mark_gap(5000)  # 5 s of host audio dropped under back-pressure.
        # The open utterance is finalized at the gap: nothing after it can be spliced in.
        self.assertEqual([(t.start_ms, t.end_ms) for t in self.turns], [(0, 200)])
        self.assertEqual(len(self.transcribe.call_args.args[1]), 200 * 32)
        self.feed(VOICE, 10)
        self.feed(SILENCE, 72)
        self.assertEqual(len(self.turns), 2)
        self.assertEqual((self.turns[1].start_ms, self.turns[1].end_ms), (5200, 5400))
        self.assertEqual(len(self.transcribe.call_args.args[1]), (200 + 1440) * 32)
        with self.assertRaises(LiveAudioError):
            self.processor.mark_gap(-1)
        self.assertTrue(self.processor.failed)

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
        self.processor.close()
        self.processor = LiveProcessor(
            LiveConfig(**(vars(self.config) | {"whisper_executable": executable})),
            self.turns.append,
        )
        self.addCleanup(self.processor.close)
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
        diarizer.segments = lambda: [{"start_ms": 0, "end_ms": THREE_HOURS_MS + 2000, "speaker": 1}]
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

    def test_worker_failures_are_value_free_and_carry_no_reason_codes(self):
        for line, expected in (
            (b'{"ok":false,"reason":"timeline-limit"}\n', LiveAudioError),
            (b'{"ok":false,"reason":"private-detail"}\n', LiveAudioError),
            (b'{"ok":false}\n', LiveAudioError),
        ):
            with self.subTest(line=line):
                reader, writer = os.pipe()
                diarizer = _Diarizer.__new__(_Diarizer)
                diarizer.timeout = 1
                diarizer.cancelled = lambda: False
                diarizer.process = Mock(stdout=os.fdopen(reader, "rb", buffering=0))
                diarizer.buffer = bytearray()
                os.write(writer, line)
                os.close(writer)
                try:
                    with self.assertRaises(expected) as error:
                        diarizer._receive()
                    self.assertIs(type(error.exception), expected)
                    self.assertNotIn("private-detail", str(error.exception))
                finally:
                    diarizer.process.stdout.close()

    def test_push_advances_the_bound_by_the_audio_actually_pushed(self):
        stream, pushed = _native_stream([(0.0, 1.5, 1)], 0)
        with self.assertRaises(LiveAudioError):
            stream.segments()
        stream.push(bytes(1000 * BYTES_PER_MS))
        self.assertEqual(pushed, [16000])
        self.assertEqual(stream.pushed_bytes, 1000 * BYTES_PER_MS)
        self.assertEqual(stream.segments(), [{"start_ms": 0, "end_ms": 1500, "speaker": 1}])


SEGMENT_MS = 300  # a speaker change every 300 ms: far denser than real conversation


def _session_timeline(until_ms):
    """Contiguous segments cycling three speakers, as the native stream would report."""
    return [
        (start / 1000, min(start + SEGMENT_MS, until_ms) / 1000, start // SEGMENT_MS % 3 + 1)
        for start in range(0, until_ms, SEGMENT_MS)
    ]


def _as_dicts(segments):
    return [
        {"start_ms": round(start * 1000), "end_ms": round(end * 1000), "speaker": speaker}
        for start, end, speaker in segments
    ]


class WindowedTimelineTests(unittest.TestCase):
    """#54: a simulated multi-hour session; no library, model or audio is involved."""

    def test_multi_hour_session_never_hits_a_segment_limit_and_stays_bounded(self):
        bound = TIMELINE_WINDOW_MS // SEGMENT_MS + 1
        for hours in (1, 3, 6):
            now = hours * 60 * 60 * 1000 + 170  # not aligned to a segment boundary
            full = _session_timeline(now)
            stream, _ = _native_stream(full, now * BYTES_PER_MS)
            window = stream.segments()
            with self.subTest(hours=hours, session_segments=len(full)):
                self.assertGreater(len(full), 18000 if hours > 1 else 11000)
                self.assertLessEqual(len(window), bound)
                self.assertEqual(window, [s for s in _as_dicts(full) if s["end_ms"] > now - 60000])
                # The segment straddling the cutoff is kept; the one before it is not.
                self.assertLess(window[0]["start_ms"], now - TIMELINE_WINDOW_MS)

    def test_attribution_over_any_utterance_matches_the_whole_session_timeline(self):
        now = 6 * 60 * 60 * 1000 + 170
        full = _as_dicts(_session_timeline(now))
        stream, _ = _native_stream(_session_timeline(now), now * BYTES_PER_MS)
        window = stream.segments()
        # Words anywhere an un-finalized utterance can reach (15 s plus a tail), at and
        # across segment boundaries, plus words spanning the whole utterance.
        earliest = now - 15000 - 40
        probes = [(start, start + 200) for start in range(earliest, now - 200, 397)]
        probes += [(earliest + 140, earliest + 160), (earliest, now), (now - 300, now)]
        for start, end in probes:
            with self.subTest(start=start, end=end):
                self.assertEqual(_attribute(start, end, window), _attribute(start, end, full))
        self.assertIn(("Speaker C", False), {_attribute(s, e, window) for s, e in probes})

    def test_a_quiet_window_keeps_one_segment_so_provenance_is_unchanged(self):
        now = 2 * 60 * 60 * 1000
        stream, _ = _native_stream([(1.0, 2.0, 2), (5.0, 9.0, 1)], now * BYTES_PER_MS)
        self.assertEqual(stream.segments(), [{"start_ms": 5000, "end_ms": 9000, "speaker": 1}])
        stream, _ = _native_stream([], now * BYTES_PER_MS)
        self.assertEqual(stream.segments(), [])

    def test_processor_turns_match_a_whole_session_timeline_across_hours(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "supplied-local-runtime"
        path.touch()
        results = {}
        for mode in ("window", "full"):
            turns = []
            clock = {}

            class SessionDiarizer:
                def __init__(self, _config):
                    pass

                def push(self, _pcm):
                    pass

                def segments(self):
                    now = clock["processor"].received_ms
                    if mode == "full":
                        return _as_dicts(_session_timeline(now))
                    stream, _ = _native_stream(_session_timeline(now), now * BYTES_PER_MS)
                    return stream.segments()

                finish = segments

                def close(self):
                    pass

            config = LiveConfig("windowed", path, path, path, path, diarizer=SessionDiarizer)
            # The whole-session baseline exceeds the per-response bound by design; only
            # the comparison run skips that check.
            check = patch("rightyo.live_audio._check_timeline") if mode == "full" else nullcontext()
            with check, patch("rightyo.live_audio._transcribe", return_value=document()):
                processor = LiveProcessor(config, turns.append)
                clock["processor"] = processor
                for hours in (1, 2, 4):
                    # Advance the stream clock instead of computing millions of RMS frames.
                    processor._received_ms = hours * 60 * 60 * 1000 + 40 * hours
                    for _ in range(10):
                        processor.push_pcm16(VOICE)
                    for _ in range(72):
                        processor.push_pcm16(SILENCE)
                processor.close()
            results[mode] = turns
        self.assertEqual(len(results["window"]), 3)
        self.assertEqual(results["window"], results["full"])
        # Both attributed and boundary-straddling (unknown) words occur.
        speakers = [t.speaker_id for t in results["window"]]
        self.assertIn(None, speakers)
        self.assertTrue(any(speakers))


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
