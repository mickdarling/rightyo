"""Configurable in-memory audio retention (#226): speaker-ID ring and timeline window.

Synthetic data only: authored tones and zero bytes generated in the test, stand-in
processors and diarizers, and a ctypes stand-in for the native library. No model,
microphone, recorded voice, native inference or hosted call is used. Nothing is written
but the test's own temporary files: its configuration and, for `StartupNoteTests`, a short
synthetic tone WAV, both in a temporary directory that is cleaned up, as in other tests.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from test_live_audio import _as_dicts, _native_stream, _session_timeline
from test_live_speaker_id import RATE, TurnEveryTwoSeconds, tone

import rightyo
from rightyo.live_audio import (
    BYTES_PER_MS,
    MAX_RESPONSE_BYTES,
    MAX_TIMELINE_SEGMENTS,
    MAX_TIMELINE_WINDOW_MS,
    MIN_TIMELINE_WINDOW_MS,
    TIMELINE_WINDOW_MS,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    _check_timeline,
    _Diarizer,
    _NativeStream,
    _response_line,
    _worker_window,
    timeline_segment_limit,
    timeline_window,
)
from rightyo.live_speaker_id import (
    MAX_RING_MS,
    MIN_RING_MS,
    RING_MS,
    PcmRing,
    ShadowSpeakerId,
    speaker_ring_ms,
)
from rightyo.prototype import (
    PrototypeConfig,
    PrototypeController,
    PrototypeError,
    RetentionConfigError,
)
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool import listen

LOCAL = (
    "whisper_executable",
    "whisper_model",
    "diarization_library",
    "diarization_model",
    "microphone_helper",
)
NOTE = re.compile(r"audio_retention speaker_ring_s=(\d+) timeline_window_s=(\d+)")
WRONG_TYPES = ("60", 60.0, True, None, [60], {"seconds": 60})


def _stream(segments, pushed_bytes, window_ms):
    """A `_NativeStream` built by its own `__init__`, with a ctypes stand-in library."""
    lib = Mock()
    for name in ("create", "stream_open", "segments"):
        getattr(lib, "nemo_speech_diar_" + name).return_value = 0
    lib.nemo_speech_diar_num_speakers.return_value = 8
    lib.nemo_speech_diar_seconds_per_frame.return_value = 0.01
    with patch("rightyo.live_audio.ctypes.CDLL", return_value=lib):
        stream = _NativeStream("stand-in-library", "stand-in-model", window_ms)
    # Reuse the shared stand-in's segment listing, keeping this stream's window.
    stand_in, _ = _native_stream(segments, pushed_bytes)
    stream.lib.nemo_speech_diar_segments = stand_in.lib.nemo_speech_diar_segments
    stream.pushed_bytes = pushed_bytes
    return stream


def _dense(count, spacing_ms=3):
    """`count` one-millisecond segments, `spacing_ms` apart, from the session start."""
    return [(i * spacing_ms / 1000, (i * spacing_ms + 1) / 1000, i % 8 + 1) for i in range(count)]


class DefaultTests(unittest.TestCase):
    """Nothing configured: exactly the previous 60 s retention."""

    def test_the_defaults_are_sixty_seconds_and_the_previous_segment_bound(self):
        self.assertEqual(RING_MS, 60000)
        self.assertEqual(TIMELINE_WINDOW_MS, 60000)
        self.assertEqual(MAX_TIMELINE_SEGMENTS, 18000)
        self.assertEqual(timeline_segment_limit(TIMELINE_WINDOW_MS), 18000)
        self.assertEqual(PcmRing().max_bytes, 60000 * BYTES_PER_MS)
        self.assertEqual(
            LiveConfig("x", transcriber=object, diarizer=object).timeline_window_ms, 60000
        )
        stream, _ = _native_stream([], 0)  # built without `__init__`: the class defaults
        self.assertEqual((stream.window_ms, stream.max_segments), (60000, 18000))

    def test_a_configuration_without_the_section_keeps_the_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset"
            asset.touch()
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({key: str(asset) for key in LOCAL}))
            config = PrototypeConfig.load(path)
            path.write_text(
                json.dumps({**{key: str(asset) for key in LOCAL}, "audio_retention": {}})
            )
            empty = PrototypeConfig.load(path)
        for loaded in (config, empty):
            self.assertEqual((loaded.speaker_ring_ms, loaded.timeline_window_ms), (60000, 60000))

    def test_the_default_window_gives_the_previous_cutoff(self):
        now = 2 * 60 * 60 * 1000 + 170
        full = _session_timeline(now)
        window = _stream(full, now * BYTES_PER_MS, TIMELINE_WINDOW_MS).segments()
        legacy, _ = _native_stream(full, now * BYTES_PER_MS)
        self.assertEqual(window, legacy.segments())
        self.assertEqual(window, [s for s in _as_dicts(full) if s["end_ms"] > now - 60000])


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        self.base = {key: str(asset) for key in LOCAL}
        self.path = Path(directory.name) / "config.json"

    def load(self, **retention):
        self.path.write_text(json.dumps({**self.base, "audio_retention": retention}))
        return PrototypeConfig.load(self.path)

    def refused(self, key, value):
        self.path.write_text(json.dumps({**self.base, "audio_retention": {key: value}}))
        with self.assertRaises(RetentionConfigError) as error:
            PrototypeConfig.load(self.path)
        self.assertIsInstance(error.exception, PrototypeError)
        message = str(error.exception)
        self.assertIn(key, message)
        return message

    def test_each_setting_is_accepted_at_both_bounds_and_independent(self):
        for seconds in (10, 60, 600):
            with self.subTest(speaker_ring_seconds=seconds):
                config = self.load(speaker_ring_seconds=seconds)
                self.assertEqual(config.speaker_ring_ms, seconds * 1000)
                self.assertEqual(config.timeline_window_ms, 60000)
        for seconds in (30, 60, 100):
            with self.subTest(timeline_window_seconds=seconds):
                config = self.load(timeline_window_seconds=seconds)
                self.assertEqual(config.timeline_window_ms, seconds * 1000)
                self.assertEqual(config.speaker_ring_ms, 60000)
        both = self.load(speaker_ring_seconds=600, timeline_window_seconds=30)
        self.assertEqual((both.speaker_ring_ms, both.timeline_window_ms), (600000, 30000))

    def test_values_just_outside_the_bounds_are_refused_not_clamped(self):
        for key, values, bounds in (
            ("speaker_ring_seconds", (9, 601, 0, -60, 10**6), "from 10 to 600"),
            ("timeline_window_seconds", (29, 101, 0, -60, 10**6), "from 30 to 100"),
        ):
            messages = set()
            for value in values:
                with self.subTest(key=key, value=value):
                    messages.add(self.refused(key, value))
            # Sanitised: one fixed message naming the rule, never the refused value.
            self.assertEqual(
                messages,
                {
                    f"The audio_retention {key} value must be a whole number of seconds {bounds}",
                },
            )

    def test_wrong_types_are_refused(self):
        for key in ("speaker_ring_seconds", "timeline_window_seconds"):
            messages = {self.refused(key, value) for value in WRONG_TYPES}
            self.assertEqual(len(messages), 1)

    def test_the_section_must_be_an_object_with_only_its_two_keys(self):
        for section in ([], "60", 60, None, {"ring_seconds": 60}, {"speaker_ring_ms": 60000}):
            with self.subTest(section=section):
                self.path.write_text(json.dumps({**self.base, "audio_retention": section}))
                with self.assertRaisesRegex(RetentionConfigError, "audio_retention section"):
                    PrototypeConfig.load(self.path)

    def test_the_library_validators_refuse_the_same_values(self):
        for value in (MIN_RING_MS, RING_MS, MAX_RING_MS):
            self.assertEqual(speaker_ring_ms(value), value)
        for value in (MIN_RING_MS - 1000, MAX_RING_MS + 1000, 10500, True, 60000.0, "60000"):
            with self.subTest(ring=value), self.assertRaises(ValueError):
                speaker_ring_ms(value)
        for value in (MIN_TIMELINE_WINDOW_MS, TIMELINE_WINDOW_MS, MAX_TIMELINE_WINDOW_MS):
            self.assertEqual(timeline_window(value), value)
            LiveConfig("x", transcriber=object, diarizer=object, timeline_window_ms=value)
        for value in (29000, 101000, 60500, True, 60000.0, "60000", None):
            with self.subTest(window=value):
                with self.assertRaisesRegex(LiveAudioError, "Invalid timeline window"):
                    timeline_window(value)
                with self.assertRaisesRegex(LiveAudioError, "Invalid timeline window"):
                    LiveConfig("x", transcriber=object, diarizer=object, timeline_window_ms=value)


class RingTests(unittest.TestCase):
    def test_a_configured_ring_holds_exactly_its_length(self):
        ring = PcmRing(speaker_ring_ms(10000))
        for second in range(12):
            ring.append(bytes([second + 1]) * (1000 * BYTES_PER_MS))
        self.assertEqual(len(ring._buffer), 10000 * BYTES_PER_MS)
        pcm, clipped = ring.span(2000, 12000)  # the last ten seconds, whole
        self.assertEqual((len(pcm), clipped), (10000 * BYTES_PER_MS, False))
        self.assertEqual(set(pcm[: 1000 * BYTES_PER_MS]), {3})
        self.assertEqual(ring.span(1000, 2000), (b"", True))  # the eleventh-oldest second

    def test_resize_shrink_drops_the_oldest_audio_and_grow_keeps_it(self):
        ring = PcmRing(3000)
        for second in range(3):
            ring.append(bytes([second + 1]) * (1000 * BYTES_PER_MS))
        ring.resize(1000)
        self.assertEqual(ring.max_bytes, 1000 * BYTES_PER_MS)
        self.assertEqual(bytes(ring._buffer), bytes([3]) * (1000 * BYTES_PER_MS))
        self.assertEqual(ring.span(2000, 3000), (bytes([3]) * (1000 * BYTES_PER_MS), False))
        self.assertEqual(ring.span(1000, 2000), (b"", True))
        ring.resize(2000)  # growing keeps what is held and fills from new audio
        self.assertEqual(len(ring._buffer), 1000 * BYTES_PER_MS)
        ring.append(bytes([4]) * (1500 * BYTES_PER_MS))
        self.assertEqual(len(ring._buffer), 2000 * BYTES_PER_MS)
        pcm, clipped = ring.span(2500, 4500)
        self.assertEqual((len(pcm), clipped), (2000 * BYTES_PER_MS, False))
        self.assertEqual(set(pcm[: 500 * BYTES_PER_MS]), {3})
        for value in (0, -1, True, 1.5, "1000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ring.resize(value)
        self.assertEqual(ring.max_bytes, 2000 * BYTES_PER_MS)  # a refused resize changes nothing

    def test_a_cleared_ring_behaves_exactly_like_a_fresh_one(self):
        def exercise(ring):
            results = []
            for second in range(5):
                ring.append(bytes([second + 1]) * (1000 * BYTES_PER_MS))
                results.append(ring.span(0, 1000))
                results.append(ring.span(second * 1000, (second + 1) * 1000))
            results.append(ring.span(0, 5000))
            return results, ring.max_bytes, ring._start, bytes(ring._buffer)

        used = PcmRing(3000)
        for _ in range(7):  # past its length, so its start offset has moved on
            used.append(bytes([9]) * (1000 * BYTES_PER_MS))
        self.assertGreater(used._start, 0)
        used.clear()
        fresh = PcmRing(3000)
        self.assertEqual(
            (used._start, used._buffer, used.max_bytes), (0, bytearray(), fresh.max_bytes)
        )
        self.assertEqual(used.span(0, 1000), fresh.span(0, 1000))
        self.assertEqual(exercise(used), exercise(fresh))

    def test_the_identifier_builds_its_ring_from_the_configured_length(self):
        settings = SpeakerIdConfig(Path("/synthetic/python"), Path("/synthetic/model"), live=True)
        shadow = ShadowSpeakerId(settings, ring_ms=speaker_ring_ms(600000))
        self.assertEqual(shadow._ring.max_bytes, 600000 * BYTES_PER_MS)


class WindowTests(unittest.TestCase):
    def test_the_configured_window_sets_the_cutoff_and_the_segment_bound(self):
        now = 2 * 60 * 60 * 1000 + 170
        full = _session_timeline(now)
        lengths = {}
        for window_ms in (MIN_TIMELINE_WINDOW_MS, 60000, MAX_TIMELINE_WINDOW_MS):
            stream = _stream(full, now * BYTES_PER_MS, window_ms)
            result = stream.segments()
            with self.subTest(window_ms=window_ms):
                self.assertEqual(stream.window_ms, window_ms)
                self.assertEqual(stream.max_segments, window_ms // 1000 * 300)
                self.assertEqual(
                    result, [s for s in _as_dicts(full) if s["end_ms"] > now - window_ms]
                )
                self.assertLess(result[0]["start_ms"], now - window_ms)
            lengths[window_ms] = len(result)
        self.assertLess(lengths[30000], lengths[60000])
        self.assertLess(lengths[60000], lengths[100000])
        self.assertEqual(timeline_segment_limit(30000), 9000)
        self.assertEqual(timeline_segment_limit(100000), 30000)

    def test_the_native_segment_bound_follows_the_window(self):
        pushed = 30000 * BYTES_PER_MS
        self.assertEqual(len(_stream(_dense(9000), pushed, 30000).segments()), 9000)
        with self.assertRaisesRegex(LiveAudioError, "Invalid local diarizer result"):
            _stream(_dense(9001), pushed, 30000).segments()
        self.assertEqual(len(_stream(_dense(9001), pushed, 60000).segments()), 9001)

    def test_the_processor_checks_every_backend_against_its_window_bound(self):
        timeline = [{"start_ms": i, "end_ms": i + 1, "speaker": 1} for i in range(9001)]
        _check_timeline(timeline, 10000)  # the default bound, 18,000
        with self.assertRaisesRegex(LiveAudioError, "Invalid diarizer timeline"):
            _check_timeline(timeline, 10000, timeline_segment_limit(30000))
        for window_ms, limit in ((None, 18000), (30000, 9000), (100000, 30000)):
            options = {} if window_ms is None else {"timeline_window_ms": window_ms}
            config = LiveConfig("x", transcriber=Mock(), diarizer=Mock(), **options)
            processor = LiveProcessor(config, lambda _turn: None)
            self.addCleanup(processor.close)
            with self.subTest(window_ms=window_ms):
                self.assertEqual(processor._timeline_limit, limit)

    def test_the_stream_validates_its_window_before_loading_anything(self):
        for value in (29000, 101000, 60500, True):
            with self.subTest(value=value):
                with patch("rightyo.live_audio.ctypes.CDLL") as load:
                    with self.assertRaisesRegex(LiveAudioError, "Invalid timeline window"):
                        _NativeStream("stand-in-library", "stand-in-model", value)
                load.assert_not_called()

    def test_the_diarizer_passes_the_configured_window_to_its_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "supplied-local-runtime"
            asset.touch()
            commands, children = [], []
            real_popen = subprocess.Popen

            def spawn(command, **options):
                commands.append(list(command))
                child = real_popen(
                    [sys.executable, "-c", "print('{\"ok\":true}', flush=True); input()"],
                    **options,
                )
                children.append(child)
                return child

            for options, expected in (({}, "60000"), ({"timeline_window_ms": 45000}, "45000")):
                config = LiveConfig("x", asset, asset, asset, asset, **options)
                with patch("rightyo.live_audio.subprocess.Popen", spawn):
                    diarizer = _Diarizer(config)
                diarizer.close()
                with self.subTest(options=options):
                    self.assertEqual(
                        commands[-1][1:4], ["-m", "rightyo.live_audio", "--native-worker"]
                    )
                    self.assertEqual(len(commands[-1]), 7)
                    self.assertEqual(commands[-1][-1], expected)
            for child in children:
                self.assertIsNotNone(child.poll())


class WorkerArgumentTests(unittest.TestCase):
    def test_only_a_canonical_in_range_window_is_accepted(self):
        for text, value in (("30000", 30000), ("60000", 60000), ("100000", 100000)):
            self.assertEqual(_worker_window(text), value)
        for text in (
            "29000",
            "101000",
            "60500",
            "0",
            "+60000",
            "-60000",
            "060000",
            " 60000",
            "60000 ",
            "60000.0",
            "6e4",
            "",
            "sixty",
            "６００００",  # full-width digits
            "1" * 400,
            None,
            60000,
        ):
            with self.subTest(text=text):
                self.assertIsNone(_worker_window(text))

    def run_worker(self, *extra):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing-runtime")
            source = str(Path(rightyo.__file__).resolve().parent.parent)
            return subprocess.run(
                [sys.executable, "-m", "rightyo.live_audio", "--native-worker", missing, missing]
                + list(extra),
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                env={"PATH": os.defpath, "PYTHONPATH": source},
            )

    def test_the_worker_refuses_an_invalid_window_before_starting(self):
        for text in ("29000", "101000", "60500", "-1", "abc"):
            with self.subTest(text=text):
                result = self.run_worker(text)
                self.assertEqual((result.returncode, result.stdout), (2, b""))
        result = self.run_worker("60000", "extra")
        self.assertEqual((result.returncode, result.stdout), (2, b""))

    def test_a_valid_window_or_none_reaches_the_worker(self):
        # The runtime path does not exist, so the worker fails closed on loading it: a
        # sanitized protocol failure, which shows the window was accepted.
        for extra in (("30000",), ("100000",), ()):
            with self.subTest(extra=extra):
                result = self.run_worker(*extra)
                self.assertEqual((result.returncode, result.stdout), (1, b'{"ok":false}\n'))


class ResponseSizeTests(unittest.TestCase):
    """The worst-case native response at the largest window fits the pipe limit."""

    def worst_case(self, timestamp_ms):
        segment = {"start_ms": timestamp_ms, "end_ms": timestamp_ms, "speaker": 8}
        count = timeline_segment_limit(MAX_TIMELINE_WINDOW_MS)
        return count, _response_line({"ok": True, "segments": [dict(segment)] * count})

    def test_the_largest_window_fits_with_any_reachable_timestamp(self):
        # 10 hours (8 digits), past three years (11 digits), and 16 digits: more stream time
        # than a session can reach. A segment is 36 bytes plus its two timestamps.
        for timestamp_ms, per_segment in (
            (10 * 60 * 60 * 1000, 52),
            (99_999_999_999, 58),
            (9_999_999_999_999_999, 68),
        ):
            count, line = self.worst_case(timestamp_ms)
            size = len(line.encode("utf-8"))
            with self.subTest(timestamp_ms=timestamp_ms):
                self.assertEqual(count, 30000)
                self.assertEqual(size, count * per_segment + 25)
                self.assertLessEqual(size, MAX_RESPONSE_BYTES)
        self.assertEqual(size, 2_040_025)
        self.assertLess(size, 2 * 1024 * 1024)

    def test_the_parent_accepts_the_worst_case_response(self):
        count, line = self.worst_case(10 * 60 * 60 * 1000 + 999)
        reader, writer = os.pipe()
        diarizer = _Diarizer.__new__(_Diarizer)
        diarizer.timeout = 10
        diarizer.cancelled = lambda: False
        diarizer.process = Mock(stdout=os.fdopen(reader, "rb", buffering=0))
        diarizer.buffer = bytearray()

        def write():
            with os.fdopen(writer, "wb") as pipe:
                pipe.write(line.encode("utf-8"))

        thread = threading.Thread(target=write)
        thread.start()
        try:
            result = diarizer._receive()
        finally:
            thread.join(10)
            diarizer.process.stdout.close()
        self.assertEqual(len(result["segments"]), count)


class StartupNoteTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.asset = self.root / "asset"
        self.asset.touch()
        self.demo = self.root / "generated-tone.wav"
        with wave.open(str(self.demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(RATE)
            audio.writeframes(tone(4.0))

    def listen(self, retention, shadow=None):
        """Run `listen` on a demo with live identification; the notes, configs and rings.

        `shadow` replaces the stand-in speaker-ID factory; the whole stderr is kept in
        `self.stderr`.
        """
        path = self.root / "config.json"
        document = {
            **{key: str(self.asset) for key in LOCAL},
            "demo_audio": str(self.demo),
            "speaker_id": {"python": str(self.asset), "model": str(self.asset), "live": True},
        }
        if retention is not None:
            document["audio_retention"] = retention
        path.write_text(json.dumps(document))
        configs, rings = [], []

        def processor(config, on_turn):
            configs.append(config)
            return TurnEveryTwoSeconds(config, on_turn)

        def stand_in(settings, *, report, **options):
            rings.append(options.get("ring_ms"))
            identifier = MagicMock()
            identifier.binding.return_value = None
            return identifier

        def factory(loaded, *, event_publisher, report=None):
            return PrototypeController(
                loaded,
                event_publisher=event_publisher,
                report=report,
                processor_factory=processor,
                capture_factory=MagicMock(side_effect=AssertionError("no microphone")),
                provider_factory=MagicMock(side_effect=AssertionError("no hosted")),
                speaker_id_factory=stand_in if shadow is None else shadow,
            )

        args = Namespace(
            config=path, mode="demo", session_id="retention", use_jev=False, allow_hosted=False
        )
        with patch("sys.stderr", io.StringIO()) as errors:
            self.assertEqual(listen(args, output=io.StringIO(), controller_factory=factory), 0)
        self.stderr = errors.getvalue()
        notes = [
            line.removeprefix("rightyo: ")
            for line in errors.getvalue().splitlines()
            if "audio_retention" in line
        ]
        return notes, configs, rings

    def test_listen_logs_both_values_once_in_seconds_only(self):
        for retention, ring, window in (
            (None, 60, 60),
            ({"speaker_ring_seconds": 600, "timeline_window_seconds": 30}, 600, 30),
            ({"timeline_window_seconds": 100}, 60, 100),
        ):
            notes, configs, rings = self.listen(retention)
            with self.subTest(retention=retention):
                self.assertEqual(len(notes), 1)
                # Seconds only: the whole line is the two fields, nothing else.
                self.assertTrue(NOTE.fullmatch(notes[0]))
                self.assertEqual(
                    notes[0], f"audio_retention speaker_ring_s={ring} timeline_window_s={window}"
                )
                self.assertEqual([config.timeline_window_ms for config in configs], [window * 1000])
                # The default ring passes nothing, so an identifier sees no new argument.
                self.assertEqual(rings, [None if ring == 60 else ring * 1000])

    def test_a_provider_without_ring_ms_keeps_speaker_id_on_and_notes_it(self):
        started = []

        def ringless(settings, *, report):
            identifier = MagicMock()
            identifier.binding.return_value = None
            identifier.start.side_effect = lambda: started.append(True)
            return identifier

        retention = {"speaker_ring_seconds": 600}
        notes, _, _ = self.listen(retention, shadow=ringless)
        self.assertEqual(started, [True])  # built without the ring and started
        self.assertNotIn("speaker_id unavailable", self.stderr)
        self.assertEqual(
            notes,
            [
                "audio_retention speaker_ring_s=600 timeline_window_s=60",
                "audio_retention speaker_ring not applied: provider takes no ring_ms",
            ],
        )
        # The default ring passes nothing and so has nothing to note.
        started.clear()
        notes, _, _ = self.listen(None, shadow=ringless)
        self.assertEqual(started, [True])
        self.assertEqual(notes, ["audio_retention speaker_ring_s=60 timeline_window_s=60"])

    def test_a_provider_that_takes_ring_ms_receives_the_configured_value(self):
        received = []

        def ringed(settings, *, report, ring_ms=RING_MS):
            received.append(ring_ms)
            identifier = MagicMock()
            identifier.binding.return_value = None
            return identifier

        class Ringed:  # a class factory is read through its constructor's signature
            def __init__(self, settings, *, report, ring_ms=RING_MS):
                received.append(ring_ms)

            def start(self):
                pass

            def binding(self, *_args, **_kwargs):
                return None

            def audio(self, _pcm):
                pass

            def turn(self, _turn, **_details):
                pass

            def close(self, **_options):
                pass

        for factory in (ringed, Ringed):
            received.clear()
            with self.subTest(factory=factory):
                notes, _, _ = self.listen({"speaker_ring_seconds": 600}, shadow=factory)
                self.assertEqual(received, [600000])
                self.assertNotIn("speaker_id unavailable", self.stderr)
                self.assertEqual(notes, ["audio_retention speaker_ring_s=600 timeline_window_s=60"])

    def test_a_provider_that_fails_otherwise_still_turns_speaker_id_off(self):
        def broken(settings, *, report):
            raise TypeError("synthetic failure")

        notes, _, _ = self.listen({"speaker_ring_seconds": 600}, shadow=broken)
        self.assertIn("speaker_id unavailable; shadow identification off", self.stderr)
        self.assertEqual(notes, ["audio_retention speaker_ring_s=600 timeline_window_s=60"])


if __name__ == "__main__":
    unittest.main()
