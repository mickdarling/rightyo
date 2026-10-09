"""Shadow live speaker identification (#137 step 4a).

Synthetic data only: audio is an authored tone generated in the test, embeddings come from
stand-in embedders that return controlled vectors, and voiceprints are injected in memory.
No model, microphone, recorded voice or enrollment store is used.
"""

from __future__ import annotations

import io
import json
import math
import shlex
import tempfile
import threading
import time
import unittest
import wave
from argparse import Namespace
from array import array
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from rightyo.contracts import Turn
from rightyo.live_speaker_id import LabelState, PcmRing, ShadowSpeakerId, bind
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents

RATE = 16000
MODEL = {"id": "synthetic/stand-in", "revision": "0" * 40, "sha256": "a" * 64}
OTHER_MODEL = {"id": "synthetic/other", "revision": "1" * 40, "sha256": "b" * 64}
PRIVATE_TEXT = "private words that must never reach a log"


def tone(seconds: float, f0: float = 140.0) -> bytes:
    samples = array("h")
    for n in range(int(seconds * RATE)):
        t = n / RATE
        samples.append(int(9000 * math.sin(2 * math.pi * f0 * t)))
    return samples.tobytes()


def unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


def turn(
    start_ms, end_ms, speaker="Speaker A", *, provenance="diarization-timeline", overlap=False
):
    return Turn(
        "shadow-test",
        f"live-{start_ms}",
        1,
        start_ms,
        end_ms,
        PRIVATE_TEXT,
        speaker,
        True,
        overlap,
        "fake-local-recognizer",
        "causal-replay",
        provenance,
    )


class ScriptedEmbedder:
    """Returns the next scripted vector per call; can block or fail on demand."""

    def __init__(self, vectors=(), *, fail_on=None, gate=None):
        self.model = dict(MODEL)
        self.vectors = list(vectors)
        self.calls = 0
        self.fail_on = fail_on
        self.gate = gate
        self.closed = False

    def embed(self, pcm: bytes) -> list[float]:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail_on is not None and self.calls >= self.fail_on:
            raise RuntimeError("synthetic embedder failure /private/path")
        return self.vectors.pop(0) if self.vectors else unit(1, 0, 0)

    def close(self):
        self.closed = True


def settings(**overrides) -> SpeakerIdConfig:
    base = SpeakerIdConfig(Path("/synthetic/python"), Path("/synthetic/model.onnx"), live=True)
    return replace(base, **overrides)


ENTRIES = [
    {"id": "owner", "model": dict(MODEL), "embedding": unit(1, 0, 0)},
    {"id": "guest", "model": dict(MODEL), "embedding": unit(0, 1, 0)},
    # Another model's voiceprint is never compared, even if it would match.
    {"id": "stale", "model": dict(OTHER_MODEL), "embedding": unit(0, 0, 1)},
]


class Harness:
    def __init__(self, embedder, *, config=None, entries=ENTRIES, queue_size=4, ring_ms=60000):
        self.lines: list[str] = []
        self.embedder = embedder
        self.shadow = ShadowSpeakerId(
            config or settings(),
            report=self.lines.append,
            embedder_factory=lambda _settings: self._make(),
            entries=lambda _settings: [dict(entry) for entry in entries],
            queue_size=queue_size,
            ring_ms=ring_ms,
        )
        self.received_ms = 0

    def _make(self):
        if isinstance(self.embedder, Exception):
            raise self.embedder
        return self.embedder

    def speak(self, seconds, speaker="Speaker A", **options):
        """Feed `seconds` of tone and offer it as one finalized turn."""
        start = self.received_ms
        pcm = tone(seconds)
        self.shadow.audio(pcm)
        self.received_ms += len(pcm) // 32
        self.shadow.turn(turn(start, self.received_ms, speaker, **options))

    def notes(self):
        return [line for line in self.lines if line.startswith("speaker_id label=")]

    @staticmethod
    def fields(line):
        """Every note is `key=value` pairs after its prefix, parseable with shlex."""
        words = shlex.split(line)
        if words[0] != "speaker_id":
            raise AssertionError("not a speaker_id note")
        return dict(word.split("=", 1) for word in words[1:] if "=" in word)


def wait_for(condition, timeout=3.0):
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.005)
    if not condition():
        raise AssertionError("condition not reached")


class RingTests(unittest.TestCase):
    def test_spans_are_addressed_by_stream_time_and_clipped_when_evicted(self):
        ring = PcmRing(max_ms=1000)
        for second in range(3):
            ring.append(bytes([second + 1]) * 32000)
        self.assertEqual(len(ring._buffer), 32000)  # bounded to one second
        pcm, clipped = ring.span(2000, 2500)
        self.assertEqual((len(pcm), clipped), (16000, False))
        self.assertEqual(set(pcm), {3})
        pcm, clipped = ring.span(1500, 2500)  # the first half was evicted
        self.assertEqual((len(pcm), clipped), (16000, True))
        self.assertEqual(ring.span(0, 1000), (b"", True))
        ring.clear()
        self.assertEqual(len(ring._buffer), 0)


class AccumulationTests(unittest.TestCase):
    def test_duration_weighted_mean_of_unit_vectors_is_renormalised(self):
        state = LabelState()
        state.add(unit(1, 0), 1.0)
        mean = state.add(unit(0, 1), 3.0)
        self.assertEqual(state.seconds, 4.0)
        self.assertEqual(state.turns, 2)
        for got, want in zip(mean, unit(1, 3)):
            self.assertAlmostEqual(got, want)

    def test_binding_needs_the_threshold_and_minimum_seconds(self):
        config = settings(bind_threshold=0.6, tentative_threshold=0.45, bind_min_seconds=3.0)
        state = LabelState(seconds=2.0)
        bind(state, {"owner": 0.9, "guest": 0.1}, config)
        self.assertEqual((state.state, state.identity), ("tentative", "owner"))
        state.seconds = 3.0
        bind(state, {"owner": 0.9, "guest": 0.1}, config)
        self.assertEqual((state.state, state.identity, state.score), ("bound", "owner", 0.9))
        fresh = LabelState(seconds=10.0)
        bind(fresh, {"owner": 0.5, "guest": 0.2}, config)
        self.assertEqual(fresh.state, "tentative")
        bind(fresh, {"owner": 0.3, "guest": 0.44}, config)
        self.assertEqual((fresh.state, fresh.identity), ("unknown", "guest"))

    def test_a_bound_label_stays_bound_until_below_the_tentative_threshold(self):
        config = settings(bind_threshold=0.6, tentative_threshold=0.45)
        state = LabelState(seconds=5.0)
        bind(state, {"owner": 0.7, "guest": 0.1}, config)
        self.assertEqual(state.state, "bound")
        # Between the thresholds, and even when another voice now scores higher.
        bind(state, {"owner": 0.5, "guest": 0.55}, config)
        self.assertEqual((state.state, state.identity, state.score), ("bound", "owner", 0.5))
        bind(state, {"owner": 0.44, "guest": 0.3}, config)
        self.assertEqual((state.state, state.identity), ("unknown", "owner"))
        bind(state, {"owner": 0.5, "guest": 0.3}, config)
        self.assertEqual(state.state, "tentative")  # rebinding needs the bind threshold


class ShadowTests(unittest.TestCase):
    def run_shadow(self, harness, *turns):
        harness.shadow.start()
        for seconds, speaker in turns:
            harness.speak(seconds, speaker)
        harness.shadow.close(drain=True)
        return harness.notes()

    def test_turns_accumulate_per_label_and_bind_after_enough_speech(self):
        embedder = ScriptedEmbedder([unit(1, 0.2, 0), unit(0, 1, 0), unit(1, 0.1, 0)])
        harness = Harness(embedder)
        notes = self.run_shadow(harness, (2.0, "Speaker A"), (2.0, "Speaker B"), (2.0, "Speaker A"))
        fields = [Harness.fields(line) for line in notes]
        self.assertEqual([f["label"] for f in fields], ["Speaker A", "Speaker B", "Speaker A"])
        # Two seconds is below the 3 s minimum: tentative, then bound at 4 s.
        self.assertEqual([f["state"] for f in fields], ["tentative", "tentative", "bound"])
        self.assertEqual([f["id"] for f in fields], ["owner", "guest", "owner"])
        self.assertEqual(fields[0]["turn_ms"], "2000")
        self.assertEqual(fields[2]["acc_s"], "4.0")
        expected = unit(*[a + b for a, b in zip(unit(1, 0.2, 0), unit(1, 0.1, 0))])[0]
        self.assertEqual(fields[2]["acc_score"], f"{expected:.3f}")
        self.assertEqual(fields[2]["turn_score"], f"{unit(1, 0.1, 0)[0]:.3f}")
        self.assertIn("speaker_id start enrolled=2", harness.lines)
        self.assertIn(
            'speaker_id final label="Speaker A" turns=2 acc_score='
            f"{expected:.3f} acc_s=4.0 state=bound id=owner",
            harness.lines,
        )
        self.assertTrue(embedder.closed)

    def test_short_overlapping_and_unlabelled_turns_are_not_scored(self):
        embedder = ScriptedEmbedder()
        harness = Harness(embedder)
        harness.shadow.start()
        harness.speak(0.6)  # below min_turn_seconds
        harness.speak(2.0, overlap=True)
        harness.speak(2.0, "u1 Speaker A", provenance="diarization-utterance")
        harness.speak(2.0, speaker=None, provenance="unknown")
        harness.shadow.close(drain=True)
        self.assertEqual(harness.notes(), [])
        self.assertEqual(embedder.calls, 0)
        self.assertIn(
            "speaker_id summary offered=2 scored=0 short=1 overlap=1 dropped=0 clipped=0",
            harness.lines,
        )

    def test_no_matching_voiceprint_switches_shadow_off(self):
        harness = Harness(ScriptedEmbedder(), entries=ENTRIES[2:])
        harness.shadow.start()
        wait_for(lambda: not harness.shadow.active)
        harness.speak(2.0)
        harness.shadow.close()
        self.assertIn(
            "speaker_id live: no voiceprint enrolled with this model; shadow off", harness.lines
        )
        self.assertEqual(harness.notes(), [])

    def test_worker_failure_is_reported_once_and_disables_cleanly(self):
        embedder = ScriptedEmbedder(fail_on=2)
        harness = Harness(embedder)
        harness.shadow.start()
        harness.speak(2.0)
        harness.speak(2.0)
        wait_for(lambda: not harness.shadow.active)
        harness.speak(2.0)  # ignored without error
        harness.shadow.close(drain=True)
        failures = [line for line in harness.lines if "unavailable" in line]
        self.assertEqual(
            failures, ["speaker_id unavailable; shadow identification off for this session"]
        )
        self.assertEqual(len(harness.notes()), 1)
        self.assertEqual(embedder.calls, 2)
        self.assertTrue(embedder.closed)
        self.assertNotIn("/private/path", "\n".join(harness.lines))

    def test_a_model_that_cannot_start_disables_cleanly(self):
        harness = Harness(RuntimeError("synthetic start failure"))
        harness.shadow.start()
        wait_for(lambda: not harness.shadow.active)
        harness.speak(2.0)
        harness.shadow.close()
        self.assertIn(
            "speaker_id unavailable; shadow identification off for this session", harness.lines
        )

    def test_backpressure_drops_and_counts_turns_without_blocking(self):
        gate = threading.Event()
        embedder = ScriptedEmbedder(gate=gate)
        harness = Harness(embedder, queue_size=1)
        harness.shadow.start()
        wait_for(lambda: any("speaker_id start" in line for line in harness.lines))
        started = time.monotonic()
        harness.speak(2.0)
        wait_for(lambda: embedder.calls == 1)  # the worker is now blocked in embed
        for _ in range(4):
            harness.speak(2.0)
        self.assertLess(time.monotonic() - started, 2.0)  # the audio side never waited
        self.assertEqual(harness.shadow.dropped, 3)
        gate.set()
        harness.shadow.close(drain=True)
        self.assertEqual(
            [line for line in harness.lines if "busy" in line],
            ["speaker_id worker busy; dropping turns (counted)"],
        )
        self.assertEqual(len(harness.notes()), 2)
        self.assertIn("dropped=3", harness.lines[-1])

    def test_turns_older_than_the_ring_are_counted_as_clipped(self):
        harness = Harness(ScriptedEmbedder(), ring_ms=800)
        harness.shadow.start()
        harness.speak(2.0)
        harness.shadow.close(drain=True)
        self.assertIn("clipped=1", harness.lines[-1])
        self.assertIn("short=1", harness.lines[-1])  # 0.8 s left: below the minimum

    def test_notes_carry_labels_identifiers_and_numbers_only(self):
        harness = Harness(ScriptedEmbedder([unit(1, 0.3, 0)] * 3))
        self.run_shadow(harness, (2.0, "Speaker A"), (2.0, "Speaker A"))
        log = "\n".join(harness.lines)
        self.assertNotIn(PRIVATE_TEXT, log)
        self.assertNotIn("/", log)
        self.assertNotIn("synthetic", log)  # neither model ids nor paths
        allowed = {"turn_ms", "speech_ms", "turn_score", "acc_score", "acc_s", "state", "id"}
        for line in harness.notes():
            self.assertIn(' label="Speaker A" ', line)
            fields = Harness.fields(line)
            self.assertEqual(set(fields) - {"label"}, allowed)
            for name in ("turn_ms", "speech_ms", "turn_score", "acc_score", "acc_s"):
                float(fields[name])

    def test_a_failure_mid_update_keeps_the_label_and_the_closing_lines(self):
        # Opposite unit vectors with equal weight leave a zero mean: normalising fails.
        embedder = ScriptedEmbedder([unit(1, 0, 0), unit(-1, 0, 0)])
        harness = Harness(embedder)
        harness.shadow.start()
        harness.speak(2.0)
        harness.speak(2.0)
        wait_for(lambda: not harness.shadow.active)
        harness.shadow.close(drain=True)
        state = harness.shadow._labels["Speaker A"]
        self.assertEqual((state.turns, state.seconds), (1, 2.0))  # the failed turn left no trace
        self.assertTrue(harness.lines[-2].startswith('speaker_id final label="Speaker A" turns=1'))
        self.assertTrue(harness.lines[-1].startswith("speaker_id summary offered=2 scored=1"))

    def test_closing_lines_print_even_for_a_label_without_a_score(self):
        harness = Harness(ScriptedEmbedder())
        harness.shadow.offered = 1
        harness.shadow._labels["Speaker B"] = LabelState()
        harness.shadow.close()
        self.assertEqual(
            harness.lines[-2:],
            [
                'speaker_id final label="Speaker B" turns=0 acc_score=- acc_s=0.0'
                " state=unknown id=-",
                "speaker_id summary offered=1 scored=0 short=0 overlap=0 dropped=0 clipped=0",
            ],
        )
        self.assertEqual(Harness.fields(harness.lines[-2])["label"], "Speaker B")

    def test_a_failing_report_channel_changes_nothing(self):
        def broken(_message):
            raise OSError("closed stderr")

        embedder = ScriptedEmbedder()
        shadow = ShadowSpeakerId(
            settings(),
            report=broken,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: ENTRIES,
        )
        shadow.start()
        shadow.audio(tone(2.0))
        shadow.turn(turn(0, 2000))
        shadow.close(drain=True)
        self.assertEqual(shadow.scored, 1)
        self.assertEqual(shadow._labels["Speaker A"].turns, 1)


class TurnEveryTwoSeconds:
    """A stand-in processor: one finalized turn per two seconds of audio."""

    def __init__(self, config, on_turn):
        self.config, self.on_turn = config, on_turn
        self.received_ms = self.emitted_ms = 0
        self.count = 0

    def push_pcm16(self, pcm):
        self.received_ms += len(pcm) // 32
        while self.received_ms - self.emitted_ms >= 2000:
            self.count += 1
            start, self.emitted_ms = self.emitted_ms, self.emitted_ms + 2000
            self.on_turn(
                Turn(
                    self.config.session_id,
                    f"live-{self.count}",
                    1,
                    start,
                    self.emitted_ms,
                    PRIVATE_TEXT,
                    "Speaker A" if self.count % 2 else "Speaker B",
                    True,
                    False,
                    "fake-local-recognizer",
                    self.config.provenance,
                    "diarization-timeline",
                )
            )

    def finish(self):
        pass

    def close(self):
        pass


class ControllerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        demo = Path(directory.name) / "generated-tone.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(RATE)
            audio.writeframes(tone(8.0))
        self.base = PrototypeConfig(asset, asset, asset, asset, asset, demo)
        self.asset = asset

    def session(self, config, factory):
        lines: list[str] = []
        controller = PrototypeController(
            config,
            processor_factory=TurnEveryTwoSeconds,
            capture_factory=MagicMock(),
            provider_factory=MagicMock(),
            event_publisher=SpeechEvents(),
            report=lines.append,
            speaker_id_factory=factory,
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "demo", "session_id": "shadow-session"})
        wait_for(lambda: controller.snapshot(heartbeat=False)["phase"] == "complete", 5)
        wait_for(lambda: not controller._audio_thread.is_alive(), 5)
        events = json.dumps(controller.drain_events(), sort_keys=True)
        return events, lines

    def test_live_false_starts_no_worker_and_events_are_byte_identical(self):
        baseline, baseline_lines = self.session(self.base, MagicMock())
        factory = MagicMock()
        configured = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset))
        events, lines = self.session(configured, factory)
        factory.assert_not_called()
        self.assertEqual(events, baseline)
        self.assertEqual(lines, baseline_lines)
        self.assertIn('"transcript"', events)

    def test_live_true_scores_in_the_shadow_and_leaves_events_unchanged(self):
        baseline, _ = self.session(self.base, MagicMock())
        embedder = ScriptedEmbedder()
        created = []

        def factory(settings, *, report):
            shadow = ShadowSpeakerId(
                settings,
                report=report,
                embedder_factory=lambda _settings: embedder,
                entries=lambda _settings: ENTRIES,
            )
            created.append(shadow)
            return shadow

        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        events, lines = self.session(live, factory)
        self.assertEqual(events, baseline)
        self.assertEqual(len(created), 1)
        notes = [line for line in lines if line.startswith("speaker_id label=")]
        self.assertEqual(len(notes), 4)
        self.assertEqual(embedder.calls, 4)
        self.assertTrue(embedder.closed)
        self.assertNotIn(PRIVATE_TEXT, "\n".join(lines))
        self.assertEqual(created[0]._ring._buffer, bytearray())

    def test_without_a_diagnostic_channel_no_worker_starts(self):
        factory = MagicMock()
        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        controller = PrototypeController(
            live,
            processor_factory=TurnEveryTwoSeconds,
            capture_factory=MagicMock(),
            provider_factory=MagicMock(),
            speaker_id_factory=factory,
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "demo"})
        wait_for(lambda: controller.snapshot(heartbeat=False)["phase"] == "complete", 5)
        factory.assert_not_called()

    def test_a_factory_failure_never_fails_the_session(self):
        baseline, _ = self.session(self.base, MagicMock())

        def factory(settings, *, report):
            raise RuntimeError("synthetic")

        live = replace(self.base, speaker_id=SpeakerIdConfig(self.asset, self.asset, live=True))
        events, lines = self.session(live, factory)
        self.assertEqual(events, baseline)
        self.assertIn("speaker_id unavailable; shadow identification off for this session", lines)


class ListenTests(unittest.TestCase):
    """`rightyo listen` wires stderr notes in every mode, so shadow ID runs in demo too."""

    def test_demo_listen_writes_shadow_notes_to_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset"
            asset.touch()
            demo = root / "generated-tone.wav"
            with wave.open(str(demo), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(RATE)
                audio.writeframes(tone(4.0))
            config = root / "config.json"
            local = (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
            config.write_text(
                json.dumps(
                    {
                        **{key: str(asset) for key in local},
                        "demo_audio": str(demo),
                        "speaker_id": {"python": str(asset), "model": str(asset), "live": True},
                    }
                )
            )
            embedder = ScriptedEmbedder()

            def shadow(settings, *, report):
                return ShadowSpeakerId(
                    settings,
                    report=report,
                    embedder_factory=lambda _settings: embedder,
                    entries=lambda _settings: ENTRIES,
                )

            def factory(loaded, *, event_publisher, report=None):
                return PrototypeController(
                    loaded,
                    event_publisher=event_publisher,
                    report=report,
                    processor_factory=TurnEveryTwoSeconds,
                    capture_factory=MagicMock(side_effect=AssertionError("no microphone")),
                    provider_factory=MagicMock(side_effect=AssertionError("no hosted")),
                    speaker_id_factory=shadow,
                )

            args = Namespace(
                config=config,
                mode="demo",
                session_id="listen-shadow",
                use_jev=False,
                allow_hosted=False,
            )
            output = io.StringIO()
            with patch("sys.stderr", io.StringIO()) as errors:
                self.assertEqual(listen(args, output=output, controller_factory=factory), 0)
                # `listen` returns once the session completes; the worker closes after.
                wait_for(lambda: "speaker_id summary" in errors.getvalue(), 5)
            log = errors.getvalue()
            self.assertIn("rightyo: speaker_id start enrolled=2", log)
            notes = [line for line in log.splitlines() if "speaker_id label=" in line]
            self.assertGreaterEqual(len(notes), 1)
            self.assertNotIn(PRIVATE_TEXT, log)
            self.assertIn(PRIVATE_TEXT, output.getvalue())  # the transcript still went out


if __name__ == "__main__":
    unittest.main()
