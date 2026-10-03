"""Loopback/controller tests use synthetic PCM and fake models, capture and Jev."""

from __future__ import annotations

import http.client
import json
import queue
import tempfile
import threading
import time
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from rightyo.contracts import Turn
from rightyo.credentials import CredentialError
from rightyo.live_audio import LiveAudioError
from rightyo.memory import MemorySessionLimitError
from rightyo.prototype import (
    PrototypeConfig,
    PrototypeController,
    PrototypeError,
    PrototypeServer,
)
from rightyo.providers import MockProvider, ProviderError
from rightyo.tool_events import SpeechEvents

THREE_HOURS_MS = 3 * 60 * 60 * 1000


def await_condition(condition):
    deadline = time.monotonic() + 3
    while not condition() and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    if not condition():
        raise AssertionError("Fake prototype worker did not reach expected state")


class TimerGate:
    """Stands in for the controller's `_closed` event so each timer tick runs on demand."""

    def __init__(self):
        self._condition = threading.Condition()
        self._closed = False
        self._grants = 0
        self._waits = 0

    def install(self, controller):
        controller._closed = self
        # The timer thread leaves its original 0.5 s wait and parks here; no tick runs
        # again until the test grants one.
        with self._condition:
            if not self._condition.wait_for(lambda: self._waits, timeout=5):
                raise AssertionError("Controller timer did not park on the gate")

    def wait(self, timeout=None):
        with self._condition:
            self._waits += 1
            self._condition.notify_all()
            self._condition.wait_for(lambda: self._closed or self._grants)
            if not self._closed:
                self._grants -= 1
            return self._closed

    def set(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def tick(self):
        """Run exactly one complete timer iteration, returning once it parks again."""
        with self._condition:
            target = self._waits + 1
            self._grants += 1
            self._condition.notify_all()
            if not self._condition.wait_for(lambda: self._waits >= target, timeout=5):
                raise AssertionError("Controller timer tick did not complete")


class FakeProcessor:
    instances = []

    def __init__(self, config, on_turn):
        self.config, self.on_turn = config, on_turn
        self.received_ms = 0
        self.index = 0
        self.closed = False
        self.finished = False
        self.instances.append(self)

    def push_pcm16(self, pcm):
        start = self.received_ms
        self.received_ms += len(pcm) // 32
        self.index += 1
        self.on_turn(
            Turn(
                self.config.session_id,
                f"synthetic-{self.index}",
                1,
                start,
                self.received_ms,
                "Rightyo, check what we discussed." if self.index == 3 else "Speaker B, hello.",
                "Speaker A" if self.index % 2 else "Speaker B",
                True,
                False,
                "fake-local-recognizer",
                self.config.provenance,
                "diarization-timeline",
            )
        )

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True


class FakeCapture:
    instances = []

    def __init__(self, helper):
        self.helper = helper
        self.started = False
        self.stopped = False
        self.pcm = queue.Queue()
        self.instances.append(self)

    def start(self):
        self.started = True

    def read(self, timeout):
        try:
            return self.pcm.get(timeout=min(timeout, 0.01))
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True


class FakeHosted:
    def __init__(self, **options):
        self.options = options
        self.requests = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.failure = False

    def decide(self, state):
        self.entered.set()
        self.requests += 1
        self.release.wait(3)
        if self.failure:
            raise ProviderError("Synthetic unavailable provider")
        return MockProvider().decide(state)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        asset = Path(self.directory.name) / "supplied-local-asset"
        asset.touch()
        demo = Path(self.directory.name) / "generated-zero-pcm.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(3 * 3200 * 2))
        self.config = PrototypeConfig(asset, asset, asset, asset, asset, demo)
        FakeProcessor.instances = []
        FakeCapture.instances = []
        self.processor = MagicMock(side_effect=FakeProcessor)
        self.capture = MagicMock(side_effect=FakeCapture)
        self.hosted = MagicMock(side_effect=FakeHosted)
        self.controller = PrototypeController(
            self.config,
            processor_factory=self.processor,
            capture_factory=self.capture,
            provider_factory=self.hosted,
        )
        self.addCleanup(self.controller.close)

    def test_startup_is_idle_without_capture_model_or_hosted_initialization(self):
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["phase"], "idle")
        self.assertEqual(snapshot["turns"], [])
        self.processor.assert_not_called()
        self.capture.assert_not_called()
        self.hosted.assert_not_called()

    def test_invalid_settings_fail_before_any_capture_or_provider_work(self):
        for options in (
            {"mode": {}},
            {"mode": []},
            {"mode": "unknown"},
            {"use_jev": 1},
            {"retention_seconds": True},
            {"retention_seconds": 59},
            {"retention_seconds": 601},
            {"confidence": float("nan")},
            {"confidence": True},
            {"confidence": -0.1},
            {"max_requests": 0},
            {"max_requests": 101},
            {"max_requests": True},
            {"unexpected": "private"},
        ):
            with self.subTest(options=options), self.assertRaises(PrototypeError):
                self.controller.start(options)
        self.processor.assert_not_called()
        self.capture.assert_not_called()
        self.hosted.assert_not_called()

    def test_generated_demo_retains_speakers_without_hosted_opt_in(self):
        self.controller.start({"mode": "demo", "retention_seconds": 60})
        await_condition(lambda: self.controller.snapshot()["phase"] == "complete")
        snapshot = self.controller.snapshot()
        self.assertEqual(len(snapshot["turns"]), 3)
        self.assertEqual(
            [t["speaker_id"] for t in snapshot["turns"]], ["Speaker A", "Speaker B", "Speaker A"]
        )
        self.assertTrue(all(t["provenance"] == "causal-replay" for t in snapshot["turns"]))
        self.assertEqual(snapshot["decision_status"], "off")
        self.assertEqual(snapshot["decisions"], {})
        self.assertEqual(snapshot["jev_requests"], 0)
        self.assertEqual(snapshot["retention"]["retention_ms"], 60000)
        self.assertTrue(FakeProcessor.instances[0].finished)
        self.capture.assert_not_called()
        self.hosted.assert_not_called()
        self.controller.stop()
        self.assertEqual(self.controller.snapshot()["turns"], [])

    def test_hosted_opt_in_and_budget_preserve_remaining_local_transcripts(self):
        self.controller.start({"mode": "demo", "use_jev": True, "max_requests": 1})
        await_condition(lambda: self.controller.snapshot()["phase"] == "complete")
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["jev_requests"], 1)
        self.assertEqual(len(snapshot["decisions"]), 1)
        self.assertEqual(len(snapshot["turns"]), 3)
        self.assertEqual(snapshot["decision_status"], "budget-exhausted")
        self.assertEqual(self.hosted.call_args.kwargs["allow_hosted"], True)
        self.assertEqual(self.hosted.call_args.kwargs["max_requests"], 1)
        self.assertEqual(self.hosted.call_args.kwargs["timeout_seconds"], 10)
        self.assertEqual(self.hosted.call_args.kwargs["min_confidence"], 0.7)
        self.assertTrue(callable(self.hosted.call_args.kwargs["cancelled"]))

    def test_hosted_failure_is_visible_and_local_demo_still_retained(self):
        provider = FakeHosted()
        provider.failure = True
        self.hosted.side_effect = None
        self.hosted.return_value = provider
        self.controller.start({"mode": "demo", "use_jev": True})
        await_condition(lambda: self.controller.snapshot()["phase"] == "complete")
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["decision_status"], "unavailable")
        self.assertEqual(snapshot["decisions"], {})
        self.assertEqual(snapshot["jev_requests"], 1)
        self.assertEqual(len(snapshot["turns"]), 3)

    def test_unexpected_provider_failure_is_visible_without_losing_transcript(self):
        class UnexpectedHosted(FakeHosted):
            def decide(self, state):
                self.requests += 1
                raise RuntimeError("synthetic-private-provider-detail")

        self.hosted.side_effect = UnexpectedHosted
        self.controller.start({"mode": "demo", "use_jev": True})
        await_condition(lambda: self.controller.snapshot()["phase"] == "complete")
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["decision_status"], "unavailable")
        self.assertEqual(snapshot["decisions"], {})
        self.assertEqual(snapshot["jev_requests"], 1)
        self.assertEqual(len(snapshot["turns"]), 3)
        self.assertNotIn("synthetic-private-provider-detail", json.dumps(snapshot))

    def test_audio_failure_preserves_valid_history_cancels_queue_and_ages(self):
        for failure in (
            LiveAudioError,
            RuntimeError,
            MemorySessionLimitError,
        ):
            with self.subTest(failure=failure):
                provider = FakeHosted()
                provider.release.clear()
                self.addCleanup(provider.release.set)
                self.hosted.side_effect = None
                self.hosted.return_value = provider

                class FailingProcessor(FakeProcessor):
                    def push_pcm16(self, pcm):
                        if self.index == 2:
                            raise failure("synthetic-private-audio-detail")
                        super().push_pcm16(pcm)
                        if self.index == 1 and not provider.entered.wait(1):
                            raise AssertionError("Fake decision did not begin")

                self.processor.side_effect = FailingProcessor
                self.controller.start(
                    {
                        "mode": "demo",
                        "use_jev": True,
                        "retention_seconds": 60,
                    }
                )
                await_condition(lambda: self.controller.snapshot()["phase"] == "error")
                old_work = self.controller._decision_queue
                snapshot = self.controller.snapshot()
                self.assertEqual(snapshot["decision_status"], "unavailable")
                self.assertEqual(len(snapshot["turns"]), 2)
                self.assertEqual(snapshot["turns"][-1]["speaker_id"], "Speaker B")
                self.assertTrue(old_work.empty())
                self.assertEqual(snapshot["decisions"], {})
                self.assertNotIn("synthetic-private-audio-detail", snapshot["error"])
                if failure is MemorySessionLimitError:
                    self.assertEqual(
                        snapshot["error"],
                        "Session reached its 1,000-turn limit; start a new session.",
                    )
                else:
                    self.assertNotIn("limit", snapshot["error"])
                provider.release.set()
                await_condition(lambda: old_work.unfinished_tasks == 0)
                self.assertEqual(provider.requests, 1)
                self.assertEqual(self.controller.snapshot()["pending_decisions"], 0)
                self.assertEqual(len(self.controller.snapshot()["turns"]), 2)
                self.assertEqual(self.controller.snapshot()["decisions"], {})
                with self.controller._lock:
                    self.assertIsNotNone(self.controller._completed_at)
                    self.controller._completed_at -= 61
                expired = self.controller.snapshot()
                self.assertEqual(expired["phase"], "error")
                self.assertEqual(expired["turns"], [])
                self.assertEqual(expired["retention"]["expired_turns"], 2)
                self.controller.stop()
                self.assertEqual(self.controller.snapshot()["phase"], "idle")
                self.assertEqual(self.controller.snapshot()["turns"], [])

    def test_audio_failure_between_decision_precheck_and_runner_entry_preserves_history(self):
        class FailingProcessor(FakeProcessor):
            def push_pcm16(self, pcm):
                if self.index == 2:
                    raise LiveAudioError("Synthetic failure after two valid turns")
                super().push_pcm16(pcm)

        provider = FakeHosted()
        self.hosted.side_effect = None
        self.hosted.return_value = provider
        self.processor.side_effect = FailingProcessor
        self.controller.start(
            {
                "mode": "microphone",
                "use_jev": True,
                "retention_seconds": 60,
            }
        )
        await_condition(lambda: bool(FakeCapture.instances and FakeCapture.instances[0].started))
        runner = self.controller._runner
        original_process = runner.process
        entered = threading.Event()
        resume = threading.Event()
        self.addCleanup(resume.set)

        def blocked_process(turn):
            # The controller has released its precheck lock, but the runner has
            # not checked cancellation or initialized its session yet.
            entered.set()
            if not resume.wait(3):
                raise AssertionError("Test did not release delayed runner entry")
            return original_process(turn)

        with patch.object(runner, "process", side_effect=blocked_process):
            FakeCapture.instances[0].pcm.put(bytes(6400))
            self.assertTrue(entered.wait(1))
            FakeCapture.instances[0].pcm.put(bytes(6400))
            await_condition(lambda: len(self.controller.snapshot()["turns"]) == 2)
            FakeCapture.instances[0].pcm.put(bytes(6400))
            await_condition(lambda: self.controller.snapshot()["phase"] == "error")
            snapshot = self.controller.snapshot()
            self.assertEqual(len(snapshot["turns"]), 2)
            self.assertIsNone(runner.session_id)
            self.assertTrue(runner.cancelled())
            self.assertTrue(self.controller._decision_queue.empty())
            resume.set()
            await_condition(lambda: self.controller.snapshot()["pending_decisions"] == 0)
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["phase"], "error")
        self.assertEqual(len(snapshot["turns"]), 2)
        self.assertEqual(
            [item["speaker_id"] for item in snapshot["turns"]], ["Speaker A", "Speaker B"]
        )
        self.assertEqual(provider.requests, 0)
        self.assertEqual(snapshot["jev_requests"], 0)
        self.assertEqual(snapshot["decisions"], {})
        with self.controller._lock:
            self.controller._completed_at -= 61
        self.assertEqual(self.controller.snapshot()["turns"], [])
        self.controller.stop()
        self.assertEqual(self.controller.snapshot()["phase"], "idle")
        self.assertEqual(self.controller.snapshot()["turns"], [])

    def test_stop_waits_for_cancelled_startup_and_blocks_overlapping_start(self):
        for raise_on_cancel in (False, True):
            with self.subTest(raise_on_cancel=raise_on_cancel):
                entered = threading.Event()
                cancelled = threading.Event()
                release = threading.Event()
                self.addCleanup(release.set)
                returned = []

                def delayed_factory(config, callback):
                    entered.set()
                    deadline = time.monotonic() + 2
                    while not config.cancelled() and time.monotonic() < deadline:
                        threading.Event().wait(0.005)
                    if not config.cancelled():
                        raise AssertionError("Startup did not receive cancellation")
                    cancelled.set()
                    if not release.wait(2):
                        raise AssertionError("Test did not release fake startup")
                    if raise_on_cancel:
                        raise LiveAudioError("Synthetic cancelled initialization")
                    processor = FakeProcessor(config, callback)
                    returned.append(processor)
                    return processor

                self.processor.side_effect = delayed_factory
                self.controller.start({"mode": "microphone"})
                self.assertTrue(entered.wait(1))
                old_audio = self.controller._audio_thread
                stopped = threading.Event()

                def stop():
                    self.controller.stop()
                    stopped.set()

                stopper = threading.Thread(target=stop)
                stopper.start()
                try:
                    self.assertTrue(cancelled.wait(1))
                    self.assertEqual(self.controller.snapshot()["phase"], "stopping")
                    self.assertEqual(self.controller.snapshot()["turns"], [])
                    with self.assertRaises(PrototypeError):
                        self.controller.start({"mode": "demo"})
                    self.assertFalse(stopped.is_set())
                    self.assertTrue(old_audio.is_alive())
                    self.capture.assert_not_called()
                finally:
                    release.set()
                    stopper.join(2)
                self.assertTrue(stopped.is_set())
                self.assertFalse(old_audio.is_alive())
                self.assertEqual(self.controller.snapshot()["phase"], "idle")
                self.assertTrue(all(processor.closed for processor in returned))
                self.processor.side_effect = FakeProcessor
                self.controller.start({"mode": "demo"})
                await_condition(lambda: self.controller.snapshot()["phase"] == "complete")
                self.assertEqual(len(self.controller.snapshot()["turns"]), 3)
                self.controller.stop()

    def test_slow_hosted_backpressure_keeps_capture_and_local_history_running(self):
        provider = FakeHosted()
        provider.release.clear()
        self.addCleanup(provider.release.set)
        self.hosted.side_effect = None
        self.hosted.return_value = provider
        self.controller.start({"mode": "microphone", "use_jev": True})
        await_condition(lambda: bool(FakeCapture.instances and FakeCapture.instances[0].started))
        capture = FakeCapture.instances[0]
        capture.pcm.put(bytes(6400))
        self.assertTrue(provider.entered.wait(1))
        for _ in range(39):
            capture.pcm.put(bytes(6400))
        await_condition(lambda: len(self.controller.snapshot()["turns"]) == 40)
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["phase"], "listening")
        self.assertIsNone(snapshot["error"])
        self.assertEqual(snapshot["decision_status"], "unavailable")
        self.assertEqual(snapshot["turns"][-1]["utterance_id"], "synthetic-40")
        self.assertEqual(snapshot["jev_requests"], 1)
        self.assertLessEqual(snapshot["pending_decisions"], 1)
        self.assertTrue(self.controller._decision_queue.empty())
        self.assertFalse(self.controller._stop.is_set())
        self.assertTrue(self.hosted.call_args.kwargs["cancelled"]())
        self.assertFalse(capture.stopped)
        self.assertFalse(FakeProcessor.instances[0].closed)
        provider.release.set()
        await_condition(lambda: self.controller.snapshot()["pending_decisions"] == 0)
        self.assertEqual(provider.requests, 1)
        self.assertEqual(self.controller.snapshot()["decisions"], {})
        capture.pcm.put(bytes(6400))
        await_condition(lambda: len(self.controller.snapshot()["turns"]) == 41)
        continued = self.controller.snapshot()
        self.assertEqual(continued["phase"], "listening")
        self.assertEqual(continued["turns"][-1]["utterance_id"], "synthetic-41")
        self.assertEqual(continued["pending_decisions"], 0)
        self.assertTrue(self.controller._decision_queue.empty())
        self.assertEqual(provider.requests, 1)
        self.assertEqual(continued["decisions"], {})
        self.assertEqual(continued["jev_requests"], 1)
        self.controller.stop()
        self.assertEqual(self.controller.snapshot()["turns"], [])

    def test_missing_hosted_credentials_are_safe_configuration_error(self):
        self.hosted.side_effect = CredentialError("Synthetic missing credential")
        with self.assertRaises(PrototypeError):
            self.controller.start({"mode": "demo", "use_jev": True})
        self.assertEqual(self.controller.snapshot()["phase"], "idle")
        self.processor.assert_not_called()
        self.capture.assert_not_called()

    def test_explicit_microphone_start_uses_fake_capture_and_stop_clears(self):
        self.controller.start({"mode": "microphone"})
        await_condition(lambda: bool(FakeCapture.instances and FakeCapture.instances[0].started))
        FakeCapture.instances[0].pcm.put(bytes(6400))
        await_condition(lambda: len(self.controller.snapshot()["turns"]) == 1)
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["phase"], "listening")
        self.assertEqual(snapshot["turns"][0]["provenance"], "live-microphone")
        self.hosted.assert_not_called()
        self.controller.stop()
        self.assertTrue(FakeCapture.instances[0].stopped)
        self.assertTrue(FakeProcessor.instances[0].closed)
        self.assertEqual(self.controller.snapshot()["turns"], [])

    def test_stop_clears_and_stale_hosted_result_cannot_repopulate(self):
        provider = FakeHosted()
        provider.release.clear()
        self.hosted.side_effect = None
        self.hosted.return_value = provider
        self.controller.start({"mode": "demo", "use_jev": True})
        self.assertTrue(provider.entered.wait(1))
        old_work = self.controller._decision_queue
        self.assertGreater(len(self.controller.snapshot()["turns"]), 0)
        started = time.monotonic()
        self.controller.stop()
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertEqual(self.controller.snapshot()["turns"], [])
        provider.release.set()
        await_condition(lambda: old_work.unfinished_tasks == 0)
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["phase"], "idle")
        self.assertEqual(snapshot["turns"], [])
        self.assertEqual(snapshot["decisions"], {})

    def test_active_session_cannot_be_silently_replaced(self):
        self.controller.start({"mode": "microphone"})
        await_condition(lambda: bool(FakeCapture.instances))
        with self.assertRaises(PrototypeError):
            self.controller.start({"mode": "demo", "use_jev": True})
        self.hosted.assert_not_called()
        self.controller.stop()

    def test_browser_lease_and_configured_wall_clock_budget_stop_observation(self):
        for expiry in ("browser", "session"):
            with self.subTest(expiry=expiry):
                if expiry == "session":
                    self.controller.config = replace(self.config, session_budget_seconds=600)
                self.controller.start({"mode": "microphone"})
                with self.controller._lock:
                    if expiry == "browser":
                        self.controller._last_browser -= 20
                    else:
                        self.controller._started -= 601
                await_condition(
                    lambda: self.controller.snapshot(heartbeat=False)["phase"] == "idle"
                )
                self.assertEqual(self.controller.snapshot()["turns"], [])

    def test_default_has_no_session_ceiling_and_accepts_three_hours_of_audio(self):
        class HourlyProcessor(FakeProcessor):
            def push_pcm16(self, pcm):
                start = self.received_ms
                self.received_ms += len(pcm) // 32
                if self.received_ms % (60 * 60 * 1000):
                    return
                self.index += 1
                self.on_turn(
                    Turn(
                        self.config.session_id,
                        f"hourly-{self.index}",
                        1,
                        start,
                        self.received_ms,
                        "Speaker B, still here.",
                        "Speaker A",
                        True,
                        False,
                        "fake-local-recognizer",
                        self.config.provenance,
                        "diarization-timeline",
                    )
                )

        self.processor.side_effect = HourlyProcessor
        events = SpeechEvents()
        controller = PrototypeController(
            self.config,
            processor_factory=self.processor,
            capture_factory=self.capture,
            provider_factory=self.hosted,
            event_publisher=events,
        )
        self.addCleanup(controller.close)
        self.assertIsNone(controller.snapshot()["session_limit_seconds"])
        controller.start({"mode": "microphone"})
        await_condition(lambda: bool(FakeCapture.instances and FakeCapture.instances[0].started))
        processor = FakeProcessor.instances[0]
        self.assertIsNone(processor.config.session_budget_ms)
        second = bytes(32000)
        for _ in range(THREE_HOURS_MS // 1000):
            FakeCapture.instances[0].pcm.put(second)
        deadline = time.monotonic() + 20
        while processor.received_ms < THREE_HOURS_MS and time.monotonic() < deadline:
            threading.Event().wait(0.01)
        self.assertEqual(processor.received_ms, THREE_HOURS_MS)
        snapshot = controller.snapshot()
        self.assertEqual(snapshot["phase"], "listening")
        self.assertIsNone(snapshot["error"])
        self.assertEqual([t["utterance_id"] for t in snapshot["turns"]], ["hourly-3"])
        self.assertEqual(snapshot["turns"][0]["end_ms"], THREE_HOURS_MS)
        self.assertEqual(snapshot["retention"]["expired_turns"], 2)
        self.assertEqual(snapshot["retention"]["session_turn_count"], 3)
        # Undrained queued events expire with their turns; the current one is delivered.
        transcripts = [e for e in controller.drain_events() if e["type"] == "transcript"]
        self.assertEqual([t["turn"]["utterance_id"] for t in transcripts], ["hourly-3"])
        self.assertEqual(transcripts[-1]["emitted_at_ms"], THREE_HOURS_MS)
        self.assertFalse(processor.closed)
        controller.stop()
        self.assertEqual(controller.drain_events()[-1]["phase"], "cancelled")

    def test_configured_budget_stops_at_the_audio_boundary_as_cancelled(self):
        events = SpeechEvents()
        controller = PrototypeController(
            replace(self.config, session_budget_seconds=2),
            processor_factory=self.processor,
            capture_factory=self.capture,
            provider_factory=self.hosted,
            event_publisher=events,
        )
        self.addCleanup(controller.close)
        self.assertEqual(controller.snapshot()["session_limit_seconds"], 2)
        controller.start({"mode": "microphone"})
        await_condition(lambda: bool(FakeCapture.instances and FakeCapture.instances[0].started))
        processor = FakeProcessor.instances[0]
        self.assertEqual(processor.config.session_budget_ms, 2000)
        for _ in range(15):
            FakeCapture.instances[0].pcm.put(bytes(6400))
        # Drain like a foreground consumer: cancellation discards undrained events.
        emitted = []
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            emitted.extend(controller.drain_events())
            if controller.snapshot(heartbeat=False)["phase"] == "idle":
                break
            threading.Event().wait(0.005)
        emitted.extend(controller.drain_events())
        self.assertEqual(controller.snapshot()["phase"], "idle")
        self.assertEqual(processor.received_ms, 2000)
        self.assertTrue(processor.closed)
        self.assertTrue(FakeCapture.instances[0].stopped)
        self.assertEqual(emitted[0]["phase"], "started")
        self.assertEqual(sum(e["type"] == "transcript" for e in emitted), 10)
        self.assertEqual(emitted[-1]["phase"], "cancelled")
        self.assertNotIn("reason", emitted[-1])
        self.assertEqual(controller.drain_events(), [])
        self.assertEqual(controller.snapshot()["turns"], [])
        self.assertIsNone(controller.snapshot()["error"])

    def test_demo_budget_truncates_replay_without_flushing_the_tail(self):
        demo = Path(self.directory.name) / "generated-two-second.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(2 * 32000))
        events = SpeechEvents()
        controller = PrototypeController(
            replace(self.config, demo_audio=demo, session_budget_seconds=1),
            processor_factory=self.processor,
            capture_factory=self.capture,
            provider_factory=self.hosted,
            event_publisher=events,
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "demo"})
        await_condition(lambda: controller.snapshot(heartbeat=False)["phase"] == "idle")
        processor = FakeProcessor.instances[0]
        self.assertEqual(processor.received_ms, 1000)
        self.assertFalse(processor.finished)
        self.assertTrue(processor.closed)
        self.assertEqual(controller.drain_events()[-1]["phase"], "cancelled")
        self.capture.assert_not_called()

    def test_slow_replay_still_receives_the_whole_budget_before_cancellation(self):
        timer = TimerGate()

        class SlowProcessor(FakeProcessor):
            pushing = False
            closed_mid_push = False

            def push_pcm16(self, pcm):
                self.pushing = True
                super().push_pcm16(pcm)
                # Slower than real time: a full timer tick runs inside every push.
                timer.tick()
                self.pushing = False

            def close(self):
                self.closed_mid_push |= self.pushing
                super().close()

        demo = Path(self.directory.name) / "generated-two-second.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(2 * 32000))
        self.processor.side_effect = SlowProcessor
        events = SpeechEvents()
        controller = PrototypeController(
            replace(self.config, demo_audio=demo, session_budget_seconds=1),
            processor_factory=self.processor,
            capture_factory=self.capture,
            provider_factory=self.hosted,
            event_publisher=events,
        )
        self.addCleanup(controller.close)
        timer.install(controller)
        controller.start({"mode": "demo"})
        with controller._lock:
            # Even a long-expired wall clock must not end a replay before its boundary.
            controller._started -= 3600
        controller._audio_thread.join(timeout=5)
        self.assertFalse(controller._audio_thread.is_alive())
        self.assertEqual(controller.snapshot(heartbeat=False)["phase"], "replaying")
        # Drain before the timer observes the boundary: cancellation discards the queue.
        emitted = controller.drain_events()
        timer.tick()
        emitted.extend(controller.drain_events())
        processor = FakeProcessor.instances[0]
        self.assertEqual(controller.snapshot()["phase"], "idle")
        self.assertEqual(processor.received_ms, 1000)
        self.assertEqual(processor.index, 5)
        self.assertFalse(processor.closed_mid_push)
        self.assertFalse(processor.finished)
        self.assertEqual(sum(e["type"] == "transcript" for e in emitted), 5)
        self.assertEqual(emitted[-1]["phase"], "cancelled")

    def test_invalid_session_budget_is_rejected_before_any_capture(self):
        for invalid in (0, -1, True, 1.5, "60"):
            self.controller.config = replace(self.config, session_budget_seconds=invalid)
            with self.subTest(invalid=invalid), self.assertRaisesRegex(PrototypeError, "budget"):
                self.controller.start({"mode": "microphone"})
        self.processor.assert_not_called()
        self.capture.assert_not_called()
        self.assertEqual(self.controller.snapshot()["phase"], "idle")

    def test_idle_expiry_discards_decision_records_with_transcript(self):
        self.controller.start({"mode": "demo", "use_jev": True})
        await_condition(lambda: self.controller.snapshot()["phase"] == "complete")
        self.assertEqual(len(self.controller.snapshot()["decisions"]), 3)
        with self.controller._lock:
            self.controller._received_ms = 300600
        snapshot = self.controller.snapshot()
        self.assertEqual(snapshot["turns"], [])
        self.assertEqual(snapshot["decisions"], {})
        self.assertEqual(snapshot["retention"]["expired_turns"], 3)

    def test_generated_demo_unavailable_is_explicit(self):
        self.controller.config = replace(self.config, demo_audio=None)
        with self.assertRaises(PrototypeError):
            self.controller.start({"mode": "demo"})
        self.processor.assert_not_called()


class ConfigTests(unittest.TestCase):
    def test_configuration_requires_only_existing_absolute_local_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "local-asset"
            asset.touch()
            config = Path(directory) / "config.json"
            valid = {
                name: str(asset)
                for name in (
                    "whisper_executable",
                    "whisper_model",
                    "diarization_library",
                    "diarization_model",
                    "microphone_helper",
                )
            }
            config.write_text(json.dumps(valid))
            self.assertEqual(PrototypeConfig.load(config).whisper_model, asset)
            self.assertIsNone(PrototypeConfig.load(config).session_budget_seconds)
            config.write_text(json.dumps(valid | {"session_budget_seconds": 4 * 3600}))
            self.assertEqual(PrototypeConfig.load(config).session_budget_seconds, 14400)
            for invalid in (
                {},
                [],
                valid | {"unknown": str(asset)},
                valid | {"session_budget_seconds": 0},
                valid | {"session_budget_seconds": -1},
                valid | {"session_budget_seconds": True},
                valid | {"session_budget_seconds": 1.5},
                valid | {"session_budget_seconds": "60"},
                valid | {"whisper_model": "relative"},
                valid | {"whisper_model": str(asset / "missing")},
                valid | {"whisper_model": {}},
            ):
                config.write_text(json.dumps(invalid))
                with self.assertRaises(PrototypeError) as error:
                    PrototypeConfig.load(config)
                self.assertNotIn(directory, str(error.exception))
            config.write_bytes(b"[" * 2000 + b"0" + b"]" * 2000)
            with self.assertRaises(PrototypeError):
                PrototypeConfig.load(config)
            config.write_bytes(b"{" * 65537)
            with self.assertRaises(PrototypeError):
                PrototypeConfig.load(config)


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.controller = MagicMock()
        self.controller.snapshot.return_value = {"phase": "idle", "turns": []}
        self.server = PrototypeServer(self.controller, port=0)
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01),
            daemon=True,
        )
        self.thread.start()
        self.addCleanup(self.close_server)

    def close_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(1)

    def request(self, method="GET", path="/api/state", *, body=None, headers=None, auth=True):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        values = {"Host": self.server.origin.removeprefix("http://")}
        if auth:
            values["Authorization"] = "Bearer " + self.server.token
        values.update(headers or {})
        connection.request(method, path, body, values)
        response = connection.getresponse()
        status, returned_headers, payload = (
            response.status,
            dict(response.getheaders()),
            response.read(),
        )
        connection.close()
        return status, returned_headers, json.loads(payload)

    def test_authenticated_state_and_no_store_headers(self):
        status, headers, payload = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(payload["phase"], "idle")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertTrue(self.server.launch_url.startswith(self.server.origin + "/#"))

    def test_host_origin_and_bearer_are_independently_enforced(self):
        for options in (
            {"auth": False},
            {"headers": {"Authorization": "Bearer invalid"}},
            {"headers": {"Authorization": "Bearer café"}},
            {"headers": {"Host": "attacker.invalid"}},
            {"headers": {"Host": "localhost:" + str(self.server.server_port)}},
            {"headers": {"Origin": "https://attacker.invalid"}},
            {"headers": {"Origin": "null"}},
        ):
            with self.subTest(options=options):
                status, headers, _ = self.request(**options)
                self.assertEqual(status, 403)
                self.assertEqual(headers["Cache-Control"], "no-store")
        self.controller.snapshot.assert_not_called()
        self.assertEqual(self.request(headers={"Origin": self.server.origin})[0], 200)

    def test_authorized_controls_and_sanitized_configuration_errors(self):
        options = {"mode": "demo", "use_jev": False}
        status, _, _ = self.request("POST", "/api/start", body=json.dumps(options))
        self.assertEqual(status, 200)
        self.controller.start.assert_called_once_with(options)
        self.assertEqual(self.request("POST", "/api/stop", body="{}")[0], 200)
        self.controller.stop.assert_called_once()
        self.controller.start.side_effect = PrototypeError("Invalid prototype setting")
        status, _, payload = self.request("POST", "/api/start", body="{}")
        self.assertEqual(status, 400)
        self.assertEqual(payload["error"], "Invalid prototype setting")

    def test_invalid_json_transfer_encoding_and_body_bounds_never_start(self):
        for body, headers in (
            ("[]", {}),
            ("invalid", {}),
            ("", {}),
            ("{}", {"Content-Length": "4097"}),
            ("{}", {"Content-Length": "-1"}),
            ("{}", {"Content-Length": "invalid"}),
            ("{}", {"Transfer-Encoding": "chunked"}),
            ('{"mode":NaN}', {}),
        ):
            with self.subTest(body=body, headers=headers):
                status, _, _ = self.request("POST", "/api/start", body=body, headers=headers)
                self.assertEqual(status, 400)
        self.controller.start.assert_not_called()

    def test_unknown_routes_and_unauthorized_stop_cannot_mutate(self):
        self.assertEqual(self.request(path="/unknown")[0], 404)
        self.assertEqual(self.request("POST", "/api/unknown", body="{}")[0], 404)
        self.assertEqual(self.request("POST", "/api/stop", body="{}", auth=False)[0], 403)
        self.assertEqual(self.request("POST", "/api/stop", body='{"extra":true}')[0], 404)
        self.controller.start.assert_not_called()
        self.controller.stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
