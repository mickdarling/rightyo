"""Executable continuous listener qualification with authored PCM and mock speech.

These tests exercise the actual child, PrototypeController, LiveProcessor framing,
trusted control pipe and finalized SpeechEvents. They do not recognize speech, query
devices, activate a microphone, fetch assets or make provider/service calls.
"""

from __future__ import annotations

import json
import math
import os
import queue
import select
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rightyo.coaching_listen import OUTPUT_STALL_SECONDS, OutputQueue, run
from rightyo.contracts import ContractError

SESSION = "11111111-1111-4111-8111-111111111111"
GENERATION = "22222222-2222-4222-8222-222222222222"
DESTINATION = "44444444-4444-4444-8444-444444444444"
QUALIFICATION = "55555555-5555-4555-8555-555555555555"
PROFILE = {
    "version": 1,
    "listener_id": "coach",
    "name": "Coach",
    "domain": "Authored conversation coaching test.",
    "speak_when": "Offer one short private suggestion after a measured pause.",
    "stay_quiet_when": "Owner speech, playback, overlap or missing permission.",
    "delivery": "owner_private",
    "priority": 1,
    "cooldown_ms": 20000,
}
ADVICE = "Ask which outcome matters most."
PCM_TURN = b"".join(
    struct.pack("<h", int(6000 * math.sin(2 * math.pi * 440 * index / 16000)))
    for index in range(6400)
) + bytes(16000 * 2 * 2)


class Child:
    """Finite test-owned process/pipe lifecycle; keep content solely in temp storage."""

    def __init__(
        self,
        directory,
        *,
        utterances=None,
        suggestions=None,
        provenance="synthetic",
        read_output=True,
    ):
        self.events = []
        self._pending = queue.Queue()
        config = directory / "config.json"
        config.write_text(
            json.dumps(
                {
                    "transcriber": {
                        "kind": "synthetic-authored",
                        "utterances": utterances
                        or ["An authored question about project priorities."],
                    },
                    "diarizer": {"kind": "synthetic-authored"},
                    "turns": {"merge_gap_ms": 0, "reply_wait_ms": 0},
                }
            )
        )
        mock = directory / "suggestions.json"
        mock.write_text(json.dumps(suggestions if suggestions is not None else {"live-1": ADVICE}))
        read_fd, self.control_fd = os.pipe()
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        # Device/identity opt-ins belong to the human host, never this qualification.
        for name in tuple(env):
            if name.startswith("RIGHTYO_"):
                env.pop(name)
        try:
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "rightyo",
                    "listen",
                    "--config",
                    str(config),
                    "--mode",
                    "stdin",
                    "--provenance",
                    provenance,
                    "--session-id",
                    SESSION,
                    "--control-fd",
                    str(read_fd),
                    "--coaching",
                    "--coaching-mock-suggestions",
                    str(mock),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                pass_fds=(read_fd,),
                env=env,
                bufsize=0,
            )
        except BaseException:
            os.close(self.control_fd)
            raise
        finally:
            os.close(read_fd)
        self.reader = threading.Thread(target=self._read, daemon=True) if read_output else None
        if self.reader is not None:
            self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    self._pending.put(json.loads(line))
                except (ValueError, UnicodeError) as error:
                    self._pending.put(error)
        except (ValueError, OSError):
            pass  # The output-failure test deliberately closes this stream.
        finally:
            self._pending.put(None)

    def send_bytes(self, data):
        view = memoryview(data)
        while view:
            written = os.write(self.control_fd, view)
            view = view[written:]

    def control(self, value):
        self.send_bytes((json.dumps(value, separators=(",", ":")) + "\n").encode())

    def qualify(self, *, valid_for_ms=2000, qualification=QUALIFICATION, generation=GENERATION):
        self.control(
            {
                "schema_version": 1,
                "type": "coaching_control",
                "action": "qualify",
                "session_id": SESSION,
                "destination_id": DESTINATION,
                "generation_id": generation,
                "qualification_id": qualification,
                "valid_for_ms": valid_for_ms,
            }
        )

    def enable(self, *, qualify=True, valid_for_ms=2000):
        self.control(
            {"version": 1, "type": "listener_profile", "action": "add", "profile": PROFILE}
        )
        self.control(
            {
                "schema_version": 1,
                "type": "coaching_control",
                "action": "enable",
                "session_id": SESSION,
                "destination_id": DESTINATION,
                "listener_id": "coach",
                "participant_awareness": True,
            }
        )
        if qualify:
            self.qualify(valid_for_ms=valid_for_ms)
            self.next(
                lambda event: event.get("type") == "session" and event.get("phase") == "started"
            )

    def audio(self, pcm=PCM_TURN):
        view = memoryview(pcm)
        while view:
            written = self.process.stdin.write(view[:6400])
            if not written:
                raise RuntimeError("child stopped accepting synthetic PCM")
            view = view[written:]

    def activity(
        self,
        *,
        utterance="live-1",
        pause_ms=1000,
        owner_mid_answer=False,
        playback=False,
        generation=GENERATION,
    ):
        self.control(
            {
                "schema_version": 1,
                "type": "coaching_activity",
                "session_id": SESSION,
                "generation_id": generation,
                "utterance_id": utterance,
                "pause_ms": pause_ms,
                "owner_mid_answer": owner_mid_answer,
                "playback": playback,
            }
        )

    def next(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                item = self._pending.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is None:
                break
            if isinstance(item, Exception):
                raise AssertionError("child produced invalid JSONL") from item
            self.events.append(item)
            if predicate(item):
                return item
        raise AssertionError(
            f"no matching child event; exit={self.process.poll()}, events={self.events!r}"
        )

    def collect(self, seconds=0.1):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                item = self._pending.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                break
            if item is None:
                break
            if isinstance(item, Exception):
                raise AssertionError("child produced invalid JSONL") from item
            self.events.append(item)

    def exited(self, expected, timeout=3):
        actual = self.process.wait(timeout=timeout)
        if self.reader is not None:
            self.reader.join(timeout=1)
        self.collect()
        if actual != expected:
            stderr = self.process.stderr.read().decode("utf-8", "replace")
            raise AssertionError(f"child exit {actual}, expected {expected}; {stderr}")
        return self.events

    def close_control(self):
        if self.control_fd is not None:
            os.close(self.control_fd)
            self.control_fd = None

    def close(self):
        self.close_control()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()
        if self.reader is not None:
            self.reader.join(timeout=1)


class CoachingListenTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def child(self, **kwargs):
        result = Child(self.root, **kwargs)
        self.addCleanup(result.close)
        return result

    def assert_no_actions(self, child):
        self.assertFalse(
            any(event.get("type") in {"attention", "request", "action"} for event in child.events)
        )
        for event in child.events:
            if event.get("type") == "coaching_advice":
                self.assertEqual(event["delivery"], "owner_private")
                self.assertEqual(event["decision"]["label"], "speak_now")
                self.assertNotIn("action", event)

    def final(self, child, utterance="live-1"):
        child.audio()
        return child.next(
            lambda event: (
                event.get("type") == "transcript" and event["turn"]["utterance_id"] == utterance
            )
        )

    def test_startup_is_inert_until_explicit_profile_awareness_and_qualification(self):
        child = self.child()
        initial = child.next(lambda event: event.get("type") == "coaching_state")
        self.assertFalse(initial["capture_permitted"])
        # Buffered authored PCM must remain unconsumed before host authorization.
        child.audio(bytes(6400))
        child.enable(qualify=False)
        child.collect(0.15)
        self.assertFalse(
            any(event.get("type") in {"session", "transcript"} for event in child.events)
        )
        child.qualify()
        child.next(lambda event: event.get("type") == "session" and event.get("phase") == "started")
        final = self.final(child)
        self.assertEqual(final["turn"]["recognizer_id"], "synthetic-authored-v1")
        self.assertEqual(final["turn"]["provenance"], "synthetic")
        self.assertEqual(final["turn"]["speaker_provenance"], "authored-fixture")
        self.assertTrue(final["turn"]["finalized"])
        child.activity()
        advice = child.next(lambda event: event.get("type") == "coaching_advice")
        self.assertEqual(advice["text"], ADVICE)
        self.assertEqual(advice["qualification_id"], QUALIFICATION)
        self.assertEqual(advice["generation_id"], GENERATION)
        self.assertGreater(advice["valid_for_ms"], 0)
        self.assertLessEqual(advice["valid_for_ms"], 2000)
        child.process.stdin.close()
        child.exited(0)
        self.assertEqual(sum(event.get("type") == "coaching_advice" for event in child.events), 1)
        self.assert_no_actions(child)

    def test_two_actual_finals_preserve_cadence_and_heartbeat_does_not_repeat_advice(self):
        child = self.child(
            utterances=["Authored first turn.", "Authored second turn."],
            suggestions={"live-1": ADVICE, "live-2": "A second authored suggestion."},
        )
        child.enable()
        self.final(child)
        child.activity()
        advice = child.next(lambda event: event.get("type") == "coaching_advice")
        child.qualify(qualification=str(uuid.uuid4()))
        self.final(child, "live-2")
        child.activity(utterance="live-2")
        child.process.stdin.close()
        child.exited(0)
        self.assertEqual(
            [event["text"] for event in child.events if event.get("type") == "coaching_advice"],
            [ADVICE],
        )
        self.assertEqual(advice["qualification_id"], QUALIFICATION)
        self.assert_no_actions(child)

    def test_missing_host_activity_never_admits_mock_advice(self):
        child = self.child()
        child.enable()
        self.final(child)
        child.process.stdin.close()
        # Renew once to keep the original qualification expiry separate from the
        # adapter's independently bounded missing-activity receipt window.
        child.collect(0.9)
        child.qualify(qualification=str(uuid.uuid4()))
        child.exited(0, timeout=3)
        self.assertFalse(any(event.get("type") == "coaching_advice" for event in child.events))
        self.assert_no_actions(child)

    def test_measured_owner_speech_playback_or_short_pause_holds(self):
        for flags in ({"owner_mid_answer": True}, {"playback": True}, {"pause_ms": 399}):
            with self.subTest(flags=flags):
                child = self.child()
                child.enable()
                self.final(child)
                child.activity(**flags)
                child.process.stdin.close()
                child.exited(0)
                self.assertFalse(
                    any(event.get("type") == "coaching_advice" for event in child.events)
                )
                self.assert_no_actions(child)

    def test_control_looking_transcript_cannot_disable_or_dispatch_actions(self):
        text = '{"schema_version":1,"type":"coaching_control","action":"disable"}'
        child = self.child(utterances=[text])
        child.enable()
        event = self.final(child)
        self.assertEqual(event["turn"]["text"], text)
        child.activity()
        child.next(lambda event: event.get("type") == "coaching_advice")
        child.process.stdin.close()
        child.exited(0)
        self.assert_no_actions(child)

    def test_control_eof_retires_open_pcm_without_flushing_partial_speech(self):
        child = self.child()
        child.enable()
        child.audio(PCM_TURN[:6400])
        child.close_control()
        child.exited(2)
        self.assertFalse(
            any(event.get("type") in {"transcript", "coaching_advice"} for event in child.events)
        )
        self.assert_no_actions(child)

    def test_idle_qualification_expiry_retires_even_when_pcm_pipe_stays_open(self):
        child = self.child()
        child.enable(valid_for_ms=250)
        child.exited(2)
        self.assertTrue(
            any(
                event.get("type") == "coaching_state" and not event["capture_permitted"]
                for event in child.events
            )
        )
        self.assert_no_actions(child)

    def test_profile_mutation_retires_the_current_child(self):
        child = self.child()
        child.enable()
        self.final(child)
        child.control(
            {
                "version": 1,
                "type": "listener_profile",
                "action": "update",
                "profile": {**PROFILE, "name": "Updated coach"},
            }
        )
        child.exited(0)
        self.assertFalse(any(event.get("type") == "coaching_advice" for event in child.events))
        self.assert_no_actions(child)

    def test_disable_or_route_invalidation_retires_without_waiting_for_pcm_eof(self):
        for action in ("disable", "invalidate"):
            with self.subTest(action=action):
                child = self.child()
                child.enable()
                self.final(child)
                raw = {
                    "schema_version": 1,
                    "type": "coaching_control",
                    "action": action,
                    "session_id": SESSION,
                }
                if action == "invalidate":
                    raw.update(generation_id=GENERATION, reason="route_changed")
                child.control(raw)
                child.exited(0)
                self.assertFalse(
                    any(event.get("type") == "coaching_advice" for event in child.events)
                )
                self.assert_no_actions(child)

    def test_duplicate_malformed_nonfinite_or_oversized_control_fails_closed(self):
        for data in (
            b'{"type":"coaching_control","type":"coaching_control"}\n',
            b"{broken}\n",
            b'{"value":NaN}\n',
            b"x" * 8193,
        ):
            with self.subTest(data=data[:48]):
                child = self.child()
                child.next(lambda event: event.get("type") == "coaching_state")
                child.send_bytes(data)
                child.exited(2)
                self.assertFalse(
                    any(
                        event.get("type") in {"session", "transcript", "coaching_advice"}
                        for event in child.events
                    )
                )

    def test_unknown_or_wrong_generation_activity_fails_closed(self):
        for flags in (
            {"utterance": "live-99"},
            {"generation": str(uuid.uuid4())},
            {"pause_ms": 60001},
        ):
            with self.subTest(flags=flags):
                child = self.child()
                child.enable()
                self.final(child)
                child.activity(**flags)
                child.exited(2)
                self.assertFalse(
                    any(event.get("type") == "coaching_advice" for event in child.events)
                )
                self.assert_no_actions(child)

    def test_duplicate_activity_cannot_generate_a_second_suggestion(self):
        child = self.child()
        child.enable()
        self.final(child)
        child.activity()
        child.next(lambda event: event.get("type") == "coaching_advice")
        child.activity()
        child.exited(2)
        self.assertEqual(sum(event.get("type") == "coaching_advice" for event in child.events), 1)
        self.assert_no_actions(child)

    def test_stale_activity_cannot_use_renewed_qualification_to_revive_old_final(self):
        child = self.child()
        child.enable()
        self.final(child)
        for _ in range(3):
            child.collect(0.75)
            child.qualify(qualification=str(uuid.uuid4()))
        child.activity()
        child.exited(2)
        self.assertFalse(any(event.get("type") == "coaching_advice" for event in child.events))
        self.assert_no_actions(child)

    def test_stdout_consumer_failure_retires_in_finite_time(self):
        child = self.child(read_output=False)
        child.enable(qualify=False)
        child.qualify()
        # No background read can retain the pipe after this foreground consumer
        # closes it. Establish real speech startup first through bounded polling.
        pending = bytearray()
        deadline = time.monotonic() + 3
        started = False
        while not started and time.monotonic() < deadline:
            ready, _, _ = select.select([child.process.stdout.fileno()], [], [], 0.05)
            if ready:
                pending.extend(os.read(child.process.stdout.fileno(), 8192))
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    event = json.loads(line)
                    child.events.append(event)
                    started = (
                        started
                        or event.get("type") == "session"
                        and event.get("phase") == "started"
                    )
        self.assertTrue(started)
        child.process.stdout.close()
        child.qualify(qualification=str(uuid.uuid4()))
        child.exited(2)
        self.assert_no_actions(child)

    def test_sigterm_cancels_open_speech_without_eof_flush(self):
        child = self.child()
        child.enable()
        child.audio(PCM_TURN[:6400])
        child.process.send_signal(signal.SIGTERM)
        child.exited(0)
        self.assertFalse(
            any(event.get("type") in {"transcript", "coaching_advice"} for event in child.events)
        )
        self.assert_no_actions(child)

    def test_synthetic_backend_refuses_non_synthetic_pcm_provenance(self):
        child = self.child(provenance="live-microphone")
        child.enable(qualify=False)
        child.qualify()
        child.exited(2)
        self.assertFalse(
            any(
                event.get("type") in {"session", "transcript", "coaching_advice"}
                for event in child.events
            )
        )


class CoachingRetirementOrderingTests(unittest.TestCase):
    def test_disable_signals_audio_cancellation_before_final_output_or_backpressure_wait(self):
        for output_blocked in (False, True):
            with self.subTest(output_blocked=output_blocked):
                started = threading.Event()
                cancelled = threading.Event()
                final_writes = []
                writer_errors = []
                control_read, control_write = os.pipe()
                output_read, output_write = os.pipe()
                self.addCleanup(os.close, control_read)
                self.addCleanup(os.close, control_write)
                self.addCleanup(os.close, output_read)
                output = os.fdopen(output_write, "wb", buffering=0)
                self.addCleanup(output.close)
                if output_blocked:
                    os.set_blocking(output_write, False)
                    while True:
                        try:
                            os.write(output_write, bytes(4096))
                        except BlockingIOError:
                            break

                class Controller:
                    input_overrun = False

                    def __init__(self, *_args, **_kwargs):
                        pass

                    def start(self, _options):
                        started.set()

                    def snapshot(self):
                        return {"phase": "listening"}

                    def drain_events(self):
                        return []

                    def close(self):
                        # This is the controller's existing cancellation boundary:
                        # output/grace must not precede the audio stop token.
                        cancelled.set()

                class ObservedOutput(OutputQueue):
                    retired = False

                    def retire(self):
                        self.retired = True
                        return super().retire()

                    def write(self):
                        if self.retired and self.items:
                            final_writes.append(cancelled.is_set())
                        return super().write()

                def host_controls():
                    try:
                        controls = [
                            {
                                "version": 1,
                                "type": "listener_profile",
                                "action": "add",
                                "profile": PROFILE,
                            },
                            {
                                "schema_version": 1,
                                "type": "coaching_control",
                                "action": "enable",
                                "session_id": SESSION,
                                "destination_id": DESTINATION,
                                "listener_id": "coach",
                                "participant_awareness": True,
                            },
                            {
                                "schema_version": 1,
                                "type": "coaching_control",
                                "action": "qualify",
                                "session_id": SESSION,
                                "destination_id": DESTINATION,
                                "generation_id": GENERATION,
                                "qualification_id": QUALIFICATION,
                                "valid_for_ms": 2000,
                            },
                        ]
                        data = "".join(json.dumps(control) + "\n" for control in controls).encode()
                        os.write(control_write, data)
                        if not started.wait(timeout=2):
                            raise AssertionError("qualified fake controller did not start")
                        disable = {
                            "schema_version": 1,
                            "type": "coaching_control",
                            "action": "disable",
                            "session_id": SESSION,
                        }
                        os.write(control_write, (json.dumps(disable) + "\n").encode())
                    except Exception as error:
                        writer_errors.append(error)

                writer = threading.Thread(target=host_controls)
                writer.start()
                try:
                    with patch("rightyo.coaching_listen.OutputQueue", ObservedOutput):
                        result = run(
                            Namespace(mode="stdin", session_id=SESSION, control_fd=control_read),
                            config=object(),
                            events=object(),
                            consent={},
                            controller_factory=Controller,
                            output=output,
                        )
                finally:
                    writer.join(timeout=3)
                self.assertFalse(writer.is_alive())
                self.assertEqual(writer_errors, [])
                self.assertTrue(started.is_set())
                self.assertTrue(cancelled.is_set())
                self.assertTrue(final_writes, "exercise the actual final-output/grace path")
                self.assertTrue(
                    all(final_writes), "audio must be cancelled before every final write"
                )
                # A trusted disable may finish its bounded final-output grace with
                # either a clean stop or a transport error. Both retire capture.
                self.assertIn(result, (0, 2) if output_blocked else (0,))


class CoachingOutputPressureTests(unittest.TestCase):
    @staticmethod
    def advice(valid_for_ms=500):
        return {
            "schema_version": 1,
            "type": "coaching_advice",
            "session_id": SESSION,
            "generation_id": GENERATION,
            "destination_id": DESTINATION,
            "qualification_id": QUALIFICATION,
            "request_id": str(uuid.uuid4()),
            "listener_id": "coach",
            "delivery": "owner_private",
            "valid_for_ms": valid_for_ms,
            "text": ADVICE,
            "decision": {
                "label": "speak_now",
                "confidence": 1.0,
                "provider": "mock",
                "model": "mock",
            },
        }

    def pipe(self):
        read_fd, write_fd = os.pipe()
        self.addCleanup(os.close, read_fd)
        self.addCleanup(os.close, write_fd)
        os.set_blocking(write_fd, False)
        return read_fd, write_fd

    def test_real_full_pipe_cannot_stall_revocation_loop_indefinitely(self):
        _read_fd, write_fd = self.pipe()
        while True:
            try:
                os.write(write_fd, bytes(4096))
            except BlockingIOError:
                break
        now = [10.0]
        sink = OutputQueue(write_fd, clock=lambda: now[0])
        sink.add([{"type": "session", "phase": "started"}])
        sink.write()
        now[0] += OUTPUT_STALL_SECONDS + 0.001
        with self.assertRaisesRegex(ContractError, "consumer stalled"):
            sink.write()

    def test_queued_advice_original_deadline_is_not_extended_by_later_output(self):
        read_fd, write_fd = self.pipe()
        now = [10.0]
        sink = OutputQueue(write_fd, clock=lambda: now[0])
        advice = self.advice()
        sink.add([advice])
        now[0] = 10.2
        sink.write()
        emitted = json.loads(os.read(read_fd, 8192))
        self.assertLessEqual(emitted["valid_for_ms"], 300)
        self.assertGreater(emitted["valid_for_ms"], 0)
        self.assertEqual(emitted["qualification_id"], QUALIFICATION)
        sink.add([advice])
        now[0] = 10.8
        sink.write()
        self.assertFalse(sink.items)
        os.set_blocking(read_fd, False)
        with self.assertRaises(BlockingIOError):
            os.read(read_fd, 8192)

    def test_advice_drain_guard_wait_cannot_extend_original_transport_deadline(self):
        read_fd, write_fd = self.pipe()
        now = [10.0]
        advice = self.advice(valid_for_ms=200)

        class GuardWait:
            def drain(self):
                # Simulate time spent waiting for the coordinator's profile guard;
                # the returned remaining budget was already measured before it.
                now[0] += 0.3
                return [advice]

        sink = OutputQueue(write_fd, clock=lambda: now[0])
        sink.add_coaching(GuardWait())
        sink.write()
        self.assertFalse(sink.items)
        os.set_blocking(read_fd, False)
        with self.assertRaises(BlockingIOError):
            os.read(read_fd, 8192)

    def test_retirement_drops_all_queued_plaintext_and_rejects_partial_json_handoff(self):
        _read_fd, write_fd = self.pipe()
        sink = OutputQueue(write_fd)
        sink.add([{"type": "transcript", "text": "An authored pending turn."}, self.advice()])
        sink.retire()
        self.assertFalse(sink.items)
        self.assertEqual(sink.bytes, 0)
        # A real nonblocking pipe accepts only a prefix of this oversized ordinary
        # event. Retirement must end the stream, not append a closed-state JSON line
        # into that incomplete transcript's tail.
        sink.add([{"type": "transcript", "text": "x" * 200000}])
        sink.write()
        self.assertGreater(sink.items[0][2], 0)
        self.assertLess(sink.items[0][2], len(sink.items[0][1]))
        with self.assertRaisesRegex(ContractError, "retired during handoff"):
            sink.retire()
        self.assertFalse(sink.items)
        self.assertEqual(sink.bytes, 0)


if __name__ == "__main__":
    unittest.main()
