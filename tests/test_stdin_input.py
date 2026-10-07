"""Stdin PCM input (#64) uses synthetic bytes, fake processors, and no audio devices."""

from __future__ import annotations

import array
import contextlib
import io
import json
import math
import os
import selectors
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import rightyo
from rightyo.capture import PCM_CHUNK_BYTES, StdinPcmCapture
from rightyo.contracts import Turn
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents


class Pieces:
    """A binary stream whose reads return deliberately awkward (odd) lengths."""

    def __init__(self, data: bytes, sizes):
        self.data, self.sizes, self.index = data, list(sizes), 0

    def read(self, _limit):
        if not self.data:
            return b""
        size = self.sizes[self.index % len(self.sizes)]
        self.index += 1
        piece, self.data = self.data[:size], self.data[size:]
        return piece


class Gated:
    """Reads pause at ``gate_at`` until the test releases them (a consumer catching up)."""

    def __init__(self, pieces, gate_at):
        self.pieces, self.gate_at, self.index = list(pieces), gate_at, 0
        self.paused, self.release = threading.Event(), threading.Event()

    def read(self, _limit):
        if self.index == self.gate_at:
            self.paused.set()
            self.release.wait(5)
        if self.index >= len(self.pieces):
            return b""
        self.index += 1
        return self.pieces[self.index - 1]


def drain(capture):
    chunks = []
    while True:
        chunk = capture.read(timeout=1)
        if chunk == b"":
            return chunks
        if chunk:
            chunks.append(chunk)


class Processor:
    instances = []

    def __init__(self, config, callback):
        self.config, self.callback = config, callback
        self.received = bytearray()
        self.finished = self.closed = False
        self.instances.append(self)

    def push_pcm16(self, pcm):
        self.received += pcm
        index = len(self.received) // PCM_CHUNK_BYTES
        if len(self.received) % PCM_CHUNK_BYTES == 0 and index % 5 == 0:
            self.callback(
                Turn(
                    self.config.session_id,
                    f"stdin-{index}",
                    1,
                    (index - 1) * 200,
                    index * 200,
                    "Synthetic stdin utterance.",
                    "Speaker A",
                    True,
                    False,
                    "authored-fixture",
                    self.config.provenance,
                    "authored-fixture",
                )
            )

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True


class StdinCaptureTests(unittest.TestCase):
    def test_odd_reads_are_regrouped_and_a_final_odd_byte_is_discarded(self):
        data = bytes(range(256)) * 60 + b"\x01"  # 15,360 bytes of samples plus one byte
        messages = []
        capture = StdinPcmCapture(Pieces(data, [1, 333, 4097, 7]), report=messages.append)
        capture.start()
        chunks = drain(capture)
        self.assertEqual(b"".join(chunks), data[:-1])
        self.assertTrue(all(len(chunk) == PCM_CHUNK_BYTES for chunk in chunks[:-1]))
        self.assertEqual(capture.discarded_tail_bytes, 1)
        self.assertEqual((capture.overrun, capture.dropped_bytes), (False, 0))
        self.assertEqual(len(messages), 1)

    def test_overrun_stops_reading_and_never_delivers_audio_after_the_drop(self):
        messages = []
        stream = io.BytesIO(b"".join(bytes([n]) * PCM_CHUNK_BYTES for n in range(10)))
        capture = StdinPcmCapture(stream, queue_chunks=2, report=messages.append)
        capture.start()
        capture._reader.join(timeout=5)  # Consumer stalled for the whole input.
        self.assertTrue(capture.overrun)
        self.assertEqual(capture.dropped_bytes, PCM_CHUNK_BYTES)
        self.assertEqual(stream.tell(), 3 * PCM_CHUNK_BYTES)  # Reading stopped at the drop.
        self.assertEqual(len(messages), 1)
        # Only the contiguous audio queued before the drop is delivered, then end of input.
        self.assertEqual([chunk[0] for chunk in drain(capture)], [0, 1])

    def test_stop_discards_pending_audio(self):
        capture = StdinPcmCapture(io.BytesIO(bytes(3 * PCM_CHUNK_BYTES)))
        capture.start()
        capture._reader.join(timeout=5)
        capture.stop()
        self.assertEqual(capture._queue.qsize(), 0)

    def test_descriptor_reader_stops_promptly_and_is_joined_while_the_pipe_stays_open(self):
        # #74/#78: a reader blocked on an open stdin must not survive stop(), or the
        # interpreter aborts at shutdown on the stdin buffer lock (exit 134).
        read_end, write_end = os.pipe()
        self.addCleanup(os.close, write_end)
        with os.fdopen(read_end, "rb") as stream:
            capture = StdinPcmCapture(stream)
            capture.start()
            self.assertFalse(capture._reader.daemon)
            os.write(write_end, bytes(PCM_CHUNK_BYTES))
            self.assertEqual(capture.read(timeout=5), bytes(PCM_CHUNK_BYTES))
            self.assertIsNone(capture.read(timeout=0.05))  # Pipe open, nothing sent.
            started = time.monotonic()
            capture.stop()
            self.assertLess(time.monotonic() - started, 2)
            self.assertFalse(capture._reader.is_alive())
            # The reader never touched the BufferedReader, so nothing is lost or held.
            os.write(write_end, b"later")
            self.assertEqual(stream.read1(16), b"later")

    def test_descriptor_reader_delivers_audio_then_end_of_input(self):
        read_end, write_end = os.pipe()
        with os.fdopen(read_end, "rb") as stream:
            capture = StdinPcmCapture(stream)
            capture.start()
            os.write(write_end, bytes(range(256)) * 50 + b"\x01")  # 12,800 bytes + 1
            os.close(write_end)
            chunks = drain(capture)
            capture.stop()
        self.assertEqual(b"".join(chunks), bytes(range(256)) * 50)
        self.assertEqual(capture.discarded_tail_bytes, 1)


class StdinListenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        asset = root / "asset"
        asset.touch()
        self.config = root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    key: str(asset)
                    for key in (
                        "whisper_executable",
                        "whisper_model",
                        "diarization_library",
                        "diarization_model",
                        "microphone_helper",
                    )
                }
            )
        )
        Processor.instances.clear()
        self.args = Namespace(
            config=self.config,
            mode="stdin",
            provenance="causal-replay",
            session_id="stdin-test",
            use_jev=False,
            allow_hosted=False,
        )

    @staticmethod
    def factory(config, *, event_publisher, audio_input, audio_provenance, report):
        def no_device(*_args, **_kwargs):
            raise AssertionError("stdin mode must not start the microphone helper")

        return PrototypeController(
            config,
            event_publisher=event_publisher,
            processor_factory=Processor,
            capture_factory=no_device,
            provider_factory=no_device,
            audio_input=audio_input,
            audio_provenance=audio_provenance,
            report=report,
        )

    def test_stdin_pcm_feeds_live_pipeline_and_eof_stops_cleanly(self):
        pcm = bytes(range(256)) * 250 + b"\x7f"  # 64,000 bytes = 2 s, plus one odd byte
        output, diagnostics = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(diagnostics):
            result = listen(
                self.args,
                output=output,
                controller_factory=self.factory,
                audio_input=Pieces(pcm, [4095, 1, 6401]),
            )
        self.assertIn("discarded the final byte", diagnostics.getvalue())
        self.assertEqual(result, 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            [e["type"] for e in events], ["session"] + ["transcript"] * 2 + ["session"]
        )
        self.assertEqual(
            events[0]["audio_input"],
            {"source": "stdin", "encoding": "s16le", "sample_rate": 16000, "channels": 1},
        )
        self.assertEqual(
            sorted(events[0]["capabilities"]), ["activation", "context", "partials", "speakers"]
        )
        self.assertTrue(all(e["turn"]["provenance"] == "causal-replay" for e in events[1:3]))
        self.assertEqual(events[-1]["phase"], "stopped")
        self.assertEqual(
            events[-1]["input_gaps"], {"gaps": 0, "dropped_bytes": 0, "discarded_tail_bytes": 1}
        )
        (processor,) = Processor.instances
        self.assertEqual(bytes(processor.received), pcm[:-1])
        self.assertTrue(processor.finished and processor.closed)

    def test_stdin_mode_never_reaches_the_web_lab_without_a_stream(self):
        config = PrototypeConfig.load(self.config)
        controller = PrototypeController(config, processor_factory=Processor)
        self.addCleanup(controller.close)
        with self.assertRaisesRegex(PrototypeError, "listen command"):
            controller.start({"mode": "stdin"})
        self.assertEqual(Processor.instances, [])

    def test_provenance_is_required_with_stdin_and_refused_without_it(self):
        for mode, provenance in (("stdin", None), ("demo", "live-microphone")):
            args = Namespace(**{**vars(self.args), "mode": mode, "provenance": provenance})
            with self.assertRaisesRegex(PrototypeError, "--provenance"):
                listen(args, controller_factory=self.factory, audio_input=io.BytesIO())
        config = PrototypeConfig.load(self.config)
        controller = PrototypeController(
            config, processor_factory=Processor, audio_input=io.BytesIO()
        )
        self.addCleanup(controller.close)
        with self.assertRaisesRegex(PrototypeError, "provenance"):
            controller.start({"mode": "stdin"})
        self.assertEqual(Processor.instances, [])

    def _stalled_session(self, provenance, wait_seconds):
        """A producer that sends nothing for ``wait_seconds`` under a 1 s session budget."""
        stream = Gated([bytes(PCM_CHUNK_BYTES)], gate_at=0)
        self.addCleanup(stream.release.set)
        args = Namespace(**{**vars(self.args), "provenance": provenance, "session_budget": 1})
        output, result = io.StringIO(), {}
        thread = threading.Thread(
            target=lambda: result.setdefault(
                "code",
                listen(args, output=output, controller_factory=self.factory, audio_input=stream),
            )
        )
        thread.start()
        thread.join(wait_seconds)
        finished_while_stalled = not thread.is_alive()
        stream.release.set()
        thread.join(10)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        return finished_while_stalled, result.get("code"), events

    def test_live_stdin_session_budget_follows_the_wall_clock(self):
        stalled, code, events = self._stalled_session("live-microphone", 4)
        self.assertTrue(stalled)
        self.assertEqual((code, events[-1]["phase"]), (0, "cancelled"))

    def test_replay_stdin_session_budget_follows_media_time_not_the_wall_clock(self):
        stalled, code, events = self._stalled_session("causal-replay", 2.5)
        self.assertFalse(stalled)
        # Released: 200 ms of media fits the 1 s budget, so EOF ends it normally.
        self.assertEqual((code, events[-1]["phase"]), (0, "stopped"))
        self.assertLess(events[-1]["emitted_at_ms"], 1000)

    def _overrun_session(self, chunks, queue_chunks, delay, during=None):
        output, diagnostics, captures = io.StringIO(), io.StringIO(), []

        def capture_factory(*args, **kwargs):
            captures.append(StdinPcmCapture(*args, queue_chunks=queue_chunks, **kwargs))
            return captures[-1]

        # A fast finite input into a small queue, with a slow consumer.
        with patch("rightyo.prototype.StdinPcmCapture", capture_factory):
            with patch.object(Processor, "push_pcm16", slow_push(Processor.push_pcm16, delay)):
                with patch.object(Processor, "finish", clipped_finish):
                    if during is not None:
                        threading.Thread(target=during, args=(captures,), daemon=True).start()
                    with contextlib.redirect_stderr(diagnostics):
                        code = listen(
                            self.args,
                            output=output,
                            controller_factory=self.factory,
                            audio_input=io.BytesIO(bytes(chunks * PCM_CHUNK_BYTES)),
                        )
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        return code, captures[0], events, diagnostics.getvalue()

    def test_overrun_discards_the_clipped_utterance_and_fails_closed(self):
        code, capture, events, diagnostics = self._overrun_session(20, 6, 0.05)
        self.assertEqual(code, 2)
        (processor,) = Processor.instances
        self.assertTrue(capture.overrun)
        # The clipped open utterance is never finalized; turns finalized before it stand.
        self.assertFalse(processor.finished)
        self.assertTrue(processor.closed)
        texts = [e["turn"]["utterance_id"] for e in events if e["type"] == "transcript"]
        self.assertEqual(texts, ["stdin-5"])
        self.assertFalse(any(e["type"] in {"attention", "request"} for e in events))
        self.assertEqual(
            (events[-1]["type"], events[-1]["phase"], events[-1]["reason"]),
            ("session", "error", "input-overrun"),
        )
        self.assertEqual(
            events[-1]["input_gaps"],
            {"gaps": 1, "dropped_bytes": capture.dropped_bytes, "discarded_tail_bytes": 0},
        )
        self.assertIn("overran", diagnostics)

    def test_sigterm_after_an_overrun_still_reports_the_overrun(self):
        def terminate_after_overrun(captures):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and not (captures and captures[0].overrun):
                time.sleep(0.01)
            os.kill(os.getpid(), signal.SIGTERM)

        # The consumer is still working through the queue when SIGTERM arrives.
        code, capture, events, _ = self._overrun_session(10, 2, 0.5, terminate_after_overrun)
        self.assertTrue(capture.overrun)
        self.assertEqual(code, 2)
        self.assertEqual((events[-1]["phase"], events[-1]["reason"]), ("error", "input-overrun"))

    def test_an_error_raised_during_the_eof_flush_is_kept(self):
        failed = threading.Event()

        class FailingProvider:
            def __init__(self, **_options):
                self.requests = 0

            def decide(self, state):
                self.requests += 1
                failed.set()
                raise RuntimeError("attention failed")

        class FlushProcessor(Processor):
            def push_pcm16(self, pcm):
                self.received += pcm

            def finish(self):
                # The EOF flush finalizes a turn whose attention fails while it is
                # still flushing; the flush returns only after that error is recorded.
                self.finished = True
                turn = Turn(
                    self.config.session_id,
                    "flush-1",
                    1,
                    0,
                    50,
                    "Rightyo, tell me what happened.",
                    "Speaker A",
                    True,
                    False,
                    "authored-fixture",
                    self.config.provenance,
                    "authored-fixture",
                )
                self.callback(turn)
                if not failed.wait(5):
                    raise AssertionError("attention did not run")
                time.sleep(0.1)

        events = SpeechEvents()
        controller = PrototypeController(
            PrototypeConfig.load(self.config),
            event_publisher=events,
            processor_factory=FlushProcessor,
            provider_factory=FailingProvider,
            audio_input=io.BytesIO(bytes(PCM_CHUNK_BYTES)),
            audio_provenance="causal-replay",
        )
        self.addCleanup(controller.close)
        controller.start({"mode": "stdin", "use_jev": True, "session_id": "stdin-flush"})
        controller._audio_thread.join(5)
        self.assertEqual(controller.snapshot()["phase"], "error")
        controller.stop()
        terminals = [
            e
            for e in controller.drain_events()
            if e["type"] == "session" and e["phase"] != "started"
        ]
        # Exactly one terminal: the error recorded during the flush, never a normal stop.
        self.assertEqual(
            [(e["phase"], e.get("reason")) for e in terminals], [("error", "attention-unavailable")]
        )

    def test_stop_with_a_blocked_reader_never_lets_it_feed_a_later_session(self):
        stream = Gated([bytes(PCM_CHUNK_BYTES)], gate_at=0)
        self.addCleanup(stream.release.set)
        captures = []

        def capture_factory(*args, **kwargs):
            captures.append(StdinPcmCapture(*args, **kwargs))
            return captures[-1]

        config = PrototypeConfig.load(self.config)
        controller = PrototypeController(
            config,
            processor_factory=Processor,
            audio_input=stream,
            audio_provenance="causal-replay",
        )
        self.addCleanup(controller.close)
        with patch("rightyo.prototype.StdinPcmCapture", capture_factory):
            controller.start({"mode": "stdin", "session_id": "stdin-first"})
            self.assertTrue(stream.paused.wait(5))  # The reader is blocked in read().
            controller.stop()
            with self.assertRaisesRegex(PrototypeError, "one session only"):
                controller.start({"mode": "stdin", "session_id": "stdin-second"})
            # The stale reader's late read is discarded, never queued or delivered.
            stream.release.set()
            captures[0]._reader.join(5)
        self.assertFalse(captures[0]._reader.is_alive())
        self.assertEqual(len(captures), 1)
        self.assertEqual(captures[0]._queue.qsize(), 0)
        self.assertEqual([bytes(p.received) for p in Processor.instances], [b""])


# A child `rightyo listen --mode stdin` with model-free stub speech backends patched in.
# The real LiveProcessor, stdin capture, controller and interpreter shutdown all run.
CHILD = """
import sys
from unittest.mock import patch

import rightyo.prototype as prototype
from rightyo.cli import main

mode, config = sys.argv[1], sys.argv[2]


class Transcriber:
    recognizer_id = "stub-recognizer"

    def transcribe(self, pcm, register=None):
        duration = len(pcm) // 32
        if mode == "systemic":
            return {"not": "a unit list"}  # A structural failure still ends the session.
        return [
            {"text": " reversed", "start_ms": 50, "end_ms": 10},
            {"text": " negative", "start_ms": -20, "end_ms": 10},
            {"text": " nan", "start_ms": float("nan"), "end_ms": 10},
            {"text": " Synthetic tone.", "start_ms": 0, "end_ms": duration},
            {"text": " beyond", "start_ms": 0, "end_ms": duration + 5000},
        ]


class Diarizer:
    speaker_provenance = "diarization-timeline"

    def push(self, frame):
        pass

    def segments(self):
        return []

    def finish(self):
        return []

    def close(self):
        pass


with (
    patch.object(prototype, "transcriber_factory", lambda *a, **k: Transcriber()),
    patch.object(prototype, "diarizer_factory", lambda *a, **k: Diarizer()),
):
    code = main(
        ["listen", "--config", config, "--mode", "stdin", "--provenance", "synthetic",
         "--session-id", "stdin-child"]
    )
raise SystemExit(code)
"""
# One second of a 440 Hz synthetic tone, then two seconds of silence (past the hangover).
TONE = array.array(
    "h", (int(8000 * math.sin(2 * math.pi * 440 * n / 16000)) for n in range(16000))
).tobytes()
UTTERANCE = TONE + bytes(2 * 16000 * 2)


class StdinProcessShutdownTests(unittest.TestCase):
    """#74/#78 end to end: no abort at interpreter shutdown, one terminal event, exit code."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        asset = root / "asset"
        asset.touch()
        self.config = root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    key: str(asset)
                    for key in (
                        "whisper_executable",
                        "whisper_model",
                        "diarization_library",
                        "diarization_model",
                        "microphone_helper",
                    )
                }
            )
        )
        self.child = root / "child.py"
        self.child.write_text(CHILD)
        source = str(Path(rightyo.__file__).resolve().parents[1])
        self.env = dict(os.environ, PYTHONPATH=source)

    def launch(self, mode):
        process = subprocess.Popen(
            [sys.executable, str(self.child), mode, str(self.config)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
        )
        self.addCleanup(lambda: process.poll() is None and process.kill())
        return process

    @staticmethod
    def wait_for_line(process, kind, timeout=20):
        """Read stdout event lines until one of ``kind``; return all read so far."""
        events, deadline = [], time.monotonic() + timeout
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while time.monotonic() < deadline:
                if not selector.select(timeout=0.5):
                    continue
                line = process.stdout.readline()
                if not line:
                    break
                events.append(json.loads(line))
                if events[-1]["type"] == kind or events[-1].get("phase") == "error":
                    break
        return events

    def finish(self, process, events):
        stdout, stderr = process.communicate(timeout=30)
        events = events + [json.loads(line) for line in stdout.splitlines()]
        stderr = stderr.decode()
        self.assertNotIn("could not acquire lock", stderr)
        self.assertNotIn("Fatal Python error", stderr)
        terminals = [e for e in events if e["type"] == "session" and e["phase"] != "started"]
        self.assertEqual(len(terminals), 1)
        return events, terminals[0], stderr

    def test_bad_timestamps_then_sigterm_with_stdin_open_exits_cleanly(self):
        process = self.launch("timestamps")
        process.stdin.write(UTTERANCE)
        process.stdin.flush()
        events = self.wait_for_line(process, "transcript")
        self.assertEqual(events[-1]["type"], "transcript")
        # The host still holds stdin open, as Hailing Station does, when it stops RightyO.
        process.send_signal(signal.SIGTERM)
        events, terminal, stderr = self.finish(process, events)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(terminal["phase"], "cancelled")
        transcripts = [e["turn"] for e in events if e["type"] == "transcript"]
        self.assertEqual([t["text"] for t in transcripts], ["Synthetic tone."])
        self.assertLessEqual(transcripts[0]["start_ms"], transcripts[0]["end_ms"])
        self.assertEqual(terminal["skipped_segments"], 4)
        self.assertIn("skipped 4 recognizer segment(s)", stderr)
        for word in ("reversed", "negative", "nan", "beyond"):
            self.assertNotIn(word, stderr)

    def test_bad_timestamps_then_eof_keeps_listening_and_stops_normally(self):
        process = self.launch("timestamps")
        process.stdin.write(UTTERANCE + UTTERANCE)
        process.stdin.close()
        events, terminal, _stderr = self.finish(process, [])
        self.assertEqual(process.returncode, 0)
        self.assertEqual(terminal["phase"], "stopped")
        transcripts = [e["turn"] for e in events if e["type"] == "transcript"]
        self.assertEqual(len(transcripts), 2)  # The session outlived the first bad batch.
        self.assertLess(transcripts[0]["end_ms"], transcripts[1]["start_ms"])
        self.assertEqual(terminal["skipped_segments"], 8)

    def test_systemic_failure_with_stdin_open_reports_error_without_aborting(self):
        process = self.launch("systemic")
        process.stdin.write(UTTERANCE)
        process.stdin.flush()
        events = self.wait_for_line(process, "transcript")
        # The session ends on its own while stdin is still open (the 2026-10-04 crash).
        events, terminal, _stderr = self.finish(process, events)
        self.assertEqual(process.returncode, 2)
        self.assertEqual((terminal["phase"], terminal["reason"]), ("error", "audio-unavailable"))
        self.assertNotIn("skipped_segments", terminal)


def slow_push(push, delay=0.05):
    def slowed(self, pcm):
        time.sleep(delay)
        push(self, pcm)

    return slowed


def clipped_finish(self):
    """A finish that would emit the open utterance; overrun sessions must never call it."""
    self.finished = True
    self.callback(
        Turn(
            self.config.session_id,
            "clipped",
            1,
            0,
            len(self.received) // 32,
            "Rightyo, delete every",
            "Speaker A",
            True,
            False,
            "authored-fixture",
            self.config.provenance,
            "authored-fixture",
        )
    )


if __name__ == "__main__":
    unittest.main()
