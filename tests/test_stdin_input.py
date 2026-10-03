"""Stdin PCM input (#64) uses synthetic bytes, fake processors, and no audio devices."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rightyo.capture import PCM_CHUNK_BYTES, StdinPcmCapture
from rightyo.contracts import Turn
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.tool import listen


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


def drain(capture, gaps=None):
    chunks = []
    while True:
        item = capture.read(timeout=1)
        if item == b"":
            return chunks
        if item:
            gap, chunk = item
            if gaps is not None:
                gaps.append(gap)
            chunks.append(chunk)


class Processor:
    instances = []

    def __init__(self, config, callback):
        self.config, self.callback = config, callback
        self.received = bytearray()
        self.gaps = []
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

    def mark_gap(self, dropped_ms):
        self.gaps.append((len(self.received), dropped_ms))

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
        self.assertEqual((capture.gaps, capture.dropped_bytes), (0, 0))
        self.assertEqual(len(messages), 1)

    def test_back_pressure_drops_counted_chunks_instead_of_buffering(self):
        messages = []
        capture = StdinPcmCapture(
            io.BytesIO(bytes(10 * PCM_CHUNK_BYTES)), queue_chunks=2, report=messages.append
        )
        capture.start()
        capture._reader.join(timeout=5)  # Consumer stalled for the whole input.
        self.assertEqual(capture._queue.qsize(), 2)
        self.assertEqual(capture.dropped_bytes, 8 * PCM_CHUNK_BYTES)
        self.assertEqual(capture.gaps, 1)
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(drain(capture)), 2)

    def test_first_chunk_after_a_drop_carries_the_gap(self):
        pieces = [bytes([n]) * PCM_CHUNK_BYTES for n in range(5)]
        stream = Gated(pieces, gate_at=4)
        capture = StdinPcmCapture(stream, queue_chunks=2, report=lambda _message: None)
        capture.start()
        self.assertTrue(stream.paused.wait(5))
        gaps = []
        first = [capture.read(timeout=1), capture.read(timeout=1)]
        stream.release.set()
        rest = drain(capture, gaps)
        self.assertEqual([gap for gap, _chunk in first], [0, 0])
        self.assertEqual([chunk[0] for _gap, chunk in first], [0, 1])
        self.assertEqual([chunk[0] for chunk in rest], [4])
        self.assertEqual(gaps, [2 * PCM_CHUNK_BYTES])
        self.assertEqual((capture.gaps, capture.dropped_bytes), (1, 2 * PCM_CHUNK_BYTES))

    def test_stop_discards_pending_audio(self):
        capture = StdinPcmCapture(io.BytesIO(bytes(3 * PCM_CHUNK_BYTES)))
        capture.start()
        capture._reader.join(timeout=5)
        capture.stop()
        self.assertEqual(capture._queue.qsize(), 0)


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

    def test_dropped_audio_marks_a_gap_and_advances_stream_time(self):
        chunk = PCM_CHUNK_BYTES
        stream = Gated([bytes(chunk)] * 6, gate_at=5)
        output, diagnostics = io.StringIO(), io.StringIO()
        result, captures = {}, []

        def capture_factory(*args, **kwargs):
            captures.append(StdinPcmCapture(*args, queue_chunks=2, **kwargs))
            return captures[-1]

        def run():
            with contextlib.redirect_stderr(diagnostics):
                result["code"] = listen(
                    self.args, output=output, controller_factory=self.factory, audio_input=stream
                )

        # Five chunks race a slow consumer and a queue of two; once the consumer has
        # caught up, a sixth chunk arrives after the gap, then EOF.
        with patch("rightyo.prototype.StdinPcmCapture", capture_factory):
            with patch.object(Processor, "push_pcm16", slow_push(Processor.push_pcm16)):
                thread = threading.Thread(target=run)
                thread.start()
                self.assertTrue(stream.paused.wait(5))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and (
                    not Processor.instances
                    or len(Processor.instances[0].received) + captures[0].dropped_bytes < 5 * chunk
                ):
                    time.sleep(0.01)
                stream.release.set()
                thread.join(10)
        self.assertEqual(result.get("code"), 0)
        (processor,) = Processor.instances
        dropped = captures[0].dropped_bytes
        self.assertGreater(dropped, 0)
        self.assertEqual(processor.gaps, [(5 * chunk - dropped, dropped // 32)])
        self.assertEqual(len(processor.received), 6 * chunk - dropped)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["phase"], "stopped")
        self.assertEqual(events[-1]["input_gaps"]["dropped_bytes"], dropped)
        # Stream time counts the dropped audio: six 200 ms chunks were spoken.
        self.assertGreaterEqual(events[-1]["emitted_at_ms"], 1200)
        self.assertIn("dropping", diagnostics.getvalue())


def slow_push(push):
    def slowed(self, pcm):
        time.sleep(0.05)
        push(self, pcm)

    return slowed


if __name__ == "__main__":
    unittest.main()
