"""Microphone lifecycle tests use fake pipes/processes, never a microphone."""

import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rightyo.capture import PCM_CHUNK_BYTES, CaptureError, MacMicrophoneCapture


class FakePipe:
    def __init__(self):
        self.items = queue.Queue()
        self.closed = False

    def read(self, size):
        value = self.items.get(timeout=3)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        if not self.closed:
            self.closed = True
            self.items.put(b"")


class FakeProcess:
    def __init__(self, returncode=0):
        self.pid = 12345
        self.stdout = FakePipe()
        self.returncode = returncode
        self.waits = []

    def wait(self, timeout):
        self.waits.append(timeout)
        self.stdout.close()
        return self.returncode


def await_condition(condition):
    deadline = time.monotonic() + 3
    while not condition() and time.monotonic() < deadline:
        threading.Event().wait(0.005)
    if not condition():
        raise AssertionError("Synthetic capture thread did not finish")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.helper = Path(self.directory.name) / "synthetic-helper"
        self.helper.write_text("not executed")
        self.helper.chmod(0o700)

    def start_fake(self, process=None, *, bound=None):
        process = process or FakeProcess()
        popen = self.enterContext(patch("rightyo.capture.subprocess.Popen", return_value=process))
        kill = self.enterContext(patch("rightyo.capture.os.killpg"))
        capture = (
            MacMicrophoneCapture(self.helper)
            if bound is None
            else MacMicrophoneCapture(self.helper, queue_chunks=bound)
        )
        self.addCleanup(capture.stop)
        capture.start()
        return capture, process, popen, kill

    def test_construction_does_not_request_permission_or_spawn(self):
        with patch("rightyo.capture.subprocess.Popen") as popen:
            capture = MacMicrophoneCapture(self.helper)
            popen.assert_not_called()
            with self.assertRaisesRegex(CaptureError, "not running"):
                capture.read()
            capture.stop()

    def test_start_uses_explicit_capture_and_isolates_credentials(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": "synthetic-secret", "HTTPS_PROXY": "x"}):
            capture, _, popen, _ = self.start_fake()
        arguments, options = popen.call_args
        self.assertEqual(arguments[0], [str(self.helper.resolve()), "--capture"])
        self.assertEqual(set(options["env"]), {"PATH", "HOME", "TMPDIR"})
        self.assertNotIn("synthetic-secret", str(popen.call_args))
        self.assertEqual(options["stderr"], subprocess.DEVNULL)
        self.assertTrue(options["start_new_session"])
        self.assertIsNone(capture.read(timeout=0))

    def test_pcm_odd_pipe_boundaries_preserve_samples(self):
        capture, process, _, _ = self.start_fake()
        payload = bytes(range(256)) * 50
        for offset in range(0, len(payload), 683):
            process.stdout.items.put(payload[offset : offset + 683])
        self.assertEqual(capture.read(timeout=1), payload[:PCM_CHUNK_BYTES])
        self.assertEqual(capture.read(timeout=1), payload[PCM_CHUNK_BYTES:])

    def test_thirty_second_pause_with_short_pipe_reads_preserves_every_sample(self):
        capture, process, _, _ = self.start_fake()
        payload = bytes(range(256)) * 3750  # 960,000 bytes: 30 seconds of PCM16.
        for offset in range(0, len(payload), 682):
            process.stdout.items.put(payload[offset : offset + 682])
        await_condition(lambda: capture._queue.qsize() == 150)
        self.assertEqual(capture._queue.maxsize, 160)
        self.assertIsNone(capture._failure)
        self.assertFalse(capture._stopped.is_set())
        received = b"".join(capture.read(timeout=1) for _ in range(150))
        self.assertEqual(received, payload)
        capture.stop()
        self.assertTrue(capture._queue.empty())

    def test_overflow_fails_closed_and_discards_samples(self):
        capture, process, _, kill = self.start_fake(bound=1)
        process.stdout.items.put(b"\x00" * PCM_CHUNK_BYTES)
        process.stdout.items.put(b"\x00" * PCM_CHUNK_BYTES)
        await_condition(lambda: capture._stopped.is_set())
        await_condition(lambda: capture._reader is not None and not capture._reader.is_alive())
        with self.assertRaisesRegex(CaptureError, "buffer bound"):
            capture.read()
        self.assertTrue(capture._queue.empty())
        kill.assert_any_call(process.pid, signal.SIGTERM)

    def test_default_capacity_still_fails_closed_on_sustained_short_reads(self):
        capture, process, _, kill = self.start_fake()
        payload = b"\x00" * (PCM_CHUNK_BYTES * 161)
        for offset in range(0, len(payload), 682):
            process.stdout.items.put(payload[offset : offset + 682])
        await_condition(lambda: capture._stopped.is_set())
        await_condition(lambda: capture._reader is not None and not capture._reader.is_alive())
        with self.assertRaisesRegex(CaptureError, "buffer bound"):
            capture.read()
        self.assertTrue(capture._queue.empty())
        kill.assert_any_call(process.pid, signal.SIGTERM)

    def test_native_failure_codes_are_sanitized(self):
        for code, expected in ((12, "denied"), (13, "No microphone"), (16, "input changed")):
            with self.subTest(code=code):
                capture, process, _, _ = self.start_fake(FakeProcess(code))
                process.stdout.items.put(b"")
                await_condition(lambda: capture._failure is not None)
                with self.assertRaisesRegex(CaptureError, expected):
                    capture.read()

    def test_pipe_failure_does_not_expose_sensitive_output(self):
        capture, process, _, _ = self.start_fake()
        process.stdout.items.put(OSError("synthetic-sensitive-device-name"))
        await_condition(lambda: capture._failure is not None)
        with self.assertRaises(CaptureError) as error:
            capture.read()
        self.assertNotIn("synthetic-sensitive", str(error.exception))
        self.assertIsNone(error.exception.__context__)

    def test_stop_clears_queue_and_private_directory(self):
        capture, process, _, kill = self.start_fake()
        process.stdout.items.put(b"\x00" * PCM_CHUNK_BYTES)
        process.stdout.items.put(b"\x01\x02\x03")  # An incomplete chunk is discarded too.
        await_condition(lambda: not capture._queue.empty())
        directory = Path(capture._directory.name)
        capture.stop()
        self.assertFalse(directory.exists())
        self.assertTrue(capture._queue.empty())
        self.assertFalse(capture._reader.is_alive())
        kill.assert_any_call(process.pid, signal.SIGTERM)
        with self.assertRaisesRegex(CaptureError, "not running"):
            capture.read()
        with self.assertRaisesRegex(CaptureError, "cannot be restarted"):
            capture.start()

    def test_stop_discards_incomplete_chunk_without_emitting_short_pcm(self):
        capture, process, _, _ = self.start_fake()
        process.stdout.items.put(b"\x01\x02\x03")
        await_condition(process.stdout.items.empty)
        self.assertIsNone(capture.read(timeout=0.02))
        capture.stop()
        self.assertTrue(capture._queue.empty())
        self.assertFalse(capture._reader.is_alive())

    def test_stop_escalates_to_kill_for_unresponsive_helper(self):
        process = FakeProcess()
        capture, _, _, kill = self.start_fake(process)
        with patch.object(
            process, "wait", side_effect=[subprocess.TimeoutExpired("synthetic", 1), 0]
        ):
            capture.stop()
        kill.assert_any_call(process.pid, signal.SIGKILL)

    def test_missing_helper_and_spawn_failure_are_sanitized(self):
        capture = MacMicrophoneCapture(Path(self.directory.name) / "missing")
        with self.assertRaisesRegex(CaptureError, "existing executable"):
            capture.start()
        with patch("rightyo.capture.subprocess.Popen", side_effect=OSError("sensitive")):
            capture = MacMicrophoneCapture(self.helper)
            with self.assertRaises(CaptureError) as error:
                capture.start()
            self.assertNotIn("sensitive", str(error.exception))
            self.assertIsNone(error.exception.__context__)
            self.assertIsNone(capture._directory)

    def test_bounds_and_timeout_are_validated(self):
        for invalid in (True, 0, 257, float("nan")):
            with self.subTest(bound=invalid), self.assertRaises(CaptureError):
                MacMicrophoneCapture(self.helper, queue_chunks=invalid)
        capture, _, _, _ = self.start_fake()
        for invalid in (True, -1, 6, float("nan"), float("inf"), "1"):
            with self.subTest(timeout=invalid), self.assertRaises(CaptureError):
                capture.read(timeout=invalid)


if __name__ == "__main__":
    unittest.main()
