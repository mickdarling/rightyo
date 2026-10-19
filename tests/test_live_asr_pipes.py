"""Authored local-child tests: no model, microphone, native inference or real speech."""

import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rightyo.live_audio import LiveAudioError, LiveConfig, _terminate, _transcribe


class MemoryRecognizerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.model = self.root / "authored-model"
        self.model.touch()
        self.processes = []

    def recognizer(self, body, **options):
        executable = self.root / "authored-recognizer"
        executable.write_text(f"#!{sys.executable}\n" + body)
        executable.chmod(0o700)
        return LiveConfig(
            "authored-pipe-test",
            executable,
            self.model,
            self.model,
            self.model,
            provenance="synthetic",
            **options,
        )

    def register(self, process):
        self.processes.append(process)

    def assert_retired(self):
        self.assertEqual(len(self.processes), 1)
        process = self.processes[0]
        self.assertIsNotNone(process.poll())
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertFalse(Path(process.args[process.args.index("--file") + 1]).is_absolute())
        self.assertEqual(set(self.root.iterdir()), {self.model, self.root / "authored-recognizer"})

    def test_wav_stdin_json_stdout_and_no_speech_or_transcript_files(self):
        config = self.recognizer(
            "import io, json, os, sys, wave\n"
            "from pathlib import Path\n"
            "assert sys.argv[sys.argv.index('--file')+1] == '-'\n"
            "assert sys.argv[sys.argv.index('--output-file')+1] == '-'\n"
            "assert '--output-json-full' in sys.argv\n"
            "assert Path.cwd() == Path(os.environ['HOME']).resolve()\n"
            "assert Path.cwd() == Path(os.environ['TMPDIR']).resolve()\n"
            "assert list(Path.cwd().iterdir()) == []\n"
            "packet = sys.stdin.buffer.read()\n"
            "assert packet[:4] == b'RIFF' and packet[8:12] == b'WAVE'\n"
            "with wave.open(io.BytesIO(packet), 'rb') as source:\n"
            " metadata = (source.getnchannels(), source.getsampwidth(), source.getframerate())\n"
            " assert metadata == (1,2,16000)\n"
            " assert source.readframes(source.getnframes()) == b'\\x11\\x00' * 1000\n"
            "assert list(Path.cwd().iterdir()) == []\n"
            "sys.stdout.write(json.dumps({'transcription': []}))\n"
        )
        with patch("rightyo.live_audio.Path.open", side_effect=AssertionError("No payload files")):
            self.assertEqual(
                _transcribe(config, b"\x11\x00" * 1000, self.register), {"transcription": []}
            )
        self.assert_retired()

    def test_overflow_drains_while_child_waits_to_receive_large_stdin(self):
        config = self.recognizer(
            "import os, sys\n"
            "from pathlib import Path\n"
            "for _ in range(256): os.write(1, b'x' * 4096)\n"
            "assert len(sys.stdin.buffer.read()) == 192044\n"
            "assert list(Path.cwd().iterdir()) == []\n",
            timeout_seconds=3,
        )
        with patch("rightyo.live_audio.MAX_RESPONSE_BYTES", 256):
            with self.assertRaisesRegex(LiveAudioError, "exceeded output limit"):
                _transcribe(config, bytes(192000), self.register)
        self.assertEqual(self.processes[0].returncode, 0)
        self.assert_retired()

    def test_timeout_reaps_term_ignoring_child_after_stdout_eof(self):
        config = self.recognizer(
            "import os, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "os.close(1)\n"
            "time.sleep(30)\n",
            timeout_seconds=0.3,
        )
        started = time.monotonic()
        with self.assertRaisesRegex(LiveAudioError, "timed out"):
            _transcribe(config, b"\x11\x00" * 1000, self.register)
        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(self.processes[0].returncode, -signal.SIGKILL)
        self.assert_retired()

    def test_cancellation_reaps_blocked_child_and_never_returns_transcript(self):
        stopped = threading.Event()
        config = self.recognizer(
            "import time\ntime.sleep(30)\n", cancelled=stopped.is_set, timeout_seconds=3
        )

        def register(process):
            self.register(process)
            timer = threading.Timer(0.05, stopped.set)
            self.addCleanup(timer.cancel)
            timer.start()

        with self.assertRaisesRegex(LiveAudioError, "stopped"):
            _transcribe(config, bytes(192000), register)
        self.assert_retired()

    def test_register_failure_retires_child_without_disclosing_error_content(self):
        config = self.recognizer("import time\ntime.sleep(30)\n")

        def reject(process):
            self.register(process)
            raise ValueError("authored sensitive detail")

        with self.assertRaisesRegex(LiveAudioError, "^Local recognizer failed$"):
            _transcribe(config, bytes(640), reject)
        self.assert_retired()

    def test_cleanup_failure_still_closes_owned_pipes(self):
        config = self.recognizer("import time\ntime.sleep(30)\n")

        def reject(process):
            self.register(process)
            raise ValueError("authored registration failure")

        def failed_cleanup(process):
            _terminate(process)
            raise subprocess.TimeoutExpired("authored cleanup", 0)

        with patch("rightyo.live_audio._terminate", side_effect=failed_cleanup):
            with self.assertRaisesRegex(LiveAudioError, "timed out"):
                _transcribe(config, bytes(640), reject)
        self.assert_retired()

    def test_invalid_child_json_has_content_free_error(self):
        config = self.recognizer(
            "import sys\nsys.stdin.buffer.read()\nsys.stdout.write('authored sensitive detail')\n"
        )
        with self.assertRaisesRegex(LiveAudioError, "^Local recognizer failed$"):
            _transcribe(config, bytes(640), self.register)
        self.assert_retired()
