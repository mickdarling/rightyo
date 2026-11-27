"""#350. Generated PCM, authored recognition and a stand-in owner. No capture or downloads."""

import contextlib
import io
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from array import array
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import test_privacy_minimum as privacy
from test_live_audio import SILENCE, VOICE
from test_whisper_window_recovery import doc, segment

from rightyo.live_speaker_id import EntryScore
from rightyo.silero_vad import CHUNK_BYTES, CHUNK_MS, VERSION, SileroVad, VadConfig, VadError

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".cache" / f"silero_vad-{VERSION}.onnx"


def tones(*components):
    """100 ms of orthogonal integer-cycle sinusoids, at 16 kHz."""
    return array(
        "h",
        (
            round(sum(amp * math.sin(2 * math.pi * hz * i / 16000) for hz, amp in components))
            for i in range(1600)
        ),
    ).tobytes()


def fake_worker(test, speech, failure_ms=None):
    """A real subprocess returning an authored sequence of decisions for 512-sample reads."""
    directory = tempfile.TemporaryDirectory()
    test.addCleanup(directory.cleanup)
    script = Path(directory.name) / "fixture-python"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "out = sys.stdout.buffer\n"
        "out.write(b'R'); out.flush()\n"
        "ms = 0\n"
        f"while len(sys.stdin.buffer.read({CHUNK_BYTES})) == {CHUNK_BYTES}:\n"
        f"    if ms == {failure_ms!r}: sys.exit(1)\n"
        f"    voiced = any(start <= ms and ms + {CHUNK_MS} <= end for start, end in {speech!r})\n"
        "    out.write(b'1' if voiced else b'0'); out.flush()\n"
        f"    ms += {CHUNK_MS}\n"
    )
    script.chmod(0o700)
    client = SileroVad(VadConfig(script, Path(directory.name) / "unused"))
    test.addCleanup(client.close)
    return client


class Replay:
    controller = privacy.PrivacyMinimumTests.controller
    arrange = privacy.PrivacyMinimumTests.arrange
    feed = privacy.PrivacyMinimumTests.feed
    replay_private_windows = privacy.PrivacyMinimumTests.replay_private_windows


class FailureTests(Replay, unittest.TestCase):
    def test_startup_failures_keep_privacy_on_and_report_fixed_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "private-path"
            wrong = root / "wrong-weights.onnx"
            wrong.write_bytes(b"authored invalid model")
            python = Path(sys.executable)
            for settings, reason in (
                (None, "model and worker not configured"),
                (VadConfig(python, missing), "model missing or unreadable"),
                (VadConfig(python, root), "model missing or unreadable"),
                (VadConfig(python, wrong), "model checksum mismatch"),
                (VadConfig(missing, wrong), "worker unavailable"),
            ):
                with self.subTest(reason=reason):
                    controller = self.controller(
                        **(
                            {
                                "silero_vad": {
                                    "python": str(settings.python),
                                    "model": str(settings.model),
                                }
                            }
                            if settings
                            else {}
                        )
                    )
                    stderr = io.StringIO()
                    with contextlib.redirect_stderr(stderr):
                        vad = controller._silero_vad(threading.Event())
                    self.assertIsNone(vad)
                    self.assertEqual(
                        stderr.getvalue(),
                        f"rightyo: Silero VAD {reason}; voice privacy exit disabled\n",
                    )
                    command = doc(segment("Jarvis, privacy off.", 0, 100))
                    self.assertEqual(self.replay_private_windows(command, vad=vad), ([], 0))
                    status = doc(segment("Are you recording?", 0, 100))
                    self.assertEqual(self.replay_private_windows(status, vad=vad), (["status"], 0))

    def test_failure_after_qualified_onset_revokes_candidate_and_following_exits(self):
        worker = fake_worker(self, [(320, 640)], failure_ms=640)
        command = doc(segment("Jarvis, privacy off.", 20, 340))
        stream = SILENCE * 16 + VOICE * 16 + SILENCE * 80
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(
                self.replay_private_windows(command, command, audio=[stream, stream], vad=worker),
                ([], 0),
            )
        self.assertEqual(
            stderr.getvalue(), "rightyo: Silero VAD worker failed; voice privacy exit disabled\n"
        )

    def test_worker_lost_after_scoring_cannot_release_an_exit(self):
        worker = fake_worker(self, [(320, 640)])
        command = doc(segment("Jarvis, privacy off.", 20, 340))

        def end_turn(_pcm):
            worker.close()
            return 1.0

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(
                self.replay_private_windows(
                    command,
                    vad=worker,
                    end_of_turn=end_turn,
                    end_of_turn_silence_ms=20,
                    audio=[SILENCE * 16 + VOICE * 16 + bytes(32 * 32)],
                ),
                ([], 0),
            )
        self.assertEqual(
            stderr.getvalue(), "rightyo: Silero VAD worker failed; voice privacy exit disabled\n"
        )

    def test_unknown_worker_reply_cannot_supply_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "fixture"
            script.write_text("#!/bin/sh\nprintf 'R?'\n")
            script.chmod(0o700)
            worker = SileroVad(VadConfig(script, script))
            self.addCleanup(worker.close)
            with self.assertRaisesRegex(VadError, "^worker unavailable$"):
                worker.is_speech(bytes(CHUNK_BYTES))


class AdversarialReplay(Replay):
    """The identical waveforms and recognition/owner gates run with either worker."""

    def exercise(self, pcm, speech, command, *, expected, machinery_ms=0):
        vad = self.worker(speech)
        calls, labels = [], []
        classify = vad.is_speech

        def observe(block):
            label = classify(block)
            labels.append(label)
            return label

        vad.is_speech = observe
        c, e, s, p, _, units = self.arrange(vad=vad, pre_roll_ms=400)
        c._enrolled_roles.exit_audio = c._private_sentence
        s.exit_score = Mock(return_value=EntryScore(0.9, None, 0, "scored"))
        p.config = replace(
            p.config,
            on_private_turn_audio=lambda t, pcm: c._private_turn_audio(c._generation, t, pcm),
        )
        c.duck(True)
        e.drain()

        def decode(audio, register=None):
            start = p._utterance_start
            end = start + len(audio) // 32
            calls.append((start, end))
            if start <= command[0] and end >= command[1]:
                return [
                    {
                        "start_ms": command[0] - start,
                        "end_ms": command[1] - start,
                        "text": "Jarvis, privacy off.",
                    }
                ]
            return []

        units.transcribe = decode
        self.feed(c, p, pcm, s)
        p.finish()
        self.assertIs(p._vad, vad, "worker failure is not an adversarial result")
        self.assertEqual(len(labels), len(pcm) // CHUNK_BYTES)
        self.assertEqual(
            ([event["action"] for event in e.drain()], s.exit_score.call_count), expected
        )
        self.assertFalse(c._private_sentence.pcm)
        self.assertFalse(
            any(end <= machinery_ms for _, end in calls), "steady machinery opened an ASR window"
        )
        return labels, calls

    def test_persistent_fundamental_with_louder_third_harmonic(self):
        # The 100 Hz voice stays 20 dB above the 50 Hz background throughout.
        base = ((50, 100), (100, 1000), (200, 80))
        voice, louder = tones(*base, (300, 50)), tones(*base, (300, 3000))
        labels, _ = self.exercise(
            voice * 4 + louder * 10 + voice * 18, [(0, 3200)], (400, 1400), expected=([], 0)
        )
        self.assertNotIn(
            b"\0" * 7,
            bytes(labels[: 400 // CHUNK_MS]),
            "VAD certified 200 ms of absence during the continuous voiced lead",
        )

    def test_omitted_low_pitched_tail_at_1_47_times_fan_rms(self):
        fan = (50, 1000)
        command = tones(fan, (200, 3000), (400, 1000))
        amp = 1000 * math.sqrt((1.465**2 - 1) / (1 + 0.2**2 + 0.1**2))
        tail = tones(fan, (80, amp), (160, 0.2 * amp), (240, 0.1 * amp))
        prefix = tones(fan) * 2 + command * 10
        with self.subTest(case="clean command over the same fan"):
            self.exercise(
                prefix + tones(fan) * 16, [(200, 1200)], (200, 1200), expected=(["exit"], 1)
            )
        audio = prefix + tail * 20 + tones(fan) * 16
        labels, calls = self.exercise(audio, [(200, 3200)], (200, 1200), expected=([], 0))
        self.assertTrue(
            any(labels[1200 // CHUNK_MS : 3200 // CHUNK_MS]), "VAD missed the voiced tail"
        )
        self.assertFalse(any(end < 3200 for _, end in calls), "ASR ended while speech continued")

    def machinery(self, hz):
        machine = tones((hz, 1000))
        # Both acoustic boundaries coincide with native VAD and ASR frame edges.
        command = tones((hz, 1000), (100, 3000), (300, 1200))
        self.exercise(
            machine * 240 + command * 16 + machine * 16,
            [(24000, 25600)],
            (24000, 25600),
            machinery_ms=24000,
            expected=(["exit"], 1),
        )

    def test_clean_command_over_200_hz_steady_machinery(self):
        self.machinery(200)

    def test_clean_command_sharing_a_fundamental_with_the_fan(self):
        self.machinery(100)


class FakeWorkerDecisions(AdversarialReplay, unittest.TestCase):
    def worker(self, speech):
        return fake_worker(self, speech)

    def test_incomplete_last_vad_block_cannot_supply_an_ending_pause(self):
        command = doc(segment("Jarvis, privacy off.", 20, 340))
        for pause_ms, expected in ((20, ([], 0)), (32, (["exit"], 1))):
            with self.subTest(pause_ms=pause_ms):
                self.assertEqual(
                    self.replay_private_windows(
                        command,
                        vad=self.worker([(320, 640)]),
                        audio=[SILENCE * 16 + VOICE * 16 + bytes(pause_ms * 32)],
                        end_of_turn=lambda _: 1,
                        end_of_turn_silence_ms=20,
                    ),
                    expected,
                )

    def test_uneven_pushes_deliver_each_original_native_block_once(self):
        vad = self.worker([])
        vad.is_speech = Mock(wraps=vad.is_speech)
        c, _, _, p, _, _ = self.arrange(vad=vad)
        c.duck(True)
        pcm = tones((100, 3456)) * 3
        for i in range(0, len(pcm), 222):
            p.push_pcm16(pcm[i : i + 222])
        p.finish()
        blocks = [args.args[0] for args in vad.is_speech.call_args_list]
        self.assertEqual(
            blocks,
            [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm) - CHUNK_BYTES + 1, CHUNK_BYTES)],
        )

    def test_partial_native_block_cannot_authenticate_a_word(self):
        for name_end, expected in ((52, (["exit"], 1)), (21, ([], 0))):
            with self.subTest(name_end=name_end):
                worker = self.worker([(320, 640)])
                words = doc(
                    segment("Jarvis,", 20, name_end),
                    segment(" privacy", name_end, 84),
                    segment(" off.", 84, 340),
                )
                self.assertEqual(
                    self.replay_private_windows(
                        words, vad=worker, audio=[SILENCE * 16 + VOICE * 16 + SILENCE * 80]
                    ),
                    expected,
                )


class RealModelDecisions(AdversarialReplay, unittest.TestCase):
    def setUp(self):
        python = Path(os.environ.get("RIGHTYO_SILERO_PYTHON", sys.executable))
        model = Path(os.environ.get("RIGHTYO_SILERO_MODEL", MODEL))
        if not model.is_file():
            self.skipTest("pinned Silero ONNX model absent; see docs/prototype.md")
        if not python.is_file():
            self.skipTest("Silero worker interpreter absent; set RIGHTYO_SILERO_PYTHON")
        probe = subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "from importlib.util import find_spec; "
                "print(' '.join(n for n in ('numpy', 'onnxruntime') if find_spec(n) is None))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if missing := probe.stdout.strip():
            self.skipTest(f"Silero worker missing {missing}; set RIGHTYO_SILERO_PYTHON")
        self.settings = VadConfig(python, model)

    def worker(self, speech):
        worker = SileroVad(self.settings)
        self.addCleanup(worker.close)
        return worker
