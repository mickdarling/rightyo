"""#350. Generated PCM, authored recognition and a stand-in owner. No capture or downloads."""

import contextlib
import io
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from array import array
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import test_privacy_minimum as privacy
from test_live_audio import SILENCE, VOICE
from test_whisper_window_recovery import doc, segment

from rightyo.live_speaker_id import EntryScore, LabelState
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
    def test_tool_api_discloses_inter_block_audio_retention(self):
        contract = " ".join((ROOT / "docs/tool-api.md").read_text().split())
        self.assertIn("last 64 samples", contract)
        self.assertIn("until its next block or shutdown", contract)

    def test_invalid_or_timed_out_readiness_kills_and_reaps_own_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "stalled-worker"
            controller = self.controller(silero_vad={"python": str(script), "model": str(script)})
            spawn = subprocess.Popen
            for invalid in (True, False):
                with self.subTest(invalid=invalid):
                    script.write_text(
                        f"#!{sys.executable}\nimport os, signal\n"
                        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                        + ("os.write(1, b'?')\n" if invalid else "")
                        + "signal.pause()\n"
                    )
                    script.chmod(0o700)
                    children = []

                    def start(*args, **kwargs):
                        child = spawn(*args, **kwargs)
                        children.append(child)
                        return child

                    diagnostic = io.StringIO()
                    try:
                        with (
                            patch("rightyo.silero_vad.subprocess.Popen", side_effect=start),
                            patch("rightyo.silero_vad.READY_TIMEOUT", 1),
                            contextlib.redirect_stderr(diagnostic),
                        ):
                            vad = controller._silero_vad(threading.Event())
                        self.assertIsNone(vad)
                        self.assertEqual(
                            diagnostic.getvalue(),
                            "rightyo: Silero VAD worker unavailable; voice privacy exit disabled\n",
                        )
                        (child,) = children
                        self.assertEqual(child.returncode, -signal.SIGKILL)
                        with self.assertRaises(ChildProcessError):
                            os.waitpid(child.pid, os.WNOHANG)
                        command = doc(segment("Jarvis, privacy off.", 0, 100))
                        self.assertEqual(self.replay_private_windows(command, vad=vad), ([], 0))
                    finally:
                        for child in children:
                            if child.poll() is None:
                                child.kill()
                            child.wait()

    def test_unconfigured_vad_diagnostic_at_session_start(self):
        for roles in (False, True):
            with self.subTest(roles=roles):
                processor = Mock()
                controller = self.controller(
                    constructor={
                        "audio_input": io.BytesIO(),
                        "audio_provenance": "synthetic",
                        "processor_factory": processor,
                        "speaker_id_factory": Mock(),
                    },
                    speakers={"owner": ["owner"]},
                    speaker_id={
                        "python": sys.executable,
                        "model": sys.executable,
                        "live": True,
                        "roles": roles,
                    },
                )
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    controller.start({"mode": "stdin"})
                    controller._audio_thread.join(3)
                self.assertFalse(controller._audio_thread.is_alive())
                processor.assert_called_once()
                self.assertIsNone(processor.call_args.args[0].silero_vad)
                missing = (
                    "rightyo: Silero VAD model and worker not configured; "
                    "voice privacy exit disabled\n"
                )
                self.assertEqual(stderr.getvalue(), missing if roles else "")

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
            # The already-open window finishes without exit evidence. The second
            # replay must not open another window through the energy gate.
            self.assertEqual(
                self.replay_private_windows(command, audio=[stream, stream], vad=worker),
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

    def exercise(self, pcm, speech, command, *, expected, words=None, close_during_check=False):
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
        worker_at_approval = []

        def score(*_):
            if close_during_check:
                vad.close()
                worker_at_approval.append(vad.available)
            return EntryScore(0.9, None, 0, "scored")

        s.exit_score = Mock(side_effect=score)
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
                        "start_ms": left - start,
                        "end_ms": right - start,
                        "text": text,
                    }
                    for left, right, text in words or [(*command, "Jarvis, privacy off.")]
                ]
            return []

        units.transcribe = decode
        self.feed(c, p, pcm, s)
        p.finish()
        if close_during_check:
            self.assertEqual(worker_at_approval, [False])
        else:
            self.assertIs(p._vad, vad, "worker failure is not an adversarial result")
            self.assertEqual(len(labels), len(pcm) // CHUNK_BYTES)
        events = e.drain()
        self.assertTrue(e.ducked)
        self.assertTrue(all(event["type"] == "privacy" for event in events))
        self.assertFalse(e._memory.retained_ids, "private transcript retained")
        self.assertEqual(([event["action"] for event in events], s.exit_score.call_count), expected)
        self.assertFalse(c._private_sentence.pcm)
        return labels, calls

    def clean_timed_control(self, shift, *, expected, close_during_check=False, late=0):
        machine = tones((50, 1000))
        voice = tones((50, 1000), (200, 3000), (400, 1000))
        start, end = 400 + shift, 1400 + shift
        return self.exercise(
            machine * 4 + machine[: shift * 32] + voice * 10 + machine * 16,
            [(416, 672)],
            (start + late, end),
            words=[
                (start + late, start + 100, "Jarvis,"),
                (start + 100, start + 180, " privacy"),
                (start + 180, end, " off."),
            ],
            close_during_check=close_during_check,
            expected=expected,
        )

    def test_worker_dies_inside_owner_verification(self):
        self.clean_timed_control(0, close_during_check=True, expected=([], 1))

    def test_18_ms_shifted_word_onset_in_first_native_speech_block(self):
        labels, _ = self.clean_timed_control(18, expected=(["exit"], 1))
        self.assertEqual(labels.index(True) * CHUNK_MS, 416)
        # Omitting the entire 416-448 ms speech block still refuses.
        self.clean_timed_control(18, late=30, expected=([], 0))

    def test_first_machine_block_and_incomplete_final_block_never_open_private_asr(self):
        for hz, ms, speech in ((200, 24000, [(0, 32)]), (50, 1600, []), (50, 1620, [])):
            with self.subTest(hz=hz, ms=ms):
                pcm = (tones((hz, 1000)) * (ms // 100 + 1))[: 32 * ms]
                _, calls = self.exercise(pcm, speech, (0, ms), expected=([], 0))
                self.assertEqual(calls, [])

    def test_privacy_off_after_silence_with_native_alignment_remainder(self):
        for ms in (160, 1000):
            with self.subTest(ms=ms):
                vad = self.worker([])
                c, _, shadow, p, notes, _ = self.arrange(vad=vad)
                shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                c.duck(True)
                epoch = shadow.epoch
                self.feed(c, p, bytes(ms * 32), shadow)
                self.assertEqual(len(p._partial), (20 if ms == 1000 else 0) * 32)
                c.duck(False)
                self.assertIn("diarizer_private_frames=0 forget=skipped", notes[-1])
                self.assertEqual(shadow.epoch, epoch)
                self.assertEqual(shadow.binding("Speaker A"), ("bound", "owner"))
                self.assertFalse(p._speaker_reset)

    def persistent_fundamental(self):
        # The 100 Hz voice stays 20 dB above the 50 Hz background throughout.
        base = ((50, 100), (100, 1000), (200, 80))
        voice, louder = tones(*base, (300, 50)), tones(*base, (300, 3000))
        return self.exercise(
            voice * 4 + louder * 10 + voice * 18, [(0, 3200)], (400, 1400), expected=([], 0)
        )

    def test_clean_command_over_50_hz_fan_with_400_ms_lead(self):
        # With a 200 ms lead-in, the native onset block can leave only 192 ms of absence.
        fan = (50, 1000)
        command = tones(fan, (200, 3000), (400, 1000))
        self.exercise(
            tones(fan) * 4 + command * 10 + tones(fan) * 16,
            [(400, 1400)],
            (400, 1400),
            expected=(["exit"], 1),
        )

    def omitted_tail(self):
        fan = (50, 1000)
        command = tones(fan, (200, 3000), (400, 1000))
        amp = 1000 * math.sqrt((1.465**2 - 1) / (1 + 0.2**2 + 0.1**2))
        tail = tones(fan, (80, amp), (160, 0.2 * amp), (240, 0.1 * amp))
        audio = tones(fan) * 2 + command * 10 + tail * 20 + tones(fan) * 16
        return self.exercise(audio, [(200, 3200)], (200, 1200), expected=([], 0))

    def machinery(self, hz, *, expected):
        machine = tones((hz, 1000))
        # Both acoustic boundaries coincide with native VAD and ASR frame edges.
        command = tones((hz, 1000), (100, 3000), (300, 1200))
        return self.exercise(
            machine * 240 + command * 16 + machine * 16,
            [(24000, 25600)],
            (24000, 25600),
            expected=expected,
        )


class FakeWorkerDecisions(AdversarialReplay, unittest.TestCase):
    def worker(self, speech):
        return fake_worker(self, speech)

    def test_off_silences_a_whole_private_frame_waiting_for_vad_alignment(self):
        c, _, s, p, notes, _ = self.arrange(vad=self.worker([]))
        c.duck(True)
        self.feed(c, p, SILENCE * 49 + VOICE, s)
        self.assertEqual(p._partial, VOICE)
        c.duck(False)
        self.assertIn("forget=skipped", notes[-1])
        # Observe the cutoff itself before the diarizer's second substitution.
        substitute = Mock(wraps=p._diarizer_frame)
        with patch.object(p, "_diarizer_frame", substitute):
            self.feed(c, p, VOICE * 3, s)
        self.assertEqual(substitute.call_args_list[0].args[0], SILENCE)
        self.assertEqual(substitute.call_args_list[1].args[0], VOICE)

    def test_persistent_fundamental_with_louder_third_harmonic(self):
        labels, _ = self.persistent_fundamental()
        self.assertNotIn(
            b"\0" * 7,
            bytes(labels[: 400 // CHUNK_MS]),
            "VAD certified 200 ms of absence during the continuous voiced lead",
        )

    def test_omitted_low_pitched_tail_at_1_47_times_fan_rms(self):
        labels, calls = self.omitted_tail()
        self.assertTrue(
            any(labels[1200 // CHUNK_MS : 3200 // CHUNK_MS]), "VAD missed the voiced tail"
        )
        self.assertFalse(any(end < 3200 for _, end in calls), "ASR ended while speech continued")

    def test_clean_command_over_200_hz_steady_machinery(self):
        _, calls = self.machinery(200, expected=(["exit"], 1))
        self.assertFalse(any(end <= 24000 for _, end in calls), "steady machinery opened ASR")

    def test_clean_command_sharing_a_fundamental_with_the_fan(self):
        _, calls = self.machinery(100, expected=(["exit"], 1))
        self.assertFalse(any(end <= 24000 for _, end in calls), "steady machinery opened ASR")

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


class RealSilero:
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

    def real_worker(self):
        worker = SileroVad(self.settings)
        self.addCleanup(worker.close)
        return worker


class RealModelDecisions(RealSilero, AdversarialReplay, unittest.TestCase):
    def worker(self, speech):
        return self.real_worker()

    def test_persistent_fundamental_with_louder_third_harmonic(self):
        self.persistent_fundamental()

    def test_omitted_low_pitched_tail_at_1_47_times_fan_rms(self):
        self.omitted_tail()

    def test_clean_command_over_200_hz_steady_machinery(self):
        _, calls = self.machinery(200, expected=([], 0))
        self.assertFalse(any(end <= 24000 for _, end in calls))

    def test_clean_command_sharing_a_fundamental_with_the_fan(self):
        _, calls = self.machinery(100, expected=([], 0))
        self.assertFalse(any(end <= 24000 for _, end in calls))
