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
import time
import unittest
from array import array
from dataclasses import replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from unittest.mock import Mock, patch

import test_privacy_minimum as privacy
from test_live_audio import SILENCE, VOICE
from test_whisper_window_recovery import doc, segment

from rightyo.live_audio import BYTES_PER_MS
from rightyo.live_speaker_id import EntryScore, LabelState
from rightyo.silero_vad import CHUNK_BYTES, CHUNK_MS, VERSION, SileroVad, VadConfig, VadError

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".cache" / f"silero_vad-{VERSION}.onnx"

# Round 18. Lag in ms from the command end to the start of a positive block.
# Values beyond 2,000 ms came from a 6 s continuation with the same 2 s prefix.
GRID_LEADS_MS = (200, 420, 1000)
GRID_LAGS_MS = {
    160: (3568, 3220, 1360),
    180: (1360, 3188, 1328),
    200: (1328, 1652, 976),
    250: (944, 1332, 944),
    300: (944, 1012, 944),
    400: (720, 980, 720),
}


def tones(*components):
    """100 ms of orthogonal integer-cycle sinusoids, at 16 kHz."""
    return array(
        "h",
        (
            round(sum(amp * math.sin(2 * math.pi * hz * i / 16000) for hz, amp in components))
            for i in range(1600)
        ),
    ).tobytes()


@lru_cache(maxsize=2)
def speechlike_wave(duration_ms, *, reference_ms=None):
    """Key7's moving formants, scaled to unit RMS over the reference interval."""
    wave = []
    pitch_sum = 0.0
    for i in range(duration_ms * 16):
        t = math.tau * (i / 16000)
        pitch = 118 + 13 * math.sin(2.2 * t) + 4 * math.sin(5.1 * t)
        pitch_sum += pitch
        phase = math.tau * pitch_sum / 16000
        f1 = 490 + 195 * math.sin(2.7 * t + 0.35)
        f2 = 1480 + 480 * math.sin(2.2 * t + 1.1)
        f3 = 2570 + 175 * math.cos(3.4 * t)
        sample = 0.0
        for h in range(1, 47):
            hz = h * pitch
            amplitude = (
                0.09 / h + math.exp(-0.5 * ((hz - f1) / 100) ** 2)
                + 0.6 * math.exp(-0.5 * ((hz - f2) / 150) ** 2)
                + 0.32 * math.exp(-0.5 * ((hz - f3) / 180) ** 2)
            )
            sample += amplitude / math.sqrt(h) * math.sin(h * phase)
        wave.append(sample * (0.28 + 0.72 * (0.5 + 0.5 * math.sin(3.15 * t)) ** 1.4))
    reference = wave if reference_ms is None else wave[:reference_ms * 16]
    rms = math.sqrt(sum(s * s for s in reference) / len(reference))
    return tuple(s / rms for s in wave)


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
        "while opcode := sys.stdin.buffer.read(1):\n"
        "    if opcode == b'P':\n"
        "        out.write(b'P'); out.flush(); continue\n"
        "    assert opcode == b'S'\n"
        f"    assert len(sys.stdin.buffer.read({CHUNK_BYTES})) == {CHUNK_BYTES}\n"
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

    def test_hung_probe_escalates_to_kill_and_reaps_before_teardown(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "unresponsive"
            script.write_text(
                f"#!{sys.executable}\nimport os, signal\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "os.write(1, b'R')\nsignal.pause()\n"
            )
            script.chmod(0o700)
            worker = SileroVad(VadConfig(script, script))
            self.addCleanup(worker.close)
            start = time.monotonic()
            self.assertFalse(worker.available)
            self.assertLess(time.monotonic() - start, 2)
            self.assertEqual(worker.process.returncode, -signal.SIGKILL)
            with self.assertRaises(ChildProcessError):
                os.waitpid(worker.process.pid, os.WNOHANG)

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
        command = doc(segment("Jarvis,", 20, 180), segment(" privacy off.", 180, 340))
        stream = SILENCE * 16 + VOICE * 16 + SILENCE * 80
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            # The already-open window finishes without exit evidence. The second
            # energy-gated window may recognize status but cannot exit.
            self.assertEqual(
                self.replay_private_windows(command, command, audio=[stream, stream], vad=worker),
                ([], 0),
            )
        self.assertEqual(
            stderr.getvalue(), "rightyo: Silero VAD worker failed; voice privacy exit disabled\n"
        )

    def test_private_status_survives_a_runtime_worker_failure(self):
        status = doc(segment("Are you recording?", 0, 100))
        command = doc(privacy.addressed_segment("Jarvis, privacy off.", 0, 100))
        for vad in (None, fake_worker(self, [], failure_ms=32)):
            with self.subTest(configured=vad is not None):
                self.assertEqual(
                    self.replay_private_windows(status, command, vad=vad), (["status"], 0)
                )

    def test_worker_lost_after_scoring_cannot_release_an_exit(self):
        worker = fake_worker(self, [(320, 640)])
        command = doc(segment("Jarvis,", 20, 180), segment(" privacy off.", 180, 340))

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

    def exercise(
        self,
        pcm,
        speech,
        command,
        *,
        expected,
        words=None,
        worker_action=None,
        during_hold=None,
        endpoint_pause=None,
        expected_status_at=None,
        pre_roll_ms=400,
        expected_acoustic_end=None,
    ):
        vad = self.worker(speech)
        calls, labels, owner_check_times = [], [], []
        hold_observations = []
        classify = vad.is_speech

        def observe(block):
            label = classify(block)
            labels.append(label)
            if during_hold and len(labels) * CHUNK_MS == during_hold[0]:
                before_change = s.exit_score.call_count, e.drain(), len(c._private_sentence.pcm)
                during_hold[1](c, e, vad)
                hold_observations.append(before_change)
            return label

        vad.is_speech = observe
        c, e, s, p, _, units = self.arrange(
            vad=vad,
            pre_roll_ms=pre_roll_ms,
            end_of_turn=(lambda _: 1.0) if endpoint_pause else None,
        )
        if endpoint_pause:
            p.config = replace(p.config, end_of_turn_silence_ms=endpoint_pause)
        c._enrolled_roles.exit_audio = c._private_sentence
        worker_at_approval = []

        def score(*_):
            owner_check_times.append(len(labels) * CHUNK_MS)
            if worker_action == "close":
                vad.close()
                worker_at_approval.append(not vad.available)
            elif worker_action == "stop":
                vad.process.send_signal(signal.SIGSTOP)
                pid, status = os.waitpid(vad.process.pid, os.WUNTRACED)
                worker_at_approval.append(pid == vad.process.pid and os.WIFSTOPPED(status))
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
                    for left, right, text in words
                    or [
                        (command[0], command[0] + 100, "Jarvis,"),
                        (command[0] + 100, command[1], " privacy off."),
                    ]
                ]
            return []

        units.transcribe = decode
        if expected_status_at is None:
            self.feed(c, p, pcm, s)
            p.finish()
        else:
            status_clock = []
            emit = e._emit

            def observe_emit(kind, **fields):
                if kind == "privacy" and fields.get("action") == "status":
                    status_clock.append(c._received_ms)
                return emit(kind, **fields)

            with patch.object(e, "_emit", side_effect=observe_emit):
                self.feed(c, p, pcm, s)
                p.finish()
            self.assertEqual(status_clock, [expected_status_at])
        if during_hold:
            selected_ms = sum(end - start for start, end, word in words if "?" not in word)
            self.assertEqual(hold_observations, [(0, [], selected_ms * BYTES_PER_MS)])
        if worker_action == "during_hold":
            self.assertIsNone(p._vad)
        elif worker_action:
            self.assertIsNone(p._vad)
            self.assertEqual(worker_at_approval, [True])
        else:
            self.assertIs(p._vad, vad, "worker failure is not an adversarial result")
            self.assertEqual(len(labels), len(pcm) // CHUNK_BYTES)
        events = e.drain()
        self.assertTrue(e.ducked)
        self.assertTrue(all(event["type"] == "privacy" for event in events))
        self.assertFalse(e._memory.retained_ids, "private transcript retained")
        self.assertEqual(([event["action"] for event in events], s.exit_score.call_count), expected)
        if owner_check_times:
            # ASR finalization and the acoustic-end window must both be complete.
            acoustic_end = command[1] if expected_acoustic_end is None else expected_acoustic_end
            ready = math.ceil(max(acoustic_end + 1728, calls[0][1]) / CHUNK_MS) * CHUNK_MS
            self.assertEqual(owner_check_times, [ready] * expected[1])
        for event in events:
            if event["action"] == "exit":
                # Publication can already be on the 20 ms input frame carrying the last block.
                self.assertGreaterEqual(event["emitted_at_ms"], owner_check_times[0])
                self.assertLess(event["emitted_at_ms"], owner_check_times[0] + 20)
            elif event["action"] == "status" and expected_status_at is not None:
                self.assertEqual(event["emitted_at_ms"], expected_status_at)
        self.assertFalse(c._private_sentence.pcm)
        return labels, calls

    def key_exit(self, tail, *, lead=400, later_speech=(), expected, **options):
        fan = tones((50, 1000))
        command = tones((50, 1000), (200, 3000), (400, 1000))
        onset = math.ceil((lead + 16) / CHUNK_MS) * CHUNK_MS
        return self.exercise(
            (fan * 10)[:lead * BYTES_PER_MS] + command * 10 + tail,
            [(onset, onset + 8 * CHUNK_MS), *later_speech],
            (lead, lead + 1000),
            words=[
                (lead, lead + 100, "Jarvis,"),
                (lead + 100, lead + 180, " privacy"),
                (lead + 180, lead + 1000, " off."),
            ],
            expected=expected,
            **options,
        )

    def speechlike_tail(self, level, duration, lag, expected):
        fan = tones((50, 1000))
        room = array("h", fan * (duration // 100))
        wave = speechlike_wave(duration)
        tail = array("h", (round(bg + level * v) for bg, v in zip(room, wave, strict=True)))
        pcm = tail.tobytes() + fan * 16
        first_positive = 1400 + lag
        for pause, close in ((None, 2120), (20, 700), (720, 1400)):
            with self.subTest(endpoint_pause=pause):
                labels, windows = self.key_exit(
                    pcm,
                    later_speech=[(first_positive, first_positive + CHUNK_MS)],
                    expected=expected if close >= 1400 else ([], 0),
                    endpoint_pause=pause,
                )
                self.assertEqual(windows[0][1], close)
                first = next(
                    i * CHUNK_MS for i, v in enumerate(labels) if v and i * CHUNK_MS >= 1400
                )
                self.assertEqual(first - 1400, lag)

    def test_speechlike_continuation_cancels_at_both_measured_levels(self):
        for rms, lag in ((400, 936), (180, 1352)):
            with self.subTest(rms=rms):
                self.speechlike_tail(rms, 2000, lag, ([], 0))

    def replay_grid_cell(self, rms, lead, *, residual=False):
        lag = GRID_LAGS_MS[rms][GRID_LEADS_MS.index(lead)]
        end = lead + 1000
        duration = 6000 if residual else 2000
        # Keep the first two seconds identical when extending a starred cell.
        wave = speechlike_wave(duration, reference_ms=2000)
        if residual:
            short = speechlike_wave(2000)
            self.assertEqual(wave[:len(short)], short)
        fan = tones((50, 1000))
        room = array("h", fan * (duration // 100))
        tail = array("h", (bg + round(rms * v) for bg, v in zip(room, wave, strict=True)))
        for pause in (None, 720):
            with self.subTest(rms=rms, lead=lead, endpoint_pause=pause):
                # The fake worker gives every command a qualified onset so the
                # tail gate is exercised even for the shortest lead-in.
                can_exit = residual and (lead != 200 or self.qualified_200_ms_onset)
                labels, windows = self.key_exit(
                    tail.tobytes() + fan * 16,
                    lead=lead,
                    later_speech=[(end + lag, end + lag + CHUNK_MS)],
                    endpoint_pause=pause,
                    expected=(["exit"], 2) if can_exit else ([], 0),
                )
                start, close = windows[0]
                self.assertLessEqual(start, lead)
                self.assertGreaterEqual(close, end, "ASR truncated the candidate")
                if pause:
                    self.assertLess(close, end + lag, "fast ASR must close before detection")
                after = math.ceil(end / CHUNK_MS)
                if residual:
                    two_seconds = math.ceil((end + 2000) / CHUNK_MS)
                    self.assertFalse(any(labels[after:two_seconds]))
                    self.assertTrue(any(labels[two_seconds:(end + duration) // CHUNK_MS]))
                    if not can_exit:
                        self.assertEqual(labels.index(True) * CHUNK_MS, 192)
                else:
                    self.assertTrue(any(labels[after:math.ceil((end + 1728) / CHUNK_MS)]))
            self.doCleanups()

    def test_grid_continuations_detected_within_window_cancel(self):
        for rms, lags in GRID_LAGS_MS.items():
            for lead, lag in zip(GRID_LEADS_MS, lags, strict=True):
                if lag <= 1728:
                    self.replay_grid_cell(rms, lead)

    def test_early_final_word_timestamp_cannot_shorten_acoustic_confirmation(self):
        fan = tones((50, 1000))
        command = tones((50, 1000), (200, 3000), (400, 1000)) * 10
        tail = array("h", (
            bg + round(200 * value)
            for bg, value in zip(array("h", fan * 20), speechlike_wave(2000), strict=True)
        ))
        prefix = (fan * 5)[:420 * BYTES_PER_MS] + command
        pcm = prefix + tail.tobytes() + fan * 16
        for reported_end in (1300, 1420):
            with self.subTest(reported_end=reported_end):
                labels, windows = self.exercise(
                    pcm,
                    [(416, 1184), (3072, 3104)],
                    (420, reported_end),
                    words=[(420, 520, "Jarvis,"), (520, 600, " privacy"),
                           (600, reported_end, " off.")],
                    expected=([], 0),
                )
                self.assertGreaterEqual(windows[0][1], 1420)
                resumed = next(
                    i * CHUNK_MS for i, voice in enumerate(labels) if voice and i * CHUNK_MS >= 1420
                )
                self.assertEqual(resumed, 3072)
        self.exercise(
            prefix + fan * 24,
            [(416, 1184)],
            (420, 1300),
            words=[(420, 520, "Jarvis,"), (520, 600, " privacy"), (600, 1300, " off.")],
            expected=(["exit"], 2),
            expected_acoustic_end=1420,
        )

    def report_playback(self, replay, vad, spans, *, use_received_clock):
        c, events, _, processor, _, _ = replay
        classify = vad.is_speech
        reports = [(edge, phase) for start, end in spans
                   for edge, phase in ((start, "started"), (end, "ended"))]
        reports = iter(reports)
        next_report = next(reports, None)
        block_end = 0

        def observe(block):
            nonlocal next_report, block_end
            block_end += CHUNK_MS
            now = c._received_ms if use_received_clock else block_end
            while next_report is not None and now >= next_report[0]:
                if next_report[0] in {spans[0][0], spans[-1][0]}:
                    self.assertIsNotNone(processor._pending_exit, "confirmation already decided")
                events.reply(next_report[1], now_ms=now)
                next_report = next(reports, None)
            return classify(block)

        vad.is_speech = observe

    def test_early_playback_evidence_survives_eight_later_spans(self):
        spans = [(1440, 1500)] + [(1760 + i * 128, 1820 + i * 128) for i in range(8)]
        arrange = self.arrange

        def with_playback(**options):
            replay = arrange(**options)
            self.report_playback(replay, options["vad"], spans, use_received_clock=True)
            return replay

        with patch.object(self, "arrange", side_effect=with_playback):
            self.key_exit(tones((50, 1000)) * 24, endpoint_pause=720, expected=([], 0))

    def test_deferred_status_survives_speaker_forget(self):
        fan = tones((50, 1000))
        command = tones((50, 1000), (200, 3000), (400, 1000))
        self.exercise(
            fan * 4 + command * 10 + fan * 24,
            [(416, 672)],
            (400, 1400),
            words=[(400, 480, "Are you recording?"), (480, 560, " Jarvis,"),
                   (560, 610, " privacy"), (610, 1400, " off.")],
            during_hold=(2496, lambda c, e, vad: e.forget_speakers(now_ms=2496)),
            expected=(["status"], 0),
        )

    def test_stream_start_status_does_not_require_preceding_absence(self):
        wave = speechlike_wave(2000)[:16 * 800]
        pcm = array("h", (round(3000 * value) for value in wave)).tobytes()
        labels, windows = self.exercise(
            pcm + bytes(2400 * BYTES_PER_MS),
            [(0, 800)], (0, 600), words=[(0, 600, "Are you recording?")],
            expected=(["status"], 0), pre_roll_ms=240,
        )
        self.assertEqual(labels[:26], [True] * 25 + [False])
        self.assertEqual(windows, [(0, 2240)])

    def test_clean_grid_commands_under_both_endpointers(self):
        for lead in GRID_LEADS_MS:
            for pause in (None, 720):
                with self.subTest(lead=lead, endpoint_pause=pause):
                    qualified = lead != 200 or self.qualified_200_ms_onset
                    self.key_exit(
                        tones((50, 1000)) * 24,
                        lead=lead,
                        endpoint_pause=pause,
                        expected=(["exit"], 2) if qualified else ([], 0),
                    )
                self.doCleanups()

    def test_boundary_lag_3568_ms_after_command_end(self):
        """Residual limit at 160 RMS / 200 ms. The real onset also lacks 200 ms of absence."""
        self.replay_grid_cell(160, 200, residual=True)

    def test_boundary_lag_3220_ms_after_command_end(self):
        """Residual limit at 160 RMS / 420 ms. Detection follows the confirmed exit."""
        self.replay_grid_cell(160, 420, residual=True)

    def test_boundary_lag_3188_ms_after_command_end(self):
        """Residual limit at 180 RMS / 420 ms. Detection follows the confirmed exit."""
        self.replay_grid_cell(180, 420, residual=True)

    def test_name_waits_for_confirmation_before_owner_check(self):
        fan = tones((50, 1000))
        voice = tones((50, 1000), (200, 3000), (400, 1000))
        wave = speechlike_wave(2000)
        tail = array("h", (round(b + 400 * v) for b, v in zip(array("h", fan * 20), wave)))
        for continuation, later, checks in (
            (tail.tobytes(), [(2336, 2368)], 0),
            (fan * 20, [], 1),
        ):
            with self.subTest(checks=checks):
                self.exercise(
                    fan * 4 + voice * 10 + continuation + fan * 16,
                    [(416, 672), *later], (400, 1400), words=[(400, 1400, "Jarvis,")],
                    endpoint_pause=720, expected=([], checks),
                )

    def test_contiguous_final_subwords_cover_the_same_native_block_as_one_word(self):
        voice = tones((200, 5000)) + tones((200, 5000), (400, 5000))
        pcm = bytes(400 * 32) + (voice * 6)[:1008 * 32] + bytes(2400 * 32)
        for ending in ([segment(" off.", 1280, 1328)],
                       [segment(" o", 1280, 1310), segment("ff.", 1310, 1328)]):
            with self.subTest(split=len(ending) == 2):
                parts = [segment("Jarvis,", 320, 520), segment(" privacy", 520, 1280), *ending]
                phrase = doc({**segment("Jarvis, privacy off.", 320, 1328), "tokens": parts})
                self.assertEqual(
                    self.replay_private_windows(
                        phrase, pre_roll_ms=400, audio=[pcm], vad=self.worker([(480, 1408)])
                    ),
                    (["exit"], 2),
                )

    def test_status_question_survives_cancelled_exit_in_same_recognition(self):
        fan = tones((50, 1000))
        voice = tones((50, 1000), (200, 3000), (400, 1000))
        words = [(400, 480, "Are you recording?"), (480, 560, " Jarvis,"),
                 (560, 610, " privacy"), (610, 1400, " off.")]
        for failure in (True, False):
            with self.subTest(failure=failure):
                tail = fan * 24 if failure else (fan * 15)[:1416 * 32] + voice * 4 + fan * 16
                self.exercise(
                    fan * 4 + voice * 10 + tail,
                    [(416, 672), (2816, 2848)], (400, 1400), words=words,
                    during_hold=(2496, lambda c, e, vad: vad.close()) if failure else None,
                    worker_action="during_hold" if failure else None, expected=(["status"], 0),
                    expected_status_at=2540 if failure else None,
                )

    def test_accepting_endpointer_close_relative_to_acoustic_end(self):
        # Same clean command, ending at 1400 ms. An early close can truncate it.
        for pause, close, expected in (
            (20, 700, ([], 0)),
            (700, 1380, ([], 0)),
            (720, 1400, (["exit"], 2)),
            (800, 1480, (["exit"], 2)),
        ):
            with self.subTest(pause=pause):
                _, windows = self.key_exit(
                    tones((50, 1000)) * 24, endpoint_pause=pause, expected=expected
                )
                self.assertEqual(windows, [(0, close)])

    def test_clean_exit_requires_the_last_observed_block(self):
        fan = tones((50, 1000)) * 24
        # The last word ends at 1400 ms. The block [3104, 3136) covers the 3128 ms deadline.
        last = (3136 - 1400) * BYTES_PER_MS
        for pause in (None, 720):
            with self.subTest(endpoint_pause=pause):
                self.key_exit(fan[: last - 2], endpoint_pause=pause, expected=([], 0))
                self.key_exit(fan[:last], endpoint_pause=pause, expected=(["exit"], 2))

    def test_speech_on_the_last_block_cancels_the_exit(self):
        fan = tones((50, 1000))
        resumed = tones((50, 1000), (200, 3000), (400, 1000))
        tail = (fan * 18)[: (3104 - 1400) * 32] + resumed * 4 + fan * 24
        for pause, close in ((None, 2120), (720, 1400)):
            with self.subTest(endpoint_pause=pause):
                labels, windows = self.key_exit(
                    tail, later_speech=[(3104, 3136)], endpoint_pause=pause, expected=([], 0)
                )
                self.assertEqual(windows[0][1], close)
                self.assertFalse(any(labels[math.ceil(1400 / CHUNK_MS) : 3104 // CHUNK_MS]))
                self.assertTrue(labels[3104 // CHUNK_MS])

    def test_worker_failure_during_hold_or_after_last_score_cancels_exit(self):
        def stop_worker(c, e, vad):
            vad.process.send_signal(signal.SIGSTOP)
            _, status = os.waitpid(vad.process.pid, os.WUNTRACED)
            self.assertTrue(os.WIFSTOPPED(status))

        def close_worker(c, e, vad):
            vad.close()

        for at_ms, fail in ((2496, close_worker), (3136, close_worker), (3136, stop_worker)):
            with self.subTest(at_ms=at_ms, failure=fail.__name__):
                self.key_exit(
                    tones((50, 1000)) * 24,
                    during_hold=(at_ms, fail),
                    worker_action="during_hold",
                    expected=([], 0),
                )

    def clean_timed_control(self, shift, *, expected, worker_action=None, late=0):
        machine = tones((50, 1000))
        voice = tones((50, 1000), (200, 3000), (400, 1000))
        start, end = 400 + shift, 1400 + shift
        return self.exercise(
            machine * 4 + machine[: shift * 32] + voice * 10 + machine * 24,
            [(416, 672)],
            (start + late, end),
            words=[
                (start + late, start + 100, "Jarvis,"),
                (start + 100, start + 180, " privacy"),
                (start + 180, end, " off."),
            ],
            worker_action=worker_action,
            expected=expected,
        )

    def test_worker_dies_inside_owner_verification(self):
        self.clean_timed_control(0, worker_action="close", expected=([], 1))

    def test_stopped_child_inside_owner_verification_cannot_publish(self):
        self.clean_timed_control(0, worker_action="stop", expected=([], 1))

    def test_failed_probe_reaps_stopped_worker_before_teardown(self):
        worker = self.worker([])
        worker.process.send_signal(signal.SIGSTOP)
        _, status = os.waitpid(worker.process.pid, os.WUNTRACED)
        self.assertTrue(os.WIFSTOPPED(status))
        start = time.monotonic()
        self.assertFalse(worker.available)
        self.assertLess(time.monotonic() - start, 2)
        self.assertIn(worker.process.returncode, (-signal.SIGTERM, -signal.SIGKILL))
        with self.assertRaises(ChildProcessError):
            os.waitpid(worker.process.pid, os.WNOHANG)

    def test_timed_period_uses_the_final_words_native_block_tolerance(self):
        voice = tones((200, 5000)) + tones((200, 5000), (400, 5000))
        pcm = bytes(400 * 32) + (voice * 6)[: 1007 * 32] + bytes(2400 * 32)
        for period_start in (1407, 1408):
            with self.subTest(period_start=period_start):
                labels, _ = self.exercise(
                    pcm,
                    [(480, 1408)],
                    (400, 1407),
                    words=[
                        (400, 600, "Jarvis,"),
                        (600, 800, " privacy"),
                        (800, 1407, " off"),
                        (period_start, period_start + 10, "."),
                    ],
                    expected=(["exit"], 2),
                )
                self.assertEqual((len(labels) - labels[::-1].index(True)) * CHUNK_MS, 1408)

    def test_last_word_ends_one_ms_inside_native_speech_block(self):
        voice = tones((200, 5000)) + tones((200, 5000), (400, 5000))
        for end in (1407, 1408):
            with self.subTest(end=end):
                labels, _ = self.exercise(
                    bytes(400 * 32) + (voice * 6)[: (end - 400) * 32] + bytes(2400 * 32),
                    [(480, 1408)],
                    (400, end),
                    words=[(400, 600, "Jarvis,"), (600, 800, " privacy"), (800, end, " off.")],
                    expected=(["exit"], 2),
                )
                self.assertEqual((len(labels) - labels[::-1].index(True)) * CHUNK_MS, 1408)

    def test_steady_200_hz_from_crest_only_opens_startup_asr(self):
        # A quarter period shifts the same 1,000-count wave from zero to its crest.
        wave = tones((200, 1000))
        shifted = wave[40:] + wave[:40]
        labels, windows = self.exercise(shifted * 240, [(0, 64)], (0, 24000), expected=([], 0))
        self.assertEqual(labels[:2], [True, True])
        self.assertEqual(windows, [(0, 1520)])

    def test_18_ms_shifted_word_onset_in_first_native_speech_block(self):
        labels, _ = self.clean_timed_control(18, expected=(["exit"], 2))
        self.assertEqual(labels.index(True) * CHUNK_MS, 416)
        # Omitting the entire 416-448 ms speech block still refuses.
        self.clean_timed_control(18, late=30, expected=([], 0))

    def test_initial_machine_block_and_incomplete_final_block(self):
        for hz, ms, speech, windows in (
            (200, 24000, [(0, 32)], [(0, 1480)]), (50, 1600, [], []), (50, 1620, [], [])
        ):
            with self.subTest(hz=hz, ms=ms):
                pcm = (tones((hz, 1000)) * (ms // 100 + 1))[: 32 * ms]
                _, calls = self.exercise(pcm, speech, (0, ms), expected=([], 0))
                self.assertEqual(calls, windows)

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
            tones(fan) * 4 + command * 10 + tones(fan) * 24,
            [(400, 1400)],
            (400, 1400),
            expected=(["exit"], 2),
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
            machine * 240 + command * 16 + machine * 24,
            [(24000, 25600)],
            (24000, 25600),
            expected=expected,
        )


class FakeWorkerDecisions(AdversarialReplay, unittest.TestCase):
    qualified_200_ms_onset = True

    def worker(self, speech):
        return fake_worker(self, speech)

    def test_split_name_keeps_playback_evidence_until_its_first_check(self):
        fragment = bytes(320 * BYTES_PER_MS) + VOICE * 16 + bytes(2400 * BYTES_PER_MS)
        spans = [(832, 896)] + [(1024 + i * 128, 1088 + i * 128) for i in range(8)]
        vad = self.worker([(320, 640), (3360, 3680)])
        replay = self.arrange(vad=vad, pre_roll_ms=20, end_of_turn=lambda _: 1.0)
        self.report_playback(replay, vad, spans, use_received_clock=False)
        with patch.object(self, "arrange", return_value=replay):
            result = self.replay_private_windows(
                doc(segment("Jarvis,", 20, 340)), doc(segment("Privacy off.", 20, 340)),
                audio=[fragment, fragment], vad=vad, end_of_turn_silence_ms=160,
            )
        self.assertEqual(result, ([], 0))

    def test_split_filler_ellipsis_and_playback_during_remainder_confirmation(self):
        fragment = bytes(320 * 32) + VOICE * 16 + bytes(2400 * 32)
        name = doc(segment("Jarvis,", 20, 340))
        for dots, late_playback in product(("...", "…"), (False, True)):
            with self.subTest(dots=dots, late_playback=late_playback):
                rest = doc(segment(f"please{dots}", 20, 116), segment(" privacy off.", 116, 340))
                vad = self.worker([(320, 640), (3360, 3680)])
                replay = self.arrange(vad=vad, pre_roll_ms=20)
                _, events, shadow, _, _, _ = replay
                classify = vad.is_speech
                clock = 0

                def observe(block):
                    nonlocal clock
                    clock += CHUNK_MS
                    if late_playback and clock == 4992:
                        # The remainder has finalized. Only the name overlaps playback.
                        self.assertEqual(shadow.exit_score.call_count, 1)
                        events.reply("started", now_ms=690)
                        events.reply("ended", now_ms=740)
                    return classify(block)

                vad.is_speech = observe
                with patch.object(self, "arrange", return_value=replay):
                    result = self.replay_private_windows(name, rest, vad=vad, audio=[fragment] * 2)
                self.assertEqual(result, ([], 1) if late_playback else (["exit"], 2))

    def test_owner_forget_or_late_playback_report_during_hold_revokes_exit(self):
        for name, revoke in (
            ("forget", lambda c, e, vad: e.forget_speakers(now_ms=2496)),
            ("playback", lambda c, e, vad: e.reply("started", now_ms=1000)),
            ("privacy epoch", lambda c, e, vad: e.duck(True, now_ms=2496)),
        ):
            with self.subTest(name=name):
                self.key_exit(tones((50, 1000)) * 24, during_hold=(2496, revoke), expected=([], 0))

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
        _, calls = self.machinery(200, expected=(["exit"], 2))
        self.assertFalse(any(end <= 24000 for _, end in calls), "steady machinery opened ASR")

    def test_clean_command_sharing_a_fundamental_with_the_fan(self):
        _, calls = self.machinery(100, expected=(["exit"], 2))
        self.assertFalse(any(end <= 24000 for _, end in calls), "steady machinery opened ASR")

    def test_incomplete_last_vad_block_cannot_supply_an_ending_pause(self):
        command = doc(segment("Jarvis,", 20, 180), segment(" privacy off.", 180, 340))
        for pause_ms, expected in (
            (20, ([], 0)), (32, ([], 0)), (736, ([], 0)), (1440, ([], 0)), (1728, (["exit"], 2)),
        ):
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
        for name_end, expected in ((52, (["exit"], 2)), (21, ([], 0))):
            with self.subTest(name_end=name_end):
                worker = self.worker([(320, 640)])
                words = doc(
                    segment("Jarvis,", 20, name_end),
                    segment(" privacy", name_end, 84),
                    segment(" off.", 84, 340),
                )
                self.assertEqual(
                    self.replay_private_windows(
                        words, vad=worker, audio=[SILENCE * 16 + VOICE * 16 + SILENCE * 110]
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
    qualified_200_ms_onset = False

    def worker(self, speech):
        return self.real_worker()

    def test_split_name_keeps_playback_evidence_with_pinned_worker(self):
        fan = tones((50, 1000))
        name = tones((50, 1000), (200, 3000), (400, 1000))
        rest = array("h", (round(3000 * value) for value in speechlike_wave(2000)[:16000]))
        pcm = fan * 4 + name * 10 + fan * 24 + rest.tobytes() + bytes(2400 * BYTES_PER_MS)
        spans = [(1440, 1500)] + [(1760 + i * 128, 1820 + i * 128) for i in range(8)]
        for with_playback in (False, True):
            with self.subTest(with_playback=with_playback):
                vad = self.real_worker()
                replay = self.arrange(vad=vad, pre_roll_ms=400, end_of_turn=lambda _: 1.0)
                if with_playback:
                    self.report_playback(replay, vad, spans, use_received_clock=True)
                with patch.object(self, "arrange", return_value=replay):
                    result = self.replay_private_windows(
                        doc(segment("Jarvis,", 400, 1400)), doc(segment("Privacy off.", 400, 1400)),
                        audio=[pcm], vad=vad, end_of_turn_silence_ms=720,
                    )
                self.assertEqual(result, ([], 0) if with_playback else (["exit"], 2))

    def test_boundary_quiet_speechlike_onset_at_stream_start(self):
        """#350 round 15 operator ruling (2026-11-27), key7 P1-1, residual onset limit.

        Silero misses the quiet lead. The authored ASR also omits it, leaving
        the configured name at the start of the recognized sentence.
        """
        wave = speechlike_wave(1400)
        pcm = array("h", (round(v * (30 if i < 6400 else 3000)) for i, v in enumerate(wave)))
        labels, _ = self.exercise(
            pcm.tobytes() + bytes(2400 * 32),
            None,
            (400, 1400),
            expected=(["exit"], 2),
        )
        self.assertEqual(labels.index(True) * CHUNK_MS, 416)

    def test_boundary_nonspeech_100_hz_tone_with_rising_harmonics_allows_exit(self):
        """#350 operator ruling (2026-11-27), P1-1.

        Pinned Silero labels the tone after the phrase as non-speech. Two owner checks
        and one exit are expected while the fundamental continues.
        """
        base = ((50, 100), (100, 1000))
        tone = tones(*base, (200, 80), (400, 50)) * 34
        louder = tones(*base, (200, 3000), (400, 1000)) * 14
        # Change only the harmonic amplitudes. Keep the phase continuous at both edges.
        start, end = 384, 1384
        left, right = start * BYTES_PER_MS, end * BYTES_PER_MS
        labels, windows = self.exercise(
            tone[:left] + louder[left:right] + tone[right:],
            None,
            (start, end),
            words=[(start, 484, "Jarvis,"), (484, 564, " privacy"), (564, end, " off.")],
            expected=(["exit"], 2),
        )
        self.assertFalse(any(labels[: start // CHUNK_MS]))
        self.assertFalse(any(labels[end // CHUNK_MS :]))
        ((_, close_ms),) = windows
        self.assertLess(close_ms, 3400, "ASR waited for the tone to end")

    def test_boundary_nonspeech_tail_at_1_465_times_fan_rms_after_400_ms_lead_allows_exit(self):
        """#350 operator ruling (2026-11-27), P1-2.

        Pinned Silero labels the synthetic voiced tail as non-speech. Two owner checks
        and one exit are expected while the tail continues.
        """
        fan = (50, 1000)
        command = tones(fan, (200, 3000), (400, 1000))
        amp = 1000 * math.sqrt((1.465**2 - 1) / (1 + 0.2**2 + 0.1**2))
        tail = tones(fan, (80, amp), (160, 0.2 * amp), (240, 0.1 * amp))
        labels, windows = self.exercise(
            tones(fan) * 4 + command * 10 + tail * 20 + tones(fan) * 16,
            None,
            (400, 1400),
            words=[(400, 500, "Jarvis,"), (500, 580, " privacy"), (580, 1400, " off.")],
            expected=(["exit"], 2),
        )
        self.assertFalse(any(labels[1400 // CHUNK_MS :]))
        ((_, close_ms),) = windows
        self.assertLess(close_ms, 3400, "ASR waited for the synthetic tail to end")

    def test_persistent_fundamental_with_louder_third_harmonic(self):
        self.persistent_fundamental()

    def test_omitted_low_pitched_tail_at_1_47_times_fan_rms(self):
        self.omitted_tail()

    def test_clean_command_over_200_hz_steady_machinery(self):
        _, calls = self.machinery(200, expected=([], 0))
        self.assertEqual(calls, [(0, 1480)])

    def test_clean_command_sharing_a_fundamental_with_the_fan(self):
        _, calls = self.machinery(100, expected=([], 0))
        self.assertFalse(any(end <= 24000 for _, end in calls))
