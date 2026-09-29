"""Offline benchmark safety and measurement tests; synthetic WAV and mocked runtimes."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from rightyo.audio import AudioError
from scripts.benchmark_stack import benchmark, main, save_report, wav_duration


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "identifying-source.wav"
        with wave.open(str(self.audio), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000)
        self.whisper = self.root / "whisper-cli"
        self.whisper_model = self.root / "whisper-model.bin"
        self.nemotron = self.root / "nemo-speech"
        self.nemotron_model = self.root / "nemotron-model.gguf"
        for path in (self.whisper, self.whisper_model, self.nemotron, self.nemotron_model):
            path.write_bytes(b"synthetic-artifact")
        self.arguments = dict(
            audio=self.audio,
            whisper_executable=self.whisper,
            whisper_model=self.whisper_model,
            nemotron_executable=self.nemotron,
            nemotron_model=self.nemotron_model,
        )
        self.transcription = [
            {
                "start_ms": 0,
                "end_ms": 300,
                "text": "Private synthetic transcript",
                "finalized": True,
            },
            {
                "start_ms": 300,
                "end_ms": 600,
                "text": "Other private synthetic transcript",
                "finalized": True,
            },
            {"start_ms": 600, "end_ms": 1000, "text": "Private tail", "finalized": True},
        ]
        self.timeline = [
            {"start_ms": 0, "end_ms": 500, "speaker_id": "source-voice-99", "finalized": True},
            {"start_ms": 500, "end_ms": 700, "speaker_id": "source-voice-88", "finalized": True},
        ]

    def mocks(self):
        stack = contextlib.ExitStack()
        stack.enter_context(patch("scripts.benchmark_stack.hardware", return_value={"os": "test"}))
        stack.enter_context(
            patch("scripts.benchmark_stack.transcribe_whisper_cpp", return_value=self.transcription)
        )
        stack.enter_context(
            patch("scripts.benchmark_stack.diarize_nemotron_cpp", return_value=self.timeline)
        )
        return stack

    def argv(self, **extra):
        fields = {**self.arguments, **extra}
        return [
            item
            for name, value in fields.items()
            for item in ("--" + name.replace("_", "-"), str(value))
        ]

    def test_timings_include_first_and_cache_warmed_processes_separately(self):
        clock = [0, 4, 10, 12, 20, 21, 30, 33, 40, 44, 50, 52]
        with self.mocks(), patch("scripts.benchmark_stack.time.perf_counter", side_effect=clock):
            report = benchmark(**self.arguments, reruns=2)
        self.assertEqual(report["configuration"]["total_invocations_per_stage"], 3)
        timing = report["nemotron"]["timing"]
        self.assertEqual(timing["initial_invocation_seconds"], 4)
        self.assertEqual(timing["cache_warmed_separate_process_seconds"], [2, 1])
        self.assertEqual(timing["cache_warmed_seconds"], {"median": 1.5, "min": 1, "max": 2})
        self.assertEqual(timing["cache_warmed_fullfile_real_time_factors"], [2, 1])
        self.assertEqual(report["whisper"]["timing"]["initial_invocation_seconds"], 3)

    def test_strict_join_preserves_switch_and_missing_tail_as_unknown(self):
        with self.mocks():
            report = benchmark(**self.arguments, reruns=0)
        self.assertEqual(
            report["nemotron"]["strict_join_counts_by_invocation"],
            [
                {
                    "turns": 3,
                    "assigned": 1,
                    "unknown": 2,
                    "overlap": 0,
                    "unique_assigned_speakers": 1,
                }
            ],
        )
        self.assertEqual(
            report["nemotron"]["timeline_counts_by_invocation"],
            [{"segments": 2, "unique_known_speakers": 2}],
        )
        self.assertIsNone(report["nemotron"]["timing"]["cache_warmed_seconds"])

    def test_summary_hides_text_paths_and_source_labels_but_pins_artifacts(self):
        with self.mocks():
            report = benchmark(**self.arguments, reruns=0)
        output = json.dumps(report)
        for private in ("Private", "source-voice", self.audio.stem, str(self.root), "Speaker A"):
            self.assertNotIn(private, output)
        self.assertEqual(
            report["artifacts"]["whisper_model"]["sha256"],
            hashlib.sha256(b"synthetic-artifact").hexdigest(),
        )
        self.assertEqual(
            report["artifacts"]["audio"]["sha256"],
            hashlib.sha256(self.audio.read_bytes()).hexdigest(),
        )
        self.assertEqual(report["configuration"]["nemotron_preset"], "v3-streaming")
        self.assertEqual(report["mode"], "offline-file-benchmark")
        self.assertTrue(report["completed"])

    def test_overlap_counts_are_separate_from_unknown_counts(self):
        self.timeline.append(
            {"start_ms": 100, "end_ms": 200, "speaker_id": "third-voice", "finalized": True}
        )
        with self.mocks():
            report = benchmark(**self.arguments, reruns=0)
        joined = report["nemotron"]["strict_join_counts_by_invocation"][0]
        self.assertEqual(joined["overlap"], 1)
        self.assertEqual(joined["unknown"], 3)

    def test_supplied_speakerkit_timeline_has_no_implied_runtime_measurement(self):
        reference = self.root / "comparison.rttm"
        reference.write_text(
            f"SPEAKER {self.audio.stem} 1 0 1 <NA> <NA> original-label <NA> <NA>\n"
        )
        with self.mocks():
            report = benchmark(**self.arguments, reruns=0, speakerkit_rttm=reference)
        baseline = report["speakerkit_supplied_rttm"]
        self.assertEqual(baseline["mode"], "import-only-no-runtime-measurement")
        self.assertNotIn("timing", baseline)
        self.assertEqual(baseline["strict_join_counts"]["assigned"], 3)
        self.assertEqual(
            report["artifacts"]["speakerkit_supplied_rttm"]["sha256"],
            hashlib.sha256(reference.read_bytes()).hexdigest(),
        )

    def test_other_recording_reference_rejected_before_inference(self):
        reference = self.root / "comparison.rttm"
        reference.write_text("SPEAKER unrelated 1 0 1 <NA> <NA> A <NA> <NA>\n")
        with (
            patch("scripts.benchmark_stack.diarize_nemotron_cpp") as inference,
            self.assertRaises(AudioError),
        ):
            benchmark(**self.arguments, speakerkit_rttm=reference)
        inference.assert_not_called()

    def test_repeat_budget_timeout_and_missing_inputs_fail_before_inference(self):
        invalid = [
            {"reruns": -1},
            {"reruns": 10},
            {"reruns": True},
            {"timeout_seconds": float("nan")},
            {"timeout_seconds": 3601},
            {"timeout_seconds": True},
            {"backend": "unknown"},
            {"whisper_model": self.root / "missing"},
        ]
        for extra in invalid:
            with (
                self.subTest(extra=extra),
                patch("scripts.benchmark_stack.diarize_nemotron_cpp") as inference,
                self.assertRaises(AudioError),
            ):
                benchmark(**{**self.arguments, **extra})
            inference.assert_not_called()

    def test_wrong_sample_rate_and_truncated_wav_rejected(self):
        with wave.open(str(self.audio), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\0\0" * 8000)
        with self.assertRaises(AudioError):
            wav_duration(self.audio)
        with wave.open(str(self.audio), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(b"\0\0" * 16000)
        self.audio.write_bytes(self.audio.read_bytes()[:-2])
        with self.assertRaises(AudioError):
            wav_duration(self.audio)

    def test_later_runtime_failure_emits_no_success_or_partial_file(self):
        output = self.root / "summary.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            self.mocks(),
            patch(
                "scripts.benchmark_stack.diarize_nemotron_cpp",
                side_effect=[self.timeline, AudioError("Private transcript and secret path")],
            ),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(self.argv(output=output, reruns=1))
        self.assertEqual(result, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Benchmark failed; no complete result was produced\n")
        self.assertFalse(output.exists())

    def test_output_is_new_complete_json_and_cannot_overwrite_input(self):
        output = self.root / "summary.json"
        save_report(output, {"completed": True})
        self.assertEqual(json.loads(output.read_text()), {"completed": True})
        with self.assertRaises(AudioError):
            save_report(output, {"completed": False})
        self.assertEqual(json.loads(output.read_text()), {"completed": True})
        with (
            patch("scripts.benchmark_stack.benchmark") as inference,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            self.assertEqual(main(self.argv(output=self.audio)), 1)
        inference.assert_not_called()

    def test_missing_output_parent_fails_before_inference_without_partial_file(self):
        output = self.root / "missing-parent" / "summary.json"
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("scripts.benchmark_stack.benchmark") as inference,
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(self.argv(output=output))
        self.assertEqual(result, 1)
        inference.assert_not_called()
        self.assertFalse(output.exists())
        self.assertFalse(output.parent.exists())
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Benchmark failed; no complete result was produced\n")

    def test_unsupported_hardlinks_fail_before_inference_and_clean_private_probes(self):
        output = self.root / "summary.json"
        before = set(self.root.iterdir())
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("scripts.benchmark_stack.benchmark") as inference,
            patch("scripts.benchmark_stack.os.link", side_effect=OSError("Private storage path")),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            result = main(self.argv(output=output))
        self.assertEqual(result, 1)
        inference.assert_not_called()
        self.assertFalse(output.exists())
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "Benchmark failed; no complete result was produced\n")


if __name__ == "__main__":
    unittest.main()
