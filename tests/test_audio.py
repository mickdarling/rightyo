"""Synthetic adapter tests; no real audio, models, or vendor runtime required."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rightyo.audio import (
    AudioError,
    join_transcript_timeline,
    load_timeline,
    parse_rttm,
    parse_whisper_cpp,
    transcribe_whisper_cpp,
)


def transcript(start=0, end=1000, **extra):
    return dict(start_ms=start, end_ms=end, text="Synthetic speech", finalized=True, **extra)


def speaker(start=0, end=1000, label="voice-1", finalized=True):
    return dict(start_ms=start, end_ms=end, speaker_id=label, finalized=finalized)


def join(turns, timeline):
    return join_transcript_timeline(
        turns, timeline, session_id="test", recognizer_id="synthetic-fixture", provenance="synthetic",
    )


class SpeakerTimelineTests(unittest.TestCase):
    def test_a_b_a_reuses_session_identity_and_hides_source_labels(self):
        turns = [transcript(0, 100), transcript(100, 200), transcript(200, 300)]
        timeline = [speaker(0, 100, "voice-7"), speaker(100, 200, "voice-2"), speaker(200, 300, "voice-7")]
        result = join(turns, timeline)
        self.assertEqual([r["speaker_id"] for r in result], ["Speaker A", "Speaker B", "Speaker A"])
        self.assertNotIn("voice-7", json.dumps(result))

    def test_overlapping_distinct_speakers_remain_unknown(self):
        result = join([transcript()], [speaker(), speaker(400, 700, "voice-2")])[0]
        self.assertIsNone(result["speaker_id"])
        self.assertTrue(result["overlap"])

    def test_adjacent_speaker_switch_does_not_invent_word_alignment(self):
        result = join([transcript()], [speaker(0, 500), speaker(500, 1000, "voice-2")])[0]
        self.assertIsNone(result["speaker_id"])
        self.assertFalse(result["overlap"])
        self.assertEqual(result["text"], "Synthetic speech")

    def test_gap_or_explicit_unknown_cannot_be_majority_voted_away(self):
        for timeline in ([speaker(0, 999)], [speaker(0, 999), speaker(999, 1000, None)]):
            with self.subTest(timeline=timeline):
                self.assertIsNone(join([transcript()], timeline)[0]["speaker_id"])

    def test_same_speaker_overlapping_windows_can_supply_full_coverage(self):
        result = join([transcript()], [speaker(0, 700), speaker(500, 1000)])[0]
        self.assertEqual(result["speaker_id"], "Speaker A")
        self.assertFalse(result["overlap"])

    def test_tentative_diarization_prevents_final_and_preserves_revision(self):
        result = join([transcript(utterance_id="stable-u", revision=4)], [speaker(finalized=False)])[0]
        self.assertFalse(result["finalized"])
        self.assertEqual((result["utterance_id"], result["revision"]), ("stable-u", 4))

    def test_invalid_times_finality_and_revisions_fail_closed(self):
        for key, value in (("start_ms", True), ("end_ms", float("nan")), ("end_ms", -1),
                           ("finalized", "true"), ("revision", True), ("revision", -1)):
            turn = transcript()
            turn[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(AudioError):
                join([turn], [])

    def test_synthetic_fixture_contains_overlap_unknown_and_tentative(self):
        fixture = Path(__file__).parents[1] / "examples" / "synthetic-speakers.json"
        result = load_timeline(fixture)
        self.assertEqual([r["speaker_id"] for r in result[:3]], ["Speaker A", "Speaker B", "Speaker A"])
        self.assertTrue(result[3]["overlap"])
        self.assertIsNone(result[4]["speaker_id"])
        self.assertFalse(result[5]["finalized"])
        self.assertTrue(all(r["provenance"] == "synthetic" for r in result))

    def test_argmax_rttm_seconds_convert_and_unknown_stays_unknown(self):
        records = parse_rttm(
            "SPEAKER fixture 1 0.125 0.875 <NA> <NA> A <NA> <NA>\n"
            "SPEAKER fixture 1 1.000 0.500 <NA> <NA> UNKNOWN <NA> <NA>"
        )
        self.assertEqual((records[0]["start_ms"], records[0]["end_ms"]), (125, 1000))
        self.assertIsNone(records[1]["speaker_id"])

    def test_rttm_rejects_multiple_recordings_channels_and_nonfinite_times(self):
        first = "SPEAKER fixture 1 0.000 1.000 <NA> <NA> A <NA> <NA>\n"
        for second in (
            "SPEAKER other 1 1.000 1.000 <NA> <NA> B <NA> <NA>",
            "SPEAKER fixture 2 1.000 1.000 <NA> <NA> B <NA> <NA>",
            "SPEAKER fixture 1 NaN 1.000 <NA> <NA> B <NA> <NA>",
        ):
            with self.subTest(second=second), self.assertRaises(AudioError):
                parse_rttm(first + second)

    def test_rttm_recording_mismatch_rejected(self):
        with self.assertRaises(AudioError):
            parse_rttm("SPEAKER other 1 0.000 1.000 <NA> <NA> A <NA> <NA>", expected_file_id="fixture")


class WhisperAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.executable = self.root / "whisper-cli"
        self.model = self.root / "model.bin"
        self.audio = self.root / "supplied.wav"
        for path in (self.executable, self.model, self.audio):
            path.touch()
        self.vendor_json = {"transcription": [{"offsets": {"from": 0, "to": 1000}, "text": "Synthetic speech"}]}

    def invoke(self, **extra):
        return transcribe_whisper_cpp(
            self.audio, executable=self.executable, model_path=self.model, session_id="test", **extra,
        )

    def test_upstream_offsets_are_ms_and_turn_marker_is_not_identity(self):
        self.vendor_json["transcription"][0]["speaker_turn_next"] = True
        result = parse_whisper_cpp(self.vendor_json, session_id="test")[0]
        self.assertEqual(result["end_ms"], 1000)
        self.assertIsNone(result["speaker_id"])
        self.assertEqual(result["speaker_provenance"], "unknown")

    def test_explicit_local_process_argv_and_private_output_cleanup(self):
        outputs = []
        def vendor_process(argv, **kwargs):
            self.assertEqual(argv[0], str(self.executable.resolve()))
            self.assertEqual(argv[argv.index("--model") + 1], str(self.model.resolve()))
            self.assertNotIn("shell", kwargs)
            self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
            self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
            output = Path(argv[argv.index("--output-file") + 1]).with_suffix(".json")
            self.assertEqual(output.parent.stat().st_mode & 0o777, 0o700)
            outputs.append(output)
            output.write_text(json.dumps(self.vendor_json))
        with patch("rightyo.audio.subprocess.run", side_effect=vendor_process):
            result = self.invoke()
        self.assertEqual(result[0]["provenance"], "recorded-file")
        self.assertFalse(outputs[0].exists())

    def test_actual_rttm_schema_can_join_whisper_output(self):
        diarization = self.root / "speaker.rttm"
        diarization.write_text("SPEAKER supplied 1 0.000 1.000 <NA> <NA> A <NA> <NA>")
        def vendor_process(argv, **kwargs):
            output = Path(argv[argv.index("--output-file") + 1]).with_suffix(".json")
            output.write_text(json.dumps(self.vendor_json))
        with patch("rightyo.audio.subprocess.run", side_effect=vendor_process):
            result = self.invoke(diarization_path=diarization)
        self.assertEqual(result[0]["speaker_id"], "Speaker A")
        self.assertEqual(result[0]["speaker_provenance"], "diarization-timeline")

    def test_missing_model_never_starts_process(self):
        self.model.unlink()
        with patch("rightyo.audio.subprocess.run") as process, self.assertRaises(AudioError):
            self.invoke()
        process.assert_not_called()

    def test_other_session_timeline_never_starts_process(self):
        diarization = self.root / "speaker.json"
        diarization.write_text(json.dumps({"schema_version": 1, "session_id": "other", "speakers": []}))
        with patch("rightyo.audio.subprocess.run") as process, self.assertRaises(AudioError):
            self.invoke(diarization_path=diarization)
        process.assert_not_called()

    def test_process_failure_never_exposes_vendor_content(self):
        error = subprocess.CalledProcessError(2, ["private-path"], output="private speech", stderr="secret")
        with patch("rightyo.audio.subprocess.run", side_effect=error), self.assertRaises(AudioError) as caught:
            self.invoke()
        self.assertEqual(str(caught.exception), "Local recognizer failed")
        self.assertIsNone(caught.exception.__cause__)

    def test_missing_or_duplicate_json_output_fails(self):
        with patch("rightyo.audio.subprocess.run"), self.assertRaises(AudioError):
            self.invoke()
        def malformed(argv, **kwargs):
            output = Path(argv[argv.index("--output-file") + 1]).with_suffix(".json")
            output.write_text('{"transcription": [], "transcription": []}')
        with patch("rightyo.audio.subprocess.run", side_effect=malformed), self.assertRaises(AudioError):
            self.invoke()


if __name__ == "__main__":
    unittest.main()
