"""#285: authored recognizer documents and synthetic PCM, without native inference."""

import io
import re
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_edge_attribution import Timeline
from test_live_audio import SILENCE, VOICE, FakeDiarizer

from rightyo.live_audio import (
    ASR_END_PADDING_MS,
    ASR_REDECODE_SECONDS,
    ASR_TIMING_JITTER_MS,
    BYTES_PER_MS,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    _checked_units,
    _recognizer_window,
)
from rightyo.tool import _stderr


def segment(text, start, end):
    return {"text": text, "offsets": {"from": start, "to": end}}


def doc(*segments):
    return {"transcription": list(segments)}


BAD = segment(" UNUSABLE_FIXTURE", -500, 100)
GOOD = segment(" RECOVERED_FIXTURE", 20, 180)


class WhisperWindowRecoveryTests(unittest.TestCase):
    def processor(self, **options):
        turns, reports = [], []
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "authored-asset"
        asset.touch()
        # Configuration requires existing assets; native inference is still fully faked.
        config = LiveConfig(
            "window-replay",
            whisper_executable=asset,
            whisper_model=asset,
            diarizer=FakeDiarizer(None),
            provenance="causal-replay",
            report=reports.append,
        )
        with patch("rightyo.live_audio.recognizer_dir", return_value="/var/empty"):
            processor = LiveProcessor(replace(config, **options), turns.append)
        self.addCleanup(processor.close)
        return processor, turns, reports

    def feed(self, processor):
        for _ in range(10):
            processor.push_pcm16(VOICE)
        for _ in range(72):
            processor.push_pcm16(SILENCE)
        self.assertFalse(processor.failed)
        self.assertFalse(processor.closed)

    def test_mixed_good_and_bad_segments_keep_good_text_and_times(self):
        processor, turns, reports = self.processor(turn_merge_gap_ms=5000)
        mixed = doc(segment(" PREFIX_FIXTURE", 0, 100), BAD, segment(" TAIL_FIXTURE", 200, 300))
        with patch("rightyo.live_audio._transcribe", return_value=mixed) as decode:
            self.feed(processor)
            processor.finish()
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(
            [(t.text, t.start_ms, t.end_ms, t.speaker_id) for t in turns],
            [
                ("PREFIX_FIXTURE", 0, 100, "Speaker A"),
                ("TAIL_FIXTURE", 200, 300, "Speaker A"),
            ],
        )
        self.assertIn("outcome=salvaged", reports[0])
        for piece in mixed["transcription"]:
            self.assertNotIn(piece["text"].strip(), reports[0])
        self.assertIn("segments=3 retry_segments=0 units=2", reports[0])
        self.assertEqual(processor.skipped_utterances, 0)

    def test_bad_first_segment_keeps_later_segments(self):
        processor, turns, _reports = self.processor()
        with patch("rightyo.live_audio._transcribe", return_value=doc(BAD, GOOD)) as decode:
            self.feed(processor)
            processor.finish()
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(
            [(t.text, t.start_ms, t.end_ms) for t in turns],
            [
                ("RECOVERED_FIXTURE", 20, 180),
            ],
        )

    def test_fixable_offsets_are_clamped_and_zero_duration_stays_a_point(self):
        value = doc(
            segment(" zero", 100, 100),
            segment(" reversal", 200, 200 - ASR_TIMING_JITTER_MS),
            segment(" neighbour", 200 - ASR_TIMING_JITTER_MS, 400),
            segment(" padding", 500, 1000 + ASR_END_PADDING_MS),
        )
        window = _recognizer_window(value, 1000)
        self.assertEqual(
            [(u["start_ms"], u["end_ms"]) for u in window.units],
            [
                (100, 100),
                (200, 200),
                (200, 400),
                (500, 1000),
            ],
        )
        self.assertEqual(window.repaired, 4)
        self.assertEqual(window.rejected, 0)
        self.assertIsNotNone(_checked_units(window.units, 1000))
        processor, turns, reports = self.processor()
        with patch("rightyo.live_audio._transcribe", return_value=value):
            self.feed(processor)
            processor.finish()
        self.assertTrue(turns)
        self.assertIsNone(turns[0].speaker_id)  # no invented duration/direct owner evidence
        self.assertIn("outcome=salvaged", reports[0])

    def test_just_outside_tolerances_are_rejected_without_poisoning_later_text(self):
        for bad in (
            segment(" bounds", 0, 1000 + ASR_END_PADDING_MS + 1),
            segment(" reversal", 200, 200 - ASR_TIMING_JITTER_MS - 1),
            segment(" order", 100 - ASR_TIMING_JITTER_MS - 1, 180),
            segment(" boolean", False, 180),
            segment(" nan", float("nan"), 180),
        ):
            with self.subTest(bad=bad):
                window = _recognizer_window(
                    doc(segment(" first", 100, 150), bad, segment(" last", 300, 400)), 1000
                )
                self.assertEqual([u["text"] for u in window.units], [" first", " last"])
                self.assertEqual(window.rejected, 1)
                self.assertEqual(window.gaps, {1})

    def test_out_of_order_segment_does_not_discard_later_segments(self):
        window = _recognizer_window(
            doc(
                segment(" first", 100, 300),
                segment(" backwards", 0, 50),
                segment(" last", 400, 500),
            ),
            1000,
        )
        self.assertEqual([u["text"] for u in window.units], [" first", " last"])
        self.assertEqual(window.reasons, (0, 0, 0, 1, 0))

    def test_valid_overlapping_segments_fall_back_without_losing_text_or_gap(self):
        first = segment(" first words", 0, 500)
        first["tokens"] = [
            {"text": " first", "offsets": {"from": 0, "to": 100}},
            {"text": " words", "offsets": {"from": 400, "to": 500}},
        ]
        window = _recognizer_window(doc(first, BAD, segment(" next", 200, 600)), 1000)
        self.assertEqual(
            window.units,
            [
                {"text": " first words", "start_ms": 0, "end_ms": 500},
                {"text": " next", "start_ms": 200, "end_ms": 600},
            ],
        )
        self.assertEqual(window.rejected, 1)
        self.assertEqual(window.gaps, {1})
        self.assertIsNotNone(_checked_units(window.units, 1000))

    def test_all_bad_then_recovered_redecodes_exactly_once_with_the_same_pcm(self):
        processor, turns, reports = self.processor()
        with patch("rightyo.live_audio._transcribe", side_effect=[doc(BAD), doc(GOOD)]) as decode:
            self.feed(processor)
            processor.finish()
        self.assertEqual(decode.call_count, 2)
        first, retry = decode.call_args_list
        self.assertIs(first.args[1], retry.args[1])
        self.assertEqual(first.args[2], retry.args[2])  # same cancellation registration
        self.assertEqual(first.args[0].whisper_executable, retry.args[0].whisper_executable)
        self.assertEqual(first.args[0].whisper_model, retry.args[0].whisper_model)
        self.assertEqual(retry.args[0].timeout_seconds, ASR_REDECODE_SECONDS)
        self.assertEqual(
            [(t.text, t.start_ms, t.end_ms) for t in turns],
            [
                ("RECOVERED_FIXTURE", 20, 180),
            ],
        )
        self.assertEqual(processor.skipped_utterances, 0)
        self.assertIn("outcome=redecoded window_id=1 start_ms=0 end_ms=1640", reports[0])
        for piece in (BAD, GOOD):
            self.assertNotIn(piece["text"].strip(), reports[0])

    def test_pcm_is_retained_until_validation_and_retry_respect_shorter_timeout(self):
        processor, turns, _reports = self.processor(timeout_seconds=0.2)
        calls = []

        def decode(config, pcm, register):
            calls.append(config)
            self.assertEqual(bytes(processor._utterance), pcm)
            self.assertEqual(len(pcm), 1640 * BYTES_PER_MS)
            return doc(BAD) if len(calls) == 1 else doc(GOOD)

        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            self.feed(processor)
        self.assertEqual(calls[-1].timeout_seconds, 0.2)
        self.assertEqual(processor.buffered_audio_bytes, 0)
        self.assertEqual(len(turns), 1)

    def test_all_bad_retry_failures_emit_gap_and_session_survives(self):
        for retry in (doc(BAD), doc(), LiveAudioError("Local recognizer timed out"), {}):
            with self.subTest(retry=retry):
                processor, turns, reports = self.processor()
                with patch(
                    "rightyo.live_audio._transcribe", side_effect=[doc(BAD), retry, doc(GOOD)]
                ) as decode:
                    self.feed(processor)
                    self.assertEqual(decode.call_count, 2)
                    self.assertEqual(turns, [])
                    self.feed(processor)
                    processor.finish()
                self.assertEqual(decode.call_count, 3)
                self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"])
                self.assertGreater(turns[0].start_ms, 1640)
                self.assertEqual(processor.skipped_utterances, 1)
                self.assertEqual(len(reports), 1)
                self.assertIn("outcome=lost window_id=1 start_ms=0 end_ms=1640", reports[0])
                self.assertRegex(reports[0], r"reasons=[0-9,]+ asr_elapsed_ms=\d+$")

    def test_gap_diagnostic_contains_no_recognizer_text(self):
        output = io.StringIO()
        processor, _turns, _reports = self.processor(report=_stderr)
        with (
            redirect_stderr(output),
            patch("rightyo.live_audio._transcribe", return_value=doc(BAD)),
        ):
            self.feed(processor)
            processor.finish()
        diagnostic = output.getvalue()
        self.assertNotIn(BAD["text"].strip(), diagnostic)
        self.assertRegex(diagnostic, r"^rightyo: asr_window outcome=lost window_id=1 ")
        # Exact grammar prevents even an arbitrary added word/text/path from leaking.
        self.assertIsNotNone(
            re.fullmatch(
                r"rightyo: asr_window outcome=lost window_id=1 start_ms=0 end_ms=1640 "
                r"segments=1 retry_segments=1 units=0 repaired=0 rejected=2 "
                r"reasons=0,2,0,0,0 asr_elapsed_ms=\d+\n",
                diagnostic,
            )
        )

    def test_loss_at_window_edges_prevents_joining_across_it(self):
        for broken in (doc(BAD, GOOD), doc(GOOD, BAD), doc(BAD)):
            with self.subTest(broken=broken):
                processor, turns, _reports = self.processor(turn_merge_gap_ms=5000)
                responses = [doc(GOOD), broken]
                if len(broken["transcription"]) == 1:
                    responses.append(doc(BAD))  # recovery fails as well
                responses.append(doc(GOOD))
                with patch("rightyo.live_audio._transcribe", side_effect=responses):
                    for _ in range(3):
                        self.feed(processor)
                    processor.finish()
                self.assertGreaterEqual(len(turns), 2)
                # No turn spans both windows on opposite sides of the unresolved loss.
                self.assertTrue(all(not (t.start_ms < 1640 and t.end_ms > 3280) for t in turns))

    def test_edge_attribution_cannot_bridge_a_dropped_middle_segment(self):
        processor, turns, _reports = self.processor(
            edge_attribution_ms=1000,
            turn_merge_gap_ms=5000,
            diarizer=Timeline((0, 150, 1)),
        )
        mixed = doc(segment(" LABELLED", 0, 100), BAD, segment(" UNKNOWN", 200, 300))
        with patch("rightyo.live_audio._transcribe", return_value=mixed):
            self.feed(processor)
            processor.finish()
        self.assertEqual(
            [(t.text, t.speaker_id) for t in turns],
            [
                ("LABELLED", "Speaker A"),
                ("UNKNOWN", None),
            ],
        )

    def test_silent_window_does_not_retry(self):
        processor, turns, reports = self.processor()
        with patch("rightyo.live_audio._transcribe", return_value=doc()) as decode:
            self.feed(processor)
            processor.finish()
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(turns, [])
        self.assertEqual(reports, [])

    def test_private_window_has_no_diagnostics_or_public_loss_counts(self):
        processor, _turns, reports = self.processor(private=lambda start, end: True)
        with patch("rightyo.live_audio._transcribe", return_value=doc(BAD)):
            self.feed(processor)
            processor.finish()
        self.assertEqual(reports, [])
        self.assertEqual((processor.skipped_segments, processor.skipped_utterances), (0, 0))


if __name__ == "__main__":
    unittest.main()
