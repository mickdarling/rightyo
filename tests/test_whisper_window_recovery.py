"""#285: authored recognizer documents and synthetic PCM, without native inference."""

import io
import re
import tempfile
import threading
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_edge_attribution import Timeline
from test_live_audio import SILENCE, VOICE, FakeDiarizer

from rightyo.live_audio import (
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

    def settle(self, processor):
        recovery = processor._recovery
        if recovery is not None:
            self.assertTrue(recovery.done.wait(3), "authored recovery did not finish")
            processor._drain_results()

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
            segment(" reversal", 200, 180),
            segment(" neighbour", 180, 400),
            segment(" padding", 500, 2000),
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
            segment(" bounds", 0, 2001),
            segment(" reversal", 200, 179),
            segment(" order", 79, 180),
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
        self.assertEqual(window.reasons, (0, 0, 0, 1, 0, 0))

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
        self.assertTrue(callable(first.args[2]))
        self.assertTrue(callable(retry.args[2]))  # separate, cancellable process ownership
        self.assertEqual(first.args[0].whisper_executable, retry.args[0].whisper_executable)
        self.assertEqual(first.args[0].whisper_model, retry.args[0].whisper_model)
        self.assertEqual(retry.args[0].timeout_seconds, 5.0)
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
            retained = processor._utterance if len(calls) == 1 else processor._waiting[0].pcm
            self.assertEqual(bytes(retained), pcm)
            self.assertEqual(len(pcm), 1640 * BYTES_PER_MS)
            return doc(BAD) if len(calls) == 1 else doc(GOOD)

        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            self.feed(processor)
            processor.finish()
        self.assertEqual(calls[-1].timeout_seconds, 0.2)
        self.assertEqual(processor.buffered_audio_bytes, 0)
        self.assertEqual(len(turns), 1)

    def test_all_bad_retry_failures_emit_gap_and_session_survives(self):
        for retry in (
            doc(BAD),
            doc(),
            LiveAudioError("Local recognizer timed out"),
            RuntimeError("AUTHORED_EXCEPTION_CONTENT"),
            {},
        ):
            with self.subTest(retry=retry):
                processor, turns, reports = self.processor()
                with patch(
                    "rightyo.live_audio._transcribe", side_effect=[doc(BAD), retry, doc(GOOD)]
                ) as decode:
                    self.feed(processor)
                    self.settle(processor)
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
                self.assertNotIn("AUTHORED_EXCEPTION_CONTENT", reports[0])

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
                r"reasons=0,2,0,0,0,0 asr_elapsed_ms=\d+\n",
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
                        self.settle(processor)
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

    def test_literal_twenty_ms_jitter_repairs_but_twenty_one_is_rejected(self):
        accepted = _recognizer_window(doc(segment(" accepted", 200, 180)), 1000)
        self.assertEqual(
            accepted.units,
            [
                {"text": " accepted", "start_ms": 200, "end_ms": 200},
            ],
        )
        self.assertEqual(accepted.rejected, 0)
        rejected = _recognizer_window(doc(segment(" rejected", 200, 179)), 1000)
        self.assertEqual(rejected.units, [])
        self.assertEqual(rejected.rejected, 1)
        # Check the neighbour-order boundary independently too.
        for onset, texts in ((80, [" first", " next"]), (79, [" first"])):
            window = _recognizer_window(
                doc(segment(" first", 100, 150), segment(" next", onset, 160)), 1000
            )
            self.assertEqual([u["text"] for u in window.units], texts)

    def test_literal_five_second_retry_and_twelve_second_first_decode_deadlines(self):
        processor, turns, _reports = self.processor(timeout_seconds=120)
        with patch("rightyo.live_audio._transcribe", side_effect=[doc(BAD), doc(GOOD)]) as decode:
            self.feed(processor)
            processor.finish()
        self.assertEqual(
            [call.args[0].timeout_seconds for call in decode.call_args_list], [12.0, 5.0]
        )
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"])

    def test_zero_length_at_exact_audio_end_is_kept_without_a_speaker_or_retry(self):
        value = doc(segment(" END_POINT", 1000, 1000))
        window = _recognizer_window(value, 1000)
        self.assertEqual(
            window.units,
            [
                {"text": " END_POINT", "start_ms": 1000, "end_ms": 1000},
            ],
        )
        self.assertEqual((window.repaired, window.rejected), (1, 0))
        processor, turns, _reports = self.processor()
        with patch("rightyo.live_audio._transcribe", return_value=value) as decode:
            for _ in range(50):
                processor.push_pcm16(VOICE)
            processor.finish()
        self.assertEqual(decode.call_count, 1)
        self.assertEqual(
            [(t.text, t.start_ms, t.end_ms, t.speaker_id) for t in turns],
            [("END_POINT", 1000, 1000, None)],
        )

    def test_review_reproduction_excludes_timeline_evidence_before_the_loss(self):
        value = doc(
            segment(" PREFIX", 0, 100),
            BAD,
            segment(" UNKNOWN", 200, 300),
            segment(" LOCAL", 1000, 1100),
        )
        for before in ((), ((0, 150, 1),)):
            with self.subTest(before=before):
                processor, turns, _reports = self.processor(
                    edge_attribution_ms=100, diarizer=Timeline(*before, (1000, 1150, 1))
                )
                with patch("rightyo.live_audio._transcribe", return_value=value):
                    self.feed(processor)
                    processor.finish()
                self.assertEqual(
                    [(t.text, t.speaker_id) for t in turns][1:],
                    [("UNKNOWN", None), ("LOCAL", "Speaker A")],
                )

    def blocked_retry(self, *, later_bad=False, **options):
        processor, turns, reports = self.processor(**options)
        entered, release = threading.Event(), threading.Event()
        first_calls, retry_calls = [], []

        def decode(config, pcm, register):
            if config.timeout_seconds == 5.0:
                retry_calls.append(pcm)
                entered.set()
                if not release.wait(5):
                    raise LiveAudioError("Authored retry watchdog")
                return doc(GOOD)
            first_calls.append(pcm)
            if len(first_calls) == 1 or later_bad:
                return doc(BAD)
            return doc(segment(" LATER_FIXTURE", 20, 180))

        self.addCleanup(release.set)
        return processor, turns, reports, entered, release, first_calls, retry_calls, decode

    def test_hung_retry_does_not_block_ingestion_or_let_later_results_overtake(self):
        processor, turns, _reports, entered, release, first, retries, decode = self.blocked_retry()
        finished = threading.Event()
        errors = []

        def ingest():
            try:
                self.feed(processor)
                self.feed(processor)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            producer = threading.Thread(target=ingest)
            producer.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertTrue(finished.wait(2), "ingestion waited for the blocked retry")
                self.assertEqual(errors, [])
                self.assertFalse(release.is_set())
                self.assertEqual(processor.received_ms, 3280)
                self.assertEqual((len(first), len(retries)), (2, 1))
                self.assertEqual(len(processor._waiting), 2)
                self.assertEqual(turns, [])
            finally:
                release.set()
                producer.join(3)
            processor.finish()
        self.assertFalse(producer.is_alive())
        self.assertEqual(
            [(t.text, t.start_ms) for t in turns],
            [("RECOVERED_FIXTURE", 20), ("LATER_FIXTURE", 1660)],
        )
        self.assertEqual([t.utterance_id for t in turns], ["live-1", "live-2"])

    def test_newer_invalid_window_skips_retry_behind_the_one_pending_recovery(self):
        processor, turns, reports, entered, release, first, retries, decode = self.blocked_retry(
            later_bad=True
        )
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                self.feed(processor)
                processor.release_pending()  # a stalled source must not bypass the hold
                self.assertEqual((len(first), len(retries)), (2, 1))
                self.assertEqual(turns, [])
                self.assertEqual(reports, [])
            finally:
                release.set()
            processor.finish()
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"])
        self.assertEqual(processor.skipped_utterances, 1)
        self.assertEqual(len(reports), 2)
        self.assertIn("outcome=redecoded window_id=1", reports[0])
        self.assertIn("outcome=lost window_id=2 start_ms=1640 end_ms=3280", reports[1])
        self.assertIn("reasons=0,1,0,0,0,1", reports[1])
        self.assertNotIn("UNUSABLE_FIXTURE", " ".join(reports))

    def test_hold_bound_resolves_oldest_loss_and_keeps_every_newer_good_result(self):
        processor, turns, reports, entered, release, first, retries, decode = self.blocked_retry()
        recovery = None
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                recovery = processor._recovery
                for count in (2, 3):
                    self.feed(processor)
                    self.assertEqual(len(processor._waiting), count)
                    self.assertEqual(turns, [])
                self.feed(processor)
                self.assertEqual(len(processor._waiting), 0)
                self.assertTrue(recovery.cancelled.is_set())
                self.assertEqual([t.text for t in turns], ["LATER_FIXTURE"] * 3)
                self.assertEqual([t.start_ms for t in turns], [1660, 3300, 4940])
                self.assertEqual((len(first), len(retries)), (4, 1))
                self.assertLessEqual(processor.buffered_audio_bytes, 1640 * 32)
                self.assertIn("outcome=lost window_id=1", reports[0])
                self.assertIn("reasons=0,1,0,0,0,1", reports[0])
            finally:
                release.set()
            self.assertTrue(recovery.done.wait(2))
            processor.finish()
        self.assertEqual(len(turns), 3)  # late recovery cannot resurrect cancelled text
        self.assertEqual(processor.buffered_audio_bytes, 0)

    def test_expired_retry_resolves_loss_without_waiting_for_the_blocked_worker(self):
        processor, turns, reports, entered, release, _first, _retries, decode = self.blocked_retry()
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                self.feed(processor)
                recovery = processor._recovery
                recovery.expires = 0  # authored supervisor deadline already expired
                processor.release_pending()
                self.assertTrue(recovery.cancelled.is_set())
                self.assertFalse(release.is_set())
                self.assertEqual([t.text for t in turns], ["LATER_FIXTURE"])
                self.assertIn("outcome=lost window_id=1", reports[0])
                processor.finish()  # cancelled worker does not delay EOF either
                self.assertEqual(processor.buffered_audio_bytes, 0)
            finally:
                release.set()
            self.assertTrue(recovery.done.wait(2))
        self.assertEqual(len(turns), 1)

    def test_speaker_renewal_cannot_reuse_a_delayed_old_channel(self):
        processor, turns, _reports, entered, release, _first, _retries, decode = self.blocked_retry(
            diarizer=FakeDiarizer
        )
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                old = processor._diarizer
                processor.reset_speakers()
                for _ in range(10):
                    processor.push_pcm16(SILENCE)
                self.assertIs(processor._diarizer, old)  # pending window defers renewal
                for _ in range(500):
                    processor.push_pcm16(SILENCE)
                self.assertIsNot(processor._diarizer, old)  # existing forced bound still applies
                self.assertTrue(old.closed)
            finally:
                release.set()
            processor.finish()
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"])
        self.assertIsNone(turns[0].speaker_id)

    def test_stop_cancels_recovery_and_discards_held_results_without_late_callbacks(self):
        processor, turns, reports, entered, release, _first, _retries, decode = self.blocked_retry()
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                self.feed(processor)
                recovery = processor._recovery
                processor.close()
                self.assertTrue(recovery.cancelled.is_set())
                self.assertEqual(processor.buffered_audio_bytes, 0)
            finally:
                release.set()
            self.assertTrue(recovery.done.wait(2))
            processor.release_pending()
        self.assertEqual((turns, reports), ([], []))

    def test_deferred_windows_keep_their_utterance_local_speaker_namespaces(self):
        diarizer = FakeDiarizer(None)
        diarizer.speaker_provenance = "diarization-utterance"
        processor, turns, _reports, entered, release, _first, _retries, decode = self.blocked_retry(
            diarizer=diarizer
        )
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                self.feed(processor)
            finally:
                release.set()
            processor.finish()
        self.assertEqual([t.speaker_id for t in turns], ["u1 Speaker A", "u2 Speaker A"])

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
