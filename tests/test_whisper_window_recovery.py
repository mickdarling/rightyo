"""#285: authored recognizer documents and synthetic PCM, without native inference."""

import io
import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from test_edge_attribution import Timeline
from test_live_audio import SILENCE, VOICE, FakeDiarizer

from rightyo.contracts import Addressing
from rightyo.live_audio import (
    BYTES_PER_MS,
    LiveAudioError,
    LiveConfig,
    LiveProcessor,
    _checked_units,
    _recognizer_window,
    _WindowRecovery,
)
from rightyo.memory import TranscriptMemory
from rightyo.tool import _stderr
from rightyo.tool_events import SpeechEvents


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

    def test_configured_first_decode_timeout_and_literal_five_second_retry_deadline(self):
        processor, turns, _reports = self.processor(timeout_seconds=120)
        with patch("rightyo.live_audio._transcribe", side_effect=[doc(BAD), doc(GOOD)]) as decode:
            self.feed(processor)
            processor.finish()
        self.assertEqual(
            [call.args[0].timeout_seconds for call in decode.call_args_list], [120.0, 5.0]
        )
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"])

    def test_default_first_decode_allows_thirteen_second_synthetic_child(self):
        processor, turns, _reports = self.processor()
        clock = [100.0]
        child = Mock(args=["authored-local-asr"], stdin=io.BytesIO(), stdout=io.BytesIO())
        child.poll.return_value = 0
        child.wait.return_value = 0

        def read(process, deadline):
            self.assertIs(process, child)
            clock[0] += 13.0  # scaled child runtime, no real 13 s sleep or model
            if clock[0] > deadline:
                raise subprocess.TimeoutExpired(child.args, 13.0)
            return json.dumps(doc(GOOD)).encode()

        with (
            patch("rightyo.live_audio.recognizer_dir", return_value="/var/empty"),
            patch("rightyo.live_audio.subprocess.Popen", return_value=child),
            patch("rightyo.live_audio._bounded_stdout", side_effect=read),
            patch("rightyo.live_audio.time.monotonic", side_effect=lambda: clock[0]),
        ):
            try:
                self.feed(processor)  # exercise the real CLI deadline/wait path with a fake child
                processor.finish()
            except LiveAudioError:
                self.fail("configured 30-second first decode rejected a 13-second synthetic child")
        self.assertEqual(processor.config.timeout_seconds, 30)
        self.assertEqual(child.wait.call_args.kwargs["timeout"], 17.0)
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
        for before in ((), ((0, 150, 1),), ((0, 200, 1),)):
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

    def test_touching_competing_speaker_blocks_edge_inference_at_both_run_edges(self):
        for units, timeline, expected in (
            (
                doc(segment(" LABELLED", 0, 100), segment(" UNKNOWN", 200, 300)),
                Timeline((0, 150, 1), (300, 400, 2)),
                [("LABELLED", "Speaker A"), ("UNKNOWN", None)],
            ),
            (
                doc(segment(" UNKNOWN", 100, 200), segment(" LABELLED", 300, 400)),
                Timeline((0, 100, 2), (250, 400, 1)),
                [("UNKNOWN", None), ("LABELLED", "Speaker A")],
            ),
            (
                doc(segment(" LABELLED", 0, 100), segment(" UNKNOWN", 1540, 1640)),
                Timeline((0, 150, 1), (1640, 1740, 2)),
                [("LABELLED", "Speaker A"), ("UNKNOWN", None)],
            ),
        ):
            with self.subTest(units=units):
                processor, turns, _reports = self.processor(
                    edge_attribution_ms=2000, diarizer=timeline
                )
                with patch("rightyo.live_audio._transcribe", return_value=units):
                    self.feed(processor)
                    processor.finish()
                self.assertEqual([(t.text, t.speaker_id) for t in turns], expected)

    def test_recovery_preserves_privacy_barriers_and_accepts_the_public_request(self):
        from test_duck_mode import DuckControllerTests

        for end_privacy in (True, False):
            for recover_before_next in (True, False):
                with self.subTest(end_privacy=end_privacy, early=recover_before_next):
                    events = SpeechEvents()
                    events.start(
                        "window-replay", now_ms=0, addressing=Addressing(names=("Jarvis",))
                    )
                    events.drain()
                    controller = DuckControllerTests.controller(
                        self, constructor={"event_publisher": events}
                    )
                    controller._decision_status = "ready"
                    work, memory, accepted = queue.Queue(), TranscriptMemory(), []
                    processor, turns, _reports, entered, release, first, retries, decode = (
                        self.blocked_retry(
                            turn_merge_gap_ms=5000, private=controller._heard_private_span
                        )
                    )
                    controller._processor = processor

                    def accept(turn):
                        turns.append(turn)
                        accepted.append(
                            controller._accept(controller._generation, work, memory, turn)
                        )

                    processor.on_turn = accept

                    def feed(frames):
                        for frame in frames:
                            self.assertTrue(
                                controller._feed(controller._generation, processor, frame, "stdin")
                            )

                    def responses(config, pcm, register):
                        if config.timeout_seconds == 5.0 or not first:
                            return decode(config, pcm, register)
                        first.append(pcm)
                        # Second window begins at 1640 (200 ms silent pre-roll).
                        return doc(segment(" Jarvis, add the authored quokka.", 240, 400))

                    with patch("rightyo.live_audio._transcribe", side_effect=responses):
                        try:
                            feed([VOICE] * 10 + [SILENCE] * 72)
                            self.assertTrue(entered.wait(2))
                            self.assertEqual(controller._received_ms, 1640)
                            controller.duck(True)
                            feed([SILENCE] * 10)
                            self.assertEqual(controller._received_ms, 1840)
                            if end_privacy:
                                controller.duck(False)
                            if recover_before_next:
                                release.set()
                                self.settle(processor)
                                self.assertEqual(
                                    [t.text for t in turns],
                                    ["RECOVERED_FIXTURE"] if end_privacy else [],
                                )
                            feed([VOICE] * 10 + [SILENCE] * 72)
                            self.assertEqual(turns if not recover_before_next else turns[1:], [])
                        finally:
                            release.set()
                        processor.finish()
                    self.assertEqual(len(retries), 1)
                    self.assertEqual(
                        [(t.start_ms, t.end_ms) for t in turns],
                        [(20, 180), (1880, 2040)] if end_privacy else [(1880, 2040)],
                    )
                    # Existing fail-closed policy: while privacy is still on, even an
                    # earlier public span is private at acceptance. Settle discards it;
                    # the private continuation must still stand alone, never joined.
                    self.assertEqual(accepted, [False, False] if end_privacy else [True])
                    queued = [work.get_nowait() for _ in range(work.qsize())]
                    self.assertEqual(queued, turns if end_privacy else [])
                    public = [
                        e["turn"]["text"] for e in events.drain() if e["type"] == "transcript"
                    ]
                    self.assertEqual(public, [t.text for t in queued])

    def test_input_stall_barrier_waits_for_recovery_and_precedes_later_text(self):
        processor, turns, _reports, entered, release, _first, _retries, decode = self.blocked_retry(
            turn_merge_gap_ms=5000
        )
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                for _ in range(100):
                    processor.release_pending()
                self.assertEqual(len(processor._waiting), 1)
                self.feed(processor)
                self.assertEqual(turns, [])
            finally:
                release.set()
            processor.finish()
        self.assertEqual(
            [(t.text, t.start_ms, t.end_ms) for t in turns],
            [("RECOVERED_FIXTURE", 20, 180), ("LATER_FIXTURE", 1660, 1820)],
        )

    def blocked_retry(self, *, later_bad=False, **options):
        processor, turns, reports = self.processor(**options)
        entered, release = threading.Event(), threading.Event()
        first_calls, retry_calls = [], []

        def decode(config, pcm, register):
            if config.timeout_seconds == 5.0:
                retry_calls.append(pcm)
                entered.set()
                deadline = time.monotonic() + 5
                while not release.wait(0.01):
                    if register.__self__.cancelled.is_set():
                        raise LiveAudioError("Authored retry cancelled")
                    if time.monotonic() >= deadline:
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

    def test_two_invalid_windows_queue_and_both_recover_in_order(self):
        processor, turns, reports, entered, release, first, retries, decode = self.blocked_retry(
            later_bad=True
        )
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                self.feed(processor)
                processor.release_pending()
                self.assertEqual((len(first), len(retries)), (2, 1))
                self.assertEqual(len(processor._recoveries), 2)
                self.assertEqual((turns, reports), ([], []))
            finally:
                release.set()
            processor.finish()
        self.assertEqual(len(retries), 2)
        self.assertEqual(
            [(t.text, t.start_ms) for t in turns],
            [("RECOVERED_FIXTURE", 20), ("RECOVERED_FIXTURE", 1660)],
        )
        self.assertEqual(processor.skipped_utterances, 0)
        self.assertEqual(len(reports), 2)
        self.assertTrue(all("outcome=redecoded" in report for report in reports))
        self.assertIn("window_id=1", reports[0])
        self.assertIn("window_id=2", reports[1])
        self.assertNotIn("UNUSABLE_FIXTURE", " ".join(reports))

    def test_recoverable_oldest_and_three_good_windows_are_all_retained(self):
        processor, turns, reports, entered, release, first, retries, decode = self.blocked_retry()
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                recovery = processor._recovery
                for count in (2, 3, 4):
                    self.feed(processor)
                    self.assertEqual(len(processor._waiting), count)
                    self.assertFalse(recovery.cancelled.is_set())
                    self.assertEqual(turns, [])
                self.assertEqual((len(first), len(retries)), (4, 1))
                self.assertEqual(processor.buffered_audio_bytes, 4 * 1640 * 32)
            finally:
                release.set()
            processor.finish()
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"] + ["LATER_FIXTURE"] * 3)
        self.assertEqual([t.start_ms for t in turns], [20, 1660, 3300, 4940])
        self.assertEqual(processor.skipped_utterances, 0)
        self.assertEqual(len(reports), 1)
        self.assertIn("outcome=redecoded window_id=1", reports[0])
        self.assertEqual(processor.buffered_audio_bytes, 0)

    def test_four_recovery_slots_drop_only_the_newest_when_full(self):
        processor, turns, reports, entered, release, first, retries, decode = self.blocked_retry(
            later_bad=True
        )
        with patch("rightyo.live_audio._transcribe", side_effect=decode):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                recovery = processor._recovery
                for _ in range(4):
                    self.feed(processor)
                self.assertEqual(len(processor._recoveries), 4)
                self.assertEqual(len(retries), 1)
                self.assertFalse(recovery.cancelled.is_set())
                self.assertEqual((turns, reports), ([], []))
            finally:
                release.set()
            processor.finish()
        self.assertEqual((len(first), len(retries)), (5, 4))
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"] * 4)
        self.assertEqual([t.start_ms for t in turns], [20, 1660, 3300, 4940])
        self.assertEqual(processor.skipped_utterances, 1)
        self.assertEqual(len(reports), 5)
        self.assertTrue(all("outcome=redecoded" in report for report in reports[:4]))
        self.assertIn("outcome=lost window_id=5 start_ms=6560 end_ms=8200", reports[4])
        self.assertIn("reasons=0,1,0,0,0,1", reports[4])
        self.assertEqual(processor.buffered_audio_bytes, 0)

    def test_result_hold_overflow_never_cancels_the_older_recovery(self):
        processor, turns, reports, entered, release, _first, _retries, decode = self.blocked_retry()
        with (
            patch("rightyo.live_audio._transcribe", side_effect=decode),
            patch("rightyo.live_audio.ASR_HELD_WINDOWS", 1),
        ):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                recovery = processor._recovery
                for _ in range(3):
                    self.feed(processor)
                self.assertFalse(recovery.cancelled.is_set())
                self.assertEqual(len(processor._waiting), 3)  # recovery, one good, tail gap
                self.assertEqual(processor._waiting[-1].loss_count, 2)
                self.assertEqual((turns, reports), ([], []))
            finally:
                release.set()
            processor.finish()
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE", "LATER_FIXTURE"])
        self.assertEqual(processor.skipped_utterances, 2)
        self.assertIn("window_id=3 last_window_id=4 windows_lost=2", reports[1])
        self.assertIn("start_ms=3280 end_ms=6560", reports[1])

    def test_result_payload_bound_preserves_oldest_and_reports_only_tail_loss(self):
        processor, turns, reports, entered, release, _first, _retries, decode = self.blocked_retry()
        with (
            patch("rightyo.live_audio._transcribe", side_effect=decode),
            patch("rightyo.live_audio.ASR_HELD_BYTES", 1),
        ):
            try:
                self.feed(processor)
                self.assertTrue(entered.wait(2))
                recovery = processor._recovery
                self.feed(processor)
                self.assertFalse(recovery.cancelled.is_set())
                self.assertEqual(len(processor._waiting), 2)
                self.assertEqual(processor._waiting[-1].pcm, b"")
            finally:
                release.set()
            processor.finish()
        self.assertEqual([t.text for t in turns], ["RECOVERED_FIXTURE"])
        self.assertEqual(processor.skipped_utterances, 1)
        self.assertIn("outcome=lost window_id=2", reports[1])

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

    def test_close_during_worker_start_reaps_child_and_releases_all_recovery_data(self):
        processor, turns, reports = self.processor()
        constructing, allow_start = threading.Event(), threading.Event()
        stop_attempted, spawned, reaped = threading.Event(), threading.Event(), threading.Event()
        allow_cleanup, stopped, stopped_again = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        repeat_join = threading.Event()
        workers, published, children, errors = [], [], [], []
        actual_lock = processor._recovery_lock

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "authored-stop":
                    stop_attempted.set()  # Stop is about to contend for the held lock.
                actual_lock.acquire()
                return self

            def __exit__(self, *_args):
                actual_lock.release()

        processor._recovery_lock = ObservedLock()
        original_start, original_join = _WindowRecovery.start, _WindowRecovery.join

        def start(worker):
            workers.append(worker)
            published.append(processor._recovery)
            constructing.set()
            if not allow_start.wait(10):
                raise AssertionError("authored startup watchdog")
            original_start(worker)
            # Keep the creation/start lock held until a real synthetic child exists.
            if not spawned.wait(5):
                raise AssertionError("authored child startup watchdog")

        def join(worker):
            if threading.current_thread().name == "authored-stop-again":
                repeat_join.set()
            original_join(worker)

        def decode(config, _pcm, register):
            if config.timeout_seconds != 5.0:
                return doc(BAD)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            children.append(child)
            spawned.set()
            try:
                try:
                    register(child)
                except LiveAudioError:
                    pass  # Stop may win the registration lock; the child still must be reaped.
                child.wait(timeout=5)
                reaped.set()
                if not allow_cleanup.wait(10):
                    raise AssertionError("authored cleanup watchdog")
                return doc(GOOD)  # Deliberate late result after Stop: never retain or emit it.
            finally:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)

        def ingest():
            try:
                for frame in [VOICE] * 10 + [SILENCE] * 72:
                    processor.push_pcm16(frame)
            except LiveAudioError:
                pass  # Concurrent Stop may end the active push.
            except BaseException as error:
                errors.append(error)

        def stop(done):
            try:
                processor.close()
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        ingest_thread = threading.Thread(target=ingest, name="authored-input")
        stop_thread = threading.Thread(target=stop, args=(stopped,), name="authored-stop")
        repeat_thread = threading.Thread(
            target=stop, args=(stopped_again,), name="authored-stop-again"
        )
        with (
            patch("rightyo.live_audio._transcribe", side_effect=decode),
            patch.object(_WindowRecovery, "start", new=start),
            patch.object(_WindowRecovery, "join", new=join),
            patch.object(
                processor._diarizer, "close", wraps=processor._diarizer.close
            ) as close_diarizer,
        ):
            ingest_thread.start()
            try:
                self.assertTrue(constructing.wait(5))
                worker = workers[0]
                stop_thread.start()
                self.assertTrue(stop_attempted.wait(5))
                self.assertFalse(processor.closed)  # Stop cannot cross construction/start.
                self.assertFalse(stopped.is_set())
                allow_start.set()
                self.assertTrue(reaped.wait(5))
                # Cleanup is paused: Stop must join OUTSIDE the lock, and PCM remains
                # accounted for even after Stop removed both owner-side queues.
                with processor._recovery_lock:
                    self.assertTrue(processor.closed)
                    self.assertEqual((len(processor._waiting), len(processor._recoveries)), (0, 0))
                self.assertEqual(processor.buffered_audio_bytes, 1640 * 32)
                self.assertEqual(worker.packet.pcm, b"")
                self.assertFalse(stopped.is_set())
                repeat_thread.start()
                self.assertTrue(repeat_join.wait(5))
                self.assertFalse(stopped_again.is_set())
                self.assertEqual((turns, reports), ([], []))
            finally:
                allow_start.set()
                allow_cleanup.set()
                for thread in (ingest_thread, stop_thread, repeat_thread):
                    if thread.ident is not None:
                        thread.join(timeout=12)
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)
            self.assertTrue(stopped.is_set())
            self.assertTrue(stopped_again.is_set())
            self.assertFalse(any(t.is_alive() for t in (ingest_thread, stop_thread, repeat_thread)))
            self.assertEqual(errors, [])
            self.assertIs(published[0], worker)  # Publication BEFORE start is an invariant.
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].poll())
            self.assertFalse(worker.thread.is_alive())
            self.assertTrue(worker.done.is_set())
            self.assertEqual(worker.pcm, b"")
            self.assertIsNone(worker._initial)
            self.assertIsNone(worker.result)
            self.assertIsNone(worker.packet.window)
            self.assertEqual(
                (worker.packet.pcm, worker.packet.units, worker.packet.timeline), (b"", [], [])
            )
            self.assertIsNone(processor._recovery)
            self.assertEqual(processor.buffered_audio_bytes, 0)
            processor.close()
            processor.close()
            close_diarizer.assert_called_once()
            processor.release_pending()
            self.assertEqual((turns, reports), ([], []))

    def test_close_before_scheduling_never_constructs_or_starts_recovery(self):
        processor, turns, reports = self.processor()
        processor.close()
        with patch("rightyo.live_audio._WindowRecovery") as factory:
            processor._drain_results()
            with processor._recovery_lock:
                processor._start_recovery_locked()
            factory.assert_not_called()
        self.assertEqual(processor.buffered_audio_bytes, 0)
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
