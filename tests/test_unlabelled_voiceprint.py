"""#334. Synthetic local PCM and controlled embeddings, never an enrollment store."""

import copy
import io
import select
import socket
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from test_conversation import decided
from test_live_speaker_id import Harness, ScriptedEmbedder, settings, tone, turn, wait_for

from rightyo.capture import PCM_CHUNK_BYTES, StdinPcmCapture
from rightyo.contracts import SpeakerPriority
from rightyo.live_speaker_id import EnrolledRoles, TurnVoiceprint
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.speaker_id import SpeakerEmbedder, SpeakerIdError, _diagnostic_requests
from rightyo.tool_events import SpeechEvents


class DiagnosticEmbedder(ScriptedEmbedder):
    def __init__(self, **options):
        super().__init__(**options)
        self.windows = []
        self.vectors = []

    def embed_diagnostic(self, pcm, cancel_fd):
        if select.select([cancel_fd], [], [], 0)[0]:
            raise ValueError("cancelled")
        self.windows.append(pcm)
        vector = [1.0, 2.0, 3.0]
        self.vectors.append(vector)
        return vector


class UnlabelledVoiceprintTests(unittest.TestCase):
    def setUp(self):
        self.model = DiagnosticEmbedder()
        self.model.embed_diagnostic = Mock(wraps=self.model.embed_diagnostic)
        self.shadow = Harness(self.model, config=settings(roles=True)).shadow
        self.shadow.start()
        self.addCleanup(self.shadow.close)
        wait_for(lambda: bool(self.shadow._compared) and self.shadow._diagnostic_idle)
        self.events = SpeechEvents()
        self.roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        self.roles.source = self.shadow
        self.events.start("shadow-test", priority=self.roles, duck=True)
        self.offer = Mock(wraps=self.shadow.try_unlabelled_score)
        self.events.unlabelled_scorer = self.offer
        self.events.drain()

    def submit(self, current):
        self.events.transcript(current, current.end_ms)
        return self.events._pending_voiceprints.get(current.utterance_id)

    def decision(self, current):
        self.events.decision(decided(current, "attend"), current.end_ms)
        stream = self.events.drain()
        evidence = next(e["decision"] for e in stream if e["type"] == "attention")
        for event in stream:
            if event["type"] == "request":
                self.assertEqual(event["decision"], evidence)
                self.assertNotIn("voiceprint_owner_score", event["turn"])
                self.assertNotIn("voiceprint_owner_score", event["context"])
        return evidence

    def test_public_unlabelled_score_and_wipes(self):
        self.shadow.audio(bytes(3200) + tone(4) + bytes(3200))
        current = turn(0, 4200, None)
        raw = self.shadow._ring.diagnostic_span(0, 4200)
        with patch.object(self.shadow._ring, "diagnostic_span", return_value=raw):
            self.submit(current)
            wait_for(lambda: self.shadow._diagnostic is None)
        evidence = self.decision(current)
        self.assertAlmostEqual(evidence["voiceprint_owner_score"], 1 / 14**0.5, places=15)
        self.assertEqual(evidence["voiceprint_speech_ms"], 4000)
        self.assertNotIn("voiceprint_unavailable", evidence)
        self.assertEqual(evidence["role"], "unknown")
        self.assertFalse(any(raw))
        self.assertEqual(len(self.model.windows), 2)
        self.assertTrue(all(not any(buf) for buf in self.model.windows + self.model.vectors))

    def capture_controller(self, capture):
        config = PrototypeConfig(None, None, None, None, None)
        controller = PrototypeController(config, event_publisher=self.events)
        controller._capture = capture
        controller._attach_capture_mask(capture)
        return controller

    def test_private_turn_and_private_capture_backlog_never_ask_worker(self):
        self.events.duck(True, now_ms=0)
        current = turn(0, 200, None)
        self.submit(current)
        self.events.decision(decided(current, "attend"), 200)
        self.assertEqual(self.events.drain(), [])

        # The reader queues two real synthetic chunks while the processing clock
        # is still at zero. Its second chunk is finalized after the empty on/off span.
        capture = StdinPcmCapture(io.BytesIO(tone(0.4)))
        controller = self.capture_controller(capture)
        capture.start()
        self.addCleanup(capture.stop)
        wait_for(capture._eof.is_set)
        self.events.duck(False, now_ms=0)
        for _ in range(2):
            pcm = capture.read()
            self.assertEqual(len(pcm), PCM_CHUNK_BYTES)
            controller._feed(controller._generation, Mock(), pcm, "stdin", self.shadow)
        queued = turn(200, 400, None)
        self.submit(queued)
        evidence = self.decision(queued)
        self.assertEqual(evidence["voiceprint_unavailable"], "private")
        self.assertIsNone(evidence["voiceprint_owner_score"])
        self.assertIsNone(evidence["voiceprint_speech_ms"])
        self.offer.assert_not_called()
        self.assertEqual(self.model.windows, [])

    def test_partial_capture_read_crossing_privacy_is_excluded(self):
        capture = StdinPcmCapture(io.BytesIO())
        controller = self.capture_controller(capture)
        reads = iter([b"x" * 300, b"y" * (PCM_CHUNK_BYTES - 300), b""])

        def read(_size):
            data = next(reads)
            if data.startswith(b"y"):
                self.events.duck(True, now_ms=0)
                self.events.duck(False, now_ms=0)
            return data

        capture._receive(read)
        controller._feed(controller._generation, Mock(), capture.read(), "stdin", self.shadow)
        current = turn(0, 200, None)
        self.submit(current)
        self.assertEqual(self.decision(current)["voiceprint_unavailable"], "private")
        self.offer.assert_not_called()

    def test_busy_labelled_worker_skips_without_queue_or_pcm_copy(self):
        gate = self.model.gate = threading.Event()
        self.addCleanup(gate.set)
        self.shadow.audio(tone(2))
        self.shadow.turn(turn(0, 1000), spans=((0, 1000),))
        wait_for(lambda: self.model.calls == 1)
        current = turn(1000, 2000, None)
        with patch.object(self.shadow._ring, "diagnostic_span") as span:
            self.submit(current)
            self.assertEqual(self.decision(current)["voiceprint_unavailable"], "busy")
            self.assertTrue(self.shadow._queue.empty())
            self.assertTrue(self.shadow._checks.empty())
            self.assertIsNone(self.shadow._diagnostic)
            span.assert_not_called()
        gate.set()
        wait_for(lambda: self.shadow.scored == 1)
        self.assertEqual(self.model.windows, [])

    def test_later_label_and_decision_do_not_wait_for_diagnostic(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def block(pcm, fd):
            entered.set()
            self.assertTrue(select.select([fd], [], [], 2)[0])
            release.wait(2)
            return [1, 2, 3]

        self.model.embed_diagnostic.side_effect = block
        self.shadow.audio(tone(5))
        current = turn(0, 4000, None)
        self.submit(current)
        self.assertTrue(entered.wait(1))
        self.shadow.turn(turn(4000, 5000), spans=((4000, 5000),))
        wait_for(lambda: self.shadow.scored == 1, timeout=1)
        self.assertEqual(self.decision(current)["voiceprint_unavailable"], "busy")
        self.assertIsNotNone(self.shadow._diagnostic)
        release.set()

    def test_fast_decision_cancels_score_and_concurrent_turn_cannot_queue(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def block(pcm, fd):
            entered.set()
            release.wait(3)
            return [1, 0, 0]

        self.model.embed_diagnostic.side_effect = block
        self.shadow.audio(tone(2))
        first, second = turn(0, 1000, None), turn(1000, 2000, None)
        self.submit(first)
        self.assertTrue(entered.wait(1))
        self.submit(second)
        for current in (first, second):
            self.assertEqual(self.decision(current)["voiceprint_unavailable"], "busy")
        self.assertTrue(self.shadow._queue.empty() and self.shadow._checks.empty())
        release.set()
        wait_for(lambda: self.shadow._diagnostic is None)
        self.model.embed_diagnostic.assert_called_once()

    def test_diagnostic_failure_wipes_window_and_leaves_worker_available(self):
        held = []

        def fail(pcm, fd):
            held.append(pcm)
            raise ValueError("synthetic diagnostic failure")

        self.model.embed_diagnostic.side_effect = fail
        self.shadow.audio(tone(2))
        current = turn(0, 1000, None)
        self.submit(current)
        wait_for(lambda: self.shadow._diagnostic is None)
        self.assertEqual(self.decision(current)["voiceprint_unavailable"], "no_turn_embedding")
        self.assertEqual(len(held), 1)
        self.assertFalse(any(held[0]))
        self.shadow.turn(turn(1000, 2000), spans=[(1000, 2000)], wait=True)
        self.assertTrue(self.shadow.active)
        self.assertEqual(self.shadow.scored, 1)

    def test_privacy_entry_revokes_an_inflight_public_turn(self):
        entered, revoked = threading.Event(), threading.Event()

        def embed(pcm, fd):
            entered.set()
            if select.select([fd], [], [], 2)[0]:
                revoked.set()
            return [1, 0, 0]

        self.model.embed_diagnostic.side_effect = embed
        self.shadow.audio(tone(6))
        current = turn(0, 6000, None)
        self.submit(current)
        self.assertTrue(entered.wait(1))
        self.events.duck(True, now_ms=6100)
        self.assertTrue(revoked.wait(1))
        self.events.duck(False, now_ms=6300)
        self.assertEqual(self.decision(current)["voiceprint_unavailable"], "private")
        wait_for(lambda: self.shadow._diagnostic is None)
        self.assertEqual(self.model.embed_diagnostic.call_count, 1)

    def test_unlabelled_scores_preserve_binding_receipts_roles_and_counts(self):
        self.shadow.audio(tone(6))
        self.shadow.turn(turn(0, 3000), spans=((0, 3000),), wait=True)
        self.assertEqual(self.shadow.binding("Speaker A"), ("bound", "owner"))
        self.assertTrue(self.shadow._first_binds)

        def snapshot():
            return (
                copy.deepcopy(self.shadow._labels),
                dict(self.shadow._first_binds),
                dict(self.shadow._binding_revisions),
                self.shadow._self_generation,
                self.shadow.epoch,
                self.shadow.owner_evidence(("owner",)),
                self.shadow.offered,
                self.shadow.scored,
                self.shadow.labelled_ms,
                self.roles._resolved,
                self.roles._spans,
            )

        before = snapshot()
        for start in (3000, 4000, 5000):
            ticket = self.shadow.try_unlabelled_score(start, start + 1000, ("owner",))
            wait_for(lambda: self.shadow._diagnostic is None)
            self.assertIsInstance(ticket.take()[0], float)
        self.assertEqual(snapshot(), before)


class DiagnosticChannelTests(unittest.TestCase):
    def test_local_worker_protocol_and_native_cancellation(self):
        # Exercise both ends of the actual inherited-fd protocol without ONNX weights.
        parent, child = socket.socketpair(socket.AF_UNIX)
        client = SpeakerEmbedder.__new__(SpeakerEmbedder)
        client._diagnostic_socket = parent
        parent.settimeout(3)
        client._diagnostic_reader = parent.makefile("rb")
        entered = threading.Event()
        np = MagicMock()
        np.asarray.return_value.reshape.return_value.tolist.side_effect = lambda: [0.25, 0.75]
        outputs = MagicMock()

        session = SimpleNamespace(run=Mock(return_value=[outputs]))
        ort = SimpleNamespace(RunOptions=lambda: SimpleNamespace(terminate=False))
        worker = threading.Thread(
            target=_diagnostic_requests, args=(child.detach(), np, session, "x", ort)
        )
        with patch("rightyo.speaker_id._kaldi_fbank", return_value=MagicMock()):
            worker.start()
            try:
                ticket = TurnVoiceprint()
                with ticket.cancellation_fd() as fd:
                    pcm = memoryview(bytearray(320))
                    self.assertEqual(client.embed_diagnostic(pcm, fd), [0.25, 0.75])
                    session.run.assert_called_once()

                    def block(_outputs, _inputs, options):
                        entered.set()
                        wait_for(lambda: options.terminate)
                        raise ValueError("interrupted")

                    session.run = block

                    def cancel():
                        if entered.wait(2):
                            ticket.cancel("private")

                    revoke = threading.Thread(target=cancel)
                    revoke.start()
                    try:
                        with self.assertRaises(SpeakerIdError):
                            client.embed_diagnostic(pcm, fd)
                    finally:
                        revoke.join(3)
                    self.assertTrue(entered.is_set())
                self.assertEqual(outputs.fill.call_args.args, (0,))
            finally:
                client._close_diagnostic()
                worker.join(3)
            self.assertFalse(worker.is_alive())
