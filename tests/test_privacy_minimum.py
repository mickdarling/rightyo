"""#291: authored turns and synthetic PCM; no capture, downloads or enrolled data."""

import contextlib
import io
import queue
import threading
import unittest
from array import array
from dataclasses import replace
from unittest.mock import Mock, patch

import test_duck_mode as duck
from test_live_audio import SILENCE, VOICE, FakeDiarizer
from test_live_speaker_id import Harness, ScriptedEmbedder, settings, tone, unit, wait_for
from test_whisper_window_recovery import BAD, GOOD, doc

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import LiveConfig, LiveProcessor, _recognizer_window
from rightyo.live_speaker_id import EnrolledRoles, EntryScore, LabelState
from rightyo.memory import TranscriptMemory
from rightyo.tool import _stderr
from rightyo.tool_events import SpeechEvents


class Units:
    recognizer_id = "authored-units"

    def __init__(self):
        self.audio = []

    def transcribe(self, pcm, register=None):
        self.audio.append(pcm)
        return [{"start_ms": 0, "end_ms": 200, "text": "Authored sentence."}]


class PrivacyMinimumTests(unittest.TestCase):
    controller = duck.DuckControllerTests.controller

    def arrange(self, *, vector=None):
        controller = self.controller()
        notes = []
        controller.report = notes.append
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        harness = Harness(
            ScriptedEmbedder([vector or unit(1, 0, 0)] * 8), config=settings(roles=True)
        )
        shadow = harness.shadow
        roles.source = shadow
        controller._shadow = shadow
        controller._enrolled_roles = roles
        events = SpeechEvents(report=notes.append)
        events.start(
            duck.SESSION,
            now_ms=0,
            addressing=duck.OURS,
            conversation=duck.MODE,
            priority=roles,
            duck=True,
        )
        events.drain()
        controller._events = events
        work, memory = queue.Queue(), TranscriptMemory()
        units = Units()
        config = LiveConfig(
            duck.SESSION,
            transcriber=units,
            diarizer=FakeDiarizer,
            private=controller._heard_private_span,
        )
        processor = LiveProcessor(
            config, lambda t: controller._turn(controller._generation, work, memory, shadow, t)
        )
        processor.on_speakers_renewed = controller._speakers_renewed
        controller._processor = processor
        self.addCleanup(processor.close)
        self.addCleanup(shadow.close)
        return controller, events, shadow, processor, notes, units

    def feed(self, controller, processor, pcm, shadow):
        for offset in range(0, len(pcm), 640):
            controller._feed(
                controller._generation, processor, pcm[offset : offset + 640], "demo", shadow
            )

    def test_empty_privacy_interval_preserves_and_publishes_public_prefix(self):
        c, e, s, p, notes, units = self.arrange()
        self.feed(c, p, VOICE * 10, s)
        public = bytes(p._utterance)
        c.duck(True)
        self.assertEqual(bytes(p._utterance), public)
        c.duck(False)
        e.drain()
        self.feed(c, p, SILENCE, s)
        self.assertEqual(bytes(p._utterance[: len(public)]), public)
        self.feed(c, p, SILENCE * 72, s)
        p.finish()
        self.assertEqual(bytes(units.audio[0][: len(public)]), public)
        self.assertEqual(
            [event["turn"]["text"] for event in e.drain() if event["type"] == "transcript"],
            ["Authored sentence."],
        )
        self.assertIn("forget=skipped", notes[-1])

    def test_any_nonzero_private_sample_requires_full_forget(self):
        for samples in ([100] * 320, [0] * 319 + [1]):
            with self.subTest(nonzero=sum(value != 0 for value in samples)):
                c, e, s, p, notes, _ = self.arrange()
                quiet = array("h", samples).tobytes()
                c.duck(True)
                with patch.object(p, "_diarizer_frame", side_effect=lambda pcm, private: pcm):
                    self.feed(c, p, quiet, s)
                self.assertEqual(p._diarizer.frames, [quiet])
                self.assertEqual(e._private_counts["discarded"], 0)
                c.duck(False)
                self.assertTrue(p._speaker_reset)
                self.assertGreater(s.epoch, 0)
                self.assertIn(
                    "diarizer_private_frames=1 forget=full reason=private_frames", notes[-1]
                )

    def test_stop_cancels_promptly_while_exit_races_a_stalled_native_push(self):
        c, e, s, p, notes, _ = self.arrange()
        c.duck(True)
        entered, release = threading.Event(), threading.Event()
        exit_started, stopped = threading.Event(), threading.Event()
        errors = []
        original_push = p._diarizer.push

        def slow_push(pcm):
            entered.set()
            if not release.wait(3):
                raise AssertionError("test did not release native push")
            original_push(pcm)

        def feed():
            try:
                self.feed(c, p, SILENCE, s)
            except Exception as error:
                errors.append(error)

        def exit_privacy():
            exit_started.set()
            c.duck(False)

        def stop():
            c.stop()
            stopped.set()

        with patch.object(p._diarizer, "push", side_effect=slow_push):
            workers = [threading.Thread(target=fn) for fn in (feed, exit_privacy, stop)]
            workers[0].start()
            try:
                self.assertTrue(entered.wait(1))
                workers[1].start()
                self.assertTrue(exit_started.wait(1))
                wait_for(lambda: not e.ducked)
                workers[2].start()
                self.assertTrue(stopped.wait(0.2), "Stop waited for native push")
                self.assertTrue(c._stop.is_set())
                self.assertTrue(c._decision_cancel.is_set())
            finally:
                release.set()
                for worker in workers:
                    if worker.ident is not None:
                        worker.join(3)
        self.assertEqual(errors, [])
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertIn("forget=full reason=privacy_unknown", notes[-1])

    def test_exit_discards_private_recovery_inputs_results_and_held_units(self):
        for queued_retry in (False, True):
            with self.subTest(queued_retry=queued_retry):
                c, e, s, p, notes, units = self.arrange()
                entered, release = threading.Event(), threading.Event()
                decoded, borrowed = [], []

                def decode(pcm, register=None):
                    decoded.append(True)
                    units._window = _recognizer_window(
                        doc(BAD if len(decoded) == 1 or queued_retry else GOOD), len(pcm) // 32
                    )
                    return units._window.units

                def retry(_config, pcm, _register):
                    borrowed.append(pcm)
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("test did not release recovery")
                    return doc(GOOD)

                units.transcribe = decode
                with patch("rightyo.live_audio._transcribe", side_effect=retry):
                    c.duck(True)
                    try:
                        self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                        self.assertTrue(entered.wait(1))
                        recovery = p._recovery
                        self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                        held = list(p._waiting)
                        self.assertEqual(len(held), 2)
                        self.assertTrue(any(recovery.pcm))
                        c.duck(False)
                        self.assertTrue(recovery.cancelled.is_set())
                        self.assertFalse(any(recovery.pcm))
                        self.assertFalse(any(borrowed[0]))
                        self.assertTrue(all(not w.pcm and not w.units for w in held))
                        self.assertIsNone(recovery.result)
                        self.assertIn("forget=full reason=privacy_unknown", notes[-1])
                    finally:
                        release.set()
                        if p._recovery is not None:
                            self.assertTrue(p._recovery.done.wait(3))
                            p._drain_results()
                    self.assertIsNone(recovery.result)
                    e.drain()
                    units.transcribe = lambda *_: [
                        {"start_ms": 200, "end_ms": 400, "text": "Authored public sentence."}
                    ]
                    self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                    p.finish()
                self.assertEqual(len(borrowed), 1)
                self.assertEqual(
                    [x["turn"]["text"] for x in e.drain() if x["type"] == "transcript"],
                    ["Authored public sentence."],
                )

    def test_exit_before_window_registration_refuses_late_private_decode(self):
        c, e, s, p, _, units = self.arrange()
        c.duck(True)
        e.drain()
        finalize = p._finalize

        def off_before_registration(timeline, complete=False):
            # All private frames are audited, but no finalized packet exists yet.
            c.duck(False)
            finalize(timeline, complete)

        with patch.object(p, "_finalize", side_effect=off_before_registration):
            self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
        self.assertEqual(units.audio, [])
        self.assertEqual(e.drain(), [])
        self.assertTrue(all(not w.pcm and not w.units for w in p._waiting))
        self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
        p.finish()
        self.assertEqual(len(units.audio), 1)
        self.assertEqual(
            [x["turn"]["text"] for x in e.drain() if x["type"] == "transcript"],
            ["Authored sentence."],
        )

    def test_exit_clears_completed_recovery_result_before_its_ordered_release(self):
        c, e, s, p, _, units = self.arrange()
        units.transcribe = lambda *_: []
        units._window = _recognizer_window(doc(BAD), 1640)
        c.duck(True)
        e.drain()
        release = threading.Event()

        def retry(*_):
            release.wait(3)
            return doc(GOOD)

        with patch("rightyo.live_audio._transcribe", side_effect=retry):
            try:
                self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                recovery = p._recovery
            finally:
                release.set()
            # No more audio: leave the completed result in the worker's result slot.
            self.assertTrue(recovery.done.wait(3))
            result = recovery.result
            self.assertTrue(bool(result.units))
            packet_pcm = recovery.packet.pcm
            c.duck(False)
            self.assertFalse(bool(result.units))
            self.assertIsNone(recovery.result)
            self.assertFalse(any(packet_pcm))
            self.assertTrue(all(not w.units and not w.pcm for w in p._waiting))
            p._drain_results()
        self.assertEqual(e.drain(), [])

    def test_cutoff_preserves_public_prefix_and_silences_only_private_samples(self):
        c, e, s, p, _, _ = self.arrange()
        self.feed(c, p, VOICE * 10, s)
        public = bytes(p._utterance)
        c.duck(True)
        self.assertEqual(bytes(p._utterance), public)
        self.feed(c, p, VOICE * 10, s)
        c.duck(False)
        self.feed(c, p, SILENCE, s)
        self.assertEqual(bytes(p._utterance[: len(public)]), public)
        self.assertFalse(any(p._utterance[len(public) :]))

    def test_entry_discards_recovery_window_spanning_the_on_boundary(self):
        c, e, s, p, _, units = self.arrange()
        entered, release = threading.Event(), threading.Event()
        units.transcribe = lambda *_: []
        units._window = _recognizer_window(doc(BAD), 1640)

        def retry(*_):
            entered.set()
            release.wait(3)
            return doc(GOOD)

        with patch("rightyo.live_audio._transcribe", side_effect=retry):
            try:
                self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                self.assertTrue(entered.wait(1))
                recovery = p._recovery
                # Exercise the processor boundary at a sample inside the held window.
                p.begin_privacy(100 * 32)
                self.assertTrue(recovery.cancelled.is_set())
                self.assertFalse(any(recovery.pcm))
                self.assertFalse(recovery.packet.units)
            finally:
                release.set()
                if p._recovery is not None:
                    self.assertTrue(p._recovery.done.wait(3))
                    p._drain_results()
        self.assertEqual(e.drain(), [])

    def test_processor_split_exit_authenticates_both_unlabelled_fragments(self):
        for scores, allowed in (((0.9, 0.9), True), ((0.1,), False), ((0.9, 0.1), False)):
            with self.subTest(scores=scores):
                c, e, s, p, _, units = self.arrange()
                s._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
                c._enrolled_roles.exit_audio = c._private_sentence
                s.exit_score = Mock(side_effect=[EntryScore(n, None, 0, "scored") for n in scores])
                phrases = iter(("Jarvis", "privacy off"))
                units.transcribe = lambda *_: [
                    {"start_ms": 0, "end_ms": 200, "text": next(phrases)}
                ]
                p.config = replace(
                    p.config,
                    on_private_turn_audio=lambda t, pcm: c._private_turn_audio(
                        c._generation, t, pcm
                    ),
                )
                turns = []
                on_turn = p.on_turn
                p.on_turn = lambda t: (turns.append(t), on_turn(t))
                c.duck(True)
                e.drain()
                self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                self.assertEqual(e._private_name_end is not None, scores[0] > 0.5)
                self.assertFalse(bool(c._private_sentence.pcm))
                self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                self.assertEqual("exit" in [x.get("action") for x in e.drain()], allowed)
                self.assertEqual(s.exit_score.call_count, len(scores))
                self.assertTrue(all(t.speaker_id is None for t in turns))
                if len(scores) == 2:
                    self.assertFalse(bool(c._private_sentence.pcm))
                self.assertEqual(s.offered, 0)

    def test_silent_privacy_keeps_every_public_label_state_and_next_owner_role(self):
        c, e, s, p, notes, _ = self.arrange()
        s._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
        s._bound_ids.add("owner")
        s.scored_seconds = 4
        e._roles["Speaker A"] = "owner"
        e._cue_voices.append((0, "Speaker A"))
        e._cooldowns["synthetic"] = (0, 100)
        e.last_request = (0, "Authored request.", "u0", "Speaker A")
        e._withdrawn["u0"] = {"speaker_id": "Speaker A"}
        e._old_labels.add("old")
        e._label_stale.add("stale")
        e._request_outcomes["r0"] = "synthetic"
        e._pending_ack = {"until_ms": 10000}
        c._runner = Mock(request_context=True)
        native, epoch = p._diarizer, s.epoch
        before = {
            key: dict(getattr(e, key))
            for key in ("_roles", "_cooldowns", "_withdrawn", "_request_outcomes")
        }
        c.duck(True)
        self.feed(c, p, SILENCE * 10, s)
        c.duck(False)
        self.assertIs(p._diarizer, native)
        self.assertFalse(p._speaker_reset)
        self.assertEqual(s.epoch, epoch)
        self.assertEqual(s.scored_seconds, 4)
        self.assertEqual(s.binding("Speaker A"), ("bound", "owner"))
        for key, value in before.items():
            self.assertEqual(getattr(e, key), value)
        self.assertEqual(e.last_request[3], "Speaker A")
        self.assertEqual(e._pending_ack, {"until_ms": 10000})
        self.assertEqual((e._old_labels, e._label_stale), ({"old"}, {"stale"}))
        self.assertEqual(tuple(e._cue_voices), ((0, "Speaker A"),))
        self.assertFalse(e._labels_renewed)
        c._runner.forget_speakers.assert_not_called()
        self.assertEqual(p._unlabelled_before_ms, 0)
        e.transcript(
            duck.heard("next", 201, 1100, "Authored public sentence."), 1100, expect_decision=False
        )
        self.assertEqual(e.drain()[-1]["turn"]["role"], "owner")
        self.assertIn("forget=skipped", notes[-1])
        self.assertNotIn("speakers forgotten", " ".join(notes))

    def test_private_speech_is_discarded_but_only_silence_reaches_diarizer_and_ring(self):
        c, e, s, p, notes, units = self.arrange()
        s._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
        native = p._diarizer
        c.duck(True)
        e.drain()
        self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
        self.assertGreaterEqual(e._private_counts["discarded"], 1)
        private = duck.heard("private-buffer", 0, 200, "Authored private sentence.")
        c._private_sentence.replace(private, VOICE * 10)
        retained = c._private_sentence.pcm
        # Also leave an open private utterance and pre-roll for the byte cutoff.
        self.feed(c, p, VOICE * 10, s)
        c.duck(False)
        self.feed(c, p, SILENCE, s)
        self.assertEqual(native.frames, [SILENCE] * len(native.frames))
        self.assertEqual(s._ring.span(0, c._received_ms)[0], bytes(c._received_bytes))
        self.assertFalse(any(p._utterance))
        self.assertFalse(any(retained))
        self.assertFalse(c._private_sentence.pcm)
        self.assertEqual(e.drain(), [])
        self.assertEqual(s.offered, 0)
        self.assertEqual(s.binding("Speaker A"), ("bound", "owner"))
        self.assertIs(p._diarizer, native)
        self.assertEqual(s.epoch, 0)
        self.assertIn("diarizer_private_frames=0 forget=skipped", notes[-1])
        self.assertTrue(units.audio)  # local private ASR remains the exit-phrase path

    def test_engagement_and_nameless_follow_up_survive_privacy(self):
        c, e, s, p, _, _ = self.arrange()
        s._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
        request = duck.heard("request", 0, 900, "Jarvis, check the authored fixture.")
        e.transcript(request, 900)
        e.decision(duck.decided(request, "attend"), 950)
        e.drain()
        engaged = dict(e._engaged)
        self.feed(c, p, SILENCE * 50, s)
        c.duck(True)
        e.drain()  # the unchanged content-free privacy-on acknowledgement
        c.duck(False)
        self.assertEqual(e._engaged, engaged)
        self.assertEqual(e.drain(), [])
        follow = duck.heard("follow", 1100, 1800, "And the second fixture?")
        e.transcript(follow, 1800)
        e.decision(duck.decided(follow, "uncertain", attend=0.8), 1900)
        self.assertIn("request", [event["type"] for event in e.drain()])

    def test_unfinished_private_voice_uses_actual_frame_count_even_with_no_discarded_turn(self):
        c, e, s, p, notes, _ = self.arrange()
        c.duck(True)
        with patch.object(p, "_diarizer_frame", side_effect=lambda frame, private: frame):
            self.feed(c, p, VOICE, s)
        self.assertEqual(e._private_counts["discarded"], 0)
        c.duck(False)
        self.assertTrue(p._speaker_reset)
        self.assertGreater(s.epoch, 0)
        self.assertTrue(e._labels_renewed)
        self.assertIn(
            "discarded=0 diarizer_private_frames=1 forget=full reason=private_frames", notes[-1]
        )

    def test_unknown_or_raising_predicate_fails_closed_even_with_substitution(self):
        for predicate in (lambda *_: None, Mock(side_effect=RuntimeError("authored secret"))):
            with self.subTest(predicate=predicate):
                c, e, s, p, notes, _ = self.arrange()
                c.duck(True)
                p.config = replace(p.config, private=predicate)
                self.feed(c, p, VOICE, s)
                self.assertEqual(p._diarizer.frames, [SILENCE])
                c.duck(False)
                self.assertTrue(p._speaker_reset)
                self.assertIn("forget=full reason=privacy_unknown", notes[-1])
                self.assertNotIn("authored secret", " ".join(notes))

    def test_fallback_binding_turn_publishes_its_own_above_bar_role(self):
        for vector, seconds, expected in (
            (unit(1, 0, 0), 1.2, "owner"),
            (unit(0.5, 0, 0.866), 1.2, "unknown"),
            (unit(0, 0, 1), 4.0, "unknown"),
        ):
            with self.subTest(expected=expected):
                c, e, s, p, _, _ = self.arrange(vector=vector)
                s.start()
                wait_for(lambda: bool(s._compared))
                c.duck(True)
                with patch.object(p, "_diarizer_frame", side_effect=lambda frame, private: frame):
                    self.feed(c, p, VOICE, s)
                c.duck(False)
                # Apply the deferred native renewal before the first new labelled turn.
                p._renew_diarizer()
                end = 1000 + round(seconds * 1000)
                current = duck.heard("rebound", 1000, end, "Authored public sentence.")
                s.audio(bytes(1000 * 32 - len(VOICE)) + tone(seconds))
                c._observe_spans(c._generation, "rebound", ((1000, end),))
                c._turn(c._generation, queue.Queue(), TranscriptMemory(), s, current)
                transcript = [event for event in e.drain() if event["type"] == "transcript"]
                self.assertEqual(transcript[-1]["turn"]["role"], expected)
                self.assertEqual(s.scored, 1)
                if expected == "owner":
                    self.assertLess(s.scored_seconds, s.settings.bind_min_seconds)

    def test_partial_private_tail_also_substitutes_and_audits(self):
        for substitute in (True, False):
            with self.subTest(substitute=substitute):
                c, e, s, p, notes, _ = self.arrange()
                c.duck(True)
                tail = VOICE[:100]
                native = p._diarizer
                self.feed(c, p, tail, s)
                if not substitute:
                    p._diarizer_frame = lambda frame, private: frame
                p.finish()
                self.assertEqual(native.frames, [bytes(len(tail)) if substitute else tail])
                c.duck(False)
                self.assertIn("diarizer_private_frames=" + str(int(not substitute)), notes[-1])
                self.assertEqual(s.epoch > 0, not substitute)

    def test_exit_with_an_unaudited_frame_in_flight_renews_fail_closed(self):
        c, e, s, p, notes, _ = self.arrange()
        c.duck(True)
        arrived, release = threading.Event(), threading.Event()
        errors = []

        def pending(frame, private):
            arrived.set()
            if not release.wait(3):
                raise AssertionError("test did not release authored frame")
            return frame  # deliberately bypass substitution

        def feed():
            try:
                self.feed(c, p, VOICE, s)
            except Exception as error:
                errors.append(error)

        with patch.object(p, "_diarizer_frame", side_effect=pending):
            worker = threading.Thread(target=feed, daemon=True)
            worker.start()
            try:
                self.assertTrue(arrived.wait(3))
                c.duck(False)
                self.assertIn("forget=full reason=privacy_unknown", notes[-1])
                self.assertGreater(s.epoch, 0)
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(p._speaker_reset)

    def test_exit_before_frame_registration_renews_for_accepted_unaudited_audio(self):
        c, e, s, p, notes, _ = self.arrange()
        c.duck(True)
        arrived, release = threading.Event(), threading.Event()
        push = p.push_pcm16

        def deferred(pcm):
            arrived.set()
            if not release.wait(3):
                raise AssertionError("test did not release accepted audio")
            push(pcm)

        with patch.object(p, "push_pcm16", side_effect=deferred):
            worker = threading.Thread(target=lambda: self.feed(c, p, VOICE, s), daemon=True)
            worker.start()
            try:
                self.assertTrue(arrived.wait(3))
                self.assertEqual(p._diarizer_pending, 0)
                c.duck(False)
                self.assertIn("forget=full reason=privacy_unknown", notes[-1])
                self.assertGreater(s.epoch, 0)
            finally:
                release.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertTrue(p._speaker_reset)

    def test_private_silence_timeline_cannot_authenticate_an_owner_exit(self):
        for score, allowed in ((0.1, False), (0.9, True)):
            with self.subTest(score=score):
                c, e, s, p, _, units = self.arrange()
                s._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
                c._enrolled_roles.exit_audio = c._private_sentence
                s.exit_score = Mock(return_value=EntryScore(score, None, 0, "scored"))
                units.transcribe = lambda *_: [
                    {"start_ms": 0, "end_ms": 200, "text": "Jarvis, privacy off"}
                ]
                p.config = replace(
                    p.config,
                    on_private_turn_audio=lambda t, pcm: c._private_turn_audio(
                        c._generation, t, pcm
                    ),
                )
                c.duck(True)
                e.drain()
                self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                actions = [event.get("action") for event in e.drain()]
                self.assertEqual("exit" in actions, allowed)
                s.exit_score.assert_called_once()
                self.assertEqual(s.offered, 0)
                self.assertFalse(c._private_sentence.pcm)

    def test_slow_fallback_score_finishes_before_its_role_is_published(self):
        c, e, s, p, _, _ = self.arrange()
        gate = threading.Event()
        self.addCleanup(gate.set)
        s._embedder_factory = lambda _: ScriptedEmbedder([unit(1, 0, 0)], gate=gate)
        s.start()
        wait_for(lambda: bool(s._compared))
        s.forget()
        current = duck.heard("slow", 0, 1200, "Authored public sentence.")
        s.audio(tone(1.2))
        c._observe_spans(c._generation, "slow", ((0, 1200),))
        worker = threading.Thread(
            target=lambda: c._turn(c._generation, queue.Queue(), TranscriptMemory(), s, current),
            daemon=True,
        )
        with patch("rightyo.live_speaker_id.ENTRY_CHECK_TIMEOUT_S", 0.001):
            worker.start()
            try:
                wait_for(lambda: s._embedder.calls == 1)
                self.assertTrue(worker.is_alive())
                self.assertEqual(e.drain(), [])
            finally:
                gate.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        transcript = [event for event in e.drain() if event["type"] == "transcript"]
        self.assertEqual(transcript[-1]["turn"]["role"], "owner")

    def test_scoring_does_not_block_privacy_stop_or_epoch_changes(self):
        for action in ("private", "stop", "renew"):
            with self.subTest(action=action):
                c, e, s, p, _, _ = self.arrange()
                gate = threading.Event()
                s._embedder_factory = lambda _: ScriptedEmbedder([unit(1, 0, 0)], gate=gate)
                s.start()
                wait_for(lambda: bool(s._compared))
                s.forget()
                current = duck.heard("waiting", 0, 1200, "Authored public sentence.")
                s.audio(tone(1.2))
                c._observe_spans(c._generation, "waiting", ((0, 1200),))
                worker = threading.Thread(
                    target=lambda: c._turn(
                        c._generation, queue.Queue(), TranscriptMemory(), s, current
                    ),
                    daemon=True,
                )
                applied = threading.Event()

                def control():
                    if action == "private":
                        c.duck(True)
                    elif action == "stop":
                        c.stop()
                    else:
                        c._speakers_renewed()
                    applied.set()

                worker.start()
                try:
                    wait_for(lambda: s._embedder.calls == 1)
                    controls = threading.Thread(target=control, daemon=True)
                    controls.start()
                    self.assertTrue(applied.wait(1))
                finally:
                    gate.set()
                    worker.join(3)
                    controls.join(3)
                self.assertFalse(worker.is_alive())
                transcripts = [event for event in e.drain() if event["type"] == "transcript"]
                if action == "renew":
                    self.assertIsNone(transcripts[-1]["turn"]["speaker_id"])
                    self.assertEqual(transcripts[-1]["turn"]["role"], "unknown")
                else:
                    self.assertEqual(transcripts, [])

    def test_worker_stop_after_dequeue_also_releases_the_scoring_waiter(self):
        c, e, s, p, _, _ = self.arrange()
        s.forget()
        current = duck.heard("stopped-score", 0, 1200, "Authored public sentence.")
        s.audio(tone(1.2))
        get = s._queue.get
        popped = []

        def stop_after_get(*args, **kwargs):
            item = get(*args, **kwargs)
            popped.append(item)
            s._stop.set()
            return item

        with patch.object(s._queue, "get", side_effect=stop_after_get):
            s.start()
            wait_for(lambda: bool(s._compared))
            waiter = threading.Thread(
                target=lambda: s.turn(current, spans=((0, 1200),), epoch=s.epoch, wait=True),
                daemon=True,
            )
            waiter.start()
            try:
                waiter.join(1)
                self.assertFalse(waiter.is_alive())
                self.assertEqual(s.scored, 0)
            finally:
                for item in popped:
                    item[-1].set()
                waiter.join(3)

    def test_fallback_enrollment_takes_precedence_over_a_recycled_configured_label(self):
        c, e, s, p, _, _ = self.arrange()
        c._enrolled_roles.priority = SpeakerPriority(owners=("owner", "Speaker A"))
        s.start()
        wait_for(lambda: bool(s._compared))
        e.forget_speakers(now_ms=0)
        s.forget()
        current = duck.heard("enrolled", 0, 1200, "Authored public sentence.")
        s.audio(tone(1.2))
        c._observe_spans(c._generation, "enrolled", ((0, 1200),))
        c._turn(c._generation, queue.Queue(), TranscriptMemory(), s, current)
        transcript = [event for event in e.drain() if event["type"] == "transcript"][-1]
        self.assertEqual(transcript["turn"]["role"], "owner")
        self.assertEqual(transcript["role_source"], "voiceprint")

    def test_worker_teardown_before_admission_cannot_strand_a_waiter(self):
        c, e, s, p, _, _ = self.arrange()
        s.start()
        wait_for(lambda: bool(s._compared))
        s.forget()
        s.audio(tone(1.2))
        current = duck.heard("late-score", 0, 1200, "Authored public sentence.")
        arrived, release = threading.Event(), threading.Event()
        put = s._queue.put_nowait

        def late_put(item):
            arrived.set()
            if not release.wait(3):
                raise AssertionError("test did not release queue admission")
            put(item)

        with patch.object(s._queue, "put_nowait", side_effect=late_put):
            waiter = threading.Thread(
                target=lambda: s.turn(current, spans=((0, 1200),), epoch=s.epoch, wait=True),
                daemon=True,
            )
            waiter.start()
            try:
                self.assertTrue(arrived.wait(1))
                s._closing.set()
                wait_for(lambda: s._off.is_set() and not s._thread.is_alive())
                release.set()
                waiter.join(1)
                self.assertFalse(waiter.is_alive())
                self.assertTrue(s._queue.empty())
                self.assertIsNone(s.binding("Speaker A"))
            finally:
                release.set()
                s._discard_queue()
                waiter.join(3)

    def test_invalid_or_failed_exit_audit_renews_without_reporting_content(self):
        for result in ((0, []), (False, None), (-1, None), RuntimeError("Authored secret")):
            with self.subTest(result=type(result).__name__):
                c, e, s, p, notes, _ = self.arrange()
                c.duck(True)
                self.feed(c, p, VOICE, s)
                audit = (
                    Mock(side_effect=result)
                    if isinstance(result, Exception)
                    else Mock(return_value=result)
                )
                with patch.object(p, "end_privacy", audit):
                    c.duck(False)
                self.assertGreater(s.epoch, 0)
                self.assertIn("forget=full reason=audit_unavailable", notes[-1])
                self.assertNotIn("Authored secret", " ".join(notes))

    def test_exit_diagnostics_use_only_fixed_tokens_and_counts_on_stderr(self):
        c, e, s, p, _, _ = self.arrange()
        c.report = e._report = _stderr
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            c.duck(True)
            self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
            c.duck(False)
        self.assertRegex(
            output.getvalue().splitlines()[-1],
            r"^rightyo: privacy off: discarded=1 diarizer_private_frames=0 forget=skipped$",
        )
        self.assertNotIn("Authored", output.getvalue())
        self.assertNotIn("Speaker", output.getvalue())
        self.assertNotIn("owner", output.getvalue().splitlines()[-1])
