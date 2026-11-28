"""#291: authored turns and synthetic PCM; no capture, downloads or enrolled data."""

import builtins
import contextlib
import io
import math
import queue
import threading
import unittest
from array import array
from dataclasses import replace
from unittest.mock import Mock, patch

import test_duck_mode as duck
from test_live_audio import SILENCE, VOICE, FakeDiarizer
from test_live_speaker_id import Harness, ScriptedEmbedder, settings, tone, unit, wait_for
from test_whisper_window_recovery import BAD, GOOD, doc, segment

from rightyo.contracts import SpeakerPriority
from rightyo.live_audio import LiveConfig, LiveProcessor, _FinalizedWindow, _recognizer_window
from rightyo.live_speaker_id import EnrolledRoles, EntryScore, LabelState
from rightyo.memory import TranscriptMemory
from rightyo.tool import _stderr
from rightyo.tool_events import SpeechEvents


class PrivateReplyHoldTests(unittest.TestCase):
    """Authored playback controls and turns; no audio or host process."""

    def engaged(self, scoped=False):
        events = SpeechEvents(request_control=scoped)
        events.start(
            duck.SESSION, now_ms=0, addressing=duck.OURS, conversation=duck.MODE, duck=True
        )
        events.drain()
        request = duck.heard("request", 0, 900, "Jarvis, check the authored fixture.")
        events.transcript(request, 900)
        events.decision(duck.decided(request, "attend"), 950)
        events.drain()
        request_id = events._engaged["request_id"] if scoped else None
        if scoped:
            events.request_outcome(
                {
                    "version": 1,
                    "session_id": duck.SESSION,
                    "request_id": request_id,
                    "state": "accepted",
                },
                960,
            )
            events.drain()
        return events, request_id

    def follow(self, events, at):
        turn = duck.heard(f"follow-{at}", at, at + 900, "And the second fixture?")
        events.transcript(turn, turn.end_ms)
        events.decision(duck.decided(turn, "uncertain", attend=0.8), turn.end_ms + 50)
        return events.drain()

    def test_private_reply_end_does_not_keep_a_late_follow_up(self):
        for scoped in (False, True):
            with self.subTest(scoped=scoped):
                events, request_id = self.engaged(scoped)
                events.reply("started", 1000, request_id=request_id)
                events.duck(True, now_ms=1500)
                events.reply("ended", 2000, request_id=request_id)
                self.assertEqual(events.drain(), [])
                events.duck(False, now_ms=3000)
                self.assertEqual(events.drain(), [])
                self.assertNotIn("request", [e["type"] for e in self.follow(events, 30000)])

    def test_private_reply_end_keeps_the_public_follow_up_window(self):
        for scoped in (False, True):
            for exit_ms in (3000, 20000):
                with self.subTest(scoped=scoped, exit_ms=exit_ms):
                    public, public_id = self.engaged(scoped)
                    private, private_id = self.engaged(scoped)
                    public.reply("started", 1000, request_id=public_id)
                    private.reply("started", 1000, request_id=private_id)
                    private.duck(True, now_ms=1500)
                    public.reply("ended", 2000, request_id=public_id)
                    private.reply("ended", 2000, request_id=private_id)
                    self.assertEqual(private.drain(), [])
                    private.duck(False, now_ms=exit_ms)
                    self.assertEqual(private.drain(), [])
                    self.assertEqual(private._engaged, public._engaged)
                    self.assertNotIn("replying_since", private._engaged)
                    self.assertEqual(private._engaged["until_ms"], 12000)
                    # The end, not privacy exit, starts the ordinary ten-second window.
                    at = 11000 if exit_ms == 3000 else 21000
                    expected = self.follow(public, at)
                    self.assertEqual(self.follow(private, at), expected)
                    self.assertEqual(any(e["type"] == "request" for e in expected), at == 11000)

    def test_private_reply_start_does_not_leave_a_hold_at_exit(self):
        for scoped in (False, True):
            for end_ms in (2000, 4000):
                with self.subTest(scoped=scoped, end_ms=end_ms):
                    events, request_id = self.engaged(scoped)
                    before = dict(events._engaged)
                    events.duck(True, now_ms=1000)
                    events.reply("started", 1500, request_id=request_id)
                    if end_ms < 3000:
                        events.reply("ended", end_ms, request_id=request_id)
                    events.duck(False, now_ms=3000)
                    self.assertEqual(events.drain(), [])
                    self.assertEqual(events._engaged, before)
                    if end_ms > 3000:
                        events.reply("ended", end_ms, request_id=request_id)
                    self.assertNotIn("replying_since", events._engaged)
                    self.assertNotIn("request", [e["type"] for e in self.follow(events, 30000)])

    def test_public_reply_still_playing_at_exit_keeps_its_hold_until_its_end(self):
        for scoped in (False, True):
            with self.subTest(scoped=scoped):
                events, request_id = self.engaged(scoped)
                events.reply("started", 1000, request_id=request_id)
                before = dict(events._engaged)
                events.duck(True, now_ms=1500)
                events.duck(False, now_ms=3000)
                self.assertEqual(events.drain(), [])
                self.assertEqual(events._engaged, before)
                events.expire(30000)
                self.assertEqual(events._engaged["replying_since"], 1000)
                events.reply("ended", 31000, request_id=request_id)
                self.assertNotIn("replying_since", events._engaged)
                self.assertEqual(events._engaged["until_ms"], 41000)
                self.assertIn("request", [e["type"] for e in self.follow(events, 40000)])

    def test_unscoped_private_end_cannot_release_a_scoped_reply_hold(self):
        for unrelated in (None, duck.SESSION + ":other"):
            with self.subTest(request_id=unrelated):
                events, request_id = self.engaged(scoped=True)
                events.reply("started", 1000, request_id=request_id)
                before = dict(events._engaged)
                events.duck(True, now_ms=1500)
                events.reply("ended", 2000, request_id=unrelated)
                events.duck(False, now_ms=3000)
                self.assertEqual(events.drain(), [])
                self.assertEqual(events._engaged, before)
                events.reply("ended", 4000, request_id=request_id)
                self.assertNotIn("replying_since", events._engaged)


class Units:
    recognizer_id = "authored-units"

    def __init__(self):
        self.audio = []

    def transcribe(self, pcm, register=None):
        self.audio.append(pcm)
        return [{"start_ms": 0, "end_ms": 200, "text": "Authored sentence."}]


class FixtureVad:
    """A stand-in worker. Match authored quiet blocks by identity, including subframe edges."""

    chunk_ms = 1
    available = True

    def __init__(self, *quiet_frames):
        self.quiet = {
            frame[i : i + 32]
            for frame in (SILENCE, *quiet_frames)
            for i in range(0, len(frame), 32)
        }

    def is_speech(self, pcm):
        return pcm not in self.quiet

    def close(self):
        pass


class PrivacyMinimumTests(unittest.TestCase):
    controller = duck.DuckControllerTests.controller

    def arrange(self, *, vector=None, end_of_turn=None, pre_roll_ms=240, vad=FixtureVad()):
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
            end_of_turn=end_of_turn,
            pre_roll_ms=pre_roll_ms,
            private=controller._heard_private_span,
            silero_vad=vad,
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

    def test_privacy_off_after_publisher_cuts_drops_units_without_closing_session(self):
        c, e, s, p, _, _ = self.arrange()
        p.config = replace(p.config, edge_attribution_ms=500)
        paused, exited = threading.Event(), threading.Event()
        packets, errors = [], []

        def exit_privacy():
            try:
                if not paused.wait(3):
                    raise AssertionError("publisher did not reach its cuts")
                c.duck(False)
            except Exception as error:
                errors.append(error)
            finally:
                exited.set()

        def unit_cuts(left, right):
            # _publish_window has computed both boundaries but not read the slice.
            packet = p._publishing
            packets.append((packet, packet.pcm, packet.units))
            paused.set()
            self.assertTrue(exited.wait(3), "privacy-off waited for the publisher")
            return builtins.zip(left, right)

        c.duck(True)
        e.drain()
        control = threading.Thread(target=exit_privacy)
        control.start()
        try:
            with patch("rightyo.live_audio.zip", side_effect=unit_cuts, create=True):
                self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
        finally:
            paused.set()
            control.join(3)
        self.assertFalse(control.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(packets), 1)
        packet, pcm, units = packets[0]
        self.assertTrue(packet.discarded)
        self.assertFalse(any(pcm))
        self.assertEqual(units, [])
        self.assertEqual(packet.units, [])
        self.assertEqual(packet.pcm, bytearray())
        self.assertFalse(p.failed)
        self.assertFalse(p.closed)
        self.assertFalse(p._merger.holding)
        self.assertFalse(c._stop.is_set())
        self.assertEqual(e.drain(), [])
        self.assertFalse(c._private_sentence.pcm)
        self.assertEqual(s.offered, 0)

    def test_edge_attribution_accepts_an_empty_unit_slice(self):
        _, e, _, p, _, _ = self.arrange()
        p.config = replace(p.config, edge_attribution_ms=500)
        p._publish_units(
            [],
            [],
            0,
            b"",
            "diarization-timeline",
            number=1,
            timeline_present=True,
            complete=True,
            packet=Mock(discarded=False),
        )
        self.assertFalse(p.failed)
        self.assertFalse(p._merger.holding)
        self.assertEqual(e.drain(), [])

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

    def test_privacy_off_between_overflow_inspection_and_append_keeps_session_public(self):
        c, e, s, p, _, units = self.arrange()
        entered, resume, inspected, exited = (threading.Event() for _ in range(4))
        errors, retained = [], []
        decoded = 0

        def decode(pcm, register=None):
            nonlocal decoded
            decoded += 1
            units._window = _recognizer_window(doc(BAD if decoded == 1 else GOOD), len(pcm) // 32)
            return units._window.units

        def retry(*_):
            entered.set()
            if not resume.wait(3):
                raise AssertionError("test did not release recovery")
            return doc(GOOD)

        def exit_privacy():
            try:
                if not inspected.wait(3):
                    raise AssertionError("queue did not inspect overflow tail")
                retained.extend(
                    (packet, packet.pcm, packet.units, packet.window)
                    for packet in (*p._waiting, p._finalizing)
                )
                c.duck(False)
            except Exception as error:
                errors.append(error)
            finally:
                exited.set()

        heard_private = p._heard_private

        def pause_after_inspection(start, end):
            if p._finalizing is not None and p._finalizing.number == 4:
                self.assertTrue(p._waiting[-1].overflow)
                inspected.set()
                self.assertTrue(exited.wait(3), "privacy-off waited for queue admission")
            return heard_private(start, end)

        units.transcribe = decode
        c.duck(True)
        e.drain()
        control = threading.Thread(target=exit_privacy)
        control.start()
        with (
            patch("rightyo.live_audio.ASR_HELD_WINDOWS", 1),
            patch("rightyo.live_audio._transcribe", side_effect=retry),
            patch.object(p, "_heard_private", side_effect=pause_after_inspection),
        ):
            try:
                for _ in range(4):
                    self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
                    self.assertTrue(entered.wait(1))
                self.assertTrue(inspected.is_set())
                self.assertEqual(errors, [])
                for packet, pcm, text, window in retained:
                    self.assertTrue(packet.discarded)
                    self.assertFalse(any(pcm))
                    self.assertEqual(text, [])
                    self.assertFalse(packet.pcm or packet.units or packet.timeline)
                    self.assertIsNone(packet.window)
                    if window is not None:
                        self.assertEqual(window.units, [])
                self.assertFalse(p.closed or p.failed or c._stop.is_set())
                self.assertEqual(e.drain(), [])
                self.assertFalse(c._private_sentence.pcm)
            finally:
                inspected.set()
                resume.set()
                control.join(3)
                if p._recovery is not None:
                    self.assertTrue(p._recovery.done.wait(3))
                    p._drain_results()
        self.assertFalse(control.is_alive())
        self.assertEqual(errors, [])
        units.transcribe = lambda *_: [
            {"start_ms": 200, "end_ms": 400, "text": "Authored public sentence."}
        ]
        self.feed(c, p, VOICE * 10 + SILENCE * 72, s)
        p.finish()
        self.assertEqual(
            [x["turn"]["text"] for x in e.drain() if x["type"] == "transcript"],
            ["Authored public sentence."],
        )

    def test_public_admission_rechecks_discarded_overflow_tail_after_callback(self):
        _, _, _, p, _, _ = self.arrange()
        turns = []
        p.on_turn = turns.append

        def packet(offset, number):
            window = _recognizer_window(doc(GOOD), 200)
            return _FinalizedWindow(
                bytearray(VOICE * 10),
                offset,
                offset + 200,
                offset,
                number,
                [],
                "causal-replay",
                True,
                window.units,
                window,
                (0, 0),
                200,
                False,
            )

        tail, public = packet(0, 1), packet(200, 2)
        private_pcm = tail.pcm
        with p._recovery_lock:
            p._append_overflow(tail)
        private_units, private_window = tail.units, tail.window

        # Simulate privacy-off while the owner is outside its lifecycle lock querying
        # the host predicate. Only the old tail intersects the closed private interval.
        def exit_before_append(*_):
            p.end_privacy(200 * 32)
            return False

        with patch.object(p, "_heard_private", side_effect=exit_before_append):
            p._queue_window(public)
        self.assertTrue(tail.discarded)
        self.assertFalse(tail.overflow)
        self.assertFalse(any(private_pcm))
        self.assertEqual(private_units, [])
        self.assertEqual(private_window.units, [])
        self.assertIsNone(tail.window)
        self.assertFalse(public.discarded or public.overflow)
        self.assertEqual([turn.text for turn in turns], ["RECOVERED_FIXTURE"])
        self.assertFalse(p.closed or p.failed)

    def test_attributed_private_units_are_dropped_if_exit_runs_in_host_predicate(self):
        for verdict in (False, True):
            with self.subTest(verdict=verdict):
                _, _, _, p, _, _ = self.arrange()
                turns, private_audio = [], Mock()
                p.on_turn = turns.append
                p.config = replace(p.config, on_private_turn_audio=private_audio)
                window = _recognizer_window(doc(GOOD), 200)
                packet = _FinalizedWindow(
                    bytearray(VOICE * 10),
                    0,
                    200,
                    0,
                    1,
                    [],
                    "causal-replay",
                    True,
                    window.units,
                    window,
                    (0, 0),
                    200,
                    False,
                )
                pcm, units = packet.pcm, packet.units

                def exit_during_attribution(*_):
                    p.end_privacy(200 * 32)
                    return verdict

                with patch.object(p, "_heard_private", side_effect=exit_during_attribution):
                    p._commit_window(packet)
                self.assertTrue(packet.discarded)
                self.assertFalse(any(pcm))
                self.assertEqual(units, [])
                self.assertIsNone(packet.window)
                self.assertFalse(p._merger.holding)
                private_audio.assert_not_called()
                self.assertEqual(turns, [])
                self.assertFalse(p.closed or p.failed)

    def test_exit_before_window_registration_refuses_late_private_decode(self):
        c, e, s, p, _, units = self.arrange()
        c.duck(True)
        e.drain()
        finalize = p._finalize

        def off_before_registration(timeline, **boundary):
            # All private frames are audited, but no finalized packet exists yet.
            c.duck(False)
            finalize(timeline, **boundary)

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

    def replay_private_windows(
        self,
        *windows,
        retry=None,
        timeline=None,
        audio=None,
        end_of_turn=None,
        pre_roll_ms=20,
        vad=FixtureVad(),
        **window_settings,
    ):
        """Replay authored Whisper documents or a replaceable adapter's unit lists."""
        c, e, s, p, _, units = self.arrange(
            end_of_turn=end_of_turn, pre_roll_ms=pre_roll_ms, vad=vad
        )
        c._enrolled_roles.exit_audio = c._private_sentence
        s.exit_score = Mock(return_value=EntryScore(0.9, None, 0, "scored"))
        pending = iter(windows)

        def decode(pcm, register=None):
            result = next(pending)
            if isinstance(result, list):
                return result
            units._window = _recognizer_window(result, len(pcm) // 32)
            return units._window.units

        def private_audio(turn, pcm):
            c._private_turn_audio(c._generation, turn, pcm)
            if not turn.recognition_complete:
                self.assertFalse(c._private_sentence.pcm)

        units.transcribe = decode
        p.config = replace(
            p.config,
            provenance="causal-replay",
            on_private_turn_audio=private_audio,
            **window_settings,
        )
        c.duck(True)
        e.drain()
        if timeline is not None:
            p._diarizer.segments = Mock(return_value=timeline)
        if audio is None:
            audio = [SILENCE * 10 + VOICE * 4 + SILENCE * 72] * len(windows)
        with patch("rightyo.live_audio._transcribe", return_value=retry) as redecode:
            for pcm in audio:
                self.feed(c, p, pcm, s)
                p._drain_results(wait=True)
            p.finish()
        self.assertEqual(list(pending), [])
        self.assertEqual(redecode.call_count, int(retry is not None))
        self.assertFalse(c._private_sentence.pcm)
        return [event["action"] for event in e.drain()], s.exit_score.call_count

    def test_processor_podcast_tail_with_valid_missing_or_invalid_timestamps(self):
        command = segment("Jarvis, start listening", 0, 100)
        for start in (100, None, -1):
            with self.subTest(tail_start=start):
                window = doc(command, segment(" to the podcast", start, 200))
                self.assertEqual(self.replay_private_windows(window), ([], 0))
        self.assertEqual(self.replay_private_windows(doc(command)), (["exit"], 1))

    def test_omitted_voiced_tail_and_stream_start(self):
        # Identical synthetic voice before/after the recognised command. Only the
        # fully observed, fully covered statement may reach the stand-in owner check.
        command = doc(segment("Jarvis, start listening", 20, 1020))
        pause = SILENCE * 72
        # #350 round 5 requires 200 ms of observed speech absence before the onset.
        for frames in (1, 9):
            self.assertEqual(
                self.replay_private_windows(command, audio=[SILENCE * frames + VOICE * 50 + pause]),
                ([], 0),
            )
        self.assertEqual(
            self.replay_private_windows(command, audio=[SILENCE * 10 + VOICE * 50 + pause]),
            (["exit"], 1),
        )
        # The same coverage rule applies when end-of-turn accepts a shorter pause.
        for length, expected in ((50, (["exit"], 1)), (51, ([], 0))):
            endpoint = Mock(return_value=1.0)
            self.assertEqual(
                self.replay_private_windows(
                    command,
                    audio=[SILENCE * 10 + VOICE * length + SILENCE * 10],
                    end_of_turn=endpoint,
                ),
                expected,
            )
            endpoint.assert_called_once()
        for result in (
            command,
            doc(*command["transcription"], segment(" to the podcast", 1020, 3020)),
        ):
            self.assertEqual(
                self.replay_private_windows(result, audio=[SILENCE * 10 + VOICE * 150 + pause]),
                ([], 0),
            )
        full = doc(segment("Did you hear Jarvis start listening", 20, 1620))
        self.assertEqual(
            self.replay_private_windows(full, audio=[SILENCE * 10 + VOICE * 80 + pause]), ([], 0)
        )
        clipped = doc(segment("Jarvis start listening", 0, 1000))
        self.assertEqual(self.replay_private_windows(clipped, audio=[VOICE * 50 + pause]), ([], 0))

    def test_allowed_trailing_word_cannot_bridge_omitted_speech(self):
        phrase = doc(segment("Jarvis, privacy off", 20, 1020), segment(" please", 1620, 1820))
        for gap, expected in ((SILENCE, (["exit"], 1)), (VOICE, ([], 0))):
            with self.subTest(expected=expected):
                audio = [SILENCE * 10 + VOICE * 50 + gap * 30 + VOICE * 10 + SILENCE * 72]
                self.assertEqual(self.replay_private_windows(phrase, audio=audio), expected)

    def test_each_exit_word_and_name_need_speech_in_their_own_interval(self):
        audio = [SILENCE * 10 + VOICE * 50 + SILENCE * 72]
        for words in (
            [("Jarvis,", 20, 1020), (" privacy", 1120, 1320), (" off.", 1320, 1420)],
            [("Jarvis,", 0, 20), (" privacy", 20, 820), (" off.", 820, 1020)],
            [("Jarvis,", 20, 820), (" privacy", 820, 1020), (" off.", 1020, 1040)],
        ):
            parts = [segment(*word) for word in words]
            phrase = segment("Jarvis, privacy off.", words[0][1], words[-1][2])
            for window in (
                doc(*parts),
                doc({**phrase, "tokens": parts}),
                [{"text": text, "start_ms": start, "end_ms": end} for text, start, end in words],
            ):
                with self.subTest(words=words, window=window):
                    self.assertEqual(self.replay_private_windows(window, audio=audio), ([], 0))
        # The internal gap lies between voiced name and final word.
        audio = [SILENCE * 10 + VOICE * 20 + SILENCE * 10 + VOICE * 20 + SILENCE * 72]
        window = doc(
            segment("Jarvis,", 20, 420), segment(" privacy", 420, 620), segment(" off.", 620, 1020)
        )
        self.assertEqual(self.replay_private_windows(window, audio=audio), ([], 0))

    def test_silent_subword_cannot_lend_voice_to_the_next_word_in_its_token(self):
        audio = [SILENCE * 10 + VOICE * 50 + SILENCE * 72]
        phrase = segment("Jarvis, privacy off.", 20, 1420)
        # A partly padded word still overlaps voice. An additional word timed
        # only by the padded token does not.
        for first, last, expected in (
            ("Jarvis, privacy o", "ff.", (["exit"], 1)),
            ("Jarvis, pri", "vacy off.", ([], 0)),
        ):
            with self.subTest(last=last):
                parts = [segment(first, 20, 1020), segment(last, 1120, 1420)]
                self.assertEqual(
                    self.replay_private_windows(doc({**phrase, "tokens": parts}), audio=audio),
                    expected,
                )

    def test_split_window_uses_acoustic_end_even_with_a_padded_name_token(self):
        rest = doc(segment("privacy off.", 20, 1020))
        for gap_ms, expected in ((4000, (["exit"], 2)), (4200, ([], 1))):
            for token_end in (1020, 2120):
                with self.subTest(gap_ms=gap_ms, token_end=token_end):
                    name = segment("Jarvis", 20, token_end)
                    audio = [
                        SILENCE * 10 + VOICE * 50 + SILENCE * (gap_ms // 20),
                        VOICE * 50 + SILENCE * 72,
                    ]
                    self.assertEqual(
                        self.replay_private_windows(
                            doc({**name, "tokens": [name]}), rest, audio=audio, timeline=[]
                        ),
                        expected,
                    )

    def test_short_pause_after_speech_cannot_reuse_the_quiet_before_it(self):
        first = doc(segment("Other speech.", 20, 1020))
        # The one silent frame closes the first turn. There is no pre-roll
        # before the second voice and no new 200 ms quiet run.
        second = doc(segment("Jarvis, privacy off.", 0, 1000))
        audio = [SILENCE * 10 + VOICE * 50 + SILENCE + VOICE * 50 + SILENCE]
        self.assertEqual(
            self.replay_private_windows(
                first, second, audio=audio, end_of_turn=lambda _: 1, end_of_turn_silence_ms=20
            ),
            ([], 0),
        )

    def test_naturally_closed_stream_start_does_not_bar_a_fresh_onset(self):
        first = doc(segment("First utterance.", 0, 1000))
        second = doc(segment("Jarvis, privacy off.", 20, 1020))
        for gap in (15, 72):
            with self.subTest(gap=gap):
                audio = [VOICE * 50 + SILENCE * gap + VOICE * 50 + SILENCE * 10]
                self.assertEqual(
                    self.replay_private_windows(
                        first, second, audio=audio, end_of_turn=lambda _: 1
                    ),
                    (["exit"], 1),
                )

    def test_full_ending_pause_exactly_at_the_cap_is_a_natural_end(self):
        for length, expected in ((126, (["exit"], 1)), (127, (["exit"], 1)), (128, ([], 0))):
            with self.subTest(voice_frames=length):
                phrase = doc(segment("Jarvis, privacy off.", 20, 20 + 20 * length))
                audio = [SILENCE * 10 + VOICE * length + SILENCE * 72]
                self.assertEqual(
                    self.replay_private_windows(phrase, audio=audio, max_utterance_ms=4000),
                    expected,
                )

    def test_natural_end_of_forced_continuation_clears_barrier_for_next_onset(self):
        cut = doc(segment("Forced cut.", 20, 4000))
        continuation = doc(segment("Jarvis, privacy off.", 0, 200))
        fresh = doc(segment("Jarvis, privacy off.", 20, 1020))
        lead = SILENCE * 10 + VOICE * 209
        self.assertEqual(
            self.replay_private_windows(
                cut,
                continuation,
                audio=[lead + SILENCE * 15],
                end_of_turn=lambda _: 1.0,
                max_utterance_ms=4000,
            ),
            ([], 0),
        )
        for gap in (15, 72):
            with self.subTest(gap_ms=20 * gap):
                audio = [lead + SILENCE * gap + VOICE * 50 + SILENCE * 10]
                self.assertEqual(
                    self.replay_private_windows(
                        cut,
                        continuation,
                        fresh,
                        audio=audio,
                        end_of_turn=lambda _: 1.0,
                        max_utterance_ms=4000,
                    ),
                    (["exit"], 1),
                )

    def test_cap_honours_completion_waiting_for_private_diarizer(self):
        for cap in (4000, 6000):
            with self.subTest(cap=cap):
                endpoint = Mock(return_value=1.0)
                self.assertEqual(
                    self.replay_private_windows(
                        doc(segment("Jarvis, privacy off.", 20, 3620)),
                        audio=[SILENCE * 10 + VOICE * 180 + SILENCE * 72],
                        end_of_turn=endpoint,
                        timeline=[],
                        max_utterance_ms=cap,
                    ),
                    (["exit"], 1),
                )
                endpoint.assert_called_once()

    @staticmethod
    def fan_frame(gain, voice_ratio=0, voice_hz=250):
        return array(
            "h",
            (
                round(
                    gain
                    * (
                        math.sin(2 * math.pi * 100 * i / 16000)
                        + voice_ratio * math.sin(2 * math.pi * voice_hz * i / 16000)
                    )
                )
                for i in range(320)
            ),
        ).tobytes()

    def test_fan_noise_and_continued_voice_at_the_same_snr(self):
        # Two orthogonal tones in each 20 ms frame. The fan alone is above the
        # ordinary public energy gate at the louder gain. Both mixes have 10 dB SNR.
        for gain in (100, 1000):
            quiet, voiced = self.fan_frame(gain), self.fan_frame(gain, math.sqrt(10))
            phrase = doc(segment("Jarvis, privacy off.", 200, 1200))
            for extra, expected in ((0, (["exit"], 1)), (100, ([], 0))):
                with self.subTest(gain=gain, extra=extra):
                    audio = [quiet * 10 + voiced * (50 + extra) + quiet * 72]
                    self.assertEqual(
                        self.replay_private_windows(
                            phrase, vad=FixtureVad(quiet), audio=audio, pre_roll_ms=240
                        ),
                        expected,
                    )
            clipped = doc(segment("Jarvis, privacy off.", 0, 1000))
            self.assertEqual(
                self.replay_private_windows(
                    clipped, vad=FixtureVad(quiet), audio=[voiced * 50 + quiet * 72]
                ),
                ([], 0),
            )

    def test_digital_silence_before_a_command_does_not_zero_the_ending_floor(self):
        fan, voice = self.fan_frame(1000), self.fan_frame(1000, math.sqrt(10))
        phrase = doc(segment("Jarvis, privacy off.", 20, 1020))
        for lead in (SILENCE * 10, SILENCE * 10 + fan):
            with self.subTest(fan_lead=lead.endswith(fan)):
                audio = [lead + voice * 50 + fan * 72]
                self.assertEqual(
                    self.replay_private_windows(phrase, vad=FixtureVad(fan), audio=audio),
                    (["exit"], 1),
                )

    def test_subframe_edges_over_50_and_100_hz_background_peaks(self):
        def frame(voice):
            phases = (2 * math.pi * i / 320 for i in range(320))
            return array(
                "h",
                (
                    round(1000 * math.cos(t) + 500 * math.cos(2 * t) + voice * math.cos(5 * t))
                    for t in phases
                ),
            ).tobytes()

        fan, voiced = frame(0), frame(5000)
        for shift in (0, 1, 2, 18, 19):
            with self.subTest(shift=shift):
                edge = shift * 32
                audio = [
                    fan * 10
                    + fan[:edge]
                    + voiced[edge:]
                    + voiced * 49
                    + voiced[:edge]
                    + fan[edge:]
                    + fan * 72
                ]
                phrase = doc(segment("Jarvis, privacy off.", 200 + shift, 1200 + shift))
                self.assertEqual(
                    self.replay_private_windows(
                        phrase, vad=FixtureVad(fan), audio=audio, pre_roll_ms=240
                    ),
                    (["exit"], 1),
                )
                if shift:
                    early = doc(segment("Jarvis, privacy off.", 200 + shift, 1199 + shift))
                    self.assertEqual(
                        self.replay_private_windows(
                            early, vad=FixtureVad(fan), audio=audio, pre_roll_ms=240
                        ),
                        ([], 0),
                    )

    def test_whisper_period_in_verified_ending_silence_is_neither_repair_nor_voice(self):
        speech = segment("Jarvis, privacy off", 20, 1020)
        audio = [SILENCE * 10 + VOICE * 50 + SILENCE * 72]
        for left, right in ((1120, 1120), (1120, 1140)):
            period = segment(".", left, right)
            phrase = segment("Jarvis, privacy off.", 20, right)
            for window in (
                doc(speech, period),
                doc(speech, {**period, "tokens": []}),
                doc({**phrase, "tokens": [speech, period]}),
            ):
                with self.subTest(window=window):
                    self.assertEqual(
                        self.replay_private_windows(window, audio=audio), (["exit"], 1)
                    )
                    # The punctuation cannot account for even one omitted voiced frame.
                    tail = [SILENCE * 10 + VOICE * 50 + SILENCE * 5 + VOICE + SILENCE * 72]
                    self.assertEqual(self.replay_private_windows(window, audio=tail), ([], 0))

    def test_continuous_voice_from_stream_start_cannot_be_its_own_floor(self):
        for gain in (50, 500):
            with self.subTest(gain=gain):
                # Voice already 20 dB over the fan, louder for one second, then back
                # to its initial level throughout the presumed ending pause.
                voice, louder = self.fan_frame(gain, 10), self.fan_frame(gain, 30)
                command = doc(segment("Jarvis, privacy off.", 400, 1400))
                audio = [voice * 20 + louder * 50 + voice * 100]
                self.assertEqual(
                    self.replay_private_windows(command, audio=audio, pre_roll_ms=400), ([], 0)
                )

    def test_low_pitched_start_must_persist_as_background_under_the_command(self):
        # An initially smooth low voice can resemble a fan. Getting louder or
        # changing pitch while it speaks does not establish a speech-absent start.
        for before, during in ((124, 124), (100, 250)):
            with self.subTest(before=before, during=during):
                voice = self.fan_frame(100, 10, before)
                louder = self.fan_frame(100, 30, during)
                phrase = doc(segment("Jarvis, privacy off.", 400, 1400))
                audio = [voice * 20 + louder * 50 + voice * 100]
                self.assertEqual(
                    self.replay_private_windows(phrase, audio=audio, pre_roll_ms=400), ([], 0)
                )

    def test_steady_fan_alone_never_opens_asr_and_startup_zero_does_not_poison_it(self):
        for gain in (100, 1000):
            with self.subTest(gain=gain):
                fan, voice = self.fan_frame(gain), self.fan_frame(gain, math.sqrt(10))
                self.assertEqual(
                    self.replay_private_windows(vad=FixtureVad(fan), audio=[fan * 1200]), ([], 0)
                )
                audio = [SILENCE + fan * 10 + voice * 50 + fan * 72]
                phrase = doc(segment("Jarvis, privacy off.", 220, 1220))
                self.assertEqual(
                    self.replay_private_windows(
                        phrase, vad=FixtureVad(fan), audio=audio, pre_roll_ms=240
                    ),
                    (["exit"], 1),
                )

    def test_zero_duration_punctuation_cannot_cover_an_omitted_voiced_tail(self):
        command = segment("Jarvis, privacy off.", 20, 3020)
        command["tokens"] = [segment("Jarvis, privacy off", 20, 1020), segment(".", 3020, 3020)]
        audio = [SILENCE * 10 + VOICE * 150 + SILENCE * 72]
        self.assertEqual(self.replay_private_windows(doc(command), audio=audio), ([], 0))
        command["tokens"][0]["offsets"]["to"] = 3020
        self.assertEqual(self.replay_private_windows(doc(command), audio=audio), (["exit"], 1))

    def test_gaps_between_original_subwords_must_be_speech_absent(self):
        command = segment("Jarvis, privacy off.", 20, 1220)
        command["tokens"] = [
            segment("Jarvis, pri", 20, 420),
            segment("vacy", 820, 1020),
            segment(" off.", 1020, 1220),
        ]
        for gap, expected in ((VOICE, ([], 0)), (SILENCE, (["exit"], 1))):
            with self.subTest(expected=expected):
                audio = [SILENCE * 10 + VOICE * 20 + gap * 20 + VOICE * 20 + SILENCE * 72]
                self.assertEqual(self.replay_private_windows(doc(command), audio=audio), expected)

    def test_punctuation_in_the_pause_cannot_extend_a_split_names_lifetime(self):
        name = segment("Jarvis,", 20, 1200)
        name["tokens"] = [segment("Jarvis", 20, 100), segment(",", 1200, 1200)]
        rest = doc(segment("privacy off.", 20, 100))
        for result in (
            doc(name),
            [
                {"text": "Jarvis", "start_ms": 20, "end_ms": 100},
                {"text": ",", "start_ms": 1200, "end_ms": 1200},
            ],
        ):
            with self.subTest(result=result):
                audio = [
                    SILENCE * 10 + VOICE * 4 + SILENCE * 210,
                    VOICE * 4 + SILENCE * 72,
                ]
                self.assertEqual(
                    self.replay_private_windows(result, rest, audio=audio, timeline=[]), ([], 1)
                )

    def test_timestamp_edges_and_internal_gaps_between_acoustic_frames(self):
        for shift in (0, 1, 10, 19):
            with self.subTest(shift=shift):
                lead = SILENCE * 10 + bytes(32 * shift)
                tail = SILENCE * 73
                start, stop = 20 + shift, 1020 + shift
                command = doc(segment("Jarvis, privacy off.", start, stop))
                audio = [lead + VOICE * 50 + tail]
                self.assertEqual(self.replay_private_windows(command, audio=audio), (["exit"], 1))
                # Even a short voiced overhang inside the same frame is outside the word.
                audio = [lead + VOICE * 50 + VOICE[:32] + tail]
                self.assertEqual(self.replay_private_windows(command, audio=audio), ([], 0))
                command = doc(
                    segment("Jarvis,", start, start + 400),
                    segment(" privacy off.", start + 600, stop),
                )
                audio = [lead + VOICE * 20 + SILENCE * 10 + VOICE * 20 + tail]
                self.assertEqual(self.replay_private_windows(command, audio=audio), (["exit"], 1))

    def test_subframe_word_edges_over_nonzero_background(self):
        fan, voice = self.fan_frame(1000), self.fan_frame(1000, math.sqrt(10))
        onset, offset = fan[:320] + voice[320:], voice[:320] + fan[320:]
        word = onset + voice * 19 + offset
        audio = [fan * 10 + word + fan * 9 + word + fan * 72]
        doc_with_gap = doc(segment("Jarvis,", 30, 430), segment(" privacy off.", 630, 1030))
        self.assertEqual(
            self.replay_private_windows(doc_with_gap, vad=FixtureVad(fan), audio=audio),
            (["exit"], 1),
        )

    def test_partial_speaker_coverage_cannot_separate_an_exit_from_its_window(self):
        phrase = segment("Jarvis, start listening", 0, 100)
        prefix = segment("Did you hear ", 0, 50)
        timeline = [{"speaker": 1, "start_ms": 180, "end_ms": 280}]
        for window in (
            doc(phrase, segment(" to the podcast", 100, 200)),
            doc(prefix, segment("Jarvis, start listening", 100, 200)),
            doc(segment("Jarvis", 0, 100), segment(" start listening", 100, 200)),
        ):
            with self.subTest(window=window):
                self.assertEqual(self.replay_private_windows(window, timeline=timeline), ([], 0))
        self.assertEqual(self.replay_private_windows(doc(phrase), timeline=timeline), (["exit"], 1))

    def test_only_a_naturally_ended_whole_utterance_can_exit(self):
        phrase = doc(segment("Jarvis start listening", 0, 200))
        suffix = doc(segment(" to the podcast", 0, 200))
        pause = SILENCE * 72
        self.assertEqual(
            self.replay_private_windows(phrase, audio=[SILENCE * 10 + VOICE * 9]), ([], 0)
        )
        self.assertEqual(
            self.replay_private_windows(
                doc(segment("Jarvis, privacy off", 20, 12000)),
                suffix,
                audio=[SILENCE * 10 + VOICE * 599, VOICE * 10 + pause],
            ),
            ([], 0),
        )
        # A naturally ended continuation still has a clipped beginning. A fresh
        # utterance following the pause can authorize an exit again.
        self.assertEqual(
            self.replay_private_windows(
                suffix, phrase, audio=[SILENCE * 10 + VOICE * 599, VOICE * 10 + pause]
            ),
            ([], 0),
        )
        self.assertEqual(
            self.replay_private_windows(
                suffix,
                doc(segment("Jarvis start listening", 240, 440)),
                audio=[SILENCE * 10 + VOICE * 590, pause + VOICE * 10 + pause],
                pre_roll_ms=240,
            ),
            (["exit"], 1),
        )

    def test_blank_segment_cannot_hide_tokens_or_timed_silence(self):
        command = segment("Jarvis, start listening", 0, 100)
        for blank in (
            {**segment("", 100, 100), "tokens": [segment(" to the podcast", 100, 200)]},
            {**segment("  ", 100, 200), "tokens": [segment(" to the podcast", 100, 200)]},
            segment("  ", 100, 200),
        ):
            with self.subTest(blank=blank):
                self.assertEqual(self.replay_private_windows(doc(command, blank)), ([], 0))

    def test_whisper_keeps_line_boundaries_in_segments_and_token_reconstruction(self):
        start = segment("Jarvis, start", 20, 60)
        for br in ("\n", "\u2028", "\u2029", "\x85"):
            for reported, token in ((br, br), (br, " "), (" ", br)):
                last = segment(reported + "listening.", 60, 100)
                last["tokens"] = [segment(token + "listening.", 60, 100)]
                with self.subTest(reported=reported, token=token):
                    window = _recognizer_window(doc(start, last), 100)
                    self.assertEqual(window.timing_incomplete, reported != token)
                    self.assertEqual(
                        "".join(u["text"] for u in window.units),
                        "Jarvis, start" + reported + "listening.",
                    )
                    self.assertEqual(self.replay_private_windows(doc(start, last)), ([], 0))
            point = segment(br, 60, 60)
            document = doc(start, point, segment(" listening.", 60, 100))
            with self.subTest(point=br):
                self.assertIn(
                    br, "".join(u["text"] for u in _recognizer_window(document, 100).units)
                )
                self.assertEqual(self.replay_private_windows(document), ([], 0))
                # A matching newline token can also separate a finished sentence
                # from a complete addressed command in the same Whisper window.
                separated = doc(
                    segment("Done.", 20, 60),
                    {**point, "tokens": [point]},
                    segment("Jarvis, privacy off.", 60, 100),
                )
                self.assertFalse(_recognizer_window(separated, 100).timing_incomplete)
                self.assertEqual(self.replay_private_windows(separated), (["exit"], 1))

    def test_token_timing_is_checked_before_joining_words(self):
        for start, end in ((40, 60), (90, 150), (110, 160)):
            with self.subTest(start=start, end=end):
                phrase = segment("Jarvis, start listening", 0, 200)
                phrase["tokens"] = [
                    segment("Jarvis, start", 0, 90),
                    segment(" listen", 90, 150),
                    segment("ing", start, end),
                ]
                self.assertEqual(self.replay_private_windows(doc(phrase)), ([], 0))
        for middle, start, last in ((" start", 40, " listening"), (" ", 50, "start listening")):
            with self.subTest(middle=middle):
                parts = [
                    segment("Jarvis,", 0, 50),
                    segment(middle, start, 100),
                    segment(last, 100, 200),
                ]
                phrase = segment("Jarvis, start listening", 0, 200)
                self.assertEqual(
                    self.replay_private_windows(doc({**phrase, "tokens": parts})), ([], 0)
                )

    def test_replaceable_adapter_cannot_hide_a_textless_or_overlapping_unit(self):
        def piece(text, start, end):
            return {"text": text, "start_ms": start, "end_ms": end}

        name, rest = piece("Jarvis", 0, 50), piece(" start listening", 100, 150)
        blank = piece(" \t ", 50, 100)
        for units in (
            [name, blank, rest],
            [name, rest, piece(" ", 150, 200)],
            [name, piece(" start listening", 40, 100)],
        ):
            with self.subTest(units=units):
                self.assertEqual(self.replay_private_windows(units), ([], 0))
        command = piece("Jarvis, start listening", 0, 100)
        self.assertEqual(self.replay_private_windows([command]), (["exit"], 1))
        for invisible in ("\u200b", "\ufeff", "\u2060"):
            with self.subTest(invisible=invisible):
                self.assertEqual(
                    self.replay_private_windows([command, piece(invisible, 100, 200)]), ([], 0)
                )
        # #350 round 12 P2-2 corrects this PR's earlier standalone-period refusal.
        self.assertEqual(
            self.replay_private_windows([command, piece(".", 100, 200)]), (["exit"], 1)
        )
        spoken = piece("Jarvis, privacy off", 20, 1020)
        period = piece(".", 1120, 1140)
        for tail, expected in ((SILENCE, (["exit"], 1)), (VOICE, ([], 0))):
            audio = [SILENCE * 10 + VOICE * 50 + SILENCE * 5 + tail + SILENCE * 72]
            self.assertEqual(self.replay_private_windows([spoken, period], audio=audio), expected)
        # Here each standalone fragment has timings for its own complete window.
        name, rest = piece("Jarvis", 0, 100), piece("start listening", 0, 100)
        self.assertEqual(self.replay_private_windows([name], [blank], [rest]), ([], 1))
        self.assertEqual(self.replay_private_windows([name], [rest]), (["exit"], 2))

    def test_exit_refuses_an_untimed_word_inside_or_after_the_phrase(self):
        for parts in (
            (("Jarvis,", 0, 100), (" start", 100, 100), (" listening", 100, 200)),
            (("Jarvis, start listening", 0, 100), (" please", 100, 100)),
        ):
            with self.subTest(parts=parts):
                self.assertEqual(
                    self.replay_private_windows(doc(*(segment(*part) for part in parts))),
                    ([], 0),
                )

    def test_segment_fallback_cannot_hide_an_untimed_token(self):
        # Whisper keeps the timed segment's full text when token timing fails.
        phrase = segment("Jarvis, start listening", 0, 200)
        phrase["tokens"] = [
            segment("Jarvis,", 0, 50),
            {"text": " start"},
            segment(" listening", 100, 200),
        ]
        # Exercise the missing timestamp with voice under all three words too.
        full_audio = [SILENCE * 10 + VOICE * 9 + SILENCE * 72]
        self.assertEqual(self.replay_private_windows(doc(phrase)), ([], 0))
        self.assertEqual(self.replay_private_windows(doc(phrase), audio=full_audio), ([], 0))
        phrase["tokens"][1] = segment(" start", 50, 100)
        # The shorter replay ends before "listening" begins.
        self.assertEqual(self.replay_private_windows(doc(phrase)), ([], 0))
        self.assertEqual(self.replay_private_windows(doc(phrase), audio=full_audio), (["exit"], 1))

    def test_split_exit_refuses_loss_on_either_fragment_or_between_them(self):
        name = segment("Jarvis", 0, 100)
        rest = segment("start listening", 0, 100)
        for documents, retry, checks in (
            ((doc(name, BAD), doc(rest)), None, 0),
            ((doc(name), doc(rest, BAD)), None, 1),
            ((doc(name), doc(BAD)), doc(rest), 1),
            ((doc(name), doc(BAD), doc(rest)), doc(BAD), 1),
            ((doc(name), doc(), doc(rest)), None, 1),
        ):
            with self.subTest(documents=documents, retry=retry):
                self.assertEqual(self.replay_private_windows(*documents, retry=retry), ([], checks))

        # A fresh pair entirely after a loss remains eligible.
        self.assertEqual(
            self.replay_private_windows(doc(BAD), doc(name), doc(rest), retry=doc(BAD)),
            (["exit"], 2),
        )

    def test_sentence_initial_split_name_and_remainder_each_reach_the_owner_check(self):
        name = doc(segment("We're done. Jarvis,", 20, 100))
        rest = doc(segment("Privacy off.", 20, 100))
        self.assertEqual(self.replay_private_windows(name, rest), (["exit"], 2))

    def test_processor_split_exit_authenticates_both_unlabelled_fragments(self):
        for scores, allowed in (((0.9, 0.9), True), ((0.1,), False), ((0.9, 0.1), False)):
            with self.subTest(scores=scores):
                c, e, s, p, _, units = self.arrange(pre_roll_ms=20)
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
                self.feed(c, p, SILENCE * 10 + VOICE * 9 + SILENCE * 72, s)
                self.assertEqual(e._private_name_end is not None, scores[0] > 0.5)
                self.assertFalse(bool(c._private_sentence.pcm))
                self.feed(c, p, SILENCE + VOICE * 9 + SILENCE * 72, s)
                self.assertEqual("exit" in [x.get("action") for x in e.drain()], allowed)
                self.assertEqual(s.exit_score.call_count, len(scores))
                self.assertTrue(all(t.speaker_id is None for t in turns))
                if len(scores) == 2:
                    self.assertFalse(bool(c._private_sentence.pcm))
                self.assertEqual(s.offered, 0)

    def test_positioned_off_after_a_silent_chunk_preserves_the_bound_owner(self):
        for timing in ("before", "during", "both", "during_next_chunk"):
            with self.subTest(timing=timing):
                controller, events, shadow, processor, notes, _ = self.arrange()
                controller._mode = "stdin"
                shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
                shadow._bound_ids.add("owner")
                events._roles["Speaker A"] = "owner"
                controller.duck(True)
                if timing in ("before", "both"):
                    controller.duck_after_sample(3200)

                edge_ms = 200
                if timing == "during_next_chunk":
                    controller._feed(
                        controller._generation, processor, bytes(6400), "stdin", shadow
                    )
                    edge_ms = 400

                push_pcm16 = processor.push_pcm16

                def push(pcm):
                    if timing != "before":
                        controller.duck_after_sample(3200)
                    self.assertEqual(controller._pending_off_sample, 3200)
                    self.assertTrue(events.ducked)
                    push_pcm16(pcm)
                    self.assertTrue(events.ducked)

                with (
                    patch.object(processor, "push_pcm16", push),
                    patch.object(events, "duck", wraps=events.duck) as apply,
                ):
                    controller._feed(
                        controller._generation, processor, bytes(6400), "stdin", shadow
                    )
                apply.assert_called_once_with(False, now_ms=edge_ms, ack=False)
                self.assertIsNone(controller._pending_off_sample)
                self.assertEqual(events._duck_spans, [[0, edge_ms]])
                self.assertIn("diarizer_private_frames=0 forget=skipped", notes[-1])
                self.assertEqual(shadow.binding("Speaker A"), ("bound", "owner"))
                self.assertEqual(events._roles["Speaker A"], "owner")
                self.assertFalse(processor._speaker_reset)

    def test_silent_privacy_keeps_every_public_label_state_and_next_owner_role(self):
        c, e, s, p, notes, _ = self.arrange()
        s._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
        s._bound_ids.add("owner")
        s.scored_seconds = 4
        e._roles["Speaker A"] = "owner"
        e._cue_voices.append((0, "Speaker A", None))
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
        self.assertEqual(tuple(e._cue_voices), ((0, "Speaker A", None),))
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
                c, e, s, p, _, units = self.arrange(pre_roll_ms=20)
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
                self.feed(c, p, SILENCE * 10 + VOICE * 9 + SILENCE * 72, s)
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
