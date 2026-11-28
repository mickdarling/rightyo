"""#946 step 1: authored PCM and fake embeddings only; no speech or model assets."""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import tempfile
import threading
import unittest
from argparse import Namespace
from array import array
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from test_edge_attribution import Timeline, Units
from test_live_speaker_id import (
    GUEST_F0,
    OWNER_F0,
    Harness,
    ScriptedEmbedder,
    ToneEmbedder,
    settings,
    tone,
    turn,
    unit,
    wait_for,
)
from test_privacy_minimum import FixtureVad
from test_silero_vad import fake_worker

from rightyo import speaker_id
from rightyo.contracts import Addressing, SpeakerPriority
from rightyo.live_audio import LiveConfig, LiveProcessor
from rightyo.live_speaker_id import (
    PRIVATE_TURN_MAX_BYTES,
    EnrolledRoles,
    EntryScore,
    LabelState,
    PrivateTurnAudio,
)
from rightyo.memory import TranscriptMemory
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.speaker_id import SpeakerEmbedder, SpeakerIdError, _read_private_pcm
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents

ADDRESSING = Addressing(names=("Unit",))
OWNERS = SpeakerPriority(owners=("owner",))
EXIT = "Unit, privacy off"


def authored_turn(current, text):
    timings = ()
    if text == EXIT:
        boundary = current.start_ms + (current.end_ms - current.start_ms) * 3 // 4
        timings = ((0, 4, current.start_ms, boundary), (6, len(EXIT), boundary, current.end_ms))
    return replace(current, text=text, exit_timings=timings)


class PrivateExitTests(unittest.TestCase):
    def harness(self, embedder=None, **options):
        harness = Harness(embedder or ToneEmbedder(), config=settings(**options))
        harness.shadow.start()
        self.addCleanup(harness.shadow.close)
        wait_for(lambda: bool(harness.shadow._compared))
        return harness

    def controller(self, harness, *, addressing=ADDRESSING):
        events = SpeechEvents(report=harness.lines.append)
        events.duck(True, now_ms=0)
        roles = EnrolledRoles(OWNERS)
        roles.source = harness.shadow
        events.start("shadow-test", now_ms=0, addressing=addressing, priority=roles, duck=True)
        events.drain()
        asset = Path("/synthetic/asset")
        controller = PrototypeController(
            PrototypeConfig(asset, asset, asset, asset, asset, asset), event_publisher=events
        )
        self.addCleanup(controller.close)
        controller._enrolled_roles = roles
        controller._shadow = harness.shadow
        roles.exit_audio = controller._private_sentence
        return controller, events, roles

    def hear(self, controller, text=EXIT, *, start=0, seconds=1, f0=OWNER_F0, inferred=False):
        current = authored_turn(turn(start, start + round(seconds * 1000)), text)
        controller._received_bytes = current.end_ms * 32
        controller._received_ms = current.end_ms
        controller._private_turn_audio(
            controller._generation, current, memoryview(tone(seconds, f0))
        )
        old = controller._private_sentence.pcm
        if inferred:
            controller._inferred_turns.add(current.utterance_id)
        controller._turn(
            controller._generation, queue.Queue(), TranscriptMemory(), controller._shadow, current
        )
        return old

    def test_queued_public_guest_binding_revokes_each_private_exit_fragment(self):
        # Reviewer probe: a public score was queued before privacy. It commits after
        # preparation, before the private fragment commits, without a forget/reset.
        for fragment in ("Unit", "privacy off", EXIT):
            with self.subTest(fragment=fragment):
                entered, release = threading.Event(), threading.Event()

                class BlockingGuest(ScriptedEmbedder):
                    def embed(self, pcm):
                        entered.set()
                        if not release.wait(3):
                            raise AssertionError("public score not released")
                        self.calls += 1
                        return unit(0, 1, 0)

                h = self.harness(BlockingGuest())
                c, e, _ = self.controller(h)
                c.duck(False)
                h.shadow._labels["Speaker A"] = LabelState(
                    state="bound", identity="owner", seconds=4, total=[4.0, 0.0, 0.0]
                )
                h.speak(20)
                self.assertTrue(entered.wait(1))
                c._received_ms = 20000
                c._received_bytes = 20000 * 32
                c.duck(True)
                e.drain()
                start = 20000
                if fragment == "privacy off":
                    self.hear(c, "Unit", start=start)
                    self.assertEqual(e._private_name_end, 21000)
                    start += 1000
                epoch = h.shadow.epoch
                commit = e._private

                def revoke(*args):
                    release.set()
                    wait_for(lambda: h.shadow.binding("Speaker A") == ("bound", "guest"))
                    return commit(*args)

                try:
                    with (
                        patch.object(e, "_private", side_effect=revoke),
                        patch.object(h.shadow, "exit_score", wraps=h.shadow.exit_score) as score,
                    ):
                        self.hear(c, fragment, start=start)
                        score.assert_not_called()  # Only the cached binding had approved it.
                finally:
                    release.set()
                self.assertEqual(h.shadow.epoch, epoch)
                self.assertIsNone(e._private_name_end)
                self.assertNotIn("exit", [event.get("action") for event in e.drain()])
                self.assertFalse(c._private_sentence.pcm)

    def test_binding_approval_requires_the_same_binding_and_known_origin_at_commit(self):
        for change in ("unchanged", "unbound", "guest", "tentative", "other_owner", "origin"):
            for fragment in ("Unit", "privacy off", EXIT):
                with self.subTest(change=change, fragment=fragment):
                    h = self.harness()
                    c, e, roles = self.controller(h)
                    h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                    roles.priority = replace(OWNERS, owners=("owner", "other_owner"))
                    if fragment == "privacy off":
                        self.hear(c, "Unit")
                    prepare = e.prepare_transcript

                    def changed(*args, **kwargs):
                        ticket = prepare(*args, **kwargs)
                        if change == "origin":
                            return replace(ticket, exit_origin=None)
                        if change == "unbound":
                            h.shadow._labels.clear()
                        elif change in {"guest", "other_owner"}:
                            h.shadow._labels["Speaker A"] = LabelState(
                                state="bound", identity=change
                            )
                        elif change == "tentative":
                            h.shadow._labels["Speaker A"] = LabelState(
                                state="tentative", identity="owner"
                            )
                        return ticket

                    with patch.object(e, "prepare_transcript", side_effect=changed):
                        self.hear(c, fragment, start=1000)
                    passed = change == "unchanged"
                    self.assertEqual(
                        e._private_name_end, 2000 if passed and fragment == "Unit" else None
                    )
                    self.assertEqual(
                        [event.get("action") for event in e.drain()],
                        ["exit"] if passed and fragment != "Unit" else [],
                    )

    def test_isolated_voice_result_is_not_replaced_by_a_new_binding(self):
        for score, expected in ((0.9, ["exit"]), (0.0, [])):
            for labelled in (True, False):
                with self.subTest(score=score, labelled=labelled):
                    h = self.harness()
                    c, e, _ = self.controller(h)
                    h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="guest")
                    candidate = authored_turn(
                        turn(0, 1000, "Speaker A" if labelled else None), EXIT
                    )
                    c._received_ms = 1000
                    c._received_bytes = 32000
                    c._private_turn_audio(c._generation, candidate, memoryview(tone(1)))

                    def scored(*_):
                        h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                        return EntryScore(score, None, 0, "scored")

                    with patch.object(h.shadow, "exit_score", side_effect=scored) as check:
                        c._turn(
                            c._generation, queue.Queue(), TranscriptMemory(), h.shadow, candidate
                        )
                        check.assert_called_once()
                    self.assertEqual([event.get("action") for event in e.drain()], expected)

    def unrelated_binding_during_approval(self, *, entry):
        for change in ("bind", "unbind"):
            with self.subTest(change=change):
                vector = unit(0, 1, 0) if change == "bind" else unit(0, 0, 1)
                h = self.harness(ScriptedEmbedder([vector] * 8))
                c, e, _ = self.controller(h)
                h.shadow._labels["Speaker A"] = LabelState(
                    state="bound", identity="owner", seconds=4
                )
                if change == "unbind":
                    h.shadow._labels["Speaker B"] = LabelState(state="bound", identity="guest")
                before = h.shadow.binding_snapshot("Speaker A")
                other_before = h.shadow.binding_snapshot("Speaker B")
                if entry:
                    c.duck(False)
                    e.drain()
                    c._observe_spans(c._generation, "live-40000", ((40000, 41000),))
                prepare = e.prepare_transcript

                def unrelated(*args, **kwargs):
                    ticket = prepare(*args, **kwargs)
                    self.assertTrue(ticket.entry if entry else ticket.exit[0])
                    h.speak(4, speaker="Speaker B")
                    wait_for(lambda: h.shadow.scored == 1)
                    self.assertEqual(
                        h.shadow.binding("Speaker B")[0], "bound" if change == "bind" else "unknown"
                    )
                    return ticket

                with (
                    patch.object(e, "prepare_transcript", side_effect=unrelated),
                    patch.object(
                        h.shadow, "entry_score", return_value=EntryScore(0.9, None, 0, "scored")
                    ),
                ):
                    self.hear(c, "privacy mode" if entry else EXIT, start=40000, inferred=entry)
                events = e.drain()
                if entry:
                    self.assertEqual(len(events), 1)
                    self.assertEqual(events[0]["turn"]["role"], "owner")
                    self.assertEqual(events[0].get("role_source"), "voiceprint_label")
                else:
                    self.assertEqual([event.get("action") for event in events], ["exit"])
                self.assertEqual(h.shadow.binding_snapshot("Speaker A"), before)
                self.assertNotEqual(h.shadow.binding_snapshot("Speaker B"), other_before)

    def test_unrelated_binding_changes_preserve_owner_privacy_entry(self):
        self.unrelated_binding_during_approval(entry=True)

    def test_unrelated_binding_changes_preserve_owner_privacy_exit(self):
        self.unrelated_binding_during_approval(entry=False)

    def test_binding_check_is_per_speaker_both_directions(self):
        class PublicAndPrivateEmbedder(ScriptedEmbedder):
            def embed_private(self, pcm):
                return ToneEmbedder().embed(pcm)

        for entry in (True, False):
            changes = ("bind", "unbind") if entry else ("bind", "owner_stays")
            phrases = (
                (("privacy mode",),)
                if entry
                else ((EXIT,), ("Unit", "privacy off"), ("privacy off",))
            )
            for change in changes:
                for words in phrases:
                    with self.subTest(entry=entry, change=change, words=words):
                        vector = unit(0, 1, 0) if entry else unit(1, 0, 0)
                        if change == "unbind":
                            vector = unit(0, 0, 1)
                        h = self.harness(PublicAndPrivateEmbedder([vector] * 8))
                        c, e, _ = self.controller(h)
                        speaker, other = (
                            ("Speaker A", "Speaker B") if entry else ("Speaker B", "Speaker A")
                        )
                        h.shadow._labels[speaker] = LabelState(
                            state="bound", identity="owner" if entry else "guest", seconds=4
                        )
                        if change != "bind":
                            h.shadow._labels[other] = LabelState(
                                state="bound", identity="guest" if entry else "owner", seconds=4
                            )
                        if entry:
                            c.duck(False)
                            e.drain()
                            c._observe_spans(c._generation, "live-40000", ((40000, 41000),))
                        elif words == ("privacy off",):
                            # Also check the guest suffix with a valid owner prefix,
                            # so its own approval must refuse independently of the name.
                            self.hear(c, "Unit", start=39000)
                            self.assertEqual(e._private_name_end, 40000)
                        prepare = e.prepare_transcript

                        def unrelated(*args, **kwargs):
                            ticket = prepare(*args, **kwargs)
                            if entry:
                                self.assertTrue(ticket.entry)
                            elif ticket.exit is not None:
                                self.assertFalse(ticket.exit[0])
                            scored = h.shadow.scored
                            h.speak(4, speaker=other)
                            wait_for(lambda: h.shadow.scored == scored + 1)
                            self.assertEqual(
                                h.shadow.binding(other),
                                ("unknown", "guest")
                                if change == "unbind"
                                else ("bound", "guest" if entry else "owner"),
                            )
                            return ticket

                        with (
                            patch.object(e, "prepare_transcript", side_effect=unrelated),
                            patch.object(
                                h.shadow,
                                "entry_score",
                                return_value=EntryScore(0.9, None, 0, "scored"),
                            ),
                        ):
                            for index, text in enumerate(words):
                                start = 40000 + index * 1000
                                if entry:
                                    self.hear(c, text, start=start, inferred=True)
                                else:
                                    candidate = authored_turn(
                                        turn(start, start + 1000, speaker), text
                                    )
                                    c._received_ms = candidate.end_ms
                                    c._received_bytes = candidate.end_ms * 32
                                    c._private_turn_audio(
                                        c._generation, candidate, memoryview(tone(1, GUEST_F0))
                                    )
                                    c._turn(
                                        c._generation,
                                        queue.Queue(),
                                        TranscriptMemory(),
                                        h.shadow,
                                        candidate,
                                    )
                                    self.assertNotIn(
                                        "exit", [event.get("action") for event in e.drain()]
                                    )
                                    self.assertTrue(e.ducked)
                                    self.assertIsNone(e._private_name_end)
                        if entry:
                            events = e.drain()
                            self.assertEqual(len(events), 1)
                            self.assertEqual(events[0]["turn"]["role"], "owner")
                            self.assertEqual(events[0].get("role_source"), "voiceprint_label")
                        self.assertEqual(
                            h.shadow.binding(speaker), ("bound", "owner" if entry else "guest")
                        )

    def test_own_binding_revocation_drops_entry_authority(self):
        h = self.harness(ScriptedEmbedder([unit(0, 1, 0)] * 8))
        c, e, _ = self.controller(h)
        c.duck(False)
        e.drain()
        h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
        c._observe_spans(c._generation, "live-40000", ((40000, 41000),))
        prepare = e.prepare_transcript

        def revoked(*args, **kwargs):
            ticket = prepare(*args, **kwargs)
            self.assertTrue(ticket.entry)
            h.speak(4)
            wait_for(lambda: h.shadow.scored == 1)
            self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "guest"))
            return ticket

        with (
            patch.object(e, "prepare_transcript", side_effect=revoked),
            patch.object(h.shadow, "entry_score", return_value=EntryScore(0.9, None, 0, "scored")),
        ):
            self.hear(c, "privacy mode", start=40000, inferred=True)
        events = e.drain()
        self.assertEqual(len(events), 1)
        self.assertNotIn("role_source", events[0])

    def test_binding_exit_without_a_revision_fails_closed(self):
        for missing in ("adapter", "ticket"):
            with self.subTest(missing=missing):
                h = self.harness()
                c, e, _ = self.controller(h)
                h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                prepare = e.prepare_transcript

                def no_revision(*args, **kwargs):
                    ticket = prepare(*args, **kwargs)
                    self.assertEqual(ticket.exit_origin, "binding")
                    return replace(
                        ticket, binding=(*ticket.binding[:3], (None, ticket.binding[3][1]))
                    )

                with (
                    patch.object(
                        h.shadow,
                        "binding_snapshot",
                        None if missing == "adapter" else h.shadow.binding_snapshot,
                    ),
                    patch.object(
                        e,
                        "prepare_transcript",
                        side_effect=no_revision if missing == "ticket" else prepare,
                    ),
                ):
                    self.hear(c)
                self.assertEqual(e.drain(), [])

    def test_binding_revision_invalidates_rebound_exit_and_entry_evidence(self):
        # Use the actual worker to change owner -> guest -> owner. Comparing only the
        # final binding tuple would wrongly accept the evidence from before both changes.
        for entry in (False, True):
            with self.subTest(entry=entry):

                class SwitchingEmbedder(ScriptedEmbedder):
                    vector = unit(0, 1, 0)

                    def embed(self, pcm):
                        self.calls += 1
                        return self.vector

                embedder = SwitchingEmbedder()
                h = self.harness(embedder)
                c, e, _ = self.controller(h)
                h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                before = h.shadow.binding_snapshot("Speaker A")
                if entry:
                    c.duck(False)
                    e.drain()
                    c._observe_spans(c._generation, "live-40000", ((40000, 41000),))
                prepare = e.prepare_transcript

                def rebound(*args, **kwargs):
                    ticket = prepare(*args, **kwargs)
                    h.speak(20)
                    wait_for(lambda: h.shadow.scored == 1)
                    self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "guest"))
                    # Remove accumulated evidence so the second public score can rebind.
                    h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="guest")
                    embedder.vector = unit(1, 0, 0)
                    h.speak(20)
                    wait_for(lambda: h.shadow.scored == 2)
                    self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "owner"))
                    return ticket

                with (
                    patch.object(e, "prepare_transcript", side_effect=rebound),
                    patch.object(
                        h.shadow, "entry_score", return_value=EntryScore(0.9, None, 0, "scored")
                    ),
                ):
                    self.hear(c, "privacy mode" if entry else EXIT, start=40000, inferred=entry)
                self.assertGreater(h.shadow.binding_snapshot("Speaker A")[0], before[0])
                self.assertEqual(h.shadow.epoch, 0)
                events = e.drain()
                self.assertNotIn("exit", [event.get("action") for event in events])
                if entry:
                    self.assertEqual(len(events), 1)
                    self.assertEqual(events[0]["turn"]["role"], "owner")
                    self.assertNotIn("role_source", events[0])

    def test_more_public_evidence_for_the_same_owner_does_not_revoke_an_exit(self):
        h = self.harness(ScriptedEmbedder())
        c, e, _ = self.controller(h)
        h.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner", seconds=4)
        before = h.shadow.binding_snapshot("Speaker A")
        prepare = e.prepare_transcript

        def more_evidence(*args, **kwargs):
            ticket = prepare(*args, **kwargs)
            h.speak(4)
            wait_for(lambda: h.shadow.scored == 1)
            return ticket

        with patch.object(e, "prepare_transcript", side_effect=more_evidence):
            self.hear(c)
        self.assertEqual(h.shadow.binding_snapshot("Speaker A"), before)
        self.assertEqual([event.get("action") for event in e.drain()], ["exit"])
        h.shadow.forget()
        self.assertGreater(h.shadow.binding_snapshot("Speaker A")[0], before[0])

    def test_stop_cancels_promptly_during_split_name_native_inference(self):
        # Exercise both split fragments and the complete exit through the real worker.
        for fragment in ("Unit", "privacy off", EXIT, "privacy mode"):
            with self.subTest(fragment=fragment):
                entered, release, stopped = (threading.Event() for _ in range(3))

                class BlockingEmbedder(ScriptedEmbedder):
                    def embed(self, pcm):
                        entered.set()
                        return super().embed(pcm)

                embedder = BlockingEmbedder([unit(1, 0, 0)], gate=release)
                harness = self.harness(embedder)
                c, e, _ = self.controller(harness)
                if fragment == "privacy off":
                    # The name fragment was already independently authenticated.
                    with patch.object(harness.shadow, "exit_score") as score:
                        score.return_value = EntryScore(0.9, None, 0, "scored")
                        self.hear(c, "Unit")
                        score.assert_called_once()
                if fragment == "privacy mode":
                    c.duck(False)
                    harness.shadow._labels["Speaker A"] = LabelState(
                        state="bound", identity="owner", seconds=4
                    )
                    harness.shadow.audio(tone(2))
                    c._observe_spans(c._generation, "live-1000", ((1000, 2000),))
                errors = []

                def hear():
                    try:
                        self.hear(c, fragment, start=1000, inferred=fragment == "privacy mode")
                    except Exception as error:
                        errors.append(error)

                def stop():
                    try:
                        c.stop()
                    except Exception as error:
                        errors.append(error)
                    finally:
                        stopped.set()

                audio, control = threading.Thread(target=hear), threading.Thread(target=stop)
                audio.start()
                try:
                    self.assertTrue(entered.wait(1), "native stand-in did not start")
                    borrowed = c._private_sentence.pcm
                    control.start()
                    self.assertTrue(stopped.wait(0.2), "Stop waited for native inference")
                    self.assertTrue(c._stop.is_set())
                    self.assertTrue(c._decision_cancel.is_set())
                    self.assertFalse(any(borrowed))
                    self.assertFalse(c._private_sentence.pcm)
                finally:
                    release.set()
                    for worker in (audio, control):
                        if worker.ident is not None:
                            worker.join(3)
                self.assertTrue(all(not worker.is_alive() for worker in (audio, control)))
                self.assertEqual(errors, [])
                self.assertNotIn("exit", [event.get("action") for event in e.drain()])

    def test_inflight_name_check_cannot_authorize_a_new_privacy_generation(self):
        for change in ("cycle", "redundant_on", "forget", "playback"):
            with self.subTest(change=change):
                harness = self.harness()
                c, e, _ = self.controller(harness)
                entered, release = threading.Event(), threading.Event()
                errors = []

                def score(*_):
                    entered.set()
                    if not release.wait(3):
                        raise AssertionError("test did not release voice check")
                    return EntryScore(0.9, None, 0, "scored")

                def hear():
                    try:
                        self.hear(c, "Unit")
                    except Exception as error:
                        errors.append(error)

                audio = threading.Thread(target=hear)
                with patch.object(harness.shadow, "exit_score", side_effect=score):
                    audio.start()
                    try:
                        self.assertTrue(entered.wait(1))
                        if change == "cycle":
                            c.duck(False)
                            c.duck(True)
                        elif change == "redundant_on":
                            c.duck(True)
                        elif change == "forget":
                            c._forget_speakers()
                        else:
                            e.reply("started", now_ms=1000)
                    finally:
                        release.set()
                        audio.join(3)
                self.assertFalse(audio.is_alive())
                self.assertEqual(errors, [])
                self.assertIsNone(e._private_name_end)
                self.hear(c, "privacy off", start=1000)
                self.assertNotIn("exit", [event.get("action") for event in e.drain()])

    def test_replacement_zeroes_the_old_object_and_holds_only_the_latest_turn(self):
        buffer = PrivateTurnAudio()
        first, second = turn(0, 1000), turn(1000, 2000)
        buffer.replace(first, memoryview(tone(1)))
        old = buffer.pcm
        buffer.replace(second, memoryview(tone(1, OWNER_F0)))
        self.assertTrue(old)
        self.assertFalse(any(old))
        self.assertIsNone(buffer.for_turn(first))
        self.assertIs(buffer.for_turn(second), buffer.pcm)
        self.assertEqual(len(buffer.pcm), 32000)
        latest = buffer.pcm
        buffer.clear()
        self.assertFalse(any(latest))
        self.assertEqual(buffer.pcm, bytearray())

    def test_cap_and_incomplete_spans_retain_nothing(self):
        buffer = PrivateTurnAudio()
        buffer.replace(turn(0, 10000), bytearray(PRIVATE_TURN_MAX_BYTES))
        self.assertEqual(len(buffer.pcm), PRIVATE_TURN_MAX_BYTES)
        old = buffer.pcm
        buffer.replace(turn(0, 10001), bytearray(PRIVATE_TURN_MAX_BYTES + 32))
        self.assertFalse(any(old))
        self.assertFalse(buffer.pcm)
        buffer.replace(turn(0, 1000), bytearray(31998))
        self.assertFalse(buffer.pcm)

    def test_start_private_owner_passes_and_guest_refuses_without_binding(self):
        for f0, allowed in ((OWNER_F0, True), (GUEST_F0, False)):
            with self.subTest(owner=allowed):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, f0=f0)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(
                    [e.get("action") for e in events.drain()], ["exit"] if allowed else []
                )
                self.assertEqual(
                    events._private_counts["exit_refused_voice_below_threshold"], int(not allowed)
                )
                self.assertEqual(harness.shadow._labels, {})
                self.assertEqual(harness.shadow._bound_ids, set())
                self.assertEqual(harness.shadow.scored_seconds, 0)
                self.assertEqual(harness.shadow.offered, 0)
                self.assertTrue(events.ducked)  # Only the host applies the off.

    def test_non_exit_status_and_unnamed_remainders_are_never_scored(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(harness.shadow, "exit_score", wraps=harness.shadow.exit_score) as score:
            turns = ("synthetic token", "status", "are you recording?", "privacy off")
            for index, text in enumerate(turns):
                old = self.hear(controller, text, start=index * 1000)
                # The matcher has ruled out an exit. Waiting for another turn to wipe
                # this one would keep unrelated private speech indefinitely in silence.
                self.assertEqual(len(old), 32000)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
            score.assert_not_called()
        self.assertEqual([event.get("action") for event in events.drain()], ["status"])
        self.assertEqual(harness.embedder.calls, 0)
        self.assertEqual(harness.shadow._labels, {})

    def test_exit_voice_never_rebinds_or_accumulates_into_existing_labels(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        previous = LabelState(state="bound", identity="guest", seconds=3, turns=2)
        harness.shadow._labels["Speaker A"] = previous
        harness.shadow._bound_ids = {"guest"}
        harness.shadow.scored_seconds = 3
        self.hear(controller, "synthetic token")
        self.hear(controller, start=1000)
        self.assertEqual([event.get("action") for event in events.drain()], ["exit"])
        self.assertIs(harness.shadow._labels["Speaker A"], previous)
        self.assertEqual(harness.shadow._bound_ids, {"guest"})
        self.assertEqual(harness.shadow.scored_seconds, 3)
        self.assertEqual(harness.shadow.offered, 0)

    def test_classifying_another_turn_does_not_wipe_the_candidate_being_verified(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        candidate = authored_turn(turn(1000, 2000), EXIT)
        controller._private_turn_audio(
            controller._generation, candidate, memoryview(tone(1, OWNER_F0))
        )
        old = controller._private_sentence.pcm
        score = harness.shadow.exit_score

        def score_after_non_exit(pcm, owners):
            events.prepare_transcript(replace(turn(0, 1000), text="synthetic token"))
            return score(pcm, owners)

        with patch.object(harness.shadow, "exit_score", side_effect=score_after_non_exit):
            checks = events.prepare_transcript(candidate)
        self.assertFalse(any(old))
        self.assertFalse(controller._private_sentence.pcm)
        events.transcript(candidate, now_ms=2000, voice_checks=checks)
        self.assertEqual([event.get("action") for event in events.drain()], ["exit"])

    def test_wipes_on_unavailable_short_and_embedder_exception(self):
        for embedder, seconds in ((ScriptedEmbedder(fail_on=1), 1), (ToneEmbedder(), 0.2)):
            with self.subTest(seconds=seconds):
                harness = self.harness(embedder)
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, seconds=seconds)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(events.drain(), [])
                self.assertEqual(events._private_counts["exit_refused_voice_unavailable"], 1)
                self.assertNotIn("synthetic embedder failure", " ".join(harness.lines))

    def test_exception_from_exit_scorer_is_sanitized_and_wiped(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(
            harness.shadow, "exit_score", side_effect=RuntimeError("synthetic token")
        ):
            old = self.hear(controller)
        self.assertFalse(any(old))
        self.assertEqual(events._private_counts["exit_refused_voice_unavailable"], 1)
        self.assertNotIn("synthetic token", " ".join(harness.lines))

    def test_oversize_and_missing_audio_refuse_without_scoring(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(harness.shadow, "exit_score") as score:
            old = self.hear(controller, seconds=10.001)
            self.assertFalse(old)
            current = replace(turn(11000, 12000), text=EXIT)
            controller._turn(
                controller._generation, queue.Queue(), TranscriptMemory(), harness.shadow, current
            )
            score.assert_not_called()
        self.assertEqual(events.drain(), [])
        self.assertEqual(events._private_counts["exit_refused_voice_unavailable"], 2)

    def test_matcher_and_event_delivery_errors_leave_no_pcm_or_exception_content(self):
        for failure in ("matcher", "delivery"):
            with self.subTest(failure=failure):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                old = self.hear(controller, "synthetic token")
                current = authored_turn(turn(1000, 2000), EXIT)
                controller._private_turn_audio(
                    controller._generation, current, memoryview(tone(1, OWNER_F0))
                )
                self.assertFalse(any(old))
                candidate = controller._private_sentence.pcm
                target = "rightyo.tool_events.privacy_phrase" if failure == "matcher" else None
                manager = (
                    patch(target, side_effect=RuntimeError("synthetic token"))
                    if target
                    else patch.object(events, "_emit", side_effect=RuntimeError("synthetic token"))
                )
                with manager:
                    if failure == "matcher":
                        events.transcript(current, now_ms=2000)
                    else:
                        with self.assertRaises(RuntimeError):
                            events.transcript(current, now_ms=2000)
                self.assertFalse(any(candidate))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertNotIn("synthetic token", " ".join(harness.lines))

    def test_binding_pass_inferred_and_playback_prechecks_wipe_without_scoring(self):
        for outcome in ("bound", "inferred", "playback"):
            with self.subTest(outcome=outcome):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                if outcome == "bound":
                    harness.shadow._labels["Speaker A"] = LabelState(
                        state="bound", identity="owner"
                    )
                if outcome == "playback":
                    events._playbacks.append([0, 1000])
                with patch.object(harness.shadow, "exit_score") as score:
                    old = self.hear(controller, inferred=outcome == "inferred")
                    score.assert_not_called()
                self.assertFalse(any(old))
                self.assertEqual(
                    [e.get("action") for e in events.drain()],
                    ["exit"] if outcome == "bound" else [],
                )
                if outcome != "bound":
                    self.assertEqual(events._private_counts[f"exit_refused_{outcome}"], 1)

    def test_preexisting_playback_refuses_before_retaining_audio_for_confirmation(self):
        c, e, _ = self.controller(self.harness())
        schedule = Mock()
        current = replace(
            authored_turn(turn(0, 1000), EXIT),
            acoustic_span_ms=(0, 1000),
            exit_available=lambda: True,
            defer_exit=schedule,
        )
        c._private_turn_audio(c._generation, current, memoryview(tone(1, OWNER_F0)))
        e.reply("started", now_ms=500)
        with patch.object(c._shadow, "exit_score") as score:
            e.transcript(current, now_ms=1000)
            schedule.assert_not_called()
            score.assert_not_called()
        self.assertFalse(c._private_sentence.pcm)
        self.assertEqual(e._private_counts["exit_refused_playback"], 1)

    def test_wipes_on_privacy_off_stop_session_end_and_forget(self):
        for action in ("off", "stop", "end", "forget", "renew"):
            with self.subTest(action=action):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                # Interrupt a pending exit before the matcher or scorer wipes its audio.
                candidate = replace(turn(0, 1000), text=EXIT)
                controller._private_turn_audio(
                    controller._generation, candidate, memoryview(tone(1))
                )
                old = controller._private_sentence.pcm
                self.assertTrue(any(old))
                if action == "off":
                    controller.duck(False)
                elif action == "stop":
                    controller.stop()
                elif action == "end":
                    events.end(phase="stopped", now_ms=1000)
                elif action == "renew":
                    controller._speakers_renewed()
                else:
                    controller._forget_speakers()
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)

    def test_privacy_on_mid_turn_never_copies_public_prefix(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        controller.duck(False)
        controller._received_bytes, controller._received_ms = 16000, 500
        controller.duck(True)
        events.drain()
        old = self.hear(controller)
        self.assertFalse(old)
        self.assertEqual(events.drain(), [])
        self.assertEqual(harness.embedder.calls, 0)
        # A turn wholly after the precise on-edge can authenticate normally.
        self.hear(controller, start=1000)
        self.assertEqual([e.get("action") for e in events.drain()], ["exit"])

    def test_privacy_on_never_publishes_a_held_public_continuation(self):
        from test_turn_merge import SILENCE, VOICE, ScriptedDiarizer, ScriptedTranscriber, audio

        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        controller.duck(False)
        events.drain()
        generation = controller._generation
        work, memory, retained, turns = queue.Queue(), TranscriptMemory(), [], []

        def private_audio(current, pcm):
            controller._private_turn_audio(generation, current, pcm)
            retained.append(bytes(controller._private_sentence.pcm))

        def accept(current):
            turns.append(current)
            controller._turn(generation, work, memory, harness.shadow, current)

        processor = LiveProcessor(
            LiveConfig(
                "shadow-test",
                provenance="causal-replay",
                transcriber=ScriptedTranscriber(("Synthetic public prefix", "synthetic tail")),
                diarizer=ScriptedDiarizer(),
                private=controller._heard_private_span,
                silero_vad=FixtureVad(),
                on_private_turn_audio=private_audio,
                turn_merge_gap_ms=2000,
            ),
            accept,
        )
        self.addCleanup(processor.close)

        def feed(pcm):
            for at in range(0, len(pcm), 6400):
                controller._feed(generation, processor, pcm[at : at + 6400], "stdin")

        feed(audio((VOICE, 200), (SILENCE, 1500)))
        self.assertTrue(processor._merger.holding)
        self.assertEqual(turns, [])
        controller.duck(True)
        events.drain()
        feed(audio((VOICE, 200), (SILENCE, 1500)))
        processor.finish()
        # Main (efbbef9) joins these same-speaker pieces and suppresses the whole turn.
        # Discarding the prefix preserves that public outcome and isolates the exit PCM.
        self.assertEqual([current.text for current in turns], ["synthetic tail"])
        self.assertEqual(events.drain(), [])
        self.assertEqual(memory.snapshot(10_000)["turns"], [])
        self.assertEqual(work.qsize(), 0)
        self.assertEqual(retained, [audio((VOICE, 200))])

    def test_name_fragment_uses_binding_or_its_own_voice_and_zeroes_audio(self):
        for identity in (None, "guest", "owner"):
            with self.subTest(binding=identity):
                harness = self.harness()
                controller, events, roles = self.controller(
                    harness, addressing=Addressing(names=("Jarvis",))
                )
                if identity is not None:
                    harness.shadow._labels["Speaker A"] = LabelState(
                        state="bound", identity=identity
                    )
                with (
                    patch.object(roles, "exit_voice", wraps=roles.exit_voice) as voice,
                    patch.object(
                        harness.shadow, "exit_score", wraps=harness.shadow.exit_score
                    ) as score,
                ):
                    old = self.hear(
                        controller, "Jarvis", f0=GUEST_F0 if identity == "guest" else OWNER_F0
                    )
                    self.assertEqual(voice.call_count, int(identity != "owner"))
                    self.assertEqual(score.call_count, int(identity != "owner"))
                self.assertTrue(old)
                self.assertFalse(any(old))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(events.drain(), [])
                self.assertEqual(harness.embedder.calls, int(identity != "owner"))
                self.assertEqual(events._private_name_end, None if identity == "guest" else 1000)
                self.assertEqual(
                    events._private_counts["exit_name_refused"], int(identity == "guest")
                )

    def test_really_question_owner_cannot_authenticate_guest_unit(self):
        owner_question = tone(2, OWNER_F0)
        for name_hz, gain, allowed in (
            (GUEST_F0, 1, False),
            (GUEST_F0, 0.01, False),
            (GUEST_F0, 0, False),
            (OWNER_F0, 1, True),
        ):
            with self.subTest(name_hz=name_hz, gain=gain):
                name = array("h", (round(x * gain) for x in array("h", tone(0.8, name_hz))))
                harness = self.harness()
                c, events, _ = self.controller(harness)
                asr = Units()
                asr.transcribe = Mock(
                    side_effect=[
                        Units(("Really?", 20, 2020), (" Unit,", 2020, 2820)).units,
                        Units(("Privacy off.", 20, 980)).units,
                    ]
                )
                speech = [(320, 3120 if gain else 2320), (4800, 5760)]
                processor = LiveProcessor(
                    LiveConfig(
                        "shadow-test",
                        transcriber=asr,
                        diarizer=Timeline(),
                        pre_roll_ms=20,
                        private=c._heard_private_span,
                        silero_vad=fake_worker(self, speech),
                        on_private_turn_audio=lambda t, pcm: c._private_turn_audio(
                            c._generation, t, pcm
                        ),
                    ),
                    lambda t: c._turn(c._generation, work, memory, harness.shadow, t),
                )
                self.addCleanup(processor.close)
                work, memory = queue.Queue(), TranscriptMemory()

                def feed(pcm):
                    for at in range(0, len(pcm), 6400):
                        c._feed(
                            c._generation, processor, pcm[at : at + 6400], "demo", harness.shadow
                        )

                feed(bytes(320 * 32) + owner_question + name.tobytes() + bytes(1680 * 32))
                self.assertEqual(events._private_name_end is not None, allowed)
                self.assertEqual(harness.embedder.frames["owner"], 40 if allowed else 0)
                feed(tone(0.96, OWNER_F0) + bytes(2400 * 32))
                processor.finish()
                self.assertEqual(asr.transcribe.call_count, 2)
                self.assertEqual([e["action"] for e in events.drain()], ["exit"] if allowed else [])
                self.assertFalse(c._private_sentence.pcm)

    def test_single_recognition_guest_name_and_owner_phrase(self):
        for hz, gain, timed, allowed in (
            (OWNER_F0, 1, True, True),
            (GUEST_F0, 1, True, False),
            (GUEST_F0, 0.01, True, False),
            (GUEST_F0, 0, True, False),
            (OWNER_F0, 1, False, False),
        ):
            with self.subTest(hz=hz, gain=gain, name_timed=timed):
                h = self.harness()
                c, e, _ = self.controller(h)
                name = array("h", (round(x * gain) for x in array("h", tone(0.8, hz)))).tobytes()
                pcm = bytes(320 * 32) + name + tone(1.6, OWNER_F0) + bytes(2400 * 32)
                # Even a positive VAD classification cannot give a silent name
                # the owner's phrase. A shared timestamp cannot locate the name.
                units = (
                    Units(("Unit,", 20, 820), (" privacy off", 820, 2420))
                    if timed
                    else Units((EXIT, 20, 2420))
                )
                work, memory = queue.Queue(), TranscriptMemory()
                p = LiveProcessor(
                    LiveConfig(
                        "shadow-test",
                        transcriber=units,
                        diarizer=Timeline(),
                        pre_roll_ms=20,
                        private=c._heard_private_span,
                        silero_vad=fake_worker(self, [(320, 2720)]),
                        on_private_turn_audio=lambda t, data: c._private_turn_audio(
                            c._generation, t, data
                        ),
                    ),
                    lambda t: c._turn(c._generation, work, memory, h.shadow, t),
                )
                self.addCleanup(p.close)
                score = h.shadow.exit_score

                isolated = []

                def verify(data, owners):
                    isolated.append(bytes(data) == name)
                    return score(data, owners)

                with patch.object(h.shadow, "exit_score", side_effect=verify) as check:
                    for at in range(0, len(pcm), 6400):
                        c._feed(c._generation, p, pcm[at : at + 6400], "demo", h.shadow)
                    p.finish()
                self.assertEqual(p._utterances, 1)
                self.assertEqual(check.call_count, int(timed))
                self.assertEqual(isolated, [True] if timed else [])
                self.assertEqual(
                    [event["action"] for event in e.drain()], ["exit"] if allowed else []
                )
                self.assertEqual(memory.snapshot(6000)["turns"], [])
                self.assertTrue(work.empty())
                self.assertFalse(c._private_sentence.pcm)

    def test_multiword_name_keeps_only_its_timed_tokens(self):
        c, e, _ = self.controller(self.harness(), addressing=Addressing(("Alpha Unit",)))
        current = replace(
            turn(0, 2000),
            text="Alpha Unit, privacy off",
            exit_timings=((0, 5, 0, 400), (6, 10, 800, 1200), (12, 23, 1200, 2000)),
        )
        name = tone(0.4, GUEST_F0)
        pcm = name + tone(0.4, OWNER_F0) + name + tone(0.8, OWNER_F0)
        c._private_turn_audio(c._generation, current, memoryview(pcm))
        received = []

        def score(data, owners):
            received.append(bytes(data) == name * 2)
            return EntryScore(0.1, None, 0, "scored")

        with patch.object(c._shadow, "exit_score", side_effect=score):
            e.transcript(current, now_ms=2000)
        self.assertEqual(received, [True])
        self.assertEqual(e.drain(), [])
        self.assertFalse(c._private_sentence.pcm)

    def test_speaker_epoch_change_between_split_fragments(self):
        h = self.harness()
        c, e, _ = self.controller(h)
        self.hear(c, "Unit", seconds=0.9)
        self.assertEqual(e._private_name_end, 900)
        h.shadow.forget()
        with patch.object(h.shadow, "exit_score", wraps=h.shadow.exit_score) as score:
            self.hear(c, "privacy off", start=1000)
            score.assert_not_called()
        self.assertEqual(e.drain(), [])
        self.assertTrue(e.ducked)

    def test_unbound_owner_split_exit_checks_each_fragments_audio(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        with patch.object(harness.shadow, "exit_score", wraps=harness.shadow.exit_score) as score:
            first = self.hear(controller, "Unit", f0=OWNER_F0)
            self.hear(controller, "privacy off", start=1100, f0=OWNER_F0)
            self.assertEqual(score.call_count, 2)
        self.assertFalse(any(first))
        self.assertEqual(harness.embedder.calls, 2)
        self.assertEqual([event.get("action") for event in events.drain()], ["exit"])
        self.assertEqual(events._private_counts["exit_name_refused"], 0)
        self.assertEqual(events._private_counts["exit_unnamed"], 0)

    def test_split_name_requires_binding_and_scores_only_the_completing_turn(self):
        for f0, allowed in ((OWNER_F0, True), (GUEST_F0, False)):
            with self.subTest(owner=allowed):
                harness = self.harness()
                controller, events, _roles = self.controller(harness)
                harness.shadow._labels["Speaker A"] = LabelState(state="bound", identity="owner")
                with patch.object(harness.shadow, "exit_score") as score:
                    first = self.hear(controller, "Unit", f0=OWNER_F0)
                    score.assert_not_called()
                self.assertFalse(any(first))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(harness.embedder.calls, 0)
                # The accepted fragment needs its binding; the completing turn can
                # authenticate its own audio without that binding still being present.
                harness.shadow._labels.clear()
                with patch.object(
                    harness.shadow, "exit_score", wraps=harness.shadow.exit_score
                ) as score:
                    second = self.hear(controller, "privacy off", start=1100, f0=f0)
                    score.assert_called_once()
                    self.assertIs(score.call_args.args[0], second)
                self.assertEqual(len(second), len(tone(1, f0)))
                self.assertFalse(any(second))
                self.assertFalse(controller._private_sentence.pcm)
                self.assertEqual(
                    [e.get("action") for e in events.drain()], ["exit"] if allowed else []
                )
                self.assertEqual(harness.embedder.calls, 1)

    def test_threshold_and_minimum_match_entry_configuration(self):
        harness = self.harness(ScriptedEmbedder(vectors=([0.6, 0.8, 0],)), entry_threshold=0.7)
        controller, events, _roles = self.controller(harness)
        self.hear(controller)
        self.assertEqual(events.drain(), [])
        self.assertEqual(events._private_counts["exit_refused_voice_below_threshold"], 1)
        self.assertIn("speaker_id exit_check score=0.600", harness.lines)

    def test_worker_timeout_cancels_queued_audio_before_a_late_score(self):
        gate = threading.Event()
        harness = Harness(ScriptedEmbedder(gate=gate))
        harness.shadow.start()
        self.addCleanup(harness.shadow.close)
        wait_for(lambda: bool(harness.shadow._compared))
        harness.speak(1)
        wait_for(lambda: harness.embedder.calls == 1)
        pcm = bytearray(tone(1))
        with patch("rightyo.live_speaker_id.ENTRY_CHECK_TIMEOUT_S", 0.01):
            result = harness.shadow.exit_score(pcm, ("owner",))
        self.assertEqual(result.outcome, "timeout")
        self.assertFalse(any(pcm))
        gate.set()
        wait_for(lambda: harness.shadow.scored == 1)
        harness.shadow.close(drain=True)
        self.assertEqual(harness.embedder.calls, 1)

    def test_private_embedding_views_are_wiped_even_on_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                views = []
                embedder = ScriptedEmbedder()

                def embed(pcm):
                    self.assertIsInstance(pcm, memoryview)
                    self.assertFalse(pcm.readonly)
                    views.append(pcm)
                    if fail:
                        raise ValueError("synthetic token")
                    return [1.0, 0.0, 0.0]

                embedder.embed_private = embed
                harness = self.harness(embedder)
                pcm = bytearray(tone(4))
                result = harness.shadow.exit_score(pcm, ("owner",))
                self.assertEqual(result.outcome, "unavailable" if fail else "scored")
                self.assertFalse(any(pcm))
                self.assertTrue(views)
                self.assertTrue(all(not any(view) for view in views))

    def test_inflight_timeout_finishes_and_wipes_before_return_without_more_windows(self):
        gate, started = threading.Event(), threading.Event()
        views = []
        embedder = ScriptedEmbedder()

        def embed(pcm):
            views.append(pcm)
            started.set()
            gate.wait(3)
            return [1.0, 0.0, 0.0]

        embedder.embed_private = embed
        harness = self.harness(embedder)
        pcm, results = bytearray(tone(8)), []
        with patch("rightyo.live_speaker_id.ENTRY_CHECK_TIMEOUT_S", 0.2):
            caller = threading.Thread(
                target=lambda: results.append(harness.shadow.exit_score(pcm, ("owner",)))
            )
            caller.start()
            self.assertTrue(started.wait(1))
            caller.join(0.25)
            self.assertTrue(caller.is_alive())  # Cannot release an actively borrowed view.
            gate.set()
            caller.join(2)
        self.assertFalse(caller.is_alive())
        self.assertEqual(results[0].outcome, "timeout")
        self.assertEqual(len(views), 1)
        self.assertFalse(any(pcm))
        self.assertFalse(any(views[0]))

    def test_live_processor_retains_only_exact_final_turn_and_ring_stays_silent(self):
        harness = self.harness()
        controller, events, _roles = self.controller(harness)
        generation = controller._generation
        work, memory = queue.Queue(), TranscriptMemory()
        retained = []

        def audio(current, pcm):
            controller._private_turn_audio(generation, current, pcm)
            retained.append(controller._private_sentence.pcm)

        recognizer = Units()
        recognizer.transcribe = Mock(
            side_effect=[
                Units(("synthetic token", 200, 600)).units,
                Units(("Unit,", 0, 750), (" privacy off", 750, 1000)).units,
            ]
        )
        processor = LiveProcessor(
            LiveConfig(
                "shadow-test",
                transcriber=recognizer,
                diarizer=Timeline((200, 600, 1), (2040, 3040, 2)),
                private=controller._heard_private_span,
                silero_vad=FixtureVad(),
                on_private_turn_audio=audio,
                turn_merge_gap_ms=2000,
                reply_wait_ms=2000,
            ),
            lambda current: controller._turn(generation, work, memory, harness.shadow, current),
        )
        self.addCleanup(processor.close)
        pause = bytes(1440 * 32)
        pcm = bytes(200 * 32) + tone(0.4, OWNER_F0) + pause + tone(1, OWNER_F0) + bytes(2144 * 32)
        for at in range(0, len(pcm), 6400):
            controller._feed(generation, processor, pcm[at : at + 6400], "stdin", harness.shadow)
        processor.finish()
        self.assertEqual([len(pcm) for pcm in retained], [400 * 32, 1000 * 32])
        self.assertTrue(all(not any(pcm) for pcm in retained))
        self.assertEqual([e.get("action") for e in events.drain()], ["exit"])
        self.assertEqual(harness.embedder.calls, 1)
        self.assertEqual(harness.shadow._labels, {})
        self.assertEqual(memory.snapshot(4000)["turns"], [])
        self.assertEqual(work.qsize(), 0)
        self.assertFalse(any(harness.shadow._ring.span(0, len(pcm) // 32)[0]))


class PrivateListenTests(unittest.TestCase):
    def test_start_private_listen_uses_voice_without_files_or_content_in_logs(self):
        for f0, allowed in ((OWNER_F0, True), (GUEST_F0, False)):
            with self.subTest(owner=allowed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                asset, config = root / "asset", root / "config.json"
                asset.touch()
                local = (
                    "whisper_executable",
                    "whisper_model",
                    "diarization_library",
                    "diarization_model",
                    "microphone_helper",
                )
                config.write_text(
                    json.dumps(
                        {
                            **{key: str(asset) for key in local},
                            "addressing": {"names": ["Unit"]},
                            "speakers": {"owner": ["owner"]},
                            "silero_vad": {"python": str(asset), "model": str(asset)},
                            "speaker_id": {
                                "python": str(asset),
                                "model": str(asset),
                                "live": True,
                                "roles": True,
                            },
                        }
                    )
                )
                expected = set(root.rglob("*"))
                harness = Harness(ToneEmbedder())
                controllers, buffers = [], []

                def shadow(_settings, *, report):
                    harness.shadow._report_to = report
                    return harness.shadow

                def processor(loaded, on_turn):
                    wait_for(lambda: bool(harness.shadow._compared))
                    callback = loaded.on_private_turn_audio

                    def audio(current, pcm):
                        callback(current, pcm)
                        buffers.append(controllers[0]._private_sentence.pcm)

                    return LiveProcessor(
                        replace(
                            loaded,
                            transcriber=Units(("Unit,", 200, 950), (" privacy off", 950, 1200)),
                            diarizer=Timeline((200, 1200, 1)),
                            on_private_turn_audio=audio,
                        ),
                        on_turn,
                    )

                def factory(loaded, **options):
                    controller = PrototypeController(
                        loaded,
                        processor_factory=processor,
                        speaker_id_factory=shadow,
                        capture_factory=Mock(side_effect=AssertionError("capture")),
                        provider_factory=Mock(side_effect=AssertionError("hosted")),
                        **options,
                    )
                    controllers.append(controller)
                    return controller

                args = Namespace(
                    config=config,
                    mode="stdin",
                    provenance="synthetic",
                    session_id="private-test",
                    use_jev=False,
                    allow_hosted=False,
                    start_private=True,
                )
                output, errors = io.StringIO(), io.StringIO()
                with (
                    patch.dict(os.environ, {"HOME": directory}),
                    contextlib.chdir(root),
                    patch("sys.stderr", errors),
                    patch("rightyo.prototype.SileroVad", return_value=FixtureVad()) as start_vad,
                ):
                    result = listen(
                        args,
                        output=output,
                        controller_factory=factory,
                        audio_input=io.BytesIO(bytes(200 * 32) + tone(1, f0) + bytes(2400 * 32)),
                    )
                self.assertEqual(result, 0)
                start_vad.assert_called_once()
                self.assertEqual(set(root.rglob("*")), expected)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(events[0]["duck"], {"version": 3, "private": True})
                self.assertEqual(
                    [e.get("action") for e in events if e["type"] == "privacy"],
                    ["exit"] if allowed else [],
                )
                self.assertTrue(buffers)
                self.assertTrue(all(not any(buffer) for buffer in buffers))
                self.assertEqual(harness.shadow._labels, {})
                self.assertEqual(harness.shadow.scored, 0)
                for line in errors.getvalue().splitlines():
                    self.assertRegex(
                        line,
                        r"^rightyo: (audio_retention speaker_ring_s=\d+ timeline_window_s=\d+"
                        r"|speaker_id start enrolled=\d+|speaker_id exit_check score=-?\d+\.\d+)$",
                    )
                self.assertNotIn(EXIT, output.getvalue() + errors.getvalue())


class PrivatePipeTests(unittest.TestCase):
    def test_private_worker_wipes_pcm_features_and_both_embedding_arrays(self):
        header = b'{"command":"embed_private","size":2}\n'
        source = io.BytesIO(header + b"\x01\x02")
        request = json.loads(header)
        protocol, np, session = io.StringIO(), Mock(), Mock()
        samples, features, output, vector = MagicMock(), MagicMock(), Mock(), Mock()
        np.frombuffer.return_value.astype.return_value = samples
        samples.__itruediv__.return_value = samples
        session.run.return_value = [output]
        np.asarray.return_value.reshape.return_value = vector
        vector.tolist.return_value = [0.25, -0.75]

        with (
            patch.object(speaker_id.json, "loads", return_value=request),
            patch.object(speaker_id, "_kaldi_fbank", return_value=features),
        ):
            self.assertTrue(speaker_id._embed_request(source, protocol, np, session, "input"))
        self.assertEqual(request, {})
        self.assertEqual(np.frombuffer.call_args.args[0], bytearray(2))
        for scratch in (samples, features, output, vector):
            scratch.fill.assert_called_once_with(0)
        self.assertEqual(vector.tolist.return_value, [])
        self.assertEqual(json.loads(protocol.getvalue()), {"ok": True, "embedding": [0.25, -0.75]})

    def test_worker_reads_mutable_pcm_and_wipes_partial_reads_on_failure(self):
        payload = tone(0.5)
        pcm = _read_private_pcm(io.BytesIO(payload), len(payload))
        self.assertIsInstance(pcm, bytearray)
        self.assertEqual(pcm, payload)
        for failure in ("eof", "exception"):
            with self.subTest(failure=failure):
                held = []

                def readinto(view):
                    held.append(view)
                    if len(held) == 1:
                        view[:2] = b"\x01\x02"
                        return 2
                    if failure == "exception":
                        raise OSError("synthetic token")
                    return 0

                with self.assertRaises((SpeakerIdError, OSError)):
                    _read_private_pcm(Mock(readinto=readinto), 32000)
                self.assertFalse(any(held[0]))
        for size in (None, True, -2, 0, 3, 320002):
            with self.assertRaises(SpeakerIdError):
                _read_private_pcm(io.BytesIO(), size)

    def test_private_request_has_fixed_header_and_borrowed_mutable_pcm_only(self):
        client = object.__new__(SpeakerEmbedder)
        client.process = Mock()
        client.process.stdin.write.side_effect = lambda view: len(view)
        client.timeout = 1
        client._receive = Mock(return_value={"embedding": [1.0, 0.0]})
        pcm = bytearray(tone(0.5))
        self.assertEqual(client.embed_private(memoryview(pcm)), [1.0, 0.0])
        header, payload = [call.args[0] for call in client.process.stdin.write.call_args_list]
        self.assertEqual(
            json.loads(header.tobytes()), {"command": "embed_private", "size": len(pcm)}
        )
        self.assertIs(payload.obj, pcm)
        for invalid in (
            memoryview(b"\0\0"),
            memoryview(bytearray(3)),
            memoryview(bytearray(320002)),
        ):
            with self.assertRaises(SpeakerIdError):
                client.embed_private(invalid)


if __name__ == "__main__":
    unittest.main()
