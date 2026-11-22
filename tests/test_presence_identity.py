"""Acceptance-time presence with synthetic turns and scripted binding revisions."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from test_conversation import SESSION, decided, turn

from rightyo.contracts import Addressing, Conversation, SpeakerPriority
from rightyo.live_speaker_id import EnrolledRoles, PresenceBinding
from rightyo.tool_events import SpeechEvents


class Bindings:
    def __init__(self):
        self.epoch = 0
        self.labels = {}
        self.revisions = {}
        self.self_labels = set()
        self.self_generation = object()

    def set(self, label, state, identity):
        if self.binding(label) == (state, identity):
            return
        self.labels[label] = (state, identity)
        self.revisions[label] = self.revisions.get(label, 0) + 1

    def binding(self, label):
        return self.labels.get(label)

    def set_self(self, label, self_bound):
        if self_bound != (label in self.self_labels):
            self.self_labels.symmetric_difference_update({label})
            self.self_generation = object()
            self.revisions[label] = self.revisions.get(label, 0) + 1

    def presence_snapshot(self, label):
        state, identity = self.binding(label) or ("unbound", None)
        return PresenceBinding(
            self.epoch,
            self.revisions.get(label, 0),
            state,
            identity,
            label in self.self_labels,
            self.self_generation,
        )


class PresenceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.start_session()

    def start_session(self):
        self.bindings = Bindings()
        roles = EnrolledRoles(
            SpeakerPriority(owners=("owner",), trusted=("trusted",), owner_only=False)
        )
        roles.source = self.bindings
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.address = Addressing(("unit",))
        self.events.start(
            SESSION,
            priority=roles,
            addressing=self.address,
            conversation=Conversation(cue_when_others_ms=120000),
        )
        self.now = 0

    def hear(
        self,
        label,
        *,
        named=False,
        duration=1000,
        words=3,
        provenance="diarization-timeline",
        overlap=False,
        inferred=False,
    ):
        tokens = ["x"] * words
        if named:
            tokens[0] = self.address.names[0]
        start = self.now
        self.now += duration
        current = turn(
            f"u{start}",
            start,
            self.now,
            " ".join(tokens),
            label,
            provenance=provenance,
            overlap=overlap,
        )
        self.events.transcript(current, self.now, inferred=inferred)
        return current

    def decide(self, current, label):
        self.events.decision(decided(current, label), self.now)
        return [event for event in self.events.drain() if event["type"] == "request"]

    def engage(self, state="tentative", identity="owner"):
        if state is not None:
            self.bindings.set("A", state, identity)
        self.assertTrue(self.decide(self.hear("A"), "attend"))
        self.assertEqual(self.events._engaged["speaker_id"], "A")

    def test_prebinding_entries_clear_for_a_named_bound_split(self):
        self.engage()
        self.hear("C")
        self.assertTrue(self.events._others_present(self.hear("C")))
        self.bindings.set("C", "bound", "owner")
        current = self.hear("C", named=True, inferred=True)
        self.assertFalse(self.events._others_present(current))
        self.assertEqual(self.notes[-1], "attention_presence others_present=0 same_identity=2")
        self.assertFalse(self.decide(self.hear("C", named=True), "ignore"))
        self.assertTrue(self.decide(current, "uncertain"))

    def assert_held(self, label):
        current = self.hear(label)
        self.assertTrue(self.events._others_present(current))
        self.assertFalse(self.decide(current, "attend"))

    def test_guest_rebind_preserves_the_guest_entry(self):
        for state in ("tentative", "bound"):
            with self.subTest(state=state):
                self.start_session()
                self.engage("bound")
                self.bindings.set("C", state, "guest")
                self.hear("C")
                self.bindings.set("C", "bound", "owner")
                self.assert_held("A")

    def test_each_entry_uses_its_own_binding_revision(self):
        self.engage("bound")
        self.hear("C")
        self.bindings.set("C", "bound", "guest")
        self.bindings.set("C", "tentative", "owner")
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        self.assert_held("A")
        self.assertIn("attention_presence others_present=1 same_identity=1", self.notes)

    def test_binding_change_during_guest_acceptance_keeps_the_bar(self):
        self.engage("bound")
        self.bindings.set("C", "bound", "guest")
        snapshot = self.bindings.presence_snapshot

        def rebind(label):
            self.bindings.set(label, "bound", "owner")
            return snapshot(label)

        with patch.object(self.bindings, "presence_snapshot", rebind):
            self.hear("C")
        self.assert_held("A")

    def test_missing_or_unreadable_snapshots_keep_the_bar(self):
        for stage in ("entry", "judgement"):
            for evidence in (None, ValueError(), "revision"):
                with self.subTest(stage=stage, evidence=evidence):
                    self.start_session()
                    self.engage("bound")
                    invalid = replace(self.bindings.presence_snapshot("C"), revision=None)
                    lookup = patch.object(
                        self.bindings,
                        "presence_snapshot",
                        side_effect=evidence if isinstance(evidence, Exception) else None,
                        return_value=invalid if evidence == "revision" else None,
                    )
                    if stage == "entry":
                        with lookup:
                            self.hear("C")
                        self.bindings.set("C", "bound", "owner")
                        self.assert_held("A")
                    else:
                        self.hear("C")
                        self.bindings.set("C", "bound", "owner")
                        with lookup:
                            self.assert_held("A")

    def test_guest_turns_require_an_engaged_identity_tie(self):
        for state in (None, "unknown", "tentative"):
            for identity in ("guest", "trusted"):
                with self.subTest(engaged=state, current=identity):
                    self.start_session()
                    self.engage(state)
                    if identity == "guest":
                        self.bindings.set("B", "bound", identity)
                    self.hear("B")
                    self.bindings.set("B", "bound", identity)
                    self.assert_held("B")

    def test_matching_guest_engagement_cannot_anchor_a_skip(self):
        self.engage("bound", "guest")
        self.bindings.set("B", "bound", "guest")
        self.hear("B")
        self.assert_held("B")

    def test_tentative_engagement_matches_only_bound_history(self):
        self.engage()
        self.bindings.set("C", "tentative", "owner")
        self.bindings.set("D", "bound", "owner")
        self.hear("C")
        pending = self.hear("D")
        self.assertTrue(self.events._others_present(pending))
        self.bindings.set("C", "bound", "owner")
        self.assertFalse(self.events._others_present(self.hear("D")))
        self.assertFalse(self.decide(pending, "attend"))

    def test_tentative_turn_cannot_anchor_a_skip(self):
        self.engage()
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        self.assert_held("A")

    def test_self_bound_tentative_engagement_keeps_the_bar(self):
        self.engage()
        self.bindings.set_self("A", True)
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        self.assert_held("C")

    def test_self_binding_flip_between_entry_and_judgement_keeps_the_bar(self):
        for label in ("A", "C", "D"):
            with self.subTest(label=label):
                self.start_session()
                self.engage()
                self.bindings.set("C", "tentative", "owner")
                self.hear("C")
                self.bindings.set_self(label, True)
                self.bindings.set_self(label, False)
                self.bindings.set("C", "bound", "owner")
                self.bindings.set("D", "bound", "owner")
                self.assert_held("D")

    def test_entry_self_bound_at_recording_keeps_the_bar(self):
        self.engage()
        self.bindings.set("C", "tentative", "owner")
        self.bindings.set_self("C", True)
        self.hear("C")
        self.bindings.set_self("C", False)
        self.bindings.set("C", "bound", "owner")
        self.assert_held("C")

    def test_unengaged_trusted_turn_compares_to_its_own_binding(self):
        self.hear("C")
        self.bindings.set("C", "bound", "trusted")
        self.bindings.set("B", "bound", "trusted")
        current = self.hear("B")
        self.assertFalse(self.events._others_present(current))
        self.assertTrue(self.decide(current, "attend"))

    def test_unbound_guest_and_different_bound_identity_keep_the_bar(self):
        self.engage("bound")
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        for state in ("unknown", "bound"):
            with self.subTest(state=state):
                self.bindings.set("B", state, "guest")
                self.hear("B")
                current = self.hear("A")
                self.assertTrue(self.events._others_present(current))
                self.assertEqual(
                    self.notes[-1], "attention_presence others_present=1 same_identity=1"
                )
                self.assertFalse(self.decide(current, "attend"))
        self.assertTrue(self.events._others_present(self.hear("B")))

    def test_dropped_binding_counts_again_on_the_next_acceptance(self):
        for change in ("revision", "epoch"):
            with self.subTest(change=change):
                self.start_session()
                self.engage("bound")
                self.hear("C")
                self.bindings.set("C", "bound", "owner")
                pending = self.hear("A")
                self.assertFalse(self.events._others_present(pending))
                if change == "epoch":
                    self.bindings.epoch += 1
                    self.bindings.labels.clear()
                    self.bindings.revisions.clear()
                    self.bindings.set("A", "bound", "owner")
                    self.bindings.set("C", "bound", "owner")
                else:
                    self.bindings.set("C", "unknown", "owner")
                self.assertTrue(self.events._others_present(self.hear("A")))
                self.assertFalse(self.events._others_present(pending))

    def test_overlapping_or_non_timeline_history_keeps_the_bar(self):
        self.engage("bound")
        self.hear("B", overlap=True)
        self.hear("C", provenance="authored-fixture")
        self.bindings.set("B", "bound", "owner")
        self.bindings.set("C", "bound", "owner")
        self.assertTrue(self.events._others_present(self.hear("A")))
        self.assertFalse(any("same_identity=" in note for note in self.notes))

    def test_revision_change_during_comparison_keeps_the_bar(self):
        self.engage("bound")
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        snapshot = self.bindings.presence_snapshot

        def changing(label):
            before = snapshot(label)
            if label == "C":
                self.bindings.set(label, "unknown", "owner")
                self.bindings.set(label, "bound", "owner")
            return before

        self.bindings.presence_snapshot = changing
        self.assertTrue(self.events._others_present(self.hear("A")))

    def test_presence_minimums_and_playback_still_exclude_guest_entries(self):
        self.engage("bound")
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        self.hear("B", duration=999)
        self.hear("B", words=2)
        self.events.reply("started", self.now)
        self.hear("B")
        self.events.reply("ended", self.now)
        self.assertFalse(self.events._others_present(self.hear("A")))
        self.assertEqual(self.notes[-1], "attention_presence others_present=0 same_identity=1")
