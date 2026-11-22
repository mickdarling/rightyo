"""Acceptance-time presence with synthetic turns and scripted binding revisions."""

import unittest

from test_conversation import SESSION, decided, turn

from rightyo.contracts import Addressing, Conversation, SpeakerPriority
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.tool_events import SpeechEvents


class Bindings:
    def __init__(self):
        self.epoch = 0
        self.labels = {}
        self.revisions = {}

    def set(self, label, state, identity):
        self.labels[label] = (state, identity)
        self.revisions[label] = self.revisions.get(label, 0) + 1

    def binding(self, label):
        return self.labels.get(label)

    def binding_snapshot(self, label):
        return (self.epoch, self.revisions.get(label, 0)), self.binding(label)


class PresenceIdentityTests(unittest.TestCase):
    def setUp(self):
        self.bindings = Bindings()
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
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

    def engage(self, state="tentative"):
        self.bindings.set("A", state, "owner")
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

    def test_judged_identity_can_resolve_history_without_an_engaged_binding(self):
        self.engage("insufficient")
        self.hear("B")
        self.bindings.set("B", "bound", "owner")
        self.bindings.set("C", "bound", "owner")
        self.assertFalse(self.events._others_present(self.hear("C")))

    def test_tentative_engagement_matches_only_bound_history(self):
        self.engage()
        self.bindings.set("C", "tentative", "owner")
        self.hear("C")
        pending = self.hear("A")
        self.assertTrue(self.events._others_present(pending))
        self.bindings.set("C", "bound", "owner")
        self.assertFalse(self.events._others_present(self.hear("A")))
        self.assertFalse(self.decide(pending, "attend"))

    def test_unbound_guest_and_different_bound_identity_keep_the_bar(self):
        self.engage("bound")
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        self.hear("B")
        for state in ("unknown", "bound"):
            with self.subTest(state=state):
                self.bindings.set("B", state, "guest")
                current = self.hear("A")
                self.assertTrue(self.events._others_present(current))
                self.assertEqual(
                    self.notes[-1], "attention_presence others_present=1 same_identity=1"
                )
                self.assertFalse(self.decide(current, "attend"))
        self.assertTrue(self.events._others_present(self.hear("B")))

    def test_dropped_binding_counts_again_on_the_next_acceptance(self):
        self.engage()
        self.hear("C")
        for change in ("revision", "epoch"):
            with self.subTest(change=change):
                self.bindings.set("C", "bound", "owner")
                pending = self.hear("A")
                self.assertFalse(self.events._others_present(pending))
                if change == "epoch":
                    self.bindings.epoch += 1
                    self.bindings.labels.clear()
                    self.bindings.revisions.clear()
                else:
                    self.bindings.set("C", "unknown", "owner")
                self.assertTrue(self.events._others_present(self.hear("A")))
                self.assertFalse(self.events._others_present(pending))

    def test_overlapping_or_non_timeline_history_keeps_the_bar(self):
        self.engage()
        self.hear("B", overlap=True)
        self.hear("C", provenance="authored-fixture")
        self.bindings.set("B", "bound", "owner")
        self.bindings.set("C", "bound", "owner")
        self.assertTrue(self.events._others_present(self.hear("A")))
        self.assertFalse(any("same_identity=" in note for note in self.notes))

    def test_revision_change_during_comparison_keeps_the_bar(self):
        self.engage()
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        snapshot = self.bindings.binding_snapshot

        def changing(label):
            before = snapshot(label)
            if label == "C":
                self.bindings.set(label, "unknown", "owner")
                self.bindings.set(label, "bound", "owner")
            return before

        self.bindings.binding_snapshot = changing
        self.assertTrue(self.events._others_present(self.hear("A")))

    def test_presence_minimums_and_playback_still_exclude_guest_entries(self):
        self.engage()
        self.hear("C")
        self.bindings.set("C", "bound", "owner")
        self.hear("B", duration=999)
        self.hear("B", words=2)
        self.events.reply("started", self.now)
        self.hear("B")
        self.events.reply("ended", self.now)
        self.assertFalse(self.events._others_present(self.hear("A")))
        self.assertEqual(self.notes[-1], "attention_presence others_present=0 same_identity=1")
