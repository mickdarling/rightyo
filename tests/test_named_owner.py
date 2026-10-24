"""Named owner abstentions (#735): authored turns and stand-in live bindings only.

No speech, enrollment store, model or network: these qualify deterministic promotion,
acknowledgement and re-engagement, not attention-model or live voiceprint quality.
"""

from __future__ import annotations

import unittest

from test_conversation import MODE, SESSION, decided, turn

from rightyo.contracts import Acknowledgement, Addressing, SpeakerPriority
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents

OURS = Addressing(names=("Jarvis",), variants=(("Jarvis", ("Jarrvis",)),))
PROMOTED = "attention_promoted reason=named_owner"


class Bindings:
    """Authored live identifier results, never actual voices or identities."""

    def __init__(self, binding):
        self.value = binding

    def binding(self, label):
        return self.value if label == "Speaker A" else None


def enrolled(binding=("bound", "owner"), priority=SpeakerPriority(owners=("owner",))):
    roles = EnrolledRoles(priority)
    roles.source = Bindings(binding)
    return roles


def heard(name, start, text):
    return turn(name, start, start + 900, text, provenance="diarization-timeline")


class NamedOwnerTests(unittest.TestCase):
    def start(self, priority=None, *, addressing=OURS):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.events.start(
            SESSION,
            now_ms=0,
            addressing=addressing,
            priority=enrolled() if priority is None else priority,
            conversation=MODE,
            acknowledgement=Acknowledgement(min_confidence=0.95, dedup_window_ms=0),
            duck=True,
        )
        self.events.drain()
        # Start an ordinary exchange, then explicitly end it by addressing another person.
        self.say(heard("t1", 0, "Open the synthetic ledger."), "attend")
        ended = self.say(
            heard("t2", 2000, "Did you see the synthetic ledger?"),
            "ignore",
            recipient="other_human",
        )
        self.assertEqual(
            [(e["state"], e["reason"]) for e in ended if e["type"] == "conversation"],
            [("ambient", "other_human")],
        )
        self.assertIsNone(self.events._engaged)
        self.notes.clear()

    def say(self, current, label, *, inferred=False, **options):
        self.events.transcript(current, current.end_ms, inferred=inferred)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return self.events.drain()

    def test_named_owner_abstentions_acknowledge_and_re_engage_after_other_human(self):
        texts = (
            "Jarvis, that's for you.",
            "Jarvis, it doesn't seem to be recognizing my voice. Can you hear me?",
            "Jarvis,",
            "Jarvis.",
            "um, Jarvis, that's for you.",
            "please, okay, now, uh, so, thanks, hey Jarvis, that's for you.",
            "Jarrvis, that's for you.",
        )
        for text in texts:
            for recipient in ("system", "unknown"):
                with self.subTest(text=text, recipient=recipient):
                    self.start()
                    current = heard("t3", 4000, text)
                    events = self.say(
                        current, "uncertain", recipient=recipient, attend=0.1, confidence=0.55
                    )
                    self.assertEqual(
                        [e["type"] for e in events],
                        ["transcript", "attention", "request", "conversation"],
                    )
                    transcript, attention, request, conversation = events
                    self.assertEqual(transcript["role_source"], "voiceprint")
                    self.assertEqual(request["turn"]["role"], "owner")
                    self.assertEqual(attention["decision"], request["decision"])
                    self.assertEqual(attention["decision"]["label"], "attend")
                    self.assertEqual(attention["decision"]["confidence"], 0.55)
                    self.assertNotIn("follow_up", attention["decision"])
                    self.assertIs(request["acknowledge"], True)
                    self.assertTrue(
                        any(n.startswith("ack outcome=ack reason=named ") for n in self.notes)
                    )
                    self.assertEqual(self.notes.count(PROMOTED), 1)
                    self.assertEqual(
                        (conversation["state"], conversation["reason"]), ("engaged", "request")
                    )
                    self.assertEqual(conversation["request_id"], request["request_id"])
                    self.assertEqual(conversation["until_ms"], current.end_ms + MODE.window_ms)
                    # A retry of the decision neither delivers nor reports a second promotion.
                    self.events.decision(decided(current, "uncertain"), current.end_ms + 200)
                    self.assertEqual(self.events.drain(), [])
                    self.assertEqual(self.notes.count(PROMOTED), 1)
                    follow_up = self.say(
                        heard("t4", 6000, "And the other ledger?"), "uncertain", attend=0.55
                    )
                    requests = [e for e in follow_up if e["type"] == "request"]
                    self.assertEqual(len(requests), 1)
                    self.assertIs(requests[0]["decision"]["follow_up"], True)
                    self.assertEqual(self.notes.count(PROMOTED), 1)

    def test_non_voiceprint_owners_and_non_owners_are_never_promoted(self):
        cases = (
            ("guest", enrolled(("bound", "guest")), False),
            ("unknown", enrolled(None), False),
            ("tentative", enrolled(("tentative", "owner")), False),
            (
                "trusted",
                enrolled(("bound", "guest"), SpeakerPriority(trusted=("guest",))),
                False,
            ),
            (
                "configured owner",
                ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",))),
                False,
            ),
            (
                "configured owner with live roles",
                enrolled(priority=SpeakerPriority(owners=("Speaker A", "owner"))),
                False,
            ),
        )
        for name, priority, inferred in cases:
            with self.subTest(case=name):
                self.start(priority)
                events = self.say(
                    heard("t3", 4000, "Jarvis, that's for you."), "uncertain", inferred=inferred
                )
                self.assert_unpromoted(events)

    def test_inferred_live_bound_owner_is_promoted(self):
        self.start()
        self.events.media(False)
        events = self.say(heard("t3", 4000, "Jarvis, that's for you."), "uncertain", inferred=True)
        self.assertEqual(
            [e["type"] for e in events],
            ["transcript", "attention", "request", "conversation"],
        )
        transcript, attention, request, conversation = events
        self.assertEqual(transcript["turn"]["role"], "owner")
        self.assertNotIn("role_source", transcript)
        self.assertEqual(request["turn"]["role"], "owner")
        self.assertEqual(attention["decision"], request["decision"])
        self.assertEqual(attention["decision"]["label"], "attend")
        self.assertIs(request["acknowledge"], True)
        self.assertEqual((conversation["state"], conversation["reason"]), ("engaged", "request"))
        self.assertEqual(self.notes.count(PROMOTED), 1)

    def test_media_blocks_inferred_owner_attends_and_promotions_equally(self):
        outcomes = []
        for label in ("attend", "uncertain"):
            with self.subTest(label=label):
                self.start()
                self.events.media(True)
                events = self.say(
                    heard("t3", 4000, "Jarvis, pause the show."), label, inferred=True
                )
                self.assert_unpromoted(events, "ignore")
                evidence = events[1]["decision"]
                self.assertIs(evidence["media"], True)
                outcomes.append((evidence["label"], evidence["media"]))
        self.assertEqual(outcomes, [("ignore", True), ("ignore", True)])

    def test_media_allows_named_owner_promotion_with_owner_authority(self):
        self.start()
        self.events.media(True)
        current = heard("t3", 4000, "Jarvis, pause the show.")
        self.assertTrue(self.events._media_attends(current, "owner"))
        events = self.say(current, "uncertain")
        request = next(e for e in events if e["type"] == "request")
        self.assertEqual(request["decision"]["label"], "attend")
        self.assertNotIn("media", request["decision"])
        self.assertEqual(self.notes.count(PROMOTED), 1)

    def test_unbound_label_is_not_promoted(self):
        self.start(enrolled(None))
        self.assert_unpromoted(self.say(heard("t3", 4000, "Jarvis, that's for you."), "uncertain"))

    def assert_unpromoted(self, events, label="uncertain"):
        self.assertEqual([e["type"] for e in events], ["transcript", "attention"])
        self.assertEqual(events[1]["decision"]["label"], label)
        self.assertNotIn(PROMOTED, self.notes)
        self.assertIsNone(self.events._engaged)

    def test_mid_sentence_names_and_ignore_are_never_promoted(self):
        for text, label in (
            ("I told Jarvis about it.", "uncertain"),
            ("well, Jarvis, that's for you.", "uncertain"),
            ("Jarvis, that's for you.", "ignore"),
        ):
            with self.subTest(text=text, label=label):
                self.start()
                self.assert_unpromoted(self.say(heard("t3", 4000, text), label), label)

    def test_no_configured_name_means_no_promotion(self):
        self.start(addressing=None)
        self.assert_unpromoted(self.say(heard("t3", 4000, "Jarvis, that's for you."), "uncertain"))

    def test_text_attached_to_names_is_never_promoted(self):
        cases = (
            (OURS, "我告诉Jarvis这件事。"),
            (OURS, "Jarvis这件事。"),
            (OURS, "我告诉Jarvis"),
            (OURS, "Jarvisonian, that's for you."),
            (
                Addressing(names=("Synthetic Ledger Assistant Unit",)),
                "我告诉Synthetic Ledger Assistant Unit, that's for you.",
            ),
        )
        for addressing, text in cases:
            with self.subTest(text=text):
                self.start(addressing=addressing)
                events = self.say(heard("t3", 4000, text), "uncertain")
                self.assertEqual(events[0]["turn"]["role"], "owner")
                self.assertEqual(events[0]["role_source"], "voiceprint")
                self.assert_unpromoted(events)

    def test_configured_multiword_and_filler_names_are_supported(self):
        for name in ("Okay", "Hey", "Synthetic Ledger Assistant Unit"):
            with self.subTest(name=name):
                self.start(addressing=Addressing(names=(name,)))
                events = self.say(heard("t3", 4000, name + ", that's for you."), "uncertain")
                self.assertEqual(len([e for e in events if e["type"] == "request"]), 1)
                self.assertEqual(self.notes.count(PROMOTED), 1)

    def test_punctuation_and_spacing_inside_names_are_supported(self):
        for name, text in (
            ("R2D2", "R2-D2, help"),
            ("R2D2", "R2 D2, help"),
            ("RightyO", "Righty O, help"),
        ):
            with self.subTest(name=name, text=text):
                self.start(addressing=Addressing(names=(name,)))
                events = self.say(heard("t3", 4000, text), "uncertain")
                requests = [e for e in events if e["type"] == "request"]
                self.assertEqual(len(requests), 1)
                self.assertIs(requests[0]["acknowledge"], True)
                self.assertEqual(self.notes.count(PROMOTED), 1)

    def test_an_unavailable_placeholder_is_never_promoted(self):
        self.start()
        current = heard("t3", 4000, "Jarvis, that's for you.")
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, "uncertain"), 5000, unavailable="timeout")
        self.assert_unpromoted(self.events.drain())

    def test_privacy_emits_no_content_or_promotion(self):
        self.start()
        self.events.duck(True, now_ms=3000)
        self.events.drain()
        self.notes.clear()
        self.assertEqual(self.say(heard("t3", 4000, "Jarvis, that's for you."), "uncertain"), [])
        self.assertNotIn(PROMOTED, self.notes)

    def test_a_public_turn_decided_while_private_is_not_promoted(self):
        self.start()
        current = heard("t3", 4000, "Jarvis, that's for you.")
        self.events.transcript(current, current.end_ms)
        self.events.drain()
        self.events.duck(True, now_ms=5000)
        self.events.drain()
        self.events.decision(decided(current, "uncertain"), 5100)
        events = self.events.drain()
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertEqual(events[0]["decision"]["label"], "uncertain")
        self.assertNotIn(PROMOTED, self.notes)

    def test_an_ordinary_attend_still_acknowledges_and_engages_without_promotion(self):
        self.start()
        events = self.say(heard("t3", 4000, "Jarvis, open the synthetic ledger."), "attend")
        request = next(e for e in events if e["type"] == "request")
        self.assertIs(request["acknowledge"], True)
        self.assertEqual(request["decision"]["label"], "attend")
        self.assertEqual(events[-1]["state"], "engaged")
        self.assertNotIn(PROMOTED, self.notes)


if __name__ == "__main__":
    unittest.main()
