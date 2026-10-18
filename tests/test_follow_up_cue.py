"""Follow-ups that must carry a cue (rightyo#199): authored text and test doubles only."""

from __future__ import annotations

import unittest

from test_conversation import SESSION, decided, turn

from rightyo.addressedness import follow_up_cued
from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.tool_events import SpeechEvents

NAMES = Addressing.from_dict({"names": ["Jarvis"], "variants": {}})


class CueShapeTests(unittest.TestCase):
    def test_a_name_second_person_reference_or_adding_opener_is_a_cue(self):
        for text in (
            "Jarvis, louder",
            "can you make it shorter",
            "no, the other one",
            "and eggs",
            "do that again",
            "Actually, Tuesday",
            "the second one",
        ):
            with self.subTest(text=text):
                self.assertTrue(follow_up_cued(text, NAMES))

    def test_talk_to_a_pet_or_passer_by_usually_carries_none(self):
        for text in ("Good boy, come here!", "Sit. Stay.", "Lovely morning", "Drop the ball", ""):
            with self.subTest(text=text):
                self.assertFalse(follow_up_cued(text, NAMES))


class RequireCueTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()

    def follow_up(self, text, conversation):
        self.events.start(SESSION, now_ms=0, conversation=conversation, addressing=NAMES)
        self.events.drain()
        for current, label, extra in (
            (turn("t1", 0, 900, "Jarvis, add milk to the shopping list"), "attend", {}),
            (turn("t2", 5000, 5800, text), "uncertain", {"attend": 0.6}),
        ):
            self.events.transcript(current, current.end_ms)
            self.events.decision(decided(current, label, **extra), current.end_ms + 100)
        requests = [e for e in self.events.drain() if e["type"] == "request"]
        return len(requests) == 2

    def test_with_the_requirement_an_uncued_follow_up_is_refused(self):
        mode = Conversation(window_ms=10000, require_cue=True)
        self.assertFalse(self.follow_up("Good boy, come here!", mode))
        self.setUp()
        self.assertTrue(self.follow_up("and eggs too", mode))

    def test_off_by_default_as_before(self):
        self.assertTrue(self.follow_up("Good boy, come here!", Conversation(window_ms=10000)))

    def test_the_setting_is_validated_and_advertised_only_when_on(self):
        with self.assertRaises(ContractError):
            Conversation(require_cue="yes")
        self.assertTrue(Conversation.from_dict({"require_cue": True}).require_cue)
        self.assertIs(Conversation(require_cue=True).to_dict()["require_cue"], True)
        self.assertNotIn("require_cue", Conversation().to_dict())


if __name__ == "__main__":
    unittest.main()
