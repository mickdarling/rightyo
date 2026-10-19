"""Follow-ups that must carry a cue (rightyo#199): authored text and test doubles only."""

from __future__ import annotations

import unittest

from test_conversation import SESSION, decided, turn

from rightyo.addressedness import follow_up_cued
from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.tool_events import SpeechEvents

NAMES = Addressing.from_dict({"names": ["Jarvis"], "variants": {}})


class CueShapeTests(unittest.TestCase):
    def test_a_name_a_question_to_the_assistant_a_reference_or_an_addition_is_a_cue(self):
        for text in (
            "Jarvis, louder",
            "can you make it shorter",
            "no, the other one",
            "and eggs",
            "do that again",
            "Actually, Tuesday",
            "the second one",
            "play that",
            "turn it up",
            "what about tomorrow",
            "how about pizza",
            "louder",
            "next one",
            "skip",
            "use the same one instead",
            "one more time",
            "show me that one",
            "repeat",
            "cancel",
            "the other one",
            "bigger",
            "smaller",
            "Run the weekly report, please.",
            "Please check the garden forecast",
            "OK, open the budget sheet",
            "Could we do the blue one, please",
            "Hold on, you mean the cheaper plan?",
            "No, I said Thursday",
            "I meant the other folder",
        ):
            with self.subTest(text=text):
                self.assertTrue(follow_up_cued(text, NAMES))

    def test_talk_to_a_pet_or_a_passer_by_usually_carries_none(self):
        for text in (
            "Good boy, come here!",
            "Sit. Stay.",
            "Lovely morning",
            "Drop the ball",
            "",
            "No! Bad dog",
            "No, sit",
            "Not on the couch",
            "Leave it",
            "Come on, drop it",
            "Go get it!",
            "Get off that",
            "Who did this?",
            "Who's a good boy, you are!",
            "Is that your dog?",
            "Does it bite?",
            "Are you lost?",
            "Thank you",
            "See you later",
            "Good morning to you",
            "Make way",
            "Also, nice hat",
            "Just a second",
            "Wait a second",
            "Same to you!",
            "On the other hand",
            "And you!",
            "Put that down",
            "Run!",
            "Play nice",
            "Sit down, please",
            "Drop it, please",
            "Give me the ball",
            "Look at me",
        ):
            with self.subTest(text=text):
                self.assertFalse(follow_up_cued(text, NAMES))


class RequireCueTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()

    def follow_up(self, text, conversation, start=5000, attend=0.6):
        self.events.start(SESSION, now_ms=0, conversation=conversation, addressing=NAMES)
        self.events.drain()
        for current, label, extra in (
            (turn("t1", 0, 900, "Jarvis, add milk to the shopping list"), "attend", {}),
            (turn("t2", start, start + 800, text), "uncertain", {"attend": attend}),
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

    def test_a_closing_is_reported_as_a_closing_not_as_uncued(self):
        notes = []
        self.events = SpeechEvents(report=notes.append)
        self.assertFalse(
            self.follow_up("that's all", Conversation(window_ms=10000, require_cue=True))
        )
        reasons = [note for note in notes if note.startswith("follow_up ")]
        self.assertIn("reason=closing", reasons[-1])

    def test_the_requirement_applies_to_an_appendix_too(self):
        # Below the follow-up bar, inside the appendix window: only the cue decides.
        mode = Conversation(window_ms=10000, require_cue=True, appendix_ms=5000)
        self.assertFalse(self.follow_up("sit, good boy", mode, start=1500, attend=0.1))
        self.setUp()
        self.assertTrue(self.follow_up("and eggs", mode, start=1500, attend=0.1))

    def test_a_strong_attend_needs_no_cue_when_the_bypass_is_set(self):
        mode = Conversation(window_ms=10000, require_cue=True, cue_min_attend=0.75)
        self.assertTrue(self.follow_up("The blue folder", mode, attend=0.8))
        self.setUp()
        self.assertFalse(self.follow_up("The blue folder", mode, attend=0.7))
        self.setUp()
        # Without the bypass, the same strong attend still needs a cue.
        plain = Conversation(window_ms=10000, require_cue=True)
        self.assertFalse(self.follow_up("The blue folder", plain, attend=0.8))

    def test_the_bypass_is_validated_and_advertised_only_when_set(self):
        for bad in (0, 1.5, "0.8", True):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                Conversation(cue_min_attend=bad)
        self.assertEqual(Conversation.from_dict({"cue_min_attend": 0.75}).cue_min_attend, 0.75)
        self.assertEqual(Conversation(cue_min_attend=0.75).to_dict()["cue_min_attend"], 0.75)
        self.assertNotIn("cue_min_attend", Conversation().to_dict())

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
