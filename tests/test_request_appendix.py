"""An appendix to a just-formed request (rightyo#199): authored text and test doubles only."""

from __future__ import annotations

import unittest

from test_conversation import SESSION, decided, turn

from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.tool_events import SpeechEvents

MODE = Conversation(window_ms=10000, appendix_ms=2000)


class AppendixTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()

    def start(self, conversation=MODE, **options):
        self.events.start(SESSION, now_ms=0, conversation=conversation, **options)
        return self.events.drain()[0]

    def say(self, current, label, **options):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return [e for e in self.events.drain() if e["type"] in {"attention", "request"}]

    def appended(self, start, label="uncertain", text="and eggs, two dozen", **options):
        """Request at 0-900 ms, then a low-attend turn at `start`; whether it became a request."""
        self.say(turn("t1", 0, 900, "Jarvis, add milk to the shopping list"), "attend")
        attend = {"attend": 0.05} if label == "uncertain" else {}
        events = self.say(turn("t2", start, start + 800, text, **options), label, **attend)
        requests = [e for e in events if e["type"] == "request"]
        if requests:
            self.assertTrue(requests[0]["decision"]["follow_up"])
        return bool(requests)

    def test_a_quick_continuation_forms_a_follow_up_whatever_its_attend_probability(self):
        self.start()
        self.assertTrue(self.appended(2900))

    def test_a_confident_ignore_is_never_an_appendix(self):
        self.start()
        self.assertFalse(self.appended(1500, label="ignore"))

    def test_an_appendix_never_opens_another(self):
        self.start()
        self.assertTrue(self.appended(1500))
        events = self.say(turn("t3", 3000, 3600, "and bread"), "uncertain", attend=0.05)
        self.assertEqual([e["type"] for e in events], ["attention"])
        # A new direct request opens a new window.
        self.say(turn("t4", 4000, 4800, "Jarvis, add flour"), "attend")
        events = self.say(turn("t5", 5500, 6000, "and sugar"), "uncertain", attend=0.05)
        self.assertEqual([e["type"] for e in events], ["attention", "request"])

    def test_outside_the_window_the_usual_follow_up_rules_apply(self):
        self.start()
        self.assertFalse(self.appended(3000))

    def test_off_by_default(self):
        self.start(conversation=Conversation(window_ms=10000))
        self.assertFalse(self.appended(1500))

    def test_another_speaker_or_another_person_never_appends(self):
        self.start()
        self.assertFalse(self.appended(1500, speaker="Speaker B"))
        self.setUp()
        self.start()
        self.say(turn("t1", 0, 900, "Jarvis, add milk to the shopping list"), "attend")
        events = self.say(
            turn("t2", 1500, 2300, "and eggs"), "uncertain", recipient="other_human", attend=0.05
        )
        self.assertEqual([e["type"] for e in events], ["attention"])

    def test_a_closing_or_another_agents_turn_never_appends(self):
        self.start()
        self.assertFalse(self.appended(1500, text="thanks, that's all"))
        self.setUp()
        dot = Addressing.from_dict({"names": ["Dot"], "variants": {}})
        self.start(other_agents=dot)
        self.assertFalse(self.appended(1500, text="Dot, and eggs"))

    def test_the_window_is_bounded_and_advertised_only_when_on(self):
        for bad in (-1, 5001, 1.5, "2000"):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                Conversation(appendix_ms=bad)
        self.assertEqual(Conversation.from_dict({"appendix_ms": 2000}).appendix_ms, 2000)
        self.assertEqual(MODE.to_dict()["appendix_ms"], 2000)
        self.assertNotIn("appendix_ms", Conversation().to_dict())


if __name__ == "__main__":
    unittest.main()
