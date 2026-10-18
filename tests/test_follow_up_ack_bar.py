"""A follow-up's own acknowledgement bar (rightyo#199): authored text and test doubles only."""

from __future__ import annotations

import unittest

from test_conversation import SESSION, decided, turn

from rightyo.contracts import Acknowledgement, Addressing, ContractError, Conversation
from rightyo.tool_events import SpeechEvents

NAMES = Addressing.from_dict({"names": ["Jarvis"], "variants": {}})


class FollowUpAckBarTests(unittest.TestCase):
    def acknowledged(self, gate, text="and eggs", attend=0.75):
        """Whether a follow-up at `attend`, after a direct request, is acknowledged."""
        events = SpeechEvents()
        events.start(
            SESSION,
            now_ms=0,
            conversation=Conversation(window_ms=60000),
            acknowledgement=gate,
            addressing=NAMES,
        )
        events.drain()
        first = turn("t1", 0, 900, "Jarvis, add milk")
        events.transcript(first, 900)
        events.decision(decided(first, "attend"), 1000)
        second = turn("t2", 30000, 30800, text)
        events.transcript(second, 30800)
        events.decision(decided(second, "uncertain", attend=attend), 30900)
        requests = [e for e in events.drain() if e["type"] == "request"]
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[1]["decision"]["follow_up"])
        return requests[1]["acknowledge"]

    def test_a_mid_confidence_follow_up_is_not_acknowledged_with_its_own_bar(self):
        self.assertTrue(self.acknowledged(Acknowledgement()))
        self.assertFalse(self.acknowledged(Acknowledgement(follow_up_min_confidence=0.9)))
        self.assertTrue(
            self.acknowledged(Acknowledgement(follow_up_min_confidence=0.9), attend=0.95)
        )

    def test_a_named_follow_up_is_still_acknowledged(self):
        gate = Acknowledgement(follow_up_min_confidence=0.9)
        self.assertTrue(self.acknowledged(gate, text="Jarvis, and eggs"))

    def test_the_bar_is_validated_and_advertised_only_when_set(self):
        for bad in (-0.1, 1.5, "0.9"):
            with self.subTest(bad=bad), self.assertRaises((ContractError, TypeError, ValueError)):
                Acknowledgement(follow_up_min_confidence=bad)
        gate = Acknowledgement.from_dict({"follow_up_min_confidence": 0.9})
        self.assertEqual(gate.to_dict()["follow_up_min_confidence"], 0.9)
        self.assertNotIn("follow_up_min_confidence", Acknowledgement().to_dict())


if __name__ == "__main__":
    unittest.main()
