"""The assistant's spoken reply as decision context (rightyo#153).

The host's reply reports (#124) reach the decision state as `assistant_reply`, and Jev is
told how to read it, so a short unnamed answer to the reply can count as addressed.
"""

from __future__ import annotations

import json
import unittest

from test_addressedness import turn

from rightyo.pipeline import REPLY_CONTEXT_MS, ReplayRunner
from rightyo.providers import REPLY_GUIDANCE, MockProvider, build_request


class Recording:
    def __init__(self):
        self.states = []

    def decide(self, state):
        self.states.append(state)
        return MockProvider().decide(state)


class ReplyContextTests(unittest.TestCase):
    def setUp(self):
        self.provider = Recording()
        self.runner = ReplayRunner(self.provider)
        self.runner.restart("addressedness")

    def decide(self, current):
        self.runner.process(current)
        return self.provider.states[-1]

    def test_nothing_is_added_until_the_host_reports_a_reply(self):
        state = self.decide(turn("Haili, what time is it?", "u1", start=0, end=900))
        self.assertNotIn("assistant_reply", state)
        self.assertNotIn(REPLY_GUIDANCE, json.dumps(build_request(state)))

    def test_a_turn_after_the_reply_ended_carries_how_long_ago(self):
        self.runner.note_reply("started", 2000)
        self.runner.note_reply("ended", 9000)
        state = self.decide(turn("I heard that.", "u2", start=11000, end=12000))
        self.assertEqual(state["assistant_reply"], {"playing": False, "ended_ms_before": 2000})
        instructions = build_request(state)["questions"]["attention"]["instructions"]
        self.assertIn(REPLY_GUIDANCE, instructions)

    def test_a_turn_while_the_reply_plays_says_so(self):
        self.runner.note_reply("started", 2000)
        state = self.decide(turn("Stop.", "u2", start=3000, end=3400))
        self.assertEqual(state["assistant_reply"], {"playing": True})

    def test_an_old_reply_and_a_turn_before_the_reply_carry_nothing(self):
        self.runner.note_reply("started", 1000)
        self.runner.note_reply("ended", 2000)
        late = 2000 + REPLY_CONTEXT_MS + 1
        self.assertNotIn(
            "assistant_reply", self.decide(turn("Later.", "u2", start=late, end=late + 500))
        )
        self.runner.note_reply("started", late + 1000)
        earlier = self.decide(turn("Before it.", "u3", start=late + 600, end=late + 900))
        self.assertNotIn("assistant_reply", earlier)

    def test_a_restart_forgets_the_reply(self):
        self.runner.note_reply("started", 1000)
        self.runner.note_reply("ended", 2000)
        self.runner.restart("addressedness")
        self.assertNotIn("assistant_reply", self.decide(turn("Hello.", "u1", start=3000, end=3500)))

    def test_an_unknown_phase_is_refused(self):
        with self.assertRaises(ValueError):
            self.runner.note_reply("paused", 1000)


if __name__ == "__main__":
    unittest.main()
