"""The requester continuing after any reply is a follow-up cue (rightyo#213); synthetic only."""

from __future__ import annotations

import dataclasses
import unittest

from test_conversation import SESSION, decided, turn

from rightyo.contracts import ContractError, Conversation
from rightyo.tool_events import SpeechEvents

MODE = Conversation(window_ms=60000, require_cue=True, continuation_cue=True)


def chose(current, choice, attend=0.6):
    """An `uncertain` decision whose raw attention choice was `choice`."""
    event = decided(current, "uncertain", attend=attend)
    return dataclasses.replace(
        event, decision=dataclasses.replace(event.decision, attention_choice=choice)
    )


class ContinuationCueTests(unittest.TestCase):
    def requests(self, steps, mode=MODE):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=mode)
        for step in steps:
            if step[0] == "reply":
                events.reply(step[1], step[2])
                continue
            _, name, start, end, text, speaker, choice = step
            current = turn(name, start, end, text, speaker=speaker)
            events.transcript(current, end)
            if choice == "request":
                events.decision(decided(current, "attend"), end + 100)
            else:
                events.decision(chose(current, choice), end + 100)
        return [e["turn"]["text"] for e in events.drain() if e["type"] == "request"]

    REQUEST = ("turn", "t1", 0, 900, "Jarvis, tidy the shared folder", "Speaker A", "request")
    REPLIED = [REQUEST, ("reply", "started", 2000), ("reply", "ended", 4000)]

    def test_the_requester_continuing_after_a_reply_is_a_follow_up(self):
        more = ("turn", "t2", 6000, 6800, "The old drafts can go too", "Speaker A", "attend")
        self.assertEqual(self.requests([*self.REPLIED, more])[-1], "The old drafts can go too")

    def test_speaking_over_the_reply_counts(self):
        steps = [
            self.REQUEST,
            ("reply", "started", 2000),
            ("turn", "t2", 3000, 3800, "The old drafts can go too", "Speaker A", "attend"),
        ]
        self.assertEqual(self.requests(steps)[-1], "The old drafts can go too")

    def test_only_the_reply_ending_is_enough(self):
        steps = [
            self.REQUEST,
            ("reply", "ended", 4000),
            ("turn", "t2", 6000, 6800, "The old drafts can go too", "Speaker A", "attend"),
        ]
        self.assertEqual(self.requests(steps)[-1], "The old drafts can go too")

    def test_before_any_reply_it_still_needs_a_cue(self):
        more = ("turn", "t2", 1500, 1900, "The old drafts can go too", "Speaker A", "attend")
        self.assertEqual(self.requests([self.REQUEST, more]), [self.REQUEST[4]])

    def test_a_turn_jev_did_not_choose_to_attend_still_needs_a_cue(self):
        for choice in ("uncertain", "ignore"):
            with self.subTest(choice=choice):
                pet = ("turn", "t2", 6000, 6800, "Good boy, sit", "Speaker A", choice)
                self.assertEqual(self.requests([*self.REPLIED, pet]), [self.REQUEST[4]])

    def test_another_or_unlabelled_speaker_never_continues(self):
        for speaker in ("Speaker B", None):
            with self.subTest(speaker=speaker):
                other = ("turn", "t2", 6000, 6800, "The old drafts can go too", speaker, "attend")
                self.assertEqual(self.requests([*self.REPLIED, other]), [self.REQUEST[4]])

    def test_off_by_default(self):
        mode = Conversation(window_ms=60000, require_cue=True)
        more = ("turn", "t2", 6000, 6800, "The old drafts can go too", "Speaker A", "attend")
        self.assertEqual(self.requests([*self.REPLIED, more], mode), [self.REQUEST[4]])

    def test_a_reply_does_not_outlive_its_conversation(self):
        renewed = ("turn", "t3", 300000, 300900, "Jarvis, lights off", "Speaker A", "request")
        more = ("turn", "t4", 302000, 302800, "The old drafts can go too", "Speaker A", "attend")
        # Lapsed, then a new named request that got no spoken reply yet.
        self.assertEqual(self.requests([*self.REPLIED, renewed, more])[-1], "Jarvis, lights off")

    def test_each_request_waits_for_its_own_reply(self):
        steps = [
            *self.REPLIED,
            ("turn", "t2", 6000, 6800, "The old drafts can go too", "Speaker A", "attend"),
            ("turn", "t3", 7500, 8300, "Leave the backups alone", "Speaker A", "attend"),
        ]
        self.assertEqual(self.requests(steps)[-1], "The old drafts can go too")


class ContinuationContractTests(unittest.TestCase):
    def test_round_trip_and_validation(self):
        mode = Conversation.from_dict({"require_cue": True, "continuation_cue": True})
        self.assertTrue(mode.continuation_cue)
        self.assertTrue(mode.to_dict()["continuation_cue"])
        self.assertNotIn("continuation_cue", Conversation().to_dict())
        with self.assertRaises(ContractError):
            Conversation(continuation_cue=1)


if __name__ == "__main__":
    unittest.main()
