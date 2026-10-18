"""An unnamed turn another speaker answers is not the assistant's (rightyo#199); synthetic."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_conversation import SESSION, decided, turn

from rightyo.contracts import Addressing, Conversation, ProviderDecision
from rightyo.pipeline import ReplayRunner
from rightyo.prototype import DecisionConfigError, PrototypeConfig
from rightyo.tool_events import SpeechEvents

NAMES = Addressing.from_dict({"names": ["Jarvis"], "variants": {}})
ATTEND = ProviderDecision(
    "attend", "system", 0.9, {"attend": 0.9, "ignore": 0.05, "uncertain": 0.05}, "m", "fake", 0.9
)


class Attending:
    def decide(self, state):
        return ATTEND


def gap(following):
    silence = 1200 if following == "none" else 400
    return {"observed": True, "window_ms": 1200, "silence_ms": silence, "following": following}


class PipelineTests(unittest.TestCase):
    """The pipeline only records when another speaker answered; it changes no label."""

    def decide(self, text, following, gate=True):
        runner = ReplayRunner(
            Attending(), addressing=NAMES, post_turn_gaps=True, other_answered_blocks=gate
        )
        return runner.process(turn("t1", 0, 900, text), gap(following)).decision

    def test_an_unnamed_turn_another_speaker_answers_records_when(self):
        decision = self.decide("What time is it?", "different_speaker")
        self.assertEqual((decision.label, decision.answered_at_ms), ("attend", 1300))

    def test_named_quiet_same_speaker_unattributed_or_off_record_nothing(self):
        for text, following, gate in (
            ("Jarvis, what time is it?", "different_speaker", True),
            ("What time is it?", "none", True),
            ("What time is it?", "same_speaker", True),
            ("What time is it?", "unattributed", True),
            ("What time is it?", "different_speaker", False),
        ):
            with self.subTest(text=text, following=following, gate=gate):
                self.assertIsNone(self.decide(text, following, gate).answered_at_ms)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()

    def run_turns(self, answered_at, conversation=None, replies=(), **options):
        """A direct request at 0-900 ms another speaker answered at `answered_at`."""
        self.events.start(SESSION, now_ms=0, conversation=conversation, **options)
        self.events.drain()
        for phase, at in replies:
            self.events.reply(phase, at)
        current = turn("t1", 0, 900, "What time is it?")
        self.events.transcript(current, 900)
        event = decided(current, "attend")
        event = replace(event, decision=replace(event.decision, answered_at_ms=answered_at))
        self.events.decision(event, 2200)
        return [e for e in self.events.drain() if e["type"] in {"attention", "request"}]

    def test_an_answered_turn_is_withheld_and_says_why(self):
        events = self.run_turns(1300)
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertEqual(events[0]["decision"]["label"], "uncertain")
        self.assertIs(events[0]["decision"]["answered_by_other"], True)

    def test_the_assistants_own_playback_is_never_taken_for_an_answer(self):
        events = self.run_turns(1300, replies=(("started", 1200), ("ended", 2000)))
        self.assertEqual([e["type"] for e in events], ["attention", "request"])
        self.assertNotIn("answered_by_other", events[0]["decision"])

    def test_without_a_recorded_answer_nothing_changes(self):
        events = self.run_turns(None)
        self.assertEqual([e["type"] for e in events], ["attention", "request"])

    def test_an_answered_turn_is_never_a_follow_up(self):
        self.events.start(SESSION, now_ms=0, conversation=Conversation(window_ms=10000))
        first = turn("t1", 0, 900, "Jarvis, add milk")
        self.events.transcript(first, 900)
        self.events.decision(decided(first, "attend"), 1000)
        second = turn("t2", 3000, 3800, "and eggs")
        self.events.transcript(second, 3800)
        answered = decided(second, "uncertain", attend=0.8)
        answered = replace(answered, decision=replace(answered.decision, answered_at_ms=4200))
        self.events.decision(answered, 5100)
        requests = [e for e in self.events.drain() if e["type"] == "request"]
        self.assertEqual(len(requests), 1)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        keys = (
            "whisper_executable",
            "whisper_model",
            "diarization_library",
            "diarization_model",
            "microphone_helper",
        )
        self.base = {key: str(asset) for key in keys}
        self.path = Path(directory.name) / "config.json"

    def load(self, decision):
        self.path.write_text(json.dumps({**self.base, "decision": decision}))
        return PrototypeConfig.load(self.path)

    def test_the_decision_section_turns_the_gate_on(self):
        jev = {"provider": "jev", "allow_hosted": True}
        self.assertFalse(self.load(jev).decision_other_answered_blocks)
        self.assertTrue(
            self.load({**jev, "other_answered_blocks": True}).decision_other_answered_blocks
        )
        with self.assertRaisesRegex(DecisionConfigError, "reply_wait_ms"):
            self.path.write_text(
                json.dumps(
                    {
                        **self.base,
                        "decision": {**jev, "other_answered_blocks": True},
                        "turns": {"reply_wait_ms": 0},
                    }
                )
            )
            PrototypeConfig.load(self.path)
        for bad in (1, "true", None):
            with (
                self.subTest(bad=bad),
                self.assertRaisesRegex(DecisionConfigError, "other_answered"),
            ):
                self.load({**jev, "other_answered_blocks": bad})


if __name__ == "__main__":
    unittest.main()
