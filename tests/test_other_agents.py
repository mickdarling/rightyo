"""Turns addressed to another agent (#161), such as the owner's dot "Puck", are not ours."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_conversation import MODE, SESSION, decided, turn

from rightyo.contracts import Addressing, ContractError
from rightyo.prototype import PrototypeConfig
from rightyo.tool_events import SpeechEvents

OURS = Addressing(names=("Jarvis",))
PUCK = Addressing(names=("Puck",), variants=(("Puck", ("POC", "Puk")),))


class OtherAgentTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()
        self.events.start(SESSION, now_ms=0, addressing=OURS, conversation=MODE, other_agents=PUCK)
        self.events.drain()

    def say(self, current, label, **options):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return [e for e in self.events.drain() if e["type"] in {"attention", "request"}]

    def test_a_turn_naming_another_agent_is_never_a_request(self):
        events = self.say(turn("p", 0, 900, "Alright, Puck, let's draft the reply."), "attend")
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertEqual(events[0]["decision"]["label"], "ignore")

    def test_an_asr_variant_counts(self):
        events = self.say(turn("p", 0, 900, "POC, what about the plan?"), "attend")
        self.assertEqual([e["type"] for e in events], ["attention"])

    def test_naming_this_assistant_too_keeps_it_ours(self):
        events = self.say(turn("j", 0, 900, "Jarvis, ask Puck to send it."), "attend")
        self.assertEqual([e["type"] for e in events], ["attention", "request"])

    def test_an_engaged_speakers_turn_to_another_agent_is_not_a_follow_up(self):
        self.say(turn("j", 0, 900, "Jarvis, what time is it?"), "attend")
        events = self.say(turn("p", 2000, 2900, "Puck, and tomorrow?"), "uncertain", attend=0.9)
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertNotIn("follow_up", events[0]["decision"])

    def test_turning_to_another_agent_ends_this_conversation(self):
        self.say(turn("j", 0, 900, "Jarvis, what time is it?"), "attend")
        self.say(turn("p", 2000, 2900, "Puck, draft a reply."), "attend")
        # The next unnamed remark is no longer a follow-up to Jarvis.
        events = self.say(turn("c", 4000, 4900, "And make it concise."), "uncertain", attend=0.9)
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertNotIn("follow_up", events[0]["decision"])

    def test_start_refuses_a_foreign_object(self):
        with self.assertRaises(ContractError):
            SpeechEvents().start(SESSION, other_agents={"names": ["Puck"]})


class OtherAgentConfigTests(unittest.TestCase):
    def test_the_config_accepts_other_agents_in_the_addressing_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset"
            asset.touch()
            base = {
                key: str(asset)
                for key in (
                    "whisper_executable",
                    "whisper_model",
                    "diarization_library",
                    "diarization_model",
                    "microphone_helper",
                )
            }
            config = root / "config.json"
            puck = {"names": ["Puck"], "variants": {"Puck": ["POC"]}}
            config.write_text(json.dumps({**base, "other_agents": puck}))
            self.assertEqual(PrototypeConfig.load(config).other_agents.names, ("Puck",))
            config.write_text(json.dumps(base))
            self.assertIsNone(PrototypeConfig.load(config).other_agents)
            config.write_text(json.dumps({**base, "other_agents": {"names": []}}))
            with self.assertRaises(Exception):
                PrototypeConfig.load(config)


if __name__ == "__main__":
    unittest.main()
