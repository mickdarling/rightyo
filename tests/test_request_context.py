"""Decision states can carry the latest request formed (rightyo#199); synthetic only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_answered_by_other import Attending
from test_conversation import SESSION, decided, turn

from rightyo.contracts import ContractError, Conversation
from rightyo.pipeline import REQUEST_CONTEXT_CHARS, REQUEST_CONTEXT_MS, ReplayRunner
from rightyo.prototype import DecisionConfigError, PrototypeConfig
from rightyo.providers import REQUEST_GUIDANCE, build_request
from rightyo.tool_events import SpeechEvents


class RunnerTests(unittest.TestCase):
    def state(self, runner, start):
        return runner._state(turn("t2", start, start + 800, "and the other one"))

    def test_the_latest_request_and_how_long_ago_reach_the_state_and_guidance(self):
        runner = ReplayRunner(Attending(), request_context=True)
        runner.note_request(900, "Jarvis, add milk to the list")
        state = self.state(runner, 30900)
        self.assertEqual(
            state["assistant_request"], {"ms_before": 30000, "text": "Jarvis, add milk to the list"}
        )
        request = build_request(state)
        self.assertIn(REQUEST_GUIDANCE, request["questions"]["attention"]["instructions"])
        self.assertEqual(request["state"]["assistant_request"]["ms_before"], 30000)

    def test_it_is_bounded_in_time_and_length_and_off_by_default(self):
        runner = ReplayRunner(Attending(), request_context=True)
        runner.note_request(0, "x" * 1000)
        self.assertEqual(
            len(self.state(runner, 100)["assistant_request"]["text"]), REQUEST_CONTEXT_CHARS
        )
        self.assertNotIn("assistant_request", self.state(runner, REQUEST_CONTEXT_MS + 1))
        off = ReplayRunner(Attending())
        off.note_request(0, "Jarvis, add milk")
        self.assertNotIn("assistant_request", self.state(off, 100))
        self.assertNotIn(
            REQUEST_GUIDANCE,
            build_request(self.state(off, 100))["questions"]["attention"]["instructions"],
        )

    def test_an_older_request_never_replaces_a_newer_one(self):
        runner = ReplayRunner(Attending(), request_context=True)
        runner.note_request(5000, "newer")
        runner.note_request(1000, "older")
        self.assertEqual(self.state(runner, 6000)["assistant_request"]["text"], "newer")
        with self.assertRaises(ContractError):
            runner.note_request("5000", "x")


class EventsTests(unittest.TestCase):
    def test_the_events_layer_remembers_the_latest_request_formed(self):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=Conversation(window_ms=10000))
        self.assertIsNone(events.last_request)
        first = turn("t1", 0, 900, "Jarvis, add milk")
        events.transcript(first, 900)
        events.decision(decided(first, "attend"), 1000)
        self.assertEqual(events.last_request, (900, "Jarvis, add milk"))
        ignored = turn("t2", 2000, 2800, "good boy")
        events.transcript(ignored, 2800)
        events.decision(decided(ignored, "ignore"), 2900)
        self.assertEqual(events.last_request, (900, "Jarvis, add milk"))


class ConfigTests(unittest.TestCase):
    def test_the_decision_section_turns_it_on(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset"
            asset.touch()
            keys = (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
            path = Path(directory) / "config.json"
            jev = {"provider": "jev", "allow_hosted": True}

            def load(decision):
                path.write_text(json.dumps({**{k: str(asset) for k in keys}, "decision": decision}))
                return PrototypeConfig.load(path)

            self.assertFalse(load(jev).decision_request_context)
            self.assertTrue(load({**jev, "request_context": True}).decision_request_context)
            with self.assertRaisesRegex(DecisionConfigError, "request_context"):
                load({**jev, "request_context": 1})


if __name__ == "__main__":
    unittest.main()
