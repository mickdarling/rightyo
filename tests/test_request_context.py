"""Decision states can carry the latest request formed (rightyo#199); synthetic only."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test_answered_by_other import Attending
from test_conversation import SESSION, decided, turn

from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.pipeline import REQUEST_CONTEXT_CHARS, REQUEST_CONTEXT_MS, ReplayRunner
from rightyo.prototype import DecisionConfigError, PrototypeConfig
from rightyo.providers import REQUEST_GUIDANCE, build_request
from rightyo.tool_events import SpeechEvents


class RunnerTests(unittest.TestCase):
    def state(self, runner, start):
        return runner._state(turn("t2", start, start + 800, "and the other one"))

    def test_the_latest_request_and_how_long_ago_reach_the_state_and_guidance(self):
        runner = ReplayRunner(Attending(), request_context=True)
        runner.note_request((900, "Jarvis, add milk to the list", "u1", "Speaker A"))
        state = self.state(runner, 30900)
        self.assertEqual(
            state["assistant_request"],
            {"ms_before": 30000, "text": "Jarvis, add milk to the list", "speaker_id": "Speaker A"},
        )
        request = build_request(state)
        self.assertIn(REQUEST_GUIDANCE, request["questions"]["attention"]["instructions"])
        self.assertEqual(request["state"]["assistant_request"]["ms_before"], 30000)

    def test_it_is_bounded_in_time_and_length_and_off_by_default(self):
        runner = ReplayRunner(Attending(), request_context=True)
        runner.note_request((0, "x" * 1000, "u1", None))
        self.assertEqual(
            len(self.state(runner, 100)["assistant_request"]["text"]), REQUEST_CONTEXT_CHARS
        )
        self.assertNotIn("assistant_request", self.state(runner, REQUEST_CONTEXT_MS + 1))
        off = ReplayRunner(Attending())
        off.note_request((0, "Jarvis, add milk", "u1", None))
        self.assertNotIn("assistant_request", self.state(off, 100))
        self.assertNotIn(
            REQUEST_GUIDANCE,
            build_request(self.state(off, 100))["questions"]["attention"]["instructions"],
        )

    def test_an_older_request_never_replaces_a_newer_one(self):
        runner = ReplayRunner(Attending(), request_context=True)
        runner.note_request((5000, "newer", "u2", None))
        runner.note_request((1000, "older", "u1", None))
        self.assertEqual(self.state(runner, 6000)["assistant_request"]["text"], "newer")
        with self.assertRaises(ContractError):
            runner.note_request(("5000", "x", "u1", None))


class ForgettingTests(unittest.TestCase):
    def test_no_decision_state_after_a_forget_names_a_label_from_before_it(self):
        # #242 round 25 (hailing-station #735 S-1): once the speakers are forgotten, the
        # runner's past turns, participants, a turn heard before the forget but decided after
        # it, and the latest request carry no label from before it; later speech keeps its own.
        runner = ReplayRunner(Attending(), request_context=True)
        runner.process(turn("t1", 0, 900, "Jarvis, add milk"))
        runner.note_request((900, "Jarvis, add milk", "t1", "Speaker A"))
        stale = turn("t2", 1000, 1900, "and the other one")
        runner.forget_speakers(2000)
        for current in (stale, turn("t3", 3000, 3800, "x", "Speaker C")):
            state = runner._state(current)
            text = json.dumps(state)
            self.assertNotIn("Speaker A", text)
            self.assertEqual([t["speaker_id"] for t in state["past_turns"]], [None])
            self.assertIsNone(state["assistant_request"]["speaker_id"])
            if current is stale:
                self.assertIsNone(state["current_turn"]["speaker_id"])
                self.assertEqual(state["known_participants"], [])
            else:
                self.assertEqual(state["current_turn"]["speaker_id"], "Speaker C")
                self.assertEqual(state["known_participants"], ["Speaker C"])
        # A later forget never moves the boundary back; a restart clears it.
        runner.forget_speakers(500)
        self.assertIsNone(runner._state(stale)["current_turn"]["speaker_id"])
        runner.restart("next-session")
        self.assertEqual(runner._state(stale)["current_turn"]["speaker_id"], "Speaker A")

    def test_none_clear_restart_and_expired_retention_forget_it(self):
        from rightyo.memory import TranscriptMemory

        memory = TranscriptMemory(retention_ms=60000)
        runner = ReplayRunner(Attending(), memory=memory, request_context=True)
        first = turn("t1", 0, 900, "Jarvis, add milk")
        runner.process(first)
        runner.note_request((900, "Jarvis, add milk", first.utterance_id, "Speaker A"))
        later = turn("t2", 30000, 30800, "and eggs")
        self.assertIn("assistant_request", runner._state(later))
        runner.expire(90000)
        self.assertNotIn("assistant_request", runner._state(turn("t3", 90000, 90800, "x")))
        plain = ReplayRunner(Attending(), request_context=True)
        for forget in (lambda: plain.note_request(None), lambda: plain.clear()):
            plain.note_request((900, "Jarvis, add milk", "u1", None))
            self.assertIn("assistant_request", plain._state(later))
            forget()
            self.assertNotIn("assistant_request", plain._state(later))


class EventsTests(unittest.TestCase):
    def test_the_events_layer_remembers_the_latest_request_formed(self):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=Conversation(window_ms=10000))
        self.assertIsNone(events.last_request)
        first = turn("t1", 0, 900, "Jarvis, add milk")
        events.transcript(first, 900)
        events.decision(decided(first, "attend"), 1000)
        self.assertEqual(events.last_request, (900, "Jarvis, add milk", "t1", "Speaker A"))
        ignored = turn("t2", 2000, 2800, "good boy")
        events.transcript(ignored, 2800)
        events.decision(decided(ignored, "ignore"), 2900)
        self.assertEqual(events.last_request, (900, "Jarvis, add milk", "t1", "Speaker A"))

    def request(self, events, current, at):
        events.transcript(current, current.end_ms)
        events.decision(decided(current, "attend"), at)

    def test_a_short_tag_on_the_latest_request_leaves_it_in_place(self):
        events = SpeechEvents()
        names = Addressing.from_dict({"names": ["Jarvis"], "variants": {}})
        events.start(
            SESSION, now_ms=0, conversation=Conversation(window_ms=60000), addressing=names
        )
        self.request(events, turn("t1", 0, 900, "Jarvis, is the plan forty dollars"), 1000)
        self.request(events, turn("t2", 1500, 1900, "right?", speaker=None), 2000)
        self.assertEqual(events.last_request[1], "Jarvis, is the plan forty dollars")
        # A longer request, a later one or another known speaker's replaces it.
        self.request(events, turn("t3", 3000, 3900, "Jarvis, add eggs too"), 4000)
        self.assertEqual(events.last_request[1], "Jarvis, add eggs too")
        # A named short request inside the tag window is a request of its own.
        self.request(events, turn("t4", 5000, 5400, "Jarvis, weather"), 5500)
        self.assertEqual(events.last_request[1], "Jarvis, weather")
        self.request(events, turn("t6", 20000, 20400, "louder"), 20500)
        self.assertEqual(events.last_request[1], "louder")
        self.request(events, turn("t5", 21000, 21400, "Jarvis, skip", speaker="Speaker B"), 21500)
        self.assertEqual(events.last_request[1], "Jarvis, skip")

    def test_a_dismissal_forgets_it(self):
        from rightyo.contracts import Dismissal

        events = SpeechEvents()
        events.start(
            SESSION, now_ms=0, conversation=Conversation(window_ms=10000), dismissal=Dismissal()
        )
        first = turn("t1", 0, 900, "Jarvis, add milk")
        events.transcript(first, 900)
        events.decision(decided(first, "attend"), 1000)
        self.assertIsNotNone(events.last_request)
        stop = turn("t2", 2000, 2500, "never mind")
        events.transcript(stop, 2500)
        self.assertIsNone(events.last_request)

    def test_talk_to_another_person_keeps_it(self):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=Conversation(window_ms=60000))
        first = turn("t1", 0, 900, "Jarvis, add milk")
        events.transcript(first, 900)
        events.decision(decided(first, "attend"), 1000)
        aside = turn("t2", 3000, 3800, "do we have eggs?")
        events.transcript(aside, 3800)
        events.decision(decided(aside, "ignore", recipient="other_human"), 3900)
        kinds = [(e["type"], e.get("reason")) for e in events.drain()]
        self.assertIn(("conversation", "other_human"), kinds)
        self.assertEqual(events.last_request[1], "Jarvis, add milk")


class ControllerTests(unittest.TestCase):
    def test_listen_hands_each_formed_request_to_the_next_decision(self):
        from unittest.mock import patch

        import test_tool

        from rightyo.providers import MockProvider

        harness = test_tool.ConfigDecisionTests(
            "test_config_jev_with_consent_constructs_the_jev_provider"
        )
        harness.setUp()
        self.addCleanup(harness.doCleanups)
        harness.write({"provider": "jev", "allow_hosted": True, "request_context": True})
        states = []
        original = MockProvider.decide

        def recording(provider, state):
            states.append(state)
            return original(provider, state)

        with patch.object(MockProvider, "decide", recording):
            harness.run_listen()
        self.assertGreaterEqual(len(states), 2)
        self.assertNotIn("assistant_request", states[0])
        self.assertEqual(states[1]["assistant_request"]["text"], "Rightyo, tell me what happened.")


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
