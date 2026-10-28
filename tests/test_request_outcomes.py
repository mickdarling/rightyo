"""Authored control/replay regressions for #262; no capture, model or real speech."""

import copy
import io
import json
import os
import threading
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from test_conversation import SESSION, decided, turn

from rightyo.contracts import (
    Acknowledgement,
    ContractError,
    Conversation,
    Dismissal,
    SpeakerPriority,
)
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.request_outcomes import validate
from rightyo.tool import _apply_control, _read_control
from rightyo.tool_events import SpeechEvents


def outcome(name="a", state="accepted", **extra):
    return {
        "version": 1,
        "session_id": SESSION,
        "request_id": SESSION + ":" + name,
        "state": state,
        **extra,
    }


class RequestOutcomeTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents(request_control=True)
        self.events.start(
            SESSION,
            now_ms=0,
            conversation=Conversation(window_ms=10000),
            acknowledgement=Acknowledgement(),
        )
        self.started = self.events.drain()[0]

    def say(self, name, at=0, speaker="Speaker A"):
        current = turn(name, at, at + 900, "Jarvis, what time is it?", speaker=speaker)
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, "attend"), current.end_ms + 10)
        return self.events.drain()

    def apply(self, name="a", state="accepted", at=1000):
        return self.events.request_outcome(outcome(name, state), at)

    def test_shared_authored_contract_fixture(self):
        fixture = json.loads(
            (Path(__file__).parents[1] / "examples/request-outcomes-v1.json").read_text()
        )
        base = fixture["accepted"]["request_outcome"]
        for state in fixture["states"]:
            self.assertEqual(validate({**base, "state": state})["state"], state)
        for value in fixture["rejected"]:
            with self.assertRaises(ContractError):
                validate(value)

    def test_advertisement_is_opt_in_and_versioned(self):
        self.assertEqual(self.started["request_outcomes"], {"version": 1})
        old = SpeechEvents()
        old.start(SESSION, now_ms=0)
        self.assertNotIn("request_outcomes", old.drain()[0])
        self.assertFalse(old.request_outcome(outcome(), 0))

    def test_failure_clears_only_exact_pending_and_new_request_is_admitted(self):
        self.say("a")
        self.assertTrue(self.apply())
        self.assertTrue(self.apply(state="timed_out"))
        self.assertIsNone(self.events._pending_ack)
        self.assertIsNone(self.events._engaged)
        self.assertIsNone(self.events._dormant)
        self.assertIsNone(self.events.last_request)
        self.assertFalse(self.apply(state="reply_started"))
        events = self.say("b", at=2000)
        self.assertIn("request", [e["type"] for e in events])
        self.assertEqual(self.events._engaged["request_id"], SESSION + ":b")

    def test_old_outcome_does_not_clear_competing_new_request(self):
        self.say("a")
        self.apply()
        self.apply(state="reply_started", at=1100)
        self.say("b", at=2000)
        self.apply("b", at=3000)
        before = copy.deepcopy(self.events._engaged)
        pending = copy.deepcopy(self.events._pending_ack)
        self.assertTrue(self.apply(state="failed", at=4000))
        self.assertEqual(self.events._engaged, before)
        self.assertEqual(self.events._pending_ack, pending)
        self.assertFalse(self.apply(state="reply_ended", at=4500))
        self.assertEqual(self.events._engaged, before)

    def test_duplicates_do_not_extend_reply_window_and_ack_clip_is_not_answer(self):
        self.say("a")
        self.apply()
        self.events.reply("started", 1100)
        self.events.reply("ended", 1200)
        self.assertIsNotNone(self.events._pending_ack)
        self.assertTrue(self.apply(state="reply_started", at=1300))
        self.assertIsNone(self.events._pending_ack)
        self.assertTrue(self.apply(state="reply_ended", at=1400))
        before = copy.deepcopy(self.events._engaged)
        self.assertTrue(self.apply(state="reply_ended", at=9000))
        self.assertEqual(self.events._engaged, before)
        self.assertFalse(self.apply(state="accepted"))

    def test_foreign_unknown_and_restarted_sessions_never_ack(self):
        self.say("a")
        before = copy.deepcopy(self.events._engaged)
        wrong = outcome(session_id="foreign", request_id="foreign:a")
        self.assertFalse(self.events.request_outcome(wrong, 1000))
        self.assertFalse(self.apply("missing"))
        self.assertEqual(self.events._engaged, before)
        self.assertFalse(any(e["type"] == "request_outcome" for e in self.events.drain()))
        self.events.end(now_ms=2000)
        self.assertFalse(self.apply())

    def test_privacy_and_forget_revoke_old_request_identity(self):
        self.say("a")
        self.apply()
        self.events.duck(True, now_ms=1000)
        self.assertFalse(self.apply(state="reply_ended"))
        self.events.duck(False, now_ms=2000)
        self.say("b", 3000)
        self.events.forget_speakers(now_ms=4000)
        self.assertFalse(self.apply("b", "reply_started", 5000))
        self.assertIsNone(self.events._engaged)

    def test_terminal_from_pending_also_revokes_unscoped_late_reply_authority(self):
        self.say("a")
        self.assertTrue(self.apply(state="timed_out"))
        self.say("b", at=2000)
        before = copy.deepcopy(self.events._engaged)
        pending = copy.deepcopy(self.events._pending_ack)
        self.events.reply("started", 4000)
        self.events.reply("ended", 5000, asked=True)
        self.assertEqual(self.events._engaged, before)
        self.assertEqual(self.events._pending_ack, pending)
        self.assertIsNone(self.events._asked_at)

    def test_old_scoped_reply_end_closes_only_its_echo_span(self):
        self.say("a")
        self.apply()
        self.apply(state="reply_started", at=1100)
        span_a = self.events._playback_scopes[SESSION + ":a"]
        self.say("b", at=2000)
        self.apply("b", at=3000)
        self.apply("b", "reply_started", at=3100)
        span_b = self.events._playback_scopes[SESSION + ":b"]
        before = copy.deepcopy(self.events._engaged)
        self.assertTrue(self.apply(state="reply_ended", at=4000))
        self.assertEqual(span_a, [1100, 4000])
        self.assertEqual(span_b, [3100, None])
        self.assertEqual(self.events._engaged, before)

    def test_old_decision_never_replaces_newer_request_identity(self):
        self.say("a")
        middle = turn("c", 1000, 1900, "Jarvis, another question?")
        newest = turn("b", 2000, 2900, "Jarvis, the newest question?")
        self.events.transcript(middle, 1900)
        self.events.transcript(newest, 2900)
        self.events.decision(decided(newest, "attend"), 3000)
        self.events.decision(decided(middle, "attend"), 3100)
        before = copy.deepcopy(self.events._engaged)
        self.assertEqual(before["request_id"], SESSION + ":b")
        self.assertTrue(self.apply("c", "failed", at=3200))
        self.assertEqual(self.events._engaged, before)

    def test_expired_and_ended_sessions_release_request_control_state(self):
        self.say("a")
        self.events.expire(self.events.retention_ms + 2000)
        self.assertFalse(self.apply(state="reply_started"))
        self.assertEqual(self.events._request_outcomes, {})
        self.events.end(now_ms=self.events.retention_ms + 3000)
        self.assertEqual(self.events._request_outcome_ends, {})

    def test_control_enforces_retention_without_a_prior_timer_tick(self):
        self.say("a")
        self.apply()
        self.assertFalse(self.apply(state="reply_started", at=self.events.retention_ms + 2000))
        self.assertEqual(self.events._request_outcomes, {})
        self.assertFalse(
            any(
                event["type"] == "request_outcome" and event["state"] == "reply_started"
                for event in self.events.drain()
            )
        )

    def test_spoken_dismissal_retires_original_control_identity(self):
        self.events = SpeechEvents(request_control=True)
        self.events.start(
            SESSION, now_ms=0, conversation=Conversation(window_ms=10000), dismissal=Dismissal()
        )
        self.events.drain()
        self.say("a")
        self.apply()
        closing = turn("stop", 2000, 2900, "Never mind.")
        self.events.transcript(closing, 2900)
        events = self.events.drain()
        self.assertIn("dismiss", [event["type"] for event in events])
        self.assertFalse(self.apply(state="reply_started", at=3000))
        self.assertFalse(self.apply(state="cancelled", at=3100))

    def test_owner_override_retires_superseded_request_controls(self):
        self.events = SpeechEvents(request_control=True)
        self.events.start(
            SESSION, priority=ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",)))
        )
        self.events.drain()
        self.say("guest", speaker="Speaker B")
        self.assertTrue(self.apply("guest"))
        self.say("owner", at=2000)
        self.assertFalse(self.apply("guest", "reply_started", 3000))
        self.assertTrue(self.apply("owner", at=3000))

    def test_controller_forwards_only_new_scoped_answer_to_runner(self):
        self.say("a")
        config = PrototypeConfig(None, None, None, None, None)
        controller = PrototypeController(config, event_publisher=self.events)
        self.addCleanup(controller.close)
        runner = SimpleNamespace(request_context=True, note_request=Mock(), note_reply=Mock())
        controller._runner = runner
        controller._event_terminal = False
        controller._phase = "replaying"
        controller._now_ms = lambda: 2000
        try:
            controller.request_outcome(outcome())
            controller.reply("started")
            controller.reply("ended")
            runner.note_reply.assert_not_called()
            controller.request_outcome(outcome(state="reply_started"))
            runner.note_reply.assert_called_once_with("started", 2000)
            controller.request_outcome(outcome(state="reply_started"))
            controller.request_outcome(outcome("foreign", state="reply_ended"))
            self.assertEqual(runner.note_reply.call_count, 1)
            controller.request_outcome(outcome(state="reply_ended"))
            self.assertEqual(runner.note_reply.call_count, 2)
            self.assertIsNotNone(self.events._engaged)
        finally:
            controller._runner = None
            controller._phase = "idle"

    def test_malformed_envelopes_refuse_without_mutation(self):
        self.say("a")
        for value in [
            outcome(version=True),
            outcome(version=2),
            outcome(state="unknown"),
            outcome(extra=1),
            outcome(asked=True),
            outcome(request_id="other:a"),
            outcome(state="reply_ended", asked=1),
        ]:
            with self.subTest(value=value), self.assertRaises(ContractError):
                validate(value)
        self.assertEqual(self.events._request_outcomes[SESSION + ":a"], "pending")

    def test_real_control_fd_is_bounded_and_continues_after_invalid_line(self):
        calls = []

        class Controller:
            def request_outcome(self, value):
                calls.append(validate(value))

        reader, writer = os.pipe()
        thread = threading.Thread(target=_read_control, args=(reader, Controller()))
        with redirect_stderr(io.StringIO()):
            thread.start()
            os.write(writer, b"x" * 600 + b"\n")
            os.write(writer, json.dumps({"request_outcome": outcome()}).encode() + b"\n")
            os.close(writer)
            thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(calls, [outcome()])
        with redirect_stderr(io.StringIO()):
            _apply_control(b'{"request_outcome":{},"request_outcome":{}}', Controller())
        self.assertEqual(calls, [outcome()])
