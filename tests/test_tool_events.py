"""Synthetic producer contract tests; no native models, capture, or hosted calls."""

import json
import unittest

from rightyo.contracts import ContractError, DecisionEvent, ProviderDecision, Turn
from rightyo.tool_events import SpeechEvents


def turn(name="request", start=1000, end=2000, text="Rightyo, check our discussion.", **extra):
    return Turn(
        session_id=extra.pop("session_id", "tool-demo"),
        utterance_id=name,
        revision=1,
        start_ms=start,
        end_ms=end,
        text=text,
        speaker_id="Speaker A",
        finalized=True,
        overlap=False,
        recognizer_id="authored-fixture",
        provenance="synthetic",
        speaker_provenance="authored-fixture",
        **extra,
    )


def decision(current, label="attend", recipient="system"):
    return DecisionEvent(
        current,
        ProviderDecision(
            label,
            recipient,
            1.0,
            {k: float(k == label) for k in ("attend", "ignore", "uncertain")},
            "mock-v1",
            "mock",
            1.0,
        ),
        current.revision,
        0.0,
        0.0,
    )


class SpeechEventsTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()
        self.events.start("tool-demo")
        self.started = self.events.drain()[0]

    def test_capabilities_truthfully_advertise_finalized_activation(self):
        self.assertEqual(self.started["schema_version"], 1)
        self.assertEqual(self.started["capabilities"]["activation"], "finalized-turn")
        self.assertFalse(self.started["capabilities"]["partials"])

    def test_late_decision_freezes_prior_context_preserves_complete_trigger(self):
        previous = turn("discussion", 0, 500, "The review is scheduled for Friday.")
        request = turn()
        future = turn("later", 2100, 3000, "Unrelated later conversation.")
        self.events.transcript(previous, 500, expect_decision=False)
        self.events.transcript(request, 2000)
        self.events.transcript(future, 3000, expect_decision=False)
        self.events.drain()
        self.events.decision(decision(request), 3500)
        attention, result = self.events.drain()
        self.assertEqual(attention["request_id"], result["request_id"])
        self.assertEqual(result["turn"], request.to_dict())
        self.assertEqual(result["context"]["turns"], [previous.to_dict()])
        self.assertEqual(result["decision_at_ms"], 3500)
        self.assertEqual(result["emitted_at_ms"], 3500)
        self.assertNotIn("target", result)

    def test_ignore_uncertain_and_conflicting_recipient_never_emit_request(self):
        for index, (label, recipient) in enumerate(
            (("ignore", "other_human"), ("uncertain", "unknown"), ("attend", "other_human"))
        ):
            current = turn(str(index), index * 2000, index * 2000 + 1000)
            self.events.transcript(current, current.end_ms)
            self.events.decision(decision(current, label, recipient), current.end_ms)
            self.assertEqual([e["type"] for e in self.events.drain()], ["transcript", "attention"])

    def test_duplicates_emit_once_and_conflicting_final_rejected(self):
        request = turn()
        self.events.transcript(request, 2000)
        self.events.transcript(request, 2000)
        self.events.decision(decision(request), 2500)
        self.events.decision(decision(request), 2500)
        self.assertEqual(
            [e["type"] for e in self.events.drain()], ["transcript", "attention", "request"]
        )
        with self.assertRaises(ContractError):
            self.events.transcript(turn(text="Changed words"), 3000)

    def test_cancel_drops_queued_requests_and_late_decisions(self):
        request = turn()
        self.events.transcript(request, 2000)
        self.events.decision(decision(request), 2500)
        self.events.end("cancelled", 2600)
        self.events.decision(decision(request), 3000)
        self.assertEqual([e["phase"] for e in self.events.drain()], ["cancelled"])
        self.assertEqual(self.events._memory.snapshot(3000)["turns"], [])
        self.assertEqual(self.events._pending, {})

    def test_normal_end_preserves_order_then_is_idempotent(self):
        request = turn()
        self.events.transcript(request, 2000)
        self.events.decision(decision(request), 2500)
        self.events.end("stopped", 2600)
        result = self.events.drain()
        self.assertEqual(
            [e["type"] for e in result], ["transcript", "attention", "request", "session"]
        )
        self.events.end("cancelled", 3000)
        self.assertEqual(self.events.drain(), [])

    def test_terminal_skip_counts_are_optional_and_validated(self):
        self.events.end("stopped", 2600)
        terminal = self.events.drain()[-1]
        self.assertNotIn("skipped_segments", terminal)
        self.assertNotIn("skipped_utterances", terminal)
        for name in ("skipped_segments", "skipped_utterances"):
            for invalid in (0, -1, 1.0, True, "3"):
                with self.subTest(name=name, value=invalid):
                    events = SpeechEvents()
                    events.start("skip-session")
                    with self.assertRaisesRegex(ContractError, name):
                        events.end("stopped", 10, **{name: invalid})
        events = SpeechEvents()
        events.start("skip-session")
        events.end("cancelled", 10, skipped_segments=3, skipped_utterances=2)
        terminal = events.drain()[-1]
        self.assertEqual((terminal["skipped_segments"], terminal["skipped_utterances"]), (3, 2))

    def test_expiry_removes_stale_decisions_queued_text_and_prior_context(self):
        previous = turn("previous", 0, 500, "Old conversation.")
        request = turn()
        self.events.transcript(previous, 500, expect_decision=False)
        self.events.transcript(request, 2000)
        self.events.decision(decision(request), 301000)
        result = self.events.drain()[-1]
        self.assertEqual(result["context"]["turns"], [])
        self.events.expire(302000)
        self.assertEqual(self.events.drain(), [])

    def test_expiring_frozen_and_queued_context_reclaims_serialized_budget(self):
        previous = turn("previous", 0, 500, "👋" * 4000)
        request = turn()
        self.events.transcript(previous, 500, expect_decision=False)
        self.events.transcript(request, 2000)
        before = self.events._pending_bytes
        self.events.expire(300600)
        self.assertLess(self.events._pending_bytes, before - 3900)
        next_turn = turn("next", 300601, 300700)
        from unittest.mock import patch

        with patch("rightyo.tool_events.MAX_PENDING_BYTES", before):
            self.events.transcript(next_turn, 300700)
        self.events.decision(decision(request), 300701)
        self.events.decision(decision(next_turn), 300701)
        actual_size = sum(
            len(json.dumps(payload, ensure_ascii=True, allow_nan=False)) + 1
            for payload, _ in self.events._queue
        )
        self.assertEqual(self.events._queue_bytes, actual_size)
        self.assertEqual(self.events._pending_bytes, 0)

    def test_unicode_context_expansion_is_bounded_before_request_emission(self):
        for index in range(64):
            previous = turn(str(index), index * 1000, index * 1000 + 500, "👋" * 4000)
            self.events.transcript(previous, previous.end_ms, expect_decision=False)
            self.events.drain()
        with self.assertRaisesRegex(ContractError, "pending attention context budget"):
            self.events.transcript(turn(start=65000, end=66000), 66000)
        self.events.end("error", 66000, "consumer-backlog")
        self.assertEqual([e["type"] for e in self.events.drain()], ["session"])
        self.assertEqual(self.events._pending_bytes, 0)

    def test_unicode_wire_queue_and_single_event_limits_fail_closed(self):
        from unittest.mock import patch

        for limit in ("MAX_QUEUE_BYTES", "MAX_EVENT_BYTES"):
            events = SpeechEvents()
            events.start("tool-demo")
            events.drain()
            with patch("rightyo.tool_events." + limit, 60000):
                previous = turn("first", 0, 500, "👋" * 4000)
                events.transcript(previous, 500, expect_decision=False)
                if limit == "MAX_EVENT_BYTES":
                    events.drain()
                    request = turn(start=1000, end=2000, text="👋" * 4000)
                    events.transcript(request, 2000)
                    events.drain()
                    with self.assertRaisesRegex(ContractError, "consumer backlog"):
                        events.decision(decision(request), 2100)
                else:
                    with self.assertRaisesRegex(ContractError, "consumer backlog"):
                        events.transcript(turn(text="👋" * 4000), 2000, expect_decision=False)
            events.end("error", 2100, "consumer-backlog")
            self.assertEqual([e["type"] for e in events.drain()], ["session"])
            self.assertEqual(events._pending, {})

    def test_expired_undecided_turn_never_activates(self):
        request = turn()
        self.events.transcript(request, 2000)
        self.events.expire(302000)
        self.events.decision(decision(request), 302001)
        self.assertEqual(self.events.drain(), [])

    def test_queue_overflow_fails_closed_without_retained_request(self):
        with self.assertRaises(ContractError):
            SpeechEvents(max_pending=4)
        events = SpeechEvents(max_pending=5)
        events.start("tool-demo")
        events.drain()
        for index in range(3):
            previous = turn(f"chat-{index}", index * 200, index * 200 + 100, "Hello.")
            events.transcript(previous, previous.end_ms, expect_decision=False)
        request = turn()
        events.transcript(request, 2000)
        with self.assertRaises(ContractError):
            events.decision(decision(request), 2500)
        events.end("error", 2600, "consumer-backlog")
        self.assertEqual([e["type"] for e in events.drain()], ["session"])
        self.assertEqual(events._pending, {})

    def test_transcript_only_sessions_do_not_hold_unused_decision_context(self):
        for index in range(80):
            current = turn(str(index), index * 1000, index * 1000 + 500)
            self.events.transcript(current, current.end_ms, expect_decision=False)
            self.events.drain()
        self.assertEqual(self.events._pending, {})

    def test_disabled_attention_and_retired_session_identity_are_enforced(self):
        events = SpeechEvents()
        events.start("tool-demo", attention_enabled=False)
        request = turn()
        events.transcript(request, 2000)
        events.decision(decision(request), 2500)
        result = events.drain()
        self.assertEqual(result[0]["capabilities"]["activation"], "disabled")
        self.assertEqual([e["type"] for e in result], ["session", "transcript"])
        self.assertEqual(events._pending, {})
        events.end("stopped", 3000)
        events.drain()
        events.start("another-session")
        events.end("stopped", 1000)
        events.drain()
        with self.assertRaises(ContractError):
            events.start("tool-demo")

    def test_cross_session_partial_or_invalid_time_rejected(self):
        with self.assertRaises(ContractError):
            self.events.transcript(turn(session_id="another"), 2000)
        with self.assertRaises(ContractError):
            self.events.transcript(turn(), True)
        with self.assertRaises(ContractError):
            self.events.start("tool-demo")

    def test_detached_drain_cannot_modify_frozen_context(self):
        previous = turn("previous", 0, 500)
        request = turn()
        self.events.transcript(previous, 500, expect_decision=False)
        self.events.transcript(request, 2000)
        drained = self.events.drain()
        drained[0]["turn"]["text"] = "Modified outside state"
        self.events.decision(decision(request), 2500)
        self.assertEqual(self.events.drain()[-1]["context"]["turns"][0]["text"], previous.text)

    def test_shared_authored_fixture_matches_producer(self):
        from pathlib import Path

        events = SpeechEvents()
        events.start("tool-demo")
        previous = turn("discussion", 0, 500, "The review is scheduled for Friday.")
        request = turn()
        events.transcript(previous, 500)
        events.decision(decision(previous, "ignore", "other_human"), 600)
        events.transcript(request, 2000)
        events.decision(decision(request), 2100)
        events.end("stopped", 2200)
        actual = events.drain()
        fixture = Path(__file__).resolve().parents[1] / "examples/tool-events.jsonl"
        self.assertEqual(actual, [json.loads(line) for line in fixture.read_text().splitlines()])


if __name__ == "__main__":
    unittest.main()
