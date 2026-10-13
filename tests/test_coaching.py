"""Synthetic coaching lifetime, resource and authority regressions; no live models."""

import json
import threading
import unittest
from dataclasses import replace
from unittest.mock import patch
from uuid import UUID

from rightyo.coaching import (
    MAX_CONTEXT_BYTES,
    CoachingSession,
    CoachSuggestion,
    MockCoachProvider,
    validate_coaching_control,
    wire_json,
)
from rightyo.coaching_input import CoachingInputAdapter
from rightyo.contracts import ContractError, ListenerProfile, Turn
from rightyo.listener_profiles import ListenerProfiles
from rightyo.tool_events import SpeechEvents


def uid(n):
    return str(UUID(int=n))


SESSION, DESTINATION, GENERATION, QUALIFICATION = map(uid, (1, 2, 3, 4))
PROFILE = ListenerProfile(
    "coach",
    "Coach",
    "Conversation",
    "One short private nudge",
    "Hold during speech",
    "owner_private",
)


class Clock:
    at = 0.0

    def __call__(self):
        return self.at


def control(action, **values):
    return {
        "schema_version": 1,
        "type": "coaching_control",
        "action": action,
        "session_id": SESSION,
        **values,
    }


def enable(**values):
    return control(
        "enable",
        **{
            "destination_id": DESTINATION,
            "listener_id": "coach",
            "participant_awareness": True,
            **values,
        },
    )


def qualify(**values):
    return control(
        "qualify",
        **{
            "destination_id": DESTINATION,
            "generation_id": GENERATION,
            "qualification_id": QUALIFICATION,
            "valid_for_ms": 2000,
            **values,
        },
    )


def turn(n=10, text="Authored conversation", **values):
    return Turn(
        **{
            "session_id": SESSION,
            "utterance_id": uid(n),
            "revision": 1,
            "start_ms": n * 10,
            "end_ms": n * 10 + 10,
            "text": text,
            "speaker_id": "Speaker B",
            "finalized": True,
            "overlap": False,
            "recognizer_id": "synthetic",
            "provenance": "synthetic",
            "speaker_provenance": "authored-fixture",
            **values,
        }
    )


class Provider:
    def __init__(self, *, blocking=False, result=None):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = None
        self.result = (
            CoachSuggestion("Ask which assumption matters most.") if result is None else result
        )
        if not blocking:
            self.release.set()

    def advise(self, work, cancelled):
        self.calls.append(work)
        self.cancelled = cancelled
        self.entered.set()
        if not self.release.wait(1):
            raise AssertionError("test provider receipt budget")
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class CoachingTests(unittest.TestCase):
    def make(self, provider=None, profiles=None):
        self.clock = Clock()
        self.profiles = ListenerProfiles([PROFILE] if profiles is None else profiles)
        self.provider = provider or Provider()
        self.session = CoachingSession(
            profiles=self.profiles, provider=self.provider, clock=self.clock
        )

        def cleanup():
            self.session.close()
            if isinstance(self.provider, Provider):
                self.provider.release.set()
            self.session.wait_idle(1)

        self.addCleanup(cleanup)
        return self.session

    def start(self):
        self.session.apply_control(enable())
        self.session.apply_control(qualify())
        self.session.drain()

    def observe(self, item=None, **values):
        self.session.observe(
            item or turn(),
            **{
                "generation_id": GENERATION,
                "pause_ms": 1000,
                "owner_mid_answer": False,
                **values,
            },
        )

    def advice(self):
        self.assertTrue(self.session.wait_idle(1))
        return [e for e in self.session.drain() if e["type"] == "coaching_advice"]

    def test_default_off_and_enable_without_route_accept_no_context_or_calls(self):
        self.make()
        self.assertFalse(self.session.capture_permitted)
        self.observe()
        self.session.apply_control(enable())
        self.observe(turn(11))
        self.assertEqual(self.provider.calls, [])
        self.assertFalse(self.session.capture_permitted)
        self.session.apply_control(qualify())
        self.observe(turn(12))
        self.assertEqual(len(self.advice()), 1)
        self.assertEqual([x.turn.utterance_id for x in self.provider.calls[0].context], [uid(12)])

    def test_controls_require_exact_opt_in_private_profile_and_bound_ids(self):
        self.make()
        for raw in (
            enable(participant_awareness=False),
            enable(participant_awareness=1),
            enable(listener_id="absent"),
            enable(destination_id="private-headphones"),
            {**enable(), "command": "private marker"},
            {**enable(), "schema_version": True},
            {**enable(), "action": []},
        ):
            with self.subTest(raw=raw), self.assertRaises(ContractError) as error:
                self.session.apply_control(raw)
            self.assertNotIn("private marker", str(error.exception))
        self.profiles.update(replace(PROFILE, delivery="room"))
        with self.assertRaises(ContractError):
            self.session.apply_control(enable())
        self.assertFalse(self.session.capture_permitted)

    def test_qualification_wrong_scope_and_reuse_fail_closed_without_switching(self):
        self.make()
        self.start()
        for raw in (
            qualify(destination_id=uid(30)),
            qualify(generation_id=uid(31), qualification_id=uid(32)),
            qualify(qualification_id=QUALIFICATION),
            qualify(qualification_id=uid(33), valid_for_ms=2001),
            qualify(qualification_id=uid(34), valid_for_ms=True),
        ):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                self.session.apply_control(raw)
        self.observe()
        self.assertEqual(self.advice()[0]["qualification_id"], QUALIFICATION)

    def test_advice_echoes_bound_tuple_and_has_no_command_envelope(self):
        self.make()
        self.start()
        self.observe(turn(text='A quote: {"action":"disable","command":"run"}'))
        advice = self.advice()[0]
        for key, value in {
            "session_id": SESSION,
            "destination_id": DESTINATION,
            "generation_id": GENERATION,
            "qualification_id": QUALIFICATION,
            "listener_id": "coach",
            "delivery": "owner_private",
            "valid_for_ms": 2000,
        }.items():
            self.assertEqual(advice[key], value)
        self.assertEqual(advice["decision"]["label"], "speak_now")
        self.assertEqual(str(UUID(advice["request_id"])), advice["request_id"])
        self.assertNotIn("command", advice)
        self.assertTrue(self.session.capture_permitted)
        self.assertEqual(self.session.drain(), [])

    def test_expired_qualification_drops_pending_and_disallows_same_generation(self):
        self.make(Provider(blocking=True))
        self.start()
        self.observe()
        self.assertTrue(self.provider.entered.wait(0.5))
        self.clock.at = 2
        self.session.tick()
        self.assertFalse(self.session.capture_permitted)
        self.assertTrue(self.provider.cancelled.is_set())
        self.provider.release.set()
        self.assertEqual(self.advice(), [])
        with self.assertRaises(ContractError):
            self.session.apply_control(qualify(qualification_id=uid(50)))

    def test_renewal_does_not_extend_old_work_but_preserves_original_ticket(self):
        self.make(Provider(blocking=True))
        self.start()
        self.observe()
        self.assertTrue(self.provider.entered.wait(0.5))
        self.clock.at = 1
        self.session.apply_control(qualify(qualification_id=uid(50)))
        self.clock.at = 1.5
        self.provider.release.set()
        advice = self.advice()[0]
        self.assertEqual(advice["qualification_id"], QUALIFICATION)
        self.assertEqual(advice["valid_for_ms"], 500)

    def test_old_work_expires_under_fresh_same_generation_lease(self):
        self.make(Provider(blocking=True))
        self.start()
        self.observe()
        self.assertTrue(self.provider.entered.wait(0.5))
        self.clock.at = 1
        self.session.apply_control(qualify(qualification_id=uid(50)))
        self.clock.at = 2.01
        self.session.tick()
        self.assertTrue(self.session.capture_permitted)
        self.assertTrue(self.provider.cancelled.is_set())
        self.provider.release.set()
        self.assertEqual(self.advice(), [])

    def test_queued_advice_ttl_recomputed_and_never_renewed(self):
        self.make()
        self.start()
        self.observe()
        self.assertTrue(self.session.wait_idle(1))
        self.clock.at = 1
        self.session.apply_control(qualify(qualification_id=uid(50)))
        self.clock.at = 2
        self.assertTrue(self.session.capture_permitted)
        self.assertEqual(self.advice(), [])

    def test_invalidation_requalification_never_revives_old_worker_or_callback(self):
        self.make(Provider(blocking=True))
        self.start()
        self.observe()
        self.assertTrue(self.provider.entered.wait(0.5))
        self.session.apply_control(
            control("invalidate", generation_id=GENERATION, reason="route_lost")
        )
        self.assertFalse(self.session.capture_permitted)
        new_generation = uid(60)
        self.session.apply_control(qualify(generation_id=new_generation, qualification_id=uid(61)))
        self.observe(turn(11))  # Old capture callback retains old generation.
        self.observe(turn(12), generation_id=new_generation)
        self.assertEqual(len(self.provider.calls), 1)  # Old uncooperative call still occupies slot.
        self.provider.release.set()
        self.assertEqual(self.advice(), [])
        self.clock.at = 20
        self.session.tick()
        self.session.apply_control(qualify(generation_id=uid(62), qualification_id=uid(63)))
        self.observe(turn(13), generation_id=uid(62))
        self.assertEqual(len(self.advice()), 1)
        self.assertNotIn(uid(10), [x.turn.utterance_id for x in self.provider.calls[-1].context])

    def test_older_invalidation_cannot_cancel_new_generation(self):
        self.make()
        self.start()
        self.session.apply_control(
            control("invalidate", generation_id=GENERATION, reason="route_changed")
        )
        self.session.apply_control(qualify(generation_id=uid(60), qualification_id=uid(61)))
        self.session.apply_control(
            control("invalidate", generation_id=GENERATION, reason="route_lost")
        )
        self.assertTrue(self.session.capture_permitted)

    def test_disable_clears_context_and_late_failure_does_not_disable_fresh_session(self):
        self.make(Provider(blocking=True, result=RuntimeError("private text")))
        self.start()
        self.observe()
        self.assertTrue(self.provider.entered.wait(0.5))
        self.session.apply_control(control("disable"))
        self.assertTrue(self.provider.cancelled.is_set())
        self.session.apply_control({**enable(), "session_id": uid(70)})
        self.session.apply_control(
            {**qualify(generation_id=uid(71), qualification_id=uid(72)), "session_id": uid(70)}
        )
        self.provider.release.set()
        self.assertEqual(self.advice(), [])
        self.assertTrue(self.session.capture_permitted)
        self.assertEqual(len(self.session._context), 0)

    def test_profile_remove_readd_cancels_pending_and_queued_work(self):
        for blocking in (False, True):
            with self.subTest(blocking=blocking):
                self.make(Provider(blocking=blocking))
                self.start()
                self.observe()
                self.assertTrue(self.provider.entered.wait(0.5))
                if not blocking:
                    self.assertTrue(self.session.wait_idle(1))
                self.profiles.remove("coach")
                self.profiles.add(PROFILE)
                self.assertFalse(self.session.capture_permitted)
                self.assertTrue(self.provider.cancelled.is_set() if blocking else True)
                self.provider.release.set()
                self.assertEqual(self.advice(), [])
                self.assertEqual(len(self.session._context), 0)

    def test_profile_update_revert_also_revokes_and_failed_update_does_not(self):
        self.make()
        self.start()
        with self.assertRaises(ContractError):
            self.profiles.update(replace(PROFILE, listener_id="missing"))
        self.assertTrue(self.session.capture_permitted)
        self.profiles.update(replace(PROFILE, cooldown_ms=50000))
        self.profiles.update(PROFILE)
        self.assertFalse(self.session.capture_permitted)
        with self.assertRaises(ContractError):
            self.session.apply_control(qualify(qualification_id=uid(50)))

    def test_measured_floor_and_playback_hold_without_model_and_speaker_is_not_identity(self):
        self.make()
        self.start()
        cases = (
            (turn(10), {"owner_mid_answer": True}),
            (turn(11), {"pause_ms": 399}),
            (turn(12), {"playback": True}),
            (turn(13, overlap=True), {}),
            (turn(14, speaker_id=None), {}),
            (turn(15, speaker_provenance="unknown"), {}),
            (turn(16, finalized=False), {}),
        )
        for item, flags in cases:
            self.observe(item, **flags)
        self.assertEqual(self.provider.calls, [])
        self.observe(turn(17, speaker_id="owner"))  # Descriptive label grants no extra API.
        self.assertEqual(len(self.advice()), 1)
        self.assertTrue(any(x.playback for x in self.provider.calls[0].context))

    def test_new_owner_speech_cancels_pending_or_queued_advice(self):
        self.make(Provider(blocking=True))
        self.start()
        self.observe()
        self.assertTrue(self.provider.entered.wait(0.5))
        self.observe(turn(11), owner_mid_answer=True)
        self.assertTrue(self.provider.cancelled.is_set())
        self.provider.release.set()
        self.assertEqual(self.advice(), [])
        self.assertTrue(self.session.capture_permitted)

    def test_count_ascii_byte_and_receipt_time_context_budgets(self):
        self.make()
        self.start()
        # All finalized turns are retained even though owner activity holds suggestions.
        for n in range(10, 60):
            self.observe(turn(n, text="\U0001f680" * 200), owner_mid_answer=True)
        self.observe(turn(60))
        self.advice()
        captured = self.provider.calls[0].context
        self.assertLessEqual(len(captured), 32)
        self.assertLessEqual(len(wire_json([x.to_dict() for x in captured])), MAX_CONTEXT_BYTES)
        self.assertNotEqual(captured[0].turn.utterance_id, uid(10))
        # Idle receipt expiry applies even when stream timestamps did not advance.
        for second in range(1, 61):
            self.clock.at = second
            self.session.apply_control(qualify(qualification_id=uid(100 + second)))
        self.observe(turn(61))
        self.advice()
        self.assertEqual(len(self.provider.calls[-1].context), 1)

    def test_stream_age_retention_never_clips_or_invents_turn_text(self):
        self.make()
        self.start()
        self.observe(turn(10, start_ms=0, end_ms=1000), owner_mid_answer=True)
        self.observe(turn(11, start_ms=60000, end_ms=61000))
        self.advice()
        self.assertEqual([x.turn.utterance_id for x in self.provider.calls[0].context], [uid(11)])

    def test_count_eviction_and_oversized_unicode_turn_hold_without_provider(self):
        self.make()
        self.start()
        for n in range(10, 50):
            self.observe(turn(n), owner_mid_answer=True)
        self.observe(turn(50))
        self.advice()
        captured = self.provider.calls[0].context
        self.assertEqual(len(captured), 32)
        self.assertEqual(captured[0].turn.utterance_id, uid(19))
        self.observe(turn(51, text="\U0001f680" * 4000))
        self.assertEqual(len(self.provider.calls), 1)
        self.assertLessEqual(
            len(wire_json([x.to_dict() for x in self.session._snapshot()])), MAX_CONTEXT_BYTES
        )

    def test_profile_can_raise_cadence_and_final_budget_clears_owned_content(self):
        self.make(profiles=[replace(PROFILE, cooldown_ms=60000)])
        self.start()
        self.observe()
        self.advice()
        for second in range(1, 61):
            self.clock.at = second
            self.session.apply_control(qualify(qualification_id=uid(100 + second)))
            self.observe(turn(10 + second))
            self.advice()
        self.assertEqual(len(self.provider.calls), 2)
        with patch("rightyo.coaching.MAX_FINALS", 1):
            with self.assertRaises(ContractError):
                self.observe(turn(71))
        self.assertFalse(self.session.capture_permitted)
        self.assertEqual(len(self.session._context), 0)

    def test_close_discards_content_with_no_file_or_socket_io(self):
        self.make(MockCoachProvider())
        with (
            patch("builtins.open", side_effect=AssertionError("unexpected retention")),
            patch("socket.socket", side_effect=AssertionError("unexpected network")),
        ):
            self.start()
            self.observe(turn(text="Discard this authored conversation."))
            self.assertTrue(self.session.wait_idle(1))
            self.session.close()
            self.assertEqual(len(self.session._context), 0)
            self.assertEqual(len(self.session._seen), 0)
            self.assertFalse(self.session.capture_permitted)
            self.assertFalse(any(e["type"] == "coaching_advice" for e in self.session.drain()))
            with self.assertRaises(ContractError):
                self.session.apply_control({**enable(), "session_id": uid(90)})

    def test_cadence_counts_hold_and_does_not_query_each_turn_or_renewal(self):
        self.make()
        self.start()
        self.observe()
        self.advice()
        for second in range(1, 21):
            self.clock.at = second
            self.session.apply_control(qualify(qualification_id=uid(100 + second)))
            self.observe(turn(10 + second))
            self.advice()
        self.assertEqual(len(self.provider.calls), 2)

    def test_session_cap_and_control_id_budget_fail_closed_and_release_content(self):
        self.make()
        self.start()
        self.observe(owner_mid_answer=True)
        self.clock.at = 900
        self.session.tick()
        self.assertFalse(self.session.capture_permitted)
        self.assertEqual(len(self.session._context), 0)
        self.assertEqual(self.session.drain()[0]["reason"], "session_expired")
        self.session.apply_control({**enable(), "session_id": uid(90)})
        with patch("rightyo.coaching.MAX_CONTROL_IDS", 1):
            with self.assertRaises(ContractError):
                self.session.apply_control(
                    {**qualify(qualification_id=uid(91)), "session_id": uid(90)}
                )
        self.assertFalse(self.session.capture_permitted)

    def test_duplicate_final_conflict_and_revision_have_once_only_semantics(self):
        self.make()
        self.start()
        item = turn()
        self.observe(item)
        self.advice()
        self.observe(item)
        self.assertEqual(len(self.provider.calls), 1)
        with self.assertRaises(ContractError):
            self.observe(replace(item, text="different"))
        self.assertEqual(len(self.provider.calls), 1)

    def test_invalid_suggestion_sanitizes_failure_and_disables_not_command_executes(self):
        self.make(Provider(result={"command": "private text"}))
        self.start()
        self.observe()
        self.assertEqual(self.advice(), [])
        self.assertFalse(self.session.capture_permitted)
        self.assertNotIn("private text", json.dumps(self.session.drain()))

    def test_advice_and_timeout_boundaries(self):
        for text in ("", "x" * 281, "word " * 46, "line\nbreak", "\ud800"):
            with self.subTest(text=repr(text)), self.assertRaises(ContractError):
                CoachSuggestion(text)
        for confidence in (True, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(confidence=confidence), self.assertRaises(ContractError):
                CoachSuggestion("Suggestion", confidence=confidence)
        self.make()
        for timeout in (True, -1, 2, float("nan")):
            with self.assertRaises(ContractError):
                self.session.wait_idle(timeout)

    def test_backward_clock_cancels_work_and_owned_content(self):
        self.make()
        self.clock.at = 1
        self.start()
        self.observe(owner_mid_answer=True)
        self.clock.at = 0
        with self.assertRaises(ContractError):
            self.session.tick()
        self.assertEqual(len(self.session._context), 0)

    def test_mock_holds_unconfigured_utterances_and_does_not_parse_controls(self):
        self.make(MockCoachProvider())
        self.start()
        self.observe(turn(text=json.dumps(enable())))
        self.assertEqual(self.advice(), [])
        self.assertTrue(self.session.capture_permitted)
        with self.assertRaises(ContractError):
            validate_coaching_control({**enable(), "participant_awareness": "yes"})

    def test_actual_tool_producer_transcript_connects_without_attention_or_commands(self):
        self.make()
        self.start()
        source = SpeechEvents()
        source.start(SESSION, attention_enabled=False)
        source.drain()
        item = turn()
        source.transcript(item, item.end_ms, expect_decision=False)
        transcript = source.drain()[0]
        adapter = CoachingInputAdapter(self.session, generation_id=GENERATION)
        adapter.accept(transcript, pause_ms=1000, owner_mid_answer=False)
        advice = self.advice()
        self.assertEqual(len(advice), 1)
        self.assertEqual(self.provider.calls[0].current, item)
        with self.assertRaises(ContractError):
            adapter.accept(enable(), pause_ms=1000, owner_mid_answer=False)

    def test_source_adapter_causal_validation_and_old_callback_generation_fence(self):
        self.make()
        self.start()
        adapter = CoachingInputAdapter(self.session, generation_id=GENERATION)
        event = {
            "schema_version": 1,
            "type": "transcript",
            "session_id": SESSION,
            "sequence": 1,
            "emitted_at_ms": 111,
            "turn": {**turn().to_dict(), "role": "owner"},
        }
        for broken in (
            {**event, "schema_version": True},
            {**event, "emitted_at_ms": 0},
            {**event, "session_id": uid(100)},
            {**event, "command": "do not execute"},
        ):
            with self.assertRaises(ContractError):
                adapter.accept(broken, pause_ms=1000, owner_mid_answer=False)
        self.session.apply_control(
            control("invalidate", generation_id=GENERATION, reason="route_lost")
        )
        self.session.apply_control(qualify(generation_id=uid(101), qualification_id=uid(102)))
        adapter.accept(event, pause_ms=1000, owner_mid_answer=False)
        self.assertEqual(self.provider.calls, [])

    def test_real_live_identifier_format_connects_and_mock_is_scoped_to_source_id(self):
        self.make(MockCoachProvider({"live-1": "Ask which outcome matters most."}))
        self.start()
        source = SpeechEvents()
        source.start(SESSION, attention_enabled=False)
        source.drain()
        item = turn(utterance_id="live-1")
        source.transcript(item, item.end_ms, expect_decision=False)
        adapter = CoachingInputAdapter(self.session, generation_id=GENERATION)
        adapter.accept(source.drain()[0], pause_ms=1000, owner_mid_answer=False)
        advice = self.advice()
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0]["text"], "Ask which outcome matters most.")
        self.observe(item)  # Same source final is still once-only.
        self.assertEqual(self.advice(), [])


if __name__ == "__main__":
    unittest.main()
