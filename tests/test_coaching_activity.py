"""Authored final/activity correlation and cancellation; no capture or models."""

import threading
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from uuid import UUID

from rightyo.coaching import CoachingSession, CoachSuggestion
from rightyo.coaching_activity import CoachingActivityAdapter, validate_coaching_activity
from rightyo.contracts import ContractError, ListenerProfile, Turn
from rightyo.listener_profiles import ListenerProfiles


def uid(n):
    return str(UUID(int=n))


SESSION, DESTINATION, GENERATION = map(uid, (1, 2, 3))
PROFILE = ListenerProfile("coach", "Coach", "Conversation", "Short nudge", "Hold", "owner_private")


class Clock:
    at = 0.0

    def __call__(self):
        return self.at


class Provider:
    def __init__(self, blocking=False):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.cancelled = None
        if not blocking:
            self.release.set()

    def advise(self, work, cancelled):
        self.calls.append(work)
        self.cancelled = cancelled
        self.entered.set()
        if not self.release.wait(1):
            raise AssertionError("authored provider wait exceeded")
        return CoachSuggestion("Ask which outcome matters most.")


def event(n=1, **changes):
    turn = Turn(
        SESSION,
        f"live-{n}",
        1,
        n * 10,
        n * 10 + 10,
        "An authored question",
        "Speaker A",
        True,
        False,
        "synthetic",
        "synthetic",
        speaker_provenance="authored-fixture",
    )
    return {
        "schema_version": 1,
        "type": "transcript",
        "session_id": SESSION,
        "sequence": n,
        "emitted_at_ms": n * 10 + 10,
        "turn": turn.to_dict(),
        **changes,
    }


def activity(n=1, **changes):
    return {
        "schema_version": 1,
        "type": "coaching_activity",
        "session_id": SESSION,
        "generation_id": GENERATION,
        "utterance_id": f"live-{n}",
        "pause_ms": 500,
        "owner_mid_answer": False,
        "playback": False,
        **changes,
    }


class CoachingActivityTests(unittest.TestCase):
    def make(self, *, blocking=False):
        self.clock = Clock()
        self.provider = Provider(blocking)
        self.profiles = ListenerProfiles([PROFILE])
        self.session = CoachingSession(
            profiles=self.profiles, provider=self.provider, clock=self.clock
        )
        self.session.apply_control(
            {
                "schema_version": 1,
                "type": "coaching_control",
                "action": "enable",
                "session_id": SESSION,
                "destination_id": DESTINATION,
                "listener_id": "coach",
                "participant_awareness": True,
            }
        )
        self.renew(4)
        self.session.drain()
        self.adapter = CoachingActivityAdapter(
            self.session, session_id=SESSION, generation_id=GENERATION, clock=self.clock
        )
        adapter, session, provider = self.adapter, self.session, self.provider

        def cleanup():
            adapter.close()
            session.close()
            provider.release.set()
            session.wait_idle(1)

        self.addCleanup(cleanup)

    def renew(self, number):
        self.session.apply_control(
            {
                "schema_version": 1,
                "type": "coaching_control",
                "action": "qualify",
                "session_id": SESSION,
                "destination_id": DESTINATION,
                "generation_id": GENERATION,
                "qualification_id": uid(number),
                "valid_for_ms": 2000,
            }
        )

    def advice(self):
        self.assertTrue(self.session.wait_idle(1))
        return [item for item in self.session.drain() if item["type"] == "coaching_advice"]

    def test_only_matching_actual_final_activity_can_start_provider(self):
        self.make()
        emitted = self.adapter.accept_transcript(event())
        self.assertEqual(emitted, event())
        self.assertTrue(self.adapter.pending)
        self.assertEqual(self.provider.calls, [])
        self.adapter.accept_activity(activity())
        self.assertFalse(self.adapter.pending)
        self.assertEqual(len(self.advice()), 1)
        self.assertEqual(self.provider.calls[0].pause_ms, 500)

    def test_superseded_activity_keeps_latest_pending_and_inflight_work(self):
        self.make(blocking=True)
        self.adapter.accept_transcript(event())
        self.adapter.accept_transcript(event(2))
        for _ in range(3):
            self.adapter.accept_activity(activity())
        self.assertTrue(self.adapter.pending)
        self.assertEqual(self.provider.calls, [])
        self.adapter.accept_activity(activity(2))
        self.assertTrue(self.provider.entered.wait(1))
        for _ in range(3):
            self.adapter.accept_activity(activity())
        self.assertFalse(self.provider.cancelled.is_set())
        self.provider.release.set()
        self.assertEqual(len(self.advice()), 1)
        self.assertEqual(self.provider.calls[0].current.utterance_id, "live-2")
        self.assertEqual([item.playback for item in self.provider.calls[0].context], [True, False])
        self.assertTrue(self.session.capture_permitted)

    def test_superseded_activity_does_not_extend_current_receipt_deadline(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.clock.at = 0.1
        self.adapter.accept_transcript(event(2))
        self.clock.at = 1.9
        self.renew(5)
        self.adapter.accept_activity(activity())
        self.clock.at = 2.2
        self.adapter.tick()
        self.assertFalse(self.adapter.pending)
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(len(self.session._context), 2)
        self.assertTrue(all(item.playback for _, item in self.session._context))
        with self.assertRaises(ContractError):
            self.adapter.accept_activity(activity(2))

    def test_known_superseded_id_cannot_bypass_scope_or_expiry(self):
        for change in ("session", "generation", "expired", "closed"):
            with self.subTest(change=change):
                self.make()
                self.adapter.accept_transcript(event())
                self.adapter.accept_transcript(event(2))
                raw = activity()
                if change == "session":
                    raw["session_id"] = uid(80)
                elif change == "generation":
                    raw["generation_id"] = uid(81)
                elif change == "expired":
                    self.clock.at = 2
                else:
                    self.adapter.close()
                with self.assertRaises(ContractError):
                    self.adapter.accept_activity(raw)
                self.assertFalse(self.adapter.pending)
                self.assertEqual(self.provider.calls, [])

    def test_activity_before_final_is_rejected_without_speculative_queue(self):
        self.make()
        with self.assertRaises(ContractError):
            self.adapter.accept_activity(activity())
        self.adapter.accept_transcript(event())
        self.assertEqual(self.provider.calls, [])

    def test_replacement_commits_safe_context_without_calling_provider(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.adapter.accept_transcript(event(2))
        self.assertEqual(self.provider.calls, [])
        self.adapter.accept_activity(activity(2))
        self.assertEqual(len(self.advice()), 1)
        context = self.provider.calls[0].context
        self.assertEqual([item.turn.utterance_id for item in context], ["live-1", "live-2"])
        self.assertEqual([item.playback for item in context], [True, False])

    def test_new_final_immediately_cancels_inflight_advice_and_preserves_busy_cadence(self):
        self.make(blocking=True)
        self.adapter.accept_transcript(event())
        self.adapter.accept_activity(activity())
        self.assertTrue(self.provider.entered.wait(1))
        self.adapter.accept_transcript(event(2))
        self.assertTrue(self.provider.cancelled.is_set())
        self.assertTrue(self.session._busy)
        self.assertEqual(len(self.session._context), 1)
        self.provider.release.set()
        self.assertEqual(self.advice(), [])
        self.adapter.accept_activity(activity(2))
        self.assertEqual(self.advice(), [])
        self.assertEqual(len(self.provider.calls), 1)
        self.assertTrue(self.session.capture_permitted)
        self.assertEqual(len(self.session._context), 2)

    def test_new_final_drops_completed_but_queued_advice(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.adapter.accept_activity(activity())
        self.assertTrue(self.session.wait_idle(1))
        self.assertIsNotNone(self.session._advice)
        self.adapter.accept_transcript(event(2))
        self.assertEqual(self.advice(), [])

    def test_duplicate_and_conflicting_final_rejected_after_activity(self):
        for changed in (False, True):
            with self.subTest(changed=changed):
                self.make(blocking=True)
                self.adapter.accept_transcript(event())
                self.adapter.accept_activity(activity())
                self.assertTrue(self.provider.entered.wait(1))
                duplicate = event(sequence=2)
                if changed:
                    duplicate["turn"]["text"] = "Different authored text"
                with self.assertRaises(ContractError):
                    self.adapter.accept_transcript(duplicate)
                self.assertTrue(self.provider.cancelled.is_set())
                self.provider.release.set()
                self.assertEqual(self.advice(), [])

    def test_duplicate_activity_cancels_work(self):
        self.make(blocking=True)
        self.adapter.accept_transcript(event())
        self.adapter.accept_activity(activity())
        self.assertTrue(self.provider.entered.wait(1))
        with self.assertRaises(ContractError):
            self.adapter.accept_activity(activity())
        self.assertTrue(self.provider.cancelled.is_set())
        self.provider.release.set()
        self.assertEqual(self.advice(), [])

    def test_wrong_session_generation_and_final_are_rejected(self):
        for changes in (
            {"session_id": uid(80)},
            {"generation_id": uid(81)},
            {"utterance_id": "live-99"},
        ):
            with self.subTest(changes=changes):
                self.make()
                self.adapter.accept_transcript(event())
                with self.assertRaises(ContractError):
                    self.adapter.accept_activity(activity(**changes))
                self.assertFalse(self.adapter.pending)
                self.assertEqual(self.provider.calls, [])

    def test_timeout_holds_context_even_after_same_generation_renewal(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.clock.at = 1
        self.renew(5)
        self.clock.at = 2
        self.adapter.tick()
        self.assertFalse(self.adapter.pending)
        self.assertEqual(len(self.session._context), 1)
        self.assertTrue(self.session._context[0][1].playback)
        self.assertEqual(self.provider.calls, [])
        with self.assertRaises(ContractError):
            self.adapter.accept_activity(activity())

    def test_late_activity_after_renewal_is_rejected(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.clock.at = 1
        self.renew(5)
        self.clock.at = 2
        with self.assertRaises(ContractError):
            self.adapter.accept_activity(activity())
        self.assertFalse(self.adapter.pending)
        self.assertEqual(self.provider.calls, [])

    def test_expiry_profile_mutation_and_new_generation_discard_pending(self):
        for reason in ("expiry", "profile", "generation"):
            with self.subTest(reason=reason):
                self.make()
                self.adapter.accept_transcript(event())
                if reason == "expiry":
                    self.clock.at = 2
                elif reason == "profile":
                    self.profiles.remove("coach")
                else:
                    self.session.apply_control(
                        {
                            "schema_version": 1,
                            "type": "coaching_control",
                            "action": "invalidate",
                            "session_id": SESSION,
                            "generation_id": GENERATION,
                            "reason": "route_changed",
                        }
                    )
                    self.session.apply_control(
                        {
                            "schema_version": 1,
                            "type": "coaching_control",
                            "action": "qualify",
                            "session_id": SESSION,
                            "destination_id": DESTINATION,
                            "generation_id": uid(90),
                            "qualification_id": uid(91),
                            "valid_for_ms": 2000,
                        }
                    )
                    self.assertTrue(self.session.capture_permitted)
                with self.assertRaises(ContractError):
                    self.adapter.accept_activity(activity())
                self.adapter.tick()
                self.assertFalse(self.adapter.pending)
                self.assertEqual(len(self.session._context), 0)
                self.assertEqual(self.provider.calls, [])

    def test_return_and_input_mutations_cannot_change_retained_final(self):
        self.make()
        original = event()
        emitted = self.adapter.accept_transcript(original)
        original["turn"]["text"] = "Changed input"
        emitted["turn"]["text"] = "Changed output"
        self.adapter.accept_activity(activity())
        self.assertEqual(len(self.advice()), 1)
        self.assertEqual(self.provider.calls[0].current.text, "An authored question")

    def test_role_and_display_are_descriptive_while_measurements_gate_provider(self):
        self.make()
        original = event(speaker_name="Authored speaker")
        original["turn"]["role"] = "owner"
        emitted = self.adapter.accept_transcript(original)
        self.assertEqual(emitted, original)
        self.adapter.accept_activity(activity(owner_mid_answer=True, playback=True, pause_ms=0))
        self.assertEqual(self.provider.calls, [])
        self.assertNotIn("role", self.session._context[0][1].turn.to_dict())

    def test_invalid_transcript_never_reaches_pending_or_context(self):
        for mutation in ("causal", "partial", "scope", "role", "display", "sequence"):
            with self.subTest(mutation=mutation):
                self.make()
                value = event()
                if mutation == "causal":
                    value["emitted_at_ms"] = 0
                elif mutation == "partial":
                    value["turn"]["finalized"] = False
                elif mutation == "scope":
                    value["session_id"] = uid(99)
                    value["turn"]["session_id"] = uid(99)
                elif mutation == "role":
                    value["turn"]["role"] = "admin"
                elif mutation == "display":
                    value["speaker_name"] = " "
                else:
                    value["sequence"] = 0
                with self.assertRaises(ContractError):
                    self.adapter.accept_transcript(value)
                self.assertFalse(self.adapter.pending)
                self.assertEqual(len(self.session._context), 0)

    def test_activity_exact_fields_types_ids_and_timing_bounds(self):
        for changes in (
            {"schema_version": True},
            {"schema_version": 2},
            {"type": "transcript"},
            {"session_id": "unknown"},
            {"generation_id": "unknown"},
            {"utterance_id": "x" * 97},
            {"pause_ms": True},
            {"pause_ms": -1},
            {"pause_ms": 60001},
            {"pause_ms": 0.1},
            {"owner_mid_answer": 0},
            {"playback": "false"},
            {"command": "ignored"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                validate_coaching_activity(activity(**changes))
        self.assertEqual(validate_coaching_activity(activity(pause_ms=60000))["pause_ms"], 60000)
        missing = activity()
        del missing["playback"]
        with self.assertRaises(ContractError):
            validate_coaching_activity(missing)

    def test_finish_holds_but_close_discards_pending_plaintext_and_rejects_future_input(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.adapter.finish_pending()
        self.assertFalse(self.adapter.pending)
        self.assertEqual(len(self.session._context), 1)
        self.assertEqual(self.provider.calls, [])
        self.adapter.accept_transcript(event(2))
        self.adapter.close()
        self.assertFalse(self.adapter.pending)
        self.assertEqual(len(self.session._context), 1)
        with self.assertRaises(ContractError):
            self.adapter.accept_transcript(event(3))

    def test_invalid_adapter_clock_drops_pending_and_raises_sanitized_error(self):
        self.make()
        self.adapter.accept_transcript(event())
        self.adapter._clock = lambda: float("nan")
        with self.assertRaisesRegex(ContractError, "invalid coaching activity clock"):
            self.adapter.tick()
        self.assertFalse(self.adapter.pending)

    def test_scoped_permission_resamples_time_after_registry_guard_wait(self):
        self.make()
        original = self.profiles.delivery_guard

        @contextmanager
        def delayed_guard(revision, cancelled=None):
            with original(revision, cancelled) as result:
                self.clock.at = 2
                yield result

        with patch.object(self.profiles, "delivery_guard", delayed_guard):
            self.assertFalse(
                self.session.capture_permitted_for(session_id=SESSION, generation_id=GENERATION)
            )
        self.assertFalse(self.session.capture_permitted)

    def test_reversed_sequence_or_timeline_rejects_pending_and_never_starts_provider(self):
        for value in (event(2, sequence=1), event(1, sequence=3)):
            with self.subTest(value=value):
                self.make()
                self.adapter.accept_transcript(event(2))
                with self.assertRaises(ContractError):
                    self.adapter.accept_transcript(value)
                self.assertFalse(self.adapter.pending)
                self.assertEqual(self.provider.calls, [])


if __name__ == "__main__":
    unittest.main()
