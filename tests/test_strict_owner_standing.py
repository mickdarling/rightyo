"""Synthetic PCM, scripted word timings and enrolled bindings for the #326 live path."""

import unittest
from dataclasses import replace

from test_conversation import SESSION, decided
from test_edge_attribution import SPEECH, Timeline, Units
from test_request_outcomes import outcome
from test_turn_merge import BoundShadow

from rightyo.contracts import Addressing, Conversation, SpeakerPriority
from rightyo.live_audio import FRAME_BYTES, LiveConfig, LiveProcessor
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.tool_events import SpeechEvents

OWNER, OTHER = "Speaker A", "Speaker B"
MODE = Conversation(require_guest_address=True, follow_up_min_probability=0.5)
BODY_TAIL = ((" Show the", 0, 400), (" chart", 480, 700))
OWNER_BODY = ((0, 420, 1),)


class StrictOwnerStandingTests(unittest.TestCase):
    def start(self, mode=MODE):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        roles.source = BoundShadow({OWNER: ("bound", "owner"), OTHER: ("bound", "guest")})
        self.events = SpeechEvents(request_control=True)
        self.events.start(
            SESSION, conversation=mode, priority=roles, addressing=Addressing.from_names(["Jarvis"])
        )
        self.events.drain()

    def replay(self, *windows, **settings):
        """One second of authored speech and two of silence per scripted ASR window."""
        turns, inferred, heads = [], set(), {}
        units, timeline = Units(), Timeline()
        processor = LiveProcessor(
            LiveConfig(
                SESSION,
                provenance="causal-replay",
                transcriber=units,
                diarizer=timeline,
                edge_attribution_ms=300,
                tail_join_ms=1000,
                pre_roll_ms=20,
                on_inferred=inferred.add,
                on_diarized_head=heads.__setitem__,
                **settings,
            ),
            turns.append,
        )
        self.addCleanup(processor.close)
        for words, segments in windows:
            units.units = Units(*words).units
            timeline.timeline = Timeline(*segments).timeline
            for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
                processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
        processor.finish()
        return [(t, t.utterance_id in inferred, heads[t.utterance_id]) for t in turns]

    def publish(self, replayed, *, label="attend", score=0.98, dispatch=True):
        turn, inferred, head = replayed
        self.assertFalse(turn.overlap)
        self.events.transcript(turn, turn.end_ms, inferred=inferred, diarized_head=head)
        event = decided(turn, label, attend=score, confidence=score)
        if label == "uncertain":
            # A raw attend plurality below the ordinary request floor.
            decision = replace(
                event.decision,
                attention_choice="attend",
                probabilities={"attend": score, "ignore": 0.30, "uncertain": 0.70 - score},
            )
            event = replace(event, decision=decision)
        self.events.decision(event, turn.end_ms)
        output = self.events.drain()
        (transcript,) = [e["turn"] for e in output if e["type"] == "transcript"]
        self.assertNotIn("word_attribution", transcript)
        (attention,) = [e for e in output if e["type"] == "attention"]
        evidence = attention["decision"]
        requests = [e for e in output if e["type"] == "request"]
        self.assertEqual(len(requests), int(dispatch))
        if dispatch:
            self.assertEqual(requests[0]["request_id"], attention["request_id"])
            self.assertEqual(requests[0]["turn"], transcript)
            self.assertEqual(requests[0]["decision"], evidence)
        else:
            self.assertNotIn("request_id", attention)
        return evidence

    def assert_standing(self, evidence, strict, inferred, speakers, reason):
        self.assertEqual(evidence["strict_word_count"], strict)
        self.assertEqual(evidence["inferred_word_count"], inferred)
        self.assertEqual(evidence["strict_speakers"], speakers)
        self.assertEqual(evidence["standing_reason"], reason)

    def test_unnamed_owner_with_edge_word_or_joined_tail(self):
        for start in (480, 860):  # After the timeline by 60 ms (edge) or 440 ms (tail join).
            with self.subTest(start=start):
                self.start()
                words = (BODY_TAIL[0], (" chart", start, 1000))
                (turn,) = self.replay((words, OWNER_BODY))
                self.assertTrue(turn[1])
                self.assert_standing(self.publish(turn), 2, 1, [OWNER], "strict_owner")

    def test_undelimited_owner_address_with_inferred_last_or_first_word(self):
        words = ((" Jarvis", 0, 180), (" the chart is", 260, 660), (" ready", 720, 920))
        for start, end, head in ((0, 680, len("Jarvis the chart is")), (220, 940, None)):
            with self.subTest(start=start):
                self.start()
                (turn,) = self.replay((words, ((start, end, 1),)))
                self.assertEqual(turn[1:], (True, head))
                self.assert_standing(self.publish(turn), 4, 1, [OWNER], "strict_owner")

    def test_joined_inferred_wake_name_promotes_owner_abstention(self):
        self.start()
        words = ((" Jarvis", 0, 150), (" the chart is ready", 720, 1000))
        (turn,) = self.replay(
            (words, ((680, 1000, 1),)),
            turn_merge_gap_ms=1000,
            turn_wake_word=lambda text: text == "Jarvis",
            turn_owner=lambda fragment: fragment["speaker"] == OWNER,
        )
        self.assertEqual(turn[1:], (True, None))
        evidence = self.publish(turn, label="uncertain", score=0.44)
        self.assertEqual(evidence["label"], "attend")
        self.assertNotIn("follow_up", evidence)
        self.assert_standing(evidence, 4, 1, [OWNER], "strict_owner")

    def test_only_owners_word_is_an_inferred_tail_of_a_guest_request(self):
        self.start()
        # B covers the body. A speaks only in the tail, with too little coverage
        # for a strict label. Tail joining publishes it with B's turn.
        words = ((" Jarvis the chart is", 0, 400), (" ready", 800, 1000))
        (turn,) = self.replay((words, ((0, 420, 2), (850, 950, 1))))
        self.assertEqual((turn[0].speaker_id, turn[1]), (OTHER, True))
        evidence = self.publish(turn, dispatch=False)
        self.assertTrue(evidence["guest_unaddressed"])
        self.assert_standing(evidence, 4, 1, [OTHER], "strict_other")

    def test_conflicting_strict_label_on_a_bound_turn_stays_guest(self):
        self.start()
        (turn,) = self.replay((BODY_TAIL, ((0, 420, 2),)))
        # The live splitter normally separates labels. Even a producer that gives
        # this composite the owner's turn label cannot erase B's strict words.
        heard, inferred, head = turn
        evidence = self.publish((replace(heard, speaker_id=OWNER), inferred, head), dispatch=False)
        self.assert_standing(evidence, 2, 1, [OTHER], "strict_other")

    def test_coverage_only_on_bound_owner_has_no_strict_words(self):
        self.start()
        # Partial coverage supplies A's label for the first unit; the edge pass
        # labels the last word. Neither supplies a strictly covered word.
        (turn,) = self.replay((BODY_TAIL, ((0, 350, 1),)), word_label_coverage=0.8)
        self.assertEqual((turn[0].speaker_id, turn[1]), (OWNER, True))
        evidence = self.publish(turn, dispatch=False)
        self.assertTrue(evidence["guest_unaddressed"])
        self.assert_standing(evidence, 0, 3, [], "no_strict_words")

    def test_collapsed_restart_discards_the_superseded_strict_words(self):
        self.start()
        (turn,) = self.replay(
            (((" Read the chart", 0, 900),), ((0, 1000, 1),)),
            (((" Read the chart now", 20, 1000),), ((3100, 4000, 1),)),
            end_of_turn=lambda pcm: 1.0,
            turn_complete_hold_ms=3000,
            turn_collapse_restarts=True,
            word_label_coverage=0.8,
        )
        self.assertEqual(turn[0].text, "Read the chart now")
        self.assertEqual(turn[0].speaker_id, OWNER)
        self.assert_standing(self.publish(turn, dispatch=False), 0, 4, [], "no_strict_words")

    def test_spoken_answer_floor_with_a_mixed_opening_and_answer(self):
        for guest_policy, floor in ((True, 0.40), (False, 0.40), (False, None)):
            with self.subTest(guest_policy=guest_policy, floor=floor):
                self.start(
                    replace(
                        MODE,
                        require_guest_address=guest_policy,
                        answer_follow_up_min_probability=floor,
                        answer_follow_up_window_ms=5000,
                    )
                )
                answer = ((" Repeat your last", 40, 480), (" answer.", 700, 920))
                opening, follow = self.replay((BODY_TAIL, OWNER_BODY), (answer, ((3000, 3500, 1),)))
                self.assertTrue(opening[1] and follow[1])
                self.publish(opening)
                for state, at in (
                    ("accepted", 1100),
                    ("reply_started", 1400),
                    ("reply_ended", 1900),
                ):
                    self.assertTrue(
                        self.events.request_outcome(outcome(opening[0].utterance_id, state), at)
                    )
                evidence = self.publish(
                    follow, label="uncertain", score=0.44, dispatch=floor is not None
                )
                if floor is not None:
                    self.assertTrue(evidence["follow_up"])
                    self.assert_standing(evidence, 3, 1, [OWNER], "strict_owner")
                else:
                    self.assertEqual(evidence["label"], "uncertain")
                    self.assertNotIn("standing_reason", evidence)

    def test_keys_off_preserve_main_direct_attend_evidence(self):
        for channel, role in ((1, "owner"), (2, "participant")):
            with self.subTest(role=role):
                self.start(replace(MODE, require_guest_address=False))
                (turn,) = self.replay((BODY_TAIL, ((0, 420, channel),)))
                self.assertTrue(turn[1])
                self.assertEqual(
                    self.publish(turn),
                    {
                        "label": "attend",
                        "recipient_kind": "system",
                        "confidence": 0.98,
                        "model": "mock-v1",
                        "provider": "mock",
                        "role": role,
                    },
                )
