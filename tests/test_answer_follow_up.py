"""Scoped owner follow-ups (#326). Authored turns, playback controls and bindings only."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_conversation import SESSION, decided, turn
from test_request_outcomes import outcome

from rightyo.contracts import (
    Acknowledgement,
    Addressing,
    ContractError,
    Conversation,
    SpeakerPriority,
)
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.tool_events import SpeechEvents

BASE = Conversation(
    follow_up_min_probability=0.5, require_cue=True, continuation_cue=True, cue_when_others_ms=6000
)
MODE = replace(BASE, answer_follow_up_min_probability=0.4, answer_follow_up_window_ms=2000)
INSTRUCTION = "Please check the timetable."
ANSWER = "Yes, both copies."


class AnswerFollowUpTests(unittest.TestCase):
    def prepare(self, mode=MODE, *, scoped=True, label_owner=False):
        self.bindings = {"Speaker A": ("bound", "owner"), "Speaker B": ("bound", "guest")}
        owners = ("Speaker A",) if label_owner else ("owner",)
        roles = EnrolledRoles(SpeakerPriority(owners=owners), follow_up_min_probability=0.5)
        roles.source = SimpleNamespace(binding=self.bindings.get)
        self.notes = []
        self.events = SpeechEvents(request_control=True, report=self.notes.append)
        self.events.start(
            SESSION,
            conversation=mode,
            priority=roles,
            acknowledgement=Acknowledgement(),
            other_agents=Addressing.from_names(["Vega"]),
        )
        opening = turn("opening", 0, 800, "List the departures.", provenance="diarization-timeline")
        self.events.transcript(opening, 800)
        self.events.decision(decided(opening, "attend"), 900)
        self.opening = {**opening.to_dict(), "role": "owner"}
        (request,) = [e for e in self.events.drain() if e["type"] == "request"]
        self.assertEqual(request["request_id"], SESSION + ":opening")
        self.assertEqual(request["context"]["turns"], [])
        self.assertIs(request["acknowledge"], True)
        if scoped:
            self.report("accepted", 1000)

    def report(self, state, at, **fields):
        self.assertTrue(self.events.request_outcome(outcome("opening", state, **fields), at))

    def playback(self, *, asked=False):
        self.report("reply_started", 1200)
        self.report("reply_ended", 2200, asked=asked)

    def candidate(
        self,
        text=INSTRUCTION,
        *,
        start=2600,
        attend=0.44,
        speaker="Speaker A",
        overlap=False,
        inferred=False,
        unbind=False,
        recipient="system",
        name="candidate",
    ):
        current = turn(
            name,
            start,
            start + 600,
            text,
            speaker,
            overlap=overlap,
            provenance="diarization-timeline",
        )
        self.events.transcript(current, current.end_ms, inferred=inferred)
        if unbind:
            self.bindings.clear()
        event = decided(current, "uncertain", recipient=recipient, attend=attend, confidence=0.3)
        event = replace(event, decision=replace(event.decision, attention_choice="attend"))
        self.events.decision(event, current.end_ms + 100)
        emitted = self.events.drain()
        self.assertTrue(
            any(e["type"] == "attention" and e["utterance_id"] == name for e in emitted)
        )
        return [e for e in emitted if e["type"] == "request"]

    def assert_dispatched(self, requests, text):
        self.assertEqual([r["request_id"] for r in requests], [SESSION + ":candidate"])
        request = requests[0]
        self.assertEqual(request["turn"]["utterance_id"], "candidate")
        self.assertEqual(request["turn"]["text"], text)
        self.assertEqual(request["context"]["turns"], [self.opening])
        self.assertIs(request["decision"]["follow_up"], True)
        self.assertEqual(request["decision"]["confidence"], 0.3)
        self.assertIs(request["acknowledge"], False)

    def test_owner_answer_to_host_marked_question(self):
        for scoped in (True, False):
            with self.subTest(scoped=scoped):
                self.prepare(scoped=scoped)
                if scoped:
                    self.playback(asked=True)
                else:
                    self.events.reply("started", 1200)
                    self.events.reply("ended", 2200, asked=True)
                self.assert_dispatched(self.candidate(ANSWER), ANSWER)

    def test_owner_instruction_after_spoken_answer(self):
        for instruction in (INSTRUCTION, "Could you please summarize both?", "Make it shorter."):
            with self.subTest(instruction=instruction):
                self.prepare()
                self.playback()
                self.assert_dispatched(self.candidate(instruction), instruction)

    def test_only_answers_and_instructions_get_the_lower_floor(self):
        for text, asked in (
            ("What a sight!", True),
            ("Come here, little pup!", True),
            ("No, sit.", False),
            ("The second one!", False),
            ("Could you sit, pup?", False),
            (ANSWER, False),
            ("Okay, that's all.", True),
        ):
            with self.subTest(text=text):
                self.prepare()
                self.playback(asked=asked)
                self.assertEqual(self.candidate(text), [])
        self.prepare()
        self.playback()
        self.assert_dispatched(self.candidate("What a sight!", attend=0.51), "What a sight!")

    def test_ack_start_and_publication_do_not_confirm_a_spoken_answer(self):
        for phase in ("legacy_ack", "scoped_ack", "reply_started", "reply_published"):
            with self.subTest(phase=phase):
                self.prepare(scoped=phase != "legacy_ack")
                if phase.endswith("_ack"):
                    self.events.reply("started", 1200)
                    self.events.reply("ended", 2200)
                elif phase == "reply_started":
                    self.report(phase, 1200)
                else:
                    self.report(phase, 2200, asked=True)
                self.assertEqual(
                    self.candidate(ANSWER if phase == "reply_published" else INSTRUCTION), []
                )

    def test_onset_and_probability_boundaries(self):
        for start, attend, passes in (
            (2100, 0.44, False),
            (2200, 0.44, True),
            (4200, 0.4, True),
            (4201, 0.44, False),
            (2600, 0.399, False),
        ):
            with self.subTest(start=start, attend=attend):
                self.prepare()
                self.playback()
                requests = self.candidate(start=start, attend=attend)
                if passes:
                    self.assert_dispatched(requests, INSTRUCTION)
                else:
                    self.assertEqual(requests, [])
        self.prepare(mode=replace(MODE, window_ms=1000))
        self.playback()
        self.assertEqual(
            self.candidate(start=3200), []
        )  # Engagement expires before the reply bound.

    def test_speaker_identity_attribution_and_recipient(self):
        for options in (
            {"speaker": "Speaker B"},
            {"overlap": True},
            {"inferred": True},
            {"unbind": True},
            {"recipient": "unknown"},
            {"recipient": "other_human"},
        ):
            with self.subTest(options=options):
                self.prepare()
                self.playback()
                self.assertEqual(self.candidate(**options), [])
        self.prepare(label_owner=True)
        self.playback()
        self.assertEqual(self.candidate(), [])

    def test_presence_meeting_other_agent_and_playback_remain_vetoes(self):
        for veto in ("presence", "meeting", "other_agent", "other_playback"):
            with self.subTest(veto=veto):
                self.prepare()
                self.playback()
                if veto == "presence":
                    guest = turn(
                        "guest",
                        2600,
                        3700,
                        "The bus is early.",
                        "Speaker B",
                        provenance="diarization-timeline",
                    )
                    self.events.transcript(guest, guest.end_ms, expect_decision=False)
                elif veto == "meeting":
                    self.events.meeting(True)
                elif veto == "other_playback":
                    self.events.reply("started", 3700)
                text = "Please check the timetable, Vega." if veto == "other_agent" else INSTRUCTION
                self.assertEqual(self.candidate(text, start=3800), [])

    def test_next_request_needs_its_own_playback(self):
        self.prepare()
        self.playback(asked=True)
        self.assert_dispatched(self.candidate(ANSWER), ANSWER)
        self.report("reply_ended", 3400, asked=True)  # Duplicate end for the old request.
        self.assertEqual(self.candidate(start=3500, name="next"), [])

    def test_disabled_keeps_below_bar_and_the_refused_turn_in_context(self):
        self.prepare(mode=BASE)
        self.playback(asked=True)
        self.assertEqual(self.candidate(ANSWER), [])
        self.assertIn("follow_up outcome=refused reason=below_bar attend=0.44 bar=0.5", self.notes)
        later = turn(
            "later", 4000, 4600, "List the return trips.", provenance="diarization-timeline"
        )
        self.events.transcript(later, later.end_ms)
        self.events.decision(decided(later, "attend"), 4700)
        (request,) = [e for e in self.events.drain() if e["type"] == "request"]
        self.assertEqual(request["request_id"], SESSION + ":later")
        self.assertEqual(
            [t["utterance_id"] for t in request["context"]["turns"]], ["opening", "candidate"]
        )

    def test_config_bounds_and_opt_in_advertisement(self):
        floor = "answer_follow_up_min_probability"
        window = "answer_follow_up_window_ms"
        self.assertNotIn(floor, BASE.to_dict())
        self.assertNotIn(window, Conversation.from_dict({window: 1}).to_dict())
        for value in (1, 5000):
            enabled = Conversation.from_dict({floor: 0.4, window: value}).to_dict()
            self.assertEqual((enabled[floor], enabled[window]), (0.4, value))
        for key, bad in (
            (floor, [0, -0.1, 1.1, True, float("nan"), "0.4"]),
            (window, [0, 5001, True, 0.5, "2000"]),
        ):
            for value in bad:
                with self.subTest(key=key, value=value), self.assertRaises(ContractError):
                    Conversation.from_dict({key: value})
