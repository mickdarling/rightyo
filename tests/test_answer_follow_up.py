"""Scoped owner follow-ups (#326). Authored turns, playback controls and bindings only."""

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

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
from rightyo.prototype import PrototypeConfig, PrototypeController
from rightyo.self_voice import SelfVoiceConfig
from rightyo.tool_events import SpeechEvents

BASE = Conversation(
    follow_up_min_probability=0.5, require_cue=True, continuation_cue=True, cue_when_others_ms=6000
)
MODE = replace(BASE, answer_follow_up_min_probability=0.4, answer_follow_up_window_ms=2000)
INSTRUCTION = "Please check the timetable."
ANSWER = "Yes, both copies."


class AnswerFollowUpTests(unittest.TestCase):
    def prepare(
        self, mode=MODE, *, scoped=True, label_owner=False, identity="owner", self_voice=False
    ):
        self.bindings = {"Speaker A": ("bound", identity), "Speaker B": ("bound", "guest")}
        owners = ("Speaker A",) if label_owner else ("owner", "second-owner")
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
            self_voice=SelfVoiceConfig(enabled=self_voice),
        )
        opening = turn("opening", 0, 800, "List the departures.", provenance="diarization-timeline")
        self.events.transcript(opening, 800)
        self.events.decision(decided(opening, "attend"), 900)
        self.opening = {
            **opening.to_dict(),
            "role": "owner" if identity in owners or label_owner else "participant",
        }
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
        rebind=None,
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
        if rebind:
            self.bindings[speaker] = ("bound", rebind)
        event = decided(current, "uncertain", recipient=recipient, confidence=0.3)
        probabilities = {"attend": attend, "ignore": 0.30, "uncertain": 1 - attend - 0.30}
        self.assertEqual(max(probabilities, key=probabilities.get), "attend")
        event = replace(
            event,
            decision=replace(
                event.decision, probabilities=probabilities, attention_choice="attend"
            ),
        )
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

    def direct_request(self, name, start, end):
        current = turn(
            name, start, end, "List the return trips.", provenance="diarization-timeline"
        )
        self.events.transcript(current, end)
        self.events.decision(decided(current, "attend"), end + 1)
        self.assertIn(
            SESSION + ":" + name,
            [e.get("request_id") for e in self.events.drain() if e["type"] == "request"],
        )

    def playback_with_clock_gap(self, mode, scoped, *, end_only=False):
        self.prepare(mode=mode, scoped=scoped)
        config = PrototypeConfig(None, None, None, None, None, conversation=mode)
        with patch.object(PrototypeController, "_tick"):
            controller = PrototypeController(config, event_publisher=self.events)
        self.addCleanup(controller.close)
        controller._phase = "listening"
        controller._event_terminal = False
        controller._now_ms = lambda: controller._received_ms + 3000
        if not end_only:
            controller._received_ms = 1200
            if scoped:
                controller.request_outcome(outcome("opening", "reply_started"))
            else:
                controller.reply("started")
        controller._received_ms = 2200
        if scoped:
            controller.request_outcome(outcome("opening", "reply_ended", asked=True))
        else:
            controller.reply("ended", asked=True)

    def test_playback_uses_received_audio_time(self):
        for scoped in (True, False):
            for start, passes in ((2100, False), (2600, True), (5500, False)):
                with self.subTest(scoped=scoped, start=start):
                    self.playback_with_clock_gap(MODE, scoped)
                    self.assertEqual(bool(self.candidate(ANSWER, start=start)), passes)

    def test_revived_window_after_a_clock_gap(self):
        mode = replace(MODE, window_ms=1000, continuation_cue=False)
        for start, passes in ((2600, True), (3200, False)):
            with self.subTest(start=start):
                self.playback_with_clock_gap(mode, scoped=True, end_only=True)
                self.assertEqual(bool(self.candidate(ANSWER, start=start)), passes)

    def test_legacy_clip_cannot_answer_a_replacement_request(self):
        self.prepare(scoped=False)
        self.events.reply("started", 1200)
        self.direct_request("replacement", 1300, 1900)
        self.events.reply("ended", 2200, asked=True)
        self.assertEqual(self.candidate(), [])

    def test_late_decision_preserves_the_current_requests_answer(self):
        for text in (INSTRUCTION, ANSWER):
            with self.subTest(text=text):
                self.prepare(mode=replace(MODE, continuation_cue=False))
                older = turn(
                    "older", 900, 1100, "List the stops.", provenance="diarization-timeline"
                )
                self.events.transcript(older, 1100)
                self.direct_request("newer", 1200, 1600)
                for state in ("accepted", "reply_ended"):
                    fields = {"asked": True} if state == "reply_ended" else {}
                    self.assertTrue(
                        self.events.request_outcome(outcome("newer", state, **fields), 2200)
                    )
                self.events.decision(decided(older, "attend"), 2400)
                self.assertEqual(
                    [e["request_id"] for e in self.events.drain() if e["type"] == "request"],
                    [SESSION + ":older"],
                )
                self.assertEqual(
                    [r["request_id"] for r in self.candidate(text)], [SESSION + ":candidate"]
                )

    def test_end_only_question_survives_revival(self):
        for scoped in (True, False):
            with self.subTest(scoped=scoped):
                self.prepare(
                    mode=replace(MODE, window_ms=1000, continuation_cue=False), scoped=scoped
                )
                if scoped:
                    self.report("reply_ended", 2200, asked=True)
                else:
                    self.events.reply("ended", 2200, asked=True)
                self.assert_dispatched(self.candidate(ANSWER), ANSWER)

    def test_reused_label_cannot_transfer_a_question(self):
        for identity in ("guest", "second-owner"):
            with self.subTest(identity=identity):
                self.prepare(identity=identity)
                self.playback(asked=True)
                self.bindings["Speaker A"] = ("bound", "owner")
                self.assertEqual(self.candidate(ANSWER), [])

    def test_candidate_binding_must_survive_its_decision(self):
        self.prepare()
        self.playback(asked=True)
        self.bindings["Speaker A"] = ("bound", "second-owner")
        self.assertEqual(self.candidate(ANSWER, rebind="owner"), [])

    def test_open_playback_outlasts_recent_clips(self):
        for close_at in (None, 2500, 2600):
            with self.subTest(close_at=close_at):
                self.prepare()
                self.report("reply_started", 1100)
                self.direct_request("newer", 1200, 1800)
                for state in ("accepted", "reply_ended"):
                    self.assertTrue(self.events.request_outcome(outcome("newer", state), 1900))
                for at in range(2000, 2400, 50):
                    self.events.reply("started", at)
                    self.events.reply("ended", at + 1)
                if close_at is not None:
                    self.report("reply_ended", close_at)
                self.assertEqual(bool(self.candidate(start=2500)), close_at == 2500)

    def test_host_sample_window_vetoes_overlapping_follow_up(self):
        for start, end, passes in ((2600, 3200, False), (3100, None, False), (2200, 2600, True)):
            with self.subTest(start=start, end=end):
                self.prepare(self_voice=True)
                self.playback()
                self.events.self_voice.reply_window("other", start, end, 0, 3200)
                self.assertEqual(bool(self.candidate()), passes)

    def test_instruction_requires_an_object_and_no_other_addressee(self):
        for text in (
            "Text messages are expensive.",
            "Run like the wind, Fido!",
            "Check the timetable, Fido!",
            "Please check.",
        ):
            with self.subTest(text=text):
                self.prepare()
                self.playback()
                self.assertEqual(self.candidate(text), [])

    def test_instructions_about_the_assistants_output(self):
        for text in ("Repeat your last answer.", "Summarize it."):
            for attend in (0.44, 0.51):
                with self.subTest(text=text, attend=attend):
                    self.prepare(mode=replace(MODE, continuation_cue=False))
                    self.playback()
                    self.assert_dispatched(self.candidate(text, attend=attend), text)
        self.prepare()
        self.playback()
        self.assertEqual(self.candidate("Call your mom back."), [])

    def test_pending_question_rejects_redirected_answers(self):
        for text in ("Okay, Fido, drop the ball.", "No, sit."):
            with self.subTest(text=text):
                self.prepare()
                self.playback(asked=True)
                self.assertEqual(self.candidate(text), [])

    def test_follow_up_log_names_the_applied_floor(self):
        for attend, bar, passes in ((0.399, 0.4, False), (0.44, 0.4, True), (0.51, 0.5, True)):
            with self.subTest(attend=attend):
                self.prepare()
                self.playback()
                requests = self.candidate(attend=attend)
                if passes:
                    self.assert_dispatched(requests, INSTRUCTION)
                else:
                    self.assertEqual(requests, [])
                result = "formed reason=passed" if passes else "refused reason=below_bar"
                self.assertIn(f"follow_up outcome={result} attend={attend} bar={bar}", self.notes)

    def test_spoken_reply_cues_without_continuation(self):
        for text, asked in ((ANSWER, True), (INSTRUCTION, False)):
            with self.subTest(text=text):
                self.prepare(mode=replace(MODE, continuation_cue=False))
                self.playback(asked=asked)
                self.assert_dispatched(self.candidate(text), text)

    def test_disabled_keeps_the_ordinary_question_cue(self):
        self.prepare(mode=replace(BASE, continuation_cue=False))
        self.playback(asked=True)
        self.assert_dispatched(self.candidate("No, sit.", attend=0.51), "No, sit.")
