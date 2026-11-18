"""Courtesy-only follow-up refusals (#296); authored turns and controls only."""

import unittest
from dataclasses import replace

from test_conversation import SESSION, decided, turn
from test_turn_merge import fragment

from rightyo.contracts import Addressing, Conversation, SpeakerPriority
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents
from rightyo.turn_merge import TurnMerger

NAMES = Addressing.from_names(["Jarvis"])
MODE = Conversation(window_ms=10000, follow_up_min_probability=0.5)
COURTESIES = ("Thanks.", "Thank you.", "Okay, thanks.")


class CourtesyFollowUpTests(unittest.TestCase):
    def prepare(
        self,
        mode=MODE,
        *,
        addressing=NAMES,
        replying=False,
        asked=False,
        provenance="authored-fixture",
    ):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.events.start(
            SESSION,
            now_ms=0,
            conversation=mode,
            addressing=addressing,
            priority=ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",))),
        )
        first = turn(
            "initial-request", 0, 900, "Jarvis, draft the authored note", provenance=provenance
        )
        self.events.transcript(first, first.end_ms)
        self.events.decision(decided(first, "attend"), 1000)
        self.events.drain()
        if replying or asked:
            self.events.reply("started", 1100)
        if asked:
            # Authored host control: the assistant asked "Shall I send it?"
            self.events.reply("ended", 1200, asked=True)
        self.events.drain()
        self.notes.clear()

    def say(self, current, *, attend=0.5, continuation=False, inferred=False):
        self.events.transcript(current, current.end_ms, inferred=inferred)
        event = decided(current, "uncertain", recipient="unknown", attend=attend)
        if continuation:
            event = replace(event, decision=replace(event.decision, attention_choice="attend"))
        self.events.decision(event, current.end_ms + 100)
        return self.events.drain()

    def assert_refused(self, output, *, attend=0.5, reason="courtesy"):
        self.assertFalse(any(event["type"] == "request" for event in output))
        attention = next(event for event in output if event["type"] == "attention")
        self.assertEqual(attention["decision"]["label"], "uncertain")
        self.assertNotIn("follow_up", attention["decision"])
        self.assertEqual(
            [note for note in self.notes if note.startswith("follow_up ")][-1],
            f"follow_up outcome=refused reason={reason} attend={attend:g} bar=0.5",
        )

    def assert_formed(self, output):
        requests = [event for event in output if event["type"] == "request"]
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["decision"]["follow_up"], True)

    def test_courtesy_only_is_refused_inside_the_window(self):
        for text in COURTESIES:
            for attend in (0.5, 0.57):
                with self.subTest(text=text, attend=attend):
                    self.prepare()
                    output = self.say(turn("courtesy-turn", 2000, 2400, text), attend=attend)
                    self.assert_refused(output, attend=attend)
                    self.assertIsNotNone(self.events._engaged)

    def test_continuation_cue_does_not_exempt_a_courtesy(self):
        for text in COURTESIES:
            with self.subTest(text=text):
                self.prepare(replace(MODE, require_cue=True, continuation_cue=True), replying=True)
                current = turn("continuing-courtesy", 2000, 2400, text)
                choice = replace(decided(current, "uncertain").decision, attention_choice="attend")
                self.assertTrue(self.events._continues(current, choice))
                self.assert_refused(self.say(current, continuation=True))

    def test_appendix_does_not_exempt_a_courtesy(self):
        for text in COURTESIES:
            for attend in (0.1, 0.5):
                with self.subTest(text=text, attend=attend):
                    self.prepare(replace(MODE, appendix_ms=5000))
                    current = turn("appendix-courtesy", 1500, 1900, text)
                    self.assertTrue(self.events._appendix(current))
                    self.assert_refused(self.say(current, attend=attend), attend=attend)

    def test_cue_min_attend_does_not_exempt_a_courtesy(self):
        for text in COURTESIES:
            with self.subTest(text=text):
                self.prepare(replace(MODE, require_cue=True, cue_min_attend=0.75))
                current = turn("confident-courtesy", 2000, 2400, text)
                self.assert_refused(self.say(current, attend=0.75), attend=0.75)

    def test_a_courtesy_naming_the_assistant_forms_a_follow_up(self):
        for addressing, text in (
            (NAMES, "Jarvis, thank you."),
            # A name/variant can itself be a courtesy unit. This exercises the exemption
            # even though the whole turn also passes the strict courtesy-only parser.
            (Addressing.from_names(["Cheers"]), "Cheers, thank you."),
            (Addressing.from_names(["Jarvis"], {"Jarvis": ["Cheers"]}), "Thanks, Cheers."),
        ):
            with self.subTest(text=text, addressing=addressing):
                self.prepare(replace(MODE, require_cue=True), addressing=addressing)
                self.assert_formed(self.say(turn("named-courtesy", 2000, 2400, text)))

    def test_a_courtesy_answer_to_the_assistants_question_forms_a_follow_up(self):
        for text in (
            "Okay.",
            "Perfect.",
            "Great.",
            "Alright.",
            "All right.",
            "Cool.",
            "Thanks.",
            "Excellent.",
            "Cheers.",
        ):
            with self.subTest(text=text):
                self.prepare(replace(MODE, require_cue=False), asked=True)
                current = turn("courtesy-answer", 2000, 2400, text)
                self.assertTrue(MODE.is_courtesy_only(text))
                self.assert_formed(self.say(current))

    def test_pending_question_keeps_existing_answer_cues(self):
        for text, answers in (("Okay.", True), ("OK.", True), ("Perfect.", False)):
            with self.subTest(text=text):
                self.prepare(replace(MODE, require_cue=True), asked=True)
                current = turn("courtesy-answer-cue", 2000, 2400, text)
                self.assertEqual(self.events._answers_question(current), answers)
                output = self.say(current)
                if answers:
                    self.assert_formed(output)
                else:
                    self.assert_refused(output, reason="no_cue")

    def test_courtesy_exemption_starts_at_the_question_time(self):
        for start in (1199, 1200):
            with self.subTest(start=start):
                self.prepare(asked=True)
                output = self.say(turn("courtesy-question-window", start, 1600, "Great."))
                if start < 1200:
                    self.assert_refused(output)
                else:
                    self.assert_formed(output)

    def test_next_reply_clears_the_pending_question_courtesy_exemption(self):
        self.prepare(asked=True)
        self.events.reply("started", 1300)
        self.events.drain()
        self.assert_refused(self.say(turn("courtesy-after-next-reply", 2000, 2400, "Great.")))

    def test_a_courtesy_with_a_real_request_forms_a_follow_up(self):
        self.prepare(replace(MODE, require_cue=True))
        self.assert_formed(
            self.say(turn("courtesy-request", 2000, 2800, "Thanks, can you open the blue folder?"))
        )

    def test_a_closing_with_courtesies_keeps_its_reason_and_ends_engagement(self):
        self.prepare()
        output = self.say(turn("closing-turn", 2000, 2800, "Okay. That's it. Thank you."))
        self.assert_refused(output, reason="closing")
        ended = [event for event in output if event["type"] == "conversation"]
        self.assertEqual(
            [(event["state"], event["reason"]) for event in ended], [("ambient", "closed")]
        )
        self.assertIsNone(self.events._engaged)

    def test_a_courtesy_ending_as_a_question_keeps_existing_behavior(self):
        for text in ("Thanks?", "Thank you?!", "Okay, thanks?…"):
            with self.subTest(text=text):
                self.prepare()
                self.assertFalse(MODE.is_courtesy_only(text))
                self.assert_formed(self.say(turn("courtesy-question", 2000, 2400, text)))

    def test_perfect_without_a_question_asked_is_still_a_courtesy(self):
        self.prepare()
        self.assert_refused(self.say(turn("perfect-courtesy", 2000, 2400, "Perfect.")))

    def test_perfect_ending_as_a_question_keeps_the_existing_cue_requirement(self):
        self.prepare(replace(MODE, require_cue=True), asked=True)
        current = turn("perfect-question", 2000, 2400, "Perfect?")
        self.assertFalse(self.events._answers_question(current))
        self.assert_refused(self.say(current), reason="no_cue")

    def test_split_courtesy_unattributed_then_owner_forms_neither_follow_up(self):
        for overlap in (False, True):
            with self.subTest(overlap=overlap):
                self.prepare(provenance="diarization-timeline")
                parts = []
                merger = TurnMerger(2000, 24000, parts.append, tail_join_ms=400)
                merger.offer(fragment("Okay.", 2000, 2200, speaker=None, overlap=overlap))
                merger.offer(fragment("Thanks.", 2300, 2500))
                merger.flush()
                # An unattributed head cannot take a labelled tail; no identity invented.
                self.assertEqual([part["speaker"] for part in parts], [None, "Speaker A"])
                for index, part in enumerate(parts):
                    output = self.say(
                        turn(
                            f"split-{index}",
                            part["start_ms"],
                            part["end_ms"],
                            part["text"],
                            speaker=part["speaker"],
                            overlap=part["overlap"],
                            provenance=part["speaker_provenance"],
                        )
                    )
                    if index == 0:
                        self.assertFalse(any(event["type"] == "request" for event in output))
                        self.assertFalse(any(note.startswith("follow_up ") for note in self.notes))
                    else:
                        self.assert_refused(output)

    def test_tail_joined_courtesies_do_not_form_a_follow_up(self):
        self.prepare(provenance="diarization-timeline")
        parts = []
        merger = TurnMerger(2000, 24000, parts.append, tail_join_ms=400)
        merger.offer(fragment("Okay.", 2000, 2200))
        merger.offer(fragment("Thanks.", 2300, 2500, speaker=None))
        merger.flush()
        self.assertEqual(len(parts), 1)
        joined = parts[0]
        self.assertEqual(joined["text"], "Okay. Thanks.")
        current = turn(
            "joined-courtesy",
            joined["start_ms"],
            joined["end_ms"],
            joined["text"],
            speaker=joined["speaker"],
            provenance=joined["speaker_provenance"],
        )
        self.assert_refused(self.say(current, inferred=True))

    def test_courtesy_diagnostic_is_content_free(self):
        self.prepare()
        current = turn("private-authored-id", 2000, 2400, "Thank you.")
        self.assert_refused(self.say(current))
        note = next(note for note in self.notes if note.startswith("follow_up "))
        self.assertEqual(note, "follow_up outcome=refused reason=courtesy attend=0.5 bar=0.5")
        for content in (current.text, current.utterance_id, current.speaker_id, SESSION, "Jarvis"):
            self.assertNotIn(content, note)


class CourtesyParsingTests(unittest.TestCase):
    def test_only_complete_courtesy_units_are_accepted(self):
        for text in (
            "okay",
            "ok",
            "alright",
            "all right",
            "great",
            "perfect",
            "excellent",
            "thanks",
            "thank you",
            "cheers",
            "cool",
            "OKAY! Thank—you. Cheers.",
            "  great\nthanks  ",
        ):
            with self.subTest(text=text):
                self.assertTrue(MODE.is_courtesy_only(text))
                self.assertFalse(MODE.is_closing(text))
        for text in (
            "",
            "...",
            "  ",
            "thank",
            "you",
            "all",
            "right",
            "thanks for that",
            "thanks, open it",
            "okay?",
            "Thank you?!",
            'Thanks?…")]',
            "cool？！",
            "Okay. That's all.",
        ):
            with self.subTest(text=text):
                self.assertFalse(MODE.is_courtesy_only(text))

    def test_a_configured_courtesy_closing_remains_a_closing(self):
        mode = replace(MODE, closing_phrases=("thanks",))
        self.assertTrue(mode.is_closing("Thanks. Okay."))
        self.assertFalse(mode.is_courtesy_only("Thanks. Okay."))
        self.assertTrue(mode.is_courtesy_only("Okay."))
        self.assertFalse(mode.is_closing("Okay."))
        empty = replace(MODE, closing_phrases=())
        self.assertTrue(empty.is_courtesy_only("Thanks."))
        self.assertFalse(empty.is_closing("Thanks."))
