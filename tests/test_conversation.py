"""Conversation mode (#82): engaged follow-ups after a request, then back to ambient.

Authored text and test doubles only: no capture, native inference, hosted calls or
recorded speech.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rightyo.contracts import (
    DEFAULT_CONVERSATION_WINDOW_MS,
    ContractError,
    Conversation,
    DecisionEvent,
    Dismissal,
    ProviderDecision,
    SpeakerPriority,
    Turn,
)
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents

SESSION = "conversation-test"
MODE = Conversation(window_ms=10000)


def turn(name, start, end, text, speaker="Speaker A", **extra):
    return Turn(
        SESSION,
        name,
        1,
        start,
        end,
        text,
        speaker,
        True,
        extra.get("overlap", False),
        "authored-fixture",
        "synthetic",
        extra.get("provenance", "authored-fixture"),
    )


def decided(current, label, recipient="system", attend=None, confidence=0.9):
    """A decision event; `attend` sets the attend probability of an `uncertain` label."""
    if attend is None:
        probabilities = {k: float(k == label) for k in ("attend", "ignore", "uncertain")}
    else:
        probabilities = {"attend": attend, "ignore": 0.0, "uncertain": 1.0 - attend}
    return DecisionEvent(
        current,
        ProviderDecision(
            label, recipient, confidence, probabilities, "mock-v1", "mock", confidence
        ),
        current.revision,
        0.0,
        0.0,
    )


class ConversationEventTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()

    def start(self, conversation=MODE, **options):
        self.events.start(SESSION, now_ms=0, conversation=conversation, **options)
        return self.events.drain()[0]

    def say(self, current, label, **options):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return [
            event
            for event in self.events.drain()
            if event["type"] in {"attention", "request", "conversation", "dismiss"}
        ]

    @staticmethod
    def kinds(events):
        return [event["type"] for event in events]

    def test_off_by_default_emits_no_conversation_events_or_follow_ups(self):
        started = self.start(conversation=None)
        self.assertNotIn("conversation", started)
        self.assertEqual(
            self.kinds(self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")),
            ["attention", "request"],
        )
        events = self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.8)
        self.assertEqual(self.kinds(events), ["attention"])

    def test_the_session_advertises_the_configuration(self):
        started = self.start()
        self.assertEqual(
            started["conversation"],
            {
                "version": 1,
                "window_ms": 10000,
                "follow_up_min_probability": 0.4,
                "closing_phrases": list(MODE.closing_phrases),
            },
        )

    def test_a_request_engages_its_speaker(self):
        self.start()
        events = self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        self.assertEqual(self.kinds(events), ["attention", "request", "conversation"])
        engaged = events[2]
        self.assertEqual(
            {k: engaged[k] for k in ("state", "reason", "speaker_id", "at_ms", "until_ms")},
            {
                "state": "engaged",
                "reason": "request",
                "speaker_id": "Speaker A",
                "at_ms": 900,
                "until_ms": 10900,
            },
        )
        self.assertEqual(engaged["request_id"], events[1]["request_id"])

    def test_an_engaged_speakers_undecided_follow_up_becomes_a_request(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        for recipient in ("system", "unknown"):
            with self.subTest(recipient=recipient):
                current = turn("t2-" + recipient, 2000, 2900, "And tomorrow?")
                events = self.say(current, "uncertain", recipient=recipient, attend=0.55)
                # Extending the window emits nothing: the state is unchanged.
                self.assertEqual(self.kinds(events), ["attention", "request"])
                self.assertEqual(events[0]["decision"]["label"], "attend")
                self.assertIs(events[0]["decision"]["follow_up"], True)
                self.assertEqual(events[0]["decision"]["recipient_kind"], recipient)
                self.assertIs(events[1]["decision"]["follow_up"], True)

    def test_follow_ups_need_the_same_speaker_and_a_leaning_toward_attend(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        cases = {
            "another speaker": (turn("a", 2000, 2900, "And then?", "Speaker B"), {}),
            "unattributed": (turn("b", 2000, 2900, "And then?", None), {}),
            "overlapping": (turn("c", 2000, 2900, "And then?", overlap=True), {}),
            "too unsure": (turn("d", 2000, 2900, "And then?"), {"attend": 0.3}),
            "another person": (
                turn("e", 2000, 2900, "And then?"),
                {"recipient": "speaker_1"},
            ),
        }
        for name, (current, options) in cases.items():
            with self.subTest(name):
                events = self.say(current, "uncertain", **{"attend": 0.8, **options})
                self.assertEqual(self.kinds(events), ["attention"])
                self.assertEqual(events[0]["decision"]["label"], "uncertain")
                self.assertNotIn("follow_up", events[0]["decision"])

    def test_a_confident_ignore_is_never_promoted(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        events = self.say(turn("t2", 2000, 2900, "Hmm."), "ignore", recipient="unknown")
        self.assertEqual(self.kinds(events), ["attention"])

    def test_the_window_lapses_to_ambient_and_each_request_extends_it(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        # A second request at 8 s extends the window to 18.9 s.
        events = self.say(turn("t2", 8000, 8900, "Haili, and the weather?"), "attend")
        self.assertEqual(self.kinds(events), ["attention", "request"])
        events = self.say(turn("t3", 15000, 15900, "And tomorrow?"), "uncertain", attend=0.6)
        self.assertEqual(self.kinds(events), ["attention", "request"])
        # Nothing for 25.9 s: the next decided turn finds the window lapsed.
        events = self.say(turn("t4", 41000, 41900, "And then?"), "uncertain", attend=0.6)
        self.assertEqual(self.kinds(events), ["conversation", "attention"])
        self.assertEqual(
            {k: events[0][k] for k in ("state", "reason", "speaker_id", "at_ms")},
            {"state": "ambient", "reason": "timeout", "speaker_id": "Speaker A", "at_ms": 25900},
        )

    def test_a_closing_phrase_returns_to_ambient_and_never_re_engages(self):
        # Whether Jev ignores the closing turn or attends it, engagement ends there.
        for label in ("ignore", "attend"):
            with self.subTest(label):
                self.events = SpeechEvents()
                self.start()
                self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
                events = self.say(turn("t2", 3000, 3900, "Thanks, that's all."), label)
                closing = [e for e in events if e["type"] == "conversation"]
                self.assertEqual(len(closing), 1)
                self.assertEqual((closing[0]["state"], closing[0]["reason"]), ("ambient", "closed"))
                events = self.say(turn("t3", 4000, 4900, "And?"), "uncertain", attend=0.9)
                self.assertEqual(self.kinds(events), ["attention"])

    def test_talking_to_another_person_returns_to_ambient(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        events = self.say(
            turn("t2", 2000, 2900, "Sam, did you see that?"), "ignore", recipient="other_human"
        )
        self.assertEqual(self.kinds(events), ["attention", "conversation"])
        self.assertEqual((events[1]["state"], events[1]["reason"]), ("ambient", "other_human"))
        self.assertEqual(events[1]["utterance_id"], "t2")

    def test_another_speaker_talking_to_someone_leaves_the_engagement(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        events = self.say(
            turn("t2", 2000, 2900, "Sam, look.", "Speaker B"), "ignore", recipient="other_human"
        )
        self.assertEqual(self.kinds(events), ["attention"])
        events = self.say(turn("t3", 3000, 3900, "And tomorrow?"), "uncertain", attend=0.6)
        self.assertEqual(self.kinds(events), ["attention", "request"])

    def test_a_stop_phrase_dismissal_returns_to_ambient(self):
        self.start(dismissal=Dismissal())
        self.say(turn("t1", 0, 900, "Haili, play some music."), "attend")
        current = turn("t2", 2000, 2400, "Stop.")
        self.events.transcript(current, current.end_ms)
        events = self.events.drain()
        self.assertEqual(self.kinds(events), ["transcript", "dismiss", "conversation"])
        self.assertEqual((events[2]["state"], events[2]["reason"]), ("ambient", "dismissed"))

    def test_another_speakers_request_moves_the_engagement(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        events = self.say(turn("t2", 2000, 2900, "Haili, the news?", "Speaker B"), "attend")
        self.assertEqual(self.kinds(events), ["attention", "request", "conversation"])
        self.assertEqual(events[2]["speaker_id"], "Speaker B")
        # Speaker A is no longer engaged.
        events = self.say(turn("t3", 3000, 3900, "And tomorrow?"), "uncertain", attend=0.6)
        self.assertEqual(self.kinds(events), ["attention"])

    def test_speech_from_before_the_engaging_request_is_never_a_follow_up(self):
        self.start()
        musing = turn("t0", 0, 900, "Hmm, I wonder about tomorrow.")
        request = turn("t1", 1500, 2400, "Haili, what time is it?")
        self.events.transcript(musing, musing.end_ms)
        self.events.transcript(request, request.end_ms)
        # Decisions may arrive out of turn order: the request is decided first.
        self.events.decision(decided(request, "attend"), 2500)
        self.events.decision(decided(musing, "uncertain", attend=0.6), 2600)
        events = [e for e in self.events.drain() if e["type"] != "transcript"]
        self.assertEqual(self.kinds(events), ["attention", "request", "conversation", "attention"])
        self.assertEqual(events[3]["decision"]["label"], "uncertain")

    def test_an_older_request_decided_late_never_takes_the_engagement(self):
        self.start()
        older = turn("t1", 0, 900, "Haili, what time is it?")
        newer = turn("t2", 2000, 2900, "Haili, the news?", "Speaker B")
        self.events.transcript(older, older.end_ms)
        self.events.transcript(newer, newer.end_ms)
        self.events.decision(decided(newer, "attend"), 3000)
        self.events.decision(decided(older, "attend"), 3100)
        events = [e for e in self.events.drain() if e["type"] == "conversation"]
        self.assertEqual([e["speaker_id"] for e in events], ["Speaker B"])
        follow = self.say(
            turn("t3", 4000, 4900, "And tomorrow?", "Speaker B"), "uncertain", attend=0.6
        )
        self.assertEqual(self.kinds(follow), ["attention", "request"])

    def test_a_participants_playback_only_stop_still_ends_their_own_engagement(self):
        owner = SpeakerPriority(owners=("Speaker Z",))
        self.start(dismissal=Dismissal(), priority=ConfiguredPriorityProvider(owner))
        self.say(turn("t1", 0, 900, "Haili, play some music."), "attend")
        current = turn("t2", 2000, 2400, "Stop.")
        self.events.transcript(current, current.end_ms)
        events = [e for e in self.events.drain() if e["type"] != "transcript"]
        self.assertEqual(self.kinds(events), ["dismiss", "conversation"])
        self.assertEqual(events[0]["scope"], ["playback"])
        self.assertEqual((events[1]["state"], events[1]["reason"]), ("ambient", "dismissed"))
        events = self.say(turn("t3", 3000, 3900, "And the other one."), "uncertain", attend=0.6)
        self.assertEqual(self.kinds(events), ["attention"])

    def test_owner_only_never_promotes_another_speaker(self):
        owner = SpeakerPriority(owners=("Speaker A",), owner_only=True)
        self.start(priority=ConfiguredPriorityProvider(owner))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        events = self.say(turn("t2", 2000, 2900, "And?", "Speaker B"), "uncertain", attend=0.9)
        self.assertEqual(self.kinds(events), ["attention"])
        # The owner's own follow-up still forms a request.
        events = self.say(turn("t3", 3000, 3900, "And tomorrow?"), "uncertain", attend=0.6)
        self.assertEqual(self.kinds(events), ["attention", "request"])

    def test_a_closing_phrase_is_not_promoted_to_a_request(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        events = self.say(turn("t2", 2000, 2600, "Never mind."), "uncertain", attend=0.9)
        self.assertEqual(self.kinds(events), ["attention", "conversation"])
        self.assertEqual(events[0]["decision"]["label"], "uncertain")
        self.assertEqual(events[1]["reason"], "closed")

    def test_a_follow_up_held_back_by_a_cool_down_is_not_reported_as_one(self):
        self.start(dismissal=Dismissal(cooldown_ms=60000))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        # Another speaker's disengage starts a cool-down for everyone without moving
        # the engagement (their label is not comparable: overlapping speech).
        self.events._cooldowns[None] = (1000, 61000)
        events = self.say(
            turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.6, confidence=0.5
        )
        self.assertEqual(self.kinds(events), ["attention"])
        self.assertIs(events[0]["decision"]["cooldown"], True)
        self.assertNotIn("follow_up", events[0]["decision"])

    def test_an_unattributed_request_engages_no_one(self):
        self.start()
        events = self.say(turn("t1", 0, 900, "Haili, what time is it?", None), "attend")
        self.assertEqual(self.kinds(events), ["attention", "request"])

    def test_an_unavailable_decision_is_never_a_follow_up(self):
        self.start()
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        current = turn("t2", 2000, 2900, "And tomorrow?")
        self.events.transcript(current, current.end_ms)
        self.events.decision(
            decided(current, "uncertain", attend=0.9), 3000, unavailable="provider-timeout"
        )
        events = [e for e in self.events.drain() if e["type"] != "transcript"]
        self.assertEqual(self.kinds(events), ["attention"])


class FollowUpNoteTests(unittest.TestCase):
    """The follow-up diagnostic (#153): every undecided turn of the engaged speaker says
    whether it became a follow-up, why not, Jev's attend probability and the bar."""

    def setUp(self):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.events.start(SESSION, now_ms=0, conversation=MODE)
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")

    def say(self, current, label, **options):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        self.events.drain()
        return [note for note in self.notes if note.startswith("follow_up ")]

    def test_a_formed_follow_up_and_a_refused_one_are_both_noted(self):
        self.assertEqual(
            self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.55),
            ["follow_up outcome=formed reason=passed attend=0.55 bar=0.4"],
        )
        self.assertEqual(
            self.say(turn("t3", 4000, 4900, "I heard that."), "uncertain", attend=0.04)[-1],
            "follow_up outcome=refused reason=below_bar attend=0.04 bar=0.4",
        )

    def test_each_refusal_reason_is_a_fixed_token(self):
        cases = {
            "recipient": ({"attend": 0.8, "recipient": "speaker_1"}, "uncertain"),
            "label": ({}, "ignore"),
            "closing": ({"attend": 0.8}, "uncertain"),
        }
        start = 2000
        for reason, (options, label) in cases.items():
            with self.subTest(reason):
                text = "That's all" if reason == "closing" else "And then?"
                notes = self.say(turn("r" + reason, start, start + 900, text), label, **options)
                self.assertIn(f"reason={reason} ", notes[-1])
                start += 2000

    def test_a_follow_up_a_cool_down_holds_back_is_noted_as_held(self):
        notes = []
        events = SpeechEvents(report=notes.append)
        events.start(SESSION, now_ms=0, conversation=MODE, dismissal=Dismissal(cooldown_ms=60000))
        for current, label, options in (
            (turn("t1", 0, 900, "Haili, what time is it?"), "attend", {}),
            (
                turn("t2", 2000, 2900, "And tomorrow?"),
                "uncertain",
                {"attend": 0.6, "confidence": 0.5},
            ),
        ):
            if current.utterance_id == "t2":
                # Another speaker's disengage: a cool-down for everyone, engagement intact.
                events._cooldowns[None] = (1000, 61000)
            events.transcript(current, current.end_ms)
            events.decision(decided(current, label, **options), current.end_ms + 100)
        follow_ups = [note for note in notes if note.startswith("follow_up ")]
        self.assertEqual(follow_ups, ["follow_up outcome=held reason=cooldown attend=0.6 bar=0.4"])

    def test_a_near_miss_never_prints_as_a_tie(self):
        notes = self.say(turn("t2", 2000, 2900, "Right."), "uncertain", attend=0.399)
        self.assertEqual(
            notes[-1], "follow_up outcome=refused reason=below_bar attend=0.399 bar=0.4"
        )

    def test_other_speakers_and_turns_outside_a_conversation_are_not_noted(self):
        self.assertEqual(self.say(turn("o", 2000, 2900, "And then?", "Speaker B"), "uncertain"), [])
        self.assertEqual(self.say(turn("late", 30000, 30900, "And then?"), "uncertain"), [])

    def test_notes_never_carry_transcript_text_or_ids(self):
        self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.55)
        for note in self.notes:
            for content in ("Haili", "tomorrow", "t1", "t2", SESSION, "Speaker"):
                self.assertNotIn(content, note)


class ReplyNoteTests(unittest.TestCase):
    """The reply-report diagnostic (rightyo#158): every report says what it did to the window."""

    def setUp(self):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.events.start(SESSION, now_ms=0, conversation=MODE)

    def replies(self):
        return [note for note in self.notes if note.startswith("reply ")]

    def engage(self):
        current = turn("t1", 0, 900, "Haili, what time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, "attend"), current.end_ms + 100)
        self.events.drain()

    def test_a_report_with_nothing_engaged_says_so(self):
        self.events.reply("ended", 500)
        self.assertEqual(
            self.replies(), ["reply phase=ended action=not_engaged window_left_ms=none"]
        )

    def test_a_reply_holds_then_extends_the_window(self):
        self.engage()  # engaged until 900 + 10000
        self.events.reply("started", 2000)
        self.events.reply("ended", 5000)
        self.assertEqual(
            self.replies(),
            [
                "reply phase=started action=held window_left_ms=180000",
                "reply phase=ended action=extended window_left_ms=10000",
            ],
        )

    def test_a_reply_after_a_timeout_revives_the_conversation(self):
        # rightyo#158: a proactive relay after the window lapsed re-engages the speaker.
        self.engage()  # engaged until 10900
        self.events.reply("started", 20000)
        self.events.reply("ended", 25000)
        self.assertEqual(
            self.replies(),
            [
                "reply phase=started action=revived window_left_ms=180000",
                "reply phase=ended action=extended window_left_ms=10000",
            ],
        )
        timeouts = [e for e in self.events.drain() if e["type"] == "conversation"]
        self.assertEqual([(e["state"], e["reason"]) for e in timeouts], [("ambient", "timeout")])
        # The speaker's answer after the reply is a follow-up; nothing from before it counts.
        answer = turn("t2", 27000, 28000, "Yes, do that.")
        self.events.transcript(answer, answer.end_ms)
        self.events.decision(decided(answer, "uncertain", attend=0.5), answer.end_ms + 100)
        requests = [e for e in self.events.drain() if e["type"] == "request"]
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0]["decision"]["follow_up"], True)

    def test_a_stale_lapse_is_not_revived(self):
        self.engage()
        self.events.reply("ended", 10900 + 600_001 + 1)
        self.assertEqual(
            self.replies()[-1], "reply phase=ended action=not_engaged window_left_ms=none"
        )

    def test_the_lapse_event_is_never_stamped_before_the_time_it_names(self):
        self.engage()  # engaged until 10900; the last decision was at ~1000
        self.events.reply("started", 20000)
        lapse = [e for e in self.events.drain() if e["type"] == "conversation"][0]
        self.assertLessEqual(lapse["at_ms"], lapse["emitted_at_ms"])

    def test_a_closing_said_after_the_lapse_is_not_revived(self):
        self.engage()
        closing = turn("t2", 15000, 15600, "That's all")
        self.events.transcript(closing, closing.end_ms)
        self.events.decision(decided(closing, "uncertain", attend=0.2), closing.end_ms + 100)
        self.events.reply("ended", 20000)
        self.assertEqual(
            self.replies()[-1], "reply phase=ended action=not_engaged window_left_ms=none"
        )

    def test_a_reply_to_another_request_does_not_revive_the_lapsed_speaker(self):
        self.engage()
        # Another, unattributed speaker's request is delivered after the lapse; it engages no one.
        other = turn("t3", 30000, 30900, "Haili, play some music.", None)
        self.events.transcript(other, other.end_ms)
        self.events.decision(decided(other, "attend"), other.end_ms + 100)
        self.events.reply("ended", 35000)
        self.assertEqual(
            self.replies()[-1], "reply phase=ended action=not_engaged window_left_ms=none"
        )

    def test_an_explicit_end_is_never_revived(self):
        self.engage()
        closing = turn("t2", 2000, 2600, "That's all")
        self.events.transcript(closing, closing.end_ms)
        self.events.decision(decided(closing, "uncertain", attend=0.8), closing.end_ms + 100)
        self.events.reply("ended", 3000)
        self.assertEqual(
            self.replies()[-1], "reply phase=ended action=not_engaged window_left_ms=none"
        )


class ReplyTimingTests(unittest.TestCase):
    """The window runs from the end of the spoken reply when the host reports it (#124)."""

    def setUp(self):
        self.events = SpeechEvents()
        self.events.start(SESSION, now_ms=0, conversation=MODE)
        self.events.drain()

    def say(self, current, label, **options):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return [e["type"] for e in self.events.drain() if e["type"] != "transcript"]

    def test_the_window_restarts_when_the_reply_ends(self):
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        self.events.reply("started", 3000)
        self.events.reply("ended", 9000)
        # 10.9 s would have lapsed; the reply ended at 9 s, so the window runs to 19 s.
        kinds = self.say(turn("t2", 17000, 17900, "And tomorrow?"), "uncertain", attend=0.6)
        self.assertEqual(kinds, ["attention", "request"])

    def test_a_playing_reply_holds_the_window_open(self):
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        self.events.reply("started", 2000)
        # Still playing at 30 s (long render): the conversation has not lapsed.
        kinds = self.say(turn("t2", 30000, 30900, "Wait, and tomorrow?"), "uncertain", attend=0.6)
        self.assertEqual(kinds, ["attention", "request"])

    def test_a_reply_that_never_ends_is_bounded(self):
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend")
        self.events.reply("started", 2000)
        kinds = self.say(turn("t2", 200000, 200900, "And then?"), "uncertain", attend=0.6)
        self.assertEqual(kinds, ["conversation", "attention"])

    def test_a_report_without_engagement_changes_nothing_and_one_after_a_lapse_revives(self):
        self.events.reply("ended", 500)
        kinds = self.say(turn("t0", 1000, 1900, "And?"), "uncertain", attend=0.9)
        self.assertEqual(kinds, ["attention"])
        self.say(turn("t1", 2000, 2900, "Haili, what time is it?"), "attend")
        # The window ended at 12.9 s. A reply reported at 20 s (a proactive one, say) says
        # the conversation lapsed, then revives it from 20 s (rightyo#158).
        self.events.reply("ended", 20000)
        kinds = self.say(turn("t2", 21000, 21900, "And then?"), "uncertain", attend=0.6)
        self.assertEqual(kinds, ["conversation", "attention", "request"])

    def test_reports_are_validated(self):
        for phase in ("finished", None, 1):
            with self.subTest(phase=phase), self.assertRaises(ContractError):
                self.events.reply(phase, 100)


class ConversationConfigTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        mode = Conversation()
        self.assertEqual(mode.window_ms, DEFAULT_CONVERSATION_WINDOW_MS)
        self.assertTrue(mode.is_closing("Thanks — that's ALL!"))
        self.assertTrue(mode.is_closing("never mind"))
        self.assertFalse(mode.is_closing("That's all the news?"))
        self.assertEqual(Conversation.from_dict({}), mode)
        self.assertEqual(
            Conversation.from_dict({"closing_phrases": ["we're good"]}).closing_phrases,
            ("we're good",),
        )
        for raw in (
            [],
            {"unknown": 1},
            {"window_ms": 999},
            {"window_ms": 120001},
            {"window_ms": 1.5},
            {"follow_up_min_probability": 0},
            {"follow_up_min_probability": 1.5},
            {"closing_phrases": "that's all"},
            {"closing_phrases": ["!!!"]},
            {"closing_phrases": [""]},
            {"closing_phrases": ["x"] * 17},
        ):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                Conversation.from_dict(raw)

    def test_the_prototype_section_is_off_unless_present(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset"
            asset.touch()
            config = Path(directory) / "config.json"
            base = {
                name: str(asset)
                for name in (
                    "whisper_executable",
                    "whisper_model",
                    "diarization_library",
                    "diarization_model",
                    "microphone_helper",
                )
            }
            config.write_text(json.dumps(base))
            self.assertIsNone(PrototypeConfig.load(config).conversation)
            config.write_text(json.dumps(base | {"conversation": {}}))
            self.assertEqual(PrototypeConfig.load(config).conversation, Conversation())
            config.write_text(json.dumps(base | {"conversation": {"window_ms": 5}}))
            with self.assertRaises(PrototypeError):
                PrototypeConfig.load(config)


class ControlInputTests(unittest.TestCase):
    """`listen --control-fd`: the host's reply reports over an inherited pipe (#124)."""

    class Recorder:
        def __init__(self):
            self.phases = []

        def reply(self, phase):
            if phase not in {"started", "ended"}:
                raise PrototypeError("invalid reply phase")
            self.phases.append(phase)

    def test_reports_are_forwarded_and_junk_is_skipped(self):
        import contextlib
        import io
        import os

        from rightyo.tool import _read_control

        read, write = os.pipe()
        os.write(
            write,
            b'{"reply": "started"}\n'
            b"not json\n"
            b'{"reply": "paused"}\n'
            b'{"reply": "ended", "extra": 1}\n' + b"x" * 300 + b"\n"
            b'{"reply": "ended"}\n',
        )
        os.close(write)
        recorder = self.Recorder()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            _read_control(read, recorder)
        self.assertEqual(recorder.phases, ["started", "ended"])
        self.assertEqual(errors.getvalue().count("skipped"), 4)

    def test_the_tail_of_an_overlong_line_is_never_applied(self):
        import contextlib
        import io
        import os

        from rightyo.tool import _read_control

        read, write = os.pipe()
        # 600 bytes whose tail is itself valid JSON, then a blank line and a real report.
        os.write(write, b"x" * 580 + b'{"reply": "ended"}\n\n{"reply": "started"}\n')
        os.close(write)
        recorder = self.Recorder()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            _read_control(read, recorder)
        self.assertEqual(recorder.phases, ["started"])
        self.assertEqual(errors.getvalue().count("too long"), 1)
        self.assertNotIn("invalid", errors.getvalue())

    def test_listen_requires_stdin_mode_and_a_free_descriptor(self):
        from argparse import Namespace

        from rightyo.tool import listen

        for mode, fd, provenance in (("demo", 3, None), ("stdin", 2, "live-microphone")):
            args = Namespace(
                config="unused",
                mode=mode,
                provenance=provenance,
                control_fd=fd,
                session_id=None,
                use_jev=False,
                allow_hosted=False,
            )
            with self.subTest(mode=mode, fd=fd), self.assertRaises(PrototypeError):
                listen(args)


if __name__ == "__main__":
    unittest.main()
