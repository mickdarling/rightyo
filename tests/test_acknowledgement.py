"""Acknowledgement gating (#132): ack a request only when a reply is likely.

Authored text and test doubles only: no capture, native inference, hosted calls or
recorded speech.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from rightyo.contracts import (
    Acknowledgement,
    Addressing,
    ContractError,
    Conversation,
    DecisionEvent,
    ProviderDecision,
    SpeakerPriority,
    Turn,
)
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents

SESSION = "acknowledgement-test"
# Gating alone, for tests whose requests fall inside the default dedup window (#122).
NO_DEDUP = Acknowledgement(dedup_window_ms=0)


def turn(name, start, end, text, speaker="Speaker A"):
    return Turn(
        SESSION,
        name,
        1,
        start,
        end,
        text,
        speaker,
        True,
        False,
        "authored-fixture",
        "synthetic",
        "authored-fixture",
    )


def decided(current, label, confidence, attend=None):
    """A decision event; `attend` sets the attend probability of an `uncertain` label."""
    if attend is None:
        probabilities = {k: float(k == label) for k in ("attend", "ignore", "uncertain")}
    else:
        probabilities = {"attend": attend, "ignore": 0.0, "uncertain": 1.0 - attend}
    return DecisionEvent(
        current,
        ProviderDecision(label, "system", confidence, probabilities, "mock-v1", "mock", confidence),
        current.revision,
        0.0,
        0.0,
    )


class SessionHarness:
    """An event session with gating on and a recorded diagnostic stream."""

    def setUp(self):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)

    def start(self, acknowledgement=Acknowledgement(), **options):
        options.setdefault("addressing", Addressing(("Haili",)))
        self.events.start(SESSION, now_ms=0, acknowledgement=acknowledgement, **options)
        return self.events.drain()[0]

    def say(self, current, label, confidence, attend=None):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, confidence, attend), current.end_ms + 100)
        requests = [e for e in self.events.drain() if e["type"] == "request"]
        return requests[0] if requests else None


class AcknowledgementEventTests(SessionHarness, unittest.TestCase):
    def test_off_by_default_changes_nothing(self):
        started = self.start(acknowledgement=None)
        self.assertNotIn("acknowledgement", started)
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.5)
        self.assertNotIn("acknowledge", request)
        self.assertEqual(self.notes, [])

    def test_the_session_advertises_the_threshold(self):
        started = self.start()
        self.assertEqual(
            started["acknowledgement"],
            {"version": 1, "min_confidence": 0.7, "dedup_window_ms": 8000},
        )

    def test_a_confident_attend_is_acknowledged(self):
        self.start()
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.92)
        self.assertIs(request["acknowledge"], True)
        self.assertEqual(
            self.notes,
            ["ack outcome=ack reason=confident attend_confidence=0.92 min=0.70 follow_up=false"],
        )

    def test_a_low_confidence_attend_is_submitted_without_an_ack(self):
        self.start()
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.6)
        # The request is still delivered; only the acknowledgement is withheld.
        self.assertIsNotNone(request)
        self.assertIs(request["acknowledge"], False)
        self.assertIn("outcome=skip reason=low_confidence", self.notes[0])

    def test_a_failing_note_never_disturbs_delivery(self):
        def broken(_note):
            raise BrokenPipeError("stderr closed")

        self.events = SpeechEvents(report=broken)
        self.start()
        first = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.92)
        second = self.say(turn("t2", 2000, 2900, "And the weather?"), "attend", 0.6)
        self.assertIs(first["acknowledge"], True)
        self.assertIs(second["acknowledge"], False)

    def test_a_name_addressed_turn_is_acknowledged_below_the_threshold(self):
        self.start()
        request = self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend", 0.5)
        self.assertIs(request["acknowledge"], True)
        self.assertIn("outcome=ack reason=named", self.notes[0])

    def test_the_threshold_is_inclusive(self):
        self.start()
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.7)
        self.assertIs(request["acknowledge"], True)

    def test_a_zero_threshold_acknowledges_every_request(self):
        self.start(acknowledgement=Acknowledgement(min_confidence=0))
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.01)
        self.assertIs(request["acknowledge"], True)

    def test_a_low_confidence_follow_up_gets_no_ack(self):
        # Live evidence (#132): "Okay, great." as a follow-up, confidence 0.22.
        self.start(conversation=Conversation(window_ms=10000))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend", 0.92)
        request = self.say(turn("t2", 2000, 2600, "Okay, great."), "uncertain", 0.22, 0.45)
        self.assertIs(request["decision"]["follow_up"], True)
        self.assertIs(request["acknowledge"], False)
        self.assertEqual(
            [note for note in self.notes if note.startswith("ack ")][-1],
            "ack outcome=skip reason=low_confidence attend_confidence=0.45 min=0.70 follow_up=true",
        )

    def test_a_follow_up_is_judged_on_its_attend_probability(self):
        # A follow-up's `confidence` belongs to its `uncertain` choice; a strong attend
        # probability still earns the acknowledgement, a high `confidence` alone does not.
        self.start(acknowledgement=NO_DEDUP, conversation=Conversation(window_ms=10000))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend", 0.92)
        strong = self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", 0.3, 0.8)
        self.assertIs(strong["acknowledge"], True)
        weak = self.say(turn("t3", 4000, 4900, "And Friday?"), "uncertain", 0.95, 0.45)
        self.assertIs(weak["acknowledge"], False)

    def test_a_named_follow_up_is_acknowledged(self):
        self.start(acknowledgement=NO_DEDUP, conversation=Conversation(window_ms=10000))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend", 0.92)
        request = self.say(turn("t2", 2000, 2900, "And tomorrow, Haili?"), "uncertain", 0.2, 0.45)
        self.assertIs(request["acknowledge"], True)

    def test_an_owner_request_carries_the_field_through_the_override_reserve(self):
        owner = SpeakerPriority(owners=("Speaker A",))
        self.start(priority=ConfiguredPriorityProvider(owner))
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.5)
        self.assertIs(request["acknowledge"], False)

    def test_notes_never_carry_transcript_text_or_ids(self):
        self.start(conversation=Conversation(window_ms=10000))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend", 0.92)
        self.say(turn("t2", 2000, 2600, "Okay, great."), "uncertain", 0.22, 0.45)
        # Two acknowledgement notes, then the follow-up note, written after its request (#153).
        self.assertEqual(len(self.notes), 3)
        self.assertTrue(self.notes[2].startswith("follow_up "))
        for note in self.notes:
            for content in ("Haili", "time", "great", "t1", "t2", SESSION, "Speaker"):
                self.assertNotIn(content, note)

    def test_unattended_turns_produce_no_note(self):
        self.start()
        self.assertIsNone(self.say(turn("t1", 0, 900, "Pass the salt."), "ignore", 0.95))
        self.assertEqual(self.notes, [])

    def test_start_rejects_a_foreign_object(self):
        with self.assertRaises(ContractError):
            self.events.start(SESSION, acknowledgement={"min_confidence": 0.7})


class OneAcknowledgementPerRequestTests(SessionHarness, unittest.TestCase):
    """One acknowledgement per spoken request (#122), within acknowledgement gating."""

    def acks(self, *requests):
        return [request["acknowledge"] for request in requests]

    def test_fragments_of_one_request_get_one_ack(self):
        # Live repro (#122): one question submitted as three requests, three acks.
        self.start(conversation=Conversation(window_ms=10000))
        first = self.say(turn("t1", 0, 1200, "Can you tell me what"), "attend", 0.9)
        second = self.say(turn("t2", 1700, 2600, "time it is?"), "attend", 0.95)
        third = self.say(turn("t3", 3400, 3900, "Please."), "uncertain", 0.3, 0.8)
        # Every fragment is still delivered; only the first is acknowledged.
        self.assertEqual(self.acks(first, second, third), [True, False, False])
        acks = [note for note in self.notes if note.startswith("ack ")]
        self.assertIn("outcome=ack reason=confident", acks[0])
        self.assertEqual(
            acks[1],
            "ack outcome=skip reason=pending_ack attend_confidence=0.95 min=0.70 follow_up=false",
        )
        self.assertIn("outcome=skip reason=pending_ack", acks[2])
        self.assertIn("follow_up=true", acks[2])

    def test_a_named_fragment_is_not_acknowledged_twice(self):
        self.start()
        first = self.say(turn("t1", 0, 900, "Haili, what time"), "attend", 0.9)
        second = self.say(turn("t2", 1500, 2400, "is it, Haili?"), "attend", 0.9)
        self.assertEqual(self.acks(first, second), [True, False])
        self.assertIn("reason=pending_ack", self.notes[1])

    def test_a_low_confidence_skip_keeps_its_reason_and_holds_nothing(self):
        self.start()
        first = self.say(turn("t1", 0, 900, "Okay then"), "attend", 0.4)
        second = self.say(turn("t2", 1500, 2400, "What time is it?"), "attend", 0.9)
        self.assertEqual(self.acks(first, second), [False, True])
        self.assertIn("reason=low_confidence", self.notes[0])
        third = self.say(turn("t3", 3000, 3500, "Hmm."), "attend", 0.4)
        # Below the threshold anyway: the gate's own reason comes first.
        self.assertIs(third["acknowledge"], False)
        self.assertIn("reason=low_confidence", self.notes[2])

    def test_a_request_after_the_window_is_acknowledged(self):
        self.start()
        first = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        # The window runs 8 s from the acknowledged turn's end: up to 8.9 s.
        inside = self.say(turn("t2", 8899, 9500, "And the date?"), "attend", 0.9)
        after = self.say(turn("t3", 9500, 10000, "And the weather?"), "attend", 0.9)
        self.assertEqual(self.acks(first, inside, after), [True, False, True])

    def test_a_skipped_fragment_does_not_extend_the_window(self):
        self.start()
        first = self.say(turn("t1", 0, 900, "What time"), "attend", 0.9)
        skipped = self.say(turn("t2", 5000, 8000, "is it?"), "attend", 0.9)
        later = self.say(turn("t3", 9000, 9800, "And the weather?"), "attend", 0.9)
        self.assertEqual(self.acks(first, skipped, later), [True, False, True])

    def test_the_configured_window_is_used(self):
        self.start(acknowledgement=Acknowledgement(dedup_window_ms=2000))
        first = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        inside = self.say(turn("t2", 2000, 2500, "Please."), "attend", 0.9)
        after = self.say(turn("t3", 3000, 3500, "And the weather?"), "attend", 0.9)
        self.assertEqual(self.acks(first, inside, after), [True, False, True])

    def test_a_zero_window_acknowledges_every_fragment_as_before(self):
        self.start(acknowledgement=NO_DEDUP)
        first = self.say(turn("t1", 0, 1200, "Can you tell me what"), "attend", 0.9)
        second = self.say(turn("t2", 1700, 2600, "time it is?"), "attend", 0.95)
        self.events.reply("started", 3000)
        self.events.reply("ended", 4000)
        third = self.say(turn("t3", 4100, 4600, "Please."), "attend", 0.9)
        self.assertEqual(self.acks(first, second, third), [True, True, True])
        self.assertNotIn("pending_ack", " ".join(self.notes))

    def test_the_answer_starting_clears_the_pending_ack(self):
        self.start()
        first = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        # The acknowledgement clip's own playback: it answers nothing.
        self.events.reply("started", 1200)
        self.events.reply("ended", 2000)
        during = self.say(turn("t2", 2100, 2600, "Please."), "attend", 0.9)
        # The answer starts playing; the next request is a new one.
        self.events.reply("started", 3000)
        self.events.reply("ended", 5000)
        after = self.say(turn("t3", 5500, 6200, "And the weather?"), "attend", 0.9)
        self.assertEqual(self.acks(first, during, after), [True, False, True])
        # And that acknowledgement is pending in turn.
        again = self.say(turn("t4", 6500, 7000, "Tomorrow, I mean."), "attend", 0.9)
        self.assertIs(again["acknowledge"], False)

    def test_the_ack_clip_starting_alone_clears_nothing(self):
        self.start()
        self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        self.events.reply("started", 1200)
        second = self.say(turn("t2", 1500, 2000, "Please."), "attend", 0.9)
        self.assertIs(second["acknowledge"], False)

    def test_a_collapsed_clip_report_still_lets_the_answer_clear(self):
        # A host may report only the last phase of a burst: the clip's `ended` alone.
        self.start()
        self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        self.events.reply("ended", 2000)
        self.events.reply("started", 2500)
        after = self.say(turn("t2", 3000, 3500, "And the weather?"), "attend", 0.9)
        self.assertIs(after["acknowledge"], True)

    def test_reports_before_any_ack_are_not_carried_over(self):
        self.start()
        self.events.reply("ended", 100)
        self.say(turn("t1", 500, 900, "What time is it?"), "attend", 0.9)
        # The earlier `ended` belongs to nothing pending; this start is the clip's.
        self.events.reply("started", 1200)
        second = self.say(turn("t2", 1500, 2000, "Please."), "attend", 0.9)
        self.assertIs(second["acknowledge"], False)

    def test_reply_reports_work_without_conversation_mode(self):
        self.start()
        self.assertIsNone(self.events._conversation)
        self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        self.events.reply("started", 1000)
        self.events.reply("ended", 1800)
        self.events.reply("started", 2500)
        after = self.say(turn("t2", 3000, 3500, "And the weather?"), "attend", 0.9)
        self.assertIs(after["acknowledge"], True)

    def test_reply_reports_without_gating_change_nothing(self):
        self.start(acknowledgement=None)
        first = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        self.events.reply("started", 1000)
        self.events.reply("ended", 1800)
        second = self.say(turn("t2", 2000, 2500, "Please."), "attend", 0.9)
        self.assertNotIn("acknowledge", first)
        self.assertNotIn("acknowledge", second)
        self.assertEqual(self.notes, [])

    def test_a_new_session_starts_with_nothing_pending(self):
        self.start()
        self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        self.events.end("stopped", 1000)
        self.events.drain()
        self.events.start("acknowledgement-next", now_ms=0, acknowledgement=Acknowledgement())
        self.events.drain()
        current = Turn(
            "acknowledgement-next",
            "t1",
            1,
            1000,
            1500,
            "Please.",
            "Speaker A",
            True,
            False,
            "authored-fixture",
            "synthetic",
            "authored-fixture",
        )
        self.events.transcript(current, 1500)
        self.events.decision(decided(current, "attend", 0.9), 1600)
        request = [e for e in self.events.drain() if e["type"] == "request"][0]
        self.assertIs(request["acknowledge"], True)

    def test_reply_reports_arrive_from_the_control_reader_thread(self):
        # The `listen --control-fd` path: a separate reader thread forwards each report
        # through the controller, which stamps it, into the same locked session state.
        from rightyo.tool import _read_control

        events = self.events

        class Controller:
            def reply(self, phase):
                events.reply(phase, 2500 if phase == "started" else 2000)

        self.start()
        self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.9)
        read, write = os.pipe()
        os.write(write, b'{"reply": "ended"}\n{"reply": "started"}\n')
        os.close(write)
        reader = threading.Thread(target=_read_control, args=(read, Controller()))
        reader.start()
        reader.join(timeout=5)
        self.assertFalse(reader.is_alive())
        after = self.say(turn("t2", 3000, 3500, "And the weather?"), "attend", 0.9)
        self.assertIs(after["acknowledge"], True)


class AcknowledgementConfigTests(unittest.TestCase):
    def test_from_dict_validates(self):
        self.assertEqual(Acknowledgement.from_dict({}), Acknowledgement(0.7))
        self.assertEqual(
            Acknowledgement.from_dict({"min_confidence": 0.5}).to_dict(),
            {"version": 1, "min_confidence": 0.5, "dedup_window_ms": 8000},
        )
        self.assertEqual(Acknowledgement.from_dict({}).dedup_window_ms, 8000)
        for window in (0, 1, 60000):
            with self.subTest(window=window):
                self.assertEqual(
                    Acknowledgement.from_dict({"dedup_window_ms": window}).dedup_window_ms, window
                )
        for raw in (
            [],
            {"min_confidence": 1.5},
            {"min_confidence": -0.1},
            {"min_confidence": True},
            {"min_confidence": "0.7"},
            {"min_confidence": float("nan")},
            {"threshold": 0.7},
            {"dedup_window_ms": -1},
            {"dedup_window_ms": 60001},
            {"dedup_window_ms": 8000.0},
            {"dedup_window_ms": True},
            {"dedup_window_ms": "8000"},
            {"dedup_window_ms": None},
        ):
            with self.subTest(raw=raw), self.assertRaises(ContractError):
                Acknowledgement.from_dict(raw)

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
            self.assertIsNone(PrototypeConfig.load(config).acknowledgement)
            config.write_text(json.dumps(base | {"acknowledgement": {}}))
            self.assertEqual(PrototypeConfig.load(config).acknowledgement, Acknowledgement())
            config.write_text(json.dumps(base | {"acknowledgement": {"min_confidence": 0.85}}))
            self.assertEqual(PrototypeConfig.load(config).acknowledgement, Acknowledgement(0.85))
            config.write_text(json.dumps(base | {"acknowledgement": {"min_confidence": 2}}))
            with self.assertRaises(PrototypeError):
                PrototypeConfig.load(config)
            config.write_text(json.dumps(base | {"acknowledgement": {"dedup_window_ms": 0}}))
            self.assertEqual(
                PrototypeConfig.load(config).acknowledgement, Acknowledgement(dedup_window_ms=0)
            )
            config.write_text(json.dumps(base | {"acknowledgement": {"dedup_window_ms": -5}}))
            with self.assertRaises(PrototypeError):
                PrototypeConfig.load(config)


if __name__ == "__main__":
    unittest.main()
