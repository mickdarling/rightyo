"""Acknowledgement gating (#132): ack a request only when a reply is likely.

Authored text and test doubles only: no capture, native inference, hosted calls or
recorded speech.
"""

from __future__ import annotations

import json
import tempfile
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


class AcknowledgementEventTests(unittest.TestCase):
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

    def test_off_by_default_changes_nothing(self):
        started = self.start(acknowledgement=None)
        self.assertNotIn("acknowledgement", started)
        request = self.say(turn("t1", 0, 900, "What time is it?"), "attend", 0.5)
        self.assertNotIn("acknowledge", request)
        self.assertEqual(self.notes, [])

    def test_the_session_advertises_the_threshold(self):
        started = self.start()
        self.assertEqual(started["acknowledgement"], {"version": 1, "min_confidence": 0.7})

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
            self.notes[-1],
            "ack outcome=skip reason=low_confidence attend_confidence=0.45 min=0.70 follow_up=true",
        )

    def test_a_follow_up_is_judged_on_its_attend_probability(self):
        # A follow-up's `confidence` belongs to its `uncertain` choice; a strong attend
        # probability still earns the acknowledgement, a high `confidence` alone does not.
        self.start(conversation=Conversation(window_ms=10000))
        self.say(turn("t1", 0, 900, "Haili, what time is it?"), "attend", 0.92)
        strong = self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", 0.3, 0.8)
        self.assertIs(strong["acknowledge"], True)
        weak = self.say(turn("t3", 4000, 4900, "And Friday?"), "uncertain", 0.95, 0.45)
        self.assertIs(weak["acknowledge"], False)

    def test_a_named_follow_up_is_acknowledged(self):
        self.start(conversation=Conversation(window_ms=10000))
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
        self.assertEqual(len(self.notes), 2)
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


class AcknowledgementConfigTests(unittest.TestCase):
    def test_from_dict_validates(self):
        self.assertEqual(Acknowledgement.from_dict({}), Acknowledgement(0.7))
        self.assertEqual(
            Acknowledgement.from_dict({"min_confidence": 0.5}).to_dict(),
            {"version": 1, "min_confidence": 0.5},
        )
        for raw in (
            [],
            {"min_confidence": 1.5},
            {"min_confidence": -0.1},
            {"min_confidence": True},
            {"min_confidence": "0.7"},
            {"min_confidence": float("nan")},
            {"threshold": 0.7},
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


if __name__ == "__main__":
    unittest.main()
