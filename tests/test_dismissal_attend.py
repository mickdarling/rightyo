"""Dismissal attend bar (#282): authored turns, scripted decisions, no speech."""

import unittest
from dataclasses import replace

from test_dismissal import NAMES, WINDOW, decision, turn

from rightyo.contracts import ContractError, Conversation, Dismissal
from rightyo.tool_events import SpeechEvents


class DismissalAttendTests(unittest.TestCase):
    def replay(self, attend, *, bar=0.6, kind="stop", text="Quiet, buddy."):
        notes = []
        events = SpeechEvents(report=notes.append)
        events.start(
            "dismissal-test",
            dismissal=replace(WINDOW, min_attend=bar),
            conversation=Conversation(),
            addressing=NAMES,
        )
        started = events.drain()[0]
        self.assertNotIn("min_attend", started["dismissal"])
        self.assertEqual(events._dismissal.min_attend, bar)
        request = turn("request", 0, 900, "Haili, read the authored list")
        events.transcript(request, 900)
        events.decision(decision(request), 1000)
        events.reply("started", 1500)
        # Also retain a request awaiting its decision, to catch late withdrawals.
        pending = turn("pending", 1600, 1900, "Haili, add the blue folder")
        events.transcript(pending, 1900)
        events.drain()
        current = turn("quiet", 2000, 2400, text)
        events.transcript(current, 2400)
        event = decision(current, label="uncertain", dismissal=(kind, 0.88))
        event = replace(
            event,
            decision=replace(
                event.decision,
                probabilities={
                    "attend": attend,
                    "ignore": 0.0,
                    "uncertain": 1.0 - attend,
                },
            ),
        )
        events.decision(event, 2500)
        events.decision(decision(pending), 2600)
        return events.drain(), notes, events

    def test_low_attend_neither_stops_nor_withdraws(self):
        for attend in (0.19, 0.44):
            for kind in ("stop", "disengage"):
                with self.subTest(attend=attend, kind=kind):
                    output, notes, events = self.replay(attend, kind=kind)
                    self.assertFalse(any(e["type"] == "dismiss" for e in output))
                    self.assertFalse(
                        any(
                            e["type"] == "conversation" and e["reason"] == "dismissed"
                            for e in output
                        )
                    )
                    evidence = next(
                        e["decision"]
                        for e in output
                        if e["type"] == "attention" and e["utterance_id"] == "quiet"
                    )
                    self.assertEqual(evidence["dismissal"], "uncertain")
                    self.assertEqual(evidence["dismissal_confidence"], 0.88)
                    self.assertIn("dismissal-test:request", events._delivered)
                    self.assertIn("dismissal-test:pending", events._delivered)
                    self.assertEqual(events._cooldowns, {})
                    self.assertIn("dismissal outcome=uncertain reason=below_attend_bar", notes)
                    self.assertNotIn("buddy", " ".join(notes))
                    self.assertNotIn("Speaker", " ".join(notes))

    def test_attend_at_or_above_the_bar_still_stops(self):
        for attend in (0.6, 0.9):
            for kind in ("stop", "disengage"):
                with self.subTest(attend=attend, kind=kind):
                    output, _, events = self.replay(attend, kind=kind)
                    dismissals = [e for e in output if e["type"] == "dismiss"]
                    self.assertTrue(dismissals)
                    self.assertIn("playback", dismissals[0]["scope"])
                    withdrawn = [r for e in dismissals for r in e["withdrawn_request_ids"]]
                    self.assertIn("dismissal-test:request", withdrawn)
                    self.assertIn("dismissal-test:pending", withdrawn)
                    self.assertNotIn("dismissal-test:request", events._delivered)

    def test_stop_phrase_stops_regardless_of_attend(self):
        output, _, _ = self.replay(0.19, text="Stop.")
        dismissals = [e for e in output if e["type"] == "dismiss"]
        self.assertEqual(dismissals[0]["reason"], "stop-phrase")
        self.assertIn("playback", dismissals[0]["scope"])
        self.assertIn("dismissal-test:request", dismissals[0]["withdrawn_request_ids"])

    def test_none_bar_preserves_dismissal(self):
        output, notes, _ = self.replay(0.19, bar=None)
        self.assertTrue(any(e["type"] == "dismiss" for e in output))
        self.assertNotIn("dismissal outcome=uncertain reason=below_attend_bar", notes)

    def test_validation_and_serialization(self):
        for bad in (-0.1, 1.1, True, "0.6", float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                Dismissal.from_dict({"min_attend": bad})
        for value in (None, 0.0, 0.6, 1.0):
            mode = Dismissal.from_dict({"min_attend": value})
            self.assertEqual(mode.min_attend, value)
            raw = mode.to_dict()
            raw.pop("version")
            self.assertEqual(Dismissal.from_dict(raw), mode)
            self.assertEqual("min_attend" in raw, value is not None)
