"""#295: authored turns, synthetic tones and scripted vectors only; no model or capture."""

import math
import queue
import unittest

import test_duck_mode as duck
from test_enrolled_roles import OWNER_ID, SESSION, turn
from test_live_speaker_id import ENTRIES, Harness, ScriptedEmbedder, settings, tone, wait_for

from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.memory import TranscriptMemory
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool_events import SpeechEvents


def vector(score):
    return [score, 0, math.sqrt(1 - score * score)]


class PresenceSubstanceTests(unittest.TestCase):
    def test_both_presence_bounds_are_required(self):
        for duration, text, present in (
            (690, "Authored fragment", False),
            (1500, "These are five authored words", True),
            (1500, "Authored fragment", False),
            (690, "These are five authored words", False),
            (1000, "Three authored words", True),
        ):
            with self.subTest(duration=duration, text=text):
                events = SpeechEvents()
                events.start(
                    SESSION, now_ms=0, conversation=Conversation(cue_when_others_ms=120000)
                )
                other = turn("other", 0, duration, text, "Speaker B")
                events.transcript(other, duration, expect_decision=False)
                owner = turn("owner", 2000, 3500, "Jarvis, check the authored list")
                events.transcript(owner, 3500)
                self.assertEqual(events._others_present(owner), present)

    def test_presence_config_validation_and_round_trip(self):
        self.assertEqual(
            (Conversation().presence_min_ms, Conversation().presence_min_words), (1000, 3)
        )
        for key, maximum in (("presence_min_ms", 120000), ("presence_min_words", 1000)):
            for bad in (-1, True, 1.5, "3", None, maximum + 1):
                with self.subTest(key=key, bad=bad), self.assertRaises(ContractError):
                    Conversation.from_dict({key: bad})
            for value in (0, 7, maximum):
                mode = Conversation.from_dict({key: value})
                raw = mode.to_dict()
                raw.pop("version")
                self.assertEqual(Conversation.from_dict(raw), mode)


class SingleTurnBindTests(unittest.TestCase):
    def score(self, score, *, seconds=1.5, threshold=None, epoch=0):
        harness = Harness(
            ScriptedEmbedder([vector(score)]),
            config=settings(single_turn_bind_threshold=threshold),
            entries=ENTRIES[:1],
        )
        harness.shadow._epoch = epoch
        harness.shadow.start()
        harness.speak(seconds)
        harness.shadow.close(drain=True)
        return harness

    def test_single_turn_default_off_and_below_bar_accumulate(self):
        for threshold, score in ((None, 0.9), (0.7, 0.69)):
            with self.subTest(threshold=threshold):
                h = self.score(score, threshold=threshold)
                self.assertEqual(h.shadow.binding("Speaker A"), ("tentative", "owner"))
                self.assertEqual(h.fields(h.notes()[0])["bind"], "-")

    def test_single_turn_binds_and_bumps_revision(self):
        for score in (0.7, 0.704, 0.95):
            with self.subTest(score=score):
                h = self.score(score, threshold=0.7)
                self.assertEqual(
                    h.shadow.binding_snapshot("Speaker A"), ((0, 1), ("bound", "owner"))
                )
                fields = h.fields(h.notes()[0])
                self.assertEqual(fields["bind"], "single")
                self.assertTrue(h.notes()[0].endswith(" bind=single"))
                self.assertNotIn("private words", h.notes()[0])
                self.assertEqual(
                    set(fields),
                    {
                        "label",
                        "turn_ms",
                        "labelled_ms",
                        "speech_ms",
                        "turn_score",
                        "acc_score",
                        "acc_s",
                        "state",
                        "id",
                        "bind",
                    },
                )

    def test_single_turn_needs_minimum_scored_speech(self):
        h = self.score(0.99, seconds=0.7, threshold=0.7)
        self.assertIsNone(h.shadow.binding("Speaker A"))
        self.assertEqual(h.shadow.short, 1)
        self.assertEqual(h.notes(), [])

    def test_guest_typical_score_never_binds(self):
        for score in (0.4, 0.60, 0.691):
            with self.subTest(score=score):
                h = self.score(score, threshold=0.7)
                self.assertNotEqual(h.shadow.binding("Speaker A")[0], "bound")

    def test_renewal_rule_uses_its_original_threshold_even_with_single_on(self):
        for threshold in (None, 0.9):
            with self.subTest(threshold=threshold):
                h = self.score(0.65, threshold=threshold, epoch=1)
                self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "owner"))
                self.assertEqual(h.fields(h.notes()[0])["bind"], "rebind")

    def test_single_config_validation(self):
        raw = {"python": "/synthetic/python", "model": "/synthetic/model", "bind_threshold": 0.6}
        for value in (0.59, 1.01, True, "0.7", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpeakerIdConfig.from_dict({**raw, "single_turn_bind_threshold": value})
        for value in (None, 0.6, 0.7, 1):
            self.assertEqual(
                SpeakerIdConfig.from_dict(
                    {**raw, "single_turn_bind_threshold": value}
                ).single_turn_bind_threshold,
                value,
            )
        self.assertIsNone(SpeakerIdConfig.from_dict(raw).single_turn_bind_threshold)


class BaseBindingPublicationTests(unittest.TestCase):
    def arrange(self, *, threshold=None, scores=(0.8,)):
        controller = duck.DuckControllerTests.controller(self)
        h = Harness(
            ScriptedEmbedder([vector(s) for s in scores]),
            config=settings(roles=True, single_turn_bind_threshold=threshold),
            entries=ENTRIES[:1],
        )
        roles = EnrolledRoles(OWNER_ID)
        roles.source = h.shadow
        events = SpeechEvents()
        events.start(
            SESSION,
            now_ms=0,
            priority=roles,
            addressing=Addressing(names=("Jarvis",)),
            conversation=Conversation(cue_when_others_ms=120000),
            duck=True,
        )
        controller._decision_status = "ready"
        controller._events = events
        controller._enrolled_roles = roles
        controller._shadow = h.shadow
        h.shadow.start()
        wait_for(lambda: bool(h.shadow._compared))
        self.addCleanup(h.shadow.close)
        return controller, events, h

    def publish(self, c, h, current):
        h.shadow.audio(tone((current.end_ms - current.start_ms) / 1000))
        c._turn_spans[current.utterance_id] = ((current.start_ms, current.end_ms),)
        c._diarized_heads[current.utterance_id] = len(current.text)
        c._turn(c._generation, queue.Queue(), TranscriptMemory(), h.shadow, current)

    def transcript(self, events):
        return next(e for e in events.drain() if e["type"] == "transcript")

    def test_renewal_binding_turn_publishes_owner_as_on_base(self):
        c, e, h = self.arrange(scores=(0.65,))
        h.shadow.forget()
        self.publish(c, h, turn("renewal", 0, 1500, "Jarvis, check the authored list"))
        transcript = self.transcript(e)
        self.assertEqual(transcript["turn"]["role"], "owner")
        self.assertEqual(transcript["role_source"], "voiceprint")

    def test_epoch_zero_binding_turn_keeps_its_pre_scoring_role(self):
        for single in (False, True):
            with self.subTest(single=single):
                c, e, h = self.arrange(
                    threshold=0.7 if single else None, scores=(0.8,) if single else (0.65, 0.65)
                )
                for index in range(1 if single else 2):
                    current = turn(
                        f"initial-{index}",
                        index * 2000,
                        (index + 1) * 2000,
                        "Jarvis, check the authored list",
                    )
                    self.publish(c, h, current)
                    self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
                    wait_for(lambda: h.shadow.scored == index + 1)
                self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "owner"))
                self.assertEqual(
                    h.fields(h.notes()[-1])["bind"], "single" if single else "accumulated"
                )
