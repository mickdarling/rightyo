"""#295: authored turns, synthetic tones and scripted vectors only; no model or capture."""

import math
import queue
import threading
import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import test_duck_mode as duck
from test_enrolled_roles import OWNER_ID, SESSION, HaildModel, decided, turn
from test_live_speaker_id import ENTRIES, Harness, ScriptedEmbedder, settings, tone, wait_for

from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.live_speaker_id import PUBLIC_SCORE_WAIT_S, EnrolledRoles
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


class BindingPublicationTests(unittest.TestCase):
    def arrange(self, *, threshold=None, scores=(0.8,), gate=None, fail_on=None):
        controller = duck.DuckControllerTests.controller(self)
        h = Harness(
            ScriptedEmbedder([vector(s) for s in scores], gate=gate, fail_on=fail_on),
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
        if gate is not None:
            self.addCleanup(gate.set)
        return controller, events, h

    def publish(self, c, h, current, *, inferred=False, spans=True):
        h.shadow.audio(tone((current.end_ms - current.start_ms) / 1000))
        if inferred:
            c._inferred_turns.add(current.utterance_id)
        if spans:
            c._turn_spans[current.utterance_id] = ((current.start_ms, current.end_ms),)
        c._diarized_heads[current.utterance_id] = len(current.text)
        c._turn(c._generation, queue.Queue(), TranscriptMemory(), h.shadow, current)

    def transcript(self, events):
        return next(e for e in events.drain() if e["type"] == "transcript")

    def test_single_binding_turn_publishes_owner(self):
        c, e, h = self.arrange(threshold=0.7)
        current = turn("single", 0, 1500, "Jarvis, check the authored list")
        self.publish(c, h, current)
        transcript = self.transcript(e)
        self.assertEqual(transcript["turn"]["role"], "owner")
        self.assertEqual(transcript["role_source"], "voiceprint")

    def test_accumulated_binding_turn_publishes_owner(self):
        c, e, h = self.arrange(scores=(0.65, 0.65))
        for index, expected in enumerate(("unknown", "owner")):
            current = turn(
                f"acc-{index}", index * 2000, (index + 1) * 2000, "Jarvis, check the authored list"
            )
            self.publish(c, h, current)
            self.assertEqual(self.transcript(e)["turn"]["role"], expected)
        self.assertEqual(h.fields(h.notes()[-1])["bind"], "accumulated")

    def test_a_later_binding_never_rewrites_an_earlier_published_role(self):
        c, e, h = self.arrange(scores=(0.65, 0.65))
        first = turn("first", 0, 2000, "Jarvis, check the authored list")
        self.publish(c, h, first)
        original = self.transcript(e)
        self.publish(c, h, turn("later", 2000, 4000, "Jarvis, check another authored list"))
        self.assertEqual(self.transcript(e)["turn"]["role"], "owner")
        e.decision(decided(first), 4100)
        attention = next(item for item in e.drain() if item["type"] == "attention")
        self.assertEqual(original["turn"]["role"], "unknown")
        self.assertEqual(attention["decision"]["role"], "unknown")

    def test_timeout_cannot_use_a_different_turns_bind_before_publication(self):
        c, e, h = self.arrange(threshold=0.7)
        current = turn("waiting", 0, 1500, "Jarvis, check the authored list")
        accept = c._accept

        def bind_later(*args):
            # A DIFFERENT queued turn binds after the wait timed out, before role resolution.
            h.shadow._score(h.embedder, {"owner": [1, 0, 0]}, 0, "Speaker A", 1500, 1500, tone(1.5))
            return accept(*args)

        with (
            patch.object(h.shadow, "turn", return_value=None),
            patch.object(c, "_accept", side_effect=bind_later),
        ):
            self.publish(c, h, current)
        self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "owner"))
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")

    def test_inferred_binding_turn_is_not_upgraded_even_if_score_finishes_first(self):
        c, e, h = self.arrange(threshold=0.7)
        current = turn("tail-joined", 0, 1500, "Jarvis, check the authored list")
        score = h.shadow.turn

        def finish(*args, **kwargs):
            result = score(*args, **{**kwargs, "inferred": False})
            self.assertIsNotNone(result)
            return result

        with patch.object(h.shadow, "turn", side_effect=finish):
            self.publish(c, h, current, inferred=True)
        self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "owner"))
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")

    def test_scoring_timeout_and_error_publish_unknown(self):
        for failure in ("timeout", "error"):
            with self.subTest(failure=failure):
                gate = threading.Event() if failure == "timeout" else None
                c, e, h = self.arrange(
                    threshold=0.7, gate=gate, fail_on=1 if failure == "error" else None
                )
                before = time.monotonic()
                self.publish(c, h, turn(failure, 0, 1500, "Jarvis, check the authored list"))
                elapsed = time.monotonic() - before
                self.assertLess(elapsed, PUBLIC_SCORE_WAIT_S + 0.7)
                self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
                if gate is not None:
                    gate.set()
                    wait_for(lambda: h.shadow.scored == 1)
                    self.assertEqual(e.drain(), [])

    def test_epoch_or_revision_change_after_scoring_refuses_upgrade(self):
        for change in ("forget", "revision"):
            with self.subTest(change=change):
                c, e, h = self.arrange(threshold=0.7)
                accept = c._accept

                def invalidate(*args):
                    if change == "forget":
                        h.shadow.forget()
                    else:
                        h.shadow._binding_revisions["Speaker A"] += 2
                    return accept(*args)

                with patch.object(c, "_accept", side_effect=invalidate):
                    self.publish(c, h, turn(change, 0, 1500, "Jarvis, check the authored list"))
                self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")

    def test_unscorable_turns_never_wait_and_guest_never_becomes_owner(self):
        for case in ("short", "no-spans", "overlap", "provenance", "guest", "inferred"):
            with self.subTest(case=case):
                c, e, h = self.arrange(threshold=0.7, scores=(0.4,))
                current = turn(
                    case,
                    0,
                    690 if case == "short" else 1500,
                    "Jarvis, check the authored list",
                    overlap=case == "overlap",
                )
                if case == "provenance":
                    current = replace(current, speaker_provenance="unknown")
                with patch(
                    "rightyo.live_speaker_id._ScoringDone.wait", return_value=False
                ) as waiting:
                    self.publish(
                        c, h, current, spans=case != "no-spans", inferred=case == "inferred"
                    )
                    if case != "guest":
                        waiting.assert_not_called()
                    else:
                        waiting.assert_called_once_with(PUBLIC_SCORE_WAIT_S)

                self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")

    def test_an_enrolled_guest_binds_as_participant(self):
        c, e, h = self.arrange(threshold=0.7)
        h.shadow.close()
        h = Harness(
            ScriptedEmbedder([[0, 1, 0]]),
            config=settings(roles=True, single_turn_bind_threshold=0.7),
        )
        c._enrolled_roles.source = h.shadow
        c._shadow = h.shadow
        h.shadow.start()
        wait_for(lambda: bool(h.shadow._compared))
        self.addCleanup(h.shadow.close)
        self.publish(c, h, turn("guest", 0, 1500, "Jarvis, check the authored list", "Speaker B"))
        self.assertEqual(h.shadow.binding("Speaker B"), ("bound", "guest"))
        self.assertEqual(self.transcript(e)["turn"]["role"], "participant")

    def test_authored_event_regressions(self):
        for single in (False, True):
            with self.subTest(single=single):
                c, e, h = self.arrange(
                    threshold=0.7 if single else None, scores=(0.704,) if single else (0.704, 0.627)
                )
                output = e.drain()
                fragment = turn("fragment", 0, 690, "Authored fragment", "Speaker B")
                self.publish(c, h, fragment, spans=False)
                output.extend(e.drain())
                start = 690
                for index in range(1 if single else 2):
                    current = turn(
                        f"named-{index}", start, start + 1700, "Jarvis, check the authored list"
                    )
                    self.publish(c, h, current)
                    e.decision(decided(current), current.end_ms + 10)
                    output.extend(e.drain())
                    start += 1700
                requests = [item for item in output if item["type"] == "request"]
                self.assertEqual(requests[-1]["turn"]["utterance_id"], current.utterance_id)
                self.assertEqual(requests[-1]["turn"]["role"], "owner")
                self.assertNotIn("others_present", requests[-1]["decision"])
                HaildModel(self).check(output)
