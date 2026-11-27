"""#295: authored turns, synthetic tones and scripted vectors only; no model or capture."""

import math
import queue
import threading
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import test_duck_mode as duck
from test_answered_by_other import gap
from test_dismissal import decision as dismissal_decision
from test_enrolled_roles import OWNER_ID, SESSION, decided, turn
from test_live_speaker_id import ENTRIES, Harness, ScriptedEmbedder, settings, tone, wait_for

from rightyo.contracts import Addressing, ContractError, Conversation, Dismissal
from rightyo.enroll import cosine
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.memory import TranscriptMemory
from rightyo.pipeline import ReplayRunner
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool_events import PRIVACY_EXIT, SpeechEvents


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

    def sequence(self, vectors, durations, *, threshold=0.7, entries=ENTRIES[:1], **config):
        h = Harness(
            ScriptedEmbedder(vectors),
            config=settings(single_turn_bind_threshold=threshold, **config),
            entries=entries,
        )
        h.shadow.start()
        try:
            for count, seconds in enumerate(durations, 1):
                h.speak(seconds)
                wait_for(lambda: h.shadow.scored == count)
        finally:
            h.shadow.close(drain=True)
        return h, [h.fields(note) for note in h.notes()]

    def test_alternating_scores_keep_accumulated_evidence(self):
        scores = [0.30, 0.75] * 4
        baseline, before = self.sequence([vector(s) for s in scores], [2.0] * 8, threshold=None)
        h, after = self.sequence([vector(s) for s in scores], [2.0] * 8)
        self.assertEqual([n["state"] for n in after], [n["state"] for n in before])
        self.assertEqual(after[-1]["state"], "tentative")
        self.assertEqual(h.shadow.binding("Speaker A"), baseline.shadow.binding("Speaker A"))
        self.assertEqual(h.shadow._bound_ids, set())

    def test_long_low_score_keeps_accumulated_evidence(self):
        # Nine seconds uses five overlapping embedding windows.
        vectors, durations = [vector(0.10)] * 5 + [vector(0.75)], [9.0, 1.5]
        baseline, before = self.sequence(vectors, durations, threshold=None)
        h, after = self.sequence(vectors, durations)
        self.assertEqual([n["state"] for n in after], [n["state"] for n in before])
        self.assertEqual(h.shadow.binding("Speaker A"), ("unknown", "owner"))
        roles = EnrolledRoles(OWNER_ID)
        for source in (baseline.shadow, h.shadow):
            roles.source = source
            self.assertEqual(
                roles.role_for(turn("low", 11000, 12500, "Authored turn")), "participant"
            )
        self.assertEqual(h.shadow._bound_ids, set())

    def test_single_bind_diagnostic_keeps_accumulated_mean(self):
        h, notes = self.sequence([vector(0.50), vector(0.75)], [1.0, 1.6])
        state = h.shadow._labels["Speaker A"]
        self.assertEqual(notes[-1]["bind"], "single")
        self.assertAlmostEqual(state.score, state.scores["owner"])
        self.assertEqual(notes[-1]["acc_score"], f"{state.scores['owner']:.3f}")
        self.assertNotEqual(notes[-1]["acc_score"], notes[-1]["turn_score"])

    def test_single_bind_needs_its_own_minimum_speech(self):
        for seconds, expected in ((1.2, "tentative"), (1.6, "bound")):
            with self.subTest(seconds=seconds):
                h = self.score(0.9, seconds=seconds, threshold=0.7)
                self.assertEqual(h.shadow.binding("Speaker A"), (expected, "owner"))
                self.assertEqual(
                    h.fields(h.notes()[0])["bind"], "single" if expected == "bound" else "-"
                )

    def test_single_minimum_config_validation(self):
        raw = {"python": "/synthetic/python", "model": "/synthetic/model", "min_turn_seconds": 0.5}
        for minimum, expected in ((0.5, 1.5), (1.0, 1.5), (2.0, 2.0), (10.0, 10.0)):
            with self.subTest(min_turn_seconds=minimum):
                parsed = SpeakerIdConfig.from_dict({**raw, "min_turn_seconds": minimum})
                self.assertEqual(parsed.single_turn_bind_min_seconds, expected)
                self.assertIsNone(parsed.single_turn_bind_threshold)
                self.assertEqual(parsed.to_dict()["single_turn_bind_min_seconds"], expected)
                self.assertEqual(SpeakerIdConfig.from_dict(parsed.to_dict()), parsed)
        for value in (None, True, "1.5", 0.49, 10.01, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpeakerIdConfig.from_dict({**raw, "single_turn_bind_min_seconds": value})
        for value in (0.5, 1.5, 10):
            parsed = SpeakerIdConfig.from_dict({**raw, "single_turn_bind_min_seconds": value})
            self.assertEqual(parsed.single_turn_bind_min_seconds, value)
            self.assertIsInstance(parsed.single_turn_bind_min_seconds, float)
            self.assertEqual(SpeakerIdConfig.from_dict(parsed.to_dict()), parsed)
        with self.assertRaises(ValueError):
            SpeakerIdConfig.from_dict(
                {**raw, "min_turn_seconds": 2.0, "single_turn_bind_min_seconds": 1.9}
            )
        parsed = SpeakerIdConfig.from_dict(
            {**raw, "min_turn_seconds": 2.0, "single_turn_bind_min_seconds": 2.0}
        )
        self.assertEqual(parsed.single_turn_bind_min_seconds, 2.0)
        self.assertEqual(SpeakerIdConfig.from_dict(parsed.to_dict()), parsed)

    def test_single_bind_needs_the_accumulated_best_identity(self):
        entries = [ENTRIES[0], {**ENTRIES[1], "embedding": [0.6, 0.8, 0]}]
        first = [0.55, 0.4, math.sqrt(1 - 0.55**2 - 0.4**2)]
        second = [0.75, 0.3125, math.sqrt(1 - 0.75**2 - 0.3125**2)]
        h, notes = self.sequence([first, second], [1.2, 1.6], entries=entries)
        state = h.shadow._labels["Speaker A"]
        self.assertGreater(state.scores["guest"], state.scores["owner"])
        self.assertGreaterEqual(state.scores["owner"], 0.6)
        self.assertEqual(h.shadow.binding("Speaker A"), ("tentative", "guest"))
        self.assertEqual(notes[-1]["bind"], "-")

    def test_single_bind_does_not_replace_a_prior_binding(self):
        vectors, durations = [[1, 0, 0], [0, 1, 0]], [1.0, 1.99]
        # The shorter single minimum isolates prior-binding hardening.
        h, notes = self.sequence(
            vectors, durations, entries=ENTRIES[:2], single_turn_bind_min_seconds=1.0
        )
        self.assertEqual(notes[0]["state"], "bound")
        self.assertEqual(notes[0]["id"], "owner")
        baseline, before = self.sequence(vectors, durations, entries=ENTRIES[:2], threshold=None)
        self.assertEqual(h.shadow.binding("Speaker A"), baseline.shadow.binding("Speaker A"))
        self.assertEqual(notes[-1]["state"], before[-1]["state"])
        self.assertEqual(notes[-1]["state"], "tentative")
        self.assertEqual(notes[-1]["bind"], "-")

    def test_accumulated_identity_change_reports_its_rule(self):
        h, notes = self.sequence(
            [[0, 1, 0], [1, 0, 0]],
            [1.0, 2.0],
            entries=ENTRIES[:2],
            threshold=None,
            bind_min_seconds=1.0,
        )
        self.assertEqual(notes[0]["id"], "guest")
        self.assertEqual(notes[0]["state"], "bound")
        self.assertEqual(h.shadow.binding("Speaker A"), ("bound", "owner"))
        self.assertEqual(notes[-1]["bind"], "accumulated")


class BaseBindingPublicationTests(unittest.TestCase):
    def arrange(
        self,
        *,
        threshold=None,
        scores=(0.8,),
        gate=None,
        priority=OWNER_ID,
        entries=ENTRIES[:1],
        dismissal=None,
    ):
        controller = duck.DuckControllerTests.controller(self)
        h = Harness(
            ScriptedEmbedder([vector(s) for s in scores], gate=gate),
            config=settings(roles=True, single_turn_bind_threshold=threshold),
            entries=entries,
        )
        roles = EnrolledRoles(priority)
        roles.source = h.shadow
        events = SpeechEvents(report=h.lines.append)
        events.start(
            SESSION,
            now_ms=0,
            priority=roles,
            addressing=Addressing(names=("Jarvis",)),
            conversation=Conversation(cue_when_others_ms=120000),
            dismissal=dismissal,
            meeting_ask_anywhere=False,
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

    @staticmethod
    def sample(index, start=0, end=1500):
        return turn(str(index), start, end, " ".join((duck.OURS.names[0], str(index), str(end))))

    def decision(self, events, current, *, role, requested, label="attend"):
        events.decision(decided(current, label), current.end_ms)
        output = events.drain()
        attention = next(event for event in output if event["type"] == "attention")
        self.assertEqual(attention["decision"]["role"], role)
        requests = [event for event in output if event["type"] == "request"]
        self.assertEqual(len(requests), int(requested))
        if requested:
            self.assertEqual(requests[0]["decision"], attention["decision"])
            self.assertEqual(requests[0]["turn"]["role"], "unknown")
        return output

    def test_first_bind_attends_without_changing_presence_or_owner_authority(self):
        for single in (False, True):
            with self.subTest(single=single):
                c, e, h = self.arrange(threshold=0.7 if single else None)
                other = replace(self.sample(0, end=1000), speaker_id="other")
                e.transcript(other, 1000)
                e.decision(decided(other), 1000)
                witness = replace(self.sample(3, 1000, 2000), speaker_id="third")
                e.transcript(witness, 2000, expect_decision=False)
                e.drain()
                h.shadow.audio(bytes(2000 * 32))
                if not single:
                    first = self.sample(1, 2000, 3500)
                    self.publish(c, h, first)
                    wait_for(lambda: h.shadow.scored == 1)
                    self.assertEqual(h.shadow.binding(first.speaker_id)[0], "tentative")
                    e.drain()
                start = 2000 if single else 3500
                current = self.sample(2, start, start + 1500)
                self.publish(c, h, current)
                transcript = self.transcript(e)
                self.assertEqual(transcript["turn"]["role"], "unknown")
                self.assertNotIn("role_source", transcript)
                wait_for(lambda: h.shadow.binding(current.speaker_id) == ("bound", "owner"))
                history = list(e._cue_voices)
                self.assertEqual(history[-1][:2], (current.end_ms, current.speaker_id))
                self.assertTrue(e._others_present(current))
                output = self.decision(
                    e,
                    current,
                    role="owner",
                    requested=True,
                    label="uncertain" if single else "attend",
                )
                evidence = next(
                    event["decision"] for event in output if event["type"] == "attention"
                )
                self.assertEqual(evidence["speaker_binding_source"], "first_bind_receipt")
                self.assertEqual(evidence["speaker_binding_state"], "bound")
                self.assertIs(evidence["speaker_bound_to_owner"], True)
                self.assertFalse(any(event["type"] == "override" for event in output))
                self.assertIn(SESSION + ":" + other.utterance_id, e._open)
                self.assertIn(SESSION + ":" + current.utterance_id, e._open)
                self.assertEqual(e._engaged["role"], "unknown")
                self.assertEqual(list(e._cue_voices), history)
                self.assertTrue(e._others_present(current))
                note = "attention_promoted reason=named_owner" if single else "attention_decision"
                self.assertIn(note + " role_at=receipt", h.lines)

    def check_binding_decision(self, scores, entries, threshold, *, rule, owner_score, receipt):
        c, e, h = self.arrange(
            scores=scores,
            entries=entries,
            threshold=threshold,
            priority=replace(OWNER_ID, trusted=("guest",), owner_only=True),
        )
        duration = 1500 if rule == "single" else 1000
        for index in range(len(scores)):
            current = self.sample(index, index * duration, (index + 1) * duration)
            self.publish(c, h, current)
            self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
            wait_for(lambda: len(h.notes()) == index + 1)
        note = h.fields(h.notes()[-1])
        self.assertEqual(note["bind"], rule)
        self.assertEqual(note["turn_score"], f"{owner_score:.3f}")
        self.assertEqual(h.shadow.binding(current.speaker_id), ("bound", "owner"))
        output = self.decision(
            e, current, role="owner" if receipt else "unknown", requested=receipt
        )
        self.assertEqual(output[0]["decision"]["label"], "attend" if receipt else "ignore")
        self.assertIn("attention_decision role_at=" + ("receipt" if receipt else "accept"), h.lines)

    def test_accumulated_owner_bind_on_guest_turn_has_no_receipt(self):
        entries = [ENTRIES[0], {**ENTRIES[1], "embedding": vector(0)}]
        for threshold in (None, 0.70):
            for score in (0.0, settings().bind_threshold + 0.01):
                with self.subTest(threshold=threshold, owner_score=score):
                    self.check_binding_decision(
                        (1, 1, score),
                        entries,
                        threshold,
                        rule="accumulated",
                        owner_score=score,
                        receipt=False,
                    )

    def test_binding_turn_just_below_threshold_has_no_receipt(self):
        score = settings().bind_threshold - 0.001
        self.check_binding_decision(
            (1, 1, score), ENTRIES[:1], None, rule="accumulated", owner_score=score, receipt=False
        )

    def test_single_enrollment_at_threshold_receives_receipt(self):
        score = settings().bind_threshold
        self.check_binding_decision(
            (1, 1, score), ENTRIES[:1], None, rule="accumulated", owner_score=score, receipt=True
        )

    def test_tied_turn_scores_withhold_receipt_for_both_binding_rules(self):
        owner = {**ENTRIES[0], "embedding": [1, 0, 1]}
        for rule, scores, threshold, guest_id in (
            ("accumulated", (0.8, 0.8, 1), None, "guest"),
            ("single", (1,), 0.70, "participant"),
        ):
            with self.subTest(rule=rule):
                guest = {**ENTRIES[1], "id": guest_id, "embedding": [1, 0, -1]}
                self.check_binding_decision(
                    scores,
                    [guest, owner],
                    threshold,
                    rule=rule,
                    owner_score=1 / math.sqrt(2),
                    receipt=False,
                )

    def test_receipt_requires_margin_over_each_competitor(self):
        owner = {**ENTRIES[0], "embedding": vector(0.85)}
        for competing_score, receipt in ((0.80, False), (0.75, True), (0.70, True)):
            with self.subTest(competing_score=competing_score):
                competitor = {**ENTRIES[1], "id": "third", "embedding": vector(competing_score)}
                self.check_binding_decision(
                    (1,),
                    [owner, ENTRIES[1], competitor],
                    0.70,
                    rule="single",
                    owner_score=0.85,
                    receipt=receipt,
                )

    def test_missing_competitor_score_withholds_receipt(self):
        guest = {**ENTRIES[1], "embedding": [1, 0]}
        self.check_binding_decision(
            (1,), [ENTRIES[0], guest], 0.70, rule="single", owner_score=1, receipt=False
        )

    def test_nonfinite_competitor_score_withholds_receipt(self):
        def score(vector, profile):
            return -math.inf if profile == ENTRIES[1]["embedding"] else cosine(vector, profile)

        with patch("rightyo.live_speaker_id.cosine", side_effect=score):
            self.check_binding_decision(
                (1,), ENTRIES[:2], 0.70, rule="single", owner_score=1, receipt=False
            )

    def test_stop_phrase_on_receipt_turn_uses_acceptance_role(self):
        c, e, h = self.arrange(threshold=0.70)
        current = replace(self.sample(1), text=OWNER_ID.stop_phrases[0])
        self.publish(c, h, current)
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
        wait_for(lambda: h.shadow.scored == 1)
        self.decision(e, current, role="owner", requested=True)
        self.assertIn("attention_decision role_at=receipt", h.lines)

    def test_dismissal_on_receipt_turn_uses_acceptance_role(self):
        c, e, h = self.arrange(threshold=0.70, dismissal=Dismissal())
        current = self.sample(1)
        self.publish(c, h, current)
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
        wait_for(lambda: h.shadow.scored == 1)
        e.decision(dismissal_decision(current, dismissal=("disengage", 1.0)), current.end_ms)
        output = e.drain()
        attention = next(event for event in output if event["type"] == "attention")
        dismissal = next(event for event in output if event["type"] == "dismiss")
        self.assertEqual(attention["decision"]["role"], "owner")
        self.assertIn("attention_decision role_at=receipt", h.lines)
        self.assertEqual((dismissal["role"], dismissal["scope"]), ("unknown", ["playback"]))

    def test_single_bind_preserves_answered_by_other(self):
        for label, text in (("attend", "1"), ("uncertain", " ".join(duck.OURS.names[0]))):
            with self.subTest(provider_label=label):
                c, e, h = self.arrange(threshold=0.70)
                current = turn("1", 0, 1500, text)
                self.publish(c, h, current)
                self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
                wait_for(lambda: bool(h.notes()))
                self.assertEqual(h.fields(h.notes()[-1])["bind"], "single")
                provider = Mock()
                provider.decide.return_value = decided(current, label).decision
                runner = ReplayRunner(
                    provider, addressing=duck.OURS, post_turn_gaps=True, other_answered_blocks=True
                )
                observed = gap("different_speaker")
                event = runner.process(current, observed)
                self.assertEqual(event.decision.answered_at_ms, current.end_ms + 400)
                e.decision(event, current.end_ms + observed["window_ms"])
                output = e.drain()
                self.assertEqual([item["type"] for item in output], ["attention"])
                decision = output[0]["decision"]
                self.assertEqual((decision["label"], decision["role"]), ("uncertain", "owner"))
                self.assertIs(decision["answered_by_other"], True)
                self.assertIn("attention_decision role_at=receipt", h.lines)

    def test_first_bind_uses_the_accepted_follow_up_bar(self):
        c, e, h = self.arrange()
        c._enrolled_roles.follow_up_min_probability = 0.2
        first = self.sample(1)
        self.publish(c, h, first)
        wait_for(lambda: h.shadow.scored == 1)
        self.decision(e, first, role="unknown", requested=True)
        current = turn("2", 1500, 3000, "2")
        self.publish(c, h, current)
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
        wait_for(lambda: h.shadow.scored == 2)
        e.decision(decided(current, "uncertain", attend=0.25), current.end_ms)
        output = e.drain()
        self.assertEqual([item["type"] for item in output], ["attention"])
        self.assertEqual(output[0]["decision"]["role"], "owner")
        self.assertIn("follow_up outcome=refused reason=below_bar attend=0.25 bar=0.4", h.lines)

    def test_decision_finishes_while_score_is_pending(self):
        gate = threading.Event()
        c, e, h = self.arrange(
            threshold=0.7, gate=gate, priority=replace(OWNER_ID, owner_only=True)
        )
        current = self.sample(1)
        self.publish(c, h, current)
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
        self.decision(e, current, role="unknown", requested=False)
        self.assertEqual(h.shadow.scored, 0)
        self.assertIn("attention_decision role_at=accept", h.lines)
        gate.set()
        wait_for(lambda: h.shadow.scored == 1)
        self.assertEqual(h.shadow.binding(current.speaker_id), ("bound", "owner"))
        e.decision(decided(current), current.end_ms)
        self.assertEqual(e.drain(), [])

    def test_later_turn_receipt_belongs_only_to_its_own_decision(self):
        c, e, h = self.arrange(priority=replace(OWNER_ID, owner_only=True))
        first, second = self.sample(1), self.sample(2, 1500, 3000)
        self.publish(c, h, first)
        wait_for(lambda: h.shadow.scored == 1)
        self.publish(c, h, second)
        wait_for(lambda: h.shadow.scored == 2)
        self.decision(e, first, role="unknown", requested=False)
        self.decision(e, second, role="owner", requested=True)

    def test_oldest_unconsumed_receipt_is_pruned(self):
        c, e, h = self.arrange(threshold=0.7, priority=replace(OWNER_ID, owner_only=True))
        first = self.sample(1)
        second = replace(self.sample(2, 1500, 3000), speaker_id="second")
        with patch("rightyo.live_speaker_id.MAX_BINDING_RECEIPTS", 1):
            for count, current in enumerate((first, second), 1):
                self.publish(c, h, current)
                wait_for(lambda: h.shadow.scored == count)
        self.decision(e, first, role="unknown", requested=False)
        self.decision(e, second, role="owner", requested=True)

    def test_binding_dropped_and_restored_invalidates_the_first_receipt(self):
        gate = threading.Event()
        c, e, h = self.arrange(
            threshold=0.7, gate=gate, scores=(1, -1, 1), priority=replace(OWNER_ID, owner_only=True)
        )
        first, drop, rebound = (
            self.sample(1),
            self.sample(2, 1500, 3500),
            self.sample(3, 3500, 5500),
        )
        for current in (first, drop, rebound):
            self.publish(c, h, current)
            self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
        gate.set()
        wait_for(lambda: h.shadow.scored == 3)
        self.assertEqual(h.shadow.binding_snapshot(first.speaker_id), ((0, 3), ("bound", "owner")))
        self.decision(e, first, role="unknown", requested=False)
        self.decision(e, rebound, role="unknown", requested=False)

    def test_self_binding_on_another_label_invalidates_a_pending_receipt(self):
        c, e, h = self.arrange(
            threshold=0.7, scores=(1, 0), priority=replace(OWNER_ID, owner_only=True)
        )
        first = self.sample(1)
        self.publish(c, h, first)
        wait_for(lambda: h.shadow.scored == 1)
        original = h.shadow.presence_snapshot(first.speaker_id)
        self.assertEqual(original.state, "bound")

        other = replace(self.sample(2, 1500, 4500), speaker_id="second")
        h.shadow._self_prints = (vector(0),)
        h.shadow.audio(tone(3))
        h.shadow.turn(other, spans=((1500, 4500),), in_window=True)
        wait_for(lambda: h.shadow.scored == 2)
        self.assertTrue(h.shadow.self_bound(other.speaker_id))
        current = h.shadow.presence_snapshot(first.speaker_id)
        self.assertEqual(current.revision, original.revision)
        self.assertIsNot(current.self_generation, original.self_generation)
        self.decision(e, first, role="unknown", requested=False)

    def test_forget_invalidates_the_first_receipt(self):
        c, e, h = self.arrange(threshold=0.7, priority=replace(OWNER_ID, owner_only=True))
        first, renewed = self.sample(1), self.sample(2, 1500, 3000)
        self.publish(c, h, first)
        wait_for(lambda: h.shadow.scored == 1)
        h.shadow.forget()
        self.publish(c, h, renewed)
        self.assertEqual(h.shadow.binding_snapshot(first.speaker_id), ((1, 1), ("bound", "owner")))
        self.decision(e, first, role="unknown", requested=False)
        e.decision(decided(renewed), renewed.end_ms)
        request = next(event for event in e.drain() if event["type"] == "request")
        self.assertEqual((request["turn"]["role"], request["decision"]["role"]), ("owner", "owner"))
        self.assertFalse(any(note.endswith(" role_at=receipt") for note in h.lines))

    def test_guest_and_inferred_first_binds_keep_their_acceptance_role(self):
        for identity, inferred in (("guest", False), ("owner", True)):
            with self.subTest(identity=identity):
                c, e, h = self.arrange(
                    threshold=0.7,
                    priority=replace(OWNER_ID, owner_only=True),
                    entries=[{**ENTRIES[0], "id": identity}],
                )
                current = self.sample(1)
                if inferred:
                    c._observe_inferred(c._generation, current.utterance_id)
                self.publish(c, h, current)
                wait_for(lambda: h.shadow.scored == 1)
                self.assertEqual(h.shadow.binding(current.speaker_id), ("bound", identity))
                self.decision(e, current, role="unknown", requested=False)
                self.assertIn("attention_decision role_at=accept", h.lines)

    def test_trusted_first_bind_carries_only_its_configured_role(self):
        c, e, h = self.arrange(
            threshold=0.7,
            priority=replace(OWNER_ID, trusted=("trusted",), owner_only=True),
            entries=[{**ENTRIES[0], "id": "trusted"}],
        )
        current = self.sample(1)
        self.publish(c, h, current)
        wait_for(lambda: h.shadow.scored == 1)
        self.decision(e, current, role="trusted", requested=False)

    def test_pending_binding_turn_cannot_exit_privacy(self):
        gate = threading.Event()
        c, e, h = self.arrange(
            threshold=0.7, gate=gate, priority=replace(OWNER_ID, owner_only=True)
        )
        current = replace(self.sample(1), text=" ".join((*duck.OURS.names, *PRIVACY_EXIT[0])))
        self.publish(c, h, current)
        self.assertEqual(self.transcript(e)["turn"]["role"], "unknown")
        e.duck(True, now_ms=current.end_ms + 1)
        gate.set()
        wait_for(lambda: h.shadow.scored == 1)
        self.assertEqual(h.shadow.binding(current.speaker_id), ("bound", "owner"))
        e.decision(decided(current), current.end_ms + 2)
        output = e.drain()
        self.assertFalse(any(event.get("action") == "exit" for event in output))
        self.assertEqual([event["type"] for event in output], ["attention"])
        self.assertEqual(output[0]["decision"]["role"], "unknown")
        self.assertTrue(e.ducked)

    def test_private_exit_never_consults_the_decision_role(self):
        c, e, h = self.arrange(threshold=0.7)
        current = replace(self.sample(1), text=" ".join((*duck.OURS.names, *PRIVACY_EXIT[0])))
        e.duck(True, now_ms=0)
        e.drain()
        with patch.object(c._enrolled_roles, "first_bind_role", return_value="owner") as receipt:
            self.publish(c, h, current)
            self.assertEqual(e.drain(), [])
            self.assertEqual(e._private_counts["exit_refused"], 1)
            receipt.assert_not_called()
        self.assertTrue(e.ducked)
        self.assertEqual(h.shadow.offered, 0)
