"""Roles and precedence from live speaker identification (#137 step 4b, #113).

Synthetic data only: authored turns, stand-in bindings or scripted embedder vectors, and
authored PCM. No model, microphone, recorded voice or enrollment store is used.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import unittest
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

from test_edge_attribution import SPEECH, Timeline, Units
from test_live_speaker_id import ENTRIES, RATE, ScriptedEmbedder, settings, tone, unit, wait_for
from test_turn_merge import fragment

from rightyo.contracts import (
    Conversation,
    DecisionEvent,
    Dismissal,
    ProviderDecision,
    SpeakerPriority,
    Turn,
)
from rightyo.live_audio import FRAME_BYTES, LiveConfig, LiveProcessor
from rightyo.live_speaker_id import EnrolledRoles, ShadowSpeakerId
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool_events import SpeechEvents
from rightyo.turn_merge import TurnMerger

ROOT = Path(__file__).resolve().parents[1]
SESSION = "enrolled-live"
OWNER_ID = SpeakerPriority(owners=("owner",))
LOCAL = (
    "whisper_executable",
    "whisper_model",
    "diarization_library",
    "diarization_model",
    "microphone_helper",
)


def turn(name, start, end, text, speaker="Speaker A", *, overlap=False):
    return Turn(
        SESSION,
        name,
        1,
        start,
        end,
        text,
        speaker,
        True,
        overlap,
        "fake-local-recognizer",
        "live-microphone",
        "diarization-timeline",
    )


def decided(current, label="attend", recipient="system", attend=None, confidence=0.9):
    if attend is None:
        probabilities = {k: float(k == label) for k in ("attend", "ignore", "uncertain")}
    else:
        probabilities = {"attend": attend, "ignore": 0.0, "uncertain": 1.0 - attend}
    return DecisionEvent(
        current,
        ProviderDecision(label, recipient, confidence, probabilities, "mock-v1", "mock", 0.9),
        current.revision,
        0.0,
        0.0,
    )


class Bindings:
    """A stand-in identifier: label -> (state, identity), changed by the test."""

    def __init__(self, **labels):
        self.labels = {label.replace("_", " "): value for label, value in labels.items()}

    def binding(self, label):
        return self.labels.get(label)


class HaildModel:
    """The checks haild's `RightyoInputConsumer` applies to an enrolled stream.

    Mirrors hailing-station `RightyoInputEvent.validate` and the consumer's correlation:
    roles are one of the four literals on enrolled sessions (only `unknown` on anonymous
    ones); a request's turn, decision and context turns must equal the admitted transcript
    and attention records; an `override` must cite an utterance whose transcript and
    attention both carried `role: owner`, and only on an enrolled session.
    """

    ROLES = {"owner", "trusted", "participant", "unknown"}

    def __init__(self, test):
        self.test = test

    @staticmethod
    def fingerprint(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def role(self, value, enrolled):
        if value is None:
            return
        self.test.assertIn(value, self.ROLES if enrolled else {"unknown"})

    def check(self, events):
        started = events[0]
        self.test.assertEqual((started["type"], started["phase"]), ("session", "started"))
        enrolled = started["capabilities"]["speakers"] == "enrolled"
        self.test.assertIn(started["capabilities"]["speakers"], {"anonymous", "enrolled"})
        finals, attentions, owners, decided = {}, {}, set(), set()
        for event in events[1:]:
            self.test.assertNotIn("inferred", json.dumps(event))
            kind = event["type"]
            if kind == "transcript":
                current = event["turn"]
                self.role(current.get("role"), enrolled)
                self.test.assertNotIn(current["utterance_id"], finals)
                finals[current["utterance_id"]] = self.fingerprint(current)
                if current.get("role") == "owner":
                    owners.add(current["utterance_id"])
            elif kind == "attention":
                utterance = event["utterance_id"]
                self.test.assertIn(utterance, finals)
                self.role(event["decision"].get("role"), enrolled)
                decided.add(utterance)
                if event["decision"].get("role") != "owner":
                    owners.discard(utterance)
                if "request_id" in event:
                    self.test.assertEqual(event["request_id"], f"{SESSION}:{utterance}")
                    attentions[event["request_id"]] = self.fingerprint(event["decision"])
            elif kind == "request":
                self.test.assertEqual(
                    finals[event["turn"]["utterance_id"]], self.fingerprint(event["turn"])
                )
                self.test.assertEqual(
                    attentions[event["request_id"]], self.fingerprint(event["decision"])
                )
                for prior in event["context"]["turns"]:
                    self.test.assertEqual(finals[prior["utterance_id"]], self.fingerprint(prior))
            elif kind == "override":
                self.test.assertTrue(enrolled)
                self.test.assertEqual(event["role"], "owner")
                self.test.assertIn(event["by_utterance_id"], owners)
                self.test.assertIn(event["by_utterance_id"], decided)
            elif kind == "dismiss":
                self.test.assertIn(event["utterance_id"], finals)
                self.role(event.get("role"), enrolled)


class HaildModelSanityTests(unittest.TestCase):
    def test_the_authored_enrolled_fixtures_pass_the_model(self):
        for name in ("enrolled-override.jsonl", "enrolled-formed-request.jsonl"):
            events = [
                json.loads(line) for line in (ROOT / "examples" / name).read_text().splitlines()
            ]
            with self.subTest(fixture=name):
                # The fixtures use their own session id.
                for event in events:
                    if "request_id" in event:
                        event["request_id"] = event["request_id"].replace(
                            event["session_id"], SESSION
                        )
                HaildModel(self).check(events)


class EnrolledRolesProviderTests(unittest.TestCase):
    def test_role_for_follows_the_current_binding(self):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",), trusted=("jev",)))
        a = turn("a", 0, 1000, "hi")
        self.assertEqual(roles.role_for(a), "unknown")  # identification not started
        roles.source = Bindings()
        self.assertEqual(roles.role_for(a), "unknown")  # not scored yet
        cases = {
            ("bound", "owner"): "owner",
            ("bound", "jev"): "trusted",
            ("bound", "guest"): "participant",  # enrolled, but given no role
            ("tentative", "owner"): "unknown",
            ("unknown", "owner"): "participant",  # scored, matches no enrolled voice
        }
        for binding, expected in cases.items():
            roles.source = Bindings(Speaker_A=binding)
            with self.subTest(binding=binding):
                self.assertEqual(roles.role_for(a), expected)

    def test_unlabelled_overlapping_and_utterance_local_turns_are_never_looked_up(self):
        roles = EnrolledRoles(OWNER_ID)
        roles.source = Bindings(Speaker_A=("bound", "owner"))
        self.assertEqual(roles.role_for(turn("a", 0, 1, "x", None)), "unknown")
        self.assertEqual(roles.role_for(turn("b", 0, 1, "x", overlap=True)), "unknown")
        local = Turn(
            SESSION,
            "c",
            1,
            0,
            1,
            "x",
            "u1 Speaker A",
            True,
            False,
            "r",
            "synthetic",
            "diarization-utterance",
        )
        self.assertEqual(roles.role_for(local), "unknown")

    def test_a_failing_source_degrades_to_unknown(self):
        roles = EnrolledRoles(OWNER_ID)
        roles.source = MagicMock(binding=MagicMock(side_effect=RuntimeError("boom")))
        self.assertEqual(roles.role_for(turn("a", 0, 1, "x")), "unknown")

    def test_session_labels_in_the_configuration_still_apply(self):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",), trusted=("Speaker B",)))
        roles.source = Bindings()
        self.assertEqual(roles.role_for(turn("b", 0, 1, "x", "Speaker B")), "trusted")

    def test_the_shadow_identifier_exposes_bindings_without_blocking(self):
        embedder = ScriptedEmbedder([unit(1, 0.1, 0), unit(0, 1, 0)])
        shadow = ShadowSpeakerId(
            settings(bind_min_seconds=1.0),
            report=lambda _line: None,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: [dict(entry) for entry in ENTRIES],
        )
        shadow.start()
        self.assertIsNone(shadow.binding("Speaker A"))
        received = 0
        for speaker in ("Speaker A", "Speaker B"):
            pcm = tone(2.0)  # one 3 s window: one scripted vector per turn
            shadow.audio(pcm)
            shadow.turn(turn(f"t-{speaker}", received, received + 2000, "x", speaker))
            received += 2000
        shadow.close(drain=True)
        self.assertEqual(shadow.binding("Speaker A"), ("bound", "owner"))
        self.assertEqual(shadow.binding("Speaker B"), ("bound", "guest"))
        roles = EnrolledRoles(OWNER_ID)
        roles.source = shadow
        self.assertEqual(roles.role_for(turn("x", 0, 1, "x", "Speaker A")), "owner")
        self.assertEqual(roles.role_for(turn("y", 0, 1, "x", "Speaker B")), "participant")

    def test_an_inferred_owner_tail_never_moves_a_guest_label_toward_owner(self):
        """The owner's tail joined onto a guest's turn must not become guest evidence."""
        guest, owner = unit(0, 1, 0), unit(1, 0, 0)
        # Were the inferred turns embedded, these owner vectors would be accumulated.
        embedder = ScriptedEmbedder([guest, owner, owner, owner, owner])
        lines = []
        shadow = ShadowSpeakerId(
            settings(bind_min_seconds=1.0),
            report=lines.append,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: [dict(entry) for entry in ENTRIES],
        )
        shadow.start()
        received = 0
        for index, inferred in enumerate((False, True, True, True, True)):
            shadow.audio(tone(2.0))
            current = turn(f"g{index}", received, received + 2000, "x", "Speaker B")
            shadow.turn(current, inferred=inferred)
            received += 2000
        shadow.close(drain=True)
        self.assertEqual(embedder.calls, 1)
        self.assertEqual((shadow.scored, shadow.inferred, shadow.offered), (1, 4, 5))
        state = shadow._labels["Speaker B"]
        self.assertEqual((state.turns, state.state, state.identity), (1, "bound", "guest"))
        self.assertLess(state.scores["owner"], 0.01)
        roles = EnrolledRoles(OWNER_ID)
        roles.source = shadow
        self.assertEqual(roles.role_for(turn("later", 0, 1, "x", "Speaker B")), "participant")
        self.assertTrue(lines[-1].endswith(" inferred=4"))
        self.assertEqual(len([line for line in lines if line.startswith("speaker_id label=")]), 1)

    def test_a_failed_identifier_reports_no_bindings(self):
        embedder = ScriptedEmbedder([unit(1, 0.1, 0)], fail_on=2)
        shadow = ShadowSpeakerId(
            settings(bind_min_seconds=1.0),
            report=lambda _line: None,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: [dict(entry) for entry in ENTRIES],
        )
        self.addCleanup(shadow.close)
        shadow.start()
        roles = EnrolledRoles(OWNER_ID)
        roles.source = shadow
        received = 0
        for name in ("one", "two"):
            shadow.audio(tone(2.0))
            shadow.turn(turn(name, received, received + 2000, "x"))
            received += 2000
            if name == "one":
                wait_for(lambda: shadow.binding("Speaker A") is not None)
                self.assertEqual(roles.role_for(turn("a", 0, 1, "x")), "owner")
        wait_for(lambda: not shadow.active)
        self.assertIsNone(shadow.binding("Speaker A"))
        self.assertEqual(roles.role_for(turn("b", 0, 1, "x")), "unknown")


class EnrolledEventTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()
        self.bindings = Bindings()
        self.roles = EnrolledRoles(OWNER_ID)
        self.roles.source = self.bindings
        self.stream = []

    def start(self, **options):
        self.events.start(SESSION, now_ms=0, priority=self.roles, **options)
        self.stream += self.events.drain()
        return self.stream[0]

    def say(self, current, label="attend", *, inferred=False, **options):
        self.events.transcript(current, current.end_ms, inferred=inferred)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        out = self.events.drain()
        self.stream += out
        return out

    def finish(self):
        self.events.end("stopped", 100000)
        self.stream += self.events.drain()
        HaildModel(self).check(self.stream)
        return self.stream

    @staticmethod
    def of(events, kind):
        return [event for event in events if event["type"] == kind]

    def test_session_is_enrolled_and_roles_follow_binding_over_time(self):
        started = self.start()
        self.assertEqual(started["capabilities"]["speakers"], "enrolled")
        # Before the identifier has bound anything, the owner's first turn is unknown.
        first = self.say(
            turn("u1", 0, 2000, "Hello there.", "Speaker A"), "ignore", recipient="other_human"
        )
        self.assertEqual(first[0]["turn"]["role"], "unknown")
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        self.bindings.labels["Speaker B"] = ("unknown", "owner")  # a video voice
        later = self.say(turn("u2", 3000, 5000, "Haili, what time is it?", "Speaker A"))
        self.assertEqual(later[0]["turn"]["role"], "owner")
        request = self.of(later, "request")[0]
        self.assertEqual(request["turn"]["role"], "owner")
        self.assertEqual(request["decision"]["role"], "owner")
        # Context keeps the role each prior turn had when it was emitted.
        self.assertEqual(request["context"]["turns"][0]["role"], "unknown")
        video = self.say(
            turn("u3", 6000, 8000, "Buy now!", "Speaker B"), "ignore", recipient="unknown"
        )
        self.assertEqual(video[0]["turn"]["role"], "participant")
        self.bindings.labels["Speaker C"] = ("tentative", "owner")
        guest = self.say(turn("u4", 9000, 10000, "Hm.", "Speaker C"), "ignore", recipient="unknown")
        self.assertEqual(guest[0]["turn"]["role"], "unknown")
        self.finish()

    def test_a_binding_change_never_rewrites_an_emitted_turn(self):
        self.start()
        current = turn("u1", 0, 2000, "Haili, lights on.", "Speaker A")
        self.events.transcript(current, 2000)
        # The identifier binds while the decision is pending: the turn keeps its role.
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        self.events.decision(decided(current), 2100)
        out = self.events.drain()
        self.stream += out
        self.assertEqual(
            {e["type"]: e.get("turn", e.get("decision", {})).get("role") for e in out},
            {"transcript": "unknown", "attention": "unknown", "request": "unknown"},
        )
        self.finish()

    def test_unenrolled_voices_are_not_blocked(self):
        self.start()
        self.bindings.labels["Speaker B"] = ("unknown", "owner")
        out = self.say(turn("u1", 0, 2000, "Haili, what's the weather?", "Speaker B"))
        self.assertEqual(len(self.of(out, "request")), 1)
        self.assertEqual(self.of(out, "request")[0]["turn"]["role"], "participant")
        self.finish()

    def test_a_bound_owner_overrides_an_earlier_participant_request(self):
        self.start()
        self.bindings.labels.update(
            {"Speaker A": ("bound", "owner"), "Speaker B": ("unknown", "owner")}
        )
        self.say(turn("u1", 0, 2000, "Haili, delete everything.", "Speaker B"))
        out = self.say(
            turn("u2", 3000, 4000, "Ignore that.", "Speaker A"), "uncertain", recipient="unknown"
        )
        overrides = self.of(out, "override")
        self.assertEqual(len(overrides), 1)
        self.assertEqual(overrides[0]["superseded_request_id"], f"{SESSION}:u1")
        self.finish()

    def test_an_inferred_owner_turn_never_overrides(self):
        self.start()
        self.bindings.labels.update(
            {"Speaker A": ("bound", "owner"), "Speaker B": ("unknown", "owner")}
        )
        self.say(turn("u1", 0, 2000, "Haili, delete everything.", "Speaker B"))
        # A stop phrase whose label was inferred: no override, no stop.
        stop = self.say(
            turn("u2", 3000, 4000, "Ignore that.", "Speaker A"),
            "uncertain",
            recipient="unknown",
            inferred=True,
        )
        self.assertEqual(self.of(stop, "override"), [])
        self.assertEqual(stop[0]["turn"]["role"], "owner")
        # An attended inferred owner turn still forms its request (attention keeps the
        # role) but supersedes nothing, and stays open to a later real owner override.
        attended = self.say(
            turn("u3", 5000, 7000, "Haili, keep the files.", "Speaker A"), inferred=True
        )
        self.assertEqual(self.of(attended, "override"), [])
        self.assertEqual(self.of(attended, "request")[0]["turn"]["role"], "owner")
        real = self.say(
            turn("u4", 8000, 9000, "Never mind.", "Speaker A"), "uncertain", recipient="unknown"
        )
        self.assertEqual(
            sorted(e["superseded_request_id"] for e in self.of(real, "override")),
            [f"{SESSION}:u1", f"{SESSION}:u3"],
        )
        self.finish()

    def test_an_inferred_owner_turn_does_not_supersede_pending_decisions(self):
        self.start()
        self.bindings.labels.update(
            {"Speaker A": ("bound", "owner"), "Speaker B": ("unknown", "owner")}
        )
        guest = turn("u1", 0, 2000, "Haili, order pizza.", "Speaker B")
        self.events.transcript(guest, 2000)
        owner = turn("u2", 3000, 4000, "Haili, lights off.", "Speaker A")
        self.events.transcript(owner, 4000, inferred=True)
        self.events.decision(decided(owner), 4100)
        self.events.decision(decided(guest), 4200)
        out = self.events.drain()
        self.stream += out
        self.assertEqual(self.of(out, "override"), [])
        self.assertEqual(len(self.of(out, "request")), 2)
        self.finish()

    def test_an_inferred_owner_stop_phrase_only_stops_playback(self):
        self.start(dismissal=Dismissal())
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        self.say(turn("u1", 0, 2000, "Haili, play music.", "Speaker A"))
        out = self.say(
            turn("u2", 3000, 3500, "Stop.", "Speaker A"),
            "uncertain",
            recipient="unknown",
            inferred=True,
        )
        dismiss = self.of(out, "dismiss")
        self.assertEqual(len(dismiss), 1)
        self.assertEqual(dismiss[0]["scope"], ["playback"])
        self.assertEqual(dismiss[0]["role"], "owner")
        # It withdraws nothing: the owner's delivered request stands.
        self.assertEqual(dismiss[0]["withdrawn_request_ids"], [])
        self.assertIn(f"{SESSION}:u1", self.events._delivered)
        self.finish()

    def test_an_inferred_stop_phrase_withdraws_no_pending_request(self):
        self.start(dismissal=Dismissal())
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        owner = turn("u1", 0, 2000, "Haili, play music.", "Speaker A")
        self.events.transcript(owner, 2000)
        stop = turn("u2", 3000, 3500, "Stop.", "Speaker A")
        self.events.transcript(stop, 3500, inferred=True)
        self.events.decision(decided(owner), 3600)
        self.events.decision(decided(stop, "uncertain", "unknown"), 3700)
        out = self.events.drain()
        self.stream += out
        self.assertEqual(self.of(out, "dismiss")[0]["withdrawn_request_ids"], [])
        self.assertEqual(len(self.of(out, "request")), 1)
        self.finish()

    def test_an_inferred_model_judged_dismissal_withdraws_nothing(self):
        self.start(dismissal=Dismissal(), conversation=Conversation(window_ms=20000))
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        self.say(turn("u1", 0, 2000, "Haili, play music.", "Speaker A"))
        current = turn("u2", 3000, 4000, "Okay that's enough of that.", "Speaker A")
        self.events.transcript(current, 4000, inferred=True)
        judged = DecisionEvent(
            current,
            ProviderDecision(
                "uncertain",
                "system",
                0.9,
                {"attend": 0.0, "ignore": 0.0, "uncertain": 1.0},
                "mock-v1",
                "mock",
                0.9,
                dismissal="disengage",
                dismissal_choice="disengage",
                dismissal_confidence=0.95,
            ),
            1,
            0.0,
            0.0,
        )
        self.events.decision(judged, 4100)
        out = self.events.drain()
        self.stream += out
        dismiss = self.of(out, "dismiss")
        self.assertEqual(len(dismiss), 1)
        self.assertEqual(dismiss[0]["scope"], ["playback"])
        self.assertEqual(dismiss[0]["withdrawn_request_ids"], [])
        self.assertNotIn("cooldown_until_ms", dismiss[0])
        # The owner's engagement is not ended by an inferred turn's dismissal.
        self.assertEqual(self.of(out, "conversation"), [])
        self.assertIsNotNone(self.events._engaged)
        self.finish()

    def test_a_non_inferred_owner_dismissal_still_withdraws(self):
        self.start(dismissal=Dismissal())
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        self.say(turn("u1", 0, 2000, "Haili, play music.", "Speaker A"))
        out = self.say(
            turn("u2", 3000, 3500, "Stop.", "Speaker A"), "uncertain", recipient="unknown"
        )
        self.assertEqual(self.of(out, "dismiss")[0]["withdrawn_request_ids"], [f"{SESSION}:u1"])
        self.finish()

    def test_a_pending_turn_that_lost_its_binding_is_superseded(self):
        """#147: the owner exemption from supersession is re-checked too."""
        self.start()
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        stale = turn("u1", 0, 2000, "Haili, delete everything.", "Speaker A")
        self.events.transcript(stale, 2000)
        # Speaker A's binding is lost; Speaker C is now the bound owner.
        self.bindings.labels["Speaker A"] = ("unknown", "owner")
        self.bindings.labels["Speaker C"] = ("bound", "owner")
        owner = turn("u2", 3000, 4000, "Haili, keep everything.", "Speaker C")
        self.events.transcript(owner, 4000)
        self.events.decision(decided(owner), 4100)
        self.events.decision(decided(stale), 4200)
        out = self.events.drain()
        self.stream += out
        self.assertEqual(
            [(e["superseded_request_id"], e["by_utterance_id"]) for e in self.of(out, "override")],
            [(f"{SESSION}:u1", "u2")],
        )
        self.assertEqual([e["request_id"] for e in self.of(out, "request")], [f"{SESSION}:u2"])
        # The stale turn's published role is unchanged.
        self.assertEqual(self.of(out, "attention")[1]["decision"]["role"], "owner")
        self.finish()

    def test_a_still_bound_pending_owner_turn_keeps_its_exemption(self):
        self.start()
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        first = turn("u1", 0, 2000, "Haili, lights on.", "Speaker A")
        self.events.transcript(first, 2000)
        second = turn("u2", 3000, 4000, "Haili, and the heating.", "Speaker A")
        self.events.transcript(second, 4000)
        self.events.decision(decided(second), 4100)
        self.events.decision(decided(first), 4200)
        out = self.events.drain()
        self.stream += out
        self.assertEqual(self.of(out, "override"), [])
        self.assertEqual(len(self.of(out, "request")), 2)
        self.finish()

    def test_owner_only_filters_on_the_rechecked_role(self):
        self.roles = EnrolledRoles(SpeakerPriority(owners=("owner",), owner_only=True))
        self.roles.source = self.bindings
        self.start()
        self.bindings.labels["Speaker A"] = ("bound", "owner")
        bound = self.say(turn("u1", 0, 2000, "Haili, lights on.", "Speaker A"))
        self.assertEqual(len(self.of(bound, "request")), 1)
        current = turn("u2", 3000, 4000, "Haili, open the door.", "Speaker A")
        self.events.transcript(current, 4000)
        self.bindings.labels["Speaker A"] = ("unknown", "owner")
        self.events.decision(decided(current), 4100)
        out = self.events.drain()
        self.stream += out
        attention = self.of(out, "attention")[0]
        self.assertEqual(attention["decision"]["label"], "ignore")
        self.assertEqual(attention["decision"]["role"], "owner")  # published role unchanged
        self.assertEqual(self.of(out, "request"), [])
        self.finish()

    def test_authority_is_rechecked_when_the_decision_arrives(self):
        self.start()
        self.bindings.labels.update(
            {"Speaker A": ("bound", "owner"), "Speaker B": ("unknown", "owner")}
        )
        self.say(turn("u1", 0, 2000, "Haili, delete everything.", "Speaker B"))
        owner = turn("u2", 3000, 4000, "Ignore that.", "Speaker A")
        self.events.transcript(owner, 4000)
        # The label loses its binding before the decision: no override fires.
        self.bindings.labels["Speaker A"] = ("unknown", "owner")
        self.events.decision(decided(owner, "uncertain", "unknown"), 4100)
        out = self.events.drain()
        self.stream += out
        self.assertEqual(self.of(out, "override"), [])
        # The published role is unchanged.
        self.assertEqual(self.of(out, "attention")[0]["decision"]["role"], "owner")
        self.finish()

    def test_inferred_marks_are_pruned_by_retention(self):
        self.events = SpeechEvents(retention_ms=60000)
        self.start()
        self.events.transcript(turn("u1", 0, 1000, "Hi.", "Speaker A"), 1000, inferred=True)
        self.assertIn("u1", self.events._inferred)
        self.events.expire(70000)
        self.assertNotIn("u1", self.events._inferred)

    def test_inferred_has_no_effect_on_anonymous_sessions(self):
        outputs = []
        for inferred in (False, True):
            events = SpeechEvents()
            events.start(SESSION, now_ms=0)
            current = turn("u1", 0, 2000, "Haili, lights on.")
            events.transcript(current, 2000, inferred=inferred)
            events.decision(decided(current), 2100)
            events.end("stopped", 3000)
            outputs.append(json.dumps(events.drain(), sort_keys=True))
        self.assertEqual(outputs[0], outputs[1])
        with self.assertRaises(Exception):
            SpeechEvents().transcript(current, 0, inferred="yes")

    def test_an_engaged_owner_keeps_the_conversation(self):
        self.start(conversation=Conversation(window_ms=10000))
        self.bindings.labels.update(
            {"Speaker A": ("bound", "owner"), "Speaker B": ("unknown", "owner")}
        )
        owner = self.say(turn("u1", 0, 2000, "Haili, set a timer.", "Speaker A"))
        self.assertEqual(self.of(owner, "conversation")[0]["state"], "engaged")
        guest = self.say(turn("u2", 3000, 4000, "Haili, what's on TV?", "Speaker B"))
        # The guest is heard (a request), but the engagement stays with the owner.
        self.assertEqual(len(self.of(guest, "request")), 1)
        self.assertEqual(self.of(guest, "conversation"), [])
        follow = self.say(
            turn("u3", 5000, 6000, "And make it ten minutes.", "Speaker A"),
            "uncertain",
            recipient="unknown",
            attend=0.5,
        )
        self.assertTrue(self.of(follow, "request")[0]["decision"]["follow_up"])
        self.finish()

    def test_without_identification_roles_a_guest_takes_the_engagement(self):
        events = SpeechEvents()
        events.start(
            SESSION,
            now_ms=0,
            priority=ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",))),
            conversation=Conversation(window_ms=10000),
        )
        for current in (
            turn("u1", 0, 2000, "Haili, set a timer.", "Speaker A"),
            turn("u2", 3000, 4000, "Haili, what's on TV?", "Speaker B"),
        ):
            events.transcript(current, current.end_ms)
            events.decision(decided(current), current.end_ms + 100)
        moved = [e for e in events.drain() if e["type"] == "conversation"]
        self.assertEqual([e["speaker_id"] for e in moved], ["Speaker A", "Speaker B"])

    def test_the_enrolled_follow_up_bar_applies_to_owners_only(self):
        self.roles.follow_up_min_probability = 0.2
        self.start(conversation=Conversation(window_ms=20000, follow_up_min_probability=0.4))
        self.bindings.labels.update(
            {"Speaker A": ("bound", "owner"), "Speaker B": ("unknown", "owner")}
        )
        self.say(turn("u1", 0, 2000, "Haili, set a timer.", "Speaker A"))
        low = self.say(
            turn("u2", 3000, 4000, "Maybe longer.", "Speaker A"),
            "uncertain",
            recipient="unknown",
            attend=0.25,
        )
        self.assertEqual(len(self.of(low, "request")), 1)
        # A participant engaged later still needs the configured bar.
        self.bindings.labels["Speaker A"] = ("unknown", "owner")
        self.say(turn("u3", 30000, 31000, "Haili, play jazz.", "Speaker B"))
        guest = self.say(
            turn("u4", 32000, 33000, "Louder.", "Speaker B"),
            "uncertain",
            recipient="unknown",
            attend=0.25,
        )
        self.assertEqual(self.of(guest, "request"), [])
        self.finish()


class InferredLabelTrackingTests(unittest.TestCase):
    def test_tail_join_marks_the_joined_turn(self):
        emitted = []
        merger = TurnMerger(2000, 24000, emitted.append, tail_join_ms=400)
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(fragment("down a bit.", 900, 1100, speaker=None))
        merger.flush()
        self.assertTrue(emitted[0]["inferred"])

    def test_plain_joins_carry_no_mark_and_marks_propagate(self):
        emitted = []
        merger = TurnMerger(2000, 24000, emitted.append)
        merger.offer(fragment("Turn the volume", 0, 600))
        merger.offer(fragment("down.", 1000, 1200))
        merger.flush()
        self.assertNotIn("inferred", emitted[0])
        marked = fragment("up.", 1500, 1700)
        marked["inferred"] = True
        merger.offer(fragment("Turn it", 1000, 1200))
        merger.offer(marked)
        merger.flush()
        self.assertTrue(emitted[1]["inferred"])

    def run_live(self, units, timeline, **options):
        turns, inferred = [], []
        config = LiveConfig(
            "inferred-test",
            provenance="causal-replay",
            transcriber=units,
            diarizer=timeline,
            on_inferred=inferred.append,
            **options,
        )
        processor = LiveProcessor(config, turns.append)
        for offset in range(0, len(SPEECH), FRAME_BYTES * 10):
            processor.push_pcm16(SPEECH[offset : offset + FRAME_BYTES * 10])
        processor.finish()
        return turns, inferred

    def test_edge_attributed_words_are_reported_before_the_turn(self):
        units = Units((" Turn the volume", 0, 600), (" down", 600, 800), (" a bit.", 850, 1000))
        turns, inferred = self.run_live(units, Timeline((0, 700, 1)), edge_attribution_ms=300)
        self.assertEqual([t.speaker_id for t in turns], ["Speaker A"])
        self.assertEqual(inferred, [turns[0].utterance_id])
        turns, inferred = self.run_live(units, Timeline((0, 1000, 1)), edge_attribution_ms=300)
        self.assertEqual(inferred, [])  # every word was labelled by the diarizer
        turns, inferred = self.run_live(units, Timeline((0, 700, 1)))
        self.assertEqual(inferred, [])  # edge attribution off


class TwoSpeakerTurns:
    """A stand-in processor: Speaker A, then B, then A, two seconds each; A's last turn
    reports an inferred label."""

    def __init__(self, config, on_turn):
        self.config, self.on_turn = config, on_turn
        self.received_ms = self.emitted_ms = 0
        self.count = 0

    def push_pcm16(self, pcm):
        self.received_ms += len(pcm) // 32
        while self.received_ms - self.emitted_ms >= 2000 and self.count < 3:
            self.count += 1
            start, self.emitted_ms = self.emitted_ms, self.emitted_ms + 2000
            utterance = f"live-{self.count}"
            if self.count == 3:
                self.config.on_inferred(utterance)
            self.on_turn(
                Turn(
                    self.config.session_id,
                    utterance,
                    1,
                    start,
                    self.emitted_ms,
                    "Ignore that." if self.count == 3 else "Some words.",
                    "Speaker B" if self.count == 2 else "Speaker A",
                    True,
                    False,
                    "fake-local-recognizer",
                    self.config.provenance,
                    "diarization-timeline",
                )
            )

    def finish(self):
        pass

    def close(self):
        pass


class FakeShadow:
    def __init__(self, bindings):
        self.bindings = bindings
        self.turns = []
        self.inferred = []
        self.closed = False

    def start(self):
        pass

    def audio(self, pcm):
        pass

    def turn(self, current, *, inferred=False):
        self.turns.append(current.utterance_id)
        if inferred:
            self.inferred.append(current.utterance_id)
            return
        # Bind after the first scored turn of each label, as a live identifier would.
        self.bindings.labels.setdefault(
            current.speaker_id,
            ("bound", "owner") if current.speaker_id == "Speaker A" else ("unknown", "owner"),
        )

    def binding(self, label):
        return self.bindings.binding(label)

    def close(self, drain=False):
        self.closed = True


class ControllerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        asset = self.root / "asset"
        asset.touch()
        demo = self.root / "generated-tone.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(RATE)
            audio.writeframes(tone(7.0))
        self.asset, self.demo = asset, demo
        self.base = PrototypeConfig(asset, asset, asset, asset, asset, demo)

    def identification(self, **options):
        return SpeakerIdConfig(self.asset, self.asset, live=True, **options)

    def session(self, config, factory, *, report=None):
        controller = PrototypeController(
            config,
            processor_factory=TwoSpeakerTurns,
            capture_factory=MagicMock(),
            provider_factory=MagicMock(),
            event_publisher=(publisher := SpeechEvents()),
            report=report,
            speaker_id_factory=factory,
        )
        self.addCleanup(controller.close)
        self.publisher = publisher
        controller.start({"mode": "demo", "session_id": SESSION})
        wait_for(lambda: controller.snapshot(heartbeat=False)["phase"] == "complete", 5)
        wait_for(lambda: not controller._audio_thread.is_alive(), 5)
        return controller.drain_events()

    def test_roles_off_is_byte_identical(self):
        baseline = json.dumps(self.session(self.base, MagicMock()), sort_keys=True)
        for speaker_id in (self.identification(), SpeakerIdConfig(self.asset, self.asset)):
            config = replace(self.base, speaker_id=speaker_id)
            events = self.session(config, lambda s, *, report: FakeShadow(Bindings()), report=None)
            with self.subTest(speaker_id=speaker_id):
                self.assertEqual(json.dumps(events, sort_keys=True), baseline)
        self.assertIn('"speakers": "anonymous"', baseline)

    def test_roles_on_apply_bindings_without_a_diagnostic_channel(self):
        shadows = []

        def factory(settings, *, report):
            shadows.append(FakeShadow(Bindings()))
            return shadows[0]

        config = replace(
            self.base,
            speakers=OWNER_ID,
            speaker_id=self.identification(roles=True),
            edge_attribution_ms=300,
            tail_join_ms=1000,
        )
        events = self.session(config, factory)
        self.assertEqual(len(shadows), 1)
        self.assertEqual(shadows[0].turns, ["live-1", "live-2", "live-3"])
        self.assertEqual(events[0]["capabilities"]["speakers"], "enrolled")
        roles = [e["turn"]["role"] for e in events if e["type"] == "transcript"]
        # The first turn of each voice comes before its binding; later turns carry it.
        self.assertEqual(roles, ["unknown", "unknown", "owner"])
        # The processor's inferred-label report reached the producer.
        self.assertEqual(set(self.publisher._inferred), {"live-3"})
        # ...and the identifier, which counts the turn instead of scoring it.
        self.assertEqual(shadows[0].inferred, ["live-3"])
        HaildModel(self).check(events)

    def test_load_requires_live_and_a_named_enrolled_role(self):
        base = {key: str(self.asset) for key in LOCAL} | {"demo_audio": str(self.demo)}
        section = {"python": str(self.asset), "model": str(self.asset), "live": True}
        path = self.root / "config.json"
        valid = base | {
            "speakers": {"owner": ["owner"]},
            "turns": {"edge_attribution_ms": 300, "tail_join_ms": 1000},
            "speaker_id": section | {"roles": True, "enrolled_follow_up_min_probability": 0.25},
        }
        path.write_text(json.dumps(valid))
        loaded = PrototypeConfig.load(path)
        self.assertTrue(loaded.voiceprint_roles)
        self.assertEqual(loaded.speaker_id.enrolled_follow_up_min_probability, 0.25)
        invalid = [
            base | {"speakers": {"owner": ["owner"]}, "speaker_id": section | {"roles": "yes"}},
            base
            | {
                "speakers": {"owner": ["owner"]},
                "speaker_id": section | {"live": False, "roles": True},
            },
            base | {"speaker_id": section | {"roles": True}},
            base | {"speakers": {"owner_only": True}, "speaker_id": section | {"roles": True}},
            base
            | {
                "speakers": {"owner": ["owner"]},
                "speaker_id": section | {"enrolled_follow_up_min_probability": 0.2},
            },
            base
            | {
                "speakers": {"owner": ["owner"]},
                "speaker_id": section | {"roles": True, "enrolled_follow_up_min_probability": 0},
            },
        ]
        for raw in invalid:
            path.write_text(json.dumps(raw))
            with self.subTest(raw=raw["speaker_id"]), self.assertRaises(PrototypeError):
                PrototypeConfig.load(path)
        self.assertFalse(SpeakerIdConfig.from_dict(section).roles)

    def test_session_label_roles_still_refuse_inferred_labels(self):
        for turns in ({"edge_attribution_ms": 300}, {"tail_join_ms": 1000}):
            config = replace(
                self.base,
                speakers=SpeakerPriority(owners=("Speaker A",)),
                speaker_id=self.identification(),
                **turns,
            )
            controller = PrototypeController(config)
            self.addCleanup(controller.close)
            with self.subTest(turns=turns), self.assertRaises(PrototypeError):
                controller.start({"mode": "demo"})

    def test_identification_roles_refuse_utterance_local_labels(self):
        config = replace(
            self.base,
            speakers=OWNER_ID,
            speaker_id=self.identification(roles=True),
            diarizer={"kind": "hosted-deepgram", "model": "nova-3"},
        )
        controller = PrototypeController(config, allow_hosted_speech=True)
        self.addCleanup(controller.close)
        with self.assertRaises(PrototypeError) as error:
            controller.start({"mode": "demo"})
        self.assertIn("session-stable", str(error.exception))

    def test_a_real_identifier_binds_from_scripted_vectors(self):
        """End to end with `ShadowSpeakerId`: synthetic vectors bind A, never B."""
        embedder = ScriptedEmbedder([unit(1, 0.1, 0), unit(0, 0, 1), unit(1, 0.1, 0)])
        created = []

        def factory(settings, *, report):
            created.append(
                ShadowSpeakerId(
                    replace(settings, bind_min_seconds=1.0),
                    report=report,
                    embedder_factory=lambda _settings: embedder,
                    entries=lambda _settings: [dict(entry) for entry in ENTRIES],
                )
            )
            return created[0]

        lines = []
        config = replace(self.base, speakers=OWNER_ID, speaker_id=self.identification(roles=True))
        self.session(config, factory, report=lines.append)
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and created[0].binding("Speaker A") is None:
            time.sleep(0.01)
        self.assertEqual(created[0].binding("Speaker A")[0], "bound")
        self.assertEqual(created[0].binding("Speaker B")[0], "unknown")
        self.assertFalse(any("Some words" in line for line in lines))
        # The third (inferred) turn was counted, not embedded or accumulated.
        self.assertEqual((created[0].scored, created[0].inferred), (2, 1))
        self.assertEqual(embedder.calls, 2)
        self.assertEqual(created[0]._labels["Speaker A"].turns, 1)
        summary = [line for line in lines if line.startswith("speaker_id summary")]
        self.assertEqual(len(summary), 1)
        self.assertIn(" inferred=1", summary[0])

    def test_shadow_mode_also_skips_inferred_turns(self):
        """Without roles too: inferred audio is never voiceprint evidence."""
        embedder = ScriptedEmbedder([unit(1, 0.1, 0), unit(0, 0, 1)])
        created = []

        def factory(settings, *, report):
            created.append(
                ShadowSpeakerId(
                    settings,
                    report=report,
                    embedder_factory=lambda _settings: embedder,
                    entries=lambda _settings: [dict(entry) for entry in ENTRIES],
                )
            )
            return created[0]

        lines = []
        self.session(
            replace(self.base, speaker_id=self.identification()), factory, report=lines.append
        )
        self.assertEqual((created[0].offered, created[0].inferred), (3, 1))
        self.assertEqual(embedder.calls, 2)


if __name__ == "__main__":
    unittest.main()
