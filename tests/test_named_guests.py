"""Named guests (hailing-station #434, slice 2): a bound guest's display name on transcripts.

Synthetic data only: authored turns, stand-in bindings or scripted embedder vectors, and
authored PCM. No model, microphone, recorded voice or enrollment store is used.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from test_enrolled_roles import (
    LOCAL,
    OWNER_ID,
    SESSION,
    Bindings,
    ControllerTests,
    FakeShadow,
    HaildModel,
    decided,
    turn,
)
from test_live_speaker_id import ENTRIES, ScriptedEmbedder, settings, tone, unit

from rightyo.contracts import SpeakerPriority
from rightyo.live_speaker_id import EnrolledRoles, ShadowSpeakerId
from rightyo.prototype import PrototypeConfig, PrototypeError
from rightyo.speaker_id import SpeakerIdConfig
from rightyo.tool_events import SpeechEvents, speaker_name

GUEST_NAME = "Ben Quillfeather"
NAMES = {"owner": "Mick Owner", "guest": GUEST_NAME, "jev": "Jev Trusted"}


class NamedBindings(Bindings):
    """Stand-in bindings that also know enrolled display names."""

    def __init__(self, names=None, **labels):
        super().__init__(**labels)
        self.names = dict(NAMES if names is None else names)

    def display_name(self, identity):
        return self.names.get(identity)


class NamedFakeShadow(FakeShadow):
    def display_name(self, identity):
        return NAMES.get(identity)


class SpeakerNameRuleTests(unittest.TestCase):
    def test_valid_names_pass_unchanged(self):
        for name in ("Ben", "Zoë", "Ana María", "李雷", "A" * 64, "Ben 👋🏽", "O'Brien-Smith"):
            with self.subTest(name=name):
                self.assertEqual(speaker_name(name), name)

    def test_invalid_names_are_refused(self):
        invalid = [
            None,
            "",
            42,
            ["Ben"],
            "A" * 65,
            " Ben",
            "Ben ",
            " Ben",
            "Ben\n",
            "Be\nn",
            "Be\tn",
            "Be\x00n",
            "Be\x7fn",
            "Be​n",  # zero-width space (Cf)
            "Be‎n",  # left-to-right mark (Cf)
            "Be‮n",  # right-to-left override (Cf)
            "Be n",  # line separator (Zl)
            "Be n",  # paragraph separator (Zp)
            "Ben",  # private use (Co)
            "Be\ud800n",  # lone surrogate (Cs)
            "Be\U000e0041n",  # tag character (Cf)
        ]
        for name in invalid:
            with self.subTest(name=repr(name)):
                self.assertIsNone(speaker_name(name))


class NameForTests(unittest.TestCase):
    def roles(self, **labels):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",), trusted=("jev",)), emit_names=True)
        roles.source = NamedBindings(**labels)
        return roles

    def test_only_a_bound_non_owner_identity_is_named(self):
        a = turn("a", 0, 1000, "hi")
        cases = {
            ("bound", "guest"): GUEST_NAME,
            ("bound", "jev"): "Jev Trusted",
            ("bound", "owner"): None,
            ("tentative", "guest"): None,
            ("unknown", "guest"): None,
            ("insufficient", "guest"): None,
        }
        for binding, expected in cases.items():
            roles = self.roles(Speaker_A=binding)
            with self.subTest(binding=binding):
                self.assertEqual(roles.name_for(a, roles.role_for(a)), expected)

    def test_overlap_provenance_labels_and_role_mismatch_get_no_name(self):
        roles = self.roles(Speaker_A=("bound", "guest"))
        self.assertIsNone(roles.name_for(turn("a", 0, 1, "x", overlap=True), "participant"))
        self.assertIsNone(roles.name_for(turn("b", 0, 1, "x", None), "participant"))
        local = replace(turn("c", 0, 1, "x"), speaker_provenance="unknown")
        self.assertIsNone(roles.name_for(local, "participant"))
        # A binding that changed since the role was resolved gives no name.
        self.assertIsNone(roles.name_for(turn("d", 0, 1, "x"), "owner"))
        self.assertIsNone(roles.name_for(turn("d", 0, 1, "x"), "trusted"))
        self.assertIsNone(roles.name_for(turn("d", 0, 1, "x"), None))

    def test_off_by_default_and_session_label_roles_are_never_named(self):
        roles = EnrolledRoles(OWNER_ID)
        roles.source = NamedBindings(Speaker_A=("bound", "guest"))
        a = turn("a", 0, 1, "x")
        self.assertIsNone(roles.name_for(a, roles.role_for(a)))
        configured = EnrolledRoles(
            SpeakerPriority(owners=("owner",), trusted=("Speaker A",)), emit_names=True
        )
        configured.source = NamedBindings(Speaker_A=("bound", "guest"))
        self.assertIsNone(configured.name_for(a, configured.role_for(a)))

    def test_a_source_without_names_or_a_failing_one_gives_none(self):
        roles = EnrolledRoles(OWNER_ID, emit_names=True)
        roles.source = Bindings(Speaker_A=("bound", "guest"))
        a = turn("a", 0, 1, "x")
        self.assertIsNone(roles.name_for(a, "participant"))
        failing = NamedBindings(Speaker_A=("bound", "guest"))
        failing.display_name = None
        failing.binding = lambda _label: (_ for _ in ()).throw(RuntimeError("boom"))
        roles.source = failing
        self.assertIsNone(roles.name_for(a, "participant"))


class NamedEventTests(unittest.TestCase):
    def setUp(self):
        self.lines = []
        self.events = SpeechEvents(report=self.lines.append)
        self.bindings = NamedBindings()
        self.stream = []

    def start(self, *, emit_names=True):
        roles = EnrolledRoles(OWNER_ID, emit_names=emit_names)
        roles.source = self.bindings
        self.events.start(SESSION, now_ms=0, priority=roles)
        self.stream += self.events.drain()
        return self.stream[0]

    def say(self, current, label="ignore", *, inferred=False):
        self.events.transcript(current, current.end_ms, inferred=inferred)
        self.events.decision(
            decided(current, label, recipient="system" if label == "attend" else "other_human"),
            current.end_ms + 100,
        )
        out = self.events.drain()
        self.stream += out
        return [event for event in out if event["type"] == "transcript"][0]

    def finish(self):
        self.events.end("stopped", 100000)
        self.stream += self.events.drain()
        HaildModel(self).check(self.stream)
        self.assertFalse(any(GUEST_NAME in line for line in self.lines))

    def test_a_bound_guest_is_named_and_others_are_not(self):
        started = self.start()
        self.assertEqual(started["speaker_names"], {"max_scalars": 64})
        self.assertNotIn("speaker_names", started["capabilities"])
        self.bindings.labels.update(
            {
                "Speaker A": ("bound", "owner"),
                "Speaker B": ("bound", "guest"),
                "Speaker C": ("tentative", "guest"),
                "Speaker D": ("unknown", "guest"),
            }
        )
        guest = self.say(turn("u1", 0, 2000, "Haili, what time is it?", "Speaker B"), "attend")
        self.assertEqual(guest["speaker_name"], GUEST_NAME)
        self.assertEqual(guest["turn"]["role"], "participant")
        self.assertNotIn("speaker_name", guest["turn"])
        # The request repeats the transcript turn unchanged: the name is never inside it.
        request = [e for e in self.stream if e["type"] == "request"][0]
        self.assertNotIn(GUEST_NAME, json.dumps(request))
        attention = [e for e in self.stream if e["type"] == "attention"][0]
        self.assertNotIn(GUEST_NAME, json.dumps(attention))
        unnamed = [
            turn("u2", 3000, 4000, "Me.", "Speaker A"),
            turn("u3", 5000, 6000, "Maybe.", "Speaker C"),
            turn("u4", 7000, 8000, "Nobody.", "Speaker D"),
            turn("u5", 9000, 10000, "Together.", "Speaker B", overlap=True),
            turn("u6", 11000, 12000, "Unlabelled.", None),
            turn("u7", 13000, 14000, "Not yet scored.", "Speaker E"),
        ]
        for current in unnamed:
            with self.subTest(utterance=current.utterance_id):
                self.assertNotIn("speaker_name", self.say(current))
        # An inferred-label turn of the bound guest carries no name either.
        inferred = self.say(turn("u8", 15000, 16000, "Joined.", "Speaker B"), inferred=True)
        self.assertNotIn("speaker_name", inferred)
        self.finish()

    def test_off_emits_no_name_and_no_advertisement(self):
        started = self.start(emit_names=False)
        self.assertNotIn("speaker_names", started)
        self.bindings.labels["Speaker B"] = ("bound", "guest")
        self.assertNotIn("speaker_name", self.say(turn("u1", 0, 2000, "Hi.", "Speaker B")))
        self.finish()

    def test_invalid_names_are_omitted_not_fatal(self):
        self.start()
        self.bindings.labels["Speaker B"] = ("bound", "guest")
        bad = ["Be​n", " Ben", "Ben\n", "A" * 65, "", None, 7]
        for index, name in enumerate(bad):
            self.bindings.names["guest"] = name
            current = turn(f"u{index}", index * 2000, index * 2000 + 1000, "Hi.", "Speaker B")
            with self.subTest(name=repr(name)):
                event = self.say(current)
                self.assertNotIn("speaker_name", event)
                self.assertEqual(event["turn"]["role"], "participant")
        self.finish()

    def test_anonymous_and_configured_sessions_never_advertise(self):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0)
        self.assertNotIn("speaker_names", events.drain()[0])


class RealIdentifierTests(unittest.TestCase):
    def test_a_bound_guest_is_named_and_stderr_never_carries_the_name(self):
        entries = [dict(entry, display_name=NAMES.get(entry["id"], "Stale")) for entry in ENTRIES]
        embedder = ScriptedEmbedder([unit(0, 1, 0), unit(0, 1, 0)])
        lines = []
        shadow = ShadowSpeakerId(
            settings(bind_min_seconds=1.0),
            report=lines.append,
            embedder_factory=lambda _settings: embedder,
            entries=lambda _settings: [dict(entry) for entry in entries],
        )
        shadow.start()
        roles = EnrolledRoles(OWNER_ID, emit_names=True)
        roles.source = shadow
        events = SpeechEvents(report=lines.append)
        events.start(SESSION, now_ms=0, priority=roles)
        first = turn("u1", 0, 2000, "Hello.", "Speaker B")
        shadow.audio(tone(2.0))
        events.transcript(first, first.end_ms, expect_decision=False)
        shadow.turn(first, spans=((0, 2000),))
        shadow.close(drain=True)
        self.assertEqual(shadow.binding("Speaker B"), ("bound", "guest"))
        self.assertEqual(shadow.display_name("guest"), GUEST_NAME)
        self.assertIsNone(shadow.display_name("stale"))  # another model's voiceprint
        second = turn("u2", 3000, 5000, "Me again.", "Speaker B")
        events.transcript(second, second.end_ms, expect_decision=False)
        events.end("stopped", 10000)
        stream = events.drain()
        transcripts = [event for event in stream if event["type"] == "transcript"]
        self.assertNotIn("speaker_name", transcripts[0])  # emitted before the binding
        self.assertEqual(transcripts[1]["speaker_name"], GUEST_NAME)
        self.assertTrue(any("state=bound id=guest" in line for line in lines))
        for line in lines:
            self.assertNotIn("Ben", line)
            self.assertNotIn("Mick", line)


class NamedControllerTests(unittest.TestCase):
    # The enrolled-roles controller harness, borrowed without inheriting its tests.
    setUp = ControllerTests.setUp
    identification = ControllerTests.identification
    session = ControllerTests.session

    def test_emit_names_config_names_the_bound_guest_only(self):
        lines = []

        def factory(settings, *, report):
            return NamedFakeShadow(NamedBindings())

        config = replace(
            self.base,
            speakers=OWNER_ID,
            speaker_id=self.identification(roles=True, emit_names=True),
        )
        events = self.session(config, factory, report=lines.append)
        self.assertEqual(events[0]["speaker_names"], {"max_scalars": 64})
        named = {
            event["turn"]["utterance_id"]: event.get("speaker_name")
            for event in events
            if event["type"] == "transcript"
        }
        # TwoSpeakerTurns: A, B, A. The fake binds A to the owner after its first turn and
        # B (matching no voiceprint) to `unknown`, so no turn is a bound guest.
        self.assertEqual(set(named.values()), {None})
        self.assertFalse(any("Mick" in line for line in lines))
        HaildModel(self).check(events)

    def test_load_accepts_emit_names_only_with_roles(self):
        base = {key: str(self.asset) for key in LOCAL} | {"demo_audio": str(self.demo)}
        section = {"python": str(self.asset), "model": str(self.asset), "live": True}
        path = self.root / "config.json"
        path.write_text(
            json.dumps(
                base
                | {
                    "speakers": {"owner": ["owner"]},
                    "speaker_id": section | {"roles": True, "emit_names": True},
                }
            )
        )
        self.assertTrue(PrototypeConfig.load(path).speaker_id.emit_names)
        self.assertFalse(SpeakerIdConfig.from_dict(section | {"roles": True}).emit_names)
        for extra in ({"emit_names": True}, {"roles": True, "emit_names": "yes"}):
            raw = base | {"speakers": {"owner": ["owner"]}, "speaker_id": section | extra}
            path.write_text(json.dumps(raw))
            with self.subTest(extra=extra), self.assertRaises(PrototypeError):
                PrototypeConfig.load(path)


if __name__ == "__main__":
    unittest.main()
