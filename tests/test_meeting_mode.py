"""hailing-station#735 meeting mode (#227): synthetic text, no capture or model calls."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_conversation import MODE, SESSION, decided, turn

from rightyo.contracts import Addressing, ContractError, SpeakerPriority
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents

OURS = Addressing(names=("Jarvis",), variants=(("Jarvis", ("Travis",)),))
OWNER = SpeakerPriority(owners=("Speaker A",))
ASK = "so what do you think, Jarvis?"


class MeetingModeTests(unittest.TestCase):
    def start(self, *, on=True, priority=OWNER, addressing=OURS, **options):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.events.meeting(on)
        self.events.start(
            SESSION,
            now_ms=0,
            priority=(
                ConfiguredPriorityProvider(priority)
                if isinstance(priority, SpeakerPriority)
                else priority
            ),
            addressing=addressing,
            conversation=MODE,
            **options,
        )
        return self.events.drain()[0]

    def say(
        self,
        text=ASK,
        label="uncertain",
        *,
        speaker="Speaker A",
        inferred=False,
        provenance="authored-fixture",
        **opts,
    ):
        current = turn("t1", 0, 900, text, speaker, provenance=provenance)
        self.events.transcript(current, current.end_ms, inferred=inferred)
        self.events.decision(decided(current, label, **opts), current.end_ms + 100)
        return self.events.drain()

    def assert_no_request(self, events, label="uncertain"):
        self.assertEqual([e["type"] for e in events], ["transcript", "attention"])
        self.assertEqual(events[1]["decision"]["label"], label)
        self.assertNotEqual(events[1]["decision"].get("meeting"), "ask_anywhere")

    def test_owner_named_anywhere_promotes_only_uncertain_and_sets_system_recipient(self):
        for name in ("Jarvis", "Travis"):
            for recipient in ("system", "unknown"):
                with self.subTest(name=name, recipient=recipient):
                    self.start()
                    events = self.say(ASK.replace("Jarvis", name), recipient=recipient)
                    self.assertEqual(
                        [e["type"] for e in events],
                        ["transcript", "attention", "request", "conversation"],
                    )
                    evidence = events[1]["decision"]
                    self.assertEqual(evidence["label"], "attend")
                    self.assertEqual(evidence["recipient_kind"], "system")
                    self.assertEqual(evidence["meeting"], "ask_anywhere")
                    self.assertEqual(events[2]["decision"], evidence)

    def test_ignore_is_never_promoted(self):
        for text in (ASK, "I told Jarvis about it."):
            with self.subTest(text=text):
                self.start()
                self.assert_no_request(self.say(text, "ignore"), "ignore")

    def test_guest_never_promotes_or_attends(self):
        for label in ("uncertain", "attend"):
            with self.subTest(label=label):
                self.start()
                self.assert_no_request(self.say(label=label, speaker="Speaker B"), "ignore")

    def test_unnamed_owner_never_promotes_or_attends(self):
        for label in ("uncertain", "attend"):
            with self.subTest(label=label):
                self.start()
                self.assert_no_request(self.say("What do you think?", label), "ignore")

    def test_inferred_owner_never_promotes_or_attends(self):
        for label in ("uncertain", "attend"):
            with self.subTest(label=label):
                self.start()
                self.assert_no_request(
                    self.say("Jarvis, what do you think?", label, inferred=True),
                    "ignore",
                )

    def test_another_configured_agent_never_promotes(self):
        self.start(other_agents=Addressing(names=("Dot",)))
        self.assert_no_request(self.say("so what do you think, Dot?"), "ignore")

    def test_start_named_live_binding_composes_with_251_and_inferred_authority_stays_closed(self):
        # Stand-in bindings only: #251 promotes start-named live owners even with inferred
        # words in normal mode. Meeting's bar must refuse their authority first.
        for inferred in (False, True):
            with self.subTest(inferred=inferred):
                roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
                roles.source = SimpleNamespace(binding=lambda label: ("bound", "owner"))
                self.start(priority=roles, duck=True)
                events = self.say(
                    "Jarvis, what do you think?",
                    inferred=inferred,
                    provenance="diarization-timeline",
                )
                self.assertEqual(sum(e["type"] == "request" for e in events), int(not inferred))
                if not inferred:
                    evidence = next(e["decision"] for e in events if e["type"] == "request")
                    self.assertEqual(evidence["recipient_kind"], "system")
                    self.assertEqual(evidence["meeting"], "ask_anywhere")
                self.assertNotIn("attention_promoted reason=named_owner", self.notes)

    def test_ask_anywhere_false_leaves_the_owner_and_name_bar(self):
        self.start(meeting_ask_anywhere=False)
        self.assert_no_request(self.say())
        self.start(meeting_ask_anywhere=False)
        self.assertTrue(any(e["type"] == "request" for e in self.say(label="attend")))

    def test_without_roles_or_names_no_promotion(self):
        for options in ({"priority": None}, {"addressing": None}):
            with self.subTest(options=options):
                self.start(**options)
                self.assert_no_request(self.say(), "ignore")

    def test_named_attends_without_roles_fail_closed_even_for_inferred_or_unlabelled_turns(self):
        for speaker in ("Speaker A", "Speaker B", None):
            for inferred in (False, True):
                with self.subTest(speaker=speaker, inferred=inferred):
                    self.start(priority=None)
                    events = self.say(
                        "Jarvis, open the synthetic ledger.",
                        "attend",
                        speaker=speaker,
                        inferred=inferred,
                    )
                    self.assert_no_request(events, "ignore")
                    self.assertIs(events[1]["decision"]["meeting"], True)

    def test_unlabelled_named_turns_with_roles_never_promote_or_attend(self):
        for label in ("uncertain", "attend"):
            with self.subTest(label=label):
                self.start()
                self.assert_no_request(self.say(label=label, speaker=None), "ignore")

    def test_pending_public_uncertain_is_not_promoted_after_privacy_entry(self):
        self.start(duck=True)
        current = turn("t1", 0, 900, ASK)
        self.events.transcript(current, 900)
        self.events.duck(True, now_ms=1000)
        self.events.decision(decided(current, "uncertain"), 1100)
        events = self.events.drain()
        self.assertTrue(self.events.ducked)
        self.assert_no_request(events)
        self.assertNotIn("meeting", events[1]["decision"])

    def test_named_owner_turn_heard_entirely_in_privacy_is_dropped_with_meeting_on(self):
        for label in ("uncertain", "attend"):
            with self.subTest(label=label):
                self.start(duck=True)
                self.events.duck(True, now_ms=1000)
                current = turn("t1", 1100, 1900, ASK)
                self.events.transcript(current, 1900)
                self.events.decision(decided(current, label), 2000)
                self.assertTrue(self.events.ducked)
                self.assertEqual(self.events.drain(), [])
                self.assertEqual(self.events._memory.retained_ids, set())
                self.assertEqual(self.events._pending, {})

    def test_follow_up_is_refused_with_meeting_token_even_with_media_on(self):
        for media in (False, True):
            with self.subTest(media=media):
                self.start()
                self.events.media(media)
                self.say(label="attend")
                current = turn("t2", 2000, 2900, "And tomorrow?")
                self.events.transcript(current, current.end_ms)
                self.events.decision(decided(current, "uncertain", attend=0.9), 3000)
                self.assert_no_request(self.events.drain(), "ignore")
                self.assertIn("follow_up outcome=refused reason=meeting", self.notes[-1])

    def test_transcripts_and_context_are_retained(self):
        self.start()
        first = self.say("Synthetic meeting detail.", "ignore")
        self.assertEqual(first[0]["turn"]["text"], "Synthetic meeting detail.")
        current = turn("t2", 2000, 2900, ASK)
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, "uncertain"), 3000)
        request = next(e for e in self.events.drain() if e["type"] == "request")
        self.assertEqual(request["context"]["turns"][0]["text"], "Synthetic meeting detail.")
        self.assertEqual(self.events._memory.retained_ids, {"t1", "t2"})

    def test_snapshot_in_both_directions_for_gate_and_promotion(self):
        for on in (False, True):
            for text, label in (("What time is it?", "attend"), (ASK, "uncertain")):
                with self.subTest(on=on, label=label):
                    self.start(on=on)
                    current = turn("t1", 0, 900, text)
                    self.events.transcript(current, 900)
                    self.events.meeting(not on)
                    self.events.decision(decided(current, label), 1000)
                    events = self.events.drain()
                    expected = on if label == "uncertain" else not on
                    self.assertEqual(any(e["type"] == "request" for e in events), expected)

    def test_snapshot_prunes_and_clears_on_new_session_but_mode_persists(self):
        self.start()
        self.say()
        self.events.expire(301000)
        self.assertEqual(self.events._meeting_turns, {})
        self.events.end(phase="stopped", now_ms=301000)
        self.events.drain()
        self.events._meeting_turns["old"] = 301000
        self.events.start("next-session", now_ms=0)
        self.assertEqual(self.events._meeting_turns, {})
        self.assertIs(self.events._meeting, True)

    def test_normal_and_media_mode_do_not_promote_anywhere(self):
        for media in (False, True):
            with self.subTest(media=media):
                self.start(on=False)
                self.events.media(media)
                self.assert_no_request(self.say())

    def test_media_and_meeting_together_are_the_stricter_of_the_two(self):
        # Meeting's bar is media's bar plus owner authority, so with both on every turn is
        # decided exactly as under meeting alone, and nothing attends past media's bar.
        cases = (
            ("owner named abstention", ASK, "uncertain", {}),
            ("owner named attend", ASK, "attend", {}),
            ("owner unnamed attend", "What time is it?", "attend", {}),
            ("guest named attend", ASK, "attend", {"speaker": "Speaker B"}),
            ("inferred owner named attend", ASK, "attend", {"inferred": True}),
            ("unlabelled named attend", ASK, "attend", {"speaker": None}),
            ("no roles named attend", ASK, "attend", {"priority": None}),
            ("no roles named abstention", ASK, "uncertain", {"priority": None}),
        )
        for name, text, label, options in cases:
            priority = options.pop("priority", OWNER)
            outcomes = {}
            for meeting, media in ((True, False), (False, True), (True, True)):
                with self.subTest(case=name, meeting=meeting, media=media):
                    self.start(on=meeting, priority=priority)
                    self.events.media(media)
                    events = self.say(text, label, **options)
                    evidence = events[1]["decision"]
                    outcomes[meeting, media] = (
                        any(e["type"] == "request" for e in events),
                        evidence["label"],
                        evidence.get("meeting"),
                        evidence.get("media"),
                    )
            with self.subTest(case=name):
                # Identical to meeting alone: meeting's refusal, never a separate media one.
                self.assertEqual(outcomes[True, True], outcomes[True, False])
                if outcomes[True, True][0] and label == "attend":
                    # A provider-labeled attend that passes both passes media alone too.
                    self.assertTrue(outcomes[False, True][0])

    def test_meeting_bar_implies_media_bar(self):
        self.start()
        for text in (ASK, "What time is it?"):
            for authority in ("owner", "trusted", "unknown", None):
                for priority in (None, OWNER):
                    with self.subTest(text=text, authority=authority, priority=priority):
                        self.events._priority = (
                            None if priority is None else ConfiguredPriorityProvider(priority)
                        )
                        current = turn("t1", 0, 900, text)
                        meeting = self.events._meeting_attends(current, authority)
                        self.assertEqual(meeting, authority == "owner" and "Jarvis" in text)
                        if meeting:
                            self.assertTrue(self.events._media_attends(current, authority))

    def test_named_owner_promotion_never_bypasses_the_meeting_bar(self):
        # #251 promotes a start-named abstention by the live voiceprint-bound owner, even
        # with inferred words or a split name. With meeting on, its turns face meeting's
        # bar first and the promotion re-checks it, so none of these turns attends.
        split = Addressing(names=("RightyO",))
        cases = (
            # (case, text, speaker, inferred, addressing, roles, promoted with meeting off)
            ("inferred owner", "Jarvis, what do you think?", "Speaker A", True, OURS, True, True),
            ("split name", "Righty O, what do you think?", "Speaker A", False, split, True, True),
            ("guest", "Jarvis, what do you think?", "Speaker B", False, OURS, True, False),
            ("unlabelled", "Jarvis, what do you think?", None, False, OURS, True, False),
            ("no roles", "Jarvis, what do you think?", "Speaker A", False, OURS, False, False),
        )

        def live_roles():
            roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
            roles.source = SimpleNamespace(
                binding=lambda label: ("bound", "owner" if label == "Speaker A" else "guest")
            )
            return roles

        for case, text, speaker, inferred, addressing, roles, promoted in cases:
            for on in (False, True):
                for ask_anywhere in (False, True):
                    with self.subTest(case=case, meeting=on, ask_anywhere=ask_anywhere):
                        self.start(
                            on=on,
                            priority=live_roles() if roles else None,
                            addressing=addressing,
                            duck=True,
                            meeting_ask_anywhere=ask_anywhere,
                        )
                        events = self.say(
                            text,
                            speaker=speaker,
                            inferred=inferred,
                            provenance="diarization-timeline",
                        )
                        named = "attention_promoted reason=named_owner" in self.notes
                        if not on:
                            # Control: #251 alone promotes exactly the owner's turns.
                            self.assertIs(named, promoted)
                            continue
                        self.assert_no_request(events, "ignore")
                        self.assertIs(events[1]["decision"]["meeting"], True)
                        self.assertNotIn("media", events[1]["decision"])
                        self.assertFalse(named)

    def test_named_owner_promotion_still_attends_when_meeting_bar_passes(self):
        # With ask-anywhere off, #251's start-of-turn promotion is the only one, and it
        # attends the bound owner's named abstention under meeting, with or without media.
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        roles.source = SimpleNamespace(binding=lambda label: ("bound", "owner"))
        for media in (False, True):
            with self.subTest(media=media):
                self.start(priority=roles, duck=True, meeting_ask_anywhere=False)
                self.events.media(media)
                events = self.say("Jarvis, what do you think?", provenance="diarization-timeline")
                request = next(e for e in events if e["type"] == "request")
                self.assertEqual(request["decision"]["label"], "attend")
                self.assertNotIn("meeting", request["decision"])
                self.assertNotIn("media", request["decision"])
                self.assertEqual(self.notes.count("attention_promoted reason=named_owner"), 1)

    def test_media_and_meeting_together_promote_and_keep_the_bar(self):
        self.start()
        self.events.media(True)
        self.assertTrue(any(e["type"] == "request" for e in self.say()))
        self.start()
        self.events.media(True)
        self.assert_no_request(self.say("What time is it?", "attend"), "ignore")

    def test_unavailable_never_promotes(self):
        self.start()
        current = turn("t1", 0, 900, ASK)
        self.events.transcript(current, 900)
        self.events.decision(decided(current, "uncertain"), 1000, unavailable="timeout")
        self.assert_no_request(self.events.drain())

    def test_advertised_only_with_host_control_beside_capabilities(self):
        for control in (False, True):
            for duck in (False, True):
                with self.subTest(control=control, duck=duck):
                    started = self.start(meeting=control, duck=duck)
                    self.assertEqual(started.get("meeting"), {"version": 1} if control else None)
                    self.assertNotIn("meeting", started["capabilities"])

    def test_control_is_silent_strict_and_forwarded_without_a_session(self):
        events = SpeechEvents()
        controller = SimpleNamespace(_lock=threading.RLock(), _events=events)
        for on in (True, False):
            PrototypeController.meeting(controller, on)
            self.assertIs(events._meeting, on)
            self.assertEqual(events.drain(), [])
        PrototypeController.meeting(SimpleNamespace(_lock=threading.RLock(), _events=None), True)
        for value in (1, "on", None):
            with self.assertRaises(PrototypeError):
                PrototypeController.meeting(controller, value)
            with self.assertRaises(ContractError):
                events.meeting(value)
            with self.assertRaises(ContractError):
                events.start(SESSION, meeting_ask_anywhere=value)
            with self.assertRaises(ContractError):
                events.start(SESSION, meeting=value)
            with self.assertRaises(ContractError):
                SpeechEvents(meeting_control=value)


class MeetingConfigTests(unittest.TestCase):
    def test_defaults_booleans_and_strict_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            asset = Path(directory) / "asset"
            asset.touch()
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
            for section, expected in (({}, True), ({"meeting": {}}, True)):
                config.write_text(json.dumps(base | section))
                self.assertIs(PrototypeConfig.load(config).meeting_ask_anywhere, expected)
            for on in (True, False):
                config.write_text(json.dumps(base | {"meeting": {"ask_anywhere": on}}))
                self.assertIs(PrototypeConfig.load(config).meeting_ask_anywhere, on)
            for invalid in (
                None,
                True,
                [],
                {"ask_anywhere": 1},
                {"ask_anywhere": "yes"},
                {"ask_anywhere": None},
                {"unknown": True},
            ):
                with self.subTest(invalid=invalid):
                    config.write_text(json.dumps(base | {"meeting": invalid}))
                    with self.assertRaises(PrototypeError):
                        PrototypeConfig.load(config)


if __name__ == "__main__":
    unittest.main()
