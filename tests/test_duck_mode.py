"""Privacy mode, internally duck v2 (hailing-station #735 S-1): while it is on, rightyo keeps
and emits nothing heard, and matches the text only locally for the owner's named exit and
anyone's status question. Synthetic text only."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_conversation import MODE, SESSION, decided, turn

from rightyo import tool_events
from rightyo.contracts import Addressing, ContractError, Dismissal, SpeakerPriority
from rightyo.live_audio import (
    MAX_RESPONSE_BYTES,
    READ_CHUNK_BYTES,
    LiveAudioError,
    LiveConfig,
    RecognizerSandboxError,
    WhisperCppTranscriber,
    _transcribe,
    recognizer_dir,
)
from rightyo.live_speaker_id import EnrolledRoles, PcmRing
from rightyo.memory import TranscriptMemory
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import MAX_DUCK_SPANS, SpeechEvents, encode_json, privacy_phrase
from rightyo.turn_merge import TurnMerger

OURS = Addressing(names=("Jarvis",), variants=(("Jarvis", ("Travis",)),))
OWNER = SpeakerPriority(owners=("Speaker A",))
# With live voiceprint roles, owners are enrolled identities, not diarizer labels.
OWNER_ID = SpeakerPriority(owners=("mick",))
# Synthetic words that never occur in an event's keys or fixed values.
MONOLOGUE = (
    "Marmalade zeppelin quokka, the saxophone budget is turquoise.",
    "Jarvis, the fjord ledger needs a narwhal spreadsheet.",
    "What do you think about the gazebo, Jarvis?",
    "Stop the walrus memo and play the kumquat draft back.",
)
SECRET_WORDS = (
    "marmalade",
    "zeppelin",
    "quokka",
    "saxophone",
    "turquoise",
    "fjord",
    "ledger",
    "narwhal",
    "spreadsheet",
    "gazebo",
    "walrus",
    "kumquat",
)
CONTENT = {"transcript", "attention", "request", "conversation", "acknowledgement"}


def leaks(blob):
    """The secret words found in `blob` (any case)."""
    lowered = blob.lower()
    return [word for word in SECRET_WORDS if word in lowered]


class RecordingProvider(ConfiguredPriorityProvider):
    """Configured roles that record every hosted role question asked."""

    def __init__(self, priority):
        super().__init__(priority)
        self.states = []

    def assign(self, state):
        self.states.append(state)
        return super().assign(state)


class Bindings:
    """A stand-in live identifier: label -> (state, identity)."""

    def __init__(self, labels):
        self.labels = labels

    def binding(self, label):
        return self.labels.get(label)


def enrolled():
    """Live voiceprint roles: Speaker A is bound to the enrolled owner, Speaker B to a
    guest's voice, anyone else is not yet identified."""
    roles = EnrolledRoles(OWNER_ID)
    roles.source = Bindings({"Speaker A": ("bound", "mick"), "Speaker B": ("unknown", None)})
    return roles


def heard(name, start, end, text, speaker="Speaker A"):
    """A turn with a diarization-timeline label, the only kind live identification binds."""
    return turn(name, start, end, text, speaker, provenance="diarization-timeline")


class PrivacyEventsTests(unittest.TestCase):
    def start(self, provider="enrolled", **options):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        if provider == "enrolled":
            provider = enrolled()
        self.events.start(
            SESSION,
            now_ms=0,
            addressing=OURS,
            conversation=MODE,
            priority=provider,
            duck=True,
            **options,
        )
        return self.events.drain()

    def say(self, current, label="attend", **options):
        self.events.transcript(current, current.end_ms)
        if current.utterance_id in self.events._pending:
            self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return self.events.drain()

    def hear(self, *texts, speaker="Speaker A", start=100, gap=1000):
        """Say each text as its own turn; return every event drained."""
        events = []
        for index, text in enumerate(texts):
            at = start + index * gap
            events += self.say(heard(f"p{at}", at, at + gap - 100, text, speaker))
        return events

    @staticmethod
    def types(events):
        return [e["type"] for e in events]

    def note(self, **counts):
        """The privacy-off note with these counts and zeros elsewhere."""
        keys = ("discarded", "exit", "exit_refused", "status", "failed")
        return "privacy off: " + " ".join(f"{key}={counts.get(key, 0)}" for key in keys)

    def test_started_advertises_version_2(self):
        (started,) = self.start()
        self.assertEqual(started["duck"], {"version": 2})
        # Beside the strictly validated capability set, never inside it.
        self.assertNotIn("duck", started["capabilities"])
        # A producer that takes no control (replay) does not advertise it.
        replay = SpeechEvents()
        replay.start("replay", now_ms=0)
        self.assertNotIn("duck", replay.drain()[0])
        with self.assertRaises(ContractError):
            SpeechEvents().start("bad", duck="yes")

    def test_a_private_monologue_emits_and_keeps_nothing(self):
        stream = self.start(dismissal=Dismissal())
        stderr, logs = io.StringIO(), io.StringIO()
        handler = logging.StreamHandler(logs)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stderr):
            self.events.duck(True, now_ms=0)
            events = []
            for index, speaker in enumerate(("Speaker A", "Speaker B", None)):
                events += self.hear(*MONOLOGUE * 3, speaker=speaker, start=index * 20_000)
            self.events.duck(False, now_ms=200_000)
            # And everything the session emits until it stops.
            self.events.end("stopped", 200_000)
            tail = self.events.drain()
        self.assertEqual(events, [])
        self.assertEqual(self.types(tail), ["session"])
        self.assertEqual(self.events._memory.snapshot(200_000)["turns"], [])
        self.assertEqual(self.events._pending, {})
        # No digest of the words, and none of the session's turn budget: ids only.
        self.assertEqual(self.events._seen, {})
        self.assertTrue(all(uid.startswith("p") for uid in self.events._private_seen))
        # Every byte the host could have read, as the JSONL stream carries it.
        everything = "".join(encode_json(e) + "\n" for e in [*stream, *events, *tail])
        self.assertEqual(leaks(everything), [])
        self.assertEqual(leaks(stderr.getvalue() + logs.getvalue()), [])
        self.assertEqual(leaks(" ".join(self.notes)), [])
        # Logs carry counts only.
        self.assertEqual(self.notes, [self.note(discarded=36)])

    def test_private_turns_never_spend_the_session_turn_budget(self):
        # Key 3's item 4: thousands of private turns never end the stream.
        self.start()
        self.events.duck(True, now_ms=0)
        for index in range(1500):
            self.events.transcript(heard(f"q{index}", index, index, "um"), index)
        self.events.duck(False, now_ms=2000)
        events = self.say(heard("t1", 3000, 3900, "Jarvis, what time is it?"))
        self.assertIn("request", self.types(events))

    def test_no_hosted_role_question_is_asked_for_a_private_turn(self):
        provider = RecordingProvider(OWNER)
        self.start(provider=provider)
        self.events.duck(True, now_ms=0)
        self.hear(*MONOLOGUE, speaker="Speaker B")
        self.assertEqual(provider.states, [])
        # Nor is a role fixed for the speaker from it.
        self.assertEqual(self.events._roles, {})

    def test_the_enrolled_owner_exit_emits_one_content_free_event(self):
        self.start()
        self.events.duck(True, now_ms=0)
        for index, text in enumerate(
            ("Jarvis, I'm back.", "Jarvis, start listening.", "Travis, privacy off!")
        ):
            with self.subTest(text=text):
                events = self.hear(text, start=index * 5000)
                self.assertEqual(len(events), 1)
                self.assertEqual(
                    {k: v for k, v in events[0].items() if k not in {"sequence", "emitted_at_ms"}},
                    {
                        "schema_version": 1,
                        "type": "privacy",
                        "session_id": SESSION,
                        "action": "exit",
                    },
                )

    def test_a_label_configured_owner_is_refused_and_counted(self):
        # Key 3's blocker 1: a TV diarized into "Speaker A" must never end privacy.
        for provider in (ConfiguredPriorityProvider(OWNER), EnrolledRoles(OWNER)):
            with self.subTest(provider=type(provider).__name__):
                self.start(provider=provider)
                self.events.duck(True, now_ms=0)
                self.assertEqual(self.hear("Jarvis, I'm back."), [])
                self.events.duck(False, now_ms=5000)
                self.assertEqual(self.notes, [self.note(discarded=1, exit_refused=1)])

    def test_a_guest_or_unnamed_exit_is_ignored(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.hear("Jarvis, I'm back.", speaker="Speaker B"), [])
        self.assertEqual(self.hear("Jarvis, I'm back.", speaker="Speaker C", start=5000), [])
        self.assertEqual(self.hear("Jarvis, I'm back.", speaker=None, start=10_000), [])
        self.assertEqual(self.hear("I'm back.", start=15_000), [])
        self.assertEqual(self.hear("Start listening.", start=20_000), [])
        self.events.duck(False, now_ms=30_000)
        self.assertEqual(self.notes, [self.note(discarded=5, exit_refused=3)])

    def test_without_roles_nobody_can_exit(self):
        self.start(provider=None)
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.hear("Jarvis, I'm back."), [])

    def test_an_inferred_owner_label_cannot_exit(self):
        self.start()
        self.events.duck(True, now_ms=0)
        current = heard("t1", 100, 900, "Jarvis, I'm back.")
        self.events.transcript(current, current.end_ms, inferred=True)
        self.assertEqual(self.events.drain(), [])

    def test_an_exit_heard_during_the_hosts_own_playback_is_dropped(self):
        # An echo of a clip such as "Privacy off, I'm back" is never the owner leaving.
        self.start()
        self.events.duck(True, now_ms=0)
        self.events.reply("started", 1000)
        self.assertEqual(self.hear("Jarvis, I'm back.", start=1200), [])
        self.events.reply("ended", 3000)
        # Within the margin after it ended: still dropped.
        self.assertEqual(self.say(heard("t1", 3200, 3900, "Jarvis, I'm back.")), [])
        events = self.say(heard("t2", 3400, 4100, "Jarvis, I'm back."))
        self.assertEqual([e["action"] for e in events], ["exit"])

    def test_loose_matching(self):
        self.start()
        self.events.duck(True, now_ms=0)
        for index, text in enumerate(
            (
                "Um, ok Jarvis, so, I'm back.",
                "That's enough of that. Jarvis I'm back",
                "JARVIS... PRIVACY OFF!!",
                "Hey Jarvis, please start listening",
                "Jarvis, I am back now",
                "Jarvis, privacy mode off.",
            )
        ):
            with self.subTest(text=text):
                events = self.hear(text, start=index * 5000)
                self.assertEqual(self.types(events), ["privacy"])
                self.assertEqual(events[0]["action"], "exit")

    def test_a_split_name_fragment_joins_the_exit(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.hear("Jarvis,"), [])
        events = self.hear("I'm back.", start=1500)
        self.assertEqual([e["action"] for e in events], ["exit"])
        # Too late after the name, or with a turn between: no exit.
        self.assertEqual(self.hear("Jarvis,", start=10_000), [])
        self.assertEqual(self.hear("I'm back.", start=20_000), [])
        self.assertEqual(self.hear("Jarvis,", start=30_000), [])
        self.assertEqual(self.hear("Something else.", start=31_000), [])
        self.assertEqual(self.hear("I'm back.", start=32_000), [])
        # A guest finishing the owner's name is still a guest.
        self.assertEqual(self.hear("Jarvis,", start=40_000), [])
        self.assertEqual(self.hear("I'm back.", speaker="Speaker B", start=41_000), [])

    def test_a_name_from_anyone_but_the_enrolled_owner_never_starts_a_split_exit(self):
        self.start()
        self.events.duck(True, now_ms=0)
        for index, speaker in enumerate(("Speaker B", "Speaker C", None)):
            with self.subTest(speaker=speaker):
                at = index * 10_000
                self.assertEqual(self.hear("Travis.", speaker=speaker, start=at), [])
                self.assertEqual(self.hear("I'm back.", start=at + 1000), [])

    def test_key_2_repro_a_stale_name_never_exits(self):
        # Key 2's blocker: on, a guest's name released after the off, a public turn, on
        # again, then the owner's unnamed "start listening".
        self.start()
        self.events.duck(True, now_ms=0)
        self.events.duck(False, now_ms=1000)
        self.assertEqual(self.say(heard("n1", 600, 1200, "Travis.", "Speaker B")), [])
        self.say(heard("g1", 1500, 1800, "A synthetic aside.", "Speaker B"))
        self.events.duck(True, now_ms=2000)
        events = self.say(heard("o1", 2100, 2900, "start listening to this song"))
        self.assertEqual(events, [])

    def test_an_owner_name_straddling_the_off_never_starts_a_split_exit(self):
        for release_after_next_on in (False, True):
            with self.subTest(release_after_next_on=release_after_next_on):
                self.start()
                self.events.duck(True, now_ms=0)
                self.events.duck(False, now_ms=1000)
                # The owner's own name, heard across the off and released late, before or
                # after privacy goes on again.
                name = heard("n1", 600, 1200, "Jarvis,")
                if not release_after_next_on:
                    self.assertEqual(self.say(name), [])
                self.events.duck(True, now_ms=1300)
                if release_after_next_on:
                    self.assertEqual(self.say(name), [])
                self.assertIsNone(self.events._private_name_end)
                self.assertEqual(self.say(heard("o1", 1400, 2000, "I'm back.")), [])

    def test_a_session_started_private_keeps_nothing_until_the_host_turns_it_off(self):
        # hailing-station #735 S-2b: `--start-private` is `{"duck": true}` at time 0.
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        self.events.duck(True, now_ms=0)
        self.events.start(
            SESSION, now_ms=0, addressing=OURS, conversation=MODE, priority=enrolled(), duck=True
        )
        (started,) = self.events.drain()
        self.assertEqual(started["duck"], {"version": 2, "private": True})
        self.assertEqual(self.events._duck_spans, [[0, None]])
        self.assertEqual(self.hear(*MONOLOGUE, start=0), [])
        self.events.duck(False, now_ms=5000)
        events = self.say(heard("t1", 6000, 6900, "Jarvis, what time is it?"))
        self.assertIn("request", self.types(events))
        (request,) = [e for e in events if e["type"] == "request"]
        self.assertEqual(request["context"]["turns"], [])
        # Not private at start: no `private` key.
        self.assertEqual(self.start()[0]["duck"], {"version": 2})

    def test_a_public_turn_between_ends_a_split_exit(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.say(heard("n1", 100, 600, "Jarvis,")), [])
        self.assertIsNotNone(self.events._private_name_end)
        self.events.duck(False, now_ms=700)
        self.say(heard("g1", 800, 900, "A synthetic aside.", "Speaker B"))
        self.events.duck(True, now_ms=1000)
        self.assertEqual(self.say(heard("o1", 1100, 1600, "I'm back.")), [])

    def test_the_off_clears_a_split_exit(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.say(heard("n1", 100, 600, "Jarvis,")), [])
        self.assertEqual(self.events._private_name_end, 600)
        self.events.duck(False, now_ms=700)
        self.assertIsNone(self.events._private_name_end)
        self.events.duck(True, now_ms=700)
        self.assertEqual(self.say(heard("o1", 800, 1400, "I'm back.")), [])

    def test_near_misses_and_negations_are_rejected(self):
        self.start()
        self.events.duck(True, now_ms=0)
        for index, text in enumerate(
            (
                "Jarvis, I'm backing up the files.",
                "Jarvis, I'm going back.",
                "Jarvis, don't start listening.",
                "Don't say Jarvis I'm back.",
                "Do not, Jarvis, start listening.",
                "Jarvis, start the listening test.",
                "Harvey, I'm back.",
                "Jarvis, privacy is off the table.",
                "Don't ask are you listening.",
                "What mode is it in?",
            )
        ):
            with self.subTest(text=text):
                self.assertEqual(self.hear(text, start=index * 5000), [])

    def test_status_works_for_any_speaker(self):
        self.start()
        self.events.duck(True, now_ms=0)
        for index, (text, speaker) in enumerate(
            (
                ("What mode am I in?", "Speaker A"),
                ("Um, are you recording?", "Speaker B"),
                ("Jarvis, are you listening", None),
                ("So what mode am I in, Jarvis?", "Speaker C"),
            )
        ):
            with self.subTest(text=text):
                events = self.hear(text, speaker=speaker, start=index * 5000)
                self.assertEqual(self.types(events), ["privacy"])
                self.assertEqual(events[0]["action"], "status")
                self.assertEqual(
                    set(events[0]),
                    {"schema_version", "type", "session_id", "sequence", "emitted_at_ms", "action"},
                )

    def test_stop_never_exits_and_does_nothing_in_privacy(self):
        # Mick: "In privacy mode, there's no recording, so stop shouldn't do anything."
        self.start(dismissal=Dismissal())
        self.say(heard("t0", 0, 900, "Jarvis, what time is it?"))
        self.events.duck(True, now_ms=1000)

        def state():
            events = self.events
            return copy.deepcopy(
                (
                    events._dismissed,
                    events._cooldowns,
                    events._delivered,
                    events._withdrawn,
                    events._engaged,
                    events.last_request,
                    events._private_name_end,
                    events._memory.snapshot(100_000),
                )
            )

        before = state()
        forms = ("Stop.", "Stop that.", "Stop listening.", "Stop recording.", "Stop it.")
        for index, form in enumerate(forms):
            for text in (form, f"Jarvis, {form[0].lower()}{form[1:]}"):
                with self.subTest(text=text):
                    events = self.hear(text, start=2000 + index * 10_000)
                    self.assertEqual(events, [])
                    self.assertEqual(state(), before)
                    self.assertTrue(self.events.ducked)

    def test_a_matcher_failure_leaks_no_words(self):
        self.start()
        self.events.duck(True, now_ms=0)

        def broken(text, _addressing):
            raise ValueError(f"cannot match {text!r}")

        stderr, logs = io.StringIO(), io.StringIO()
        handler = logging.StreamHandler(logs)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        with (
            mock.patch.object(tool_events, "privacy_phrase", broken),
            contextlib.redirect_stderr(stderr),
        ):
            events = self.hear(*MONOLOGUE)
        self.events.duck(False, now_ms=100_000)
        self.assertEqual(events, [])
        self.assertEqual(leaks(stderr.getvalue() + logs.getvalue() + " ".join(self.notes)), [])
        self.assertEqual(self.notes, [self.note(discarded=4, failed=4)])

    def test_role_source_marks_only_a_live_voiceprint_owner(self):
        # The host's owner-only privacy entry (HS #807): "voiceprint" or absent, nothing else.
        (started,) = self.start()
        self.assertNotIn("role_source", started)
        sources = {}
        for name, speaker, inferred in (
            ("owner", "Speaker A", False),
            ("inferred", "Speaker A", True),
            ("guest", "Speaker B", False),
            ("unbound", "Speaker C", False),
            ("unlabelled", None, False),
        ):
            at = len(sources) * 1000
            self.events.transcript(
                heard(name, at, at + 900, "x", speaker), at + 900, inferred=inferred
            )
            (event, *_rest) = self.events.drain()
            sources[name] = (event["turn"]["role"], event.get("role_source"))
        self.assertEqual(
            sources,
            {
                "owner": ("owner", "voiceprint"),
                "inferred": ("owner", None),
                "guest": ("participant", None),
                "unbound": ("unknown", None),
                "unlabelled": ("unknown", None),
            },
        )
        # A label configured as the owner, with live roles or without, is never one.
        for provider in (EnrolledRoles(OWNER), ConfiguredPriorityProvider(OWNER)):
            with self.subTest(provider=type(provider).__name__):
                self.start(provider=provider)
                self.events.transcript(heard("l", 0, 900, "x"), 900)
                event = self.events.drain()[0]
                self.assertEqual(event["turn"]["role"], "owner")
                self.assertNotIn("role_source", event)
        # Nor where privacy is not advertised (replay, coaching).
        replay = SpeechEvents()
        replay.start(SESSION, now_ms=0, priority=enrolled())
        replay.drain()
        replay.transcript(heard("r", 0, 900, "x"), 900)
        self.assertNotIn("role_source", replay.drain()[0])

    def test_after_off_normal_behaviour_resumes_without_the_private_window(self):
        self.start()
        self.say(heard("t0", 0, 900, "A synthetic opener."), "ignore")
        self.events.duck(True, now_ms=1000)
        self.hear(*MONOLOGUE, start=1000)
        self.events.duck(False, now_ms=6000)
        events = self.say(heard("t9", 7000, 7900, "Jarvis, what time is it?"))
        self.assertEqual(self.types(events), ["transcript", "attention", "request", "conversation"])
        (request,) = [e for e in events if e["type"] == "request"]
        self.assertEqual([t["text"] for t in request["context"]["turns"]], ["A synthetic opener."])
        self.assertEqual(leaks("".join(encode_json(e) for e in events)), [])

    def test_a_turn_accepted_before_it_was_turned_on_is_still_decided(self):
        self.start()
        current = turn("t1", 0, 900, "Jarvis, what time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.duck(True, now_ms=1000)
        self.events.decision(decided(current, "attend"), 1100)
        self.assertIn("request", self.types(self.events.drain()))

    def test_speech_held_or_joined_across_the_off_edge_stays_private(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.events.duck(False, now_ms=3000)
        # Held by the merger (the owner pause), released after the off.
        self.assertEqual(self.say(turn("t1", 1000, 1900, "the gazebo budget too")), [])
        # Joined across the off.
        self.assertEqual(self.say(turn("t2", 2000, 5000, "Jarvis, so the walrus plan")), [])
        # Said before the on and joined with speech after it.
        self.events.duck(True, now_ms=8000)
        self.events.duck(False, now_ms=8500)
        self.assertEqual(self.say(turn("t3", 6000, 8100, "Jarvis, the kumquat")), [])
        # Said after the off: published and decided.
        events = self.say(turn("t4", 9000, 9900, "Jarvis, what time is it?"))
        self.assertIn("request", self.types(events))

    def test_a_session_start_keeps_it(self):
        self.events = SpeechEvents()
        self.events.duck(True, now_ms=0)
        self.events.start(SESSION, now_ms=0, addressing=OURS, conversation=MODE)
        self.events.drain()
        self.events.duck(False, now_ms=500)
        self.assertEqual(self.say(turn("t1", 0, 900, "Jarvis, hi.")), [])
        events = self.say(turn("t2", 1000, 1900, "Jarvis, what time is it?"))
        self.assertIn("request", self.types(events))

    def test_turned_off_before_any_audio_of_a_new_session_covers_nothing(self):
        self.events = SpeechEvents()
        self.events.duck(True, now_ms=0)
        self.events.start(SESSION, now_ms=0, addressing=OURS, conversation=MODE)
        self.events.drain()
        self.events.duck(False, now_ms=0)
        self.assertEqual(self.events._duck_spans, [[0, 0]])
        events = self.say(turn("t1", 0, 900, "Jarvis, what time is it?"))
        self.assertIn("request", self.types(events))

    def test_the_on_edge_is_inclusive_and_the_off_edge_exclusive(self):
        self.start()
        self.events.duck(True, now_ms=1000)
        self.events.duck(False, now_ms=2000)
        heard = self.events.heard_ducked
        self.assertTrue(heard(turn("a", 500, 1000, "x")))
        self.assertFalse(heard(turn("b", 500, 999, "x")))
        self.assertTrue(heard(turn("c", 1999, 2000, "x")))
        self.assertTrue(heard(turn("d", 1500, 2600, "x")))
        self.assertFalse(heard(turn("e", 2000, 2600, "x")))
        self.assertTrue(heard(turn("f", 2000, 2000, "x")))
        self.assertFalse(heard(turn("g", 2001, 2001, "x")))

    def test_an_empty_span_covers_nothing_even_inside_a_turn(self):
        self.start()
        self.events.duck(True, now_ms=500)
        self.events.duck(False, now_ms=500)
        self.assertEqual(self.events._duck_spans, [[500, 500]])
        self.assertFalse(self.events.heard_ducked(turn("t1", 0, 1000, "x")))

    def test_the_change_time_is_required(self):
        self.start()
        with self.assertRaises(TypeError):
            self.events.duck(True)

    def test_spans_are_bounded_and_merging_only_widens_them(self):
        self.start()
        for index in range(MAX_DUCK_SPANS + 6):
            self.events.duck(True, now_ms=index * 1000)
            self.events.duck(False, now_ms=index * 1000 + 100)
        self.assertEqual(len(self.events._duck_spans), MAX_DUCK_SPANS)
        self.assertTrue(self.events.heard_ducked(turn("t1", 500, 600, "x")))
        self.assertFalse(self.events.heard_ducked(turn("t2", 999_000, 999_100, "x")))

    def test_a_non_boolean_or_bad_time_is_refused(self):
        for bad in ("on", 1, None):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                SpeechEvents().duck(bad, now_ms=0)
        for bad in (-1, None):
            with self.subTest(now_ms=bad), self.assertRaises(ContractError):
                SpeechEvents().duck(True, now_ms=bad)


class PrivacyPhraseTests(unittest.TestCase):
    def test_kinds(self):
        cases = {
            "Jarvis, I'm back.": {"exit"},
            "I'm back.": {"rest"},
            "Jarvis,": {"name"},
            "Travis.": {"name"},
            "What mode am I in?": {"status"},
            "Jarvis, privacy mode off.": {"exit"},
            "": set(),
        }
        for text, kinds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(privacy_phrase(text, OURS), frozenset(kinds))
        # Without names nothing is named.
        self.assertEqual(privacy_phrase("Jarvis, I'm back.", None), frozenset())

    def test_entry_is_intentional_only(self):
        # The host's S-2 rule: a standalone turn, after an optional name or address, with
        # small fillers at the edges only. Never mid-sentence, with an object, or a question.
        standalone = (
            "Stop recording.",
            "Stop listening",
            "Privacy mode.",
            "Go private!",
            "Stop the recording.",
            "Stop recording now.",
            "Stop all recording.",
            "Stop the recordings.",
            "Um, stop recording please.",
            "Ok so go private, thanks",
            "Jarvis, privacy mode.",
            "Hey Jarvis, stop listening.",
            "Okay Travis, stop recording now please.",
            "Please, Jarvis, go private.",
        )
        embedded = (
            "We should stop recording the meeting.",
            "Stop listening to podcasts.",
            "Stop listening on 443.",
            "I said stop recording.",
            "Don't stop recording.",
            "Stop recording?",
            "Can you stop recording?",
            "Privacy mode off.",
            "I'm not in privacy mode.",
            "What's privacy mode?",
            "Jarvis, what is privacy mode",
            "Go private and stop recording.",
            "Stop all the recording.",
            "Stop the walrus recording.",
            "Jarvis, um, can you go private",
        )
        for text in standalone:
            with self.subTest(text=text):
                self.assertIn("entry", privacy_phrase(text, OURS))
        for text in embedded:
            with self.subTest(text=text):
                self.assertNotIn("entry", privacy_phrase(text, OURS))
        # Without names, only the unnamed forms.
        self.assertIn("entry", privacy_phrase("Stop recording.", None))
        self.assertNotIn("entry", privacy_phrase("Jarvis, stop recording.", None))


class DuckControllerTests(unittest.TestCase):
    """The controller asks no decision for, keeps no memory or audio of, a private turn."""

    def controller(self, constructor=None, **extra):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        base = {
            key: str(asset)
            for key in (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
        }
        path = Path(directory.name) / "config.json"
        path.write_text(json.dumps({**base, "demo_audio": str(asset), **extra}))
        controller = PrototypeController(PrototypeConfig.load(path), **(constructor or {}))
        self.addCleanup(controller.close)
        return controller

    @staticmethod
    def types(events):
        return [e["type"] for e in events]

    def test_a_private_turn_is_never_published_queued_or_remembered(self):
        notes = []
        controller = self.controller()
        controller.report = notes.append
        events = SpeechEvents(report=notes.append)
        events.start(SESSION, now_ms=0, addressing=OURS)
        events.drain()
        controller._events = events
        controller._decision_status = "ready"
        work, memory = queue.Queue(), TranscriptMemory()

        def accept(name, start, text):
            current = turn(name, start, start + 900, text)
            PrototypeController._accept(controller, controller._generation, work, memory, current)

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            accept("t1", 0, "A")
            controller._received_ms = 1000
            controller.duck(True)
            for index, text in enumerate(MONOLOGUE):
                accept(f"p{index}", 1000 + index * 1000, text)
            controller._received_ms = 9000
            controller.duck(False)
            # Heard at 8,000 ms, while private; released only after the off edge.
            accept("t3", 8000, "The quokka again")
            accept("t4", 9500, "D")
        self.assertEqual(work.qsize(), 2)
        self.assertEqual(controller._pending, 2)
        drained = events.drain()
        self.assertEqual(self.types(drained), ["transcript", "transcript"])
        self.assertEqual([t["text"] for t in memory.snapshot(20_000)["turns"]], ["A", "D"])
        blob = "".join(encode_json(e) for e in drained)
        self.assertEqual(leaks(blob + stderr.getvalue() + " ".join(notes)), [])

    def test_privacy_is_advertised_only_with_local_speech(self):
        # The guard itself, through `start`, not a mock (key 2's nit 3).
        hosted = {
            "transcriber": {
                "kind": "hosted-openai-compatible",
                "endpoint": "https://transcribe.example.test/v1/audio/transcriptions",
                "model": "whisper-1",
            }
        }
        for extra, advertised in (({}, True), (hosted, False)):
            with self.subTest(hosted=bool(extra)):
                events = SpeechEvents()

                def no_audio(*_args, **_options):
                    raise PrototypeError("no audio in this test")

                controller = self.controller(
                    constructor={
                        "event_publisher": events,
                        "processor_factory": no_audio,
                        "allow_hosted_speech": True,
                    },
                    **extra,
                )
                controller.start({"mode": "demo", "session_id": "advertise"})
                started = events.drain()[0]
                self.assertEqual(started["phase"], "started")
                self.assertEqual("duck" in started, advertised)
                if advertised:
                    self.assertEqual(started["duck"], {"version": 2})

    def test_the_named_owner_exit_reaches_the_host_through_the_controller(self):
        controller = self.controller()
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, addressing=OURS, priority=enrolled())
        events.drain()
        controller._events = events
        controller.duck(True)
        work, memory = queue.Queue(), TranscriptMemory()
        current = heard("t1", 100, 900, "Um, Jarvis, I'm back.")
        PrototypeController._accept(controller, controller._generation, work, memory, current)
        self.assertEqual([e.get("action") for e in events.drain()], ["exit"])
        self.assertEqual(work.qsize(), 0)
        self.assertEqual(memory.snapshot(1000)["turns"], [])

    def test_no_audio_heard_in_privacy_is_kept_in_the_speaker_ring(self):
        controller = self.controller()
        events = SpeechEvents()
        events.start(SESSION, now_ms=0)
        events.drain()
        controller._events = events
        ring = PcmRing(60_000)
        shadow = SimpleNamespace(audio=ring.append)
        processor = SimpleNamespace(pushed=[])
        processor.push_pcm16 = processor.pushed.append
        voice = b"\x01\x02" * 16_000  # one second of synthetic non-silence

        def feed():
            controller._feed(controller._generation, processor, voice, "stdin", shadow)

        feed()
        controller.duck(True)
        feed()
        feed()
        controller.duck(False)
        feed()
        # The ASR still hears it all; the ring keeps silence for the private window.
        self.assertEqual(processor.pushed, [voice] * 4)
        self.assertEqual(ring.span(0, 1000)[0], voice)
        self.assertEqual(ring.span(1000, 3000)[0], bytes(len(voice) * 2))
        self.assertEqual(ring.span(3000, 4000)[0], voice)
        self.assertEqual(events._duck_spans, [[1000, 3000]])

    def test_spans_use_the_audio_clock_when_the_wall_clock_runs_ahead(self):
        controller = self.controller()
        events = SpeechEvents()
        events.start(SESSION, now_ms=0)
        events.drain()
        controller._events = events
        controller._decision_status = "ready"
        controller._mode = "microphone"
        base = time.monotonic()
        work, memory = queue.Queue(), TranscriptMemory()

        def accept(name, start, end):
            current = turn(name, start, end, "synthetic")
            PrototypeController._accept(controller, controller._generation, work, memory, current)

        controller._audio_started, controller._received_ms = base - 180, 60_000
        self.assertGreater(controller._now_ms(), controller._received_ms)
        controller.duck(True)
        controller._audio_started, controller._received_ms = base - 240, 120_000
        controller.duck(False)
        self.assertEqual(events._duck_spans, [[60_000, 120_000]])
        accept("h1", 110_000, 118_000)
        self.assertEqual(work.qsize(), 0)
        accept("t1", 121_000, 122_000)
        self.assertEqual(work.qsize(), 1)

    def test_the_controller_forwards_or_ignores_without_a_publisher(self):
        events = SpeechEvents()
        controller = SimpleNamespace(_lock=threading.RLock(), _events=events, _received_ms=7)
        PrototypeController.duck(controller, True)
        self.assertIs(events._duck, True)
        self.assertEqual(events._duck_spans, [[7, None]])
        PrototypeController.duck(SimpleNamespace(_lock=threading.RLock(), _events=None), True)
        with self.assertRaises(PrototypeError):
            PrototypeController.duck(controller, "yes")

    def breaks(self, private, **extra):
        controller = self.controller(**extra)
        controller._events = SpeechEvents()
        controller._events.duck(private, now_ms=0)
        return controller._turn_break()

    def test_the_way_back_breaks_only_in_privacy_and_the_entry_only_outside(self):
        named = {"addressing": {"names": ["Jarvis"]}}
        private, normal = self.breaks(True, **named), self.breaks(False, **named)
        for text in ("Jarvis, I'm back.", "um so Jarvis start listening", "What mode am I in?"):
            with self.subTest(text=text):
                self.assertTrue(private(text))
                self.assertFalse(normal(text))
        for text in ("Stop recording.", "ok, go private", "Jarvis, privacy mode"):
            with self.subTest(text=text):
                self.assertTrue(normal(text))
                self.assertFalse(private(text))
        for predicate in (private, normal):
            self.assertTrue(predicate("Stop."))
            self.assertFalse(predicate("Jarvis, what time is it?"))
            self.assertFalse(predicate("Don't stop recording."))
            self.assertFalse(predicate("I'm back."))
            # The old duck phrases are gone.
            self.assertFalse(predicate("Jarvis, done thinking."))
            self.assertFalse(predicate("Jarvis, let me think out loud."))
        self.assertFalse(private("What do you think?"))
        # Without names nothing is named, but the status still breaks.
        self.assertFalse(self.breaks(True)("Jarvis, I'm back."))
        self.assertTrue(self.breaks(True)("Are you recording?"))

    def test_a_breaking_matcher_failure_leaks_no_words(self):
        predicate = self.breaks(True, addressing={"names": ["Jarvis"]})

        def broken(text, _addressing):
            raise ValueError(f"cannot match {text!r}")

        stderr = io.StringIO()
        with (
            mock.patch("rightyo.prototype.privacy_phrase", broken),
            contextlib.redirect_stderr(stderr),
        ):
            self.assertFalse(predicate(MONOLOGUE[0]))
        self.assertEqual(leaks(stderr.getvalue()), [])


FAKE_WHISPER = """#!{python}
import json, os, sys, time
audio = sys.stdin.buffer.read()
if os.path.exists(os.path.join(os.path.dirname(sys.argv[0]), "hang")):
    time.sleep(30)
json.dump(
    {{
        "transcription": [],
        "argv": sys.argv[1:],
        "cwd": os.getcwd(),
        "home": os.environ.get("HOME"),
        "tmpdir": os.environ.get("TMPDIR"),
        "riff": audio[:4].decode("latin-1"),
        "bytes": len(audio),
    }},
    sys.stdout,
)
"""


class TranscriptionWritesNothingTests(unittest.TestCase):
    """Key 3's blocker 2: the local recognizer never puts words or audio on disk, even
    briefly or after a kill: the WAV goes in on stdin and the JSON comes back on stdout."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.whisper = self.root / "whisper-cli"
        self.whisper.write_text(FAKE_WHISPER.format(python=sys.executable))
        self.whisper.chmod(0o700)
        (self.root / "model").touch()
        # A watched temporary root: nothing may appear in it.
        self.shared = self.root / "shared-tmp"
        self.shared.mkdir()

    def config(self, **extra):
        return LiveConfig(
            session_id="no-files",
            whisper_executable=self.whisper,
            whisper_model=self.root / "model",
            diarization_library=self.root / "model",
            diarization_model=self.root / "model",
            **extra,
        )

    def guarded(self):
        """No temporary file or directory may be made, and the temp root is `shared`."""

        def refuse(*_args, **_options):
            raise AssertionError("a temporary file was requested")

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.dict(os.environ, {"TMPDIR": str(self.shared)}))
        stack.enter_context(mock.patch.object(tempfile, "tempdir", str(self.shared)))
        for name in ("mkstemp", "mkdtemp", "NamedTemporaryFile", "TemporaryDirectory"):
            stack.enter_context(mock.patch.object(tempfile, name, refuse))
        return stack

    def test_audio_in_on_stdin_and_json_out_on_stdout(self):
        pcm = b"\x01\x02" * 1600
        with self.guarded():
            document = _transcribe(self.config(), pcm)
        self.assertEqual(document["transcription"], [])
        argv = document["argv"]
        self.assertEqual(argv[argv.index("--file") + 1], "-")
        self.assertEqual(argv[argv.index("--output-file") + 1], "-")
        self.assertEqual(document["riff"], "RIFF")
        self.assertGreater(document["bytes"], len(pcm))
        self.assertEqual(
            {document["cwd"], document["home"], document["tmpdir"]},
            {os.path.realpath(recognizer_dir())} | {recognizer_dir()},
        )
        self.assertEqual(list(self.shared.iterdir()), [])
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            sorted(["whisper-cli", "model", "shared-tmp"]),
        )

    def test_a_kill_mid_transcription_leaves_nothing_behind(self):
        (self.root / "hang").touch()
        processes = []
        killer = threading.Timer(0.5, lambda: processes and processes[0].kill())
        killer.start()
        self.addCleanup(killer.cancel)
        with (
            self.guarded(),
            self.assertRaises(LiveAudioError),
        ):
            _transcribe(self.config(), b"\x01\x02" * 1600, processes.append)
        # It really ran, and was killed mid-transcription.
        self.assertEqual(len(processes), 1)
        self.assertLess(processes[0].returncode, 0)
        self.assertEqual(list(self.shared.iterdir()), [])
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()),
            sorted(["whisper-cli", "model", "shared-tmp", "hang"]),
        )


FAILING_WHISPER = """#!{python}
import json, os, signal, sys, time
mode = open(os.path.join(os.path.dirname(sys.argv[0]), "mode")).read().strip()
sys.stdin.buffer.read()
doc = json.dumps({{"transcription": [{{"text": " walrus kumquat", "offsets": {{}}}}]}})
if mode == "crash":
    sys.stderr.write("fatal: walrus kumquat\\n")
    sys.exit(3)
if mode == "killed":
    sys.stdout.write(doc[:20])
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)
if mode == "truncated":
    sys.stdout.write(doc[:20])
if mode == "timeout":
    time.sleep(30)
if mode == "noise":
    sys.stderr.write("noise walrus kumquat\\n" * 5000)
    sys.stdout.write(doc)
if mode == "oversize":
    sys.stdout.write(json.dumps({{"pad": "x" * (2 * 1024 * 1024 + 10)}}))
if mode == "nondict":
    sys.stdout.write("[]")
if mode == "flood":
    chunk = "x" * 65536
    while True:
        sys.stdout.write(chunk)
        sys.stdout.flush()
"""


class TranscriptionFailureParityTests(unittest.TestCase):
    """The stdin/stdout recognizer fails exactly as the old file mode did (#242 round 2):
    the same `LiveAudioError` message per case, with no words in it, its traceback, stderr
    or the logs, and stderr noise ignored."""

    CASES = {
        "crash": "Local recognizer failed",
        "killed": "Local recognizer failed",
        "empty": "Local recognizer failed",
        "truncated": "Local recognizer failed",
        "timeout": "Local recognizer timed out",
        "oversize": "Local recognizer failed",
        "nondict": "Local recognizer failed",
    }

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.whisper = self.root / "whisper-cli"
        self.whisper.write_text(FAILING_WHISPER.format(python=sys.executable))
        self.whisper.chmod(0o700)
        self.model = self.root / "model"
        self.model.touch()

    def run_mode(self, mode):
        (self.root / "mode").write_text(mode)
        config = LiveConfig(
            session_id="parity",
            whisper_executable=self.whisper,
            whisper_model=self.model,
            diarization_library=self.model,
            diarization_model=self.model,
            timeout_seconds=1 if mode == "timeout" else 30,
        )
        stderr, logs = io.StringIO(), io.StringIO()
        handler = logging.StreamHandler(logs)
        logging.getLogger().addHandler(handler)
        self.addCleanup(logging.getLogger().removeHandler, handler)
        with contextlib.redirect_stderr(stderr):
            try:
                return _transcribe(
                    config, b"\\x01\\x02" * 1600
                ), stderr.getvalue() + logs.getvalue()
            except LiveAudioError as error:
                trace = "".join(traceback.format_exception(error))
                return error, trace + stderr.getvalue() + logs.getvalue()

    def test_each_failure_raises_the_old_error_and_leaks_nothing(self):
        for mode, message in self.CASES.items():
            with self.subTest(mode=mode):
                error, seen = self.run_mode(mode)
                self.assertIsInstance(error, LiveAudioError)
                self.assertEqual(str(error), message)
                self.assertIsNone(error.__cause__)
                self.assertTrue(error.__suppress_context__)
                self.assertNotIn("walrus", seen)
                self.assertNotIn("kumquat", seen)

    def test_stderr_noise_is_ignored(self):
        document, seen = self.run_mode("noise")
        self.assertEqual(document["transcription"][0]["text"], " walrus kumquat")
        self.assertNotIn("walrus", seen)


class BoundedRecognizerTests(TranscriptionFailureParityTests):
    """Codex P1 on #242: stdout is read in chunks and the recognizer is stopped at the cap,
    and a large WAV on stdin never deadlocks against stdout."""

    def test_a_flooding_recognizer_is_stopped_at_the_cap(self):
        from rightyo import live_audio

        read, total = os.read, [0]

        def counted(descriptor, size):
            chunk = read(descriptor, size)
            total[0] += len(chunk)
            return chunk

        processes = []
        (self.root / "mode").write_text("flood")
        config = LiveConfig(
            session_id="flood",
            whisper_executable=self.whisper,
            whisper_model=self.model,
            diarization_library=self.model,
            diarization_model=self.model,
            timeout_seconds=60,
        )
        started = time.monotonic()
        with (
            mock.patch.object(live_audio.os, "read", counted),
            self.assertRaises(LiveAudioError) as caught,
        ):
            _transcribe(config, b"\x01\x02" * 1600, processes.append)
        self.assertEqual(str(caught.exception), "Local recognizer failed")
        # Never more than one chunk past the cap was read, and the process was stopped.
        self.assertGreater(total[0], MAX_RESPONSE_BYTES)
        self.assertLessEqual(total[0], MAX_RESPONSE_BYTES + READ_CHUNK_BYTES)
        self.assertIsNotNone(processes[0].poll())
        self.assertLess(time.monotonic() - started, 30)

    def test_a_large_wav_on_stdin_does_not_deadlock(self):
        self.whisper.write_text(FAKE_WHISPER.format(python=sys.executable))
        config = LiveConfig(
            session_id="large",
            whisper_executable=self.whisper,
            whisper_model=self.model,
            diarization_library=self.model,
            diarization_model=self.model,
            timeout_seconds=30,
        )
        pcm = b"\x01\x02" * (5 * 1024 * 1024)
        document = _transcribe(config, pcm)
        self.assertEqual(document["bytes"], len(pcm) + 44)


class RecognizerSandboxTests(unittest.TestCase):
    """Codex P2 on #242: the recognizer runs only in a directory verified unwritable by this
    process; there is no fallback, so without one transcription is refused."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def candidates(self, *paths):
        from rightyo import live_audio

        return mock.patch.object(live_audio, "NO_FILES_CANDIDATES", tuple(map(str, paths)))

    def test_a_writable_or_missing_or_linked_candidate_is_refused(self):
        writable = self.root / "writable"
        writable.mkdir()
        link = self.root / "link"
        link.symlink_to(writable)
        for path in (writable, self.root / "missing", link):
            with self.subTest(path=path.name), self.candidates(path):
                with self.assertRaises(RecognizerSandboxError):
                    recognizer_dir()

    def test_an_unwritable_directory_is_accepted_and_preferred_in_order(self):
        if os.geteuid() == 0:
            self.skipTest("root can write any directory")
        locked = self.root / "locked"
        locked.mkdir()
        locked.chmod(0o555)
        self.addCleanup(locked.chmod, 0o755)
        with self.candidates(self.root / "missing", self.root, locked):
            self.assertEqual(recognizer_dir(), str(locked))

    @unittest.skipUnless(os.path.isdir("/var/empty"), "macOS /var/empty")
    def test_var_empty_is_accepted_on_macos(self):
        if os.geteuid() == 0:
            self.skipTest("root can write /var/empty")
        self.assertEqual(recognizer_dir(), "/var/empty")

    def test_without_one_transcription_is_refused_before_any_audio(self):
        model = self.root / "model"
        model.touch()
        config = LiveConfig(
            session_id="sandbox",
            whisper_executable=model,
            whisper_model=model,
            diarization_library=model,
            diarization_model=model,
        )
        with self.candidates(self.root):
            with self.assertRaises(RecognizerSandboxError):
                WhisperCppTranscriber(config)
            with self.assertRaises(RecognizerSandboxError):
                _transcribe(config, b"\x01\x02" * 1600)
            # And the controller refuses to start a session that would transcribe.
            path = self.root / "config.json"
            path.write_text(
                json.dumps(
                    {
                        key: str(model)
                        for key in (
                            "whisper_executable",
                            "whisper_model",
                            "diarization_library",
                            "diarization_model",
                            "microphone_helper",
                            "demo_audio",
                        )
                    }
                )
            )
            controller = PrototypeController(PrototypeConfig.load(path))
            self.addCleanup(controller.close)
            with self.assertRaisesRegex(PrototypeError, "cannot write, such as /var/empty"):
                controller.start({"mode": "demo"})


def fragment(text, start, end, speaker="Speaker A"):
    return {
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "speaker": speaker,
        "overlap": False,
        "speaker_provenance": "diarization-timeline",
    }


class PrivacyMergeTests(unittest.TestCase):
    """In privacy, the way back is released alone and at once, even with the owner pause."""

    def merger(self, emitted, **options):
        stop = SpeakerPriority().is_stop_phrase

        def breaks(text):
            return stop(text) or bool(privacy_phrase(text, OURS) & {"exit", "status"})

        return TurnMerger(
            2000,
            24000,
            emitted.append,
            breaks,
            owner=lambda item: item["speaker"] == "Speaker A",
            owner_pause_ms=6000,
            **options,
        )

    @staticmethod
    def texts(emitted):
        return [e["text"] for e in emitted]

    def test_the_exit_is_never_joined_onto_held_owner_speech(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(fragment("So that is the plan for the week", 0, 2000), complete=True)
        merger.offer(fragment("Jarvis, I'm back.", 3000, 4000))
        self.assertEqual(
            self.texts(emitted), ["So that is the plan for the week", "Jarvis, I'm back."]
        )

    def test_a_split_exit_is_emitted_as_soon_as_it_is_whole(self):
        emitted = []
        merger = self.merger(emitted, wake_word=lambda text: OURS.name_for(text) is not None)
        merger.offer(fragment("Jarvis,", 0, 500, speaker=None), complete=True)
        merger.offer(fragment("I'm back.", 900, 1600))
        self.assertEqual(self.texts(emitted), ["Jarvis, I'm back."])


class DuckControlTests(unittest.TestCase):
    class Recorder:
        def __init__(self):
            self.calls = []

        def reply(self, phase):
            self.calls.append(("reply", phase))

        def media(self, on):
            self.calls.append(("media", on))

        def duck(self, on):
            if type(on) is not bool:
                raise PrototypeError("invalid duck mode")
            self.calls.append(("duck", on))

    def test_duck_lines_are_forwarded_and_junk_is_skipped(self):
        from rightyo.tool import _read_control

        read, write = os.pipe()
        os.write(
            write,
            b'{"duck": true}\n'
            b'{"duck": "yes"}\n'
            b'{"duck": 1}\n'
            b'{"duck": null}\n'
            b'{"duck": false, "media": true}\n'
            b'{"duck": true, "extra": 1}\n'
            b'{"media": true}\n'
            b'{"duck": false}\n',
        )
        os.close(write)
        recorder = self.Recorder()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            _read_control(read, recorder)
        self.assertEqual(recorder.calls, [("duck", True), ("media", True), ("duck", False)])
        self.assertEqual(errors.getvalue().count("skipped"), 5)


if __name__ == "__main__":
    unittest.main()
