"""Privacy mode, internally duck v2 (hailing-station #735 S-1): while it is on, rightyo keeps
and emits nothing heard, and matches the text only locally for the owner's named exit and
anyone's status question. Synthetic text only."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_conversation import MODE, SESSION, decided, turn

from rightyo import tool_events
from rightyo.contracts import Addressing, ContractError, Dismissal, SpeakerPriority
from rightyo.live_speaker_id import PcmRing
from rightyo.memory import TranscriptMemory
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import MAX_DUCK_SPANS, SpeechEvents, encode_json, privacy_phrase
from rightyo.turn_merge import TurnMerger

OURS = Addressing(names=("Jarvis",), variants=(("Jarvis", ("Travis",)),))
OWNER = SpeakerPriority(owners=("Speaker A",))
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


class PrivacyEventsTests(unittest.TestCase):
    def start(self, priority=OWNER, provider=None, **options):
        self.notes = []
        self.events = SpeechEvents(report=self.notes.append)
        if provider is None and priority is not None:
            provider = ConfiguredPriorityProvider(priority)
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
            events += self.say(turn(f"p{at}", at, at + gap - 100, text, speaker))
        return events

    @staticmethod
    def types(events):
        return [e["type"] for e in events]

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
        # No digest of the words either: only a constant per private turn.
        self.assertEqual(set(self.events._seen.values()), {tool_events.PRIVATE_DIGEST})
        # Every byte the host could have read, as the JSONL stream carries it.
        everything = "".join(encode_json(e) + "\n" for e in [*stream, *events, *tail])
        self.assertEqual(leaks(everything), [])
        self.assertEqual(leaks(stderr.getvalue() + logs.getvalue()), [])
        self.assertEqual(leaks(" ".join(self.notes)), [])
        # Logs carry counts only.
        self.assertEqual(self.notes, ["privacy off: discarded=36 exit=0 status=0 failed=0"])

    def test_no_hosted_role_question_is_asked_for_a_private_turn(self):
        provider = RecordingProvider(OWNER)
        self.start(provider=provider)
        self.events.duck(True, now_ms=0)
        self.hear(*MONOLOGUE, speaker="Speaker B")
        self.assertEqual(provider.states, [])
        # Nor is a role fixed for the speaker from it.
        self.assertEqual(self.events._roles, {})

    def test_the_named_owner_exit_emits_one_content_free_event(self):
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

    def test_a_guest_or_unnamed_exit_is_ignored(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.hear("Jarvis, I'm back.", speaker="Speaker B"), [])
        self.assertEqual(self.hear("Jarvis, I'm back.", speaker=None, start=5000), [])
        self.assertEqual(self.hear("I'm back.", start=10_000), [])
        self.assertEqual(self.hear("Start listening.", start=15_000), [])

    def test_without_roles_nobody_can_exit(self):
        self.start(priority=None)
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.hear("Jarvis, I'm back."), [])

    def test_an_inferred_owner_label_cannot_exit(self):
        self.start()
        self.events.duck(True, now_ms=0)
        current = turn("t1", 100, 900, "Jarvis, I'm back.")
        self.events.transcript(current, current.end_ms, inferred=True)
        self.assertEqual(self.events.drain(), [])

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
        self.assertEqual(self.hear("Jarvis,", speaker=None), [])
        events = self.hear("I'm back.", start=1500)
        self.assertEqual([e["action"] for e in events], ["exit"])
        # Too late after the name, or with a turn between: no exit.
        self.assertEqual(self.hear("Jarvis,", start=10_000), [])
        self.assertEqual(self.hear("I'm back.", start=20_000), [])
        self.assertEqual(self.hear("Jarvis,", start=30_000), [])
        self.assertEqual(self.hear("Something else.", start=31_000), [])
        self.assertEqual(self.hear("I'm back.", start=32_000), [])
        # A guest finishing someone's name is still a guest.
        self.assertEqual(self.hear("Jarvis,", start=40_000), [])
        self.assertEqual(self.hear("I'm back.", speaker="Speaker B", start=41_000), [])

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

    def test_a_stop_phrase_only_stops_playback_and_carries_no_words(self):
        self.start(dismissal=Dismissal())
        self.events.duck(True, now_ms=0)
        (dismiss,) = self.hear("Stop.")
        self.assertEqual(dismiss["type"], "dismiss")
        self.assertEqual(dismiss["scope"], ["playback"])
        self.assertEqual(dismiss["withdrawn_request_ids"], [])
        self.assertNotIn("text", json.dumps(dismiss))
        # Without natural dismissal, nothing.
        self.start()
        self.events.duck(True, now_ms=0)
        self.assertEqual(self.hear("Stop."), [])

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
        self.assertEqual(self.notes, ["privacy off: discarded=4 exit=0 status=0 failed=4"])

    def test_after_off_normal_behaviour_resumes_without_the_private_window(self):
        self.start()
        self.say(turn("t0", 0, 900, "A synthetic opener."), "ignore")
        self.events.duck(True, now_ms=1000)
        self.hear(*MONOLOGUE, start=1000)
        self.events.duck(False, now_ms=6000)
        events = self.say(turn("t9", 7000, 7900, "Jarvis, what time is it?"))
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
            "Um, stop recording.": {"entry"},
            "Ok so go private": {"entry"},
            "Don't stop recording.": set(),
            "We should not, um, do not stop listening": set(),
            "Jarvis, privacy mode.": {"entry"},
            "Jarvis, privacy mode off.": {"exit", "entry"},
            "": set(),
        }
        for text, kinds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(privacy_phrase(text, OURS), frozenset(kinds))
        # Without names nothing is named.
        self.assertEqual(privacy_phrase("Jarvis, I'm back.", None), frozenset())


class DuckControllerTests(unittest.TestCase):
    """The controller asks no decision for, keeps no memory or audio of, a private turn."""

    def controller(self, **extra):
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
        path.write_text(json.dumps({**base, **extra}))
        controller = PrototypeController(PrototypeConfig.load(path))
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

    def test_hosted_speech_is_recognized_so_privacy_is_not_advertised(self):
        self.assertFalse(self.controller()._hosted_speech())
        hosted = {
            "transcriber": {
                "kind": "hosted-openai-compatible",
                "endpoint": "https://transcribe.example.test/v1/audio/transcriptions",
                "model": "whisper-1",
            }
        }
        self.assertTrue(self.controller(**hosted)._hosted_speech())
        diarizer = {"diarizer": {"kind": "hosted-deepgram", "model": "nova-3"}}
        self.assertTrue(self.controller(**diarizer)._hosted_speech())

    def test_the_named_owner_exit_reaches_the_host_through_the_controller(self):
        controller = self.controller()
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, addressing=OURS, priority=ConfiguredPriorityProvider(OWNER))
        events.drain()
        controller._events = events
        controller.duck(True)
        work, memory = queue.Queue(), TranscriptMemory()
        current = turn("t1", 100, 900, "Um, Jarvis, I'm back.")
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
