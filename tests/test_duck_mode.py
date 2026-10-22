"""Duck mode (hailing-station #735 R-2): while the owner thinks out loud, every turn is
still transcribed with its role, but nothing is decided or requested. Synthetic text only."""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from test_conversation import MODE, SESSION, decided, turn

from rightyo.contracts import Addressing, ContractError, Dismissal, SpeakerPriority
from rightyo.memory import TranscriptMemory
from rightyo.prototype import (
    PrototypeConfig,
    PrototypeController,
    PrototypeError,
    duck_cue_shaped,
    duck_phrase_shaped,
)
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents
from rightyo.turn_merge import TurnMerger

OURS = Addressing(names=("Jarvis",), variants=(("Jarvis", ("Travis",)),))
OWNER = SpeakerPriority(owners=("Speaker A",))
DECIDED = {"attention", "request", "conversation", "dismiss"}


class DuckModeTests(unittest.TestCase):
    def start(self, priority=OWNER, **options):
        self.events = SpeechEvents()
        provider = None if priority is None else ConfiguredPriorityProvider(priority)
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
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return self.events.drain()

    def test_started_advertises_the_duck_capability(self):
        (started,) = self.start()
        self.assertEqual(started["duck"], {"version": 1})
        # Beside the strictly validated capability set, never inside it.
        self.assertNotIn("duck", started["capabilities"])
        # A producer that takes no control (replay) does not advertise it.
        replay = SpeechEvents()
        replay.start("replay", now_ms=0)
        self.assertNotIn("duck", replay.drain()[0])
        with self.assertRaises(ContractError):
            SpeechEvents().start("bad", duck="yes")

    def test_an_owner_request_is_transcribed_but_never_decided(self):
        self.start()
        self.events.duck(True)
        events = self.say(turn("t1", 0, 900, "Jarvis, what time is it?"))
        self.assertEqual([e["type"] for e in events], ["transcript"])
        self.assertEqual(events[0]["turn"]["role"], "owner")
        self.assertEqual(self.events._pending, {})

    def test_no_follow_up_and_no_conversation_window(self):
        self.start()
        self.say(turn("t1", 0, 900, "Jarvis, what time is it?"))
        self.events.duck(True)
        events = self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.9)
        self.assertEqual([e["type"] for e in events], ["transcript"])

    def test_a_guest_turn_is_transcribed_with_its_role(self):
        self.start()
        self.events.duck(True)
        events = self.say(turn("t1", 0, 900, "Jarvis, open the door.", "Speaker B"))
        self.assertEqual([e["type"] for e in events], ["transcript"])
        self.assertEqual(events[0]["turn"]["role"], "participant")

    def test_a_turn_accepted_before_it_was_turned_on_is_still_decided(self):
        self.start()
        current = turn("t1", 0, 900, "Jarvis, what time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.duck(True)
        self.events.decision(decided(current, "attend"), current.end_ms + 100)
        events = [e["type"] for e in self.events.drain()]
        self.assertIn("request", events)

    def test_a_turn_heard_in_duck_mode_stays_undecided_after_it_is_turned_off(self):
        self.start()
        self.events.duck(True)
        current = turn("t1", 0, 900, "Jarvis, what time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.duck(False)
        self.events.decision(decided(current, "attend"), current.end_ms + 100)
        self.assertEqual([e["type"] for e in self.events.drain()], ["transcript"])

    def test_turning_it_off_restores_decisions(self):
        self.start()
        self.events.duck(True)
        self.events.duck(False)
        events = self.say(turn("t1", 0, 900, "Jarvis, what time is it?"))
        self.assertEqual(
            [e["type"] for e in events], ["transcript", "attention", "request", "conversation"]
        )

    def test_a_stop_phrase_still_dismisses_at_once(self):
        self.start(dismissal=Dismissal())
        self.events.duck(True)
        current = turn("t1", 0, 900, "Stop.")
        self.events.transcript(current, current.end_ms)
        events = [e for e in self.events.drain() if e["type"] in DECIDED | {"transcript"}]
        self.assertEqual([e["type"] for e in events], ["transcript", "dismiss"])

    def test_a_session_start_keeps_it(self):
        # The host may report duck mode before a (re)started session begins.
        self.events = SpeechEvents()
        self.events.duck(True)
        self.events.start(SESSION, now_ms=0, addressing=OURS, conversation=MODE)
        self.events.drain()
        events = self.say(turn("t1", 0, 900, "Jarvis, what time is it?"))
        self.assertEqual([e["type"] for e in events], ["transcript"])

    def test_it_is_independent_of_media_mode(self):
        # The host keeps the two exclusive; if both were on, duck mode decides nothing.
        self.start()
        self.events.media(True)
        self.events.duck(True)
        self.assertEqual(
            [e["type"] for e in self.say(turn("t1", 0, 900, "Jarvis, what's on?"))],
            ["transcript"],
        )
        self.events.duck(False)
        events = self.say(turn("t2", 2000, 2900, "What time is it?"))
        self.assertEqual(events[1]["decision"].get("media"), True)

    def test_a_non_boolean_is_refused(self):
        for bad in ("on", 1, None):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                SpeechEvents().duck(bad)


class DuckControllerTests(unittest.TestCase):
    """The controller asks no hosted decision for a turn accepted in duck mode."""

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

    def test_a_ducked_turn_is_published_but_never_queued(self):
        controller = self.controller()
        events = SpeechEvents()
        events.start(SESSION, now_ms=0)
        events.drain()
        controller._events = events
        controller._decision_status = "ready"
        work = queue.Queue()

        def accept(name, start, text):
            current = turn(name, start, start + 900, text)
            generation = controller._generation
            PrototypeController._accept(controller, generation, work, TranscriptMemory(), current)

        accept("t1", 0, "A")
        self.assertEqual(work.qsize(), 1)
        controller.duck(True)
        self.assertIs(events.ducked, True)
        accept("t2", 1000, "B")
        self.assertEqual(work.qsize(), 1)
        self.assertEqual(controller._pending, 1)
        self.assertEqual([e["type"] for e in events.drain()], ["transcript", "transcript"])
        controller.duck(False)
        accept("t3", 2000, "C")
        self.assertEqual(work.qsize(), 2)

    def test_the_controller_forwards_or_ignores_without_a_publisher(self):
        events = SpeechEvents()
        controller = SimpleNamespace(_lock=threading.RLock(), _events=events)
        PrototypeController.duck(controller, True)
        self.assertIs(events._duck, True)
        PrototypeController.duck(SimpleNamespace(_lock=threading.RLock(), _events=None), True)
        with self.assertRaises(PrototypeError):
            PrototypeController.duck(controller, "yes")

    def test_the_turn_break_releases_named_duck_phrases_only_with_names(self):
        named = self.controller(addressing={"names": ["Jarvis"]})._turn_break()
        self.assertTrue(named("Jarvis, done thinking."))
        self.assertTrue(named("Stop."))
        self.assertFalse(named("Jarvis, what time is it?"))
        self.assertFalse(named("Jarvis, go away"))  # dismissal shapes need dismissal on
        unnamed = self.controller()._turn_break()
        self.assertFalse(unnamed("Jarvis, done thinking."))

    def test_the_turn_break_releases_duck_cues_only_while_ducking(self):
        controller = self.controller(addressing={"names": ["Jarvis"]})
        controller._events = SpeechEvents()
        breaks = controller._turn_break()
        cues = ("What do you think?", "Jarvis, play that back.", "Mute.", "Done thinking.")
        for text in cues:
            with self.subTest(text=text):
                self.assertFalse(breaks(text))
        controller.duck(True)
        for text in cues:
            with self.subTest(text=text):
                self.assertTrue(breaks(text))
        self.assertFalse(breaks("What do you think about the plan?"))
        unnamed = self.controller()
        unnamed._events = controller._events
        self.assertTrue(unnamed._turn_break()("What do you think?"))


class DuckPhraseTests(unittest.TestCase):
    def test_named_entry_and_exit_phrases_match_as_the_host_trims_them(self):
        for text in (
            "Jarvis, done thinking.",
            "Jarvis done thinking please",
            "Hey Jarvis, please end duck mode.",
            "Okay Travis, I’m done thinking.",
            "Jarvis, let me think out loud.",
            "JARVIS, DUCK MODE!",
        ):
            with self.subTest(text=text):
                self.assertTrue(duck_phrase_shaped(text, OURS))

    def test_anything_else_is_not_one(self):
        for text in (
            "Done thinking.",
            "Jarvis.",
            "Jarvis, done thinking about the budget.",
            "So that's the plan, Jarvis, done thinking.",
            "Ducks are thinking out loud.",
            "",
        ):
            with self.subTest(text=text):
                self.assertFalse(duck_phrase_shaped(text, OURS))
        self.assertFalse(duck_phrase_shaped("Jarvis, done thinking.", None))

    def test_cues_match_named_or_not(self):
        for text in ("What do you think?", "Hey Jarvis, play that back please.", "Mute"):
            with self.subTest(text=text):
                self.assertTrue(duck_cue_shaped(text, OURS))
        self.assertTrue(duck_cue_shaped("what do you think", None))
        self.assertFalse(duck_cue_shaped("What do you think of it?", OURS))


class DuckPhraseMergeTests(unittest.TestCase):
    """A named duck phrase is released alone, at once, even with the owner pause (R-1)."""

    def fragment(self, text, start, end, speaker="Speaker A"):
        return {
            "text": text,
            "start_ms": start,
            "end_ms": end,
            "speaker": speaker,
            "overlap": False,
            "speaker_provenance": "diarization-timeline",
        }

    def merger(self, emitted, **options):
        stop = SpeakerPriority().is_stop_phrase
        return TurnMerger(
            2000,
            24000,
            emitted.append,
            lambda text: stop(text) or duck_phrase_shaped(text, OURS),
            owner=lambda item: item["speaker"] == "Speaker A",
            owner_pause_ms=6000,
            **options,
        )

    def test_the_exit_is_never_joined_onto_held_owner_speech(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(self.fragment("So that is the plan for the week", 0, 2000), complete=True)
        merger.offer(self.fragment("Jarvis, done thinking.", 3000, 4000))
        self.assertEqual(
            [e["text"] for e in emitted],
            ["So that is the plan for the week", "Jarvis, done thinking."],
        )

    def test_nothing_joins_the_exit_after_it(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(self.fragment("Jarvis, done thinking.", 0, 1000))
        self.assertEqual([e["text"] for e in emitted], ["Jarvis, done thinking."])
        merger.offer(self.fragment("and one more thing", 1500, 2500))
        merger.flush()
        self.assertEqual(
            [e["text"] for e in emitted], ["Jarvis, done thinking.", "and one more thing"]
        )

    def test_a_split_exit_is_emitted_as_soon_as_it_is_whole(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(self.fragment("Jarvis, done", 0, 800))
        self.assertEqual(emitted, [])
        merger.offer(self.fragment("thinking.", 1200, 1800))
        self.assertEqual([e["text"] for e in emitted], ["Jarvis, done thinking."])

    def test_a_cue_while_ducking_is_released_alone_and_at_once(self):
        emitted = []
        merger = TurnMerger(
            2000,
            24000,
            emitted.append,
            lambda text: duck_cue_shaped(text, OURS),
            owner=lambda item: item["speaker"] == "Speaker A",
            owner_pause_ms=6000,
        )
        merger.offer(self.fragment("I keep coming back to the second option", 0, 2000))
        merger.offer(self.fragment("What do you think?", 3000, 3800), complete=True)
        self.assertEqual(
            [e["text"] for e in emitted],
            ["I keep coming back to the second option", "What do you think?"],
        )

    def test_without_the_break_the_owner_pause_would_hide_it(self):
        # The failure this guards against (hailing-station #783 review).
        emitted = []
        merger = TurnMerger(
            2000,
            24000,
            emitted.append,
            SpeakerPriority().is_stop_phrase,
            owner=lambda item: item["speaker"] == "Speaker A",
            owner_pause_ms=6000,
        )
        merger.offer(self.fragment("So that is the plan for the week", 0, 2000), complete=True)
        merger.offer(self.fragment("Jarvis, done thinking.", 3000, 4000))
        merger.flush()
        self.assertEqual(
            [e["text"] for e in emitted],
            ["So that is the plan for the week Jarvis, done thinking."],
        )


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
