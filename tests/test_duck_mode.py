"""Duck mode (hailing-station #735 R-2): while the owner thinks out loud, every turn is
still transcribed with its role, but nothing heard then is decided, requested or used as
context. Synthetic text only."""

from __future__ import annotations

import contextlib
import io
import json
import os
import queue
import tempfile
import threading
import time
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
from rightyo.tool_events import MAX_DUCK_SPANS, SpeechEvents
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

    def types(self, events):
        return [e["type"] for e in events]

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
        self.events.duck(True, now_ms=0)
        events = self.say(turn("t1", 100, 900, "Jarvis, what time is it?"))
        self.assertEqual(self.types(events), ["transcript"])
        self.assertEqual(events[0]["turn"]["role"], "owner")
        self.assertEqual(self.events._pending, {})

    def test_no_follow_up_and_no_conversation_window(self):
        self.start()
        self.say(turn("t1", 0, 900, "Jarvis, what time is it?"))
        self.events.duck(True, now_ms=1000)
        events = self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.9)
        self.assertEqual(self.types(events), ["transcript"])

    def test_a_guest_turn_is_transcribed_with_its_role(self):
        self.start()
        self.events.duck(True, now_ms=0)
        events = self.say(turn("t1", 100, 900, "Jarvis, open the door.", "Speaker B"))
        self.assertEqual(self.types(events), ["transcript"])
        self.assertEqual(events[0]["turn"]["role"], "participant")

    def test_a_turn_heard_and_accepted_before_it_was_turned_on_is_still_decided(self):
        self.start()
        current = turn("t1", 0, 900, "Jarvis, what time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.duck(True, now_ms=1000)
        self.events.decision(decided(current, "attend"), 1100)
        self.assertIn("request", self.types(self.events.drain()))

    def test_a_turn_heard_in_duck_mode_stays_undecided_after_it_is_turned_off(self):
        self.start()
        self.events.duck(True, now_ms=0)
        current = turn("t1", 100, 900, "Jarvis, what time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.duck(False, now_ms=1000)
        self.events.decision(decided(current, "attend"), 1100)
        self.assertEqual(self.types(self.events.drain()), ["transcript"])

    def test_speech_held_past_the_off_edge_is_never_decided(self):
        # Key 2's minimal case: heard while ducking, held by the merger (the owner pause),
        # and released only after the host turned duck mode off.
        self.start()
        self.events.duck(True, now_ms=0)
        self.events.duck(False, now_ms=3000)
        events = self.say(turn("t1", 1000, 1900, "and the budget too"))
        self.assertEqual(self.types(events), ["transcript"])

    def test_a_turn_joined_across_the_off_edge_is_never_decided(self):
        # A ducked piece held, the mode turned off, then a continuation joined into it.
        self.start()
        self.events.duck(True, now_ms=0)
        self.events.duck(False, now_ms=3000)
        joined = turn("t1", 2000, 5000, "Jarvis, so that is the plan and the budget too")
        self.assertEqual(self.types(self.say(joined)), ["transcript"])

    def test_pre_duck_speech_joined_with_ducked_speech_stays_undecided(self):
        self.start()
        self.events.duck(True, now_ms=2000)
        self.events.duck(False, now_ms=2500)
        joined = turn("t1", 0, 2100, "Jarvis, what time is it and the budget")
        self.assertEqual(self.types(self.say(joined)), ["transcript"])

    def test_turning_it_off_restores_decisions_for_speech_heard_after(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.events.duck(False, now_ms=500)
        events = self.say(turn("t1", 1000, 1900, "Jarvis, what time is it?"))
        self.assertEqual(self.types(events), ["transcript", "attention", "request", "conversation"])

    def test_ducked_turns_are_never_context(self):
        self.start()
        self.events.duck(True, now_ms=0)
        self.say(turn("t1", 100, 900, "A guest's synthetic aside.", "Speaker B"))
        self.say(turn("t2", 1000, 1900, "The owner's synthetic thought."))
        self.events.duck(False, now_ms=2000)
        events = self.say(turn("t3", 3000, 3900, "Jarvis, what time is it?"))
        (request,) = [e for e in events if e["type"] == "request"]
        self.assertEqual(request["context"]["turns"], [])
        self.assertEqual(self.events._memory.snapshot(4000)["turns"], [request["turn"]])

    def test_a_stop_phrase_still_dismisses_at_once(self):
        self.start(dismissal=Dismissal())
        self.events.duck(True, now_ms=0)
        current = turn("t1", 100, 900, "Stop.")
        self.events.transcript(current, current.end_ms)
        events = [e for e in self.events.drain() if e["type"] in DECIDED | {"transcript"}]
        self.assertEqual(self.types(events), ["transcript", "dismiss"])

    def test_a_session_start_keeps_it(self):
        # The host may report duck mode before a (re)started session begins.
        self.events = SpeechEvents()
        self.events.duck(True, now_ms=0)
        self.events.start(SESSION, now_ms=0, addressing=OURS, conversation=MODE)
        self.events.drain()
        self.events.duck(False, now_ms=500)
        self.assertEqual(self.types(self.say(turn("t1", 0, 900, "Jarvis, hi."))), ["transcript"])
        events = self.say(turn("t2", 1000, 1900, "Jarvis, what time is it?"))
        self.assertIn("request", self.types(events))

    def test_turned_off_before_any_audio_of_a_new_session_covers_nothing(self):
        # Codex P2 on #237: kept on into a new session, then off before audio: [0, 0].
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
        # Ending as it went on: counted (fail closed).
        self.assertTrue(heard(turn("a", 500, 1000, "x")))
        self.assertFalse(heard(turn("b", 500, 999, "x")))
        # The off is stamped at or after the end of all audio received by then, so a piece
        # heard in duck mode ends at or before it and starts before it.
        self.assertTrue(heard(turn("c", 1999, 2000, "x")))
        self.assertTrue(heard(turn("d", 1500, 2600, "x")))  # joined across the off
        # Starting at the off: heard after it. An empty piece exactly at the off may have
        # ended there, so it counts (#238, fail closed).
        self.assertFalse(heard(turn("e", 2000, 2600, "x")))
        self.assertTrue(heard(turn("f", 2000, 2000, "x")))
        self.assertFalse(heard(turn("g", 2001, 2001, "x")))

    def test_an_empty_span_covers_nothing_even_inside_a_turn(self):
        # #238: [500, 500] (on and off with no audio between) must not duck a turn around it.
        self.start()
        self.events.duck(True, now_ms=500)
        self.events.duck(False, now_ms=500)
        self.assertEqual(self.events._duck_spans, [[500, 500]])
        self.assertFalse(self.events.heard_ducked(turn("t1", 0, 1000, "x")))
        self.assertFalse(self.events.heard_ducked(turn("t2", 500, 500, "x")))

    def test_the_change_time_is_required(self):
        # #238: no default to the latest event time, which could drift from the audio clock.
        self.start()
        with self.assertRaises(TypeError):
            self.events.duck(True)

    def test_it_is_independent_of_media_mode(self):
        # The host keeps the two exclusive; if both were on, duck mode decides nothing.
        self.start()
        self.events.media(True)
        self.events.duck(True, now_ms=0)
        self.assertEqual(
            self.types(self.say(turn("t1", 100, 900, "Jarvis, what's on?"))), ["transcript"]
        )
        self.events.duck(False, now_ms=1000)
        events = self.say(turn("t2", 2000, 2900, "What time is it?"))
        self.assertEqual(events[1]["decision"].get("media"), True)

    def test_spans_are_bounded_and_merging_only_widens_them(self):
        self.start()
        for index in range(MAX_DUCK_SPANS + 6):
            self.events.duck(True, now_ms=index * 1000)
            self.events.duck(False, now_ms=index * 1000 + 100)
        self.assertEqual(len(self.events._duck_spans), MAX_DUCK_SPANS)
        # Between the first two spans: now covered, so held undecided (fail closed).
        self.assertTrue(self.events.heard_ducked(turn("t1", 500, 600, "x")))
        self.assertFalse(self.events.heard_ducked(turn("t2", 999_000, 999_100, "x")))

    def test_a_non_boolean_or_bad_time_is_refused(self):
        for bad in ("on", 1, None):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                SpeechEvents().duck(bad, now_ms=0)
        for bad in (-1, None):
            with self.subTest(now_ms=bad), self.assertRaises(ContractError):
                SpeechEvents().duck(True, now_ms=bad)


class DuckControllerTests(unittest.TestCase):
    """The controller asks no hosted decision for, and keeps no memory of, a ducked turn."""

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

    def test_a_ducked_turn_is_published_but_never_queued_or_remembered(self):
        controller = self.controller()
        events = SpeechEvents()
        events.start(SESSION, now_ms=0)
        events.drain()
        controller._events = events
        controller._decision_status = "ready"
        work, memory = queue.Queue(), TranscriptMemory()

        def accept(name, start, text):
            current = turn(name, start, start + 900, text)
            PrototypeController._accept(controller, controller._generation, work, memory, current)

        accept("t1", 0, "A")
        self.assertEqual(work.qsize(), 1)
        controller._received_ms = 1000
        controller.duck(True)
        self.assertEqual(events._duck_spans, [[1000, None]])
        accept("t2", 1000, "B")
        self.assertEqual(work.qsize(), 1)
        controller._received_ms = 3000
        controller.duck(False)
        # Heard at 2,000 ms, while ducking; released only after the off edge.
        accept("t3", 2000, "C")
        self.assertEqual(work.qsize(), 1)
        accept("t4", 3500, "D")
        self.assertEqual(work.qsize(), 2)
        self.assertEqual(controller._pending, 2)
        self.assertEqual(self.types(events.drain()), ["transcript"] * 4)
        self.assertEqual([t["text"] for t in memory.snapshot(5000)["turns"]], ["A", "D"])

    @staticmethod
    def types(events):
        return [e["type"] for e in events]

    def test_spans_use_the_audio_clock_when_the_wall_clock_runs_ahead(self):
        # Key 1's round-3 case: two minutes of audio lost earlier, so the wall clock (and
        # `_now_ms`) runs D = 120 s ahead of the received audio turns are timed in.
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
        # Heard while ducking, held by the owner pause, released after the off: undecided.
        accept("h1", 110_000, 118_000)
        self.assertEqual(work.qsize(), 0)
        # Said after the off in audio time, though within D of its wall-clock time: decided.
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

    def breaks(self, ducking, **extra):
        controller = self.controller(**extra)
        controller._events = SpeechEvents()
        controller._events.duck(ducking, now_ms=0)
        return controller._turn_break()

    def test_the_exit_breaks_only_while_ducking_and_the_entry_only_while_not(self):
        named = {"addressing": {"names": ["Jarvis"]}}
        ducking, normal = self.breaks(True, **named), self.breaks(False, **named)
        self.assertTrue(ducking("Jarvis, done thinking."))
        self.assertFalse(normal("Jarvis, done thinking."))
        self.assertTrue(normal("Jarvis, let me think out loud."))
        self.assertFalse(ducking("Jarvis, let me think out loud."))
        for predicate in (ducking, normal):
            self.assertTrue(predicate("Stop."))
            self.assertFalse(predicate("Jarvis, what time is it?"))
            self.assertFalse(predicate("Jarvis, go away"))  # dismissal shapes need dismissal
        self.assertFalse(self.breaks(True)("Jarvis, done thinking."))

    def test_cues_break_only_while_ducking_named_or_not(self):
        cues = (
            "What do you think?",
            "Jarvis, play that back.",
            "Mute.",
            "Are you still listening?",
            "Discard that.",
        )
        ducking = self.breaks(True, addressing={"names": ["Jarvis"]})
        normal = self.breaks(False, addressing={"names": ["Jarvis"]})
        for text in cues:
            with self.subTest(text=text):
                self.assertTrue(ducking(text))
                self.assertFalse(normal(text))
        self.assertFalse(ducking("What do you think about the plan?"))
        # The unnamed exit is not a cue, so a lone wake word can still join it (#230).
        self.assertFalse(ducking("done thinking."))
        self.assertTrue(self.breaks(True)("What do you think?"))


class DuckPhraseTests(unittest.TestCase):
    def test_named_exit_phrases_match_as_the_host_trims_them(self):
        for text in (
            "Jarvis, done thinking.",
            "Jarvis done thinking please",
            "Hey Jarvis, please end duck mode.",
            "Okay Travis, I'm done thinking.",
            "JARVIS, DUCK MODE OFF!",
        ):
            with self.subTest(text=text):
                self.assertTrue(duck_phrase_shaped(text, OURS, True))
                self.assertFalse(duck_phrase_shaped(text, OURS, False))

    def test_named_entry_phrases_match_only_while_not_ducking(self):
        for text in ("Jarvis, let me think out loud.", "Hey Jarvis, duck mode."):
            with self.subTest(text=text):
                self.assertTrue(duck_phrase_shaped(text, OURS, False))
                self.assertFalse(duck_phrase_shaped(text, OURS, True))

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
                self.assertFalse(duck_phrase_shaped(text, OURS, True))
        self.assertFalse(duck_phrase_shaped("Jarvis, done thinking.", None, True))

    def test_cues_match_named_or_not(self):
        for text in ("What do you think?", "Hey Jarvis, play that back please.", "Mute"):
            with self.subTest(text=text):
                self.assertTrue(duck_cue_shaped(text, OURS))
        self.assertTrue(duck_cue_shaped("are you still listening", None))
        self.assertFalse(duck_cue_shaped("What do you think of it?", OURS))
        self.assertFalse(duck_cue_shaped("Done thinking.", OURS))


def fragment(text, start, end, speaker="Speaker A"):
    return {
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "speaker": speaker,
        "overlap": False,
        "speaker_provenance": "diarization-timeline",
    }


class DuckPhraseMergeTests(unittest.TestCase):
    """While ducking, the exit and the cues are released alone and at once, even with the
    owner pause (R-1), and a lone wake word still joins the exit (#230)."""

    def merger(self, emitted, breaks=None, **options):
        stop = SpeakerPriority().is_stop_phrase
        if breaks is None:

            def breaks(text):
                return (
                    stop(text)
                    or duck_phrase_shaped(text, OURS, True)
                    or duck_cue_shaped(text, OURS)
                )

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
        merger.offer(fragment("Jarvis, done thinking.", 3000, 4000))
        self.assertEqual(
            self.texts(emitted), ["So that is the plan for the week", "Jarvis, done thinking."]
        )

    def test_nothing_joins_the_exit_after_it(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(fragment("Jarvis, done thinking.", 0, 1000))
        self.assertEqual(self.texts(emitted), ["Jarvis, done thinking."])
        merger.offer(fragment("and one more thing", 1500, 2500))
        merger.flush()
        self.assertEqual(self.texts(emitted), ["Jarvis, done thinking.", "and one more thing"])

    def test_a_split_exit_is_emitted_as_soon_as_it_is_whole(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(fragment("Jarvis, done", 0, 800))
        self.assertEqual(emitted, [])
        merger.offer(fragment("thinking.", 1200, 1800))
        self.assertEqual(self.texts(emitted), ["Jarvis, done thinking."])

    def test_a_comma_pause_exit_still_joins_the_lone_wake_word(self):
        # Key 1's B2: "Jarvis," unattributed, then the owner's "done thinking." (#230).
        emitted = []
        merger = self.merger(emitted, wake_word=lambda text: OURS.name_for(text) is not None)
        merger.offer(fragment("Jarvis,", 0, 500, speaker=None), complete=True)
        merger.offer(fragment("done thinking.", 900, 1600))
        self.assertEqual(self.texts(emitted), ["Jarvis, done thinking."])

    def test_a_cue_while_ducking_is_released_alone_and_at_once(self):
        emitted = []
        merger = self.merger(emitted)
        merger.offer(fragment("I keep coming back to the second option", 0, 2000))
        merger.offer(fragment("What do you think?", 3000, 3800), complete=True)
        self.assertEqual(
            self.texts(emitted), ["I keep coming back to the second option", "What do you think?"]
        )

    def test_without_the_break_the_owner_pause_would_hide_it(self):
        # The failure this guards against (hailing-station #783 review).
        emitted = []
        merger = self.merger(emitted, breaks=SpeakerPriority().is_stop_phrase)
        merger.offer(fragment("So that is the plan for the week", 0, 2000), complete=True)
        merger.offer(fragment("Jarvis, done thinking.", 3000, 4000))
        merger.flush()
        self.assertEqual(
            self.texts(emitted), ["So that is the plan for the week Jarvis, done thinking."]
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
