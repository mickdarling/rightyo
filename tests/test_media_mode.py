"""Media mode (hailing-station #551): while the owner watches TV, only the owner's voice
naming this assistant attends, and nothing is a follow-up."""

from __future__ import annotations

import contextlib
import io
import os
import threading
import unittest
from types import SimpleNamespace

from test_conversation import MODE, SESSION, decided, turn

from rightyo.contracts import Addressing, ContractError, SpeakerPriority
from rightyo.prototype import PrototypeController, PrototypeError
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents

OURS = Addressing(names=("Jarvis",), variants=(("Jarvis", ("Travis",)),))
OWNER = SpeakerPriority(owners=("Speaker A",))


class MediaModeTests(unittest.TestCase):
    def start(self, priority=None, addressing=OURS):
        self.events = SpeechEvents()
        provider = None if priority is None else ConfiguredPriorityProvider(priority)
        self.events.start(
            SESSION, now_ms=0, addressing=addressing, conversation=MODE, priority=provider
        )
        self.events.drain()

    def say(self, current, label, **options):
        self.events.transcript(current, current.end_ms)
        self.events.decision(decided(current, label, **options), current.end_ms + 100)
        return [e for e in self.events.drain() if e["type"] in {"attention", "request"}]

    def test_an_unnamed_attend_is_ignored_and_says_why(self):
        self.start()
        self.events.media(True)
        events = self.say(turn("t1", 0, 900, "What time is it?"), "attend")
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertEqual(events[0]["decision"]["label"], "ignore")
        self.assertIs(events[0]["decision"]["media"], True)

    def test_a_named_attend_still_gets_through_and_a_variant_counts(self):
        self.start()
        self.events.media(True)
        events = self.say(turn("t1", 0, 900, "Travis, pause the show."), "attend")
        self.assertEqual([e["type"] for e in events], ["attention", "request"])
        self.assertNotIn("media", events[0]["decision"])

    def test_with_roles_only_the_owners_voice_naming_it_attends(self):
        self.start(OWNER)
        self.events.media(True)
        tv = self.say(turn("t1", 0, 900, "Jarvis, open the pod bay doors.", "Speaker B"), "attend")
        self.assertEqual([e["type"] for e in tv], ["attention"])
        self.assertIs(tv[0]["decision"]["media"], True)
        owner = self.say(turn("t2", 2000, 2900, "Jarvis, what's the score?"), "attend")
        self.assertEqual([e["type"] for e in owner], ["attention", "request"])

    def test_no_turn_is_a_follow_up(self):
        self.start()
        self.say(turn("t1", 0, 900, "Jarvis, what time is it?"), "attend")
        self.events.media(True)
        events = self.say(turn("t2", 2000, 2900, "And tomorrow?"), "uncertain", attend=0.9)
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertNotIn("follow_up", events[0]["decision"])

    def test_a_turn_heard_in_media_mode_keeps_it_when_its_decision_comes_later(self):
        # Codex review of #196: the hosted decision can arrive after the host turned it off.
        self.start()
        self.events.media(True)
        current = turn("t1", 0, 900, "What time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.media(False)
        self.events.decision(decided(current, "attend"), current.end_ms + 100)
        events = [e for e in self.events.drain() if e["type"] in {"attention", "request"}]
        self.assertEqual([e["type"] for e in events], ["attention"])
        self.assertIs(events[0]["decision"]["media"], True)

    def test_a_request_made_before_it_was_turned_on_still_stands(self):
        # Codex review of #196: the snapshot works both ways.
        self.start()
        current = turn("t1", 0, 900, "What time is it?")
        self.events.transcript(current, current.end_ms)
        self.events.media(True)
        self.events.decision(decided(current, "attend"), current.end_ms + 100)
        events = [e for e in self.events.drain() if e["type"] in {"attention", "request"}]
        self.assertEqual([e["type"] for e in events], ["attention", "request"])

    def test_turning_it_off_restores_unnamed_attends(self):
        self.start()
        self.events.media(True)
        self.events.media(False)
        events = self.say(turn("t1", 0, 900, "What time is it?"), "attend")
        self.assertEqual([e["type"] for e in events], ["attention", "request"])

    def test_without_names_nothing_attends(self):
        self.start(addressing=None)
        self.events.media(True)
        events = self.say(turn("t1", 0, 900, "Jarvis, what time is it?"), "attend")
        self.assertEqual([e["type"] for e in events], ["attention"])

    def test_a_session_start_keeps_it(self):
        # The host may report media mode before a (re)started session begins.
        self.events = SpeechEvents()
        self.events.media(True)
        self.events.start(SESSION, now_ms=0, addressing=OURS, conversation=MODE)
        self.events.drain()
        events = self.say(turn("t1", 0, 900, "What time is it?"), "attend")
        self.assertEqual(events[0]["decision"]["label"], "ignore")

    def test_a_non_boolean_is_refused(self):
        with self.assertRaises(ContractError):
            SpeechEvents().media("on")


class MediaControlTests(unittest.TestCase):
    class Recorder:
        def __init__(self):
            self.calls = []

        def reply(self, phase):
            self.calls.append(("reply", phase))

        def media(self, on):
            if type(on) is not bool:
                raise PrototypeError("invalid media mode")
            self.calls.append(("media", on))

    def test_media_lines_are_forwarded_and_junk_is_skipped(self):
        from rightyo.tool import _read_control

        read, write = os.pipe()
        os.write(
            write,
            b'{"media": true}\n'
            b'{"media": "yes"}\n'
            b'{"media": false, "reply": "ended"}\n'
            b'{"reply": "started"}\n'
            b'{"media": false}\n',
        )
        os.close(write)
        recorder = self.Recorder()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            _read_control(read, recorder)
        self.assertEqual(recorder.calls, [("media", True), ("reply", "started"), ("media", False)])
        self.assertEqual(errors.getvalue().count("skipped"), 2)

    def test_the_controller_forwards_to_its_publisher_or_ignores_without_one(self):
        events = SpeechEvents()
        controller = SimpleNamespace(_lock=threading.RLock(), _events=events)
        PrototypeController.media(controller, True)
        self.assertIs(events._media, True)
        PrototypeController.media(SimpleNamespace(_lock=threading.RLock(), _events=None), True)
        with self.assertRaises(PrototypeError):
            PrototypeController.media(controller, 1)


if __name__ == "__main__":
    unittest.main()
