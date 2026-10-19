"""An answer to the assistant's own question is a follow-up cue (rightyo#199); synthetic only."""

from __future__ import annotations

import contextlib
import io
import os
import unittest

from test_conversation import SESSION, decided, turn

from rightyo.addressedness import answer_shaped
from rightyo.contracts import Addressing, ContractError, Conversation
from rightyo.tool_events import SpeechEvents

NAMES = Addressing.from_dict({"names": ["Jarvis"], "variants": {}})
MODE = Conversation(window_ms=60000, require_cue=True)


class AnswerShapeTests(unittest.TestCase):
    def test_yes_no_and_go_ahead_openers_are_answers(self):
        for text in (
            "Yes, please do both",
            "yeah",
            "Yep, the blue one",
            "Sure.",
            "No, leave it",
            "Nope",
            "Okay, send it",
            "OK",
            "Go ahead",
            "Do it",
            "Sounds good to me",
        ):
            with self.subTest(text=text):
                self.assertTrue(answer_shaped(text))

    def test_other_openings_are_not(self):
        for text in ("", "Good boy", "The blue folder", "I guess so", "Not now", "Go away"):
            with self.subTest(text=text):
                self.assertFalse(answer_shaped(text))


class AnswerCueTests(unittest.TestCase):
    def follow_up(self, text, *, asked, started_again=False):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=MODE, addressing=NAMES)
        request = turn("t1", 0, 900, "Jarvis, tidy the shared folder")
        events.transcript(request, 900)
        events.decision(decided(request, "attend"), 1000)
        events.reply("started", 2000)
        events.reply("ended", 4000, asked=asked)
        if started_again:
            events.reply("started", 4500)
            events.reply("ended", 5000)
        answer = turn("t2", 6000, 6800, text)
        events.transcript(answer, 6800)
        events.decision(decided(answer, "uncertain", attend=0.6), 6900)
        return sum(e["type"] == "request" for e in events.drain()) == 2

    def test_an_answer_after_a_question_is_a_follow_up(self):
        self.assertTrue(self.follow_up("Yes, keep both copies", asked=True))
        self.assertTrue(self.follow_up("Go ahead", asked=True))

    def test_without_the_question_it_still_needs_a_cue(self):
        self.assertFalse(self.follow_up("Yes, keep both copies", asked=False))

    def test_a_later_reply_that_asked_nothing_clears_it(self):
        self.assertFalse(self.follow_up("Yes, keep both copies", asked=True, started_again=True))

    def test_talk_after_the_question_that_is_no_answer_still_needs_a_cue(self):
        self.assertFalse(self.follow_up("Good boy, sit", asked=True))

    def requests(self, steps):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=MODE, addressing=NAMES)
        for step in steps:
            if step[0] == "reply":
                events.reply(step[1], step[2], **step[3])
            else:
                _, name, start, end, text, label, attend = step
                current = turn(name, start, end, text)
                events.transcript(current, end)
                events.decision(decided(current, label, attend=attend), end + 100)
        return [e["turn"]["text"] for e in events.drain() if e["type"] == "request"]

    def test_a_question_does_not_outlive_its_conversation(self):
        asked = [
            ("turn", "t1", 0, 900, "Jarvis, tidy the shared folder", "attend", None),
            ("reply", "started", 2000, {}),
            ("reply", "ended", 4000, {"asked": True}),
        ]
        pet = ("turn", "t3", 305000, 305800, "Okay buddy, dinner time", "uncertain", 0.6)
        # Lapsed, then a new named request that got no spoken reply.
        renewed = ("turn", "t2", 300000, 300900, "Jarvis, lights off", "attend", None)
        self.assertEqual(self.requests([*asked, renewed, pet])[-1], "Jarvis, lights off")
        # Ended by a closing inside the window, then the same.
        closing = ("turn", "c1", 6000, 6500, "that's all, thanks", "uncertain", 0.6)
        self.assertEqual(self.requests([*asked, closing, renewed, pet])[-1], "Jarvis, lights off")

    def test_the_answer_itself_uses_the_question_up(self):
        steps = [
            ("turn", "t1", 0, 900, "Jarvis, tidy the shared folder", "attend", None),
            ("reply", "started", 2000, {}),
            ("reply", "ended", 4000, {"asked": True}),
            ("turn", "t2", 6000, 6800, "Yes, keep both", "uncertain", 0.6),
            ("turn", "t3", 9000, 9800, "Okay buddy, sit", "uncertain", 0.6),
        ]
        self.assertEqual(self.requests(steps)[-1], "Yes, keep both")

    def test_asked_only_rides_on_an_ended_reply(self):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=MODE)
        for bad in (("started", True), ("ended", "yes"), ("ended", 1)):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                events.reply(bad[0], 100, asked=bad[1])


class AnswerControlTests(unittest.TestCase):
    class Recorder:
        def __init__(self):
            self.calls = []

        def reply(self, phase, **fields):
            self.calls.append((phase, fields))

        def media(self, on):
            self.calls.append(("media", on))

    def test_the_asked_flag_is_read_from_an_ended_reply_only(self):
        from rightyo.tool import _read_control

        read, write = os.pipe()
        os.write(
            write,
            b'{"reply": "ended", "asked": true}\n'
            b'{"reply": "ended", "asked": false}\n'
            b'{"reply": "started", "asked": true}\n'
            b'{"reply": "ended", "asked": "yes"}\n'
            b'{"reply": "ended", "asked": true, "media": true}\n'
            b'{"reply": "ended"}\n',
        )
        os.close(write)
        recorder = self.Recorder()
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            _read_control(read, recorder)
        self.assertEqual(
            recorder.calls,
            [("ended", {"asked": True}), ("ended", {"asked": False}), ("ended", {})],
        )
        self.assertEqual(errors.getvalue().count("skipped"), 3)


class AnswerControllerTests(unittest.TestCase):
    def test_the_controller_forwards_asked_only_when_set(self):
        import threading
        from types import SimpleNamespace

        from rightyo.prototype import PrototypeController, PrototypeError

        calls = []
        controller = SimpleNamespace(
            _lock=threading.RLock(),
            _events=object(),
            _event_terminal=False,
            _stop=threading.Event(),
            _phase="listening",
            _runner=None,
            _publish=lambda method, value, **fields: calls.append((method, value, fields)),
        )
        PrototypeController.reply(controller, "ended", asked=True)
        PrototypeController.reply(controller, "ended")
        self.assertEqual(calls, [("reply", "ended", {"asked": True}), ("reply", "ended", {})])
        for phase, asked in (("started", True), ("ended", 1)):
            with self.subTest(phase=phase), self.assertRaises(PrototypeError):
                PrototypeController.reply(controller, phase, asked=asked)


if __name__ == "__main__":
    unittest.main()
