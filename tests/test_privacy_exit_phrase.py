"""#350: match a named privacy exit at the end of an authored owner turn."""

import unittest

import test_duck_mode as duck

from rightyo.tool_events import SpeechEvents


class NamedPrivacyExitTests(unittest.TestCase):
    def setUp(self):
        self.events = SpeechEvents()
        self.events.duck(True, now_ms=0)
        self.events.start(
            duck.SESSION, now_ms=0, addressing=duck.OURS, priority=duck.enrolled(), duck=True
        )
        self.events.drain()

    def hear(self, text, start):
        turn = duck.heard(f"turn-{start}", start, start + 900, text)
        self.events.transcript(turn, turn.end_ms)
        return self.events.drain()

    def test_podcast_request_is_silent_in_private_and_published_in_public(self):
        text = "Jarvis, start listening to the podcast."
        self.assertEqual(self.hear(text, 100), [])
        self.assertTrue(self.events.ducked)
        self.events.duck(False, now_ms=1500)
        (transcript,) = self.hear(text, 2000)
        self.assertEqual(transcript["type"], "transcript")
        self.assertEqual(transcript["turn"]["text"], text)

    def test_other_trailing_words_keep_the_turn_private(self):
        tails = ("and turn it up", "um", "okay, please", "to start listening")
        for index, tail in enumerate(tails):
            with self.subTest(tail=tail):
                self.assertEqual(self.hear(f"Jarvis, start listening {tail}.", index * 1000), [])

    def test_allowed_trailing_words_still_exit(self):
        for index, word in enumerate(("now", "please", "again")):
            with self.subTest(word=word):
                (event,) = self.hear(f"Jarvis, start listening, {word}!", index * 1000)
                self.assertEqual((event["type"], event["action"]), ("privacy", "exit"))
