"""#350: match a privacy exit at the end of an authored owner turn."""

import unittest
from dataclasses import replace

import test_duck_mode as duck

from rightyo.contracts import ContractError, Turn
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

    def test_split_follow_up_requires_the_phrase_to_end_the_turn(self):
        for index, tail in enumerate(("to the podcast", "okay, please", "again um")):
            with self.subTest(tail=tail):
                at = index * 3000
                self.assertEqual(self.hear("Jarvis,", at), [])
                self.assertEqual(self.hear(f"start listening {tail}", at + 1000), [])
                self.assertEqual(self.hear("start listening", at + 2000), [])

    def test_exact_split_exit_with_or_without_allowed_trailing_words(self):
        for index, tail in enumerate(("", " now", " please", " again, please!")):
            with self.subTest(tail=tail):
                at = index * 2000
                self.assertEqual(self.hear("Jarvis,", at), [])
                (event,) = self.hear(f"start listening{tail}", at + 1000)
                self.assertEqual((event["type"], event["action"]), ("privacy", "exit"))

    def test_questions_cannot_start_or_complete_an_exit(self):
        self.assertEqual(self.hear("Did you hear Jarvis start listening?  ", 0), [])
        self.assertEqual(self.hear("Jarvis?", 1000), [])
        self.assertEqual(self.hear("start listening.", 2000), [])
        self.assertEqual(self.hear("Jarvis,", 3000), [])
        self.assertEqual(self.hear("start listening, please?", 4000), [])
        self.assertEqual(self.hear("start listening.", 5000), [])

    def test_timing_loss_blocks_a_bound_owner_exit_and_preserves_status(self):
        turn = replace(
            duck.heard("partial", 0, 100, "Jarvis, start listening"), recognition_loss_end_ms=200
        )
        self.events.transcript(turn, turn.end_ms)
        self.assertEqual(self.events.drain(), [])
        status = replace(turn, utterance_id="status", text="Are you recording?")
        self.events.transcript(status, status.end_ms)
        self.assertEqual([event["action"] for event in self.events.drain()], ["status"])

        self.assertNotIn("recognition_loss_end_ms", turn.to_dict())
        with self.assertRaises(ContractError):
            Turn.from_dict({**turn.to_dict(), "recognition_loss_end_ms": 0})
