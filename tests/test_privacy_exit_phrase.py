"""#350: match a privacy exit at the end of an authored owner turn."""

import unittest
from dataclasses import replace

import test_duck_mode as duck

from rightyo.contracts import Addressing, ContractError, Turn
from rightyo.tool_events import SpeechEvents, privacy_phrase


class NamedPrivacyExitTests(unittest.TestCase):
    def setUp(self):
        self.start(duck.OURS)

    def start(self, addressing):
        self.events = SpeechEvents()
        self.events.duck(True, now_ms=0)
        self.events.start(
            duck.SESSION, now_ms=0, addressing=addressing, priority=duck.enrolled(), duck=True
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

    def test_question_marks_anywhere_and_unpunctuated_questions_stay_private(self):
        for index, text in enumerate(
            (
                "Did you hear Jarvis start listening?!",
                "She asked, “did Jarvis start listening?”  ",
                "(Jarvis, start listening？)  ",
                "¿Jarvis, start listening",
                "Really? Jarvis, start listening.",
                "Did you hear Jarvis start listening",
                "Well, could you hear Jarvis start listening",
                "Yesterday did you hear Jarvis start listening",
                "After lunch did anyone hear Jarvis start listening",
                "After lunch did anyone hear that. Jarvis, start listening",
                "She asked did Jarvis start listening",
            )
        ):
            with self.subTest(text=text):
                self.assertEqual(self.hear(text, index * 1000), [])

    def test_address_starts_a_sentence(self):
        for index, sep in enumerate((".", "!", ";", "\n")):
            with self.subTest(sep=sep):
                (event,) = self.hear(
                    f"That's enough of that{sep} Well, Jarvis, I'm back", index * 1000
                )
                self.assertEqual(event["action"], "exit")
        for sep in (" ", ", ", ": ", " — "):
            with self.subTest(sep=sep):
                self.assertFalse(
                    privacy_phrase(f"That's enough of that{sep}Jarvis I'm back", duck.OURS)
                )
        for text in (
            "Yesterday I heard Jarvis start listening.",
            "When I give the signal, Jarvis, start listening.",
            "Jarvis сказал «start listening».",
            "Jarvis, privacy off 谢谢",
            "Jarvis, privacy off, क्",
        ):
            with self.subTest(text=text):
                self.assertFalse(privacy_phrase(text, duck.OURS))

    def test_only_the_eight_fillers_can_surround_the_address(self):
        for word in ("um", "uh", "ok", "okay", "hey", "so", "please", "well"):
            self.assertEqual(privacy_phrase(f"{word} Jarvis {word} I'm back", duck.OURS), {"exit"})
        for word in ("umm", "uhm", "er", "erm", "ah", "hmm", "mm", "oh", "now", "again"):
            for text in (f"{word} Jarvis, privacy off", f"Jarvis {word} privacy off"):
                with self.subTest(text=text):
                    self.assertFalse(privacy_phrase(text, duck.OURS))

    def test_unknown_words_cannot_arm_or_complete_a_split(self):
        for index, word in enumerate(("谢谢", "сказал", "قال", "क्", "7", "\u0301")):
            with self.subTest(word=word):
                at = index * 4000
                self.assertEqual(self.hear(f"Jarvis {word}", at), [])
                self.assertEqual(self.hear("I'm back", at + 1000), [])
                self.assertEqual(self.hear("Jarvis", at + 2000), [])
                self.assertEqual(self.hear(f"I'm back {word}", at + 3000), [])

    def test_question_word_as_name_or_variant(self):
        for naming in (
            Addressing(("Will", "May")),
            Addressing(("Jarvis",), (("Jarvis", ("Will", "May")),)),
        ):
            with self.subTest(naming=naming):
                self.start(naming)
                (event,) = self.hear("Will, start listening.", 0)
                self.assertEqual(event["action"], "exit")
                self.assertEqual(self.hear("May,", 1000), [])
                (event,) = self.hear("I'm back.", 2000)
                self.assertEqual(event["action"], "exit")
                self.assertEqual(self.hear("Will, start listening?", 3000), [])
                self.assertEqual(self.hear("May I ask. Will, start listening.", 4000), [])

    def test_marked_questions_disarm_each_split_fragment(self):
        for index, (name, rest) in enumerate(
            (
                ("“Jarvis?!”", "start listening"),
                ("Jarvis", "[start listening？]"),
                ("Jarvis", "¿start listening"),
            )
        ):
            with self.subTest(name=name, rest=rest):
                at = index * 3000
                self.assertEqual(self.hear(name, at), [])
                self.assertEqual(self.hear(rest, at + 1000), [])
                self.assertEqual(self.hear("start listening", at + 2000), [])

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
