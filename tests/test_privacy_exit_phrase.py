"""#350: match a privacy exit at the end of an authored owner turn."""

import unittest
from dataclasses import replace
from unittest.mock import patch

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

    def test_forget_between_name_and_remainder_revokes_the_name_approval(self):
        self.assertEqual(self.hear("Jarvis,", 0), [])
        self.assertEqual(self.events._private_name_end, 900)
        self.events.forget_speakers(now_ms=950)
        with patch.object(self.events, "_exit_approval", wraps=self.events._exit_approval) as owner:
            self.assertEqual(self.hear("privacy off.", 1000), [])
            owner.assert_not_called()
        self.assertTrue(self.events.ducked)

    def test_late_playback_within_name_margin_revokes_split(self):
        for start, allowed in ((950, False), (1200, False), (1201, True)):
            with self.subTest(playback=start):
                self.start(duck.OURS)
                self.assertEqual(self.hear("Jarvis,", 0), [])
                self.events.reply("started", now_ms=start)
                self.events.reply("ended", now_ms=start + 50)
                self.assertEqual(
                    [e["action"] for e in self.hear("privacy off.", 2500)],
                    ["exit"] if allowed else [],
                )

    def test_split_vocative_filler_before_ellipsis(self):
        cases = (("please", True), ("um, well", True), ("please; well", False), ("really", False))
        for dots in ("...", "…"):
            self.start(duck.OURS)
            for i, (prefix, allowed) in enumerate(cases):
                with self.subTest(prefix=prefix, dots=dots):
                    at = 5000 * i
                    rest = f"{prefix}{dots} privacy off."
                    expected = ["exit"] if allowed else []
                    single = self.hear("Jarvis, " + rest, at)
                    self.assertEqual([e["action"] for e in single], expected)
                    self.assertEqual(self.hear("Jarvis,", at + 1000), [])
                    self.assertEqual([e["action"] for e in self.hear(rest, at + 2500)], expected)

    def test_questions_cannot_start_or_complete_an_exit(self):
        self.assertEqual(self.hear("Did you hear Jarvis start listening?  ", 0), [])
        self.assertEqual(self.hear("Jarvis?", 1000), [])
        self.assertEqual(self.hear("start listening.", 2000), [])
        self.assertEqual(self.hear("Jarvis,", 3000), [])
        self.assertEqual(self.hear("start listening, please?", 4000), [])
        self.assertEqual(self.hear("start listening.", 5000), [])

    def test_question_marks_and_unpunctuated_questions_in_the_exit_sentence_stay_private(self):
        for index, text in enumerate(
            (
                "Did you hear Jarvis start listening?!",
                "She asked, “did Jarvis start listening?”  ",
                "(Jarvis, start listening？)  ",
                "¿Jarvis, start listening",
                "Did you hear Jarvis start listening",
                "Well, could you hear Jarvis start listening",
                "Yesterday did you hear Jarvis start listening",
                "After lunch did anyone hear Jarvis start listening",
                "After lunch did Alice hear Jarvis start listening",
                "She asked did Jarvis start listening",
            )
        ):
            with self.subTest(text=text):
                self.assertEqual(self.hear(text, index * 1000), [])

    def test_earlier_sentences_do_not_question_the_exit(self):
        # #350 round 5: question punctuation and interrogatives belong to their sentence.
        for intro in (
            "Really?",
            "After lunch did anyone hear that.",
            "After lunch did Alice call.",
            "The train arrived\nDid the meeting end\n",
            "When I finish, I’ll call.",
            "What a relief!",
        ):
            with self.subTest(intro=intro):
                self.assertEqual(
                    privacy_phrase(f"{intro} Jarvis, start listening.", duck.OURS), {"exit"}
                )
                self.assertFalse(privacy_phrase(f"{intro} ¿Jarvis, start listening.", duck.OURS))

    def test_auxiliary_question_after_the_name_stays_private(self):
        for word in "did do does can could will would is are should may might".split():
            with self.subTest(word=word):
                self.assertFalse(
                    privacy_phrase(f"Ready. Jarvis, {word} Alice start listening", duck.OURS)
                )

    def test_sentence_break_between_phrase_words_refuses_single_and_split_exit(self):
        for i, boundary in enumerate(".!?;\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
            with self.subTest(boundary=boundary):
                at = 4000 * i
                phrase = f"start{boundary} listening."
                self.assertEqual(self.hear("Jarvis, " + phrase, at), [])
                self.assertEqual(self.hear("Jarvis,", at + 1000), [])
                self.assertEqual(self.hear(phrase, at + 2000), [])
                self.assertEqual(self.hear(f"Jarvis{boundary} privacy off.", at + 3000), [])

    def test_period_inside_exact_configured_name(self):
        self.start(Addressing(("A.B",)))
        self.assertEqual([e["action"] for e in self.hear("A.B, privacy off.", 0)], ["exit"])
        for index, text in enumerate(("A. B, privacy off.", "A.B; privacy off.", "A.B?")):
            self.assertEqual(self.hear(text, (index + 1) * 1000), [])

    def test_ellipsis_inside_exact_configured_name(self):
        for dots in ("...", "…"):
            with self.subTest(dots=dots):
                self.start(Addressing(("A...B",)))
                self.assertEqual(
                    [e["action"] for e in self.hear(f"A{dots}B, privacy off.", 0)], ["exit"]
                )
                self.assertEqual(self.hear(f"A{dots}B,", 1000), [])
                self.assertEqual([e["action"] for e in self.hear("privacy off.", 2000)], ["exit"])

    def test_only_a_vocative_ellipsis_can_join_address_and_phrase(self):
        for dots in ("...", "…"):
            with self.subTest(dots=dots):
                self.start(duck.OURS)
                rejected = (
                    "Jarvis; privacy off.",
                    "Jarvis. Privacy off.",
                    f"Jarvis{dots}\nprivacy off.",
                    f"Jarvis; {dots} privacy off.",
                    f"Jarvis, privacy{dots}off.",
                )
                accepted = (
                    f"Jarvis{dots} privacy off.",
                    f"JARVIS{dots} PRIVACY OFF!!",
                    f"Jarvis, privacy off{dots}",
                )
                with patch.object(
                    self.events, "_exit_approval", wraps=self.events._exit_approval
                ) as owner:
                    for i, text in enumerate(rejected):
                        self.assertEqual(self.hear(text, i * 1000), [])
                    owner.assert_not_called()
                    for i, text in enumerate(accepted, start=len(rejected)):
                        owner.reset_mock()
                        self.assertEqual([e["action"] for e in self.hear(text, i * 1000)], ["exit"])
                        owner.assert_called_once()

    def test_earlier_negated_question_or_refused_address_does_not_veto_exit(self):
        for name, text in (
            ("Jarvis", "Don't you agree? Jarvis, privacy off."),
            ("Hey", "Hey? Hey, privacy off."),
        ):
            with self.subTest(text=text):
                self.start(Addressing((name,)))
                self.assertEqual([event["action"] for event in self.hear(text, 0)], ["exit"])

    def test_name_fragment_does_not_suppress_status(self):
        self.assertEqual(
            [event["action"] for event in self.hear("Are you recording? Jarvis.", 0)], ["status"]
        )

    def test_complete_exit_wins_even_when_the_phrase_is_another_name(self):
        self.start(Addressing(("Hey", "Privacy Off")))
        self.assertEqual([event["action"] for event in self.hear("Hey, privacy off.", 0)], ["exit"])

    def test_configured_name_uses_addressing_casefold(self):
        self.start(Addressing(("Strasse",)))
        self.assertEqual(
            [event["action"] for event in self.hear("Straße, privacy off.", 0)], ["exit"]
        )

    def test_split_after_a_sentence_opening_name_matches_the_single_turn(self):
        for i, (intro, allowed) in enumerate(
            (
                ("We’re done. ", True),
                ("Really? ", True),
                ("I heard... ", True),
                ("I heard… ", True),
                ("Don't... ", True),
                ("Don't… ", True),
                ("Did Alice hear ", False),
                ("We're done, ", False),
                ("We're done. ¿", False),
            )
        ):
            with self.subTest(intro=intro):
                name = f"{intro}Jarvis,"
                at = i * 4000
                self.assertEqual(bool(self.hear(name + " privacy off.", at)), allowed)
                self.assertEqual(self.hear(name, at + 1000), [])
                events = self.hear("Privacy off.", at + 2000)
                self.assertEqual([e["action"] for e in events], ["exit"] if allowed else [])

    def test_address_starts_a_sentence(self):
        for index, sep in enumerate((".", "!", ";", "\n", "...", "…")):
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
                (event,) = self.hear("May I ask. Will, start listening.", 4000)
                self.assertEqual(event["action"], "exit")
                self.assertEqual(self.hear("May I ask Will, start listening.", 5000), [])

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

    def test_acoustic_split_clock_is_private_processor_evidence(self):
        turn = replace(duck.heard("name", 100, 1900, "Jarvis"), acoustic_span_ms=(200, 1200))
        self.assertNotIn("acoustic_span_ms", turn.to_dict())
        with self.assertRaises(ContractError):
            Turn.from_dict({**turn.to_dict(), "acoustic_span_ms": (200, 1200)})
        self.assertNotIn("exit_timings", turn.to_dict())
        with self.assertRaises(ContractError):
            Turn.from_dict({**turn.to_dict(), "exit_timings": [[0, 6, 100, 200]]})
        self.assertNotIn("exit_available", turn.to_dict())
        with self.assertRaises(ContractError):
            Turn.from_dict({**turn.to_dict(), "exit_available": True})
        self.assertNotIn("defer_exit", turn.to_dict())
        with self.assertRaises(ContractError):
            Turn.from_dict({**turn.to_dict(), "defer_exit": True})
