"""Automatic meeting attention around other voices (#283); authored turns/controls only."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_conversation import SESSION, decided, turn
from test_follow_up_cue import NAMES

from rightyo.contracts import (
    MAX_CONVERSATION_WINDOW_MS,
    Addressing,
    ContractError,
    Conversation,
    SpeakerPriority,
)
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.providers import ConfiguredPriorityProvider
from rightyo.tool_events import SpeechEvents

MODE = Conversation(window_ms=60000, cue_when_others_ms=5000)
OWNER = ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",)))


class OthersCueTests(unittest.TestCase):
    def replay(
        self,
        *,
        mode=MODE,
        other=True,
        speaker="Speaker B",
        playback=None,
        text="The blue folder",
        start=5000,
        attend=0.6,
        asked=False,
        continuation=False,
        priority=OWNER,
        addressing=NAMES,
        label="uncertain",
        current_speaker="Speaker A",
        inferred=False,
        engage=True,
        meeting=False,
        media=False,
        ask_anywhere=True,
        provenance="authored-fixture",
    ):
        notes = []
        events = SpeechEvents(report=notes.append)
        events.start(
            SESSION,
            now_ms=0,
            conversation=mode,
            addressing=addressing,
            priority=priority,
            meeting_ask_anywhere=ask_anywhere,
        )
        first = turn("request", 0, 900, "Jarvis, tidy the shared folder", provenance=provenance)
        events.transcript(first, 900)
        events.decision(decided(first, "attend" if engage else "ignore"), 1000)
        if playback is not None:
            events.reply("started", 1500)
        if playback == "ended":
            events.reply("ended", 2000)
        if other:
            # This turn spans playback even when it starts before playback's margin.
            second = turn(
                "other", 1100, 2400, "Lovely morning", speaker=speaker, provenance=provenance
            )
            events.transcript(second, 2400, expect_decision=False)
        if (asked or continuation) and playback is None:
            # The person is heard before the reply and outside its echo margin, so the
            # question/continuation tests really do exercise others-present mode.
            events.reply("started", 3000)
        if asked or continuation:
            events.reply("ended", 4000, asked=asked)
        events.meeting(meeting)
        events.media(media)
        current = turn(
            "follow", start, start + 800, text, speaker=current_speaker, provenance=provenance
        )
        events.transcript(current, current.end_ms, inferred=inferred)
        event = decided(current, label, attend=attend, confidence=attend)
        if continuation:
            event = replace(event, decision=replace(event.decision, attention_choice="attend"))
        events.decision(event, current.end_ms + 100)
        output = events.drain()
        self.output = output
        return [e for e in output if e["type"] == "request"], notes, events

    def test_others_present_uncued_is_refused(self):
        requests, notes, _ = self.replay()
        self.assertEqual(len(requests), 1)
        self.assertIn("reason=no_cue_others", notes[-1])
        self.assertNotIn("Speaker", notes[-1])
        self.assertNotIn("folder", notes[-1])

    def test_naming_the_assistant_forms(self):
        self.assertEqual(len(self.replay(text="Jarvis, the blue folder")[0]), 2)

    def test_the_name_must_match_configured_addressing_including_variants(self):
        names = Addressing.from_names(["Jarvis"], {"Jarvis": ["Jervis"]})
        self.assertEqual(len(self.replay(addressing=names, text="Jervis, the blue folder")[0]), 2)
        for addressing, text in (
            (names, "Helper, the blue folder"),
            (None, "Jarvis, the blue folder"),
        ):
            with self.subTest(addressing=addressing):
                requests, notes, _ = self.replay(addressing=addressing, text=text)
                self.assertEqual(len(requests), 1)
                self.assertIn("reason=no_cue_others", notes[-1])

    def test_owner_alone_forms_as_before(self):
        self.assertEqual(len(self.replay(other=False)[0]), 2)

    def test_same_and_unattributed_voices_never_count(self):
        for speaker in ("Speaker A", None):
            with self.subTest(speaker=speaker):
                self.assertEqual(len(self.replay(speaker=speaker)[0]), 2)

    def test_playback_overlapping_voice_never_counts(self):
        for phase in ("started", "ended"):
            with self.subTest(phase=phase):
                self.assertEqual(len(self.replay(playback=phase)[0]), 2)

    def test_window_lapse_forms_again(self):
        self.assertEqual(len(self.replay(start=7400)[0]), 1)
        self.assertEqual(len(self.replay(start=7401)[0]), 2)

    def test_static_require_cue_wins(self):
        mode = replace(MODE, require_cue=True)
        for other in (True, False):
            requests, notes, _ = self.replay(mode=mode, other=other)
            self.assertEqual(len(requests), 1)
            self.assertIn("reason=no_cue ", notes[-1])

    def test_default_off_is_unchanged(self):
        self.assertEqual(len(self.replay(mode=Conversation(window_ms=60000))[0]), 2)

    def assert_name_required(self, **options):
        # Static require_cue may pass on a waiver or generic cue; the presence gate must
        # still demand the assistant's name with that setting either on or off.
        mode = options.pop("mode", MODE)
        for static in (False, True):
            with self.subTest(require_cue=static):
                requests, notes, _ = self.replay(mode=replace(mode, require_cue=static), **options)
                self.assertEqual(len(requests), 1)
                self.assertIn("reason=no_cue_others", notes[-1])
                named = {**options, "text": "Jarvis, the blue folder"}
                self.assertEqual(
                    len(self.replay(mode=replace(mode, require_cue=static), **named)[0]), 2
                )

    def test_continuation_cue_never_waives_the_name_with_others_present(self):
        self.assert_name_required(mode=replace(MODE, continuation_cue=True), continuation=True)

    def test_cue_min_attend_never_waives_the_name_with_others_present(self):
        for attend in (0.6, 0.8):
            with self.subTest(attend=attend):
                self.assert_name_required(mode=replace(MODE, cue_min_attend=0.6), attend=attend)

    def test_an_answer_to_the_assistants_question_still_needs_the_name(self):
        self.assert_name_required(asked=True, text="Yes, the blue folder")

    def test_generic_follow_up_cues_still_need_the_name(self):
        for text in (
            "You mean the green folder?",
            "Can you make it shorter?",
            "the other one",
            "and eggs",
        ):
            with self.subTest(text=text):
                self.assert_name_required(text=text)

    def test_owner_alone_keeps_the_static_cue_waivers(self):
        self.assertEqual(
            len(
                self.replay(mode=replace(MODE, require_cue=True, cue_min_attend=0.6), other=False)[
                    0
                ]
            ),
            2,
        )
        self.assertEqual(
            len(
                self.replay(
                    mode=replace(MODE, require_cue=True, continuation_cue=True),
                    continuation=True,
                    other=False,
                )[0]
            ),
            2,
        )
        self.assertEqual(
            len(
                self.replay(
                    mode=replace(MODE, require_cue=True),
                    asked=True,
                    text="Yes, the blue folder",
                    other=False,
                )[0]
            ),
            2,
        )

    def test_per_turn_owner_and_trusted_roles_do_not_count(self):
        from rightyo.contracts import SpeakerPriority

        class Roles:
            per_turn = True
            priority = SpeakerPriority()

            def assign(self, state):
                return {}

            def role_for(self, current):
                return self.other_role if current.speaker_id == "Speaker B" else "owner"

        for role, count in (("owner", 2), ("trusted", 2), ("participant", 1)):
            roles = Roles()
            roles.other_role = role
            with self.subTest(role=role):
                self.assertEqual(len(self.replay(priority=roles)[0]), count)

    def test_later_voice_does_not_hide_prior_voice_from_a_pending_decision(self):
        _, _, events = self.replay(text="Jarvis, the blue folder")
        other = turn("other2", 7000, 7400, "Lovely afternoon", speaker="Speaker B")
        events.transcript(other, 7400, expect_decision=False)
        pending = turn("pending", 8000, 8400, "The green folder")
        events.transcript(pending, 8400)
        later = turn("later", 9000, 9400, "Lovely evening", speaker="Speaker B")
        events.transcript(later, 9400, expect_decision=False)
        events.decision(decided(pending, "uncertain", attend=0.6), 9500)
        self.assertFalse(any(e["type"] == "request" for e in events.drain()))

    def test_voice_heard_before_engagement_is_retained(self):
        for before_request in (True, False):
            with self.subTest(before_request=before_request):
                events = SpeechEvents()
                events.start(SESSION, now_ms=0, conversation=MODE, addressing=NAMES, priority=OWNER)
                other = turn("other", 0, 400, "Lovely morning", speaker="Speaker B")
                request = turn("request", 500, 900, "Jarvis, tidy the folder")
                if before_request:
                    events.transcript(other, 400, expect_decision=False)
                events.transcript(request, 900)
                if not before_request:
                    other = turn("other", 1100, 1400, "Lovely morning", speaker="Speaker B")
                    events.transcript(other, 1400, expect_decision=False)
                events.decision(decided(request, "attend"), 1500)
                events.drain()
                more = turn("more", 2000, 2400, "The blue folder")
                events.transcript(more, 2400)
                events.decision(decided(more, "uncertain", attend=0.6), 2500)
                self.assertFalse(any(e["type"] == "request" for e in events.drain()))

    def test_direct_attend_without_name_is_blocked_by_others_present(self):
        for engage in (False, True):
            with self.subTest(engage=engage):
                requests, notes, _ = self.replay(label="attend", attend=0.98, engage=engage)
                self.assertEqual(len(requests), int(engage))
                evidence = self.output[-1]["decision"]
                self.assertEqual(evidence["label"], "ignore")
                self.assertIs(evidence["others_present"], True)
                self.assertIn("attention_blocked reason=others_present", notes)

    def test_direct_attend_with_owner_and_name_forms(self):
        for engage in (False, True):
            with self.subTest(engage=engage):
                requests, _, _ = self.replay(
                    label="attend",
                    attend=0.98,
                    text="The blue folder, Jarvis",
                    engage=engage,
                )
                self.assertEqual(len(requests), 1 + int(engage))
                self.assertNotIn("follow_up", requests[-1]["decision"])

    def test_participant_inferred_unlabelled_and_no_roles_never_pass(self):
        for options in (
            {"current_speaker": "Speaker B"},
            {"inferred": True},
            {"current_speaker": None},
            {"priority": None},
        ):
            for label in ("attend", "uncertain"):
                with self.subTest(options=options, label=label):
                    requests, _, _ = self.replay(
                        label=label,
                        text="The blue folder, Jarvis",
                        **options,
                    )
                    self.assertEqual(len(requests), 1)
                    if options.get("current_speaker") == "Speaker B":
                        attention = next(
                            e for e in reversed(self.output) if e["type"] == "attention"
                        )
                        self.assertEqual(attention["decision"]["role"], "participant")

    def test_bound_owner_talking_to_a_participant_is_still_blocked(self):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner",)))
        roles.source = SimpleNamespace(
            binding=lambda label: ("bound", "owner" if label == "Speaker A" else "guest")
        )
        requests, notes, _ = self.replay(
            priority=roles, label="attend", attend=0.98, provenance="diarization-timeline"
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(self.output[-1]["decision"]["role"], "owner")
        self.assertIs(self.output[-1]["decision"]["others_present"], True)
        self.assertIn("attention_blocked reason=others_present", notes)

    def test_unlabelled_turn_never_inherits_owner_authority_from_a_role_provider(self):
        class Roles:
            per_turn = True
            priority = SpeakerPriority()

            def assign(self, state):
                return {}

            def role_for(self, current):
                return "participant" if current.speaker_id == "Speaker B" else "owner"

        for meeting in (False, True):
            with self.subTest(meeting=meeting):
                requests, _, _ = self.replay(
                    label="attend",
                    attend=0.98,
                    text="Jarvis, the green folder",
                    current_speaker=None,
                    priority=Roles(),
                    meeting=meeting,
                )
                self.assertEqual(len(requests), 1)

    def test_direct_attend_forms_after_presence_window_lapses(self):
        self.assertEqual(len(self.replay(label="attend", attend=0.98, start=7400)[0]), 1)
        self.assertEqual(len(self.replay(label="attend", attend=0.98, start=7401)[0]), 2)

    def test_later_engagement_cannot_erase_presence_for_a_pending_direct_attend(self):
        events = SpeechEvents()
        events.start(SESSION, now_ms=0, conversation=MODE, addressing=NAMES, priority=OWNER)
        first = turn("first", 0, 900, "Jarvis, open the blue folder")
        events.transcript(first, 900)
        events.decision(decided(first, "attend"), 1000)
        other = turn("other", 1100, 2400, "Lovely morning", speaker="Speaker B")
        events.transcript(other, 2400, expect_decision=False)
        pending = turn("pending", 6000, 6800, "The green folder")
        events.transcript(pending, 6800)
        # The presence window has elapsed by this later request; it can engage B.
        later = turn("later", 9000, 9800, "Jarvis, open the authored ledger", speaker="Speaker B")
        events.transcript(later, 9800)
        events.decision(decided(later, "attend"), 9900)
        self.assertEqual(events._engaged["speaker_id"], "Speaker B")
        events.drain()
        events.decision(decided(pending, "attend", attend=0.98), 10000)
        output = events.drain()
        self.assertFalse(any(e["type"] == "request" for e in output))
        self.assertIs(
            next(e for e in output if e["type"] == "attention")["decision"]["others_present"], True
        )

    def test_named_owner_abstention_reuses_meeting_promotion_not_a_follow_up(self):
        requests, _, _ = self.replay(text="The green folder, Jarvis", attend=0.1)
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[-1]["decision"]["others_present"], "ask_anywhere")
        self.assertNotIn("follow_up", requests[-1]["decision"])
        self.assertEqual(len(self.replay(text="The green folder, Jarvis", label="ignore")[0]), 1)
        self.assertEqual(
            len(self.replay(text="The green folder, Jarvis", ask_anywhere=False)[0]), 1
        )

    def test_explicit_meeting_and_media_refusals_keep_precedence(self):
        for meeting, media, reason in (
            (True, False, "meeting"),
            (False, True, "media"),
            (True, True, "meeting"),
        ):
            with self.subTest(meeting=meeting, media=media):
                self.replay(label="attend", attend=0.98, meeting=meeting, media=media)
                evidence = self.output[-1]["decision"]
                self.assertIs(evidence[reason], True)
                self.assertNotIn("others_present", evidence)

    def test_configuration_validation_and_round_trip(self):
        for bad in (-1, True, 1.5, "5000", None, MAX_CONVERSATION_WINDOW_MS + 1):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                Conversation.from_dict({"cue_when_others_ms": bad})
        for value in (0, 5000, MAX_CONVERSATION_WINDOW_MS):
            mode = Conversation.from_dict({"cue_when_others_ms": value})
            self.assertEqual(mode.cue_when_others_ms, value)
            raw = mode.to_dict()
            raw.pop("version")
            self.assertEqual(Conversation.from_dict(raw), mode)
            self.assertEqual("cue_when_others_ms" in raw, bool(value))


class OthersPresentSnapshotTests(unittest.TestCase):
    """Per-turn policy survives public pending decisions, not reusable speaker labels."""

    def pending(self, *, others=True, meeting=False, media=False, max_pending=128):
        self.notes = []
        events = SpeechEvents(report=self.notes.append, max_pending=max_pending, retention_ms=10000)
        events.meeting(meeting)
        events.media(media)
        events.start(
            SESSION, now_ms=0, conversation=MODE, addressing=NAMES, priority=OWNER, duck=True
        )
        events.drain()
        owner = turn("owner", 0, 900, "Jarvis, open the authored folder")
        events.transcript(owner, 900)
        events.decision(decided(owner, "attend"), 1000)
        events.drain()
        if others:
            participant = turn("participant", 1100, 2400, "Lovely morning", speaker="Speaker B")
            events.transcript(participant, 2400, expect_decision=False)
        current = turn("pending", 3000, 3800, "The green folder")
        events.transcript(current, 3800)
        events.drain()
        self.assertIn(current.utterance_id, events._pending)
        self.assertIs(events._others_present(current), others)
        return events, current

    def assert_blocked(self, events, current, *, now=5000, reason="others_present"):
        events.drain()
        events.decision(decided(current, "attend", attend=0.98, confidence=0.98), now)
        output = events.drain()
        self.assertFalse(any(e["type"] == "request" for e in output))
        evidence = next(e["decision"] for e in output if e["type"] == "attention")
        self.assertEqual(evidence["label"], "ignore")
        self.assertIs(evidence[reason], True)
        return evidence

    def test_pending_attend_stays_blocked_after_privacy_and_speaker_reset(self):
        events, current = self.pending()
        events.duck(True, now_ms=4000)
        events.duck(False, now_ms=4100)
        events.forget_speakers(now_ms=4200)
        self.assertEqual(list(events._cue_voices), [])
        evidence = self.assert_blocked(events, current)
        self.assertEqual(evidence["role"], "unknown")
        self.assertIn("attention_blocked reason=others_present", self.notes)
        self.assertNotIn(current.text, " ".join(self.notes))

    def test_other_reset_and_toggle_paths_keep_the_accepted_bar(self):
        for reset in (
            "forget",
            "forget_twice",
            "privacy_on",
            "privacy_cycle",
            "meeting",
            "media",
            "meeting_media_cycle",
        ):
            with self.subTest(reset=reset):
                events, current = self.pending()
                if reset.startswith("forget"):
                    events.forget_speakers(now_ms=4000)
                    if reset == "forget_twice":
                        events.forget_speakers(now_ms=4100)
                elif reset.startswith("privacy"):
                    events.duck(True, now_ms=4000)
                    if reset == "privacy_cycle":
                        events.duck(False, now_ms=4100)
                else:
                    if "meeting" in reset:
                        events.meeting(True)
                    if "media" in reset:
                        events.media(True)
                    if reset.endswith("cycle"):
                        events.meeting(False)
                        events.media(False)
                self.assertIs(events._others_present(current), True)
                self.assert_blocked(events, current)

    def test_existing_meeting_and_media_snapshots_keep_precedence_after_reset(self):
        for meeting, media, reason in (
            (True, False, "meeting"),
            (False, True, "media"),
            (True, True, "meeting"),
        ):
            with self.subTest(meeting=meeting, media=media):
                events, current = self.pending(meeting=meeting, media=media)
                events.meeting(False)
                events.media(False)
                events.forget_speakers(now_ms=4000)
                evidence = self.assert_blocked(events, current, reason=reason)
                self.assertNotIn("others_present", evidence)
                self.assertIs(events._others_present(current), True)

    def test_reset_clears_history_for_new_turns_but_not_the_old_verdict(self):
        events, current = self.pending()
        events.forget_speakers(now_ms=4000)
        self.assert_blocked(events, current)
        fresh = turn("fresh", 6000, 6800, "The blue folder")
        events.transcript(fresh, 6800)
        self.assertIs(events._others_present(fresh), False)
        events.decision(decided(fresh, "attend", attend=0.98), 6900)
        request = next(e for e in events.drain() if e["type"] == "request")
        self.assertNotIn("others_present", request["decision"])

    def test_a_turn_accepted_alone_does_not_acquire_a_bar_after_reset_or_later_speech(self):
        for reset in (False, True):
            with self.subTest(reset=reset):
                events, current = self.pending(others=False)
                if reset:
                    events.forget_speakers(now_ms=4000)
                later = turn("later", 4000, 4800, "Lovely morning", speaker="Speaker B")
                events.transcript(later, 4800, expect_decision=False)
                self.assertIs(events._others_present(current), False)
                events.decision(decided(current, "attend", attend=0.98), 5000)
                request = next(e for e in events.drain() if e["type"] == "request")
                self.assertNotIn("others_present", request["decision"])

    def test_private_turns_never_add_presence_snapshots_or_history(self):
        events, current = self.pending()
        snapshots = dict(events._others_turns)
        voices = list(events._cue_voices)
        events.duck(True, now_ms=4000)
        private = turn("private", 4100, 4800, "Synthetic private sentence", speaker="Speaker B")
        events.transcript(private, 4800)
        self.assertEqual(events._others_turns, snapshots)
        self.assertEqual(list(events._cue_voices), voices)
        events.duck(False, now_ms=4900)
        self.assert_blocked(events, current, now=5000)

    def test_presence_snapshot_expires_with_the_other_per_turn_snapshots(self):
        events, current = self.pending(meeting=True, media=True)
        events.forget_speakers(now_ms=4000)
        expiry = current.end_ms + events.retention_ms
        events.expire(expiry - 1)
        for snapshots in (events._others_turns, events._meeting_turns, events._media_turns):
            self.assertEqual(snapshots, {current.utterance_id: current.end_ms})
        self.assertIn(current.utterance_id, events._pending)
        events.expire(expiry)
        for snapshots in (events._others_turns, events._meeting_turns, events._media_turns):
            self.assertEqual(snapshots, {})
        self.assertEqual(events._pending, {})
        events.decision(decided(current, "attend", attend=0.98), expiry)
        self.assertFalse(any(e["type"] == "request" for e in events.drain()))

    def test_session_end_and_restart_discard_pending_verdicts_and_history(self):
        for phase in ("stopped", "cancelled", "error"):
            with self.subTest(phase=phase):
                events, current = self.pending()
                events.end(phase, now_ms=4000)
                self.assertEqual(events._others_turns, {})
                self.assertEqual(list(events._cue_voices), [])
                self.assertEqual(events._pending, {})
                events.drain()
                session = SESSION + "-" + phase
                events.start(session, now_ms=0, conversation=MODE, addressing=NAMES, priority=OWNER)
                events.drain()
                with self.assertRaises(ContractError):
                    events.decision(decided(current, "attend", attend=0.98), 1000)
                fresh = replace(current, session_id=session, start_ms=0, end_ms=800)
                events.transcript(fresh, 800)
                self.assertIs(events._others_present(fresh), False)
                events.decision(decided(fresh, "attend", attend=0.98), 900)
                request = next(e for e in events.drain() if e["type"] == "request")
                self.assertNotIn("others_present", request["decision"])

    def test_backlog_teardown_discards_the_pending_verdict_with_its_turn(self):
        events, current = self.pending(max_pending=5)
        with self.assertRaises(ContractError):
            for index in range(6):
                extra = turn(
                    f"extra-{index}",
                    4000 + index * 1000,
                    4800 + index * 1000,
                    "Authored extra turn",
                )
                events.transcript(extra, extra.end_ms, expect_decision=False)
        self.assertEqual(events._others_turns, {})
        self.assertEqual(list(events._cue_voices), [])
        self.assertEqual(events._pending, {})
        events.decision(decided(current, "attend", attend=0.98), 11000)
        self.assertEqual(events.drain(), [])
