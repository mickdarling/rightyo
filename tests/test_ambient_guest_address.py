"""#326 request-stream replays with authored turns and synthetic binding results."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from test_conversation import SESSION, decided, turn

from rightyo.contracts import (
    Acknowledgement,
    Addressing,
    ContractError,
    Conversation,
    Dismissal,
    SpeakerPriority,
)
from rightyo.live_speaker_id import EnrolledRoles
from rightyo.tool_events import SpeechEvents

MODE = Conversation(
    window_ms=10000,
    follow_up_min_probability=0.5,
    cue_when_others_ms=30000,
    require_guest_address=True,
)
NAMES = Addressing.from_names(["Beacon"], {"Beacon": ["Beakon"]})
OWNER, GUEST, LABEL_OWNER = "Speaker A", "Speaker B", "Speaker C"


class AmbientGuestAddressTests(unittest.TestCase):
    def start(self, mode=MODE, *, owner_only=False, request_control=True, **settings):
        self.bindings = {OWNER: ("bound", "owner"), GUEST: ("bound", "guest")}
        roles = EnrolledRoles(
            SpeakerPriority(
                owners=("owner", LABEL_OWNER), trusted=("trusted",), owner_only=owner_only
            )
        )
        roles.source = SimpleNamespace(binding=self.bindings.get)
        self.events = SpeechEvents(request_control=request_control)
        self.events.start(
            SESSION,
            conversation=mode,
            addressing=NAMES,
            priority=roles,
            duck=True,
            other_agents=Addressing.from_names(["Atlas"]),
            **settings,
        )
        self.transcripts = {}
        return self.events.drain()[0]

    def say(
        self,
        uid,
        at,
        text,
        label="attend",
        *,
        speaker=OWNER,
        context=None,
        follow_up=False,
        overlap=False,
        inferred=False,
        **decision,
    ):
        """Assert no request, or one matching request with the specified prior transcripts."""
        current = turn(
            uid, at, at + 1000, text, speaker, provenance="diarization-timeline", overlap=overlap
        )
        self.events.transcript(current, current.end_ms, inferred=inferred)
        self.events.decision(decided(current, label, **decision), current.end_ms)
        output = self.events.drain()
        (transcript,) = (e["turn"] for e in output if e["type"] == "transcript")
        self.transcripts[uid] = transcript
        (attention,) = (e for e in output if e["type"] == "attention")
        requests = [e for e in output if e["type"] == "request"]
        if context is None:
            self.assertNotIn("request_id", attention)
            self.assertEqual(requests, [])
            self.assertNotIn("follow_up", attention["decision"])
        else:
            (request,) = requests
            self.assertEqual(request["request_id"], SESSION + ":" + uid)
            self.assertEqual(attention["request_id"], request["request_id"])
            self.assertEqual(request["turn"], transcript)
            self.assertEqual(request["decision"], attention["decision"])
            self.assertEqual(request["decision"].get("follow_up", False), follow_up)
            self.assertEqual(request["context"]["turns"], [self.transcripts[k] for k in context])
        return output

    def fragment(self):
        self.say("fragment", 0, "The last column was shaded", "ignore", speaker=GUEST)

    def test_inferred_owner_label_needs_a_guest_address(self):
        self.start()
        self.fragment()
        held = self.say("read", 1200, "Read the chart", inferred=True)
        self.assertFalse(any(e["type"] == "conversation" for e in held))
        named = self.say(
            "named", 2500, "Beacon, read the chart", inferred=True, context=["fragment", "read"]
        )
        self.assertFalse(any(e["type"] == "conversation" for e in named))
        self.say("more", 4000, "And the legend?", "uncertain", attend=0.9)

        # The pre-existing named-owner promotion explicitly permits inferred words.
        self.start(replace(MODE, require_guest_address=False))
        self.say("named", 0, "Beacon, read the chart", "uncertain", inferred=True, context=[])

    def test_guest_opening_name_needs_vocative_evidence(self):
        for text, addressed in (
            ("Beacon was offline yesterday", False),
            ("Beacon can read the chart", False),
            ("Beacon was there yesterday", False),
            ("We told Beacon about the chart", False),
            ("Beacon, the chart please", True),
            ("Hey Beakon the chart please", True),
            ("Beacon read the chart", True),
            ("Beacon do it again", True),
            ("Beakon can you read the chart", True),
            ("Beacon is the chart ready?", True),
            ("Beacon?", True),
        ):
            with self.subTest(text=text):
                self.start()
                self.say("guest", 0, text, speaker=GUEST, context=[] if addressed else None)

    def test_guest_delimited_opening_name(self):
        for text, addressed in (
            ("Beacon, describe the grid", True),
            ("Beacon, summarize the meeting", True),
            ("Beacon: translate this", True),
            ("Beacon, did you describe the grid?", True),
            ("Beacon, are you there?", True),
            ("Beacon, is the chart ready?", True),
            ("Beacon, where is the chart?", True),
            ("Beacon, what time is it?", True),
            ("Beacon, can you hear me", True),
            ("Beacon, that is the last chart?", True),
            ("Beacon, is the chart ready", True),
            ("Beacon, where is the chart", True),
            ("Beacon, where is the chart, exactly?", True),
            ("Beacon, who called", True),
            ("Beacon, which we retired, was slow", False),
            ("Beacon, which is our old assistant, is offline?", False),
            ("Beacon, where we used to work, closed", False),
            ("Beacon, who we replaced, was slow", False),
            ("Beacon, is offline today", False),
            ("Beacon, was great yesterday", False),
            ("Beacon, isn’t online today", False),
            ("Beacon, formerly Atlas, seems slow", False),
            ("Beacon, did you a favor yesterday", False),
        ):
            with self.subTest(text=text):
                self.start()
                self.say("guest", 0, text, speaker=GUEST, context=[] if addressed else None)

    def test_verified_owner_direct_and_post_reply_follow_up_with_presence(self):
        self.start()
        self.fragment()
        self.say("open", 1200, "Display the sample chart", context=["fragment"])
        self.events.reply("started", 2400)
        self.events.reply("ended", 3100, asked=True)
        # The live-95 shape was a direct model attend. Also exercise the distinct
        # uncertain-to-follow-up path with the same probability and different-label history.
        self.say("answer", 3500, "Use the wider column", attend=0.93, context=["fragment", "open"])
        self.say(
            "more",
            5000,
            "And the last row",
            "uncertain",
            attend=0.93,
            follow_up=True,
            context=["fragment", "open", "answer"],
        )

    def test_guests_and_label_only_owners_need_a_direct_address(self):
        for speaker in (GUEST, None, LABEL_OWNER, "unbound", "trusted label"):
            with self.subTest(speaker=speaker):
                self.start()
                self.bindings["trusted label"] = ("bound", "trusted")
                self.fragment()
                self.say("talk", 1200, "Is the chart ready?", speaker=speaker)
                self.say("mention", 2400, "We told Beacon about the chart", speaker=speaker)
                addressed = self.say(
                    "ask",
                    3600,
                    "Hey Beakon, display the chart",
                    speaker=speaker,
                    context=["fragment", "talk", "mention"],
                )
                self.assertFalse(any(e["type"] == "conversation" for e in addressed))
                self.say(
                    "more", 4800, "And the first row", "uncertain", speaker=speaker, attend=0.99
                )
                self.say(
                    "vocative",
                    6000,
                    "Read the heading, Beacon?",
                    speaker=speaker,
                    context=["fragment", "talk", "mention", "ask", "more"],
                )
                self.say(
                    "abstain",
                    7200,
                    "Beacon, the footer?",
                    "uncertain",
                    speaker=speaker,
                    attend=0.99,
                )
                self.say("ignore", 8400, "Beacon, read the row", "ignore", speaker=speaker)

    def test_guest_rule_without_a_presence_window_still_respects_owner_only(self):
        for owner_only in (False, True):
            with self.subTest(owner_only=owner_only):
                self.start(replace(MODE, cue_when_others_ms=0), owner_only=owner_only)
                self.say("first", 0, "Display the map", speaker=LABEL_OWNER)
                self.say(
                    "ask",
                    1500,
                    "Beacon, display the map",
                    speaker=LABEL_OWNER,
                    context=None if owner_only else ["first"],
                )
                self.say("owner", 3000, "Zoom in", context=["first", "ask"])

    def test_guest_on_engaged_label_cannot_borrow_or_extend_the_window(self):
        self.start()
        self.say("open", 0, "Open the example chart", context=[])
        self.bindings[OWNER] = ("bound", "guest")
        self.say("borrow", 1500, "And the other chart", "uncertain", attend=0.99)
        self.say("named", 5000, "Beacon, read the scale", context=["open", "borrow"])
        self.bindings[OWNER] = ("bound", "owner")
        self.say("expired", 11000, "And the units?", "uncertain", attend=0.99)

    def test_addressed_guest_and_reply_preserve_owner_question_and_deadline(self):
        # A trusted guest could previously take the engagement from another trusted/owner
        # speaker. Their direct request and its playback must leave the owner's question open.
        for scoped in (False, True):
            for inside in (False, True):
                with self.subTest(scoped=scoped, inside=inside):
                    self.start(replace(MODE, require_cue=True))
                    self.bindings[GUEST] = ("bound", "trusted")
                    self.say("open", 0, "Display the example chart", context=[])
                    self.reply_to("open", 1300, 1800, scoped=scoped, asked=True)
                    self.say(
                        "guest", 5000, "Beacon, read the legend", speaker=GUEST, context=["open"]
                    )
                    self.reply_to("guest", 6300, 7000, scoped=scoped)
                    self.say(
                        "answer",
                        11799 if inside else 11800,
                        "Yes",
                        "uncertain",
                        attend=0.8,
                        context=["open", "guest"] if inside else None,
                        follow_up=inside,
                    )

    def reply_to(self, uid, start, end, *, scoped, asked=False):
        if scoped:
            for state, at in (("accepted", start), ("reply_started", start), ("reply_ended", end)):
                control = {
                    "version": 1,
                    "session_id": SESSION,
                    "request_id": SESSION + ":" + uid,
                    "state": state,
                }
                if state == "reply_ended":
                    control["asked"] = asked
                self.assertTrue(self.events.request_outcome(control, at))
        else:
            self.events.reply("started", start)
            self.events.reply("ended", end, asked=asked)

    def test_guest_playback_and_owner_answer_floor(self):
        mode = replace(MODE, require_cue=True, answer_follow_up_min_probability=0.4)
        instruction = "Repeat your last answer."
        for scoped in (False, True):
            for owner_answered in (False, True):
                with self.subTest(scoped=scoped, owner_answered=owner_answered):
                    self.start(mode)
                    self.say("open", 0, "Draw the grid", context=[])
                    if owner_answered:
                        self.reply_to("open", 1100, 1400, scoped=scoped, asked=not scoped)
                    self.say(
                        "guest", 1800, "Beacon, read the scale", speaker=GUEST, context=["open"]
                    )
                    self.reply_to("guest", 3000, 3200, scoped=scoped, asked=True)
                    self.say(
                        "owner-follow",
                        3600,
                        instruction,
                        "uncertain",
                        attend=0.44,
                        context=["open", "guest"] if owner_answered else None,
                        follow_up=owner_answered,
                    )
                    if owner_answered:
                        # Keep a confirmed answer live while a guest reuses the owner's label.
                        self.reply_to("owner-follow", 4700, 4900, scoped=scoped, asked=not scoped)
                    self.bindings[OWNER] = self.bindings[GUEST]
                    self.say(
                        "guest-follow", 5100, instruction, "uncertain", speaker=OWNER, attend=0.44
                    )

    def test_guest_playback_end_after_an_owner_request_cannot_extend_it(self):
        self.start()
        self.say("guest", 0, "Beacon, describe the grid", speaker=GUEST, context=[])
        self.events.reply("started", 1500)
        self.say("open", 2000, "Draw a second grid", context=["guest"])
        self.events.reply("ended", 5000)
        self.say("expired", 13000, "And the legend?", "uncertain", attend=0.9)

    def test_delayed_legacy_guest_reply_cannot_answer_or_renew_owner(self):
        for request_control, scoped in ((False, False), (True, False), (True, True)):
            for at, probe in ((8000, "Yes"), (14000, "And the legend?"), (14000, "Yes")):
                with self.subTest(
                    request_control=request_control, scoped=scoped, at=at, probe=probe
                ):
                    self.start(replace(MODE, require_cue=True), request_control=request_control)
                    owner = turn(
                        "owner", 0, 1000, "Read the chart", provenance="diarization-timeline"
                    )
                    self.events.transcript(owner, 1000)
                    (event,) = self.events.drain()
                    self.transcripts["owner"] = event["turn"]
                    self.say(
                        "guest", 2000, "Beacon, read the scale", speaker=GUEST, context=["owner"]
                    )
                    self.events.decision(decided(owner, "attend"), 4000)
                    output = self.events.drain()
                    self.assertFalse(any(e["type"] == "override" for e in output))
                    (engaged,) = (e for e in output if e["type"] == "conversation")
                    self.assertEqual(engaged["until_ms"], 11000)
                    self.reply_to(
                        "owner" if scoped else "guest", 5000, 6000, scoped=scoped, asked=True
                    )
                    self.assertTrue(self.events._playback_overlaps(5500, 5500))
                    self.assertFalse(self.events._playback_overlaps(6500, 6500))
                    self.say(
                        "probe",
                        at,
                        probe,
                        "uncertain",
                        attend=0.9,
                        context=["owner", "guest"] if scoped else None,
                        follow_up=scoped,
                    )

    def test_replacing_owner_playback_releases_hold_at_second_start(self):
        for at, inside in ((14999, True), (15000, False), (20000, False)):
            with self.subTest(at=at):
                self.start()
                self.say("owner", 0, "Read the chart", context=[])
                self.events.reply("started", 1500)
                self.say("guest", 3000, "Beacon, read the scale", speaker=GUEST, context=["owner"])
                self.reply_to("guest", 5000, 6000, scoped=False)
                self.say(
                    "probe",
                    at,
                    "And the legend?",
                    "uncertain",
                    attend=0.9,
                    context=["owner", "guest"] if inside else None,
                    follow_up=inside,
                )

    def test_guest_cannot_close_owner_question_on_reused_label(self):
        for action in ("closing", "stop", "disengage"):
            with self.subTest(action=action):
                self.start(
                    replace(MODE, require_cue=True, closing_phrases=("all done beacon",)),
                    dismissal=Dismissal(),
                )
                self.say("owner", 0, "Display the chart", context=[])
                self.reply_to("owner", 1300, 1800, scoped=False, asked=True)
                self.bindings[OWNER] = ("bound", "trusted" if action == "disengage" else "guest")
                if action == "closing":
                    output = self.say("guest", 3000, "All done, Beacon", context=["owner"])
                else:
                    phrase = "Stop" if action == "stop" else "Go away"
                    stop = turn("guest", 3000, 4000, phrase, provenance="diarization-timeline")
                    self.events.transcript(stop, 4000)
                    if action == "disengage":
                        event = decided(stop, "ignore")
                        self.events.decision(
                            replace(
                                event,
                                decision=replace(
                                    event.decision,
                                    dismissal="disengage",
                                    dismissal_choice="disengage",
                                    dismissal_confidence=0.95,
                                ),
                            ),
                            4000,
                        )
                    output = self.events.drain()
                    self.transcripts["guest"] = next(
                        e["turn"] for e in output if e["type"] == "transcript"
                    )
                    (dismiss,) = (e for e in output if e["type"] == "dismiss")
                    self.assertEqual(dismiss["scope"], ["playback"])
                    self.assertEqual(dismiss["withdrawn_request_ids"], [])
                self.assertFalse(any(e["type"] == "conversation" for e in output))
                self.bindings[OWNER] = ("bound", "owner")
                self.say(
                    "answer",
                    5000,
                    "Yes",
                    "uncertain",
                    attend=0.8,
                    context=["owner", "guest"],
                    follow_up=True,
                )

    def test_guest_stop_withdraws_own_requests_on_reused_label(self):
        self.start(dismissal=Dismissal())
        self.say("owner", 0, "Read the chart", context=[])
        pending_owner = turn(
            "older", 1200, 2000, "Show the chart", provenance="diarization-timeline"
        )
        self.events.transcript(pending_owner, 2000)
        self.transcripts["older"] = self.events.drain()[0]["turn"]
        self.bindings[OWNER] = ("bound", "guest")
        self.say("delivered", 2500, "Beacon, read the scale", context=["owner", "older"])
        pending_guest = turn(
            "pending", 4000, 5000, "Beacon, read the units", provenance="diarization-timeline"
        )
        self.events.transcript(pending_guest, 5000)
        stop = turn("stop", 5500, 6500, "Stop", provenance="diarization-timeline")
        self.events.transcript(stop, 6500)
        (dismiss,) = (e for e in self.events.drain() if e["type"] == "dismiss")
        self.assertEqual(dismiss["withdrawn_request_ids"], [SESSION + ":delivered"])
        self.events.decision(decided(pending_guest, "attend"), 7000)
        output = self.events.drain()
        self.assertFalse(any(e["type"] == "request" for e in output))
        (withdrawn,) = (e for e in output if e["type"] == "dismiss")
        self.assertEqual(withdrawn["withdrawn_request_ids"], [SESSION + ":pending"])
        self.bindings[OWNER] = ("bound", "owner")
        self.events.decision(decided(pending_owner, "attend"), 7500)
        (request,) = (e for e in self.events.drain() if e["type"] == "request")
        self.assertEqual(request["request_id"], SESSION + ":older")

    def test_guest_legacy_playback_prunes_queued_context_at_cutoff(self):
        for phase in ("started", "ended"):
            for at, keep_old in ((300999, True), (301000, False)):
                with self.subTest(phase=phase, at=at):
                    self.start()
                    self.fragment()
                    self.say("recent", 1500, "A different heading", "ignore", speaker=GUEST)
                    guest = turn(
                        "guest",
                        3000,
                        4000,
                        "Beacon, read the chart",
                        GUEST,
                        provenance="diarization-timeline",
                    )
                    self.events.transcript(guest, guest.end_ms)
                    self.events.decision(decided(guest, "attend"), guest.end_ms)
                    self.events.reply(phase, at)
                    (request,) = (e for e in self.events.drain() if e["type"] == "request")
                    expected = ["fragment", "recent"] if keep_old else ["recent"]
                    self.assertEqual(
                        request["context"]["turns"], [self.transcripts[k] for k in expected]
                    )

    def test_plain_floor_and_cue_requirement_for_low_score_owner_turns(self):
        for enabled, presence in ((False, False), (True, False), (True, True)):
            for require_cue in (False, True):
                with self.subTest(enabled=enabled, presence=presence, require_cue=require_cue):
                    self.start(
                        replace(MODE, require_guest_address=enabled, require_cue=require_cue)
                    )
                    past = []
                    if presence:
                        self.fragment()
                        past.append("fragment")
                    self.say("open", 1500, "Draw the sample chart", context=past)
                    past.append("open")
                    for uid, at, text, score, confidence in (
                        ("pet", 3000, "Paws down, kitten.", 0.69, 0.53),
                        ("exclaim", 5000, "Holy mackerel!", 0.51, 0.26),
                    ):
                        self.say(
                            uid,
                            at,
                            text,
                            "uncertain",
                            attend=score,
                            confidence=confidence,
                            context=None if require_cue or presence else past,
                            follow_up=not (require_cue or presence),
                        )
                        past.append(uid)
                    self.say("below", 7000, "And the labels?", "uncertain", attend=0.44)

    def test_uncued_low_score_owner_follow_ups_stay_out_with_presence(self):
        for enabled in (False, True):
            for text, score, confidence in (
                ("Paws down, kitten.", 0.69, 0.53),
                ("Holy mackerel!", 0.51, 0.26),
            ):
                with self.subTest(enabled=enabled, text=text):
                    self.start(replace(MODE, require_guest_address=enabled))
                    self.say("open", 0, "Beacon, draw the sample chart", context=[])
                    self.say("guest", 1500, "The last column was shaded", "ignore", speaker=GUEST)
                    self.say("probe", 3000, text, "uncertain", attend=score, confidence=confidence)
                    self.say(
                        "directed",
                        5000,
                        "And the scale?",
                        "uncertain",
                        attend=score,
                        confidence=confidence,
                        context=["open", "guest", "probe"] if enabled else None,
                        follow_up=enabled,
                    )
                    self.say("below", 7000, "And the units?", "uncertain", attend=0.44)

    def test_speaker_overlap_recipient_and_other_agent_still_gate_follow_ups(self):
        self.start()
        self.fragment()
        self.say("open", 1500, "Draw the sample chart", context=["fragment"])
        self.bindings["other label"] = ("bound", "owner")
        self.say("mismatch", 3000, "And the lines?", "uncertain", speaker="other label", attend=0.9)
        self.say("mixed", 4500, "And the lines?", "uncertain", overlap=True, attend=0.9)
        self.say(
            "recipient", 6000, "And the lines?", "uncertain", recipient="other_human", attend=0.9
        )
        self.say("agent", 7500, "Atlas, draw the lines", "attend")

    def test_explicit_meeting_and_media_keep_their_bars(self):
        for mode in ("meeting", "media"):
            with self.subTest(mode=mode):
                self.start()
                self.fragment()
                getattr(self.events, mode)(True)
                self.say("owner", 1500, "Draw the sample chart")
                self.say("guest", 3000, "Beacon, draw the chart", speaker=GUEST)
                self.say(
                    "named", 4500, "Beacon, draw the chart", context=["fragment", "owner", "guest"]
                )
                self.say("follow", 6000, "And the lines?", "uncertain", attend=0.93)

    def test_private_duck_control_excludes_both_voices_and_their_context(self):
        self.start()
        self.fragment()
        self.events.duck(True, now_ms=1200)
        for speaker, uid, at, text in (
            (OWNER, "owner", 1500, "Beacon, draw the private chart"),
            (GUEST, "guest", 3000, "Beacon, play that back"),
            (GUEST, "exit", 4500, "Beacon, privacy off"),
        ):
            current = turn(uid, at, at + 1000, text, speaker, provenance="diarization-timeline")
            self.events.transcript(current, current.end_ms)
            self.events.decision(decided(current, "attend"), current.end_ms)
            self.assertEqual(self.events.drain(), [])
        self.events.duck(False, now_ms=6000)
        self.say("public", 6500, "Beacon, draw the public chart", context=["fragment"])

    def test_default_keeps_the_automatic_meeting_bar(self):
        self.start(Conversation(cue_when_others_ms=30000))
        self.fragment()
        self.say("direct", 1500, "Draw the chart")
        self.say("named", 3000, "Beacon, draw the chart", context=["fragment", "direct"])
        self.say("guest", 4500, "Beacon, read the chart", speaker=GUEST)
        self.say("follow", 6000, "And the units?", "uncertain", attend=0.93)

    def test_guest_address_setting_is_strict_and_advertised_only_when_enabled(self):
        for value in (None, 0, 1, "true", []):
            with self.subTest(value=value), self.assertRaises(ContractError):
                Conversation.from_dict({"require_guest_address": value})
        for enabled in (False, True):
            mode = Conversation.from_dict({"require_guest_address": enabled})
            advertised = self.start(mode)["conversation"]
            self.assertEqual(advertised.get("require_guest_address"), True if enabled else None)
            advertised.pop("version")
            self.assertEqual(Conversation.from_dict(advertised), mode)

    def test_unpaired_guest_end_after_override_cannot_renew_owner(self):
        for started in (False, True):
            with self.subTest(started=started):
                self.start(replace(MODE, require_cue=True), request_control=False)
                self.say("guest", 0, "Beacon, read the scale", speaker=GUEST, context=[])
                output = self.say("owner", 2000, "Draw a second grid", context=["guest"])
                (override,) = (e for e in output if e["type"] == "override")
                self.assertEqual(override["superseded_request_id"], SESSION + ":guest")
                if started:
                    self.events.reply("started", 5000)
                self.events.reply("ended", 6000, asked=True)
                self.say("unasked", 7500, "Yes", "uncertain", attend=0.9)
                self.say("expired", 14500, "And the legend?", "uncertain", attend=0.9)

    def test_label_only_guest_cannot_override_first_bound_owner(self):
        for phrase in ("Stop", "Beacon, read the scale"):
            with self.subTest(phrase=phrase):
                self.start(replace(MODE, require_cue=True), dismissal=Dismissal())
                del self.bindings[OWNER]
                owner = turn("owner", 0, 1000, "Draw the grid", provenance="diarization-timeline")
                self.events.transcript(owner, 1000)
                self.bindings[OWNER] = ("bound", "owner")
                self.events._priority.source.first_bind_identity = lambda label, uid: (
                    "owner" if (label, uid) == (OWNER, "owner") else None
                )
                self.events.decision(decided(owner, "attend"), 1200)
                output = self.events.drain()
                (request,) = (e for e in output if e["type"] == "request")
                self.assertEqual(request["turn"]["role"], "unknown")
                self.assertEqual(request["decision"]["role"], "owner")
                self.transcripts["owner"] = request["turn"]
                control = {"session_id": SESSION, "version": 1, "request_id": SESSION + ":owner"}
                self.assertTrue(self.events.request_outcome(dict(control, state="accepted"), 1250))
                guest = self.say(
                    "guest",
                    1500,
                    phrase,
                    speaker=LABEL_OWNER,
                    context=None if phrase == "Stop" else ["owner"],
                )
                self.assertFalse(any(e["type"] == "override" for e in guest))
                self.assertTrue(
                    self.events.request_outcome(
                        dict(control, state="reply_published", asked=True), 3000
                    )
                )
                self.say(
                    "answer",
                    3500,
                    "Yes",
                    "uncertain",
                    attend=0.9,
                    context=["owner", "guest"],
                    follow_up=True,
                )

    def test_rebound_guest_cannot_withdraw_another_speakers_turn(self):
        for first in ("owner", "guest-one", "guest-two"):
            for pending in (True, False):
                with self.subTest(first=first, pending=pending):
                    self.start(dismissal=Dismissal())
                    self.bindings[OWNER] = ("bound", first)
                    if first == "guest-two":
                        # The worker observed a rebind away and back to the same ID.
                        revision = 1
                        self.events._priority.source.binding_snapshot = lambda label: (
                            revision,
                            self.bindings.get(label),
                        )
                    earlier = turn(
                        "earlier",
                        0,
                        1000,
                        "Beacon, read the chart",
                        provenance="diarization-timeline",
                    )
                    self.events.transcript(earlier, 1000)
                    if not pending:
                        self.events.decision(decided(earlier, "attend"), 1000)
                    output = self.events.drain()
                    self.transcripts["earlier"] = next(
                        e["turn"] for e in output if e["type"] == "transcript"
                    )
                    self.bindings[OWNER] = ("bound", "guest-two")
                    revision = 3
                    output = self.say("stop", 1500, "Stop", speaker=OWNER)
                    (dismiss,) = (e for e in output if e["type"] == "dismiss")
                    self.assertEqual(dismiss["withdrawn_request_ids"], [])
                    if pending:
                        self.events.decision(decided(earlier, "attend"), 3000)
                        output = self.events.drain()
                        self.assertFalse(any(e["type"] == "dismiss" for e in output))
                        (request,) = (e for e in output if e["type"] == "request")
                        self.assertEqual(request["request_id"], SESSION + ":earlier")
                        self.assertEqual(request["turn"], self.transcripts["earlier"])
                        self.assertEqual(request["context"]["turns"], [])

    def test_completed_guest_playback_releases_later_owner_question(self):
        for speaker in (LABEL_OWNER, GUEST):
            with self.subTest(speaker=speaker):
                self.start(replace(MODE, require_cue=True), request_control=False)
                self.say("guest", 0, "Beacon, read the scale", speaker=speaker, context=[])
                self.reply_to("guest", 1200, 1500, scoped=False)
                self.say("owner", 2000, "Draw the grid", context=["guest"])
                self.reply_to("owner", 3200, 3500, scoped=False, asked=True)
                self.say(
                    "answer",
                    4000,
                    "Yes",
                    "uncertain",
                    attend=0.9,
                    context=["guest", "owner"],
                    follow_up=True,
                )

    def test_owner_follow_up_after_reply_with_and_without_guest(self):
        for presence in (False, True):
            for text, score, confidence, side_talk in (
                ("Use the wider column", 0.93, 0.9, False),
                ("Use the wider column", 0.51, 0.26, False),
                ("Zoom in", 0.93, 0.9, False),
                ("Zoom in", 0.70, 0.6, False),
                ("Paws down, kitten.", 0.69, 0.53, True),
                ("Holy mackerel!", 0.51, 0.26, True),
                ("Paws down kitten", 0.69, 0.53, True),
                ("Holy mackerel", 0.51, 0.26, True),
            ):
                with self.subTest(presence=presence, text=text, score=score):
                    self.start(replace(MODE, require_cue=False))
                    self.say("open", 0, "Draw the grid", context=[])
                    self.reply_to("open", 1100, 1400, scoped=False)
                    past = ["open"]
                    if presence:
                        self.say(
                            "guest", 2100, "The last column was shaded", "ignore", speaker=GUEST
                        )
                        past.append("guest")
                    admitted = not (side_talk and presence)
                    self.say(
                        "follow",
                        3500,
                        text,
                        "uncertain",
                        attend=score,
                        confidence=confidence,
                        context=past if admitted else None,
                        follow_up=admitted,
                    )

    def test_ask_anywhere_keeps_owner_promotion_with_guest_presence(self):
        for policy in (False, True):
            for anywhere in (False, True):
                with self.subTest(policy=policy, anywhere=anywhere):
                    self.start(
                        replace(MODE, require_guest_address=policy), meeting_ask_anywhere=anywhere
                    )
                    self.fragment()
                    self.say(
                        "mention",
                        1400,
                        "Read the chart, Beacon?",
                        "uncertain",
                        speaker=LABEL_OWNER if policy else GUEST,
                        attend=0.8,
                    )
                    self.say("unnamed", 2700, "Read the chart", "uncertain", attend=0.8)
                    output = self.say(
                        "owner",
                        4000,
                        "Read the chart, Beacon?",
                        "uncertain",
                        attend=0.8,
                        context=["fragment", "mention", "unnamed"] if anywhere else None,
                    )
                    if anywhere:
                        request = next(e for e in output if e["type"] == "request")
                        self.assertEqual(request["decision"]["others_present"], "ask_anywhere")

    def test_guest_name_in_a_statement_is_context(self):
        for text in (
            "Beacon did you a favor yesterday",
            "Beacon did you a favor?",
            "Beacon has you on file",
            "Beacon read the chart yesterday",
            "Beacon, which is our old assistant, is offline",
            "Beacon, my old assistant, was offline",
            "We named the assistant after our town, Beacon.",
            "Beacon and yourself",
        ):
            with self.subTest(text=text):
                self.start()
                self.say("mention", 0, text, speaker=GUEST)

    def test_guest_imperatives_without_asr_comma(self):
        for text, addressed in (
            ("Beacon repeat what you just said", True),
            ("Beacon introduce yourself", True),
            ("Beacon did you read the chart?", True),
            ("Beacon, read the labels on the chart", True),
            ("Beacon rewind ten seconds", False),
            ("Beacon can you rewind ten seconds", True),
        ):
            with self.subTest(text=text):
                self.start()
                self.say("command", 0, text, speaker=GUEST, context=[] if addressed else None)

    def test_guest_ack_clip_and_answer_release_the_next_ack(self):
        for speaker in (GUEST, OWNER):
            for answer_played in (False, True):
                with self.subTest(speaker=speaker, answer_played=answer_played):
                    self.start(
                        replace(MODE, require_cue=True),
                        acknowledgement=Acknowledgement(min_confidence=1.0),
                    )
                    self.say("owner", 0, "Draw the grid", context=[])
                    self.reply_to("owner", 1100, 1400, scoped=False, asked=True)
                    guest = self.say(
                        "guest", 2000, "Beacon, read the scale", speaker=GUEST, context=["owner"]
                    )
                    self.assertTrue(next(e["acknowledge"] for e in guest if e["type"] == "request"))
                    self.reply_to("clip", 3100, 3300, scoped=False)
                    if answer_played:
                        self.reply_to("guest", 3500, 3800, scoped=False)
                    if speaker == OWNER:
                        # Only the owner's unanswered question makes this an eligible follow-up.
                        text, label = "Yes", "uncertain"
                    else:
                        text, label = "Beacon, read the units", "attend"
                    output = self.say(
                        "next",
                        4000,
                        text,
                        label,
                        speaker=speaker,
                        attend=1.0,
                        context=["owner", "guest"],
                        follow_up=speaker == OWNER,
                    )
                    self.assertEqual(
                        next(e["acknowledge"] for e in output if e["type"] == "request"),
                        answer_played,
                    )

    def test_superseded_guest_ack_end_does_not_complete_the_answer(self):
        for dedup_ms in (0, 8000):
            with self.subTest(dedup_ms=dedup_ms):
                self.start(
                    replace(MODE, require_cue=True),
                    request_control=False,
                    acknowledgement=Acknowledgement(dedup_window_ms=dedup_ms),
                )
                guest = self.say("guest", 0, "Beacon, read the scale", speaker=GUEST, context=[])
                self.assertTrue(next(e["acknowledge"] for e in guest if e["type"] == "request"))
                self.events.reply("started", 1100)
                self.say("owner", 2000, "Draw the grid", context=["guest"])
                self.events.reply("ended", 3200)
                self.reply_to("guest", 5000, 6000, scoped=False, asked=True)
                self.say("expired", 14500, "Yes", "uncertain", attend=0.9)

    def test_guest_ack_with_competing_unanswered_owner(self):
        self.start(
            replace(MODE, require_cue=True), acknowledgement=Acknowledgement(min_confidence=1.0)
        )
        self.say("owner", 0, "Draw the grid", context=[])
        output = self.say("guest", 1500, "Beacon, read the scale", speaker=GUEST, context=["owner"])
        self.assertTrue(next(e["acknowledge"] for e in output if e["type"] == "request"))
        self.reply_to("clip", 2800, 3000, scoped=False)
        self.reply_to("answer", 3200, 3400, scoped=False, asked=True)
        self.say("unasked", 4000, "Yes", "uncertain", attend=0.9)
        output = self.say(
            "next", 5200, "Beacon, draw another grid", context=["owner", "guest", "unasked"]
        )
        self.assertTrue(next(e["acknowledge"] for e in output if e["type"] == "request"))
