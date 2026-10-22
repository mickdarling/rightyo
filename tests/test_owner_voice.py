"""Owner-voice status and input route (rightyo#164, slice 2).

Synthetic data only: stand-in evidence, scripted embedder vectors and authored tones. No
model, microphone, recorded voice or enrollment store is used.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from test_enrolled_roles import LOCAL, OWNER_ID, Bindings, FakeShadow
from test_live_speaker_id import (
    ENTRIES,
    PRIVATE_TEXT,
    RATE,
    Harness,
    ScriptedEmbedder,
    TurnEveryTwoSeconds,
    settings,
    tone,
    unit,
    wait_for,
)

from rightyo.contracts import ContractError, SpeakerPriority
from rightyo.live_speaker_id import INPUT_ROUTES, EnrolledRoles, ShadowSpeakerId, input_route
from rightyo.prototype import PrototypeController
from rightyo.tool import (
    INPUT_ROUTE_ENV,
    OWNER_VOICE_EVENTS_ENV,
    OWNER_VOICE_THRESHOLD_ENV,
    host_options,
    listen,
)
from rightyo.tool_events import OwnerVoice, SpeechEvents

SESSION = "owner-voice"
# HailProtocol's diagnostic `route` vocabulary (hailing-station PR #488).
HAIL_ROUTES = {
    "none",
    "built_in_mic",
    "built_in_speaker",
    "built_in_receiver",
    "headphones",
    "headset_mic",
    "line_in",
    "line_out",
    "bluetooth_a2dp",
    "bluetooth_hfp",
    "bluetooth_le",
    "airplay",
    "hdmi",
    "car_audio",
    "usb_audio",
    "other",
}
HOST_ENV = (OWNER_VOICE_EVENTS_ENV, OWNER_VOICE_THRESHOLD_ENV, INPUT_ROUTE_ENV)


class Evidence:
    """A stand-in identifier's owner evidence, changed by the test."""

    def __init__(self, enrolled=True, bound=False, seconds=0.0):
        self.enrolled, self.bound, self.seconds = enrolled, bound, seconds
        self.asked = []

    def owner_evidence(self, owners):
        self.asked.append(tuple(owners))
        return self.enrolled, self.bound, self.seconds


def owner_events(events):
    return [event for event in events if event["type"] == "owner_voice"]


class OwnerVoiceSettingsTests(unittest.TestCase):
    def test_threshold_is_bounded_whole_seconds(self):
        self.assertEqual(OwnerVoice().to_dict(), {"threshold_s": 20})
        self.assertEqual(OwnerVoice(5).threshold_s, 5)
        self.assertEqual(OwnerVoice(600).threshold_s, 600)
        for bad in (4, 601, 20.0, "20", True, None):
            with self.subTest(bad=bad), self.assertRaises(ContractError):
                OwnerVoice(bad)


class HostOptionsTests(unittest.TestCase):
    def options(self, environ):
        with patch("sys.stderr", io.StringIO()) as errors:
            return host_options(environ), errors.getvalue()

    def test_nothing_without_the_opt_in(self):
        for environ in ({}, {OWNER_VOICE_EVENTS_ENV: "0"}, {OWNER_VOICE_EVENTS_ENV: "true"}):
            with self.subTest(environ=environ):
                self.assertEqual(self.options(environ), ({}, ""))
        # A threshold alone opts in to nothing.
        options, _ = self.options({OWNER_VOICE_THRESHOLD_ENV: "30"})
        self.assertEqual(options, {})

    def test_opt_in_with_default_and_configured_threshold(self):
        options, errors = self.options({OWNER_VOICE_EVENTS_ENV: "1"})
        self.assertEqual(options["owner_voice"].threshold_s, 20)
        self.assertEqual(errors, "")
        options, _ = self.options({OWNER_VOICE_EVENTS_ENV: "1", OWNER_VOICE_THRESHOLD_ENV: "45"})
        self.assertEqual(options["owner_voice"].threshold_s, 45)

    def test_an_invalid_threshold_keeps_the_default_and_is_not_echoed(self):
        for raw in ("4", "601", "20.5", "-20", "twenty", "", "２０", "20s"):
            with self.subTest(raw=raw):
                options, errors = self.options(
                    {OWNER_VOICE_EVENTS_ENV: "1", OWNER_VOICE_THRESHOLD_ENV: raw}
                )
                self.assertEqual(options["owner_voice"].threshold_s, 20)
                self.assertIn(f"invalid {OWNER_VOICE_THRESHOLD_ENV}; using 20", errors)
                if raw:
                    self.assertNotIn(raw, errors.replace("using 20", ""))

    def test_route_tokens_match_hail_protocol(self):
        self.assertEqual(set(INPUT_ROUTES), HAIL_ROUTES)
        for route in HAIL_ROUTES:
            options, errors = self.options({INPUT_ROUTE_ENV: route})
            self.assertEqual((options, errors), ({"input_route": route}, ""))

    def test_an_invalid_route_is_ignored_and_not_echoed(self):
        for raw in ("USB_AUDIO", "Mick's AirPods", "usb_audio ", "", "usb_audio\nx", "Scarlett"):
            with self.subTest(raw=raw):
                options, errors = self.options({INPUT_ROUTE_ENV: raw})
                self.assertEqual(options, {})
                self.assertEqual(errors, f"rightyo: invalid {INPUT_ROUTE_ENV}; ignored\n")
        for value in (None, 1, ["usb_audio"]):
            self.assertIsNone(input_route(value))


class EnrolledRolesOwnerVoiceTests(unittest.TestCase):
    def test_status_from_evidence(self):
        roles = EnrolledRoles(SpeakerPriority(owners=("owner", "owner-usb")))
        self.assertIsNone(roles.owner_voice(20))  # identification not started
        roles.source = evidence = Evidence(seconds=19.9)
        self.assertIsNone(roles.owner_voice(20))
        self.assertEqual(evidence.asked, [("owner", "owner-usb")])
        evidence.seconds = 20.0
        self.assertEqual(roles.owner_voice(20), "unrecognized")
        evidence.bound = True
        self.assertEqual(roles.owner_voice(20), "recognized")
        evidence.enrolled = False
        self.assertIsNone(roles.owner_voice(20))

    def test_a_failing_source_says_nothing(self):
        roles = EnrolledRoles(OWNER_ID)
        source = MagicMock()
        source.owner_evidence.side_effect = RuntimeError("synthetic")
        roles.source = source
        self.assertIsNone(roles.owner_voice(20))


class OwnerVoiceEventTests(unittest.TestCase):
    def start(self, *, owner_voice=None, evidence=None, priority=None):
        roles = EnrolledRoles(OWNER_ID) if priority is None else priority
        if evidence is not None:
            roles.source = evidence
        events = SpeechEvents()
        options = {} if owner_voice is None else {"owner_voice": owner_voice}
        events.start(SESSION, priority=roles, **options)
        return events, events.drain()[0]

    def test_opt_in_off_advertises_and_emits_nothing(self):
        evidence = Evidence(seconds=600.0)
        events, started = self.start(evidence=evidence)
        self.assertNotIn("owner_voice", started)
        events.expire(1000)
        self.assertEqual(events.drain(), [])
        self.assertEqual(evidence.asked, [])

    def test_unrecognized_fires_after_the_threshold_once(self):
        evidence = Evidence(seconds=0.0)
        events, started = self.start(owner_voice=OwnerVoice(), evidence=evidence)
        self.assertEqual(started["owner_voice"], {"threshold_s": 20})
        self.assertNotIn("owner_voice", started["capabilities"])
        events.expire(1000)
        evidence.seconds = 19.5
        events.expire(2000)
        self.assertEqual(events.drain(), [])
        evidence.seconds = 21.0
        for now in (3000, 3500, 4000):
            events.expire(now)
        evidence.seconds = 80.0
        events.expire(5000)
        emitted = events.drain()
        self.assertEqual(len(emitted), 1)
        self.assertEqual(
            emitted[0],
            {
                "schema_version": 1,
                "type": "owner_voice",
                "session_id": SESSION,
                "sequence": 2,
                "emitted_at_ms": 3000,
                "state": "unrecognized",
            },
        )

    def test_recognized_follows_a_later_owner_binding_once(self):
        evidence = Evidence(seconds=25.0)
        events, _ = self.start(owner_voice=OwnerVoice(), evidence=evidence)
        events.expire(1000)
        evidence.bound = True
        events.expire(2000)
        events.expire(3000)
        states = [event["state"] for event in owner_events(events.drain())]
        self.assertEqual(states, ["unrecognized", "recognized"])
        events.expire(4000)
        self.assertEqual(events.drain(), [])

    def test_not_polled_while_private(self):
        # rightyo#244: a private turn's handling polls too, so the change would be timed
        # by private speech; the first poll after the exit says it.
        evidence = Evidence(seconds=0.0)
        events, _ = self.start(owner_voice=OwnerVoice(), evidence=evidence)
        events.duck(True, now_ms=1000)
        evidence.seconds = 30.0
        events.expire(2000)
        self.assertEqual(events.drain(), [])
        events.duck(False, now_ms=3000)
        events.expire(3000)
        emitted = owner_events(events.drain())
        self.assertEqual(
            [(e["state"], e["emitted_at_ms"]) for e in emitted], [("unrecognized", 3000)]
        )

    def test_an_owner_bound_before_the_threshold_emits_nothing(self):
        evidence = Evidence(seconds=5.0, bound=True)
        events, _ = self.start(owner_voice=OwnerVoice(), evidence=evidence)
        for now, seconds in ((1000, 5.0), (2000, 30.0), (3000, 300.0)):
            evidence.seconds = seconds
            events.expire(now)
        self.assertEqual(events.drain(), [])

    def test_no_event_when_no_owner_is_enrolled(self):
        evidence = Evidence(enrolled=False, seconds=300.0)
        events, started = self.start(owner_voice=OwnerVoice(), evidence=evidence)
        self.assertIn("owner_voice", started)
        events.expire(1000)
        evidence.bound = True
        events.expire(2000)
        self.assertEqual(events.drain(), [])

    def test_a_threshold_is_configurable(self):
        evidence = Evidence(seconds=6.0)
        events, started = self.start(owner_voice=OwnerVoice(5), evidence=evidence)
        self.assertEqual(started["owner_voice"], {"threshold_s": 5})
        events.expire(1000)
        self.assertEqual([e["state"] for e in events.drain()], ["unrecognized"])

    def test_no_advertisement_without_live_identification_roles(self):
        from rightyo.providers import ConfiguredPriorityProvider

        events, started = self.start(
            owner_voice=OwnerVoice(), priority=ConfiguredPriorityProvider(OWNER_ID)
        )
        self.assertNotIn("owner_voice", started)
        events = SpeechEvents()
        events.start(SESSION, owner_voice=OwnerVoice())
        self.assertNotIn("owner_voice", events.drain()[0])
        with self.assertRaises(ContractError):
            SpeechEvents().start("bad", owner_voice={"threshold_s": 20})

    def test_nothing_after_the_session_ended(self):
        evidence = Evidence(seconds=0.0)
        events, _ = self.start(owner_voice=OwnerVoice(), evidence=evidence)
        events.end("stopped", 1000)
        evidence.seconds = 100.0
        events.expire(2000)
        self.assertEqual(owner_events(events.drain()), [])


class VoiceEmbedder(ScriptedEmbedder):
    """Returns `vector` for every window, so a test switches voices between turns."""

    def __init__(self, vector):
        super().__init__()
        self.vector = vector

    def embed(self, pcm: bytes) -> list[float]:
        self.calls += 1
        return list(self.vector)


class RouteHarness(Harness):
    def __init__(self, embedder, *, route, **options):
        super().__init__(embedder, **options)
        self.shadow = ShadowSpeakerId(
            settings(),
            report=self.lines.append,
            embedder_factory=lambda _settings: self._make(),
            entries=lambda _settings: [dict(entry) for entry in ENTRIES],
            route=route,
        )


class ShadowOwnerEvidenceTests(unittest.TestCase):
    def test_scored_speech_and_owner_binding_are_tracked(self):
        # Four 5 s turns of a voice matching neither enrolled print, then the owner.
        embedder = VoiceEmbedder(unit(0, 0, 1))
        harness = RouteHarness(embedder, route=None)
        shadow = harness.shadow
        roles = EnrolledRoles(OWNER_ID)
        roles.source = shadow
        self.assertEqual(shadow.owner_evidence(("owner",)), (False, False, 0.0))
        shadow.start()
        for _ in range(4):
            harness.speak(5.0, "Speaker A")
        wait_for(lambda: shadow.scored == 4)
        self.assertEqual(shadow.owner_evidence(("owner",)), (True, False, 20.0))
        self.assertEqual(shadow.owner_evidence(("nobody",)), (False, False, 20.0))
        self.assertEqual(roles.owner_voice(20), "unrecognized")
        self.assertIsNone(roles.owner_voice(21))
        # Overlapped and unlabelled speech is never scored, so never counts.
        harness.speak(5.0, "Speaker B", overlap=True)
        harness.speak(5.0, "Speaker B", spans=None)
        wait_for(lambda: shadow.offered == 6)
        embedder.vector = unit(1, 0, 0)
        harness.speak(4.0, "Speaker B")
        wait_for(lambda: shadow.scored == 5)
        self.assertEqual(shadow.owner_evidence(("owner",)), (True, True, 24.0))
        self.assertEqual(roles.owner_voice(20), "recognized")
        shadow.close(drain=True)

    def test_a_guest_binding_is_not_an_owner_binding(self):
        harness = RouteHarness(VoiceEmbedder(unit(0, 1, 0)), route=None)
        harness.shadow.start()
        for _ in range(5):
            harness.speak(5.0)
        wait_for(lambda: harness.shadow.scored == 5)
        self.assertEqual(harness.shadow.owner_evidence(("owner",)), (True, False, 25.0))
        harness.shadow.close(drain=True)

    def test_stderr_notes_carry_the_route(self):
        harness = RouteHarness(VoiceEmbedder(unit(1, 0, 0)), route="usb_audio")
        harness.shadow.start()
        harness.speak(4.0)
        harness.shadow.close(drain=True)
        notes = [line for line in harness.lines if line.startswith("speaker_id ")]
        self.assertGreaterEqual(len(notes), 4)  # start, label, final, summary
        for line in notes:
            with self.subTest(line=line):
                fields = dict(w.split("=", 1) for w in shlex.split(line)[1:] if "=" in w)
                self.assertEqual(fields["route"], "usb_audio")
        self.assertIn("speaker_id start enrolled=2 route=usb_audio", harness.lines)

    def test_an_invalid_route_is_never_logged(self):
        for route in ("Mick's AirPods", "USB_AUDIO", 7):
            harness = RouteHarness(VoiceEmbedder(unit(1, 0, 0)), route=route)
            harness.shadow.start()
            harness.speak(4.0)
            harness.shadow.close(drain=True)
            with self.subTest(route=route):
                self.assertTrue(harness.lines)
                self.assertFalse(any("route=" in line for line in harness.lines))


class EvidenceShadow(FakeShadow):
    """A stand-in identifier for the controller: no bindings, fixed owner evidence."""

    def __init__(self, evidence):
        super().__init__(Bindings())
        self.evidence = evidence

    def turn(self, current, *, inferred=False, spans=None):
        self.turns.append(current.utterance_id)

    def owner_evidence(self, owners):
        return self.evidence.owner_evidence(owners)


class ListenOwnerVoiceTests(unittest.TestCase):
    """`rightyo listen` reads the host's opt-in and route from the environment."""

    def run_listen(self, environ, *, roles=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset"
            asset.touch()
            demo = root / "generated-tone.wav"
            with wave.open(str(demo), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(RATE)
                audio.writeframes(tone(4.0))
            section = {"python": str(asset), "model": str(asset), "live": True}
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        **{key: str(asset) for key in LOCAL},
                        "demo_audio": str(demo),
                        "speakers": {"owner": ["owner"]},
                        "speaker_id": section | ({"roles": True} if roles else {}),
                    }
                )
            )
            shadows, routes = [], []

            def shadow(loaded_settings, *, report, **options):
                routes.append(options.get("route"))
                shadows.append(EvidenceShadow(Evidence(seconds=30.0)))
                return shadows[-1]

            def factory(loaded, *, event_publisher, report=None, **options):
                return PrototypeController(
                    loaded,
                    event_publisher=event_publisher,
                    report=report,
                    processor_factory=TurnEveryTwoSeconds,
                    capture_factory=MagicMock(side_effect=AssertionError("no microphone")),
                    provider_factory=MagicMock(side_effect=AssertionError("no hosted")),
                    speaker_id_factory=shadow,
                    **options,
                )

            args = Namespace(
                config=config,
                mode="demo",
                session_id="listen-owner-voice",
                use_jev=False,
                allow_hosted=False,
            )
            output = io.StringIO()
            cleared = {key: value for key, value in os.environ.items() if key not in HOST_ENV}
            with (
                patch.dict(os.environ, cleared | environ, clear=True),
                patch("sys.stderr", io.StringIO()) as errors,
            ):
                self.assertEqual(listen(args, output=output, controller_factory=factory), 0)
            events = [json.loads(line) for line in output.getvalue().splitlines()]
            return events, errors.getvalue(), routes

    def test_opted_in_listen_advertises_and_emits_once(self):
        events, log, routes = self.run_listen(
            {OWNER_VOICE_EVENTS_ENV: "1", INPUT_ROUTE_ENV: "car_audio"}
        )
        self.assertEqual(events[0]["owner_voice"], {"threshold_s": 20})
        self.assertEqual([e["state"] for e in owner_events(events)], ["unrecognized"])
        self.assertEqual(routes, ["car_audio"])
        self.assertNotIn(PRIVATE_TEXT, log)

    def test_listen_without_the_opt_in_is_unchanged(self):
        events, _, routes = self.run_listen({})
        self.assertNotIn("owner_voice", events[0])
        self.assertEqual(owner_events(events), [])
        self.assertEqual(routes, [None])

    def test_opt_in_without_roles_is_reported_and_silent(self):
        events, log, _ = self.run_listen({OWNER_VOICE_EVENTS_ENV: "1"}, roles=False)
        self.assertNotIn("owner_voice", events[0])
        self.assertEqual(owner_events(events), [])
        self.assertIn("rightyo: owner_voice off: speaker identification roles are not on", log)

    def test_an_invalid_route_is_ignored(self):
        _, log, routes = self.run_listen({INPUT_ROUTE_ENV: "Scarlett 2i2"})
        self.assertEqual(routes, [None])
        self.assertIn(f"rightyo: invalid {INPUT_ROUTE_ENV}; ignored", log)
        self.assertNotIn("Scarlett", log)


if __name__ == "__main__":
    unittest.main()
