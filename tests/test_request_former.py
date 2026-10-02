"""Optional request forming: a bounded convenience string beside the raw turns (#49 item 4)."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rightyo.cli import forming_from_args, main
from rightyo.contracts import (
    MAX_FORMED_REQUEST_CHARS,
    MAX_TEXT_CHARS,
    Addressing,
    ContractError,
    DecisionEvent,
    ProviderDecision,
    RequestForming,
    SpeakerPriority,
    Turn,
    formed_request_text,
)
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import (
    CONTEXT_MARKER,
    OMITTED_MARKER,
    ConfiguredPriorityProvider,
    MockProvider,
    TemplateRequestFormer,
    request_former_for,
)
from rightyo.tool import listen
from rightyo.tool_events import SpeechEvents, encode_json

ROOT = Path(__file__).resolve().parents[1]
MARK = f" {CONTEXT_MARKER}."
# Digests of the shared fixtures at the head this change builds on: request forming is
# off by default, so neither byte changes.
UNCHANGED_FIXTURES = {
    "tool-events.jsonl": "47a16fc4383db1d141f6f8b215c621ba84750bcb6c6aa128dfc47567eb6c8311",
    "enrolled-override.jsonl": "d003d834cc3b826f09cd45df426d8dc4644eb93af5dfdebb1134832ca2bc53b3",
}
FIXTURE_TURNS = [
    {
        "session_id": "formed-demo",
        "utterance_id": "discussion",
        "revision": 1,
        "start_ms": 0,
        "end_ms": 500,
        "text": "Speaker A, the project is finished.",
        "speaker_id": "Speaker B",
        "finalized": True,
        "overlap": False,
        "recognizer_id": "authored-fixture",
        "provenance": "synthetic",
        "speaker_provenance": "authored-fixture",
    },
    {
        "session_id": "formed-demo",
        "utterance_id": "request",
        "revision": 1,
        "start_ms": 1000,
        "end_ms": 2000,
        "text": "Rightyo, archive the project.",
        "speaker_id": "Speaker A",
        "finalized": True,
        "overlap": False,
        "recognizer_id": "authored-fixture",
        "provenance": "synthetic",
        "speaker_provenance": "authored-fixture",
    },
]
FIXTURE_FORMED = (
    'Owner (Speaker A) asked: "Rightyo, archive the project.". '
    'Earlier, participant (Speaker B) said: "Speaker A, the project is finished."' + MARK
)


def record(text, speaker="Speaker B", role=None, overlap=False, start=0, end=500):
    turn = {
        "session_id": "formed-demo",
        "utterance_id": f"turn-{start}",
        "revision": 1,
        "start_ms": start,
        "end_ms": end,
        "text": text,
        "speaker_id": speaker,
        "finalized": True,
        "overlap": overlap,
        "recognizer_id": "authored-fixture",
        "provenance": "synthetic",
        "speaker_provenance": "authored-fixture",
    }
    if role is not None:
        turn["role"] = role
    return turn


def state(current, *context, speakers="enrolled", addressing=None):
    return {
        "current_turn": current,
        "context_turns": list(context),
        "addressing": addressing,
        "speakers": speakers,
    }


def turn(name, start, end, text, speaker="Speaker A", session="formed-demo", overlap=False):
    return Turn(
        session,
        name,
        1,
        start,
        end,
        text,
        speaker,
        True,
        overlap,
        "authored-fixture",
        "synthetic",
        "authored-fixture",
    )


def decision(current, label="attend", recipient="system"):
    return DecisionEvent(
        current,
        ProviderDecision(
            label,
            recipient,
            1.0,
            {k: float(k == label) for k in ("attend", "ignore", "uncertain")},
            "mock-v1",
            "mock",
            1.0,
        ),
        current.revision,
        0.0,
        0.0,
    )


class TemplateRequestFormerTests(unittest.TestCase):
    def setUp(self):
        self.former = TemplateRequestFormer()

    def test_owner_request_with_participant_context_matches_golden(self):
        current = record(
            "Rightyo, archive the project.", "Speaker A", "owner", start=1000, end=2000
        )
        context = record("Speaker A, the project is finished.", "Speaker B", "participant")
        self.assertEqual(self.former.form(state(current, context)), FIXTURE_FORMED)

    def test_empty_context_renders_only_the_request(self):
        current = record("Rightyo, what time is it?", "Speaker A", "owner")
        self.assertEqual(
            self.former.form(state(current)),
            'Owner (Speaker A) asked: "Rightyo, what time is it?".',
        )

    def test_unknown_overlap_trusted_and_anonymous_speakers_are_labelled(self):
        current = record("Rightyo, go on.", "Speaker A", "owner", start=3000, end=3500)
        unknown = record("Nobody knows who said this.", None, "unknown", start=0, end=500)
        overlap = record("Two at once.", "Speaker C", "participant", True, 1000, 1500)
        trusted = record("I trust this.", "Speaker B", "trusted", start=2000, end=2500)
        self.assertEqual(
            self.former.form(state(current, unknown, overlap, trusted)),
            'Owner (Speaker A) asked: "Rightyo, go on.". '
            'Earlier, an unknown speaker said: "Nobody knows who said this."'
            + MARK
            + ' Earlier, participant (Speaker C, overlapping speech) said: "Two at once."'
            + MARK
            + ' Earlier, trusted speaker (Speaker B) said: "I trust this."'
            + MARK,
        )
        anonymous = record("Rightyo, hello.", "Speaker A", overlap=True, start=1000, end=1500)
        unlabelled = record("Hi.", None, overlap=True)
        self.assertEqual(
            self.former.form(state(anonymous, unlabelled, speakers="anonymous")),
            'Speaker (Speaker A, overlapping speech) asked: "Rightyo, hello.". '
            'Earlier, an unknown speaker (overlapping speech) said: "Hi."' + MARK,
        )

    def test_non_owner_request_is_marked_and_owner_context_stays_context(self):
        current = record("Rightyo, delete it.", "Speaker B", "participant", start=1000, end=1500)
        context = record("The project is finished.", "Speaker A", "owner")
        self.assertEqual(
            self.former.form(state(current, context)),
            'Participant (Speaker B) asked: "Rightyo, delete it.". The requester is not an '
            'owner. Earlier, owner (Speaker A) said: "The project is finished."' + MARK,
        )

    def test_most_recent_context_is_last_whatever_the_supplied_order(self):
        current = record("Rightyo, which?", "Speaker A", "owner", start=5000, end=5500)
        first = record("First.", "Speaker B", "participant", start=0, end=500)
        second = record("Second.", "Speaker B", "participant", start=1000, end=1500)
        formed = self.former.form(state(current, second, first))
        self.assertLess(formed.index('"First."'), formed.index('"Second."'))

    def test_truncation_drops_oldest_context_first_and_never_the_request(self):
        request_text = "Rightyo, " + "r" * (MAX_TEXT_CHARS - 9)
        current = record(request_text, "Speaker A", "owner", start=9000, end=9500)
        context = [
            record(str(index) * MAX_TEXT_CHARS, "Speaker B", "participant", False, index * 1000)
            for index in range(1, 6)
        ]
        formed = self.former.form(state(current, *context))
        self.assertLessEqual(len(formed), MAX_FORMED_REQUEST_CHARS)
        self.assertTrue(formed.startswith(f'Owner (Speaker A) asked: "{request_text}". '))
        self.assertIn(OMITTED_MARKER, formed)
        self.assertNotIn("1" * MAX_TEXT_CHARS, formed)
        self.assertNotIn("2" * MAX_TEXT_CHARS, formed)
        self.assertIn("5" * MAX_TEXT_CHARS, formed)
        self.assertTrue(formed.endswith(MARK))
        kept = [index for index in range(1, 6) if str(index) * MAX_TEXT_CHARS in formed]
        self.assertEqual(kept, list(range(kept[0], 6)))
        # Everything fits: nothing is dropped and no omission is claimed.
        small = self.former.form(state(current, context[-1]))
        self.assertNotIn(OMITTED_MARKER, small)

    def test_formed_text_validation_and_configuration_values(self):
        self.assertEqual(formed_request_text("x"), "x")
        for invalid in (None, 5, "", "x" * (MAX_FORMED_REQUEST_CHARS + 1), "\udc80"):
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ContractError):
                formed_request_text(invalid)
        self.assertEqual(
            RequestForming.from_dict({"kind": "template"}).to_dict(), {"kind": "template"}
        )
        for invalid in ({"kind": "bogus"}, {"kind": "template", "x": 1}, "template", [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(ContractError):
                RequestForming.from_dict(invalid)
        with self.assertRaises(ContractError):
            RequestForming(None)
        self.assertIsNone(request_former_for(None))
        self.assertIsInstance(request_former_for(RequestForming("template")), TemplateRequestFormer)


class _Recorder:
    """A former that records its state and returns whatever it was told to."""

    kind = "template"

    def __init__(self, result=FIXTURE_FORMED):
        self.result = result
        self.states = []

    def form(self, state):
        self.states.append(state)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class SpeechEventsFormingTests(unittest.TestCase):
    def run_session(self, former=None, priority=None, addressing=None):
        events = SpeechEvents()
        events.start("formed-demo", addressing=addressing, priority=priority, former=former)
        context = turn("discussion", 0, 500, "Speaker A, the project is finished.", "Speaker B")
        request = turn("request", 1000, 2000, "Rightyo, archive the project.")
        events.transcript(context, 500)
        events.decision(decision(context, "ignore", "other_human"), 500)
        events.transcript(request, 2000)
        events.decision(decision(request), 2000)
        events.end("stopped", 2000)
        return events, events.drain()

    def test_off_by_default_emits_neither_field_on_anonymous_or_enrolled_sessions(self):
        owner = ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",)))
        for priority in (None, owner):
            with self.subTest(enrolled=priority is not None):
                _events, stream = self.run_session(priority=priority)
                self.assertNotIn("request_forming", stream[0])
                request = next(e for e in stream if e["type"] == "request")
                self.assertNotIn("formed_request", request)

    def test_template_former_adds_the_string_and_leaves_the_raw_turns_unchanged(self):
        owner = ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",)))
        _plain, without = self.run_session(priority=owner)
        _formed, with_former = self.run_session(TemplateRequestFormer(), owner)
        self.assertEqual(with_former[0]["request_forming"], {"kind": "template"})
        self.assertEqual(with_former[0]["capabilities"], without[0]["capabilities"])
        request = next(e for e in with_former if e["type"] == "request")
        plain = next(e for e in without if e["type"] == "request")
        self.assertEqual(request.pop("formed_request"), FIXTURE_FORMED)
        self.assertEqual(request, plain)
        self.assertEqual(
            [e for e in with_former[1:] if e["type"] != "request"], without[1:-2] + without[-1:]
        )

    def test_former_state_is_bounded_to_the_event_fields(self):
        owner = ConfiguredPriorityProvider(SpeakerPriority(owners=("Speaker A",)))
        recorder = _Recorder()
        self.run_session(recorder, owner, Addressing(("Hailing Station",)))
        (state,) = recorder.states
        self.assertEqual(set(state), {"current_turn", "context_turns", "addressing", "speakers"})
        self.assertEqual(state["speakers"], "enrolled")
        self.assertEqual(state["addressing"], {"names": ["Hailing Station"]})
        self.assertEqual(state["current_turn"]["role"], "owner")
        self.assertEqual([t["role"] for t in state["context_turns"]], ["participant"])
        anonymous = _Recorder()
        self.run_session(anonymous)
        self.assertEqual(anonymous.states[0]["speakers"], "anonymous")
        self.assertIsNone(anonymous.states[0]["addressing"])
        self.assertNotIn("role", anonymous.states[0]["current_turn"])

    def test_invalid_former_output_fails_closed_without_a_dropped_field(self):
        for result in (RuntimeError("boom"), 5, "", "x" * (MAX_FORMED_REQUEST_CHARS + 1)):
            with self.subTest(result=type(result).__name__):
                events = SpeechEvents()
                events.start("formed-demo", former=_Recorder(result))
                request = turn("request", 1000, 2000, "Rightyo, archive the project.")
                events.transcript(request, 2000)
                events.drain()
                with self.assertRaisesRegex(ContractError, "request forming failed"):
                    events.decision(decision(request), 2000)
                self.assertEqual(events._pending, {})
                events.end("error", 2100, "replay-failed")
                self.assertEqual([e["type"] for e in events.drain()], ["session"])
                with self.assertRaises(ContractError):
                    events.transcript(turn("later", 3000, 4000, "Rightyo, again."), 4000)

    def test_invalid_former_objects_are_rejected_at_start(self):
        class NoForm:
            kind = "template"

        class BadKind:
            kind = "bad\nkind"

            def form(self, state):
                return "x"

        for former in (NoForm(), BadKind(), object()):
            with self.subTest(former=type(former).__name__), self.assertRaises(ContractError):
                SpeechEvents().start("formed-demo", former=former)

    def test_formed_request_counts_toward_the_event_byte_budget(self):
        _events, without = self.run_session()
        plain = next(e for e in without if e["type"] == "request")
        limit = len(encode_json(plain)) + 1
        with patch("rightyo.tool_events.MAX_EVENT_BYTES", limit):
            self.run_session()
            with self.assertRaisesRegex(ContractError, "consumer backlog"):
                self.run_session(TemplateRequestFormer())


class _Processor:
    def __init__(self, config, callback):
        self.config, self.callback = config, callback
        self.index = 0

    def push_pcm16(self, _pcm):
        self.index += 1
        speaker = "Speaker A" if self.index % 2 else "Speaker B"
        self.callback(
            turn(
                f"live-{self.index}",
                (self.index - 1) * 200,
                self.index * 200,
                "Rightyo, what happened?",
                speaker,
                self.config.session_id,
            )
        )

    def finish(self):
        pass

    def close(self):
        pass


def _refuse(*_args, **_kwargs):
    raise AssertionError("request former tests must not capture audio or use hosted inference")


class _LocalDecider:
    """Stands in for the hosted provider with the mock rule; never touches the network."""

    def __init__(self):
        self.requests = 0
        self.min_confidence = 0.7

    def decide(self, state):
        self.requests += 1
        return MockProvider().decide(state)


class CommandLineAndConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.input = self.root / "formed-input.json"
        self.input.write_text(json.dumps({"schema_version": 1, "turns": FIXTURE_TURNS}))

    def write_config(self, extra, name="config.json"):
        asset = self.root / "asset"
        asset.touch()
        demo = self.root / "authored.wav"
        with wave.open(str(demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(12800))
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
        config = self.root / name
        config.write_text(json.dumps({**base, "demo_audio": str(demo), **extra}))
        return config

    def test_flags_parse_validate_and_default_to_none(self):
        self.assertIsNone(forming_from_args(Namespace()))
        self.assertIsNone(forming_from_args(Namespace(request_former=None)))
        self.assertEqual(
            forming_from_args(Namespace(request_former="template")), RequestForming("template")
        )
        with self.assertRaises(ContractError):
            forming_from_args(Namespace(request_former="bogus"))

    def test_prototype_config_accepts_template_and_rejects_other_values(self):
        loaded = PrototypeConfig.load(self.write_config({"request_former": {"kind": "template"}}))
        self.assertEqual(loaded.request_former, RequestForming("template"))
        self.assertIsNone(PrototypeConfig.load(self.write_config({})).request_former)
        for invalid in ({"kind": "bogus"}, {"kind": "template", "extra": 1}, "template", []):
            with self.subTest(invalid=invalid), self.assertRaises(PrototypeError):
                PrototypeConfig.load(self.write_config({"request_former": invalid}))

    def test_tool_replay_flag_selects_the_former_and_rejects_unknown_kinds(self):
        for flags in ([], ["--request-former", "template"]):
            with self.subTest(flags=flags):
                output = io.StringIO()
                with patch("sys.stdout", output):
                    code = main(["tool-replay", "--input", str(self.input), *flags])
                self.assertEqual(code, 0)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                request = next(e for e in events if e["type"] == "request")
                self.assertEqual("request_forming" in events[0], bool(flags))
                self.assertEqual("formed_request" in request, bool(flags))
                self.assertEqual(events[-1]["phase"], "stopped")
        with patch("sys.stderr", io.StringIO()) as error, self.assertRaises(SystemExit) as exit:
            main(["tool-replay", "--input", str(self.input), "--request-former", "bogus"])
        self.assertEqual(exit.exception.code, 2)
        self.assertIn("invalid choice", error.getvalue())

    def test_listen_applies_the_file_section_and_the_flag_replaces_it(self):
        loaded = []

        def factory(config, *, event_publisher):
            loaded.append(config.request_former)
            oracle = _LocalDecider()
            return PrototypeController(
                config,
                event_publisher=event_publisher,
                processor_factory=_Processor,
                capture_factory=_refuse,
                provider_factory=lambda **_options: oracle,
            )

        plain = self.write_config({"speakers": {"owner": ["Speaker A"]}})
        configured = self.write_config(
            {"speakers": {"owner": ["Speaker A"]}, "request_former": {"kind": "template"}},
            "configured.json",
        )
        cases = ((plain, None, False), (plain, "template", True), (configured, None, True))
        for index, (config, flag, formed) in enumerate(cases):
            with self.subTest(flag=flag, configured=config is configured):
                args = Namespace(
                    config=config,
                    mode="demo",
                    session_id=f"formed-{index}",
                    use_jev=True,
                    allow_hosted=True,
                    names=None,
                    request_former=flag,
                )
                output = io.StringIO()
                self.assertEqual(listen(args, output=output, controller_factory=factory), 0)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual("request_forming" in events[0], formed)
                requests = [e for e in events if e["type"] == "request"]
                self.assertTrue(requests)
                self.assertTrue(all(("formed_request" in e) == formed for e in requests))
                if formed:
                    self.assertEqual(events[0]["request_forming"], {"kind": "template"})
                    for request in requests:
                        who = "Owner" if request["turn"]["role"] == "owner" else "Participant"
                        self.assertTrue(request["formed_request"].startswith(f"{who} (Speaker "))
                self.assertEqual(events[-1]["phase"], "stopped")
        self.assertEqual(loaded, [None, RequestForming("template"), RequestForming("template")])
        with self.assertRaises(ContractError):
            listen(
                Namespace(
                    config=plain,
                    mode="demo",
                    session_id="formed-bad",
                    use_jev=False,
                    allow_hosted=False,
                    request_former="bogus",
                ),
                output=io.StringIO(),
                controller_factory=factory,
            )

    def test_authored_formed_request_fixture_matches_tool_replay_bytes(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            code = main(
                [
                    "tool-replay",
                    "--input",
                    str(self.input),
                    "--owner",
                    "Speaker A",
                    "--request-former",
                    "template",
                ]
            )
        self.assertEqual(code, 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        canonical = "".join(
            json.dumps(e, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n" for e in events
        )
        fixture = ROOT / "examples" / "enrolled-formed-request.jsonl"
        self.assertEqual(canonical, fixture.read_text())
        self.assertEqual(events[0]["request_forming"], {"kind": "template"})
        self.assertEqual(events[0]["capabilities"]["speakers"], "enrolled")
        (request,) = [e for e in events if e["type"] == "request"]
        self.assertEqual(request["formed_request"], FIXTURE_FORMED)
        self.assertEqual(request["turn"]["role"], "owner")
        self.assertEqual([t["role"] for t in request["context"]["turns"]], ["participant"])

    def test_shared_fixtures_are_byte_identical_and_carry_no_forming_fields(self):
        for name, digest in UNCHANGED_FIXTURES.items():
            with self.subTest(fixture=name):
                content = (ROOT / "examples" / name).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), digest)
                for event in (json.loads(line) for line in content.decode().splitlines()):
                    self.assertNotIn("formed_request", event)
                    self.assertNotIn("request_forming", event)


if __name__ == "__main__":
    unittest.main()
