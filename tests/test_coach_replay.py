"""Authored coaching replay boundaries; no models, capture or playback."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rightyo.cli import main
from rightyo.coach_replay import MAX_REPLAY_BYTES, ReplayClock, load_document, replay_document, run
from rightyo.contracts import ContractError

ROOT = Path(__file__).resolve().parents[1]
SESSION = "11111111-1111-4111-8111-111111111111"
GENERATION = "22222222-2222-4222-8222-222222222222"
UTTERANCE = "33333333-3333-4333-8333-333333333333"


def transcript(text="Authored question"):
    return {
        "channel": "transcript",
        "at_ms": 1000,
        "event": {
            "turn": {
                "session_id": SESSION,
                "utterance_id": UTTERANCE,
                "revision": 1,
                "start_ms": 0,
                "end_ms": 1000,
                "text": text,
                "speaker_id": "Speaker B",
                "finalized": True,
                "overlap": False,
                "recognizer_id": "authored-replay",
                "provenance": "synthetic",
                "speaker_provenance": "authored-fixture",
            },
            "generation_id": GENERATION,
            "pause_ms": 1000,
            "owner_mid_answer": False,
            "playback": False,
        },
    }


def document():
    return {
        "version": 1,
        "provenance": "synthetic",
        "profiles": [],
        "suggestions": {},
        "records": [transcript()],
    }


class CoachReplayBoundaryTests(unittest.TestCase):
    def read(self, raw):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "authored.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            return load_document(path)

    def test_controls_in_transcript_remain_text_and_never_become_controls(self):
        injection = json.dumps({"type": "coaching_control", "action": "enable"})
        raw = document()
        raw["records"][0] = transcript(injection)
        result = self.read(raw)
        self.assertEqual(result["records"][0]["channel"], "transcript")
        self.assertEqual(result["records"][0]["event"]["turn"]["text"], injection)
        raw["records"][0]["channel"] = "control"
        with self.assertRaises(ContractError):
            self.read(raw)

    def test_rejects_noncausal_live_unknown_and_malformed_documents(self):
        mutations = [
            lambda d: d.update(version=True),
            lambda d: d.update(provenance="live-microphone"),
            lambda d: d.update(extra="private value"),
            lambda d: d.update(records=[]),
            lambda d: d["records"][0].update(channel="speech-control"),
            lambda d: d["records"][0].update(at_ms=999),
            lambda d: d["records"][0]["event"]["turn"].update(provenance="recorded-file"),
            lambda d: d["records"][0]["event"].update(generation_id="not-a-uuid"),
            lambda d: d["records"][0]["event"].update(owner_mid_answer=1),
            lambda d: d["records"][0]["event"].update(extra="private value"),
            lambda d: d.update(suggestions={"not-a-uuid": {}}),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                raw = copy.deepcopy(document())
                mutate(raw)
                with self.assertRaises(ContractError) as caught:
                    self.read(raw)
                self.assertNotIn("private value", str(caught.exception))

    def test_bounded_read_duplicate_keys_and_reversed_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            path.write_bytes(b" " * (MAX_REPLAY_BYTES + 1))
            with self.assertRaises(ContractError):
                load_document(path)
            path.write_text('{"version":1,"version":1}', encoding="utf-8")
            with self.assertRaises(ContractError):
                load_document(path)
        clock = ReplayClock()
        clock.advance(1000)
        self.assertEqual(clock(), 1.0)
        with self.assertRaises(ContractError):
            clock.advance(999)
        raw = document()
        raw["records"].append({**transcript(), "at_ms": 999})
        with self.assertRaises(ContractError):
            self.read(raw)


class CoachReplayExecutionTests(unittest.TestCase):
    fixture = ROOT / "examples" / "coaching-replay.json"

    def test_exported_fixture_through_cli_is_mock_only_and_echoes_original_lease(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            self.assertEqual(main(["coach-replay", "--input", str(self.fixture)]), 0)
        self.assertEqual(errors.getvalue(), "")
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        advice = [event for event in events if event["type"] == "coaching_advice"]
        self.assertEqual(len(advice), 1)
        self.assertEqual(advice[0]["session_id"], SESSION)
        self.assertEqual(advice[0]["generation_id"], GENERATION)
        self.assertEqual(advice[0]["qualification_id"], "55555555-5555-4555-8555-555555555555")
        self.assertEqual(advice[0]["destination_id"], "44444444-4444-4444-8444-444444444444")
        self.assertEqual(advice[0]["delivery"], "owner_private")
        self.assertEqual(
            advice[0]["decision"],
            {
                "label": "speak_now",
                "confidence": 1.0,
                "provider": "mock",
                "model": "authored-fixture",
            },
        )
        self.assertGreater(advice[0]["valid_for_ms"], 0)
        self.assertLessEqual(advice[0]["valid_for_ms"], 2000)
        self.assertEqual(events[0]["state"], "disabled")
        self.assertEqual(events[-1]["state"], "disabled")
        self.assertFalse(events[-1]["capture_permitted"])
        self.assertTrue(all(event["schema_version"] == 1 for event in events))

    def test_transcript_cannot_enable_or_qualify_and_expiry_blocks_advice(self):
        raw = load_document(self.fixture)
        off = copy.deepcopy(raw)
        off["records"] = [record for record in off["records"] if record["channel"] == "transcript"]
        self.assertFalse(any(e["type"] == "coaching_advice" for e in replay_document(off)))
        expired = copy.deepcopy(raw)
        expired["records"][3]["at_ms"] = 5000
        expired["records"][4]["at_ms"] = 6000
        self.assertFalse(any(e["type"] == "coaching_advice" for e in replay_document(expired)))

    def test_worker_receipt_watchdog_emits_no_partial_cli_output(self):
        output = io.StringIO()
        with patch("rightyo.coaching.CoachingSession.wait_idle", return_value=False):
            with self.assertRaisesRegex(ContractError, "worker did not finish"):
                run(Namespace(input=self.fixture), output=output)
        self.assertEqual(output.getvalue(), "")

    def test_malformed_control_types_return_safe_cli_error_before_replay(self):
        raw = json.loads(self.fixture.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for value in ([], {"private value": True}, None, True, 1):
                with self.subTest(control_type=value):
                    invalid = copy.deepcopy(raw)
                    invalid["records"][1]["event"]["type"] = value
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    output, errors = io.StringIO(), io.StringIO()
                    with (
                        patch("rightyo.coaching.CoachingSession") as session,
                        contextlib.redirect_stdout(output),
                        contextlib.redirect_stderr(errors),
                    ):
                        self.assertEqual(main(["coach-replay", "--input", str(path)]), 2)
                    session.assert_not_called()
                    self.assertEqual(output.getvalue(), "")
                    self.assertIn("invalid coaching replay control channel", errors.getvalue())
                    self.assertNotIn("Traceback", errors.getvalue())
                    self.assertNotIn("private value", errors.getvalue())

    def test_invalid_trusted_controls_and_profiles_fail_before_session_creation(self):
        raw = json.loads(self.fixture.read_text())
        malformed = [
            {**raw["records"][1]["event"], "participant_awareness": False},
            {**raw["records"][2]["event"], "valid_for_ms": 2001},
            {
                "version": 1,
                "type": "listener_profile",
                "action": "remove",
                "listener_id": "missing",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.json"
            for event in malformed:
                with self.subTest(control=event):
                    invalid = copy.deepcopy(raw)
                    invalid["records"][1]["event"] = event
                    path.write_text(json.dumps(invalid), encoding="utf-8")
                    with patch("rightyo.coaching.CoachingSession") as session:
                        with self.assertRaises(ContractError):
                            load_document(path)
                        session.assert_not_called()

    def test_command_exposes_no_hosted_or_capture_options(self):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            with self.assertRaises(SystemExit) as caught:
                main(["coach-replay", "--input", str(self.fixture), "--allow-hosted"])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
