"""The opt-in decision state log (rightyo#199): synthetic states, temporary files only."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from test_answered_by_other import ATTEND, Attending, gap
from test_conversation import turn

from rightyo.contracts import ContractError
from rightyo.pipeline import ReplayRunner
from rightyo.prototype import DecisionConfigError, PrototypeConfig
from rightyo.state_log import DecisionStateLog


class StateLogTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "states.jsonl"
        self.notes = []

    def test_each_decision_appends_its_state_and_answer_to_a_private_file(self):
        log = DecisionStateLog(self.path, report=self.notes.append)
        runner = ReplayRunner(Attending(), post_turn_gaps=True, state_log=log)
        runner.process(turn("t1", 0, 900, "What time is it?"), gap("none"))
        runner.process(turn("t2", 2000, 2900, "And tomorrow?"), gap("none"))
        lines = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["state"]["current_turn"]["text"], "What time is it?")
        self.assertEqual(lines[1]["state"]["past_turns"][0]["text"], "What time is it?")
        self.assertEqual(lines[0]["decision"]["label"], "attend")
        self.assertEqual(lines[0]["state"]["post_turn_gap"]["following"], "none")
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(self.notes, [])

    def test_writing_stops_at_the_size_cap_and_says_so_once(self):
        log = DecisionStateLog(self.path, max_bytes=600, report=self.notes.append)
        for _ in range(5):
            log({"current_turn": {"text": "x" * 100}}, ATTEND)
        self.assertLessEqual(self.path.stat().st_size, 600)
        self.assertEqual(self.notes, ["decision_state_log stopped reason=full"])

    def test_a_symlink_is_never_followed(self):
        target = self.root / "elsewhere"
        target.write_text("")
        self.path.symlink_to(target)
        log = DecisionStateLog(self.path, report=self.notes.append)
        log({"current_turn": {"text": "x"}}, ATTEND)
        log({"current_turn": {"text": "y"}}, ATTEND)
        self.assertEqual(target.read_text(), "")
        self.assertEqual(len(self.notes), 1)
        self.assertTrue(self.notes[0].startswith("decision_state_log stopped reason="))

    def test_a_file_that_is_not_private_or_has_another_link_is_never_written(self):
        loose = self.root / "loose.jsonl"
        loose.write_text("")
        os.chmod(loose, 0o644)
        DecisionStateLog(loose, report=self.notes.append)({"x": 1}, ATTEND)
        self.assertEqual(loose.read_text(), "")
        other = self.root / "other"
        other.write_text("")
        os.chmod(other, 0o600)
        linked = self.root / "linked.jsonl"
        os.link(other, linked)
        DecisionStateLog(linked, report=self.notes.append)({"x": 1}, ATTEND)
        self.assertEqual(other.read_text(), "")
        self.assertEqual(self.notes, ["decision_state_log stopped reason=not_a_private_file"] * 2)

    def test_a_path_inside_a_git_checkout_is_refused(self):
        checkout = self.root / "repo"
        (checkout / ".git").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            DecisionStateLog(checkout / "logs" / "states.jsonl")

    def test_the_record_keeps_the_raw_choice_and_the_answered_time(self):
        from dataclasses import replace

        DecisionStateLog(self.path)({"x": 1}, replace(ATTEND, answered_at_ms=1300))
        record = json.loads(self.path.read_text())["decision"]
        self.assertEqual((record["attention_choice"], record["answered_at_ms"]), (None, 1300))

    def test_a_failing_log_never_affects_the_decision(self):
        def broken(state, decision):
            raise RuntimeError("disk on fire")

        runner = ReplayRunner(Attending(), state_log=broken)
        event = runner.process(turn("t1", 0, 900, "What time is it?"))
        self.assertEqual(event.decision.label, "attend")
        with self.assertRaises(ContractError):
            ReplayRunner(Attending(), state_log="yes")
        with self.assertRaises(ValueError):
            DecisionStateLog(Path("relative.jsonl"))


class StateLogConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        asset = Path(directory.name) / "asset"
        asset.touch()
        keys = (
            "whisper_executable",
            "whisper_model",
            "diarization_library",
            "diarization_model",
            "microphone_helper",
        )
        self.base = {key: str(asset) for key in keys}
        self.path = Path(directory.name) / "config.json"

    def load(self, decision):
        self.path.write_text(json.dumps({**self.base, "decision": decision}))
        return PrototypeConfig.load(self.path)

    def test_off_unless_an_absolute_path_is_named_with_jev(self):
        jev = {"provider": "jev", "allow_hosted": True}
        self.assertIsNone(self.load(jev).decision_state_log)
        loaded = self.load({**jev, "state_log": "/tmp/states.jsonl"})
        self.assertEqual(loaded.decision_state_log, Path("/tmp/states.jsonl"))
        for bad in ("states.jsonl", 7, ""):
            with self.subTest(bad=bad), self.assertRaisesRegex(DecisionConfigError, "state_log"):
                self.load({**jev, "state_log": bad})
        checkout = Path(self.path.parent) / "repo"
        (checkout / ".git").mkdir(parents=True)
        with self.assertRaisesRegex(DecisionConfigError, "outside the repository"):
            self.load({**jev, "state_log": str(checkout / "states.jsonl")})
        with self.assertRaisesRegex(DecisionConfigError, "only to the jev"):
            self.load({"provider": "mock", "allow_hosted": False, "state_log": "/tmp/x.jsonl"})


if __name__ == "__main__":
    unittest.main()
