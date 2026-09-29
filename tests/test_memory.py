from __future__ import annotations

import threading
import unittest
from dataclasses import replace
from unittest.mock import MagicMock

from rightyo.contracts import ContractError, Turn
from rightyo.memory import TranscriptMemory
from rightyo.pipeline import ReplayRunner
from rightyo.providers import MockProvider, ProviderError


def turn(index=0, **changes):
    return replace(
        Turn(
            "session",
            f"turn-{index}",
            1,
            index * 10000,
            (index + 1) * 10000,
            "Speaker B, what do you think?",
            "Speaker A",
            True,
            False,
            "authored-fixture",
            "synthetic",
            "authored-fixture",
        ),
        **changes,
    )


class TranscriptMemoryTests(unittest.TestCase):
    def test_end_time_expiry_discloses_whole_boundary_turn(self):
        memory = TranscriptMemory(retention_ms=60000)
        memory.append(turn(0))
        memory.append(turn(1, start_ms=50000, end_ms=80000))
        snapshot = memory.snapshot(120000)
        self.assertEqual([item["utterance_id"] for item in snapshot["turns"]], ["turn-1"])
        self.assertEqual(snapshot["retention"]["boundary_overlap_ms"], 10000)
        self.assertEqual(snapshot["retention"]["expired_turns"], 1)
        self.assertEqual(memory.snapshot(140000)["turns"], [])

    def test_snapshot_is_detached_and_idle_expiry_cannot_move_clock_back(self):
        memory = TranscriptMemory()
        memory.append(turn())
        snapshot = memory.snapshot(10000)
        snapshot["turns"][0]["text"] = "changed outside memory"
        snapshot["retention"]["max_turns"] = 0
        self.assertNotEqual(
            memory.snapshot(10000)["turns"][0]["text"], snapshot["turns"][0]["text"]
        )
        memory.expire(310000)
        memory.append(turn())
        self.assertEqual(memory.snapshot(0)["turns"], [])
        self.assertEqual(memory.snapshot(0)["retention"]["now_ms"], 310000)

    def test_capacity_accounts_for_utf8_and_reports_early_eviction(self):
        memory = TranscriptMemory(max_bytes=17000)
        memory.append(turn(0, text="😀" * 4000))
        memory.append(turn(1, text="😀" * 4000))
        snapshot = memory.snapshot(20000)
        self.assertEqual(len(snapshot["turns"]), 1)
        self.assertLessEqual(snapshot["retention"]["retained_bytes"], 17000)
        self.assertEqual(snapshot["retention"]["capacity_evicted_turns"], 1)
        memory = TranscriptMemory(max_turns=1)
        memory.append(turn(0))
        memory.append(turn(1))
        self.assertEqual(memory.snapshot(20000)["turns"][0]["utterance_id"], "turn-1")

    def test_final_only_session_order_duplicate_and_revision_rules(self):
        memory = TranscriptMemory()
        with self.assertRaises(ContractError):
            memory.append(turn(finalized=False))
        memory.append(turn())
        memory.append(turn())
        self.assertEqual(len(memory.snapshot(10000)["turns"]), 1)
        for changed in (
            turn(revision=2),
            turn(text="conflicting"),
            turn(session_id="other"),
            turn(1, start_ms=0, end_ms=9999),
        ):
            with self.assertRaises(ContractError):
                memory.append(changed)
        memory.clear()
        memory.append(turn(session_id="other"))
        self.assertEqual(memory.snapshot(10000)["retention"]["expired_turns"], 0)

    def test_memory_budget_validation(self):
        for changes in (
            {"retention_ms": True},
            {"retention_ms": 900001},
            {"max_turns": 1001},
            {"max_bytes": 4194305},
            {"max_bytes": 0},
        ):
            with self.assertRaises(ContractError):
                TranscriptMemory(**changes)


class RunnerMemoryTests(unittest.TestCase):
    def test_minute_discussion_is_retained_separately_from_small_decision_context(self):
        memory = TranscriptMemory()
        provider = MagicMock(wraps=MockProvider())
        runner = ReplayRunner(provider, memory=memory)
        for index in range(12):
            event = runner.process(turn(index, speaker_id=f"Speaker {index % 2}"))
            self.assertEqual(event.decision.label, "ignore")
        event = runner.process(turn(12, text="Rightyo, check what we were discussing."))
        self.assertEqual(event.decision.label, "attend")
        self.assertEqual(len(provider.decide.call_args.args[0]["past_turns"]), 8)
        snapshot = runner.snapshot(130000)
        self.assertEqual(len(snapshot["turns"]), 13)
        self.assertEqual(snapshot["turns"][0]["speaker_id"], "Speaker 0")
        self.assertNotIn("retention", provider.decide.call_args.args[0])
        self.assertFalse(snapshot["decisions"]["incomplete"])

    def test_idle_expiry_clears_plaintext_but_old_ids_never_emit_again(self):
        memory = TranscriptMemory()
        provider = MagicMock(wraps=MockProvider())
        runner = ReplayRunner(provider, memory=memory)
        runner.process(turn(text="private retained transcript"))
        runner.expire(310000)
        self.assertEqual(len(runner._history), 0)
        self.assertNotIn("private retained transcript", repr(runner._latest))
        self.assertIsNone(runner.process(turn(text="private retained transcript")))
        with self.assertRaises(ContractError):
            runner.process(turn(revision=2, end_ms=400000))
        self.assertEqual(provider.decide.call_count, 1)
        self.assertEqual(runner.snapshot(400000)["turns"], [])

    def test_partial_plaintext_and_rejected_revision_do_not_corrupt_retention(self):
        runner = ReplayRunner(MockProvider(), memory=TranscriptMemory())
        runner.process(turn(text="keep this committed discussion"))
        runner.process(turn(1, finalized=False, text="private uncommitted text"))
        self.assertNotIn("private uncommitted text", repr(runner._latest))
        self.assertEqual(len(runner.snapshot(20000)["turns"]), 1)
        with self.assertRaises(ContractError):
            runner.process(turn(text="conflict", end_ms=500000))
        self.assertEqual(len(runner.snapshot(20000)["turns"]), 1)

    def test_byte_capacity_removes_small_context_copy_as_well(self):
        runner = ReplayRunner(MockProvider(), memory=TranscriptMemory(max_bytes=1))
        runner.process(turn())
        self.assertEqual(len(runner._history), 0)
        self.assertEqual(runner.snapshot(10000)["turns"], [])

    def test_provider_failure_preserves_transcript_but_is_incomplete(self):
        provider = MagicMock()
        provider.decide.side_effect = ProviderError("Jev request failed")
        runner = ReplayRunner(provider, memory=TranscriptMemory())
        with self.assertRaises(ProviderError):
            runner.process(turn())
        snapshot = runner.snapshot(10000)
        self.assertEqual(len(snapshot["turns"]), 1)
        self.assertEqual(snapshot["decisions"], {"completed": 0, "failed": 1, "incomplete": True})
        self.assertIsNone(runner.process(turn()))
        self.assertEqual(provider.decide.call_count, 1)
        with self.assertRaises(ContractError):
            runner.process(turn(revision=2))

    def test_no_speaker_ablation_retains_local_labels_without_provider_leak(self):
        provider = MagicMock(wraps=MockProvider())
        runner = ReplayRunner(provider, memory=TranscriptMemory(), no_speakers=True)
        runner.process(turn())
        runner.process(turn(1, speaker_id="Speaker B", text="Ambiguous follow-up"))
        state = provider.decide.call_args.args[0]
        self.assertEqual(state["known_participants"], [])
        self.assertTrue(all(item["speaker_id"] is None for item in state["past_turns"]))
        self.assertIsNone(state["current_turn"]["speaker_id"])
        self.assertEqual(runner.snapshot(20000)["turns"][0]["speaker_id"], "Speaker A")

    def test_stop_clears_memory_without_waiting_for_inflight_provider(self):
        entered, release = threading.Event(), threading.Event()
        provider = MagicMock()
        runner = ReplayRunner(provider, memory=TranscriptMemory(), expected_reply="private context")
        results = []

        def decide(state):
            entered.set()
            release.wait(2)
            return MockProvider().decide(state)

        provider.decide.side_effect = decide
        worker = threading.Thread(target=lambda: results.append(runner.process(turn())))
        worker.start()
        self.assertTrue(entered.wait(1))
        runner.clear()
        self.assertEqual(runner.snapshot(10000)["turns"], [])
        self.assertIsNone(runner.expected_reply)
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results, [None])
        self.assertEqual(runner.snapshot(10000)["turns"], [])

    def test_shared_memory_can_show_turn_before_decision_without_duplication(self):
        memory = TranscriptMemory()
        runner = ReplayRunner(MockProvider(), memory=memory)
        runner.restart("session")
        memory.append(turn())
        runner.process(turn())
        self.assertEqual(len(runner.snapshot(10000)["turns"]), 1)
        runner.restart("new-session")
        self.assertEqual(runner.snapshot(0)["turns"], [])


if __name__ == "__main__":
    unittest.main()
