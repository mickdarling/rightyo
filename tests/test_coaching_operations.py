"""Authored schema/state tests; no model, network, transcript export or audio."""

import copy
import json
import threading
import unittest
from dataclasses import FrozenInstanceError, replace
from unittest.mock import patch

from rightyo.coaching_operations import (
    MAX_OPERATION_DISPATCHES,
    ActiveCoachingOperations,
    CoachingCategory,
    CompiledCoachingGoal,
    parse_compiled_goal,
)
from rightyo.contracts import ContractError


def goal(operation_id="brief", category_id="clarity"):
    return {
        "schema_version": 1,
        "categories": [{"category_id": category_id, "description": "Authored clarity focus"}],
        "operations": [
            {
                "operation_id": operation_id,
                "category_id": category_id,
                "goal": "Help me explain the point briefly",
                "speak_when": "A shorter explanation would help",
                "stay_quiet_when": "The owner is speaking",
                "delivery": "owner_private",
            }
        ],
    }


def compiled(operation_id="brief", category_id="clarity"):
    return CompiledCoachingGoal.from_dict(goal(operation_id, category_id))


def control(action, revision=0, **extra):
    return {
        "schema_version": 1,
        "type": "coaching_operations_control",
        "session_id": "authored-session",
        "expected_revision": revision,
        "action": action,
        **extra,
    }


class CompiledGoalTests(unittest.TestCase):
    def test_round_trip_detached_immutable_model_neutral_descriptions(self):
        raw = goal()
        result = parse_compiled_goal(json.dumps(raw).encode())
        raw["operations"][0]["goal"] = "Changed external input"
        self.assertEqual(result, compiled())
        exported = result.to_dict()
        exported["categories"][0]["description"] = "Changed export"
        self.assertEqual(result, compiled())
        with self.assertRaises(FrozenInstanceError):
            result.operations[0].goal = "Changed immutable operation"

    def test_two_authored_adapter_fixtures_consume_same_neutral_schema(self):
        # Distinct adapter representations, not model/provider execution.
        data = compiled().to_dict()
        choice_fixture = {item["operation_id"]: item["goal"] for item in data["operations"]}
        prose_fixture = {
            "focuses": [item["description"] for item in data["categories"]],
            "instructions": [
                {
                    "purpose": item["goal"],
                    "helpful": item["speak_when"],
                    "hold": item["stay_quiet_when"],
                }
                for item in data["operations"]
            ],
        }
        self.assertEqual(choice_fixture["brief"], prose_fixture["instructions"][0]["purpose"])
        self.assertEqual(prose_fixture["focuses"], ["Authored clarity focus"])
        self.assertEqual(CompiledCoachingGoal.from_dict(data), compiled())
        self.assertNotIn("jev", json.dumps(data).casefold())

    def test_exact_fields_types_and_version_fail_closed(self):
        mutations = [
            lambda raw: raw.update(schema_version=True),
            lambda raw: raw.update(schema_version=2),
            lambda raw: raw.update(model="guessed-provider"),
            lambda raw: raw.update(categories={}),
            lambda raw: raw["categories"][0].update(description=7),
            lambda raw: raw["operations"][0].update(execute="host command"),
            lambda raw: raw["operations"][0].pop("speak_when"),
            lambda raw: raw["operations"][0].update(delivery="room"),
            lambda raw: raw["operations"][0].update(delivery=[]),
            lambda raw: raw["operations"][0].update(operation_id="has space"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = goal()
                mutation(raw)
                with self.assertRaises(ContractError):
                    CompiledCoachingGoal.from_dict(raw)

    def test_count_duplicate_and_reference_integrity(self):
        mutations = [
            lambda raw: raw.update(categories=[]),
            lambda raw: raw.update(operations=[]),
            lambda raw: raw.update(categories=raw["categories"] * 9),
            lambda raw: raw.update(operations=raw["operations"] * 17),
            lambda raw: raw["categories"].append(copy.deepcopy(raw["categories"][0])),
            lambda raw: raw["operations"].append(copy.deepcopy(raw["operations"][0])),
            lambda raw: raw["operations"][0].update(category_id="absent"),
            lambda raw: raw["categories"].append(
                {"category_id": "unused", "description": "Authored unused category"}
            ),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                raw = goal()
                mutation(raw)
                with self.assertRaises(ContractError):
                    CompiledCoachingGoal.from_dict(raw)

    def test_text_bounds_controls_surrogates_and_error_privacy(self):
        for text in ("", " ", "x" * 513, "private\x00detail", "private\nline", "private\ud800"):
            with self.subTest(text_size=len(text)):
                raw = goal()
                raw["operations"][0]["goal"] = text
                with self.assertRaisesRegex(
                    ContractError, "^invalid coaching operation description$"
                ):
                    CompiledCoachingGoal.from_dict(raw)

    def test_ascii_wire_budget_counts_nonascii_expansion(self):
        raw = goal()
        raw["operations"] = []
        for index in range(6):
            operation = goal(f"op{index}")["operations"][0]
            operation.update(goal="\u4e2d" * 512, speak_when="\u4e2d" * 512)
            raw["operations"].append(operation)
        with self.assertRaisesRegex(ContractError, "byte budget"):
            CompiledCoachingGoal.from_dict(raw)

    def test_json_duplicates_constants_malformed_depth_and_size_are_rejected(self):
        cases = [
            b'{"schema_version":1,"schema_version":1}',
            json.dumps(goal()).replace('"goal":', '"goal": "duplicate", "goal":'),
            '{"schema_version":NaN}',
            '{"schema_version":Infinity}',
            b"\xff",
            "[" * 1200 + "]" * 1200,
            "x" * 16385,
            "authored private malformed response",
            {},
        ]
        for raw in cases:
            with self.subTest(type=type(raw).__name__):
                with self.assertRaises(ContractError) as error:
                    parse_compiled_goal(raw)
                self.assertNotIn("private", str(error.exception))

    def test_direct_construction_enforces_tuple_immutable_validated_items(self):
        for categories, operations in (
            (list(compiled().categories), compiled().operations),
            (compiled().categories, list(compiled().operations)),
            (({},), compiled().operations),
            (compiled().categories, ({},)),
        ):
            with self.assertRaises(ContractError):
                CompiledCoachingGoal(categories, operations)
        with self.assertRaises(ContractError):
            CoachingCategory("valid", "")


class ActiveOperationTests(unittest.TestCase):
    def setUp(self):
        self.registry = ActiveCoachingOperations("authored-session")

    def test_host_add_list_remove_and_visible_detached_state(self):
        first = self.registry.apply_control(control("add", compiled_goal=goal()))
        second = self.registry.apply_control(control("add", 1, compiled_goal=goal("question")))
        listed = self.registry.apply_control(control("list", 2))
        self.assertEqual(listed, second)
        self.assertEqual(listed.revision, 2)
        self.assertEqual(len(listed.categories), 1)
        self.assertEqual([item.operation_id for item in listed.operations], ["brief", "question"])
        exported = listed.to_dict()
        self.assertEqual(exported["type"], "coaching_operations_state")
        self.assertEqual(exported["session_id"], "authored-session")
        exported["operations"].clear()
        self.assertEqual(self.registry.snapshot(), second)
        remaining = self.registry.apply_control(control("remove", 2, operation_id="brief"))
        self.assertEqual(remaining.revision, 3)
        empty = self.registry.apply_control(control("remove", 3, operation_id="question"))
        self.assertEqual((empty.categories, empty.operations), ((), ()))
        self.assertEqual(len(first.operations), 1)
        self.assertEqual(first.revision, 1)

    def test_unreferenced_categories_are_removed_with_last_operation(self):
        self.registry.add(compiled(), expected_revision=0)
        self.registry.add(compiled("question", "questions"), expected_revision=1)
        state = self.registry.remove("brief", expected_revision=2)
        self.assertEqual([item.category_id for item in state.categories], ["questions"])

    def test_late_compilation_and_wrong_session_cannot_change_active_set(self):
        delayed = compiled("delayed")
        self.registry.add(compiled(), expected_revision=0)
        before = self.registry.snapshot()
        with self.assertRaisesRegex(ContractError, "stale"):
            self.registry.add(delayed, expected_revision=0)
        with self.assertRaisesRegex(ContractError, "scope"):
            self.registry.apply_control(
                control("add", 1, session_id="another-session", compiled_goal=goal("other"))
            )
        self.assertEqual(self.registry.snapshot(), before)

    def test_ambiguous_compiler_response_preserves_state_and_content_free_feedback(self):
        self.registry.add(compiled(), expected_revision=0)
        before = self.registry.snapshot()
        ambiguous = {"schema_version": 1, "uncertain": "authored private goal"}
        with self.assertRaises(ContractError) as error:
            self.registry.apply_control(control("add", 1, compiled_goal=ambiguous))
        self.assertEqual(str(error.exception), "invalid coaching operation fields")
        self.assertEqual(self.registry.apply_control(control("list", 1)), before)

    def test_failed_changes_are_atomic_and_do_not_cancel_admitted_work(self):
        self.registry.add(compiled(), expected_revision=0)
        before = self.registry.snapshot()
        cancelled = threading.Event()
        admission = self.registry.admit_dispatch(1, cancelled)
        conflict = goal("another")
        conflict["categories"][0]["description"] = "Different same-id category"
        edits = [
            lambda: self.registry.add(compiled(), expected_revision=1),
            lambda: self.registry.add(
                CompiledCoachingGoal.from_dict(conflict), expected_revision=1
            ),
            lambda: self.registry.remove("absent", expected_revision=1),
            lambda: self.registry.remove("brief", expected_revision=True),
            lambda: self.registry.apply_control(control("add", 1, compiled_goal={})),
            lambda: self.registry.apply_control(control("list", 1, hidden_authority=True)),
        ]
        for edit in edits:
            with self.assertRaises(ContractError):
                edit()
            self.assertEqual(self.registry.snapshot(), before)
            self.assertFalse(cancelled.is_set())
        admission.close()

    def test_active_count_and_wire_limits_fail_without_partial_add(self):
        for index in range(16):
            self.registry.add(compiled(f"op{index}"), expected_revision=index)
        before = self.registry.snapshot()
        with self.assertRaises(ContractError):
            self.registry.add(compiled("overflow"), expected_revision=16)
        self.assertEqual(self.registry.snapshot(), before)
        other = ActiveCoachingOperations("authored-session")
        valid = compiled()
        with patch("rightyo.coaching_operations.MAX_OPERATION_BYTES", 1):
            with self.assertRaisesRegex(ContractError, "byte budget"):
                other.add(valid, expected_revision=0)
        self.assertEqual(other.snapshot().revision, 0)

    def test_revocation_remove_readd_rejects_old_work_and_results(self):
        self.registry.add(compiled(), expected_revision=0)
        cancelled = threading.Event()
        admission = self.registry.admit_dispatch(1, cancelled)
        self.registry.remove("brief", expected_revision=1)
        self.assertTrue(cancelled.is_set())
        self.registry.add(compiled(), expected_revision=2)
        self.assertIsNone(self.registry.admit_dispatch(1, threading.Event()))
        with self.registry.delivery_guard(1) as allowed:
            self.assertFalse(allowed)
        admission.close()
        fresh = self.registry.admit_dispatch(3, threading.Event())
        self.assertIsNotNone(fresh)
        fresh.close()

    def test_admission_cap_retains_revoked_workers_until_they_retire(self):
        self.registry.add(compiled(), expected_revision=0)
        tokens = [threading.Event() for _ in range(MAX_OPERATION_DISPATCHES)]
        admissions = [self.registry.admit_dispatch(1, token) for token in tokens]
        self.registry.add(compiled("another"), expected_revision=1)
        self.assertTrue(all(token.is_set() for token in tokens))
        with self.assertRaisesRegex(ContractError, "dispatch budget"):
            self.registry.admit_dispatch(2, threading.Event())
        admissions[0].close()
        admissions[0].close()
        fresh = self.registry.admit_dispatch(2, threading.Event())
        self.assertIsNotNone(fresh)
        for admission in admissions[1:] + [fresh]:
            admission.close()

    def test_empty_cancelled_and_invalid_admission_are_rejected(self):
        self.assertIsNone(self.registry.admit_dispatch(0, threading.Event()))
        with self.registry.delivery_guard(0) as allowed:
            self.assertFalse(allowed)
        self.registry.add(compiled(), expected_revision=0)
        cancelled = threading.Event()
        cancelled.set()
        self.assertIsNone(self.registry.admit_dispatch(1, cancelled))
        with self.registry.delivery_guard(1, cancelled) as allowed:
            self.assertFalse(allowed)
        for revision, token in ((True, threading.Event()), (1, None), (-1, threading.Event())):
            with self.assertRaises(ContractError):
                self.registry.admit_dispatch(revision, token)

    def test_revision_overflow_preserves_existing_state(self):
        self.registry.add(compiled(), expected_revision=0)
        self.registry._revision = 2**53 - 1
        before = self.registry.snapshot()
        with self.assertRaisesRegex(ContractError, "revision budget"):
            self.registry.remove("brief", expected_revision=2**53 - 1)
        self.assertEqual(self.registry.snapshot(), before)

    def test_control_schema_and_list_revision_are_strict(self):
        for raw in (
            [],
            control("list", schema_version=True),
            control("list", schema_version=2),
            control("list", 1),
            control("list", True),
            control("replace"),
            {**control("list"), "action": []},
        ):
            with self.assertRaises(ContractError):
                self.registry.apply_control(raw)
        self.assertEqual(self.registry.snapshot().revision, 0)

    def test_model_output_and_ordinary_transcript_cannot_act_as_host_control(self):
        self.registry.add(compiled(), expected_revision=0)
        before = self.registry.snapshot()
        for raw in (
            goal(),
            {"type": "transcript", "text": json.dumps(control("remove", 1, operation_id="brief"))},
            {**control("remove", 1, operation_id="brief"), "type": "coaching_advice"},
        ):
            with self.assertRaises(ContractError):
                self.registry.apply_control(raw)
        self.assertEqual(self.registry.snapshot(), before)

    def test_result_handoff_serializes_with_mutation_without_running_provider(self):
        self.registry.add(compiled(), expected_revision=0)
        entered = threading.Event()
        finished = threading.Event()
        handoffs = []

        def revoke():
            entered.set()
            self.registry.remove("brief", expected_revision=1)
            finished.set()

        with self.registry.delivery_guard(1) as allowed:
            self.assertTrue(allowed)
            worker = threading.Thread(target=revoke)
            worker.start()
            self.assertTrue(entered.wait(1))
            self.assertFalse(finished.wait(0.05))
            handoffs.append("authored already handed-off result")
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())
        with self.registry.delivery_guard(1) as allowed:
            self.assertFalse(allowed)
        self.assertEqual(len(handoffs), 1)

    def test_concurrent_expected_revision_edits_have_one_atomic_winner(self):
        barrier = threading.Barrier(3)
        accepted = []
        rejected = []

        def add(operation_id):
            barrier.wait()
            try:
                accepted.append(self.registry.add(compiled(operation_id), expected_revision=0))
            except ContractError:
                rejected.append(operation_id)

        workers = [threading.Thread(target=add, args=(name,)) for name in ("first", "second")]
        for worker in workers:
            worker.start()
        barrier.wait()
        for worker in workers:
            worker.join(1)
            self.assertFalse(worker.is_alive())
        self.assertEqual((len(accepted), len(rejected)), (1, 1))
        self.assertEqual(self.registry.snapshot().revision, 1)
        self.assertEqual(len(self.registry.snapshot().operations), 1)

    def test_category_conflicts_cannot_be_hidden_by_direct_dataclass_update(self):
        self.registry.add(compiled(), expected_revision=0)
        changed = replace(compiled("other"), categories=(CoachingCategory("clarity", "Changed"),))
        with self.assertRaisesRegex(ContractError, "conflicting"):
            self.registry.add(changed, expected_revision=1)
        self.assertEqual(len(self.registry.snapshot().operations), 1)
