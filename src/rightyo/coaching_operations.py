"""Pure compiled-goal schema and explicit host-owned operation lifecycle (#178).

No command recognition, compiler/provider call, transcript export or execution lives
here. Compiler output is descriptive untrusted data, never control authority.
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from rightyo.contracts import ContractError, integer

MAX_CATEGORIES = 8
MAX_OPERATIONS = 16
MAX_TEXT_CHARS = 512
MAX_OPERATION_BYTES = 16384
MAX_OPERATION_DISPATCHES = 8
_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


def _key(value: Any) -> str:
    if type(value) is not str or not _KEY.fullmatch(value):
        raise ContractError("invalid coaching operation identity")
    return value


def _text(value: Any) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or len(value) > MAX_TEXT_CHARS
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ContractError("invalid coaching operation description")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ContractError("invalid coaching operation description") from None
    return value


def _object(raw: Any, fields: set[str]) -> dict[str, Any]:
    if type(raw) is not dict or set(raw) != fields:
        raise ContractError("invalid coaching operation fields")
    return raw


def _version(value: Any) -> None:
    if type(value) is not int or value != 1:
        raise ContractError("unsupported coaching operation schema")


def _budget(raw: Any) -> None:
    if len(json.dumps(raw, ensure_ascii=True, separators=(",", ":"))) > MAX_OPERATION_BYTES:
        raise ContractError("coaching operation byte budget exceeded")


@dataclass(frozen=True)
class CoachingCategory:
    category_id: str
    description: str

    def __post_init__(self) -> None:
        _key(self.category_id)
        _text(self.description)

    @classmethod
    def from_dict(cls, raw: Any) -> CoachingCategory:
        return cls(**_object(raw, {"category_id", "description"}))

    def to_dict(self) -> dict[str, str]:
        return {"category_id": self.category_id, "description": self.description}


@dataclass(frozen=True)
class CoachingOperation:
    operation_id: str
    category_id: str
    goal: str
    speak_when: str
    stay_quiet_when: str
    delivery: str = "owner_private"

    def __post_init__(self) -> None:
        _key(self.operation_id)
        _key(self.category_id)
        for text in (self.goal, self.speak_when, self.stay_quiet_when):
            _text(text)
        if type(self.delivery) is not str or self.delivery != "owner_private":
            raise ContractError("invalid coaching operation delivery")

    @classmethod
    def from_dict(cls, raw: Any) -> CoachingOperation:
        fields = {
            "operation_id",
            "category_id",
            "goal",
            "speak_when",
            "stay_quiet_when",
            "delivery",
        }
        return cls(**_object(raw, fields))

    def to_dict(self) -> dict[str, str]:
        return {
            "operation_id": self.operation_id,
            "category_id": self.category_id,
            "goal": self.goal,
            "speak_when": self.speak_when,
            "stay_quiet_when": self.stay_quiet_when,
            "delivery": self.delivery,
        }


def _checked(categories: Any, operations: Any, *, empty: bool = False) -> None:
    if (
        type(categories) is not tuple
        or type(operations) is not tuple
        or len(categories) > MAX_CATEGORIES
        or len(operations) > MAX_OPERATIONS
        or (not empty and (not categories or not operations))
        or any(type(value) is not CoachingCategory for value in categories)
        or any(type(value) is not CoachingOperation for value in operations)
    ):
        raise ContractError("invalid coaching operation set")
    category_ids = {value.category_id for value in categories}
    if (
        len(category_ids) != len(categories)
        or len({value.operation_id for value in operations}) != len(operations)
        or {value.category_id for value in operations} != category_ids
    ):
        raise ContractError("invalid coaching operation references")


@dataclass(frozen=True)
class CompiledCoachingGoal:
    categories: tuple[CoachingCategory, ...]
    operations: tuple[CoachingOperation, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _checked(self.categories, self.operations)
        _budget(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Any) -> CompiledCoachingGoal:
        value = _object(raw, {"schema_version", "categories", "operations"})
        _version(value["schema_version"])
        if (
            type(value["categories"]) is not list
            or not 1 <= len(value["categories"]) <= MAX_CATEGORIES
            or type(value["operations"]) is not list
            or not 1 <= len(value["operations"]) <= MAX_OPERATIONS
        ):
            raise ContractError("invalid compiled coaching goal")
        return cls(
            tuple(CoachingCategory.from_dict(item) for item in value["categories"]),
            tuple(CoachingOperation.from_dict(item) for item in value["operations"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "categories": [value.to_dict() for value in self.categories],
            "operations": [value.to_dict() for value in self.operations],
        }


def parse_compiled_goal(raw: Any) -> CompiledCoachingGoal:
    """Reject bounded untrusted JSON before the separate host activation boundary."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ContractError("duplicate coaching compiler field")
            result[key] = value
        return result

    def nonfinite(_):
        raise ContractError("invalid coaching compiler JSON")

    if type(raw) not in (str, bytes) or len(raw) > MAX_OPERATION_BYTES:
        raise ContractError("invalid coaching compiler response")
    try:
        if type(raw) is bytes:
            raw = raw.decode("utf-8")
        return CompiledCoachingGoal.from_dict(
            json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
        )
    except ContractError:
        raise
    except (ValueError, RecursionError):
        raise ContractError("invalid coaching compiler JSON") from None


@dataclass(frozen=True)
class OperationSnapshot:
    session_id: str
    revision: int
    categories: tuple[CoachingCategory, ...]
    operations: tuple[CoachingOperation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "type": "coaching_operations_state",
            "session_id": self.session_id,
            "revision": self.revision,
            "categories": [value.to_dict() for value in self.categories],
            "operations": [value.to_dict() for value in self.operations],
        }


class OperationDispatch:
    """Bounded cancellation record; close only when the admitted worker is retired."""

    def __init__(self, registry: ActiveCoachingOperations, cancelled: threading.Event) -> None:
        self._registry = registry
        self.cancelled = cancelled

    def close(self) -> None:
        with self._registry._lock:
            self._registry._admissions.discard(self)


class ActiveCoachingOperations:
    """Explicit host configuration, never ordinary conversation/model control.

    The caller authenticates the operator command separately. Revision comparison
    makes a delayed compiler response stale after any successful edit. No transcript
    parser or wire input is connected to this registry by this groundwork.
    """

    def __init__(self, session_id: str) -> None:
        self._session_id = _key(session_id)
        self._revision = 0
        self._categories: tuple[CoachingCategory, ...] = ()
        self._operations: tuple[CoachingOperation, ...] = ()
        self._lock = threading.Lock()
        self._admissions: set[OperationDispatch] = set()

    def _snapshot(self) -> OperationSnapshot:
        return OperationSnapshot(
            self._session_id, self._revision, self._categories, self._operations
        )

    def snapshot(self) -> OperationSnapshot:
        with self._lock:
            return self._snapshot()

    def _expect(self, revision: Any) -> None:
        integer(revision, "coaching operation revision")
        if revision != self._revision:
            raise ContractError("stale coaching operation revision")

    def _replace(self, categories: tuple, operations: tuple) -> OperationSnapshot:
        _checked(categories, operations, empty=True)
        next_state = OperationSnapshot(self._session_id, self._revision + 1, categories, operations)
        _budget(next_state.to_dict())
        if self._revision >= 2**53 - 1:
            raise ContractError("coaching operation revision budget exceeded")
        self._categories, self._operations = categories, operations
        self._revision += 1
        # Revoked workers retain their bounded admission until actual retirement.
        for admission in self._admissions:
            admission.cancelled.set()
        return next_state

    def add(self, goal: CompiledCoachingGoal, *, expected_revision: int) -> OperationSnapshot:
        if type(goal) is not CompiledCoachingGoal:
            raise ContractError("invalid compiled coaching goal")
        with self._lock:
            self._expect(expected_revision)
            categories = {item.category_id: item for item in self._categories}
            for item in goal.categories:
                if item.category_id in categories and categories[item.category_id] != item:
                    raise ContractError("conflicting coaching category")
                categories[item.category_id] = item
            return self._replace(tuple(categories.values()), (*self._operations, *goal.operations))

    def remove(self, operation_id: str, *, expected_revision: int) -> OperationSnapshot:
        _key(operation_id)
        with self._lock:
            self._expect(expected_revision)
            operations = tuple(
                item for item in self._operations if item.operation_id != operation_id
            )
            if len(operations) == len(self._operations):
                raise ContractError("coaching removal requires an active operation")
            used = {item.category_id for item in operations}
            return self._replace(
                tuple(item for item in self._categories if item.category_id in used), operations
            )

    def apply_control(self, raw: Any) -> OperationSnapshot:
        """Explicit authorized host call; compiler/transcript bytes cannot authorize it."""
        base = {"schema_version", "type", "session_id", "expected_revision", "action"}
        if type(raw) is not dict:
            raise ContractError("invalid coaching operation control")
        action = raw.get("action")
        if type(action) is not str or action not in {"add", "remove", "list"}:
            raise ContractError("invalid coaching operation control action")
        extra = (
            {"compiled_goal"}
            if action == "add"
            else {"operation_id"}
            if action == "remove"
            else set()
        )
        _object(raw, base | extra)
        _version(raw["schema_version"])
        if raw["type"] != "coaching_operations_control" or raw["session_id"] != self._session_id:
            raise ContractError("invalid coaching operation control scope")
        if action == "add":
            return self.add(
                CompiledCoachingGoal.from_dict(raw["compiled_goal"]),
                expected_revision=raw["expected_revision"],
            )
        if action == "remove":
            return self.remove(raw["operation_id"], expected_revision=raw["expected_revision"])
        with self._lock:
            self._expect(raw["expected_revision"])
            return self._snapshot()

    def admit_dispatch(self, revision: int, cancelled: threading.Event) -> OperationDispatch | None:
        integer(revision, "coaching operation revision")
        if type(cancelled) is not threading.Event:
            raise ContractError("invalid coaching operation cancellation")
        with self._lock:
            if revision != self._revision or cancelled.is_set() or not self._operations:
                return None
            if len(self._admissions) >= MAX_OPERATION_DISPATCHES:
                raise ContractError("coaching operation dispatch budget exceeded")
            admission = OperationDispatch(self, cancelled)
            self._admissions.add(admission)
            return admission

    @contextmanager
    def delivery_guard(self, revision: int, cancelled: threading.Event | None = None):
        """Atomic local result handoff; only bounded in-memory work inside this guard.

        Consumers must independently recheck their original deadline, session/route
        authority and cancellation after waiting for this lock. Never run provider,
        playback, I/O or callbacks under it. A later edit cannot retract handed-off data.
        """
        integer(revision, "coaching operation revision")
        if cancelled is not None and type(cancelled) is not threading.Event:
            raise ContractError("invalid coaching operation cancellation")
        with self._lock:
            yield (
                bool(self._operations)
                and revision == self._revision
                and (cancelled is None or not cancelled.is_set())
            )
