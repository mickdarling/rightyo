"""Bounded listener registry updated only through explicit host API/control calls."""

from __future__ import annotations

import json
import threading
from typing import Any

from rightyo.contracts import ContractError, ListenerProfile, identifier

MAX_LISTENERS = 8
MAX_LISTENER_PROFILES_BYTES = 16384
MAX_LISTENER_DISPATCHES = 8


class ListenerDispatch:
    """One committed provider handoff, released when its worker finishes.

    Admission, not the later Python provider invocation, is the linearization point.
    Revocation cannot retract this already admitted payload; it immediately signals
    cancellation. A provider that ignores cancellation can retain its bounded work
    until it returns, so its admission continues occupying the registry's small cap.
    """

    def __init__(self, registry: ListenerProfiles, cancelled: threading.Event) -> None:
        self._registry = registry
        self.cancelled = cancelled

    def close(self) -> None:
        """Idempotently release the admission; no provider code runs under this lock."""
        with self._registry._lock:
            self._registry._admissions.discard(self)


def _validated_profile(value: Any) -> ListenerProfile:
    if type(value) is ListenerProfile:
        return value
    return ListenerProfile.from_dict(value)


def _checked_profiles(profiles: tuple[ListenerProfile, ...]) -> tuple[ListenerProfile, ...]:
    if len(profiles) > MAX_LISTENERS:
        raise ContractError("listener count budget exceeded")
    if len({profile.listener_id for profile in profiles}) != len(profiles):
        raise ContractError("duplicate listener id")
    encoded = json.dumps(
        [profile.to_dict() for profile in profiles], ensure_ascii=True, separators=(",", ":")
    )
    if len(encoded) > MAX_LISTENER_PROFILES_BYTES:
        raise ContractError("listener profile set byte budget exceeded")
    return profiles


class ListenerProfiles:
    """Atomic host-owned configuration, distinct from any transcript input boundary.

    Snapshots are immutable and detached from subsequent changes. A removal is visible
    to the next snapshot; an in-flight consumer retains its already captured snapshot.
    Failed updates leave the previous set intact. No truncation or implicit eviction.
    """

    def __init__(self, initial: Any = ()) -> None:
        if type(initial) not in (list, tuple) or len(initial) > MAX_LISTENERS:
            raise ContractError("listener profiles must be a bounded list")
        self._lock = threading.Lock()
        self._profiles = _checked_profiles(tuple(_validated_profile(value) for value in initial))
        self._revision = 0
        self._admissions: set[ListenerDispatch] = set()

    def snapshot(self) -> tuple[ListenerProfile, ...]:
        with self._lock:
            return self._profiles

    def revisioned_snapshot(self) -> tuple[int, tuple[ListenerProfile, ...]]:
        """Detect even remove/re-add or update/revert between consumer observations."""
        with self._lock:
            return self._revision, self._profiles

    def admit_dispatch(self, revision: int, cancelled: threading.Event) -> ListenerDispatch | None:
        """Commit one bounded handoff atomically with profile mutation/revocation.

        A mutation that wins this lock prevents admission of its previous revision.
        A handoff that wins first is already in flight, even if its provider function
        is scheduled later. Mutations immediately cancel it without waiting for that
        function. Only fixed-size bookkeeping and Event.set run under the lock;
        callers may hold their session lock, but registry operations never acquire it.
        """
        if type(revision) is not int or not 0 <= revision <= 2**53 - 1:
            raise ContractError("invalid listener dispatch revision")
        if type(cancelled) is not threading.Event:
            raise ContractError("invalid listener dispatch cancellation")
        with self._lock:
            if revision != self._revision or cancelled.is_set():
                return None
            if len(self._admissions) >= MAX_LISTENER_DISPATCHES:
                raise ContractError("listener dispatch budget exceeded")
            admission = ListenerDispatch(self, cancelled)
            self._admissions.add(admission)
            return admission

    def _replace(self, profiles: tuple[ListenerProfile, ...]) -> None:
        checked = _checked_profiles(profiles)
        if self._revision >= 2**53 - 1:
            raise ContractError("listener profile revision budget exceeded")
        self._profiles = checked
        self._revision += 1
        # Retain the records until their workers finish, including uncooperative
        # workers. Clearing them here would allow repeated edits to evade the cap.
        for admission in self._admissions:
            admission.cancelled.set()

    def add(self, profile: ListenerProfile | dict[str, Any]) -> None:
        validated = _validated_profile(profile)
        with self._lock:
            self._replace((*self._profiles, validated))

    def update(self, profile: ListenerProfile | dict[str, Any]) -> None:
        validated = _validated_profile(profile)
        with self._lock:
            if not any(p.listener_id == validated.listener_id for p in self._profiles):
                raise ContractError("listener update requires an existing id")
            self._replace(
                tuple(
                    validated if p.listener_id == validated.listener_id else p
                    for p in self._profiles
                )
            )

    def remove(self, listener_id: str) -> None:
        identifier(listener_id, "listener id")
        with self._lock:
            remaining = tuple(p for p in self._profiles if p.listener_id != listener_id)
            if len(remaining) == len(self._profiles):
                raise ContractError("listener removal requires an existing id")
            self._replace(remaining)

    def apply_control(self, raw: Any) -> None:
        """Apply one explicit host event. Callers must not derive these events from speech."""
        if not isinstance(raw, dict):
            raise ContractError("invalid listener profile control")
        if type(raw.get("version")) is not int or raw["version"] != 1:
            raise ContractError("unsupported listener profile control version")
        if raw.get("type") != "listener_profile":
            raise ContractError("invalid listener profile control type")
        action = raw.get("action")
        if action == "add" or action == "update":
            if set(raw) != {"version", "type", "action", "profile"}:
                raise ContractError("invalid listener profile control fields")
            if action == "add":
                self.add(raw["profile"])
            else:
                self.update(raw["profile"])
        elif action == "remove":
            if set(raw) != {"version", "type", "action", "listener_id"}:
                raise ContractError("invalid listener profile control fields")
            self.remove(raw["listener_id"])
        else:
            raise ContractError("invalid listener profile control action")
