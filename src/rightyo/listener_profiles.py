"""Bounded listener registry updated only through explicit host API/control calls."""

from __future__ import annotations

import json
import threading
from typing import Any

from rightyo.contracts import ContractError, ListenerProfile, identifier

MAX_LISTENERS = 8
MAX_LISTENER_PROFILES_BYTES = 16384


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

    def snapshot(self) -> tuple[ListenerProfile, ...]:
        with self._lock:
            return self._profiles

    def revisioned_snapshot(self) -> tuple[int, tuple[ListenerProfile, ...]]:
        """Detect even remove/re-add or update/revert between consumer observations."""
        with self._lock:
            return self._revision, self._profiles

    def _replace(self, profiles: tuple[ListenerProfile, ...]) -> None:
        checked = _checked_profiles(profiles)
        if self._revision >= 2**53 - 1:
            raise ContractError("listener profile revision budget exceeded")
        self._profiles = checked
        self._revision += 1

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
