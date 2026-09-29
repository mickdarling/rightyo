"""Validated turn and decision boundaries, separate from downstream actions."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

LABELS = frozenset({"attend", "ignore", "uncertain"})
PROVENANCE = frozenset({"synthetic", "recorded-file", "causal-replay", "live-microphone"})
SPEAKER_PROVENANCE = frozenset({"authored-fixture", "diarization-timeline", "unknown"})
MAX_TEXT_CHARS = 4000


class ContractError(ValueError):
    """A public-safe boundary error: never includes the input value."""


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_. -]{1,96}", value):
        raise ContractError(f"invalid {name}")
    return value


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ContractError(f"invalid {name}")
    return value


def probability(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ContractError("invalid provider probability")
    return float(value)


@dataclass(frozen=True)
class Turn:
    session_id: str
    utterance_id: str
    revision: int
    start_ms: int
    end_ms: int
    text: str
    speaker_id: str | None
    finalized: bool
    overlap: bool
    recognizer_id: str
    provenance: str
    speaker_provenance: str = "unknown"

    def __post_init__(self) -> None:
        for name in ("session_id", "utterance_id", "recognizer_id", "speaker_provenance"):
            identifier(getattr(self, name), name)
        if self.speaker_id is not None:
            identifier(self.speaker_id, "speaker_id")
        integer(self.revision, "revision", 1)
        integer(self.start_ms, "start_ms")
        integer(self.end_ms, "end_ms")
        if self.end_ms < self.start_ms:
            raise ContractError("turn timestamps are reversed")
        if not isinstance(self.text, str) or len(self.text) > MAX_TEXT_CHARS:
            raise ContractError("invalid turn text length")
        valid_encoding = False
        try:
            self.text.encode("utf-8")
            valid_encoding = True
        except UnicodeError:
            pass
        if not valid_encoding:
            raise ContractError("turn text must be valid UTF-8")
        if type(self.finalized) is not bool or type(self.overlap) is not bool:
            raise ContractError("invalid turn flags")
        if self.provenance not in PROVENANCE:
            raise ContractError("invalid turn provenance")
        if self.speaker_provenance not in SPEAKER_PROVENANCE:
            raise ContractError("invalid speaker provenance")

    @classmethod
    def from_dict(cls, raw: Any) -> Turn:
        if not isinstance(raw, dict):
            raise ContractError("turn must be an object")
        try:
            return cls(**raw)
        except TypeError:
            raise ContractError("missing or unrecognized turn fields") from None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderDecision:
    label: str
    recipient: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    provider_id: str
    recipient_confidence: float
    recipient_speaker_id: str | None = None
    attention_choice: str | None = None

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ContractError("invalid attention label")
        if self.attention_choice is not None and self.attention_choice not in LABELS:
            raise ContractError("invalid raw attention choice")
        identifier(self.recipient, "recipient")
        identifier(self.model, "model")
        identifier(self.provider_id, "provider_id")
        if self.recipient_speaker_id is not None:
            identifier(self.recipient_speaker_id, "recipient_speaker_id")
        probability(self.confidence)
        probability(self.recipient_confidence)
        if set(self.probabilities) != LABELS:
            raise ContractError("invalid attention probability options")
        if abs(sum(probability(p) for p in self.probabilities.values()) - 1) > 1e-5:
            raise ContractError("attention probabilities do not sum to one")


@dataclass(frozen=True)
class DecisionEvent:
    turn: Turn
    decision: ProviderDecision
    decision_revision: int
    provider_ms: float
    total_ms: float

    def public_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        # IDs and speaker labels can encode private metadata. Output them only with text consent.
        result: dict[str, Any] = {
            "label": self.decision.label,
            "attention_choice": self.decision.attention_choice or self.decision.label,
            "policy_abstained": self.decision.label
            != (self.decision.attention_choice or self.decision.label),
            "recipient_kind": (
                "known_speaker"
                if self.decision.recipient.startswith("speaker_")
                else self.decision.recipient
            ),
            "confidence": self.decision.confidence,
            "recipient_confidence": self.decision.recipient_confidence,
            "probabilities": self.decision.probabilities,
            "model": self.decision.model,
            "provider": self.decision.provider_id,
            "revision": self.decision_revision,
            "start_ms": self.turn.start_ms,
            "end_ms": self.turn.end_ms,
            "speaker_known": self.turn.speaker_id is not None,
            "overlap": self.turn.overlap,
            "provenance": self.turn.provenance,
            "provider_ms": round(self.provider_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }
        if include_text:
            result["turn"] = self.turn.to_dict()
            result["recipient"] = self.decision.recipient
            result["recipient_speaker_id"] = self.decision.recipient_speaker_id
        return result
