"""Validated turn and decision boundaries, separate from downstream actions."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any

LABELS = frozenset({"attend", "ignore", "uncertain"})
# The dismissal judgement (#98): `stop` and `disengage` dismiss the assistant; `none` is
# not a dismissal of the assistant; `uncertain` abstains.
DISMISSAL_LABELS = frozenset({"stop", "disengage", "none", "uncertain"})
DISMISSING = frozenset({"stop", "disengage"})
DEFAULT_DISMISSAL_WINDOW_MS = 10000
MAX_DISMISSAL_WINDOW_MS = 60000
DEFAULT_DISMISSAL_COOLDOWN_MS = 30000
MAX_DISMISSAL_COOLDOWN_MS = 600000
DEFAULT_COOLDOWN_MIN_CONFIDENCE = 0.9
PROVENANCE = frozenset({"synthetic", "recorded-file", "causal-replay", "live-microphone"})
SPEAKER_PROVENANCE = frozenset(
    {"authored-fixture", "diarization-timeline", "diarization-utterance", "unknown"}
)
# Labels from a per-utterance diarizer are scoped by utterance number so that equal
# labels from independent requests never merge into one participant.
UTTERANCE_SCOPE = re.compile(r"u[0-9]+ ")


def utterance_scoped_speaker(utterance: int, label: str) -> str:
    """The speaker id a `diarization-utterance` turn must carry: `u<n> <label>`."""
    return f"u{utterance} {label}"


MAX_TEXT_CHARS = 4000
MAX_ADDRESS_NAMES = 8
MAX_ADDRESS_NAME_CHARS = 48
MAX_NAME_VARIANTS = 8
MAX_TOTAL_NAME_VARIANTS = 32
_NAME_KEY_SEPARATORS = re.compile(r"[^0-9a-z]+")
_ADDRESS_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_. -]{0,%d}" % (MAX_ADDRESS_NAME_CHARS - 1))
# Speaker roles are allowlisted literals that describe precedence inside RightyO. They
# are configuration or model output, never authentication, and unlock nothing downstream.
SPEAKER_ROLES = frozenset({"owner", "trusted", "participant", "unknown"})
MODEL_SPEAKER_ROLES = frozenset({"trusted", "participant", "unknown"})
ROLE_SOURCES = frozenset({"configured", "model"})
DEFAULT_STOP_PHRASES = ("stop", "cancel", "ignore that", "never mind")
MAX_OWNER_SPEAKERS = 8
MAX_TRUSTED_SPEAKERS = 32
MAX_STOP_PHRASES = 16
_STOP_PHRASE = re.compile(r"[A-Za-z0-9' ]{1,48}")
_PHRASE_SEPARATORS = re.compile(r"[^\w]+")


class ContractError(ValueError):
    """A public-safe boundary error: never includes the input value."""


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_. -]{1,96}", value):
        raise ContractError(f"invalid {name}")
    return value


def address_name(value: Any) -> str:
    """One runtime form of address: a sanitized display name, not a wake-word grammar."""
    if (
        not isinstance(value, str)
        or not _ADDRESS_NAME.fullmatch(value)
        or value != value.strip()
        or "  " in value
        or value.endswith((".", "-"))
    ):
        # Single spaces and no trailing punctuation: a name never reads as prompt prose
        # and the mock prefix never becomes "name.," or "name-:".
        raise ContractError("invalid address name")
    return value


def speaker_role(value: Any) -> str:
    # Type first: an unhashable value must raise the sanitized error, not a TypeError.
    if not isinstance(value, str) or value not in SPEAKER_ROLES:
        raise ContractError("invalid speaker role")
    return value


def normalize_phrase(text: str) -> str:
    """Casefolded words only: apostrophes removed, punctuation and whitespace collapsed."""
    stripped = text.replace("'", "").replace("’", "")
    return " ".join(_PHRASE_SEPARATORS.sub(" ", stripped).casefold().split())


def _speaker_list(values: Any, name: str, limit: int) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ContractError(f"{name} speakers must be a list")
    if len(values) > limit:
        raise ContractError(f"too many {name} speakers")
    return tuple(identifier(value, f"{name} speaker") for value in values)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ContractError(f"invalid {name}")
    return value


def probability(value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ContractError("invalid provider probability")
    return float(value)


def name_key(text: str) -> str:
    """Casefolded ASCII letters and digits only, so "Righty-O" and "righty o" compare equal."""
    return _NAME_KEY_SEPARATORS.sub("", text.casefold())


@dataclass(frozen=True)
class Addressing:
    """Names the system answers to, supplied at runtime and never hard-coded.

    A name is evidence of addressing, not a requirement or a transcript filter; the
    decision provider still judges the addressee from the complete turn and context.

    `variants` optionally lists, per name, other spellings a speech recognizer is known
    to produce for it (#72), as `((name, (variant, ...)), ...)`. They are evidence of the
    same name, never a separate name, and are configuration, not a built-in list.
    """

    names: tuple[str, ...]
    variants: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if type(self.names) is not tuple or not 1 <= len(self.names) <= MAX_ADDRESS_NAMES:
            raise ContractError(f"addressing requires between 1 and {MAX_ADDRESS_NAMES} names")
        # Matching ignores case, spaces and punctuation (`name_key`), so uniqueness does
        # too: "Righty O" and "RightyO" would be the same name to every matcher.
        seen = set()
        for name in self.names:
            key = name_key(address_name(name))
            if key in seen:
                raise ContractError("duplicate address name")
            seen.add(key)
        if type(self.variants) is not tuple:
            raise ContractError("invalid address name variants")
        owners = set()
        total = 0
        for entry in self.variants:
            if type(entry) is not tuple or len(entry) != 2 or type(entry[1]) is not tuple:
                raise ContractError("invalid address name variants")
            name, spellings = entry
            if not isinstance(name, str) or name not in self.names or name in owners:
                raise ContractError("address name variants must name a configured name once")
            owners.add(name)
            if not 1 <= len(spellings) <= MAX_NAME_VARIANTS:
                raise ContractError(f"each name takes between 1 and {MAX_NAME_VARIANTS} variants")
            total += len(spellings)
            for spelling in spellings:
                key = name_key(address_name(spelling))
                if key in seen:
                    raise ContractError("duplicate address name")
                seen.add(key)
        if total > MAX_TOTAL_NAME_VARIANTS:
            raise ContractError(f"at most {MAX_TOTAL_NAME_VARIANTS} address name variants")

    @classmethod
    def from_names(cls, names: Any, variants: Any = None) -> Addressing:
        if isinstance(names, (str, bytes)) or not isinstance(names, (list, tuple)):
            raise ContractError("addressing names must be a list")
        if variants is None:
            return cls(tuple(names))
        if not isinstance(variants, dict) or not all(
            isinstance(spellings, list) for spellings in variants.values()
        ):
            raise ContractError("address name variants must map names to lists")
        return cls(
            tuple(names),
            tuple((name, tuple(spellings)) for name, spellings in variants.items()),
        )

    @classmethod
    def from_dict(cls, raw: Any) -> Addressing:
        if not isinstance(raw, dict) or "names" not in raw or set(raw) - {"names", "variants"}:
            raise ContractError("addressing must be an object with names and optional variants")
        if "variants" in raw and raw["variants"] is None:
            raise ContractError("address name variants must map names to lists")
        return cls.from_names(raw["names"], raw.get("variants"))

    def spellings(self, name: str) -> tuple[str, ...]:
        """The configured variants of one name, or none."""
        return next((spellings for owner, spellings in self.variants if owner == name), ())

    def name_for(self, phrase: str) -> str | None:
        """The configured name a whole phrase spells, by name or variant, ignoring case,
        spaces and punctuation; None when it spells none of them."""
        key = name_key(phrase)
        if not key:
            return None
        for name in self.names:
            if key in {name_key(spelling) for spelling in (name, *self.spellings(name))}:
                return name
        return None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"names": list(self.names)}
        if self.variants:
            result["variants"] = {name: list(spellings) for name, spellings in self.variants}
        return result


# A formed request is a convenience rendering of the raw turns for hosts that cannot
# reason over them; it is bounded like turn text and carries no authority of its own.
REQUEST_FORMER_KINDS = frozenset({"template"})
MAX_FORMED_REQUEST_CHARS = 4 * MAX_TEXT_CHARS


def formed_request_text(value: Any) -> str:
    """The same text validation as turn text, plus a non-empty, bounded length."""
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_FORMED_REQUEST_CHARS:
        raise ContractError("invalid formed request length")
    try:
        value.encode("utf-8")
    except UnicodeError:
        raise ContractError("formed request must be valid UTF-8") from None
    return value


@dataclass(frozen=True)
class RequestForming:
    """Which optional request former renders ``formed_request``; absent means off."""

    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in REQUEST_FORMER_KINDS:
            raise ContractError("invalid request former kind")

    @classmethod
    def from_dict(cls, raw: Any) -> RequestForming:
        if not isinstance(raw, dict) or set(raw) != {"kind"}:
            raise ContractError("request_former must be an object with only kind")
        return cls(raw["kind"])

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class SpeakerPriority:
    """Hard-coded speaker roles and the owner's precedence rules.

    Owners and trusted speakers are anonymous session labels or enrolled identifiers
    supplied by configuration. A role is descriptive data for precedence and routing
    inside RightyO; it is not an authenticated identity and never unlocks a host gate.
    """

    owners: tuple[str, ...] = ()
    trusted: tuple[str, ...] = ()
    owner_only: bool = False
    stop_phrases: tuple[str, ...] = DEFAULT_STOP_PHRASES
    source: str = "configured"

    def __post_init__(self) -> None:
        if type(self.owners) is not tuple or type(self.trusted) is not tuple:
            raise ContractError("speaker lists must be tuples")
        _speaker_list(self.owners, "owner", MAX_OWNER_SPEAKERS)
        _speaker_list(self.trusted, "trusted", MAX_TRUSTED_SPEAKERS)
        listed = [*self.owners, *self.trusted]
        if len(set(listed)) != len(listed):
            raise ContractError("a speaker may hold only one configured role")
        if type(self.owner_only) is not bool:
            raise ContractError("invalid owner_only flag")
        if type(self.stop_phrases) is not tuple or not 1 <= len(self.stop_phrases) <= (
            MAX_STOP_PHRASES
        ):
            raise ContractError(f"stop phrases require between 1 and {MAX_STOP_PHRASES} entries")
        normalized = set()
        for phrase in self.stop_phrases:
            if not isinstance(phrase, str) or not _STOP_PHRASE.fullmatch(phrase):
                raise ContractError("invalid stop phrase")
            words = normalize_phrase(phrase)
            if not words or words in normalized:
                raise ContractError("empty or duplicate stop phrase")
            normalized.add(words)
        if self.source not in ROLE_SOURCES:
            raise ContractError("invalid speaker role source")

    def configured_role(self, speaker_id: str | None) -> str | None:
        if speaker_id in self.owners:
            return "owner"
        if speaker_id in self.trusted:
            return "trusted"
        return None

    def is_stop_phrase(self, text: str) -> bool:
        """Whole-utterance match after casefolding and punctuation removal."""
        words = normalize_phrase(text)
        return any(words == normalize_phrase(phrase) for phrase in self.stop_phrases)

    @classmethod
    def from_dict(cls, raw: Any) -> SpeakerPriority:
        allowed = {"owner", "trusted", "owner_only", "stop_phrases", "source"}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ContractError("speakers must be an object with known keys only")
        values: dict[str, Any] = {}
        if "owner" in raw:
            values["owners"] = _speaker_list(raw["owner"], "owner", MAX_OWNER_SPEAKERS)
        if "trusted" in raw:
            values["trusted"] = _speaker_list(raw["trusted"], "trusted", MAX_TRUSTED_SPEAKERS)
        if "owner_only" in raw:
            values["owner_only"] = raw["owner_only"]
        if "stop_phrases" in raw:
            phrases = raw["stop_phrases"]
            if isinstance(phrases, (str, bytes)) or not isinstance(phrases, (list, tuple)):
                raise ContractError("stop phrases must be a list")
            values["stop_phrases"] = tuple(phrases)
        if "source" in raw:
            values["source"] = raw["source"]
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner": list(self.owners),
            "trusted": list(self.trusted),
            "owner_only": self.owner_only,
            "stop_phrases": list(self.stop_phrases),
            "source": self.source,
        }


@dataclass(frozen=True)
class Dismissal:
    """Natural dismissal and barge-in (#98); absent means off and nothing changes.

    `window_ms` bounds self-withdrawal: a speaker's dismissal withdraws that speaker's
    own request whose turn ended at most this long before the dismissal started.
    `cooldown_ms` (0 turns it off) is how long after a `disengage` dismissal an attended
    turn that uses no configured name needs at least `cooldown_min_confidence` to form a
    request.
    """

    window_ms: int = DEFAULT_DISMISSAL_WINDOW_MS
    cooldown_ms: int = DEFAULT_DISMISSAL_COOLDOWN_MS
    cooldown_min_confidence: float = DEFAULT_COOLDOWN_MIN_CONFIDENCE

    def __post_init__(self) -> None:
        if type(self.window_ms) is not int or not 1 <= self.window_ms <= MAX_DISMISSAL_WINDOW_MS:
            raise ContractError("invalid dismissal window")
        if (
            type(self.cooldown_ms) is not int
            or not 0 <= self.cooldown_ms <= MAX_DISMISSAL_COOLDOWN_MS
        ):
            raise ContractError("invalid dismissal cooldown")
        probability(self.cooldown_min_confidence)

    @classmethod
    def from_dict(cls, raw: Any) -> Dismissal:
        allowed = {"window_ms", "cooldown_ms", "cooldown_min_confidence"}
        if not isinstance(raw, dict) or set(raw) - allowed:
            raise ContractError("dismissal must be an object with known keys only")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "window_ms": self.window_ms,
            "cooldown_ms": self.cooldown_ms,
            "cooldown_min_confidence": self.cooldown_min_confidence,
        }


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
        if (
            self.speaker_provenance == "diarization-utterance"
            and self.speaker_id is not None
            and not UTTERANCE_SCOPE.match(self.speaker_id)
        ):
            # Enforced at the contract so replayed or imported turns cannot merge
            # unrelated voices, or match a configured role, through a bare label.
            raise ContractError("utterance-local speaker ids must be utterance-scoped")

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
    # The dismissal judgement (#98), present only when the question was asked: the
    # policy-applied label, the raw choice and the probability mass of the dismissing
    # options (`stop` plus `disengage`).
    dismissal: str | None = None
    dismissal_choice: str | None = None
    dismissal_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ContractError("invalid attention label")
        dismissal = (self.dismissal, self.dismissal_choice, self.dismissal_confidence)
        if dismissal != (None, None, None):
            if (
                self.dismissal not in DISMISSAL_LABELS
                or self.dismissal_choice not in DISMISSAL_LABELS
            ):
                raise ContractError("invalid dismissal label")
            probability(self.dismissal_confidence)
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
        if self.decision.dismissal is not None:
            # Present only when the dismissal question was asked (#98).
            result["dismissal"] = self.decision.dismissal
            result["dismissal_confidence"] = self.decision.dismissal_confidence
        if include_text:
            result["turn"] = self.turn.to_dict()
            result["recipient"] = self.decision.recipient
            result["recipient_speaker_id"] = self.decision.recipient_speaker_id
        return result
