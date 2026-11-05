"""Attach existing final transcript events to one explicitly qualified coaching capture.

The embedding host constructs a new adapter per generation. Old callbacks keep their
original generation, rather than being relabelled when private output reconnects.
"""

from __future__ import annotations

from typing import Any

from rightyo.coaching import CoachingSession, canonical_uuid
from rightyo.contracts import ContractError, Turn, integer, speaker_role


def validate_coaching_transcript(event: Any) -> Turn:
    """Validate the ordinary transcript envelope, without granting speaker authority."""
    if not isinstance(event, dict) or event.get("type") != "transcript":
        raise ContractError("coaching input requires a transcript event")
    required = {"schema_version", "type", "session_id", "sequence", "emitted_at_ms", "turn"}
    if not required <= set(event) or set(event) - required - {"speaker_name"}:
        raise ContractError("invalid coaching input fields")
    if type(event["schema_version"]) is not int or event["schema_version"] != 1:
        raise ContractError("unsupported coaching input version")
    integer(event["sequence"], "coaching input sequence", 1)
    integer(event["emitted_at_ms"], "coaching input time")
    canonical_uuid(event["session_id"])
    raw = event["turn"]
    if not isinstance(raw, dict):
        raise ContractError("invalid coaching input turn")
    if "role" in raw:
        speaker_role(raw["role"])
    turn = Turn.from_dict({key: value for key, value in raw.items() if key != "role"})
    if turn.session_id != event["session_id"] or turn.end_ms > event["emitted_at_ms"]:
        raise ContractError("invalid coaching input causal identity")
    return turn


class CoachingInputAdapter:
    def __init__(self, session: CoachingSession, *, generation_id: str) -> None:
        if not isinstance(session, CoachingSession):
            raise ContractError("invalid coaching input session")
        self._session = session
        self._generation = canonical_uuid(generation_id)

    def accept(
        self,
        event: Any,
        *,
        pause_ms: int,
        owner_mid_answer: bool,
        playback: bool = False,
    ) -> None:
        """Transcript channel only; no text, session event or model result is a control."""
        # Ordinary tool events can carry a descriptive role and display name. They
        # are never owner binding, playback permission or command authorization.
        turn = validate_coaching_transcript(event)
        self._session.observe(
            turn,
            generation_id=self._generation,
            pause_ms=pause_ms,
            owner_mid_answer=owner_mid_answer,
            playback=playback,
        )
