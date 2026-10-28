"""Strict provider-neutral host request control, version 1 (#262)."""

from rightyo.contracts import ContractError, identifier

STATES = frozenset({"accepted", "failed", "timed_out", "cancelled", "reply_started", "reply_ended"})
TERMINAL = frozenset({"failed", "timed_out", "cancelled", "reply_ended"})


def validate(value):
    """Return a validated envelope; never infer request identity from text or time."""
    if not isinstance(value, dict) or set(value) not in (
        {"version", "session_id", "request_id", "state"},
        {"version", "session_id", "request_id", "state", "asked"},
    ):
        raise ContractError("invalid request outcome")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ContractError("invalid request outcome")
    session = identifier(value["session_id"], "session_id")
    request = value["request_id"]
    if not isinstance(request, str) or not request.startswith(session + ":"):
        raise ContractError("invalid request outcome")
    identifier(request[len(session) + 1 :], "utterance_id")
    state = value["state"]
    if not isinstance(state, str) or state not in STATES:
        raise ContractError("invalid request outcome")
    if "asked" in value and (state != "reply_ended" or type(value["asked"]) is not bool):
        raise ContractError("invalid request outcome")
    return value
