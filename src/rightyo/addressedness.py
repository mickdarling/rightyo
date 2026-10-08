"""Addressedness evidence beyond the words of one turn (#96).

Two operator-side inputs help the decision model judge whether unnamed speech is meant
for the assistant, without making a name a trigger:

- a **scene**: plain-language text, configured by the operator (never derived from a
  transcript), describing the setting and what usually counts as addressing the
  assistant. It is rendered into the decision instructions, after the rule that
  transcripts are untrusted data.
- a **post-turn gap**: what was heard right after the current turn, within a short
  bounded window. A request that nobody else answers in a quiet gap is evidence it was
  meant for the assistant; a different speaker starting to talk is evidence it was meant
  for that person.

`request_shaped` is a small deterministic placeholder for the per-turn intent of #85: it
only decides which turns are worth holding a little longer to observe the gap. It is not
an addressedness judgement and never forms a request on its own.
"""

from __future__ import annotations

import re
from typing import Any

from .contracts import ContractError

# The default setting for the single-user assistant pilot (Hailing Station).
DEFAULT_SCENE = (
    "One primary user is talking to an AI assistant through a phone or tablet. Most of "
    "the user's directed speech that is not clearly aimed at another person present is "
    "meant for the assistant, including questions and requests that do not use its name. "
    "Other voices may be the assistant's own audio playback, other AI agents or media, "
    "rather than people in the room. A question or request that no other person answers "
    "is likely meant for the assistant."
)
MAX_SCENE_CHARS = 1000

# How long a request-shaped turn is held after its last word to observe the gap (#96).
# The default turn merge hold (2,000 ms, #89) already covers it, so by default it adds
# no latency; it only lengthens the hold when merging is off or shorter.
DEFAULT_REPLY_WAIT_MS = 1200
MAX_REPLY_WAIT_MS = 3000
# The longest observation window any hold can report: the merge gap ceiling.
MAX_GAP_WINDOW_MS = 5000

FOLLOWING = ("none", "same_speaker", "different_speaker", "unattributed")
UNOBSERVED = {"observed": False}


def scene_text(value: Any) -> str | None:
    """A validated scene: None (off) or 1..1000 printable characters, stripped."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ContractError("invalid scene")
    text = " ".join(value.split())
    if not text or len(text) > MAX_SCENE_CHARS or not text.isprintable():
        raise ContractError("invalid scene")
    return text


def reply_wait(value: Any) -> int:
    """A validated reply wait in milliseconds: 0 turns the post-turn gap signal off."""
    if type(value) is not int or not 0 <= value <= MAX_REPLY_WAIT_MS:
        raise ValueError("invalid reply wait")
    return value


def post_turn_gap(value: Any) -> dict[str, Any]:
    """A validated, normalized post-turn gap observation for the decision state."""
    if value is None:
        return dict(UNOBSERVED)
    if not isinstance(value, dict):
        raise ContractError("invalid post-turn gap")
    if value.get("observed") is False and set(value) == {"observed"}:
        return dict(UNOBSERVED)
    if set(value) != {"observed", "window_ms", "silence_ms", "following"}:
        raise ContractError("invalid post-turn gap")
    window, silence, following = value["window_ms"], value["silence_ms"], value["following"]
    if (
        value["observed"] is not True
        or type(window) is not int
        or type(silence) is not int
        or not 1 <= window <= MAX_GAP_WINDOW_MS
        or not 0 <= silence <= window
        or following not in FOLLOWING
        or (following == "none" and silence != window)
    ):
        raise ContractError("invalid post-turn gap")
    return {"observed": True, "window_ms": window, "silence_ms": silence, "following": following}


def observe_gap(
    held: dict[str, Any], following: dict[str, Any] | None, window_ms: int
) -> dict[str, Any]:
    """The gap after `held`, given the next fragment heard (None: quiet through the window).

    Fragments are the turn merger's dictionaries (`speaker`, `overlap`,
    `speaker_provenance`, `start_ms`, `end_ms`). Speech starting after the window counts as
    silence for the whole window. Without a speaker label on either side, or with overlap,
    the next speaker is `unattributed`: no different speaker is invented. Utterance-local
    labels (`diarization-utterance`, namespaced `u<n> <label>`) compare only within one
    utterance, so labels from two different utterances are `unattributed` too.
    """
    if following is None or following["start_ms"] - held["end_ms"] > window_ms:
        return post_turn_gap(
            {"observed": True, "window_ms": window_ms, "silence_ms": window_ms, "following": "none"}
        )
    silence = max(0, following["start_ms"] - held["end_ms"])
    if (
        held["speaker"] is None
        or following["speaker"] is None
        or held["overlap"]
        or following["overlap"]
        or held["speaker_provenance"] != following["speaker_provenance"]
        or (
            held["speaker_provenance"] == "diarization-utterance"
            and _utterance_scope(held["speaker"]) != _utterance_scope(following["speaker"])
        )
    ):
        who = "unattributed"
    elif held["speaker"] == following["speaker"]:
        who = "same_speaker"
    else:
        who = "different_speaker"
    return post_turn_gap(
        {"observed": True, "window_ms": window_ms, "silence_ms": silence, "following": who}
    )


def _utterance_scope(speaker: str) -> str:
    """The `u<n>` namespace of an utterance-local label (the whole label if it has none)."""
    return speaker.split(" ", 1)[0]


_WORDS = re.compile(r"[a-z']+")
_FILLERS = frozenset(
    "hey ok okay so and um uh oh well now then also actually alright right yeah yes".split()
)
_OPENERS = frozenset(
    """
    can could would will should shall may might please
    what what's whats when where where's which who who's whom whose why how how's
    is are am was were do does did have has
    tell show give make set turn open close play stop start pause resume skip remind find
    search look check call send read add remove delete create write let put get take bring
    help schedule cancel email text message note book order explain summarize translate
    """.split()
)


def request_shaped(text: str) -> bool:
    """Whether a turn reads as a question or request (English, deterministic, approximate).

    True for a turn ending in a question mark, containing "please", or opening (after
    fillers and a short comma-delimited vocative such as a name) with a question word,
    auxiliary or common imperative verb.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    lowered = stripped.lower()
    words = _WORDS.findall(lowered)
    if "please" in words:
        return True
    head, comma, rest = lowered.partition(",")
    if comma and 1 <= len(_WORDS.findall(head)) <= 3 and _WORDS.findall(rest):
        words = _WORDS.findall(rest)
    index = 0
    while index < len(words) and words[index] in _FILLERS:
        index += 1
    return index < len(words) and words[index] in _OPENERS
