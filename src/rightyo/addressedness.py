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

`dismissal_shaped` is its counterpart for natural dismissals (#98): a short turn that
sounds like "stop", "never mind" or "go away" is released at once instead of waiting for
the merge or reply-wait hold. It only shortens the wait; whether the turn dismisses the
assistant is the decision model's judgement (or the exact configured stop phrases).
"""

from __future__ import annotations

import re
from typing import Any

from .contracts import Addressing, ContractError, normalize_phrase

# The default setting for the single-user assistant pilot (Hailing Station).
DEFAULT_SCENE = (
    "One primary user is talking to an AI assistant through a phone or tablet. Most of "
    "the user's directed speech that is not clearly aimed at another person present is "
    "meant for the assistant, including questions and requests that do not use its name. "
    "Other voices may be the assistant's own audio playback, other AI agents or media, "
    "rather than people in the room. A question or request that no other person answers "
    "is likely meant for the assistant."
)
# A precision-first setting (#199): only clear requests to the assistant are addressed to it;
# thinking aloud, narration, reading aloud, talk about it and media are not. Evaluated on a
# private hold-out of real turns it raised precision to 0.86 at 0.52 recall, so it is a
# per-session choice for situations that need few false acts, not the default.
CLEAR_REQUESTS_SCENE = (
    "One owner and possibly other people are in the room with an AI assistant listening "
    "through a device. The assistant acts only on clear requests or questions addressed to "
    "it, with or without its name. Thinking aloud, narrating what is on a screen, reading "
    "text or errors aloud, notes to self, talk between people, talk about the assistant in "
    "the third person, and media playback are not addressed to the assistant. If a person "
    "present could be the intended recipient, assume they are. A request nobody present "
    "answers, in a quiet gap, is more likely meant for the assistant. When the assistant has "
    "just acted or replied, a short continuation from the same speaker that refers back to "
    'that request ("and the second one too", "no, the other one", "make it shorter", "not '
    'you", "thanks") is addressed to the assistant. When in doubt, the assistant stays silent.'
)
# The built-in scenes `listen --scene` selects by name for one session.
SCENES = {"default": DEFAULT_SCENE, "clear-requests": CLEAR_REQUESTS_SCENE}
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


# Words a finished request rarely ends on: articles, connectives, a few prepositions,
# fillers, modals and possessives ("add the", "remind me to", "and"). Particles and object
# pronouns ("turn it on", "do it", "thank you", "play this") end complete commands, so
# they are left out.
_DANGLING = frozenset(
    "a an the and or but because if when like um uh to of with for at "
    "can could would should will my your our".split()
)
# Words that open a request and may come alone, or beside a name, before it.
_OPENING = frozenset({"can", "could", "you", "hey", "so", "um", "uh"})


def unfinished_shaped(text: str, addressing: Addressing | None = None) -> bool:
    """Whether a turn looks cut off mid-request (English, deterministic, approximate, #199).

    True when its last word is an article, connective, preposition, filler, modal or
    possessive that expects more, or when it is one or two words, each a name of the
    assistant or a request opener ("Jarvis,", "hey Jarvis", "can you"). A question mark
    ends a turn ("What time is it?"); other trailing punctuation is ignored, since
    recognition often ends a fragment with a period.
    """
    words = _WORDS.findall(text.lower())
    if not words or text.rstrip().endswith("?"):
        return False
    if words[-1] in _DANGLING:
        return True
    if len(words) > 2:
        return False
    # A bare name or opener, or one beside the other ("hey Jarvis"), but not "Jarvis stop".
    named = [word for word in words if mentions_name(addressing, word)]
    rest = [word for word in words if word not in named]
    return all(word in _OPENING for word in rest) and bool(named or rest)


# A follow-up's cue that it continues the exchange with the assistant (#199). Precision
# first: talk to a pet or a passer-by often has "it", "that", "you" or "no" in it ("No,
# sit", "Is that your dog?"), so those count only in the shapes below.
_LETTERS = re.compile(r"[a-z]+")
# Words that refer back to the request or its result, anywhere in the turn; "other",
# "same", "second" and "first" count only in the phrases below ("just a second", "same to
# you" and "on the other hand" are everyday talk).
_REFERRING = frozenset({"again", "instead", "previous"})
_REFERRING_PAIRS = frozenset(
    {
        ("the", "same"),
        ("the", "second"),
        ("second", "one"),
        ("the", "first"),
        ("first", "one"),
        ("other", "one"),
        ("one", "more"),
    }
)
# A bare "it", "that" or "this" counts after a verb that acts on the assistant's output.
_OUTPUT_VERBS = frozenset(
    "turn make change play read repeat send set cancel skip undo save open show fix delete "
    "add remove move".split()
)
_OBJECTS = frozenset({"it", "that", "this", "those", "these", "them"})
_MODALS = frozenset({"can", "could", "would", "will"})
# Short adjustments of what the assistant just did ("louder", "next one", "more please").
_ADJUSTMENTS = frozenset(
    (
        "louder quieter softer more less another next skip repeat cancel slower faster "
        "shorter longer bigger smaller"
    ).split()
)
# Verbs that open a request to the assistant ("Run the review", "Check my calendar") when
# the turn has at least three words and no "your" ("Call your mom back" is to a person).
# Verbs of pet commands and everyday talk ("drop", "make", "put", "get", "go", "come",
# "leave", "look", "give", "show", "play", "tell") are left out.
_REQUEST_VERBS = frozenset(
    (
        "run check search open close write create draft summarize summarise review update "
        "list explain call email text message remind schedule book read pause resume "
        "send add remove delete fix"
    ).split()
)
# Courtesy and transition words that may come before the opening verb ("please run ...").
_VERB_LEAD = frozenset("please ok okay now so alright".split())
# A correction of what the assistant just said ("you mean $134", "I meant the other one").
# "I said ..." is left out: "No, I said sit" is how a dog is told twice. "What do you mean"
# and "do you mean" ask a person.
_CORRECTIONS = frozenset({("you", "mean"), ("i", "meant")})


# How an answer to the assistant's own question opens ("Yes, do both", "Go ahead") (#199).
_ANSWER_WORDS = frozenset("yes yeah yep yup sure no nope okay ok".split())
_ANSWER_PAIRS = frozenset({("go", "ahead"), ("do", "it"), ("sounds", "good")})


def answer_shaped(text: str) -> bool:
    """Whether a turn opens as an answer to a yes/no question (English, approximate, #199).

    True when its first word is "yes", "yeah", "yep", "yup", "sure", "no", "nope", "okay"
    or "ok", or it opens "go ahead", "do it" or "sounds good". Only meaningful right after
    the assistant asked the owner a question; contractions are split as in `follow_up_cued`.
    """
    words = _LETTERS.findall(text.lower())
    return bool(words) and (words[0] in _ANSWER_WORDS or tuple(words[:2]) in _ANSWER_PAIRS)


def follow_up_cued(text: str, addressing: Addressing | None = None) -> bool:
    """Whether a turn carries a cue that it continues an exchange with the assistant (#199).

    True when it names the assistant; asks the assistant ("can you", "could you"); refers
    back ("the other one", "again", "instead"); acts on the result ("play that", "turn it
    up"); opens by adding to it ("and ...", "what about ...", "how about ...", "actually
    ..."); is a short adjustment ("louder", "next one"); opens a request of three or more
    words with a request verb ("run the review"); says "please" beside a request verb or
    a modal ("please have it fix both"); or corrects it ("you mean ...",
    "I meant ..."). Talk to a pet or a passer-by
    ("No, sit", "Is that your dog?") usually carries none. English, deterministic,
    approximate; contractions are split ("that's" is "that s").
    """
    words = _LETTERS.findall(text.lower())
    if not words:
        return False
    pairs = list(zip(words, words[1:]))
    # "And eggs" adds to a request; "and you" is a reply to someone else.
    adds = words[0] == "and" and len(words) > 1 and words[1] != "you"
    lead = next((word for word in words if word not in _VERB_LEAD), "")
    return (
        mentions_name(addressing, text)
        or adds
        or words[0] == "actually"
        or not _REFERRING_PAIRS.isdisjoint(pairs)
        or tuple(words[:2]) in {("what", "about"), ("how", "about")}
        or not _REFERRING.isdisjoint(words)
        or any(a in _MODALS and b == "you" for a, b in pairs)
        or any(a in _OUTPUT_VERBS and b in _OBJECTS for a, b in pairs)
        # One word between ("show me that", "read me this").
        or any(a in _OUTPUT_VERBS and c in _OBJECTS for a, c in zip(words, words[2:]))
        or (len(words) <= 3 and words[0] in _ADJUSTMENTS)
        or (len(words) >= 3 and lead in _REQUEST_VERBS and "your" not in words)
        or ("please" in words and not (_REQUEST_VERBS | _MODALS).isdisjoint(words))
        or any(
            pair in _CORRECTIONS and (index == 0 or words[index - 1] not in {"do", "what"})
            for index, pair in enumerate(pairs)
        )
    )


# Cue phrases of a dismissal, as `normalize_phrase` writes them (apostrophes removed,
# casefolded). A turn is dismissal-shaped only when one of them opens it (after fillers
# and an optional short vocative) and nothing but tail words follow, so a request that
# merely contains "cancel", "quiet" or "hold on" is never split or released early.
_DISMISSAL_CUES = tuple(
    tuple(cue.split())
    for cue in (
        "stop",
        "quiet",
        "hush",
        "shush",
        "shh",
        "shhh",
        "cancel",
        "nevermind",
        "never mind",
        "shut up",
        "go away",
        "not now",
        "not right now",
        "not you",
        "i wasnt talking to you",
        "i was not talking to you",
        "wasnt talking to you",
        "not talking to you",
        "hang on",
        "hold on",
        "forget it",
        "leave me alone",
        "leave it",
        "ignore that",
        "thats all",
        "thats enough",
        "enough",
        "no thanks",
        "no thank you",
        "no no",
        "be quiet",
    )
)
# Words that may open a dismissal before its cue.
_DISMISSAL_LEAD = frozenset("no oh ok okay please just um uh hey so sorry wait".split())
# Words that may follow the cue: politeness, emphasis and "that", nothing with content.
_DISMISSAL_TAIL = frozenset(
    """
    please now right thanks thank you sorry that it this already enough thats all just a
    second sec minute moment for then again ok okay no not talking to stop quiet
    """.split()
)
# The longest vocative ("Okay Haili", "hey there Jarvis") dropped before or after the cue.
_MAX_VOCATIVE_WORDS = 3
MAX_DISMISSAL_WORDS = 10


def _cue_at(words: list[str], index: int) -> int:
    """The length of the longest cue starting at `index`, or 0."""
    return max(
        (len(cue) for cue in _DISMISSAL_CUES if tuple(words[index : index + len(cue)]) == cue),
        default=0,
    )


def _has_cue(words: list[str]) -> bool:
    return any(_cue_at(words, index) for index in range(len(words)))


def dismissal_shaped(text: str) -> bool:
    """Whether a short turn sounds like a dismissal (English, deterministic, approximate).

    True for a turn of at most ten words that, after leading fillers ("no", "okay",
    "please") and an optional comma-delimited vocative of up to three words at either end
    ("Jarvis, ...", "..., Haley"), opens with a cue such as "stop", "quiet", "never mind",
    "go away", "not now", "not you", "I wasn't talking to you", "hang on" or "forget it",
    followed only by tail words ("please", "now", "that", "thanks"). "Stop the timer",
    "Cancel my meeting" and "Hold on, what's the weather?" are not dismissal-shaped. It
    decides only that the turn is released without the merge and reply holds; it never
    dismisses anything on its own.
    """
    segments = [normalize_phrase(part).split() for part in text.split(",")]
    segments = [segment for segment in segments if segment]
    if not segments or sum(len(segment) for segment in segments) > MAX_DISMISSAL_WORDS:
        return False
    # Drop a short vocative segment without a cue at either end ("Jarvis," / ", Haley").
    if (
        len(segments) > 1
        and len(segments[0]) <= _MAX_VOCATIVE_WORDS
        and not _has_cue(segments[0])
        and not set(segments[0]) <= _DISMISSAL_LEAD
    ):
        segments = segments[1:]
    if (
        len(segments) > 1
        and len(segments[-1]) <= _MAX_VOCATIVE_WORDS
        and not _has_cue(segments[-1])
        and not set(segments[-1]) <= _DISMISSAL_TAIL
    ):
        segments = segments[:-1]
    words = [word for segment in segments for word in segment]
    index = 0
    while index < len(words) and words[index] in _DISMISSAL_LEAD and not _cue_at(words, index):
        index += 1
    size = _cue_at(words, index)
    if not size:
        return False
    index += size
    while index < len(words):
        size = _cue_at(words, index)
        if size:
            index += size
        elif words[index] in _DISMISSAL_TAIL:
            index += 1
        else:
            return False
    return True


def mentions_name(addressing: Addressing | None, text: str) -> bool:
    """Whether any run of words in `text` spells a configured name or variant.

    Runs are as long as the longest configured name or variant, so a long name is found.
    """
    if addressing is None:
        return False
    spellings = [s for name in addressing.names for s in (name, *addressing.spellings(name))]
    longest = max(len(normalize_phrase(spelling).split()) for spelling in spellings)
    words = normalize_phrase(text).split()
    return any(
        addressing.name_for(" ".join(words[i : i + size])) is not None
        for size in range(1, longest + 1)
        for i in range(len(words) - size + 1)
    )
