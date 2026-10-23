"""Versioned bounded speech events; no routing, replies, or application actions."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import unicodedata
from collections import deque
from dataclasses import replace
from threading import RLock

from rightyo.addressedness import answer_shaped, follow_up_cued, mentions_name
from rightyo.contracts import (
    DISMISSING,
    Acknowledgement,
    Addressing,
    ContractError,
    Conversation,
    DecisionEvent,
    Dismissal,
    SpeakerPriority,
    Turn,
    formed_request_text,
    identifier,
    integer,
    speaker_role,
)
from rightyo.memory import TranscriptMemory
from rightyo.providers import ConfiguredPriorityProvider, ProviderError

MAX_EVENT_BYTES = 1200000
# The duck-mode control version advertised on `started` (hailing-station #735). Version 2
# is privacy mode, which keeps nothing; version 3 (S-8) also acks every `{"duck": true}`
# line it applies with a `privacy` `on` event, and a host enters privacy only for 3+.
DUCK_VERSION = 3
# Privacy mode's local phrases (hailing-station #735 S-1), as filler-free word tuples. The
# exit needs a configured name and the owner; the status works for anyone, named or not;
# the entry is the host's (S-2) and is only released as a whole turn here.
PRIVACY_EXIT = (
    ("i'm", "back"),
    ("i", "am", "back"),
    ("start", "listening"),
    ("privacy", "off"),
    ("privacy", "mode", "off"),
)
PRIVACY_STATUS = (
    ("what", "mode", "am", "i", "in"),
    ("are", "you", "recording"),
    ("are", "you", "listening"),
)
# Privacy entry is intentional-only, as the host's rule (S-2): one of these phrases as the
# whole turn, after an optional leading name or address, with only `PRIVACY_EDGE` words
# before and after it. Never mid-sentence, with a trailing object, or as a question.
PRIVACY_ENTRY = frozenset(
    {
        ("stop", "recording"),
        ("stop", "the", "recording"),
        ("stop", "all", "recording"),
        ("stop", "the", "recordings"),
        ("stop", "listening"),
        ("privacy", "mode"),
        ("go", "private"),
    }
)
PRIVACY_EDGE = frozenset({"please", "okay", "ok", "now", "um", "uh", "so", "thanks"})
# Words that may open an address before the name ("hey Jarvis, privacy mode").
PRIVACY_ADDRESS = frozenset({"hey"})
# Words dropped anywhere before matching, so "um, ok Jarvis, so I'm back" still matches.
PRIVACY_FILLER = frozenset(
    {
        "um",
        "umm",
        "uh",
        "uhm",
        "er",
        "erm",
        "ah",
        "hmm",
        "mm",
        "oh",
        "ok",
        "okay",
        "so",
        "hey",
        "well",
        "please",
    }
)
# A lone name fragment ("Jarvis,") and an exit said this soon after it are one named exit.
PRIVACY_SPLIT_NAME_MS = 4000
# Private turns already matched, by utterance id only, so a turn offered twice matches
# once; bounded, and never part of the session's 1,000-turn `_seen` budget.
MAX_PRIVATE_SEEN = 64
# Duck-mode spans remembered per session; beyond this many the oldest two are merged into
# one, which can only make more turns count as heard in duck mode. They are never pruned by
# retention: a held or joined turn can start well before the retention cutoff it ends after.
MAX_DUCK_SPANS = 64
# How long a reply reported as playing can keep a conversation engaged when its end is
# never reported (#124).
MAX_REPLY_HOLD_MS = 180000
# How long after a conversation lapsed by timeout a spoken reply can still revive it (#158).
REVIVE_WINDOW_MS = 600_000
MAX_QUEUE_BYTES = 4194304
MAX_PENDING_BYTES = 1048576
# The context a role provider sees when a speaker first appears.
ROLE_CONTEXT_TURNS = 8
# What a host should do on a `dismiss` (#98), by kind. A late withdrawal of a request
# whose decision was still pending names only `pending_request`.
DISMISS_SCOPES = {
    "stop": ["playback", "pending_request"],
    "disengage": ["playback", "pending_request", "engagement"],
}
# Delivered requests a later dismissal can still withdraw; the oldest is dropped beyond.
MAX_WITHDRAWABLE = 32
# Speakers whose last explicit conversation end is remembered (#167).
MAX_ENDED_SPEAKERS = 64
# The assistant's own playback remembered for the answered gate (#199), and how close to it
# another speaker's start still counts as its echo rather than a person.
MAX_PLAYBACKS = 8
PLAYBACK_MARGIN_MS = 300
# A request of at most this many words, without a name, starting this soon after the latest
# one ended, by the same or an unidentified speaker, is a tag on it ("right?"), so the latest
# request in a decision's context stays the one it tags (#199).
TAG_MAX_WORDS = 2
TAG_MS = 10_000
# A bound guest's display name on a `transcript` event (hailing-station #434): the bound
# HailProtocol's `AmbientOverheard.name` accepts, in Unicode scalars, and the categories it
# refuses (control, format, surrogate, private use, unassigned, line/paragraph separator).
MAX_SPEAKER_NAME_SCALARS = 64
REFUSED_NAME_CATEGORIES = frozenset({"Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp"})
# Owner-voice status (rightyo#164): the default and allowed seconds of scored speech before
# an enrolled owner never bound counts as "not recognised on this input".
DEFAULT_OWNER_VOICE_THRESHOLD_S = 20
MIN_OWNER_VOICE_THRESHOLD_S = 5
MAX_OWNER_VOICE_THRESHOLD_S = 600


class OwnerVoice:
    """Owner-voice status events (rightyo#164); absent means off and nothing changes.

    With it, a session whose role provider reports owner-voice evidence advertises
    `owner_voice: {"threshold_s": N}` in `started` and may emit `owner_voice` events:
    `unrecognized` once, after at least `threshold_s` seconds of speech were scored by live
    identification while an enrolled owner voiceprint was compared and no label bound to
    it; then `recognized` once, if an owner binding follows. Never scores, ids or names.
    """

    __slots__ = ("threshold_s",)

    def __init__(self, threshold_s=DEFAULT_OWNER_VOICE_THRESHOLD_S):
        integer(threshold_s, "owner voice threshold", MIN_OWNER_VOICE_THRESHOLD_S)
        if threshold_s > MAX_OWNER_VOICE_THRESHOLD_S:
            raise ContractError("invalid owner voice threshold")
        self.threshold_s = threshold_s

    def to_dict(self):
        return {"threshold_s": self.threshold_s}


def speaker_name(value):
    """`value` when it is a display name a host may show, else None (never raises).

    1 to 64 Unicode scalars, no control, format or line-breaking characters, and no
    leading or trailing whitespace.
    """
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_SPEAKER_NAME_SCALARS
        or value != value.strip()
        or any(unicodedata.category(char) in REFUSED_NAME_CATEGORIES for char in value)
    ):
        return None
    return value


def _privacy_words(text, filler=PRIVACY_FILLER):
    """`text` as lowercase words of letters, digits and apostrophes, with `filler` dropped.

    Digits are kept: a configured name may hold them ("R2D2"), and a number in a turn is
    a word of its own, never silently removed from around a phrase.
    """
    cleaned = "".join(c if c.isalnum() or c == "'" else " " for c in text.replace("\u2019", "'"))
    return [word for word in cleaned.lower().split() if word not in filler]


def _filler_for(addressing):
    """The filler dropped before matching, less any word of a configured name or variant: a
    name such as "Okay" or "Hey" is still a name, so its owner keeps the way back in."""
    if addressing is None:
        return PRIVACY_FILLER
    named = {
        word
        for name in addressing.names
        for spelling in (name, *addressing.spellings(name))
        for word in _privacy_words(spelling, ())
    }
    return PRIVACY_FILLER - named


def _matched_words(text, addressing):
    """`text` as privacy words with filler dropped, except where a filler word is part of a
    configured name or variant matched at that place (round 18): "please" stays in "Please
    Jarvis, I'm back", but is still filler in "Jarvis, please, I'm back"."""
    filler = _filler_for(addressing)
    words = _privacy_words(text, filler)
    if filler == PRIVACY_FILLER:
        return words
    named = set()
    for at in range(len(words)):
        for width in range(1, min(3, len(words) - at) + 1):
            if addressing.name_for(" ".join(words[at : at + width])) is not None:
                named.update(range(at, at + width))
    return [word for at, word in enumerate(words) if word not in PRIVACY_FILLER or at in named]


def _negated(words, at):
    """Whether "don't" or "do not" comes in the three words before `at`."""
    before = words[max(0, at - 3) : at]
    return any(
        word in {"don't", "dont"} or (word == "do" and before[index + 1 : index + 2] == ["not"])
        for index, word in enumerate(before)
    )


def _phrase_at(words, at, phrases):
    return any(tuple(words[at : at + len(phrase)]) == phrase for phrase in phrases)


def _phrase_in(words, phrases):
    """Whether one of `phrases` appears anywhere in `words`, not negated."""
    return any(
        _phrase_at(words, at, phrases) and not _negated(words, at) for at in range(len(words))
    )


def privacy_phrase(text, addressing):
    """Which of privacy mode's local phrases `text` holds (hailing-station #735 S-1).

    A set of: "exit" (a configured name or variant of up to three words, then an exit
    phrase), "rest" (the turn starts with an exit phrase, for a split name), "name" (the
    turn is only a name), "status" (named or not) and "entry". For all but the entry,
    case, punctuation and filler are ignored, and the phrase may come anywhere in the turn;
    "don't" or "do not" in the three words before it (before the name, for the exit) means
    no match. The entry counts only as the whole turn (`_standalone_entry`) and never as a
    question. Nothing is kept: the words exist only inside this call.
    """
    words = _matched_words(text, addressing)
    kinds = set()
    if addressing is not None:
        if words and addressing.name_for(" ".join(words)) is not None:
            kinds.add("name")
        for at in range(len(words)):
            for width in range(1, 4):
                if (
                    at + width < len(words)
                    and addressing.name_for(" ".join(words[at : at + width])) is not None
                    and _phrase_at(words, at + width, PRIVACY_EXIT)
                    and not _negated(words, at)
                ):
                    kinds.add("exit")
    if _phrase_at(words, 0, PRIVACY_EXIT):
        kinds.add("rest")
    if _phrase_in(words, PRIVACY_STATUS):
        kinds.add("status")
    if not text.rstrip().endswith("?") and _standalone_entry(_privacy_words(text, ()), addressing):
        kinds.add("entry")
    return frozenset(kinds)


def _standalone_entry(words, addressing):
    """Whether `words` (filler kept) are an entry phrase as the whole turn: edge words, an
    optional address and name, edge words, the phrase, then only edge words."""

    def edges(at):
        while at < len(words) and words[at] in PRIVACY_EDGE:
            at += 1
        return at

    at = edges(0)
    starts = {at}
    if addressing is not None:
        named = at + 1 if at < len(words) and words[at] in PRIVACY_ADDRESS else at
        for width in range(1, 4):
            if named + width <= len(words) and (
                addressing.name_for(" ".join(words[named : named + width])) is not None
            ):
                starts.add(edges(named + width))
    for start in starts:
        for phrase in PRIVACY_ENTRY:
            end = start + len(phrase)
            if tuple(words[start:end]) == phrase and all(w in PRIVACY_EDGE for w in words[end:]):
                return True
    return False


def _role_state(record):
    """The bounded fields a role provider needs; session/source identifiers are omitted."""
    state = {name: record[name] for name in ("text", "speaker_id", "start_ms", "end_ms", "overlap")}
    if "role" in record:
        state["role"] = record["role"]
    return state


def _with_role(turn, role):
    record = turn.to_dict()
    if role is not None:
        record["role"] = role
    return record


def _unlabelled(record):
    """#242 round 25: a record heard before the speakers were forgotten (#735 S-1), as what is
    derived from it after the forget sees it: no speaker label, which may now name someone
    else, and no owner or trusted role, which only that label gave."""
    record = {**record, "speaker_id": None}
    if record.get("role") in {"owner", "trusted"}:
        record["role"] = "unknown"
    return record


def encode_json(payload):
    """The ASCII-safe JSON representation shared by wire output and byte budgets."""
    return json.dumps(payload, ensure_ascii=True, allow_nan=False)


def _speech_summary(value):
    """A validated, detached copy of the advertised speech backends, or None."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"transcriber", "diarizer"}:
        raise ContractError("invalid speech backend summary")
    summary = {}
    for stage, entry in value.items():
        if not isinstance(entry, dict) or set(entry) != {"kind", "id"}:
            raise ContractError("invalid speech backend summary")
        identifier(entry["kind"], f"{stage} kind")
        identifier(entry["id"], f"{stage} id")
        summary[stage] = {"kind": entry["kind"], "id": entry["id"]}
    return summary


def _audio_input(value):
    """A detached copy of the advertised host-supplied audio input format."""
    expected = {"source": "stdin", "encoding": "s16le", "sample_rate": 16000, "channels": 1}
    if value != expected:
        raise ContractError("invalid audio input summary")
    return dict(expected)


class SpeechEvents:
    """Local v1 producer. Frozen request context never observes later turns.

    Consumers still need their own stale-session and once-only dispatch policy.
    Overflow fails closed rather than silently losing a request. Cancellation
    releases pending plaintext; a normal end preserves already queued delivery.
    """

    def __init__(
        self,
        *,
        retention_ms=300000,
        max_pending=128,
        report=None,
        retain_transcripts=True,
        duck_control=True,
    ):
        if type(retain_transcripts) is not bool:
            raise ContractError("invalid transcript retention setting")
        if type(duck_control) is not bool:
            raise ContractError("invalid duck control setting")
        # Whether this publisher's host control channel takes `{"duck": ...}` lines
        # (hailing-station #735): false for a coaching listener, whose control descriptor
        # carries only coaching records, so a live controller never advertises duck mode.
        self.duck_control = duck_control
        self._retain_transcripts = retain_transcripts
        # At least five: one open request plus the owner's transcript, attention and
        # request, with a terminal slot, so the static and dynamic override bounds agree.
        integer(max_pending, "max_pending", 5)
        if max_pending > 128:
            raise ContractError("invalid event queue budget")
        self.retention_ms = retention_ms
        self.max_pending = max_pending
        # Optional content-free diagnostics (labels and numbers only, never text or ids).
        self._report = report
        self._acknowledgement = None
        # Open non-owner requests are bounded below the queue capacity, with headroom for
        # the owner's transcript, attention and request, so one owner decision's burst of
        # overrides can never overflow the queue; exceeding the bound fails closed.
        self.max_open = max_pending - 4
        self._lock = RLock()
        self._memory = TranscriptMemory(retention_ms=retention_ms)
        self._session = None
        self._sequence = self._now = 0
        self._active = False
        self._terminal = False
        self._attention_enabled = True
        self._sessions = set()
        self._queue = deque()
        self._queue_bytes = 0
        self._pending = {}
        self._pending_bytes = 0
        self._seen = {}
        self._decided = set()
        # Speaker roles are fixed per speaker when first emitted; open requests are the
        # emitted non-owner requests an owner's attended turn or stop phrase supersedes.
        self._priority = None
        self._roles = {}
        # Live identification roles (#137 step 4b): resolved per turn from the provider's
        # current bindings instead of fixed per speaker, with enrolled precedence.
        self._per_turn = False
        # Turns whose speaker label was partly inferred (edge attribution or tail join):
        # they keep their role for attention and precedence, never for owner authority.
        # Utterance id -> turn end, pruned by retention like the other per-turn state.
        self._inferred = {}
        self._open = {}
        # Pending non-owner turns an owner superseded before their decisions arrived,
        # mapped to the owner's utterance: their late decisions never emit a request.
        self._superseded = {}
        # Per speaker key, the end of their latest direct (non-follow-up) request: the one
        # appendix it may take (#199) is measured from here, and taking it clears the entry.
        self._appendix_from = {}
        # (end_ms, text, utterance_id, speaker_id) of the latest request formed this session,
        # or None, also after a dismissal or an explicit end (#199).
        self.last_request = None
        # Recent [start, end] stream times of the assistant's own playback (end None while
        # it plays), as the host reports them (#124); bounded.
        self._playbacks = deque(maxlen=MAX_PLAYBACKS)
        # Stream time the latest reply ended with a question to the owner (#199), or None;
        # cleared when the next reply starts.
        self._asked_at = None
        # Stream time the assistant began replying to the latest request (#213), or None;
        # cleared with the question, but a later reply to the same request keeps it.
        self._replied_at = None
        # "off" without a provider, "ready" with one, "unavailable" after a hosted role
        # question failed, or "rejected" after a provider contract violation; either
        # degrades roles to configured/unknown for the rest of the session.
        self.role_status = "off"
        self._degraded_turn = None
        # Optional request forming renders a convenience string beside the raw turns;
        # off by default, and the raw turn/context are unchanged either way.
        self._former = None
        self._addressing = None
        # Bound guests' display names on transcripts (hailing-station #434), off by default.
        self._names = False
        # Other agents' names (#161): a turn naming one, and not this assistant, is never ours.
        self._other_agents = None
        # Media mode (hailing-station #551), set by the host while the owner watches TV or
        # videos: only the owner's voice naming this assistant attends, and no follow-ups.
        self._media = False
        # Turns accepted while media mode was on (utterance id -> turn end, pruned by
        # retention): the mode each pending turn is decided under (see `_media_for`).
        self._media_turns = {}
        # Privacy mode (internally duck, v3; hailing-station #735 S-1), set by the host: a
        # turn heard while it is on is never emitted, decided or kept, and is matched only
        # for the local exit and status phrases (see `duck`). Like media mode it outlasts
        # sessions.
        self._duck = False
        # The stream time the enrolled owner's lone name fragment ended, heard while privacy
        # was on (a split exit), or None; a time only, never the words. Cleared by every
        # other turn, private or not, and whenever the mode changes.
        self._private_name_end = None
        # Utterance ids of private turns already matched (see MAX_PRIVATE_SEEN).
        self._private_seen = deque(maxlen=MAX_PRIVATE_SEEN)
        # Whether `transcript` events may carry `role_source` this session: a live session
        # with privacy control and roles (see `start` and `_role_source`).
        self._role_sources = False
        # Whether this session advertised `duck`, so a host's `{"duck": true}` gets a
        # `privacy` `on` ack (#735 S-8; see `duck`).
        self._duck_acks = False
        # Counts only, reported when privacy goes off.
        self._private_counts = dict.fromkeys(
            ("discarded", "exit", "exit_refused", "status", "failed"), 0
        )
        # This session's duck-mode spans, [on_ms, off_ms or None] in stream time, bounded by
        # MAX_DUCK_SPANS: a turn overlapping one was heard in duck mode (`heard_ducked`).
        self._duck_spans = []
        # Owner-voice status (rightyo#164), off by default: the settings, and the last state
        # emitted this session (None, "unrecognized" or "recognized"; each at most once).
        self._owner_voice = None
        self._owner_voice_state = None
        # Natural dismissal (#98), off unless configured at start.
        self._dismissal = None
        self._stop_rules = SpeakerPriority()
        # Delivered requests a dismissal can withdraw: request id -> request turn facts.
        self._delivered = {}
        # Pending turns withdrawn before their decisions arrived -> the dismissal's fields.
        self._withdrawn = {}
        # Turns already dismissed, mapped to whether the turn's own decision may still
        # dismiss it fully: only the stop-phrase fast path limited by incomparable
        # attribution, not by the speaker's role, which would limit the decision too.
        self._dismissed = {}
        # Cool-downs after `disengage`: speaker label -> (from_ms, until_ms); the key None
        # applies to every speaker (a dismissal whose speaker cannot be compared).
        self._cooldowns = {}
        # Conversation mode (#82), off unless configured at start, and the one engaged
        # speaker: {"key", "speaker_id", "from_ms", "until_ms"}, plus "replying_since" while
        # the host reports a reply playing (#124), or None when ambient.
        self._conversation = None
        self._engaged = None
        # The last engagement that lapsed by timeout, kept so a later spoken reply can revive
        # it (rightyo#158): {"engaged": <the engagement>, "at_ms": <when it lapsed>}, or None.
        self._dormant = None
        # Utterances accepted before the speakers were last forgotten (hailing-station #735 S-1)
        # and not yet decided: their labels may now name someone else. Always within `_pending`.
        self._label_stale = set()
        # #242 round 25: every utterance transcribed before the speakers were last forgotten
        # this session. What is derived from one after the forget (a formed request's context,
        # a role question's past turns) carries no label or role of it (`_unlabelled`); the
        # published transcript, request turn and context keep them, as fingerprinted.
        self._old_labels = set()
        # #242 round 25: per queued request with a formed string, the request record and the
        # context utterances it was formed with unlabelled, so a re-render after a pruning
        # (`expire`) says what the first one did, never the published turn's old label.
        self._formed = {}
        # Set once the speakers are forgotten (round 17, Jarvis): a diarizer label configured as
        # the owner or trusted may now name anyone, so until the session restarts only a live
        # voiceprint binding gives those roles.
        self._labels_renewed = False
        # Per speaker key, the start of the latest explicit end of that speaker's
        # conversation (#167): their request said before it, decided out of order after it,
        # never re-opens it. Other speakers' requests are unaffected. Bounded (`_mark_ended`).
        self._ended = {}
        # One acknowledgement per spoken request (#122), with acknowledgement gating on:
        # the last acknowledged request's un-answered acknowledgement,
        # {"until_ms", "heard_end"}, or None. See `_hold_ack` and `reply`.
        self._pending_ack = None

    def _payload(self, kind, sequence, fields):
        return {
            "schema_version": 1,
            "type": kind,
            "session_id": self._session,
            "sequence": sequence,
            "emitted_at_ms": self._now,
            **fields,
        }

    @staticmethod
    def _size(payload):
        return len(encode_json(payload)) + 1  # Include the JSONL newline.

    def _role_source(self, turn, role, inferred):
        """ "voiceprint" when `turn` is the owner by live voiceprint identification, else
        None (hailing-station #735, #807): the published role is `owner`, it came from a
        live identification binding of the label to an enrolled owner voice when the role
        was resolved for this turn, and no word of the turn has an inferred label. Never for
        a label configured as the owner, an inferred label, a guest or an unlabelled
        speaker, and only in a live session with privacy control and roles. Never raises.
        """
        if role != "owner" or inferred or not self._role_sources or not self._per_turn:
            return None
        try:
            check = getattr(self._priority, "resolved_by_voiceprint", None)
            if callable(check) and check(turn.utterance_id) is True:
                return "voiceprint"
        except Exception:  # noqa: BLE001 - a source is descriptive and never fails the turn
            pass
        return None

    def _speaker_name(self, turn, role):
        """A bound guest's validated display name for this turn, or None (#434).

        Only when names are on, never for an owner, and never for an inferred-label turn,
        whose words may not all be the bound voice's. Never raises.
        """
        if not self._names or role in (None, "owner") or turn.utterance_id in self._inferred:
            return None
        try:
            return speaker_name(self._priority.name_for(turn, role))
        except Exception:  # noqa: BLE001 - a name is optional and never fails the turn
            return None

    def _emit(self, kind, **fields):
        if kind == "dismiss" or (
            kind == "conversation"
            and fields.get("state") == "ambient"
            and fields.get("reason") in {"dismissed", "closed"}
        ):
            # A dismissed, withdrawn or explicitly closed exchange is no context (#199).
            # Talk to another person or a lapsed window keeps it: a follow-up 30 to 120 s
            # later still refers back. Any dismissal clears it, which errs on the safe side.
            self.last_request = None
        self._sequence += 1
        payload = self._payload(kind, self._sequence, fields)
        size = self._size(payload)
        if (
            size > MAX_EVENT_BYTES
            or len(self._queue) >= self.max_pending
            or self._queue_bytes + size > MAX_QUEUE_BYTES
        ):
            self._clear_content()
            self._active = False
            raise ContractError("speech event consumer backlog exceeded")
        self._queue.append((payload, size))
        self._queue_bytes += size

    def _clear_content(self):
        self._memory.clear()
        self._pending.clear()
        self._pending_bytes = 0
        self._queue.clear()
        self._queue_bytes = 0
        self._open.clear()
        self._superseded.clear()
        self._appendix_from.clear()
        self.last_request = None
        self._playbacks.clear()
        self._asked_at = None
        self._replied_at = None
        self._delivered.clear()
        self._withdrawn.clear()
        self._formed.clear()

    def start(
        self,
        session_id,
        now_ms=0,
        *,
        attention_enabled=True,
        addressing=None,
        priority=None,
        former=None,
        speech=None,
        audio_input=None,
        dismissal=None,
        conversation=None,
        acknowledgement=None,
        other_agents=None,
        owner_voice=None,
        duck=False,
    ):
        with self._lock:
            identifier(session_id, "session_id")
            if owner_voice is not None and not isinstance(owner_voice, OwnerVoice):
                raise ContractError("invalid owner voice settings")
            if acknowledgement is not None and not isinstance(acknowledgement, Acknowledgement):
                raise ContractError("invalid acknowledgement")
            if dismissal is not None and not isinstance(dismissal, Dismissal):
                raise ContractError("invalid dismissal")
            if conversation is not None and not isinstance(conversation, Conversation):
                raise ContractError("invalid conversation")
            speech = _speech_summary(speech)
            if audio_input is not None:
                audio_input = _audio_input(audio_input)
            integer(now_ms, "now_ms")
            if type(attention_enabled) is not bool:
                raise ContractError("invalid attention capability")
            if not self._retain_transcripts and attention_enabled:
                raise ContractError("attention requires transcript retention")
            if type(duck) is not bool:
                raise ContractError("invalid duck capability")
            if other_agents is not None and not isinstance(other_agents, Addressing):
                raise ContractError("invalid other agents")
            if addressing is not None and not isinstance(addressing, Addressing):
                raise ContractError("invalid addressing")
            if priority is not None and (
                not callable(getattr(priority, "assign", None))
                or not isinstance(getattr(priority, "priority", None), SpeakerPriority)
            ):
                raise ContractError("invalid speaker priority provider")
            if former is not None and not callable(getattr(former, "form", None)):
                raise ContractError("invalid request former")
            if former is not None:
                identifier(getattr(former, "kind", None), "request former kind")
            if self._active or self._queue:
                raise ContractError("finish and drain the previous event session")
            session_key = hashlib.sha256(session_id.encode("utf-8")).digest()
            if session_key in self._sessions:
                raise ContractError("restart requires a new event session identity")
            if len(self._sessions) >= 1000:
                raise ContractError("event publisher session budget exceeded")
            self._sessions.add(session_key)
            self._attention_enabled = attention_enabled
            self._clear_content()
            self._seen.clear()
            self._decided.clear()
            self._roles.clear()
            self._inferred.clear()
            self._media_turns.clear()
            # A new session's stream clock starts again; a mode still on covers all of it.
            self._duck_spans = [[0, None]] if self._duck else []
            self._private_name_end = None
            self._private_seen.clear()
            self._role_sources = duck and priority is not None
            self._duck_acks = duck
            self._priority = priority
            self._per_turn = (
                priority is not None
                and getattr(priority, "per_turn", False) is True
                and callable(getattr(priority, "role_for", None))
            )
            self._former = former
            self._names = (
                self._per_turn
                and getattr(priority, "emit_names", False) is True
                and callable(getattr(priority, "name_for", None))
            )
            self._addressing = addressing
            self._other_agents = other_agents
            # Only a per-turn provider that reports owner-voice evidence (live speaker
            # identification roles) can ever say anything, so only it is advertised.
            self._owner_voice = (
                owner_voice
                if self._per_turn and callable(getattr(priority, "owner_voice", None))
                else None
            )
            self._owner_voice_state = None
            self._dismissal = dismissal
            # The configured stop phrases, or the defaults when roles are off.
            self._stop_rules = SpeakerPriority() if priority is None else priority.priority
            self._dismissed.clear()
            self._cooldowns.clear()
            self._conversation = conversation
            self._acknowledgement = acknowledgement
            self._pending_ack = None
            self._engaged = None
            self._dormant = None
            self._label_stale.clear()
            self._old_labels.clear()
            self._labels_renewed = False
            self._ended.clear()
            self.role_status = "off" if priority is None else "ready"
            self._degraded_turn = None
            self._session = session_id
            self._sequence = 0
            self._now = now_ms
            self._active = True
            self._terminal = False
            self._emit(
                "session",
                phase="started",
                capabilities={
                    "activation": "finalized-turn" if attention_enabled else "disabled",
                    "partials": False,
                    "speakers": "anonymous" if priority is None else "enrolled",
                    "context": True,
                },
                # Configured forms of address are advertised beside, not inside, the
                # existing capability set so strict consumers of that set are unchanged.
                **({} if addressing is None else {"addressing": addressing.to_dict()}),
                # And for guest names (hailing-station #434): a host that accepts this
                # object must accept `speaker_name` on `transcript` events, which is never
                # emitted without it.
                **(
                    {"speaker_names": {"max_scalars": MAX_SPEAKER_NAME_SCALARS}}
                    if self._names
                    else {}
                ),
                # And for owner-voice status (rightyo#164): a host that accepts this object
                # must accept `owner_voice` events, which are never emitted without it.
                **(
                    {}
                    if self._owner_voice is None
                    else {"owner_voice": self._owner_voice.to_dict()}
                ),
                # Likewise for request forming: a separate top-level object, never a key
                # inside the strictly validated capability set.
                **({} if former is None else {"request_forming": {"kind": former.kind}}),
                # And for the selected speech backends: kinds and display-safe ids only.
                **({} if speech is None else {"speech": speech}),
                # And for a host-supplied PCM stream: where the live audio came from.
                **({} if audio_input is None else {"audio_input": audio_input}),
                # And for natural dismissal (#98): a host that accepts this object must
                # accept `dismiss` events, which are never emitted without it.
                **({} if dismissal is None else {"dismissal": dismissal.to_dict()}),
                # And for conversation mode (#82): a host that accepts this object must
                # accept `conversation` events, which are never emitted without it.
                **({} if conversation is None else {"conversation": conversation.to_dict()}),
                # And for acknowledgement gating (#132): a host that accepts this object
                # plays its instant acknowledgement only for a request whose
                # `acknowledge` is true; the field is never emitted without it.
                **(
                    {}
                    if acknowledgement is None
                    else {"acknowledgement": acknowledgement.to_dict()}
                ),
                # And for privacy mode (internally duck, hailing-station #735), when the
                # producer takes the `{"duck": true|false}` control (live `listen`, not
                # replay): a host enters it only for a child that advertised version 3+, and
                # must then accept `privacy` events, which are never emitted without it.
                # `private` reports a session that starts in privacy mode.
                **(
                    {"duck": {"version": DUCK_VERSION, **({"private": True} if self._duck else {})}}
                    if duck
                    else {}
                ),
            )

    def expire(self, now_ms):
        with self._lock:
            integer(now_ms, "now_ms")
            self._now = max(self._now, now_ms)
            self._memory.expire(self._now)
            cutoff = self._now - self.retention_ms
            for key, (turn, _context, size, _role) in list(self._pending.items()):
                if turn.end_ms <= cutoff:
                    del self._pending[key]
                    self._pending_bytes -= size
            # Reclaim accounting as well as plaintext when a frozen context shrinks.
            self._pending_bytes = 0
            for key, (turn, context, _size, role) in list(self._pending.items()):
                context["turns"] = [t for t in context["turns"] if t["end_ms"] > cutoff]
                size = len(encode_json(context))
                self._pending[key] = (turn, context, size, role)
                self._pending_bytes += size
            for request_id, end_ms in list(self._open.items()):
                if end_ms <= cutoff:
                    del self._open[request_id]
            for request_id, facts in list(self._delivered.items()):
                if facts["end_ms"] <= cutoff:
                    del self._delivered[request_id]
            for key in list(self._withdrawn):
                if key not in self._pending:
                    del self._withdrawn[key]
            for key, end_ms in list(self._inferred.items()):
                if end_ms <= cutoff:
                    del self._inferred[key]
            for key, end_ms in list(self._media_turns.items()):
                if end_ms <= cutoff:
                    del self._media_turns[key]
            for key in list(self._superseded):
                if key not in self._pending:
                    del self._superseded[key]
            kept = deque()
            self._queue_bytes = 0
            for payload, _size in self._queue:
                if payload["type"] in {"transcript", "request", "attention"}:
                    end = payload.get("turn", {}).get("end_ms", payload.get("speech_end_ms"))
                    if end is not None and end <= cutoff:
                        continue
                    if payload["type"] == "request":
                        turns = [t for t in payload["context"]["turns"] if t["end_ms"] > cutoff]
                        if len(turns) != len(payload["context"]["turns"]):
                            payload["context"]["turns"] = turns
                            if "formed_request" in payload:
                                # The convenience string must not outlive the retention
                                # window either: re-render it from the pruned context,
                                # failing closed exactly as at emission. An unpruned
                                # request keeps its string byte-identical.
                                # #242 round 25: from the record it was first formed with,
                                # never the published turn, which keeps a label the forget
                                # may have made stale.
                                record, unlabelled = self._formed[payload["request_id"]]
                                # #242 round 26: a request formed before a later forget is
                                # re-rendered as one formed after it: its asker and every
                                # context turn heard before the forget unlabelled, with no
                                # owner or trusted role. Its `turn` and `context` keep theirs.
                                if record["utterance_id"] in self._old_labels:
                                    record = _unlabelled(record)
                                unlabelled = unlabelled | {
                                    t["utterance_id"]
                                    for t in turns
                                    if t["utterance_id"] in self._old_labels
                                }
                                payload["formed_request"] = self._render_request(
                                    record, turns, unlabelled
                                )
                size = len(encode_json(payload)) + 1
                # A re-rendered string may grow, so the rebuilt queue is held to the
                # same per-event, count and aggregate bounds as emission, failing closed
                # the same way rather than letting the backlog exceed its budget.
                if (
                    size > MAX_EVENT_BYTES
                    or len(kept) >= self.max_pending
                    or self._queue_bytes + size > MAX_QUEUE_BYTES
                ):
                    self._clear_content()
                    self._active = False
                    raise ContractError("speech event consumer backlog exceeded")
                kept.append((payload, size))
                self._queue_bytes += size
            self._queue = kept
            queued = {p["request_id"] for p, _size in kept if p["type"] == "request"}
            for request_id in [r for r in self._formed if r not in queued]:
                del self._formed[request_id]
            self._poll_owner_voice()

    def _poll_owner_voice(self):
        """Emit an owner-voice status change (rightyo#164), each state at most once.

        Polled with `expire`, which the session's timer and consumer call often. Reads the
        provider's current evidence without blocking: `unrecognized` first, then
        `recognized` only after it; a session whose owner bound before the threshold
        never emits either.
        """
        if self._owner_voice is None or not self._active or self._owner_voice_state == "recognized":
            return
        try:
            state = self._priority.owner_voice(self._owner_voice.threshold_s)
        except Exception:  # noqa: BLE001 - a status is optional and never fails the session
            return
        if state == ("recognized" if self._owner_voice_state else "unrecognized"):
            self._owner_voice_state = state
            self._emit("owner_voice", state=state)

    def transcript(self, turn, now_ms, *, expect_decision=True, inferred=False):
        """Emit one final turn; `inferred` marks a speaker label partly inferred (#137).

        An inferred turn (edge-attributed words or a tail join) keeps its role for
        attention and precedence, but never exercises owner or trusted authority:
        no override, no owner stop phrase, and a playback-only dismissal that withdraws
        nothing and leaves the speaker's engagement in place.
        """
        with self._lock:
            if not self._active:
                raise ContractError("speech event session is not active")
            if not isinstance(turn, Turn) or not turn.finalized or turn.session_id != self._session:
                raise ContractError("event producer requires a final turn in its active session")
            if type(expect_decision) is not bool:
                raise ContractError("invalid decision expectation")
            if type(inferred) is not bool:
                raise ContractError("invalid inferred-label flag")
            integer(now_ms, "now_ms")
            self.expire(max(now_ms, turn.end_ms))
            if self.heard_ducked(turn):
                # Privacy mode (hailing-station #735 S-1): matched locally, then dropped. No
                # transcript, no memory, no decision, no digest of the words, and no use of
                # the session's turn budget.
                # Matched only while privacy is on (round 18): a turn of an ended period
                # released while it is still off is dropped unmatched, since matching it
                # would repeat its status on every offer (its id is not kept).
                # #242 round 23: and only if heard entirely since the mode last went on. A
                # turn of an earlier period released once privacy is on again is dropped
                # unmatched too, with no id kept: no `status`, no exit, and no count in the
                # new period, which it was not heard in.
                uid = turn.utterance_id
                if (
                    self._duck
                    and self._in_current_privacy(turn)
                    and uid not in self._seen
                    and uid not in self._private_seen
                ):
                    # Kept only while privacy is on; cleared at its end (#735 S-1).
                    self._private_seen.append(uid)
                    self._private(turn, inferred)
                return
            # Any turn heard outside privacy ends a split exit.
            self._private_name_end = None
            digest = hashlib.sha256(
                json.dumps(turn.to_dict(), sort_keys=True).encode("utf-8")
            ).digest()
            previous = self._seen.get(turn.utterance_id)
            if previous is not None:
                if previous != digest:
                    raise ContractError("final event turn cannot be revised")
                return
            if len(self._seen) >= 1000:
                raise ContractError("event session turn budget exceeded")
            context = self._memory.snapshot(self._now)
            context["turns"] = [t for t in context["turns"] if t["end_ms"] <= turn.start_ms]
            # The role is fixed before the first emission and repeated unchanged on the
            # decision, the request turn and every later context snapshot.
            # With live identification the role is resolved per turn instead, from the
            # bindings at this moment, and then repeated unchanged the same way.
            role = (
                None
                if self._priority is None
                else speaker_role(self._priority.role_for(turn))
                if self._per_turn
                else self._assign_role(turn, context["turns"])
            )
            if (
                role in {"owner", "trusted"}
                and self._labels_renewed
                and not self._voiceprinted(turn)
            ):
                # A configured label after the speakers were forgotten may name anyone (#735
                # S-1, round 17): fail closed until the session restarts.
                role = "unknown"
            if inferred and role is not None:
                self._inferred[turn.utterance_id] = turn.end_ms
            if self._media:
                self._media_turns[turn.utterance_id] = turn.end_ms
            if self._retain_transcripts:
                self._memory.append(turn, role=role)
            self._seen[turn.utterance_id] = digest
            if turn.end_ms <= self._now - self.retention_ms:
                return
            if expect_decision and self._attention_enabled:
                size = len(encode_json(context))
                if len(self._pending) >= 32 or self._pending_bytes + size > MAX_PENDING_BYTES:
                    self._clear_content()
                    self._active = False
                    raise ContractError("pending attention context budget exceeded")
                self._pending[turn.utterance_id] = (turn, context, size, role)
                self._pending_bytes += size
            name = self._speaker_name(turn, role)
            source = self._role_source(turn, role, inferred)
            self._emit(
                "transcript",
                turn=_with_role(turn, role),
                # Beside the turn, never inside it, so turn fingerprints are unchanged.
                **({} if name is None else {"speaker_name": name}),
                **({} if source is None else {"role_source": source}),
            )
            if self._dismissal is not None and self._stop_rules.is_stop_phrase(turn.text):
                # The deterministic fast path (#98): an exact stop phrase dismisses at
                # once, without waiting for (or depending on) the decision model.
                plan = self._plan(
                    turn,
                    role,
                    "stop",
                    "stop-phrase",
                    None,
                    loose=False,
                    authority=self._authority_role(turn.utterance_id, role),
                )
                if plan is not None:
                    self._dismissed[turn.utterance_id] = plan["upgradable"]
                    self._emit("dismiss", **self._apply(turn, plan))
                    # Ending one's own engagement affects no one else, so even a dismissal
                    # limited to playback ends it; an inferred label's dismissal may not
                    # be the engaged speaker's own (#137), so it leaves it.
                    if turn.utterance_id not in self._inferred:
                        self._disengage(turn, "dismissed")

    @staticmethod
    def _comparable(first, second):
        """Whether two turns' speaker labels can be compared (session-stable, labelled)."""
        return (
            first["speaker_id"] is not None
            and second["speaker_id"] is not None
            and not first["overlap"]
            and not second["overlap"]
            and first["speaker_provenance"] == second["speaker_provenance"]
            and (
                first["speaker_provenance"] != "diarization-utterance"
                or first["speaker_id"].split(" ", 1)[0] == second["speaker_id"].split(" ", 1)[0]
            )
        )

    @staticmethod
    def _cooldown_key(facts):
        """The per-speaker cool-down key, or None when the label holds for one utterance.

        The key carries the speaker provenance: two sources may reuse a label, and labels
        from different provenances are never the same speaker (`_comparable`).
        """
        if (
            facts["speaker_id"] is None
            or facts["overlap"]
            or facts["speaker_provenance"] == "diarization-utterance"
        ):
            return None
        return (facts["speaker_provenance"], facts["speaker_id"])

    def _withdrawable(self, dismissing, facts, loose):
        """Whether a dismissal may withdraw a request turn: the speaker's own, recent one.

        The request must have ended at or before the dismissal started (a request that
        overlaps the dismissal is concurrent work, not withdrawn) and at most `window_ms`
        before it, from the same speaker label. When the labels cannot be compared
        (no label, overlap, or utterance-local labels from different utterances), it is
        withdrawn only when `loose`: a model-judged dismissal addressed to the system on an
        anonymous session. A different known speaker's request is never withdrawn.
        """
        dismisser = self._facts(dismissing)
        if facts["end_ms"] > dismissing.start_ms:
            return False
        if dismissing.start_ms - facts["end_ms"] > self._dismissal.window_ms:
            return False
        if self._comparable(dismisser, facts):
            return dismisser["speaker_id"] == facts["speaker_id"]
        return loose

    @staticmethod
    def _facts(turn):
        return {
            "end_ms": turn.end_ms,
            "speaker_id": turn.speaker_id,
            "speaker_provenance": turn.speaker_provenance,
            "overlap": turn.overlap,
        }

    def _authority(self, role):
        """What a speaker's dismissal may do: "full", "playback" only, or None (nothing).

        Anonymous sessions: full. Enrolled: owners and trusted speakers have full effect;
        other speakers only stop playback (and withdraw their own requests); with
        `owner_only`, only owners dismiss at all.
        """
        if self._priority is None:
            return "full"
        rules = self._priority.priority
        if role == "owner":
            return "full"
        if rules.owner_only:
            return None
        return "full" if role == "trusted" else "playback"

    def _voiceprinted(self, turn):
        """Whether `turn`'s role came from a live voiceprint binding, not a configured label."""
        resolved = getattr(self._priority, "resolved_by_voiceprint", None)
        return self._per_turn and callable(resolved) and resolved(turn.utterance_id) is True

    def _current_role(self, turn, role):
        """A turn's emitted owner or trusted role, re-checked against the binding now.

        With live identification (#137), a label whose binding no longer gives it that
        role is treated as `unknown` for policy (owner-only, follow-ups, authority and the
        owner's supersession exemption). Never blocks; never upgrades a role; the
        published role is unchanged.
        """
        if not self._per_turn or role not in {"owner", "trusted"}:
            return role
        try:
            now = self._priority.role_for(turn)
        except Exception:  # noqa: BLE001 - identification never fails the session
            now = "unknown"
        return role if now == role else "unknown"

    def _authority_role(self, key, role):
        """The role a turn exercises authority with: an inferred label never grants one.

        A turn with inferred-label words (#137) has no override, owner stop phrase,
        dismissal authority or owner supersession, whatever role it carries, and its
        dismissal withdraws nothing (see `_plan`).
        """
        if key in self._inferred and role in {"owner", "trusted"}:
            return "unknown"
        return role

    def _plan(self, turn, role, kind, reason, confidence, *, loose, authority=None):
        """What a dismissal does, without changing state; None when it does nothing.

        `role` is the turn's published role; `authority`, when given, is the role it acts
        with (see `_authority_role`).
        """
        authority = self._authority(role if authority is None else authority)
        if authority is None:
            return None
        dismisser = self._facts(turn)
        # Only a model-judged dismissal by a fully trusted speaker withdraws on
        # incomparable attribution; the fast path never does.
        loose = loose and authority == "full" and self._priority is None
        # A turn with inferred-label words (#137) may be another voice under the
        # speaker's label: it withdraws nothing, delivered or pending.
        inferred = turn.utterance_id in self._inferred
        delivered = [
            request_id
            for request_id, facts in self._delivered.items()
            if not inferred and self._withdrawable(turn, facts, loose)
        ]
        pending = [
            key
            for key, (earlier, _context, _size, _role) in self._pending.items()
            if not inferred
            and key != turn.utterance_id
            and key not in self._withdrawn
            # Its label may now name someone else (round 17): no dismissal by label reaches it.
            and key not in self._label_stale
            and self._withdrawable(turn, self._facts(earlier), loose)
        ]
        # The stop-phrase fast path from a speaker whose label cannot be compared only
        # stops playback; so does any dismissal from a speaker without full authority.
        limited = authority != "full" or (
            reason == "stop-phrase" and self._cooldown_key(dismisser) is None
        )
        scope = ["playback"] if limited else list(DISMISS_SCOPES[kind])
        fields = {
            "utterance_id": turn.utterance_id,
            "speech_end_ms": turn.end_ms,
            "speaker_id": turn.speaker_id,
            **({} if role is None else {"role": role}),
            "scope": scope,
            "withdrawn_request_ids": delivered,
            "reason": reason,
            "confidence": confidence,
        }
        if kind == "disengage" and not limited and self._dismissal.cooldown_ms:
            fields["cooldown_until_ms"] = turn.end_ms + self._dismissal.cooldown_ms
        # Only a limit from incomparable attribution can be lifted by the turn's decision.
        upgradable = limited and authority == "full"
        return {"fields": fields, "pending": pending, "limited": limited, "upgradable": upgradable}

    def _apply(self, turn, plan):
        """Withdraw what a planned dismissal covers and return its `dismiss` fields."""
        fields = plan["fields"]
        for request_id in fields["withdrawn_request_ids"]:
            del self._delivered[request_id]
            self._open.pop(request_id, None)
        for key in plan["pending"]:
            # Its decision is still to come: if it would form a request, it is withdrawn.
            self._withdrawn[key] = {
                **{k: fields[k] for k in ("utterance_id", "speech_end_ms", "speaker_id")},
                **({"role": fields["role"]} if "role" in fields else {}),
                "scope": ["pending_request"],
                "reason": fields["reason"],
                "confidence": fields["confidence"],
            }
        if "cooldown_until_ms" in fields:
            key = self._cooldown_key(self._facts(turn))
            until = fields["cooldown_until_ms"]
            current = self._cooldowns.get(key)
            if current is None or until > current[1]:
                self._cooldowns[key] = (turn.end_ms, until)
        return fields

    def _cooled(self, turn, decision):
        """Whether a cool-down after `disengage` holds this attended turn back (#98).

        A cool-down started by a speaker with a comparable label applies to that speaker
        only; one whose speaker cannot be compared applies to everyone.
        """
        if not self._cooldowns or self._dismissal is None:
            return False
        keys = [None]
        speaker = self._cooldown_key(self._facts(turn))
        if speaker is not None:
            keys.append(speaker)
        active = any(
            start <= turn.start_ms < until
            for start, until in (self._cooldowns[k] for k in keys if k in self._cooldowns)
        )
        return (
            active
            and decision.confidence < self._dismissal.cooldown_min_confidence
            and not mentions_name(self._addressing, turn.text)
        )

    def _engaged_with(self, turn):
        """Whether `turn` is the engaged speaker's, within the engagement window (#82).

        Only a session-stable, unoverlapped speaker label is ever engaged, so an
        unattributed turn or another speaker's never matches (rightyo#113).
        """
        engaged = self._engaged
        return (
            engaged is not None
            and self._cooldown_key(self._facts(turn)) == engaged["key"]
            # Decisions may arrive out of turn order: speech from before the request
            # that engaged is never a follow-up.
            and engaged["from_ms"] <= turn.start_ms < self._window_end(engaged)
        )

    def _unattributed_closing(self, turn):
        """Whether `turn` is a closing with no speaker label at all said inside the window
        while one speaker is engaged (#167). It ends that engagement: if someone else said
        it, the cost is one lost follow-up window, which fails safe. A labelled turn without
        a cooldown key (overlapping, utterance-local) names someone, so it does not count."""
        engaged = self._engaged
        return (
            engaged is not None
            and turn.speaker_id is None
            and self._closing_floor(engaged) <= turn.start_ms < self._window_end(engaged)
            and self._conversation.is_closing(turn.text)
        )

    @staticmethod
    def _closing_floor(engaged):
        """The earliest start an explicit end needs to end `engaged` (#167): the
        engagement's start or the speaker's latest request (which `_engage` records as
        `revived_over_ms`), so one decided out of order never ends a newer conversation."""
        return max(engaged["from_ms"], engaged.get("revived_over_ms", 0))

    def _ends_for(self, turn, engaged, reason):
        """Whether an explicit end said in `turn` belongs to `engaged`: the speaker's own
        labelled turn, or, for a closing only, a turn with no speaker label (#167)."""
        return self._cooldown_key(self._facts(turn)) == engaged["key"] or (
            turn.speaker_id is None and reason == "closed"
        )

    @staticmethod
    def _window_end(engaged):
        """The engagement window's end: held open while a reply plays, up to a bound."""
        replying = engaged.get("replying_since")
        if replying is None:
            return engaged["until_ms"]
        return max(engaged["until_ms"], replying + MAX_REPLY_HOLD_MS)

    def media(self, on):
        """The host's media mode (hailing-station #551), on or off; it outlasts sessions.

        While on, an attend needs this assistant's name and, with speaker roles, the
        owner's authority; every follow-up is refused. Stop phrases and dismissals are
        unchanged. With no names configured, nothing attends. Emits nothing.
        """
        if type(on) is not bool:
            raise ContractError("invalid media mode")
        with self._lock:
            self._media = on

    def duck(self, on, *, now_ms, ack=False):
        """The host's privacy mode (internally duck, v3; hailing-station #735), on or off;
        it outlasts sessions.

        `now_ms` is the stream time it changed, on the received-audio clock turns are timed
        in, and is required (#238): a default to the latest event time could drift from it. A
        turn heard in privacy (`heard_ducked`) keeps nothing (see `_private`): no
        `transcript`, attention, request, follow-up, acknowledgement or conversation event,
        no memory or context, and no hosted decision or role question. Only a local match
        of the exit and status phrases may emit a content-free `privacy` event, and an exact
        stop phrase a playback-only `dismiss`. A turn accepted before it was turned on and
        heard before it is still decided. Turning it off reports counts only.

        `ack` (#735 S-8) is true only for a host's applied `{"duck": true}` control line. Each
        such on, a redundant one while already on included, then emits one content-free
        `privacy` `on` event with `at_ms` = `now_ms`, after the mode and its span are set, so
        it follows every event queued before the stamp: the host counts one ack per on-line
        it wrote. Only in a running session that advertised `duck`. Never for off, nor for
        a session that starts private (`--start-private`), whose `started` is the proof. A
        backlog failure raises after the mode is applied (the session ends; the host's ack
        deadline fails closed).
        """
        if type(on) is not bool or type(ack) is not bool:
            raise ContractError("invalid duck mode")
        integer(now_ms, "now_ms")
        with self._lock:
            at = now_ms
            # A change of mode always ends a split exit.
            self._private_name_end = None
            if on and not self._duck:
                self._duck_spans.append([at, None])
                if len(self._duck_spans) > MAX_DUCK_SPANS:
                    first, second = self._duck_spans[0], self._duck_spans.pop(1)
                    first[1] = second[1]
            elif not on and self._duck and self._duck_spans and self._duck_spans[-1][1] is None:
                # The off edge is exclusive (see `heard_ducked`); [t, t] covers nothing.
                self._duck_spans[-1][1] = max(at, self._duck_spans[-1][0])
            if not on and self._duck:
                # Ids of turns heard in privacy are kept only while it is on (#735 S-1).
                self._private_seen.clear()
                counts, self._private_counts = (
                    self._private_counts,
                    dict.fromkeys(self._private_counts, 0),
                )
                if self._report is not None:
                    # Counts only, never words, ids or speakers.
                    with contextlib.suppress(Exception):
                        self._report(
                            "privacy off: "
                            + " ".join(f"{key}={value}" for key, value in counts.items())
                        )
            self._duck = on
            if on and ack and self._active and self._duck_acks:
                self._emit("privacy", action="on", at_ms=at)

    def _private(self, turn, inferred):
        """Match a turn heard in privacy against the local phrases, then drop it.

        Called with the lock held. Emits at most one content-free `privacy` event: "exit"
        for a named exit by the enrolled owner (see `_may_exit`), or "status" for anyone.
        Nothing else does anything: a stop phrase is a no-op, since nothing is playing or
        being recorded. A matcher failure counts and matches nothing; its exception, which
        might carry the words, never leaves here.
        """
        self._private_counts["discarded"] += 1
        try:
            kinds = privacy_phrase(turn.text, self._addressing)
        except Exception:  # noqa: BLE001 - the words must never reach an error or a log
            kinds = frozenset()
            self._private_counts["failed"] += 1
        split, self._private_name_end = self._private_name_end, None
        # Only a turn heard entirely since the mode last went on, while it is still on, is
        # matched at all (#242 round 23, see `transcript`): one heard in an earlier private
        # period and released late can never end this one, answer its status or be counted.
        if "name" in kinds:
            # Only the enrolled owner's own name fragment can begin a split exit.
            if self._may_exit(turn, inferred):
                self._private_name_end = turn.end_ms
            return
        named = "exit" in kinds or (
            "rest" in kinds
            and split is not None
            and 0 <= turn.start_ms - split <= PRIVACY_SPLIT_NAME_MS
        )
        if named:
            if self._may_exit(turn, inferred):
                self._private_counts["exit"] += 1
                self._emit("privacy", action="exit")
                return
            self._private_counts["exit_refused"] += 1
        if "status" in kinds:
            self._private_counts["status"] += 1
            self._emit("privacy", action="status")

    def _in_current_privacy(self, turn):
        """Whether `turn` was heard entirely since privacy last went on, and it is still on
        (#735 S-1). Called with the lock held."""
        return self._duck and bool(self._duck_spans) and turn.start_ms >= self._duck_spans[-1][0]

    def _may_exit(self, turn, inferred):
        """Whether `turn` may leave privacy: the enrolled owner's voice, by live voiceprint
        identification (a label configured as the owner never is, so a TV in the owner's
        diarizer slot cannot), with no inferred-label words, and not heard while the host
        reported its own playback (an echo of a clip). Never raises."""
        check = getattr(self._priority, "enrolled_owner", None)
        if inferred or not self._per_turn or not callable(check):
            return False
        if any(
            start - PLAYBACK_MARGIN_MS <= turn.end_ms
            and (end is None or turn.start_ms <= end + PLAYBACK_MARGIN_MS)
            for start, end in self._playbacks
        ):
            return False
        try:
            return check(turn) is True
        except Exception:  # noqa: BLE001 - identification never fails the session
            return False

    def heard_ducked(self, turn):
        """Whether `turn` was heard in duck mode: it is on now (a turn accepted while it is
        on counts, whenever it was said), or the turn's span overlaps a span it was on.

        A turn joined from pieces said before and after a change counts as heard in duck
        mode, so the mode's end never releases speech heard during it, held or joined. The
        on edge is inclusive (a turn ending as it went on counts) and the off edge is
        exclusive: the off is stamped at or after the end of all audio received by then,
        so a piece heard in duck mode ends at or before it and, unless it is empty, starts
        before it; speech starting at the off was heard after it. An empty piece exactly at
        the off may have ended there, so it counts (#238, fail closed). A span [t, t] (on
        and off with no audio between) covers nothing.
        """
        with self._lock:
            return self._duck or any(
                turn.end_ms >= on_ms
                and (
                    off_ms is None
                    or turn.start_ms < off_ms
                    or turn.start_ms == turn.end_ms == off_ms
                )
                for on_ms, off_ms in self._duck_spans
                if off_ms is None or off_ms > on_ms
            )

    def forget_speakers(self):
        """Drop everything held per speaker label (hailing-station #735 S-1): when privacy
        ends the diarizer starts afresh, so a label may now name someone else. The roles fixed
        per label, the engaged speaker and a lapsed engagement a reply could revive, cool-downs,
        explicit ends, appendix windows, and delivered requests a later dismissal by label could
        withdraw: none may carry to a voice that inherits the label (round 16)."""
        with self._lock:
            self._roles.clear()
            self._engaged = None
            self._dormant = None
            self._cooldowns.clear()
            self._ended.clear()
            self._appendix_from.clear()
            self._delivered.clear()
            # A turn accepted under the old labels and decided later keeps nothing by its label
            # either (round 17): its request still goes, without engagement, appendix,
            # explicit end, cool-down or a withdrawable record.
            self._label_stale = set(self._pending)
            # #242 round 25: nothing derived after the forget names a label from before it,
            # or a role only that label gave: not a formed request's context or a role
            # question's past turns (`_old_labels`), not the latest request a later decision
            # is told of, nor a pending turn's withdrawal, whose `dismiss` is still to come.
            self._old_labels.update(self._seen)
            if self.last_request is not None:
                self.last_request = (*self.last_request[:3], None)
            for key, fields in self._withdrawn.items():
                self._withdrawn[key] = _unlabelled(fields)
            if not self._labels_renewed and self._report is not None:
                # Counts and states only: which labels, never.
                with contextlib.suppress(Exception):
                    self._report(
                        "speakers forgotten: configured owner and trusted labels give no role "
                        "until the session restarts"
                    )
            self._labels_renewed = True

    @property
    def ducked(self):
        """Whether duck mode is on now (see `duck`)."""
        with self._lock:
            return self._duck

    def _media_for(self, turn):
        """Whether media mode applies to `turn`: whether it was on when the turn was accepted.

        Only that snapshot counts (Codex review of #196), so a change while a hosted decision
        is pending applies to neither direction: a turn heard during the show stays gated after
        the host turns it off, and a request made just before the owner turns it on still stands.
        """
        return turn.utterance_id in self._media_turns

    def _media_attends(self, turn, authority):
        """Whether an attend turn passes media mode's bar (see `media`)."""
        named = mentions_name(self._addressing, turn.text)
        return named and (self._priority is None or authority == "owner")

    def reply(self, phase, now_ms, asked=False):
        """The host's report that the assistant's spoken reply `started` or `ended` (#124).

        While engaged, a playing reply holds the window open, and its end restarts the
        window from that moment, so follow-ups are timed from the end of what was spoken,
        not from the request. Ignored when conversation mode is off or nothing is engaged.

        With acknowledgement gating on, a reply that starts after an acknowledgement's own
        clip has ended answers it (#122), so the next request may be acknowledged again.
        `asked` (#199): the reply ended with a question to the owner, so an answer-shaped
        follow-up ("yes", "go ahead") carries its own cue until the next reply starts.

        Emits nothing.
        """
        with self._lock:
            if phase not in {"started", "ended"} or type(asked) is not bool:
                raise ContractError("invalid reply phase")
            if asked and phase != "ended":
                raise ContractError("invalid reply phase")
            integer(now_ms, "now_ms")
            if self._active:
                self._asked_at = now_ms if asked else None
                if self._replied_at is None:
                    # The first phase reported since the request: a host may forward only
                    # a reply's end, and then it had begun by then.
                    self._replied_at = now_ms
            if self._active:
                # The assistant's own audio, so its echo is never taken for a person's answer.
                if phase == "started":
                    if self._playbacks and self._playbacks[-1][1] is None:
                        # A host may forward only the last phase of a burst: an unclosed
                        # span ends where the next begins, so it never covers all time.
                        self._playbacks[-1][1] = now_ms
                    self._playbacks.append([now_ms, None])
                elif self._playbacks and self._playbacks[-1][1] is None:
                    self._playbacks[-1][1] = now_ms
            if self._active and self._pending_ack is not None:
                self._answer_ack(phase)
            engaged = self._engaged
            if self._conversation is None:
                return
            if not self._active:
                self._reply_note(phase, "inactive", None)
                return
            # The stream clock moves with the report through the normal expiry path, so
            # anything emitted here is stamped no earlier than the time it names (haild
            # requires at_ms <= emitted_at_ms) and content past retention is reclaimed first.
            self.expire(now_ms)
            if engaged is not None and now_ms >= self._window_end(engaged):
                # Lapsed already, though no decision has said so yet: say so now.
                self._lapse(now_ms)
                engaged = None
            if engaged is None:
                engaged = self._revive(now_ms)
                if engaged is None:
                    self._reply_note(phase, "not_engaged", None)
                    return
                action = "revived"
            else:
                action = None
            if phase == "started":
                engaged.setdefault("replying_since", now_ms)
                self._reply_note(phase, action or "held", self._window_end(engaged) - now_ms)
            else:
                engaged.pop("replying_since", None)
                engaged["until_ms"] = max(
                    engaged["until_ms"], now_ms + self._conversation.window_ms
                )
                self._reply_note(phase, action or "extended", self._window_end(engaged) - now_ms)

    def _forget_dormant(self, turn, reason):
        """An explicit end by the lapsed speaker, even after the lapse, is never revived (#158).

        So does an unattributed closing (#167). Either counts only when said after the
        speaker's latest request in the lapsed engagement (`_closing_floor`): an older end
        decided late never forgets the newer conversation.
        """
        dormant = self._dormant
        if dormant is None:
            return
        engaged = dormant["engaged"]
        if self._ends_for(turn, engaged, reason) and turn.start_ms >= self._closing_floor(engaged):
            self._dormant = None
            self._mark_ended(engaged["key"], turn)

    def _end_revived(self, turn, reason):
        """An explicit end decided after a reply revived its speaker's lapsed conversation
        ends the revival (rightyo#158): the turn predates the revival, so it is no follow-up,
        but it was said to the conversation the reply revived. Emits nothing, as reviving
        emitted nothing."""
        engaged = self._engaged
        if (
            engaged is not None
            and "revived_over_ms" in engaged
            and self._ends_for(turn, engaged, reason)
            and engaged["revived_over_ms"] <= turn.start_ms < engaged["from_ms"]
        ):
            self._engaged = None
            self._mark_ended(engaged["key"], turn)
            if self._report is not None:
                try:
                    self._report(f"revival action=ended reason={reason}")
                except Exception:  # noqa: BLE001
                    pass

    def _revive(self, now_ms):
        """A spoken reply re-engages the speaker whose conversation lapsed by timeout within
        REVIVE_WINDOW_MS (rightyo#158): the assistant spoke to them again, so their answer is a
        follow-up. Only speech after the reply counts. An explicit end (closing, dismissal,
        another person) left nothing dormant. Emits nothing: engagement events name a request."""
        dormant = self._dormant
        if dormant is None or now_ms - dormant["at_ms"] > REVIVE_WINDOW_MS:
            return None
        self._dormant = None
        engaged = {
            key: value for key, value in dormant["engaged"].items() if key != "replying_since"
        }
        # Any explicit end said since the lapsed engagement began, decided only after this
        # reply, ends the revival (`_end_revived`). A revival of a revival keeps the first
        # boundary (`_lapse` carries it), or the end of a request said since (`_engage`).
        engaged.setdefault("revived_over_ms", engaged["from_ms"])
        engaged["from_ms"] = engaged["until_ms"] = now_ms
        self._engaged = engaged
        return engaged

    def _reply_note(self, phase, action, left_ms):
        """The content-free reply-report diagnostic (rightyo#158): the phase, what it did to
        the conversation window, and how long the window then had left. Best-effort."""
        if self._report is None:
            return
        left = "none" if left_ms is None else str(left_ms)
        try:
            self._report(f"reply phase={phase} action={action} window_left_ms={left}")
        except Exception:  # noqa: BLE001
            pass

    def _answer_ack(self, phase):
        """Track the pending acknowledgement through the host's playback reports (#122).

        The host reports every playback, the acknowledgement clip included, so the first
        `started` after an acknowledgement is usually that clip. A `started` clears the
        pending acknowledgement only after an `ended` has been heard: clip, then answer.
        When the reports run together (a host collapsing a burst to its last phase, or one
        continuous playback), the answer is not told apart and the window decides instead.
        """
        if phase == "ended":
            self._pending_ack["heard_end"] = True
        elif self._pending_ack["heard_end"]:
            self._pending_ack = None

    def _hold_ack(self, turn):
        """Hold an emitted acknowledgement as pending until answered or the window ends."""
        window = self._acknowledgement.dedup_window_ms
        if window > 0:
            # Only reached with nothing pending or a lapsed hold (an acknowledged turn
            # never starts inside a pending window), so this never shortens a hold.
            self._pending_ack = {"until_ms": turn.end_ms + window, "heard_end": False}

    def _lapse(self, now_ms):
        """Return to ambient once stream time passes the engagement window (#82)."""
        engaged = self._engaged
        if engaged is not None and now_ms >= self._window_end(engaged):
            self._engaged = None
            # A question belongs to its conversation (#199); a revival starts a new reply.
            self._asked_at = None
            self._replied_at = None
            self._dormant = {"engaged": engaged, "at_ms": self._window_end(engaged)}
            self._emit(
                "conversation",
                state="ambient",
                reason="timeout",
                speaker_id=engaged["speaker_id"],
                at_ms=self._window_end(engaged),
            )

    def _mark_ended(self, key, turn):
        """Record, for the speaker whose conversation `turn` ended, the end's start (#167).

        `key` is the ended engagement's, which an unattributed closing's turn lacks. Each
        speaker's boundary only moves forward. Beyond MAX_ENDED_SPEAKERS, the least recently
        ended speaker with no turn awaiting its decision is dropped: fewer turns than that
        can be pending (`_pending` holds at most 32), so one always qualifies, and a
        boundary a pending request may still need is never lost.
        """
        previous = self._ended.pop(key, None)
        self._ended[key] = turn.start_ms if previous is None else max(previous, turn.start_ms)
        if len(self._ended) > MAX_ENDED_SPEAKERS:
            waiting = {self._cooldown_key(self._facts(t)) for t, *_ in self._pending.values()}
            stale = next(k for k in self._ended if k not in waiting)
            del self._ended[stale]

    def _disengage(self, turn, reason):
        """Return the engaged speaker to ambient because of their own `turn`, or because of
        an unattributed closing inside the window (#167)."""
        # An engagement whose window ended before this turn began lapses first, so an end
        # said just after the window (the stop-phrase fast path, before any decision has
        # lapsed it) still forgets it rather than leaving it to be revived (#158).
        self._lapse(turn.start_ms)
        self._forget_dormant(turn, reason)
        unattributed = reason == "closed" and self._unattributed_closing(turn)
        if self._engaged is None or not (unattributed or self._engaged_with(turn)):
            own = self._cooldown_key(self._facts(turn))
            if own is not None:
                # Not the engaged speaker (another holds the engagement, or none does): the
                # end still bounds the speaker's own older requests (#167), as when decided.
                self._mark_ended(own, turn)
            return
        if turn.start_ms < self._closing_floor(self._engaged):
            # Said before the speaker's newer request, decided out of order after it (#167):
            # it ended the conversation that request has since moved past, not this one.
            return
        speaker_id = self._engaged["speaker_id"]
        self._asked_at = None
        self._replied_at = None
        self._mark_ended(self._engaged["key"], turn)
        self._engaged = None
        # An explicit end is never revived by a later reply.
        self._dormant = None
        self._emit(
            "conversation",
            state="ambient",
            reason=reason,
            # The engaged speaker's label, which an unattributed closing's turn lacks.
            speaker_id=speaker_id,
            utterance_id=turn.utterance_id,
            at_ms=turn.end_ms,
        )

    def _engage(self, turn, request_id, role=None):
        """Engage a request's speaker, or extend their window; one speaker at a time.

        With live identification roles (#113), an engaged owner or trusted speaker keeps
        the engagement for their window: another speaker's request is still delivered, but
        does not move the engagement to them.
        """
        key = self._cooldown_key(self._facts(turn))
        if self._conversation is None or key is None:
            return
        if turn.start_ms < self._ended.get(key, turn.start_ms):
            # Said before an explicit end of this speaker's conversation that was decided
            # first (#167): the request is still delivered, but does not re-open it.
            return
        until = turn.end_ms + self._conversation.window_ms
        if self._engaged is not None and turn.end_ms < self._engaged["from_ms"]:
            # Decided out of order: an older request never takes the engagement from a
            # newer one (or shortens it).
            return
        if self._engaged is not None and self._engaged["key"] == key:
            # Each exchange extends the window; the state itself does not change. A new
            # request outranks any older end still to be decided against a revival (#158):
            # its end becomes the boundary. While engaged, [its end, from_ms) is empty, so it
            # ends nothing; after a later lapse, a revival keeps it (`_revive`'s setdefault).
            # It only moves forward: an older request decided late never lowers it.
            self._engaged["until_ms"] = max(self._engaged["until_ms"], until)
            self._engaged["revived_over_ms"] = max(
                self._engaged.get("revived_over_ms", turn.end_ms), turn.end_ms
            )
            if self._per_turn:
                self._engaged["role"] = role
            return
        if (
            self._per_turn
            and self._engaged is not None
            and self._engaged.get("role") in {"owner", "trusted"}
            and role not in {"owner", "trusted"}
        ):
            # Enrolled precedence: the window has not lapsed (`_lapse` ran first).
            return
        self._dormant = None
        self._engaged = {
            "key": key,
            "speaker_id": turn.speaker_id,
            "from_ms": turn.end_ms,
            "until_ms": until,
        }
        if self._per_turn:
            self._engaged["role"] = role
        self._emit(
            "conversation",
            state="engaged",
            reason="request",
            speaker_id=turn.speaker_id,
            utterance_id=turn.utterance_id,
            request_id=request_id,
            at_ms=turn.end_ms,
            until_ms=until,
        )

    def _follow_up_refusal(self, turn, role, decision, evidence, unavailable):
        """Why an undecided turn is not a follow-up (a fixed token), or None when it is one.

        `not_engaged` covers conversation mode off and any turn but the engaged speaker's
        within the window; every other token is about the engaged speaker's own turn (#153).
        """
        conversation = self._conversation
        if conversation is None or not self._engaged_with(turn):
            return "not_engaged"
        if self._media_for(turn):
            # Media mode: a show's dialogue in the window is never a follow-up.
            return "media"
        if unavailable is not None:
            return "unavailable"
        if evidence.get("answered_by_other"):
            # Another person answered it within the reply window, unnamed (#199).
            return "answered"
        if evidence["label"] != "uncertain":
            return "label"
        if decision.recipient not in {"system", "unknown"}:
            return "recipient"
        if decision.probabilities["attend"] < self._follow_up_bar(role) and not self._appendix(
            turn
        ):
            return "below_bar"
        if conversation.is_closing(turn.text):
            # A closing phrase ends the conversation; it is never a request of its own.
            return "closing"
        if (
            conversation.require_cue
            and not self._cue_waived(decision)
            and not follow_up_cued(turn.text, self._addressing)
            and not self._answers_question(turn)
            and not self._continues(turn, decision)
        ):
            # Talk to a pet or a passer-by in the window carries no cue that it continues
            # the exchange with the assistant (#199). It applies to an appendix too.
            return "no_cue"
        if role is not None and self._priority.priority.owner_only and role != "owner":
            # Owner-only mode never lets another speaker's turn become a request.
            return "owner_only"
        return None

    def _cue_waived(self, decision):
        """Whether Jev's attend probability reaches the conversation's `cue_min_attend`."""
        bar = self._conversation.cue_min_attend
        return bar is not None and decision.probabilities["attend"] >= bar

    def _answers_question(self, turn):
        """Whether a turn starting after a reply that asked the owner a question answers it."""
        asked = self._asked_at
        return asked is not None and turn.start_ms >= asked and answer_shaped(turn.text)

    def _continues(self, turn, decision):
        """Whether the engaged speaker's own turn continues the exchange (#213): with
        `continuation_cue` on, it starts once the assistant began replying to the latest
        request (an acknowledgement clip's report counts), even over that reply, and Jev
        chose to attend it. The caller has already
        checked it is the engaged speaker's, inside the window; an unattributed turn or
        another speaker's never is."""
        replied = self._replied_at
        return (
            self._conversation.continuation_cue
            and replied is not None
            and turn.start_ms >= replied
            and (decision.attention_choice or decision.label) == "attend"
        )

    def _is_tag(self, turn):
        """Whether a request is a short tag on the latest one ("right?"), not its own (#199)."""
        last = self.last_request
        return (
            last is not None
            and len(turn.text.split()) <= TAG_MAX_WORDS
            and not mentions_name(self._addressing, turn.text)
            and 0 <= turn.start_ms - last[0] <= TAG_MS
            and turn.speaker_id in {None, last[3]}
        )

    def _follow_up_note(self, refusal, held, role, decision):
        """The content-free diagnostic for an engaged speaker's undecided turn (#153): the
        outcome, the reason, Jev's attend probability and the bar in force. Never text or ids.

        `formed` means the turn became a request; `held` means it passed the follow-up checks
        but a cooldown, stop, dismissal, supersession or withdrawal kept it back. Values carry
        up to six significant digits, so a near miss never prints as a tie.
        """
        attend = decision.probabilities.get("attend", 0.0)
        outcome = "refused" if refusal else "held" if held else "formed"
        return (
            f"follow_up outcome={outcome} reason={refusal or held or 'passed'}"
            f" attend={attend:.6g} bar={self._follow_up_bar(role):.6g}"
        )

    def _answered(self, turn, decision, role):
        """Whether another person's answer withholds `turn` (#199): the pipeline saw a
        different speaker start within the reply window, no playback of the assistant's own
        (a reply or acknowledgement the host reported) was running then, so it was not our
        own audio, and the speaker is not an enrolled owner or trusted speaker, whose
        requests a guest's quick word never vetoes."""
        at = decision.answered_at_ms
        if at is None or (self._per_turn and role in {"owner", "trusted"}):
            return False
        return not any(
            start - PLAYBACK_MARGIN_MS <= at and (end is None or at <= end + PLAYBACK_MARGIN_MS)
            for start, end in self._playbacks
        )

    def _appendix(self, turn):
        """Whether the engaged speaker's undecided `turn` starts within the appendix window
        after their latest direct request ended (#199) and names no other agent. Only a
        direct request opens the window, so an appendix never opens another."""
        window = self._conversation.appendix_ms
        if not window or mentions_name(self._other_agents, turn.text):
            return False
        since = self._appendix_from.get(self._cooldown_key(self._facts(turn)))
        return since is not None and 0 <= turn.start_ms - since <= window

    def _follow_up_bar(self, role):
        """The follow-up attend bar: optionally lower for enrolled owners and trusted (#113)."""
        bar = self._conversation.follow_up_min_probability
        enrolled = getattr(self._priority, "follow_up_min_probability", None)
        if self._per_turn and enrolled is not None and role in {"owner", "trusted"}:
            return min(bar, enrolled)
        return bar

    def _assign_role(self, turn, past):
        """Fix a speaker's role the first time that speaker is emitted in this session."""
        if turn.speaker_id is None:
            return "unknown"
        role = self._roles.get(turn.speaker_id)
        if role is not None:
            return role
        # #242 round 25: a past turn from before the forget carries no label or role here.
        recent = [
            _role_state(
                _unlabelled(record) if record["utterance_id"] in self._old_labels else record
            )
            for record in past[-ROLE_CONTEXT_TURNS:]
        ]
        current = _role_state(turn.to_dict())
        participants = sorted(
            {r["speaker_id"] for r in [*recent, current] if r["speaker_id"] is not None}
        )
        state = {
            "past_turns": recent,
            "current_turn": current,
            "known_participants": participants,
            # Only the retained participants' roles travel with the request: the whole
            # session's role map could exceed the payload budget, which prunes turns only.
            "roles": {s: self._roles[s] for s in participants if s in self._roles},
        }
        rules = self._priority.priority
        degraded = None
        try:
            assigned = self._priority.assign(state)
            if not isinstance(assigned, dict):
                raise ContractError("invalid speaker role assignment")
            validated = {
                identifier(speaker, "speaker_id"): speaker_role(value)
                for speaker, value in assigned.items()
            }
            if any(v == "owner" and s not in rules.owners for s, v in validated.items()):
                # Owners come only from configuration; a provider naming one is rejected.
                degraded = "rejected"
            if not set(validated) <= set(state["known_participants"]):
                # A role for a speaker not yet observed would let that speaker's first
                # real turn skip the provider; treat it as a contract violation.
                degraded = "rejected"
            # The configured overlay always wins: a configured owner or trusted speaker
            # keeps that role whatever the provider answered or omitted.
            validated = {s: rules.configured_role(s) or v for s, v in validated.items()}
            configured = rules.configured_role(turn.speaker_id)
            if configured is not None:
                validated[turn.speaker_id] = configured
        except ProviderError:
            # A hosted role question failed (timeout, unavailability, budget, cancellation).
            degraded = "unavailable"
        if degraded is not None:
            # Degrade rather than abort: this speaker takes the configured role or unknown,
            # later speakers use configured roles only, and the session keeps listening.
            self._priority = ConfiguredPriorityProvider(rules)
            self.role_status = degraded
            # The turn whose decision evidence will carry the degradation on the stream.
            self._degraded_turn = turn.utterance_id
            validated = {turn.speaker_id: rules.configured_role(turn.speaker_id) or "unknown"}
        for speaker, value in validated.items():
            # An earlier fixed role is never revised by a later answer.
            self._roles.setdefault(speaker, value)
        # A provider that omits the current speaker still fixes that speaker's role.
        return self._roles.setdefault(turn.speaker_id, "unknown")

    def decision(self, event, now_ms, *, unavailable=None):
        """Emit one turn's attention evidence and, when attended, its request.

        ``unavailable`` names why a live turn's hosted decision was transiently
        unavailable (#71). The placeholder decision must be `uncertain`, so it never
        forms a request, and its evidence carries `decision_status: "unavailable"` and
        that `reason` beside the unchanged keys.
        """
        with self._lock:
            if not self._active or not self._attention_enabled:
                return
            if not isinstance(event, DecisionEvent) or event.turn.session_id != self._session:
                raise ContractError("decision event belongs to another session")
            if unavailable is not None:
                identifier(unavailable, "unavailable reason")
                if event.decision.label != "uncertain":
                    raise ContractError("an unavailable decision must be uncertain")
            integer(now_ms, "now_ms")
            self.expire(max(now_ms, event.turn.end_ms))
            key = event.turn.utterance_id
            if key in self._decided:
                return
            pending = self._pending.pop(key, None)
            if pending is None:
                return
            turn, context, size, role = pending
            self._pending_bytes -= size
            if turn != event.turn:
                raise ContractError("decision does not match the committed transcript")
            self._decided.add(key)
            heard = turn
            stale = key in self._label_stale
            if stale:
                # Accepted before the speakers were forgotten (#735 S-1, round 17): its label may
                # now name someone else, so it is treated as unlabelled for everything keyed by
                # the label. The request itself still carries the turn as heard.
                self._label_stale.discard(key)
                turn = replace(turn, speaker_id=None)
            # Keep public evidence separate from recipient-based host target selection.
            evidence = {
                name: value
                for name, value in event.public_dict().items()
                if name in {"label", "recipient_kind", "confidence", "provider", "model"}
            }
            if unavailable is not None:
                # Optional keys, like role_status: strict hosts ignore unknown keys.
                evidence["decision_status"] = "unavailable"
                evidence["reason"] = unavailable
            if event.decision.dismissal is not None:
                # Optional keys too, present only when the dismissal question was asked.
                evidence["dismissal"] = event.decision.dismissal
                evidence["dismissal_confidence"] = event.decision.dismissal_confidence
                if event.decision.dismissal_malformed:
                    # A degraded answer, as distinct from a genuine `uncertain`.
                    evidence["dismissal_status"] = "malformed"
            stop = False
            # The published role stays as emitted (the host fingerprints it); policy uses
            # the role re-checked now (#137), and authority also drops inferred labels.
            current = self._current_role(turn, role)
            if stale and current in {"owner", "trusted"} and not self._voiceprinted(heard):
                # #242 round 23: a configured label's owner or trusted role was fixed before the
                # speakers were forgotten, and the label may now name someone else. Like a turn
                # accepted after it (round 17), it is `unknown` for policy: no override or
                # supersession, owner-only gating, owner stop phrase, follow-up or authority.
                current = "unknown"
            # #242 round 24: a stale turn's decision publishes the role policy uses, so one
            # downgraded above, or by `_current_role` for its now-unlabelled turn, is `unknown`
            # (the host keeps a `role: "owner"` decision's utterance as the owner's). It goes in
            # the attention and request `decision` alike, which must fingerprint the same
            # decision, and in its dismiss. The request's `turn` keeps the role its transcript
            # published before the forget, so it still matches that transcript's fingerprint.
            decided = current if stale else role
            authority = self._authority_role(key, current)
            if role is not None:
                evidence["role"] = decided
                if key == self._degraded_turn:
                    # The stream shows where model lookups degraded, not only the object.
                    evidence["role_status"] = self.role_status
                rules = self._priority.priority
                if rules.owner_only and current != "owner" and evidence["label"] == "attend":
                    # Owner-only mode: other speakers remain context, never a request.
                    evidence["label"] = "ignore"
                stop = authority == "owner" and rules.is_stop_phrase(turn.text)
            request_id = self._session + ":" + key
            superseded_by = self._superseded.pop(key, None)
            withdrawn_by = self._withdrawn.pop(key, None)
            # A model-judged dismissal (#98), unless the stop-phrase fast path already
            # dismissed this turn at its transcript.
            kind = None
            if (
                self._dismissal is not None
                and event.decision.dismissal in DISMISSING
                and self._dismissed.get(key, True)
            ):
                kind = event.decision.dismissal
            dismissed = kind is not None or key in self._dismissed
            to_other_agent = mentions_name(self._other_agents, turn.text) and not mentions_name(
                self._addressing, turn.text
            )
            if to_other_agent:
                # Addressed to another agent (#161), such as the owner's dot: never a request,
                # a follow-up or an acknowledgement here, and it ends this assistant's
                # conversation with them below. Naming this assistant too keeps it ours.
                evidence["label"] = "ignore"
            if (
                self._media_for(turn)
                and evidence["label"] == "attend"
                and not self._media_attends(turn, authority)
            ):
                # Media mode (hailing-station #551): TV dialogue stays context. An optional
                # key, like role_status: strict hosts ignore unknown keys.
                evidence["label"] = "ignore"
                evidence["media"] = True
            if self._answered(turn, event.decision, current):
                # Another person answered this unnamed turn at once (#199): it is theirs. An
                # optional key, like media: strict hosts ignore unknown keys.
                if evidence["label"] == "attend":
                    evidence["label"] = "uncertain"
                evidence["answered_by_other"] = True
            would_attend = evidence["label"] == "attend" and evidence["recipient_kind"] == "system"
            # False: no follow-up note; None: a formed follow-up; else the refusal token.
            follow_up_note: str | None | bool = False
            follow_up_text = None
            if self._conversation is not None:
                # The window is judged in stream time at each decided turn.
                self._lapse(turn.start_ms)
                refusal = (
                    None
                    if would_attend
                    else self._follow_up_refusal(
                        turn, current, event.decision, evidence, unavailable
                    )
                )
                if not would_attend and refusal != "not_engaged":
                    # Noted once the hold-backs below have had their say (#153).
                    follow_up_note = refusal
                if not would_attend and refusal is None:
                    # An engaged speaker's undecided follow-up (#82): the request evidence
                    # says so, beside Jev's unchanged recipient and confidence.
                    evidence["label"] = "attend"
                    evidence["follow_up"] = True
                    would_attend = True
            if (
                would_attend
                and not dismissed
                and withdrawn_by is None
                and self._cooled(turn, event.decision)
            ):
                # After "go away", an unnamed attend needs more confidence for a while.
                evidence["label"] = "uncertain"
                evidence["cooldown"] = True
                # A follow-up held back is not reported as one.
                evidence.pop("follow_up", None)
                would_attend = False
            attended = (
                would_attend
                and not stop
                and not dismissed
                and superseded_by is None
                and withdrawn_by is None
            )
            if follow_up_note is not False and self._report is not None:
                held = None
                if follow_up_note is None and not attended:
                    held = (
                        "cooldown"
                        if evidence.get("cooldown")
                        else "stop"
                        if stop
                        else "dismissed"
                        if dismissed
                        else "superseded"
                        if superseded_by is not None
                        else "withdrawn"
                    )
                # Written last, like the acknowledgement note, once the request is out.
                follow_up_text = self._follow_up_note(follow_up_note, held, current, event.decision)
            overriding = authority == "owner" and (attended or stop or dismissed)
            plan = None
            if kind is not None:
                plan = self._plan(
                    turn,
                    decided,
                    kind,
                    "decision",
                    event.decision.dismissal_confidence,
                    loose=event.decision.recipient == "system",
                    authority=authority,
                )
            # A pending turn withdrawn earlier only produces an event when it would have
            # formed a request.
            late = (
                withdrawn_by is not None
                and would_attend
                and not dismissed
                and not stop
                and superseded_by is None
            )
            # Dismiss events this decision emits, in order: a late withdrawal of this
            # turn's own request, then this turn's own dismissal.
            extra = []
            if late:
                extra.append({**withdrawn_by, "withdrawn_request_ids": [request_id]})
            if plan is not None:
                extra.append(plan["fields"])
            # Form before any reservation or emission so the burst reserve below counts
            # the exact request bytes, formed string included, and a failing former
            # fails closed before anything of this decision is queued.
            # #242 round 25: formed with the decided role and the label as decided (none for a
            # stale turn), like everything else derived at the decision; only the request's
            # `turn` and `context` keep what their transcripts published.
            formed = self._form_request(request_id, turn, decided, context) if attended else {}
            ack, ack_note = (
                self._acknowledge(turn, event.decision, evidence) if attended else ({}, None)
            )
            to_supersede = []
            if overriding:
                # Earlier is decided by turn time (end_ms at or before the owner's), not by
                # emission order: decisions arrive out of order, and a request spoken after
                # the owner's turn is not what the owner was superseding.
                to_supersede = [r for r, end in self._open.items() if end <= turn.end_ms]
                # The whole burst plus the owner's own attention/request/terminal must fit
                # the undrained queue; otherwise fail closed before emitting any override,
                # never a partial batch.
                # Plus one `conversation` event when conversation mode is on (#82).
                spare = 1 if self._conversation is not None else 0
                if len(to_supersede) + 3 + len(extra) + spare > self.max_pending - len(self._queue):
                    self._clear_content()
                    self._active = False
                    raise ContractError("speech event consumer backlog exceeded")
                # Byte accounting with the exact payloads about to be emitted, in the
                # sequence order they will receive, plus a terminal event upper bound
                # (longest identifier reason, sequence digit growth), so the burst can
                # never trip the queue byte bound part way through.
                first = self._sequence + 1
                burst = sum(
                    self._size(
                        self._payload(
                            "override",
                            first + 1 + offset,
                            {"superseded_request_id": r, "by_utterance_id": key, "role": "owner"},
                        )
                    )
                    for offset, r in enumerate(to_supersede)
                )
                reserve = self._size(
                    self._payload(
                        "attention",
                        first,
                        {
                            "utterance_id": key,
                            "speech_end_ms": turn.end_ms,
                            "decision": evidence,
                            **({"request_id": request_id} if attended else {}),
                        },
                    )
                )
                after = first + 1 + len(to_supersede)
                for offset, fields in enumerate(extra):
                    reserve += self._size(self._payload("dismiss", after + offset, fields))
                after += len(extra)
                if attended:
                    reserve += self._size(
                        self._payload(
                            "request",
                            after,
                            {
                                "request_id": request_id,
                                # #242 round 25: the exact request emitted below, whose
                                # turn is the heard one, label included.
                                "turn": _with_role(heard, role),
                                "decision": evidence,
                                "context": context,
                                "decision_at_ms": self._now,
                                **formed,
                                **ack,
                            },
                        )
                    )
                if spare:
                    after += 1
                    reserve += self._size(
                        self._payload(
                            "conversation",
                            after,
                            {
                                "state": "engaged",
                                "reason": "other_human",
                                "speaker_id": turn.speaker_id,
                                "utterance_id": key,
                                "request_id": request_id,
                                "at_ms": 2**53 - 1,
                                "until_ms": 2**53 - 1,
                            },
                        )
                    )
                reserve += 32 + self._size(
                    self._payload(
                        "session",
                        after + 1,
                        {
                            "phase": "cancelled",
                            "reason": "x" * 96,
                            "role_status": "unavailable",
                            "input_gaps": {
                                "gaps": 1,
                                "dropped_bytes": 2**53 - 1,
                                "discarded_tail_bytes": 1,
                            },
                            "skipped_segments": 2**53 - 1,
                            "skipped_utterances": 2**53 - 1,
                        },
                    )
                )
                if self._queue_bytes + burst + reserve > MAX_QUEUE_BYTES:
                    self._clear_content()
                    self._active = False
                    raise ContractError("speech event consumer backlog exceeded")
            self._emit(
                "attention",
                utterance_id=key,
                speech_end_ms=turn.end_ms,
                decision=evidence,
                **({"request_id": request_id} if attended else {}),
            )
            if superseded_by is not None and would_attend:
                # The owner superseded this turn before its decision arrived: keep the
                # evidence trail, name the request it would have carried, deliver nothing.
                self._emit(
                    "override",
                    superseded_request_id=request_id,
                    by_utterance_id=superseded_by,
                    role="owner",
                )
            if late:
                # A dismissal withdrew this turn while its decision was pending (#98).
                self._emit("dismiss", **extra[0])
            if overriding:
                # The owner's own attended turn or a stop phrase supersedes every earlier
                # open non-owner request before any new request of the owner's is
                # delivered, and every earlier non-owner turn still awaiting its decision.
                for superseded in to_supersede:
                    self._emit(
                        "override",
                        superseded_request_id=superseded,
                        by_utterance_id=key,
                        role="owner",
                    )
                    del self._open[superseded]
                    self._delivered.pop(superseded, None)
                for other, (earlier, _context, _size, other_role) in self._pending.items():
                    if (
                        # #242 round 24: a turn pending across a forget has no owner's
                        # exemption, as it has no owner authority at its decision (round
                        # 23): its label may now name someone else.
                        other in self._label_stale
                        or self._authority_role(other, self._current_role(earlier, other_role))
                        != "owner"
                    ) and earlier.end_ms <= turn.end_ms:
                        self._superseded.setdefault(other, key)
            if plan is not None:
                self._dismissed[key] = plan["upgradable"]
                self._emit("dismiss", **self._apply(turn, plan))
            if self._conversation is not None and self._engaged_with(turn):
                # A close ends engagement even if the closing turn itself was attended,
                # and never re-engages. With the timeout above (emitted before any
                # reservation), this decision emits at most one more `conversation`
                # event: a disengaging turn never also engages.
                if plan is not None:
                    if key not in self._inferred:
                        self._disengage(turn, "dismissed")
                elif self._conversation.is_closing(turn.text):
                    self._disengage(turn, "closed")
                elif to_other_agent or (
                    evidence["label"] == "ignore"
                    and (
                        event.decision.recipient == "other_human"
                        or event.decision.recipient.startswith("speaker_")
                    )
                ):
                    self._disengage(turn, "other_human")
            elif self._conversation is not None and self._unattributed_closing(turn):
                # A closing with no speaker label inside the window ends the one engagement
                # (#167); a dismissal or a turn to another person still needs the label.
                self._disengage(turn, "closed")
            elif self._conversation is not None:
                # The same explicit ends, said after the conversation lapsed (#158), whether
                # it is still dormant or a reply already revived it before this decision.
                if plan is not None:
                    reason = "dismissed"
                elif self._conversation.is_closing(turn.text):
                    reason = "closed"
                elif to_other_agent or (
                    evidence["label"] == "ignore"
                    and (
                        event.decision.recipient == "other_human"
                        or event.decision.recipient.startswith("speaker_")
                    )
                ):
                    reason = "other_human"
                else:
                    reason = None
                # As when engaged, an inferred turn's dismissal (#137) ends nothing, dormant
                # or revived: its words may not be the labelled speaker's.
                if reason is not None and (plan is None or key not in self._inferred):
                    self._forget_dormant(turn, reason)
                    self._end_revived(turn, reason)
                    own = self._cooldown_key(self._facts(turn))
                    if own is not None:
                        # The speaker's own end bounds their own older requests (#167) even
                        # when another speaker holds the engagement now: decided in order,
                        # such a request would have engaged first and then ended here.
                        self._mark_ended(own, turn)
            if attended and self._conversation is not None and self._conversation.appendix_ms:
                own = self._cooldown_key(self._facts(turn))
                if evidence.get("follow_up"):
                    # One appendix per direct request (#199): a follow-up never reopens it.
                    self._appendix_from.pop(own, None)
                elif own is not None:
                    self._appendix_from[own] = max(
                        self._appendix_from.get(own, turn.end_ms), turn.end_ms
                    )
            if attended:
                # The latest request formed, for a decision's context (#199): the heard
                # words, as past turns carry them, not a former's rewrite. A short tag on
                # it ("right?") leaves it in place. Any request answers or replaces the
                # assistant's question; the next reply answers this request.
                self._asked_at = None
                self._replied_at = None
                if not self._is_tag(turn):
                    self.last_request = (
                        turn.end_ms,
                        turn.text,
                        turn.utterance_id,
                        turn.speaker_id,
                    )
                # Any delivered request makes the next reply that request's, so no lapsed
                # conversation is revived by it (rightyo#158); a request that engages its own
                # speaker replaces the dormant one anyway.
                self._dormant = None
                self._emit(
                    "request",
                    request_id=request_id,
                    turn=_with_role(heard, role),
                    decision=evidence,
                    context=context,
                    decision_at_ms=self._now,
                    **formed,
                    **ack,
                )
                if self._dismissal is not None and not stale:
                    if len(self._delivered) >= MAX_WITHDRAWABLE:
                        # Bounded: the oldest delivered request can no longer be withdrawn.
                        del self._delivered[next(iter(self._delivered))]
                    self._delivered[request_id] = self._facts(turn)
                if role is not None and authority != "owner":
                    if len(self._open) >= self.max_open:
                        # Fail closed before an owner's override burst could overflow the
                        # queue; nothing is silently dropped.
                        self._clear_content()
                        self._active = False
                        raise ContractError("open request budget exceeded")
                    self._open[request_id] = turn.end_ms
                if self._conversation is not None and not self._conversation.is_closing(turn.text):
                    self._engage(turn, request_id, decided)
                if ack.get("acknowledge") is True:
                    self._hold_ack(turn)
                if ack_note is not None and self._report is not None:
                    # Best-effort and last: a failed diagnostic write (a closed stderr)
                    # must never undo or skip the request's state transitions above.
                    try:
                        self._report(ack_note)
                    except Exception:  # noqa: BLE001
                        pass
            if follow_up_text is not None:
                # After the request, if any, is out: a former or queue failure that ends the
                # session raises before this, so `formed` is never reported for a lost one.
                try:
                    self._report(follow_up_text)
                except Exception:  # noqa: BLE001
                    pass

    def _acknowledge(self, turn, decision, evidence):
        """The optional ``acknowledge`` field and its diagnostic note (#132).

        Both are absent unless acknowledgement gating is configured.

        A request is acknowledged when its turn uses a configured name or its attend
        confidence reaches the threshold. That confidence is the decision's own for a
        direct `attend`, and Jev's attend probability for a follow-up, whose
        `confidence` belongs to the `uncertain` choice it was promoted from. Either way it
        is not acknowledged while an earlier acknowledgement is still pending (#122).
        """
        gate = self._acknowledgement
        if gate is None:
            return {}, None
        follow_up = evidence.get("follow_up") is True
        score = decision.probabilities["attend"] if follow_up else decision.confidence
        named = mentions_name(self._addressing, turn.text)
        bar = gate.min_confidence
        if follow_up and gate.follow_up_min_confidence is not None:
            # A follow-up's own, usually stricter, bar (#199).
            bar = gate.follow_up_min_confidence
        acknowledge = named or score >= bar
        reason = "named" if named else "confident" if acknowledge else "low_confidence"
        pending = self._pending_ack
        if acknowledge and pending is not None and turn.start_ms < pending["until_ms"]:
            # One acknowledgement per spoken request (#122): a fragment or quick follow-up
            # of a request still awaiting its answer is delivered without another one.
            acknowledge = False
            reason = "pending_ack"
        # The diagnostic note is content-free: labels and numbers, never text or ids.
        note = (
            f"ack outcome={'ack' if acknowledge else 'skip'} reason={reason}"
            f" attend_confidence={score:.2f} min={bar:.2f}"
            f" follow_up={'true' if follow_up else 'false'}"
        )
        return {"acknowledge": acknowledge}, note

    def _form_request(self, request_id, turn, role, context):
        """The optional ``formed_request`` field, present only when a former is configured.

        `turn` and `role` are as decided: a stale turn's has no label and its decided role
        (#242 round 25). Context turns from before the speakers were forgotten are rendered
        unlabelled. What it was formed with is kept for a re-render (`expire`).

        A former that raises or returns an invalid value fails closed: the session's
        content is released and the error ends the session, never a silently dropped field.
        """
        if self._former is None:
            return {}
        record = _with_role(turn, role)
        unlabelled = frozenset(
            t["utterance_id"] for t in context["turns"] if t["utterance_id"] in self._old_labels
        )
        formed = self._render_request(record, context["turns"], unlabelled)
        self._formed[request_id] = (record, unlabelled)
        return {"formed_request": formed}

    def _render_request(self, record, context_turns, unlabelled):
        """Run the former on the bounded state for one request record and its context;
        context turns in `unlabelled` are rendered without their label and owner role."""
        context_turns = [
            _unlabelled(t) if t["utterance_id"] in unlabelled else t for t in context_turns
        ]
        state = {
            "current_turn": copy.deepcopy(record),
            "context_turns": copy.deepcopy(context_turns),
            "addressing": None if self._addressing is None else self._addressing.to_dict(),
            "speakers": "anonymous" if self._priority is None else "enrolled",
        }
        formed = None
        try:
            formed = formed_request_text(self._former.form(state))
        except Exception:
            pass
        if formed is None:
            self._clear_content()
            self._active = False
            raise ContractError("request forming failed")
        return formed

    def end(
        self,
        phase="cancelled",
        now_ms=0,
        reason=None,
        input_gaps=None,
        skipped_segments=None,
        skipped_utterances=None,
    ):
        with self._lock:
            if phase not in {"stopped", "cancelled", "error"}:
                raise ContractError("invalid event terminal phase")
            integer(now_ms, "now_ms")
            if skipped_segments is not None:
                integer(skipped_segments, "skipped_segments", 1)
            if skipped_utterances is not None:
                integer(skipped_utterances, "skipped_utterances", 1)
            if reason is not None:
                identifier(reason, "reason")
            if input_gaps is not None:
                if not isinstance(input_gaps, dict) or set(input_gaps) != {
                    "gaps",
                    "dropped_bytes",
                    "discarded_tail_bytes",
                }:
                    raise ContractError("invalid input gap report")
                for name, value in input_gaps.items():
                    integer(value, name)
                input_gaps = dict(input_gaps)
            if self._session is None or self._terminal:
                return
            self.expire(now_ms)
            if phase != "stopped":
                self._queue.clear()
                self._queue_bytes = 0
                # #242 round 26: no queued request is left to re-render.
                self._formed.clear()
            self._memory.clear()
            self._pending.clear()
            self._pending_bytes = 0
            self._open.clear()
            self._delivered.clear()
            self._withdrawn.clear()
            self._private_seen.clear()
            self._private_name_end = None
            self._active = False
            self._terminal = True
            self._emit(
                "session",
                phase=phase,
                **({"reason": reason} if reason else {}),
                # Only enrolled sessions report role health; anonymous output is unchanged.
                **({"role_status": self.role_status} if self._priority is not None else {}),
                # Stdin input only: audio dropped under back-pressure, never buffered unbounded.
                **({} if input_gaps is None else {"input_gaps": input_gaps}),
                # Live input only, and only when nonzero (#78): whisper.cpp segments skipped
                # and whole utterances suppressed for unusable timestamps while the session
                # kept listening.
                **({} if skipped_segments is None else {"skipped_segments": skipped_segments}),
                **(
                    {} if skipped_utterances is None else {"skipped_utterances": skipped_utterances}
                ),
            )

    def drain(self):
        with self._lock:
            result = [payload for payload, _size in self._queue]
            self._queue.clear()
            self._queue_bytes = 0
            # Delivered: never re-rendered again.
            self._formed.clear()
            return copy.deepcopy(result)
