# RightyO local input API, version 1

RightyO accepts explicitly started speech input and emits transcription, anonymous
speaker labels, attention evidence and attended requests. The host owns responses,
acknowledgements, target selection and authorized actions. The browser lab is a
client of the shared controller, not a required component of the headless tool.

See [the quickstart](tool-quickstart.md) for installation, the secretless authored
replay and the explicitly started local Mac listener. This is a local developer
interface; it does not claim whisper.cpp binary/API compatibility, validated
conversational accuracy, or early activation before a complete transcript turn.

## Transport and lifecycle

`rightyo tool-replay` and `rightyo listen` emit one UTF-8 JSON object per stdout
line. Stdout contains speaker-labelled text by design; pipe it to an explicitly
selected local consumer rather than a log. Diagnostics use stderr and omit input
content. No remote listener or network audio transport is added.

Every event has `schema_version: 1`, `type`, `session_id`, a positive `sequence`,
and nonnegative `emitted_at_ms`. Sequence increases within one session; cancellation
and expiry can discard queued events, so gaps are permitted. All times are stream
relative in milliseconds. Live emission time follows the controller's session
clock; authored replay uses the supplied transcript timeline and is not a measured
live-latency experiment. A new session identity is required after termination.

The events are:

| Type | Additional fields | Meaning |
| --- | --- | --- |
| `session` | `phase`, initial `capabilities`, optional terminal `reason` | `started`, `stopped`, `cancelled`, or `error` |
| `transcript` | `turn` | One immutable finalized transcript turn |
| `attention` | `utterance_id`, `speech_end_ms`, `decision`, optional `request_id` | `attend`, `ignore`, or `uncertain` evidence |
| `request` | `request_id`, `turn`, `decision`, `context`, `decision_at_ms`, optional `formed_request` | Complete attended input available for host handling |
| `override` | `superseded_request_id`, `by_utterance_id`, `role` | An owner's turn supersedes an earlier open non-owner request |

Initial capabilities declare `activation: "finalized-turn"` when decisions are
available, or `"disabled"` for transcription-only listening, `partials: false`,
`speakers: "anonymous"` (or `"enrolled"` when [speaker roles](#speaker-roles-and-owner-override)
are configured), and `context: true`. Mock replay decisions are identified
by `provider: "mock"` and `model: "mock-v1"`; they are fixture rules. A local
listener without Jev opt-in emits no invented attention decisions.

When the host configures forms of address, the `started` event also carries a
separate top-level `addressing` object, for example
`"addressing": {"names": ["Hailing Station", "Station", "computer"]}`. The names
are supplied at runtime through repeatable `--name` flags on `listen`,
`tool-replay` and `prototype`, or through the prototype configuration's
`addressing.names`; the command line takes precedence over the file. Each name is
one to 48 ASCII letters, digits, single spaces, dots, underscores or hyphens,
starting with an ASCII letter or digit, not ending in a dot or hyphen, with at
most eight distinct names; non-ASCII names are rejected. The names
are passed to the decision provider as evidence that speech using one of them is
addressed to the system; a name alone is neither required nor a transcript
filter, and the complete turn is still judged from context. The mock fixture rule
accepts a configured name followed by a comma or colon, case-insensitively, and
keeps its authored `Rightyo,` prefix when nothing is configured. Without
configuration the field is absent and the existing `capabilities` set is
unchanged; no name is built into the tool.

A transcript precedes its decision. For an accepted system-addressed request,
attention precedes request delivery and both reference the same `request_id`.
Decisions may arrive after later transcripts. Receivers correlate identities rather
than assuming adjacent transcript/decision pairs. The complete triggering turn is
retained, including any wake phrase; no request-span extraction is implied.

A normal `stopped` follows already accepted delivery. `cancelled` and `error`
discard queued content and pending requests. No further request is valid after a
terminal event. Hosted unavailability and budget exhaustion terminate the headless
hosted-decision session conservatively; the interactive lab can separately continue
local transcription. If a pipe closes, the producer cancels. Consumers treat EOF
without a normal terminal event as incomplete and never automatically retry an
uncertain application action.

A session has no default duration ceiling; it listens until the host stops it or a
terminal condition occurs. `--session-budget SECONDS` on `listen` and `prototype`, or
`session_budget_seconds` in the prototype configuration, ends a session with the ordinary
`cancelled` event, the same as a host-initiated stop. In microphone mode the budget is
wall clock from start, including diarizer start-up, so somewhat less audio than the budget
is processed; in replay/demo mode it is the exact audio boundary, however slowly the replay
processes. The budget is a positive whole number of seconds; the command line
replaces the file's value. Timestamps are plain integers in stream milliseconds and do not
wrap. Memory is bounded independently of session length by the rolling retention limits
below, the per-utterance audio window, and the per-session unique-turn count.

## Turn and request fields

`turn` uses the existing validated `Turn` contract: `session_id`, `utterance_id`,
`revision`, `start_ms`, `end_ms`, `text`, nullable `speaker_id`, `finalized`,
`overlap`, `recognizer_id`, `provenance`, and `speaker_provenance`. Current emitted
turns are final. Labels are anonymous within a session, not verified identities; they
are numbered like spreadsheet columns (`Speaker A`..`Speaker Z`, then `Speaker AA`,
`Speaker AB`, ...), so a backend reporting more than 26 voices still yields valid labels;
overlap and unknown speakers retain their explicit meaning. Provenance distinguishes
`synthetic`, `recorded-file`, `causal-replay`, and `live-microphone`.
`speaker_provenance` says how far a label reaches: `diarization-timeline` labels come
from one session-long speaker timeline (the native stream), so the same label across
turns is the same anonymous voice for the session; `diarization-utterance` labels
(the hosted per-request diarizer) are stable only within one utterance, and a label in
one utterance does not identify the same voice in another. Such labels are namespaced
by utterance (`u7 Speaker A`, `u8 Speaker A`), so they never compare equal across
utterances and a decision provider sees each utterance's voices as distinct
participants; `authored-fixture` and `unknown` keep their meanings. None is an identity.

Decision evidence contains `label`, `recipient_kind`, `confidence`, `provider`, and
`model`. A probability is not demonstrated accuracy or authority. A request is
emitted only for `attend` with recipient `system`. No target platform/session or
executable action is supplied. The host still validates the full envelope and
applies its own submission policy.

`context.turns` is chronological prior speaker-labelled conversation, frozen when
the request transcript was accepted. Only prior turns ending at or before the
request's start are included; later-arriving turns never enter the snapshot.
Ignored and uncertain discussion can be useful prior context for retrospective
requests. Expiry also removes stale context while awaiting a decision.

`context.retention` describes the memory snapshot at acceptance, including its
cutoff, capacity/expiry counts and whole-turn boundary policy. It is not a claim
that every retained turn is included: overlap filtering or subsequent expiry can
reduce the handoff. Compare the supplied turns themselves for actual coverage.
Two per-session counts end a native session with a distinct error and require a new
session identity: 1,000 unique finalized turns, and the native diarizer's whole-session
speaker timeline of at most 18,000 segments, which is returned in full at every utterance
and so is reached by very long sessions with frequent speaker changes (a windowed timeline
is tracked in [#54](https://github.com/mickdarling/rightyo/issues/54)).
The default history is five minutes, with 1,000 unique turns and 1 MiB of retained
transcript data. There are additional independent bounds: at most 32 frozen pending
contexts totalling 1 MiB, 128 queued events (configurable from 5 to 128) totalling
4 MiB, and 1,200,000 bytes per event. Exceeding a delivery bound fails closed; it does not silently drop an attended
request. Drain continuously and start a new bounded session when necessary.

[The authored shared fixture](../examples/tool-events.jsonl) demonstrates ordinary
discussion, ignored attention, an attended retrospective request with prior context,
and normal termination. Producer tests generate and compare this exact fixture;
the Hailing Station consumer uses the same authored contract fixture.

## Speaker roles and owner override

Configured roles name session-stable labels, so they require a diarizer whose labels
persist for the session (the native stream, `diarization-timeline`). A diarizer with
utterance-local labels (`diarization-utterance`, namespaced as `u7 Speaker A`) can never
match a configured `Speaker A`, and `owner_only` would then silence every request, so
`listen` and the lab refuse to start a session that combines configured owners, trusted
speakers or `owner_only` with such a diarizer.

Without configuration the tool behaves exactly as above: `speakers` is `"anonymous"`
and no `role` field exists. When speaker roles are configured, through the prototype
configuration's `speakers` object or the `--owner`, `--trusted`, `--owner-only` and
`--role-source` options of `tool-replay`, the started event declares
`speakers: "enrolled"` and every turn object (in `transcript`, the `request` turn and
`context.turns`) and every decision object carries `role`, one of the literals `owner`,
`trusted`, `participant` or `unknown`. A role is fixed the first time a speaker is
emitted in a session and never changes afterwards; a turn without a speaker label is
`unknown`. Roles are descriptive data from configuration or a model answer. They are
not authentication, they do not verify who is speaking, and they never unlock anything
on the host: the host's own policy and confirmation flow decide what any request may do.

Owners and trusted speakers are configured by session speaker label or enrolled
identifier. With `"source": "configured"` (the default) every other speaker is a
`participant`. With `"source": "model"` the opted-in Jev provider is asked once per
newly observed unconfigured speaker, from the bounded recent conversation, whether that
speaker is `trusted`, `participant` or `unknown`; low confidence is `unknown`. The model
is never offered `owner`, and configured roles replace its answer, so an owner cannot be
downgraded by a transcript that claims otherwise. Each role question shares the hosted
request budget and consent; without hosted opt-in only the configured roles apply.

Precedence is applied by the producer before emission. An owner's attended turn, or an
owner turn that is only a stop phrase (default `stop`, `cancel`, `ignore that`, `never
mind`; matched case-insensitively against the whole utterance after punctuation is
removed), emits one `override` event per earlier non-owner request that is still open:
`superseded_request_id` names the request, `by_utterance_id` names the owner's turn and
`role` is `owner`. Overrides follow the owner's `attention` event and precede the owner's
own `request`; a stop phrase produces no request even when the decision was `attend`, and
its attention evidence is emitted unchanged without a `request_id`. A non-owner request
stays open until an override, retention expiry or a terminal event. Open requests are
bounded to the event queue capacity minus four (124 by default) so that one owner
decision's burst of overrides always fits the queue; a request that would exceed the
bound fails closed with an error, never a silent drop. An
owner override also supersedes every earlier non-owner turn whose decision is still
pending: when that late decision arrives, its `attention` evidence is emitted without a
`request_id`, an `override` names the request id the turn would have carried, and no
`request` follows. Hosts therefore treat an `override` whose `superseded_request_id`
they never received as already handled. A stop phrase with nothing open or pending
emits no override. With
`owner_only: true`, a non-owner `attend` decision is emitted as `ignore` and no request
is delivered; non-owner turns remain ordinary context. The producer adds no free-text
markers: interpreting non-owner context as information rather than instructions is the
host's responsibility, informed by the `role` on every context turn.

A host that accepts `speakers: "enrolled"` must also accept the `override` event; it is
part of the enrolled contract. An `override` is never emitted on an anonymous session.
An owner stop phrase supersedes open requests regardless of the decision label on the
owner's turn (`attend`, `ignore` or `uncertain`): the configured phrase, not the
attention decision, is the signal, as chosen in
[#50](https://github.com/mickdarling/rightyo/issues/50). If a hosted role question
fails (timeout, unavailability, budget exhaustion or cancellation), the session keeps
listening: that speaker receives the configured role if any, otherwise `unknown`, fixed
as usual, and no further model questions are asked in that session; the producer
records this as `role_status: "unavailable"`. A provider answer that names an
unconfigured speaker as `owner` is rejected the same way (`role_status: "rejected"`):
owners come only from configuration. Model-sourced roles are available to `tool-replay`
only; `listen` and the lab refuse `"source": "model"` before capture starts (see
[the lab notes](prototype.md)). The degradation is also visible on the stream: when a
priority provider is configured, the terminal `session` event carries an optional
`role_status` of `ready`, `unavailable` or `rejected`, and the decision evidence of the
turn where degradation occurred carries the same key, so a consumer can distinguish a
disabled model lookup from a legitimate `unknown` or `participant`. The key is absent on
anonymous sessions, and hosts tolerate it as an unknown optional key elsewhere. Because
decisions can arrive out of order, an
owner override supersedes only turns earlier than the owner's turn, compared by turn
time (a turn whose `end_ms` is at or before the owner turn's `end_ms`), never by
emission order; a later non-owner request that happened to be decided first stays open.

[The enrolled fixture](../examples/enrolled-override.jsonl) shows a participant's
attended request followed by the owner's "Ignore that." override; the shared anonymous
fixture above is byte-identical to before.

## Request forming

Request forming is off by default. When it is off, nothing above changes: no `request`
event carries a `formed_request` key and the `started` event carries no `request_forming`
object, so the shared anonymous and enrolled fixtures are byte-identical to before. It is
selected through the prototype configuration's `request_former` object, for example
`"request_former": {"kind": "template"}`, or through `--request-former template` on
`listen` and `tool-replay`; the command line replaces the file's value, like `--name`.
Only the `kind` key is accepted and `template` is the only kind; an unknown kind or an extra
key is rejected before any session starts.

A live session (`listen` or the lab) also advertises its selected speech backends on the
`started` event as a separate top-level `speech` object beside the capability set, for
example `"speech": {"transcriber": {"kind": "whisper.cpp", "id":
"whisper.cpp-live-window"}, "diarizer": {"kind": "nemotron.cpp", "id": "nemotron.cpp
v3-streaming"}}`. The transcriber `id` is the `recognizer_id` its turns carry; the
diarizer `id` names the backend and, for the hosted diarizer, the configured model and
diarizer version (`hosted-deepgram <model> <diarize_model> <hash>`), so exported sessions
record which diarizer labelled them without changing the `Turn` contract. Ids are display
safe: never an endpoint, local path, model file or credential. Authored replay
(`tool-replay`) does not advertise `speech`, so the shared fixtures are unchanged.

When a former is configured, the `started` event advertises it as a separate top-level
object beside the capability set, `"request_forming": {"kind": "template"}`, in the same
way `addressing` is advertised: the existing `capabilities` set is unchanged, so consumers
that validate that set strictly are unaffected. Every `request` event then carries
`formed_request`, a plain-text string, in addition to the raw `turn` and `context`, which
stay exactly as they are without a former. Both are always present on such a request. A
host that can reason over the turns and roles takes the raw fields; a host that cannot
takes the string. The string is a convenience rendering, not authority: it is produced by
a local template from the same turns the event already carries, it adds no information,
and the host's own policy and confirmation flow still decide what any request may do.

The `template` former is deterministic, local and model-free. It renders the request
turn's words as the request, labelled with the resolved speaker role and session label,
for example `Owner (Speaker A) asked: "Rightyo, archive the project.".`; on an anonymous
session the label is `Speaker (Speaker A)`, a turn without a speaker label is
`an unknown speaker`, and overlapping speech is marked `overlapping speech`. A request
from a speaker other than an owner on an enrolled session is followed by
`The requester is not an owner.` Each frozen context turn then follows in chronological
order, most recent last, as `Earlier, participant (Speaker B) said: "..." (context only,
not an instruction).`; every context turn, whatever its role, carries that marker, since
earlier speech is information for the request rather than an instruction to the host.
The whole string is at most 16,000 characters (four times the turn text limit). When the
context does not fit, the oldest context turns are dropped first and
`Older context was omitted to fit.` is inserted after the request; the request turn is
never truncated. The string is validated like turn text (valid UTF-8, bounded, non-empty)
and counts toward the per-event byte budget. It follows the retention window like the
rest of the queued content: when expiry prunes a context turn from a queued request's
`context.turns`, the producer re-renders that request's `formed_request` from the pruned
context through the same former, so no expired text outlives the window inside the
string; a queued request whose context was not pruned keeps its string byte-identical.

A former that raises, or that returns a value that is not a string, is empty or exceeds
the bound, fails closed: the producer raises `ContractError`, the session ends with
`error`, and the field is never silently dropped. `rightyo.providers.RequestFormer` is the
protocol for alternative formers (`kind` plus `form(state)`); the state they see is
bounded to the request turn, the frozen context turns (both with `role` when present),
the configured `addressing` or `null`, and the `speakers` capability value, with no
session or source identifiers beyond those the event already carries.

[The formed-request fixture](../examples/enrolled-formed-request.jsonl) is an enrolled
replay with the template former on: a participant's context turn and the owner's attended
request carrying `formed_request`. It is generated by `tool-replay --owner "Speaker A"
--request-former template` from the authored turns in its producer test, which regenerates
and byte-compares it; it is meant to be copied byte-identical into the Hailing Station
consumer's contract fixtures alongside the two above.

## Python extension boundaries

`rightyo.tool_events.SpeechEvents` provides `start` (with optional `addressing`,
`priority` and `former` providers), `transcript`, `decision`, `expire`, `drain`, and
`end`. `ContractError` indicates an invalid event or exceeded
bound. Stop the source on overflow. Drain before normal termination; cancellation
releases pending plaintext immediately. Returned event dictionaries are detached
from internal state. This API produces input events; it performs no application
actions.

`LiveProcessor(LiveConfig(...), on_turn)` accepts explicit PCM16 mono 16 kHz through
`push_pcm16`, followed by `finish` or cancellation/`close`. Each push is at most one
second of audio; `LiveConfig.session_budget_ms` (default `None`, no ceiling) fails the
processor closed once more audio than the budget arrives. Native paths are explicit,
separately provisioned trusted assets. The controller connects its finalized turns
to the replaceable `DecisionProvider` and event publisher. Alternative backends can
produce the same validated `Turn`/`DecisionEvent` values without changing consumers.

Speech recognition and speaker labelling are replaceable at the same kind of boundary.
`rightyo.providers.Transcriber` turns one finalized utterance of PCM16 mono 16 kHz bytes
into utterance-relative `{"text", "start_ms", "end_ms"}` units and names the
`recognizer_id` of emitted turns; `rightyo.providers.Diarizer` receives every pushed
frame (`push`) and returns a stream-relative `{"speaker", "start_ms", "end_ms"}` timeline
(`segments`, `finish`, `close`). `LiveConfig.transcriber` and `LiveConfig.diarizer` take
an instance or a factory called with the config; `None` keeps the local defaults,
`WhisperCppTranscriber` and `NemotronCppDiarizer` in `rightyo.live_audio`, whose
behaviour is unchanged. The `transcriber`/`diarizer` sections of the prototype
configuration select an implementation by name, described in
[the prototype document](prototype.md#speech-backends); `rightyo.speech_backends`
also provides the hosted implementations and the factories that section maps to.
Word-to-speaker attribution, grouping and the `Turn` contract are the same for every
backend. Hosted backends refuse to send audio without explicit consent (`--allow-hosted`),
and no hosted backend has been accuracy-tested here. Every Deepgram request carries
`mip_opt_out=true`, which per Deepgram's documentation excludes it from the Model
Improvement Program (participation is otherwise the default) and gives it zero data
retention after the response. The OpenAI-compatible transcription request has no
request-level training or retention control in the cited schema: the operator must
configure data-use and retention controls on the provider account, and confirm the
provider's policy, before pointing this backend at it. Turns from the hosted transcriber
carry `recognizer_id` `hosted-openai-compatible <model> <hash>`: the configured model name
with characters outside the identifier charset replaced by `-`, followed by the first 12
hex digits of the SHA-256 of the original name so sanitized or truncated names cannot
collide, the whole within 96 characters; transcripts from different models therefore stay
distinguishable in exported events, and the endpoint never appears.
`DeepgramDiarizer.diarizer_id` names that backend the same way
(`hosted-deepgram <model> <diarize_model> <hash>`) while `speaker_provenance` keeps its
allowlisted value. The hosted transcriber ignores the `register` cancellation hook: the
`cancelled` guard is checked before each request, during a login Keychain credential
lookup and every 50 ms while the response body is read, and the credential lookup plus
the whole exchange share one wall-clock deadline of `timeout_seconds` (default 30, at
most 120) from the start of the request. A Keychain lookup that is cancelled or outlives
the deadline is terminated; the transport is given only the budget that lookup left, as
its inactivity timeout for connect, TLS, headers and body, so a pause shorter than the
remaining budget is tolerated. The open phase (name resolution, connect, TLS, upload and
response headers) and the body read each run on a helper thread; a stop or the deadline
is noticed within about 50 ms, the connection is shut, and the caller returns at once. A
name lookup cannot be interrupted, so an abandoned open thread ends when the resolver
returns, without holding up the stop.

The current native implementation still uses conservative endpointing and completed
Whisper windows. Persistent ASR, early attention, lower endpoint latency and other
SDK adapters remain measured follow-ups, not requirements for this first tool.

Refs [#46](https://github.com/mickdarling/rightyo/issues/46),
[#43](https://github.com/mickdarling/rightyo/issues/43),
[#40](https://github.com/mickdarling/rightyo/issues/40).
