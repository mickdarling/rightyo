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
| `request` | `request_id`, `turn`, `decision`, `context`, `decision_at_ms` | Complete attended input available for host handling |
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

## Turn and request fields

`turn` uses the existing validated `Turn` contract: `session_id`, `utterance_id`,
`revision`, `start_ms`, `end_ms`, `text`, nullable `speaker_id`, `finalized`,
`overlap`, `recognizer_id`, `provenance`, and `speaker_provenance`. Current emitted
turns are final. Labels are anonymous within a session, not verified identities;
overlap and unknown speakers retain their explicit meaning. Provenance distinguishes
`synthetic`, `recorded-file`, `causal-replay`, and `live-microphone`.

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
The default history is five minutes, with 1,000 unique turns and 1 MiB of retained
transcript data. There are additional independent bounds: at most 32 frozen pending
contexts totalling 1 MiB, 128 queued events totalling 4 MiB, and 1,200,000 bytes per
event. Exceeding a delivery bound fails closed; it does not silently drop an attended
request. Drain continuously and start a new bounded session when necessary.

[The authored shared fixture](../examples/tool-events.jsonl) demonstrates ordinary
discussion, ignored attention, an attended retrospective request with prior context,
and normal termination. Producer tests generate and compare this exact fixture;
the Hailing Station consumer uses the same authored contract fixture.

## Speaker roles and owner override

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
stays open until an override, retention expiry or a terminal event; every open request
within the session's existing turn budget is tracked, none is silently dropped. An
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
as usual, and no further model questions are asked in that session; the prototype
reports this as `role_status: "unavailable"`. A provider answer that names an
unconfigured speaker as `owner` is rejected the same way (`role_status: "rejected"`):
owners come only from configuration. Because decisions can arrive out of order, an
owner override supersedes only turns earlier than the owner's turn, compared by turn
time (a turn whose `end_ms` is at or before the owner turn's `end_ms`), never by
emission order; a later non-owner request that happened to be decided first stays open.

[The enrolled fixture](../examples/enrolled-override.jsonl) shows a participant's
attended request followed by the owner's "Ignore that." override; the shared anonymous
fixture above is byte-identical to before.

## Python extension boundaries

`rightyo.tool_events.SpeechEvents` provides `start`, `transcript`, `decision`,
`expire`, `drain`, and `end`. `ContractError` indicates an invalid event or exceeded
bound. Stop the source on overflow. Drain before normal termination; cancellation
releases pending plaintext immediately. Returned event dictionaries are detached
from internal state. This API produces input events; it performs no application
actions.

`LiveProcessor(LiveConfig(...), on_turn)` accepts explicit PCM16 mono 16 kHz through
`push_pcm16`, followed by `finish` or cancellation/`close`. Native paths are explicit,
separately provisioned trusted assets. The controller connects its finalized turns
to the replaceable `DecisionProvider` and event publisher. Alternative backends can
produce the same validated `Turn`/`DecisionEvent` values without changing consumers.

The current native implementation still uses conservative endpointing and completed
Whisper windows. Persistent ASR, early attention, lower endpoint latency and other
SDK adapters remain measured follow-ups, not requirements for this first tool.

Refs [#46](https://github.com/mickdarling/rightyo/issues/46),
[#43](https://github.com/mickdarling/rightyo/issues/43),
[#40](https://github.com/mickdarling/rightyo/issues/40).
