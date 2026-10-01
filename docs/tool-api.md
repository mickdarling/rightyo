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

Initial capabilities declare `activation: "finalized-turn"` when decisions are
available, or `"disabled"` for transcription-only listening, `partials: false`,
`speakers: "anonymous"`, and `context: true`. Mock replay decisions are identified
by `provider: "mock"` and `model: "mock-v1"`; they are fixture rules. A local
listener without Jev opt-in emits no invented attention decisions.

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
