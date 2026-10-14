# Compiled coaching goals and active operations

Part of [#178](https://github.com/mickdarling/rightyo/issues/178), extending
[#69](https://github.com/mickdarling/rightyo/issues/69) and
[#70](https://github.com/mickdarling/rightyo/issues/70). Host integration relates to
[Hailing Station #266](https://github.com/mickdarling/hailing-station/issues/266) and
[#268](https://github.com/mickdarling/hailing-station/issues/268).

`coaching_operations.py` implements a pure schema and host-owned registry. It does
not recognize voice commands, call an LLM, export transcripts, integrate a decision
provider or change the Hailing UI. Current Jev coaching selects finite authored
catalog entries; it does not generate advice or compile freeform goals.

The intended flow is an authorized spoken operator request, an explicitly selected
LLM compiler, validation, and a host activation against the same session/revision
observed before compilation. Ordinary conversation and compiler/advice output remain
untrusted data. A field describing an operator or a well-formed control object does
not authenticate anyone. The host must authorize control separately; this module is
not attached to a transcript or transport boundary.

## Version 1 compiler output

The exact object contains `schema_version: 1`, `categories` and `operations`.
Categories have `category_id` and `description`. Operations have `operation_id`,
`category_id`, `goal`, `speak_when`, `stay_quiet_when` and `delivery` (exactly
`owner_private`). These are descriptive strings for replaceable decision adapters,
not tool calls, prompt authority or executable actions. No provider name, endpoint,
credential, transcript or advice text is required by this contract.

For example, this is an authored goal, not real conversation or model output:

```json
{
  "schema_version": 1,
  "categories": [
    {"category_id": "clarity", "description": "Explain the main point clearly"}
  ],
  "operations": [
    {
      "operation_id": "brief",
      "category_id": "clarity",
      "goal": "Help me explain the point briefly",
      "speak_when": "A shorter explanation would help",
      "stay_quiet_when": "The owner is speaking",
      "delivery": "owner_private"
    }
  ]
}
```

Each goal has 1–8 categories and 1–16 operations. Every category must be referenced;
every operation reference must exist. Identities are unique, 1–64 ASCII letters,
digits, dots, underscores or hyphens, starting with a letter/digit. Each description
has 1–512 characters, nonblank, valid UTF-8, without control characters. The compiler
response and canonical ASCII JSON representation are bounded at 16 KiB; non-ASCII
expansion counts toward that budget. Unknown/missing fields, duplicate JSON keys,
unsupported versions, nonfinite JSON and malformed/deep output fail with content-free
errors. No truncation or repair is performed. The caller must also bound transport
retention before calling `parse_compiled_goal`; this module performs no transport.

## Explicit lifecycle and visible state

Create `ActiveCoachingOperations` with a fresh host session identifier. It starts
empty at revision 0. Its immutable snapshot and detached `to_dict()` export expose
the same session, revision, categories and operation definitions to future list,
UI and model consumers. This is an in-memory export, not a persistence layer.

The exact host control has `schema_version: 1`,
`type: "coaching_operations_control"`, `session_id`, `expected_revision` and
`action`. `add` also requires `compiled_goal`; `remove` requires `operation_id`;
`list` has no additional fields. A rejected or stale call changes nothing. A
successful add/remove advances revision once, without rollback or reuse. Listing
checks the expected revision without advancing it. Session identifiers must be
fresh across host restarts; reuse would defeat session replay separation.

An add atomically adds the complete validated goal. Equal categories may be shared;
conflicting category definitions or duplicate operation identities are rejected.
The total active set retains the same count and 16 KiB state-wire budgets. Removing
an operation removes any category no longer used. No eviction occurs to fit a
budget. A compiler response returned after another edit cannot activate against
its old expected revision, even if the user removes and re-adds equal operations.

## Revocation and future consumers

`admit_dispatch` commits at most eight work admissions against a current nonempty
revision. Successful edits immediately set all their cancellation events. Revoked
workers keep their admission slots until actual retirement calls `close`; repeated
edits cannot evade the cap. A provider may retain already admitted data if it ignores
cancellation, so admission is a handoff, not a promise to retract disclosed data.

`delivery_guard` serializes a bounded in-memory result handoff with edits and rejects
old revisions or cancellation. Provider, I/O, playback and callbacks never belong
inside that guard. Future consumers must independently check original deadlines,
host/session/route authority and cancellation after lock waits and immediately before
provider disclosure. This module does not replace any existing deadline or consent
guard, and is not yet connected to `CoachingSession` or its providers.

The remaining product work is spoken list/add/remove, an actual explicit generative
compiler, the approved TypeSafe transcript/context export boundary, replaceable
realtime decisions, and visible current/pending/rejected/revoked state in the same
Hailing UI. Ambiguous or rejected compilation preserves the prior set and returns a
content-free `ContractError`; displaying that feedback remains host integration.
Listing uses stored state, not model regeneration. Two distinct authored adapter
fixtures (choice mapping and prose instruction mapping) consume the same schema in
tests; they are not real providers. Product integration requires separate
qualification. Authored pure JSON and controlled thread tests establish schema/lifecycle
behavior only; they do not establish recognition, model usefulness, coaching quality
or physical privacy.
