# Private coaching wire, version 1

This separate, opt-in interface carries short owner-private suggestions. It does not
change the ordinary attended-request API or authorize commands. The shared authored
fixture is [coaching-wire-v1.jsonl](../examples/coaching-wire-v1.jsonl). It contains
trusted host controls and one synthetic producer advice item, not a live recording.

Every line is one JSON object. UUIDs are canonical lowercase strings. Unknown fields,
unsupported versions, duplicate JSON keys, nonfinite numbers and incorrect types are
rejected. Control and transcript channels are separate: text that looks like a control
remains text. The embedding process must supply controls from its trusted application
boundary, never through speech or model output.

## Host controls

All controls have `schema_version: 1` and `type: "coaching_control"`.

| Action | Additional required fields |
| --- | --- |
| `enable` | `session_id`, `destination_id`, `listener_id`, `participant_awareness: true` |
| `qualify` | `session_id`, `destination_id`, `generation_id`, `qualification_id`, `valid_for_ms` |
| `invalidate` | `session_id`, `generation_id`, `reason` |
| `disable` | `session_id` |

Enable selects an existing `owner_private` listener profile but grants no capture
permission until the host qualifies private output. Qualification lasts 1–2,000 ms
from receipt on RightyO's local monotonic clock. The host retains its own original
observation deadline and checks it independently; clocks need not share an epoch.
Refresh at most one second apart from actual private-headphone observations. A device
name, microphone selection or Bluetooth transport alone does not prove private output.

Each qualification has a fresh host-issued UUID. Renewing a generation preserves
context, but never extends the deadline of work carrying an earlier qualification.
Any route notification invalidates the generation before rebuilding it. Requalification
after invalidation uses a new generation UUID. Retired generations cannot be reused in
the same coordinator instance. Allowed invalidation reasons are `route_changed`,
`route_lost`, `participant_awareness_lost`, `owner_cancelled`, `capture_stopped` and
`host_invalidated`.

Profile controls are the separate version-1 `listener_profile` add/update/remove
contract. Profiles are limited to eight, 4 KiB per profile and 16 KiB collectively
(compact ASCII JSON); each description is at most 1,024 Unicode characters. Any
successful registry mutation disables coaching and clears owned context, including
remove/re-add and update/revert between ticks. Re-enable with a fresh session UUID.

## Producer advice

`coaching_advice` has exactly these fields: `schema_version`, `type`, `session_id`,
`generation_id`, `destination_id`, `qualification_id`, `request_id`, `listener_id`,
`delivery`, `valid_for_ms`, `text`, `decision`. Version is 1 and delivery is
`owner_private`. `decision` has exactly `label: "speak_now"`, `confidence` (finite
0–1), `provider` and `model`. Provider/model/listener identifiers are bounded ASCII.
Identifiers are 1–96 characters matching `[A-Za-z0-9_. -]`. Advice text is printable
valid Unicode, nonempty after trimming, at most 280 characters and 45 whitespace words.
The complete compact ASCII JSON line is at most 8,192 bytes.

`request_id` is producer-issued, once per work item. The host verifies an issued
qualification, then deduplicates producer request IDs in a bounded ledger. It must not
require a request ID secretly preallocated only by the host. A fixture may pre-register
its known ID to test the same matching and cancellation paths.

`valid_for_ms` is remaining producer lifetime at delivery, rounded down, never an
extension of the host's original qualification deadline. Before queuing, scheduling,
resuming or unmuting, the host verifies the complete session/generation/destination/
qualification tuple, original deadline, explicit mode, participant awareness and
current private output. Unknown qualifications and stale, duplicate or expired advice
are discarded. Hailing Station owns playback, capture cancellation, routing and action
authorization. Speech, profiles, speaker labels and suggestions grant no action authority.

RightyO admits one physical provider call and one queued advice item. Context is at
most 60 seconds, 32 finalized turns and 12 KiB of compact ASCII JSON. Cadence is at
least 20 seconds (a profile can increase it), and an enabled session lasts at most
15 minutes. The host supplies measured pause and owner/playback activity separately;
RightyO does not guess private routes or acoustic timing from transcript text.

## State and stopping

`coaching_state` has exactly `schema_version`, `type`, `session_id` (UUID or null),
`state` (`disabled`, `enabled`, `qualified`, `suspended`), `capture_permitted` and
`reason`. Capture permission is advisory state for the host adapter, not a microphone
start command. State reasons are content-free allowlisted tokens. Local `tick()` must
run while idle to detect expired qualification/session/context.

Disable, cancellation, invalidation, expiry and active-profile changes discard owned
plaintext context and queued advice and cancel pending work. A single uncooperative
provider may retain its bounded work snapshot until it returns; cancellation is not
forensic zeroization or a promise to erase data already handed to another process.
Providers must cooperate and impose their own finite transport timeout. No retry or
new physical call is launched until the prior call finishes. There is no transcript
archive or recording default.
