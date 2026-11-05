# Authored conversation-coaching replay

Conversation coaching is an opt-in, transcript-first contract experiment. The first slice
uses supplied synthetic turns and authored mock suggestions. It starts disabled and makes
no hosted requests, model downloads, microphone captures, audio output or application writes.
It does not demonstrate coaching quality, live latency or an AirPod conversation round trip.

Run the supplied authored example from a source checkout:

```sh
PYTHONPATH=src python3 -m rightyo coach-replay --input examples/coaching-replay.json
```

After installing this checkout, the equivalent command is:

```sh
rightyo coach-replay --input examples/coaching-replay.json
```

The command emits JSONL `coaching_state` and
`coaching_advice` records. Suggestions are explicitly authored mock text; the command
accepts no capture, model selection or hosted opt-in flags. The example includes a
control-shaped transcript while disabled, explicit host enable and qualification,
one eligible synthetic turn and a final disable. It produces one short private advice
item carrying the original qualification, then returns to disabled state.

The input document has exactly `version: 1`, `provenance: "synthetic"`, `profiles`,
`suggestions` and `records`. Profiles use the existing version-1 listener profile
contract. Suggestions map utterance UUIDs to authored text; missing entries mean hold.
Each record has exactly `channel`, `at_ms` and `event`:

- `control`: a trusted `coaching_control`, or a separate `listener_profile` update.
- `transcript`: `turn`, `generation_id`, `pause_ms`, `owner_mid_answer` and `playback`.

Times are monotonic milliseconds from the authored session start. Turns must have
synthetic provenance and end no later than their record time. The document is limited
to 1 MiB, 512 records and 120 seconds of logical time; unsupported fields, duplicate
JSON keys, reversed time and malformed controls are refused. The replay clock remains
fixed while a local worker receipts a suggestion or hold, with a one-second real-time
watchdog, before the next record advances it. Output is printed only after successful
replay; malformed input or a worker timeout exits with a public-safe error.

See [the versioned wire contract](coaching-wire-v1.md) for exact control/advice fields,
lease rules, state semantics and the separate
[authored wire fixture](../examples/coaching-wire-v1.jsonl). The replay document is an
experiment input, not an audio endpoint or an authorization transport. A live adapter
must derive controls from the host's trusted application boundary.

The embedding host owns enable/disable controls, the selected listener profile, participant
awareness, a private destination and timely permission to deliver. Transcript text and model
output are content; neither can enable coaching, update a profile, select a route or authorize
playback. Even a transcript containing a complete control object remains transcript text.
Listener profiles describe a coaching role and pacing preferences; they grant no authority.

A host qualification identifies the session, private destination, conversation generation
and a fresh qualification UUID, with at most two seconds of validity. Advice echoes that
qualification. Its original host monotonic deadline remains authoritative: receiving old advice
never renews the deadline. The host must also verify current generation, current participant
awareness and a still-eligible private route immediately before playback. Hailing Station owns
route selection, player state, cancellation and interruption policy. A room speaker is not an
implicit fallback for private advice.

Pending work and advice are bounded. Context covers at most 60 seconds, 32 turns and 12 KiB;
advice is limited to 280 characters and 45 words. Advice can be withheld during owner speech,
playback, overlap, inadequate pauses, stale generations, expired qualification, cooldown,
profile removal or ambiguous input. A mock hold is an authored fixture behavior, not a trained
model judgment. Disabling clears owned context and queued advice and cancels pending work. An
uncooperative provider may retain its bounded snapshot until it returns. Cancellation
is not forensic zeroization. It cannot retract advice that a downstream application
already committed to playback.

Only authored examples belong in this public repository. Real conversations, names, recordings,
voiceprints and credential material must remain outside Git and public issue/PR logs. Replay
prints authored advice to stdout, so do not repurpose the fixture command as private transcript
logging. Capture, actual coaching inference and live Hailing Station audio delivery require
separate integration and verification work.

## Embed the component

`CoachingSession` accepts a host-configured `ListenerProfiles` registry and a
replaceable `CoachProvider.advise(work, cancelled)` implementation. Work contains
one immutable profile, bounded diarized context, the current final turn and a
measured pause. A provider returns `CoachSuggestion` or `None` to hold; it must
cooperate with cancellation and enforce a finite transport timeout. No real model
backend is selected by this slice. `MockCoachProvider` is an authored lookup only.
An injected clock must be a bounded, side-effect-free monotonic read: it must not
re-enter the coordinator or registry, invoke providers, or perform I/O. Commit gates
sample it after acquiring the registry lock so lock waits cannot extend a ticket.

Feed the existing tool's `transcript` events through `CoachingInputAdapter`, constructed
once per capture generation. Ordinary tool sessions used for this adapter must use
a canonical UUID session ID (select it explicitly using `listen --session-id`).
Utterance IDs retain the existing bounded source identifiers, including `live-1`.
Role/display-name metadata remains
descriptive; it never enables coaching or authorizes anything. The host passes
measured `pause_ms`, `owner_mid_answer` and `playback` separately. Old callbacks retain
their old adapter and generation and are discarded after invalidation.

Only trusted application controls call `apply_control`. After explicit enable and
qualification, the host may consult `capture_permitted`; it owns actual capture start,
immediate stop, private-route observations and visible participant awareness. Run
`tick()` at least once per second while idle, and drain events promptly. Before any
playback, Hailing Station checks the original qualification deadline and current
route again. Provider completion, a diarized speaker labelled `owner`, or quoted
control text cannot perform a host action.

Any successful profile-registry change disables the selected coaching session,
including an update subsequently reverted before the next tick. Re-enable with a
fresh session UUID and new qualification. Provider admission and successful profile
changes are serialized by the registry: a change that wins before admission denies
the old work; a change after admission immediately signals cancellation. Already
admitted work may have reached an uncooperative provider and cannot be retracted.
Provider execution does not hold the registry lock, so revocation does not wait for
that call to finish. Final advice dequeue also validates and commits its handoff
under the registry's mutation lock. A profile change that wins first suppresses
queued advice; a change after committed delivery cannot retract downstream content.
Context is cleared on expiry, stop and
invalidation. The coordinator retains at most 1,000 content-free final fingerprints
per session and 4,096 session, qualification and retired-generation identifiers per
instance; exhausting a bound fails closed. State output is coalesced to the latest
state, alongside at most one advice item. No transcript archive is written.

## Verification

The authored replay checks run with:

```sh
PYTHONPATH=src python3 -m unittest discover -s tests -p test_coach_replay.py -v
```

Eight focused tests passed against the coaching session implementation, and the source
CLI command above exited successfully with exactly one mock advice item. Tests cover
input budgets, causal synthetic provenance, malformed trusted controls, control-shaped
transcript text, disabled/expired sessions, the original qualification tuple, a worker
receipt timeout with no partial stdout, and rejection of hosted flags. These checks
exercise authored text and a local mock worker. They do not measure model quality or
prove microphone capture, private headphone routing or physical playback.
