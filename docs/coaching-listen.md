# Continuous private coaching tool

Refs #69 and #70. This opt-in foreground mode uses the existing `listen`
`PrototypeController`, `LiveProcessor` and `SpeechEvents` pipeline. The website is a
test rig; the embedding application owns controls, capture and private playback.
This implementation qualifies software using authored ASR and advice. It does not
establish real conversation quality, enrolled owner identity or private device safety.

## Process boundary

```sh
PYTHONPATH=src python3 -m rightyo listen --config /absolute/path/synthetic.json \
  --mode stdin --provenance synthetic --session-id CANONICAL_UUID \
  --coaching --control-fd 3 --coaching-mock-suggestions /absolute/path/suggestions.json
```

Stdin is the existing headerless mono 16 kHz signed little-endian PCM16 stream.
Stdout carries ordinary version-1 session/final transcript events and unchanged
[coaching state/advice v1](coaching-wire-v1.md). Attention and requests are disabled
in coaching mode; suggestions never enter application action dispatch. Without
`--coaching`, existing reply-started/ended fd controls and ordinary behavior stay intact.
`--coaching` requires stdin, an inherited descriptor above 2, and a canonical nonzero
session UUID. The host must explicitly select this capability before spawning it.
Ordinary dismissal, conversation, request-forming and acknowledgement configuration
is disabled in this mode. A spoken stop phrase remains transcript evidence; trusted
host controls own coaching revocation.

The child initially emits `coaching_state` disabled/initial. This signals the control
loop, not ASR model readiness. No controller or PCM consumer starts until the host
adds a private listener, enables the specified session with participant awareness,
and qualifies a current private output destination. Startup has a 10-second deadline.
Each child owns exactly one session and capture generation. Invalidation, disable,
profile mutation after enable, or qualification loss retires it. A new enable uses a
new child/source/session/generation; queued old PCM is never relabelled.
An invalidation for another generation is ignored, preserving the current lease and
any admitted work's original deadline; a matching invalidation retires the child.

## Trusted fd controls

One ASCII/UTF-8 JSON object per newline, at most 8,192 bytes excluding newline, with
at most 4,096 records per child. Duplicate keys, nonfinite numbers, malformed JSON,
blank lines, unknown fields/types and oversized input stop the child. The fd is a
trusted capability supplied by the embedding host; neither speech nor model output
may write it. EOF/read failure also stops capture and revokes advice.

Use the existing exact profile envelopes:

```json
{"version":1,"type":"listener_profile","action":"add","profile":{"version":1,"listener_id":"coach","name":"Coach","domain":"Authored conversation coaching","speak_when":"A helpful short observation at a measured pause","stay_quiet_when":"The owner is answering or playback is active","delivery":"owner_private","priority":0,"cooldown_ms":0}}
```

`update` has the same fields; `remove` has only `version`, `type`, `action`,
`listener_id`. Existing profile count/description/byte limits apply. Follow with
the existing exact `coaching_control` enable/qualify envelopes in the wire document.
Renew qualified observations at most one second apart with fresh qualification IDs.
Renewal never extends an already admitted work item's original deadline. RightyO
ticks every 50 ms while idle; the host independently checks its original route and
qualification deadlines before capture forwarding, rendering and playback.

Only after receiving a real finalized `transcript`, the host may supply its measured
activity on the trusted fd:

```json
{"schema_version":1,"type":"coaching_activity","session_id":"11111111-1111-4111-8111-111111111111","generation_id":"33333333-3333-4333-8333-333333333333","utterance_id":"live-1","pause_ms":500,"owner_mid_answer":false,"playback":false}
```

These are the exact fields. Session/generation are canonical UUIDs; utterance IDs use
the existing bounded Turn identifier grammar (1–96 ASCII `[A-Za-z0-9_. -]`). Pause
is an integer 0–60,000 ms; flags are booleans. Match the actual final ID, child scope,
and current qualification. Receipt must be strictly less than 2,000 ms after the
final. One final may await activity; speculative, duplicate, stale, unknown or
mismatched measurements are rejected. A new final immediately cancels old advice.
Missing measurements preserve a conservative context turn with no permission to
speak. Speaker names/roles and quoted controls remain untrusted transcript content.
The host measures owner activity/playback and silence; RightyO never derives them
from text or anonymous speaker labels.

## Authored qualification backend

An executable synthetic config needs no model/native paths:

```json
{"transcriber":{"kind":"synthetic-authored","utterances":["Could we clarify the next step?","I think we should wait for an answer."]},"diarizer":{"kind":"synthetic-authored"}}
```

The pair is required together, only with stdin and synthetic provenance. Optional
native end-of-turn/speaker-identification settings and hosted decisions are refused.
The script has 1–32 printable nonempty texts, at most 1,024 characters each and 16 KiB
compact ASCII JSON collectively. Each actually finalized PCM utterance consumes one
text; silence consumes none. Exhaustion fails closed. ASR/diarization identifiers
explicitly say synthetic; the single `Speaker A` is an authored stand-in. This
exercises PCM framing/endpointing/finalization and the existing transcript adapter;
it is not speech recognition accuracy or speaker discrimination evidence.

The optional suggestion file is an authored JSON object keyed by actual final ID,
for example `{"live-1":"Ask what outcome they need next."}`. It is at most 64 KiB,
uses the existing MockCoachProvider count/text limits, and defaults to hold when
absent or unmatched. Mock remains the default provider.

## Optional Jev catalog selector

The separate opt-in `--coaching-provider jev` selects one locally authored suggestion
or holds/abstains. It uses the existing pinned `jev-1.13.0` model and fixed
`https://api.typesafe.ai/v1/systemone` endpoint through the documented
[Choice API](https://docs.typesafe.ai/api). It does not generate advice text. A strict
parser verifies model, question, distribution, confidence and bounded usage; only a
catalog entry can become advice. Transcript/profile content stays untrusted evidence.

In addition to the normal coaching process/control requirements, Jev requires all
three explicit options: `--coaching-allow-hosted-text`, `--coaching-max-requests N`
(1–100 attempts per child), and `--coaching-advice-catalog /absolute/path/catalog.json`.
The catalog is a JSON object such as
`{"clarify":"Ask which outcome matters most.","pause":"Give them time to finish."}`.
It contains 1–8 distinct ASCII IDs (1–64 letters/digits/underscore/dot/hyphen, starting
with a letter or digit); `hold` and `uncertain` are reserved. Each authored text meets
the existing 280-character/45-word limit. File and encoded catalog each fit 8 KiB.
Duplicate keys, malformed text and nonfinite JSON fail closed. Mock suggestion files
cannot be combined with Jev. Jev-only options are rejected in mock/ordinary listening.

This consent sends the bounded coaching transcript and listener profile to Jev.
Ordinary `--allow-hosted` does not supply that consent; it separately authorizes any
configured hosted speech backend. Ordinary Jev attention decisions must remain off
in coaching mode. Selecting the catalog provider does not perform credential lookup,
inference or transmission; those occur only for actual admitted, still-qualified work.
The existing local Jev credential mechanism is used at request time, without a new
service or credential configuration. Do not put credentials in these files or argv.

Each admitted work item carries its immutable original monotonic qualification
deadline internally, never in model input or wire events. Credential lookup, TLS/DNS,
send/read and parsing share the remaining original lifetime, capped at 1.5 seconds;
renewal cannot extend it. Cancellation is checked during lookup/transport and again
before accepting a suggestion; the coordinator independently checks ticket/profile,
route and deadline before delivery. Failures consume an attempt and are not retried.
Responses are capped at 64 KiB and abandoned transport helpers are process-bounded.
The coaching credential wrapper also stops waiting at the original deadline even
when the existing Keychain subprocess is still terminating. Such bounded cleanup
helpers carry no work/profile/transcript/request bytes, discard late keys, and
cannot resume the abandoned transport. Ordinary credential behavior is unchanged.
Data already transmitted cannot be retracted by later cancellation.

All provider qualification uses authored text and fake credential/HTTPS transport,
including actual credential-loader cleanup through fake subprocesses and late-DNS/body
cases. This establishes the software path;
hosted model usefulness/latency and real speech/private hardware quality remain
unvalidated. The source/test slice does not authorize hosted experiments.

The original cadence of at least 20 seconds and all context/session/qualification
limits still apply. Multi-turn authored tests can demonstrate holds and cancellation
without lowering those limits.

The upstream controller and ordinary speech publisher disable their usual transcript
history in this mode. Coaching owns the sole historical context (60 seconds, 32 turns,
12 KiB); the pending final and bounded transport queues are transient delivery buffers.
Ordinary listening retains its existing context behavior.

## Shutdown and bounded transport

Output is nonblocking and bounded to 256 KiB; a consumer making no progress for
250 ms ends the child. Advice lifetime is reduced before its first write using the
original producer deadline, including time spent waiting in the output queue.
Unsent advice is discarded on newer evidence or revocation; a partially handed-off
line cannot be retracted, so the child retires and the host must reject stale data.
Host route/activity checks apply again before mock or real rendering/playback.

SIGTERM/disable/invalidation cancel advice and stop the speech worker before final
output or diagnostics. Stderr diagnostics are nonblocking and best effort. Control loss,
malformed input, speech failure, output failure or expiry fail closed with content-free
errors. PCM EOF finalizes only the still-qualified source, then allows at most two
seconds for activity/provider receipt; cancelled sources discard open speech. The
existing controller owns stopping its audio worker. No source is restarted within
the same child, and no transcript/audio retention is introduced.

After cancellation the child drains and sends exactly one ordinary terminal speech
`session` event, including stdin `input_gaps` diagnostics, before final coaching state
when output remains available. A terminal already produced by a speech error is
preserved. EOF without a complete terminal line remains incomplete at the host;
broken or stalled output cannot establish clean completion.
