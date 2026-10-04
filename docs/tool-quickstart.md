# Local speech tool

RightyO supplies speech input: transcript, anonymous speaker labels, attention decisions,
and complete attended turns with bounded preceding context. The website is a test/demo
rig over the same controller. Replies, prerecorded acknowledgement voices, agent routing,
and application actions belong to the host. This is an experimental finalized-turn tool,
not a trained streaming attention model or a Whisper-compatible drop-in library.

## First developer experiment

From a clone of this repository, using Python 3.11 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/rightyo tool-replay --input examples/synthetic-turns.json --provider mock
```

This supplies authored text and fixture rules, with no capture, models, credentials, or
hosted inference. Partial revisions and duplicate turns are validated but only committed
final turns produce events. Each JSONL line is an independently parseable versioned event.
The transcript remains untrusted input, including when an attention decision accepts it.

The command writes plaintext transcript events intentionally. A host should consume the
pipe in memory; redirecting stdout to a file creates a transcript archive owned by that
host. Keep private conversation exports outside this public checkout.

## Explicit local audio

Follow [the native setup guide](prototype.md) to provision existing local Whisper, Nemotron,
and capture assets. This package does not download or bundle them. The configuration is
the same absolute-path `local/prototype.json` used by the lab. Then choose one source:

```sh
# Explicit foreground microphone capture; may request macOS microphone permission.
.venv/bin/rightyo listen --config local/prototype.json --mode microphone \
  --session-id local-session-001

# Process a supplied authored WAV in causal order, faster than wall clock.
.venv/bin/rightyo listen --config local/prototype.json --mode demo \
  --session-id demo-session-001

# A local process pipes raw PCM in and declares where it came from.
some-local-pcm-source | .venv/bin/rightyo listen --config local/prototype.json \
  --mode stdin --provenance causal-replay --session-id stdin-session-001
```

`--mode stdin` is a local adapter only. It adds no network transport, authentication, or
source authorization, and the provenance it reports is whatever the local caller declares.
Hailing Station's ambient-audio path, which will pipe a phone's microphone into this
command, is in progress (hailing-station #203) and is not yet available. In that setup the
Hailing Station daemon owns transport and authentication, runs this command as a local
child process, and declares `--provenance live-microphone` for the audio it relays.

`--mode stdin` reads headerless mono 16,000 Hz signed 16-bit little-endian PCM (the format
the Mac capture helper produces and `push_pcm16` accepts) from stdin. Nothing else is
accepted or detected. Reads of any size are regrouped into 200 ms chunks, and an odd byte
is carried to the next read. EOF finishes the open utterance and ends with the ordinary
`stopped` event. stdout carries only JSONL; diagnostics go to stderr. The bytes do not
say where they came from, so `--provenance` is required with `--mode stdin` and refused
otherwise. There is no default. Pass `live-microphone` only for a person speaking live
(for example a phone microphone relayed by the Hailing Station daemon), and
`causal-replay`, `recorded-file` or `synthetic` for anything else. Turns carry exactly that value. Timing follows it too.
`live-microphone` uses the wall clock for the session budget and retention, as microphone
mode does. The replay values use media time, as demo mode does, so a slow producer is never
cut off early. The started session
event adds a top-level `audio_input` object
(`{"source": "stdin", "encoding": "s16le", "sample_rate": 16000, "channels": 1}`) beside
the unchanged capability set. At most 32 seconds of audio is queued. If processing falls
further behind, the input overruns and the session fails closed. RightyO stops reading,
drops the unqueued audio, and processes the audio already queued. Turns finalized before
the drop stand. The open utterance is clipped mid-speech, so it is discarded and never
emitted. The session then ends with a `session` `error` event with reason
`input-overrun` and exits non-zero, even if it is stopped or sent SIGTERM while finishing.
No audio after the drop reaches speech recognition or the diarizer, so speech is never
spliced across a gap and speaker labels stay consistent for the session.
A notice goes to stderr. The terminal session event of every stdin session carries
`input_gaps` (`gaps`, 0 or 1; `dropped_bytes`; `discarded_tail_bytes`).

Repeatable `--name` flags (for example `--name "Hailing Station" --name computer`) declare
the forms of address the system answers to for `listen`, `tool-replay` and `prototype`. They
are advertised in the started session event and given to the decision provider as evidence,
not as a required wake word; see [the API reference](tool-api.md). Nothing is configured by
default. Sessions have no default duration ceiling; `--session-budget SECONDS` on `listen`
and `prototype` ends a session after that many seconds with the ordinary `cancelled` event.
`--request-former template` on `listen` and `tool-replay` adds a `formed_request` string to
each request beside the unchanged raw turns; it is off by default (see
[request forming](tool-api.md#request-forming)).

No attention is fabricated in these commands: local-only audio emits transcript and session
events. Optional Jev inference requires both `--use-jev --allow-hosted`, or the equivalent
`decision` section in the configuration file (below), with the existing
[secure credential setup](mvp.md#secure-jev-setup-on-macos). The native audio tool remains
macOS-only; authored transcript replay uses standard Python on other platforms. Hosted
replay likewise requires `--provider jev --allow-hosted`; `--max-requests` bounds that run.

### Jev decisions from the configuration file

A host that runs `listen` with a fixed argument list, such as the Hailing Station daemon,
cannot add `--use-jev --allow-hosted`. The owner can give the same consent in the
configuration file that the host passes with `--config`:

```json
"decision": {"provider": "jev", "allow_hosted": true}
```

This is equivalent to `--use-jev --allow-hosted` for `listen`: finalized transcript text
and bounded context go to Jev, and attention/request events are emitted. Both keys are
required and exactly typed. An optional `"max_requests"`, a positive whole number, caps the
Jev requests of each session, ending it with `attention-budget-exhausted` when reached; it
is accepted only with `"provider": "jev"`. Without it, live microphone and stdin sessions
have no request cap, since Jev is called once per finalized turn for as long as the session
listens, and a demo keeps its bounded default of 20. `provider` is `"jev"` or `"mock"` (the default when the section
is absent); `allow_hosted` is a JSON boolean. `"provider": "jev"` without
`"allow_hosted": true`, a missing or extra key, another provider name, a non-boolean
value or an invalid `max_requests` is refused with a message naming the rule before any capture starts. `"mock"` with
`"allow_hosted": true` is refused, as `--allow-hosted` alone is.

The flags keep their rules. `--use-jev` without `--allow-hosted` is still refused before the
file is read, and the file never supplies the missing half. Both flags select Jev even when
the file names `"mock"`; neither source can switch the other's Jev selection off. With the
file selecting Jev, a redundant `--allow-hosted` is accepted. The section consents to Jev
decisions only: hosted speech backends still require `--allow-hosted` on the command line.
The `prototype` lab ignores the section and keeps its per-session page choice.

The credential lookup is unchanged: `TYPESAFE_API_KEY` from the process environment when
set, otherwise the login Keychain item with service `rightyo.jev` and account `api-key`.
The key is read for each Jev request, not cached. When a process the daemon started reads
the Keychain item, macOS shows its access prompt for `rightyo.jev` on the Mac. **Allow**
permits that one read, so the prompt returns on the next request; **Always Allow** stops
the prompts, but it grants the shared `/usr/bin/security` executable persistent access to
the item (see the caution in [the credential setup](mvp.md#secure-jev-setup-on-macos)).
Decide which before an unattended run: if nobody answers, the lookup times out and the
hosted decision fails, ending the session with an error event rather than silently falling
back to the mock provider. A credential failure is not treated as transient; see
[hosted unavailability](tool-api.md#hosted-unavailability) for the failures that degrade
one turn instead.

The foreground consumer refreshes the controller lease and drains events every 50 ms;
there is no HTTP server or browser requirement. Ctrl-C or SIGTERM cancels capture and
clears pending content. End of a demo emits normal stopped status after its final decisions.
Errors and consumer backlog cancel the session rather than quietly dropping attended
requests. Restart with a fresh session identifier; a host may supply `--session-id`, or
the command generates one. No listening or hosted inference starts on import.

## Version 1 event stream

The formal Python and JSONL boundary is documented in [the API reference](tool-api.md).

Every event has `schema_version: 1`, `type`, `session_id`, an increasing `sequence`, and
stream-relative `emitted_at_ms`. Sequence gaps can occur when retention discards queued
content; they do not imply transport reconnection or license a retry. Only one foreground
session runs per command. These are local process events, not an authenticated remote
audio API or a promise of reconnect/replay durability.

| Type | Meaning |
| --- | --- |
| `session` | Started capabilities, or terminal `stopped`, `cancelled`, or `error` phase. |
| `transcript` | A committed final `turn`, including text, speaker, timing, and provenance. |
| `attention` | A decision for one utterance; includes a request ID only for accepted system attention. |
| `request` | Complete accepted `turn`, decision evidence, and frozen bounded preceding `context`. |

Only `attend` with recipient kind `system` produces a `request`. Ignore and uncertain
speech remain eligible preceding context for a later attended request. The decision's
`provider` and `model` identify fixture/mock evidence or the configured provider.
Speaker IDs are anonymous session labels, not authentication or named-person recognition.
Hosts must deduplicate `request_id`, reject stale sessions and malformed events, and handle
cancellation. They must not treat model attention as permission for privileged operations.

Context is frozen when the triggering transcript arrives, before later turns arrive or
hosted inference completes. The opening words of that entire finalized turn are preserved;
request-span extraction across multiple turns is not implemented. Future speech appears
as later transcript events, and a host can associate it with its own interaction state.
The default history retains five minutes with 1,000 turns / 1 MiB caps. Pending decisions
and event output are independently bounded; expiration and cancellation release owned
plaintext references, without claiming forensic memory zeroization.

## Delivery and validation

The first adapter target is Hailing Station's explicit local Mac input path; its mobile
on-device speech recognition remains unchanged. A consumer should process `request`
events rather than scrape the lab or infer activation from transcript strings. A remote
transport, mobile input, general assistant framework, and response generation are outside
this tool contract.

Run the repository's [CI-equivalent checks](ci.md) before contributing. Tests use authored
turns and fake capture/models/providers. Native speaker/ASR smoke results remain in
[the prototype guide](prototype.md); whole-file throughput does not establish live response
latency, false-activation rates, or everyday conversation quality. The existing 1.44-second
endpoint silence wait and fresh Whisper process per utterance remain measured optimization
work rather than silently revised behavior in this delivery.

Tracked scope: [tool delivery #46](https://github.com/mickdarling/rightyo/issues/46),
[activation interface #43](https://github.com/mickdarling/rightyo/issues/43), and
[host handoff #40](https://github.com/mickdarling/rightyo/issues/40).
