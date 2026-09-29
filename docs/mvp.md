# Transcript-first MVP

The first test uses existing local transcription, anonymous speaker labels when available,
and Jev for addressedness/recipient decisions. It does not train a new encoder or gate the
Whisper decoder. This keeps the first experiment small while measuring whether a custom
audio model is necessary. Speaker identity across sessions is outside the initial scope.

## Secure Jev setup on macOS

Compile and run the native entry dialog:

```sh
mkdir -p local/bin
swiftc scripts/store-jev-key.swift -o local/bin/store-jev-key
local/bin/store-jev-key
local/bin/store-jev-key --status
```

Paste into the masked native field with Command-V. The field sends the value directly to
the login Keychain, service `rightyo.jev`, account `api-key`. It never prints the key or
writes it into this checkout. Status reads attributes only. The item trusts no application
by default; when macOS asks whether `security` may read it for evaluation, choose **Allow**
for that use. Avoid **Always Allow**, which grants that shared executable persistent access.
The login Keychain's lock and access controls protect the item. Replacing a credential
recreates its access policy; a failed replacement reports that the old item was removed.

The Python provider captures credential output internally and emits sanitized errors.
It does not load `.env` files. User-managed deployments may supply `TYPESAFE_API_KEY`
through a secret manager; never put the value in shell arguments, chat, PRs or logs.

The hosted test below sends authored synthetic fixture text to TypeSafe. It uses the pinned
`jev-1.13.0`, batches attention and recipient Choice questions, defaults to at most 20 calls
with a 10-second timeout, and performs no automatic retries. It rejects redirects and
environment proxies. A confidence threshold is an experiment setting, not measured accuracy.

```sh
rightyo evaluate --input examples/synthetic-turns.json --provider jev --allow-hosted
rightyo evaluate --input examples/synthetic-turns.json --provider jev --allow-hosted --no-speakers
```

Every committed turn uses bounded past context. Partials, duplicate/stale turns and
superseded responses cannot create a fresh decision. Unknown recipients, conflicting
answers or low confidence abstain. This is a decision experiment, with no tool execution.
Default output excludes transcript text and source IDs; `--include-text` explicitly opts in.

## Local audio and anonymous speakers

Provision whisper.cpp and a model outside Git, recording upstream revision, license,
model size and checksum. Then use a supplied audio file; no capture or download happens
implicitly. File inference is offline and its segment timestamps are not live finalization
times. Put the generated transcript outside every public checkout.

```sh
rightyo audio-import --audio /private/conversation.wav \
  --whisper-executable /local/whisper-cli --model /local/ggml-base.en.bin \
  --session-id experiment-1 --output /private/turns.json
rightyo evaluate --input /private/turns.json --provider mock
```

Whisper alone does not supply stable speaker A/B identity. The importer accepts an actual
Argmax-compatible RTTM timeline with `--diarization-input /private/conversation.rttm`.
Its recording ID must match the audio filename stem. A neutral JSON speaker timeline is
also supported by `timeline-import`; see the authored example
[synthetic-speakers.json](../examples/synthetic-speakers.json). Imported A/B/A labels remain
stable within the session. Gaps, speaker changes within an ASR segment and overlaps retain
unknown attribution; turn boundaries never fabricate speaker identities.

The RTTM import is a provider boundary, not an installed or validated diarization model.
Compare real local diarization with plain ASR and an explicitly labeled oracle-speaker
baseline under [#24](https://github.com/mickdarling/rightyo/issues/24) and
[#25](https://github.com/mickdarling/rightyo/issues/25). ASR error, speaker confusion and
recipient ambiguity are separate sources of error. Text and speaker labels cannot reveal
unobserved gaze or acoustics. Passing synthetic tests does not establish accuracy.

## Verification and next decisions

[CI](ci.md) runs repository/workflow checks, Ruff, unit tests and an installed-wheel mock
smoke without secrets, speech capture, models or paid inference. Separate bounded Mac/model
benchmarks measure ASR runtime, speaker finalization, decision runtime and full end-to-end
latency. Do not sum file-inference timings and call them causal live latency.

Choose local ASR plus Jev first for fast iteration. It pays ASR cost for every turn and
adds hosted latency/text processing. Only move to an audio attention head and decoder gate
if held-out quality and measured avoided decode costs exceed attention, buffering and
speaker-processing overhead. A shared encoder can save duplicate features but couples
training/runtime integration. End-to-end custom training adds data and licensing costs;
keep it conditional on this baseline's failures. See the
[compute comparison (#21)](https://github.com/mickdarling/rightyo/issues/21).
