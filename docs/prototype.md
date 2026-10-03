# Local microphone lab

The Mac prototype runs three separate components: persistent Nemotron 3 speaker
processing, local Whisper transcription, and optional Jev attention decisions. It provides
Start, Stop & clear, an audio replay, speaker-labelled text, and a rolling context view to
copy into Hailing Station. It does not execute commands or summarize the conversation.
Speaker A/B identities last for one session; identifying a named person is out of scope.

## Start the lab

Use Python 3.11+ and macOS 14+ with Xcode command-line tools. Build the native capture helper;
this step does not request microphone permission or start capture:

```sh
.venv/bin/python scripts/build_microphone.py
.venv/bin/python -m pip install .
```

Provision the local runtimes and models described in [the benchmark](nemotron-benchmark.md).
The live diarizer needs the native runtime's `libnemo_speech_asr_c.dylib` C ABI library,
built at the recorded revision, rather than its file-processing executable. Downloaded
assets and compiled helpers remain outside Git. No runtime downloads anything automatically.

Create the ignored `local/prototype.json` with absolute paths to your existing assets:

```json
{
  "whisper_executable": "/local/whisper-cli",
  "whisper_model": "/local/ggml-base.en.bin",
  "diarization_library": "/local/libnemo_speech_asr_c.dylib",
  "diarization_model": "/local/Nemotron-3-Diarization.q8_0.gguf",
  "microphone_helper": "/checkout/local/RightyOMicrophone.app/Contents/MacOS/RightyOMicrophone",
  "demo_audio": "/private/authored-demo.wav"
}
```

`demo_audio` is optional, as is `"addressing": {"names": ["Hailing Station", "computer"]}`,
the runtime forms of address described in [the tool API](tool-api.md); `--name` flags on
the command line replace the file's names. An optional `"speakers"` object declares
hard-coded speaker roles for the headless tool, for example
`"speakers": {"owner": ["Speaker A"], "trusted": [], "owner_only": false}`; it may also
set `"stop_phrases"`. Session speaker labels such as `Speaker A` are anonymous and are
assigned per session by the diarizer, so a configured label names whichever voice
receives it; roles are precedence data for the host, not a verified identity. Roles
need the session-stable native diarizer: with `hosted-deepgram`, whose labels hold only
within one utterance, a `speakers` object naming owners or trusted speakers or setting
`owner_only` is refused at Start rather than silently never matching.
`"source": "model"` is refused by the lab and by `listen`, in both microphone and demo
modes, with a clear error before any capture starts: a hosted role question would run on
the audio path under the controller lock, where Stop and lease expiry cannot reach it.
Model-sourced roles remain available to `tool-replay`, which has no audio thread; the
live path is tracked in [#55](https://github.com/mickdarling/rightyo/issues/55).
An optional `"request_former": {"kind": "template"}` turns on
[request forming](tool-api.md#request-forming) for the headless tool: each `request`
event then also carries a `formed_request` string beside the unchanged raw turns, and the
started event advertises `request_forming`. It is off when the key is absent; `template`
is the only kind, and `--request-former template` on `listen` replaces the file's value.
An optional `"session_budget_seconds": 7200`
ends a session after that many seconds; `--session-budget SECONDS` on the command line
replaces the file's value. The budget is a positive whole number of seconds; omitting it
means there is no session ceiling, which is the default. In microphone mode the budget is
wall clock from Start, which includes diarizer start-up, so somewhat less than the
budget's worth of audio is processed. In replay/demo mode the budget is the exact audio
boundary: the replay is trimmed there however slowly it processes, and the wall clock
does not apply. Demo audio whose length equals the budget exactly ends `cancelled`
(budget reached) rather than `complete`.
Replay accepts mono PCM16 WAV at
16 kHz, at most three minutes; it feeds real PCM in causal order as quickly as processing
permits. It is not a wall-clock streaming latency measurement. Playback through speakers is
a separate physical test.

```sh
.venv/bin/rightyo prototype --config local/prototype.json --port 8766
```

## Speech backends

Recognition and speaker labelling are selected by two optional sections. Omitting them,
as every existing configuration does, keeps the local runtimes above:

```json
{
  "transcriber": {"kind": "whisper.cpp"},
  "diarizer": {"kind": "nemotron.cpp"}
}
```

The hosted alternatives are opt-in. Selecting one makes that side's local asset paths
optional, and the lab or `listen` then refuses to start without `--allow-hosted`, which
authorizes sending session audio to the named service for that run:

```json
{
  "transcriber": {
    "kind": "hosted-openai-compatible",
    "endpoint": "https://api.openai.com/v1/audio/transcriptions",
    "model": "whisper-1",
    "language": "en",
    "timeout_seconds": 30
  },
  "diarizer": {"kind": "hosted-deepgram", "model": "nova-3", "diarize_model": "latest"}
}
```

`hosted-openai-compatible` posts each finalized utterance as a WAV file to the given
`https` endpoint in the OpenAI `POST /v1/audio/transcriptions` schema with
`response_format=verbose_json` and `timestamp_granularities[]` `word` and `segment`;
per the API reference read on 2026-10-02, that is what `whisper-1` accepts, while
`gpt-4o-transcribe` and `gpt-4o-mini-transcribe` return only `json` and
`gpt-4o-transcribe-diarize` does not offer timestamp granularities. `endpoint` and
`model` are required; `language` (ISO 639-1) and `timeout_seconds` (at most 120) are
optional. The request has no training or retention opt-out parameter in the cited schema;
configure data-use controls on the provider account and confirm its policy before use. The credential is read from `RIGHTYO_TRANSCRIBER_API_KEY` or the login
Keychain item with service `rightyo.transcriber` and account `api-key`.

`hosted-deepgram` posts the trailing utterance window to Deepgram's pre-recorded
`https://api.deepgram.com/v1/listen` with `model` (default `nova-3`), `diarize_model`
(`latest`, `v1` or `v2`; default `latest`) and always `mip_opt_out=true`, which Deepgram
documents as excluding the request from its Model Improvement Program (participation is
otherwise the default) with zero data retention after the response; it reads the
word-level `speaker` labels,
merging consecutive words of one speaker that are at most 300 ms apart into timeline
segments (a wider gap stays uncovered, so speech inside it is unknown). A response
without Deepgram's `metadata.diarize_info` marker, which Deepgram documents as absent
when the diarizer did not run, fails the session as "diarization unavailable" rather than
passing as unknown-speaker output. `endpoint` may be
overridden with another `https` URL. The credential is read from
`RIGHTYO_DIARIZER_API_KEY` or the login Keychain item with service `rightyo.diarizer`
and account `api-key`; create it with `security add-generic-password -s rightyo.diarizer
-a api-key -w` (prompted, never on the command line). Deepgram labels speakers per
request, so with this backend Speaker A in one utterance is not known to be the same
person as Speaker A in the next; its turns carry `speaker_provenance:
"diarization-utterance"` with utterance-namespaced labels such as `u7 Speaker A`, and
the native Nemotron stream is the only backend whose
labels persist for the session. The 18,000-segment timeline cap does not apply to it.
A hosted request's Keychain credential lookup and whole exchange share a wall-clock
deadline of `timeout_seconds`; a stop is honoured during the lookup and within about 50 ms
while the body is read, and a mid-body pause shorter than the remaining budget is
tolerated. The endpoint must not carry its own query string or fragment.

The lab page labels the two speech stages from the loaded configuration: `LOCAL` with the
local runtime name, or `HOSTED` with the service family ("OpenAI-compatible hosted",
"Deepgram hosted"); the masthead then reads "Audio leaves this Mac" and the Start status
says "Connecting to hosted speech service" instead of "Loading local models". The
controller snapshot carries the same `transcriber`/`diarizer` summaries (`kind`, `hosted`,
`service`, and `utterance_local`, from which the page words its speaker-label legend) and
never the endpoint, model name or credential.

Hosted calls use the standard library only, send no environment proxy, refuse redirects,
cap responses at 2 MiB, and report failures as "Hosted speech backend failed" without
audio, transcript, URL or credential content; a failure stops the session like a local
one. Unknown kinds, unknown keys and non-`https` endpoints are rejected when the file is
loaded. No hosted backend has been accuracy-tested in this repository; the local smoke
evidence below is for the local runtimes only.

Open the printed local URL, including its one-time session fragment. The dashboard initially
sits idle. **Start microphone** explicitly enables the default input and may prompt for
macOS permission. A pending permission prompt can be cancelled with Stop. The capture helper
must be launched through this lab; it is not a background application. Input-route changes
require Stop and another Start. No setting changes system volume.

**Enable Jev** is initially unchecked. Enabling it before Start explicitly authorizes sending
the current finalized transcript and bounded recent text to TypeSafe. Credentials use the
[masked Keychain dialog](mvp.md#secure-jev-setup-on-macos), never the dashboard or its API.
Local transcription continues if hosted processing fails, reaches its request limit,
or falls behind enough to fill its bounded decision queue; queued hosted work is then
cancelled and decisions disabled for this session.
Attention is then unavailable, rather than fabricated. There are no automatic retries.
The confidence slider is an experiment setting, not a calibrated accuracy guarantee.

## Context and stopping

The local transcript retains five minutes by default, configurable from one to ten minutes,
with retained text capped at 1,000 turns / 1 MiB. These retention caps evict older
turns. Separately, each session permits at most 1,000 unique finalized turns in total,
including turns already expired or evicted. Hash-only duplicate bookkeeping keeps this
session bound; word-timed ASR fragments each count as a turn, so fragmented speech can
reach it well within a long ambient session. Reaching 1,000 turns stops processing with
an explicit message to start a new session, preserving valid history until expiry or Stop.
The native diarizer separately keeps one whole-session speaker timeline capped at 18,000
segments, returned in full at every utterance; very long sessions with frequent speaker
changes reach it, stop with a distinct timeline-limit message, and likewise need a new
session. A windowed timeline is tracked in [#54](https://github.com/mickdarling/rightyo/issues/54).
The whole turn overlapping the time boundary is retained and its overlap is reported. Ignore and uncertain decisions remain in local context: a later
request can refer to the preceding discussion. Copy context produces speaker-labelled text
for manual use downstream, including unknown-speaker and overlap indications.

A local audio/ASR failure stops capture and marks the session incomplete while preserving
already valid transcript turns until their normal expiry or an explicit Stop. Malformed
or padded-only ASR segments fail closed; they are not invented, silently dropped or
converted into future speech.

Jev receives a separate short context: up to eight past turns, at most 12,000 characters
including the current turn, and at most 32 KiB encoded. The five-minute local history is
not automatically sent to the hosted provider. A later downstream summarizer could consume
that longer local history; this prototype provides manual copy only.

Stop cancels capture and native processing, discards an unfinished utterance, and clears
retained session text and queued decisions. An already-sent hosted request can finish;
its result is discarded. A bounded context copy used by in-flight inference or Keychain
lookup can remain until that operation returns (10-second HTTP timeout; Keychain lookup
up to 120 seconds). Closing the page attempts Stop, and loss of browser heartbeats
stops an active session after 15 seconds. Browser background timer throttling can also
trigger this lease, so keep the lab visible while listening. This is a prototype limitation,
including when switching tabs to a downstream application. There is no default session
duration ceiling: a session listens until Stop, a lease loss, an error, or the configured
session budget. A configured budget ends the session the same way Stop does, discarding
the unfinished utterance and emitting the ordinary `cancelled` tool event. A completed replay's text ages out too. Refreshing loses the browser token; use the
launch URL again. Only one session runs at a time.

Capture reads are aggregated into 200 ms PCM blocks. The pending capture queue holds
at most 32 seconds (1,024,000 bytes) by default, accommodating the local 30-second recognizer
timeout. Sustained slower-than-real-time processing still fails closed at that bound.
PCM buffers are bounded and temporary Whisper files are deleted after use; microphone audio
is not archived. Expiry and Stop release application-owned plaintext references. This is
not forensic zeroization of Python/native memory or a guarantee about OS swap or crash
collection. Copying text intentionally transfers it to the clipboard. The HTTP endpoint
binds only to loopback with Host/Origin checks, bearer controls, and no-store responses;
local processes with access to the session token remain within the trust boundary.

## Interpretation and verification

Whisper runs on completed speech windows, not every incoming token. Energy-based endpointing
uses 240 ms pre-roll and roughly 1.44 seconds of silence, with a 12-second continuous-speech
cut. Nemotron uses its persistent native `v3-streaming` state, keeping speaker channels
stable across windows. Word timestamps intersect speaker intervals conservatively. Gaps,
overlap and zero-duration word timestamps remain unknown instead of assigning the majority
speaker. Tail attribution at a forced speech cut can remain unknown.

This stack decodes speech before deciding whether to attend. It tests product usefulness
quickly, but does not demonstrate the compute savings of a future audio attention gate.
Diarization tells us who spoke, while Jev infers an addressee from text; neither reveals gaze.
Speaker-to-microphone playback adds room acoustics and echo, but two synthetic voices do not
establish accuracy on real overlapping conversation. Playback is never authority to execute
an action. Test Hailing Station requests alongside third-party conversation and quoted
requests, then inspect false activations and missed requests.

[CI](ci.md) checks the Python implementation, mocked capture/cancellation, retention,
controller and HTTP boundaries, packaging and workflows without microphone access, downloads,
or hosted credentials. Native helper compilation/signing and real model/capture smoke tests
are additional Mac checks; Linux CI does not validate AVFoundation or Metal.

Follow [#37](https://github.com/mickdarling/rightyo/issues/37) for the lab,
[#36](https://github.com/mickdarling/rightyo/issues/36) for retrospective context,
[#27](https://github.com/mickdarling/rightyo/issues/27) for alignment, and
[#25](https://github.com/mickdarling/rightyo/issues/25) for calibration on consented examples.

## Local smoke evidence

On the development Mac (Apple M4 Max, macOS 15.8), causal replay of the authored
23.697-second two-voice fixture completed through the controller in 1.878 seconds:
ten transcript groups, six speaker-known, stable A/B/A identity and zero Jev calls.
This feeds audio faster than real time and is a throughput smoke check.

With explicit owner authorization, the same clip was played through the physical speakers
and captured for 29.38 seconds through the default microphone. The lab stayed listening,
produced eleven groups (six speaker-known, five unknown), observed both A/B labels,
and made zero Jev calls. Stop returned idle with zero retained turns and zero pending
decisions. No raw microphone text or recording was archived. This verifies the physical
capture path, not word accuracy, diarization error rate or real-conversation quality.


### VoiceBox replacement fixture

The development lab now uses two existing VoiceBox profiles with its locally cached
Qwen 1.7B engine for the replay button. The 22.962-second authored fixture uses the
same words as the earlier fixture and alternates A/B/A, with short pauses between
turns. Generated voice audio stays in the owner's private local cache; neither audio
nor voice reference clips are distributed with this repository. VoiceBox generation
is a fixture preparation step, separate from the runtime recognition pipeline.

Causal controller replay completed in 1.496 seconds, producing eleven transcript groups,
six speaker-known, with the known labels returning A/B/A and zero Jev calls. Comparing
recognized words to the 62 authored words after lowercasing and removing punctuation
produced four token edits, including a contraction. This is a small synthetic smoke
check, not a held-out speech recognition benchmark or proof that synthesis preserved
every word.

The replacement was also played through the physical speakers and microphone for a
30.64-second capture. The lab remained listening and produced four groups: two labelled
Speaker A, one Speaker B and one unknown, with zero Jev calls. Stop returned idle with
zero retained turns and pending decisions. Only capture counts were recorded; raw
microphone text and audio were not archived. Room acoustics, endpointing and playback
conditions make group counts unsuitable as a direct accuracy comparison with the
previous fixture. Real conversations and overlap still need consented evaluation.
