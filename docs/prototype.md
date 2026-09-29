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

`demo_audio` is optional. Replay accepts mono PCM16 WAV at 16 kHz, at most three minutes;
it feeds real PCM in causal order as quickly as processing permits. It is not a wall-clock
streaming latency measurement. Playback through speakers is a separate physical test.

```sh
.venv/bin/rightyo prototype --config local/prototype.json --port 8766
```

Open the printed local URL, including its one-time session fragment. The dashboard initially
sits idle. **Start microphone** explicitly enables the default input and may prompt for
macOS permission. A pending permission prompt can be cancelled with Stop. The capture helper
must be launched through this lab; it is not a background application. Input-route changes
require Stop and another Start. No setting changes system volume.

**Enable Jev** is initially unchecked. Enabling it before Start explicitly authorizes sending
the current finalized transcript and bounded recent text to TypeSafe. Credentials use the
[masked Keychain dialog](mvp.md#secure-jev-setup-on-macos), never the dashboard or its API.
Local transcription continues if hosted processing fails or reaches its request limit;
attention is then unavailable, rather than fabricated. There are no automatic retries.
The confidence slider is an experiment setting, not a calibrated accuracy guarantee.

## Context and stopping

The local transcript retains five minutes by default, configurable from one to ten minutes,
with a further 1,000-turn / 1 MiB cap. The whole turn overlapping the time boundary is retained
and its overlap is reported. Ignore and uncertain decisions remain in local context: a later
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
including when switching tabs to a downstream application. Active sessions also stop after 15 minutes.
A completed replay's text ages out too. Refreshing loses the browser token; use the launch
URL again. Only one session runs at a time.

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
