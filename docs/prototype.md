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
the command line replace the file's names. The object may also list recognizer spellings
of a name, for example `"variants": {"Hailing Station": ["Haling Station"]}`; see
[name variants](tool-api.md#transport-and-lifecycle). An optional
`"meeting": {"ask_anywhere": true}` configures meeting mode
(hailing-station#735 meeting mode (#227)). The boolean defaults to true: while the host's
meeting control is on, an owner-authority turn naming the assistant anywhere promotes
`uncertain` to `attend`, never `ignore`. False keeps the name/owner bar without promotion.
With no speaker roles, meeting mode attends nothing; inferred or unlabelled turns never pass.
The section accepts only `ask_anywhere`; it does not turn meeting mode on. See the
[control contract](tool-api.md#meeting-mode). An optional
`"turns": {"merge_gap_ms": 2000}` sets how long a finalized turn waits for a continuation
by the same speaker before it is emitted and decided (default 2,000 ms, 0 to 5,000, 0
off); see [joined turns](tool-api.md#joined-turns), which also covers `complete_hold_ms`,
`collapse_restarts` and `bridge_backchannels` for one thought split into several
requests (#216), `wake_join` (default true) for a lone wake word split from the owner's
sentence (#230), `owner_pause_ms` (default 0, off; 0 to 10,000) for a longer pause
on the owner's turns only, which joins them even with `merge_gap_ms` 0
(hailing-station#735), and `word_label_coverage` (default 0, off; 0.5 suggested) for one-word
turns left without a speaker (#220). The same object's `reply_wait_ms`
(default 1,200 ms, 0 to 3,000, 0 off) sets how long a question or request is held to
observe the [post-turn gap](tool-api.md#scene-and-post-turn-gap). The same object's
`edge_attribution_ms` (default 0, off; 0 to 2,000 in 20 ms steps) turns on edge
attribution, an inference rather than an observation. The streaming diarizer's timeline
trails the audio and the recognizer's word timestamps run slightly past it, so a
sentence's last word ("bit.", "you.") often arrives as a separate unlabelled turn that
can't be joined. With this set, an unlabelled word takes the utterance's only labelled
speaker when it overlaps no one and either touches only that speaker's timeline segments
or, touching none, lies within that many milliseconds of one. Inferred labels carry the
same `diarization-timeline` provenance as observed ones. The risk: a second speaker's
first word inside that slack, before the diarizer segments them, is labelled as the
first speaker. For that reason it is refused together with configured speaker roles
(`owner`, `trusted`, `owner_only`) that name session labels, unless roles come from
[live speaker identification](tool-api.md#roles-from-live-speaker-identification), which
withholds owner authority from any turn with inferred words. It never applies to
utterance-level diarizers.
Keep the slack small (a few hundred milliseconds). The same object's `tail_join_ms`
(default 0, off; 0 to 2,000) turns on tail join
([#129](https://github.com/mickdarling/rightyo/issues/129)), also an inference. Edge
attribution only reaches words near the speaker's timeline segment, so a longer trailing
piece can still arrive unlabelled and become a second turn, with a second request and
acknowledgement. With this set, an unlabelled, non-overlapping turn that starts within
that many milliseconds of a held labelled turn's end is joined into it and takes its
speaker and provenance (see [joined turns](tool-api.md#joined-turns)). With merging off,
a labelled turn is held for at least that long. It carries the same risk as edge
attribution, a second speaker's short reply inside the window becoming the first
speaker's, and is refused together with configured speaker roles the same way, with the
same exception for live speaker identification roles. An optional
`"dismissal": {}` object turns on [natural dismissal](tool-api.md#natural-dismissal-and-barge-in)
(the decision model's dismissal question, the `dismiss` event, self-withdrawal and the
cool-down), with optional `window_ms`, `cooldown_ms`, `cooldown_min_confidence` and
`min_attend`; it is off when absent. An optional `"conversation": {}` object turns on
[conversation mode](tool-api.md#conversation-mode) (#82): engaged follow-ups after a
request and the `conversation` event, with optional `window_ms`,
`follow_up_min_probability`, `closing_phrases`, `appendix_ms`, `require_cue`,
`cue_min_attend`, `continuation_cue` and `cue_when_others_ms`; it is off when absent.
With `cue_when_others_ms` on, while another voice is present, RightyO applies the meeting bar
automatically: every request needs owner authority and the assistant's name, even a direct
model `attend`. There are no follow-ups in this window; owner + name anywhere may promote
abstention through the same meeting policy. Static cue shapes and waivers never bypass it.
The opt-in `conversation.require_guest_address` replaces this automatic meeting bar in
ambient listening. The voiceprint-bound owner can make unnamed requests and follow-ups
at the usual probability floor. With another voice present, a low-scoring ambiguous
aside still needs a cue. Owner standing requires at least one strictly diarized word,
with all strict words on the enrolled owner's bound label. Inferred opening names and
trailing words preserve that standing. Guests must open each request with a configured
name or variant. The [tool contract](tool-api.md#conversation-mode)
describes the cue checks.

An optional
`"acknowledgement": {}` object turns on [acknowledgement gating](tool-api.md#acknowledgement-gating)
(#132): each request carries `acknowledge`, true only when the turn uses a configured name
or its attend confidence reaches `min_confidence` (default 0.7, 0 to 1), so the host skips
the instant acknowledgement for unlikely replies, and, with optional `dedup_window_ms`
(default 8,000, 0 to 60,000, 0 off), only once per spoken request split into several
requests (#122); it is off when absent. An optional `"end_of_turn"` object (#117) puts the Smart Turn v3
end-of-turn model in front of the silence rules:
`{"enabled": true, "python": "/abs/venv/bin/python", "model": "/abs/smart-turn-v3.2-cpu.onnx"}`
with optional `threshold` (default 0.5), `silence_ms` (default 200, 20 to 1,000) and
`threads` (default 4). It is off when absent or `"enabled": false`. The interpreter must
have `numpy` and `onnxruntime`; RightyO itself installs neither and downloads no model.
After `silence_ms` of silence the model scores the open utterance. At or above the
threshold the turn is finalized once the diarizer timeline covers its last word, and is
not held for a continuation, so its post-turn gap is unobserved. Below it, the hangover
and merge hold apply unchanged. Each score is reported on stderr without content
(`end_of_turn p=… silence_ms=… outcome=complete|wait`, then
`end_of_turn finalized silence_ms=…`). A model that fails to start or answer falls back to
the silence rules for the session. An optional `"speaker_id"` object (#137) configures the
local voiceprint model and store for [speaker enrollment](speaker-enrollment.md). With
`"live": true` it also runs [shadow identification](speaker-enrollment.md#shadow-identification-in-live-sessions)
in `listen` sessions in every mode: per-label scores on stderr (`speaker_id label="…" state=… id=…`), with no
change to turns, roles, requests or events. An optional `"speakers"` object declares
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

An optional `"audio_retention"` object ([#226](https://github.com/mickdarling/rightyo/issues/226))
sets how much audio RightyO keeps in memory, with two independent keys. Both are whole
seconds and both default to 60, the fixed value before:

- `speaker_ring_seconds` (10 to 600): how much of the session's PCM
  [live speaker identification](speaker-enrollment.md#shadow-identification-in-live-sessions)
  keeps to cut each turn's audio from. It costs memory only: PCM16 at 16 kHz is 32 KB/s,
  so about 19 MB at 600 s. A shorter ring clips the audio of long turns (counted as
  `clipped`).
- `timeline_window_seconds` (30 to 100): how far back the diarizer's speaker timeline is
  returned at each utterance for attribution. It also sets the per-response segment bound,
  300 segments per second of window (18,000 at 60 s). The lower bound is twice the longest
  utterance (15 s), so no segment that overlaps an open utterance is dropped. The upper
  bound keeps the native diarizer worker's reply inside its 2 MiB pipe limit: the worker
  returns the whole window as one JSON line, and at 100 s its worst case is 30,000
  segments of at most 68 bytes each, 2,040,025 bytes.

For example `"audio_retention": {"speaker_ring_seconds": 120, "timeline_window_seconds": 90}`.
A value out of range or not a whole number is refused when the configuration loads, with
an error that names the rule; nothing is clamped. Both buffers stay in memory and are
never written. The values are read when RightyO starts, so a change takes effect at the
next restart. Each `listen` session logs them once on stderr, in seconds only:
`audio_retention speaker_ring_s=120 timeline_window_s=90`. The lab has no report channel
for these retention settings.

An optional `"decision": {"provider": "jev", "allow_hosted": true}` is the file form of
`listen --use-jev --allow-hosted`, for hosts that start `listen` with a fixed argument list.
Both keys are required and strictly typed; `"provider": "jev"` without
`"allow_hosted": true` is refused. An optional positive `max_requests` caps the Jev requests
of each `listen` session; without it, live microphone and stdin sessions have no request
cap. It covers Jev decisions only, not hosted speech, and the lab ignores it. An optional
`scene` string (1 to 1,000 characters, or `null` for none) replaces the default
[scene](tool-api.md#scene-and-post-turn-gap) given to the decision model. See [the tool guide](tool-quickstart.md#jev-decisions-from-the-configuration-file)
for the rules and the macOS Keychain prompt.
Replay accepts mono PCM16 WAV at
16 kHz, at most three minutes; it feeds real PCM in causal order as quickly as processing
permits. It is not a wall-clock streaming latency measurement. Playback through speakers is
a separate physical test.

```sh
.venv/bin/rightyo prototype --config local/prototype.json --port 8766
```

## Silero VAD for leaving Privacy by voice

A voice exit needs speech and silence observations from a local Silero VAD worker.
Provision the official `snakers4/silero-vad` ONNX file from tag `v5.1.2`. From the
checkout root, fetch and verify the file in the ignored cache.

```sh
mkdir -p .cache
curl -q --fail --show-error --proto '=https' \
  --output .cache/silero_vad-v5.1.2.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/v5.1.2/src/silero_vad/data/silero_vad.onnx
printf '%s  %s\n' \
  2623a2953f6ff3d2c1e61740c6cdb7168133479b267dfef114a4a3cc5bdd788f \
  .cache/silero_vad-v5.1.2.onnx | shasum -a 256 -c -
```

Add this section to the local configuration, substituting absolute paths. The worker
interpreter needs NumPy and ONNX Runtime. RightyO launches it in isolated mode,
verifies the SHA-256 again and passes those same bytes to the CPU inference session.

```json
{
  "silero_vad": {
    "python": "/absolute/path/to/worker/bin/python",
    "model": "/absolute/path/to/rightyo/.cache/silero_vad-v5.1.2.onnx"
  }
}
```

An absent configuration, missing or unreadable model, checksum mismatch, or worker
failure disables voice exits for the session. Sessions with enrolled roles report
a missing configuration on stderr. An explicitly configured worker reports startup
or runtime failures regardless of roles. These diagnostics include only a fixed
reason. Host privacy controls remain available. Private status questions remain available both without a configured VAD
and after a runtime worker failure. Once a worker is judging private speech,
only its observed blocks can open a private recognition window.

Every voice exit takes about **1,440 ms** from the acoustic end of the command.
The window starts at the end of the last word supported by Silero speech evidence.
ASR closure timing cannot shorten it. The processor waits for the entire block
covering the deadline before the owner check. Resumed speech on any of those
blocks cancels the exit. This errs toward staying private. All exit preconditions,
including a fresh worker liveness round-trip, must still pass at commit.

In the round 17 generated replay, the last word ended at 1,400 ms. The pinned
worker's first speech-positive continuation block began 936 ms later at 400
counts RMS, 1,352 ms later at 180 counts RMS and 5,416 ms later at 160 counts
RMS. The 2026-11-27 operator ruling adds two 32 ms blocks to the 180 RMS lag
and rounds up to 45 whole blocks. Normal ASR close was 2,120 ms. The fast
“complete” endpointer closed as early as 700 ms, truncating the command. Its
earliest close retaining every word was 1,400 ms. Both qualifying endpointers
reached the owner check at 2,848 ms on the clean replay.

The pinned VAD missed a very quiet speech-like onset from stream start and
detected a quieter, 160 counts RMS continuation only after the hold expired.
The onset replay also required the recognizer to omit the quiet words. The
owner-clip replay below remains the real-speech install gate.

R4 and R4b cover continuing speech as judged by the pinned VAD. Continuing
non-speech sound such as a tone, hum or music does not block an exit after a
genuine exit phrase.

### Reproduce the Silero test worker on Apple Silicon

The #350 round-8 replay used an existing CPython 3.11.14 on macOS 15.8.3 arm64,
NumPy 2.2.6 and ONNX Runtime 1.24.4 with the CPU provider. ONNX Runtime also
requires flatbuffers, packaging, protobuf and SymPy. SymPy requires mpmath.
The requirements file pins all seven versions and records the PyPI SHA-256 of
each wheel alongside the model SHA-256. [Test requirements](silero-test-macos-arm64.txt)

From the checkout root, start with a fresh `.cache/vad-venv`. The existing
interpreter's pip (25.3 in this run) installs the wheels into the test venv.

```sh
(
  set -e
  test "$(python3.11 -I -B --version)" = "Python 3.11.14"
  mkdir -p .cache/test-home .cache/tmp
  python3.11 -I -B -m venv --without-pip .cache/vad-venv
  env -i PATH="$PATH" HOME="$PWD/.cache/test-home" TMPDIR="$PWD/.cache/tmp" \
    PIP_CONFIG_FILE=/dev/null python3.11 -I -B -m pip \
    --disable-pip-version-check --no-cache-dir --python .cache/vad-venv/bin/python \
    install --index-url https://pypi.org/simple --only-binary=:all: --require-hashes \
    -r docs/silero-test-macos-arm64.txt
)
```

Run the generated-tone fixtures with authored recognition and a stand-in owner.
The tests use the cache model above by default. Set `RIGHTYO_SILERO_MODEL` to
test a different local copy of the pinned file.

```sh
cd tests
HOME="$PWD/../.cache/test-home" TMPDIR="$PWD/../.cache/tmp" \
  RIGHTYO_SILERO_PYTHON="$PWD/../.cache/vad-venv/bin/python" PYTHONPATH=../src \
  taskpolicy -b python3 -m unittest -v test_silero_vad.RealModelDecisions
```

The cases skip when the model or runtime is absent. With the pinned worker, the
100 and 200 Hz machinery replays, the omitted low-pitched tail with a 200 ms
lead-in and the persistent fundamental with a louder third harmonic must all
keep privacy on, make zero owner checks and emit no transcript or exit.
Two boundary replays use a persistent 100 Hz tone with raised second and fourth
harmonics and a synthetic tail at 1.465× fan RMS after a 400 ms lead-in.
Silero labels the continuing tone in each as non-speech. Both expect one
stand-in owner check and one exit, as does the 50 Hz clean control with a
400 ms lead-in. The fake worker supplies known speech intervals and
exercises the positive machinery, onset, omitted-tail and early-close decisions in CI.

### Owner-clip replay for #350

The optional file tests read six consented owner recordings from
`~/.local/share/rightyo-fixtures/350/` or the directory named by
`RIGHTYO_350_FIXTURES`. Use uncompressed mono 16 kHz signed 16-bit PCM WAV.
Trim leading and trailing quiet to the spoken boundaries. The replay supplies
the lead-in, background, exact inter-clip gaps and trailing quiet in memory.
Keep the clips and the separate existing enrollment outside Git.

| Filename | Spoken words | Approximate spoken duration |
| --- | --- | --- |
| `privacy-off.wav` | Jarvis, privacy off. | 1.2–1.6 s |
| `podcast.wav` | Jarvis, start listening to the podcast. | 2.5–3.5 s, one uninterrupted sentence |
| `jarvis.wav` | Jarvis. | 0.7–1.0 s |
| `start-listening.wav` | Start listening. | 0.8–1.2 s |
| `stop-recording.wav` | Jarvis, stop recording. | 1.2–1.8 s |
| `go-private.wav` | Go private. | 0.8–1.2 s |

Each split fragment needs at least 0.5 s of speech after the verifier's silence
trim. Set these variables to existing local files before running the tests.

| Variable | File |
| --- | --- |
| `RIGHTYO_SILERO_PYTHON` | Worker interpreter for Silero and WeSpeaker, with NumPy and ONNX Runtime |
| `RIGHTYO_SILERO_MODEL` | Optional override for the pinned Silero model in the checkout cache |
| `RIGHTYO_350_WHISPER_CLI` | Local whisper.cpp CLI with the stdin WAV and stdout full-JSON support used by the prototype |
| `RIGHTYO_350_WHISPER_MODEL` | Whisper model for that CLI |
| `RIGHTYO_350_SPEAKER_MODEL` | Pinned WeSpeaker ResNet34-LM ONNX model used for the enrollment |
| `RIGHTYO_350_OWNER_PRINT` | Existing RightyO `<id>.json` enrollment from independent owner speech, read individually |

With those variables exported, run the suite from the checkout root.

```sh
PYTHONPATH=src:tests python3 -m unittest -v test_privacy_owner_clips
```

The single exit must authenticate once and emit one content-free exit clean and
over 50, 100 and 200 Hz tones. At 100 and 200 Hz, 24 seconds of machinery alone
must open zero recognition windows. The synthetic 200 Hz onset has triggered
Silero before (see below). This same background check is part of owner qualification.

The full podcast sentence and its final words alone must stay private. A second
pass drops Whisper's words and timings after “listening” while Silero hears the
whole recording. Before any install, this owner-clip speech-continues negative
(`test_omitted_continuing_speech_stays_private`) must prove that Silero detects
real continuing speech, the ASR window stays open through the clip, and the
processor makes zero owner checks and emits zero exits. Whisper's timing for “to” also locates
the excerpt used for the side-talk replay.

With a 2 s gap, the name must authenticate silently and the remainder must
authenticate independently and emit one exit. With a 4.2 s gap, only the name
is checked. The remainder alone stays private. Each public entry must publish
its own transcript before the merge hold expires. Replayed in privacy, it must
produce zero checks or events.

These replays use a single-speaker diarization stand-in. Silero, local ASR and
isolated owner scoring use their production adapters. A missing clip, model,
interpreter or enrollment skips the affected tests with a reason. A supplied
fixture that misses an expected result fails qualification.

### Round-8 synthetic replay results (2026-11-27)

All four real-model tests failed on the five replays below. Privacy stayed on and
the owner verifier received zero calls in every replay. The test threshold was
0.5. In a separate run of the same fixtures, a test-only observer sampled the ONNX
speech probability inside the worker. Thresholding each score reproduced every
32 ms label on the worker socket. Counts below include blocks wholly inside each
stated interval. Times are relative to the start of generated audio.

| Synthetic input | Peak speech probability and timing | Observed decision |
| --- | --- | --- |
| Persistent 100 Hz fundamental, louder third harmonic | Peak 0.1661 in the first 400 ms, 0.4890 in the louder section (400–1400 ms). | All 100 blocks over 3200 ms scored absent, certifying a false onset gap. No exit. |
| Low-pitched voiced tail at 1.47× fan RMS | Command peaked at 0.9896. Tail peaked at 0.4159 with 0/62 full blocks labelled speech from 1200–3200 ms. | No exit. The recognition window ended at 2220 ms, 980 ms before the authored tail ended. |
| Steady 200 Hz machinery, then command | First machine block (0–32 ms) scored 0.5733, the only positive in 24 s. Command peaked at 0.1656 with 0/50 speech blocks from 24000–25600 ms. | Machinery alone opened a 0–1480 ms recognition window. The expected exit never occurred by 27200 ms. |
| Clean command over the same 50 Hz fan (tail control) | Command peaked at 0.9896 at 224–256 ms. The last speech block ended at 768 ms. The command continued to 1200 ms. | The expected exit never occurred by 2800 ms. |
| Command sharing the fan's 100 Hz fundamental | Fan peaked at 0.0477 in 24 s. Command peaked at 0.2619 with 0/50 speech blocks from 24000–25600 ms. | No recognition window or exit by 27200 ms. |

With the uninstrumented worker, unittest took 9.964 s for all four methods.
Wall times including worker startup were 1.104 s for the persistent fundamental,
3.014 s for the tail and its control together, 3.443 s for 200 Hz machinery and
2.401 s for the shared 100 Hz fundamental.

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
labels persist for the session. The Deepgram backend sends and diarizes only a trailing
utterance window.
A hosted request's Keychain credential lookup and whole exchange share a wall-clock
deadline of `timeout_seconds`; a stop is honoured during the lookup and within about 50 ms
while the connection is opened or the body is read, and a mid-body pause shorter than the
remaining budget is tolerated. The endpoint must not carry its own query string or fragment.
Its host is a DNS name (internationalized names in their ASCII `xn--` form), an IPv4
address, or a bracketed IPv6 literal such as `https://[::1]:8443/v1` without a zone
identifier; whitespace, control characters, user information and characters outside
RFC 3986 paths are rejected when the configuration loads.

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
A single transiently unavailable decision (timeout, connection failure, HTTP 429/529 or
5xx) marks only that turn `uncertain` with its reason, and the next turn is sent as usual.
Local transcription continues if hosted processing fails otherwise or five times in a row,
reaches its request limit, or falls behind enough to fill its bounded decision queue;
queued hosted work is then cancelled and decisions disabled for this session.
Attention is then unavailable, rather than fabricated. There are no automatic retries.
The confidence slider is an experiment setting, not a calibrated accuracy guarantee.

## Context and stopping

Everything here is in memory unless the operator opts in to the
[decision state log](tool-quickstart.md) (`decision.state_log`, #199). That log writes
each decided turn's state, including transcript text, to a private local file, and
nothing in the lab clears it: Clear and Stop leave it in place.

The local transcript retains five minutes by default, configurable from one to ten minutes,
with retained text capped at 1,000 turns / 1 MiB. These retention caps evict older
turns. Separately, each session permits at most 1,000 unique finalized turns in total,
including turns already expired or evicted. Hash-only duplicate bookkeeping keeps this
session bound; word-timed ASR fragments each count as a turn, so fragmented speech can
reach it well within a long ambient session. Reaching 1,000 turns stops processing with
an explicit message to start a new session, preserving valid history until expiry or Stop.
The native diarizer returns only the trailing 60 seconds of its speaker timeline at each
utterance (an utterance spans at most 15 seconds), so it imposes no session-length limit
([#54](https://github.com/mickdarling/rightyo/issues/54)). The configuration's
`audio_retention.timeline_window_seconds` sets that window from 30 to 100 seconds.
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
