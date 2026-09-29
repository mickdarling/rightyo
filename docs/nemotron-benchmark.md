# Nemotron 3 + Whisper + Jev: first Mac benchmark

Date: 2026-09-29. This implements the first local feasibility comparison in
[#34](https://github.com/mickdarling/rightyo/issues/34). All audio was authored speech
rendered by macOS TTS, or generated zero PCM. No microphone, real conversation, downloaded
evaluation dataset or hosted audio processing was used. Jev received synthetic transcript
text only. Raw audio, transcripts, credentials, model assets and detailed run records remain
outside Git. This is a small integration benchmark, not held-out conversational accuracy.

## Result and decision

The stack works: Nemotron 3 supplies anonymous speaker timelines, whisper.cpp supplies
transcripts, and Jev supplies bounded attention/recipient decisions. Retain these replaceable
components for the next experiment. Nemotron costs more whole-file time and RSS than the
existing SpeakerKit baseline on this sample, but its more continuous spans improve the
unchanged conservative transcript join. No ASR decoder gate or shared encoder is implemented.

Hardware: Apple M4 Max, macOS 15.8, arm64; Python 3.14.7. Input: one 23.6970625-second,
16 kHz mono PCM16 WAV, with two distinct generated voices in A/B/A order and two-second gaps.

| Standalone diarization, same recording | Nemotron 3 / Metal | SpeakerKit |
| --- | --- | --- |
| Median elapsed time, three repeated fresh processes | 0.606 s | 0.343 s |
| Repeated-process range | 0.604–0.607 s | 0.339–0.391 s |
| Largest observed peak RSS | 182.2 MiB | 111.5 MiB |
| Speakers found | 2 | 2 |
| Returning speaker pattern | A/B/A | A/B/A |
| Timeline spans | 3 | 9 |
| ASR turns assigned by the strict join | 5 / 7 | 1 / 7 |
| ASR turns retained as unknown | 2 / 7 | 6 / 7 |

Times above include process launch and model loading, measured around `/usr/bin/time -l`.
The OS/model file caches were warmed; each invocation reloaded its model into a new process.
No verified cold-cache or resident-model timing is claimed. Native Metal peak memory footprint
was about 334 MiB; RSS alone is not a complete unified-memory/GPU allocation measurement.
These runtimes use different preprocessing and postprocessing; more continuous spans are
not, by themselves, evidence of better diarization accuracy.

Nemotron's three repeated Metal outputs were byte-identical on this recording. A single
`--backend cpu` run took 2.115 s, with the returning-speaker boundary differing by 10 ms.
That selection disables GPU use but still permits Apple Accelerate; the Metal selection
also permits CPU/Accelerate work. No cross-backend bitwise reproducibility is claimed.

The repository benchmark harness, using the isolated adapter rather than the timing wrapper,
independently measured Nemotron at 0.605 s median (0.561–0.614 s) and Whisper `base.en`
at 0.391 s median (0.334–0.398 s), with one initial invocation plus three repeated fresh
processes per stage. Attribution was 5/7 in every Nemotron repeat. Its optional supplied
SpeakerKit RTTM comparison measures attribution only; the separate native runs above
provide actual SpeakerKit timing. Whole-file real-time factors were approximately 0.0255
for Nemotron and 0.0165 for Whisper: elapsed processing time divided by audio duration.

Whisper had zero word edits across the 62 authored reference words in this one easy sample.
The diagnostic concatenates transcript segments, lowercases ASCII word tokens, removes
punctuation and retains internal apostrophes, without number expansion. It says nothing
about noisy, accented, overlapping or held-out conversational speech.

## Jev and the complete file trial

The first labelled transcript replay made seven Jev requests, each batching attention and
recipient questions, at the existing confidence threshold of 0.7. It emitted two `attend`
decisions and five abstentions, with 2.072 s total provider time. Hiding the speaker labels
for the same transcript emitted one `attend` and six abstentions, with 1.767 s provider time.
This single ordered ablation is not statistical evidence that labels improve decision quality.
No threshold or prompt was tuned to make this sample pass.

A separately timed complete run processed the supplied file through Nemotron, Whisper,
private transcript export/reload and Jev in 2.919 s: 1.045 s for local diarization/ASR/join,
and 1.865 s provider time across seven requests. It also emitted two `attend` and five
abstentions. This is serial processing after the full file already exists, not time from
spoken onset to a live decision. All three hosted trials completed; 21 requests in total,
no retries, no application actions. The model was pinned to `jev-1.13.0`; no credential
value entered chat, tool output or Git.

The two explicit assistant-directed command segments attended in the labelled replay.
Other segments mostly abstained. A useful product still needs held-out operating-point
selection and assessment of missed requests, false activations, recipient confusion and
abstention under [#25](https://github.com/mickdarling/rightyo/issues/25).

## Silence, overlap and alignment limits

- Five seconds of zero PCM produced no Nemotron speaker spans.
- An 8.2585-second authored mixture produced two speakers and 5.408 seconds of simultaneous
  predicted activity. The generated voice-file envelopes overlap for 5.149 seconds, but
  contain phrase pauses and are not frame-level speech annotations. No DER is calculated
  from these envelopes, and this does not establish overlap-transcription quality.
- Two full Whisper segments still cross speaker boundaries and remain unknown. The join
  does not fill gaps, split text without word timestamps or assign by majority vote.
  Word/turn alignment remains [#27](https://github.com/mickdarling/rightyo/issues/27).
- RTTM rounds the final boundary to 23.700 s for a 23.6970625 s WAV. This did not change
  assignment of the existing Whisper segments; it is not exact sample-level endpoint timing.

The selected native `v3-streaming` preset has 13 center frames and one right-context frame,
each 80 ms: 1.04 s center audio plus 80 ms lookahead gives **1.12 s buffering before compute**.
It uses 264 speaker-cache frames, 80 FIFO frames and a 40-frame cache-update period.
The adapter runs stateful chunk inference over a supplied file and exports final RTTM;
it does not expose incremental labels or measure label revisions/finalization.
Published lower-latency model settings are not implemented as RightyO settings here.
Persistent causal PCM sessions, live latency, longer recordings, similar voices,
background media and representative consented tests remain in #34.

## Provisioning and upstream notices

Acquisition was explicitly authorized for this experiment and happened outside Git.
The inference adapter and benchmark harness never provision dependencies or models.

- [NVIDIA NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) revision
  `0f706e43cf1fbc031bad1423e05460d3acaeaa1c`, built with the `metal-diar` preset and
  Unix Makefiles. NVIDIA runtime code uses Apache-2.0; dependency notices remain separate.
- Its ggml submodule is pinned by the runtime checkout. SentencePiece v0.2.0,
  revision `17d7580d6407802f85855d2cc9190634e2c95624`, was built into a separate local
  cache prefix with Apache-2.0 notices. No global packages were installed.
- [Official Nemotron 3 model](https://huggingface.co/nvidia/Nemotron-3-Diarization),
  revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`, file
  `Nemotron-3-Diarization.q8_0.gguf`, 107,012,128 bytes, SHA-256
  `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`.
  These weights retain OpenMDW-1.1 terms; they are not redistributed or relicensed as AGPL.
- Whisper and SpeakerKit revisions, assets and their MIT/CC-BY-4.0 notices are recorded in
  [the earlier smoke record](mvp-smoke.md). The same provisioned assets were reused.

External manifests retain source commits, dependency/build configuration, model hashes,
binary hashes, effective geometry, exact argv, audio hashes and detailed timings. Original
RightyO adapter, benchmark code and documentation retain AGPL-3.0-or-later. Distribution
of any upstream runtime/model is a separate notice/license review.

## Repeat with explicitly supplied files

The native binary must be a trusted, separately provisioned build of the pinned runtime.
The adapter passes an absolute existing model path and isolates native HOME/configuration,
avoiding the inspected CLI's model-name download path. It is not a sandbox for arbitrary
user-supplied executables. Put transcript outputs outside all Git checkouts.

```sh
PYTHONPATH=src .venv/bin/python scripts/benchmark_stack.py \
  --audio /private/authored-two-voices.wav \
  --whisper-executable /local/whisper-cli --whisper-model /local/ggml-base.en.bin \
  --nemotron-executable /local/nemo-speech \
  --nemotron-model /local/Nemotron-3-Diarization.q8_0.gguf \
  --backend metal --reruns 3 --timeout-seconds 120 \
  --speakerkit-rttm /private/authored-two-voices.rttm \
  --output /private/new-metrics.json
```

The supplied baseline RTTM must use the WAV filename stem as its recording ID.
`--speakerkit-rttm` is optional. The harness limits file size, duration, reruns and
per-invocation timeouts; it records hashes, hardware, stage times and attribution counts,
with no transcript text, source labels or local input paths. A failed run does not produce
a complete result. Metrics files must be new and existing files are never overwritten.
The harness makes no Jev calls; hosted replay is a separate explicit command in
[the MVP guide](mvp.md). Offline CI exercises adapter/harness failure boundaries with
synthetic stubs; it downloads no model and performs no hosted inference.
