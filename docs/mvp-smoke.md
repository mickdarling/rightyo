# First MVP smoke record

Date: 2026-09-29. These are integration checks using authored synthetic inputs, not a
held-out benchmark or live conversational result. No real conversation or microphone
capture was used. No credential value was returned to tooling.

## Hosted transcript decisions

The installed Jev boundary successfully called the official endpoint with pinned
`jev-1.13.0`. Four committed turns produced four requests, each batching attention and
recipient questions; a partial event made no call. Default threshold: 0.7. No retries.

| Input | Requests | Provider total | Per-call range | Final policy labels |
| --- | --- | --- | --- | --- |
| Authored speaker-tagged fixture | 4 | 1005.589 ms | 201.906–320.782 ms | 4 uncertain |
| Same fixture with speaker labels hidden | 4 | 1007.636 ms | 185.753–307.495 ms | 4 uncertain |

With speaker labels the raw attention choices were attend, ignore, uncertain, uncertain.
Without labels they were attend, uncertain, uncertain, uncertain. The conservative policy
abstained throughout because confidence, unknown recipient or cross-question consistency
did not meet its requirements. The first raw attend had attention confidence 0.39 and
recipient confidence 0.23, so a high threshold did not admit it.

This proves connectivity, authentication, structured response parsing and abstention behavior.
It does not prove useful sensitivity or a speaker-label benefit. Do not lower thresholds to
make these four examples pass and call that calibration. A separate held-out evaluation must
select an operating point and count false activations, missed activations and abstentions.
These timings exclude speech decoding, diarization and utterance finalization. They are
single-run network timings and cannot establish end-to-end live latency.

## Local model provenance

whisper.cpp runtime: v1.9.4, commit `927cfce34f31707e17f2bff35c349632fb9e2c3a`,
compiled locally with Apple Silicon CPU/Accelerate/Metal support. Model: `base.en`,
147,964,211 bytes, from `ggerganov/whisper.cpp` revision
`5359861c739e955e79d9a303bcbc70fb988958b1`, SHA-256
`a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002`.
The model and local provenance manifest are outside Git. Runtime and model retain their
upstream MIT notices. Acquisition is separate from ASR runtime validation.

## Real local ASR and diarization on generated speech

The Mac test used authored text rendered by built-in macOS voices, with all WAV/RTTM/turn
files outside Git. A 4.606-second single-voice clip passed Whisper import in 0.366 seconds.
It produced one finalized transcript turn with unknown speaker attribution.

Argmax OSS v1.1.0, commit `1e2a163736dfa5a198e637ae44c114e1c6d5cc2d`, was built
with Swift 6.2 using Swift 5 language compatibility. Strict Swift 6 compilation failed in
an unrelated upstream TTS Sendable boundary; the compatibility build used unmodified sources.
SpeakerKit assets were pinned to `argmaxinc/speakerkit-coreml` revision
`556fc52a13327837688f02289457cded017802e9`: 24 files, 11,243,910 bytes, LFS SHA-256
and small-file Git blob hashes checked, attribution manifest outside Git. The model assets
retain their upstream CC-BY-4.0 terms; they are not redistributed in this repository.

On a 23.697-second Samantha/Daniel/Samantha clip, the actual diarizer automatically found
two speakers and produced nine RTTM spans with stable A/B/A ordering. Warmed whole-file
diarization took 0.305 seconds. Whisper plus the RTTM importer took 0.622 seconds and
produced seven turns. Only one retained a speaker label; six were unknown because full
ASR-segment coverage crossed pauses or diarizer boundaries. This reveals the need for
word/turn alignment rather than relaxing uncertainty silently.
The follow-up is [#27](https://github.com/mickdarling/rightyo/issues/27).

The imported seven turns then made seven Jev calls: 1887.756 milliseconds total, individual
calls 176.696–343.286 milliseconds. At the unchanged 0.7 threshold two attended and five
abstained. This exercises local audio → real diarization timeline → transcript → hosted
decision plumbing. It is not a held-out accuracy, live latency, energy or causal streaming
benchmark. All model/file processing was offline; no captured conversation was used.

See [the MVP guide](mvp.md), [evaluation](evaluation.md), and
[#25](https://github.com/mickdarling/rightyo/issues/25) for the comparison work remaining.
