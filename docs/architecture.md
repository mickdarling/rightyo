# Architecture: attention, then transcription

## Scope and assumptions

The public-facing result is a transcript of speech the system has elected to attend to. Internal attention decisions are tightly typed and separate from application actions. There is no required trigger phrase and no open-ended language generation in the attention component.

Audio can be intrinsically ambiguous: the same words may address a person or a device. The detector must support abstention and evaluation of failures. A valid type or high score is not proof of correct addressee detection.

Initially run a local, opt-in research rig on a Mac. Hailing Station's mobile clients retain their existing on-device speech recognition. Connecting a mobile audio stream to this separate rig requires a later explicit integration, privacy, and transport decision.

## Replaceable boundaries

1. Audio source provides timestamped PCM frames and an explicit source identity.
2. A bounded in-memory buffer holds recent frames; expired frames are discarded without archival.
3. An attention backend analyzes causal audio windows and returns a typed decision. Optional bounded context may describe an expected reply or playback activity; it must not silently introduce a full transcription prerequisite.
4. A deterministic controller applies thresholds, debounce/hysteresis, session ownership, and termination rules.
5. An ASR backend receives the selected buffered onset and subsequent live frames, producing partial and final transcripts.
6. A consumer receives transcript events. Tool execution and reply generation stay downstream.

Attention and ASR implementations are separately injectable. The first candidate uses a small Whisper audio encoder plus a trained attention head, followed by Whisper decoding only for selected audio. A later shared encoder/decoder packaging is a research option, not a claim of existing code. Evaluate frozen features and head training before full encoder fine-tuning or training from scratch.

Standard Whisper encoding/transcription uses fixed audio windows rather than an inherently causal streaming encoder. An experiment may re-encode sliding windows made only from already received audio, with measured cost. No future frames may enter a real-time decision. Establish latency, padding, window length, history, and memory behavior explicitly; an offline whole-utterance score is not evidence of real-time attention acquisition. Preserve a replaceable backend if a smaller distilled streaming student becomes necessary.

## Proposed contract (not implemented)

```typescript
type AttentionLabel = "attend" | "ignore" | "uncertain";

interface AttentionDecision {
  schemaVersion: 1;
  streamId: string;
  windowStartMs: number;
  windowEndMs: number;
  evaluatedThroughMs: number;
  label: AttentionLabel;
  score: number; // [0, 1]; backend-specific until separately calibrated
  backendId: string;
  modelVersion: string;
  proposedSpeechStartMs?: number;
}

type TranscriptEvent = {
  schemaVersion: 1;
  streamId: string;
  utteranceId: string;
  revision: number;
  speechStartMs: number;
  speechEndMs: number;
  text: string;
  kind: "partial" | "final";
  onsetTruncated: boolean;
  recognizerId: string;
};
```

All times use the same stream-relative monotonic clock. Reset timestamps only with a new stream identity. Validate finite scores, label allowlists, timestamp order, maximum frame/event sizes, revisions, and provenance at the boundary. Never accept executable instructions from a model result.

`proposedSpeechStartMs` cannot reach outside the retained buffer. If the true onset is older, record truncation instead of claiming complete capture. Stable utterance IDs and monotonic revisions prevent partial/final duplication; finality is emitted only once per utterance. Explicit errors/cancellation must be distinguishable from a final transcript in the implemented contract.

## Controller lifecycle

An explicit user start enables observation. `ignore` continues observation without ASR decoding. `uncertain` abstains while bounded buffering continues. An accepted `attend` opens one utterance: feed retained onset once, then live frames without overlap or gaps.

While transcribing, use a tested endpointer and hangover interval to retain sentence tails. Do not end capture solely because a later attention window becomes uncertain. Explicit stop, maximum utterance duration, stream failure, permission revocation, and interruption have defined cancellation/finalization policies. All buffers are bounded; source stop clears them.

Thresholds and pre-roll length are experimental configuration, not universal fixed constants. Measure acquisition latency and retained onset together. Debounce must not hide short requests. Rapid back-to-back utterances, pauses, interruptions, and restarts need deterministic tests.

## Playback, speakers, and authority

Assistant playback and television are hard negative examples. Pass playback reference/timing where available and evaluate acoustic echo handling; never interpret the application's own speech as fresh user authorization. Addressee detection and speaker identity are different tasks. Enrollment is optional future work, not assumed identity verification.

The first consumer explicitly chooses a target host/platform/session. A transcript is input content, not authority to perform privileged operations. Attention confidence cannot authorize destructive actions. Multiple sources require provenance and a single capture owner per intended interaction to avoid duplicate delivery.

## Resource and privacy limits

Observation is always opt-in with a visible running/stopped indicator and immediate stop. No persistent recording or transcript logging by default. Require consent and retention controls for an evaluation corpus. Keep raw data outside Git. A hosted backend requires an explicit data-processing choice; a remote audio source additionally requires authenticated encrypted transport and source authorization. These requirements precede ambient-stream integration, not follow it.
