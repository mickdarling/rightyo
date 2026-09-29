# Evaluation plan

The first milestone is a reproducible feasibility result, not a live demo presented as a reliable attention model.

## Corpus and labels

Use licensed public data or consented recordings stored outside Git. Track consent, permitted uses, source license, deletion requirements, speaker/scene grouping, and manifest version. Do not collect ambient conversations from unsuspecting people. Begin with explicit sessions and controlled examples.

Labels: `addressed`, `not_addressed`, and `ambiguous`. Include speech onset/offset and the earliest point at which addressedness is reasonably inferable. Obtain multiple judgments for ambiguous samples and retain disagreement; do not force every sample into a binary answer. The same words spoken to a person and to the system should appear as contrastive examples.

Separate train, calibration, and held-out evaluation by speaker and recording scene. Streaming evaluation must reveal only frames already received. Offline utterance classification cannot establish real-time detection performance.

Include direct questions, implicit follow-ups, dictation, commands without assistant names, ordinary nearby conversations, other humans' questions, TV/podcasts, quoted commands, assistant playback, overlapping speakers, noise, accents, pauses, quiet speech, and interruptions. Annotate provenance; synthetic or replayed samples are never reported as live user speech.

## Required measurements

| Measure | Why it matters |
| --- | --- |
| False activations per ambient hour | Reject ordinary conversations, media, and self-playback |
| Missed addressed utterances | Measure requests the system fails to hear |
| Abstention rate and ambiguity handling | Avoid confident guesses with insufficient evidence |
| Attention latency distribution | Measure time from annotated inferable point to accepted attention |
| First/last-word completeness | Catch onset and tail truncation in the gated ASR path |
| WER on attended utterances | Measure transcription quality independently of attention accuracy |
| Compute, peak memory, and decoder duty cycle | Test the compact, efficient system hypothesis |
| Stop/restart and duplicate-event behavior | Establish pipeline reliability |

Report thresholds, model versions, hardware, window/pre-roll settings, sample counts, confidence intervals where supported, and class distributions. Compare against VAD-only, always-transcribe, and transcript-based decision baselines. VAD detects speech presence, not who it addresses. A transcript baseline is useful but does not demonstrate attention before transcription.

No accuracy, latency, memory, or battery targets have yet been demonstrated. Set release gates from measured baselines and intended risk tolerance, document them before held-out evaluation, and report failures alongside successes.

## Gate sequence

1. Audit existing audio-addressedness methods, accessible weights/code, datasets, and their licenses.
2. Build a deterministic streaming replay and buffer rig with stubbed attention/ASR backends; prove timing and complete selected audio delivery.
3. Run a real audio attention candidate against held-out streams and compare baselines.
4. Add gated ASR and measure whether the beginning and end survive attention acquisition and endpointing.
5. Conduct explicitly consented live Mac sessions with stop/visibility and no default recording.
6. Consider Hailing Station integration only after feasibility and authenticated transport/privacy controls are satisfied.
