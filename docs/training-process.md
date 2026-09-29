# Model development and training process

This is the plan for a new model repository, not implemented training software. No model downloads, jobs, capture, or paid compute were started during bootstrap.

## 1. Architecture and feasibility decision

Pin a small official Whisper checkpoint family as the first candidate; compare sizes after inspecting local hardware capability and rights. Test frozen-encoder features with a small attention head before encoder fine-tuning. Record windows/stride, history features, score semantics, and the mapping to `attend`, `ignore`, `uncertain`.

Whisper's standard encoder is not a native causal streaming encoder. Only observed audio may enter sliding-window features; pad missing audio without observing future samples. Measure repeated-encoding cost and acquisition latency. An entire recorded utterance evaluated after it ends is an offline baseline, not the target behavior.

## 2. Reproducible execution environment

Create a Python training/inference package with pinned dependencies and locked environment, CPU tests, and explicit Apple Silicon MPS support where verified. CUDA/GPU capability can be added as an optional training backend with budget approval; runtime first targets a Mac. Record framework/device/OS versions, source commit, seeds, numerical settings, model revision/checksum, and dataset manifest hash per run.

Cross-backend results are not necessarily bitwise identical. State deterministic settings and known unsupported operations; measure reproducibility rather than claiming it from a seed alone. CI uses small synthetic fixtures, no live speech, real credentials, downloaded weights, or expensive training.

## 3. Dataset, consent, and annotation

Version a manifest/label schema, not raw speech in Git. Track lawful access, training/redistribution rights, consent scope, speaker/scene IDs, source hashes, labels, onset/offset, ambiguity, annotator disagreement, and deletion requirements. Retain real participant records in protected external storage, not public manifests.

Collect addressed speech and hard negatives: identical words spoken to another human, media, quoted instructions, self-playback, overlapping speakers, accents, and noise. Do not confuse VAD labels with addressee labels. ASR transcripts and licensed teacher-model labels can assist annotation, but human verification and clear provenance are required. Hosted teacher processing is optional and needs explicit permission; Jev cannot ingest audio directly.

Assign speaker/scene-disjoint train, validation/calibration, and sealed test groups before feature generation. Hash/audio-similarity checks prevent duplicates across splits. Cropping and augmentation cannot leak a test speaker/recording into training. Deletion/revoked-consent procedures must address cached features and checkpoints, including whether retraining is necessary.

## 4. Deterministic rig and feature pipeline

Build causal replay, bounded pre-roll, validated event contracts, an ASR port, and an attention port with explicit stream provenance. Test timing with stubbed backends before a real model. Version preprocessing, sample-rate conversion, windowing, padding/masks, feature caches, and augmentations. Features can reveal identity/content; protect them like audio and keep them out of public Git.

## 5. Train a small attention head

Run a tiny smoke training job on explicitly synthetic/licensed fixtures. Then train the selected head on real consented/licensed features with a recorded compute/time budget. Use loss/imbalance handling justified by data, retain ambiguous labels/disagreement, and decide whether ambiguity is a trained class or a calibration/abstention policy. Do not conflate class probability with calibrated confidence.

Use validation for hyperparameters, thresholds, and early stopping. Save resumable checkpoints atomically with source/config/data hashes and optimizer state in external artifact storage. Evaluate multiple seeds and report failure modes. Keep original Whisper frozen initially to isolate what the head contributes.

## 6. Gated ASR integration

An accepted attention event starts transcription from buffered onset, then follows observed live audio. Test short requests, pauses, long turns, back-to-back turns, overlap, playback, and interruptions. No lost initial words, clipped tails, duplicate finals, or invented silence/media transcripts may be hidden in a demo. Silence/non-speech decoding and echo handling need explicit tests; confidence alone does not authenticate a speaker or authorize actions.

## 7. Calibration and held-out evaluation

Measure false activations per ambient hour, misses, abstentions, causal acquisition latency, transcript WER/onset/tail completeness, peak memory, sustained compute, and decoder duty cycle. Compare VAD-only, always-transcribe, and transcript-decision baselines; keep them labeled as comparisons. Freeze operating-point selection on validation before opening held-out test results. See evaluation.md.

## 8. Fine-tuning, distillation, and export

Only if baseline results justify it, selectively fine-tune the encoder or distill a streaming student and quantify quality/resource tradeoffs. Preserve replaceable model endpoints and upstream licensing. Quantization/export candidates must be regression-tested at the actual causal operating point, not only on offline tensors. Document CPU/MPS/runtime support and any retraining needed to change the encoder's temporal assumptions.

## 9. Model registry and open release

Keep weights and private datasets out of Git. Select artifact storage, retention, access controls, versioning, and optional public registry through an explicit issue. Do not provision paid resources implicitly. A public release contains versioned weights/checksums, model card, preprocessing and runtime contract, training source/configs, license/notice inventory, data statement, evaluation reports, hardware settings, and honest reproducibility limitations. Complete the licensing/privacy release gate before uploading anything.

## 10. Live pilot and Hailing Station integration

After feasibility, conduct opt-in visible live testing on a Mac with immediate stop and no default recording. Hailing Station integration is a separate adapter issue: host-side attention, authenticated encrypted audio transfer if needed, session selection and source provenance, client speech-recognition fallback, and deliberate capture ownership. Do not put a semantic model on today's iPhone/iPad or remove the current on-device STT workflow.

## Independent-session handoff

Read AGENTS.md, README.md, architecture.md, licensing.md, evaluation.md, and this process. Start with the architecture/license audit and reproducible environment issues; do not start long training or publish a checkpoint first. Follow issue-linked PRs, exact-head independent review, and small-budget verification. Keep Hailing Station changes in its own repository/PR flow.
