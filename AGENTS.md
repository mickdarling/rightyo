# Agent instructions

Read README.md, CONTRIBUTING.md, SECURITY.md, and the architecture, training-process, evaluation, and licensing documents before implementation.

This is a design-only, public Whisper-based audio-attention model research bootstrap named RightyO (repository rightyo). Do not imply a model is trained, inference is implemented, or performance is validated without evidence. Current Hailing Station iPhone/iPad on-device STT remains unchanged; mobile semantic deployment is out of scope.

Use issue-linked PRs for changes after bootstrap, exact-head independent reviews, and honest verification records. Do not treat failed reviews as approval or bypass checks. Preserve source provenance and distinguish replay/stubs from live speech.

Use replaceable attention and ASR boundaries. Keep application routing/tool execution downstream. No implicit live capture, hosted API calls with real speech, dependency/model downloads, or ambient recording. Secure transport and explicit consent precede remote continuous audio experiments.

Never add raw audio, private transcripts, credentials, or model weights to Git. The repository is public and original code/docs are AGPL-3.0-or-later. Released project-owned weights are intended to use the same license after their release audit; upstream notices and dataset rights remain separate. Read the licensing policy. Do not publish a checkpoint or upload datasets without a release/consent decision.

Use small explicit training budgets, versioned configs and manifests, and resumable checkpoints. No paid GPU provisioning, large downloads, live capture, or lengthy training jobs are authorized by this planning bootstrap. Start with local CPU/MPS smoke tests and causal replay. Record backend limitations rather than claiming bitwise reproducibility across different hardware.
