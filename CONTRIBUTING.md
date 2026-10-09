# Contribution workflow

Use an issue to record the problem, scope, acceptance criteria, dependencies, and verification plan before implementation. Link work in a focused PR. Do not use role-based story boilerplate unless it genuinely helps.

Every implementation PR must identify its issue, describe changes and verification, and record an independent review against the exact current head. Resolve valid findings, then obtain a fresh review if code changes. Missing or failed review is not approval. Do not merge your own unchecked change or bypass repository protections. Mechanical enforcement is a tracked setup task; this bootstrap does not claim branch protection or CI already exists.

Keep experiments reproducible. Distinguish stubs, synthetic examples, prerecorded replay, and live speech. Do not claim a research hypothesis is implemented or a benchmark result is a physical-device test.

Never commit recordings, private transcripts, credentials, source/device identifiers, or unreviewed downloaded weights. License review covers code, datasets, and weights separately. Every issue/PR is public; use sanitized summaries and artifact hashes rather than private content.

Contributions of original code/documentation use AGPL-3.0-or-later. Preserve upstream notices. Changes to trained weights require the model-release checklist, not only a code review. See docs/licensing.md and docs/training-process.md.

The initial commit is a design-only bootstrap. Subsequent changes follow the issue/PR workflow above.

## Voice data never enters the repository

No voice sample, speaker embedding or voiceprint of a real person may be committed, attached to an issue or PR, uploaded as a CI artifact, or logged ([#109](https://github.com/mickdarling/rightyo/issues/109)). `scripts/verify.py`, which CI runs, refuses any tracked file with an audio, embedding or model extension (`.wav`, `.pcm`, `.flac`, `.mp3`, `.m4a`, `.aac`, `.ogg`, `.opus`, `.caf`, `.aif`/`.aiff`, `.npy`, `.npz`, `.emb`, `.pt`, `.pth`, `.onnx`, `.mlmodel`, `.mlpackage`, `.safetensors` and similar) and any path segment named `enrollment`, `enrollments`, `voiceprint` or `voiceprints`, unless the path sits under an entry in `SYNTHETIC_VOICE_FIXTURES` in `scripts/repository_checks.py`. That allow-list is empty today because no tracked file is voice data. Adding an entry takes a reviewed PR that records why everything under it is synthetic (an authored signal or a TTS voice). The binary and size rules still apply to allow-listed paths. The repository has no pre-commit hook; run `scripts/verify.py` before pushing.

Enrollment data lives only on your machine. `rightyo enroll` ([speaker enrollment](docs/speaker-enrollment.md), [#137](https://github.com/mickdarling/rightyo/issues/137) step 3) stores it in `~/Library/Application Support/RightyO/enrollment/` (mode 700, files 600) and provides `delete --id` and `delete --all`. Its tests use authored signals generated in the test and a stand-in embedder, never a recorded voice. To remove everything at once without RightyO, delete that directory: `rm -rf ~/Library/Application\ Support/RightyO/enrollment`. Also remove any enrollment files you kept in the gitignored `local/` directory.
