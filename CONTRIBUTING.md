# Contribution workflow

Use an issue to record the problem, scope, acceptance criteria, dependencies, and verification plan before implementation. Link work in a focused PR. Do not use role-based story boilerplate unless it genuinely helps.

Every implementation PR must identify its issue, describe changes and verification, and record an independent review against the exact current head. Resolve valid findings, then obtain a fresh review if code changes. Missing or failed review is not approval. Do not merge your own unchecked change or bypass repository protections. Mechanical enforcement is a tracked setup task; this bootstrap does not claim branch protection or CI already exists.

Keep experiments reproducible. Distinguish stubs, synthetic examples, prerecorded replay, and live speech. Do not claim a research hypothesis is implemented or a benchmark result is a physical-device test.

Never commit recordings, private transcripts, credentials, source/device identifiers, or unreviewed downloaded weights. License review covers code, datasets, and weights separately. Every issue/PR is public; use sanitized summaries and artifact hashes rather than private content.

Contributions of original code/documentation use AGPL-3.0-or-later. Preserve upstream notices. Changes to trained weights require the model-release checklist, not only a code review. See docs/licensing.md and docs/training-process.md.

The initial commit is a design-only bootstrap. Subsequent changes follow the issue/PR workflow above.
