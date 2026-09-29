# Continuous integration

The baseline workflow runs on every PR, including drafts, documentation-only changes,
dependency updates and forks, and on pushes to main. It uses ephemeral Linux runners,
read-only repository permission, no repository secrets, no credentialed checkout,
timeouts and cancellation of superseded runs. No path filter suppresses its aggregate.

The stable status is `rightyo/ci`. It explicitly requires `rightyo/repository` and
`rightyo/traceability` to succeed; a failed, cancelled or skipped child fails it. A cancelled
workflow does not count as passing. Configure protection against this actual status after
the workflow has run; a file in Git does not establish branch protection.

## Local verification

Use Python 3.11 or newer. Python 3.11 is the CI baseline. Initial setup explicitly downloads
small development tools from PyPI and the official actionlint release; it downloads no
models or datasets. The actionlint archive's SHA-256 is checked before installation.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements-ci.txt
.venv/bin/python scripts/install_actionlint.py --dest .venv/bin/actionlint
.venv/bin/python scripts/verify.py --actionlint .venv/bin/actionlint
```

The last command is the same verification entry point CI uses. It checks tracked files;
stage intended additions with `git add` before running it. It validates repository metadata,
the declared license, ignore rules with positive and negative examples, tracked artifact
paths/extensions/size/binary content, simple Markdown whitespace/fences and local-link
existence, YAML/JSON syntax, baseline workflow policy, actionlint, Ruff and discovered
unittest cases and prototype JavaScript syntax (Node.js 20+ required). Optional host shellcheck/pyflakes integration is disabled for parity.

Once `pyproject.toml` and `src` exist, the same command builds a wheel with the pinned
local backend, installs it without dependencies into temporary storage, and runs the
explicit mock evaluation on `examples/synthetic-turns.json` from outside the source tree,
plus a check that the installed prototype includes its browser assets.
Before a package exists it states that package checks are inapplicable. Real-model accuracy,
Mac/MPS behavior and hardware performance are not established by these checks.

PRs link an issue using a line such as `Refs #15` or
`Refs https://github.com/mickdarling/rightyo/issues/15`. The metadata job checks out only
the base revision and parses the event as data, without running PR-controlled programs.
The bootstrap uses an inline syntax parser while main lacks the helper. Local parity for
an event JSON file is:

```sh
python3 scripts/check_pr.py --event /path/to/pr-event.json --repository mickdarling/rightyo
```

The linkage check validates syntax, not issue existence or completed independent reviews.
CI logs the PR head/base separately from the checked-out integration revision. A PR normally
tests GitHub's merge revision; independent reviews still bind to the exact PR head.

## Artifact checks and limits

Recordings, transcripts, consent records, feature caches and model artifacts belong outside
Git. Unknown extensions in protected directories are rejected; tracked files are inspected
even if ignore rules would hide new files. Saved notebook outputs, binary files and files
over 1 MiB require a deliberately reviewed policy change. Synthetic JSON/source examples and
`.env.example` remain trackable. Failures print an entry index/category instead of data or
potentially identifying paths. Ignore rules and these checks cannot prove that arbitrary
text lacks private content or that data consent exists.

The workflow policy intentionally allows only the secretless Linux baseline. A future
privileged review or hardware workflow needs a separate reviewed policy boundary; simply
adding an exception to pass CI is insufficient. Actions use full commit SHAs, Python tools
use explicit versions, and Dependabot proposes bounded weekly updates. There are no shared
caches or uploaded artifacts in this baseline.

## Remaining enforcement work

This implements the first layer of [#15](https://github.com/mickdarling/rightyo/issues/15)
and narrow checks from [#14](https://github.com/mickdarling/rightyo/issues/14) and
[#16](https://github.com/mickdarling/rightyo/issues/16). Native secret scanning, dependency
advisory audits, a full static type checker and external-link network checks remain follow-up
work. No broad privacy/security certification is claimed.

The dual Codex/Claude adapters in [#17](https://github.com/mickdarling/rightyo/issues/17)
and trusted review gate/protections in [#18](https://github.com/mickdarling/rightyo/issues/18)
remain separate. PR-modified workflow/code can alter ordinary test results; this baseline
does not solve status spoofing or enforce independent approval. The initial PR needs actual
independent exact-head review before merging. Merge queues and hardware lanes are not
configured. There are no live recordings, remote inference, training or model downloads.

See [GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
for the trust boundaries and
[actionlint documentation](https://github.com/rhysd/actionlint/blob/v1.7.7/docs/usage.md)
for workflow validation.

The microphone helper is additionally compiled and ad-hoc signed locally on macOS.
A secretless Mac build lane is tracked in [#38](https://github.com/mickdarling/rightyo/issues/38);
its workflow policy change needs review before adding it to the required aggregate.
