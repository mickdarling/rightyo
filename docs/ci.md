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
unittest cases. Optional host shellcheck/pyflakes integration is disabled for parity.

Once `pyproject.toml` and `src` exist, the same command builds a wheel with the pinned
local backend, installs it without dependencies into temporary storage, and runs the
explicit mock evaluation on `examples/synthetic-turns.json` from outside the source tree.
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

Ordinary CI remains secretless. The narrowly specified AI review workflow is a separate
privileged boundary, validated by structural policy and mutation tests. Hardware workflows
still need their own reviewed policy boundary. Actions use full commit SHAs, Python tools
use explicit versions, and Dependabot proposes bounded weekly updates. There are no shared
caches or uploaded artifacts in this baseline.

## Remaining enforcement work

This implements the first layer of [#15](https://github.com/mickdarling/rightyo/issues/15)
and narrow checks from [#14](https://github.com/mickdarling/rightyo/issues/14) and
[#16](https://github.com/mickdarling/rightyo/issues/16). Native secret scanning, dependency
advisory audits, a full static type checker and external-link network checks remain follow-up
work. No broad privacy/security certification is claimed.

The dual reviewer workflow implements the Actions layer of
[#17](https://github.com/mickdarling/rightyo/issues/17); deployment and credentials must be
verified on real PRs. The stronger independent publisher identity in
[#18](https://github.com/mickdarling/rightyo/issues/18) remains open. PR-modified ordinary
workflows can spoof statuses under the shared GitHub Actions app identity; these check names
alone do not solve that attack. Merge queues and hardware lanes are not configured. CI
does not capture audio, acquire models or train; reviewer jobs send bounded public source
to their explicitly configured providers.

See [GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
for the trust boundaries and
[actionlint documentation](https://github.com/rhysd/actionlint/blob/v1.7.7/docs/usage.md)
for workflow validation.


## Claude and Codex PR checks

After the reviewed workflow is installed on the default branch, `ai-review.yml` triggers
for opened, updated, reopened, edited and ready PRs, including drafts and documentation
changes. Ordinary `rightyo/ci` continues independently. It publishes three check runs
on the exact PR head: `rightyo/codex-review`, `rightyo/claude-review`, and
`rightyo/review-gate`. Target-event workflow jobs themselves run against the trusted base,
so the explicit head check runs are necessary for visibility in the PR Checks tab.

The preparer uses the immutable workflow revision, reads PR metadata and public source
through GitHub's API, and never checks out or executes PR programs. Both reviewers receive
the same digest-checked snapshot of complete before/after changed files and trusted policy.
Immutable Git tree modes reject symlinks and submodules before any blob content is fetched;
the immutable comparison determines file coverage rather than a moving PR endpoint.
Coverage is bounded to fewer than 100 files and 900,000 encoded bytes; an oversized,
private-path, binary, non-UTF-8 or unsupported change fails rather than silently truncating.
This source filter cannot prove that arbitrary public text lacks secrets or private content.

Codex uses the official pinned Action and matching CLI/proxy 0.159.2. Its fresh configuration
turns off command/image/app/plugin/MCP-related capabilities, denies reads outside minimal
system runtime files, denies writes and command network access, and drops elevated privileges.
A Linux control-command/canary-read probe runs before provider credentials are supplied.
Codex retains its patch tool registration; filesystem denial is the boundary, not a claim
that every tool is absent. Claude uses the official locked CLI 2.1.285, without the Action's
GitHub MCP integration: restricted mode, no tools or MCP, no settings sources, no persistent
session, empty working directory, and an environment excluding GitHub credentials.

Reviewers cannot publish checks. A separate trusted publisher validates successful jobs,
structured completion, exact head/base, snapshot digest, run provenance and complete changed
file coverage, then refetches current head/base before updating checks. Missing credentials,
timeout, quota failure, cancelled/skipped jobs, malformed results or stale revisions fail.
Priorities 0–2 block; priority 3 findings remain visible. A no-findings response counts only
after completed coverage. This is static AI review, not proof of correctness or reproduced
hardware measurements. The check summary records limitations and immutable identities.

Each provider has a ten-minute job timeout; Claude inference is bounded to five minutes,
two turns and a $3 API budget. Codex uses a bounded snapshot and no agent command execution;
set a separate OpenAI project spending limit because job duration is not a dollar cap.
There are no automatic provider retries. Same-repository updates cancel superseded reviews.
Automatic fork events queue behind a maintainer dispatch and preserve existing checks for
the exact head/base; metadata edits do not cancel or overwrite an authorized review.
Source snapshots and validated result artifacts expire after one day; no raw CLI diagnostic
logs or credential-bearing artifacts are uploaded.

Automatic credentialed review requires a same-repository PR and a maintainer trigger
(or the explicitly allowed Dependabot identity). Dependabot reviews use a trusted
`workflow_run` callback after its ordinary CI completes, avoiding target-event credential
restrictions; the source run, PR author and current head are checked before snapshotting.
Non-bot callbacks do not cancel or displace real reviews. Forks receive failing review checks until
a maintainer runs **Independent AI reviews → Run workflow** with the current PR number.
Dispatch rechecks maintainer permission and binds the snapshot to the head/base observed at
that run; an update invalidates it and requires a new dispatch. Do not use persistent fork
approval labels. The same dispatch can backfill existing PRs, including stacked PRs whose
base branch predates this workflow. Base/head are checked again at publication; stronger
continuous base invalidation and exclusive publisher attestation remain #18 work.

### Credentials and rollout

Configure repository secrets `OPENAI_API_KEY` and exactly one of
`CLAUDE_CODE_OAUTH_TOKEN` / `ANTHROPIC_API_KEY`. The OpenAI Action uses API billing;
Claude subscription authentication uses a token generated locally with `claude setup-token`.
Never paste either value into a PR, terminal command argument or chat. On this Mac:

```sh
python3 scripts/setup_review_secrets.py --claude-auth subscription
# Alternatively: --claude-auth api
```

The native dialog has masked fields and explicitly names GitHub.com / `mickdarling/rightyo`.
Its Store button pipes values directly into `gh secret set`, discarding CLI diagnostics and
keeping values out of files, arguments and environment. Cancel performs no writes. It does
not read existing local credentials or reuse the Jev key. Compilation checks are available
with `--check`. The owner must already authenticate GitHub CLI. If changing Claude auth modes,
remove the obsolete alternate GitHub secret yourself; ambiguous dual credentials fail.

Install through an issue-linked, independently reviewed PR. Bootstrap can use manual
exact-head reviews because target/dispatch workflows must first exist on the default branch.
Then configure secrets, dispatch existing PRs, verify actual completed head check runs, and
add `rightyo/review-gate` to branch protection after it has run. A workflow file or local
review does not establish active GitHub checks. Keep #17 open until live setup is verified,
and keep #18 open until its independent identity and protection tests are implemented.

References: [Codex Action](https://learn.chatgpt.com/docs/github-action),
[Codex permission profiles](https://learn.chatgpt.com/docs/permissions),
[Claude authentication](https://code.claude.com/docs/en/github-actions).
