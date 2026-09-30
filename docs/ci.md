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
Released actionlint parsers, including current v1.7.12, do not recognize GitHub's new
`concurrency.queue` field. Verification keeps the existing checksummed v1.7.7: repository
policy first validates the exact reviewed workflow mutex, then actionlint reads that one
workflow through stdin with only its literal `queue: max` line removed. All other source
bytes and other workflows are linted normally. Invalid queue values, missing cancellation
policy, other scopes or added executable surface fail before normalization. Remove this
narrow compatibility path once a pinned official release supports the field.

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

Ordinary CI remains secretless. The subscription review gate is a separate metadata-only
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

The subscription review gate implements the current Actions layer of
[#17](https://github.com/mickdarling/rightyo/issues/17); native subscription setup and completed
reviews must be verified on real PRs. The stronger independent publisher identity in
[#18](https://github.com/mickdarling/rightyo/issues/18) remains open. PR-modified ordinary
workflows can spoof statuses under the shared GitHub Actions app identity; these check names
alone do not solve that attack. Merge queues and hardware lanes are not configured. CI
does not capture audio, acquire models or train. The native Codex GitHub integration reviews
public repository source through the connected account. The gate makes no model requests.

See [GitHub's status-check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks),
[commit-status API](https://docs.github.com/en/rest/commits/statuses),
[GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
for the trust boundaries and
[workflow concurrency and queue limits](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#concurrency),
[actionlint documentation](https://github.com/rhysd/actionlint/blob/v1.7.7/docs/usage.md)
for workflow validation.


## Subscription-backed Codex PR reviews

Codex reviews use the built-in GitHub integration connected to the owner's Codex subscription,
not `openai/codex-action`. Enable RightyO in Codex code-review settings, select **Review all
PRs**, and select **On every push**. Automatic native reviews and manual `@codex review`
requests use the subscription's review allowance. No `OPENAI_API_KEY` is required by any
active workflow. Separately purchased credits, quota behavior and account preferences remain
managed in Codex settings; the workflow does not grant extra allowance.

`ai-review.yml` now runs only a metadata validator. It checks native results after trusted
PR updates, issue-comment creation/edit/deletion, and completed ordinary PR CI or the read-only native review activity relay. It never
runs or checks out PR-controlled programs, installs a model CLI, requests inference, or
passes provider credentials. Its checkout is the immutable trusted workflow SHA. Review and inline-review events run a separate fixed notification relay with empty
permissions, no checkout, artifacts, credentials or event-controlled commands. The privileged
publisher subscribes only to its completion callback and refetches the source run through
GitHub API, validating repository, workflow name/path, event, completion and unique PR
association. A fixed repository-wide workflow concurrency group serializes the entire
bootstrap, resolver, recorder and publisher pipeline. GitHub's `queue: max` retains up to
100 waiting runs with cancellation disabled, rather than replacing a single pending run.
Every status writer shares that lock, so an older publisher cannot overwrite a newer
pipeline's pending denial after collecting its evidence. The read-only resolver and
publisher independently re-resolve the same numeric PR before status writes; there is no
PR-specific or job-level lock outside the shared workflow scope. Relay outputs never supply
a verdict and the relay's PR merge tree never executes in the privileged lane. Queue order
follows waiting start, not event creation; all callbacks reread current metadata and durable
barriers. Overflow beyond 100 waiting runs is canceled by GitHub and requires a later
recheck; this bounded service limit remains part of #18.
Immediately before a success status POST, the publisher validates its own active Actions
run and inventories every documented unfinished lifecycle of this same trusted workflow.
Any other queued, requested, waiting, pending or active run withholds approval, including
callbacks for other PRs; IDs, timestamps and event order are not used as a freshness proxy.
Malformed metadata, API failure or bounded pagination overflow also block success. The
queue can then drain and a later callback can revalidate current native evidence. If the
last queued callback belongs to another PR, manually recheck the suppressed PR after the
queue drains; suppression does not manufacture a fresh native review.
This guard removes the deliberate queued-denial delay after known callback arrival.
GitHub's inventory and status POST remain separate, non-atomic API operations: an event can
arrive between the final read and write, and webhook/API propagation is asynchronous. The
adapter does not claim a zero-window synchronous gate; that stronger guarantee remains #18.
Completion comments and summary updates also supply the trusted comment-event route. A manual dispatch rechecks a PR using its full current head and base
SHAs, without requesting or running another model.

The stable `rightyo/review-gate` commit status is published on the exact current PR head.
It starts pending before validation, succeeds only with positive native evidence, and reports
failure/error for missing evidence, findings or validation errors. Its link opens the actual
Actions run with detailed head/base, native evidence and limitations in the log. The publisher
has only `statuses: write`, alongside repository metadata read access; it cannot create
check runs. Passing requires
positive evidence from the verified connector bot identity and GitHub app, through either:

- An explicit **Codex Review: Didn't find any major issues.** completion comment with its
  reviewed commit uniquely resolved through GitHub's commit API to the full current head.
- An authentic summary containing exactly one **Code Review / Completed** row for the
  uniquely resolved current head, together with a genuine connector thumbs-up reaction on
  the PR created at or after that completion. The app's own summary describes thumbs-up
  as completion without findings; this path is verified against RightyO's automatic review.

Neither a request, summary alone, reaction alone, copied verdict nor old-head evidence
passes. All native app-authored comments are checked for edit provenance, including metadata whose
header was removed or rows malformed. Human edits or ambiguous native summaries fail closed
even when an older explicit clean verdict remains. A current authenticated Completed review
summary is required for both positive paths; a deleted or Running cycle indicator cannot
resurrect a legacy verdict. Every accepted comment's exact body, update time and identity are checked through
GitHub GraphQL. Explicit verdicts must never have been edited; summaries may be unedited
or last edited by the connector's immutable Bot identity. A maintainer-edited bot comment
cannot supply positive evidence. A new authorized first-line `@codex review` / `@codex security review` request invalidates
older clean evidence during validation. A separate recorder runs within the shared workflow mutex before publication and
persists authorized request time plus comment ID as a typed pending marker in immutable
`rightyo/review-gate` commit-status history. It can write only denial markers and gate pending, never approval.
Editing or deleting the command cannot erase that barrier. Both the requester and event
actor need current repository write/maintain/admin permission. Removed commands retain an
already-recorded timestamp, so cleanup after a completed review does not demand another
review; if initial capture was missed, the original event payload gives a conservative cutoff.
Both bounded best-effort writes use the same required-context pending marker: any successful
write simultaneously blocks approval and preserves the cutoff. Fresh validated success can
replace the latest required status without erasing its immutable request-marker history. Authorization/history API failures conservatively
persist denial and fail the recorder; known unauthorized or benign comments do not create
permanent markers. If both writes are unavailable, capture fails and cannot approve; durable
external atomic publication remains an #18 limitation.
The publisher rereads bounded marker history on every evidence collection, using the maximum
trusted Actions marker cutoff. Malformed trusted history or capture failure blocks approval. Current write/maintain/admin permission is checked through
GitHub; quoted examples and unauthorized requests do not establish a review barrier. A
current-head native Running, Queued or unknown activity summary also invalidates older
verdicts. Fresh completion must postdate the barrier at whole-second precision.
Formal connector reviews bind to their immutable reviewed commit. Inline findings bind to
`original_commit_id` and the matching authenticated parent review, because GitHub can move
their rendered `commit_id` forward when an unchanged line survives later commits. Missing,
malformed or inconsistent original/parent provenance fails closed. Old findings forwarded
to a new head do not impersonate a review of that head.
Current-head connector reviews or original inline findings block
even after thread resolution or dismissal: fix on a new head and obtain a fresh review.

Summary completion times include fractions of a second while reaction timestamps use
whole seconds; comparison floors the completion time to that same precision. PR reactions
are not cryptographically bound to a commit, so this is a conservative observed native
protocol adapter, not immutable review attestation. Stronger verification remains #18 work.
The workflow briefly retries metadata up to three times at five-second intervals because
native reactions can appear after the completion-comment event; it never retries inference.
Finding streams and comment provenance are refetched before positive publication. If the
native format changes, the reaction is delayed beyond this window, or review is incomplete,
the gate blocks; a later trusted event or exact-revision manual gate dispatch can recheck.
Unknown formats, quota failure, API/GraphQL errors and bounded-history overflow fail closed.
Current head/base are refetched before publication; revision changes invalidate the attempt.
Review/inline-review callbacks recheck findings even when a failed re-review posts no issue
completion comment. Callback PR association must be verified on real review and inline
activity before rollout; unknown or ambiguous source metadata fails instead of guessing.

Before any fallible callback routing or checkout, a trusted inline bootstrap validates the
review-relay webhook's repository/name/path/event and full GitHub-owned revision hints, then
publishes required gate pending. It has only status-write permission and cannot approve.
The resolver depends on successful bootstrap invalidation, so source-run API failure or
resolver timeout cannot leave a previous green status after new review activity. Hint data
provides denial authority only; later routing, exact-head provenance and native positive
proof still require authoritative API validation. The documented webhook schema and live
relay run metadata establish these fields; actual default-branch callback execution remains
a rollout verification. Status-API unavailability remains the #18 service/atomicity limit.

The publisher uses GitHub's commit-status API rather than custom Actions check runs.
GitHub documents that checks created by Actions jobs on dispatch, comment and workflow-run
events may not satisfy required protection. The initial live reviewed-branch dispatch showed
successful custom check runs still left the required gate Expected. Commit statuses are a
separate supported API reflected on PRs, with the same stable context and existing publisher
identity. Verify actual protected-branch recognition during rollout; no protection bypass or
manual fabricated approval is part of this change. If a check run and a commit status share
the required name, GitHub requires both to pass, so old blocked check runs on an existing
head cannot be erased by a success status; a fresh head and fresh review avoid that conflict.

Native Codex normally reports P0/P1 findings. It does not provide the previous API workflow's
structured complete-file attestation or P0–P2 policy, and the gate does not claim those
properties. Its completion attests the reviewed head, not the current merge base. Ordinary
`rightyo/ci` remains separately required with strict branch freshness. AI review does not
prove correctness. Shared GitHub Actions publisher identity remains the unresolved
[#18](https://github.com/mickdarling/rightyo/issues/18) limitation.

### Claude deferred

Claude review is intentionally deferred until the owner configures subscription authentication
with `claude setup-token`. It is not required by the current gate and no Claude job runs
implicitly. The old dual-provider snapshot helpers and masked secret-setup utility remain
available as inactive groundwork; invoking them is not part of the subscription Codex lane.
Do not add an OpenAI API key to use native Codex reviews. Reintroducing Claude requires a
reviewed workflow change and verification of a completed exact-head subscription review.

Roll out using an issue-linked, independently reviewed PR. Enable the reviewed workflow,
after queued PR/comment/CI events have settled while the old lane remains disabled. Inspect
and cancel legacy callbacks; never overwrite their failed reviews as successful. Then
dispatch that immutable reviewed branch with the PR's full current revisions, verify its
actual native completion and classic checks, and merge through ordinary protection. The
workflow file alone does not configure the account's native review settings or branch
protection. Keep #17 open until live setup is verified and #18 open until stronger publisher
identity and protection tests exist.

References: [Codex GitHub integration](https://learn.chatgpt.com/docs/third-party/github),
[Codex review billing](https://learn.chatgpt.com/docs/pricing),
[Claude authentication](https://code.claude.com/docs/en/authentication#generate-a-long-lived-token).
