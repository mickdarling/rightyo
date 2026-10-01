# Codex and Claude Code review automation

**Historical dual-provider design:** the active lane now uses subscription-backed native
Codex reviews with a metadata-only gate. Claude is deferred until subscription credentials
are configured. Follow [current CI setup](ci.md); this design is not an active API setup guide.

This is the concrete implementation plan for
[#17](https://github.com/mickdarling/rightyo/issues/17). No provider authentication or
automated review result is established by this document. The baseline CI in
[ci.md](ci.md) remains secretless. The trusted merge gate and protections belong to
[#18](https://github.com/mickdarling/rightyo/issues/18).

## First executable lane

Start with a maintainer-triggered `workflow_dispatch` workflow from the protected default
branch. Inputs identify the PR number and expected full head/base SHAs. Verify the actor's
repository write permission, fetch current PR metadata through GitHub's API, and reject
closed PRs, mismatched revisions or an untrusted workflow revision. Resolve fork PR content
through the same immutable snapshot path; never give fork workflow code a provider secret.

Use a separate preparation job without provider credentials. It creates a bounded public
source diff and sufficient surrounding source context from the expected revisions, without
executing the PR's code, setup scripts, hooks or tests. Review policy, prompts and output
schema come from the trusted workflow revision. PR changes to AGENTS/Claude guidance are
review data, not instructions. Reject symlinks, binary/private artifacts and snapshots too
large to review completely; never truncate silently into a clean result. Record snapshot
SHA-256, merge-base SHA and precise file/coverage inventory.

Fresh independent provider jobs review the identical snapshot without implementation
conversation history. Their working directory contains only the prepared review input and
trusted configuration. Do not check out PR configuration or expose developer home state.
Give each provider only its own authentication. Initially produce validated review artifacts
with short retention, without PR write permission or automatic approval/merge. A separately
privileged publisher can be added after its trust boundary is independently verified.

## Provider adapters

The [official OpenAI Action documentation](https://learn.chatgpt.com/docs/github-action)
supports pinned CLI versions, permission restrictions and schema-shaped output. Use a
full-commit action pin, a fresh trusted Codex home, a read-only sandbox and `drop-sudo` or
an unprivileged account. Read-only alone is not secret protection. Run Codex last in its
job. Before enabling this lane, prove PR input cannot execute commands, load PR-controlled
configuration, access provider credentials or change the captured result; a prompt saying
"do not execute" is insufficient. If the selected CLI cannot enforce the required tool
boundary, use a tool-free review adapter or keep this lane unavailable.

Use the [official Claude Code Action](https://code.claude.com/docs/en/github-actions)
with a plain explicit review prompt rather than treating the stock review plugin's job
success as completion: that plugin may skip trivial/automated/already-commented PRs.
Extract structured output and independently check the Action conclusion. Account/app
authentication and action/CLI pin compatibility need actual setup verification.

The [Claude CLI reference](https://code.claude.com/docs/en/cli-reference) provides restricted
execution, explicit tool selection, structured output and turn/spend caps. For a snapshot
review, configure restricted mode, remove command/edit tools and all MCP tools, and disable
project/user customization discovery. Confirm the selected pinned CLI enforces this with
adversarial fixtures; `allowedTools` alone does not define the available tool set. Do not
enable the generic implementation/mention workflow or broad bot allowlists for this lane.

Both adapters must record selected models/versions and review coverage/severity thresholds.
Choose explicit time, input, output and provider spend budgets before activation. Allow at
most one bounded retry for a transient failure; quota exhaustion remains incomplete. Cancel
superseded head runs and deduplicate by provider, head, base, policy and snapshot hashes.

## Result contract and validation

The provider supplies a small structured assessment: schema version, `completed` or
`inconclusive`, findings with priority/path/line/evidence and coverage limitations. The trusted
adapter adds repository/PR, provider identity, head/base/merge-base SHAs, policy/snapshot
hashes, actual runtime/model versions and source workflow/run IDs. Do not let model output
choose those provenance fields or claim an authenticated identity.

A trusted validator requires a successful provider run, a valid bounded result, confirmed
expected provenance and complete declared coverage. Missing credentials, failure, timeout,
quota/turn limit, malformed output, truncated input/output, skip or an inconclusive assessment
cannot become a clean completed review. Empty findings are acceptable only after successful
completion; a reaction, silence or successful installer process does not count.

Before publication, refetch PR metadata and reject a changed head or invalidated base/diff.
Store evidence outside the reviewed commit so recording its SHA does not create a new head.
Fix confirmed findings and obtain fresh reviews, or record an authenticated independent
maintainer's reasoned disposition. A PR-body checkbox does not clear findings.

## Activation and enforcement

Owner setup must establish provider access, permitted scoped credentials and actual GitHub
app/settings behavior without posting secrets. Test one representative PR through both
providers, including successful no-findings output, changed-head rerun, fork handling, stale
result, skipped provider, malformed response, unauthorized actor and hostile configuration.
Inspect logs/artifacts for exposure using invented credentials only. Until these pass, native
or manually orchestrated independent reviews need explicit provider/head/outcome evidence;
an unavailable provider is reported as incomplete.

After validated opt-in runs, add automatic triggers through the trusted orchestration boundary,
then implement the distinct `rightyo/review-gate`. It must reject spoofed evidence and a
PR-modified workflow manufacturing success under the shared GitHub Actions App identity.
Native COMMENTED bot reviews and provider completion are separate from GitHub APPROVED
review requirements. Only then configure and demonstrate branch protection. See
[GitHub's secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
for privileged event, checkout and token boundaries.
