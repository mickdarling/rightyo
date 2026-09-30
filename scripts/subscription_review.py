"""Publish a metadata-only gate for an explicitly completed native Codex review.

No model is invoked here. Provider credentials and PR-controlled code are never used.
"""

import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPOSITORY = "mickdarling/rightyo"
BOT_LOGIN = "chatgpt-codex-connector[bot]"
BOT_ID = 199175422
APP_ID = 1144995
SHA = re.compile(r"[0-9a-f]{40}\Z")
COMPLETION = re.compile(
    r"\ACodex Review: Didn't find any major issues\.[^\n]*\s+"
    r"\*\*Reviewed commit:\*\* `([0-9a-f]{10,40})`(?:\s|\Z)"
)


class GateError(ValueError):
    """Only public-safe categories are emitted; never response bodies or tokens."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GateError("GitHub API redirect refused")


def api(path, method="GET", data=None):
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token or not path.startswith(f"/repos/{REPOSITORY}/"):
        raise GateError("GitHub authentication or repository scope unavailable")
    request = urllib.request.Request(
        "https://api.github.com" + path,
        data=None if data is None else json.dumps(data).encode(),
        method=method,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read(3_000_001)
        if len(raw) > 3_000_000:
            raise GateError("GitHub API response exceeds bound")
        return json.loads(raw)
    except (urllib.error.URLError, ValueError) as error:
        raise GateError("GitHub API request failed") from error


def pages(path):
    records = []
    for page in range(1, 11):
        batch = api(f"{path}?per_page=100&page={page}")
        if not isinstance(batch, list):
            raise GateError("Invalid GitHub collection")
        records.extend(batch)
        if len(batch) < 100:
            return records
    raise GateError("Review history exceeds bounded collection")


def native(record, *, require_app=False):
    user = record.get("user", {})
    return (
        user.get("login") == BOT_LOGIN
        and user.get("id") == BOT_ID
        and user.get("type") == "Bot"
        and (not require_app or (record.get("performed_via_github_app") or {}).get("id") == APP_ID)
    )


def clean_completion(head, comments, reviews, inline_comments, resolve):
    """Require positive authenticated completion; absence of findings alone is insufficient."""
    if not isinstance(head, str) or not SHA.fullmatch(head):
        raise GateError("Invalid immutable PR head")
    # A completed native review with suggestions is never silently converted to approval,
    # even if its threads were resolved or its review dismissed. A fresh fixed head is needed.
    for record in reviews + inline_comments:
        if native(record) and record.get("commit_id") == head:
            return None
    for comment in reversed(comments):
        if not native(comment, require_app=True):
            continue
        created = comment.get("created_at")
        if not isinstance(created, str) or created != comment.get("updated_at"):
            continue
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created):
            continue
        try:
            datetime.fromisoformat(created)
        except ValueError:
            continue
        match = COMPLETION.match(comment.get("body") or "")
        if not match or not head.startswith(match[1]):
            continue
        # The displayed abbreviation must uniquely resolve to the entire current commit.
        # An ambiguous commit prefix, API failure or another SHA fails closed.
        if resolve(match[1]) == head:
            return comment
    return None


def event_pr(event_name, event):
    if event_name == "workflow_dispatch":
        inputs = event.get("inputs", {})
        expected = inputs.get("expected_head", "")
        if not isinstance(expected, str) or not SHA.fullmatch(expected):
            raise GateError("Dispatch requires a full immutable head")
        raw_number = inputs.get("pr_number", "")
        if not isinstance(raw_number, str) or not re.fullmatch(r"[1-9][0-9]*", raw_number):
            raise GateError("Dispatch requires a positive PR number")
        base = inputs.get("expected_base", "")
        if not isinstance(base, str) or not SHA.fullmatch(base):
            raise GateError("Dispatch requires a full immutable base")
        return int(raw_number), (expected, base)
    if event_name == "workflow_run":
        run = event.get("workflow_run", {})
        prs = run.get("pull_requests", [])
        if run.get("event") != "pull_request" or len(prs) != 1:
            return None, None
        return prs[0]["number"], None
    if event_name == "issue_comment":
        issue = event.get("issue", {})
        return (issue["number"], None) if issue.get("pull_request") else (None, None)
    if event_name == "pull_request_target":
        return event["pull_request"]["number"], None
    raise GateError("Unsupported gate event")


def current_pr(number):
    if type(number) is not int or number < 1:
        raise GateError("Invalid PR number")
    pr = api(f"/repos/{REPOSITORY}/pulls/{number}")
    if pr.get("base", {}).get("repo", {}).get("full_name") != REPOSITORY:
        raise GateError("PR repository mismatch")
    if not SHA.fullmatch(pr.get("head", {}).get("sha", "")):
        raise GateError("Invalid immutable PR head")
    if not SHA.fullmatch(pr.get("base", {}).get("sha", "")):
        raise GateError("Invalid immutable PR base")
    return pr


def revisions(pr):
    return pr["head"]["sha"], pr["base"]["sha"]


def publish(number, expected=None):
    pr = current_pr(number)
    if pr.get("state") != "open":
        print("Closed PR ignored")
        return
    head, base = revisions(pr)
    if expected is not None and (head, base) != expected:
        raise GateError("Dispatch revisions are stale")
    check = api(
        f"/repos/{REPOSITORY}/check-runs",
        "POST",
        {
            "name": "rightyo/review-gate",
            "head_sha": head,
            "status": "in_progress",
            "output": {
                "title": "Checking subscription Codex review",
                "summary": (
                    "Requires an explicit current-head native completion. Claude is deferred."
                ),
            },
        },
    )
    conclusion = "failure"
    message = "Review metadata could not be validated"
    failure = None
    try:
        prefix = f"/repos/{REPOSITORY}"
        comments = pages(f"{prefix}/issues/{number}/comments")
        reviews = pages(f"{prefix}/pulls/{number}/reviews")
        inline = pages(f"{prefix}/pulls/{number}/comments")
        completion = clean_completion(
            head,
            comments,
            reviews,
            inline,
            lambda short: api(f"{prefix}/commits/{short}")["sha"],
        )
        if revisions(current_pr(number)) != (head, base):
            raise GateError("PR revisions changed while validating review")
        if completion is None:
            conclusion = "action_required"
            message = (
                "Awaiting an explicit clean native Codex review of this exact head. "
                "A request, reaction, summary table, old review or resolved finding "
                "is insufficient. "
                "If native review only left a reaction/summary, request @codex review on the PR."
            )
        else:
            conclusion = "success"
            message = (
                f"Native Codex reported no major issues for full head {head}. "
                f"Completion comment ID: {completion['id']}; current base: {base}. "
                "This is subscription-backed native review, normally P0/P1; "
                "it does not attest complete file coverage or a review of the current base. "
                "Claude remains deferred. Classic CI is independently required."
            )
    except (GateError, KeyError, TypeError, ValueError) as error:
        failure = error
        if isinstance(error, GateError):
            message = str(error)
    api(
        f"/repos/{REPOSITORY}/check-runs/{check['id']}",
        "PATCH",
        {
            "status": "completed",
            "conclusion": conclusion,
            "output": {"title": "Subscription Codex review", "summary": message},
        },
    )
    print(message)
    if failure or conclusion != "success":
        raise GateError("Subscription review gate has not passed")


def main():
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
            raise GateError("Workflow repository mismatch")
        number, expected = event_pr(os.environ["GITHUB_EVENT_NAME"], event)
        if number is not None:
            publish(number, expected)
        else:
            print("Non-PR event ignored")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error) if isinstance(error, GateError) else "Subscription gate processing failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
