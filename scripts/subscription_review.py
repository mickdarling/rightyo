"""Publish a metadata-only gate for an explicitly completed native Codex review.

No model is invoked here. Provider credentials and PR-controlled code are never used.
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

REPOSITORY = "mickdarling/rightyo"
BOT_LOGIN = "chatgpt-codex-connector[bot]"
BOT_ID = 199175422
BOT_NODE_ID = "BOT_kgDOC98s_g"
APP_ID = 1144995
ACTIONS_BOT_ID = 41898282
REQUEST_CONTEXT = "rightyo/review-request"
REQUEST_MARKER = re.compile(r"v1 comment:([1-9][0-9]*) requested:(.+)\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
COMPLETION = re.compile(
    r"\ACodex Review: Didn't find any major issues\.[^\n]*\s+"
    r"\*\*Reviewed commit:\*\* `([0-9a-f]{10,40})`(?:\s|\Z)"
)

SUMMARY = re.compile(
    r"^\| 📝 \*\*Code Review\*\* \| ✅ \*\*Completed\*\* "
    r'<relative-time datetime="([^"\n]+)">\1</relative-time> '
    r"\| `([0-9a-f]{7,40})` \| [^|\n]+\|$",
    re.M,
)
COMMENT_QUERY = (
    "query($id:ID!){node(id:$id){... on IssueComment{"
    "id body databaseId updatedAt lastEditedAt editor{__typename login ... on Bot{id}}}}}"
)


class GateError(ValueError):
    """Only public-safe categories are emitted; never response bodies or tokens."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise GateError("GitHub API redirect refused")


def api(path, method="GET", data=None):
    token = os.environ.get("GITHUB_TOKEN", "")
    graphql_request = (
        path == "/graphql"
        and method == "POST"
        and isinstance(data, dict)
        and set(data) == {"query", "variables"}
        and data["query"] == COMMENT_QUERY
    )
    if not token or not (path.startswith(f"/repos/{REPOSITORY}/") or graphql_request):
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


def timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z", value
    ):
        raise GateError("Invalid native review timestamp")
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise GateError("Invalid native review timestamp") from error


def comment_provenance(comment, summary):
    """Bind edit provenance to the exact REST body, not just the original author."""
    node_id = comment.get("node_id")
    if not isinstance(node_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", node_id):
        raise GateError("Native comment node is unavailable")
    payload = api("/graphql", "POST", {"query": COMMENT_QUERY, "variables": {"id": node_id}})
    if payload.get("errors"):
        raise GateError("Native comment provenance query failed")
    node = payload.get("data", {}).get("node")
    if (
        not isinstance(node, dict)
        or not {"id", "databaseId", "body", "updatedAt", "lastEditedAt", "editor"}.issubset(node)
        or (
            node.get("id") != node_id
            or node.get("databaseId") != comment.get("id")
            or node.get("body") != comment.get("body")
            or node.get("updatedAt") != comment.get("updated_at")
        )
    ):
        raise GateError("Native comment changed during provenance validation")
    if node.get("lastEditedAt") is None:
        return node.get("editor") is None
    if not summary:
        return False
    timestamp(node["lastEditedAt"])
    editor = node.get("editor") or {}
    return (
        editor.get("__typename") == "Bot"
        and editor.get("id") == BOT_NODE_ID
        and editor.get("login") == "chatgpt-codex-connector"
    )


def authorized_request(comment):
    user = comment.get("user") or {}
    login = user.get("login")
    if (
        user.get("type") != "User"
        or not isinstance(login, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}", login)
    ):
        return False
    permission = api(f"/repos/{REPOSITORY}/collaborators/{login}/permission")
    return permission.get("permission") in {"admin", "maintain", "write"}


def request_command(body):
    return (
        isinstance(body, str)
        and re.match(r"\A\s*@codex (?:security )?review(?:\s|\Z)", body, re.I) is not None
    )


def request_markers(head):
    if not isinstance(head, str) or not SHA.fullmatch(head):
        raise GateError("Invalid persisted request revision")
    markers = {}
    for status in pages(f"/repos/{REPOSITORY}/commits/{head}/statuses"):
        context = status.get("context")
        description = status.get("description") or ""
        if context != REQUEST_CONTEXT and not (
            context == "rightyo/review-gate" and description.startswith("v1 comment:")
        ):
            continue
        creator = status.get("creator") or {}
        if creator.get("id") != ACTIONS_BOT_ID or creator.get("login") != "github-actions[bot]":
            continue
        match = REQUEST_MARKER.fullmatch(status.get("description") or "")
        if not match:
            raise GateError("Persisted review request marker is malformed")
        comment_id, cutoff = int(match[1]), timestamp(match[2]).replace(microsecond=0)
        markers[comment_id] = max(markers.get(comment_id, cutoff), cutoff)
    return markers


def persist_request(head, comment_id, cutoff):
    description = f"v1 comment:{comment_id} requested:{cutoff.isoformat().replace('+00:00', 'Z')}"
    # Each immutable record independently carries the denial cutoff. The second write
    # still runs if the first fails, preserving best-effort durability without retries.
    try:
        commit_status(head, "pending", description)
    finally:
        commit_status(head, "pending", description)


def record_request(number, event_name, event):
    if event_name != "issue_comment":
        return
    comment = event.get("comment") or {}
    action = event.get("action")
    old_body = event.get("changes", {}).get("body", {}).get("from")
    current_request = request_command(comment.get("body"))
    removed_request = (request_command(old_body) and not current_request) or (
        action == "deleted" and current_request
    )
    if action not in {"created", "edited", "deleted"} or not (current_request or removed_request):
        return
    pr = current_pr(number)
    if pr.get("state") != "open":
        return
    head = pr["head"]["sha"]
    comment_id = comment.get("id")
    if type(comment_id) is not int or comment_id < 1:
        raise GateError("Review request comment identity is invalid")
    cutoff = timestamp(comment.get("updated_at")).replace(microsecond=0)
    # Freeze approval before permission/history requests that might fail. This job never
    # writes gate success, and a failed recorder prevents the publisher from running.
    try:
        allowed = authorized_request(comment) and authorized_request({"user": event.get("sender")})
    except GateError:
        persist_request(head, comment_id, cutoff)
        raise
    if not allowed:
        return
    try:
        markers = request_markers(head)
    except GateError:
        persist_request(head, comment_id, cutoff)
        raise
    if removed_request and comment_id in markers:
        cutoff = markers[comment_id]
    if markers.get(comment_id) == cutoff:
        return
    persist_request(head, comment_id, cutoff)


def freshness_barrier(comments, head, resolve, attest, authorize, persisted=None):
    barriers = list((persisted or {}).values())
    summaries = []
    for comment in comments:
        body = comment.get("body") or ""
        # Only a command at the start of the comment is a request; quoted/code examples
        # and unprivileged copied text are not treated as reviewer authorization.
        if request_command(body):
            if authorize(comment):
                barriers.append(timestamp(comment.get("updated_at")))
        if not native(comment, require_app=True):
            continue
        if not attest(comment, True):
            raise GateError("Native comment edit provenance is invalid")
        if not body.startswith(
            "<!-- codex-pull-request-review-summary -->\n\n## Codex Review Summary\n"
        ):
            continue
        row = re.findall(r"(?m)^\|[^\n]*\*\*Code Review\*\*[^\n]*\| `([0-9a-f]{7,40})` \|", body)
        if len(row) != 1:
            raise GateError("Native review activity summary is ambiguous or unsupported")
        if not head.startswith(row[0]) or resolve(row[0]) != head:
            continue
        updated = timestamp(comment.get("updated_at"))
        completed = SUMMARY.findall(body)
        if len(completed) == 1:
            summaries.append((updated, timestamp(completed[0][0]).replace(microsecond=0)))
        else:
            barriers.append(updated)
            summaries.append((updated, None))
    barrier = max(barriers) if barriers else None
    latest = max((updated for updated, _ in summaries), default=None)
    completed_cycle = latest is not None and all(
        completion is not None and (barrier is None or completion > barrier)
        for updated, completion in summaries
        if updated == latest
    )
    return barrier, completed_cycle


def inline_revision(comment, reviews):
    """Bind a forwarded inline comment to its immutable original native review."""
    original = comment.get("original_commit_id")
    if not isinstance(original, str) or not SHA.fullmatch(original):
        raise GateError("Native inline original revision is unavailable")
    review_id = comment.get("pull_request_review_id")
    if type(review_id) is not int or review_id < 1:
        raise GateError("Native inline parent review identity is unavailable")
    parents = [review for review in reviews if review.get("id") == review_id]
    if len(parents) != 1 or not native(parents[0]) or parents[0].get("commit_id") != original:
        raise GateError("Native inline parent review provenance is inconsistent")
    return original


def review_candidates(
    head,
    comments,
    reviews,
    inline_comments,
    resolve,
    attest,
    authorize=authorized_request,
    persisted=None,
):
    """Require positive authenticated evidence, never absence of findings alone."""
    if not isinstance(head, str) or not SHA.fullmatch(head):
        raise GateError("Invalid immutable PR head")
    for record in reviews:
        if native(record) and record.get("commit_id") == head:
            return []
    for record in inline_comments:
        if native(record) and inline_revision(record, reviews) == head:
            return []
    barrier, completed_cycle = freshness_barrier(
        comments, head, resolve, attest, authorize, persisted
    )
    if not completed_cycle:
        return []
    candidates = []
    for comment in reversed(comments):
        if not native(comment, require_app=True):
            continue
        body = comment.get("body") or ""
        match = COMPLETION.match(body)
        summary = False
        if not match:
            if not body.startswith(
                "<!-- codex-pull-request-review-summary -->\n\n## Codex Review Summary\n"
            ):
                continue
            rows = SUMMARY.findall(body)
            if len(rows) != 1 or len(re.findall(r"(?m)^\|[^\n]*\*\*Code Review\*\*", body)) != 1:
                continue
            completed_at, short = rows[0]
            completed = timestamp(completed_at).replace(microsecond=0)
            if completed > timestamp(comment.get("updated_at")):
                raise GateError("Native summary completion timestamp is inconsistent")
            summary = True
        else:
            short = match[1]
            completed = timestamp(comment.get("created_at"))
        if barrier is not None and completed <= barrier:
            continue
        if not head.startswith(short) or resolve(short) != head:
            continue
        if not attest(comment, summary):
            continue
        candidates.append(
            {
                "id": comment["id"],
                "kind": "summary" if summary else "explicit",
                "completed_at": completed,
            }
        )
    return candidates


def select_completion(candidates, reactions):
    for candidate in candidates:
        if candidate["kind"] == "explicit":
            return candidate
        for reaction in reactions:
            user = reaction.get("user") or {}
            # GitHub's reactions REST endpoint exposes the genuine connector as User,
            # unlike comment/review endpoints. Its immutable ID and login still match.
            if (
                reaction.get("content") == "+1"
                and user.get("id") == BOT_ID
                and user.get("login") == BOT_LOGIN
                and user.get("type") in {"User", "Bot"}
                and timestamp(reaction.get("created_at")) >= candidate["completed_at"]
            ):
                return {**candidate, "reaction_id": reaction["id"]}
    return None


def clean_completion(
    head,
    comments,
    reviews,
    inline_comments,
    resolve,
    attest,
    reactions,
    authorize=authorized_request,
):
    return select_completion(
        review_candidates(head, comments, reviews, inline_comments, resolve, attest, authorize),
        reactions,
    )


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
        notification = event.get("workflow_run", {})
        run_id = notification.get("id")
        if type(run_id) is not int or run_id < 1:
            raise GateError("Review callback run identity is invalid")
        run = api(f"/repos/{REPOSITORY}/actions/runs/{run_id}")
        routes = {
            ".github/workflows/ci.yml": ("RightyO CI", {"pull_request"}),
            ".github/workflows/review-activity.yml": (
                "Native review activity relay",
                {"pull_request_review", "pull_request_review_comment"},
            ),
        }
        if run.get("event") == "push" and run.get("path") == ".github/workflows/ci.yml":
            return None, None
        route = routes.get(run.get("path"))
        if (
            run.get("id") != run_id
            or run.get("repository", {}).get("full_name") != REPOSITORY
            or run.get("status") != "completed"
            or route is None
            or run.get("name") != route[0]
            or run.get("event") not in route[1]
        ):
            raise GateError("Review callback source workflow is invalid")
        prs = run.get("pull_requests", [])
        if not isinstance(prs, list) or len(prs) != 1:
            revision = run.get("head_sha", "")
            if not isinstance(revision, str) or not SHA.fullmatch(revision):
                raise GateError("Review callback PR association is unavailable")
            # No source-run artifacts or programs are consulted. A bounded GitHub-owned
            # association is accepted only for one open PR with the exact source head.
            prs = [
                pr
                for pr in pages(f"/repos/{REPOSITORY}/commits/{revision}/pulls")
                if pr.get("state") == "open"
                and pr.get("head", {}).get("sha") == revision
                and pr.get("base", {}).get("repo", {}).get("full_name") == REPOSITORY
            ]
        if len(prs) != 1 or type(prs[0].get("number")) is not int or prs[0]["number"] < 1:
            raise GateError("Review callback PR association is ambiguous")
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


def commit_status(head, state, description, *, context="rightyo/review-gate"):
    """Publish the stable context without workflow-event check-suite restrictions."""
    if not isinstance(head, str) or not SHA.fullmatch(head):
        raise GateError("Invalid status revision")
    if context not in {"rightyo/review-gate", REQUEST_CONTEXT} or (
        (context == REQUEST_CONTEXT or description.startswith("v1 comment:")) and state != "pending"
    ):
        raise GateError("Invalid denial-marker status scope")
    if state not in {"pending", "success", "failure", "error"}:
        raise GateError("Invalid status state")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not re.fullmatch(r"[1-9][0-9]*", run_id):
        raise GateError("Status publisher run identity is unavailable")
    return api(
        f"/repos/{REPOSITORY}/statuses/{head}",
        "POST",
        {
            "state": state,
            "context": context,
            "description": description[:140],
            "target_url": f"https://github.com/{REPOSITORY}/actions/runs/{run_id}",
        },
    )


def publish(number, expected=None):
    pr = current_pr(number)
    if pr.get("state") != "open":
        print("Closed PR ignored")
        return
    head, base = revisions(pr)
    if expected is not None and (head, base) != expected:
        raise GateError("Dispatch revisions are stale")
    commit_status(head, "pending", "Validating current-head subscription Codex completion")
    conclusion = "failure"
    message = "Review metadata could not be validated"
    failure = None
    try:
        prefix = f"/repos/{REPOSITORY}"

        def collect():
            comments = pages(f"{prefix}/issues/{number}/comments")
            reviews = pages(f"{prefix}/pulls/{number}/reviews")
            inline = pages(f"{prefix}/pulls/{number}/comments")
            return review_candidates(
                head,
                comments,
                reviews,
                inline,
                lambda short: api(f"{prefix}/commits/{short}")["sha"],
                comment_provenance,
                persisted=request_markers(head),
            )

        candidates = collect()
        completion = select_completion(candidates, pages(f"{prefix}/issues/{number}/reactions"))
        for _ in range(3):
            if completion is not None or not any(c["kind"] == "summary" for c in candidates):
                break
            # Summary updates can precede the clean PR reaction by a few seconds.
            # There is no reaction webhook event, so briefly retry metadata, not inference.
            time.sleep(5)
            if revisions(current_pr(number)) != (head, base):
                raise GateError("PR revisions changed while waiting for native reaction")
            candidates = collect()
            completion = select_completion(candidates, pages(f"{prefix}/issues/{number}/reactions"))
        if completion is not None:
            # Refetch finding streams and current body/provenance before publication.
            completion = select_completion(collect(), pages(f"{prefix}/issues/{number}/reactions"))
        if revisions(current_pr(number)) != (head, base):
            raise GateError("PR revisions changed while validating review")
        if completion is None:
            conclusion = "failure"
            message = (
                "Awaiting clean native Codex completion evidence for this exact head. "
                "Requires a native explicit clean verdict, or an authentic completed summary "
                "plus a fresh connector thumbs-up reaction. "
                "Old or incomplete evidence is insufficient."
            )
        else:
            conclusion = "success"
            message = (
                f"Native Codex reported no major issues for full head {head}. "
                f"Evidence: {completion['kind']}; comment ID: {completion['id']}; "
                f"reaction ID: {completion.get('reaction_id', 'not required')}; "
                f"current base: {base}. "
                "This is subscription-backed native review, normally P0/P1; "
                "it does not attest complete file coverage or a review of the current base. "
                "Claude remains deferred. Classic CI is independently required."
            )
    except (GateError, KeyError, TypeError, ValueError) as error:
        failure = error
        conclusion = "error"
        if isinstance(error, GateError):
            message = str(error)
    # Full native evidence and limitations live in the linked run log; commit statuses
    # have a short description rather than the Checks API's rich output fields.
    print(message)
    description = {
        "success": "Validated clean subscription Codex completion; see run log for native evidence",
        "failure": "Awaiting clean native review; missing evidence or findings; see run log",
        "error": "Native review metadata validation failed; see run log",
    }[conclusion]
    commit_status(head, conclusion, description)
    if failure or conclusion != "success":
        raise GateError("Subscription review gate has not passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=["resolve", "record", "publish"], default="publish"
    )
    args = parser.parse_args()
    try:
        event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
        if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
            raise GateError("Workflow repository mismatch")
        number, expected = event_pr(os.environ["GITHUB_EVENT_NAME"], event)
        if args.command == "resolve":
            with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
                output.write(f"pr_number={number if number is not None else ''}\n")
            print("Trusted PR association resolved")
        elif number is not None:
            routed = os.environ.get("GATE_PR_NUMBER", "")
            if not re.fullmatch(r"[1-9][0-9]*", routed) or int(routed) != number:
                raise GateError("Publisher PR association changed after serialized routing")
            if args.command == "record":
                record_request(number, os.environ["GITHUB_EVENT_NAME"], event)
            else:
                publish(number, expected)
        else:
            print("Non-PR event ignored")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error) if isinstance(error, GateError) else "Subscription gate processing failed")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
