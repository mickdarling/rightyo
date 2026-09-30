"""Metadata-only regressions; no real provider or credentials are used."""

import os
import unittest
from unittest.mock import patch

from scripts.subscription_review import (
    APP_ID,
    BOT_ID,
    BOT_LOGIN,
    BOT_NODE_ID,
    GateError,
    clean_completion,
    comment_provenance,
    commit_status,
    event_pr,
    pages,
    publish,
)

HEAD = "a" * 40
BASE = "b" * 40


def completion(head=HEAD):
    return {
        "id": 12,
        "node_id": "IC_fixture",
        "created_at": "2026-09-30T07:07:50Z",
        "updated_at": "2026-09-30T07:07:50Z",
        "user": {"id": BOT_ID, "login": BOT_LOGIN, "type": "Bot"},
        "performed_via_github_app": {"id": APP_ID},
        "body": f"Codex Review: Didn't find any major issues. :tada:\n\n"
        f"**Reviewed commit:** `{head[:10]}`\n\n<details>About</details>",
    }


class CompletionTests(unittest.TestCase):
    def check(self, comments, reviews=None, inline=None, resolved=HEAD):
        return clean_completion(
            HEAD,
            comments,
            reviews or [],
            inline or [],
            lambda short: resolved,
            lambda item, summary: item["created_at"] == item["updated_at"],
            [],
        )

    def test_authenticated_positive_completion_resolves_full_current_head(self):
        self.assertEqual(self.check([completion()])["id"], 12)
        item = completion()
        item["body"] = item["body"].replace(":tada:", "Keep it up!")
        self.assertIsNotNone(self.check([item]))

    def test_missing_activity_request_reaction_or_short_summary_are_not_approval(self):
        for text in (
            "@codex review",
            "👍",
            "<!-- codex-pull-request-review-summary --> Completed `aaaaaaa`",
            "Codex Review: Didn't find any major issues.\n\n**Reviewed commit:** `aaaaaaa`",
        ):
            item = completion()
            item["body"] = text
            self.assertIsNone(self.check([item]))
        self.assertIsNone(self.check([]))

    def test_copied_verdict_or_wrong_app_identity_cannot_pass(self):
        for field, value in (
            ("user", {"id": 3, "login": BOT_LOGIN, "type": "Bot"}),
            ("user", {"id": BOT_ID, "login": "other", "type": "Bot"}),
            ("performed_via_github_app", None),
            ("performed_via_github_app", {"id": 3}),
        ):
            item = completion()
            item[field] = value
            self.assertIsNone(self.check([item]))

    def test_edited_completion_or_invalid_timestamp_cannot_pass(self):
        for stamp in (None, "2026-09-30T07:08:50Z", "invalid"):
            item = completion()
            item["updated_at"] = stamp
            self.assertIsNone(self.check([item]))
        item = completion()
        item["created_at"] = item["updated_at"] = "2026-99-99T07:07:50Z"
        with self.assertRaises(GateError):
            self.check([item])

    def test_old_or_ambiguous_commit_cannot_pass(self):
        self.assertIsNone(self.check([completion(BASE)]))
        self.assertIsNone(self.check([completion()], resolved=HEAD[:10] + "c" * 30))
        self.assertIsNone(self.check([completion()], resolved=None))

    def test_current_head_findings_block_even_with_clean_comment_or_dismissal(self):
        finding = {"user": completion()["user"], "commit_id": HEAD, "state": "DISMISSED"}
        self.assertIsNone(self.check([completion()], reviews=[finding]))
        self.assertIsNone(self.check([completion()], inline=[finding]))
        finding["commit_id"] = BASE
        self.assertIsNotNone(self.check([completion()], reviews=[finding]))

    def test_api_failure_is_not_empty_approval(self):
        def fail(short):
            raise GateError("GitHub API request failed")

        with self.assertRaises(GateError):
            clean_completion(HEAD, [completion()], [], [], fail, lambda item, summary: True, [])
        with patch("scripts.subscription_review.api", return_value=[{}] * 100):
            with self.assertRaises(GateError):
                pages("/collection")


class RoutingTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GITHUB_RUN_ID": "12345"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_dispatch_requires_full_head_and_base(self):
        event = {"inputs": {"pr_number": "44", "expected_head": HEAD, "expected_base": BASE}}
        self.assertEqual(event_pr("workflow_dispatch", event), (44, (HEAD, BASE)))
        for field in ("pr_number", "expected_head", "expected_base"):
            changed = {"inputs": dict(event["inputs"])}
            changed["inputs"][field] = "invalid"
            with self.assertRaises(GateError):
                event_pr("workflow_dispatch", changed)

    def test_non_pr_events_do_not_publish(self):
        self.assertEqual(event_pr("issue_comment", {"issue": {"number": 4}}), (None, None))
        self.assertEqual(
            event_pr("workflow_run", {"workflow_run": {"event": "push"}}), (None, None)
        )
        with self.assertRaises(GateError):
            event_pr("pull_request_review", {})

    def test_revision_change_fails_existing_head_check(self):
        initial = {
            "state": "open",
            "head": {"sha": HEAD},
            "base": {"sha": BASE, "repo": {"full_name": "mickdarling/rightyo"}},
        }
        changed = {**initial, "head": {"sha": "c" * 40}}
        calls = []

        def request(path, method="GET", data=None):
            calls.append((path, method, data))
            if path.endswith("/pulls/44"):
                return initial if len([m for _, m, _ in calls if m == "GET"]) == 1 else changed
            if method == "POST":
                return {"id": 8}
            if method == "PATCH":
                return {}
            return {"sha": HEAD}

        with (
            patch("scripts.subscription_review.api", side_effect=request),
            patch(
                "scripts.subscription_review.pages", side_effect=[[completion()], [], [], []] * 2
            ),
            patch("scripts.subscription_review.comment_provenance", return_value=True),
        ):
            with self.assertRaises(GateError):
                publish(44)
        self.assertEqual(calls[-1][2]["state"], "error")
        self.assertEqual(calls[1][0], f"/repos/mickdarling/rightyo/statuses/{HEAD}")
        self.assertEqual(calls[1][2]["state"], "pending")


def summary():
    item = completion()
    item["updated_at"] = "2026-09-30T07:08:50Z"
    item["body"] = (
        "<!-- codex-pull-request-review-summary -->\n\n## Codex Review Summary\n\n"
        "| Review | Status | Commit | Review trigger |\n"
        "| --- | --- | --- | --- |\n"
        "| 📝 **Code Review** | ✅ **Completed** "
        '<relative-time datetime="2026-09-30T07:08:49.384854Z">'
        "2026-09-30T07:08:49.384854Z</relative-time> | `aaaaaaa` | PR opened |\n"
    )
    return item


def reaction(stamp="2026-09-30T07:08:49Z"):
    return {
        "id": 23,
        "content": "+1",
        "created_at": stamp,
        "user": {"id": BOT_ID, "login": BOT_LOGIN, "type": "User"},
    }


class AutomaticCompletionTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GITHUB_RUN_ID": "12345"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def check(self, item=None, reactions=None, reviews=None, attest=True, resolved=HEAD):
        return clean_completion(
            HEAD,
            [item or summary()],
            reviews or [],
            [],
            lambda short: resolved,
            lambda comment, is_summary: attest,
            [reaction()] if reactions is None else reactions,
        )

    def test_completed_current_summary_with_fresh_native_reaction_passes(self):
        result = self.check()
        self.assertEqual(result["kind"], "summary")
        self.assertEqual(result["reaction_id"], 23)
        self.assertIsNotNone(self.check(reactions=[reaction("2026-09-30T07:08:52Z")]))

    def test_summary_or_reaction_alone_and_old_positive_do_not_pass(self):
        self.assertIsNone(self.check(reactions=[]))
        self.assertIsNone(self.check(reactions=[reaction("2026-09-30T07:08:48Z")]))
        self.assertIsNone(
            clean_completion(
                HEAD, [], [], [], lambda short: HEAD, lambda item, summary: True, [reaction()]
            )
        )
        item = summary()
        item["body"] = item["body"].replace("`aaaaaaa`", "`bbbbbbb`")
        self.assertIsNone(self.check(item))
        self.assertIsNone(self.check(resolved=BASE))

    def test_spoofed_reaction_noncompleted_or_duplicate_rows_and_findings_block(self):
        for field, value in (
            ("content", "eyes"),
            ("user", {"id": 3, "login": BOT_LOGIN, "type": "User"}),
        ):
            fake = reaction()
            fake[field] = value
            self.assertIsNone(self.check(reactions=[fake]))
        item = summary()
        item["body"] = item["body"].replace("**Completed**", "**Running**")
        self.assertIsNone(self.check(item))
        item = summary()
        item["body"] += item["body"].splitlines()[-1] + "\n"
        self.assertIsNone(self.check(item))
        finding = {"user": completion()["user"], "commit_id": HEAD}
        self.assertIsNone(self.check(reviews=[finding]))
        self.assertIsNone(self.check(attest=False))

    def test_summary_timestamp_must_be_valid_and_consistent(self):
        item = summary()
        item["body"] = item["body"].replace("07:08:49.384854Z", "07:09:49.384854Z")
        with self.assertRaises(GateError):
            self.check(item)

    def test_graphql_editor_and_exact_body_binding(self):
        item = summary()
        node = {
            "id": item["node_id"],
            "databaseId": item["id"],
            "body": item["body"],
            "updatedAt": item["updated_at"],
            "lastEditedAt": item["updated_at"],
            "editor": {"__typename": "Bot", "login": "chatgpt-codex-connector", "id": BOT_NODE_ID},
        }

        def check(changed, is_summary=True):
            with patch("scripts.subscription_review.api", return_value={"data": {"node": changed}}):
                return comment_provenance(item, is_summary)

        self.assertTrue(check(node))
        for field in ("lastEditedAt", "editor"):
            missing = dict(node)
            del missing[field]
            with self.assertRaises(GateError):
                check(missing)
        self.assertFalse(check(node, False))
        for editor in (
            {"__typename": "User", "login": "chatgpt-codex-connector", "id": BOT_NODE_ID},
            {"__typename": "Bot", "login": "chatgpt-codex-connector", "id": "wrong"},
            {"__typename": "Bot", "login": "other", "id": BOT_NODE_ID},
            None,
        ):
            self.assertFalse(check({**node, "editor": editor}))
        for field, value in (("body", "copied verdict"), ("id", "other"), ("databaseId", 3)):
            with self.assertRaises(GateError):
                check({**node, field: value})
        self.assertTrue(check({**node, "lastEditedAt": None, "editor": None}, False))
        with patch("scripts.subscription_review.api", return_value={"errors": [{}]}):
            with self.assertRaises(GateError):
                comment_provenance(item, True)

    def test_late_reaction_metadata_retry_publishes_success(self):
        pr = {
            "state": "open",
            "head": {"sha": HEAD},
            "base": {"sha": BASE, "repo": {"full_name": "mickdarling/rightyo"}},
        }
        changes = []

        def request(path, method="GET", data=None):
            if method == "POST":
                changes.append(data)
                return {}
            if "/commits/" in path:
                return {"sha": HEAD}
            return pr

        # Initial valid summary has no reaction, then the native +1 arrives after 5 seconds.
        batches = [
            [summary()],
            [],
            [],
            [],
            [summary()],
            [],
            [],
            [reaction()],
            [summary()],
            [],
            [],
            [reaction()],
        ]
        with (
            patch("scripts.subscription_review.api", side_effect=request),
            patch("scripts.subscription_review.pages", side_effect=batches),
            patch("scripts.subscription_review.comment_provenance", return_value=True),
            patch("scripts.subscription_review.time.sleep") as sleep,
        ):
            publish(44)
        sleep.assert_called_once_with(5)
        self.assertEqual(changes[-1]["state"], "success")
        self.assertEqual(changes[0]["state"], "pending")

    def test_late_current_head_finding_invalidates_initial_clean_summary(self):
        pr = {
            "state": "open",
            "head": {"sha": HEAD},
            "base": {"sha": BASE, "repo": {"full_name": "mickdarling/rightyo"}},
        }
        changes = []

        def request(path, method="GET", data=None):
            if method == "POST":
                changes.append(data)
                return {}
            return {"sha": HEAD} if "/commits/" in path else pr

        finding = {"user": completion()["user"], "commit_id": HEAD}
        batches = [[summary()], [], [], [], [summary()], [finding], [], [reaction()]]
        with (
            patch("scripts.subscription_review.api", side_effect=request),
            patch("scripts.subscription_review.pages", side_effect=batches),
            patch("scripts.subscription_review.comment_provenance", return_value=True),
            patch("scripts.subscription_review.time.sleep"),
        ):
            with self.assertRaises(GateError):
                publish(44)
        self.assertEqual(changes[-1]["state"], "failure")


class StatusPublicationTests(unittest.TestCase):
    def test_status_context_exact_head_and_run_link_are_preserved(self):
        with (
            patch.dict(os.environ, {"GITHUB_RUN_ID": "12345"}),
            patch("scripts.subscription_review.api", return_value={}) as request,
        ):
            commit_status(HEAD, "pending", "public description")
        request.assert_called_once_with(
            f"/repos/mickdarling/rightyo/statuses/{HEAD}",
            "POST",
            {
                "state": "pending",
                "context": "rightyo/review-gate",
                "description": "public description",
                "target_url": "https://github.com/mickdarling/rightyo/actions/runs/12345",
            },
        )

    def test_invalid_run_identity_revision_or_state_never_writes(self):
        with patch("scripts.subscription_review.api") as request:
            for head, state, run in (
                (HEAD, "success", ""),
                ("bad", "success", "123"),
                (HEAD, "neutral", "123"),
                (HEAD, "success", "https://other"),
            ):
                with patch.dict(os.environ, {"GITHUB_RUN_ID": run}), self.assertRaises(GateError):
                    commit_status(head, state, "example")
            request.assert_not_called()
