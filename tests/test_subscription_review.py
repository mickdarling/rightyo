"""Metadata-only regressions; no real provider or credentials are used."""

import unittest
from unittest.mock import patch

from scripts.subscription_review import (
    APP_ID,
    BOT_ID,
    BOT_LOGIN,
    GateError,
    clean_completion,
    event_pr,
    pages,
    publish,
)

HEAD = "a" * 40
BASE = "b" * 40


def completion(head=HEAD):
    return {
        "id": 12,
        "created_at": "2026-09-30T07:07:50Z",
        "updated_at": "2026-09-30T07:07:50Z",
        "user": {"id": BOT_ID, "login": BOT_LOGIN, "type": "Bot"},
        "performed_via_github_app": {"id": APP_ID},
        "body": f"Codex Review: Didn't find any major issues. :tada:\n\n"
        f"**Reviewed commit:** `{head[:10]}`\n\n<details>About</details>",
    }


class CompletionTests(unittest.TestCase):
    def check(self, comments, reviews=None, inline=None, resolved=HEAD):
        return clean_completion(HEAD, comments, reviews or [], inline or [], lambda short: resolved)

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
        self.assertIsNone(self.check([item]))

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
            clean_completion(HEAD, [completion()], [], [], fail)
        with patch("scripts.subscription_review.api", return_value=[{}] * 100):
            with self.assertRaises(GateError):
                pages("/collection")


class RoutingTests(unittest.TestCase):
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
            calls.append((method, data))
            if path.endswith("/pulls/44"):
                return initial if len([m for m, _ in calls if m == "GET"]) == 1 else changed
            if method == "POST":
                return {"id": 8}
            if method == "PATCH":
                return {}
            return {"sha": HEAD}

        with (
            patch("scripts.subscription_review.api", side_effect=request),
            patch("scripts.subscription_review.pages", side_effect=[[completion()], [], []]),
        ):
            with self.assertRaises(GateError):
                publish(44)
        self.assertEqual(calls[-1][1]["conclusion"], "failure")
        self.assertEqual(calls[1][1]["head_sha"], HEAD)
