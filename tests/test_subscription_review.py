"""Metadata-only regressions; no real provider or credentials are used."""

import os
import unittest
from unittest.mock import patch

from scripts.subscription_review import (
    ACTIONS_BOT_ID,
    APP_ID,
    BOT_ID,
    BOT_LOGIN,
    BOT_NODE_ID,
    REQUEST_CONTEXT,
    GateError,
    authorized_request,
    clean_completion,
    comment_provenance,
    commit_status,
    event_pr,
    inline_revision,
    main,
    pages,
    persist_request,
    publish,
    record_request,
    request_markers,
    review_candidates,
    timestamp,
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
            [summary()] + comments,
            reviews or [],
            inline or [],
            lambda short: resolved,
            lambda item, is_summary: item["body"].startswith(
                "<!-- codex-pull-request-review-summary -->"
            )
            or item["created_at"] == item["updated_at"],
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
            with self.assertRaises(GateError):
                self.check([item])
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
        inline = {
            **finding,
            "original_commit_id": HEAD,
            "pull_request_review_id": 100,
        }
        self.assertIsNone(
            self.check([completion()], reviews=[{**finding, "id": 100}], inline=[inline])
        )
        finding["commit_id"] = BASE
        self.assertIsNotNone(self.check([completion()], reviews=[finding]))

    def test_api_failure_is_not_empty_approval(self):
        def fail(short):
            raise GateError("GitHub API request failed")

        with self.assertRaises(GateError):
            clean_completion(
                HEAD, [summary(), completion()], [], [], fail, lambda item, summary: True, []
            )
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
        with patch(
            "scripts.subscription_review.api",
            return_value={"event": "push", "path": ".github/workflows/ci.yml"},
        ):
            self.assertEqual(event_pr("workflow_run", {"workflow_run": {"id": 123}}), (None, None))
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
                "scripts.subscription_review.pages",
                side_effect=[[completion()], [], [], [], []] * 2,
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
        with self.assertRaises(GateError):
            self.check(item)
        finding = {"user": completion()["user"], "commit_id": HEAD}
        self.assertIsNone(self.check(reviews=[finding]))
        with self.assertRaises(GateError):
            self.check(attest=False)

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
        batches = (
            [[summary()], [], [], [], []]
            + [[summary()], [], [], [], [reaction()]]
            + [[summary()], [], [], [], [reaction()]]
        )
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
        batches = [[summary()], [], [], [], [], [summary()], [finding], [], [], [reaction()]]
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


class ReviewRefreshTests(unittest.TestCase):
    def check(self, comments, reviews=None, authorize=True):
        return clean_completion(
            HEAD,
            [summary()] + comments,
            reviews or [],
            [],
            lambda short: HEAD,
            lambda item, is_summary: True,
            [reaction()],
            lambda item: authorize,
        )

    def request(self, stamp="2026-09-30T07:09:50Z"):
        return {
            "body": "@codex review\n\nPlease review this exact head.",
            "updated_at": stamp,
            "created_at": stamp,
            "user": {"type": "User", "login": "maintainer"},
        }

    def test_authorized_rereview_request_invalidates_old_explicit_and_automatic_clean(self):
        self.assertIsNone(self.check([completion(), self.request()]))
        self.assertIsNone(self.check([summary(), self.request()]))
        fresh = completion()
        fresh["created_at"] = fresh["updated_at"] = "2026-09-30T07:10:50Z"
        anchor = summary()
        anchor["updated_at"] = "2026-09-30T07:10:50Z"
        anchor["body"] = anchor["body"].replace("07:08:49.384854Z", "07:10:49.384854Z")
        self.assertIsNotNone(self.check([completion(), self.request(), fresh, anchor]))
        fresh["created_at"] = fresh["updated_at"] = "2026-09-30T07:09:50Z"
        self.assertIsNone(self.check([self.request(), fresh]))

    def test_unprivileged_or_quoted_request_does_not_invalidate(self):
        self.assertIsNotNone(self.check([completion(), self.request()], authorize=False))
        for prefix in ("> ", "```text\n", "An example: "):
            request = self.request()
            request["body"] = prefix + request["body"]
            self.assertIsNotNone(self.check([completion(), request]))

    def test_authentic_current_head_running_or_unknown_summary_blocks_old_clean(self):
        for status in ("Running", "Queued", "Unknown"):
            running = summary()
            running["body"] = running["body"].replace("**Completed**", f"**{status}**")
            self.assertIsNone(self.check([completion(), running]))
        running = summary()
        running["body"] = running["body"].replace("**Completed**", "**Running**")
        running["body"] = running["body"].replace("`aaaaaaa`", "`bbbbbbb`")
        self.assertIsNotNone(self.check([completion(), running]))

    def test_rereview_finding_blocks_even_with_previous_clean_and_no_completion_comment(self):
        finding = {"user": completion()["user"], "commit_id": HEAD}
        self.assertIsNone(self.check([completion(), self.request()], [finding]))

    def test_request_authorization_uses_current_repository_permission(self):
        for permission, expected in (
            ("write", True),
            ("admin", True),
            ("maintain", True),
            ("read", False),
            ("triage", False),
        ):
            with patch("scripts.subscription_review.api", return_value={"permission": permission}):
                self.assertEqual(authorized_request(self.request()), expected)
        bad = self.request()
        bad["user"]["login"] = "../other"
        with patch("scripts.subscription_review.api") as api:
            self.assertFalse(authorized_request(bad))
            api.assert_not_called()


class ReviewCallbackTests(unittest.TestCase):
    def source(self, event="pull_request_review"):
        return {
            "id": 123,
            "repository": {"full_name": "mickdarling/rightyo"},
            "status": "completed",
            "name": "Native review activity relay",
            "path": ".github/workflows/review-activity.yml",
            "event": event,
            "pull_requests": [{"number": 44}],
            "head_sha": HEAD,
        }

    def test_authoritative_review_and_inline_callbacks_recheck_associated_pr(self):
        for event in ("pull_request_review", "pull_request_review_comment"):
            with patch("scripts.subscription_review.api", return_value=self.source(event)) as api:
                self.assertEqual(
                    event_pr("workflow_run", {"workflow_run": {"id": 123}}), (44, None)
                )
                api.assert_called_once_with("/repos/mickdarling/rightyo/actions/runs/123")

    def test_forged_wrong_workflow_repository_or_event_is_rejected(self):
        for field, value in (
            ("path", ".github/workflows/attacker.yml"),
            ("event", "workflow_dispatch"),
            ("name", "Other"),
            ("status", "in_progress"),
            ("id", 9),
            ("repository", {"full_name": "other/repository"}),
        ):
            source = {**self.source(), field: value}
            with patch("scripts.subscription_review.api", return_value=source):
                with self.assertRaises(GateError):
                    event_pr("workflow_run", {"workflow_run": {"id": 123}})

    def test_empty_association_requires_unique_open_exact_head_repository_pr(self):
        source = {**self.source(), "pull_requests": []}
        pr = {
            "number": 44,
            "state": "open",
            "head": {"sha": HEAD},
            "base": {"repo": {"full_name": "mickdarling/rightyo"}},
        }
        with (
            patch("scripts.subscription_review.api", return_value=source),
            patch("scripts.subscription_review.pages", return_value=[pr]),
        ):
            self.assertEqual(event_pr("workflow_run", {"workflow_run": {"id": 123}}), (44, None))
        for associated in ([], [pr, pr], [{**pr, "head": {"sha": BASE}}]):
            with (
                patch("scripts.subscription_review.api", return_value=source),
                patch("scripts.subscription_review.pages", return_value=associated),
            ):
                with self.assertRaises(GateError):
                    event_pr("workflow_run", {"workflow_run": {"id": 123}})


class NativeMetadataTamperingTests(unittest.TestCase):
    def test_removed_header_or_duplicate_rows_never_resurrect_old_clean(self):
        for removed in (True, False):
            edited = summary()
            if removed:
                edited["body"] = edited["body"].replace("Codex Review Summary", "Other summary")
            else:
                edited["body"] += edited["body"].splitlines()[-1] + "\n"
            with self.assertRaises(GateError):
                clean_completion(
                    HEAD,
                    [completion(), edited],
                    [],
                    [],
                    lambda short: HEAD,
                    lambda item, is_summary: item is not edited,
                    [],
                )

    def test_explicit_verdict_without_current_completed_cycle_cannot_pass(self):
        self.assertIsNone(
            clean_completion(
                HEAD, [completion()], [], [], lambda short: HEAD, lambda item, summary: True, []
            )
        )


class PublisherRouteBindingTests(unittest.TestCase):
    def test_changed_resolved_pr_never_publishes_outside_serialized_route(self):
        with (
            patch.dict(
                os.environ,
                {
                    "GITHUB_REPOSITORY": "mickdarling/rightyo",
                    "GITHUB_EVENT_NAME": "issue_comment",
                    "GITHUB_EVENT_PATH": "/unused/event.json",
                    "GATE_PR_NUMBER": "45",
                },
            ),
            patch("sys.argv", ["subscription_review.py"]),
            patch("scripts.subscription_review.Path.read_text", return_value="{}"),
            patch("scripts.subscription_review.event_pr", return_value=(44, None)),
            patch("scripts.subscription_review.publish") as publish,
        ):
            with self.assertRaises(SystemExit):
                main()
            publish.assert_not_called()


class InlineSourceBindingTests(unittest.TestCase):
    def parent(self, revision=BASE):
        return {
            "id": 100,
            "user": completion()["user"],
            "commit_id": revision,
            "state": "DISMISSED",
        }

    def inline(self, original=BASE):
        return {
            "user": completion()["user"],
            "commit_id": HEAD,
            "original_commit_id": original,
            "pull_request_review_id": 100,
        }

    def test_forwarded_old_inline_uses_original_review_not_current_rendered_commit(self):
        self.assertEqual(inline_revision(self.inline(), [self.parent()]), BASE)
        result = clean_completion(
            HEAD,
            [summary(), completion()],
            [self.parent()],
            [self.inline()],
            lambda short: HEAD,
            lambda item, is_summary: True,
            [],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["kind"], "explicit")

    def test_original_current_head_remains_blocking_despite_resolution_or_dismissal(self):
        original = self.inline(HEAD)
        original["commit_id"] = BASE
        original["resolved"] = True
        self.assertEqual(inline_revision(original, [self.parent(HEAD)]), HEAD)
        self.assertIsNone(
            clean_completion(
                HEAD,
                [summary(), completion()],
                [self.parent(HEAD)],
                [original],
                lambda short: HEAD,
                lambda item, is_summary: True,
                [],
            )
        )

    def test_missing_malformed_or_mismatched_inline_source_fails_closed(self):
        for field, value in (
            ("original_commit_id", None),
            ("original_commit_id", "invalid"),
            ("pull_request_review_id", None),
            ("pull_request_review_id", True),
        ):
            item = {**self.inline(), field: value}
            with self.assertRaises(GateError):
                inline_revision(item, [self.parent()])
        for parents in (
            [],
            [self.parent(), self.parent()],
            [self.parent(HEAD)],
            [{**self.parent(), "user": {"id": 3, "login": "other", "type": "User"}}],
        ):
            with self.assertRaises(GateError):
                inline_revision(self.inline(), parents)


class DurableRequestTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"GITHUB_RUN_ID": "12345"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def marker(self, cutoff="2026-09-30T07:09:50Z"):
        return {
            "context": "rightyo/review-gate",
            "description": f"v1 comment:99 requested:{cutoff}",
            "creator": {"id": ACTIONS_BOT_ID, "login": "github-actions[bot]", "type": "Bot"},
        }

    def event(self, action="created", body="@codex review"):
        return {
            "action": action,
            "comment": {
                "id": 99,
                "body": body,
                "updated_at": "2026-09-30T07:09:50Z",
                "user": {"login": "maintainer", "type": "User"},
            },
            "sender": {"login": "maintainer", "type": "User"},
        }

    def test_immutable_marker_blocks_old_completion_after_command_edit_or_delete(self):
        for current_comments in ([summary(), completion()], [summary()]):
            with patch("scripts.subscription_review.pages", return_value=[self.marker()]):
                persisted = request_markers(HEAD)
            self.assertIsNone(
                review_candidates(
                    HEAD,
                    current_comments,
                    [],
                    [],
                    lambda short: HEAD,
                    lambda item, summary: True,
                    persisted=persisted,
                )
                or None
            )

    def test_request_recording_is_denial_only_and_not_inside_publisher_mutex(self):
        pr = {"state": "open", "head": {"sha": HEAD}}
        with (
            patch("scripts.subscription_review.current_pr", return_value=pr),
            patch("scripts.subscription_review.authorized_request", return_value=True),
            patch("scripts.subscription_review.request_markers", return_value={}),
            patch("scripts.subscription_review.commit_status") as status,
        ):
            record_request(44, "issue_comment", self.event())
        self.assertEqual(status.call_count, 2)
        self.assertTrue(all(call.args[1] == "pending" for call in status.call_args_list))
        self.assertTrue(
            all(
                call.kwargs.get("context", "rightyo/review-gate") == "rightyo/review-gate"
                for call in status.call_args_list
            )
        )

    def test_removed_command_payload_is_recorded_if_original_capture_was_missed(self):
        edited = self.event("edited", "request removed")
        edited["changes"] = {"body": {"from": "@codex review"}}
        pr = {"state": "open", "head": {"sha": HEAD}}
        for event in (edited, self.event("deleted")):
            with (
                patch("scripts.subscription_review.current_pr", return_value=pr),
                patch("scripts.subscription_review.authorized_request", return_value=True),
                patch("scripts.subscription_review.request_markers", return_value={}),
                patch("scripts.subscription_review.commit_status") as status,
            ):
                record_request(44, "issue_comment", event)
                self.assertEqual(
                    status.call_args.kwargs.get("context", "rightyo/review-gate"),
                    "rightyo/review-gate",
                )

    def test_cleanup_preserves_existing_request_time_without_new_pending_write(self):
        edited = self.event("edited", "request removed")
        edited["changes"] = {"body": {"from": "@codex review"}}
        with (
            patch(
                "scripts.subscription_review.current_pr",
                return_value={"state": "open", "head": {"sha": HEAD}},
            ),
            patch("scripts.subscription_review.authorized_request", return_value=True),
            patch(
                "scripts.subscription_review.request_markers",
                return_value={99: timestamp("2026-09-30T07:08:00Z")},
            ),
            patch("scripts.subscription_review.commit_status") as status,
        ):
            record_request(44, "issue_comment", edited)
            status.assert_not_called()
        candidates = review_candidates(
            HEAD,
            [summary(), completion()],
            [],
            [],
            lambda short: HEAD,
            lambda item, summary: True,
            persisted={99: timestamp("2026-09-30T07:08:00Z")},
        )
        self.assertTrue(candidates)

    def test_unprivileged_or_benign_comment_is_noop(self):
        with (
            patch(
                "scripts.subscription_review.current_pr",
                return_value={"state": "open", "head": {"sha": HEAD}},
            ),
            patch("scripts.subscription_review.authorized_request", return_value=False),
            patch("scripts.subscription_review.commit_status") as status,
        ):
            record_request(44, "issue_comment", self.event())
            record_request(44, "issue_comment", self.event(body="ordinary discussion"))
            status.assert_not_called()

    def test_malformed_trusted_marker_fails_closed_and_other_creators_cannot_approve(self):
        bad = {**self.marker(), "description": "v1 comment:unsupported"}
        with patch("scripts.subscription_review.pages", return_value=[bad]):
            with self.assertRaises(GateError):
                request_markers(HEAD)
        outsider = {**self.marker(), "creator": {"id": 3, "login": "other"}}
        with patch("scripts.subscription_review.pages", return_value=[outsider]):
            self.assertEqual(request_markers(HEAD), {})

    def test_marker_context_can_never_write_success(self):
        with patch("scripts.subscription_review.api") as api:
            with self.assertRaises(GateError):
                commit_status(HEAD, "success", "forged approval", context=REQUEST_CONTEXT)
            api.assert_not_called()

    def test_edited_command_records_new_cutoff_then_removal_preserves_that_time(self):
        edited = self.event("edited", "@codex review\nnew instructions")
        edited["changes"] = {"body": {"from": "@codex review"}}
        old = timestamp("2026-09-30T07:08:00Z")
        with (
            patch(
                "scripts.subscription_review.current_pr",
                return_value={"state": "open", "head": {"sha": HEAD}},
            ),
            patch("scripts.subscription_review.authorized_request", return_value=True),
            patch("scripts.subscription_review.request_markers", return_value={99: old}),
            patch("scripts.subscription_review.commit_status") as status,
        ):
            record_request(44, "issue_comment", edited)
        self.assertIn("07:09:50Z", status.call_args.args[2])

    def test_permission_or_history_failure_persists_denial_for_later_removed_comment(self):
        for failure in ("authorization", "history"):
            with (
                patch(
                    "scripts.subscription_review.current_pr",
                    return_value={"state": "open", "head": {"sha": HEAD}},
                ),
                patch(
                    "scripts.subscription_review.authorized_request",
                    side_effect=GateError("API failure") if failure == "authorization" else None,
                    return_value=True,
                ),
                patch(
                    "scripts.subscription_review.request_markers",
                    side_effect=GateError("API failure"),
                ),
                patch("scripts.subscription_review.commit_status") as status,
            ):
                with self.assertRaises(GateError):
                    record_request(44, "issue_comment", self.event())
                self.assertEqual(status.call_count, 2)
                self.assertTrue(all(call.args[1] == "pending" for call in status.call_args_list))

    def test_required_pending_marker_alone_preserves_cutoff_if_second_write_fails(self):
        fallback = {**self.marker(), "context": "rightyo/review-gate"}
        with patch("scripts.subscription_review.pages", return_value=[fallback]):
            markers = request_markers(HEAD)
        self.assertEqual(markers[99], timestamp("2026-09-30T07:09:50Z"))

    def test_durable_request_also_requires_a_newer_completed_summary_cycle(self):
        fresh = completion()
        fresh["created_at"] = fresh["updated_at"] = "2026-09-30T07:10:50Z"
        self.assertEqual(
            review_candidates(
                HEAD,
                [summary(), fresh],
                [],
                [],
                lambda short: HEAD,
                lambda item, summary: True,
                persisted={99: timestamp("2026-09-30T07:09:50Z")},
            ),
            [],
        )

    def test_second_successful_required_marker_both_denies_and_persists_after_first_failure(self):
        calls = []

        def write(head, state, description, **kwargs):
            calls.append((head, state, description, kwargs.get("context", "rightyo/review-gate")))
            if len(calls) == 1:
                raise GateError("First status write failed")

        with patch("scripts.subscription_review.commit_status", side_effect=write):
            with self.assertRaises(GateError):
                persist_request(HEAD, 99, timestamp("2026-09-30T07:09:50Z"))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[-1][1], "pending")
        self.assertEqual(calls[-1][3], "rightyo/review-gate")
        self.assertTrue(calls[-1][2].startswith("v1 comment:99 requested:"))
