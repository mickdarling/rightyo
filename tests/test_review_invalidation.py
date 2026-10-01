"""Execute the trusted denial-only bootstrap without networking or real credentials."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.repository_checks import REVIEW_INVALIDATION_COMMAND
from scripts.subscription_review import GateError, event_pr

HEAD = "a" * 40
OTHER = "b" * 40
REPOSITORY = "mickdarling/rightyo"
PROGRAM = REVIEW_INVALIDATION_COMMAND.split("\n", 1)[1].rsplit("\nPY", 1)[0]


def notification():
    return {
        "workflow": {"path": ".github/workflows/review-activity.yml"},
        "workflow_run": {
            "id": 31,
            "name": "Native review activity relay",
            "repository": {"full_name": REPOSITORY},
            "path": ".github/workflows/review-activity.yml",
            "event": "pull_request_review",
            "status": "completed",
            "head_sha": HEAD,
            "pull_requests": [{"number": 44, "head": {"sha": HEAD}}],
        },
    }


class ReviewInvalidationTests(unittest.TestCase):
    def execute(
        self, payload, *, event="workflow_run", repository=REPOSITORY, failure=None, read=None
    ):
        self.requests = []

        def send(request, timeout):
            self.requests.append(request)
            self.assertEqual(timeout, 5)
            if failure and failure(request):
                raise OSError("synthetic API outage")
            response = MagicMock()
            response.__enter__.return_value.status = 201 if request.get_method() == "POST" else 200
            if read is not None and request.get_method() == "GET":
                response.__enter__.return_value.read.return_value = json.dumps(
                    read(request)
                ).encode()
            return response

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "event.json"
            path.write_text(json.dumps(payload))
            environment = {
                "GITHUB_REPOSITORY": repository,
                "GITHUB_EVENT_NAME": event,
                "GITHUB_EVENT_PATH": str(path),
                "GITHUB_RUN_ID": "35",
                "GITHUB_TOKEN": "synthetic-test-token",
            }
            opener = MagicMock()
            opener.open.side_effect = send
            with (
                patch.dict(os.environ, environment, clear=True),
                patch("urllib.request.build_opener", return_value=opener),
                redirect_stdout(io.StringIO()),
            ):
                exec(compile(PROGRAM, "trusted-bootstrap", "exec"), {})

    def test_review_and_inline_callbacks_only_deny(self):
        for source_event in ("pull_request_review", "pull_request_review_comment"):
            payload = notification()
            payload["workflow_run"]["event"] = source_event
            self.execute(payload)
            self.assertEqual(len(self.requests), 1)
            request = self.requests[0]
            self.assertEqual(
                request.full_url, f"https://api.github.com/repos/{REPOSITORY}/statuses/{HEAD}"
            )
            self.assertEqual(request.get_method(), "POST")
            status = json.loads(request.data)
            self.assertEqual(status["state"], "pending")
            self.assertEqual(status["context"], "rightyo/review-gate")
            self.assertEqual(
                status["target_url"], f"https://github.com/{REPOSITORY}/actions/runs/35"
            )

    def test_resolver_api_failure_cannot_retain_old_success_after_invalidation(self):
        payload = notification()
        self.execute(payload)
        with patch("scripts.subscription_review.api", side_effect=GateError("synthetic failure")):
            with self.assertRaises(GateError):
                event_pr("workflow_run", payload)
        self.assertEqual(
            [json.loads(request.data)["state"] for request in self.requests], ["pending"]
        )

    def test_both_full_revision_hints_are_denied_even_if_one_write_fails(self):
        payload = notification()
        payload["workflow_run"]["pull_requests"][0]["head"]["sha"] = OTHER
        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(payload, failure=lambda request: request.full_url.endswith(HEAD))
        self.assertEqual(
            {request.full_url.rsplit("/", 1)[1] for request in self.requests}, {HEAD, OTHER}
        )
        self.assertTrue(
            all(json.loads(request.data)["state"] == "pending" for request in self.requests)
        )

    def empty_relay(self):
        payload = notification()
        payload["workflow_run"]["head_sha"] = OTHER
        payload["workflow_run"]["pull_requests"] = []
        return payload

    def association(self, *, head=HEAD, number=44, merge=OTHER):
        return {
            "number": number,
            "state": "open",
            "head": {"sha": head},
            "base": {"repo": {"full_name": REPOSITORY}},
            "merge_commit_sha": merge,
        }

    def empty_read(self, request):
        if "/commits/" in request.full_url:
            return []  # Actual GitHub merge-ref commits have no PR associations.
        if "state=open" in request.full_url:
            return [self.association()]
        return self.association()

    def test_empty_merge_association_denies_actual_head_before_resolver_failure(self):
        payload = self.empty_relay()
        self.execute(payload, read=self.empty_read)
        self.assertEqual(
            [
                (request.get_method(), request.full_url.rsplit("/", 1)[1])
                for request in self.requests
            ],
            [
                ("POST", OTHER),
                ("GET", "pulls?per_page=100&page=1"),
                ("GET", "pulls?state=open&per_page=100&page=1"),
                ("GET", "44"),
                ("POST", HEAD),
            ],
        )
        with patch(
            "scripts.subscription_review.api", side_effect=GateError("checkout/API failure")
        ):
            with self.assertRaises(GateError):
                event_pr("workflow_run", payload)
        writes = [request for request in self.requests if request.get_method() == "POST"]
        self.assertEqual([json.loads(request.data)["state"] for request in writes], ["pending"] * 2)

    def test_empty_commit_association_uses_authoritative_current_head(self):
        def read(request):
            if "/commits/" in request.full_url:
                return [self.association(head=OTHER)]
            return self.association()

        self.execute(self.empty_relay(), read=read)
        self.assertEqual(self.requests[-1].full_url.rsplit("/", 1)[1], HEAD)
        self.assertEqual(len(self.requests), 4)

    def test_failed_source_hint_write_still_denies_discovered_current_head(self):
        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(
                self.empty_relay(),
                read=self.empty_read,
                failure=lambda request: request.get_method() == "POST"
                and request.full_url.endswith(OTHER),
            )
        self.assertEqual(self.requests[-1].full_url.rsplit("/", 1)[1], HEAD)
        self.assertEqual(json.loads(self.requests[-1].data)["state"], "pending")

    def test_missing_ambiguous_overflow_and_nonmatching_inventory_fail_after_hint_denial(self):
        for inventory in (
            [],
            [self.association(), self.association(number=45)],
            [self.association(merge="c" * 40, head="c" * 40)],
            [self.association()] * 100,
        ):

            def read(request):
                return [] if "/commits/" in request.full_url else inventory

            with (
                self.subTest(inventory=inventory),
                self.assertRaisesRegex(SystemExit, "capture failed"),
            ):
                self.execute(self.empty_relay(), read=read)
            self.assertEqual(json.loads(self.requests[0].data)["state"], "pending")
            self.assertEqual(sum(request.get_method() == "POST" for request in self.requests), 1)

    def test_empty_association_lookup_failure_preserves_hint_denial(self):
        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(
                self.empty_relay(),
                read=self.empty_read,
                failure=lambda request: request.get_method() == "GET",
            )
        self.assertEqual(json.loads(self.requests[0].data)["state"], "pending")
        self.assertEqual(len(self.requests), 2)

    def test_discovered_current_pr_must_match_repository_number_and_full_revision(self):
        for changes in (
            {"number": 45},
            {"state": "closed"},
            {"head": {"sha": "a" * 7}},
            {"base": {"repo": {"full_name": "other/repo"}}},
        ):

            def read(request):
                if "state=open" in request.full_url or "/commits/" in request.full_url:
                    return self.empty_read(request)
                return {**self.association(), **changes}

            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(SystemExit, "capture failed"),
            ):
                self.execute(self.empty_relay(), read=read)
            self.assertEqual(sum(request.get_method() == "POST" for request in self.requests), 1)

    def test_top_level_workflow_path_can_supply_missing_run_path(self):
        payload = notification()
        del payload["workflow_run"]["path"]
        self.execute(payload)
        self.assertEqual(len(self.requests), 1)

    def test_nullable_top_level_workflow_uses_valid_run_path(self):
        payload = notification()
        payload["workflow"] = None
        self.execute(payload)
        self.assertEqual(len(self.requests), 1)

    def test_invalid_route_or_revision_never_publishes(self):
        for field, value in (
            ("repository", {"full_name": "outsider/repo"}),
            ("path", ".github/workflows/ci.yml"),
            ("event", "push"),
            ("status", "in_progress"),
            ("head_sha", "a" * 7),
            ("head_sha", {"invalid": "shape"}),
            ("pull_requests", None),
            ("pull_requests", [{"head": {"sha": "invalid"}}]),
        ):
            payload = notification()
            payload["workflow_run"][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(SystemExit):
                self.execute(payload)
            self.assertEqual(self.requests, [])
        payload = notification()
        del payload["workflow_run"]["path"]
        del payload["workflow"]
        with self.assertRaises(SystemExit):
            self.execute(payload)
        self.assertEqual(self.requests, [])

    def test_non_relay_events_do_not_publish(self):
        for event in ("issue_comment", "pull_request_target", "workflow_dispatch"):
            with self.assertRaises(SystemExit) as raised:
                self.execute({}, event=event)
            self.assertEqual(raised.exception.code, 0)
            self.assertEqual(self.requests, [])
        payload = copy.deepcopy(notification())
        payload["workflow_run"]["name"] = "RightyO CI"
        with self.assertRaises(SystemExit) as raised:
            self.execute(payload)
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(self.requests, [])


class RequestBootstrapTests(unittest.TestCase):
    execute = ReviewInvalidationTests.execute

    def request_event(self, action="created", body="@codex review", old=None):
        result = {
            "action": action,
            "issue": {"number": 44, "pull_request": {"url": "synthetic"}},
            "comment": {
                "id": 55,
                "body": body,
                "updated_at": "2026-09-30T07:10:00Z",
                "user": {"type": "User", "login": "owner"},
            },
            "sender": {"type": "User", "login": "owner"},
        }
        if old is not None:
            result["changes"] = {"body": {"from": old}}
        return result

    def read(self, request):
        if "/pulls/44" in request.full_url:
            return {
                "state": "open",
                "head": {"sha": HEAD},
                "base": {"repo": {"full_name": REPOSITORY}},
            }
        if "/permission" in request.full_url:
            return {"permission": "admin"}
        return []

    def writes(self):
        return [
            json.loads(request.data) for request in self.requests if request.get_method() == "POST"
        ]

    def test_authorized_create_edit_away_delete_capture_before_checkout(self):
        for payload in (
            self.request_event(),
            self.request_event("edited", "done", "@codex review"),
            self.request_event("deleted"),
            self.request_event(body="@codex security review"),
        ):
            self.execute(payload, event="issue_comment", read=self.read)
            statuses = self.writes()
            self.assertEqual([status["state"] for status in statuses], ["pending"] * 3)
            self.assertTrue(all(status["context"] == "rightyo/review-gate" for status in statuses))
            self.assertEqual(
                statuses[-1]["description"], "v1 comment:55 requested:2026-09-30T07:10:00Z"
            )
            self.assertEqual(self.requests[0].get_method(), "GET")
            self.assertIn("/pulls/44", self.requests[0].full_url)
            self.assertEqual(self.requests[1].get_method(), "POST")
            self.assertIn("/permission", self.requests[2].full_url)
            # A later source-resolution failure cannot erase the captured immutable marker.
            with patch(
                "scripts.subscription_review.api", side_effect=GateError("resolution failed")
            ):
                with self.assertRaises(GateError):
                    event_pr("workflow_run", {"workflow_run": {"id": 31}})
            self.assertTrue(self.writes()[-1]["description"].startswith("v1 comment:"))

    def test_known_unauthorized_has_no_permanent_marker(self):
        def denied(request):
            if "/permission" in request.full_url:
                return {"permission": "read"}
            return self.read(request)

        with self.assertRaises(SystemExit) as raised:
            self.execute(self.request_event(), event="issue_comment", read=denied)
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(len(self.writes()), 1)
        self.assertFalse(self.writes()[0]["description"].startswith("v1 comment:"))

    def test_permission_failure_preserves_pending_and_conservative_durable_cutoff(self):
        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(
                self.request_event(),
                event="issue_comment",
                read=self.read,
                failure=lambda request: "/permission" in request.full_url,
            )
        self.assertEqual(len(self.writes()), 3)
        self.assertTrue(self.writes()[-1]["description"].startswith("v1 comment:55"))
        self.assertTrue(all(status["state"] == "pending" for status in self.writes()))

    def test_history_failure_on_removed_command_captures_conservative_cutoff(self):
        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(
                self.request_event("edited", "done", "@codex review"),
                event="issue_comment",
                read=self.read,
                failure=lambda request: "/commits/" in request.full_url,
            )
        self.assertEqual(len(self.writes()), 3)
        self.assertTrue(self.writes()[-1]["description"].startswith("v1 comment:55"))

    def test_removal_preserves_existing_marker_time_after_completed_review(self):
        def previous(request):
            if "/commits/" in request.full_url:
                return [
                    {
                        "context": "rightyo/review-gate",
                        "creator": {"id": 41898282, "login": "github-actions[bot]"},
                        "description": "v1 comment:55 requested:2026-09-30T07:00:00Z",
                    }
                ]
            return self.read(request)

        with self.assertRaises(SystemExit) as raised:
            self.execute(self.request_event("deleted"), event="issue_comment", read=previous)
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(len(self.writes()), 1)
        self.assertFalse(self.writes()[0]["description"].startswith("v1 comment:"))

    def test_second_required_marker_write_survives_first_marker_write_failure(self):
        posts = 0

        def fail(request):
            nonlocal posts
            if request.get_method() == "POST":
                posts += 1
                return posts == 2
            return False

        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(self.request_event(), event="issue_comment", read=self.read, failure=fail)
        self.assertEqual(posts, 3)
        self.assertEqual(self.writes()[-1]["context"], "rightyo/review-gate")
        self.assertTrue(self.writes()[-1]["description"].startswith("v1 comment:55"))

    def test_head_lookup_failure_is_explicit_availability_boundary(self):
        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(
                self.request_event(),
                event="issue_comment",
                read=self.read,
                failure=lambda request: "/pulls/44" in request.full_url,
            )
        self.assertEqual(self.writes(), [])

    def test_copied_benign_or_nonpr_commands_are_noops(self):
        for payload in (
            self.request_event(body="> @codex review"),
            self.request_event(body="```\n@codex review\n```"),
            self.request_event(body="Thanks"),
        ):
            with self.assertRaises(SystemExit) as raised:
                self.execute(payload, event="issue_comment", read=self.read)
            self.assertEqual(raised.exception.code, 0)
            self.assertEqual(self.requests, [])
        payload = self.request_event()
        del payload["issue"]["pull_request"]
        with self.assertRaises(SystemExit):
            self.execute(payload, event="issue_comment", read=self.read)
        self.assertEqual(self.requests, [])

    def test_initial_pending_failure_still_attempts_required_durable_denial(self):
        posts = 0

        def fail(request):
            nonlocal posts
            if request.get_method() == "POST":
                posts += 1
                return posts == 1
            return False

        with self.assertRaisesRegex(SystemExit, "capture failed"):
            self.execute(self.request_event(), event="issue_comment", read=self.read, failure=fail)
        self.assertEqual(posts, 3)
        self.assertTrue(self.writes()[-1]["description"].startswith("v1 comment:55"))
        self.assertTrue(all(status["state"] == "pending" for status in self.writes()))

    def test_unknown_permission_metadata_gets_conservative_denial(self):
        for malformed in ({}, {"permission": "future-role"}):

            def read(request):
                return malformed if "/permission" in request.full_url else self.read(request)

            with self.assertRaisesRegex(SystemExit, "capture failed"):
                self.execute(self.request_event(), event="issue_comment", read=read)
            self.assertEqual(len(self.writes()), 3)
            self.assertTrue(self.writes()[-1]["description"].startswith("v1 comment:55"))
