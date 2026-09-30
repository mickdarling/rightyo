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
    def execute(self, payload, *, event="workflow_run", repository=REPOSITORY, failure=None):
        self.requests = []

        def send(request, timeout):
            self.requests.append(request)
            self.assertEqual(timeout, 20)
            if failure and failure(request):
                raise OSError("synthetic API outage")
            response = MagicMock()
            response.__enter__.return_value.status = 201
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
        with self.assertRaisesRegex(SystemExit, "publication failed"):
            self.execute(payload, failure=lambda request: request.full_url.endswith(HEAD))
        self.assertEqual(
            {request.full_url.rsplit("/", 1)[1] for request in self.requests}, {HEAD, OTHER}
        )
        self.assertTrue(
            all(json.loads(request.data)["state"] == "pending" for request in self.requests)
        )

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
