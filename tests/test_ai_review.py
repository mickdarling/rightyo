"""Offline trust-boundary tests; no provider calls, credentials or GitHub writes."""

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("ai_review", SCRIPTS / "ai_review.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
HEAD, BASE, WORKFLOW = "a" * 40, "b" * 40, "c" * 40
META = {
    "repository": review.REPOSITORY,
    "pr": 39,
    "head": HEAD,
    "base": BASE,
    "workflow_sha": WORKFLOW,
    "run_id": 123,
    "run_attempt": 1,
    "check_ids": {"codex-review": 1, "claude-review": 2, "review-gate": 3},
}
DATA = {"metadata": META, "files": [{"path": "src/demo.py", "before": "old\n", "after": "new\n"}]}
RESULT = {
    "status": "completed",
    "head": HEAD,
    "base": BASE,
    "findings": [],
    "limitations": ["Static review only"],
    "reviewed_files": ["src/demo.py"],
}
JOBS = {job: {"result": "success"} for job in ("prepare", "codex", "claude")}


class ReviewValidationTests(unittest.TestCase):
    def test_clean_completed_coverage_is_valid(self):
        self.assertEqual(review.validate(copy.deepcopy(RESULT), DATA), RESULT)

    def test_empty_findings_do_not_approve_partial_stale_or_malformed_review(self):
        for changes in (
            {"status": "inconclusive"},
            {"head": BASE},
            {"base": HEAD},
            {"reviewed_files": []},
            {"reviewed_files": ["other.py"]},
            {"limitations": "none"},
            {"findings": {}},
            {"extra": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(review.ReviewError):
                review.validate({**RESULT, **changes}, DATA)

    def test_findings_must_locate_actual_source_and_use_typed_severity(self):
        finding = {
            "priority": 2,
            "path": "src/demo.py",
            "line": 1,
            "side": "after",
            "evidence": "Concrete defect",
        }
        review.validate({**RESULT, "findings": [finding]}, DATA)
        for change in (
            {"priority": True},
            {"priority": 4},
            {"path": "other.py"},
            {"line": 2},
            {"line": True},
            {"side": "unknown"},
            {"evidence": ""},
        ):
            with self.subTest(change=change), self.assertRaises(review.ReviewError):
                review.validate({**RESULT, "findings": [{**finding, **change}]}, DATA)

    def test_protected_paths_are_refused_before_fetch(self):
        for path in (
            "../secret",
            "/etc/passwd",
            "recordings/demo.wav",
            ".env",
            "nested/transcripts/demo.json",
            "models/checkpoint.gguf",
            "bad\npath",
        ):
            with self.subTest(path=path), self.assertRaises(review.ReviewError):
                review.source_path(path)
        self.assertEqual(review.source_path("src/demo.py"), "src/demo.py")

    def test_digest_mismatch_or_oversize_prevents_review(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw = json.dumps(DATA).encode()
            (directory / "snapshot.json").write_bytes(raw)
            self.assertEqual(review.load_snapshot(directory, hashlib.sha256(raw).hexdigest()), DATA)
            with self.assertRaises(review.ReviewError):
                review.load_snapshot(directory, "0" * 64)
            with patch.object(review, "MAX_SNAPSHOT", 1), self.assertRaises(review.ReviewError):
                review.load_snapshot(directory, hashlib.sha256(raw).hexdigest())

    def test_api_never_follows_a_redirect_to_another_destination(self):
        with self.assertRaises(review.ReviewError):
            review.NoRedirect().redirect_request(None, None, 302, "", {}, "https://elsewhere")
        with patch.dict(os.environ, {"GITHUB_TOKEN": "invented-test-value"}):
            with self.assertRaises(review.ReviewError):
                review.api("/repos/another/repo/pulls/39")

    def test_complete_file_count_is_required_not_truncated_api_patch(self):
        pr = {"changed_files": 2}
        replies = [{"merge_base_commit": {"sha": BASE}}, [{"filename": "src/demo.py"}]]
        with (
            patch.object(review, "api", side_effect=replies),
            self.assertRaises(review.ReviewError),
        ):
            review.snapshot(pr, META)

    def test_fork_authorization_requires_current_maintainer_dispatch(self):
        fork = {"head": {"repo": {"full_name": "someone/fork"}}}
        with patch.object(review, "api", return_value={"permission": "write"}):
            self.assertFalse(review.allowed(fork, "pull_request_target", "maintainer"))
            self.assertTrue(review.allowed(fork, "workflow_dispatch", "maintainer"))
        with patch.object(review, "api", return_value={"permission": "read"}):
            self.assertFalse(review.allowed(fork, "workflow_dispatch", "outsider"))


class PublisherTests(unittest.TestCase):
    def publish(
        self, *, result=None, jobs=None, current_head=HEAD, current_base=BASE, corrupt_provider=None
    ):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            raw = json.dumps(DATA).encode()
            digest = hashlib.sha256(raw).hexdigest()
            (directory / "snapshot.json").write_bytes(raw)
            for provider in review.PROVIDERS:
                record = {
                    "provider": provider,
                    "metadata": META,
                    "snapshot_sha": digest,
                    "result": copy.deepcopy(result or RESULT),
                }
                if provider == corrupt_provider:
                    record["provider"] = "spoofed"
                (directory / f"{provider}.json").write_text(json.dumps(record))
            checks = []
            with (
                patch.object(
                    review,
                    "current_pr",
                    return_value={"head": {"sha": current_head}, "base": {"sha": current_base}},
                ),
                patch.object(review, "check", side_effect=lambda *a, **k: checks.append((a, k))),
            ):
                try:
                    review.publish(directory, META, digest, jobs or JOBS)
                    failed = False
                except review.ReviewError:
                    failed = True
            return failed, checks

    def test_clean_results_publish_three_current_head_successes(self):
        failed, checks = self.publish()
        self.assertFalse(failed)
        self.assertEqual(len(checks), 3)
        self.assertTrue(
            all(args[1] == HEAD and kwargs["conclusion"] == "success" for args, kwargs in checks)
        )

    def test_each_missing_failed_skipped_timeout_or_cancelled_job_blocks_gate(self):
        for provider in ("prepare", "codex", "claude"):
            for outcome in ("failure", "cancelled", "skipped", "timed_out", ""):
                jobs = copy.deepcopy(JOBS)
                jobs[provider]["result"] = outcome
                with self.subTest(provider=provider, outcome=outcome):
                    failed, checks = self.publish(jobs=jobs)
                    self.assertTrue(failed)
                    self.assertEqual(checks[-1][1]["conclusion"], "failure")

    def test_changed_head_or_base_invalidates_both_reviews(self):
        for changes in ({"current_head": WORKFLOW}, {"current_base": WORKFLOW}):
            with self.subTest(changes=changes):
                failed, checks = self.publish(**changes)
                self.assertTrue(failed)
                self.assertTrue(all(kwargs["conclusion"] == "failure" for _, kwargs in checks))

    def test_spoofed_provider_record_blocks_gate(self):
        failed, checks = self.publish(corrupt_provider="codex")
        self.assertTrue(failed)
        self.assertEqual(checks[0][1]["conclusion"], "failure")

    def test_medium_findings_block_low_findings_are_visible(self):
        finding = {
            "priority": 2,
            "path": "src/demo.py",
            "line": 1,
            "side": "after",
            "evidence": "Concrete defect <untrusted>",
        }
        result = {**RESULT, "findings": [finding]}
        failed, checks = self.publish(result=result)
        self.assertTrue(failed)
        self.assertIn("&lt;untrusted&gt;", checks[0][1]["summary"])
        finding["priority"] = 3
        failed, checks = self.publish(result=result)
        self.assertFalse(failed)
        self.assertIn("P3", checks[0][1]["summary"])


if __name__ == "__main__":
    unittest.main()
