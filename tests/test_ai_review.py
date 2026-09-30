"""Offline trust-boundary tests; no provider calls, credentials or GitHub writes."""

import base64
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
    def test_linux_missing_or_correct_optional_sysctls_need_no_write(self):
        for result in (
            review.subprocess.CompletedProcess([], 1, b""),
            None,
        ):
            with self.subTest(missing=result is not None):
                results = (
                    [result, result]
                    if result
                    else [
                        review.subprocess.CompletedProcess([], 0, b"1\n"),
                        review.subprocess.CompletedProcess([], 0, b"0\n"),
                    ]
                )
                with patch.object(review.subprocess, "run", side_effect=results) as run:
                    review.configure_linux_sandbox()
                    self.assertEqual(run.call_count, 2)
                    self.assertTrue(
                        all(call.args[0][:2] == ["sysctl", "-n"] for call in run.call_args_list)
                    )

    def test_linux_present_wrong_setting_must_be_updated_or_fail(self):
        query = review.subprocess.CompletedProcess([], 0, b"0\n")
        missing = review.subprocess.CompletedProcess([], 1, b"")
        with patch.object(review.subprocess, "run", side_effect=[query, missing, missing]) as run:
            review.configure_linux_sandbox()
            self.assertEqual(
                run.call_args_list[1].args[0],
                ["sudo", "sysctl", "-w", "kernel.unprivileged_userns_clone=1"],
            )
        with patch.object(
            review.subprocess,
            "run",
            side_effect=[query, review.subprocess.CalledProcessError(1, "sysctl")],
        ):
            with self.assertRaisesRegex(review.ReviewError, "kernel setup failed"):
                review.configure_linux_sandbox()

    def test_line_locations_use_git_lf_coordinates_not_unicode_separators(self):
        self.assertEqual(review.github_line_count("one\u2028two\vthree\nlast\n"), 2)
        self.assertEqual(review.github_line_count(""), 0)
        self.assertEqual(review.github_line_count("one\n"), 1)

    def test_dispatch_rejects_changed_approved_head_or_base_before_writes(self):
        pr = {"head": {"sha": HEAD}, "base": {"sha": BASE}}
        with (
            patch.dict(os.environ, {"GITHUB_REPOSITORY": review.REPOSITORY}),
            patch.object(review, "current_pr", return_value=pr),
            patch.object(review, "check") as check,
        ):
            for changes in ({"expected_head": WORKFLOW}, {"expected_base": WORKFLOW}):
                with self.assertRaises(review.ReviewError):
                    review.prepare(Path("unused"), 39, **changes)
            check.assert_not_called()

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

    def test_symlinks_and_submodules_are_refused_before_blob_fetch(self):
        for mode in ("120000", "160000"):
            tree = {"src/demo.py": {"type": "blob", "mode": mode, "sha": HEAD}}
            with (
                self.subTest(mode=mode),
                patch.object(review, "source_tree", return_value=tree),
                patch.object(review, "api") as api,
                self.assertRaises(review.ReviewError),
            ):
                review.source("src/demo.py", HEAD)
            api.assert_not_called()
        tree = {"src/demo.py": {"type": "blob", "mode": "100644", "sha": HEAD}}
        with (
            patch.object(review, "source_tree", return_value=tree),
            patch.object(
                review,
                "api",
                return_value={
                    "encoding": "base64",
                    "content": base64.b64encode(b"ordinary source").decode(),
                },
            ),
        ):
            self.assertEqual(review.source("src/demo.py", HEAD), "ordinary source")

    def test_truncated_tree_cannot_silently_omit_source(self):
        review.source_tree.cache_clear()
        with (
            patch.object(review, "api", return_value={"truncated": True, "tree": []}),
            self.assertRaises(review.ReviewError),
        ):
            review.source_tree(HEAD)
        review.source_tree.cache_clear()

    def test_dependabot_allowlist_precedes_collaborator_lookup(self):
        pr = {"head": {"repo": {"full_name": review.REPOSITORY}}}
        with patch.object(review, "api") as api:
            self.assertTrue(review.allowed(pr, "workflow_run", "dependabot[bot]"))
            api.assert_not_called()

    def test_file_coverage_comes_from_immutable_comparison_not_live_pr_endpoint(self):
        comparison = {
            "merge_base_commit": {"sha": BASE},
            "files": [{"filename": "src/demo.py", "status": "modified"}],
        }
        with (
            patch.object(review, "api", return_value=comparison) as api,
            patch.object(review, "source", side_effect=["old", "new"]),
        ):
            data = json.loads(review.snapshot({"changed_files": 1}, META))
        self.assertEqual(data["files"][0]["after"], "new")
        self.assertEqual(api.call_count, 1)
        self.assertIn(f"/compare/{BASE}...{HEAD}", api.call_args.args[0])

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
        replies = [{"merge_base_commit": {"sha": BASE}, "files": [{"filename": "src/demo.py"}]}]
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


class ForkPreservationTests(unittest.TestCase):
    def test_stacked_pr_callback_binds_current_source_head(self):
        run = {
            "event": "pull_request",
            "repository": {"full_name": review.REPOSITORY},
            "pull_requests": [{"number": 39}],
            "head_sha": HEAD,
            "actor": {"login": "maintainer"},
        }
        with tempfile.TemporaryDirectory() as temp:
            event = Path(temp) / "event.json"
            event.write_text(json.dumps({"workflow_run": run}))
            with (
                patch.dict(
                    os.environ,
                    {"GITHUB_EVENT_NAME": "workflow_run", "GITHUB_EVENT_PATH": str(event)},
                ),
                patch.object(sys, "argv", ["ai_review.py", "prepare", "--directory", temp]),
                patch.object(review, "current_pr", return_value={"head": {"sha": HEAD}}),
                patch.object(review, "prepare") as prepare,
            ):
                review.main()
                prepare.assert_called_once_with(Path(temp), 39, HEAD, None)

    def test_stale_or_non_pr_callback_never_prepares_review(self):
        for source_event, source_head in (("push", HEAD), ("pull_request", BASE)):
            with self.subTest(source_event=source_event), tempfile.TemporaryDirectory() as temp:
                event = Path(temp) / "event.json"
                event.write_text(
                    json.dumps(
                        {
                            "workflow_run": {
                                "event": source_event,
                                "repository": {"full_name": review.REPOSITORY},
                                "pull_requests": [{"number": 39}],
                                "head_sha": source_head,
                            }
                        }
                    )
                )
                with (
                    patch.dict(
                        os.environ,
                        {"GITHUB_EVENT_NAME": "workflow_run", "GITHUB_EVENT_PATH": str(event)},
                    ),
                    patch.object(sys, "argv", ["ai_review.py", "prepare", "--directory", temp]),
                    patch.object(review, "current_pr", return_value={"head": {"sha": HEAD}}),
                    patch.object(review, "prepare") as prepare,
                    patch("builtins.print"),
                    self.assertRaises(SystemExit),
                ):
                    review.main()
                prepare.assert_not_called()

    def test_auto_fork_metadata_event_preserves_current_dispatched_checks(self):
        fork = {"head": {"sha": HEAD, "repo": {"full_name": "someone/fork"}}, "base": {"sha": BASE}}
        prefix = f"Head: `{HEAD}`; Base: `{BASE}`."
        existing = {
            "check_runs": [
                {"name": "rightyo/" + name, "app": {"id": 15368}, "output": {"summary": prefix}}
                for name in ("codex-review", "claude-review", "review-gate")
            ]
            + [{"name": "unrelated", "app": {"id": 15368}, "output": {"summary": None}}]
        }
        environment = {
            "GITHUB_REPOSITORY": review.REPOSITORY,
            "GITHUB_EVENT_NAME": "pull_request_target",
            "GITHUB_WORKFLOW_SHA": WORKFLOW,
            "GITHUB_RUN_ID": "123",
            "GITHUB_RUN_ATTEMPT": "1",
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.dict(os.environ, environment),
            patch.object(review, "current_pr", return_value=fork),
            patch.object(review, "api", return_value=existing),
            patch.object(review, "check") as check,
            patch.object(review, "output") as output,
        ):
            for event_name in ("pull_request_target", "workflow_run"):
                os.environ["GITHUB_EVENT_NAME"] = event_name
                review.prepare(Path(temp), 39)
                meta = json.loads(output.call_args.args[0]["metadata"])
                self.assertTrue(meta["preserve_checks"])
                check.assert_not_called()
                review.publish(Path(temp), meta, "", {})
                check.assert_not_called()


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
