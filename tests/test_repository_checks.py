"""Synthetic regressions for repository policy, not scans of private material."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

from scripts.repository_checks import (
    AI_REVIEW_WORKFLOW,
    MAX_FILE_BYTES,
    artifact_errors,
    artifact_reason,
    ignore_errors,
    issue_references,
    markdown_errors,
    workflow_errors,
)

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_unfamiliar_extensions_in_private_directories_fail(self):
        for path in ("features/example.unknown", "nested/consent/example.json"):
            self.assertEqual(artifact_reason(path, b"synthetic"), "private/artifact directory")

    def test_case_and_known_extensions_fail(self):
        for path in ("sample.WAV", "example.npz", "sample.p8", ".env.dev"):
            self.assertIsNotNone(artifact_reason(path, b"synthetic"))

    def test_binary_and_size_fail(self):
        self.assertEqual(artifact_reason("example.txt", b"x\0y"), "binary file")
        self.assertEqual(
            artifact_reason("example.txt", b"x" * (MAX_FILE_BYTES + 1)), "oversized file"
        )

    def test_public_examples_stay_allowed(self):
        for path in ("examples/synthetic.json", "src/rightyo/model.py", ".env.example"):
            self.assertIsNone(artifact_reason(path, b"synthetic"))

    def test_notebook_outputs_fail_but_clean_notebook_passes(self):
        self.assertEqual(
            artifact_reason("example.ipynb", json.dumps({"cells": [{"outputs": [{}]}]}).encode()),
            "saved notebook output",
        )
        self.assertIsNone(artifact_reason("example.ipynb", b'{"cells": [{"outputs": []}]}'))

    def test_failures_do_not_print_paths_or_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "synthetic.wav").write_bytes(b"example-private-content")
            errors = artifact_errors(root, [Path("synthetic.wav")])
            self.assertEqual(errors, ["tracked entry 1: private/artifact extension"])

    def test_ignore_rules_and_negative_cases(self):
        self.assertEqual(ignore_errors(ROOT), [])

    def test_missing_ignore_rule_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", root], check=True)
            (root / ".gitignore").write_text(".env\n")
            self.assertTrue(ignore_errors(root))


class MarkdownTests(unittest.TestCase):
    def check_text(self, text, *, target=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text(text)
            if target:
                (root / "target.md").write_text("# Target\n")
            return markdown_errors(root, Path("README.md"))

    def test_links_and_formatting(self):
        self.assertEqual(self.check_text("# Example\n\n[Target](target.md)\n", target=True), [])
        self.assertTrue(self.check_text("# Example \n\n[Missing](missing.md)\n"))

    def test_external_links_and_code_examples_are_not_local_paths(self):
        self.assertEqual(
            self.check_text("[Web](https://example.com)\n\n```text\n[code](missing.md)\n```\n"),
            [],
        )

    def test_link_cannot_escape_checkout(self):
        self.assertTrue(self.check_text("[Escape](../README.md)\n"))

    def test_unclosed_fence_fails(self):
        self.assertTrue(self.check_text("```text\nexample\n"))


class TraceabilityTests(unittest.TestCase):
    def refs(self, body):
        return issue_references({"pull_request": {"body": body}}, "mickdarling/rightyo")

    def test_refs_and_own_urls(self):
        self.assertEqual(self.refs("Refs #15\nCloses #14"), [14, 15])
        self.assertEqual(self.refs("Refs https://github.com/mickdarling/rightyo/issues/15"), [15])
        self.assertEqual(self.refs("Refs other/repo#15, #14"), [14])

    def test_template_foreign_reference_and_examples_do_not_pass(self):
        for body in (
            "Refs #",
            "Refs #0",
            "Refs https://github.com/mickdarling/rightyo/issues/0",
            "Refs https://github.com/mickdarling/rightyo/issues/01",
            "Refs https://github.com/other/repo/issues/15",
            "Refs other/repo#15",
            "Refs other-repo#15",
            "<!-- Refs #15 -->",
            "```text\nRefs #15\n```",
            "This example says Refs #15",
        ):
            self.assertEqual(self.refs(body), [])

    def test_untrusted_text_is_only_data(self):
        self.assertEqual(self.refs("Refs #15 `$(touch impossible)`"), [15])


class WorkflowPolicyTests(unittest.TestCase):
    def document(self):
        return {
            "on": {"pull_request": {}},
            "permissions": {"contents": "read"},
            "jobs": {
                "checks": {
                    "runs-on": "ubuntu-24.04",
                    "timeout-minutes": 10,
                    "steps": [{"uses": "actions/checkout@" + "a" * 40}],
                }
            },
        }

    def test_secretless_pinned_workflow_passes(self):
        self.assertEqual(workflow_errors("ci.yml", self.document()), [])

    def test_privileged_event_fails(self):
        for event in ("pull_request_target", "workflow_run"):
            with self.subTest(event=event):
                doc = self.document()
                doc["on"] = {event: {}}
                self.assertTrue(workflow_errors("ci.yml", doc))

    def test_secret_references_fail_outside_steps(self):
        cases = (
            ("root-env", {"env": {"TOKEN": "${{ secrets.EXAMPLE }}"}}),
            ("job-env", {"env": {"TOKEN": "${{ secrets.EXAMPLE }}"}}),
            ("container", {"container": {"credentials": {"password": "${{ secrets.EXAMPLE }}"}}}),
            ("service", {"services": {"db": {"env": {"TOKEN": "${{ secrets.EXAMPLE }}"}}}}),
        )
        for location, fragment in cases:
            with self.subTest(location=location):
                doc = self.document()
                target = doc if location == "root-env" else doc["jobs"]["checks"]
                target.update(fragment)
                self.assertEqual(
                    workflow_errors("ci.yml", doc),
                    ["ci.yml: baseline workflow must not reference secrets"],
                )

    def test_secret_context_forms_fail_at_every_scope(self):
        expressions = (
            "${{ secrets['INVENTED_ONLY'] }}",
            '${{ secrets["INVENTED_ONLY"] }}',
            "${{ secrets . INVENTED_ONLY }}",
            "${{ SECRETS.INVENTED_ONLY }}",
            "${{ toJSON(secrets) }}",
            "${{\nsecrets\n[ 'INVENTED_ONLY' ] }}",
        )
        for expression in expressions:
            for location in ("root", "job", "step", "container", "service"):
                with self.subTest(expression=expression, location=location):
                    doc = self.document()
                    job = doc["jobs"]["checks"]
                    if location == "root":
                        doc["env"] = {"EXAMPLE": expression}
                    elif location == "job":
                        job["env"] = {"EXAMPLE": expression}
                    elif location == "step":
                        job["steps"][0]["env"] = {"EXAMPLE": expression}
                    elif location == "container":
                        job["container"] = {"credentials": {"password": expression}}
                    else:
                        job["services"] = {"db": {"env": {"EXAMPLE": expression}}}
                    self.assertEqual(
                        workflow_errors("ci.yml", doc),
                        ["ci.yml: baseline workflow must not reference secrets"],
                    )

    def test_mutable_action_permissions_timeout_and_interpolation_fail(self):
        doc = self.document()
        job = doc["jobs"]["checks"]
        job["permissions"] = {"contents": "write"}
        job.pop("timeout-minutes")
        job["steps"] = [
            {"uses": "actions/checkout@v4"},
            {"run": "echo '${{ github.event.pull_request.title }}'"},
            {"run": "echo example", "env": {"TOKEN": "${{ secrets.EXAMPLE }}"}},
        ]
        self.assertGreaterEqual(len(workflow_errors("ci.yml", doc)), 5)


class AIReviewPolicyTests(unittest.TestCase):
    def document(self):
        document = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
        document["on"] = document.pop(True)
        return document

    def check(self, document):
        return workflow_errors(AI_REVIEW_WORKFLOW, document)

    def test_trusted_lane_passes_without_relaxing_other_workflows(self):
        doc = self.document()
        self.assertEqual(self.check(doc), [])
        for path in ("ai-review.yml", ".github/workflows/another-review.yml"):
            self.assertTrue(workflow_errors(path, doc))

    def test_pr_checkout_or_persistent_credentials_are_rejected(self):
        for key, value in (
            ("ref", "${{ github.event.pull_request.head.sha }}"),
            ("persist-credentials", True),
            ("repository", "${{ github.event.pull_request.head.repo.full_name }}"),
        ):
            doc = self.document()
            doc["jobs"]["prepare"]["steps"][0]["with"][key] = value
            self.assertTrue(self.check(doc))

    def test_extra_events_filters_or_unbounded_dispatch_are_rejected(self):
        for change in ("push", "workflow_run", "paths", "dispatch"):
            doc = self.document()
            if change in ("push", "workflow_run"):
                doc["on"][change] = {}
            elif change == "paths":
                doc["on"]["pull_request_target"]["paths"] = ["src/**"]
            else:
                doc["on"]["workflow_dispatch"]["inputs"]["shell"] = {"type": "string"}
            self.assertTrue(self.check(doc))

    def test_workflow_callback_scope_and_bot_routing_cannot_be_broadened(self):
        for change in ("workflow-name", "workflow-type", "prepare-route", "publish-route"):
            with self.subTest(change=change):
                doc = self.document()
                if change == "workflow-name":
                    doc["on"]["workflow_run"]["workflows"] = ["*"]
                elif change == "workflow-type":
                    doc["on"]["workflow_run"]["types"] = ["requested", "completed"]
                elif change == "prepare-route":
                    doc["jobs"]["prepare"].pop("if")
                else:
                    doc["jobs"]["publish"]["if"] = "${{ always() }}"
                self.assertTrue(self.check(doc))

    def test_ignored_callbacks_and_fork_events_cannot_cancel_authorized_runs(self):
        for change in ("no-partition", "cancel-everything", "missing-concurrency"):
            with self.subTest(change=change):
                doc = self.document()
                if change == "no-partition":
                    doc["concurrency"]["group"] = (
                        "rightyo-ai-${{ github.event.pull_request.number }}"
                    )
                elif change == "cancel-everything":
                    doc["concurrency"]["cancel-in-progress"] = True
                else:
                    doc.pop("concurrency")
                self.assertTrue(self.check(doc))

    def test_missing_review_job_skip_or_publisher_dependency_fails(self):
        for change in ("missing", "skip", "needs", "publisher-if", "authorization-output"):
            doc = self.document()
            if change == "missing":
                del doc["jobs"]["claude"]
            elif change == "skip":
                doc["jobs"]["codex"]["if"] = "false"
            elif change == "needs":
                doc["jobs"]["publish"]["needs"] = ["prepare"]
            elif change == "publisher-if":
                doc["jobs"]["publish"].pop("if")
            else:
                doc["jobs"]["prepare"]["outputs"]["allowed"] = "true"
            self.assertTrue(self.check(doc))

    def test_runner_or_permission_escalation_fails(self):
        for field, value in (
            ("runs-on", "self-hosted"),
            ("permissions", {"contents": "write"}),
            ("container", {"image": "example"}),
            ("services", {"example": {}}),
            ("timeout-minutes", 60),
            ("environment", "production"),
        ):
            doc = self.document()
            doc["jobs"]["codex"][field] = value
            self.assertTrue(self.check(doc))

    def test_provider_credentials_cannot_move_scopes_or_commands(self):
        for location in ("root", "job", "other-step", "wrong-provider", "command"):
            doc = self.document()
            expression = "${{ secrets.OPENAI_API_KEY }}"
            if location == "root":
                doc["env"] = {"KEY": expression}
            elif location == "job":
                doc["jobs"]["codex"]["env"] = {"KEY": expression}
            elif location == "other-step":
                doc["jobs"]["prepare"]["steps"][0]["env"] = {"KEY": expression}
            elif location == "wrong-provider":
                doc["jobs"]["claude"]["steps"][3]["env"]["KEY"] = expression
            else:
                doc["jobs"]["claude"]["steps"][3]["run"] += "; echo unsafe"
            self.assertTrue(self.check(doc))

    def test_secret_fallback_bracket_access_or_extra_secret_fails(self):
        for expression in (
            "${{ secrets['OPENAI_API_KEY'] }}",
            "${{ secrets.OPENAI_API_KEY || secrets.OTHER }}",
            "${{ toJSON(secrets) }}",
        ):
            doc = self.document()
            doc["jobs"]["codex"]["steps"][4]["with"]["openai-api-key"] = expression
            self.assertTrue(self.check(doc))

    def test_privileged_steps_and_codex_restrictions_cannot_be_extended(self):
        for change in (
            "extra-command",
            "token-env",
            "skip-validator",
            "profile",
            "unsafe-args",
            "artifact-path",
            "step-condition",
            "install-scripts",
        ):
            with self.subTest(change=change):
                doc = self.document()
                if change == "extra-command":
                    doc["jobs"]["publish"]["steps"].append({"run": "echo forge-checks"})
                elif change == "token-env":
                    doc["jobs"]["codex"]["steps"][2]["env"] = {
                        "GITHUB_TOKEN": "${{ github.token }}",
                    }
                elif change == "skip-validator":
                    doc["jobs"]["codex"]["steps"].pop(5)
                elif change == "profile":
                    doc["jobs"]["codex"]["steps"][4]["with"]["permission-profile"] = "danger"
                elif change == "unsafe-args":
                    doc["jobs"]["codex"]["steps"][4]["with"]["codex-args"] = '["--yolo"]'
                elif change == "artifact-path":
                    doc["jobs"]["claude"]["steps"][4]["with"]["path"] = "/tmp/**"
                elif change == "step-condition":
                    doc["jobs"]["publish"]["steps"][2]["if"] = "false"
                else:
                    doc["jobs"]["claude"]["steps"][2]["run"] = "npm ci --prefix .github/reviews"
                self.assertTrue(self.check(doc))

    def test_malformed_settings_fail_closed_instead_of_crashing(self):
        for change in ("types", "inputs", "needs", "steps", "with", "env", "run"):
            with self.subTest(change=change):
                doc = self.document()
                if change == "types":
                    doc["on"]["pull_request_target"]["types"] = [{}]
                elif change == "inputs":
                    doc["on"]["workflow_dispatch"]["inputs"] = None
                elif change == "needs":
                    doc["jobs"]["codex"]["needs"] = [{}]
                elif change == "steps":
                    doc["jobs"]["codex"]["steps"] = None
                else:
                    doc["jobs"]["codex"]["steps"][1][change] = None
                self.assertTrue(self.check(doc))

    def test_local_mutable_or_unapproved_actions_and_shell_interpolation_fail(self):
        for action in (
            "./.github/actions/review",
            "openai/codex-action@main",
            "attacker/action@" + "a" * 40,
        ):
            doc = self.document()
            doc["jobs"]["codex"]["steps"][4]["uses"] = action
            self.assertTrue(self.check(doc))
        doc = self.document()
        doc["jobs"]["prepare"]["steps"].append(
            {
                "run": "echo '${{ github.event.pull_request.title }}'",
            }
        )
        self.assertTrue(self.check(doc))


if __name__ == "__main__":
    unittest.main()
