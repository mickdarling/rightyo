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
    REVIEW_RELAY_WORKFLOW,
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

    def test_trusted_checkout_and_token_boundary_are_frozen(self):
        for mutation in ("checkout-head", "persistent", "extra-command", "model-key", "job-token"):
            with self.subTest(mutation=mutation):
                doc = self.document()
                job = doc["jobs"]["gate"]
                if mutation == "checkout-head":
                    job["steps"][0]["with"]["ref"] = "${{ github.event.pull_request.head.sha }}"
                elif mutation == "persistent":
                    job["steps"][0]["with"]["persist-credentials"] = True
                elif mutation == "extra-command":
                    job["steps"].append({"run": "echo forged-check"})
                elif mutation == "model-key":
                    job["steps"][1]["env"]["KEY"] = "${{ secrets.OPENAI_API_KEY }}"
                else:
                    job["env"] = {"GITHUB_TOKEN": "${{ github.token }}"}
                self.assertTrue(self.check(doc))

    def test_event_routes_revision_input_and_publisher_permissions_are_frozen(self):
        for mutation in (
            "event",
            "dispatch",
            "permission",
            "checks-permission",
            "runner",
            "concurrency",
            "condition",
        ):
            with self.subTest(mutation=mutation):
                doc = self.document()
                if mutation == "event":
                    doc["on"]["push"] = {}
                elif mutation == "dispatch":
                    del doc["on"]["workflow_dispatch"]["inputs"]["expected_head"]
                elif mutation == "concurrency":
                    doc["jobs"]["gate"]["concurrency"]["cancel-in-progress"] = True
                else:
                    field, value = {
                        "permission": ("permissions", {"contents": "write"}),
                        "checks-permission": (
                            "permissions",
                            {"contents": "read", "pull-requests": "read", "checks": "write"},
                        ),
                        "runner": ("runs-on", "self-hosted"),
                        "condition": ("if", "false"),
                    }[mutation]
                    doc["jobs"]["gate"][field] = value
                self.assertTrue(self.check(doc))


if __name__ == "__main__":
    unittest.main()


class ReviewRelayPolicyTests(unittest.TestCase):
    def test_only_fixed_unprivileged_notification_is_allowed(self):
        doc = yaml.safe_load((ROOT / REVIEW_RELAY_WORKFLOW).read_text())
        self.assertEqual(workflow_errors(REVIEW_RELAY_WORKFLOW, doc), [])
        for mutation in ("permissions", "checkout", "secret", "event-code", "wrong-trigger"):
            changed = yaml.safe_load((ROOT / REVIEW_RELAY_WORKFLOW).read_text())
            if mutation == "permissions":
                changed["jobs"]["notify"]["permissions"] = {"statuses": "write"}
            elif mutation == "checkout":
                changed["jobs"]["notify"]["steps"].append({"uses": "actions/checkout@" + "a" * 40})
            elif mutation == "secret":
                changed["env"] = {"KEY": "${{ secrets.OPENAI_API_KEY }}"}
            elif mutation == "event-code":
                changed["jobs"]["notify"]["steps"][0]["run"] = "${{ github.event.review.body }}"
            else:
                changed[True]["pull_request_target"] = {}
            self.assertTrue(workflow_errors(REVIEW_RELAY_WORKFLOW, changed))


class ResolverPolicyTests(unittest.TestCase):
    def test_resolver_cannot_publish_and_gate_mutex_requires_authoritative_route(self):
        for mutation in ("resolver-write", "global-mutex", "wrong-job-mutex", "unchecked-route"):
            doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
            if mutation == "resolver-write":
                doc["jobs"]["resolve"]["permissions"]["statuses"] = "write"
            elif mutation == "global-mutex":
                doc["concurrency"] = {"group": "all", "cancel-in-progress": False}
            elif mutation == "wrong-job-mutex":
                doc["jobs"]["gate"]["concurrency"]["group"] = "${{ github.run_id }}"
            else:
                del doc["jobs"]["gate"]["steps"][1]["env"]["GATE_PR_NUMBER"]
            self.assertTrue(workflow_errors(AI_REVIEW_WORKFLOW, doc))


class RequestRecorderPolicyTests(unittest.TestCase):
    def test_marker_is_required_before_publisher_and_cannot_be_dropped_by_gate_mutex(self):
        for mutation in ("skip-recorder", "recorder-mutex", "gate-without-recorder"):
            doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
            if mutation == "skip-recorder":
                doc["jobs"]["record"]["if"] = "false"
            elif mutation == "recorder-mutex":
                doc["jobs"]["record"]["concurrency"] = {"group": "all", "cancel-in-progress": True}
            else:
                doc["jobs"]["gate"]["needs"] = ["resolve"]
            self.assertTrue(workflow_errors(AI_REVIEW_WORKFLOW, doc))


class ReviewInvalidationPolicyTests(unittest.TestCase):
    def test_denial_precedes_routing_and_cannot_checkout_or_approve(self):
        for mutation in ("skip-denial", "checkout", "write-more", "approve", "skip-dependency"):
            doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
            job = doc["jobs"]["invalidate"]
            if mutation == "skip-denial":
                job["if"] = "false"
            elif mutation == "checkout":
                job["steps"].insert(0, {"uses": "actions/checkout@v4"})
            elif mutation == "write-more":
                job["permissions"]["contents"] = "write"
            elif mutation == "approve":
                job["steps"][0]["run"] = job["steps"][0]["run"].replace('"pending"', '"success"')
            else:
                del doc["jobs"]["resolve"]["needs"]
            self.assertTrue(workflow_errors(AI_REVIEW_WORKFLOW, doc))
