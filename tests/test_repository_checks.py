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
    SYNTHETIC_VOICE_FIXTURES,
    VOICE_DATA_REASON,
    VOICE_DATA_SUFFIXES,
    actionlint_source,
    artifact_errors,
    artifact_reason,
    git_files,
    ignore_errors,
    issue_references,
    markdown_errors,
    voice_data_reason,
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
            self.assertEqual(errors, [f"tracked entry 1: {VOICE_DATA_REASON}"])

    def test_ignore_rules_and_negative_cases(self):
        self.assertEqual(ignore_errors(ROOT), [])

    def test_missing_ignore_rule_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q", root], check=True)
            (root / ".gitignore").write_text(".env\n")
            self.assertTrue(ignore_errors(root))


class VoiceDataGuardTests(unittest.TestCase):
    """#109: real voice audio, embeddings and voiceprints never reach the repository."""

    FIXTURES = {"tests/fixtures/synthetic-voice": "authored test tones"}

    def test_required_extensions_are_covered(self):
        required = (
            ".wav .pcm .flac .mp3 .m4a .aac .ogg .opus .caf .aif .aiff .npy .npz .emb"
            " .pt .pth .onnx .mlmodel .mlpackage .safetensors .wave .oga .amr .3gp .wma .m4b .spx"
        ).split()
        self.assertTrue(set(required) <= VOICE_DATA_SUFFIXES)

    def test_voice_extensions_fail_with_privacy_reason(self):
        for path in ("owner.WAV", "nested/owner.emb", "a/b.npy", "speaker.onnx", "x.m4a"):
            self.assertEqual(artifact_reason(path, b"synthetic"), VOICE_DATA_REASON)
        self.assertIn("#109", VOICE_DATA_REASON)

    def test_model_bundle_directories_fail(self):
        for path in ("embedder.mlpackage/Manifest.json", "a/embedder.mlmodelc/coremldata.bin"):
            self.assertEqual(voice_data_reason(path), VOICE_DATA_REASON)

    def test_enrollment_and_voiceprint_segments_fail(self):
        for path in (
            "enrollment/owner.json",
            "tests/Enrollment/notes.txt",
            "docs/voiceprints/readme.md",
            "a/voiceprint/x.txt",
        ):
            self.assertEqual(artifact_reason(path, b"synthetic"), VOICE_DATA_REASON)

    def test_similar_names_are_not_segments(self):
        for path in (
            "examples/enrolled-override.jsonl",
            "docs/enrollment-plan.md",
            "src/rightyo/voiceprints.py",
            "tests/test_speaker_priority.py",
        ):
            self.assertIsNone(voice_data_reason(path))

    def test_allow_list_admits_only_paths_strictly_under_a_prefix(self):
        fixtures = self.FIXTURES
        allowed = "tests/fixtures/synthetic-voice/tone.emb"
        self.assertIsNone(voice_data_reason(allowed, fixtures))
        self.assertIsNone(artifact_reason(allowed, b"synthetic text", fixtures))
        self.assertIsNone(
            voice_data_reason("tests/fixtures/synthetic-voice/enrollment/a.json", fixtures)
        )
        for path in (
            "tests/fixtures/synthetic-voice.wav",
            "tests/fixtures/synthetic-voice-real/owner.wav",
            "tests/fixtures/owner.wav",
            "enrollment/tests/fixtures/synthetic-voice/x.wav",
        ):
            self.assertEqual(voice_data_reason(path, fixtures), VOICE_DATA_REASON)

    def test_allow_list_does_not_lift_binary_size_or_private_directory_rules(self):
        fixtures = self.FIXTURES
        path = "tests/fixtures/synthetic-voice/tone.wav"
        self.assertEqual(artifact_reason(path, b"RIFF\0", fixtures), "binary file")
        self.assertEqual(
            artifact_reason(path, b"x" * (MAX_FILE_BYTES + 1), fixtures), "oversized file"
        )
        self.assertEqual(
            artifact_reason("tests/fixtures/synthetic-voice/recordings/a.wav", b"x", fixtures),
            "private/artifact directory",
        )

    def test_every_allow_list_entry_is_a_strict_relative_prefix(self):
        # An empty, "." or escaping entry would match every path and switch the guard off.
        for prefix in SYNTHETIC_VOICE_FIXTURES:
            parts = prefix.split("/")
            self.assertTrue(prefix and not prefix.startswith("/"), prefix)
            self.assertNotIn("", parts, prefix)
            self.assertNotIn(".", parts, prefix)
            self.assertNotIn("..", parts, prefix)

    def test_allow_list_matches_current_inventory(self):
        # No tracked file is voice data today; any future entry must document why.
        self.assertEqual(SYNTHETIC_VOICE_FIXTURES, {})
        tracked = [path.as_posix() for path in git_files(ROOT)]
        self.assertEqual([path for path in tracked if voice_data_reason(path)], [])


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
                    doc["concurrency"]["cancel-in-progress"] = True
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
    def test_resolver_cannot_publish_and_workflow_mutex_requires_authoritative_route(self):
        for mutation in ("resolver-write", "global-mutex", "wrong-job-mutex", "unchecked-route"):
            doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
            if mutation == "resolver-write":
                doc["jobs"]["resolve"]["permissions"]["statuses"] = "write"
            elif mutation == "global-mutex":
                doc["concurrency"] = {"group": "all", "cancel-in-progress": False}
            elif mutation == "wrong-job-mutex":
                doc["jobs"]["gate"]["concurrency"] = {"group": "${{ github.run_id }}"}
            else:
                del doc["jobs"]["gate"]["steps"][1]["env"]["GATE_PR_NUMBER"]
            self.assertTrue(workflow_errors(AI_REVIEW_WORKFLOW, doc))


class RequestRecorderPolicyTests(unittest.TestCase):
    def test_marker_is_required_before_publisher_without_independent_job_mutex(self):
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
        for mutation in (
            "skip-denial",
            "checkout",
            "write-more",
            "no-head-read",
            "approve",
            "skip-dependency",
        ):
            doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
            job = doc["jobs"]["invalidate"]
            if mutation == "skip-denial":
                job["if"] = "false"
            elif mutation == "checkout":
                job["steps"].insert(0, {"uses": "actions/checkout@v4"})
            elif mutation == "write-more":
                job["permissions"]["contents"] = "write"
            elif mutation == "no-head-read":
                del job["permissions"]["pull-requests"]
            elif mutation == "approve":
                job["steps"][0]["run"] = job["steps"][0]["run"].replace('"pending"', '"success"')
            else:
                del doc["jobs"]["resolve"]["needs"]
            self.assertTrue(workflow_errors(AI_REVIEW_WORKFLOW, doc))


class WorkflowSerializationPolicyTests(unittest.TestCase):
    def test_all_status_writers_share_fixed_workflow_mutex_and_retain_pending_callbacks(self):
        doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
        self.assertEqual(
            doc["concurrency"],
            {"group": "rightyo-subscription-status", "queue": "max", "cancel-in-progress": False},
        )
        writers = {
            name
            for name, job in doc["jobs"].items()
            if job.get("permissions", {}).get("statuses") == "write"
        }
        self.assertEqual(writers, {"invalidate", "record", "gate"})
        self.assertTrue(all("concurrency" not in job for job in doc["jobs"].values()))
        for mutation in ("no-lock", "job-only", "no-queue", "queue-one", "dynamic", "cancel"):
            changed = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
            if mutation == "no-lock":
                del changed["concurrency"]
            elif mutation == "job-only":
                changed["jobs"]["gate"]["concurrency"] = changed.pop("concurrency")
            elif mutation == "no-queue":
                del changed["concurrency"]["queue"]
            elif mutation == "queue-one":
                changed["concurrency"]["queue"] = "single"
            elif mutation == "dynamic":
                changed["concurrency"]["group"] = "${{ github.run_id }}"
            else:
                changed["concurrency"]["cancel-in-progress"] = True
            self.assertTrue(workflow_errors(AI_REVIEW_WORKFLOW, changed))

    def test_new_denial_cannot_be_overwritten_between_old_read_and_success(self):
        # Enumerate schedules for an in-flight publisher and a later callback.
        # The former job-only mutex allowed old-read/new-denial/old-success.
        from itertools import permutations

        doc = yaml.safe_load((ROOT / AI_REVIEW_WORKFLOW).read_text())
        self.assertEqual(workflow_errors(AI_REVIEW_WORKFLOW, doc), [])
        schedules = [
            order
            for order in permutations(("old-read", "old-success", "new-denial"))
            if order.index("old-read") < order.index("old-success")
        ]
        unsafe = ("old-read", "new-denial", "old-success")
        self.assertIn(unsafe, schedules)
        # Workflow scope forbids another run's bootstrap inside a running pipeline.
        shared = doc["concurrency"]["group"] == "rightyo-subscription-status"
        serialized = [
            order
            for order in schedules
            if not shared
            or not order.index("old-read") < order.index("new-denial") < order.index("old-success")
        ]
        self.assertNotIn(unsafe, serialized)
        for order in serialized:
            if order[0] == "old-read":
                latest = None
                for event in order:
                    if event == "old-success":
                        latest = "success"
                    elif event == "new-denial":
                        latest = "pending"
                self.assertEqual(latest, "pending")


class ActionlintQueueCompatibilityTests(unittest.TestCase):
    def source(self):
        return (ROOT / AI_REVIEW_WORKFLOW).read_text()

    def test_only_literal_validated_queue_line_is_removed(self):
        source = self.source()
        result = actionlint_source(AI_REVIEW_WORKFLOW, source)
        self.assertEqual(result, source.replace("  queue: max\n", "", 1))
        self.assertIn("on:\n", result)
        self.assertIn("${{ github.workflow_sha }}", result)
        self.assertIn("python3 - <<'PY'", result)

    def test_invalid_or_mis_scoped_queue_cannot_be_normalized(self):
        for source in (
            self.source().replace("  queue: max", "  queue: single"),
            self.source().replace("  cancel-in-progress: false", "  cancel-in-progress: true"),
            self.source().replace("  group: rightyo-subscription-status", "  group: other"),
            self.source().replace("  queue: max\n", ""),
            self.source().replace("  queue: max\n", "  queue: max\n  queue: max\n"),
            self.source().replace("    needs: invalidate", "    needs: []"),
        ):
            with self.assertRaises(ValueError):
                actionlint_source(AI_REVIEW_WORKFLOW, source)

    def test_other_workflows_are_byte_unchanged_including_unknown_queue(self):
        source = "on: push\nconcurrency:\n  queue: invalid\n"
        self.assertEqual(actionlint_source(".github/workflows/other.yml", source), source)
        for name in (".github/workflows/ci.yml", REVIEW_RELAY_WORKFLOW):
            source = (ROOT / name).read_text()
            self.assertEqual(actionlint_source(name, source), source)
