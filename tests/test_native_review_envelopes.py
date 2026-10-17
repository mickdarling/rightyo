"""Authored native-review metadata only; no GitHub calls, credentials or inference."""

import copy
import json
import unittest
from unittest.mock import patch

from test_subscription_review import BASE, HEAD, completion, reaction, summary

from scripts.subscription_review import (
    BOT_NODE_ID,
    GateError,
    clean_completion,
    comment_provenance,
    review_candidates,
    timestamp,
)


def wrapped_summary(status="completed"):
    item = summary()
    code = {
        "commit_sha": HEAD,
        "has_any_findings": False if status == "completed" else None,
        "has_unresolved_findings": False if status == "completed" else None,
        "status": status,
    }
    state = {"code_review": code, "security_review": dict.fromkeys(code)}
    item["body"] = item["body"].replace(
        "<!-- codex-pull-request-review-summary -->\n",
        "<!-- codex-pull-request-review-summary -->\n"
        f"<!-- codex-review-state:v1 {json.dumps(state, separators=(',', ':'))} -->\n",
    )
    if status != "completed":
        item["body"] = item["body"].replace("**Completed**", f"**{status.title()}**")
    return item


def wrapped_receipt():
    item = completion()
    item["body"] = "<!-- codex-review-receipt:v1 -->\n\n" + item["body"]
    return item


def change_state(item, edit):
    item = copy.deepcopy(item)
    lines = item["body"].splitlines()
    prefix, suffix = "<!-- codex-review-state:v1 ", " -->"
    state = json.loads(lines[1][len(prefix) : -len(suffix)])
    edit(state)
    lines[1] = prefix + json.dumps(state, separators=(",", ":")) + suffix
    item["body"] = "\n".join(lines) + "\n"
    return item


class NativeEnvelopeTests(unittest.TestCase):
    def check(self, comments, *, reactions=(), attest=None, reviews=(), inline=()):
        return clean_completion(
            HEAD,
            comments,
            reviews,
            inline,
            lambda short: HEAD if HEAD.startswith(short) else BASE,
            attest or (lambda item, is_summary: True),
            reactions,
            lambda item: True,
        )

    def test_current_summary_requires_native_reaction_and_supports_receipt(self):
        anchor, receipt = wrapped_summary(), wrapped_receipt()
        self.assertEqual(self.check([anchor], reactions=[reaction()])["kind"], "summary")
        self.assertIsNone(self.check([anchor]))
        self.assertIsNone(self.check([receipt], reactions=[reaction()]))
        self.assertEqual(self.check([anchor, receipt])["kind"], "explicit")
        self.assertIsNotNone(self.check([summary(), receipt]))
        self.assertIsNotNone(self.check([anchor, completion()]))

    def test_wrapper_never_substitutes_connector_identity_or_app(self):
        for field, value in [("user", {}), ("performed_via_github_app", {"id": 1})]:
            anchor, receipt = wrapped_summary(), wrapped_receipt()
            anchor[field] = receipt[field] = value
            self.assertIsNone(self.check([anchor, receipt], reactions=[reaction()]))

    def test_unsupported_receipts_cannot_fall_back_to_older_clean_evidence(self):
        for prefix in [
            "<!-- codex-review-receipt:v2 -->\n\n",
            "<!-- codex-review-receipt:v1 -->\n",
            "<!-- codex-review-receipt:v1 -->\n\n" * 2,
            " <!-- codex-review-receipt:v1 -->\n\n",
            "<!-- codex-review-receipt:v1\n\n",
        ]:
            receipt = completion()
            receipt["body"] = prefix + receipt["body"]
            for older in [[], [completion()]]:
                with self.subTest(prefix=prefix, older=bool(older)), self.assertRaises(GateError):
                    self.check([wrapped_summary(), *older, receipt], reactions=[reaction()])

    def test_unknown_malformed_duplicate_and_oversized_state_cannot_resurrect_clean(self):
        anchor = wrapped_summary()
        cases = [
            anchor["body"].replace("codex-review-state:v1", "codex-review-state:v2"),
            anchor["body"].replace('"code_review":', '"code_review":{},"code_review":'),
            anchor["body"].replace('{"code_review":', '{"extra":null,"code_review":'),
            anchor["body"].replace(
                '{"code_review":', '{"padding":"' + "x" * 4097 + '","code_review":'
            ),
            anchor["body"].replace('{"code_review":', '{"code_review":['),
        ]
        for body in cases:
            with self.subTest(body=body[:90]), self.assertRaises(GateError):
                item = {**anchor, "body": body}
                self.check([summary(), completion(), item], reactions=[reaction()])

    def test_state_commit_status_and_findings_must_agree_with_completion(self):
        edits = [
            lambda s: s["code_review"].update(commit_sha=BASE),
            lambda s: s["code_review"].update(commit_sha=HEAD[:7]),
            lambda s: s["code_review"].update(status="running"),
            lambda s: s["code_review"].update(status="unknown"),
            lambda s: s["code_review"].update(status=[]),
            lambda s: s["code_review"].update(has_any_findings=True),
            lambda s: s["code_review"].update(has_unresolved_findings=True),
            lambda s: s["code_review"].update(has_any_findings=None),
            lambda s: s["code_review"].update(has_unresolved_findings="false"),
            lambda s: s["security_review"].update(
                commit_sha=HEAD,
                status="completed",
                has_any_findings=True,
                has_unresolved_findings=False,
            ),
            lambda s: s["security_review"].update(commit_sha=HEAD, status="running"),
            lambda s: s["security_review"].update(commit_sha=HEAD, status="queued"),
            lambda s: s["security_review"].update(commit_sha=HEAD, status="completed"),
        ]
        for edit in edits:
            with self.subTest(edit=edit), self.assertRaises(GateError):
                self.check([completion(), change_state(wrapped_summary(), edit)])
        item = wrapped_summary()
        item["body"] = item["body"].replace("**Completed**", "**Running**")
        with self.assertRaises(GateError):
            self.check([completion(), item])

    def test_running_and_queued_cycles_and_newer_requests_still_withhold(self):
        for status in ["running", "queued"]:
            self.assertIsNone(self.check([summary(), completion(), wrapped_summary(status)]))
        request = {
            "body": "@codex review",
            "updated_at": "2026-09-30T07:09:50Z",
            "user": {"type": "User", "login": "maintainer"},
        }
        self.assertIsNone(self.check([wrapped_summary(), wrapped_receipt(), request]))
        candidates = review_candidates(
            HEAD,
            [wrapped_summary(), wrapped_receipt()],
            [],
            [],
            lambda short: HEAD,
            lambda item, is_summary: True,
            lambda item: True,
            persisted={1: timestamp(request["updated_at"])},
        )
        self.assertEqual(candidates, [])

    def test_current_security_state_blocks_even_when_code_row_is_for_another_commit(self):
        def mixed(status, findings, revision=HEAD):
            def edit(state):
                state["code_review"]["commit_sha"] = BASE
                state["security_review"].update(
                    commit_sha=revision,
                    status=status,
                    has_any_findings=findings,
                    has_unresolved_findings=findings,
                )

            item = change_state(wrapped_summary(), edit)
            item["body"] = item["body"].replace(f"`{HEAD[:7]}`", f"`{BASE[:7]}`")
            return item

        clean = [wrapped_summary(), completion()]
        for status, findings in [
            ("queued", None),
            ("running", None),
            ("completed", None),
            ("completed", True),
        ]:
            with self.subTest(status=status, findings=findings), self.assertRaises(GateError):
                self.check([*clean, mixed(status, findings)], reactions=[reaction()])
        self.assertIsNotNone(self.check([*clean, mixed("completed", False)]))
        self.assertIsNotNone(self.check([*clean, mixed("running", None, revision=BASE)]))

    def test_findings_stale_resolution_and_graphql_provenance_still_block(self):
        anchor, receipt = wrapped_summary(), wrapped_receipt()
        with self.assertRaises(GateError):
            clean_completion(
                HEAD,
                [anchor, receipt],
                [],
                [],
                lambda short: BASE,
                lambda item, is_summary: True,
                [reaction()],
            )
        finding = {"user": receipt["user"], "commit_id": HEAD, "state": "DISMISSED"}
        self.assertIsNone(self.check([anchor, receipt], reviews=[finding]))
        parent = {**finding, "id": 123}
        inline = {**finding, "original_commit_id": HEAD, "pull_request_review_id": 123}
        self.assertIsNone(self.check([anchor, receipt], reviews=[parent], inline=[inline]))
        with self.assertRaises(GateError):
            self.check([anchor, receipt], attest=lambda item, is_summary: False)
        for is_summary, item in [(True, anchor), (False, receipt)]:
            node = {
                "id": item["node_id"],
                "databaseId": item["id"],
                "body": item["body"],
                "updatedAt": item["updated_at"],
                "lastEditedAt": None,
                "editor": None,
            }
            with patch("scripts.subscription_review.api", return_value={"data": {"node": node}}):
                self.assertTrue(comment_provenance(item, is_summary))
            node.update(
                lastEditedAt=item["updated_at"],
                editor={
                    "__typename": "User",
                    "login": "maintainer",
                    "id": BOT_NODE_ID,
                },
            )
            with patch("scripts.subscription_review.api", return_value={"data": {"node": node}}):
                self.assertFalse(comment_provenance(item, is_summary))

    def test_current_code_state_cannot_hide_behind_an_older_markdown_row(self):
        for status, findings in [
            ("queued", None),
            ("running", None),
            ("completed", None),
            ("completed", True),
            ("completed", False),
        ]:

            def edit(state):
                state["code_review"].update(
                    status=status, has_any_findings=findings, has_unresolved_findings=findings
                )

            item = change_state(wrapped_summary(), edit)
            item["body"] = item["body"].replace(f"`{HEAD[:7]}`", f"`{BASE[:7]}`")
            with self.subTest(status=status, findings=findings), self.assertRaises(GateError):
                self.check([wrapped_summary(), completion(), item], reactions=[reaction()])

    def test_summary_like_unsupported_prefixes_cannot_fall_back_to_clean_evidence(self):
        anchor = wrapped_summary()
        cases = [
            " " + anchor["body"],
            anchor["body"].replace(
                "codex-pull-request-review-summary", "codex-pull-request-review-summary:v2"
            ),
            anchor["body"].replace("<!-- codex-pull-request-review-summary -->\n", ""),
            anchor["body"].replace(
                "<!-- codex-pull-request-review-summary -->\n",
                "<!-- codex-pull-request-review-summary -->",
            ),
        ]
        for body in cases:
            with self.subTest(body=body[:90]), self.assertRaises(GateError):
                self.check([anchor, completion(), {**anchor, "body": body}], reactions=[reaction()])


if __name__ == "__main__":
    unittest.main()
