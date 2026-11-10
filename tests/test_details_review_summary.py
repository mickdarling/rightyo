"""Public connector bodies from PR #270, with authored REST/GraphQL metadata.

The .txt fixtures are verbatim public comments supplied for #271. Variants below
are synthetic. All API access is mocked; no provider, speech or network is used.
"""

import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch

from test_native_review_envelopes import change_state
from test_subscription_review import BASE, completion, reaction

from scripts.subscription_review import (
    AUDITED_RECHECK_PAIRS,
    BOT_NODE_ID,
    DETAILS_SUMMARY,
    SUMMARY,
    GateError,
    clean_completion,
    comment_provenance,
    queued_metadata_recheck,
    summary_state,
)

FIXTURES = Path(__file__).parent / "fixtures" / "native-review"
HEAD = "f00a63802e10e825936b234838298514de45bbd7"
COMPLETED = "2026-11-10T19:37:15.391539Z"


def public_comment(kind="summary"):
    item = completion(HEAD)
    item.update(
        id=11 if kind == "summary" else 12,
        created_at="2026-11-10T19:37:13Z",
        updated_at="2026-11-10T19:37:16Z" if kind == "summary" else "2026-11-10T19:37:13Z",
        body=(FIXTURES / f"connector-{kind}-270.txt").read_text(),
    )
    return item


class DetailsReviewSummaryTests(unittest.TestCase):
    def setUp(self):
        # Any unexpected API request fails the test instead of using the network.
        self.api_patch = patch("scripts.subscription_review.api", side_effect=self.graphql)
        self.api = self.api_patch.start()
        self.addCleanup(self.api_patch.stop)
        self.comments = []
        self.human_edited = None

    def graphql(self, path, method, data):
        self.assertEqual((path, method), ("/graphql", "POST"))
        item = next(c for c in self.comments if c["node_id"] == data["variables"]["id"])
        node = {
            "id": item["node_id"],
            "databaseId": item["id"],
            "body": item["body"],
            "updatedAt": item["updated_at"],
            "lastEditedAt": None,
            "editor": None,
        }
        if item is self.human_edited:
            node.update(
                lastEditedAt=item["updated_at"],
                editor={"__typename": "User", "login": "maintainer", "id": BOT_NODE_ID},
            )
        return {"data": {"node": node}}

    def check(self, comments, *, reactions=(), resolve=None):
        self.comments = [dict(item, node_id=f"IC_fixture_{i}") for i, item in enumerate(comments)]
        if self.human_edited is not None:
            self.human_edited = self.comments[self.human_edited]
        return clean_completion(
            HEAD,
            self.comments,
            [],
            [],
            resolve or (lambda short: HEAD if HEAD.startswith(short) else BASE),
            comment_provenance,
            reactions,
            lambda item: True,
        )

    def assert_blocked(self, item, message=None):
        # A malformed current envelope must abort even with older usable evidence.
        with self.assertRaises(GateError) as raised:
            self.check(
                [public_comment(), public_comment("receipt"), item],
                reactions=[reaction("2026-11-10T19:37:15Z")],
            )
        if message is not None:
            self.assertEqual(str(raised.exception), message)

    def test_freshness_regex_extracts_the_details_grammar_completion_time(self):
        for stamp in [COMPLETED, "2026-11-10T19:37:15Z", "2026-11-10T19:37:15.1Z"]:
            with self.subTest(stamp=stamp):
                body = public_comment()["body"].replace(COMPLETED, stamp)
                rest = body.split("\n", 2)[2]
                details = DETAILS_SUMMARY.fullmatch(rest)
                self.assertIsNotNone(details)
                completed = SUMMARY.findall(body)
                self.assertEqual(len(completed), 1)
                self.assertEqual(completed[0][0], details["time"])
                self.assertEqual(details["time"], stamp)

    def test_verbatim_public_fixtures(self):
        expected = {
            "summary": "7c9066d77838a54543c6467776847dcaed9b75b40608724137d0d4ce2955561f",
            "receipt": "62f9a490c8db6715315cfbaa5ae559045e31f83af9eea77456da7bbc436293f0",
        }
        for kind, digest in expected.items():
            raw = (FIXTURES / f"connector-{kind}-270.txt").read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            self.assertEqual(public_comment(kind)["body"].encode(), raw)

    def test_real_completed_summary_and_receipt_use_both_positive_paths(self):
        anchor, receipt = public_comment(), public_comment("receipt")
        result = self.check([anchor], reactions=[reaction("2026-11-10T19:37:15Z")])
        self.assertEqual(result["kind"], "summary")
        self.assertEqual(self.check([anchor, receipt])["kind"], "explicit")
        self.assertIsNone(self.check([anchor]))
        self.assertIsNone(self.check([], reactions=[reaction("2026-11-10T19:37:15Z")]))
        self.assertIsNone(self.check([receipt], reactions=[reaction("2026-11-10T19:37:15Z")]))
        self.assertIsNone(self.check([anchor], reactions=[reaction("2026-11-10T19:37:14Z")]))
        self.assertGreater(self.api.call_count, 0)

    def test_human_edited_body_fails_even_with_clean_receipt(self):
        self.human_edited = 2
        item = public_comment()
        item["body"] = item["body"].replace("Manual request", "PR opened")
        self.assertIsNotNone(summary_state(item["body"]))
        self.assert_blocked(item, "Native comment edit provenance is invalid")

    def test_source_pair_change_does_not_register_original_main_or_new_helper(self):
        # Immutable source digests read locally from 1c72ac8, the repair's base.
        workflow = "e1542e4e6abea1df91bc8c022b31ee171afb248580a5c758a7a9dfb0b1ecbb78"
        old_helper = "c7adffe3d4ceff9aaa7c72ddd264aca153a7e5c5945f21bb5e9e1a533dc6fcfe"
        root = Path(__file__).resolve().parents[1]
        new_helper = hashlib.sha256(
            (root / "scripts/subscription_review.py").read_bytes()
        ).hexdigest()
        old_pair, new_pair = (workflow, old_helper), (workflow, new_helper)
        self.assertNotEqual(old_pair, new_pair)
        self.assertNotIn(old_pair, AUDITED_RECHECK_PAIRS)
        self.assertNotIn(new_pair, AUDITED_RECHECK_PAIRS)
        run = {
            "name": "Subscription Codex review gate",
            "status": "queued",
            "event": "workflow_dispatch",
            "head_sha": HEAD,
            "head_repository": {"full_name": "mickdarling/rightyo"},
        }
        for producer, trusted, expected in [
            (new_pair, new_pair, True),
            (new_pair, old_pair, False),
            (old_pair, new_pair, False),
        ]:
            with patch("scripts.subscription_review.recheck_source_digest", side_effect=producer):
                self.assertEqual(queued_metadata_recheck(run, trusted), expected)

    def test_findings_running_queued_unknown_status_and_header_mismatch_fail(self):
        body = public_comment()["body"]
        cases = [
            body.replace("✅ **Completed**", "⚠️ **Findings**"),
            body.replace("✅ Completed", "⏳ Running").replace(
                "✅ **Completed**", "⏳ **Running**"
            ),
            body.replace("✅ Completed", "⏳ Queued").replace("✅ **Completed**", "⏳ **Queued**"),
            body.replace("✅ **Completed**", "✅ **Unknown**"),
            body.replace("✅ Completed</summary>", "✅ Running</summary>"),
            body.replace("✅ **Completed**", "✅ **Running**"),
        ]
        for changed in cases:
            with self.subTest(body=changed):
                self.assert_blocked({**public_comment(), "body": changed})
        for status in ["running", "queued"]:
            self.assert_blocked(
                change_state(public_comment(), lambda s: s["code_review"].update(status=status))
            )

    def test_markers_schema_versions_and_clean_flags_remain_strict(self):
        body = public_comment()["body"]
        lines = body.splitlines(keepends=True)
        cases = [
            lines[0] + body,
            lines[0] + lines[1] + body[len(lines[0]) :],
            body + lines[0],
            body + lines[1],
            body + "<!-- codex-review-receipt:v1 -->\n",
            body.replace("codex-review-state:v1", "codex-review-state:v2"),
            body.replace(
                "codex-pull-request-review-summary -->", "codex-pull-request-review-summary:v2 -->"
            ),
            body.replace('"code_review":', '"code_review":{},"code_review":'),
            body.replace('{"code_review":', '{"extra":null,"code_review":'),
            " " + body,
            "".join(lines[1:]),
            lines[0] + "\n" + "".join(lines[1:]),
        ]
        for changed in cases:
            with self.subTest(body=changed[:100]):
                self.assert_blocked({**public_comment(), "body": changed})
        for flag in ["has_any_findings", "has_unresolved_findings"]:
            for value in [True, None, "false", 0]:
                self.assert_blocked(
                    change_state(public_comment(), lambda s: s["code_review"].update({flag: value}))
                )

    def test_commit_mismatch_old_head_and_nonunique_resolution_do_not_approve(self):
        item = public_comment()
        self.assert_blocked({**item, "body": item["body"].replace("`f00a638`", "`bbbbbbb`")})
        self.assert_blocked(change_state(item, lambda s: s["code_review"].update(commit_sha=BASE)))
        with self.assertRaises(GateError):
            self.check([item], resolve=lambda short: BASE)
        self.assertIsNone(
            self.check(
                [
                    {
                        **item,
                        "body": item["body"].replace(HEAD, BASE).replace("`f00a638`", "`bbbbbbb`"),
                    }
                ]
            )
        )

    def test_table_rows_trigger_and_trailing_blocks_are_fully_consumed(self):
        item = public_comment()
        body = item["body"]
        row = next(line for line in body.splitlines(keepends=True) if line.startswith("| 📝"))
        cases = [
            body.replace(row, row + row),
            body.replace(row, row + row.replace("📝 **Code Review**", "🔧 **Other Review**")),
            body.replace(
                "| Review | Status | Commit | Review trigger |",
                "| Review | State | Commit | Review trigger |",
            ),
            body.replace("| --- | --- | --- | --- |", "| --- | --- | --- |"),
            body.replace("Manual request", "Manual request | Extra"),
            body.replace("Manual request", "<b>Manual request</b>"),
            body + "\n<details><summary>Extra</summary>clean</details>\n",
            body.replace("\n</details>\n\n</details>", "\nextra\n</details>\n\n</details>"),
            body.replace("once all reviews finish with no findings", "once reviews finish"),
            body.replace("\n</details>\n\n</details>", "\n</details>"),
        ]
        for changed in cases:
            with self.subTest(body=changed[-200:]):
                self.assert_blocked({**item, "body": changed})

    def test_optional_time_cannot_supply_a_completed_cycle_and_bad_times_fail(self):
        body = public_comment()["body"]
        element = f' <relative-time datetime="{COMPLETED}">{COMPLETED}</relative-time>'
        cases = [
            body.replace(element, ""),
            body.replace(
                f'">{COMPLETED}</relative-time>', '">2026-11-10T19:37:14Z</relative-time>'
            ),
            body.replace(COMPLETED, "2026-99-10T19:37:15Z"),
            body.replace(COMPLETED, "2026-11-10T19:38:15Z"),
        ]
        for changed in cases:
            with self.subTest(body=changed):
                self.assert_blocked({**public_comment(), "body": changed})

    def test_security_state_without_a_details_row_blocks_even_with_old_code_row(self):
        for code_revision in [HEAD, BASE]:
            for revision, status, findings in [
                (BASE, "completed", True),
                (HEAD, "completed", False),
                (HEAD, "running", None),
                (BASE, "running", None),
                (HEAD, "queued", None),
                (HEAD, "completed", True),
                (HEAD, "completed", None),
            ]:
                with self.subTest(
                    code=code_revision, security=revision, status=status, flags=findings
                ):
                    item = public_comment()
                    if code_revision == BASE:
                        item["body"] = (
                            item["body"].replace(HEAD, BASE).replace("`f00a638`", "`bbbbbbb`")
                        )
                    item = change_state(
                        item,
                        lambda s: s["security_review"].update(
                            commit_sha=revision,
                            status=status,
                            has_any_findings=findings,
                            has_unresolved_findings=findings,
                        ),
                    )
                    self.assert_blocked(
                        item, "Native details summary carries unsupported security state"
                    )

    def test_connector_edit_provenance_preserves_exact_fixture_body(self):
        item = public_comment()
        node = {
            "id": item["node_id"],
            "databaseId": item["id"],
            "body": item["body"],
            "updatedAt": item["updated_at"],
            "lastEditedAt": item["updated_at"],
            "editor": {"__typename": "Bot", "id": BOT_NODE_ID, "login": "chatgpt-codex-connector"},
        }
        with patch("scripts.subscription_review.api", return_value={"data": {"node": node}}):
            self.assertTrue(comment_provenance(item, True))
        node["body"] = node["body"].strip()
        with patch("scripts.subscription_review.api", return_value={"data": {"node": node}}):
            with self.assertRaises(GateError):
                comment_provenance(item, True)


if __name__ == "__main__":
    unittest.main()
