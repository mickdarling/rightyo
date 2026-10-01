from __future__ import annotations

import contextlib
import http.client
import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

from rightyo.cli import load_turns, main, save_turns
from rightyo.contracts import ContractError, Turn
from rightyo.pipeline import ReplayRunner
from rightyo.providers import (
    JEV_MODEL,
    JevProvider,
    MockProvider,
    ProviderError,
    bounded_request,
    build_request,
    parse_response,
)

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "synthetic-turns.json"


def turn(**kwargs) -> Turn:
    return replace(
        Turn(
            "session",
            "one",
            1,
            0,
            1000,
            "Rightyo, help me.",
            "Speaker A",
            True,
            False,
            "authored-fixture",
            "synthetic",
            "authored-fixture",
        ),
        **kwargs,
    )


def answer(choice: str, options: set[str], confidence: float = 1) -> dict:
    return {
        "type": "choice",
        "choice": choice,
        "confidence": confidence,
        "probabilities": {option: float(choice == option) for option in options},
    }


def response(request: dict, label="attend", recipient="system") -> dict:
    return {
        "model": JEV_MODEL,
        "answers": {
            "attention": answer(label, set(request["questions"]["attention"]["criteria"])),
            "recipient": answer(recipient, set(request["questions"]["recipient"]["criteria"])),
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


class RunnerTests(unittest.TestCase):
    def test_turn_rejects_surrogate_without_retaining_text_in_exception(self):
        with self.assertRaises(ContractError) as error:
            turn(text="private-invalid-\ud800")
        self.assertNotIn("private-invalid", str(error.exception))
        self.assertIsNone(error.exception.__context__)

    def test_partial_duplicate_and_stale_turns_never_issue_extra_decisions(self):
        provider = MagicMock(wraps=MockProvider())
        runner = ReplayRunner(provider)
        initial = turn(finalized=False)
        final = turn(revision=2)
        self.assertIsNone(runner.process(initial))
        self.assertIsNotNone(runner.process(final))
        self.assertIsNone(runner.process(initial))
        self.assertIsNone(runner.process(final))
        self.assertEqual(provider.decide.call_count, 1)
        with self.assertRaises(ContractError):
            runner.process(turn(revision=3))

    def test_restart_drops_in_flight_result(self):
        provider = MagicMock()
        runner = ReplayRunner(provider)

        def restart(state):
            runner.restart("next-session")
            return MockProvider().decide(state)

        provider.decide.side_effect = restart
        self.assertIsNone(runner.process(turn()))
        self.assertEqual(len(runner._history), 0)

    def test_superseded_revision_drops_old_response(self):
        provider = MagicMock()
        runner = ReplayRunner(provider)
        results = []

        def supersede(state):
            provider.decide.side_effect = MockProvider().decide
            results.append(runner.process(turn(revision=2, text="Could you help me?")))
            return MockProvider().decide(state)

        provider.decide.side_effect = supersede
        self.assertIsNone(runner.process(turn()))
        self.assertEqual(results[0].decision.label, "uncertain")
        self.assertEqual(len(runner._emitted), 1)

    def test_context_is_past_only_bounded_and_speaker_ablation_does_not_fabricate_labels(self):
        provider = MagicMock(wraps=MockProvider())
        runner = ReplayRunner(provider, max_context_chars=4000, no_speakers=True)
        runner.process(turn(text="a" * 3000))
        runner.process(turn(utterance_id="two", start_ms=1000, end_ms=2000, text="b" * 3000))
        state = provider.decide.call_args.args[0]
        self.assertEqual(state["past_turns"], [])
        self.assertEqual(state["known_participants"], [])
        self.assertIsNone(state["current_turn"]["speaker_id"])
        self.assertNotIn("session_id", state["current_turn"])

    def test_context_contains_prior_committed_turn_once(self):
        provider = MagicMock(wraps=MockProvider())
        runner = ReplayRunner(provider)
        runner.process(turn())
        runner.process(turn())
        runner.process(turn(utterance_id="two", start_ms=1000, end_ms=2000))
        self.assertEqual(len(provider.decide.call_args.args[0]["past_turns"]), 1)

    def test_session_change_timestamp_regression_and_budget_rejected(self):
        runner = ReplayRunner(MockProvider(), max_utterances=1)
        runner.process(turn())
        for changed in (turn(session_id="other"), turn(utterance_id="two")):
            with self.assertRaises(ContractError):
                runner.process(changed)
        ordered = ReplayRunner(MockProvider())
        ordered.process(turn())
        with self.assertRaises(ContractError):
            ordered.process(turn(utterance_id="two", end_ms=999))

    def test_boundary_rejects_nan_boolean_times_and_unknown_fields(self):
        for changes in (
            {"start_ms": True},
            {"end_ms": float("nan")},
            {"revision": 0},
            {"text": "a" * 4001},
        ):
            with self.assertRaises(ContractError):
                turn(**changes)
        with self.assertRaises(ContractError):
            Turn.from_dict({**turn().to_dict(), "execute": "anything"})


class JevTests(unittest.TestCase):
    def test_cancellation_before_credentials_never_sends_text(self):
        provider = JevProvider(allow_hosted=True, cancelled=lambda: True)
        provider._opener = MagicMock()
        with patch("rightyo.credentials.load_jev_api_key") as key:
            with self.assertRaisesRegex(ProviderError, "cancelled"):
                provider.decide({})
        key.assert_not_called()
        provider._opener.open.assert_not_called()

    def test_cancellation_after_keychain_wait_never_sends_text(self):
        stopped = False
        provider = JevProvider(allow_hosted=True, cancelled=lambda: stopped)
        provider._opener = MagicMock()

        def credential():
            nonlocal stopped
            stopped = True
            return "fictitious-test-key"

        state = ReplayRunner(MockProvider())._state(turn())
        with patch("rightyo.credentials.load_jev_api_key", side_effect=credential):
            with self.assertRaisesRegex(ProviderError, "cancelled"):
                provider.decide(state)
        provider._opener.open.assert_not_called()
        self.assertEqual(provider.requests, 0)

    def setUp(self):
        runner = ReplayRunner(MockProvider())
        self.state = runner._state(turn())
        self.request = build_request(self.state)

    def test_uses_batched_documented_choices_and_pinned_version(self):
        self.assertEqual(self.request["model"], "jev-1.13.0")
        self.assertEqual(set(self.request["questions"]), {"attention", "recipient"})
        self.assertTrue(all(q["type"] == "choice" for q in self.request["questions"].values()))
        self.assertEqual(parse_response(response(self.request), self.request, 0.7).label, "attend")

    def test_multilingual_context_prunes_oldest_without_phantom_recipient_or_current_loss(self):
        old = {**self.state["current_turn"], "speaker_id": "Speaker C", "text": "😀" * 4000}
        recent = {**self.state["current_turn"], "speaker_id": "Speaker B", "text": "😀" * 4000}
        current = {**self.state["current_turn"], "text": "😀" * 1000}
        state = {
            **self.state,
            "past_turns": [old, recent],
            "current_turn": current,
            "known_participants": ["Speaker A", "Speaker B", "Speaker C"],
        }
        request, payload = bounded_request(state)
        self.assertLessEqual(len(payload), 32768)
        self.assertEqual(request["state"]["past_turns"], [recent])
        self.assertEqual(request["state"]["current_turn"], current)
        self.assertEqual(request["state"]["known_participants"], ["Speaker A", "Speaker B"])
        criteria = request["questions"]["recipient"]["criteria"]
        self.assertNotIn("Speaker C", json.dumps(criteria))
        result = parse_response(
            response(request, label="ignore", recipient="speaker_1"), request, 0.7
        )
        self.assertEqual(result.recipient_speaker_id, "Speaker B")
        self.assertEqual(state["past_turns"], [old, recent])

    def test_unshrinkable_current_context_fails_before_credential_or_network(self):
        state = {
            **self.state,
            # An oversized direct provider caller must still fail safely even
            # if it bypasses the normal Turn character limit.
            "current_turn": {**self.state["current_turn"], "text": "\x00" * 6000},
            "expected_reply": "\x00" * 1000,
        }
        provider = JevProvider(allow_hosted=True)
        provider._opener = MagicMock()
        with patch("rightyo.credentials.load_jev_api_key") as key:
            with self.assertRaisesRegex(ProviderError, "payload budget"):
                provider.decide(state)
            key.assert_not_called()
            provider._opener.open.assert_not_called()

    def test_unknown_low_confidence_or_conflicting_recipient_abstains(self):
        for recipient in ("other_human", "unknown"):
            parsed = parse_response(response(self.request, recipient=recipient), self.request, 0.7)
            self.assertEqual(parsed.label, "uncertain")
        raw = response(self.request)
        raw["answers"]["attention"]["confidence"] = 0.1
        self.assertEqual(parse_response(raw, self.request, 0.7).label, "uncertain")

    def test_known_recipient_retains_actual_anonymous_speaker_mapping(self):
        self.state["known_participants"] = ["Speaker A", "Speaker B"]
        request = build_request(self.state)
        result = parse_response(
            response(request, label="ignore", recipient="speaker_1"), request, 0.7
        )
        self.assertEqual(result.recipient_speaker_id, "Speaker B")

    def test_malformed_distributions_labels_and_version_rejected(self):
        mutations = (
            lambda raw: raw.update(model="jev-latest"),
            lambda raw: raw["answers"]["attention"].update(confidence=float("nan")),
            lambda raw: raw["answers"]["attention"].update(choice="execute"),
            lambda raw: raw["answers"]["attention"].update(probabilities={"attend": 1}),
            lambda raw: raw["answers"]["attention"]["probabilities"].update(ignore=0.5),
        )
        for mutate in mutations:
            raw = response(self.request)
            mutate(raw)
            with self.assertRaises(ContractError):
                parse_response(raw, self.request, 0.7)

    def test_no_hosted_consent_never_accesses_credentials_or_network(self):
        with patch("rightyo.credentials.load_jev_api_key") as key:
            with self.assertRaises(ProviderError):
                JevProvider()
            key.assert_not_called()

    def test_budget_timeout_http_errors_are_bounded_and_sanitized(self):
        provider = JevProvider(allow_hosted=True, max_requests=1, timeout_seconds=2)
        secret = "fictitious-test-credential"
        with patch("rightyo.credentials.load_jev_api_key", return_value=secret):
            provider._opener = MagicMock()
            provider._opener.open.side_effect = urllib.error.HTTPError(
                "https://example.invalid", 429, secret, {}, io.BytesIO(b"private response")
            )
            with self.assertRaisesRegex(ProviderError, "no automatic retry") as error:
                provider.decide(self.state)
            self.assertNotIn(secret, str(error.exception))
            self.assertNotIn("private response", str(error.exception))
            self.assertIsNone(error.exception.__context__)
            self.assertEqual(provider._opener.open.call_args.kwargs["timeout"], 2)
            with self.assertRaisesRegex(ProviderError, "budget exhausted"):
                provider.decide(self.state)
            self.assertEqual(provider._opener.open.call_count, 1)

    def test_reflected_token_in_protocol_error_and_json_cannot_enter_exception_chain(self):
        secret = "fictitious-test-credential"
        provider = JevProvider(allow_hosted=True)
        provider._opener = MagicMock()
        with patch("rightyo.credentials.load_jev_api_key", return_value=secret):
            provider._opener.open.side_effect = http.client.BadStatusLine(secret)
            with self.assertRaises(ProviderError) as error:
                provider.decide(self.state)
            self.assertNotIn(secret, str(error.exception))
            self.assertIsNone(error.exception.__context__)
            request = provider._opener.open.call_args.args[0]
            self.assertIsNone(request.get_header("Authorization"))
            provider._opener.open.side_effect = None
            stream = provider._opener.open.return_value.__enter__.return_value
            stream.read.return_value = ('{"reflected":"' + secret).encode()
            with self.assertRaises(ProviderError) as error:
                provider.decide(self.state)
            self.assertNotIn(secret, str(error.exception))
            self.assertIsNone(error.exception.__context__)

    def test_valid_http_response_and_oversize_rejection_without_external_call(self):
        provider = JevProvider(allow_hosted=True)
        provider._opener = MagicMock()
        stream = provider._opener.open.return_value.__enter__.return_value
        stream.read.return_value = json.dumps(response(self.request)).encode()
        with patch("rightyo.credentials.load_jev_api_key", return_value="fictitious-test-key"):
            self.assertEqual(provider.decide(self.state).recipient, "system")
            stream.read.return_value = b"x" * 65537
            with self.assertRaisesRegex(ProviderError, "size limit"):
                provider.decide(self.state)


class CliTests(unittest.TestCase):
    def test_hosted_preflight_budget_overflow_and_late_revision_error_have_no_side_effects(self):
        ordered = [
            turn(utterance_id=f"turn-{i}", start_ms=i * 1000, end_ms=(i + 1) * 1000)
            for i in range(21)
        ]
        invalid = [*ordered[:20], replace(ordered[0], revision=1, text="conflicting content")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            for turns in (ordered, invalid):
                save_turns(path, turns)
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    patch("rightyo.cli.JevProvider") as provider,
                    patch("rightyo.credentials.load_jev_api_key") as key,
                ):
                    code = main(
                        [
                            "evaluate",
                            "--input",
                            str(path),
                            "--provider",
                            "jev",
                            "--allow-hosted",
                            "--max-requests",
                            "20",
                        ]
                    )
                    self.assertEqual(code, 2)
                    provider.assert_not_called()
                    key.assert_not_called()
                path.unlink()

    def test_preflight_counts_committed_turns_instead_of_partial_or_duplicate_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            final = turn(revision=2)
            save_turns(path, [turn(finalized=False), final, final])
            actual = MagicMock(wraps=MockProvider())
            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch("rightyo.cli.JevProvider", return_value=actual),
            ):
                code = main(
                    [
                        "evaluate",
                        "--input",
                        str(path),
                        "--provider",
                        "jev",
                        "--allow-hosted",
                        "--max-requests",
                        "1",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(actual.decide.call_count, 1)

    def test_oversized_exports_are_rejected_before_file_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.json"
            for turns in ([turn()] * 1001, [turn(text="😀" * 4000)] * 100):
                with self.assertRaises(ContractError):
                    save_turns(path, turns)
                self.assertFalse(path.exists())

    def test_fixture_smoke_output_omits_text_and_source_ids_by_default(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["evaluate", "--input", str(FIXTURE), "--provider", "mock"])
        self.assertEqual(code, 0)
        rendered = output.getvalue()
        self.assertNotIn("what time", rendered)
        self.assertNotIn("synthetic-demo", rendered)
        self.assertNotIn("Speaker A", rendered)
        result = json.loads(rendered)
        self.assertEqual(result["metrics"]["committed_decisions"], 4)
        self.assertEqual(
            result["metrics"]["label_counts"], {"attend": 1, "ignore": 1, "uncertain": 2}
        )
        self.assertFalse(result["hosted_text_processing"])
        self.assertIsNone(result["metrics"]["asr_ms"])

    def test_invalid_private_input_error_does_not_echo_content_or_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-person.json"
            path.write_text('{"private text":"do not echo",', encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stderr(output):
                code = main(["evaluate", "--input", str(path)])
            self.assertEqual(code, 2)
            self.assertNotIn("private", output.getvalue())
            self.assertNotIn(directory, output.getvalue())

    def test_hosted_mode_requires_explicit_opt_in(self):
        with (
            contextlib.redirect_stderr(io.StringIO()),
            patch("rightyo.credentials.load_jev_api_key") as key,
        ):
            self.assertEqual(main(["evaluate", "--input", str(FIXTURE), "--provider", "jev"]), 2)
            key.assert_not_called()

    def test_exports_are_private_and_do_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            save_turns(path, [turn()])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(load_turns(path), [turn()])
            with self.assertRaises(ContractError):
                save_turns(path, [turn()])
        with self.assertRaises(ContractError):
            save_turns(FIXTURE.parent / "never-write-private.json", [turn()])

    def test_exports_reject_other_checkout_and_worktree_when_installed(self):
        for is_worktree in (False, True):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                marker = root / ".git"
                if is_worktree:
                    marker.write_text("gitdir: /external/worktrees/example\n")
                else:
                    marker.mkdir()
                output = root / "nested" / "private.json"
                output.parent.mkdir()
                with patch("rightyo.cli.__file__", "/venv/lib/site-packages/rightyo/cli.py"):
                    with self.assertRaises(ContractError):
                        save_turns(output, [turn()])
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
