"""Live Jev budget (#75) and transient-failure resilience (#71) with fake Jev and no audio.

No test reaches the real Keychain, /usr/bin/security or the Jev endpoint: live sessions use
a scripted provider, and provider tests replace the opener and the credential lookup.
"""

from __future__ import annotations

import contextlib
import io
import json
import socket
import tempfile
import threading
import unittest
import urllib.error
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from rightyo.capture import PCM_CHUNK_BYTES
from rightyo.cli import main
from rightyo.contracts import Turn
from rightyo.prototype import (
    MAX_CONSECUTIVE_DECISION_FAILURES,
    PrototypeConfig,
    PrototypeController,
    PrototypeError,
)
from rightyo.providers import (
    JevProvider,
    MockProvider,
    ProviderError,
    ProviderUnavailable,
    replay_request_budget,
)
from rightyo.tool import listen

FIXTURE = Path(__file__).resolve().parents[1] / "examples" / "synthetic-turns.json"
ATTENDED = "Rightyo, what is on the calendar?"
BACKGROUND = "Synthetic background conversation."
OK = "ok"


def transient(reason="timeout"):
    return ProviderUnavailable("Jev connection failed or timed out", reason)


def malformed():
    return ProviderUnavailable("Jev returned an invalid structured response", "malformed-response")


class Paced:
    """One PCM chunk per read, each released only after the previous turn's decision."""

    def __init__(self, count, decided):
        self.remaining, self.decided = count, decided

    def read(self, _limit):
        if self.remaining == 0 or not self.decided.acquire(timeout=5):
            return b""
        self.remaining -= 1
        return bytes(PCM_CHUNK_BYTES)


class TurnPerChunk:
    """A fake live processor that finalizes one turn per pushed chunk."""

    texts: list[str] = []

    def __init__(self, config, callback):
        self.config, self.callback, self.index = config, callback, 0

    def push_pcm16(self, _pcm):
        index = self.index
        self.index += 1
        text = self.texts[index % len(self.texts)]
        self.callback(
            Turn(
                self.config.session_id,
                f"live-{index}",
                1,
                index * 200,
                index * 200 + 150,
                text,
                "Speaker A",
                True,
                False,
                "authored-fixture",
                self.config.provenance,
                "authored-fixture",
            )
        )

    def finish(self):
        pass

    def close(self):
        pass


class ScriptedJev:
    """Answers like the fixture rule, or raises the scripted failure for that call."""

    def __init__(self, script, decided, built, **options):
        self.script, self.decided, self.options = script, decided, options
        self.max_requests = options["max_requests"]
        self.requests = 0
        built.append(self)

    def decide(self, state):
        try:
            if self.max_requests is not None and self.requests >= self.max_requests:
                raise ProviderError("Jev request budget exhausted")
            outcome = self.script[self.requests] if self.requests < len(self.script) else OK
            self.requests += 1
            if outcome is not OK:
                raise outcome
            return MockProvider().decide(state)
        finally:
            self.decided.release()


class LiveDecisionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        asset = root / "asset"
        asset.touch()
        self.config = root / "config.json"
        self.assets = {
            key: str(asset)
            for key in (
                "whisper_executable",
                "whisper_model",
                "diarization_library",
                "diarization_model",
                "microphone_helper",
            )
        }
        self.write_config()

    def write_config(self, **extra):
        self.config.write_text(json.dumps({**self.assets, **extra}))

    def run_listen(self, turns, script=(), texts=(BACKGROUND,)):
        """Run `listen --mode stdin --use-jev --allow-hosted` over synthetic paced turns."""
        TurnPerChunk.texts = list(texts)
        decided = threading.Semaphore(1)
        built = []

        def factory(config, *, event_publisher, audio_input, audio_provenance, report):
            return PrototypeController(
                config,
                event_publisher=event_publisher,
                processor_factory=TurnPerChunk,
                provider_factory=lambda **options: ScriptedJev(
                    list(script), decided, built, **options
                ),
                audio_input=audio_input,
                audio_provenance=audio_provenance,
                report=report,
            )

        args = Namespace(
            config=self.config,
            mode="stdin",
            provenance="live-microphone",
            session_id="live-decisions",
            use_jev=True,
            allow_hosted=True,
        )
        output = io.StringIO()
        with contextlib.redirect_stderr(io.StringIO()):
            code = listen(
                args,
                output=output,
                controller_factory=factory,
                audio_input=Paced(turns, decided),
            )
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        return code, events, built

    @staticmethod
    def of(events, kind):
        return [e for e in events if e["type"] == kind]

    def test_live_session_is_not_capped_after_more_than_100_decisions(self):
        code, events, built = self.run_listen(105)
        self.assertEqual(code, 0)
        self.assertIsNone(built[0].options["max_requests"])
        self.assertEqual(built[0].requests, 105)
        self.assertEqual(len(self.of(events, "attention")), 105)
        self.assertEqual((events[-1]["phase"], events[-1].get("reason")), ("stopped", None))

    def test_configuration_can_cap_a_live_session(self):
        self.write_config(decision={"provider": "jev", "allow_hosted": True, "max_requests": 3})
        code, events, built = self.run_listen(5)
        self.assertEqual(code, 2)
        self.assertEqual(built[0].options["max_requests"], 3)
        self.assertEqual(len(self.of(events, "attention")), 3)
        self.assertEqual(
            (events[-1]["phase"], events[-1]["reason"]), ("error", "attention-budget-exhausted")
        )

    def test_one_transient_failure_degrades_its_turn_and_the_session_stays_alive(self):
        code, events, _built = self.run_listen(
            3, script=[OK, transient("timeout")], texts=(ATTENDED,)
        )
        self.assertEqual(code, 0)
        self.assertEqual(events[-1]["phase"], "stopped")
        attention = self.of(events, "attention")
        self.assertEqual([a["utterance_id"] for a in attention], ["live-0", "live-1", "live-2"])
        degraded = attention[1]
        self.assertNotIn("request_id", degraded)
        self.assertEqual(
            degraded["decision"],
            {
                "label": "uncertain",
                "recipient_kind": "unknown",
                "confidence": 0.0,
                "provider": "jev",
                "model": "jev-1.13.0",
                "decision_status": "unavailable",
                "reason": "timeout",
            },
        )
        # The failed turn never becomes a request; the turns around it still do.
        self.assertEqual(
            [r["turn"]["utterance_id"] for r in self.of(events, "request")], ["live-0", "live-2"]
        )
        self.assertNotIn("decision_status", attention[0]["decision"])
        # The degraded turn remains ordinary context for the next request.
        context = self.of(events, "request")[1]["context"]["turns"]
        self.assertIn("live-1", [t["utterance_id"] for t in context])

    def test_a_success_resets_the_consecutive_failure_count(self):
        below = MAX_CONSECUTIVE_DECISION_FAILURES - 1
        failures = [transient("server-error")] * below
        code, events, _built = self.run_listen(2 * below + 2, script=[*failures, OK, *failures])
        self.assertEqual(code, 0)
        self.assertEqual(events[-1]["phase"], "stopped")
        reasons = [a["decision"].get("reason") for a in self.of(events, "attention")]
        self.assertEqual(reasons.count("server-error"), 2 * below)

    def test_consecutive_failures_end_the_session(self):
        limit = MAX_CONSECUTIVE_DECISION_FAILURES
        self.assertEqual(limit, 5)
        code, events, built = self.run_listen(
            limit + 3, script=[OK, *[transient("rate-limited")] * limit]
        )
        self.assertEqual(code, 2)
        self.assertEqual(built[0].requests, limit + 1)
        self.assertEqual(
            (events[-1]["phase"], events[-1]["reason"]), ("error", "attention-unavailable")
        )

    def test_a_malformed_answer_degrades_its_turn_and_later_turns_still_decide(self):
        code, events, _built = self.run_listen(3, script=[OK, malformed()], texts=(ATTENDED,))
        self.assertEqual(code, 0)
        self.assertEqual(events[-1]["phase"], "stopped")
        attention = self.of(events, "attention")
        self.assertEqual(len(attention), 3)
        self.assertEqual(
            (attention[1]["decision"]["label"], attention[1]["decision"]["decision_status"]),
            ("uncertain", "unavailable"),
        )
        self.assertEqual(attention[1]["decision"]["reason"], "malformed-response")
        self.assertEqual(attention[2]["decision"]["label"], "attend")
        self.assertEqual(
            [r["turn"]["utterance_id"] for r in self.of(events, "request")], ["live-0", "live-2"]
        )

    def test_consecutive_malformed_answers_end_the_session(self):
        limit = MAX_CONSECUTIVE_DECISION_FAILURES
        code, events, built = self.run_listen(limit + 3, script=[OK, *[malformed()] * limit])
        self.assertEqual(code, 2)
        self.assertEqual(built[0].requests, limit + 1)
        self.assertEqual(
            (events[-1]["phase"], events[-1]["reason"]), ("error", "attention-unavailable")
        )

    def test_a_permanent_provider_error_still_ends_the_session_at_once(self):
        code, events, built = self.run_listen(
            4, script=[OK, ProviderError("Jev request failed (HTTP 401)")]
        )
        self.assertEqual(code, 2)
        self.assertEqual(built[0].requests, 2)
        self.assertEqual(
            (events[-1]["phase"], events[-1]["reason"]), ("error", "attention-unavailable")
        )

    def test_decision_max_requests_is_validated(self):
        cases = {
            "positive whole number": (0, -1, True, 1.5, "3", None),
        }
        for message, values in cases.items():
            for value in values:
                with self.subTest(value=value):
                    self.write_config(
                        decision={"provider": "jev", "allow_hosted": True, "max_requests": value}
                    )
                    with self.assertRaisesRegex(PrototypeError, message):
                        PrototypeConfig.load(self.config)
        self.write_config(decision={"provider": "mock", "allow_hosted": False, "max_requests": 3})
        with self.assertRaisesRegex(PrototypeError, "only to the jev"):
            PrototypeConfig.load(self.config)
        self.write_config(decision={"provider": "jev", "allow_hosted": True, "max_requests": 500})
        self.assertEqual(PrototypeConfig.load(self.config).decision_max_requests, 500)
        self.write_config(decision={"provider": "jev", "allow_hosted": True})
        self.assertIsNone(PrototypeConfig.load(self.config).decision_max_requests)


class ProviderBudgetAndFailureTests(unittest.TestCase):
    state = {
        "past_turns": [],
        "current_turn": {
            "text": "Rightyo, help me.",
            "speaker_id": "Speaker A",
            "start_ms": 0,
            "end_ms": 1000,
            "overlap": False,
        },
        "known_participants": ["Speaker A"],
        "expected_reply": False,
        "playback_active": False,
        "addressing": None,
    }

    def test_uncapped_provider_reserves_past_100_and_replay_budgets_stay_bounded(self):
        provider = JevProvider(allow_hosted=True, max_requests=None)
        for _ in range(150):
            provider._reserve()
        self.assertEqual(provider.requests, 150)
        bounded = JevProvider(allow_hosted=True, max_requests=2)
        bounded._reserve()
        bounded._reserve()
        with self.assertRaisesRegex(ProviderError, "budget exhausted"):
            bounded._reserve()
        for value in (0, 101, True, None):
            with self.subTest(value=value), self.assertRaisesRegex(ProviderError, "1 and 100"):
                replay_request_budget(value)
        self.assertEqual(replay_request_budget(100), 100)

    def test_replay_commands_still_refuse_more_than_100_requests(self):
        for command in ("evaluate", "tool-replay"):
            errors = io.StringIO()
            with (
                self.subTest(command=command),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(errors),
                patch("rightyo.credentials.load_jev_api_key") as key,
                patch("urllib.request.OpenerDirector.open") as network,
            ):
                code = main(
                    [
                        command,
                        "--input",
                        str(FIXTURE),
                        "--provider",
                        "jev",
                        "--allow-hosted",
                        "--max-requests",
                        "101",
                    ]
                )
                self.assertEqual(code, 2)
                self.assertIn("request budget must be between 1 and 100", errors.getvalue())
                key.assert_not_called()
                network.assert_not_called()

    def failure(self, side_effect, body=b""):
        provider = JevProvider(allow_hosted=True, max_requests=5, timeout_seconds=2)
        provider._opener = MagicMock()
        provider._opener.open.side_effect = side_effect
        provider._opener.open.return_value.__enter__.return_value.read.return_value = body
        with (
            patch("rightyo.credentials.load_jev_api_key", return_value="fictitious-test-key"),
            self.assertRaises(ProviderError) as raised,
        ):
            provider.decide(self.state)
        self.assertIsNone(raised.exception.__context__)
        return raised.exception

    def http(self, code):
        return urllib.error.HTTPError(
            "https://example.invalid", code, "reason", {}, io.BytesIO(b"private")
        )

    def test_transient_failures_are_classified_with_a_reason(self):
        cases = {
            "rate-limited": (self.http(429), self.http(529)),
            "server-error": (self.http(500), self.http(503)),
            "timeout": (TimeoutError(), socket.timeout(), urllib.error.URLError(TimeoutError())),
            "connection-failed": (urllib.error.URLError("refused"), ConnectionResetError()),
        }
        for reason, errors in cases.items():
            for error in errors:
                with self.subTest(error=error):
                    raised = self.failure(error)
                    self.assertIsInstance(raised, ProviderUnavailable)
                    self.assertEqual(raised.reason, reason)

    def test_malformed_answers_are_unavailable_but_oversized_ones_stay_permanent(self):
        valid = {"model": "jev-1.13.0", "answers": {}}
        for body in (b"not json", b"[]", json.dumps(valid).encode()):
            with self.subTest(body=body):
                raised = self.failure(None, body)
                self.assertIsInstance(raised, ProviderUnavailable)
                self.assertEqual(raised.reason, "malformed-response")
        oversized = self.failure(None, b" " * 65537)
        self.assertNotIsInstance(oversized, ProviderUnavailable)

    def test_permanent_failures_stay_plain_provider_errors(self):
        for error in (self.http(400), self.http(401), self.http(403), self.http(404)):
            with self.subTest(error=error):
                self.assertNotIsInstance(self.failure(error), ProviderUnavailable)


if __name__ == "__main__":
    unittest.main()
