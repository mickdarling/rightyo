"""Tool integration uses authored text and fake native/hosted resources only."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rightyo.cli import main
from rightyo.contracts import ContractError, Turn
from rightyo.prototype import PrototypeConfig, PrototypeController, PrototypeError
from rightyo.providers import JevProvider, MockProvider, ProviderError
from rightyo.tool import listen, replay
from rightyo.tool_events import SpeechEvents


class Processor:
    instances = []

    def __init__(self, config, callback):
        self.config, self.callback = config, callback
        self.index = 0
        self.closed = False
        self.instances.append(self)

    def push_pcm16(self, pcm):
        self.index += 1
        self.callback(
            Turn(
                self.config.session_id,
                f"authored-{self.index}",
                1,
                (self.index - 1) * 200,
                self.index * 200,
                "Rightyo, tell me what happened.",
                "Speaker A",
                True,
                False,
                "authored-fixture",
                "causal-replay",
                "authored-fixture",
            )
        )

    def finish(self):
        pass

    def close(self):
        self.closed = True


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.asset = self.root / "asset"
        self.asset.touch()
        self.demo = self.root / "authored.wav"
        with wave.open(str(self.demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(12800))
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    **{
                        key: str(self.asset)
                        for key in (
                            "whisper_executable",
                            "whisper_model",
                            "diarization_library",
                            "diarization_model",
                            "microphone_helper",
                        )
                    },
                    "demo_audio": str(self.demo),
                }
            )
        )
        Processor.instances.clear()
        self.args = Namespace(
            config=self.config,
            mode="demo",
            session_id="tool-test",
            use_jev=False,
            allow_hosted=False,
        )

    def test_transcript_only_burst_never_uses_attention_queue_or_provider(self):
        with wave.open(str(self.demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(40 * 6400))
        output = io.StringIO()
        with patch.object(
            MockProvider, "decide", side_effect=AssertionError("unused provider")
        ) as provider:
            self.assertEqual(listen(self.args, output=output, controller_factory=self.factory), 0)
        provider.assert_not_called()
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[0]["capabilities"]["activation"], "disabled")
        self.assertEqual(events[-1]["phase"], "stopped")
        self.assertEqual(sum(e["type"] == "transcript" for e in events), 40)
        self.assertFalse(any(e["type"] in {"attention", "request"} for e in events))

    def factory(self, config, *, event_publisher, report=None):
        return PrototypeController(
            config,
            event_publisher=event_publisher,
            processor_factory=Processor,
            capture_factory=self.no_capture,
            provider_factory=self.no_hosted,
        )

    @staticmethod
    def no_capture(*_args, **_kwargs):
        raise AssertionError("Tool test must not start a microphone")

    @staticmethod
    def no_hosted(*_args, **_kwargs):
        raise AssertionError("Tool test must not use hosted inference")

    def test_explicit_demo_headless_emits_transcripts_only_and_one_terminal(self):
        output = io.StringIO()
        self.assertEqual(listen(self.args, output=output, controller_factory=self.factory), 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            [event["type"] for event in events], ["session", "transcript", "transcript", "session"]
        )
        self.assertEqual(events[0]["phase"], "started")
        self.assertEqual(events[0]["capabilities"]["activation"], "disabled")
        self.assertEqual(events[-1]["phase"], "stopped")
        self.assertTrue(all(event["session_id"] == "tool-test" for event in events))
        self.assertTrue(all(processor.closed for processor in Processor.instances))

    def test_hosted_consent_is_checked_before_configuration_or_runtime(self):
        self.args.config = self.root / "missing"
        for use_jev, allow_hosted in ((True, False), (False, True)):
            self.args.use_jev, self.args.allow_hosted = use_jev, allow_hosted
            with self.assertRaisesRegex(PrototypeError, "requires"):
                listen(self.args, controller_factory=self.no_capture)
        self.assertEqual(Processor.instances, [])

    def test_slow_consumer_backlog_cancels_and_emits_only_safe_terminal(self):
        with wave.open(str(self.demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(10 * 6400))

        def factory(config, *, event_publisher, report=None):
            controller = self.factory(config, event_publisher=SpeechEvents(max_pending=5))
            original_start = controller.start

            def start(options):
                original_start(options)
                controller._audio_thread.join(timeout=1)

            controller.start = start
            return controller

        output = io.StringIO()
        self.assertEqual(listen(self.args, output=output, controller_factory=factory), 2)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["phase"], "error")
        self.assertEqual(events[0]["reason"], "event-delivery-failed")
        self.assertNotIn("tell me", output.getvalue())
        self.assertTrue(all(processor.closed for processor in Processor.instances))

    def test_cli_fixture_works_without_capture_credentials_or_native_assets(self):
        output = io.StringIO()
        fixture = Path(__file__).resolve().parents[1] / "examples" / "synthetic-turns.json"
        with patch("sys.stdout", output):
            self.assertEqual(main(["tool-replay", "--input", str(fixture)]), 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(sum(event["type"] == "transcript" for event in events), 4)
        request = next(event for event in events if event["type"] == "request")
        self.assertEqual(request["decision"]["provider"], "mock")
        self.assertEqual(request["turn"]["revision"], 2)
        self.assertEqual(events[-1]["phase"], "stopped")

    def test_unicode_replay_survives_ascii_stdout_and_preserves_request(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "synthetic-turns.json"
        raw = json.loads(fixture.read_text())
        text = "Rightyo, résume cette conversation — 你好 👋."
        raw["turns"][1]["text"] = text
        supplied = self.root / "unicode.json"
        supplied.write_text(json.dumps(raw), encoding="utf-8")
        encoded = io.BytesIO()
        output = io.TextIOWrapper(encoded, encoding="ascii")
        self.addCleanup(output.close)
        with patch("sys.stdout", output):
            self.assertEqual(main(["tool-replay", "--input", str(supplied)]), 0)
        events = [json.loads(line) for line in encoded.getvalue().splitlines()]
        request = next(event for event in events if event["type"] == "request")
        self.assertEqual(request["turn"]["text"], text)
        self.assertEqual(events[-1]["phase"], "stopped")

    def test_hosted_failure_and_budget_exhaustion_are_terminal_not_silent(self):
        for failure in (True, False):
            with self.subTest(failure=failure):

                class Provider:
                    def __init__(self):
                        self.requests = 0

                    def decide(self, state):
                        self.requests += 1
                        if failure:
                            raise ProviderError("Authored provider failure")
                        return MockProvider().decide(state)

                provider = Provider()
                self.args.use_jev = self.args.allow_hosted = True

                def factory(config, *, event_publisher, report=None):
                    controller = PrototypeController(
                        config,
                        event_publisher=event_publisher,
                        processor_factory=Processor,
                        capture_factory=self.no_capture,
                        provider_factory=lambda **_options: provider,
                    )
                    original_start = controller.start

                    def start(options):
                        original_start(options | {"max_requests": 1})

                    controller.start = start
                    return controller

                output = io.StringIO()
                self.assertEqual(listen(self.args, output=output, controller_factory=factory), 2)
                events = [json.loads(line) for line in output.getvalue().splitlines()]
                self.assertEqual(events[-1]["phase"], "error")
                self.assertEqual(
                    events[-1]["reason"],
                    "attention-unavailable" if failure else "attention-budget-exhausted",
                )
                self.assertEqual(provider.requests, 1)
                self.assertTrue(all(processor.closed for processor in Processor.instances))

    def test_broken_pipe_always_closes_controller_and_stops_capture(self):
        class Output:
            def write(self, _text):
                raise BrokenPipeError

        controllers = []

        def factory(config, *, event_publisher, report=None):
            controller = self.factory(config, event_publisher=event_publisher)
            controllers.append(controller)
            return controller

        with self.assertRaises(BrokenPipeError):
            listen(self.args, output=Output(), controller_factory=factory)
        self.assertTrue(controllers[0]._closed.is_set())
        self.assertTrue(controllers[0]._stop.is_set())
        self.assertEqual(controllers[0].snapshot()["turns"], [])

    def test_slow_provider_context_backlog_stops_native_source_and_clears_text(self):
        with wave.open(str(self.demo), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(16000)
            audio.writeframes(bytes(40 * 6400))
        entered = threading.Event()
        finished = threading.Event()
        providers = []
        controllers = []

        class SlowProvider:
            def __init__(self, *, cancelled, **_options):
                self.cancelled = cancelled
                self.requests = 0
                providers.append(self)

            def decide(self, state):
                self.requests += 1
                entered.set()
                while not self.cancelled():
                    threading.Event().wait(0.005)
                finished.set()
                return MockProvider().decide(state)

        class CausalProcessor(Processor):
            def push_pcm16(self, pcm):
                super().push_pcm16(pcm)
                if self.index == 1 and not entered.wait(1):
                    raise AssertionError("Fake provider did not begin")

        def factory(config, *, event_publisher, report=None):
            controller = PrototypeController(
                config,
                event_publisher=event_publisher,
                processor_factory=CausalProcessor,
                capture_factory=self.no_capture,
                provider_factory=SlowProvider,
            )
            controllers.append(controller)
            return controller

        self.args.use_jev = self.args.allow_hosted = True
        output = io.StringIO()
        self.assertEqual(listen(self.args, output=output, controller_factory=factory), 2)
        self.assertTrue(finished.wait(1))
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["phase"], "error")
        self.assertEqual(events[-1]["reason"], "event-delivery-failed")
        self.assertFalse(any(event["type"] == "request" for event in events))
        self.assertEqual(providers[0].requests, 1)
        self.assertEqual(controllers[0]._memory.retained_ids, set())
        self.assertTrue(controllers[0]._decision_queue.empty())
        self.assertTrue(all(processor.closed for processor in Processor.instances))

    def test_keyboard_interruption_emits_cancelled_and_clears(self):
        controllers = []

        def factory(config, *, event_publisher, report=None):
            controller = self.factory(config, event_publisher=event_publisher)
            controllers.append(controller)
            controller.snapshot = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
            return controller

        output = io.StringIO()
        self.assertEqual(listen(self.args, output=output, controller_factory=factory), 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        # It may finish the supplied fixture before the interruption; either terminal
        # means no pending capture/content survives the foreground operation.
        self.assertIn(events[-1]["phase"], {"cancelled", "stopped"})
        self.assertTrue(controllers[0]._closed.is_set())
        self.assertEqual(controllers[0]._memory.retained_ids, set())

    def test_session_budget_flag_parses_validates_and_overrides_configuration(self):
        for command in (
            ["listen", "--config", str(self.config), "--mode", "demo"],
            ["prototype", "--config", str(self.config)],
        ):
            for invalid in ("0", "-5", "abc", "1.5"):
                with self.subTest(command=command[0], invalid=invalid):
                    with patch("sys.stderr", io.StringIO()) as error:
                        with self.assertRaises(SystemExit) as exit:
                            main([*command, "--session-budget", invalid])
                    self.assertEqual(exit.exception.code, 2)
                    self.assertIn("positive number of seconds", error.getvalue())
        Processor.instances.clear()
        with patch("rightyo.prototype.serve") as serve:
            self.assertEqual(
                main(["prototype", "--config", str(self.config), "--session-budget", "30"]), 0
            )
            serve.assert_called_once_with(
                self.config, 8765, addressing=None, session_budget_seconds=30, allow_hosted=False
            )
            serve.reset_mock()
            self.assertEqual(main(["prototype", "--config", str(self.config)]), 0)
            serve.assert_called_once_with(
                self.config, 8765, addressing=None, session_budget_seconds=None, allow_hosted=False
            )
        self.assertEqual(Processor.instances, [])
        raw = json.loads(self.config.read_text())
        self.config.write_text(json.dumps({**raw, "session_budget_seconds": 60}))
        loaded = []

        def factory(config, *, event_publisher, report=None):
            loaded.append(config.session_budget_seconds)
            return self.factory(config, event_publisher=event_publisher)

        for flag in (None, 5):
            args = Namespace(**vars(self.args), session_budget=flag)
            with self.subTest(flag=flag):
                self.assertEqual(listen(args, output=io.StringIO(), controller_factory=factory), 0)
        self.assertEqual(listen(self.args, output=io.StringIO(), controller_factory=factory), 0)
        self.assertEqual(loaded, [60, 5, 60])
        self.assertTrue(
            all(p.config.session_budget_ms == s * 1000 for p, s in zip(Processor.instances, loaded))
        )
        with self.assertRaisesRegex(PrototypeError, "budget"):
            listen(Namespace(**vars(self.args), session_budget=0), controller_factory=factory)

    def test_replay_rejects_late_invalid_input_before_emission_or_provider(self):
        fixture = Path(__file__).resolve().parents[1] / "examples" / "synthetic-turns.json"
        raw = json.loads(fixture.read_text())
        raw["turns"].append({})
        invalid = self.root / "invalid.json"
        invalid.write_text(json.dumps(raw))
        args = Namespace(
            input=invalid,
            provider="jev",
            max_requests=20,
            allow_hosted=True,
            timeout=10,
            min_confidence=0.7,
        )
        output = io.StringIO()
        with patch("rightyo.tool.JevProvider", side_effect=self.no_hosted):
            with self.assertRaises(ContractError):
                replay(args, output=output)
        self.assertEqual(output.getvalue(), "")


class ConfigDecisionTests(unittest.TestCase):
    """The configuration's `decision` section; no network, Keychain, or capture."""

    def setUp(self):
        self.tool = ToolTests("test_explicit_demo_headless_emits_transcripts_only_and_one_terminal")
        self.tool.setUp()
        self.addCleanup(self.tool.doCleanups)
        self.args = self.tool.args
        self.base = json.loads(self.tool.config.read_text())

    def write(self, decision):
        self.tool.config.write_text(json.dumps({**self.base, "decision": decision}))

    def run_listen(self):
        """Run demo `listen`; return the provider_factory calls and what each built."""
        built = []

        def provider_factory(**options):
            provider = JevProvider(**options)
            built.append((options, provider))
            return provider

        def factory(config, *, event_publisher, report=None):
            return PrototypeController(
                config,
                event_publisher=event_publisher,
                processor_factory=Processor,
                capture_factory=ToolTests.no_capture,
                provider_factory=provider_factory,
            )

        no_process = AssertionError("must not run /usr/bin/security")
        no_network = AssertionError("must not open a network connection")
        # The real constructor runs; decisions are answered by the fixture provider so
        # no credential lookup or request is ever attempted.
        with (
            patch("rightyo.credentials.subprocess.Popen", side_effect=no_process),
            patch("urllib.request.OpenerDirector.open", side_effect=no_network),
            patch.object(JevProvider, "decide", lambda _self, state: MockProvider().decide(state)),
        ):
            output = io.StringIO()
            self.assertEqual(listen(self.args, output=output, controller_factory=factory), 0)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(events[-1]["phase"], "stopped")
        return built

    def test_config_jev_with_consent_constructs_the_jev_provider(self):
        self.write({"provider": "jev", "allow_hosted": True})
        self.assertTrue(PrototypeConfig.load(self.tool.config).hosted_decisions)
        built = self.run_listen()
        self.assertEqual(len(built), 1)
        options, provider = built[0]
        self.assertIsInstance(provider, JevProvider)
        self.assertIs(options["allow_hosted"], True)

    def test_config_mock_and_absent_section_keep_the_mock_provider(self):
        self.assertFalse(PrototypeConfig.load(self.tool.config).hosted_decisions)
        self.assertEqual(self.run_listen(), [])
        self.write({"provider": "mock", "allow_hosted": False})
        self.assertFalse(PrototypeConfig.load(self.tool.config).hosted_decisions)
        self.assertEqual(self.run_listen(), [])

    def test_config_jev_without_consent_is_refused_before_runtime(self):
        for decision in (
            {"provider": "jev", "allow_hosted": False},
            {"provider": "jev"},
        ):
            with self.subTest(decision=decision):
                self.write(decision)
                with self.assertRaisesRegex(PrototypeError, "allow_hosted"):
                    listen(self.args, controller_factory=ToolTests.no_capture)
        self.assertEqual(Processor.instances, [])

    def test_bad_types_unknown_provider_and_extra_keys_are_refused(self):
        cases = {
            "must be an object": (None, ["jev", True], "jev", True),
            "exactly the keys": (
                {},
                {"allow_hosted": True},
                {"provider": "jev", "allow_hosted": True, "model": "x"},
            ),
            "Unknown decision provider": (
                {"provider": "openai", "allow_hosted": True},
                {"provider": "Jev", "allow_hosted": True},
                {"provider": 1, "allow_hosted": True},
                {"provider": None, "allow_hosted": True},
            ),
            "true or false": (
                {"provider": "jev", "allow_hosted": "true"},
                {"provider": "jev", "allow_hosted": 1},
                {"provider": "jev", "allow_hosted": None},
            ),
            "applies only to the jev": ({"provider": "mock", "allow_hosted": True},),
        }
        for message, decisions in cases.items():
            for decision in decisions:
                with self.subTest(decision=decision):
                    self.write(decision)
                    with self.assertRaisesRegex(PrototypeError, message):
                        PrototypeConfig.load(self.tool.config)
                    with self.assertRaisesRegex(PrototypeError, message):
                        listen(self.args, controller_factory=ToolTests.no_capture)
        self.assertEqual(Processor.instances, [])

    def test_cli_flags_keep_their_rules_and_precedence(self):
        # --use-jev alone is refused before the configuration is read, even if the
        # file consents; the file never supplies the missing CLI half.
        self.write({"provider": "jev", "allow_hosted": True})
        self.args.use_jev, self.args.allow_hosted = True, False
        with self.assertRaisesRegex(PrototypeError, "both --use-jev and --allow-hosted"):
            listen(self.args, controller_factory=ToolTests.no_capture)
        # --allow-hosted alone stays refused when nothing hosted is selected.
        self.write({"provider": "mock", "allow_hosted": False})
        self.args.use_jev, self.args.allow_hosted = False, True
        with self.assertRaisesRegex(PrototypeError, "--allow-hosted requires"):
            listen(self.args, controller_factory=ToolTests.no_capture)
        # Both flags select Jev even when the file names the mock provider.
        self.args.use_jev = self.args.allow_hosted = True
        self.assertEqual(len(self.run_listen()), 1)

    def test_section_never_consents_to_hosted_speech(self):
        self.tool.config.write_text(
            json.dumps(
                {
                    **self.base,
                    "decision": {"provider": "jev", "allow_hosted": True},
                    "diarizer": {"kind": "hosted-deepgram"},
                }
            )
        )
        with self.assertRaisesRegex(PrototypeError, "Hosted speech backends require"):
            listen(self.args, controller_factory=ToolTests.no_capture)


if __name__ == "__main__":
    unittest.main()
