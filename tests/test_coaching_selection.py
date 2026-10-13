"""Explicit provider selection cannot borrow ordinary hosted consent or start capture."""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rightyo.cli import main
from rightyo.coaching import MockCoachProvider
from rightyo.coaching_jev import JevCatalogCoachProvider
from rightyo.coaching_listen import provider_from_args
from rightyo.contracts import ContractError
from rightyo.prototype import PrototypeError
from rightyo.tool import listen


class CoachingSelectionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.catalog = self.root / "catalog.json"
        self.catalog.write_text(json.dumps({"clarify": "Ask which outcome matters most."}))
        self.config = self.root / "config.json"
        self.config.write_text(
            json.dumps(
                {
                    "transcriber": {"kind": "synthetic-authored", "utterances": ["A question."]},
                    "diarizer": {"kind": "synthetic-authored"},
                }
            )
        )

    def options(self, **updates):
        return Namespace(
            **{
                "coaching_provider": "jev",
                "coaching_allow_hosted_text": True,
                "coaching_max_requests": 1,
                "coaching_advice_catalog": self.catalog,
                "coaching_mock_suggestions": None,
                **updates,
            }
        )

    def test_default_is_mock_and_jev_construction_has_no_credential_or_network_work(self):
        with (
            patch("rightyo.coaching_jev.load_jev_api_key") as credential,
            patch("rightyo.coaching_jev._JevClient") as transport,
        ):
            self.assertIsInstance(provider_from_args(Namespace()), MockCoachProvider)
            provider = provider_from_args(self.options())
            self.assertIsInstance(provider, JevCatalogCoachProvider)
            self.assertEqual(provider.max_requests, 1)
            credential.assert_not_called()
            transport.assert_not_called()

    def test_jev_requires_its_own_consent_explicit_budget_and_catalog(self):
        for changes in (
            {"coaching_allow_hosted_text": False, "allow_hosted": True},
            {"coaching_max_requests": None},
            {"coaching_max_requests": 0},
            {"coaching_max_requests": 101},
            {"coaching_max_requests": True},
            {"coaching_advice_catalog": None},
            {"coaching_mock_suggestions": self.catalog},
            {"coaching_provider": "other"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                provider_from_args(self.options(**changes))

    def test_mock_rejects_jev_only_options_instead_of_silently_ignoring_them(self):
        for changes in (
            {"coaching_allow_hosted_text": True},
            {"coaching_max_requests": 1},
            {"coaching_advice_catalog": self.catalog},
        ):
            options = {
                "coaching_provider": "mock",
                "coaching_allow_hosted_text": False,
                "coaching_max_requests": None,
                "coaching_advice_catalog": None,
                **changes,
            }
            with self.subTest(changes=changes), self.assertRaises(ContractError):
                provider_from_args(self.options(**options))

    def test_catalog_has_bounded_reads_strict_json_and_content_free_failures(self):
        for content in (
            b" " * 8193,
            b'{"x":"A tip.","x":"Another tip."}',
            b'{"x":NaN}',
            b'{"x":"private-marker\\n"}',
            b"[]",
        ):
            self.catalog.write_bytes(content)
            with self.subTest(size=len(content)), self.assertRaises(ContractError) as caught:
                provider_from_args(self.options())
            self.assertNotIn("private-marker", str(caught.exception))
        self.catalog.unlink()
        with self.assertRaises(ContractError) as caught:
            provider_from_args(self.options())
        self.assertNotIn(str(self.catalog), str(caught.exception))

    def test_coaching_options_cannot_enter_ordinary_listener_or_construct_controller(self):
        base = {
            "config": self.config,
            "mode": "stdin",
            "provenance": "synthetic",
            "session_id": "11111111-1111-4111-8111-111111111111",
            "use_jev": False,
            "allow_hosted": False,
        }
        for changes in (
            {"coaching_provider": "jev"},
            {"coaching_allow_hosted_text": True},
            {"coaching_max_requests": 1},
            {"coaching_advice_catalog": self.catalog},
        ):
            with (
                self.subTest(changes=changes),
                patch("rightyo.tool.PrototypeController") as factory,
                self.assertRaises(PrototypeError),
            ):
                listen(
                    Namespace(**base, **changes), output=io.StringIO(), controller_factory=factory
                )
            factory.assert_not_called()

    def test_cli_preserves_mock_default_and_exposes_separate_consent_and_budget(self):
        with patch("rightyo.tool.listen", return_value=0) as call:
            self.assertEqual(main(["listen", "--config", str(self.config), "--mode", "stdin"]), 0)
            self.assertEqual(call.call_args.args[0].coaching_provider, "mock")
            self.assertFalse(call.call_args.args[0].coaching_allow_hosted_text)
            self.assertIsNone(call.call_args.args[0].coaching_max_requests)
            self.assertEqual(
                main(
                    [
                        "listen",
                        "--config",
                        str(self.config),
                        "--mode",
                        "stdin",
                        "--coaching",
                        "--coaching-provider",
                        "jev",
                        "--coaching-allow-hosted-text",
                        "--coaching-max-requests",
                        "3",
                        "--coaching-advice-catalog",
                        str(self.catalog),
                    ]
                ),
                0,
            )
            args = call.call_args.args[0]
            self.assertEqual(args.coaching_provider, "jev")
            self.assertTrue(args.coaching_allow_hosted_text)
            self.assertEqual(args.coaching_max_requests, 3)
            self.assertEqual(args.coaching_advice_catalog, self.catalog)
            self.assertFalse(args.allow_hosted)


if __name__ == "__main__":
    unittest.main()
