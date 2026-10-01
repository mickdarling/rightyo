"""Review secret setup tests use temporary executables and invented environment tokens."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "setup_review_secrets.py"
SPEC = importlib.util.spec_from_file_location("review_secret_setup", SOURCE)
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


class ReviewSecretSetupTests(unittest.TestCase):
    def test_rejects_writable_cli_and_ignores_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gh"
            path.write_text("#!/bin/sh\nexit 0\n")
            path.chmod(0o777)
            with (
                patch.object(setup, "GH_CANDIDATES", (path,)),
                patch.dict(os.environ, {"PATH": directory}),
            ):
                with self.assertRaises(RuntimeError):
                    setup.trusted_gh()
            path.chmod(0o755)
            with patch.object(setup, "GH_CANDIDATES", (path,)):
                self.assertEqual(setup.trusted_gh(), path.resolve())

    def test_authentication_mode_is_explicit_and_closed(self):
        self.assertEqual(
            setup.helper_command("/tmp/ui", "/opt/homebrew/bin/gh", "subscription", check=True),
            ["/tmp/ui", "--gh", "/opt/homebrew/bin/gh", "--claude-auth", "subscription", "--check"],
        )
        self.assertEqual(setup.helper_command("/tmp/ui", "/usr/bin/gh", "api")[-1], "api")
        with self.assertRaises(ValueError):
            setup.helper_command("/tmp/ui", "/usr/bin/gh", "unexpected")

    def test_subprocess_environment_does_not_forward_tokens_or_debug_overrides(self):
        invented = "invented-secret-never-use"
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": invented,
                "GH_TOKEN": invented,
                "GH_DEBUG": "api",
                "GH_HOST": "unexpected.example",
                "GH_CONFIG_DIR": "/tmp/untrusted",
            },
        ):
            environment = setup.clean_environment()
        for name in ("OPENAI_API_KEY", "GH_TOKEN", "GH_DEBUG", "GH_CONFIG_DIR"):
            self.assertNotIn(name, environment)
        self.assertNotIn(invented, repr(environment))
        self.assertEqual(environment["GH_PROMPT_DISABLED"], "1")
        self.assertEqual(environment["GH_HOST"], "github.com")


if __name__ == "__main__":
    unittest.main()
