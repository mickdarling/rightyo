"""Credential regressions use invented tokens and never access a real Keychain."""

import os
import subprocess
import unittest
from unittest.mock import patch

from rightyo.credentials import CredentialError, load_jev_api_key

TOKEN = "synthetic-test-token-do-not-use"


class CredentialTests(unittest.TestCase):
    def test_explicit_environment_does_not_access_keychain(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": TOKEN}, clear=True):
            with patch("rightyo.credentials.subprocess.run") as runner:
                self.assertEqual(load_jev_api_key(), TOKEN)
                runner.assert_not_called()

    def test_keychain_value_is_captured_not_in_argv(self):
        response = subprocess.CompletedProcess([], 0, (TOKEN + "\n").encode(), b"")
        with patch.dict(os.environ, {}, clear=True), patch("rightyo.credentials.sys.platform", "darwin"):
            with patch("rightyo.credentials.subprocess.run", return_value=response) as runner:
                self.assertEqual(load_jev_api_key(), TOKEN)
                arguments, options = runner.call_args
                self.assertNotIn(TOKEN, str(arguments))
                self.assertTrue(options["capture_output"])

    def test_failure_does_not_expose_subprocess_output(self):
        response = subprocess.CompletedProcess([], 1, TOKEN.encode(), TOKEN.encode())
        with patch.dict(os.environ, {}, clear=True), patch("rightyo.credentials.sys.platform", "darwin"):
            with patch("rightyo.credentials.subprocess.run", return_value=response):
                with self.assertRaises(CredentialError) as error:
                    load_jev_api_key()
                self.assertNotIn(TOKEN, str(error.exception))

    def test_invalid_environment_is_redacted(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": TOKEN + "\nINJECTED"}, clear=True):
            with self.assertRaises(CredentialError) as error:
                load_jev_api_key()
            self.assertNotIn(TOKEN, str(error.exception))

    def test_timeout_is_sanitized(self):
        with patch.dict(os.environ, {}, clear=True), patch("rightyo.credentials.sys.platform", "darwin"):
            with patch("rightyo.credentials.subprocess.run", side_effect=subprocess.TimeoutExpired(TOKEN, 30)):
                with self.assertRaises(CredentialError) as error:
                    load_jev_api_key()
                self.assertNotIn(TOKEN, str(error.exception))


if __name__ == "__main__":
    unittest.main()
