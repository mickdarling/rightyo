"""Credential regressions use invented tokens and never access a real Keychain."""

import io
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rightyo.credentials import CredentialError, load_jev_api_key

TOKEN = "synthetic-test-token-do-not-use"


class FakeProcess:
    """A stand-in for `security`: completes, or hangs until terminated."""

    def __init__(self, args, returncode=0, output=b"", hang=False, **options):
        self.args, self.options = args, options
        self.exit_status = returncode
        self.stdout = io.BytesIO(output)
        self.hang = hang
        self.terminated = False
        self.returncode = None

    def poll(self):
        if self.hang and not self.terminated:
            return None
        self.returncode = -15 if self.terminated else self.exit_status
        return self.returncode

    def terminate(self):
        self.terminated = True

    kill = terminate

    def wait(self, timeout=None):
        return self.poll()


def fake_popen(**defaults):
    spawned = []

    def popen(args, **options):
        process = FakeProcess(args, **defaults, **options)
        spawned.append(process)
        return process

    return patch("rightyo.credentials.subprocess.Popen", side_effect=popen), spawned


class CredentialTests(unittest.TestCase):
    def test_explicit_environment_does_not_access_keychain(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": TOKEN}, clear=True):
            with patch("rightyo.credentials.subprocess.Popen") as runner:
                self.assertEqual(load_jev_api_key(), TOKEN)
                runner.assert_not_called()

    def test_keychain_value_is_captured_not_in_argv(self):
        popen, spawned = fake_popen(output=(TOKEN + "\n").encode())
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            popen,
        ):
            self.assertEqual(load_jev_api_key(), TOKEN)
        (process,) = spawned
        self.assertNotIn(TOKEN, str(process.args))
        self.assertEqual(process.options["stdout"], -1)  # subprocess.PIPE
        self.assertEqual(process.args[-1], str(Path.home() / "Library/Keychains/login.keychain-db"))

    def test_failure_does_not_expose_subprocess_output(self):
        popen, _ = fake_popen(returncode=1, output=TOKEN.encode())
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            popen,
        ):
            with self.assertRaises(CredentialError) as error:
                load_jev_api_key()
            self.assertNotIn(TOKEN, str(error.exception))
            self.assertIsNone(error.exception.__context__)

    def test_invalid_environment_is_redacted(self):
        with patch.dict(os.environ, {"TYPESAFE_API_KEY": TOKEN + "\nINJECTED"}, clear=True):
            with self.assertRaises(CredentialError) as error:
                load_jev_api_key()
            self.assertNotIn(TOKEN, str(error.exception))
            self.assertNoSecretInFrames(error.exception)

    def assertNoSecretInFrames(self, exception):
        """No frame on the exception's traceback may hold the credential in a local."""
        import traceback

        held = []
        for frame, _line in traceback.walk_tb(exception.__traceback__):
            held.extend(repr(value) for value in frame.f_locals.values())
        self.assertFalse(any(TOKEN in value for value in held), held)

    def test_malformed_keychain_value_leaves_no_secret_in_tracebacks(self):
        for output in (TOKEN.encode() + b"\xff", (TOKEN + " with space").encode(), b"short"):
            popen, _ = fake_popen(output=output)
            with (
                self.subTest(output=output),
                patch.dict(os.environ, {}, clear=True),
                patch("rightyo.credentials.sys.platform", "darwin"),
                popen,
            ):
                with self.assertRaises(CredentialError) as error:
                    load_jev_api_key()
                self.assertNotIn(TOKEN, str(error.exception))
                self.assertIsNone(error.exception.__context__)
                self.assertNoSecretInFrames(error.exception)

    def test_timeout_terminates_the_lookup_and_is_sanitized(self):
        clock = [0.0]

        def monotonic():
            clock[0] += 50.0
            return clock[0]

        popen, spawned = fake_popen(hang=True, output=TOKEN.encode())
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch("rightyo.credentials.time.monotonic", monotonic),
            patch("rightyo.credentials.time.sleep"),
            popen,
        ):
            with self.assertRaises(CredentialError) as error:
                load_jev_api_key()
            self.assertNotIn(TOKEN, str(error.exception))
            self.assertIn("could not be accessed", str(error.exception))
            self.assertIsNone(error.exception.__context__)
        self.assertTrue(spawned[0].terminated)

    def test_cancellation_ends_the_lookup_before_its_deadline(self):
        polls = []
        popen, spawned = fake_popen(hang=True)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            patch("rightyo.credentials.time.sleep", lambda _s: polls.append(1)),
            popen,
        ):
            with self.assertRaisesRegex(CredentialError, "cancelled"):
                load_jev_api_key(cancelled=lambda: len(polls) >= 3)
        self.assertTrue(spawned[0].terminated)
        self.assertEqual(len(polls), 3)

    def test_invalid_keychain_encoding_has_no_secret_exception_context(self):
        popen, _ = fake_popen(output=TOKEN.encode() + b"\xff")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("rightyo.credentials.sys.platform", "darwin"),
            popen,
        ):
            with self.assertRaises(CredentialError) as error:
                load_jev_api_key()
            self.assertNotIn(TOKEN, str(error.exception))
            self.assertIsNone(error.exception.__context__)


if __name__ == "__main__":
    unittest.main()
