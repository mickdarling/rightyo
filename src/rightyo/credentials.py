"""Read an explicitly configured Jev key without exposing it in diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

KEYCHAIN_SERVICE = "rightyo.jev"
KEYCHAIN_ACCOUNT = "api-key"


class CredentialError(RuntimeError):
    """A credential is missing or inaccessible; messages never include its value."""


def _validate_key(value: str) -> str:
    key = value.strip()
    if not 8 <= len(key) <= 8192 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise CredentialError("The configured Jev credential is invalid.")
    return key


def load_jev_api_key() -> str:
    """Return the key to the calling process; never print or persist it.

    On macOS the native secure dialog stores the item in the login Keychain.
    An explicit environment variable also supports user-managed deployments.
    Neither path loads .env files, and the key is never passed in process argv.
    """
    configured = os.environ.get("TYPESAFE_API_KEY")
    if configured is not None:
        return _validate_key(configured)
    if sys.platform != "darwin":
        raise CredentialError("Configure TYPESAFE_API_KEY through your secret manager.")
    result = None
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                KEYCHAIN_ACCOUNT,
                "-w",
                str(Path.home() / "Library" / "Keychains" / "login.keychain-db"),
            ],
            capture_output=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Raise outside the handler: timeout exceptions can carry partial stdout.
        pass
    if result is None:
        raise CredentialError("The Jev login Keychain item could not be accessed.")
    if result.returncode != 0:
        raise CredentialError("Save a Jev API key with the native RightyO credential dialog.")
    decoded = None
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if decoded is None:
        raise CredentialError("The configured Jev credential is invalid.")
    return _validate_key(decoded)
