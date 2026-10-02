"""Read explicitly configured hosted credentials without exposing them in diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

KEYCHAIN_SERVICE = "rightyo.jev"
KEYCHAIN_ACCOUNT = "api-key"
# Hosted speech backends use their own items: the transcriber and diarizer services
# are configured independently, so each has a separate environment variable and item.
TRANSCRIBER_ENV = "RIGHTYO_TRANSCRIBER_API_KEY"
TRANSCRIBER_KEYCHAIN_SERVICE = "rightyo.transcriber"
DIARIZER_ENV = "RIGHTYO_DIARIZER_API_KEY"
DIARIZER_KEYCHAIN_SERVICE = "rightyo.diarizer"


class CredentialError(RuntimeError):
    """A credential is missing or inaccessible; messages never include its value."""


def _validate_key(value: str, label: str) -> str:
    key = value.strip()
    if not 8 <= len(key) <= 8192 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise CredentialError(f"The configured {label} credential is invalid.")
    return key


def _load_api_key(environment: str, service: str, label: str, hint: str) -> str:
    configured = os.environ.get(environment)
    if configured is not None:
        return _validate_key(configured, label)
    if sys.platform != "darwin":
        raise CredentialError(f"Configure {environment} through your secret manager.")
    result = None
    try:
        result = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                service,
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
        raise CredentialError(f"The {label} login Keychain item could not be accessed.")
    if result.returncode != 0:
        raise CredentialError(hint)
    decoded = None
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeDecodeError:
        pass
    if decoded is None:
        raise CredentialError(f"The configured {label} credential is invalid.")
    return _validate_key(decoded, label)


def load_jev_api_key() -> str:
    """Return the key to the calling process; never print or persist it.

    On macOS the native secure dialog stores the item in the login Keychain.
    An explicit environment variable also supports user-managed deployments.
    Neither path loads .env files, and the key is never passed in process argv.
    """
    return _load_api_key(
        "TYPESAFE_API_KEY",
        KEYCHAIN_SERVICE,
        "Jev",
        "Save a Jev API key with the native RightyO credential dialog.",
    )


def load_transcriber_api_key() -> str:
    """The hosted transcriber key: `RIGHTYO_TRANSCRIBER_API_KEY` or login Keychain item
    service `rightyo.transcriber`, account `api-key`. Same rules as the Jev key."""
    return _load_api_key(
        TRANSCRIBER_ENV,
        TRANSCRIBER_KEYCHAIN_SERVICE,
        "hosted transcriber",
        "Save a hosted transcriber API key in the login Keychain "
        f"(service {TRANSCRIBER_KEYCHAIN_SERVICE}, account {KEYCHAIN_ACCOUNT}).",
    )


def load_diarizer_api_key() -> str:
    """The hosted diarizer key: `RIGHTYO_DIARIZER_API_KEY` or login Keychain item
    service `rightyo.diarizer`, account `api-key`. Same rules as the Jev key."""
    return _load_api_key(
        DIARIZER_ENV,
        DIARIZER_KEYCHAIN_SERVICE,
        "hosted diarizer",
        "Save a hosted diarizer API key in the login Keychain "
        f"(service {DIARIZER_KEYCHAIN_SERVICE}, account {KEYCHAIN_ACCOUNT}).",
    )
