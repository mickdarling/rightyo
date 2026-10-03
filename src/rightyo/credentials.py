"""Read explicitly configured hosted credentials without exposing them in diagnostics."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

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


def _validate_key(value: str) -> str | None:
    """The usable key, or None when malformed; never raises while holding the value."""
    key = value.strip()
    if not 8 <= len(key) <= 8192 or any(ord(char) < 33 or ord(char) > 126 for char in key):
        return None
    return key


def _invalid(label: str) -> CredentialError:
    """Built in a frame that holds no secret, so tracebacks never carry the value."""
    return CredentialError(f"The configured {label} credential is invalid.")


def _keychain_lookup(
    service: str, cancelled: Callable[[], bool] | None, timeout_seconds: float
) -> tuple[int, bytes] | str:
    """Run `security` under a wall-clock deadline that cancellation can cut short.

    Returns (exit status, stdout) or a reason word; never raises with process output.
    """
    deadline = time.monotonic() + timeout_seconds
    process = None
    reason = None
    try:
        process = subprocess.Popen(
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
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        while process.poll() is None:
            if cancelled is not None and cancelled():
                reason = "cancelled"
                break
            if time.monotonic() >= deadline:
                reason = "timeout"
                break
            time.sleep(0.05)
        if reason is not None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            return reason
        assert process.stdout is not None
        return process.returncode, process.stdout.read()
    except (OSError, subprocess.SubprocessError, ValueError):
        return "inaccessible"
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()


def _load_api_key(
    environment: str,
    service: str,
    label: str,
    hint: str,
    *,
    cancelled: Callable[[], bool] | None = None,
    timeout_seconds: float = 120,
) -> str:
    # Every raise below happens after the secret-bearing locals are dropped, so a
    # retained traceback can show this frame without the credential in it.
    configured = os.environ.get(environment)
    if configured is not None:
        key = _validate_key(configured)
        del configured
        if key is None:
            raise _invalid(label)
        return key
    if sys.platform != "darwin":
        raise CredentialError(f"Configure {environment} through your secret manager.")
    result = _keychain_lookup(service, cancelled, timeout_seconds)
    if result == "cancelled":
        raise CredentialError(f"The {label} credential lookup was cancelled.")
    if isinstance(result, str):
        raise CredentialError(f"The {label} login Keychain item could not be accessed.")
    status, output = result
    del result
    if status != 0:
        del output
        raise CredentialError(hint)
    key = None
    try:
        key = _validate_key(output.decode("utf-8"))
    except UnicodeDecodeError:
        pass
    del output
    if key is None:
        raise _invalid(label)
    return key


def load_jev_api_key(
    *, cancelled: Callable[[], bool] | None = None, timeout_seconds: float = 120
) -> str:
    """Return the key to the calling process; never print or persist it.

    On macOS the native secure dialog stores the item in the login Keychain.
    An explicit environment variable also supports user-managed deployments.
    Neither path loads .env files, and the key is never passed in process argv.
    A Keychain lookup ends early when `cancelled` reports true or `timeout_seconds` pass.
    """
    return _load_api_key(
        "TYPESAFE_API_KEY",
        KEYCHAIN_SERVICE,
        "Jev",
        "Save a Jev API key with the native RightyO credential dialog.",
        cancelled=cancelled,
        timeout_seconds=timeout_seconds,
    )


def load_transcriber_api_key(
    *, cancelled: Callable[[], bool] | None = None, timeout_seconds: float = 120
) -> str:
    """The hosted transcriber key: `RIGHTYO_TRANSCRIBER_API_KEY` or login Keychain item
    service `rightyo.transcriber`, account `api-key`. Same rules as the Jev key."""
    return _load_api_key(
        TRANSCRIBER_ENV,
        TRANSCRIBER_KEYCHAIN_SERVICE,
        "hosted transcriber",
        "Save a hosted transcriber API key in the login Keychain "
        f"(service {TRANSCRIBER_KEYCHAIN_SERVICE}, account {KEYCHAIN_ACCOUNT}).",
        cancelled=cancelled,
        timeout_seconds=timeout_seconds,
    )


def load_diarizer_api_key(
    *, cancelled: Callable[[], bool] | None = None, timeout_seconds: float = 120
) -> str:
    """The hosted diarizer key: `RIGHTYO_DIARIZER_API_KEY` or login Keychain item
    service `rightyo.diarizer`, account `api-key`. Same rules as the Jev key."""
    return _load_api_key(
        DIARIZER_ENV,
        DIARIZER_KEYCHAIN_SERVICE,
        "hosted diarizer",
        "Save a hosted diarizer API key in the login Keychain "
        f"(service {DIARIZER_KEYCHAIN_SERVICE}, account {KEYCHAIN_ACCOUNT}).",
        cancelled=cancelled,
        timeout_seconds=timeout_seconds,
    )
