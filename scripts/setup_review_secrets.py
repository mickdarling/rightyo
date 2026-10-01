"""Open a native masked dialog that sends review credentials directly to GitHub Secrets."""

import argparse
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "mickdarling/rightyo"
GH_CANDIDATES = (Path("/opt/homebrew/bin/gh"), Path("/usr/local/bin/gh"), Path("/usr/bin/gh"))
SOURCE = Path(__file__).with_name("review_secrets.swift")


def trusted_gh():
    """Select an installed CLI from fixed locations, never cwd or a caller's PATH."""
    for candidate in GH_CANDIDATES:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            resolved = candidate.resolve()
            if resolved.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                continue
            return resolved
    raise RuntimeError("Install the official GitHub CLI in a standard location first.")


def helper_command(binary, gh, claude_auth, *, check=False):
    if claude_auth not in ("subscription", "api"):
        raise ValueError("Unsupported Claude authentication mode.")
    command = [str(binary), "--gh", str(gh), "--claude-auth", claude_auth]
    if check:
        command.append("--check")
    return command


def clean_environment():
    # gh uses its normal login configuration. No inherited provider/debug/token override.
    return {
        "HOME": str(Path.home()),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "GH_HOST": "github.com",
        "GH_PROMPT_DISABLED": "1",
        "GH_NO_UPDATE_NOTIFIER": "1",
        "GH_NO_EXTENSION_UPDATE_NOTIFIER": "1",
        "NO_COLOR": "1",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-auth", choices=("subscription", "api"), required=True)
    parser.add_argument("--check", action="store_true", help="compile/check only; opens no dialog")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("the secure credential-entry dialog requires macOS")
    try:
        gh = trusted_gh()
        env = clean_environment()
        compiler = subprocess.run(
            ["/usr/bin/xcrun", "--find", "swiftc"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        if not Path(compiler).is_absolute() or not compiler.startswith(
            ("/Library/Developer/", "/Applications/Xcode.app/Contents/Developer/")
        ):
            raise RuntimeError("The official Apple Swift compiler could not be located.")
        sdk = subprocess.run(
            ["/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
        if not Path(sdk).is_dir() or not sdk.startswith(
            ("/Library/Developer/", "/Applications/Xcode.app/Contents/Developer/")
        ):
            raise RuntimeError("The official Apple macOS SDK could not be located.")
        # This directory holds only source-derived executable code, never credential data.
        with tempfile.TemporaryDirectory(
            prefix="rightyo-review-secret-ui-", dir="/tmp"
        ) as directory:
            binary = Path(directory) / "review-secrets"
            subprocess.run(
                [compiler, "-sdk", sdk, str(SOURCE), "-o", str(binary)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=120,
            )
            result = subprocess.run(
                helper_command(binary, gh, args.claude_auth, check=args.check), env=env, check=False
            )
            return result.returncode
    except (OSError, RuntimeError, subprocess.SubprocessError):
        print(
            "Review credential setup failed before completion; no credential value is shown.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
