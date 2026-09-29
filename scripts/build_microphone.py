#!/usr/bin/env python3
"""Build an ignored, ad-hoc signed Mac microphone helper; never start capture."""

from __future__ import annotations

import plistlib
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if sys.platform != "darwin":
        print("The microphone helper requires macOS.", file=sys.stderr)
        return 1
    root = Path(__file__).resolve().parents[1]
    contents = root / "local/RightyOMicrophone.app/Contents"
    binary = contents / "MacOS/RightyOMicrophone"
    binary.parent.mkdir(parents=True, exist_ok=True)
    info = contents / "Info.plist"
    info.write_bytes(
        plistlib.dumps(
            {
                "CFBundleIdentifier": "io.rightyo.microphone",
                "CFBundleName": "RightyO Microphone",
                "CFBundleDisplayName": "RightyO Microphone",
                "CFBundleExecutable": "RightyOMicrophone",
                "CFBundlePackageType": "APPL",
                "CFBundleVersion": "1",
                "CFBundleShortVersionString": "0.1.0",
                "LSMinimumSystemVersion": "14.0",
                "LSUIElement": True,
                "NSMicrophoneUsageDescription": (
                    "RightyO uses your microphone only after you click Start. "
                    "Speech is transcribed locally; hosted decisions send text only "
                    "when you explicitly enable them. Stop discards session buffers."
                ),
            }
        )
    )
    try:
        subprocess.run(
            [
                "/usr/bin/xcrun",
                "swiftc",
                "-O",
                "-framework",
                "AVFoundation",
                str(root / "scripts/microphone.swift"),
                "-o",
                str(binary),
                "-Xlinker",
                "-sectcreate",
                "-Xlinker",
                "__TEXT",
                "-Xlinker",
                "__info_plist",
                "-Xlinker",
                str(info),
            ],
            check=True,
        )
        subprocess.run(
            [
                "/usr/bin/codesign",
                "--force",
                "--sign",
                "-",
                "--identifier",
                "io.rightyo.microphone",
                "--requirements",
                '=designated => identifier "io.rightyo.microphone"',
                str(contents.parent),
            ],
            check=True,
        )
        subprocess.run(
            ["/usr/bin/codesign", "--verify", "--strict", str(contents.parent)], check=True
        )
    except (OSError, subprocess.CalledProcessError):
        print("Microphone helper build failed; capture was not started.", file=sys.stderr)
        return 1
    print(f"Built {binary}; microphone capture was not started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
